#!/usr/bin/env python3
"""
LERNA Full Experiment Orchestrator v2

Upgraded from v1 to fix all issues from the March 2026 5-day experiment:

  FIX 1: DISK SPACE
    --clean-checkpoints removes checkpoint-* dirs after each successful run
    --min-disk-gb N stops before disk fills (default 15 GB)
    Disk usage printed in every progress line

  FIX 2: PER-TASK LEARNING RATE
    TASK_LR dict mirrors TASK_HP_OVERRIDES in run_baseline_glue.py
    CoLA uses lr=1e-05, all others use lr=2e-05
    Orchestrator passes the correct LR to each subprocess

  FIX 3: W&B GROUP FRAGMENTATION
    --wandb-group is deterministic (hash of output-dir) so restarts reuse it
    Prints the exact resume command on exit
    No more auto-generated timestamp groups on every restart

  FIX 4: FLEXIBLE RUN DETECTION
    find_completed_run() globs {task}_s{seed}_lr*/results.json
    Finds CoLA runs saved as cola_s42_lr1e-05 even when expected lr is 2e-05
    Writes .completed sentinel for extra safety

Usage:
  # Clean start (recommended)
  python scripts/run_full_experiment_v2.py \\
    --output-dir /home/sheheryar/lerna_clean_experiment \\
    --num-seeds 10 --wandb --wandb-project lerna-2026-clean \\
    --clean-checkpoints

  # Resume after crash (same command, auto-skips finished runs)
  python scripts/run_full_experiment_v2.py \\
    --output-dir /home/sheheryar/lerna_clean_experiment \\
    --num-seeds 10 --wandb --wandb-project lerna-2026-clean \\
    --wandb-group full-a1b2c \\
    --clean-checkpoints
"""

import os, sys, json, glob, time, shutil, signal, subprocess, argparse
import psutil, logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
import numpy as np

# ======================================================================
# FIX 2 -- Per-task learning rates (synced with TASK_HP_OVERRIDES)
# ======================================================================
TASK_LR = {
    "sst2": 2e-05,
    "qnli": 2e-05,
    "qqp":  2e-05,
    "mnli": 2e-05,
    "rte":  2e-05,
    "mrpc": 2e-05,
    "cola": 1e-05,
    "stsb": 2e-05,
}

GLUE_TASKS = ["sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb"]
DEFAULT_LR = 2e-5
MAX_RETRIES = 3
MEMORY_WARN_GB = 4.0
COOLDOWN_S = 5
DISK_MIN_GB = 15.0


# ======================================================================
# Logging
# ======================================================================
def setup_logging(output_dir):
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"orchestrator_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = logging.getLogger("orch_v2")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S"))
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    return logger, log_file


# ======================================================================
# System helpers
# ======================================================================
def get_system_stats():
    mem = psutil.virtual_memory()
    s = {"ram_avail_gb": mem.available / 1024**3, "ram_total_gb": mem.total / 1024**3, "ram_pct": mem.percent}
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=memory.used,memory.total,temperature.gpu,power.draw",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            p = r.stdout.strip().split(",")
            if len(p) >= 4:
                s["gpu_mem_mb"] = float(p[0]); s["gpu_total_mb"] = float(p[1])
                s["gpu_temp"] = float(p[2]); s["gpu_power"] = float(p[3])
    except Exception:
        pass
    return s


def disk_free_gb(path):
    try:
        return shutil.disk_usage(path).free / 1024**3
    except Exception:
        return float("inf")


def check_memory(logger):
    avail = get_system_stats()["ram_avail_gb"]
    if avail < MEMORY_WARN_GB:
        logger.warning(f"Low RAM: {avail:.1f}GB. Waiting up to 60s...")
        for i in range(6):
            time.sleep(10)
            avail = get_system_stats()["ram_avail_gb"]
            if avail >= MEMORY_WARN_GB:
                logger.info(f"RAM recovered: {avail:.1f}GB"); return True
        logger.error(f"RAM still low ({avail:.1f}GB). Continuing anyway."); return False
    return True


# ======================================================================
# FIX 1 -- Checkpoint cleanup
# ======================================================================
def slim_checkpoints(run_dir, logger):
    """Remove optimizer bloat from checkpoints while PRESERVING model weights.

    Your research needs model.safetensors at each checkpoint for:
      - TracIn step attribution (Phase 4)
      - Checkpoint forking experiments (Phase 4)
      - Post-hoc analysis of parameter drift during plateaus

    What we DELETE (only needed to resume training, not for analysis):
      - optimizer.pt          (~1GB each, the main disk hog)
      - scheduler.pt          (~1KB)
      - rng_state_*.pth       (~1KB each)

    What we KEEP:
      - model.safetensors     (~500MB, the actual model weights)
      - config.json           (needed to load the model)
      - trainer_state.json    (step/epoch metadata, tiny)
      - training_args.bin     (hyperparameter record, tiny)

    Returns bytes freed.
    """
    DELETABLE = {"optimizer.pt", "scheduler.pt"}
    DELETABLE_PREFIXES = ("rng_state",)
    freed = 0

    for ckpt_dir in sorted(Path(run_dir).glob("checkpoint-*")):
        if not ckpt_dir.is_dir():
            continue
        for f in ckpt_dir.iterdir():
            if not f.is_file():
                continue
            should_delete = (
                f.name in DELETABLE
                or f.name.startswith(DELETABLE_PREFIXES)
            )
            if should_delete:
                try:
                    sz = f.stat().st_size
                    f.unlink()
                    freed += sz
                    logger.debug(f"    Removed {ckpt_dir.name}/{f.name} ({sz/1024**2:.1f}MB)")
                except Exception as e:
                    logger.warning(f"    Could not remove {f}: {e}")

    # Also clean wandb local cache inside run dir (already uploaded to cloud)
    for sub in ["runs", "wandb"]:
        p = Path(run_dir) / sub
        if p.is_dir():
            try:
                sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p); freed += sz
            except Exception:
                pass

    if freed > 0:
        logger.info(f"  Slimmed {freed/1024**3:.2f}GB from {Path(run_dir).name} (model weights preserved)")
    return freed


def cleanup_checkpoints(run_dir, logger):
    """FULL deletion of checkpoint-* dirs. Use only if you do NOT need
    model weights for TracIn/forking. Prefer slim_checkpoints() instead."""
    freed = 0
    for d in sorted(Path(run_dir).glob("checkpoint-*")):
        if d.is_dir():
            try:
                sz = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                shutil.rmtree(d); freed += sz
            except Exception as e:
                logger.warning(f"  Could not remove {d}: {e}")
    for sub in ["runs", "wandb"]:
        p = Path(run_dir) / sub
        if p.is_dir():
            try:
                sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
                shutil.rmtree(p); freed += sz
            except Exception:
                pass
    if freed > 0:
        logger.info(f"  Cleaned {freed/1024**3:.2f}GB from {Path(run_dir).name}")
    return freed


# ======================================================================
# FIX 4 -- Flexible run detection (wildcard LR)
# ======================================================================
def get_run_id(task, seed, lr):
    return f"{task}_s{seed}_lr{lr:.0e}"


def _valid_results(path):
    """True if path is a results.json with eval_metrics and no error."""
    try:
        with open(path) as f:
            d = json.load(f)
        return "eval_metrics" in d and "error" not in d
    except Exception:
        return False


def find_completed_run(output_dir, task, seed, expected_lr=None):
    """Return (run_dir, actual_lr) or (None, None).

    First checks the expected LR path, then globs across all LRs.
    """
    base = Path(output_dir)
    # Fast path: exact match
    if expected_lr is not None:
        rid = get_run_id(task, seed, expected_lr)
        rf = base / rid / "results.json"
        if _valid_results(rf):
            return str(base / rid), expected_lr
    # Slow path: wildcard
    for rd in sorted(base.glob(f"{task}_s{seed}_lr*")):
        if _valid_results(rd / "results.json"):
            try:
                alr = float(rd.name.split("_lr")[-1])
            except Exception:
                alr = None
            return str(rd), alr
    # Sentinel check
    if expected_lr is not None:
        rid = get_run_id(task, seed, expected_lr)
        if (base / rid / ".completed").exists():
            return str(base / rid), expected_lr
    return None, None


def is_run_completed(output_dir, task, seed, lr):
    d, _ = find_completed_run(output_dir, task, seed, expected_lr=lr)
    return d is not None


# ======================================================================
# Pre-flight checks
# ======================================================================
def preflight(output_dir, tasks, seeds, logger):
    ok = True
    free = disk_free_gb(output_dir)
    logger.info(f"  Disk free: {free:.1f}GB")
    if free < 20:
        logger.warning(f"  Low disk! Need ~5GB peak per run."); ok = False
    try:
        r = subprocess.run(["nvidia-smi","--query-gpu=name,memory.total","--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        if r.returncode == 0:
            logger.info(f"  GPU: {r.stdout.strip()}")
        else:
            logger.warning("  nvidia-smi failed"); ok = False
    except FileNotFoundError:
        logger.warning("  No GPU detected"); ok = False
    if not Path("scripts/run_baseline_glue.py").exists():
        logger.warning("  Training script not found"); ok = False
    for t in tasks:
        if t not in TASK_LR:
            logger.warning(f"  No LR configured for task: {t}"); ok = False
    if ok:
        logger.info("  Pre-flight OK")
    return ok


# ======================================================================
# Subprocess runner
# ======================================================================
def run_subprocess(task, seed, lr, output_dir, use_wandb, wandb_project,
                   wandb_group, max_samples, script, logger, idx, total, timeout=7200):
    run_id = get_run_id(task, seed, lr)
    cmd = [sys.executable, script, "--mode", "custom", "--tasks", task,
           "--seeds", str(seed), "--lr", str(lr), "--output-dir", output_dir]
    if use_wandb:
        cmd += ["--wandb", "--wandb-project", wandb_project, "--wandb-group", wandb_group]
    if max_samples is not None:
        cmd += ["--max-samples", str(max_samples)]
    else:
        cmd.append("--unlimited")

    logger.info(f"[{idx}/{total}] Starting: {run_id}")
    logger.debug(f"  cmd: {' '.join(cmd)}")
    t0 = time.time()
    try:
        env = os.environ.copy()
        env.update({"TORCH_COMPILE_DISABLE": "1", "TORCHDYNAMO_DISABLE": "1",
                    "WANDB_START_METHOD": "thread", "WANDB_LOG_MODEL": "false",
                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        lines = []
        for line in iter(proc.stdout.readline, ""):
            line = line.rstrip()
            if line:
                lines.append(line)
                if any(k in line for k in ["Final Results","ETA","LERNA step","FAILED","Error","energy","Eval metrics","Saved:"]):
                    logger.info(f"  {line}")
        proc.stdout.close()
        rc = proc.wait(timeout=timeout)
        dur = time.time() - t0
        if rc == 0:
            rf = Path(output_dir) / run_id / "results.json"
            if _valid_results(rf):
                (Path(output_dir) / run_id / ".completed").write_text(datetime.now().isoformat())
                logger.info(f"[{idx}/{total}] Done: {run_id} in {timedelta(seconds=int(dur))}")
                with open(rf) as f:
                    return True, json.load(f), dur
            logger.warning(f"[{idx}/{total}] {run_id}: exit 0 but no valid results.json")
            return False, None, dur
        logger.error(f"[{idx}/{total}] {run_id}: exit {rc} after {timedelta(seconds=int(dur))}")
        for l in lines[-20:]:
            logger.error(f"  | {l}")
        return False, None, dur
    except subprocess.TimeoutExpired:
        logger.error(f"[{idx}/{total}] {run_id}: TIMEOUT {timeout}s")
        proc.kill(); proc.wait()
        return False, None, time.time() - t0
    except Exception as e:
        logger.error(f"[{idx}/{total}] {run_id}: {e}")
        return False, None, time.time() - t0


# ======================================================================
# Summary
# ======================================================================
def generate_summary(output_dir, tasks, seeds, logger):
    results = []
    for task in tasks:
        lr = TASK_LR.get(task, DEFAULT_LR)
        for seed in seeds:
            rd, _ = find_completed_run(output_dir, task, seed, expected_lr=lr)
            if rd:
                rf = Path(rd) / "results.json"
                if rf.exists():
                    try:
                        with open(rf) as f:
                            results.append(json.load(f))
                    except Exception:
                        pass
    sp = Path(output_dir) / "baseline_summary.json"
    with open(sp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    ok = [r for r in results if "error" not in r]
    fail = [r for r in results if "error" in r]
    logger.info(f"\n{'='*80}")
    logger.info(f"  SUMMARY: {len(ok)} succeeded, {len(fail)} failed out of {len(results)}")
    if ok:
        logger.info(f"  {'Task':<8} {'LR':>9} {'N':>3} {'Mean':>9} {'Std':>8} {'kWh':>10} {'Waste':>7}")
        logger.info(f"  {'-'*60}")
        for task in tasks:
            tr = [r for r in ok if r["task"] == task]
            if not tr:
                continue
            accs = []
            for r in tr:
                em = r.get("eval_metrics", {})
                accs.append(em.get("eval_accuracy", em.get("eval_matthews_correlation", em.get("eval_pearsonr", 0))))
            kwhs = [r.get("energy_kwh", 0) for r in tr]
            wastes = [r.get("waste_metrics", {}).get("waste_ratio", 0) for r in tr]
            tlr = tr[0].get("learning_rate", TASK_LR.get(task, DEFAULT_LR))
            logger.info(f"  {task:<8} {tlr:>9.0e} {len(accs):>3} {np.mean(accs):>9.4f} "
                        f"{np.std(accs):>8.4f} {np.mean(kwhs):>10.6f} {np.mean(wastes):>7.3f}")
    logger.info(f"  Saved: {sp}")
    logger.info(f"{'='*80}")
    return results


# ======================================================================
# FIX 3 -- Deterministic W&B group name
# ======================================================================
def make_wandb_group(output_dir, user_group=None):
    if user_group:
        return user_group
    h = abs(hash(str(Path(output_dir).resolve()))) % 100000
    return f"full-{h:05d}"


# ======================================================================
# Main
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="LERNA Orchestrator v2",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--num-seeds", type=int, default=10)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="lerna-baseline")
    ap.add_argument("--wandb-group", default=None,
                    help="Reuse the SAME group on restart to avoid fragmentation")
    ap.add_argument("--max-samples", type=int, default=25000, help="0 = unlimited")
    ap.add_argument("--max-retries", type=int, default=MAX_RETRIES)
    ap.add_argument("--cooldown", type=int, default=COOLDOWN_S)
    ap.add_argument("--timeout", type=int, default=7200)
    ap.add_argument("--script", default="scripts/run_baseline_glue.py")
    # NEW flags
    ap.add_argument("--slim-checkpoints", action="store_true",
                    help="Remove optimizer bloat but KEEP model weights for TracIn/forking (saves ~160GB)")
    ap.add_argument("--clean-checkpoints", action="store_true",
                    help="FULL delete of checkpoints (use --slim-checkpoints instead if you need weights)")
    ap.add_argument("--min-disk-gb", type=float, default=DISK_MIN_GB,
                    help="Stop if free disk drops below this (default 15)")
    ap.add_argument("--skip-preflight", action="store_true")
    ap.add_argument("--phase", default=None, help="Experiment phase label (e.g. baseline)")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="Explicit list of seeds (e.g. 42 123 456 789 1024). Overrides --num-seeds.")
    ap.add_argument("--cleanup", action="store_true", help="Alias for --slim-checkpoints")
    ap.add_argument("--cleanup-wandb", action="store_true",
                    help="Remove W&B local runs folder after each completed run")
    args = ap.parse_args()

    if args.cleanup:
        args.slim_checkpoints = True

    tasks = args.tasks or GLUE_TASKS
    seeds = args.seeds if args.seeds else list(range(42, 42 + args.num_seeds))
    max_samples = None if args.max_samples == 0 else args.max_samples
    wg = make_wandb_group(args.output_dir, args.wandb_group)
    logger, log_file = setup_logging(args.output_dir)

    # Graceful shutdown
    stop = [False]
    def _sig(s, f):
        if stop[0]: logger.warning("Force quit!"); sys.exit(1)
        stop[0] = True
        logger.warning("Shutdown requested. Finishing current run...")
    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    # Build queue with per-task LRs
    queue = [(t, s, TASK_LR.get(t, DEFAULT_LR)) for t in tasks for s in seeds]
    total = len(queue)
    done = [(t, s, lr) for t, s, lr in queue if is_run_completed(args.output_dir, t, s, lr)]
    pending = [(t, s, lr) for t, s, lr in queue if not is_run_completed(args.output_dir, t, s, lr)]

    # Banner
    logger.info("")
    logger.info("=" * 65)
    logger.info("  LERNA Orchestrator v2")
    logger.info("=" * 65)
    logger.info(f"  Tasks:       {tasks}")
    logger.info(f"  Seeds:       {seeds}")
    if args.phase:
        logger.info(f"  Phase:       {args.phase}")
    logger.info(f"  Per-task LR: { {t: f'{TASK_LR[t]:.0e}' for t in tasks} }")
    logger.info(f"  Total/Done/Pending: {total}/{len(done)}/{len(pending)}")
    cleanup_mode = "SLIM (keep weights)" if args.slim_checkpoints else ("FULL DELETE" if args.clean_checkpoints else "OFF")
    logger.info(f"  Checkpoint cleanup: {cleanup_mode}")
    logger.info(f"  Min disk: {args.min_disk_gb}GB")
    logger.info(f"  Disk free: {disk_free_gb(args.output_dir):.1f}GB")
    if args.wandb:
        logger.info(f"  W&B project: {args.wandb_project}")
        logger.info(f"  W&B group:   {wg}")
    ss = get_system_stats()
    logger.info(f"  RAM: {ss['ram_avail_gb']:.1f}/{ss['ram_total_gb']:.1f}GB")
    if "gpu_total_mb" in ss:
        logger.info(f"  GPU: {ss['gpu_mem_mb']:.0f}/{ss['gpu_total_mb']:.0f}MB  {ss.get('gpu_temp',0):.0f}C")
    logger.info(f"  Log: {log_file}")
    logger.info("=" * 65 + "\n")

    if not args.skip_preflight and not preflight(args.output_dir, tasks, seeds, logger):
        logger.error("Pre-flight failed. Use --skip-preflight to override.")
        sys.exit(1)

    if not pending:
        logger.info("All runs already completed!")
        generate_summary(args.output_dir, tasks, seeds, logger)
        return

    # Main loop
    t0 = time.time()
    tracker = {"done": list(done), "failed": [], "durs": [], "freed_gb": 0.0}
    counter = len(done)

    for task, seed, lr in pending:
        if stop[0]:
            logger.warning("Stopped by user."); break

        counter += 1
        run_id = get_run_id(task, seed, lr)

        # Disk check
        dfree = disk_free_gb(args.output_dir)
        if dfree < args.min_disk_gb:
            logger.error(f"Disk too low: {dfree:.1f}GB < {args.min_disk_gb}GB. Stopping.")
            logger.error("TIP: use --clean-checkpoints or free space manually.")
            break

        check_memory(logger)

        # ETA
        if tracker["durs"]:
            avg = np.mean(tracker["durs"])
            eta = timedelta(seconds=int(avg * (total - counter + 1)))
            logger.info(f"Progress: {counter}/{total} | ETA: {eta} | Disk: {dfree:.1f}GB")

        # Retry loop
        success = False
        for attempt in range(1, args.max_retries + 1):
            if stop[0]: break
            if attempt > 1:
                w = args.cooldown * attempt
                logger.info(f"Retry {attempt}/{args.max_retries} for {run_id} (wait {w}s)")
                time.sleep(w)

            ok, result, dur = run_subprocess(
                task, seed, lr, args.output_dir, args.wandb, args.wandb_project,
                wg, max_samples, args.script, logger, counter, total, args.timeout)

            if ok:
                success = True
                tracker["done"].append((task, seed, lr))
                tracker["durs"].append(dur)
                # FIX 1: disk cleanup (slim = keep model weights, clean = delete all)
                if args.slim_checkpoints or args.cleanup_wandb:
                    freed = slim_checkpoints(str(Path(args.output_dir) / run_id), logger)
                    tracker["freed_gb"] += freed / 1024**3
                elif args.clean_checkpoints:
                    freed = cleanup_checkpoints(str(Path(args.output_dir) / run_id), logger)
                    tracker["freed_gb"] += freed / 1024**3
                break
            else:
                logger.warning(f"  Attempt {attempt}/{args.max_retries} failed for {run_id}")

        if not success:
            logger.error(f"{run_id}: all {args.max_retries} attempts failed!")
            tracker["failed"].append((task, seed, lr))

        if not stop[0]:
            time.sleep(args.cooldown)

    # Final
    elapsed = time.time() - t0
    logger.info(f"\n{'='*80}")
    logger.info(f"  ORCHESTRATOR v2 COMPLETE")
    logger.info(f"  Wall time: {timedelta(seconds=int(elapsed))}")
    logger.info(f"  Done: {len(tracker['done'])}/{total}  Failed: {len(tracker['failed'])}")
    if tracker["failed"]:
        logger.info(f"  Failed: {tracker['failed']}")
    if args.clean_checkpoints:
        logger.info(f"  Disk freed: {tracker['freed_gb']:.2f}GB")
    logger.info(f"{'='*80}")

    generate_summary(args.output_dir, tasks, seeds, logger)

    # Save metadata
    meta = Path(args.output_dir) / "orchestrator_v2_meta.json"
    with open(meta, "w") as f:
        json.dump({
            "version": "2.0",
            "total": total, "completed": len(tracker["done"]), "failed": len(tracker["failed"]),
            "failed_runs": tracker["failed"],
            "wall_s": elapsed,
            "avg_run_s": float(np.mean(tracker["durs"])) if tracker["durs"] else 0,
            "slim_checkpoints": args.slim_checkpoints,
            "clean_checkpoints": args.clean_checkpoints,
            "freed_gb": tracker["freed_gb"],
            "task_lr": {t: TASK_LR.get(t, DEFAULT_LR) for t in tasks},
            "wandb_group": wg,
            "wandb_project": args.wandb_project if args.wandb else None,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    logger.info(f"  Meta: {meta}")
    logger.info(f"  Log:  {log_file}")

    # Print resume command
    if tracker["failed"] or stop[0]:
        cmd = (f"python scripts/run_full_experiment_v2.py "
               f"--output-dir {args.output_dir} --num-seeds {args.num_seeds} "
               f"--wandb-group {wg}")
        if args.wandb: cmd += f" --wandb --wandb-project {args.wandb_project}"
        if args.slim_checkpoints: cmd += " --slim-checkpoints"
        elif args.clean_checkpoints: cmd += " --clean-checkpoints"
        logger.info(f"\n  To resume:\n  {cmd}")


if __name__ == "__main__":
    main()
