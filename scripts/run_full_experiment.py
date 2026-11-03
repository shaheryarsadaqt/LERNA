#!/usr/bin/env python3
"""
LERNA Full Experiment Orchestrator — Production Grade

Runs all GLUE tasks × seeds with:
  ✅ Subprocess isolation (no memory leaks between runs)
  ✅ Auto-resume (skips completed runs on restart)
  ✅ Retry on failure (up to 3 attempts per run)
  ✅ Memory monitoring (warns before OOM)
  ✅ Graceful shutdown (Ctrl+C saves progress)
  ✅ Final summary generation
  ✅ Detailed logging
  ✅ Faster cooldown (5s between runs)

Usage:
  python scripts/run_full_experiment.py \
    --output-dir /ssd_xs/home/scvi383/scvi383/experiments/full_baseline \
    --num-seeds 10 \
    --wandb
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import psutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════

GLUE_TASKS = ["sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb"]
DEFAULT_LR = 2e-5
MAX_RETRIES = 3
MEMORY_WARN_THRESHOLD_GB = 4.0  # Warn if less than 4GB RAM free
COOLDOWN_SECONDS = 5  # Pause between runs for cleanup (5s sufficient with subprocess isolation)
GPU_COOLDOWN_SECONDS = 3  # Extra pause for GPU memory release

# ═══════════════════════════════════════════════════════════════════════
# Logging Setup
# ═══════════════════════════════════════════════════════════════════════

def setup_logging(output_dir):
    """Setup dual logging: console + file."""
    log_dir = Path(output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"orchestrator_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.DEBUG)
    
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%H:%M:%S"
    ))
    logger.addHandler(ch)
    
    # File handler (detailed)
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s │ %(levelname)-7s │ %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)
    
    return logger, log_file


# ═══════════════════════════════════════════════════════════════════════
# System Monitoring
# ═══════════════════════════════════════════════════════════════════════

def get_system_stats():
    """Get current system resource usage."""
    mem = psutil.virtual_memory()
    stats = {
        "ram_used_gb": mem.used / (1024**3),
        "ram_total_gb": mem.total / (1024**3),
        "ram_available_gb": mem.available / (1024**3),
        "ram_percent": mem.percent,
        "cpu_percent": psutil.cpu_percent(interval=0.5),
    }
    
    # GPU stats (if nvidia-smi available)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            if len(parts) >= 4:
                stats["gpu_mem_used_mb"] = float(parts[0].strip())
                stats["gpu_mem_total_mb"] = float(parts[1].strip())
                stats["gpu_temp_c"] = float(parts[2].strip())
                stats["gpu_power_w"] = float(parts[3].strip())
    except Exception:
        pass
    
    return stats


def check_memory_safe(logger, threshold_gb=MEMORY_WARN_THRESHOLD_GB):
    """Check if there's enough memory to proceed."""
    stats = get_system_stats()
    available = stats["ram_available_gb"]
    
    if available < threshold_gb:
        logger.warning(
            f"⚠️  Low RAM: {available:.1f}GB available (threshold: {threshold_gb}GB). "
            f"Waiting for memory to free up..."
        )
        # Wait and retry
        for attempt in range(6):  # Wait up to 60 seconds
            time.sleep(10)
            stats = get_system_stats()
            available = stats["ram_available_gb"]
            if available >= threshold_gb:
                logger.info(f"✅ RAM recovered: {available:.1f}GB available")
                return True
            logger.warning(f"   Still waiting... {available:.1f}GB available (attempt {attempt+1}/6)")
        
        logger.error(f"❌ RAM still low after 60s: {available:.1f}GB. Proceeding anyway...")
        return False
    
    return True


# ═══════════════════════════════════════════════════════════════════════
# Run Management
# ═══════════════════════════════════════════════════════════════════════

def get_run_id(task, seed, lr=DEFAULT_LR):
    """Generate consistent run ID."""
    return f"{task}_s{seed}_lr{lr:.0e}"


def is_run_completed(output_dir, task, seed, lr=DEFAULT_LR):
    """Check if a run has already completed successfully."""
    run_id = get_run_id(task, seed, lr)
    results_file = Path(output_dir) / run_id / "results.json"
    
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
            # Verify it has actual results (not just an error)
            if "eval_metrics" in data and "error" not in data:
                return True
        except (json.JSONDecodeError, KeyError):
            pass
    
    return False


def run_single_experiment_subprocess(
    task, seed, lr, output_dir, use_wandb, wandb_project, wandb_group,
    max_samples, script_path, logger, run_idx, total_runs, timeout=7200
):
    """Run a single experiment in an isolated subprocess.
    
    Returns: (success: bool, result: dict or None, duration: float)
    """
    run_id = get_run_id(task, seed, lr)
    
    cmd = [
        sys.executable, script_path,
        "--mode", "custom",
        "--tasks", task,
        "--seeds", str(seed),
        "--lr", str(lr),
        "--output-dir", output_dir,
    ]
    
    if use_wandb:
        cmd.append("--wandb")
        cmd.extend(["--wandb-project", wandb_project])
        cmd.extend(["--wandb-group", wandb_group])
    
    if max_samples is not None:
        cmd.extend(["--max-samples", str(max_samples)])
    else:
        cmd.append("--unlimited")
    
    logger.info(f"🚀 [{run_idx}/{total_runs}] Starting: {run_id}")
    logger.debug(f"   Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Set environment to prevent torch compile issues
        env = os.environ.copy()
        env["TORCH_COMPILE_DISABLE"] = "1"
        env["TORCHDYNAMO_DISABLE"] = "1"
        env["WANDB_START_METHOD"] = "thread"
        env["WANDB_LOG_MODEL"] = "false"
        # Prevent CUDA caching from growing unbounded
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        
        # Stream output in real-time
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.rstrip()
            if line:
                output_lines.append(line)
                # Print key lines to console
                if any(kw in line for kw in ["Final Results", "ETA", "LERNA step", "FAILED", "Error", "energy"]):
                    logger.info(f"   {line}")
                elif "Eval metrics" in line or "Saved:" in line:
                    logger.info(f"   {line}")
        
        process.stdout.close()
        return_code = process.wait(timeout=timeout)
        duration = time.time() - start_time
        
        if return_code == 0:
            # Verify results file exists
            results_file = Path(output_dir) / run_id / "results.json"
            if results_file.exists():
                with open(results_file) as f:
                    result = json.load(f)
                if "error" not in result:
                    logger.info(
                        f"✅ [{run_idx}/{total_runs}] Completed: {run_id} "
                        f"in {timedelta(seconds=int(duration))} "
                    )
                    return True, result, duration
            
            logger.warning(f"⚠️  [{run_idx}/{total_runs}] {run_id}: Process exited OK but no valid results.json")
            return False, None, duration
        else:
            logger.error(
                f"❌ [{run_idx}/{total_runs}] {run_id}: Exit code {return_code} "
                f"after {timedelta(seconds=int(duration))}"
            )
            # Save last 20 lines of output for debugging
            if output_lines:
                logger.error(f"   Last output lines:")
                for line in output_lines[-20:]:
                    logger.error(f"   | {line}")
            return False, None, duration
    
    except subprocess.TimeoutExpired:
        logger.error(f"⏰ [{run_idx}/{total_runs}] {run_id}: TIMEOUT after {timeout}s")
        process.kill()
        process.wait()
        return False, None, time.time() - start_time
    
    except Exception as e:
        logger.error(f"💥 [{run_idx}/{total_runs}] {run_id}: Exception: {e}")
        return False, None, time.time() - start_time


# ═══════════════════════════════════════════════════════════════════════
# Summary Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_final_summary(output_dir, tasks, seeds, lr, logger):
    """Generate comprehensive final summary from all results."""
    all_results = []
    
    for task in tasks:
        for seed in seeds:
            run_id = get_run_id(task, seed, lr)
            results_file = Path(output_dir) / run_id / "results.json"
            if results_file.exists():
                try:
                    with open(results_file) as f:
                        data = json.load(f)
                    all_results.append(data)
                except Exception as e:
                    logger.warning(f"Could not read {results_file}: {e}")
    
    # Save combined results
    summary_path = Path(output_dir) / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary table
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    logger.info(f"\n{'='*80}")
    logger.info(f"  FINAL SUMMARY")
    logger.info(f"{'='*80}")
    logger.info(f"  Total runs: {len(all_results)} ({len(successful)} succeeded, {len(failed)} failed)")
    
    if successful:
        total_kwh = sum(r.get("energy_kwh", 0) for r in successful)
        total_time = sum(r.get("train_runtime_s", 0) for r in successful)
        logger.info(f"  Total energy: {total_kwh:.6f} kWh")
        logger.info(f"  Total compute: {total_time:.1f}s ({total_time/3600:.2f}h)")
        
        logger.info(f"\n  {'Task':<8} {'Seeds':>5} {'Mean':>10} {'Std':>8} {'95% CI':>20} {'kWh':>10} {'Waste':>8}")
        logger.info(f"  {'-'*80}")
        
        for task in tasks:
            task_results = [r for r in successful if r["task"] == task]
            if not task_results:
                continue
            
            accs = []
            for r in task_results:
                em = r.get("eval_metrics", {})
                acc = em.get("eval_accuracy", em.get("eval_matthews_correlation", em.get("eval_pearsonr", 0)))
                accs.append(acc)
            
            kwhs = [r.get("energy_kwh", 0) for r in task_results]
            wastes = [r.get("waste_metrics", {}).get("waste_ratio", 0) for r in task_results]
            
            n = len(accs)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            
            # 95% CI
            if n >= 2:
                from scipy import stats as scipy_stats
                ci = scipy_stats.t.interval(0.95, df=n-1, loc=mean_acc, scale=scipy_stats.sem(accs))
                ci_str = f"[{ci[0]:.4f}, {ci[1]:.4f}]"
            else:
                ci_str = f"[{mean_acc:.4f}, {mean_acc:.4f}]"
            
            logger.info(
                f"  {task:<8} {n:>5} {mean_acc:>10.4f} {std_acc:>8.4f} "
                f"{ci_str:>20} {np.mean(kwhs):>10.6f} {np.mean(wastes):>8.3f}"
            )
    
    if failed:
        logger.info(f"\n  Failed runs:")
        for r in failed:
            logger.info(f"    ❌ {r.get('task', '?')} seed={r.get('seed', '?')}: {r.get('error', 'unknown')}")
    
    logger.info(f"\n  Summary saved: {summary_path}")
    logger.info(f"{'='*80}")
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="LERNA Full Experiment Orchestrator")
    parser.add_argument("--output-dir", required=True, help="Output directory for all experiments")
    parser.add_argument("--num-seeds", type=int, default=10, help="Number of seeds (default: 10)")
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to run (default: all 8 GLUE tasks)")
    parser.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="lerna-baseline", help="W&B project name")
    parser.add_argument("--wandb-group", default=None, help="W&B group name")
    parser.add_argument("--max-samples", type=int, default=25000, help="Max samples per task (default: 25000, 0=unlimited)")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Max retries per failed run")
    parser.add_argument("--cooldown", type=int, default=COOLDOWN_SECONDS, help="Seconds between runs (default: 5)")
    parser.add_argument("--timeout", type=int, default=7200, help="Timeout per run in seconds (default: 2h)")
    parser.add_argument("--script", default="scripts/run_baseline_glue.py", help="Path to training script")
    args = parser.parse_args()
    
    # Setup
    tasks = args.tasks or GLUE_TASKS
    seeds = list(range(42, 42 + args.num_seeds))
    max_samples = None if args.max_samples == 0 else args.max_samples
    wandb_group = args.wandb_group or f"full-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    logger, log_file = setup_logging(args.output_dir)
    
    # Graceful shutdown handler
    shutdown_requested = [False]
    def signal_handler(sig, frame):
        if shutdown_requested[0]:
            logger.warning("Force quit!")
            sys.exit(1)
        shutdown_requested[0] = True
        logger.warning("\n⚠️  Shutdown requested. Finishing current run then stopping...")
        logger.warning("   Press Ctrl+C again to force quit.")
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Build run queue
    all_runs = [(task, seed) for task in tasks for seed in seeds]
    total_runs = len(all_runs)
    
    # Check which are already done
    completed = [(t, s) for t, s in all_runs if is_run_completed(args.output_dir, t, s, args.lr)]
    pending = [(t, s) for t, s in all_runs if not is_run_completed(args.output_dir, t, s, args.lr)]
    
    logger.info(f"")
    logger.info(f"═══════════════════════════════════════════════════════════")
    logger.info(f"  LERNA Full Experiment Orchestrator")
    logger.info(f"═══════════════════════════════════════════════════════════")
    logger.info(f"  Tasks:        {tasks}")
    logger.info(f"  Seeds:        {seeds[0]}-{seeds[-1]} ({len(seeds)} seeds)")
    logger.info(f"  Total runs:   {total_runs}")
    logger.info(f"  Completed:    {len(completed)} (will skip)")
    logger.info(f"  Pending:      {len(pending)}")
    logger.info(f"  Max retries:  {args.max_retries}")
    logger.info(f"  Cooldown:     {args.cooldown}s between runs")
    logger.info(f"  Timeout:      {args.timeout}s per run")
    logger.info(f"  Max samples:  {max_samples or 'unlimited'}")
    logger.info(f"  W&B:          {'enabled' if args.wandb else 'disabled'}")
    if args.wandb:
        logger.info(f"  W&B group:    {wandb_group}")
    logger.info(f"  Log file:     {log_file}")
    
    stats = get_system_stats()
    logger.info(f"  RAM:          {stats['ram_used_gb']:.1f}/{stats['ram_total_gb']:.1f} GB ({stats['ram_percent']}%)")
    if "gpu_mem_total_mb" in stats:
        logger.info(f"  GPU Memory:   {stats['gpu_mem_used_mb']:.0f}/{stats['gpu_mem_total_mb']:.0f} MB")
        logger.info(f"  GPU Temp:     {stats.get('gpu_temp_c', 'N/A')}°C")
    logger.info(f"═══════════════════════════════════════════════════════════\n")
    
    if not pending:
        logger.info("🎉 All runs already completed! Generating summary...")
        generate_final_summary(args.output_dir, tasks, seeds, args.lr, logger)
        return
    
    # ── Main execution loop ───────────────────────────────────────────
    overall_start = time.time()
    results_tracker = {
        "completed": list(completed),
        "failed": [],
        "durations": [],
    }
    
    run_counter = len(completed)
    
    for task, seed in pending:
        if shutdown_requested[0]:
            logger.warning("🛑 Shutdown requested. Stopping.")
            break
        
        run_counter += 1
        run_id = get_run_id(task, seed, args.lr)
        
        # Memory check
        check_memory_safe(logger)
        
        # ETA calculation
        if results_tracker["durations"]:
            avg_duration = np.mean(results_tracker["durations"])
            remaining_runs = total_runs - run_counter + 1
            eta = timedelta(seconds=int(avg_duration * remaining_runs))
            logger.info(f"📊 Progress: {run_counter}/{total_runs} | ETA: {eta}")
        
        # System stats
        stats = get_system_stats()
        logger.debug(
            f"System: RAM {stats['ram_available_gb']:.1f}GB free, "
            f"CPU {stats['cpu_percent']}%"
            + (f", GPU {stats['gpu_mem_used_mb']:.0f}MB, {stats.get('gpu_temp_c', '?')}°C"
               if "gpu_mem_used_mb" in stats else "")
        )
        
        # Retry loop
        success = False
        for attempt in range(1, args.max_retries + 1):
            if shutdown_requested[0]:
                break
            
            if attempt > 1:
                wait_time = args.cooldown * attempt  # Exponential backoff
                logger.info(f"🔄 Retry {attempt}/{args.max_retries} for {run_id} (waiting {wait_time}s)")
                time.sleep(wait_time)
            
            ok, result, duration = run_single_experiment_subprocess(
                task=task,
                seed=seed,
                lr=args.lr,
                output_dir=args.output_dir,
                use_wandb=args.wandb,
                wandb_project=args.wandb_project,
                wandb_group=wandb_group,
                max_samples=max_samples,
                script_path=args.script,
                logger=logger,
                run_idx=run_counter,
                total_runs=total_runs,
                timeout=args.timeout,
            )
            
            if ok:
                success = True
                results_tracker["completed"].append((task, seed))
                results_tracker["durations"].append(duration)
                break
            else:
                logger.warning(f"   Attempt {attempt}/{args.max_retries} failed for {run_id}")
        
        if not success:
            logger.error(f"💀 {run_id}: All {args.max_retries} attempts failed!")
            results_tracker["failed"].append((task, seed))
        
        # Cooldown between runs
        if not shutdown_requested[0]:
            logger.debug(f"💤 Cooldown: {args.cooldown}s")
            time.sleep(args.cooldown)
    
    # ── Final Summary ─────────────────────────────────────────────────
    total_elapsed = time.time() - overall_start
    
    logger.info(f"\n{'='*80}")
    logger.info(f"  ORCHESTRATOR COMPLETE")
    logger.info(f"  Wall time: {timedelta(seconds=int(total_elapsed))}")
    logger.info(f"  Completed: {len(results_tracker['completed'])}/{total_runs}")
    logger.info(f"  Failed:    {len(results_tracker['failed'])}")
    if results_tracker["failed"]:
        logger.info(f"  Failed runs: {results_tracker['failed']}")
    logger.info(f"{'='*80}")
    
    # Generate final summary
    generate_final_summary(args.output_dir, tasks, seeds, args.lr, logger)
    
    # Save orchestrator metadata
    meta_path = Path(args.output_dir) / "orchestrator_meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "total_runs": total_runs,
            "completed": len(results_tracker["completed"]),
            "failed": len(results_tracker["failed"]),
            "failed_runs": [(t, s) for t, s in results_tracker["failed"]],
            "wall_time_s": total_elapsed,
            "avg_run_duration_s": float(np.mean(results_tracker["durations"])) if results_tracker["durations"] else 0,
            "cooldown_s": args.cooldown,
            "timestamp": datetime.now().isoformat(),
        }, f, indent=2)
    
    logger.info(f"  Orchestrator metadata: {meta_path}")
    logger.info(f"  Full log: {log_file}")


if __name__ == "__main__":
    main()
