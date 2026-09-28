#!/usr/bin/env python3
"""Gate 3: QNLI confirmatory run (4 arms x 5 seeds x rate 0.30).

Pre-registered: see /new_raid/nanhangproj/experiments/gate3_qnli_pre_registration.md

Arms: full_finetune, exact_random, ler_guided_stratified, phase_strat_guarded
Rate: 0.30 only
Seeds: [11, 13, 17, 19, 23] (5 pre-registered; reserves 29, 31; replace if >2 fail)
Config: 2 epochs, batch 32, fp16+eager, max_grad_norm=1.0, lr=2e-5, warmup=0.06
Hypothesis: paired Δacc(lgs - exact_random) != 0, two-sided, alpha=0.05
"""

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/raid/hf_cache")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_ENDPOINT", None)

import sys
import json
import glob
import subprocess
import time
import shutil
import traceback
from pathlib import Path

import torch
torch.cuda.set_device(0)

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerna.utils.phase1_3_operations import check_forward_loss_numerics
from lerna.utils import ettin_constants as ETTIN
from scripts.run_ablation_study import (
    plan_phase1_3_cell,
    run_ablation_single,
    ABLATIONS,
    ONLINE_LER_MODE_AUTO,
    CLASSIFICATION_MATCHED_CLAIM,
    SchedulerStepPolicy,
    resolve_task_data_facts,
    TASK_HP_OVERRIDES,
)
from scripts.run_baseline_glue import (
    detect_device_profile,
    get_training_config,
    TASK_HP_OVERRIDES as RGB_TASK_HP_OVERRIDES,
)
from lerna.utils.model_loader import load_tokenizer

CONFIRMATORY_ARMS = [
    "full_finetune",
    "exact_random",
    "ler_guided_stratified",
    "phase_strat_guarded",
]
CONFIRMATORY_SEEDS = [11, 13, 17, 19, 23]
CONFIRMATORY_SEED_RESERVES = [29, 31]
TARGET_SKIP_RATE = 0.30

QNLI_EPOCHS = 2
QNLI_LR = 2e-5
QNLI_WARMUP = 0.06


def resolve_git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


def is_git_dirty():
    result = subprocess.run(
        ["git", "diff", "--quiet", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    return result.returncode != 0


def patch_qnli():
    """Override QNLI to 2 epochs, lr=2e-5 per pre-registration (decided before running)."""
    original = TASK_HP_OVERRIDES.get("qnli", {})
    TASK_HP_OVERRIDES["qnli"] = {
        "learning_rate": QNLI_LR,
        "num_epochs": QNLI_EPOCHS,
        "warmup_ratio": QNLI_WARMUP,
        "early_stopping_patience": 8,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
    }
    RGB_TASK_HP_OVERRIDES["qnli"] = TASK_HP_OVERRIDES["qnli"]
    return original


def build_environment(profile):
    hw = get_training_config(profile)
    return {
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
        "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", "1"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "false"),
        "HF_HOME": os.environ.get("HF_HOME", "/raid/hf_cache"),
        "dataloader_num_workers": hw["dataloader_num_workers"],
        "torch_dtype": "float16" if hw["fp16"] else "float32",
        "model_id": ETTIN.ETTIN_MODEL_ID,
        "model_revision": ETTIN.ETTIN_REVISION,
        "dataset": "glue/qnli",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_grad_norm": hw.get("max_grad_norm", 1.0),
        "learning_rate": QNLI_LR,
        "attn_implementation": "eager",
        "num_epochs": QNLI_EPOCHS,
    }


def build_plan(base_output_dir, git_sha, profile):
    tokenizer = load_tokenizer(
        ETTIN.ETTIN_MODEL_ID, revision=ETTIN.ETTIN_REVISION, local_files_only=True,
    )
    data_facts_provider = lambda task: resolve_task_data_facts(task, tokenizer, None, profile)
    hw_config = dict(get_training_config(profile))
    qnli_facts = data_facts_provider("qnli")

    plan = []
    for seed in CONFIRMATORY_SEEDS:
        for arm in CONFIRMATORY_ARMS:
            cell = plan_phase1_3_cell(
                task_name="qnli",
                training_seed=seed,
                policy_seed=seed,
                ablation_name=arm,
                target_skip_rate=TARGET_SKIP_RATE,
                model_name=ETTIN.ETTIN_MODEL_ID,
                model_revision=ETTIN.ETTIN_REVISION,
                data_facts=qnli_facts,
                git_sha=git_sha,
                base_output_dir=base_output_dir,
                scheduler_step_policy=SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP,
                max_consecutive_skips=4,
                probe_interval=8,
                rho_veto_threshold=-0.2,
                risk_gamma=0.0,
                online_ler_mode=ONLINE_LER_MODE_AUTO,
                online_ler_parameter_sample_size=4096,
                online_ler_update_interval=1,
                use_rho_vg=True,
                use_safety_horizon=True,
                provenance_classification=CLASSIFICATION_MATCHED_CLAIM,
                hw_cfg=hw_config,
                lr=QNLI_LR,
                warmup_ratio=QNLI_WARMUP,
            )
            plan.append(cell)
    return plan, qnli_facts, hw_config


def find_completed_attempt(base_output_dir, arm, fingerprint):
    """Return (attempt_name, results_path) for the first completed attempt, or None."""
    arm_dir = Path(base_output_dir) / arm / fingerprint
    if not arm_dir.exists():
        return None
    for attempt_dir in sorted(arm_dir.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt-"):
            continue
        rpath = attempt_dir / "results.json"
        ipath = attempt_dir / "instrumentation.json"
        if rpath.exists() and ipath.exists():
            return attempt_dir.name, rpath
    return None


def validate_cell(cell, base_output_dir):
    arm = cell["arm"]
    fingerprint = cell["fingerprint"]
    completed = find_completed_attempt(base_output_dir, arm, fingerprint)
    if completed is None:
        return False, f"results.json not found in any attempt"
    _, results_path = completed
    with open(results_path) as f:
        results = json.load(f)
    errors = []
    try:
        total_steps = results.get("controller_config", {}).get("configured_total_steps", 6548)
        check_forward_loss_numerics(results, expected_observations=total_steps)
    except Exception as exc:
        errors.append(f"forward_loss_numerics: {exc}")
    instr = results.get("true_skip_instrumentation", {})
    if instr.get("eval_seconds", 0.0) <= 0.0:
        errors.append(f"eval_seconds={instr.get('eval_seconds')}")
    if instr.get("mean_step_ms_executed") is None:
        errors.append("mean_step_ms_executed is None")
    fwd = instr.get("forward_calls", 0)
    bck = instr.get("backward_calls", 0)
    skp = instr.get("skipped_backward_steps", 0)
    if fwd != bck + skp:
        errors.append(f"invariant: forward({fwd}) != backward({bck}) + skipped({skp})")
    return (len(errors) == 0), "; ".join(errors)


def discover_existing_cells(base_output_dir):
    """Discover all valid cells in the output directory by scanning manifests.

    Used in --analyze-only mode where the current git_sha differs from the
    frozen SHA the cells were run at.  Returns a list of cell dicts compatible
    with the plan format.
    """
    cells = []
    for arm in CONFIRMATORY_ARMS:
        arm_dir = Path(base_output_dir) / arm
        if not arm_dir.exists():
            continue
        for fp_dir in sorted(arm_dir.iterdir()):
            if not fp_dir.is_dir() or len(fp_dir.name) != 16:
                continue
            manifest = glob.glob(str(fp_dir / "attempt-001" / "run_manifest.json"))
            if not manifest:
                continue
            with open(manifest[0]) as f:
                m = json.load(f)
            seed = m.get("identity_inputs", {}).get("training_seed")
            if seed is None or seed not in CONFIRMATORY_SEEDS:
                continue
            cells.append(
                {
                    "arm": arm,
                    "fingerprint": fp_dir.name,
                    "training_seed": seed,
                    "model_id": arm,
                    "model_revision": "",
                }
            )
    return cells


def collect_cell_metrics(base_output_dir, arm, fingerprint):
    """Collect metrics for a specific cell (arm + fingerprint).
    
    The fingerprint includes seed, so passing it ensures we read the correct
    results.json rather than matching any arbitrary attempt for the arm.
    """
    completed = find_completed_attempt(base_output_dir, arm, fingerprint)
    if completed is None:
        return None
    _, rpath = completed
    d = json.load(open(rpath))
    instr = d.get("true_skip_instrumentation", {})
    return {
        "acc": d["eval_metrics"]["eval_accuracy"],
        "f1": d["eval_metrics"].get("eval_f1", None),
        "mean_step_ms_executed": instr.get("mean_step_ms_executed"),
        "mean_step_ms_skipped": instr.get("mean_step_ms_skipped"),
        "train_loop_seconds": instr.get("train_loop_seconds"),
        "eval_seconds": instr.get("eval_seconds"),
        "avg_dataloader_wait_ms": instr.get("avg_dataloader_wait_ms"),
        "forward_calls": instr.get("forward_calls"),
        "backward_calls": instr.get("backward_calls"),
        "skipped_backward_steps": instr.get("skipped_backward_steps"),
        "energy_kwh": d.get("energy_kwh"),
        "energy_valid": d.get("power_evidence", {}).get("energy_valid", False),
        "skip_ratio_by_batch": d.get("skip_ratio_by_batch"),
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gate 3: QNLI confirmatory run")
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip preflight checks and just re-analyze cached results (no new runs)",
    )
    args = parser.parse_args()

    base_output_dir = os.environ.get(
        "GATE3_OUTPUT_DIR",
        "/new_raid/nanhangproj/experiments/gate3_qnli_confirmatory",
    )
    git_sha = resolve_git_sha()
    profile = detect_device_profile()

    print("=" * 60)
    print("Gate 3: QNLI Confirmatory Run")
    print(f"  Git SHA: {git_sha}")
    print(f"  Profile: {profile}")
    print(f"  Output: {base_output_dir}")
    print(f"  Arms: {CONFIRMATORY_ARMS}")
    print(f"  Max seeds: {CONFIRMATORY_SEEDS}")
    print(f"  Patches: QNLI epochs={QNLI_EPOCHS} lr={QNLI_LR} warmup={QNLI_WARMUP}")
    if args.analyze_only:
        print(f"  Mode: ANALYZE ONLY (no preflight, no new runs)")
    print("=" * 60)

    orig = patch_qnli()
    env = build_environment(profile)
    print(f"  Environment: {json.dumps(env, indent=2)}")

    # Preflight checks
    preflight_skipped = False
    if not args.analyze_only:
        preflight_errors = []
        # 1. Clean git tree
        if is_git_dirty():
            preflight_errors.append("git tree is dirty (commit or stash changes first)")
        # 2. Disk space check (>=100 GB free)
        disk = shutil.disk_usage("/new_raid/nanhangproj")
        if disk.free < 100 * 1024 ** 3:
            preflight_errors.append(
                f"insufficient disk: {disk.free / 1024 ** 3:.0f} GB free (need 100 GB)"
            )
        # 3. No foreign-SHA cells in output dir
        for arm_dir in Path(base_output_dir).iterdir():
            if arm_dir.is_dir():
                for fp_dir in arm_dir.iterdir():
                    if fp_dir.is_dir():
                        for attempt_dir in fp_dir.iterdir():
                            if attempt_dir.is_dir() and attempt_dir.name.startswith(
                                "attempt-"
                            ):
                                manifest = attempt_dir / "run_manifest.json"
                                if manifest.exists():
                                    try:
                                        m = json.load(open(manifest))
                                        m_sha = m.get("git_sha", "")
                                        if m_sha != git_sha:
                                            preflight_errors.append(
                                                f"foreign SHA cell: {manifest} has git_sha={m_sha[:12]}, expected={git_sha[:12]}"
                                            )
                                    except json.JSONDecodeError:
                                        preflight_errors.append(
                                            f"corrupt manifest: {manifest}"
                                        )
        if preflight_errors:
            print("\n  PREFLIGHT CHECKS FAILED:")
            for err in preflight_errors:
                print(f"    - {err}", flush=True)
            sys.exit(2)
        print("  Preflight: clean git, disk OK, no foreign-SHA cells")
    else:
        preflight_skipped = True
        print("  Preflight: SKIPPED (analyze-only mode)")

    # In analyze-only mode, discover existing cells from manifests rather
    # than building a plan whose fingerprints depend on the current git_sha.
    if args.analyze_only:
        plan = discover_existing_cells(base_output_dir)
        print(f"\nDiscovered {len(plan)} existing cells (analyze-only)")
        for cell in plan:
            print(f"  [{cell['arm']}] seed={cell['training_seed']} fp={cell['fingerprint'][:8]}")
        qnli_facts = {
            "num_epochs": QNLI_EPOCHS,
            "total_steps": 6548,
            "train_samples_realized": 104743,
            "eval_samples_realized": 5463,
        }
        hw_config = {}
    else:
        plan, qnli_facts, hw_config = build_plan(base_output_dir, git_sha, profile)
        print(f"\nPlan: {len(plan)} cells (4 arms x up to {len(CONFIRMATORY_SEEDS)} seeds)")
        print(f"  QNLI: {qnli_facts['num_epochs']} epochs, {qnli_facts['total_steps']} total_steps, "
              f"{qnli_facts['train_samples_realized']} train, {qnli_facts['eval_samples_realized']} eval")

    completed_seeds = []
    failed_seeds = []
    fp16_nan_count = 0
    all_cell_metrics = {}

    for seed in CONFIRMATORY_SEEDS:
        seed_cells = [c for c in plan if c["training_seed"] == seed]
        seed_all_valid = True
        seed_results = []

        print(f"\n{'='*60}")
        print(f"  Seed {seed}: 4 arms")
        print(f"{'='*60}")

        for cell in seed_cells:
            arm = cell["arm"]
            fingerprint = cell["fingerprint"]
            t0 = time.perf_counter()
            completed = find_completed_attempt(base_output_dir, arm, fingerprint)
            if completed is not None:
                attempt_name, _ = completed
                print(f"\n  [{arm}] seed={seed} SKIP (already complete: {attempt_name})", flush=True)
                t1 = time.perf_counter()
                valid, msg = validate_cell(cell, base_output_dir)
                metrics = collect_cell_metrics(base_output_dir, arm, fingerprint)
                all_cell_metrics[(seed, arm)] = metrics
                if valid:
                    seed_results.append((arm, True, ""))
                    print(f"  [{arm}] PASS (cached) | acc={metrics['acc']:.4f} | step={metrics['mean_step_ms_executed']:.1f}ms | eval_s={metrics['eval_seconds']:.1f}", flush=True)
                else:
                    seed_all_valid = False
                    seed_results.append((arm, False, msg))
                    print(f"  [{arm}] FAIL (cached): {msg}", flush=True)
                continue
            print(f"\n  [{arm}] seed={seed} starting...", flush=True)
            if args.analyze_only:
                print(f"  [{arm}] SKIP (analyze-only mode: no execution)", flush=True)
                seed_all_valid = False
                seed_results.append((arm, False, "analyze-only mode: cell not found for re-execution"))
                continue
            try:
                run_ablation_single(
                    task_name="qnli",
                    seed=seed,
                    ablation_name=arm,
                    ablation_overrides=ABLATIONS[arm],
                    model_name=cell["model_id"],
                    profile=profile,
                    base_output_dir=base_output_dir,
                    use_wandb=False,
                    num_epochs=QNLI_EPOCHS,
                    warmup_ratio=QNLI_WARMUP,
                    metric_for_best_model="eval_accuracy",
                    greater_is_better=True,
                    no_early_stopping=True,
                    target_skip_rate=TARGET_SKIP_RATE,
                    max_consecutive_skips=cell["max_consecutive_skips"],
                    probe_interval=cell["probe_interval"],
                    policy="hybrid",
                    rho_veto_threshold=cell["rho_veto_threshold"],
                    risk_gamma=cell["risk_gamma"],
                    skip_update_mode="freeze",
                    scheduler_step_policy=SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP,
                    allow_early_stopping_with_skipping=False,
                    rvd_veto_mode="none",
                    online_ler_mode=ONLINE_LER_MODE_AUTO,
                    online_ler_parameter_sample_size=cell["online_ler_parameter_sample_size"],
                    online_ler_update_interval=cell["online_ler_update_interval"],
                    planned_cell=cell,
                    model_revision=cell["model_revision"],
                )
            except Exception as exc:
                t1 = time.perf_counter()
                err_str = str(exc).lower()
                if "nonfinite" in err_str or "nan" in err_str:
                    fp16_nan_count += 1
                print(f"  [{arm}] FAILED in {t1-t0:.1f}s: {type(exc).__name__}: {exc}", flush=True)
                traceback.print_exc()
                seed_all_valid = False
                seed_results.append((arm, False, str(exc)))
                continue

            t1 = time.perf_counter()
            valid, msg = validate_cell(cell, base_output_dir)
            metrics = collect_cell_metrics(base_output_dir, arm, fingerprint)
            all_cell_metrics[(seed, arm)] = metrics

            f1 = metrics["acc"] if metrics else "?"
            ms = metrics["mean_step_ms_executed"] if metrics else 0
            print(f"  [{arm}] {'PASS' if valid else 'FAIL'} in {t1-t0:.0f}s | "
                  f"acc={f1:.4f} | step={ms:.1f}ms | eval_s={metrics['eval_seconds']:.1f}" if metrics and valid else
                  f"  [{arm}] FAIL in {t1-t0:.0f}s: {msg}", flush=True)

            if valid:
                seed_results.append((arm, True, ""))
            else:
                seed_all_valid = False
                seed_results.append((arm, False, msg))

        if seed_all_valid and len(seed_results) == 4:
            completed_seeds.append(seed)
            print(f"\n  Seed {seed}: COMPLETE")
        else:
            failed_seeds.append(seed)
            print(f"\n  Seed {seed}: INCOMPLETE")

        if len(failed_seeds) > 2:
            print(f"\n  >2 seeds failed ({len(failed_seeds)}).", flush=True)
            if fp16_nan_count > 0:
                print("  fp16 NaN detected. STOPPING. Consider FP32 re-run.", flush=True)
            break

        # Early check: if first completed seed has GPU busy < 85%, report host-bound
        if len(completed_seeds) >= 1:
            for arm in CONFIRMATORY_ARMS:
                m = all_cell_metrics.get((completed_seeds[0], arm))
                if m and m["mean_step_ms_executed"]:
                    # Quick estimate
                    pass

    # Summary
    print("\n" + "=" * 60)
    print("GATE 3 SUMMARY")
    print("=" * 60)
    print(f"  Completed seeds: {len(completed_seeds)}/{len(CONFIRMATORY_SEEDS)}")
    print(f"  Failed seeds: {len(failed_seeds)}")
    print(f"  fp16 NaN events: {fp16_nan_count}")
    if completed_seeds:
        print(f"  Completed: {completed_seeds}")
    if failed_seeds:
        print(f"  Failed: {failed_seeds}")

    # Validator rule: identical accuracy across >=2 seeds for the same arm => invalid
    # Triggers if ANY two seeds produce the same accuracy (statistical coincidence
    # or seed propagation bug). Investigated: ler_guided_stratified seeds 11 & 13
    # both yield 0.9322716456159619 despite different train_loss/eval_loss.
    seed_identity_violation = False
    for arm in CONFIRMATORY_ARMS:
        accs = []
        for s in completed_seeds:
            m = all_cell_metrics.get((s, arm))
            if m and m.get("acc") is not None:
                accs.append((s, m["acc"]))
        if len(accs) >= 2:
            vals = [a for _, a in accs]
            if len(set(vals)) < len(vals):
                seed_identity_violation = True
                dup_seeds = [str(s) for s, a in accs if vals.count(a) > 1]
                print(f"\n  VALIDATOR RULE VIOLATION: {arm} has identical acc across seeds [{', '.join(dup_seeds)}]")
    if seed_identity_violation:
        print("\n  MATRIX INVALIDATED: identical results across seeds detected")
        print("  (Likely cause: seed not propagated to trainer, or fingerprint collision)")
        matrix_invalidated = True
    else:
        matrix_invalidated = False

    # Paired analysis (only if >= 2 completed seeds)
    if len(completed_seeds) >= 2:
        print("\n--- Paired Analysis ---")
        print(f"\n  {'Arm':25s} | " + " | ".join(f"s{s}" for s in completed_seeds))
        print("  " + "-" * 70)
        for arm in CONFIRMATORY_ARMS:
            vals = []
            for s in completed_seeds:
                m = all_cell_metrics.get((s, arm))
                if m:
                    vals.append(f"{m['acc']:.4f}")
                else:
                    vals.append("N/A")
            print(f"  {arm:25s} | " + " | ".join(vals))

        # Primary: paired t-test on Δacc(lgs - exact_random)
        import math
        deltas = []
        for s in completed_seeds:
            lgs = all_cell_metrics.get((s, "ler_guided_stratified"))
            rnd = all_cell_metrics.get((s, "exact_random"))
            if lgs and rnd:
                deltas.append(lgs["acc"] - rnd["acc"])

        if len(deltas) >= 2:
            n = len(deltas)
            mean_d = sum(deltas) / n
            se = math.sqrt(sum(d**2 for d in deltas) / n - mean_d**2) / math.sqrt(n)
            t_stat = mean_d / se if se > 0 else float('inf')
            t_crit = 3.182 if n == 4 else (2.776 if n == 5 else 2.447)  # t(n-1, 0.025)
            print(f"\n  Primary: Δacc(lgs − exact_random) per seed:")
            print(f"    " + " + ".join(f"{d:+.4f}" for d in deltas))
            print(f"    Mean Δ: {mean_d:+.6f} | SE: {se:.6f} | t({n-1}): {t_stat:.3f} | crit: {t_crit}")
            sig = "SIGNIFICANT" if abs(t_stat) > t_crit else "NULL (not significant)"
            print(f"    Result: {sig} at α=0.05 (two-sided)")

            # Secondary: lgs vs full_finetune non-inferiority
            deltas_ni = []
            for s in completed_seeds:
                lgs = all_cell_metrics.get((s, "ler_guided_stratified"))
                ftf = all_cell_metrics.get((s, "full_finetune"))
                if lgs and ftf:
                    deltas_ni.append(lgs["acc"] - ftf["acc"])
            if deltas_ni:
                mean_ni = sum(deltas_ni) / len(deltas_ni)
                ci_lower = mean_ni - 2.776 * se  # approximate
                print(f"\n  Secondary: Δacc(lgs − full_finetune): {mean_ni:+.6f}")
                print(f"    Non-inferiority (margin 0.5pt): {'PASS' if ci_lower >= -0.005 else 'FAIL'}")

            # Overhead: mean_step_ms_executed(lgs) vs exact_random
            ov_lgs = [all_cell_metrics[(s, "ler_guided_stratified")] for s in completed_seeds]
            ov_rnd = [all_cell_metrics[(s, "exact_random")] for s in completed_seeds]
            ov_lgs_ms = [m["mean_step_ms_executed"] for m in ov_lgs if m]
            ov_rnd_ms = [m["mean_step_ms_executed"] for m in ov_rnd if m]
            if ov_lgs_ms and ov_rnd_ms:
                mean_lgs = sum(ov_lgs_ms) / len(ov_lgs_ms)
                mean_rnd = sum(ov_rnd_ms) / len(ov_rnd_ms)
                print(f"\n  Overhead: mean_step_ms_executed")
                print(f"    lgs: {mean_lgs:.1f} ms | exact_random: {mean_rnd:.1f} ms")
                print(f"    Δ: {mean_lgs - mean_rnd:+.1f} ms per executed step")

    summary = {
        "gate": 3,
        "task": "qnli",
        "plan_sha": git_sha,
        "provenance_classification": "matched_claim",
        "arms": CONFIRMATORY_ARMS,
        "max_seeds": CONFIRMATORY_SEEDS,
        "completed_seeds": completed_seeds,
        "failed_seeds": failed_seeds,
        "target_skip_rate": TARGET_SKIP_RATE,
        "num_epochs": QNLI_EPOCHS,
        "training_config": env,
        "n_completed_seeds": len(completed_seeds),
        "n_validated_cells": len(completed_seeds) * 4,
        "total_cells_planned": len(plan),
        "fp16_nan_events": fp16_nan_count,
        "matrix_invalidated": matrix_invalidated,
    }
    os.makedirs(base_output_dir, exist_ok=True)
    summary_path = Path(base_output_dir) / "gate3_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Summary saved: {summary_path}")

    if len(completed_seeds) < 2:
        print("\n  INSUFFICIENT DATA for paired analysis")
        sys.exit(1)


if __name__ == "__main__":
    main()
