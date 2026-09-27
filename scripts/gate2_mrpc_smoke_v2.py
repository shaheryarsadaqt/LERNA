#!/usr/bin/env python3
"""Gate 2: MRPC smoke test on tracker v3 (6 arms x rate 0.30 x seed 42).

Builds a 6-cell plan using build_phase1_3_matrix_plan with a single skip
rate (0.30) and single seed (42), persists it as a fresh plan, runs all
6 cells sequentially on a single V100, then validates each cell's
results.json for the M3 checks.

Usage:
    CUDA_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \\
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \\
    python scripts/gate2_mrpc_smoke_v2.py \\
        --output-dir /new_raid/nanhangproj/experiments/gate2_mrpc_v2
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

import sys
import json
import subprocess
import time
from pathlib import Path

import torch
torch.cuda.set_device(0)

# Ensure the repo root is on sys.path so scripts.* imports work
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from lerna.utils.phase1_3_matrix import PHASE1_3_CANONICAL_ARMS
from lerna.utils.phase1_3_operations import (
    check_forward_loss_numerics,
    file_sha256,
)
from lerna.utils import ettin_constants as ETTIN
from scripts.run_ablation_study import (
    build_phase1_3_matrix_plan,
    run_ablation_single,
    ABLATIONS,
    ONLINE_LER_MODE_AUTO,
    CLASSIFICATION_MATCHED_CLAIM,
    SchedulerStepPolicy,
    resolve_task_data_facts,
)
from scripts.run_baseline_glue import (
    detect_device_profile,
    get_training_config,
)
from lerna.utils.model_loader import load_tokenizer


def resolve_git_sha():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()


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
        "dataset": "glue/mrpc",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "max_grad_norm": hw["max_grad_norm"],
        "learning_rate": hw.get("learning_rate", 2e-5),
        "attn_implementation": "eager",
    }


def plan_smoke(base_output_dir, git_sha, profile):
    """Build 6-arm x 0.30 x seed 42 plan and persist."""
    tokenizer = load_tokenizer(
        ETTIN.ETTIN_MODEL_ID,
        revision=ETTIN.ETTIN_REVISION,
        local_files_only=True,
    )
    resolved_facts = {}
    def data_facts_provider(task):
        facts = resolve_task_data_facts(task, tokenizer, None, profile)
        resolved_facts[task] = facts
        return facts

    hw_config = dict(get_training_config(profile))

    plan = build_phase1_3_matrix_plan(
        tasks=["mrpc"],
        seeds=[42],
        target_skip_rates=[0.30],
        model_name=ETTIN.ETTIN_MODEL_ID,
        model_revision=ETTIN.ETTIN_REVISION,
        base_output_dir=base_output_dir,
        data_facts_provider=data_facts_provider,
        git_sha=git_sha,
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
        lr=2e-5,
        warmup_ratio=0.06,
    )

    # Validate all 6 arms present
    arms_in_plan = [c["arm"] for c in plan]
    assert set(arms_in_plan) == set(PHASE1_3_CANONICAL_ARMS), (
        f"Expected all 6 canonical arms, got {arms_in_plan}"
    )
    assert len(plan) == 6, f"Expected 6 cells, got {len(plan)}"
    assert all(c["target_skip_rate"] == 0.30 for c in plan), "All cells must have rate 0.30"
    assert all(c["training_seed"] == 42 for c in plan), "All cells must have seed 42"

    return plan, resolved_facts["mrpc"], hw_config


def run_cell(cell, base_output_dir, profile, num_epochs, warmup_ratio):
    """Run a single cell and return (success, error_msg)."""
    arm = cell["arm"]
    task = cell["task"]
    seed = cell["training_seed"]
    target_skip_rate = cell["target_skip_rate"]

    print(f"\n{'='*60}")
    print(f"  Cell: {arm} | {task} | seed={seed} | rate={target_skip_rate}")
    print(f"{'='*60}")

    start = time.perf_counter()
    try:
        run_ablation_single(
            task_name=task,
            seed=seed,
            ablation_name=arm,
            ablation_overrides=ABLATIONS[arm],
            model_name=cell["model_id"],
            profile=profile,
            base_output_dir=base_output_dir,
            use_wandb=False,
            num_epochs=num_epochs,
            warmup_ratio=warmup_ratio,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            no_early_stopping=True,
            target_skip_rate=target_skip_rate,
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
        elapsed = time.perf_counter() - start
        print(f"  Completed in {elapsed:.1f}s")
        return True, ""
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return False, f"{type(exc).__name__}: {exc}"


def validate_cell(cell, base_output_dir):
    """Validate a completed cell result against M3 checks."""
    arm = cell["arm"]
    arm_dir = Path(base_output_dir) / arm

    # Scan for results.json (fingerprint dir is computed at runtime by run_ablation_single)
    results_path = None
    manifest_path = None
    if arm_dir.exists():
        for fp_dir in sorted(arm_dir.iterdir()):
            if fp_dir.is_dir() and len(fp_dir.name) == 16:
                candidate = fp_dir / "attempt-001" / "results.json"
                cand_manifest = fp_dir / "attempt-001" / "run_manifest.json"
                if candidate.exists():
                    results_path = candidate
                    manifest_path = cand_manifest
                    break

    if results_path is None:
        return False, "results.json not found"

    with open(results_path) as f:
        results = json.load(f)

    errors = []

    # M2c: forward-loss numerics (use cell's total_steps, not hard-coded 575)
    total_steps = cell.get("total_steps", 575)
    try:
        check_forward_loss_numerics(results, expected_observations=total_steps)
    except Exception as exc:
        errors.append(f"forward_loss_numerics: {exc}")

    instr = results.get("true_skip_instrumentation", {})

    # eval_seconds > 0 (the timing bug fix)
    eval_s = instr.get("eval_seconds", 0.0)
    if eval_s <= 0.0:
        errors.append(f"eval_seconds={eval_s} (must be > 0)")

    # mean_step_ms_executed populated (CUDA-event-based)
    ms_exec = instr.get("mean_step_ms_executed")
    if ms_exec is not None:
        print(f"  CUDA mean_step_ms_executed: {ms_exec:.1f} ms")
    else:
        errors.append("mean_step_ms_executed is None")

    # Power evidence coverage >= 0.9
    pe = results.get("power_evidence", {})
    coverage = pe.get("coverage")
    if coverage is None or coverage < 0.9:
        errors.append(f"power_evidence.coverage={coverage} (must be >= 0.9)")

    # git_sha in manifest
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        if not manifest.get("git_sha"):
            errors.append("manifest missing git_sha")
        if not manifest.get("tracker_revision"):
            errors.append("manifest missing tracker_revision")
        if manifest.get("nonfinite_forward_loss_policy") != "fail":
            errors.append(
                f"manifest nonfinite_forward_loss_policy="
                f"{manifest.get('nonfinite_forward_loss_policy')} (expected 'fail')"
            )
    else:
        errors.append("run_manifest.json not found")

    # LER overhead check: backward_calls > 0 implies executed steps
    backward_calls = instr.get("backward_calls", 0)
    batches_seen = instr.get("batches_seen", 0)
    skipped = instr.get("skipped_backward_steps", 0)
    forward = instr.get("forward_calls", 0)

    # Invariant: forward == backward + skipped
    if forward != backward_calls + skipped:
        errors.append(
            f"invariant violated: forward({forward}) != "
            f"backward({backward_calls}) + skipped({skipped})"
        )

    if errors:
        return False, "; ".join(errors)
    return True, ""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Gate 2: MRPC smoke v2")
    parser.add_argument("--output-dir", default="/new_raid/nanhangproj/experiments/gate2_mrpc_v2")
    args = parser.parse_args()

    base_output_dir = args.output_dir
    git_sha = resolve_git_sha()
    profile = detect_device_profile()
    print(f"Gate 2: MRPC smoke test on tracker v3")
    print(f"  Git SHA: {git_sha}")
    print(f"  Profile: {profile}")
    print(f"  Output: {base_output_dir}")

    # Build plan
    plan, mrpc_facts, hw_config = plan_smoke(
        base_output_dir, git_sha, profile
    )
    print(f"\nPlan: {len(plan)} cells (6 arms x rate 0.30 x seed 42)")

    # Run each cell
    results = []
    for cell in plan:
        arm = cell["arm"]
        success, err = run_cell(
            cell, base_output_dir, profile,
            num_epochs=mrpc_facts["num_epochs"],
            warmup_ratio=0.06,
        )
        valid = False
        msg = err if not success else ""
        if success:
            valid, msg = validate_cell(cell, base_output_dir)
        results.append((arm, success, valid, msg))

    # Summary
    print("\n=== SMOKE TEST RESULTS ===")
    all_pass = True
    for arm, ran, valid, msg in results:
        status = "PASS" if (ran and valid) else "FAIL"
        if not (ran and valid):
            all_pass = False
        print(f"  {arm:30s} {status}  {msg}")

    n_completed = sum(1 for _, ran, _, _ in results if ran)
    n_validated = sum(1 for _, ran, valid, _ in results if ran and valid)
    print(f"\n  Completed: {n_completed}/{len(results)}")
    print(f"  Validated: {n_validated}/{len(results)}")

    if all_pass:
        print("\nGate 2: ALL CHECKS PASSED")
    else:
        print("\nGate 2: FAILURES DETECTED")
        sys.exit(1)


if __name__ == "__main__":
    main()
