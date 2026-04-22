#!/usr/bin/env python3
"""
LERNA Phase 1: Run all 6 simple baselines on GLUE tasks.

Runs each baseline callback on the same GLUE tasks with the same
configuration as the LERNA experiments, for direct comparison.

Usage:
  # Smoke test: 1 baseline, 1 task, 1 seed
  python scripts/run_baseline_comparison.py --mode smoke

  # Run specific baseline on specific tasks
  python scripts/run_baseline_comparison.py --baselines grad_norm_skip random_skip --tasks sst2 mrpc --seeds 42 43

  # Full comparison: all baselines x 4 tasks x 5 seeds
  python scripts/run_baseline_comparison.py --mode full --wandb

  # Run all baselines x all 8 GLUE tasks x 10 seeds (production)
  python scripts/run_baseline_comparison.py --mode production --wandb
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false"

import sys
import json
import time
import argparse
import gc
import numpy as np
from datetime import datetime, timedelta

import torch
torch._dynamo.config.disable = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from transformers import EarlyStoppingCallback

# Import from existing baseline runner
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_baseline_glue import (
    run_single_experiment,
    detect_device_profile,
    GLUE_TASK_CONFIG,
    TASK_HP_OVERRIDES,
    _ensure_wandb_finished,
)
from lerna.callbacks.simple_baselines import (
    GradientNormSkippingCallback,
    RandomStepSkippingCallback,
    WeightFreezingCallback,
    ReducedTotalStepsCallback,
    CosineAnnealingWarmRestartsCallback,
    create_all_baselines,
)


# Baseline configurations
BASELINE_CONFIGS = {
    "grad_norm_skip": {
        "description": "Skip backward when ||g|| < threshold",
        "class": GradientNormSkippingCallback,
        "params": {"grad_norm_threshold": 0.01},
    },
    "random_skip": {
        "description": "Skip backward randomly at matched rate",
        "class": RandomStepSkippingCallback,
        "params": {"target_skip_rate": 0.33},
    },
    "weight_freeze": {
        "description": "Freeze weights during LER-detected plateaus (no momentum)",
        "class": WeightFreezingCallback,
        "params": {},  # needs ler_tracker, added at runtime
    },
    "reduced_steps": {
        "description": "Train fewer total steps (same compute budget)",
        "class": ReducedTotalStepsCallback,
        "params": {"reduction_fraction": 0.33},
    },
    "cosine_warm_restarts": {
        "description": "Cosine annealing LR with warm restarts",
        "class": CosineAnnealingWarmRestartsCallback,
        "params": {},
    },
    "early_stopping": {
        "description": "Early stopping with optimal patience (3, 5, 10, 20)",
        "class": EarlyStoppingCallback,
        "params": {},  # patience set per-run
    },
}


def build_baseline_callback(baseline_name: str, config: dict):
    """Build a baseline callback instance from config.
    
    Used by both run_baseline_comparison.py and run_phase1_2_simple_baselines.py.
    """
    from lerna.utils.metrics import LERTracker
    import torch.nn as nn
    
    callback_class = config["class"]
    params = config.get("params", {})
    
    # Some callbacks need additional setup
    if baseline_name == "weight_freeze":
        # WeightFreezingCallback needs a dummy model/ler_tracker
        # It will be updated when attached to trainer
        return callback_class(ler_tracker=None, **params)
    
    return callback_class(**params)


def main():
    parser = argparse.ArgumentParser(description="LERNA Phase 1: Baseline Comparison")
    parser.add_argument(
        "--mode", choices=["smoke", "full", "production"], default="smoke",
        help="smoke=1 baseline/task/seed, full=all baselines x 4 tasks x 5 seeds, "
             "production=all baselines x 8 tasks x 10 seeds",
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        choices=list(BASELINE_CONFIGS.keys()),
        help="Which baselines to run (default: all)",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="GLUE tasks")
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--output-dir", default="./experiments/baseline_comparison")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerna-2026")
    parser.add_argument("--max-samples", type=int, default=None)
    args = parser.parse_args()

    profile = detect_device_profile()

    # Mode presets
    if args.mode == "smoke":
        baselines = ["grad_norm_skip"]
        tasks = ["sst2"]
        seeds = [42]
    elif args.mode == "full":
        baselines = list(BASELINE_CONFIGS.keys())
        tasks = ["sst2", "mrpc", "rte", "qnli"]
        seeds = list(range(42, 47))  # 5 seeds
    else:  # production
        baselines = list(BASELINE_CONFIGS.keys())
        tasks = list(GLUE_TASK_CONFIG.keys())
        seeds = list(range(42, 52))  # 10 seeds

    # Override with CLI args
    if args.baselines:
        baselines = args.baselines
    if args.tasks:
        tasks = args.tasks
    if args.seeds:
        seeds = args.seeds

    total_runs = len(baselines) * len(tasks) * len(seeds)
    wandb_group = f"phase1-baselines-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n{'=' * 60}")
    print(f"  LERNA Phase 1: Baseline Comparison")
    print(f"{'=' * 60}")
    print(f"  Baselines: {baselines}")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"  Profile: {profile}")
    if args.wandb:
        print(f"  W&B project: {args.wandb_project}")
        print(f"  W&B group: {wandb_group}")
    print(f"{'=' * 60}\n")

    all_results = []
    run_idx = 0
    overall_start = time.time()

    for baseline_name in baselines:
        for task in tasks:
            for seed in seeds:
                run_idx += 1

                # ETA
                if run_idx > 1:
                    elapsed = time.time() - overall_start
                    avg_per_run = elapsed / (run_idx - 1)
                    remaining = (total_runs - run_idx + 1) * avg_per_run
                    print(f"\n  === Run {run_idx}/{total_runs} | "
                          f"ETA: {timedelta(seconds=int(remaining))} ===")

                print(f"  Baseline: {baseline_name} | Task: {task} | Seed: {seed}")

                try:
                    from scripts.run_baseline_glue import run_single_experiment
                    # run_single_experiment must accept extra_callbacks; see §3b below.
                    result = run_single_experiment(
                        task_name=task,
                        seed=seed,
                        lr=BASELINE_CONFIGS[baseline_name]["params"].get("lr", 2e-5),
                        profile=profile,
                        base_output_dir=str(output_dir / baseline_name / task / f"seed{seed}"),
                        use_wandb=args.wandb,
                        extra_callbacks=[
                            build_baseline_callback(baseline_name, BASELINE_CONFIGS[baseline_name])
                        ],
                    )
                    result.update({
                        "baseline": baseline_name,
                        "task": task,
                        "seed": seed,
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                    })
                    all_results.append(result)

                except Exception as e:
                    # Do NOT swallow silently; log and record failure but continue sweep.
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "baseline": baseline_name,
                        "task": task,
                        "seed": seed,
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })

    # Save summary
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, "baseline_comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"  BASELINE COMPARISON CONFIGURED: {len(all_results)} runs")
    print(f"  Summary: {summary_path}")
    print(f"  Wall time: {timedelta(seconds=int(total_elapsed))}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
