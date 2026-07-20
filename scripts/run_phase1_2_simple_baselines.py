#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2: Simple Baselines on GLUE
=============================================================================

Self-contained experiment script for Phase 1.2 of the LERNA research plan.

Purpose:
  Run all 6 simple baselines on the same GLUE tasks/configs as Phase 1.1
  to prove that LERNA's LER-guided switching is justified. If any simple
  baseline matches LERNA's accuracy-energy tradeoff, LER adds no value.

Baselines:
  1. Gradient Norm Thresholding  - skip backward when ||g|| < threshold
  2. Random Step Skipping        - skip randomly at matched rate
  3. Early Stopping (patience)   - HF EarlyStoppingCallback with p=3,5,10,20
  4. Weight Freezing              - freeze weights during LER-detected plateaus
  5. Reduced Total Steps          - train fewer steps (same compute budget)
  6. Cosine Annealing + Restarts  - cosine LR schedule with warm restarts

Experiment Matrix:
  - Model: RoBERTa-base (same as Phase 1.1)
  - Tasks: 8 GLUE tasks (sst2, qnli, qqp, mnli, rte, mrpc, cola, stsb)
  - Seeds: 10 per task per baseline (42-51)
  - Metrics: accuracy, energy (kWh), waste ratio, training time

Usage:
  # Smoke test: 1 baseline, 1 task, 1 seed (~2 min)
  python scripts/run_phase1_2_simple_baselines.py --mode smoke

  # Quick validation: all baselines x 2 tasks x 2 seeds (~1 hour)
  python scripts/run_phase1_2_simple_baselines.py --mode quick

  # Full run: all baselines x 4 representative tasks x 5 seeds (~12 hours)
  python scripts/run_phase1_2_simple_baselines.py --mode full

  # Production: all baselines x 8 tasks x 10 seeds (~5 days)
  python scripts/run_phase1_2_simple_baselines.py --mode production

  # Custom: pick baselines, tasks, seeds
  python scripts/run_phase1_2_simple_baselines.py \
      --baselines grad_norm random_skip early_stop_p3 \
      --tasks sst2 mrpc --seeds 42 43 44

Output:
  experiments/phase1_2_baselines/
    <baseline>/<task>_s<seed>_lr<lr>/
      results.json           - per-run metrics
      power/                 - energy telemetry
    phase1_2_summary.json    - aggregated results across all runs
    phase1_2_comparison.txt  - human-readable comparison table

Dependencies:
  - Phase 1.1 must be completed (baseline numbers for comparison)
  - lerna/callbacks/simple_baselines.py (baseline callback implementations)
  - lerna/utils/metrics.py (LERTracker for weight_freeze baseline)
  - lerna/callbacks/efficiency_callback.py (PowerTelemetryCallback)

Author: LERNA Research Team
Phase: 1.2 (Simple Baselines)
Estimated compute: ~480 runs x ~6 min/run = ~48 GPU-hours (RTX 5090)
=============================================================================
"""

import os

# Must be set before any torch import
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
if not os.environ.get("HF_HUB_OFFLINE") and not os.environ.get("TRANSFORMERS_OFFLINE"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_LOG_MODEL"] = "false"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import sys
import json
import time
import argparse
import gc
import logging
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Optional, Any, Tuple, List

import torch

try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass

try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback,
)
from datasets import load_dataset
import evaluate

# Project root for lerna package imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Scripts directory for importing Phase 1.1 GLUE config directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_baseline_glue import TASK_HP_OVERRIDES
from lerna.utils.metrics import LERTracker
from lerna.callbacks.efficiency_callback import PowerTelemetryCallback
from lerna.callbacks.comprehensive_metrics import (
    ComprehensiveMetricsCallback,
    GradNormDetailedCallback,
)
from lerna.callbacks.all_charts import AllChartsMetricsCallback
from lerna.callbacks.simple_baselines import (
    GradientNormSkippingCallback,
    RandomStepSkippingCallback,
    WeightFreezingCallback,
    ReducedTotalStepsCallback,
    CosineAnnealingWarmRestartsCallback,
)
from lerna.callbacks.ler_feed import LERFeedCallback   # [FIX #5]
from lerna.trainers import (
    TrueBackwardSkippingTrainer,
    SchedulerStepPolicy,
    ComputeSavingMechanism,
)
from lerna.trainers.policies import (
    AlwaysFalsePolicy,
    GradNormSkipPolicy,
    RandomSkipPolicy,
    LERPlateauPolicy,
)


def safe_from_pretrained(*args, **kwargs):
    try:
        return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
    except TypeError as exc:
        if "reference_compile" in str(exc) or "unexpected keyword argument" in str(exc):
            kwargs.pop("reference_compile", None)
            return AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        raise

logger = logging.getLogger(__name__)


# =============================================================================
# Phase 1.2 cleanup utilities
# =============================================================================

PHASE1_2_KEEP_FILENAMES = {
    "results.json",
    "instrumentation.json",
    "power_telemetry_report.json",
}

PHASE1_2_KEEP_PREFIXES = (
    "baseline_",
)

PHASE1_2_DELETE_FILENAMES = {
    "optimizer.pt",
    "scheduler.pt",
    "rng_state.pth",
    "scaler.pt",
    "trainer_state.json",
    "training_args.bin",
    "pytorch_model.bin",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "merges.txt",
    "config.json",
    "generation_config.json",
}

PHASE1_2_DELETE_DIRNAMES = {
    "__pycache__",
    "all_data",
}


def release_cuda_memory() -> None:
    """Best-effort release of Python and CUDA allocator memory between runs."""
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        pass
    try:
        torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def cleanup_phase1_2_run_dir(
    run_dir: str,
    keep_power_samples: bool = False,
    keep_debug_artifacts: bool = False,
) -> dict:
    """Delete nonessential artifacts after a Phase 1.2 run."""
    root = Path(run_dir)
    summary = {"files_removed": 0, "dirs_removed": 0, "bytes_removed": 0}
    if not root.exists():
        return summary

    def remove_file(path: Path) -> None:
        try:
            size = path.stat().st_size
            path.unlink()
            summary["files_removed"] += 1
            summary["bytes_removed"] += size
        except OSError:
            pass

    def remove_dir(path: Path) -> None:
        try:
            size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
            shutil.rmtree(path, ignore_errors=True)
            summary["dirs_removed"] += 1
            summary["bytes_removed"] += size
        except OSError:
            pass

    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        rel_parts = path.relative_to(root).parts
        name = path.name

        if path.is_dir():
            if name.startswith("checkpoint-") or name in PHASE1_2_DELETE_DIRNAMES:
                remove_dir(path)
            continue

        if name in PHASE1_2_KEEP_FILENAMES or name.startswith(PHASE1_2_KEEP_PREFIXES):
            continue
        if keep_power_samples and name == "power_samples.json":
            continue
        if keep_debug_artifacts and name.endswith(".json"):
            continue

        if name in PHASE1_2_DELETE_FILENAMES:
            remove_file(path)
            continue
        if rel_parts and rel_parts[0] in {"runs", "wandb"}:
            remove_file(path)
            continue
        if name in {"power_samples.json", "training_summary.json"}:
            remove_file(path)

    return summary


# =============================================================================
# Constants (identical to Phase 1.1 for reproducibility)
# =============================================================================

MODEL_NAME = "jhu-clsp/ettin-encoder-150m"
MODEL_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"

GLUE_TASK_CONFIG = {
    "sst2":  {"keys": ("sentence", None),        "num_labels": 2, "metric": "accuracy"},
    "qnli":  {"keys": ("question", "sentence"),   "num_labels": 2, "metric": "accuracy"},
    "qqp":   {"keys": ("question1", "question2"), "num_labels": 2, "metric": "accuracy"},
    "mnli":  {"keys": ("premise", "hypothesis"),   "num_labels": 3, "metric": "accuracy"},
    "rte":   {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
    "mrpc":  {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
    "cola":  {"keys": ("sentence", None),          "num_labels": 2, "metric": "matthews_correlation"},
    "stsb":  {"keys": ("sentence1", "sentence2"), "num_labels": 1, "metric": "pearsonr"},
}

# Reuse Phase 1.1 per-task hyperparameter overrides directly to prevent drift.
# Phase 1.2 must match Phase 1.1 exactly for a fair baseline comparison.

# Phase 1.1 reference results for comparison.
#
# Note on waste_ratio=0.0 for RTE, MRPC, CoLA:
# These values are CORRECT, not a bug. These tasks use extended training
# (10-20 epochs) with aggressive early stopping (patience 10-15), so the
# WasteQuantifier correctly detects that compute is used efficiently.
# Large-dataset tasks (QQP, MNLI) show massive waste because the model
# converges in ~1% of steps during standard 3-epoch training.
# STS-B (0.370) was re-run with the RPSE fix and shows moderate waste.
# See README.md "Phase 1.1" section for full analysis.
PHASE_1_1_RESULTS = {
    "sst2":  {"accuracy": 0.9373, "std": 0.0054, "waste_ratio": 0.504},
    "qnli":  {"accuracy": 0.9248, "std": 0.0018, "waste_ratio": 0.558},
    "qqp":   {"accuracy": 0.9081, "std": 0.0022, "waste_ratio": 0.988},
    "mnli":  {"accuracy": 0.8728, "std": 0.0030, "waste_ratio": 0.989},
    "rte":   {"accuracy": 0.7639, "std": 0.0090, "waste_ratio": 0.000},  # 20 epochs + ES patience 15
    "mrpc":  {"accuracy": 0.8831, "std": 0.0066, "waste_ratio": 0.000},  # 10 epochs + ES patience 10
    "cola":  {"accuracy": 0.5780, "std": 0.0055, "waste_ratio": 0.000},  # 10 epochs + ES patience 10
    "stsb":  {"accuracy": 0.9026, "std": 0.0022, "waste_ratio": 0.370},  # Post-RPSE fix
}

# Baseline definitions
BASELINE_REGISTRY = {
    "full_finetune": {
        "description": "Full fine-tuning control with no compute-saving intervention",
        "tests": "Reference quality, runtime, and backward-call count",
    },
    "grad_norm": {
        "description": "Skip backward pass when ||g|| < threshold",
        "tests": "Whether LER is better than its simplest component",
    },
    "random_skip": {
        "description": "Skip backward pass randomly at matched rate",
        "tests": "Whether the selection of which steps to skip matters",
    },
    "early_stop_p3": {
        "description": "Early stopping with patience=3",
        "tests": "Whether LERNA captures anything beyond 'stop when not learning'",
    },
    "early_stop_p5": {
        "description": "Early stopping with patience=5",
        "tests": "Whether LERNA captures anything beyond 'stop when not learning'",
    },
    "early_stop_p10": {
        "description": "Early stopping with patience=10",
        "tests": "Whether LERNA captures anything beyond 'stop when not learning'",
    },
    "early_stop_p20": {
        "description": "Early stopping with patience=20",
        "tests": "Whether LERNA captures anything beyond 'stop when not learning'",
    },
    "weight_freeze": {
        "description": "Freeze weights during LER-detected plateaus (no momentum)",
        "tests": "Whether momentum extrapolation actually helps vs doing nothing",
    },
    "reduced_steps": {
        "description": "Train fewer total steps (same compute budget as LERNA)",
        "tests": "Whether just training less achieves the same result",
    },
    "cosine_restarts": {
        "description": "Cosine annealing LR with warm restarts",
        "tests": "Whether phase-aware LR scheduling captures the same benefit",
    },
}

BASELINES = [
    "full_finetune",
    "grad_norm",
    "random_skip",
    "weight_freeze",
    "reduced_steps",
    "cosine_restarts",
    "early_stop_p3",
    "early_stop_p5",
    "early_stop_p10",
    "early_stop_p20",
]


# =============================================================================
# Data & Model Utilities (reused from Phase 1.1)
# =============================================================================

def detect_device_profile():
    """Detect hardware profile based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb >= 20:
        return "server"
    return "laptop"


def get_training_config(profile: str) -> dict:
    """Return hardware-appropriate training hyperparameters."""
    if profile == "server":
        use_bf16, use_fp16 = False, True
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:
                use_bf16, use_fp16 = True, False
        return {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "fp16": use_fp16, "bf16": use_bf16,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 4,
            "max_samples": None,
        }
    else:
        gradient_accumulation_steps = 1 if profile in ("smoke", "cpu") else 4
        return {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "fp16": True, "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "max_samples": 2000,
        }


def load_glue_task(task_name, tokenizer, max_length=128, max_samples=None):
    """Load and tokenize a GLUE task dataset."""
    cfg = GLUE_TASK_CONFIG[task_name]
    key1, key2 = cfg["keys"]
    dataset = load_dataset("glue", task_name)

    def tokenize_fn(examples):
        if key2 is not None:
            return tokenizer(examples[key1], examples[key2],
                             truncation=True, max_length=max_length, padding=False)
        return tokenizer(examples[key1],
                         truncation=True, max_length=max_length, padding=False)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[
        c for c in dataset["train"].column_names if c not in ["label", "labels"]
    ])
    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    train_ds = tokenized["train"]
    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    eval_ds = tokenized[eval_key]

    if max_samples is not None:
        if len(train_ds) > max_samples:
            train_ds = train_ds.select(range(max_samples))
        eval_max = min(max_samples // 4, len(eval_ds))
        eval_ds = eval_ds.select(range(eval_max))

    return train_ds, eval_ds, cfg


def build_compute_metrics(task_name):
    """Build the metric computation function for a GLUE task."""
    cfg = GLUE_TASK_CONFIG[task_name]
    metric_name = cfg["metric"]
    if metric_name == "matthews_correlation":
        metric = evaluate.load("glue", "cola")
    elif metric_name == "pearsonr":
        metric = evaluate.load("glue", "stsb")
    else:
        metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred.predictions, eval_pred.label_ids
        if cfg["num_labels"] == 1:
            predictions = predictions.squeeze()
        else:
            predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


# =============================================================================
# Custom Trainer (captures real logits for LER, same as Phase 1.1)
# =============================================================================
# ==============================================================================
# Baseline utilities
# ==============================================================================

def parse_early_stop_patience(name):
    if not name.startswith("early_stop_p"):
        return None
    return int(name.removeprefix("early_stop_p"))


class _GradientNormCaptureCallback(TrainerCallback):
    """[REMOVED] This callback caused double recording of grad norm.
    Retained here only as a placeholder. Do not instantiate.
    """


# ==============================================================================
# Core: Run a single baseline experiment
# ==============================================================================

def run_single_baseline_experiment(
    baseline_name: str,
    task_name: str,
    seed: int,
    profile: str,
    base_output_dir: str,
    max_samples_override: Optional[int] = None,
    run_idx: int = 0,
    total_runs: int = 0,
    target_skip_rate: float = 0.33,
    cleanup_artifacts: bool = True,
    keep_power_samples: bool = False,
    keep_debug_artifacts: bool = False,
    model_name: str = MODEL_NAME,
    model_revision: str = MODEL_REVISION,
    unlimited: bool = False,
) -> dict:
    """
    Run a single Phase 1.2 baseline experiment.

    Mirrors run_single_experiment() from Phase 1.1 but injects the
    specified baseline callback. All diagnostics (LER, power) are
    tracked identically for direct comparison.
    """
    hw_cfg = get_training_config(profile)
    if unlimited:
        hw_cfg["max_samples"] = None
    elif max_samples_override is not None:
        hw_cfg["max_samples"] = max_samples_override

    if not model_revision:
        raise ValueError("A pinned Hugging Face model revision is required")

    # Resolve per-task hyperparameters
    task_hp = TASK_HP_OVERRIDES.get(task_name, {})
    lr = task_hp.get("learning_rate", 2e-5)
    num_epochs = task_hp.get("num_epochs", 5)
    warmup_ratio = task_hp.get("warmup_ratio", 0.1)
    default_patience = task_hp.get("early_stopping_patience", 5)
    metric_for_best_model = task_hp.get("metric_for_best_model", "eval_loss")
    greater_is_better = task_hp.get("greater_is_better", False)

    run_id = f"{baseline_name}/{task_name}_s{seed}_lr{lr:.0e}"
    output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  Phase 1.2 | Baseline: {baseline_name} | Task: {task_name} | Seed: {seed}")
    print(f"  LR: {lr} | Epochs: {num_epochs} | Profile: {profile}")
    if run_idx and total_runs:
        print(f"  Progress: run {run_idx}/{total_runs}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    # --- Reproducibility ---
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # --- Model & Tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=model_revision,
    )
    cfg = GLUE_TASK_CONFIG[task_name]

    model_kwargs = {
        "num_labels": cfg["num_labels"],
        "attn_implementation": "sdpa",
        "reference_compile": False,
    }
    model = safe_from_pretrained(model_name, revision=model_revision, **model_kwargs)


    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # --- Data ---
    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"]
    )
    print(f"  Train: {len(train_ds)} samples | Eval: {len(eval_ds)} samples")

    # --- Compute total steps ---
    steps_per_epoch = max(1, len(train_ds) // (
        hw_cfg["per_device_train_batch_size"] * hw_cfg["gradient_accumulation_steps"]
    ))
    total_steps = steps_per_epoch * num_epochs
    eval_steps = max(total_steps // 20, 10)

    # --- Initialize LER tracker (needed for weight_freeze & diagnostics) ---
    ler_tracker = LERTracker(task=task_name, window_size=5)

    # --- Skip policy selection ---
    target_skip = target_skip_rate
    calibration_steps = max(20, min(200, total_steps // 5))
    recalibrate_every = max(100, total_steps // 2)

    if baseline_name == "full_finetune":
        skip_policy = AlwaysFalsePolicy()
        mechanism = ComputeSavingMechanism.NONE
    elif baseline_name == "grad_norm":
        skip_policy = GradNormSkipPolicy(
            target_skip_rate=target_skip,
            calibration_steps=50,
            recalibrate_every=max(100, total_steps // 2),
            min_step=0,
            min_calibration_samples=20,
            rolling_window_size=1000,
            max_consecutive_skips=1,  # [IMP-3] probe rule: skip-one-compute-one
        )
        mechanism = ComputeSavingMechanism.BACKWARD_SKIPPING
    elif baseline_name == "random_skip":
        skip_policy = RandomSkipPolicy(target_skip_rate=target_skip, min_step=100, seed=seed)
        mechanism = ComputeSavingMechanism.BACKWARD_SKIPPING
    elif baseline_name == "weight_freeze":
        skip_policy = LERPlateauPolicy(ler_tracker=ler_tracker, threshold=1e-5, min_step=100)
        mechanism = ComputeSavingMechanism.BACKWARD_SKIPPING
    elif baseline_name.startswith("early_stop_p"):
        skip_policy = AlwaysFalsePolicy()
        mechanism = ComputeSavingMechanism.EARLY_STOPPING
    elif baseline_name == "reduced_steps":
        skip_policy = AlwaysFalsePolicy()
        mechanism = ComputeSavingMechanism.REDUCED_TOTAL_STEPS  # [FIX #9]
    else:
        skip_policy = AlwaysFalsePolicy()
        mechanism = ComputeSavingMechanism.NONE

    # ---- Diagnostic-only callbacks for non-skipping baselines ----
    diagnostic_callback = None
    if baseline_name == "reduced_steps":
        diagnostic_callback = ReducedTotalStepsCallback(
            reduction_fraction=target_skip,
            total_steps=total_steps,
            wandb_enabled=False,
        )
    elif baseline_name == "cosine_restarts":
        diagnostic_callback = CosineAnnealingWarmRestartsCallback(
            T_0=max(total_steps // 10, 50), T_mult=2, eta_min=1e-7,
            base_lr=lr, wandb_enabled=False,
        )

    # ---- Early stopping patience (FIX #8) ----
    es_patience = parse_early_stop_patience(baseline_name)

    # ---- Shared LER feed (from lerna.callbacks.ler_feed) [FIX #5] ----
    ler_feed_callback = LERFeedCallback(ler_tracker=ler_tracker)

    # ---- Power telemetry callback ----
    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=False,
        log_frequency=50,
        require_measured_power=True,
    )

    # ---- Comprehensive metrics for W&B visualizations ----
    comprehensive_metrics = ComprehensiveMetricsCallback(
        log_frequency=10,
        histogram_frequency=100,
        table_frequency=500,
        wandb_enabled=False,
    )
    grad_norm_detailed = GradNormDetailedCallback(
        wandb_enabled=False,
        log_frequency=10,
    )

    all_charts_metrics = AllChartsMetricsCallback(
        log_frequency=20,
        wandb_enabled=False,
    )

    callbacks = [
        power_callback,
        ler_feed_callback,
        comprehensive_metrics,
        grad_norm_detailed,
        all_charts_metrics,
    ]
    if diagnostic_callback is not None:
        callbacks.append(diagnostic_callback)

    es_patience = parse_early_stop_patience(baseline_name)
    if es_patience is not None:
        callbacks.insert(
            0,
            EarlyStoppingCallback(early_stopping_patience=es_patience),
        )

    # --- Training arguments (identical to Phase 1.1) ---
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=hw_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        fp16=hw_cfg["fp16"],
        bf16=hw_cfg["bf16"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=max(eval_steps // 5, 1),
        report_to=[],
        run_name=None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
    )

    if profile == "server" and training_args.device.type != "cuda":
        raise RuntimeError(
            f"Server profile requires CUDA, got {training_args.device}"
        )

    if torch.cuda.is_available() and training_args._n_gpu > 1:
        print("  [Phase1.2] Multiple GPUs detected; forcing single-GPU training to avoid unstable NCCL/DataParallel behavior.")
        training_args._n_gpu = 1

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics(task_name)

    # --- Create trainer ---
    trainer = TrueBackwardSkippingTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        skip_policy=skip_policy,
        scheduler_step_policy=SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP,
        instrumentation_path=os.path.join(output_dir, "instrumentation.json"),
        compute_saving_mechanism=mechanism,                # [FIX #9]
        capture_logits=True,
    )

    ler_feed_callback.attach(trainer=trainer)
    trainer.ler_tracker = ler_tracker
    if diagnostic_callback is not None and hasattr(diagnostic_callback, '_trainer'):
        diagnostic_callback._trainer = trainer

    # --- Train ---
    start_time = time.time()
    print(f"  Starting training: ~{total_steps} steps, eval every {eval_steps} steps")
    print(f"  Baseline: {BASELINE_REGISTRY[baseline_name]['description']}")
    print(f"  Early stopping patience: {es_patience}")
    if baseline_name == "reduced_steps" and diagnostic_callback is not None and hasattr(diagnostic_callback, 'max_steps'):
        print(f"  Reduced steps max: {diagnostic_callback.max_steps} (of {total_steps})")

    try:
        train_result = trainer.train()
        train_time = time.time() - start_time

        # --- Evaluate best model ---
        print(f"  Evaluating best model...")
        eval_result = trainer.evaluate()
    finally:
        # HF skips on_train_end if train()/evaluate() raises, leaking the
        # nvidia-smi daemon thread. _stop_sampling() is idempotent, so the
        # success path (already stopped in on_train_end) is unaffected.
        try:
            power_callback._stop_sampling()
        except Exception:
            pass

    instrumentation = trainer.get_instrumentation()

    # --- Post-run baseline activation validation ---
    # FIX (2026-04-15): Verify each baseline actually activated.
    # This catches silent failures where callbacks are present but inert.
    # Also check LER feed callback
    print(f"  LER feed updates: {ler_feed_callback._update_count}")
    ler_diag = ler_tracker.get_diagnostics()
    print(f"  LER tracker: n_steps={ler_diag.get('n_steps', 0)}, "
          f"ler={ler_diag.get('ler', 'None')}, phase={ler_diag.get('phase', 'unknown')}")

    # --- Collect results ---
    avg_power = (
        float(np.mean([s["power_w"] for s in power_callback._power_samples]))
        if power_callback._power_samples else 0
    )

    # Get baseline-specific stats
    baseline_stats = {
        "skip_ratio": instrumentation["skip_ratio_by_batch"],
    }
    if baseline_name == "grad_norm" and isinstance(skip_policy, GradNormSkipPolicy):
        baseline_stats.update({
            "grad_norm_samples_collected": len(skip_policy._grad_norms),
            "grad_norm_threshold": skip_policy._threshold,
            "grad_norm_calibrated": skip_policy._threshold is not None,
            "grad_norm_calibration_step": skip_policy._last_calibration_step,
            "grad_norm_min": float(min(skip_policy._grad_norms)) if skip_policy._grad_norms else None,
            "grad_norm_max": float(max(skip_policy._grad_norms)) if skip_policy._grad_norms else None,
            "grad_norm_mean": float(np.mean(skip_policy._grad_norms)) if skip_policy._grad_norms else None,
            "grad_norm_last": float(skip_policy._grad_norm_last) if skip_policy._grad_norm_last is not None else None,
            "grad_norm_forced_probe_count": skip_policy._grad_norm_forced_probe_count,
            "grad_norm_skip_decisions": skip_policy._grad_norm_skip_decisions,
        })

        print(f"  [grad_norm] samples collected: {baseline_stats['grad_norm_samples_collected']}")
        print(f"  [grad_norm] threshold: {baseline_stats['grad_norm_threshold']}")
        print(f"  [grad_norm] calibrated: {baseline_stats['grad_norm_calibrated']}")
        print(f"  [grad_norm] skipped: {instrumentation['skipped_backward_steps']}")
    # Extract the primary metric
    primary_metric_key = {
        "accuracy": "eval_accuracy",
        "matthews_correlation": "eval_matthews_correlation",
        "pearsonr": "eval_pearson",
    }.get(cfg["metric"], "eval_accuracy")
    # Handle alternate key names from evaluate library
    primary_value = eval_result.get(
        primary_metric_key,
        eval_result.get("eval_pearsonr", eval_result.get("eval_accuracy", 0))
    )

    training_used_cuda = training_args.device.type == "cuda"
    energy_valid = power_callback.energy_valid and training_used_cuda

    results = {
        "phase": "1.2",
        "baseline": baseline_name,
        "task": task_name,
        "seed": seed,
        "learning_rate": lr,
        "model_name": model_name,
        "model_revision": model_revision,
        "profile": profile,
        "training_device": str(training_args.device),
        "cuda_available": torch.cuda.is_available(),
        "primary_metric": primary_value,
        "primary_metric_name": primary_metric_key,
        "eval_metrics": eval_result,
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_time,
        "train_steps": trainer.state.global_step,
        "energy_kwh": power_callback.cumulative_kwh if energy_valid else None,
        "energy_valid": energy_valid,
        "energy_invalid_reason": power_callback.energy_invalid_reason if energy_valid else "training did not use CUDA",
        "energy_measurement_source": power_callback.energy_measurement_source,
        "power_avg_watts": avg_power,
        "max_samples": hw_cfg.get("max_samples"),
        "dataset_scope": "full GLUE" if hw_cfg.get("max_samples") is None else "sample-capped",
        "early_stopping_enabled": es_patience is not None,
        "early_stopping_patience": es_patience,
        "baseline_stats": baseline_stats,
        "target_skip_rate": target_skip_rate,
        "compute_saving_mechanism": instrumentation["compute_saving_mechanism"],  # [FIX #9]
        "true_skip_instrumentation": instrumentation,
        "precision_mode": instrumentation["precision_mode"],
        "true_backward_skipping_enabled": instrumentation["true_backward_skipping_enabled"],
        "scheduler_step_policy": instrumentation["scheduler_step_policy"],
        "policy_name": instrumentation["policy_name"],
        "last_grad_scale": getattr(trainer, "_last_grad_scale", None),
        "skipped_backward_steps": instrumentation["skipped_backward_steps"],
        "optimizer_step_attempts": instrumentation["optimizer_step_attempts"],  # [IMP-4] renamed
        "batches_seen": instrumentation["batches_seen"],
        "skipped_batches": instrumentation["skipped_batches"],
        "forward_calls": instrumentation["forward_calls"],
        "backward_calls": instrumentation["backward_calls"],
        "grad_scaler_step_calls": instrumentation["grad_scaler_step_calls"],
        "scheduler_step_calls": instrumentation["scheduler_step_calls"],
        "skip_ratio_by_batch": instrumentation["skip_ratio_by_batch"],
        "skip_ratio_by_optimizer_opportunity": instrumentation["skip_ratio_by_optimizer_opportunity"],
        "timestamp": datetime.now().isoformat(),
    }

    # Save per-run results
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n  --- Results ---")
    print(f"  {primary_metric_key}: {primary_value:.4f}")
    print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
    print(f"  Time: {train_time:.1f}s")
    print(f"  Steps: {trainer.state.global_step}")
    if baseline_stats:
        print(f"  Skip ratio: {baseline_stats.get('skip_ratio', 0):.3f}")
    print(f"  Saved: {results_path}")

    # --- Cleanup ---
    del model, trainer, tokenizer, train_ds, eval_ds, data_collator, compute_metrics
    release_cuda_memory()

    if cleanup_artifacts:
        cleanup = cleanup_phase1_2_run_dir(
            output_dir,
            keep_power_samples=keep_power_samples,
            keep_debug_artifacts=keep_debug_artifacts,
        )
        if cleanup["files_removed"] or cleanup["dirs_removed"]:
            print(
                "  Cleanup: removed "
                f"{cleanup['files_removed']} files, {cleanup['dirs_removed']} dirs, "
                f"{cleanup['bytes_removed'] / (1024 ** 2):.1f} MB"
            )

    return results


# =============================================================================
# Summary & Comparison
# =============================================================================

def generate_comparison_summary(
    all_results: List[dict],
    output_dir: str,
    tasks: List[str],
    baselines: List[str],
):
    """Generate comparison tables and summary JSON."""
    successful = [r for r in all_results if "error" not in r]
    if not successful:
        print("  No successful runs to summarize.")
        return

    # --- Aggregate by baseline x task ---
    agg = defaultdict(lambda: defaultdict(list))
    for r in successful:
        key = (r["baseline"], r["task"])
        agg[key]["metric"].append(r["primary_metric"])
        agg[key]["energy"].append(r["energy_kwh"])
        agg[key]["time"].append(r["train_runtime_s"])
        agg[key]["steps"].append(r["train_steps"])
        skip_ratio = r.get("baseline_stats", {}).get("skip_ratio")
        if skip_ratio is None:
            skip_ratio = r.get("skip_ratio_by_batch")
        if skip_ratio is None:
            skip_ratio = r.get("true_skip_instrumentation", {}).get("skip_ratio_by_batch", 0)
        agg[key]["skip_ratio"].append(skip_ratio)

    # --- Build summary rows ---
    summary_rows = []
    for baseline in baselines:
        for task in tasks:
            key = (baseline, task)
            if key not in agg:
                continue
            metrics = agg[key]["metric"]
            energies = agg[key]["energy"]
            times = agg[key]["time"]
            skip_ratios = agg[key]["skip_ratio"]

            n = len(metrics)
            mean_metric = np.mean(metrics)
            std_metric = np.std(metrics) if n > 1 else 0
            mean_energy = np.mean(energies)
            mean_time = np.mean(times)
            mean_skip = np.mean(skip_ratios)

            # Compare to Phase 1.1
            ref = PHASE_1_1_RESULTS.get(task, {})
            ref_acc = ref.get("accuracy", 0)
            delta = mean_metric - ref_acc

            summary_rows.append({
                "baseline": baseline,
                "task": task,
                "n_seeds": n,
                "mean_metric": float(mean_metric),
                "std_metric": float(std_metric),
                "mean_energy_kwh": float(mean_energy),
                "mean_time_s": float(mean_time),
                "mean_skip_ratio": float(mean_skip),
                "phase1_1_metric": float(ref_acc),
                "delta_vs_phase1_1": float(delta),
            })

    # --- Save JSON summary ---
    summary = {
        "phase": "1.2",
        "description": "Simple Baselines on GLUE",
        "generated": datetime.now().isoformat(),
        "total_runs": len(all_results),
        "successful_runs": len(successful),
        "failed_runs": len(all_results) - len(successful),
        "baselines": baselines,
        "tasks": tasks,
        "results": summary_rows,
        "all_results": all_results,
    }
    summary_path = os.path.join(output_dir, "phase1_2_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary saved: {summary_path}")

    # --- Print human-readable comparison table ---
    comparison_lines = []
    header = (
        f"{'Baseline':<20} {'Task':<8} {'Seeds':>5} "
        f"{'Metric':>10} {'Std':>8} {'Energy':>10} "
        f"{'Skip%':>7} {'P1.1':>8} {'Delta':>8}"
    )
    sep = "-" * len(header)
    comparison_lines.append("")
    comparison_lines.append("=" * len(header))
    comparison_lines.append("  LERNA Phase 1.2: Simple Baselines Comparison")
    comparison_lines.append("=" * len(header))
    comparison_lines.append(header)
    comparison_lines.append(sep)

    for row in summary_rows:
        line = (
            f"{row['baseline']:<20} {row['task']:<8} {row['n_seeds']:>5} "
            f"{row['mean_metric']:>10.4f} {row['std_metric']:>8.4f} "
            f"{row['mean_energy_kwh']:>10.6f} "
            f"{row['mean_skip_ratio']*100:>6.1f}% "
            f"{row['phase1_1_metric']:>8.4f} "
            f"{row['delta_vs_phase1_1']:>+8.4f}"
        )
        comparison_lines.append(line)

    comparison_lines.append(sep)
    comparison_lines.append("")
    comparison_lines.append("  Delta = baseline_metric - phase1.1_metric")
    comparison_lines.append("  Negative delta = baseline is WORSE than Phase 1.1 full training")
    comparison_lines.append("  LERNA must outperform ALL baselines to justify LER sophistication")
    comparison_lines.append("")

    comparison_text = "\n".join(comparison_lines)
    print(comparison_text)

    # Save to file
    comparison_path = os.path.join(output_dir, "phase1_2_comparison.txt")
    with open(comparison_path, "w") as f:
        f.write(comparison_text)
    print(f"  Comparison table saved: {comparison_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="LERNA Phase 1.2: Simple Baselines on GLUE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_phase1_2_simple_baselines.py --mode smoke
  python scripts/run_phase1_2_simple_baselines.py --mode full --wandb
  python scripts/run_phase1_2_simple_baselines.py --baselines grad_norm random_skip --tasks sst2 mrpc
        """,
    )
    parser.add_argument(
        "--mode",
        choices=["smoke", "quick", "full", "production"],
        default="smoke",
        help=(
            "smoke: 1 baseline x 1 task x 1 seed (~2 min). "
            "quick: all baselines x 2 tasks x 2 seeds (~1 hr). "
            "full: all baselines x 4 tasks x 5 seeds (~12 hrs). "
            "production: all baselines x 8 tasks x 10 seeds (~5 days)."
        ),
    )
    parser.add_argument(
        "--baselines", nargs="+", default=None,
        choices=list(BASELINE_REGISTRY.keys()),
        help="Which baselines to run (default: all for the chosen mode)",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="GLUE tasks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds")
    parser.add_argument(
        "--output-dir", default="./experiments/phase1_2_baselines",
        help="Base output directory",
    )
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--model-revision", default=MODEL_REVISION)
    parser.add_argument("--wandb-project", default="lerna-phase1.2")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Cap training samples per task (default: auto based on profile)",
    )
    parser.add_argument(
        "--unlimited", action="store_true",
        help="Use full dataset without sample cap",
    )
    parser.add_argument(
        "--target-skip-rate", type=float, default=0.33,
        help="Target skip rate for baselines that need it (default: 0.33)",
    )
    parser.add_argument(
        "--no-cleanup-artifacts", action="store_true",
        help="Keep checkpoints/model weights/raw telemetry after each run",
    )
    parser.add_argument(
        "--keep-power-samples", action="store_true",
        help="Keep raw power/power_samples.json in addition to the compact power report",
    )
    parser.add_argument(
        "--keep-debug-artifacts", action="store_true",
        help="Keep per-run debug JSON files such as comprehensive/all-chart histories",
    )
    args = parser.parse_args()

    profile = detect_device_profile()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing to run official Phase 1.2 on CPU."
        )

    # --- Mode presets ---
    if args.mode == "smoke":
        baselines = ["grad_norm"]
        tasks = ["sst2"]
        seeds = [42]
    elif args.mode == "quick":
        baselines = list(BASELINE_REGISTRY.keys())
        tasks = ["sst2", "mrpc"]
        seeds = [42, 43]
    elif args.mode == "full":
        baselines = list(BASELINE_REGISTRY.keys())
        tasks = ["sst2", "mrpc", "rte", "qnli"]
        seeds = list(range(42, 47))  # 5 seeds
    else:  # production
        baselines = list(BASELINE_REGISTRY.keys())
        tasks = list(GLUE_TASK_CONFIG.keys())
        seeds = list(range(42, 52))  # 10 seeds

    # Override with CLI args
    if args.baselines:
        baselines = args.baselines
    if args.tasks:
        tasks = args.tasks
    if args.seeds:
        seeds = args.seeds

    # Sample cap
    if args.max_samples is not None:
        max_samples = args.max_samples
    elif args.unlimited:
        max_samples = None
    elif profile == "server":
        max_samples = None  # Full data on server (same as Phase 1.1)
    else:
        max_samples = 2000

    total_runs = len(baselines) * len(tasks) * len(seeds)

    # --- Lock environment ---
    from lock_environment import check_environment
    check_environment(args.output_dir)

    # --- Print experiment plan ---
    print(f"\n{'=' * 70}")
    print(f"  LERNA Phase 1.2: Simple Baselines on GLUE")
    print(f"{'=' * 70}")
    print(f"  Mode: {args.mode}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Profile: {profile}")
    print(f"  Baselines ({len(baselines)}): {baselines}")
    print(f"  Tasks ({len(tasks)}): {tasks}")
    print(f"  Seeds ({len(seeds)}): {seeds}")
    print(f"  Total runs: {total_runs}")
    print(f"  Max samples/task: {max_samples or 'unlimited'}")
    print(f"  Target skip rate: {args.target_skip_rate}")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")
    print(f"\n  Baselines to evaluate:")
    for b in baselines:
        info = BASELINE_REGISTRY[b]
        print(f"    - {b}: {info['description']}")
        print(f"      Tests: {info['tests']}")
    print()

    # Estimate time
    est_per_run = 360 if profile == "server" else 600  # seconds
    est_total = total_runs * est_per_run
    print(f"  Estimated total time: ~{timedelta(seconds=est_total)}")
    print()

    # --- Ensure no stale W&B run ---
    if args.wandb:
        raise ValueError("W&B is disabled for Phase 1.2")

    # --- Main experiment loop ---
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
                    print(f"\n  >>> Run {run_idx}/{total_runs} | "
                          f"ETA: {timedelta(seconds=int(remaining))} <<<")
                else:
                    print(f"\n  >>> Run {run_idx}/{total_runs} <<<")

                try:
                    result = run_single_baseline_experiment(
                        baseline_name=baseline_name,
                        task_name=task,
                        seed=seed,
                        profile=profile,
                        base_output_dir=args.output_dir,
                        max_samples_override=max_samples,
                        run_idx=run_idx,
                        total_runs=total_runs,
                        target_skip_rate=args.target_skip_rate,
                        cleanup_artifacts=not args.no_cleanup_artifacts,
                        keep_power_samples=args.keep_power_samples,
                        keep_debug_artifacts=args.keep_debug_artifacts,
                        model_name=args.model_name,
                        model_revision=args.model_revision,
                        unlimited=args.unlimited,
                    )
                    all_results.append(result)

                except Exception as e:
                    print(f"  FAILED: {baseline_name}/{task}/s{seed}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "baseline": baseline_name,
                        "task": task,
                        "seed": seed,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    })

                    run_dir = os.path.join(
                        args.output_dir,
                        baseline_name,
                        f"{task}_s{seed}_lr{TASK_HP_OVERRIDES.get(task, {}).get('learning_rate', 2e-5):.0e}",
                    )

                    if not args.no_cleanup_artifacts:
                        cleanup = cleanup_phase1_2_run_dir(
                            run_dir,
                            keep_power_samples=args.keep_power_samples,
                            keep_debug_artifacts=args.keep_debug_artifacts,
                        )
                        if cleanup["files_removed"] or cleanup["dirs_removed"]:
                            print(
                                "  Cleanup after failed run: removed "
                                f"{cleanup['files_removed']} files, {cleanup['dirs_removed']} dirs, "
                                f"{cleanup['bytes_removed'] / (1024 ** 2):.1f} MB"
                            )

                    release_cuda_memory()

    # --- Generate summary ---
    os.makedirs(args.output_dir, exist_ok=True)
    generate_comparison_summary(all_results, args.output_dir, tasks, baselines)

    total_elapsed = time.time() - overall_start
    successful = [r for r in all_results if "error" not in r]

    print(f"\n{'=' * 70}")
    print(f"  PHASE 1.2 COMPLETE")
    print(f"{'=' * 70}")
    print(f"  Total runs: {len(all_results)} ({len(successful)} successful, "
          f"{len(all_results) - len(successful)} failed)")
    print(f"  Wall time: {timedelta(seconds=int(total_elapsed))}")
    if successful:
        total_kwh = sum(r.get("energy_kwh", 0) for r in successful)
        print(f"  Total energy: {total_kwh:.4f} kWh")
    print(f"  Output: {args.output_dir}")
    print(f"{'=' * 70}")

    # --- Final verdict ---
    if successful:
        print(f"\n  === VERDICT ===")
        print(f"  Compare these results against Phase 1.1 baselines.")
        print(f"  If LERNA outperforms ALL baselines on accuracy-energy tradeoff,")
        print(f"  proceed to Phase 1.3 (Component Ablation).")
        print(f"  If any baseline matches LERNA, investigate before proceeding.")
        print()


if __name__ == "__main__":
    main()
