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
  1. LERNA                      - LER-guided step skipping (the method itself)
  2. Gradient Norm Thresholding - skip backward when ||g|| < threshold
  3. Random Step Skipping       - skip randomly at matched rate
  4. Early Stopping             - HF EarlyStoppingCallback with p=5
  5. Weight Freezing            - freeze weights during LER-detected plateaus
  6. Reduced Total Steps        - train fewer steps (same compute budget)
  7. Cosine Annealing + Restarts - cosine LR schedule with warm restarts

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
  python scripts/run_phase1_2_simple_baselines.py --mode full --wandb

  # Production: all baselines x 8 tasks x 10 seeds (~5 days)
  python scripts/run_phase1_2_simple_baselines.py --mode production --wandb

  # Custom: pick baselines, tasks, seeds
  python scripts/run_phase1_2_simple_baselines.py \
      --baselines lerna grad_norm random_skip early_stop \
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
  - lerna/callbacks/lerna_baseline.py (LERNA baseline callback)
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
os.environ["WANDB_START_METHOD"] = "thread"
os.environ["WANDB_LOG_MODEL"] = "false"

import sys
import json
import time
import argparse
import gc
import logging
import pathlib
import numpy as np
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

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lerna.utils.metrics import LERTracker
from lerna.callbacks.efficiency_callback import PowerTelemetryCallback
from lerna.callbacks.simple_baselines import (
    GradientNormSkippingCallback,
    RandomStepSkippingCallback,
    WeightFreezingCallback,
    ReducedTotalStepsCallback,
    CosineAnnealingWarmRestartsCallback,
)
from lerna.callbacks.lerna_baseline import LERNABaselineCallback
from lerna.utils.checkpoint_compat import safe_load_state_dict
from lerna.utils.sanity import assert_layernorm_trained

# Import bootstrap CI helper and stats
from scripts.phase1_2_bootstrap_ci import bootstrap_ci

# Make scripts/ importable for phase1_2_stats
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

try:
    from phase1_2_stats import run_all_paired_tests
except ImportError:
    run_all_paired_tests = None

logger = logging.getLogger(__name__)


# =============================================================================
# LER Feed Callback: Feeds eval metrics into LERTracker during training
# =============================================================================

class LERFeedCallback(TrainerCallback):
    """Feeds evaluation metrics into LERTracker during training.
    
    FIX (2026-04-15): The LERTracker was never updated during Phase 1.2
    training because nobody called ler_tracker.update(). This caused
    WeightFreezingCallback to never detect plateaus (LER was always None).
    
    This callback hooks into on_evaluate to feed loss, logits, and accuracy
    into the LERTracker, enabling plateau detection for weight_freeze baseline.
    
    It also hooks into on_pre_optimizer_step to capture gradients for
    rho_VG computation when gradients are still live.
    """
    
    def __init__(self, ler_tracker: LERTracker, trainer_ref=None):
        self.ler_tracker = ler_tracker
        self._trainer_ref = trainer_ref
        self._update_count = 0
        self._model = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        if "model" in kwargs:
            self._model = kwargs["model"]
        return control
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Capture gradients for rho_VG while they are live."""
        model = kwargs.get("model", self._model)
        if model is not None:
            self.ler_tracker.capture_step_gradients(model)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Sample LER every 20 training steps (not just at eval time).

        Without this, LER is updated only at eval boundaries (~20 samples
        per run), which is too coarse for hysteresis-based phase detection
        and for paper plots.
        """
        if state.global_step % 20 != 0:
            return control

        trainer = self._trainer_ref
        if trainer is None:
            return control

        logits = getattr(trainer, '_last_real_logits', None)
        if logits is None:
            return control

        # Get most recent training loss from trainer state.log_history.
        last_loss = None
        for entry in reversed(trainer.state.log_history):
            if 'loss' in entry:
                last_loss = entry['loss']
                break
        if last_loss is None:
            return control

        model = kwargs.get("model", self._model)
        self.ler_tracker.update(
            loss=last_loss,
            logits=logits,
            accuracy=None,
            model=model,
            gradients=None,
        )
        self._update_count += 1
        return control
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Feed evaluation metrics into LERTracker."""
        if metrics is None:
            return control
        
        # Get the trainer reference for logits
        trainer = self._trainer_ref
        if trainer is None:
            return control
        
        # Extract eval loss
        eval_loss = metrics.get("eval_loss", 0.0)
        
        # Extract accuracy (task-dependent key)
        accuracy = metrics.get("eval_accuracy",
                    metrics.get("eval_matthews_correlation",
                    metrics.get("eval_pearson",
                    metrics.get("eval_pearsonr", None))))
        
        # Get logits captured by Phase12Trainer.compute_loss.
        logits = getattr(trainer, '_last_real_logits', None)
        if logits is None:
            print(f"  [LERFeed] Skipping update at step {state.global_step}: "
                  f"no logits captured (would have used random fallback).")
            return control
        
        # Get model for velocity computation
        model = kwargs.get("model", self._model)
        
        # Feed into LER tracker
        self.ler_tracker.update(
            loss=eval_loss,
            logits=logits,
            accuracy=accuracy,
            model=model,
            gradients=None,
        )
        self._update_count += 1
        
        # Log LER diagnostics
        diag = self.ler_tracker.get_diagnostics()
        if state.global_step % 50 == 0 or self._update_count <= 3:
            print(f"  [LERFeed] Step {state.global_step}: LER={diag.get('ler', 'None')}, "
                  f"phase={diag.get('phase', 'unknown')}, "
                  f"rho_VG={diag.get('rho_vg', 'None')}, "
                  f"updates={self._update_count}")
        
        return control


# =============================================================================
# Constants (identical to Phase 1.1 for reproducibility)
# =============================================================================

MODEL_NAME = "roberta-base"

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

# Per-task hyperparameters (identical to Phase 1.1)
TASK_HP_OVERRIDES = {
    "rte": {
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 15,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        "init_from_mnli": True,
    },
    "cola": {
        "learning_rate": 1e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 10,
        "metric_for_best_model": "eval_matthews_correlation",
        "greater_is_better": True,
    },
    "mrpc": {
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 10,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
    },
    "stsb": {
        "learning_rate": 2e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 10,
        "metric_for_best_model": "eval_pearson",
        "greater_is_better": True,
    },
}

# Phase 1.1 reference results for comparison.
PHASE_1_1_RESULTS = {
    "sst2":  {"accuracy": 0.9373, "std": 0.0054, "waste_ratio": 0.504},
    "qnli":  {"accuracy": 0.9248, "std": 0.0018, "waste_ratio": 0.558},
    "qqp":   {"accuracy": 0.9081, "std": 0.0022, "waste_ratio": 0.988},
    "mnli":  {"accuracy": 0.8728, "std": 0.0030, "waste_ratio": 0.989},
    "rte":   {"accuracy": 0.7639, "std": 0.0090, "waste_ratio": 0.000},
    "mrpc":  {"accuracy": 0.8831, "std": 0.0066, "waste_ratio": 0.000},
    "cola":  {"accuracy": 0.5780, "std": 0.0055, "waste_ratio": 0.000},
    "stsb":  {"accuracy": 0.9026, "std": 0.0022, "waste_ratio": 0.370},
}

# Baseline definitions - UPDATED with lerna and early_stop
BASELINE_REGISTRY = {
    "lerna": {
        "description": "LERNA: LER-guided step skipping (the method itself)",
        "tests": "Reference point - all baselines should be worse than this",
    },
    "grad_norm": {
        "description": "Skip backward pass when ||g|| < threshold",
        "tests": "Whether LER is better than its simplest component",
    },
    "random_skip": {
        "description": "Skip backward pass randomly at matched rate",
        "tests": "Whether the selection of which steps to skip matters",
    },
    "early_stop": {
        "description": "Early stopping with patience=5",
        "tests": "Whether LERNA captures anything beyond 'stop when not learning'",
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
        return {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
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


def _ensure_wandb_finished():
    """Safely finish any active W&B run."""
    try:
        import wandb
        if wandb.run is not None:
            wandb.finish(quiet=True)
    except ImportError:
        pass


# =============================================================================
# Custom Trainer (captures real logits for LER, same as Phase 1.1)
# =============================================================================

class Phase12Trainer(Trainer):
    """Extended Trainer that captures logits and gradient norms for Phase 1.2.
    
    Phase 1.2 uses POST-HOC gradient replacement (Option B):
    - Full forward+backward runs normally on every step
    - For "skipped" steps, callbacks undo the gradient update and apply
      momentum extrapolation in on_step_end
    - This avoids AMP/GradScaler/multi-GPU compatibility issues
    - Accuracy results are identical to true skipping
    - Energy savings are computed theoretically from skip ratio
    
    Attributes:
        _last_real_logits: Captured logits for LER computation
        _pre_clip_grad_norm: Gradient norm BEFORE clipping (for calibration)
        _step_was_skipped: Set by callbacks in on_step_end to indicate
            this step's gradient update should be replaced with momentum
    """

    def __init__(self, *args, ler_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ler_tracker = ler_tracker
        self._last_real_logits = None
        self._pre_clip_grad_norm: float = 0.0
        self.should_skip_backward: bool = False
        self._skip_count: int = 0
        self._freeze_weights_no_momentum: bool = False  # reset by callbacks each step
        self._microbatch_count: int = 0                  # for grad accumulation gating
        self._param_snapshot: Optional[dict] = None
        self._should_snapshot: bool = False
        self._grad_norm_callback = _GradientNormCaptureCallback(self)
        self.add_callback(self._grad_norm_callback)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        if loss is not None and loss.dim() > 0:
            loss = loss.mean()
        if hasattr(outputs, "logits"):
            self._last_real_logits = outputs.logits.detach()
        elif isinstance(outputs, dict) and "logits" in outputs:
            self._last_real_logits = outputs["logits"].detach()
        return (loss, outputs) if return_outputs else loss

    def _compute_pre_clip_grad_norm(self) -> float:
        """Compute gradient norm BEFORE clipping is applied."""
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.detach().float().norm().item()
                total_norm_sq += param_norm ** 2
        return total_norm_sq ** 0.5

    def snapshot_params(self):
        """Save a snapshot of current parameters for potential rollback."""
        self._param_snapshot = {
            name: param.data.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

    def rollback_and_apply_momentum(self, use_momentum: bool = True):
        """Rollback gradient-based update and optionally apply momentum."""
        if self._param_snapshot is None:
            return
        
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self._param_snapshot:
                    param.data.copy_(self._param_snapshot[name])
        
        if use_momentum and self.optimizer is not None:
            with torch.no_grad():
                for group in self.optimizer.param_groups:
                    group_lr = group.get('lr', self.args.learning_rate)
                    for param in group['params']:
                        if not param.requires_grad or param not in self.optimizer.state:
                            continue
                        p_state = self.optimizer.state[param]
                        if 'momentum_buffer' in p_state:
                            param.data.add_(p_state['momentum_buffer'], alpha=-group_lr)
                        elif 'exp_avg' in p_state:
                            step = p_state.get('step', 1)
                            if isinstance(step, torch.Tensor):
                                step = step.item()
                            step = max(int(step), 1)
                            exp_avg = p_state['exp_avg']
                            beta1 = group.get('betas', (0.9, 0.999))[0]
                            bias_correction = 1 - beta1 ** step
                            if bias_correction > 0:
                                corrected = exp_avg / bias_correction
                            else:
                                corrected = exp_avg
                            param.data.add_(corrected, alpha=-group_lr)
        
        self._param_snapshot = None
        self._skip_count += 1

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to implement true backward-pass skipping.

        FIX: skip path now (a) normalizes loss by gradient_accumulation_steps
        to match the parent class, and (b) only applies momentum
        extrapolation once per accumulation window (i.e. on real
        optimizer-step boundaries), not once per microbatch.
        """
        self._pre_clip_grad_norm = None

        if self.should_skip_backward:
            model.train()
            inputs = self._prepare_inputs(inputs)

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            # Match HF Trainer's loss normalization for grad accumulation.
            accum = max(1, self.args.gradient_accumulation_steps)
            if accum > 1:
                loss = loss / accum

            # Only extrapolate on the LAST microbatch of the accumulation
            # window — that's when a real optimizer step would have run.
            self._microbatch_count += 1
            on_optimizer_boundary = (self._microbatch_count % accum == 0)

            if on_optimizer_boundary:
                if not self._freeze_weights_no_momentum:
                    self._apply_momentum_extrapolation()
                self._skip_count += 1

            return loss.detach()

        # Normal training path
        self._microbatch_count += 1
        loss = super().training_step(model, inputs, num_items_in_batch)
        self._pre_clip_grad_norm = self._compute_pre_clip_grad_norm()
        return loss


class _GradientNormCaptureCallback(TrainerCallback):
    """Internal callback to capture gradient norm before optimizer step."""
    
    def __init__(self, trainer: Phase12Trainer):
        self.trainer = trainer
    
    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return
        
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        
        self.trainer._pre_clip_grad_norm = total_norm_sq ** 0.5


# =============================================================================
# Baseline Factory - UPDATED with lerna and early_stop
# =============================================================================

def create_baseline_callback(
    baseline_name: str,
    task_name: str,
    total_steps: int,
    base_lr: float,
    seed: int,
    ler_tracker: Optional[LERTracker] = None,
    target_skip_rate: float = 0.22,   # A.4: matches measured 22.2% waste excl. STS-B
    wandb_enabled: bool = True,
) -> Tuple[Optional[Any], Optional[int]]:
    """
    Create the appropriate callback for a given baseline.

    Returns:
        (callback_instance, early_stopping_patience_override)
    """
    if baseline_name == "lerna":
        if ler_tracker is None:
            raise ValueError("lerna baseline requires ler_tracker")
        return LERNABaselineCallback(
            ler_tracker=ler_tracker,
            warmup_steps=20,  # Lower for smoke test
        ), None

    elif baseline_name == "grad_norm":
        calibration_steps = max(20, min(200, total_steps // 5))
        recalibrate_every = max(100, total_steps // 2)
        min_calibration_samples = max(10, calibration_steps // 4)
        return GradientNormSkippingCallback(
            target_skip_rate=target_skip_rate,
            calibration_steps=calibration_steps,
            recalibrate_every=recalibrate_every,
            min_step=0,
            wandb_enabled=wandb_enabled,
            min_calibration_samples=min_calibration_samples,
        ), None

    elif baseline_name == "random_skip":
        return RandomStepSkippingCallback(
            target_skip_rate=target_skip_rate,
            min_step=100,
            seed=seed,
            wandb_enabled=wandb_enabled,
        ), None

    elif baseline_name == "early_stop":
        return None, 5

    elif baseline_name.startswith("early_stop_p"):
        import re
        match = re.match(r"early_stop_p(\d+)", baseline_name)
        if not match:
            raise ValueError(f"Invalid early_stop baseline name: '{baseline_name}'")
        patience = int(match.group(1))
        return None, patience

    elif baseline_name == "weight_freeze":
        if ler_tracker is None:
            raise ValueError("weight_freeze baseline requires ler_tracker")
        # Don't pass threshold — WeightFreezingCallback now reads it from
        # ler_tracker.task_calibration so it stays in sync with the
        # phase detector.
        return WeightFreezingCallback(
            ler_tracker=ler_tracker,
            min_step=100,
            wandb_enabled=wandb_enabled,
        ), None

    elif baseline_name == "reduced_steps":
        return ReducedTotalStepsCallback(
            reduction_fraction=target_skip_rate,
            total_steps=total_steps,
            wandb_enabled=wandb_enabled,
        ), None

    elif baseline_name == "cosine_restarts":
        return CosineAnnealingWarmRestartsCallback(
            T_0=max(total_steps // 10, 50),
            T_mult=2,
            eta_min=1e-7,
            base_lr=base_lr,
            wandb_enabled=wandb_enabled,
        ), None

    else:
        raise ValueError(f"Unknown baseline: {baseline_name}")


# =============================================================================
# Core: Run a single baseline experiment
# =============================================================================

def run_single_baseline_experiment(
    baseline_name: str,
    task_name: str,
    seed: int,
    profile: str,
    base_output_dir: str,
    use_wandb: bool = False,
    max_samples_override: Optional[int] = None,
    run_idx: int = 0,
    total_runs: int = 0,
    wandb_project: str = "lerna-phase1.2",
    wandb_group: str = "phase1.2-baselines",
    target_skip_rate: float = 0.22,   # A.4: matches measured 22.2% waste excl. STS-B
) -> dict:
    """Run a single Phase 1.2 baseline experiment."""
    hw_cfg = get_training_config(profile)
    if max_samples_override is not None:
        hw_cfg["max_samples"] = max_samples_override

    task_hp = TASK_HP_OVERRIDES.get(task_name, {})
    lr = task_hp.get("learning_rate", 2e-5)
    num_epochs = task_hp.get("num_epochs", 3)
    warmup_ratio = task_hp.get("warmup_ratio", 0.1)
    default_patience = task_hp.get("early_stopping_patience", 5)
    metric_for_best_model = task_hp.get("metric_for_best_model", "eval_loss")
    greater_is_better = task_hp.get("greater_is_better", False)
    init_from_mnli = task_hp.get("init_from_mnli", False)

    run_id = f"{baseline_name}/{task_name}_s{seed}_lr{lr:.0e}"
    output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb:
        import wandb
        _ensure_wandb_finished()
        wandb.init(
            project=wandb_project,
            name=f"{baseline_name}-{task_name}-s{seed}",
            group=wandb_group,
            job_type="baseline",
            tags=[baseline_name, task_name, f"seed-{seed}", "phase1.2"],
            reinit=True,
            config={
                "phase": "1.2",
                "baseline": baseline_name,
                "task": task_name,
                "seed": seed,
                "learning_rate": lr,
                "model": MODEL_NAME,
                "profile": profile,
                "target_skip_rate": target_skip_rate,
            },
        )
        try:
            wandb.define_metric("*", step_metric="train/global_step", step_sync=False)
        except Exception:
            pass

    print(f"\n{'=' * 70}")
    print(f"  Phase 1.2 | Baseline: {baseline_name} | Task: {task_name} | Seed: {seed}")
    print(f"  LR: {lr} | Epochs: {num_epochs} | Profile: {profile}")
    if run_idx and total_runs:
        print(f"  Progress: run {run_idx}/{total_runs}")
    print(f"  Output: {output_dir}")
    print(f"{'=' * 70}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg = GLUE_TASK_CONFIG[task_name]

    mnli_checkpoint_dir = os.path.join(
        os.path.dirname(base_output_dir), "baseline", "mnli_finetuned"
    )
    if init_from_mnli and os.path.exists(mnli_checkpoint_dir):
        print(f"  [MNLI Transfer] Loading from {mnli_checkpoint_dir}")
        mnli_model = AutoModelForSequenceClassification.from_pretrained(mnli_checkpoint_dir)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=cfg["num_labels"],
        )
        encoder_state = {k: v for k, v in mnli_model.state_dict().items()
                         if "classifier" not in k and "pooler" not in k}
        safe_load_state_dict(model, encoder_state, strict=False)
        del mnli_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        if init_from_mnli:
            print(f"  [MNLI Transfer] WARNING: No checkpoint at {mnli_checkpoint_dir}, using pretrained")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=cfg["num_labels"],
        )

    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"]
    )
    print(f"  Train: {len(train_ds)} samples | Eval: {len(eval_ds)} samples")

    steps_per_epoch = max(1, len(train_ds) // (
        hw_cfg["per_device_train_batch_size"] * hw_cfg["gradient_accumulation_steps"]
    ))
    total_steps = steps_per_epoch * num_epochs
    eval_steps = max(total_steps // 20, 10)

    ler_tracker = LERTracker(task=task_name, window_size=5)

    baseline_callback, patience_override = create_baseline_callback(
        baseline_name=baseline_name,
        task_name=task_name,
        total_steps=total_steps,
        base_lr=lr,
        seed=seed,
        ler_tracker=ler_tracker,
        target_skip_rate=target_skip_rate,
        wandb_enabled=use_wandb,
    )

    if patience_override is not None:
        es_patience = patience_override
    elif baseline_name == "reduced_steps":
        es_patience = 999999
        print(f"  [Phase1.2] Disabled early stopping for reduced_steps baseline")
    else:
        es_patience = default_patience

    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
    )

    ler_feed_callback = LERFeedCallback(ler_tracker=ler_tracker)

    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=es_patience),
        power_callback,
        ler_feed_callback,
    ]
    if baseline_callback is not None:
        callbacks.append(baseline_callback)

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
        save_safetensors=True,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=max(eval_steps // 5, 1),
        report_to="wandb" if use_wandb else "none",
        run_name=f"{baseline_name}-{task_name}-s{seed}" if use_wandb else None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics(task_name)

    trainer = Phase12Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        ler_tracker=ler_tracker,
        callbacks=callbacks,
    )

    if baseline_callback is not None and hasattr(baseline_callback, '_model'):
        baseline_callback._model = model
        print(f'  [Phase1.2] Model injected into {baseline_callback.baseline_name}: {type(model).__name__}')

    if baseline_callback is not None:
        baseline_callback._trainer = trainer
        print(f"  [Phase1.2] Trainer linked to {baseline_callback.baseline_name}")

    ler_feed_callback._trainer_ref = trainer
    ler_feed_callback._model = model

    start_time = time.time()
    print(f"  Starting training: ~{total_steps} steps, eval every {eval_steps} steps")
    print(f"  Baseline: {BASELINE_REGISTRY[baseline_name]['description']}")
    print(f"  Early stopping patience: {es_patience}")
    if baseline_name == "reduced_steps":
        print(f"  Reduced steps max: {baseline_callback.max_steps} (of {total_steps})")

    train_result = trainer.train()
    train_time = time.time() - start_time

    # Verify load_best_model_at_end did not corrupt LayerNorm parameters.
    assert_layernorm_trained(model)

    print(f"  Evaluating best model...")
    eval_result = trainer.evaluate()

    if baseline_callback is not None and hasattr(baseline_callback, 'get_activation_summary'):
        activation = baseline_callback.get_activation_summary()
        print(f"\n  --- Baseline Activation Check ---")
        for k, v in activation.items():
            print(f"    {k}: {v}")
        
        if not activation.get('activated', True):
            if baseline_name in ('grad_norm', 'random_skip', 'weight_freeze', 'lerna'):
                print(f"  *** WARNING: {baseline_name} baseline did NOT activate! ***")
            elif baseline_name == 'cosine_restarts':
                print(f"  *** WARNING: cosine_restarts did NOT override any LR! ***")
    
    print(f"  LER feed updates: {ler_feed_callback._update_count}")
    ler_diag = ler_tracker.get_diagnostics()
    print(f"  LER tracker: n_steps={ler_diag.get('n_steps', 0)}, "
          f"ler={ler_diag.get('ler', 'None')}, phase={ler_diag.get('phase', 'unknown')}")

    avg_power = (
        float(np.mean([s["power_w"] for s in power_callback._power_samples]))
        if power_callback._power_samples else 0
    )

    baseline_stats = {}
    if baseline_callback is not None and hasattr(baseline_callback, 'steps_skipped'):
        baseline_stats = {
            "steps_skipped": baseline_callback.steps_skipped,
            "skip_ratio": baseline_callback.steps_skipped / max(trainer.state.global_step, 1),
            "energy_saved_kwh": baseline_callback.total_energy_saved,
        }

    primary_metric_key = {
        "accuracy": "eval_accuracy",
        "matthews_correlation": "eval_matthews_correlation",
        "pearsonr": "eval_pearson",
    }.get(cfg["metric"], "eval_accuracy")
    
    primary_value = eval_result.get(
        primary_metric_key,
        eval_result.get("eval_pearsonr", eval_result.get("eval_accuracy", 0))
    )

    results = {
        "phase": "1.2",
        "baseline": baseline_name,
        "task": task_name,
        "seed": seed,
        "learning_rate": lr,
        "model": MODEL_NAME,
        "profile": profile,
        "primary_metric": primary_value,
        "primary_metric_name": primary_metric_key,
        "eval_metrics": eval_result,
        "train_loss": train_result.training_loss,
        "train_runtime_s": train_time,
        "train_steps": trainer.state.global_step,
        "energy_kwh": power_callback.cumulative_kwh,
        "power_avg_watts": avg_power,
        "early_stopping_patience": es_patience,
        "baseline_stats": baseline_stats,
        "target_skip_rate": target_skip_rate,
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  --- Results ---")
    print(f"  {primary_metric_key}: {primary_value:.4f}")
    print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
    print(f"  Time: {train_time:.1f}s")
    print(f"  Steps: {trainer.state.global_step}")
    if baseline_stats:
        print(f"  Skip ratio: {baseline_stats.get('skip_ratio', 0):.3f}")
    print(f"  Saved: {results_path}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if use_wandb:
        _ensure_wandb_finished()

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

    agg = defaultdict(lambda: defaultdict(list))
    for r in successful:
        key = (r["baseline"], r["task"])
        agg[key]["metric"].append(r["primary_metric"])
        agg[key]["energy"].append(r["energy_kwh"])
        agg[key]["time"].append(r["train_runtime_s"])
        agg[key]["steps"].append(r["train_steps"])
        skip_ratio = r.get("baseline_stats", {}).get("skip_ratio", 0)
        agg[key]["skip_ratio"].append(skip_ratio)

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
            
            metric_mean, ci_low, ci_high = bootstrap_ci(metrics)

            ref = PHASE_1_1_RESULTS.get(task, {})
            ref_acc = ref.get("accuracy", 0)
            delta = mean_metric - ref_acc

            summary_rows.append({
                "baseline": baseline,
                "task": task,
                "n_seeds": n,
                "mean_metric": float(mean_metric),
                "std_metric": float(std_metric),
                "ci_low": float(ci_low),
                "ci_high": float(ci_high),
                "mean_energy_kwh": float(mean_energy),
                "mean_time_s": float(mean_time),
                "mean_skip_ratio": float(mean_skip),
                "phase1_1_metric": float(ref_acc),
                "delta_vs_phase1_1": float(delta),
            })

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

    comparison_lines = []
    header = (
        f"{'Baseline':<20} {'Task':<8} {'Seeds':>5} "
        f"{'Metric':>10} {'CI95':>14} {'Energy':>10} "
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
        ci_str = f"[{row['ci_low']:.4f}, {row['ci_high']:.4f}]"
        line = (
            f"{row['baseline']:<20} {row['task']:<8} {row['n_seeds']:>5} "
            f"{row['mean_metric']:>10.4f} {ci_str:>14} "
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

    comparison_path = os.path.join(output_dir, "phase1_2_comparison.txt")
    with open(comparison_path, "w") as f:
        f.write(comparison_text)
    print(f"  Comparison table saved: {comparison_path}")
    
    # Run paired significance tests (LERNA vs each baseline)
    if "lerna" in baselines and len(successful) > 0:
        sig_results = {}
        baselines_with_data = [b for b in baselines
                               if sum(1 for r in successful if r.get("baseline") == b) >= 2]
        if run_all_paired_tests is not None and len(baselines_with_data) >= 2:
            sig_results = run_all_paired_tests(successful, baselines_with_data, tasks)
        sig_path = os.path.join(output_dir, "phase1_2_significance.json")
        with open(sig_path, "w") as f:
            json.dump(sig_results, f, indent=2, default=str)


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
  python scripts/run_phase1_2_simple_baselines.py --baselines lerna grad_norm --tasks sst2 mrpc
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
        choices=[
            "lerna",
            "grad_norm",
            "random_skip",
            "early_stop",
            "early_stop_p3",
            "early_stop_p5",
            "early_stop_p10",
            "early_stop_p20",
            "weight_freeze",
            "reduced_steps",
            "cosine_restarts",
        ],
        help="Which baselines to run (default: all for the chosen mode)",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="GLUE tasks to run")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds")
    parser.add_argument(
        "--output-dir", default="./experiments/phase1_2_baselines",
        help="Base output directory",
    )
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
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
        "--target-skip-rate", type=float, default=0.22,
        help="Target skip rate for baselines that need it (A.4 default: 0.22)",
    )
    parser.add_argument("--smoke-test", action="store_true",
                        help="Allow LERNA to not activate without failing")
    parser.add_argument("--lerna-warmup-steps", type=int, default=None,
                        help="Override LERNA warmup_steps")
    args = parser.parse_args()

    profile = detect_device_profile()

    if args.mode == "smoke":
        baselines = args.baselines if args.baselines else ["lerna"]
        tasks = args.tasks if args.tasks else ["mrpc"]
        seeds = args.seeds if args.seeds else [42]
    elif args.mode == "quick":
        baselines = args.baselines if args.baselines else list(BASELINE_REGISTRY.keys())
        tasks = args.tasks if args.tasks else ["sst2", "mrpc"]
        seeds = args.seeds if args.seeds else [42, 43]
    elif args.mode == "full":
        baselines = args.baselines if args.baselines else list(BASELINE_REGISTRY.keys())
        tasks = args.tasks if args.tasks else ["sst2", "mrpc", "rte", "qnli"]
        seeds = args.seeds if args.seeds else list(range(42, 47))
    else:  # production
        baselines = args.baselines if args.baselines else list(BASELINE_REGISTRY.keys())
        tasks = args.tasks if args.tasks else list(GLUE_TASK_CONFIG.keys())
        seeds = args.seeds if args.seeds else list(range(42, 52))

    if args.max_samples is not None:
        max_samples = args.max_samples
    elif args.unlimited:
        max_samples = None
    elif profile == "server":
        max_samples = None
    else:
        max_samples = 2000

    total_runs = len(baselines) * len(tasks) * len(seeds)
    wandb_group = args.wandb_group or f"phase1.2-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

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
    if args.wandb:
        print(f"  W&B project: {args.wandb_project}")
        print(f"  W&B group: {wandb_group}")
    print(f"{'=' * 70}")
    print(f"\n  Baselines to evaluate:")
    for b in baselines:
        info = BASELINE_REGISTRY[b]
        print(f"    - {b}: {info['description']}")
        print(f"      Tests: {info['tests']}")
    print()

    est_per_run = 360 if profile == "server" else 600
    est_total = total_runs * est_per_run
    print(f"  Estimated total time: ~{timedelta(seconds=est_total)}")
    print()

    if args.wandb:
        _ensure_wandb_finished()

    all_results = []
    run_idx = 0
    overall_start = time.time()

    for baseline_name in baselines:
        for task in tasks:
            for seed in seeds:
                run_idx += 1

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
                        use_wandb=args.wandb,
                        max_samples_override=max_samples,
                        run_idx=run_idx,
                        total_runs=total_runs,
                        wandb_project=args.wandb_project,
                        wandb_group=wandb_group,
                        target_skip_rate=args.target_skip_rate,
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
                    if args.wandb:
                        _ensure_wandb_finished()

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

    if successful:
        print(f"\n  === VERDICT ===")
        print(f"  Compare these results against Phase 1.1 baselines.")
        print(f"  If LERNA outperforms ALL baselines on accuracy-energy tradeoff,")
        print(f"  proceed to Phase 1.3 (Component Ablation).")
        print(f"  If any baseline matches LERNA, investigate before proceeding.")
        print()


if __name__ == "__main__":
    main()