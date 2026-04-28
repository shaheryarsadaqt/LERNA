#!/usr/bin/env python3
"""
LERNA Baseline: ModernBERT-base on GLUE with FULL diagnostics.

Enhancements over original:
  1. GSNR tracking (Gradient Signal-to-Noise Ratio) per layer and global
  2. Per-layer gradient analysis (norm distributions, dead neurons, saturation)
  3. Waste quantification with 95% confidence intervals
  4. Real logits passed to LER tracker (not dummy)
  5. Rich W&B visualizations (heatmaps, training dynamics, layer-wise analysis)
  6. Gradient norm distribution tracking
  7. Learning rate vs loss correlation
  8. Phase transition detection visualization
  9. load_best_model_at_end=True (evaluate best model, not last)
  10. ETA/progress estimation with time remaining

Usage:
  # Smoke test on RTX 3050 (1 seed, SST-2 only, small subset)
  python scripts/run_baseline_glue.py --mode smoke

  # Full baseline on RTX 5090 (3 seeds x 8 tasks, 25k samples/task)
  python scripts/run_baseline_glue.py --mode full --num-seeds 3

  # Full baseline 10 seeds (production)
  python scripts/run_baseline_glue.py --mode full --wandb

  # Full data, no cap
  python scripts/run_baseline_glue.py --mode full --unlimited --wandb

  # Custom: specific tasks and seeds
  python scripts/run_baseline_glue.py --tasks sst2 rte mrpc --seeds 42 43 44
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"
# NOTE: WANDB_LOG_MODEL left unset so W&B can log checkpoint artifacts
# when explicitly requested. The orchestrator controls this via env vars.

import sys
import json
import time
import argparse
import gc
import math
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import torch

# Conditionally disable dynamo (only available in PyTorch 2.4+)
try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass

# Conditionally disable SDP kernels (only available on Ampere+ GPUs)
# V100 (Volta) does not support these APIs
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
)
from datasets import load_dataset
import evaluate


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

# Per-task hyperparameter overrides for small datasets.
# Keys override the defaults (lr=2e-5, num_epochs=3, warmup_ratio=0.1, early_stopping_patience=5).
# Only tasks listed here get overrides; all others use the global defaults.
#
# FIX: Tuned based on GLUE benchmark analysis (2026-03-06):
#   - RTE: MNLI transfer + lower LR + more epochs (was 60.8%, target >70%)
#   - CoLA: Lower LR + more epochs + MCC-based model selection (was 59.4 MCC, target >65)
#   - MRPC: Lower LR + more epochs + F1-based model selection (was 86.2%, target >89%)
TASK_HP_OVERRIDES = {
    "rte": {
        "learning_rate": 2e-5,
        "num_epochs": 20,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 15,
        "metric_for_best_model": "eval_accuracy",
        "greater_is_better": True,
        # Standard practice: initialize from MNLI-finetuned model for RTE
        # (Devlin et al. 2019, Liu et al. 2019, Wang et al. 2019)
        "init_from_mnli": True,
    },
    "cola": {
        "learning_rate": 1e-5,
        "num_epochs": 10,
        "warmup_ratio": 0.1,
        "early_stopping_patience": 10,
        # CoLA uses MCC as primary metric; model selection must use MCC, not loss
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
        # STS-B is a regression task; model selection must use Pearson, not loss
        "metric_for_best_model": "eval_pearson",
        "greater_is_better": True,
    },
}

MODEL_NAME = "roberta-base"


# ═══════════════════════════════════════════════════════════════════════
# GSNR Tracker (NEW)
# ═══════════════════════════════════════════════════════════════════════

class _EMAWelfordAccumulator:
    """EMA-Welford accumulator for smooth online GSNR estimation.

    Uses exponentially weighted moving averages instead of the discontinuous
    halving approach.  Each new gradient's contribution decays smoothly with
    factor alpha = 2 / (window_size + 1), giving an effective window equal
    to *window_size* samples.

    Also tracks E[||g||^2] (Fisher Information trace) as a byproduct.

    Memory: O(dim) regardless of window size.
    """

    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.alpha = 2.0 / (window_size + 1)  # EMA decay factor
        self.n = 0
        self._ema_mean = None      # EMA of gradient (1-D tensor, CPU)
        self._ema_var = None       # EMA of (g - mean)^2 per element
        self._ema_grad_sq = 0.0    # EMA of ||g||^2 (scalar, for Fisher Info)

    def update(self, grad_vec: torch.Tensor):
        """Incorporate a new gradient vector (already on CPU, float32)."""
        self.n += 1
        if self._ema_mean is None:
            self._ema_mean = grad_vec.clone()
            self._ema_var = torch.zeros_like(grad_vec)
            self._ema_grad_sq = grad_vec.norm().item() ** 2
            return

        # Update EMA mean
        delta = grad_vec - self._ema_mean
        self._ema_mean += self.alpha * delta

        # Update EMA variance: Var_ema = (1-alpha) * (Var_ema + alpha * delta^2)
        delta2 = grad_vec - self._ema_mean
        self._ema_var = (1 - self.alpha) * (self._ema_var + self.alpha * delta * delta2)

        # Update Fisher Information trace: EMA of ||g||^2
        g_sq = grad_vec.norm().item() ** 2
        self._ema_grad_sq = (1 - self.alpha) * self._ema_grad_sq + self.alpha * g_sq

    def compute_gsnr(self) -> float:
        """Return GSNR = ||E[g]||^2 / sum(Var[g_i])."""
        if self.n < 2 or self._ema_mean is None:
            return 0.0
        signal = self._ema_mean.norm().item() ** 2
        noise = self._ema_var.sum().item()
        return signal / (noise + 1e-10)

    def compute_fisher_info(self) -> float:
        """Return trace of empirical Fisher Information: E[||g||^2].

        This is a theoretically grounded metric (information geometry)
        that NeurIPS/ICLR reviewers expect when discussing gradient
        signal analysis.  It measures the expected curvature of the
        loss landscape.
        """
        if self.n < 1:
            return 0.0
        return self._ema_grad_sq

    @property
    def ready(self) -> bool:
        return self.n >= 2


class GSNRTracker:
    """Gradient Signal-to-Noise Ratio tracker (memory-safe).

    Uses Welford's online algorithm so memory is O(total_params) regardless
    of window size, instead of the previous O(total_params * window_size)
    which caused OOM on large models like ModernBERT.
    """

    def __init__(self, model, window_size=50):
        self.window_size = window_size
        self.layer_names = []
        self.layer_param_map = {}  # layer_name -> list of param names

        # Discover layer structure
        self._discover_layers(model)

        # Per-layer EMA-Welford accumulators (O(layer_params) each)
        self._layer_accum = {}
        for ln in self.layer_names:
            self._layer_accum[ln] = _EMAWelfordAccumulator(window_size)
        self._global_accum = _EMAWelfordAccumulator(window_size)

        # GSNR history (scalar values only, negligible memory)
        self.gsnr_per_layer_history = defaultdict(list)
        self.gsnr_global_history = []

        # Gradient norm history (scalars)
        self.grad_norm_history = defaultdict(list)
        self.global_grad_norm_history = []

        # Step counter
        self.step = 0

    def _discover_layers(self, model):
        """Discover named layers for per-layer analysis."""
        layer_params = defaultdict(list)
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            parts = name.split(".")
            if "layer" in parts:
                idx = parts.index("layer")
                if idx + 1 < len(parts):
                    layer_name = ".".join(parts[:idx + 2])
                else:
                    layer_name = ".".join(parts[:3])
            elif "embeddings" in parts:
                layer_name = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts[:2])
            elif "classifier" in parts or "head" in parts:
                layer_name = ".".join(parts[:2]) if len(parts) >= 2 else parts[0]
            else:
                layer_name = ".".join(parts[:3]) if len(parts) >= 3 else ".".join(parts[:2])
            layer_params[layer_name].append(name)

        self.layer_names = sorted(layer_params.keys())
        self.layer_param_map = dict(layer_params)

    def capture_scalar_norms(self, model):
        """Lightweight: capture only scalar gradient norms (no Welford update).

        Call this frequently (every step or every few steps) to feed the
        PhaseTransitionDetector with high-resolution gradient norm data
        without the cost of full EMA-Welford accumulation.
        """
        global_norm_sq = 0.0
        has_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                pnorm = param.grad.detach().float().norm().item()
                global_norm_sq += pnorm ** 2
                has_grad = True
                # Per-layer norm (find which layer this param belongs to)
                for layer_name, param_names in self.layer_param_map.items():
                    if name in param_names:
                        self.grad_norm_history[layer_name].append(pnorm)
                        if len(self.grad_norm_history[layer_name]) > self.window_size * 10:
                            self.grad_norm_history[layer_name] = self.grad_norm_history[layer_name][-self.window_size * 5:]
                        break
        if has_grad:
            self.global_grad_norm_history.append(global_norm_sq ** 0.5)
            if len(self.global_grad_norm_history) > self.window_size * 10:
                self.global_grad_norm_history = self.global_grad_norm_history[-self.window_size * 5:]

    def capture_gradients(self, model):
        """Full capture: update EMA-Welford accumulators + scalar norms.

        Call this less frequently (at eval intervals) since it builds
        concatenated gradient vectors for the Welford update.
        """
        self.step += 1

        # Build a name -> grad lookup once to avoid repeated iteration
        grad_lookup = {}
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                grad_lookup[name] = param.grad

        # Per-layer accumulation
        for layer_name, param_names in self.layer_param_map.items():
            layer_grads = []
            for pn in param_names:
                if pn in grad_lookup:
                    layer_grads.append(grad_lookup[pn].detach().float().flatten())
            if layer_grads:
                layer_grad_vec = torch.cat(layer_grads)
                grad_norm = layer_grad_vec.norm().item()
                self.grad_norm_history[layer_name].append(grad_norm)
                if len(self.grad_norm_history[layer_name]) > self.window_size * 10:
                    self.grad_norm_history[layer_name] = self.grad_norm_history[layer_name][-self.window_size * 5:]
                self._layer_accum[layer_name].update(layer_grad_vec.cpu())
                del layer_grad_vec

        # Global accumulation
        all_grads = []
        for name in sorted(grad_lookup.keys()):
            all_grads.append(grad_lookup[name].detach().float().flatten())
        if all_grads:
            global_grad = torch.cat(all_grads)
            self.global_grad_norm_history.append(global_grad.norm().item())
            if len(self.global_grad_norm_history) > self.window_size * 10:
                self.global_grad_norm_history = self.global_grad_norm_history[-self.window_size * 5:]
            self._global_accum.update(global_grad.cpu())
            del global_grad
        del all_grads

    def compute_gsnr(self):
        """Compute GSNR for all layers and globally (memory-safe)."""
        results = {}
        try:
            for layer_name in self.layer_names:
                acc = self._layer_accum.get(layer_name)
                if acc is not None and acc.ready:
                    gsnr = acc.compute_gsnr()
                    results[layer_name] = gsnr
                    self.gsnr_per_layer_history[layer_name].append(gsnr)

            if self._global_accum.ready:
                global_gsnr = self._global_accum.compute_gsnr()
                results["__global__"] = global_gsnr
                self.gsnr_global_history.append(global_gsnr)
        except (RuntimeError, MemoryError) as e:
            print(f"  [GSNR warn] compute_gsnr failed: {e}")
        return results

    def compute_fisher_info(self):
        """Compute empirical Fisher Information trace per layer and globally.

        FI = E[||g||^2] measures the expected curvature of the loss surface.
        It is a theoretically grounded metric from information geometry that
        complements GSNR for NeurIPS/ICLR-level analysis.

        Returns dict with per-layer and global Fisher Information values.
        """
        results = {}
        try:
            for layer_name in self.layer_names:
                acc = self._layer_accum.get(layer_name)
                if acc is not None and acc.n >= 1:
                    results[layer_name] = acc.compute_fisher_info()
            if self._global_accum.n >= 1:
                results["__global__"] = self._global_accum.compute_fisher_info()
        except Exception as e:
            print(f"  [Fisher warn] compute_fisher_info failed: {e}")
        return results

    def get_gradient_norm_stats(self):
        """Get gradient norm statistics per layer."""
        stats = {}
        for layer_name in self.layer_names:
            norms = self.grad_norm_history.get(layer_name, [])
            if not norms:
                continue
            recent = norms[-self.window_size:]
            stats[layer_name] = {
                "mean": float(np.mean(recent)),
                "std": float(np.std(recent)),
                "min": float(np.min(recent)),
                "max": float(np.max(recent)),
                "current": float(recent[-1]),
            }
        return stats

    def get_summary(self):
        """Get full GSNR summary."""
        gsnr = self.compute_gsnr()
        grad_stats = self.get_gradient_norm_stats()
        return {
            "gsnr_per_layer": {k: v for k, v in gsnr.items() if k != "__global__"},
            "gsnr_global": gsnr.get("__global__"),
            "grad_norm_stats": grad_stats,
            "step": self.step,
            "num_layers_tracked": len(self.layer_names),
        }


# ═══════════════════════════════════════════════════════════════════════
# Plateau Detectors — Theil–Sen slope + Mann–Kendall + GSNR collapse
# ═══════════════════════════════════════════════════════════════════════

def _theil_sen_slope(y):
    """Robust slope (median of pairwise slopes). O(W^2); W ≤ 64 ⇒ fast."""
    n = len(y)
    if n < 3:
        return 0.0
    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dx = j - i
            slopes.append((y[j] - y[i]) / dx)
    return float(np.median(slopes)) if slopes else 0.0


def _mann_kendall_pvalue(y):
    """Two-sided Mann–Kendall p-value (no scipy needed).
    Large p ⇒ fail to reject 'no monotone trend' (= plateau evidence).
    """
    n = len(y)
    if n < 4:
        return 1.0
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            d = y[j] - y[i]
            if d > 0:
                s += 1
            elif d < 0:
                s -= 1
    var_s = n * (n - 1) * (2 * n + 5) / 18.0
    if var_s <= 0:
        return 1.0
    if s > 0:
        z = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / math.sqrt(var_s)
    else:
        z = 0.0
    p = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(z) / math.sqrt(2.0))))
    return float(max(0.0, min(1.0, p)))


class SlopePlateauDetector:
    """Dual-signal train-loss plateau detector.

        T1: Theil–Sen slope of log(EMA(loss)) over the trailing window
            is below ε_slope (in nat-log per 1000 SGD steps).
        T2: Mann–Kendall on raw recent losses fails to reject the null
            'no monotone trend' (p > α_MK).
        T3: GSNR has collapsed to < γ · gsnr_peak for ≥ K consecutive
            evaluations (fed externally via update_gsnr).

    plateau ⇔ (T1 ∧ T2) ∨ (T1 ∧ T3)
    """

    def __init__(self,
                 window_W=32,
                 ema_alpha=0.061,
                 eps_slope_per_1k=5e-3,
                 alpha_mk=0.10,
                 gsnr_gamma=0.10,
                 gsnr_K=3):
        self.W = int(window_W)
        self.ema_alpha = float(ema_alpha)
        self.eps_slope_per_1k = float(eps_slope_per_1k)
        self.alpha_mk = float(alpha_mk)
        self.gsnr_gamma = float(gsnr_gamma)
        self.gsnr_K = int(gsnr_K)

        self._raw = []
        self._log_ema = []
        self._ema = None

        self._gsnr_peak = 0.0
        self._gsnr_below = 0

        self.t1_fired_at = None
        self.t2_fired_at = None
        self.t3_fired_at = None
        self.plateau_step = None

    def update_loss(self, step, loss):
        if self._ema is None:
            self._ema = float(loss)
        else:
            self._ema = self.ema_alpha * float(loss) + (1 - self.ema_alpha) * self._ema
        self._log_ema.append(math.log(max(self._ema, 1e-12)))
        self._raw.append(float(loss))

        if len(self._log_ema) > self.W * 4:
            self._log_ema = self._log_ema[-self.W * 2:]
            self._raw = self._raw[-self.W * 2:]

        if len(self._log_ema) < self.W:
            return

        slope_per_step = _theil_sen_slope(self._log_ema[-self.W:])
        slope_per_1k = abs(slope_per_step) * 1000.0
        t1 = slope_per_1k < self.eps_slope_per_1k
        if t1 and self.t1_fired_at is None:
            self.t1_fired_at = step

        p_mk = _mann_kendall_pvalue(self._raw[-self.W:])
        t2 = p_mk > self.alpha_mk
        if t2 and self.t2_fired_at is None:
            self.t2_fired_at = step

        t3 = self.t3_fired_at is not None
        if (t1 and t2) or (t1 and t3):
            if self.plateau_step is None:
                self.plateau_step = step

    def update_gsnr(self, step, gsnr_value):
        if gsnr_value is None:
            return
        g = float(gsnr_value)
        if g > self._gsnr_peak:
            self._gsnr_peak = g
        threshold = self.gsnr_gamma * self._gsnr_peak
        if self._gsnr_peak > 0 and g < threshold:
            self._gsnr_below += 1
        else:
            self._gsnr_below = 0
        if self._gsnr_below >= self.gsnr_K and self.t3_fired_at is None:
            self.t3_fired_at = step


# ═══════════════════════════════════════════════════════════════════════
# Waste Quantifier (NEW)
# ═══════════════════════════════════════════════════════════════════════

class WasteQuantifier:
    """Quantifies computational waste during training with confidence intervals.
    
    Waste is defined as compute spent AFTER the model has converged (plateau),
    not step-by-step loss fluctuations (which are normal SGD noise).
    
    Uses EMA-smoothed loss to detect plateau onset, then computes:
        waste_ratio = (total_steps - plateau_step) / total_steps
    
    Also preserves the raw per-step improving ratio as a secondary metric
    for backward compatibility with existing W&B panels.
    """

    def __init__(self, ema_alpha=0.05, plateau_patience=50, plateau_min_improvement=0.001,
                 min_steps_before_plateau=100,
                 detector="dual_signal", slope_kwargs=None):
        """
        Args:
            ema_alpha: Smoothing factor for EMA loss (lower = smoother).
            plateau_patience: Number of steps with no EMA improvement to declare plateau.
            plateau_min_improvement: Minimum relative improvement in EMA loss to count
                                     as "still improving" (0.001 = 0.1%).
            min_steps_before_plateau: Minimum number of steps before plateau detection
                                     activates. Prevents false early detection on
                                     large-dataset tasks where EMA needs warmup time.
            detector: Which plateau detector to use ("dual_signal", "relative", or "slope").
            slope_kwargs: Keyword arguments for SlopePlateauDetector.
        """
        self.ema_alpha = ema_alpha
        self.plateau_patience = plateau_patience
        self.plateau_min_improvement = plateau_min_improvement
        self.min_steps_before_plateau = min_steps_before_plateau
        self.detector_name = str(detector)
        self.epsilon_slope = 5e-3
        self.alpha_mk = 0.10
        self._plateau_window = plateau_patience

        self.loss_history = []
        self.grad_norm_history = []
        self.energy_per_step = []
        self.step_times = []

        # True step counter (not affected by history capping)
        self._total_steps_seen = 0
        self._sgd_step_at_obs = []  # parallel to loss_history; SGD step at which that loss was logged
        self._eval_greater_is_better = True  # default; updated via set_eval_direction

        # EMA-based plateau detection state
        self._ema_loss = None
        self._best_ema_loss = None
        self._steps_since_ema_improvement = 0
        self._plateau_step = None  # Step at which plateau was first detected

        # Dual-signal change-point detection state
        self._log_ema_loss_history = []  # log(EMA(loss)) for Theil-Sen
        self._gsnr_history = []  # GSNR_t for T3 test
        self._consecutive_gsnr_low = 0  # Counter for K=3 consecutive evals

        # Legacy per-step improving flags (kept for backward compat / raw metric)
        self.improving_steps = []  # True/False per step
        self.waste_reasons = defaultdict(int)

        # Eval-metric–based plateau tracking (separate from train-loss EMA)
        self._best_eval_metric = None
        self._best_eval_step = None
        self._last_global_step = None

        self._slope_det = SlopePlateauDetector(**(slope_kwargs or {}))
        self._plateau_step_train = None

    def record_step(self, loss, grad_norm=None, energy_j=None, step_time=None, sgd_step=None):
        """Record metrics for a single training step."""
        self._total_steps_seen += 1
        if sgd_step is not None:
            self._sgd_step_at_obs.append(int(sgd_step))
        self.loss_history.append(loss)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
        if energy_j is not None:
            self.energy_per_step.append(energy_j)
        if step_time is not None:
            self.step_times.append(step_time)

        # Cap histories to prevent memory growth on large datasets (MNLI/QQP)
        _MAX_HIST = 5000
        if len(self.loss_history) > _MAX_HIST * 2:
            self.loss_history = self.loss_history[-_MAX_HIST:]
        if len(self.grad_norm_history) > _MAX_HIST * 2:
            self.grad_norm_history = self.grad_norm_history[-_MAX_HIST:]
        if len(self.step_times) > _MAX_HIST * 2:
            self.step_times = self.step_times[-_MAX_HIST:]
        if len(self.improving_steps) > _MAX_HIST * 2:
            self.improving_steps = self.improving_steps[-_MAX_HIST:]

        # --- Legacy EMA-based detector (kept for `detector="relative"`) ---
        if self.detector_name == "relative":
            if self._ema_loss is None:
                self._ema_loss = loss
                self._best_ema_loss = loss
            else:
                self._ema_loss = self.ema_alpha * loss + (1 - self.ema_alpha) * self._ema_loss
                if self._best_ema_loss is not None and self._best_ema_loss > 0:
                    relative_improvement = (self._best_ema_loss - self._ema_loss) / self._best_ema_loss
                    if relative_improvement > self.plateau_min_improvement:
                        self._best_ema_loss = self._ema_loss
                        self._steps_since_ema_improvement = 0
                    else:
                        self._steps_since_ema_improvement += 1
                else:
                    if self._ema_loss < self._best_ema_loss:
                        self._best_ema_loss = self._ema_loss
                        self._steps_since_ema_improvement = 0
                    else:
                        self._steps_since_ema_improvement += 1

                if (self._plateau_step is None
                        and self._steps_since_ema_improvement >= self.plateau_patience
                        and self._total_steps_seen >= self.min_steps_before_plateau):
                    self._plateau_step = self._total_steps_seen - self.plateau_patience

        # --- New slope / dual-signal detector ---
        if self.detector_name in ("slope", "dual_signal"):
            sgd_now = sgd_step if sgd_step is not None else self._total_steps_seen
            self._slope_det.update_loss(sgd_now, loss)
            if (self._plateau_step is None
                    and self._slope_det.plateau_step is not None
                    and self._total_steps_seen >= self.min_steps_before_plateau):
                self._plateau_step = self._total_steps_seen - 1
                self._plateau_step_train = self._slope_det.plateau_step

        # --- Legacy per-step comparison (kept as secondary metric) ---
        if len(self.loss_history) >= 2:
            improved = self.loss_history[-1] < self.loss_history[-2]
            self.improving_steps.append(improved)
            if not improved:
                if loss > self.loss_history[-2] * 1.1:
                    self.waste_reasons["loss_spike"] += 1
                elif grad_norm is not None and grad_norm < 1e-7:
                    self.waste_reasons["vanishing_gradient"] += 1
                else:
                    self.waste_reasons["no_improvement"] += 1

    def _normal_cdf(self, x):
        """Standard normal CDF approximation."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))

    def compute_waste_metrics(self):
        """Compute waste metrics with 95% confidence intervals.
        
        Primary metric (waste_ratio): fraction of steps after plateau onset.
        Secondary metric (raw_improving_ratio): legacy per-step comparison.
        """
        n = self._total_steps_seen
        if n == 0:
            return {"waste_ratio": 0, "ci_95_low": 0, "ci_95_high": 0, "total_steps": 0,
                    "wasted_steps": 0, "plateau_step": None, "improving_steps": 0,
                    "raw_improving_ratio": 0,
                    "wasted_energy_j": 0, "total_energy_j": 0, "energy_waste_pct": 0,
                    "wasted_time_s": 0, "total_time_s": 0, "waste_reasons": {}}

        # --- Primary: plateau-based waste (translate obs-index to SGD steps) ---
        # Backward-compat: use SGD-step translation only if _sgd_step_at_obs is fully wired
        if (self._plateau_step is not None and self._sgd_step_at_obs
                and len(self._sgd_step_at_obs) == self._total_steps_seen):
            obs_idx = max(0, min(self._plateau_step, len(self._sgd_step_at_obs) - 1))
            plateau_step_sgd = self._sgd_step_at_obs[obs_idx]
            total_steps_sgd = self._sgd_step_at_obs[-1]
            wasted_steps = max(0, total_steps_sgd - plateau_step_sgd)
            waste_ratio = wasted_steps / total_steps_sgd if total_steps_sgd else 0.0
        else:
            # Fallback: obs-index units (old behaviour for legacy callers)
            if self._plateau_step is not None:
                plateau_step_sgd = max(0, self._plateau_step)
                wasted_steps = n - self._plateau_step
                waste_ratio = wasted_steps / n if n else 0.0
            else:
                plateau_step_sgd = None
                wasted_steps = 0
                waste_ratio = 0.0
            total_steps_sgd = n

        # 95% CI for waste ratio using Wilson score interval
        if n >= 2 and waste_ratio > 0:
            z = 1.96
            p_hat = waste_ratio
            denominator = 1 + z**2 / n
            center = (p_hat + z**2 / (2 * n)) / denominator
            spread = z * math.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
            ci_low = max(0, center - spread)
            ci_high = min(1, center + spread)
        else:
            ci_low = ci_high = waste_ratio

        # --- Secondary: legacy raw per-step improving ratio ---
        n_legacy = len(self.improving_steps)
        if n_legacy > 0:
            raw_waste_flags = [0 if imp else 1 for imp in self.improving_steps]
            raw_improving_ratio = 1.0 - np.mean(raw_waste_flags)
        else:
            raw_improving_ratio = 0.0

        # Energy waste (attributed to post-plateau steps)
        total_energy = sum(self.energy_per_step) if self.energy_per_step else 0
        wasted_energy = 0.0
        if self.energy_per_step and self._plateau_step is not None:
            obs_idx = max(0, min(self._plateau_step, len(self.energy_per_step) - 1))
            wasted_energy = sum(self.energy_per_step[obs_idx:])

        # Time waste (attributed to post-plateau steps)
        total_time = sum(self.step_times) if self.step_times else 0
        wasted_time = 0.0
        if self.step_times and self._plateau_step is not None:
            obs_idx = max(0, min(self._plateau_step, len(self.step_times) - 1))
            wasted_time = sum(self.step_times[obs_idx:])

        # --- Eval-metric-based waste (the meaningful one) ---
        best_eval_step = getattr(self, "_best_eval_step", None)
        best_eval_metric = getattr(self, "_best_eval_metric", None)
        if best_eval_step is not None and n > 0:
            last_step = getattr(self, "_last_global_step", None)
            if last_step and last_step > 0:
                eval_waste_ratio = max(0.0, (last_step - best_eval_step) / last_step)
            else:
                eval_waste_ratio = 0.0
        else:
            eval_waste_ratio = 0.0

        return {
            "waste_ratio": float(waste_ratio),
            "ci_95_low": float(ci_low),
            "ci_95_high": float(ci_high),
            "total_steps": total_steps_sgd if self._sgd_step_at_obs else n,
            "wasted_steps": int(wasted_steps),
            "plateau_step": plateau_step_sgd,
            "improving_steps": int((total_steps_sgd if self._sgd_step_at_obs else n) - wasted_steps),
            "raw_improving_ratio": float(raw_improving_ratio),
            "wasted_energy_j": float(wasted_energy),
            "total_energy_j": float(total_energy),
            "energy_waste_pct": float(wasted_energy / total_energy * 100) if total_energy > 0 else 0,
            "wasted_time_s": float(wasted_time),
            "total_time_s": float(total_time),
            "waste_reasons": dict(self.waste_reasons),
            "best_eval_step": self._best_eval_step,
            "best_eval_metric": self._best_eval_metric,
            "eval_waste_ratio": float(eval_waste_ratio),
        }


    def set_eval_direction(self, greater_is_better):
        """Set the evaluation metric direction for waste tracking."""
        self._eval_greater_is_better = greater_is_better

    def record_eval_improvement(self, global_step, eval_metric, greater_is_better=None):
        """Track best-eval-metric step for the secondary, eval-based waste ratio.

        Called from on_evaluate() in the diagnostics callback. This metric
        answers the question 'how much compute happened after the eval
        metric stopped improving?' and is independent of training-loss
        smoothing artefacts.
        """
        if greater_is_better is None:
            greater_is_better = self._eval_greater_is_better
        if self._best_eval_metric is None:
            self._best_eval_step = global_step
            self._best_eval_metric = eval_metric
            self._last_global_step = global_step
            return
        improved = (
            (greater_is_better and eval_metric > self._best_eval_metric)
            or (not greater_is_better and eval_metric < self._best_eval_metric)
        )
        if improved:
            self._best_eval_step = global_step
            self._best_eval_metric = eval_metric
        self._last_global_step = global_step


# ═══════════════════════════════════════════════════════════════════════
# Phase Transition Detector (NEW)
# ═══════════════════════════════════════════════════════════════════════

class PhaseTransitionDetector:
    """Detects phase transitions in training dynamics.
    
    Monitors loss curvature, gradient norm changes, and learning rate
    to identify transitions between training phases:
    - Warmup -> Active learning
    - Active learning -> Plateau
    - Plateau -> Fine-tuning
    - Potential divergence
    """

    def __init__(self, smoothing_window=20, min_phase_duration=20):
        self.smoothing_window = smoothing_window
        self.min_phase_duration = min_phase_duration
        self.loss_history = []
        self.grad_norm_history = []
        self.lr_history = []
        self.phase_history = []  # (step, phase_name)
        self.transition_points = []  # (step, from_phase, to_phase, evidence)
        self.current_phase = "warmup"
        # Hysteresis state
        self._candidate_phase = None
        self._candidate_count = 0

    def update(self, step, loss, grad_norm=None, lr=None):
        """Update detector with new step data."""
        self.loss_history.append(loss)
        if grad_norm is not None:
            self.grad_norm_history.append(grad_norm)
        if lr is not None:
            self.lr_history.append(lr)

        # Cap histories to prevent memory growth on large datasets
        _MAX_HIST = 5000
        if len(self.loss_history) > _MAX_HIST * 2:
            self.loss_history = self.loss_history[-_MAX_HIST:]
        if len(self.grad_norm_history) > _MAX_HIST * 2:
            self.grad_norm_history = self.grad_norm_history[-_MAX_HIST:]
        if len(self.lr_history) > _MAX_HIST * 2:
            self.lr_history = self.lr_history[-_MAX_HIST:]
        if len(self.phase_history) > _MAX_HIST * 2:
            self.phase_history = self.phase_history[-_MAX_HIST:]
        
        raw_phase = self._detect_phase(step)
        
        # Hysteresis: require min_phase_duration consecutive observations
        # of the same new phase before committing the transition.
        if raw_phase == self.current_phase:
            self._candidate_phase = None
            self._candidate_count = 0
        elif raw_phase == self._candidate_phase:
            self._candidate_count += 1
            if self._candidate_count >= self.min_phase_duration:
                evidence = self._gather_evidence(step)
                self.transition_points.append((step, self.current_phase, raw_phase, evidence))
                self.current_phase = raw_phase
                self._candidate_phase = None
                self._candidate_count = 0
        else:
            self._candidate_phase = raw_phase
            self._candidate_count = 1
        
        self.phase_history.append((step, self.current_phase))

    def _smooth(self, values, window=None):
        """Apply moving average smoothing."""
        if not values:
            return []
        w = window or self.smoothing_window
        if len(values) < w:
            return values
        return [np.mean(values[max(0, i - w):i + 1]) for i in range(len(values))]

    def _detect_phase(self, step):
        """Detect current training phase."""
        if len(self.loss_history) < self.smoothing_window:
            return "warmup"
        
        smoothed = self._smooth(self.loss_history)
        recent = smoothed[-self.smoothing_window:]
        
        # Check for divergence (loss increasing significantly)
        if len(recent) >= 5:
            recent_trend = np.polyfit(range(len(recent)), recent, 1)[0]
            if recent_trend > 0.01:
                return "diverging"
        
        # Check for plateau (very small loss changes)
        if len(recent) >= 10:
            loss_std = np.std(recent[-10:])
            loss_mean = np.mean(recent[-10:])
            if loss_mean > 0 and loss_std / loss_mean < 0.005:
                return "plateau"
        
        # Check if still in warmup (LR still increasing)
        if self.lr_history and len(self.lr_history) >= 2:
            if self.lr_history[-1] > self.lr_history[-2]:
                return "warmup"
        
        # Active learning (loss decreasing)
        if len(recent) >= 5:
            recent_trend = np.polyfit(range(len(recent)), recent, 1)[0]
            if recent_trend < -0.001:
                return "active_learning"
        
        return "fine_tuning"

    def _gather_evidence(self, step):
        """Gather evidence for phase transition."""
        evidence = {"step": step}
        if self.loss_history:
            evidence["loss"] = self.loss_history[-1]
        if self.grad_norm_history:
            evidence["grad_norm"] = self.grad_norm_history[-1]
        if self.lr_history:
            evidence["lr"] = self.lr_history[-1]
        return evidence

    def get_summary(self):
        """Get phase transition summary."""
        return {
            "current_phase": self.current_phase,
            "transitions": [
                {"step": s, "from": f, "to": t, "evidence": e}
                for s, f, t, e in self.transition_points
            ],
            "phase_durations": self._compute_phase_durations(),
        }

    def _compute_phase_durations(self):
        """Compute how long each phase lasted."""
        if not self.phase_history:
            return {}
        durations = defaultdict(int)
        for _, phase in self.phase_history:
            durations[phase] += 1
        return dict(durations)


# ═══════════════════════════════════════════════════════════════════════
# ETA Estimator (NEW)
# ═══════════════════════════════════════════════════════════════════════

class ETAEstimator:
    """Estimates time remaining for training with exponential moving average."""

    def __init__(self, total_steps):
        self.total_steps = total_steps
        self.start_time = time.time()
        self.step_times = []
        self.ema_step_time = None
        self.alpha = 0.1  # EMA smoothing factor

    def step_completed(self, current_step):
        """Record completion of a step."""
        now = time.time()
        if self.step_times:
            dt = now - self.step_times[-1][1]
            if self.ema_step_time is None:
                self.ema_step_time = dt
            else:
                self.ema_step_time = self.alpha * dt + (1 - self.alpha) * self.ema_step_time
        self.step_times.append((current_step, now))
        # Cap to prevent unbounded memory growth on long runs
        if len(self.step_times) > 2000:
            self.step_times = self.step_times[-1000:]

    def get_eta(self, current_step):
        """Get estimated time remaining."""
        if self.ema_step_time is None or current_step >= self.total_steps:
            return None
        remaining_steps = self.total_steps - current_step
        eta_seconds = remaining_steps * self.ema_step_time
        return {
            "eta_seconds": eta_seconds,
            "eta_formatted": str(timedelta(seconds=int(eta_seconds))),
            "elapsed_seconds": time.time() - self.start_time,
            "elapsed_formatted": str(timedelta(seconds=int(time.time() - self.start_time))),
            "progress_pct": current_step / self.total_steps * 100,
            "steps_per_second": 1.0 / self.ema_step_time if self.ema_step_time > 0 else 0,
            "current_step": current_step,
            "total_steps": self.total_steps,
        }


# ═══════════════════════════════════════════════════════════════════════
# LR-Loss Correlation Tracker (NEW)
# ═══════════════════════════════════════════════════════════════════════

class LRLossCorrelationTracker:
    """Tracks correlation between learning rate and loss dynamics."""

    def __init__(self):
        self.lr_history = []
        self.loss_history = []
        self.lr_loss_pairs = []

    def update(self, lr, loss):
        """Record a learning rate and loss pair."""
        self.lr_history.append(lr)
        self.loss_history.append(loss)
        self.lr_loss_pairs.append((lr, loss))

        # Cap to prevent memory growth on large datasets
        _MAX = 10000
        if len(self.lr_loss_pairs) > _MAX * 2:
            self.lr_loss_pairs = self.lr_loss_pairs[-_MAX:]
            self.lr_history = self.lr_history[-_MAX:]
            self.loss_history = self.loss_history[-_MAX:]

    def compute_correlation(self, window=None):
        """Compute Pearson correlation between LR and loss."""
        try:
            from scipy import stats as scipy_stats
        except ImportError:
            return None
        
        if len(self.lr_loss_pairs) < 5:
            return None
        
        pairs = self.lr_loss_pairs[-window:] if window else self.lr_loss_pairs
        lrs = [p[0] for p in pairs]
        losses = [p[1] for p in pairs]
        
        if np.std(lrs) < 1e-10 or np.std(losses) < 1e-10:
            return {"correlation": 0, "p_value": 1.0, "n_samples": len(pairs)}
        
        corr, p_val = scipy_stats.pearsonr(lrs, losses)
        return {
            "correlation": float(corr),
            "p_value": float(p_val),
            "n_samples": len(pairs),
        }


# ═══════════════════════════════════════════════════════════════════════
# Original utility functions (preserved)
# ═══════════════════════════════════════════════════════════════════════

def detect_device_profile():
    """Detect hardware profile based on available GPU VRAM."""
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb >= 20:
        return "server"
    return "laptop"


def get_training_config(profile: str):
    """Return hardware-appropriate training hyperparameters."""
    if profile == "server":
        # Detect bf16 support: V100 (compute capability 7.x) only supports fp16
        use_bf16 = False
        use_fp16 = True
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            if cc[0] >= 8:  # Ampere+ (A100, RTX 3090, RTX 4090, RTX 5090)
                use_bf16 = True
                use_fp16 = False
        return {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "fp16": use_fp16,
            "bf16": use_bf16,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 4,
            "max_samples": None,
        }
    else:
        return {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "max_samples": 2000,
        }


def load_glue_task(task_name: str, tokenizer, max_length: int = 128, max_samples=None):
    """Load and tokenize a GLUE task dataset."""
    cfg = GLUE_TASK_CONFIG[task_name]
    key1, key2 = cfg["keys"]

    dataset = load_dataset("glue", task_name)

    def tokenize_fn(examples):
        if key2 is not None:
            return tokenizer(
                examples[key1], examples[key2],
                truncation=True, max_length=max_length, padding=False,
            )
        return tokenizer(
            examples[key1],
            truncation=True, max_length=max_length, padding=False,
        )

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


def build_compute_metrics(task_name: str):
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
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
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


# ═══════════════════════════════════════════════════════════════════════
# Enhanced W&B Visualizations (NEW)
# ═══════════════════════════════════════════════════════════════════════

def log_training_dynamics_plots(
    gsnr_tracker, waste_quantifier, phase_detector, lr_loss_tracker,
    ler_tracker, task_name, use_wandb
):
    """Log rich training dynamics visualizations to W&B."""
    if not use_wandb:
        return
    
    try:
        import wandb
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        if wandb.run is None:
            return

        # ── 1. GSNR Heatmap (per-layer over time) ────────────────────
        if gsnr_tracker.gsnr_per_layer_history:
            layers = sorted(gsnr_tracker.gsnr_per_layer_history.keys())
            max_len = max(len(v) for v in gsnr_tracker.gsnr_per_layer_history.values())
            
            # Shorten layer names for display
            short_names = []
            for l in layers:
                parts = l.split(".")
                if len(parts) > 3:
                    short_names.append(".".join(parts[-3:]))
                else:
                    short_names.append(l)
            
            z_data = []
            for layer in layers:
                vals = gsnr_tracker.gsnr_per_layer_history[layer]
                # Pad with None if needed
                padded = vals + [None] * (max_len - len(vals))
                # Log scale for better visualization
                z_data.append([math.log10(v + 1e-10) if v is not None else None for v in padded])
            
            fig = go.Figure(data=go.Heatmap(
                z=z_data,
                y=short_names,
                colorscale="Viridis",
                colorbar=dict(title="log10(GSNR)"),
            ))
            fig.update_layout(
                title=f"GSNR Heatmap (Per-Layer) - {task_name}",
                xaxis_title="Measurement Step",
                yaxis_title="Layer",
                height=max(400, len(layers) * 25),
                width=900,
            )
            wandb.log({f"dynamics/gsnr_heatmap": wandb.Plotly(fig)})

        # ── 2. Gradient Norm Distribution (per-layer) ─────────────────
        if gsnr_tracker.grad_norm_history:
            fig = go.Figure()
            for layer_name in list(gsnr_tracker.grad_norm_history.keys())[:15]:  # Top 15 layers
                norms = gsnr_tracker.grad_norm_history[layer_name]
                short = layer_name.split(".")[-2:] if len(layer_name.split(".")) > 2 else [layer_name]
                fig.add_trace(go.Box(
                    y=norms[-100:],  # Last 100 values
                    name=".".join(short),
                    boxpoints="outliers",
                ))
            fig.update_layout(
                title=f"Gradient Norm Distribution (Per-Layer) - {task_name}",
                yaxis_title="Gradient Norm",
                yaxis_type="log",
                height=500,
                width=1000,
                showlegend=False,
            )
            wandb.log({f"dynamics/grad_norm_distribution": wandb.Plotly(fig)})

        # ── 3. Training Dynamics Multi-Panel ──────────────────────────
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                "Loss Trajectory", "Global GSNR",
                "Global Gradient Norm", "Phase Transitions",
                "LR vs Loss Correlation", "Waste Accumulation",
            ),
            vertical_spacing=0.08,
        )

        # Loss trajectory
        if phase_detector.loss_history:
            fig.add_trace(go.Scatter(
                y=phase_detector.loss_history,
                mode="lines",
                name="Loss",
                line=dict(color="#1565C0"),
            ), row=1, col=1)

        # Global GSNR over time
        if gsnr_tracker.gsnr_global_history:
            fig.add_trace(go.Scatter(
                y=gsnr_tracker.gsnr_global_history,
                mode="lines",
                name="Global GSNR",
                line=dict(color="#E65100"),
            ), row=1, col=2)

        # Global gradient norm
        if gsnr_tracker.global_grad_norm_history:
            fig.add_trace(go.Scatter(
                y=gsnr_tracker.global_grad_norm_history,
                mode="lines",
                name="Grad Norm",
                line=dict(color="#2E7D32"),
            ), row=2, col=1)

        # Phase transitions
        if phase_detector.phase_history:
            phase_map = {"warmup": 0, "active_learning": 1, "fine_tuning": 2, "plateau": 3, "diverging": 4}
            steps = [p[0] for p in phase_detector.phase_history]
            phases = [phase_map.get(p[1], -1) for p in phase_detector.phase_history]
            fig.add_trace(go.Scatter(
                x=steps, y=phases,
                mode="lines+markers",
                name="Phase",
                marker=dict(size=3),
                line=dict(color="#6A1B9A"),
            ), row=2, col=2)
            fig.update_yaxes(
                tickvals=list(phase_map.values()),
                ticktext=list(phase_map.keys()),
                row=2, col=2,
            )
            # Mark transitions
            for step, from_p, to_p, evidence in phase_detector.transition_points:
                fig.add_vline(x=step, line_dash="dash", line_color="red",
                              row=2, col=2)

        # LR vs Loss
        if lr_loss_tracker.lr_loss_pairs:
            lrs = [p[0] for p in lr_loss_tracker.lr_loss_pairs]
            losses = [p[1] for p in lr_loss_tracker.lr_loss_pairs]
            fig.add_trace(go.Scatter(
                x=lrs, y=losses,
                mode="markers",
                name="LR vs Loss",
                marker=dict(size=3, color="#00695C", opacity=0.5),
            ), row=3, col=1)

        # Waste accumulation
        if waste_quantifier.improving_steps:
            cumulative_waste = np.cumsum([0 if imp else 1 for imp in waste_quantifier.improving_steps])
            fig.add_trace(go.Scatter(
                y=cumulative_waste.tolist(),
                mode="lines",
                name="Cumulative Waste",
                line=dict(color="#C62828"),
            ), row=3, col=2)

        fig.update_layout(
            height=1000, width=1100,
            title_text=f"Training Dynamics Dashboard - {task_name}",
            showlegend=False,
        )
        wandb.log({f"dynamics/training_dashboard": wandb.Plotly(fig)})

        # ── 4. Waste Breakdown Pie Chart ──────────────────────────────
        waste_metrics = waste_quantifier.compute_waste_metrics()
        if waste_metrics["total_steps"] > 0:
            fig = make_subplots(rows=1, cols=2, specs=[[{"type": "pie"}, {"type": "bar"}]],
                                subplot_titles=("Step Classification", "Waste with 95% CI"))
            
            fig.add_trace(go.Pie(
                labels=["Improving", "Wasted"],
                values=[waste_metrics["improving_steps"], waste_metrics["wasted_steps"]],
                marker_colors=["#2E7D32", "#C62828"],
                hole=0.4,
            ), row=1, col=1)
            
            fig.add_trace(go.Bar(
                x=["Waste Ratio"],
                y=[waste_metrics["waste_ratio"]],
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[waste_metrics["ci_95_high"] - waste_metrics["waste_ratio"]],
                    arrayminus=[waste_metrics["waste_ratio"] - waste_metrics["ci_95_low"]],
                ),
                marker_color="#C62828",
            ), row=1, col=2)
            
            fig.update_layout(
                title_text=f"Waste Analysis - {task_name}",
                height=400, width=800,
            )
            wandb.log({f"dynamics/waste_analysis": wandb.Plotly(fig)})

        # ── 5. LER + rho_VG over time ────────────────────────────────
        if hasattr(ler_tracker, 'ler_history') and ler_tracker.ler_history:
            fig = make_subplots(rows=1, cols=2,
                                subplot_titles=("LER Over Time", "rho_VG Over Time"))
            
            fig.add_trace(go.Scatter(
                y=ler_tracker.ler_history,
                mode="lines+markers",
                name="LER",
                line=dict(color="#1565C0"),
                marker=dict(size=4),
            ), row=1, col=1)
            
            if hasattr(ler_tracker, 'rho_vg_history') and ler_tracker.rho_vg_history:
                fig.add_trace(go.Scatter(
                    y=ler_tracker.rho_vg_history,
                    mode="lines+markers",
                    name="rho_VG",
                    line=dict(color="#E65100"),
                    marker=dict(size=4),
                ), row=1, col=2)
            
            fig.update_layout(
                title_text=f"LER & rho_VG Dynamics - {task_name}",
                height=400, width=900,
            )
            wandb.log({f"dynamics/ler_rho_dynamics": wandb.Plotly(fig)})

    except Exception as e:
        print(f"  [W&B dynamics plots warn] {e}")
        import traceback
        traceback.print_exc()


def log_cross_run_summary(all_results, tasks, wandb_project, wandb_group):
    """Log cross-run summary visualizations to W&B."""
    try:
        import wandb
        import plotly.graph_objects as go
        import plotly.express as px
        from plotly.subplots import make_subplots
        from scipy import stats as scipy_stats
        
        successful = [r for r in all_results if "error" not in r]
        if not successful:
            return

        wandb.init(
            project=wandb_project,
            name=f"summary-{wandb_group}",
            group=wandb_group,
            job_type="summary",
            tags=["summary"],
            reinit=True,
        )

        # ── Cross-task performance heatmap ────────────────────────────
        seeds = sorted(set(r["seed"] for r in successful))
        matrix = []
        for task in tasks:
            row = []
            for seed in seeds:
                match = [r for r in successful if r["task"] == task and r["seed"] == seed]
                if match:
                    em = match[0].get("eval_metrics", {})
                    acc = em.get("eval_accuracy", em.get("eval_matthews_correlation",
                                 em.get("eval_pearsonr", 0)))
                    row.append(acc)
                else:
                    row.append(None)
            matrix.append(row)

        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[f"seed-{s}" for s in seeds],
            y=tasks,
            colorscale="RdYlGn",
            zmin=0, zmax=1,
            text=[[f"{v:.3f}" if v is not None else "" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
        ))
        fig.update_layout(
            title="Cross-Task Performance Heatmap",
            height=max(300, len(tasks) * 50),
            width=max(500, len(seeds) * 80),
        )
        wandb.log({"summary/performance_heatmap": wandb.Plotly(fig)})

        # ── GSNR cross-task comparison ────────────────────────────────
        task_gsnr = {}
        for r in successful:
            task = r["task"]
            gsnr = r.get("gsnr_final", {}).get("gsnr_global")
            if gsnr is not None:
                if task not in task_gsnr:
                    task_gsnr[task] = []
                task_gsnr[task].append(gsnr)
        
        if task_gsnr:
            fig = go.Figure()
            for task in tasks:
                if task in task_gsnr:
                    fig.add_trace(go.Box(
                        y=task_gsnr[task],
                        name=task,
                        boxpoints="all",
                    ))
            fig.update_layout(
                title="Final GSNR Distribution Across Tasks",
                yaxis_title="GSNR (log scale)",
                yaxis_type="log",
                height=400, width=800,
            )
            wandb.log({"summary/gsnr_comparison": wandb.Plotly(fig)})

        # ── Waste comparison across tasks ─────────────────────────────
        task_waste = {}
        task_waste_ci = {}
        for r in successful:
            task = r["task"]
            wm = r.get("waste_metrics", {})
            wr = wm.get("waste_ratio")
            if wr is not None:
                if task not in task_waste:
                    task_waste[task] = []
                    task_waste_ci[task] = []
                task_waste[task].append(wr)
                task_waste_ci[task].append((wm.get("ci_95_low", wr), wm.get("ci_95_high", wr)))
        
        if task_waste:
            fig = go.Figure()
            for task in tasks:
                if task in task_waste:
                    mean_w = np.mean(task_waste[task])
                    ci_lows = [c[0] for c in task_waste_ci[task]]
                    ci_highs = [c[1] for c in task_waste_ci[task]]
                    fig.add_trace(go.Bar(
                        x=[task],
                        y=[mean_w],
                        error_y=dict(
                            type="data",
                            symmetric=False,
                            array=[np.mean(ci_highs) - mean_w],
                            arrayminus=[mean_w - np.mean(ci_lows)],
                        ),
                        name=task,
                    ))
            fig.update_layout(
                title="Waste Ratio Across Tasks (with 95% CI)",
                yaxis_title="Waste Ratio",
                height=400, width=800,
            )
            wandb.log({"summary/waste_comparison": wandb.Plotly(fig)})

        # ── Energy breakdown ──────────────────────────────────────────
        task_kwh = {}
        task_kwh_std = {}
        task_acc = {}
        task_acc_std = {}
        for task in tasks:
            task_results = [r for r in successful if r["task"] == task]
            kwhs = [r.get("energy_kwh", 0) for r in task_results]
            accs = []
            for r in task_results:
                em = r.get("eval_metrics", {})
                accs.append(em.get("eval_accuracy", em.get("eval_matthews_correlation",
                            em.get("eval_pearsonr", 0))))
            task_kwh[task] = np.mean(kwhs) if kwhs else 0
            task_kwh_std[task] = np.std(kwhs) if len(kwhs) > 1 else 0
            task_acc[task] = np.mean(accs) if accs else 0
            task_acc_std[task] = np.std(accs) if len(accs) > 1 else 0

        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=("Energy per Task", "Accuracy vs Energy"))

        fig.add_trace(go.Bar(
            x=tasks,
            y=[task_kwh[t] for t in tasks],
            error_y=dict(type="data", array=[task_kwh_std[t] for t in tasks]),
            marker_color="#1565C0",
            name="Energy (kWh)",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=[task_kwh[t] for t in tasks],
            y=[task_acc[t] for t in tasks],
            mode="markers+text",
            text=tasks,
            textposition="top center",
            marker=dict(size=12, color="#E65100"),
            name="Tasks",
        ), row=1, col=2)

        fig.update_layout(height=450, width=1000,
                          title_text="Energy Analysis")
        fig.update_yaxes(title_text="kWh", row=1, col=1)
        fig.update_yaxes(title_text="Performance", row=1, col=2)
        fig.update_xaxes(title_text="kWh", row=1, col=2)

        wandb.log({"summary/energy_analysis": wandb.Plotly(fig)})

        # ── Results table with 95% CI ─────────────────────────────────
        summary_table = wandb.Table(columns=[
            "task", "n_seeds", "mean_acc", "std_acc", "ci_95_low", "ci_95_high",
            "mean_kwh", "mean_runtime_s", "mean_waste_ratio", "mean_gsnr",
        ])

        for task in tasks:
            task_results = [r for r in successful if r["task"] == task]
            if not task_results:
                continue
            accs = []
            for r in task_results:
                em = r.get("eval_metrics", {})
                accs.append(em.get("eval_accuracy", em.get("eval_matthews_correlation",
                            em.get("eval_pearsonr", 0))))
            kwhs = [r.get("energy_kwh", 0) for r in task_results]
            runtimes = [r.get("train_runtime_s", 0) for r in task_results]
            wastes = [r.get("waste_metrics", {}).get("waste_ratio", 0) for r in task_results]
            gsnrs_raw = [r.get("gsnr_final", {}).get("gsnr_global") for r in task_results]
            # Filter None values: GSNR can be None when the tracker has
            # insufficient gradient captures (short runs, large log interval).
            gsnrs = [g for g in gsnrs_raw if g is not None]

            n = len(accs)
            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            if n >= 2:
                ci = scipy_stats.t.interval(0.95, df=n - 1, loc=mean_acc,
                                            scale=scipy_stats.sem(accs))
            else:
                ci = (mean_acc, mean_acc)

            summary_table.add_data(
                task, n, mean_acc, std_acc, ci[0], ci[1],
                np.mean(kwhs), np.mean(runtimes),
                np.mean(wastes), np.mean(gsnrs) if gsnrs else 0.0,
            )

        wandb.log({"summary/results_table": summary_table})
        wandb.finish(quiet=True)
        
    except Exception as e:
        print(f"  [W&B summary warn] {e}")
        import traceback
        traceback.print_exc()


# ═══════════════════════════════════════════════════════════════════════
# Custom Trainer to capture real logits (FIX #4)
# ═══════════════════════════════════════════════════════════════════════

class LERNATrainer(Trainer):
    """Extended Trainer that captures real logits for LER tracking."""

    def __init__(self, *args, ler_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ler_tracker = ler_tracker
        self._last_real_logits = None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Override to capture real logits."""
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        
        # Capture real logits for LER tracker
        if hasattr(outputs, "logits"):
            self._last_real_logits = outputs.logits.detach()
        elif isinstance(outputs, dict) and "logits" in outputs:
            self._last_real_logits = outputs["logits"].detach()
        
        return (loss, outputs) if return_outputs else loss


_UNSET = object()


def run_single_experiment(
    task_name: str,
    seed: int,
    lr: float,
    profile: str,
    base_output_dir: str,
    use_wandb: bool = False,
    max_samples_override=_UNSET,
    run_idx: int = 0,
    total_runs: int = 0,
    wandb_project: str = "lerna-baseline",
    wandb_group: str = "glue-baseline",
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 5,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    init_from_mnli: bool = False,
):
    """Run a single GLUE fine-tuning experiment with FULL diagnostics.

    Includes: LER, GSNR, per-layer gradient analysis, waste quantification,
    phase transition detection, LR-loss correlation, ETA estimation,
    and rich W&B visualizations.
    """
    from lerna.utils.metrics import LERTracker
    from lerna.callbacks.efficiency_callback import PowerTelemetryCallback

    hw_cfg = get_training_config(profile)
    if max_samples_override is not _UNSET:
        hw_cfg["max_samples"] = max_samples_override

    run_id = f"{task_name}_s{seed}_lr{lr:.0e}"
    output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    # ── W&B: explicitly create a NEW run for this experiment ──────────
    if use_wandb:
        import wandb
        _ensure_wandb_finished()
        wandb.init(
            project=wandb_project,
            name=run_id,
            group=wandb_group,
            job_type="train",
            tags=[task_name, f"seed-{seed}", profile, MODEL_NAME.split("/")[-1]],
            reinit=True,
            config={
                "task": task_name,
                "seed": seed,
                "learning_rate": lr,
                "model": MODEL_NAME,
                "profile": profile,
                "max_samples": hw_cfg["max_samples"],
                "batch_size": hw_cfg["per_device_train_batch_size"],
                "gradient_accumulation_steps": hw_cfg["gradient_accumulation_steps"],
                "run_index": run_idx,
                "total_runs": total_runs,
            },
        )
        # Fix WandB step-conflict warnings: let wandb accept any step order
        try:
            wandb.define_metric("*", step_metric="train/global_step", step_sync=False)
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  LERNA Baseline: {task_name} | seed={seed} | lr={lr}")
    print(f"  Profile: {profile} | Output: {output_dir}")
    print(f"  Epochs: {num_epochs} | Warmup: {warmup_ratio} | ES patience: {early_stopping_patience}")
    if run_idx and total_runs:
        print(f"  Progress: run {run_idx}/{total_runs}")
    print(f"{'='*60}")

    # ── Reproducibility ───────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # ── Model & tokenizer ─────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg = GLUE_TASK_CONFIG[task_name]

    # FIX: For RTE, optionally initialize from MNLI-finetuned model
    # This is standard practice (Devlin et al. 2019) and typically adds 10-20 points
    model_name_or_path = MODEL_NAME
    mnli_checkpoint_dir = os.path.join(base_output_dir, "mnli_finetuned")
    if init_from_mnli and os.path.exists(mnli_checkpoint_dir):
        print(f"  [MNLI Transfer] Loading MNLI-finetuned model from {mnli_checkpoint_dir}")
        # Load the MNLI model but replace the classifier head for the target task
        from transformers import AutoConfig
        mnli_model = AutoModelForSequenceClassification.from_pretrained(
            mnli_checkpoint_dir,
        )
        # Create target model with correct num_labels
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=cfg["num_labels"],
        )
        # Transfer encoder weights from MNLI model (skip classifier head)
        encoder_state = {k: v for k, v in mnli_model.state_dict().items()
                        if "classifier" not in k and "pooler" not in k}
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"  [MNLI Transfer] Loaded encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        del mnli_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif init_from_mnli and not os.path.exists(mnli_checkpoint_dir):
        print(f"  [MNLI Transfer] WARNING: init_from_mnli=True but no MNLI checkpoint found at {mnli_checkpoint_dir}")
        print(f"  [MNLI Transfer] Run MNLI first, then save best model to {mnli_checkpoint_dir}")
        print(f"  [MNLI Transfer] Falling back to pretrained {MODEL_NAME}")
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=cfg["num_labels"],
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=cfg["num_labels"],
        )

    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    # ── Data ──────────────────────────────────────────────────────────
    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"]
    )
    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    # ── Compute total steps (needed by trackers below) ───────────────
    steps_per_epoch = len(train_ds) // (
        hw_cfg["per_device_train_batch_size"] * hw_cfg["gradient_accumulation_steps"]
    )
    total_steps = steps_per_epoch * num_epochs
    eval_steps = max(total_steps // 20, 10)

    # ── Initialize ALL trackers ───────────────────────────────────────
    ler_tracker = LERTracker(task=task_name, window_size=5)
    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
    )
    
    # NEW trackers
    gsnr_tracker = GSNRTracker(model, window_size=50)
    # min_steps_before_plateau: 10% of total steps (min 100) to prevent
    # false early plateau detection on large-dataset tasks like MNLI/QQP
    # Plateau detection parameters scaled to dataset size.
    # Previous values (patience=50, min_steps=max(100, total//10)) were too
    # aggressive for small datasets like MRPC (~345 steps) and RTE (~234 steps),
    # causing 0% waste detection across all seeds for those tasks.
    #
    # New logic:
    #   - min_steps_before_plateau: 15% of total steps, capped at [30, 200]
    #   - plateau_patience: 8% of total steps, capped at [20, 100]
    #   - plateau_min_improvement: 0.0005 (was 0.001)
    # CRITICAL: The stale-loss dedup fix means the WasteQuantifier only
    # receives one data point per unique loss value, NOT one per training
    # step. The actual number of unique observations depends on how often
    # the training loss changes between logging intervals:
    #   - Classification (large datasets): ~hundreds of unique values
    #   - Regression (STS-B): only ~20-30 unique values across 450 steps
    #     because MSE loss is averaged and changes slowly
    #
    # For regression tasks, use small fixed parameters since we know the
    # data resolution is extremely low. For classification, scale to the
    # estimated unique observations.
    logging_steps = max(eval_steps // 5, 1)  # matches TrainingArguments.logging_steps
    expected_unique_obs = max(1, total_steps // logging_steps)
    is_regression = GLUE_TASK_CONFIG[task_name]["num_labels"] == 1
    if is_regression:
        # Regression tasks produce very few unique loss values (~20-30)
        # due to MSE averaging. Use minimal parameters to ensure plateau
        # detection can trigger within the available data points.
        # patience=1 because MSE loss oscillates: it alternates between
        # 5-7% improvements and 2-3% non-improvements, so consecutive
        # patience >1 never triggers. patience=1 is safe because the
        # rapid-learning phase (steps 0-9) always shows >10% improvement.
        waste_min_steps = 5
        waste_patience = 1
        waste_min_improvement = 0.04
        waste_ema_alpha = 0.5
    else:
        waste_min_steps = max(3, min(50, int(expected_unique_obs * 0.15)))
        waste_patience = max(3, min(30, int(expected_unique_obs * 0.10)))
        waste_min_improvement = 0.0005
        target_window = max(4, 2 * waste_patience)
        waste_ema_alpha = min(0.5, 2.0 / (target_window + 1))
    # Regression tasks (MSE loss) need a much higher min_improvement
    # threshold than classification. The stale-loss dedup means each
    # unique observation spans multiple training steps, so the loss
    # difference between consecutive observations is larger than between
    # consecutive raw steps. For STS-B with ~26 unique observations,
    # the per-interval relative improvements in the plateau zone are
    # typically 2-5%, so we need a threshold above that noise floor.
    #
    # Threshold history (STS-B seed 42 smoke tests, 10 epochs / ~450 steps):
    #   0.0005 (0.05%) -> waste=0.000 (original, never triggers)
    #   0.005  (0.5%)  -> waste=0.000 (2026-04-06, still too sensitive)
    #   0.015  (1.5%)  -> waste=0.000 (2026-04-07, still below per-interval noise)
    #   0.04   (4.0%)  -> waste>0     (2026-04-08, above MSE inter-interval noise)
    #
    # Classification tasks use 0.0005 (0.05%) which correctly detects
    # plateaus on QQP (98.8%), MNLI (98.9%), SST-2 (50.4%), QNLI (55.8%).
    waste_quantifier = WasteQuantifier(
        ema_alpha=waste_ema_alpha,
        plateau_patience=waste_patience,
        plateau_min_improvement=waste_min_improvement,
        min_steps_before_plateau=waste_min_steps,
    )
    phase_detector = PhaseTransitionDetector(smoothing_window=20, min_phase_duration=20)
    lr_loss_tracker = LRLossCorrelationTracker()
    eta_estimator = ETAEstimator(total_steps)

    # ── Enhanced Diagnostics Callback ─────────────────────────────────
    class FullDiagnosticsCallback:
        """Comprehensive diagnostics: LER + GSNR + waste + phase + ETA."""

        def __init__(self, ler_trk, gsnr_trk, waste_q, phase_det, lr_loss_trk, eta_est, model_ref, trainer_ref_holder, greater_is_better=True):
            self.ler_tracker = ler_trk
            self.gsnr_tracker = gsnr_trk
            self.waste_quantifier = waste_q
            self.phase_detector = phase_det
            self.lr_loss_tracker = lr_loss_trk
            self.eta_estimator = eta_est
            self._model = model_ref
            self._trainer_holder = trainer_ref_holder  # mutable list to hold trainer ref
            self._greater_is_better = greater_is_better
            self.step_count = 0
            self._last_loss = None
            self._last_loss_fed_to_waste = None  # Track last value fed to WasteQuantifier
            self._step_start_time = None
            # Scale GSNR interval with dataset size to avoid CPU bottleneck
            # Small datasets (~800 steps): every ~50 steps
            # Large datasets (~36K steps): every eval_steps
            self._gsnr_log_interval = max(1, min(max(eval_steps, 200), max(1, total_steps // 4)))

        # ── Trainer callback interface ────────────────────────────────
        def on_init_end(self, args, state, control, **kwargs):
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            # A.2.1 FIX: HF Trainer creates the optimizer lazily inside
            # trainer.train(). It is exposed to callbacks via kwargs at
            # on_train_begin. Bind it here so LERTracker can compute ρ_VG
            # against the Adam-effective update direction.
            opt = kwargs.get("optimizer", None)
            if opt is not None and hasattr(self.ler_tracker, "set_optimizer"):
                self.ler_tracker.set_optimizer(opt)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            # Save all diagnostics
            self._save_all_diagnostics()
            
            # Log training dynamics plots to W&B
            log_training_dynamics_plots(
                self.gsnr_tracker, self.waste_quantifier, self.phase_detector,
                self.lr_loss_tracker, self.ler_tracker, task_name, use_wandb,
            )
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            self._step_start_time = time.time()
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            """Capture gradients BEFORE optimizer step (when grads are fresh)."""
            try:
                self.ler_tracker.capture_step_gradients(self._model)
            except Exception:
                pass

            # Lightweight scalar norms every step for phase detector
            try:
                self.gsnr_tracker.capture_scalar_norms(self._model)
            except Exception:
                pass

            # Full EMA-Welford GSNR update less frequently
            if self.step_count % self._gsnr_log_interval == 0:
                try:
                    self.gsnr_tracker.capture_gradients(self._model)
                except Exception:
                    pass
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.step_count += 1
            
            # ETA tracking
            self.eta_estimator.step_completed(state.global_step)
            
            # Fallback gradient capture if pre_optimizer didn't fire
            # Also capture GSNR gradients as belt-and-braces fallback
            if self.ler_tracker._cached_rho_vg is None and self._model is not None:
                has_grad = any(
                    p.grad is not None for p in self._model.parameters() if p.requires_grad
                )
                if has_grad:
                    try:
                        self.ler_tracker.capture_step_gradients(self._model)
                    except Exception:
                        pass
            # Belt-and-braces GSNR fallback: capture if on_pre_optimizer_step never fired
            if self.step_count % self._gsnr_log_interval == 0:
                try:
                    self.gsnr_tracker.capture_gradients(self._model)
                except Exception:
                    pass
            
            # Step time for waste tracking
            step_time = time.time() - self._step_start_time if self._step_start_time else None
            
            # Get current LR from scheduler
            current_lr = None
            trainer = self._trainer_holder[0] if self._trainer_holder else None
            if trainer is not None and hasattr(trainer, 'lr_scheduler') and trainer.lr_scheduler is not None:
                try:
                    current_lr = trainer.lr_scheduler.get_last_lr()[0]
                except Exception:
                    pass
            
            # Record waste and phase data.
            # CRITICAL: Only feed the WasteQuantifier when _last_loss has
            # actually changed since the last feed. The Trainer logs loss
            # every logging_steps (e.g., 4 steps), so between log events
            # _last_loss is stale. Feeding the same value repeatedly to
            # the EMA creates artificial "no improvement" periods followed
            # by artificial "improvement" when a new value arrives, which
            # prevents plateau detection from ever triggering.
            if self._last_loss is not None:
                # Get global grad norm
                global_grad_norm = None
                if self.gsnr_tracker.global_grad_norm_history:
                    global_grad_norm = self.gsnr_tracker.global_grad_norm_history[-1]
                
                # Only feed WasteQuantifier on genuinely new loss values
                loss_is_new = (self._last_loss_fed_to_waste is None or
                               self._last_loss != self._last_loss_fed_to_waste)
                if loss_is_new:
                    # Pull last per-step energy from power callback if available
                    energy_j = None
                    try:
                        pc = power_callback
                        if hasattr(pc, "_power_samples") and pc._power_samples:
                            last = pc._power_samples[-1]
                            if step_time and "power_w" in last:
                                energy_j = float(last["power_w"]) * float(step_time)
                    except Exception:
                        energy_j = None
                    self.waste_quantifier.record_step(
                        loss=self._last_loss,
                        grad_norm=global_grad_norm,
                        energy_j=energy_j,
                        step_time=step_time,
                        sgd_step=state.global_step,
                    )
                    self._last_loss_fed_to_waste = self._last_loss
                
                # Phase detector and LR-loss tracker can still use every step
                self.phase_detector.update(
                    step=state.global_step,
                    loss=self._last_loss,
                    grad_norm=global_grad_norm,
                    lr=current_lr,
                )
                if current_lr is not None:
                    self.lr_loss_tracker.update(current_lr, self._last_loss)
            
            # Periodic ETA print
            if state.global_step % max(eval_steps, 10) == 0 and state.global_step > 0:
                eta = self.eta_estimator.get_eta(state.global_step)
                if eta:
                    print(
                        f"  [ETA] Step {eta['current_step']}/{eta['total_steps']} "
                        f"({eta['progress_pct']:.1f}%) | "
                        f"Elapsed: {eta['elapsed_formatted']} | "
                        f"ETA: {eta['eta_formatted']} | "
                        f"{eta['steps_per_second']:.1f} steps/s"
                    )
            
            # Periodic GSNR computation and logging
            if state.global_step % self._gsnr_log_interval == 0 and state.global_step > 0:
                try:
                    gsnr_results = self.gsnr_tracker.compute_gsnr()
                    global_gsnr = gsnr_results.get("__global__")
                    
                    if use_wandb and global_gsnr is not None:
                        import wandb
                        if wandb.run is not None:
                            log_data = {
                                "gsnr/global": global_gsnr,
                                "gsnr/global_log10": math.log10(global_gsnr + 1e-10),
                            }
                            # Log top/bottom layers
                            layer_gsnrs = {k: v for k, v in gsnr_results.items() if k != "__global__"}
                            if layer_gsnrs:
                                sorted_layers = sorted(layer_gsnrs.items(), key=lambda x: x[1])
                                for i, (name, val) in enumerate(sorted_layers[:3]):
                                    short = name.split(".")[-2:]
                                    log_data[f"gsnr/bottom_{i+1}_{'_'.join(short)}"] = val
                                for i, (name, val) in enumerate(sorted_layers[-3:]):
                                    short = name.split(".")[-2:]
                                    log_data[f"gsnr/top_{i+1}_{'_'.join(short)}"] = val

                            # Fisher Information (computed from same accumulators)
                            fisher_results = self.gsnr_tracker.compute_fisher_info()
                            fi_global = fisher_results.get("__global__")
                            if fi_global is not None:
                                log_data["fisher/global"] = fi_global
                                log_data["fisher/global_log10"] = math.log10(fi_global + 1e-10)

                            wandb.log(log_data, step=state.global_step)
                except Exception:
                    pass
            
            # Log waste and phase to W&B
            if use_wandb and state.global_step % max(eval_steps // 2, 5) == 0:
                try:
                    import wandb
                    if wandb.run is not None:
                        wm = self.waste_quantifier.compute_waste_metrics()
                        wandb.log({
                            "waste/ratio": wm["waste_ratio"],
                            "waste/ci_95_low": wm["ci_95_low"],
                            "waste/ci_95_high": wm["ci_95_high"],
                            "waste/cumulative_wasted_steps": wm["wasted_steps"],
                            "phase/current": self.phase_detector.current_phase,
                            "phase/num_transitions": len(self.phase_detector.transition_points),
                        }, step=state.global_step)
                        
                        # LR-loss correlation
                        corr = self.lr_loss_tracker.compute_correlation(window=100)
                        if corr:
                            wandb.log({
                                "lr_loss/correlation": corr["correlation"],
                                "lr_loss/p_value": corr["p_value"],
                            }, step=state.global_step)
                except Exception:
                    pass
            
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            return control

        def on_save(self, args, state, control, **kwargs):
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self._last_loss = logs.get("loss", self._last_loss)
                
                # Log gradient norms from trainer logs
                if use_wandb:
                    try:
                        import wandb
                        if wandb.run is not None:
                            grad_norm_val = logs.get("grad_norm")
                            if grad_norm_val is not None:
                                wandb.log({
                                    "gradients/trainer_grad_norm": grad_norm_val,
                                }, step=state.global_step)
                    except Exception:
                        pass
            return control

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if metrics is None:
                return control

            eval_loss = metrics.get("eval_loss", 0)
            accuracy = metrics.get("eval_accuracy", metrics.get("eval_matthews_correlation", 0))

            # Wire eval-metric into WasteQuantifier (eval-based waste ratio).
            # Pick the same metric the trainer uses for best-model selection.
            try:
                self.waste_quantifier.set_eval_direction(self._greater_is_better)
                _eval_metric_name = "eval_" + str(metric_for_best_model)
                _gib = bool(greater_is_better)
                _val = metrics.get(_eval_metric_name)
                if _val is None:
                    _val = accuracy if accuracy else (-eval_loss)
                    _gib = True
                self.waste_quantifier.record_eval_improvement(
                    global_step=state.global_step,
                    eval_metric=float(_val),
                    greater_is_better=_gib,
                )
            except Exception as _e:
                print(f"  [WARN] record_eval_improvement failed: {_e}")


            # FIX #4: Use REAL logits from the custom trainer instead of dummy
            trainer = self._trainer_holder[0] if self._trainer_holder else None
            real_logits = None
            if trainer is not None and hasattr(trainer, '_last_real_logits') and trainer._last_real_logits is not None:
                real_logits = trainer._last_real_logits
            
            # Fallback: generate logits from a small eval batch if no cached logits
            if real_logits is None and model is not None:
                try:
                    model.eval()
                    # Get a small batch from eval dataset
                    small_batch_size = min(8, len(eval_ds))
                    small_batch = eval_ds.select(range(small_batch_size))
                    from torch.utils.data import DataLoader
                    dl = DataLoader(
                        small_batch,
                        batch_size=small_batch_size,
                        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
                    )
                    batch = next(iter(dl))
                    batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                    with torch.no_grad():
                        outputs = model(**batch)
                    real_logits = outputs.logits.detach()
                except Exception as e:
                    # Last resort: use dummy logits (but log warning)
                    print(f"  [WARN] Could not get real logits, using dummy: {e}")
                    num_labels = GLUE_TASK_CONFIG[task_name]["num_labels"]
                    real_logits = torch.randn(8, num_labels)

            loss_for_ler = self._last_loss if self._last_loss is not None else eval_loss

            try:
                self.ler_tracker.update(
                    loss=loss_for_ler,
                    logits=real_logits,
                    accuracy=accuracy,
                    model=model,
                )
            except Exception as e:
                print(f"  [LER warn] {e}")

            diag = self.ler_tracker.get_diagnostics()
            ler_val = diag.get("ler")
            vel_val = diag.get("param_velocity")
            rho_val = diag.get("rho_vg")
            phase = diag.get("phase", "?")

            ler_str = f"{ler_val:.6f}" if ler_val is not None else "warming"
            vel_str = f"{vel_val:.6f}" if vel_val is not None else "N/A"
            rho_str = f"{rho_val:.4f}" if rho_val is not None else "N/A"

            # Also get GSNR and waste info for the print
            gsnr_global = None
            if gsnr_tracker.gsnr_global_history:
                gsnr_global = gsnr_tracker.gsnr_global_history[-1]
            gsnr_str = f"{gsnr_global:.4f}" if gsnr_global is not None else "N/A"
            
            waste_metrics = waste_quantifier.compute_waste_metrics()
            waste_str = f"{waste_metrics['waste_ratio']:.3f}"
            
            phase_det_str = phase_detector.current_phase

            print(
                f"  [LERNA step={state.global_step}] "
                f"LER={ler_str} | vel={vel_str} | rho_VG={rho_str} | "
                f"GSNR={gsnr_str} | waste={waste_str} | "
                f"phase_LER={phase} | phase_det={phase_det_str} | acc={accuracy:.3f}"
            )

            # Log comprehensive diagnostics to W&B
            if use_wandb:
                try:
                    import wandb
                    if wandb.run is not None:
                        log_data = {
                            "lerna/ler": ler_val,
                            "lerna/velocity": vel_val,
                            "lerna/rho_vg": rho_val,
                            "lerna/phase": phase,
                            "lerna/eval_accuracy": accuracy,
                            "lerna/eval_loss": eval_loss,
                        }
                        
                        # Per-layer gradient norm stats
                        grad_stats = gsnr_tracker.get_gradient_norm_stats()
                        for layer_name, stats in list(grad_stats.items())[:10]:  # Top 10
                            short = "_".join(layer_name.split(".")[-2:])
                            log_data[f"grad_norms/{short}_mean"] = stats["mean"]
                            log_data[f"grad_norms/{short}_std"] = stats["std"]
                        
                        wandb.log(log_data, step=state.global_step)
                except Exception:
                    pass

            return control

        def on_predict(self, args, state, control, **kwargs):
            return control

        def on_prediction_step(self, args, state, control, **kwargs):
            return control

        def _save_all_diagnostics(self):
            """Save all diagnostics to JSON files."""
            # LER diagnostics
            diag_path = os.path.join(output_dir, "ler_diagnostics.json")
            final = self.ler_tracker.get_diagnostics()
            final["ler_history"] = self.ler_tracker.ler_history
            final["velocity_history"] = self.ler_tracker.velocity_history
            final["rho_vg_history"] = self.ler_tracker.rho_vg_history
            final["loss_history"] = self.ler_tracker.loss_history
            final["entropy_history"] = self.ler_tracker.entropy_history
            with open(diag_path, "w") as f:
                json.dump(final, f, indent=2, default=str)
            print(f"  LER diagnostics saved: {diag_path}")

            # GSNR diagnostics
            gsnr_path = os.path.join(output_dir, "gsnr_diagnostics.json")
            gsnr_summary = self.gsnr_tracker.get_summary()
            gsnr_summary["gsnr_global_history"] = self.gsnr_tracker.gsnr_global_history
            gsnr_summary["gsnr_per_layer_history"] = {
                k: v for k, v in self.gsnr_tracker.gsnr_per_layer_history.items()
            }
            with open(gsnr_path, "w") as f:
                json.dump(gsnr_summary, f, indent=2, default=str)
            print(f"  GSNR diagnostics saved: {gsnr_path}")

            # Waste diagnostics
            waste_path = os.path.join(output_dir, "waste_diagnostics.json")
            waste_metrics = self.waste_quantifier.compute_waste_metrics()
            with open(waste_path, "w") as f:
                json.dump(waste_metrics, f, indent=2, default=str)
            print(f"  Waste diagnostics saved: {waste_path}")

            # Phase transition diagnostics
            phase_path = os.path.join(output_dir, "phase_diagnostics.json")
            phase_summary = self.phase_detector.get_summary()
            with open(phase_path, "w") as f:
                json.dump(phase_summary, f, indent=2, default=str)
            print(f"  Phase diagnostics saved: {phase_path}")

            # LR-Loss correlation
            lr_loss_path = os.path.join(output_dir, "lr_loss_correlation.json")
            lr_loss_corr = self.lr_loss_tracker.compute_correlation()
            lr_loss_data = {
                "final_correlation": lr_loss_corr,
                "lr_history": self.lr_loss_tracker.lr_history[-1000:],  # Last 1000
                "loss_history": self.lr_loss_tracker.loss_history[-1000:],
            }
            with open(lr_loss_path, "w") as f:
                json.dump(lr_loss_data, f, indent=2, default=str)
            print(f"  LR-Loss correlation saved: {lr_loss_path}")

    # Mutable holder for trainer reference (needed inside callback)
    trainer_holder = [None]
    
    diag_callback = FullDiagnosticsCallback(
        ler_trk=ler_tracker,
        gsnr_trk=gsnr_tracker,
        waste_q=waste_quantifier,
        phase_det=phase_detector,
        lr_loss_trk=lr_loss_tracker,
        eta_est=eta_estimator,
        model_ref=model,
        trainer_ref_holder=trainer_holder,
        greater_is_better=greater_is_better,
    )

    # ── Training arguments ────────────────────────────────────────────
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
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,  # Keep 3 checkpoints (warmup/active/plateau phases for TracIn)
        # FIX #9: load_best_model_at_end=True to evaluate best model
        # FIX: Use task-specific metric for model selection (not hardcoded eval_loss)
        # CoLA -> eval_matthews_correlation, RTE/MRPC -> eval_accuracy, etc.
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=max(eval_steps // 5, 1),
        report_to="wandb" if use_wandb else "none",
        run_name=run_id if use_wandb else None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
        # Include grad norm in logs for tracking
        #include_for_metrics=["loss"],
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics(task_name)

    # Use custom trainer that captures real logits (FIX #4)
    trainer = LERNATrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        ler_tracker=ler_tracker,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            power_callback,
            diag_callback,
        ],
    )
    
    # Set trainer reference in callback
    trainer_holder[0] = trainer

    # A.2.1 FIX: trainer.optimizer is None until trainer.train() runs. The
    # real binding happens in FullDiagnosticsCallback.on_train_begin via
    # kwargs["optimizer"]. We only attempt an eager bind here as a no-op
    # safety in case some path constructs the optimizer up-front.
    if hasattr(trainer, "optimizer") and trainer.optimizer is not None \
            and hasattr(ler_tracker, "set_optimizer"):
        ler_tracker.set_optimizer(trainer.optimizer)

    # ── Train & evaluate ──────────────────────────────────────────────
    start_time = time.time()
    print(f"\n  Starting training: {total_steps} total steps, eval every {eval_steps} steps")
    print(f"  ETA will be shown after first eval checkpoint")
    print(f"  Diagnostics: EMA-Welford GSNR + Fisher Info + optimizer velocity\n")
    
    train_result = trainer.train()
    total_time = time.time() - start_time

    # FIX #9: Now evaluating the BEST model (loaded automatically)
    print(f"\n  Evaluating best model (loaded via load_best_model_at_end=True)...")
    eval_result = trainer.evaluate()

    # ── Collect results ───────────────────────────────────────────────
    avg_power = (
        float(np.mean([s["power_w"] for s in power_callback._power_samples]))
        if power_callback._power_samples
        else 0
    )

    # Compute final GSNR
    gsnr_final = gsnr_tracker.get_summary()
    
    # Compute final waste metrics
    waste_metrics = waste_quantifier.compute_waste_metrics()
    
    # Phase transition summary
    phase_summary = phase_detector.get_summary()
    
    # LR-Loss correlation
    lr_loss_corr = lr_loss_tracker.compute_correlation()

    results = {
        "task": task_name,
        "seed": seed,
        "learning_rate": lr,
        "profile": profile,
        "model": MODEL_NAME,
        "train_runtime_s": total_time,
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_result,
        "energy_kwh": power_callback.cumulative_kwh,
        "power_avg_watts": avg_power,
        "ler_final": ler_tracker.get_diagnostics(),
        "gsnr_final": {
            "gsnr_global": gsnr_final.get("gsnr_per_layer", {}).get("__global__", 
                           gsnr_tracker.gsnr_global_history[-1] if gsnr_tracker.gsnr_global_history else None),
            "num_layers_tracked": gsnr_final["num_layers_tracked"],
            "step": gsnr_final["step"],
        },
        "waste_metrics": waste_metrics,
        "phase_summary": phase_summary,
        "lr_loss_correlation": lr_loss_corr,
        "timestamp": datetime.now().isoformat(),
        "hw_config": {k: v for k, v in hw_cfg.items() if k != "max_samples"},
        "evaluated_best_model": True,  # FIX #9 flag
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Log final summary to W&B
    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary.update({
                    "final/eval_loss": eval_result.get("eval_loss"),
                    "final/eval_accuracy": eval_result.get(
                        "eval_accuracy",
                        eval_result.get("eval_matthews_correlation",
                                        eval_result.get("eval_pearsonr")),
                    ),
                    "final/train_loss": train_result.training_loss,
                    "final/energy_kwh": power_callback.cumulative_kwh,
                    "final/power_avg_watts": avg_power,
                    "final/runtime_s": total_time,
                    "final/ler": ler_tracker.get_diagnostics().get("ler"),
                    "final/rho_vg": ler_tracker.get_diagnostics().get("rho_vg"),
                    "final/gsnr_global": gsnr_tracker.gsnr_global_history[-1] if gsnr_tracker.gsnr_global_history else None,
                    "final/waste_ratio": waste_metrics["waste_ratio"],
                    "final/waste_ci_95": f"[{waste_metrics['ci_95_low']:.3f}, {waste_metrics['ci_95_high']:.3f}]",
                    "final/phase": phase_summary["current_phase"],
                    "final/num_phase_transitions": len(phase_summary["transitions"]),
                    "final/lr_loss_correlation": lr_loss_corr["correlation"] if lr_loss_corr else None,
                    "final/evaluated_best_model": True,
                })
        except Exception:
            pass

    print(f"\n  ── Final Results ──────────────────────────────────────")
    print(f"  Eval metrics: {eval_result}")
    print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
    print(f"  Time: {total_time:.1f}s")
    if gsnr_tracker.gsnr_global_history:
        print(f"  GSNR (global): {gsnr_tracker.gsnr_global_history[-1]:.4f}")
    print(f"  Waste ratio (train-loss): {waste_metrics['waste_ratio']:.3f} "
          f"(95% CI: [{waste_metrics['ci_95_low']:.3f}, {waste_metrics['ci_95_high']:.3f}])")
    print(f"  Waste ratio (eval-metric): {waste_metrics.get('eval_waste_ratio', 0.0):.3f} "
          f"(best at step {waste_metrics.get('best_eval_step')})")
    print(f"  Phase: {phase_summary['current_phase']} "
          f"({len(phase_summary['transitions'])} transitions)")
    if lr_loss_corr:
        print(f"  LR-Loss correlation: {lr_loss_corr['correlation']:.4f} "
              f"(p={lr_loss_corr['p_value']:.4f})")
    print(f"  Saved: {results_path}")
    print(f"  Best model evaluated: Yes (load_best_model_at_end=True)")

    # ── Cleanup ───────────────────────────────────────────────────────
    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── W&B: finish this run so the next experiment gets a fresh one ──
    if use_wandb:
        _ensure_wandb_finished()

    return results


def main():
    parser = argparse.ArgumentParser(description="LERNA Baseline: ModernBERT on GLUE (Full Diagnostics)")
    parser.add_argument(
        "--mode", choices=["smoke", "full", "custom"], default="smoke",
        help="smoke=1 seed SST-2 only (3050 OK), full=10 seeds x 8 tasks (5090), custom=use --tasks/--seeds",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to run (e.g., sst2 qnli)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds (e.g., 42 43 44)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./experiments/baseline", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", default="lerna-baseline", help="W&B project name")
    parser.add_argument("--wandb-group", default=None,
                        help="W&B group name (default: auto-generated from mode + timestamp)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples per task (default: 2000 laptop, 25000 server-full, None for --unlimited)")
    parser.add_argument("--unlimited", action="store_true",
                        help="Use full dataset without sample cap (server only)")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Override number of seeds in full mode (default: 10)")
    args = parser.parse_args()

    profile = detect_device_profile()

    # ── Determine tasks and seeds ─────────────────────────────────────
    if args.mode == "smoke":
        tasks = ["sst2"]
        seeds = [42]
        print("\n  SMOKE TEST MODE (1 seed, SST-2 only)")
    elif args.mode == "full":
        tasks = list(GLUE_TASK_CONFIG.keys())
        n_seeds = args.num_seeds or 10
        seeds = list(range(42, 42 + n_seeds))
        print(f"\n  FULL MODE ({len(seeds)} seeds x {len(tasks)} tasks = {len(seeds)*len(tasks)} runs)")
    else:
        tasks = args.tasks or ["sst2"]
        seeds = args.seeds or [42]

    # ── W&B group: unique per invocation so runs are grouped properly ─
    wandb_group = args.wandb_group or f"{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  LERNA Baseline v2 (Full Diagnostics)")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  LR: {args.lr}")
    print(f"  Profile: {profile}")
    print(f"  Features: LER + GSNR + Waste CI + Phase Detection + ETA")
    print(f"  Best model eval: YES (load_best_model_at_end=True)")
    if args.wandb:
        print(f"  W&B project: {args.wandb_project}")
        print(f"  W&B group: {wandb_group}")
        print(f"  W&B viz: heatmaps, training dynamics, layer analysis")

    # ── Sample cap logic ──────────────────────────────────────────────
    if args.max_samples is not None:
        effective_max_samples = args.max_samples
    elif args.unlimited:
        effective_max_samples = None
    elif profile == "server":
        effective_max_samples = 25000
    else:
        effective_max_samples = 2000

    print(f"  Max samples/task: {effective_max_samples or 'unlimited'}")

    # ── Estimate total time ───────────────────────────────────────────
    total_runs = len(tasks) * len(seeds)
    est_time_per_run = 120 if profile == "server" else 300  # rough estimate in seconds
    est_total = total_runs * est_time_per_run
    print(f"  Total runs: {total_runs}")
    print(f"  Estimated total time: ~{timedelta(seconds=est_total)}")
    print(f"  ═══════════════════════════════════════════════════════\n")

    # ── Ensure no stale W&B run from a previous crash ─────────────────
    if args.wandb:
        _ensure_wandb_finished()

    # ── Main experiment loop ──────────────────────────────────────────
    all_results = []
    run_idx = 0
    overall_start = time.time()

    for task in tasks:
        for seed in seeds:
            run_idx += 1
            
            # Overall ETA
            if run_idx > 1:
                elapsed = time.time() - overall_start
                avg_per_run = elapsed / (run_idx - 1)
                remaining = (total_runs - run_idx + 1) * avg_per_run
                print(f"\n  ═══ Run {run_idx}/{total_runs} | "
                      f"Overall ETA: {timedelta(seconds=int(remaining))} ═══")
            else:
                print(f"\n  ═══ Run {run_idx}/{total_runs} ═══")
            
            # Resolve per-task hyperparameters
            task_hp = TASK_HP_OVERRIDES.get(task, {})
            task_lr = task_hp.get("learning_rate", args.lr)
            task_epochs = task_hp.get("num_epochs", 3)
            task_warmup = task_hp.get("warmup_ratio", 0.1)
            task_patience = task_hp.get("early_stopping_patience", 5)
            task_best_metric = task_hp.get("metric_for_best_model", "eval_loss")
            task_greater_is_better = task_hp.get("greater_is_better", False)
            task_init_mnli = task_hp.get("init_from_mnli", False)

            try:
                result = run_single_experiment(
                    task_name=task,
                    seed=seed,
                    lr=task_lr,
                    profile=profile,
                    base_output_dir=args.output_dir,
                    use_wandb=args.wandb,
                    max_samples_override=effective_max_samples,
                    run_idx=run_idx,
                    total_runs=total_runs,
                    wandb_project=args.wandb_project,
                    wandb_group=wandb_group,
                    num_epochs=task_epochs,
                    warmup_ratio=task_warmup,
                    early_stopping_patience=task_patience,
                    metric_for_best_model=task_best_metric,
                    greater_is_better=task_greater_is_better,
                    init_from_mnli=task_init_mnli,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  FAILED: {task} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"task": task, "seed": seed, "error": str(e)})
                if args.wandb:
                    _ensure_wandb_finished()

    # ── Final summary ─────────────────────────────────────────────────
    summary_path = os.path.join(args.output_dir, "baseline_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # ── Cross-run W&B summary ─────────────────────────────────────────
    if args.wandb:
        log_cross_run_summary(all_results, tasks, args.wandb_project, wandb_group)

    total_elapsed = time.time() - overall_start

    print(f"\n{'='*60}")
    print(f"  BASELINE COMPLETE: {len(all_results)} runs")
    print(f"  Summary: {summary_path}")
    print(f"  Total wall time: {timedelta(seconds=int(total_elapsed))}")

    successful = [r for r in all_results if "error" not in r]
    if successful:
        total_kwh = sum(r.get("energy_kwh", 0) for r in successful)
        total_time = sum(r.get("train_runtime_s", 0) for r in successful)
        print(f"  Total energy: {total_kwh:.6f} kWh")
        print(f"  Total compute time: {total_time:.1f}s ({total_time/3600:.2f}h)")

        # Print per-task summary table with NEW metrics
        print(f"\n  {'Task':<8} {'Seeds':>5} {'Avg Acc':>10} {'Std':>8} {'Avg kWh':>10} {'GSNR':>10} {'Waste':>8} {'Phase':>14}")
        print(f"  {'-'*75}")
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
            gsnrs = [r.get("gsnr_final", {}).get("gsnr_global", 0) for r in task_results]
            wastes = [r.get("waste_metrics", {}).get("waste_ratio", 0) for r in task_results]
            phases = [r.get("phase_summary", {}).get("current_phase", "?") for r in task_results]
            # Most common phase
            from collections import Counter
            common_phase = Counter(phases).most_common(1)[0][0] if phases else "?"
            
            gsnr_vals = [g for g in gsnrs if g is not None and g > 0]
            gsnr_str = f"{np.mean(gsnr_vals):.4f}" if gsnr_vals else "N/A"
            
            # GSNR outlier detection: flag seeds with GSNR > 5x median
            gsnr_outlier_warning = ""
            if len(gsnr_vals) >= 3:
                gsnr_median = np.median(gsnr_vals)
                gsnr_outliers = [
                    (r["seed"], r.get("gsnr_final", {}).get("gsnr_global", 0))
                    for r in task_results
                    if r.get("gsnr_final", {}).get("gsnr_global", 0) is not None
                    and r.get("gsnr_final", {}).get("gsnr_global", 0) > gsnr_median * 5
                ]
                if gsnr_outliers:
                    gsnr_outlier_warning = f" \u26a0\ufe0f GSNR outliers: {gsnr_outliers}"
            
            print(
                f"  {task:<8} {len(task_results):>5} "
                f"{np.mean(accs):>10.4f} {np.std(accs):>8.4f} "
                f"{np.mean(kwhs):>10.6f} {gsnr_str:>10} "
                f"{np.mean(wastes):>8.3f} {common_phase:>14}"
                f"{gsnr_outlier_warning}"
            )

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
