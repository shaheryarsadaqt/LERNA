"""
Phase 1: Simple Baselines for LERNA Comparison

These 6 baselines test whether LERNA's LER-guided switching is justified.
If any simple baseline matches LERNA's accuracy-energy tradeoff, then
LER's sophistication adds no value.

All baselines implement the HuggingFace TrainerCallback interface and
track the same metrics (steps_skipped, energy_saved, accuracy) for
direct comparison with LERNASwitchingCallback.

Baselines:
  1. GradientNormSkippingCallback  - skip when ||g|| < threshold
  2. RandomStepSkippingCallback    - skip randomly at matched rate
  3. WeightFreezingCallback        - freeze weights during LER plateaus
  4. ReducedTotalStepsCallback     - train fewer total steps
  5. CosineAnnealingWarmRestartsCallback - cosine LR with restarts
  6. Early stopping with optimal patience (use HF EarlyStoppingCallback)

FIXES (2026-04-15):
  - RandomStepSkippingCallback: TRUE backward-pass skipping via
    trainer.should_skip_backward in on_step_begin (was only recording
    skips after backward already ran).
  - WeightFreezingCallback: TRUE backward-pass skipping via
    trainer.should_skip_backward in on_step_begin.
  - CosineAnnealingWarmRestartsCallback: Robust optimizer capture via
    on_train_begin + trainer reference fallback.
  - All callbacks: Added get_activation_summary() for post-run validation.
"""

import math
import json
import os
import random
import logging
from collections import defaultdict
from typing import Optional, Dict, Any

import torch
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from transformers import TrainerCallback

# Import EnergyTracker for real power telemetry
try:
    from lerna.callbacks.lerna_switching import EnergyTracker
    ENERGY_TRACKER_AVAILABLE = True
except ImportError:
    EnergyTracker = None
    ENERGY_TRACKER_AVAILABLE = False

# Import single source of truth for momentum extrapolation
from lerna.utils.momentum import apply_momentum_extrapolation

logger = logging.getLogger(__name__)


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


# ===================================================================
# Shared mixin for consistent stats tracking across all baselines
# ===================================================================

class _BaselineStatsMixin:
    """Common statistics tracking for all baseline callbacks."""

    def _init_stats(self, baseline_name: str, wandb_enabled: bool = True,
                    energy_per_skip: float = None, use_real_energy: bool = True):
        self.baseline_name = baseline_name
        self.wandb_enabled = wandb_enabled
        self.steps_skipped = 0
        self.total_energy_saved = 0.0
        self.skip_decisions = []          # True/False per step
        self.accuracy_during_skip = []
        self.accuracy_during_normal = []
        self.last_accuracy = 0.0
        self.active_skipping = False
        
        # Real energy tracking via pynvml
        self._energy_tracker: Optional[Any] = None
        self._use_real_energy = use_real_energy and ENERGY_TRACKER_AVAILABLE
        
        if self._use_real_energy:
            try:
                self._energy_tracker = EnergyTracker()
                logger.info(f"  [{baseline_name}] Using real power telemetry via pynvml")
            except Exception as e:
                logger.warning(f"  [{baseline_name}] Failed to init EnergyTracker: {e}")
                self._use_real_energy = False
        
        # Fallback: static estimate (kWh). ~0.6 Wh based on RTX 5090
        # at ~575W TDP with backward pass ~60% of step cost.
        self._energy_per_skip = energy_per_skip if energy_per_skip is not None else 0.0006
        
        # Track energy per step for averaging
        self._measured_energy_per_skip: list = []
        
        # Trainer reference for TRUE backward-pass skipping
        self._trainer = None

    def _record_skip(self, state):
        self.steps_skipped += 1
        self.skip_decisions.append(True)
        # Defer energy measurement to on_step_end so we sample power
        # AFTER forward pass (during which GPU is fully loaded).
        self._pending_skip_record = True

    def _flush_pending_energy(self):
        """Call from on_step_end to actually record energy for a deferred skip."""
        if not getattr(self, "_pending_skip_record", False):
            return
        self._pending_skip_record = False
        if self._use_real_energy and self._energy_tracker is not None:
            try:
                current_power_w = self._energy_tracker.get_current_power_w()
                backward_duration_s = 0.060
                energy_kwh = (current_power_w * backward_duration_s) / 3600 / 1000
                self._measured_energy_per_skip.append(energy_kwh)
                self.total_energy_saved += energy_kwh
            except Exception:
                self.total_energy_saved += self._energy_per_skip
        else:
            self.total_energy_saved += self._energy_per_skip

    def _record_normal(self, state):
        self.skip_decisions.append(False)

    def _apply_momentum_extrapolation(self, optimizer, lr):
        """Apply momentum-driven weight extrapolation during skip phases.

        Uses the optimizer's momentum buffer (SGD) or first moment
        estimate (Adam/AdamW) as a proxy for the gradient direction.
        Delegates to single source of truth in lerna.utils.momentum.
        """
        if optimizer is None:
            return
        apply_momentum_extrapolation(optimizer)

    def _log_periodic(self, state, extra: dict = None):
        if not (self.wandb_enabled and _wandb_active()):
            return
        if state.global_step % 10 != 0:
            return
        data = {
            f"baseline/{self.baseline_name}/steps_skipped": self.steps_skipped,
            f"baseline/{self.baseline_name}/energy_saved_kwh": self.total_energy_saved,
            f"baseline/{self.baseline_name}/skip_ratio": (
                self.steps_skipped / max(state.global_step, 1)
            ),
            "step": state.global_step,
        }
        if extra:
            data.update(extra)
        wandb.log(data)

    def _on_evaluate_stats(self, metrics, state):
        """Track primary metric, regardless of task. Handles eval_accuracy
        (most tasks), eval_matthews_correlation (CoLA), eval_pearson (STS-B)."""
        if not metrics:
            return
        metric_value = (
            metrics.get("eval_accuracy")
            or metrics.get("eval_matthews_correlation")
            or metrics.get("eval_pearson")
            or metrics.get("eval_pearsonr")
        )
        if metric_value is None:
            return
        self.last_accuracy = metric_value
        if self.active_skipping:
            self.accuracy_during_skip.append(metric_value)
        else:
            self.accuracy_during_normal.append(metric_value)

    def get_activation_summary(self) -> Dict[str, Any]:
        """Return a summary of whether this baseline actually activated.
        
        Used for post-run validation to catch silent failures.
        """
        total_decisions = len(self.skip_decisions)
        return {
            "baseline_name": self.baseline_name,
            "steps_skipped": self.steps_skipped,
            "total_decisions": total_decisions,
            "skip_ratio": self.steps_skipped / max(total_decisions, 1),
            "energy_saved_kwh": self.total_energy_saved,
            "activated": self.steps_skipped > 0,
            "n_evals_during_skip": len(self.accuracy_during_skip),
            "n_evals_during_normal": len(self.accuracy_during_normal),
        }

    def _on_train_end_stats(self, args, state):
        total_steps = state.global_step
        skip_ratio = self.steps_skipped / max(total_steps, 1)

        avg_acc_skip = (
            sum(self.accuracy_during_skip) / len(self.accuracy_during_skip)
            if self.accuracy_during_skip else 0.0
        )
        avg_acc_normal = (
            sum(self.accuracy_during_normal) / len(self.accuracy_during_normal)
            if self.accuracy_during_normal else 0.0
        )
        
        avg_energy_per_skip = self._energy_per_skip
        energy_source = "static estimate"
        if self._measured_energy_per_skip:
            avg_energy_per_skip = sum(self._measured_energy_per_skip) / len(self._measured_energy_per_skip)
            energy_source = "measured"

        stats = {
            "baseline_name": self.baseline_name,
            "total_steps": total_steps,
            "steps_skipped": self.steps_skipped,
            "skip_ratio": skip_ratio,
            "energy_saved_kwh": self.total_energy_saved,
            "avg_energy_per_skip_kwh": avg_energy_per_skip,
            "energy_source": energy_source,
            "avg_acc_during_skip": avg_acc_skip,
            "avg_acc_during_normal": avg_acc_normal,
            "acc_difference": avg_acc_skip - avg_acc_normal,
            "n_evals_during_skip": len(self.accuracy_during_skip),
            "n_evals_during_normal": len(self.accuracy_during_normal),
        }

        print(f"\n{'=' * 60}")
        print(f"  BASELINE: {self.baseline_name}")
        print(f"{'=' * 60}")
        print(f"  Total steps: {total_steps}")
        print(f"  Steps skipped: {self.steps_skipped} ({skip_ratio * 100:.1f}%)")
        print(f"  Energy saved: {self.total_energy_saved:.6f} kWh ({energy_source})")
        print(f"  Avg energy/skip: {avg_energy_per_skip:.6f} kWh")
        print(f"  Acc during skip: {avg_acc_skip:.4f} ({len(self.accuracy_during_skip)} evals)")
        print(f"  Acc during normal: {avg_acc_normal:.4f} ({len(self.accuracy_during_normal)} evals)")
        if self.steps_skipped == 0:
            print(f"  *** WARNING: Baseline did NOT activate! 0 steps skipped. ***")
        print(f"{'=' * 60}")

        # Save stats
        stats_path = os.path.join(args.output_dir, f"baseline_{self.baseline_name}_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"  Stats saved: {stats_path}")

        # Log to W&B
        if self.wandb_enabled and _wandb_active():
            wandb.log({
                f"baseline/{self.baseline_name}/final/skip_ratio": skip_ratio,
                f"baseline/{self.baseline_name}/final/energy_saved_kwh": self.total_energy_saved,
                f"baseline/{self.baseline_name}/final/avg_acc_skip": avg_acc_skip,
                f"baseline/{self.baseline_name}/final/avg_acc_normal": avg_acc_normal,
            })

        return stats


# ===================================================================
# Baseline 1: Gradient Norm Thresholding (Adaptive)
# ===================================================================

class GradientNormSkippingCallback(TrainerCallback, _BaselineStatsMixin):
    """Skip backward pass when gradient norm falls below an adaptive threshold.

    Tests whether LER is better than its simplest component: just
    checking if gradients are small.

    Uses adaptive percentile-based thresholding for fair comparison:
    during a calibration window, gradient norms are collected. The
    threshold is then set at the percentile that produces the target
    skip rate. Recalibrated periodically for non-stationarity.

    HOOK LIFECYCLE (Transformers 4.41+):
        on_step_begin -> forward -> loss -> backward -> on_pre_optimizer_step
        -> optimizer.step -> optimizer.zero_grad -> on_step_end
    """

    def __init__(
        self,
        target_skip_rate: float = 0.33,
        calibration_steps: int = 200,
        recalibrate_every: int = 500,
        min_step: int = 0,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
        min_calibration_samples: int = 50,
        min_coefficient_of_variation: float = 0.01,
        rolling_window_size: int = 1000,
    ):
        super().__init__()
        self._init_stats("grad_norm_skip", wandb_enabled, energy_per_skip)
        self.target_skip_rate = target_skip_rate
        self.calibration_steps = calibration_steps
        self.recalibrate_every = recalibrate_every
        self.min_step = min_step
        self.min_calibration_samples = min_calibration_samples
        self.min_coefficient_of_variation = min_coefficient_of_variation
        self.rolling_window_size = rolling_window_size
        self._optimizer = None
        self._model = None
        self._grad_norm_history = []
        self._rolling_grad_norms = []
        self._adaptive_threshold = None
        self._last_calibration_step = 0
        self._current_grad_norm = None
        self._calibration_attempts = 0
        self._max_calibration_attempts = 3
        self._predicted_skip = False

    def _compute_grad_norm(self, model) -> float:
        """Compute total gradient norm from model parameters."""
        total_norm_sq = 0.0
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
                has_grad = True
        return float(total_norm_sq ** 0.5) if has_grad else 0.0

    def _validate_calibration_data(self, norms: list) -> tuple:
        if len(norms) < self.min_calibration_samples:
            return False, f"insufficient samples ({len(norms)} < {self.min_calibration_samples})"
        norms_arr = np.array(norms)
        mean_norm = np.mean(norms_arr)
        std_norm = np.std(norms_arr)
        if mean_norm < 1e-10:
            return False, f"mean gradient norm is near zero ({mean_norm:.2e})"
        cv = std_norm / mean_norm
        if cv < self.min_coefficient_of_variation:
            return False, f"insufficient variation (CV={cv:.4f} < {self.min_coefficient_of_variation})"
        return True, f"valid (CV={cv:.4f}, mean={mean_norm:.6f}, std={std_norm:.6f})"

    def _compute_threshold_with_validation(self, norms: list) -> tuple:
        if len(norms) < self.min_calibration_samples:
            return None, 0.0, False
        percentile = self.target_skip_rate * 100
        threshold = float(np.percentile(norms, percentile))
        norms_arr = np.array(norms)
        actual_skip_rate = np.sum(norms_arr < threshold) / len(norms_arr)
        is_valid = abs(actual_skip_rate - self.target_skip_rate) < 0.10
        return threshold, actual_skip_rate, is_valid

    def _calibrate_threshold(self, use_rolling: bool = False) -> bool:
        norms = self._rolling_grad_norms if use_rolling else self._grad_norm_history
        is_valid, reason = self._validate_calibration_data(norms)
        if not is_valid:
            logger.warning(f"  [grad_norm_skip] Calibration validation failed: {reason}")
            return False
        threshold, actual_rate, is_valid = self._compute_threshold_with_validation(norms)
        if threshold is None:
            return False
        if not is_valid:
            logger.warning(
                f"  [grad_norm_skip] Threshold produces skip rate {actual_rate:.2%} "
                f"vs target {self.target_skip_rate:.2%}"
            )
        self._adaptive_threshold = threshold
        logger.info(
            f"  [grad_norm_skip] Calibrated threshold={self._adaptive_threshold:.6f} "
            f"(p{self.target_skip_rate*100:.0f} of {len(norms)} norms, "
            f"actual_skip_rate={actual_rate:.2%})"
        )
        return True

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        if "model" in kwargs:
            self._model = kwargs["model"]
        logger.info(f"  [grad_norm_skip] Initialized with target_skip_rate={self.target_skip_rate:.0%}")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Decide whether to skip backward pass BEFORE it runs."""
        self._current_grad_norm = None
        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        
        trainer = getattr(self, '_trainer', None)
        if trainer is None:
            return control
        
        trainer.should_skip_backward = False
        
        if self._adaptive_threshold is None:
            return control
        if state.global_step < self.min_step:
            return control
        if len(self._grad_norm_history) == 0:
            return control
        if state.global_step < self.min_step + self.calibration_steps:
            return control
        
        prev_grad_norm = float(self._grad_norm_history[-1])
        threshold = float(self._adaptive_threshold)
        should_skip = prev_grad_norm < threshold
        
        trainer.should_skip_backward = should_skip
        
        if should_skip:
            self._predicted_skip = True
            self._record_skip(state)
            self.active_skipping = True
        else:
            self._predicted_skip = False
            self._record_normal(state)
            self.active_skipping = False
        
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model", self._model)
        if model is None:
            return control
        self._current_grad_norm = self._compute_grad_norm(model)
        if self._current_grad_norm is not None and self._current_grad_norm > 0:
            grad_norm_float = float(self._current_grad_norm)
            self._grad_norm_history.append(grad_norm_float)
            self._rolling_grad_norms.append(grad_norm_float)
            if len(self._rolling_grad_norms) > self.rolling_window_size:
                self._rolling_grad_norms = self._rolling_grad_norms[-self.rolling_window_size:]
            if len(self._grad_norm_history) > 5000:
                self._grad_norm_history = self._grad_norm_history[-2500:]
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._flush_pending_energy()
        if state.global_step < self.min_step:
            return control
        if (state.global_step < self.min_step + self.calibration_steps
                and self._adaptive_threshold is None):
            return control
        if self._adaptive_threshold is None:
            is_valid, reason = self._validate_calibration_data(self._grad_norm_history)
            if not is_valid:
                self._calibration_attempts += 1
                if self._calibration_attempts <= self._max_calibration_attempts:
                    return control
                else:
                    if len(self._grad_norm_history) >= 10:
                        self._adaptive_threshold = float(np.median(self._grad_norm_history))
                        self._last_calibration_step = state.global_step
                    else:
                        return control
            success = self._calibrate_threshold(use_rolling=False)
            self._last_calibration_step = state.global_step
            if success:
                print(
                    f"\n  [grad_norm_skip] === CALIBRATION COMPLETE ==="
                    f"\n  [grad_norm_skip] Step: {state.global_step}"
                    f"\n  [grad_norm_skip] Threshold: {self._adaptive_threshold:.6f}"
                    f"\n  [grad_norm_skip] Target skip rate: {self.target_skip_rate:.0%}"
                    f"\n  [grad_norm_skip] Norms collected: {len(self._grad_norm_history)}"
                )
            else:
                return control
        if (self.recalibrate_every > 0
                and state.global_step - self._last_calibration_step >= self.recalibrate_every):
            success = self._calibrate_threshold(use_rolling=True)
            if success:
                self._last_calibration_step = state.global_step
        grad_norm = self._current_grad_norm
        if grad_norm is not None and grad_norm > 0:
            self._log_periodic(state, {
                f"baseline/{self.baseline_name}/grad_norm": grad_norm,
                f"baseline/{self.baseline_name}/threshold": self._adaptive_threshold or 0,
                f"baseline/{self.baseline_name}/predicted_skip": self._predicted_skip,
            })
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        stats = self._on_train_end_stats(args, state)
        if self._adaptive_threshold is not None:
            print(f"  Final adaptive threshold: {self._adaptive_threshold:.6f}")
        return control


# ===================================================================
# Baseline 2: Random Step Skipping
# ===================================================================

class RandomStepSkippingCallback(TrainerCallback, _BaselineStatsMixin):
    """Skip backward pass randomly at a target skip rate.

    Tests whether the *selection* of which steps to skip matters,
    or whether just reducing compute by any fraction is sufficient.
    If random skipping matches LERNA, then LER adds no value.

    FIX (2026-04-15): Now implements TRUE backward-pass skipping by
    setting trainer.should_skip_backward = True in on_step_begin,
    BEFORE the backward pass runs. Previously only recorded skips
    in on_step_end after backward had already executed.
    """

    def __init__(
        self,
        target_skip_rate: float = 0.22,
        min_step: int = 100,
        seed: int = 42,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("random_skip", wandb_enabled, energy_per_skip)
        self.target_skip_rate = target_skip_rate
        self.min_step = min_step
        self._rng = random.Random(seed)
        self._optimizer = None
        self._model = None

    def on_step_end(self, args, state, control, **kwargs):
        self._flush_pending_energy()
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._on_train_end_stats(args, state)
        return control


# ===================================================================
# Baseline 3: Weight Freezing During Skip Phases (TRUE backward skip)
# ===================================================================

class RandomStepSkippingCallback(TrainerCallback, _BaselineStatsMixin):
    """Skip backward pass randomly at a target skip rate.

    Tests whether the *selection* of which steps to skip matters,
    or whether just reducing compute by any fraction is sufficient.
    If random skipping matches LERNA, then LER adds no value.

    FIX (2026-04-15): Now implements TRUE backward-pass skipping by
    setting trainer.should_skip_backward = True in on_step_begin,
    BEFORE the backward pass runs. Previously only recorded skips
    in on_step_end after backward had already executed.
    """

    def __init__(
        self,
        target_skip_rate: float = 0.22,
        min_step: int = 100,
        seed: int = 42,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("random_skip", wandb_enabled, energy_per_skip)
        self.target_skip_rate = target_skip_rate
        self.min_step = min_step
        self._rng = random.Random(seed)
        self._optimizer = None
        self._model = None

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        if "model" in kwargs:
            self._model = kwargs["model"]
        print(f"  [random_skip] Initialized: target_skip_rate={self.target_skip_rate:.0%}, "
              f"min_step={self.min_step}, trainer={'linked' if self._trainer else 'NOT linked'}")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Decide whether to skip backward BEFORE it runs.
        
        FIX: Sets trainer.should_skip_backward = True so that
        Phase12Trainer.training_step() skips loss.backward() entirely.
        The trainer applies momentum extrapolation instead.
        """
        trainer = getattr(self, '_trainer', None)
        if trainer is not None:
            trainer.should_skip_backward = False  # Default: don't skip
        
        if state.global_step < self.min_step:
            self._record_normal(state)
            return control
        
        should_skip = self._rng.random() < self.target_skip_rate
        
        if should_skip:
            # Set the flag BEFORE backward runs
            if trainer is not None:
                trainer.should_skip_backward = True
            self._record_skip(state)
            self.active_skipping = True
        else:
            self._record_normal(state)
            self.active_skipping = False
        
        self._log_periodic(state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._on_train_end_stats(args, state)
        return control


# ===================================================================
# Baseline 3: Weight Freezing During Skip Phases (TRUE backward skip)
# ===================================================================

class WeightFreezingCallback(TrainerCallback, _BaselineStatsMixin):
    """Freeze weights entirely during LER-detected plateaus.

    When LER detects a plateau, instead of momentum extrapolation,
    simply freeze all weights (no update at all).

    Tests whether momentum extrapolation actually helps vs. doing nothing.
    Uses the same LER tracker as LERNA for plateau detection.

    FIX (2026-04-15): Now implements TRUE backward-pass skipping via
    trainer.should_skip_backward. Also, the LERTracker is now properly
    fed during training via LERFeedCallback (see run_phase1_2_simple_baselines.py).
    """

    def __init__(
        self,
        ler_tracker,
        threshold: float = None,   # falls back to ler_tracker.task_calibration
        min_step: int = 100,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("weight_freeze", wandb_enabled, energy_per_skip)
        self.ler_tracker = ler_tracker
        self.threshold = threshold
        self.min_step = min_step
        self._ler_update_count = 0  # Track how many times LER was updated

    def on_step_begin(self, args, state, control, **kwargs):
        """Decide whether to skip backward BEFORE it runs.
        
        FIX: Sets trainer.should_skip_backward = True for TRUE skipping.
        Unlike LERNA, we do NOT apply momentum extrapolation (weights frozen).
        The Phase12Trainer checks should_skip_backward and skips loss.backward().
        We also override the trainer's momentum extrapolation by setting a
        special flag.
        """
        trainer = getattr(self, '_trainer', None)
        if trainer is not None:
            trainer.should_skip_backward = False
        
        if state.global_step < self.min_step:
            return control

        diag = self.ler_tracker.get_diagnostics()
        current_ler = diag.get("ler")
        
        if current_ler is None:
            # LER not yet computed (not enough data points)
            return control

        # Use task-calibrated threshold (consistent with phase detector).
        task_threshold = self.ler_tracker.task_calibration.get(
            self.ler_tracker.task, {}
        ).get("ler_threshold", self.threshold)

        if current_ler < task_threshold:
            # Signal to skip backward pass AND freeze weights
            if trainer is not None:
                trainer.should_skip_backward = True
                # Set a flag so trainer does NOT apply momentum extrapolation
                trainer._freeze_weights_no_momentum = True
            self._record_skip(state)
            self.active_skipping = True
        else:
            if trainer is not None:
                trainer._freeze_weights_no_momentum = False
            self._record_normal(state)
            self.active_skipping = False

        self._log_periodic(state, {
            f"baseline/{self.baseline_name}/ler": current_ler,
        })
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Report LER tracker status
        diag = self.ler_tracker.get_diagnostics()
        print(f"  [weight_freeze] LER tracker status: n_steps={diag.get('n_steps', 0)}, "
              f"current_ler={diag.get('ler', 'None')}, phase={diag.get('phase', 'unknown')}")
        self._on_train_end_stats(args, state)
        return control

    def on_step_end(self, args, state, control, **kwargs):
        self._flush_pending_energy()
        return control


# ===================================================================
# Baseline 4: Reduced Total Steps
# ===================================================================

class ReducedTotalStepsCallback(TrainerCallback, _BaselineStatsMixin):
    """Stop training early to match LERNA's compute budget.

    If LERNA skips ~33% of steps, this baseline simply trains for
    67% of the total steps. Tests whether just training less achieves
    the same result as LERNA's selective skipping.
    """

    def __init__(
        self,
        reduction_fraction: float = 0.22,
        total_steps: int = 1000,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("reduced_steps", wandb_enabled, energy_per_skip)
        self.reduction_fraction = reduction_fraction
        self.total_steps = total_steps
        self.max_steps = int(total_steps * (1.0 - reduction_fraction))
        self._stopped = False

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.max_steps and not self._stopped:
            print(
                f"  [{self.baseline_name}] Stopping at step {state.global_step} "
                f"(max_steps={self.max_steps}, reduction={self.reduction_fraction:.0%})"
            )
            control.should_training_stop = True
            self._stopped = True

            remaining = self.total_steps - state.global_step
            self.steps_skipped = remaining
            self.total_energy_saved = remaining * self._energy_per_skip

        self._log_periodic(state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._on_train_end_stats(args, state)
        return control


# ===================================================================
# Baseline 5: Cosine Annealing with Warm Restarts
# ===================================================================

class CosineAnnealingWarmRestartsCallback(TrainerCallback, _BaselineStatsMixin):
    """Replace the default LR schedule with cosine annealing + warm restarts.

    Tests whether phase-aware LR scheduling captures the same benefit
    as LERNA's explicit phase detection and backward-pass skipping.

    This callback overrides the learning rate at each step using
    cosine annealing with warm restarts (Loshchilov & Hutter, 2017).
    It does NOT skip any backward passes.

    FIX (2026-04-15): Robust optimizer capture. HF Trainer does not
    reliably pass optimizer in on_step_begin kwargs. Now captures it
    in on_train_begin and also tries trainer._trainer reference.
    """

    def __init__(
        self,
        T_0: int = 100,
        T_mult: int = 2,
        eta_min: float = 1e-7,
        base_lr: float = 2e-5,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("cosine_warm_restarts", wandb_enabled, energy_per_skip)
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lr = base_lr
        self._optimizer = None
        self._lr_history = []
        self._lr_overrides_applied = 0

    def _compute_lr(self, step: int) -> float:
        """SGDR: cosine annealing with warm restarts."""
        T_cur = step
        T_i = self.T_0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i = max(int(T_i * self.T_mult), T_i + 1)  # guarantee growth even if T_mult==1
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * T_cur / T_i)
        )
        return lr

    def on_train_begin(self, args, state, control, **kwargs):
        """Capture optimizer reference at training start.
        
        FIX: This is the most reliable place to get the optimizer.
        HF Trainer passes it here in transformers >= 4.41.
        """
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
            print(f"  [cosine_restarts] Optimizer captured in on_train_begin: {type(self._optimizer).__name__}")
        self.base_lr = args.learning_rate
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Override learning rate at each step.
        
        FIX: Try multiple sources for optimizer:
        1. kwargs (may be passed by some Trainer versions)
        2. Cached from on_train_begin
        3. Trainer reference (set by run_phase1_2_simple_baselines.py)
        """
        # Try to get optimizer from multiple sources
        if self._optimizer is None:
            if "optimizer" in kwargs:
                self._optimizer = kwargs["optimizer"]
            elif hasattr(self, '_trainer') and self._trainer is not None:
                # Get from trainer reference
                if hasattr(self._trainer, 'optimizer') and self._trainer.optimizer is not None:
                    self._optimizer = self._trainer.optimizer
                    print(f"  [cosine_restarts] Optimizer captured from trainer at step {state.global_step}")

        if self._optimizer is None:
            return control

        new_lr = self._compute_lr(state.global_step)
        self._lr_history.append(new_lr)

        # Override optimizer LR
        for group in self._optimizer.param_groups:
            group["lr"] = new_lr
        self._lr_overrides_applied += 1

        if self.wandb_enabled and _wandb_active() and state.global_step % 10 == 0:
            wandb.log({
                f"baseline/{self.baseline_name}/lr": new_lr,
                "step": state.global_step,
            })

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def get_activation_summary(self) -> Dict[str, Any]:
        """Override to report LR override count instead of skip count."""
        return {
            "baseline_name": self.baseline_name,
            "lr_overrides_applied": self._lr_overrides_applied,
            "optimizer_captured": self._optimizer is not None,
            "lr_history_len": len(self._lr_history),
            "activated": self._lr_overrides_applied > 0,
            "steps_skipped": 0,  # Cosine doesn't skip steps
            "n_evals_during_skip": len(self.accuracy_during_skip),
            "n_evals_during_normal": len(self.accuracy_during_normal),
        }

    def on_train_end(self, args, state, control, **kwargs):
        stats = self._on_train_end_stats(args, state)
        print(f"  [cosine_restarts] LR overrides applied: {self._lr_overrides_applied}")
        print(f"  [cosine_restarts] Optimizer was {'captured' if self._optimizer else 'NOT captured'}")
        if self._lr_overrides_applied == 0:
            print(f"  *** WARNING: Cosine restarts did NOT activate! Optimizer was never captured. ***")

        lr_path = os.path.join(args.output_dir, f"baseline_{self.baseline_name}_lr_schedule.json")
        with open(lr_path, "w") as f:
            json.dump(self._lr_history, f)
        print(f"  LR schedule saved: {lr_path}")
        return control


# ===================================================================
# Factory function to create all baselines
# ===================================================================

def create_all_baselines(
    ler_tracker=None,
    target_skip_rate: float = 0.22,
    total_steps: int = 1000,
    base_lr: float = 2e-5,
    seed: int = 42,
    wandb_enabled: bool = True,
    energy_per_skip: float = None,
) -> dict:
    """Create all 6 baseline callbacks for comparison.

    Returns a dict mapping baseline name to callback instance.
    Baseline 6 (early stopping) uses HF's built-in EarlyStoppingCallback
    and is not included here; add it separately with different patience values.
    """
    baselines = {}

    baselines["grad_norm_skip"] = GradientNormSkippingCallback(
        target_skip_rate=target_skip_rate,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    baselines["random_skip"] = RandomStepSkippingCallback(
        target_skip_rate=target_skip_rate,
        seed=seed,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    if ler_tracker is not None:
        baselines["weight_freeze"] = WeightFreezingCallback(
            ler_tracker=ler_tracker,
            wandb_enabled=wandb_enabled,
            energy_per_skip=energy_per_skip,
        )

    baselines["reduced_steps"] = ReducedTotalStepsCallback(
        reduction_fraction=target_skip_rate,
        total_steps=total_steps,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    baselines["cosine_warm_restarts"] = CosineAnnealingWarmRestartsCallback(
        T_0=max(total_steps // 10, 50),
        base_lr=base_lr,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    return baselines
