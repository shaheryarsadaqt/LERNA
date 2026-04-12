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

    def _record_skip(self, state):
        self.steps_skipped += 1
        self.skip_decisions.append(True)
        
        # Use real energy measurement if available
        if self._use_real_energy and self._energy_tracker is not None:
            try:
                # Get current power and compute energy for a typical backward pass duration
                # Backward pass is ~60% of total step time, typically 50-100ms on GPU
                current_power_w = self._energy_tracker.get_current_power_w()
                # Estimate backward pass duration: ~60% of step, assume 100ms typical
                backward_duration_s = 0.060  # 60ms for backward pass
                energy_kwh = (current_power_w * backward_duration_s) / 3600 / 1000
                self._measured_energy_per_skip.append(energy_kwh)
                self.total_energy_saved += energy_kwh
            except Exception as e:
                # Fallback to static estimate
                self.total_energy_saved += self._energy_per_skip
        else:
            self.total_energy_saved += self._energy_per_skip

    def _record_normal(self, state):
        self.skip_decisions.append(False)

    def _apply_momentum_extrapolation(self, optimizer, lr):
        """Apply momentum-driven weight extrapolation during skip phases.

        Uses the optimizer's momentum buffer (SGD) or first moment
        estimate (Adam/AdamW) as a proxy for the gradient direction.
        This is the same mechanism used by LERNA during plateau phases.
        """
        if optimizer is None:
            return
        with torch.no_grad():
            for group in optimizer.param_groups:
                for param in group["params"]:
                    if not param.requires_grad or param not in optimizer.state:
                        continue
                    p_state = optimizer.state[param]
                    if "momentum_buffer" in p_state:
                        param.data.add_(p_state["momentum_buffer"], alpha=-lr)
                    elif "exp_avg" in p_state:
                        param.data.add_(p_state["exp_avg"], alpha=-lr)

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
        if metrics and "eval_accuracy" in metrics:
            self.last_accuracy = metrics["eval_accuracy"]
            if self.active_skipping:
                self.accuracy_during_skip.append(self.last_accuracy)
            else:
                self.accuracy_during_normal.append(self.last_accuracy)

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
        
        # Compute average measured energy per skip if available
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
    during a calibration window (first `calibration_steps` steps),
    gradient norms are collected. The threshold is then set at the
    percentile that produces the target skip rate. The threshold is
    recalibrated periodically to handle non-stationarity.

    This ensures the skip rate approximately matches LERNA's observed
    rate regardless of model architecture, hardware, or grad clipping.

    HOOK LIFECYCLE (Transformers 4.41+):
        on_step_begin -> forward -> loss -> backward -> on_pre_optimizer_step
        -> optimizer.step -> optimizer.zero_grad -> on_step_end

    The on_pre_optimizer_step hook fires AFTER backward but BEFORE
    optimizer.step/zero_grad, making it the ideal place to capture
    gradient norms while they are still live.
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
        self._model = None  # Cached from on_train_begin
        self._grad_norm_history = []
        self._rolling_grad_norms = []  # Rolling window for recalibration
        self._adaptive_threshold = None
        self._last_calibration_step = 0
        self._current_grad_norm = None  # Captured in on_pre_optimizer_step
        self._calibration_attempts = 0
        self._max_calibration_attempts = 3

    def _compute_grad_norm(self, model) -> float:
        """Compute total gradient norm from model parameters.

        Returns 0.0 if no gradients are found.
        """
        total_norm_sq = 0.0
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
                has_grad = True
        return total_norm_sq ** 0.5 if has_grad else 0.0

    def _validate_calibration_data(self, norms: list) -> tuple:
        """Validate that calibration data has sufficient diversity.

        Returns (is_valid, reason) tuple.
        """
        if len(norms) < self.min_calibration_samples:
            return False, f"insufficient samples ({len(norms)} < {self.min_calibration_samples})"

        norms_arr = np.array(norms)
        mean_norm = np.mean(norms_arr)
        std_norm = np.std(norms_arr)

        # Check for zero mean (all gradients are zero)
        if mean_norm < 1e-10:
            return False, f"mean gradient norm is near zero ({mean_norm:.2e})"

        # Check coefficient of variation (CV = std/mean)
        cv = std_norm / mean_norm
        if cv < self.min_coefficient_of_variation:
            return False, f"insufficient variation (CV={cv:.4f} < {self.min_coefficient_of_variation})"

        return True, f"valid (CV={cv:.4f}, mean={mean_norm:.6f}, std={std_norm:.6f})"

    def _compute_threshold_with_validation(self, norms: list) -> tuple:
        """Compute threshold and validate it produces target skip rate.

        Returns (threshold, actual_skip_rate, is_valid) tuple.
        """
        if len(norms) < self.min_calibration_samples:
            return None, 0.0, False

        percentile = self.target_skip_rate * 100
        threshold = float(np.percentile(norms, percentile))

        # Validate: compute actual skip rate this threshold would produce
        norms_arr = np.array(norms)
        actual_skip_rate = np.sum(norms_arr < threshold) / len(norms_arr)

        # Allow 10% tolerance from target
        is_valid = abs(actual_skip_rate - self.target_skip_rate) < 0.10

        return threshold, actual_skip_rate, is_valid

    def _calibrate_threshold(self, use_rolling: bool = False) -> bool:
        """Set threshold at the percentile matching target_skip_rate.

        If target_skip_rate=0.33, we want to skip the bottom 33% of
        gradient norms, so threshold = 33rd percentile of observed norms.

        Args:
            use_rolling: If True, use rolling window for recalibration.
                        If False, use full history for initial calibration.

        Returns:
            True if calibration succeeded, False otherwise.
        """
        norms = self._rolling_grad_norms if use_rolling else self._grad_norm_history

        # Validate calibration data diversity
        is_valid, reason = self._validate_calibration_data(norms)
        if not is_valid:
            logger.warning(f"  [grad_norm_skip] Calibration validation failed: {reason}")
            return False

        # Compute threshold with validation
        threshold, actual_rate, is_valid = self._compute_threshold_with_validation(norms)
        if threshold is None:
            logger.warning(f"  [grad_norm_skip] Failed to compute threshold")
            return False

        percentile = self.target_skip_rate * 100

        # Log warning if actual skip rate deviates significantly
        if not is_valid:
            logger.warning(
                f"  [grad_norm_skip] Threshold produces skip rate {actual_rate:.2%} "
                f"vs target {self.target_skip_rate:.2%}"
            )

        self._adaptive_threshold = threshold
        logger.info(
            f"  [grad_norm_skip] Calibrated threshold={self._adaptive_threshold:.6f} "
            f"(p{percentile:.0f} of {len(norms)} norms, "
            f"range=[{min(norms):.6f}, {max(norms):.6f}], "
            f"actual_skip_rate={actual_rate:.2%})"
        )
        return True

    def on_train_begin(self, args, state, control, **kwargs):
        """Capture optimizer and model references at training start."""
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        if "model" in kwargs:
            self._model = kwargs["model"]
        logger.info(f"  [grad_norm_skip] Initialized with target_skip_rate={self.target_skip_rate:.0%}")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        """Reset per-step state."""
        self._current_grad_norm = None
        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Capture gradient norm AFTER backward, BEFORE optimizer.step.

        FLAW 6 FIX: Use pre-clip gradient norm from trainer if available.
        The trainer's _pre_clip_grad_norm is captured BEFORE clipping.
        The on_pre_optimizer_step hook sees CLIPPED gradients, so we must
        use the trainer's pre-clip value for accurate calibration.
        """
        model = kwargs.get("model", self._model)
        if model is None:
            return control

        # FLAW 6 FIX: Use pre-clip gradient norm from trainer
        # This is the TRUE gradient norm before clipping corrupted it
        trainer = getattr(self, '_trainer', None)
        if trainer is not None and hasattr(trainer, '_pre_clip_grad_norm'):
            self._current_grad_norm = trainer._pre_clip_grad_norm
        else:
            # Fallback: compute from model (will be clipped if max_grad_norm set)
            self._current_grad_norm = self._compute_grad_norm(model)

        # Store in history for adaptive threshold calibration
        if self._current_grad_norm is not None and self._current_grad_norm > 0:
            self._grad_norm_history.append(self._current_grad_norm)
            # Rolling window for non-stationarity handling
            self._rolling_grad_norms.append(self._current_grad_norm)
            if len(self._rolling_grad_norms) > self.rolling_window_size:
                self._rolling_grad_norms = self._rolling_grad_norms[-self.rolling_window_size:]
            # Cap full history to prevent unbounded memory growth
            if len(self._grad_norm_history) > 5000:
                self._grad_norm_history = self._grad_norm_history[-2500:]

        return control

    def on_step_end(self, args, state, control, **kwargs):
        """Make skip decision based on gradient norm captured in on_pre_optimizer_step.

        FLAW 8 FIX: The actual backward skipping is handled in training_step
        via the should_skip_backward flag. This method now just records stats.
        """
        grad_norm = self._current_grad_norm

        if grad_norm is None or grad_norm == 0:
            self._record_normal(state)
            return control

        # Skip calibration during min_step period
        if state.global_step < self.min_step:
            self._record_normal(state)
            return control

        # Calibration phase: collect norms, don't skip yet
        if (state.global_step < self.min_step + self.calibration_steps
                and self._adaptive_threshold is None):
            self._record_normal(state)
            return control

        # First calibration: fires on the first step at or after the
        # calibration window. Uses >= (not ==) to handle cases where
        # gradient accumulation causes step counts to skip.
        # Will retry calibration up to _max_calibration_attempts times.
        if self._adaptive_threshold is None:
            # Check if we have enough samples and diversity
            is_valid, reason = self._validate_calibration_data(self._grad_norm_history)

            if not is_valid:
                self._calibration_attempts += 1
                if self._calibration_attempts <= self._max_calibration_attempts:
                    logger.warning(
                        f"  [grad_norm_skip] Calibration attempt {self._calibration_attempts} "
                        f"deferred: {reason}. Extending calibration window..."
                    )
                    self._record_normal(state)
                    return control
                else:
                    logger.error(
                        f"  [grad_norm_skip] Calibration failed after {self._calibration_attempts} "
                        f"attempts. Using fallback threshold."
                    )
                    # Fallback: use median as threshold (50th percentile)
                    if len(self._grad_norm_history) >= 10:
                        self._adaptive_threshold = float(np.median(self._grad_norm_history))
                        self._last_calibration_step = state.global_step
                    else:
                        self._record_normal(state)
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
                    f"\n  [grad_norm_skip] Norm range: [{min(self._grad_norm_history):.6f}, "
                    f"{max(self._grad_norm_history):.6f}]"
                    f"\n  [grad_norm_skip] Norm median: {np.median(self._grad_norm_history):.6f}"
                    f"\n  [grad_norm_skip] Calibration attempts: {self._calibration_attempts}"
                )
            else:
                print(f"  [grad_norm_skip] WARNING: Calibration failed at step {state.global_step}")
                self._record_normal(state)
                return control

        # Recalibrate periodically to handle non-stationarity
        # Uses rolling window for faster adaptation
        if (self.recalibrate_every > 0
                and state.global_step - self._last_calibration_step >= self.recalibrate_every):
            success = self._calibrate_threshold(use_rolling=True)
            if success:
                self._last_calibration_step = state.global_step
                logger.info(
                    f"  [grad_norm_skip] Recalibrated at step {state.global_step}, "
                    f"new threshold: {self._adaptive_threshold:.6f}"
                )

        # No threshold yet (shouldn't happen, but be safe)
        if self._adaptive_threshold is None:
            self._record_normal(state)
            return control

        if grad_norm < self._adaptive_threshold:
            self._record_skip(state)
            self.active_skipping = True

            # Momentum extrapolation (same as LERNA)
            optimizer = kwargs.get("optimizer", self._optimizer)
            if optimizer is not None:
                lr = optimizer.param_groups[0].get("lr", args.learning_rate)
                self._apply_momentum_extrapolation(optimizer, lr)
        else:
            self._record_normal(state)
            self.active_skipping = False

        self._log_periodic(state, {
            f"baseline/{self.baseline_name}/grad_norm": grad_norm,
            f"baseline/{self.baseline_name}/threshold": self._adaptive_threshold,
        })
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        stats = self._on_train_end_stats(args, state)
        # Add calibration info to stats
        if self._adaptive_threshold is not None:
            print(f"  Final adaptive threshold: {self._adaptive_threshold:.6f}")
            print(f"  Grad norm range: [{min(self._grad_norm_history):.6f}, "
                  f"{max(self._grad_norm_history):.6f}]")
        return control


# ===================================================================
# Baseline 2: Random Step Skipping
# ===================================================================

class RandomStepSkippingCallback(TrainerCallback, _BaselineStatsMixin):
    """Skip backward pass randomly at a target skip rate.

    Tests whether the *selection* of which steps to skip matters,
    or whether just reducing compute by any fraction is sufficient.
    If random skipping matches LERNA, then LER adds no value.
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
        self._model = None  # Cached from on_train_begin

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        if "model" in kwargs:
            self._model = kwargs["model"]
        return control

    def on_step_end(self, args, state, control, **kwargs):
        # Random skipping doesn't need model, but we need optimizer for momentum
        if state.global_step < self.min_step:
            self._record_normal(state)
            return control

        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]

        should_skip = self._rng.random() < self.target_skip_rate

        if should_skip:
            self._record_skip(state)
            self.active_skipping = True

            optimizer = kwargs.get("optimizer", self._optimizer)
            if optimizer is not None:
                lr = optimizer.param_groups[0].get("lr", args.learning_rate)
                self._apply_momentum_extrapolation(optimizer, lr)
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
# Baseline 3: Weight Freezing During Skip Phases
# ===================================================================

class WeightFreezingCallback(TrainerCallback, _BaselineStatsMixin):
    """Freeze weights entirely during LER-detected plateaus.

    When LER detects a plateau, instead of momentum extrapolation,
    simply freeze all weights (no update at all).

    Tests whether momentum extrapolation actually helps vs. doing nothing.
    Uses the same LER tracker as LERNA for plateau detection.
    """

    def __init__(
        self,
        ler_tracker,
        threshold: float = 1e-5,
        min_step: int = 100,
        wandb_enabled: bool = True,
        energy_per_skip: float = None,
    ):
        super().__init__()
        self._init_stats("weight_freeze", wandb_enabled, energy_per_skip)
        self.ler_tracker = ler_tracker
        self.threshold = threshold
        self.min_step = min_step

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.min_step:
            return control

        diag = self.ler_tracker.get_diagnostics()
        current_ler = diag.get("ler")
        if current_ler is None:
            return control

        if current_ler < self.threshold:
            # Signal to skip backward pass
            # Unlike LERNA, we do NOT apply momentum extrapolation
            self._record_skip(state)
            self.active_skipping = True
        else:
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
        self._on_train_end_stats(args, state)
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

            # Count remaining steps as "skipped" for fair energy comparison
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

    def _compute_lr(self, step: int) -> float:
        """Compute LR using SGDR (cosine annealing with warm restarts)."""
        T_cur = step
        T_i = self.T_0

        # Find which restart cycle we're in
        while T_cur >= T_i:
            T_cur -= T_i
            T_i = int(T_i * self.T_mult)

        # Cosine annealing within current cycle
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (
            1 + math.cos(math.pi * T_cur / T_i)
        )
        return lr

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        self.base_lr = args.learning_rate
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]

        if self._optimizer is None:
            return control

        new_lr = self._compute_lr(state.global_step)
        self._lr_history.append(new_lr)

        # Override optimizer LR
        for group in self._optimizer.param_groups:
            group["lr"] = new_lr

        if self.wandb_enabled and _wandb_active() and state.global_step % 10 == 0:
            wandb.log({
                f"baseline/{self.baseline_name}/lr": new_lr,
                "step": state.global_step,
            })

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        stats = self._on_train_end_stats(args, state)

        # Also save LR schedule
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

    Args:
        ler_tracker: LERTracker instance (needed for weight freezing baseline).
        target_skip_rate: Target skip rate to match LERNA's observed rate.
        total_steps: Total training steps (for reduced steps baseline).
        base_lr: Base learning rate (for cosine annealing baseline).
        seed: Random seed for reproducibility.
        wandb_enabled: Whether to log to W&B.
        energy_per_skip: Energy per skipped step in kWh (default: 0.0006 for RTX 5090).
    """
    baselines = {}

    # Baseline 1: Gradient Norm Thresholding (adaptive)
    baselines["grad_norm_skip"] = GradientNormSkippingCallback(
        target_skip_rate=target_skip_rate,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    # Baseline 2: Random Step Skipping
    baselines["random_skip"] = RandomStepSkippingCallback(
        target_skip_rate=target_skip_rate,
        seed=seed,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    # Baseline 3: Weight Freezing (requires LER tracker)
    if ler_tracker is not None:
        baselines["weight_freeze"] = WeightFreezingCallback(
            ler_tracker=ler_tracker,
            wandb_enabled=wandb_enabled,
            energy_per_skip=energy_per_skip,
        )

    # Baseline 4: Reduced Total Steps
    baselines["reduced_steps"] = ReducedTotalStepsCallback(
        reduction_fraction=target_skip_rate,
        total_steps=total_steps,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    # Baseline 5: Cosine Annealing with Warm Restarts
    baselines["cosine_warm_restarts"] = CosineAnnealingWarmRestartsCallback(
        T_0=max(total_steps // 10, 50),
        base_lr=base_lr,
        wandb_enabled=wandb_enabled,
        energy_per_skip=energy_per_skip,
    )

    return baselines
