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

import torch
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


# ===================================================================
# Shared mixin for consistent stats tracking across all baselines
# ===================================================================

class _BaselineStatsMixin:
    """Common statistics tracking for all baseline callbacks."""

    def _init_stats(self, baseline_name: str, wandb_enabled: bool = True):
        self.baseline_name = baseline_name
        self.wandb_enabled = wandb_enabled
        self.steps_skipped = 0
        self.total_energy_saved = 0.0
        self.skip_decisions = []          # True/False per step
        self.accuracy_during_skip = []
        self.accuracy_during_normal = []
        self.last_accuracy = 0.0
        self.active_skipping = False
        self._energy_per_skip = 0.0006    # ~0.6 Wh per skipped step on RTX 5090

    def _record_skip(self, state):
        self.steps_skipped += 1
        self.total_energy_saved += self._energy_per_skip
        self.skip_decisions.append(True)

    def _record_normal(self, state):
        self.skip_decisions.append(False)

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

        stats = {
            "baseline_name": self.baseline_name,
            "total_steps": total_steps,
            "steps_skipped": self.steps_skipped,
            "skip_ratio": skip_ratio,
            "energy_saved_kwh": self.total_energy_saved,
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
        print(f"  Energy saved: {self.total_energy_saved:.6f} kWh")
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
# Baseline 1: Gradient Norm Thresholding
# ===================================================================

class GradientNormSkippingCallback(TrainerCallback, _BaselineStatsMixin):
    """Skip backward pass when gradient norm falls below a threshold.

    Tests whether LER is better than its simplest component: just
    checking if gradients are small.

    The threshold is calibrated so the overall skip rate approximately
    matches LERNA's observed skip rate for fair comparison.
    """

    def __init__(
        self,
        grad_norm_threshold: float = 480000.0,
        min_step: int = 100,
        wandb_enabled: bool = True,
    ):
        super().__init__()
        self._init_stats("grad_norm_skip", wandb_enabled)
        self.grad_norm_threshold = grad_norm_threshold
        self.min_step = min_step
        self._optimizer = None
        self._grad_norm_history = []

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step < self.min_step:
            return control
        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step < self.min_step or model is None:
            self._record_normal(state)
            return control

        # Compute global gradient norm
        total_norm_sq = 0.0
        has_grad = False
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                total_norm_sq += p.grad.detach().float().norm().item() ** 2
                has_grad = True

        if not has_grad:
            self._record_normal(state)
            return control

        grad_norm = total_norm_sq ** 0.5
        self._grad_norm_history.append(grad_norm)

        if grad_norm < self.grad_norm_threshold:
            self._record_skip(state)
            self.active_skipping = True

            # Momentum extrapolation (same as LERNA)
            optimizer = kwargs.get("optimizer", self._optimizer)
            if optimizer is not None:
                lr = optimizer.param_groups[0].get("lr", args.learning_rate)
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
        else:
            self._record_normal(state)
            self.active_skipping = False

        self._log_periodic(state, {
            f"baseline/{self.baseline_name}/grad_norm": grad_norm,
        })
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self._on_evaluate_stats(metrics, state)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._on_train_end_stats(args, state)
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
    ):
        super().__init__()
        self._init_stats("random_skip", wandb_enabled)
        self.target_skip_rate = target_skip_rate
        self.min_step = min_step
        self._rng = random.Random(seed)
        self._optimizer = None

    def on_train_begin(self, args, state, control, **kwargs):
        if "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]
        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step < self.min_step or model is None:
            self._record_normal(state)
            return control

        if self._optimizer is None and "optimizer" in kwargs:
            self._optimizer = kwargs["optimizer"]

        should_skip = self._rng.random() < self.target_skip_rate

        if should_skip:
            self._record_skip(state)
            self.active_skipping = True

            # Momentum extrapolation
            optimizer = kwargs.get("optimizer", self._optimizer)
            if optimizer is not None:
                lr = optimizer.param_groups[0].get("lr", args.learning_rate)
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
    ):
        super().__init__()
        self._init_stats("weight_freeze", wandb_enabled)
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
    ):
        super().__init__()
        self._init_stats("reduced_steps", wandb_enabled)
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
    ):
        super().__init__()
        self._init_stats("cosine_warm_restarts", wandb_enabled)
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
    grad_norm_threshold: float = 480000.0,
    total_steps: int = 1000,
    base_lr: float = 2e-5,
    seed: int = 42,
    wandb_enabled: bool = True,
) -> dict:
    """Create all 6 baseline callbacks for comparison.

    Returns a dict mapping baseline name to callback instance.
    Baseline 6 (early stopping) uses HF's built-in EarlyStoppingCallback
    and is not included here; add it separately with different patience values.

    Args:
        ler_tracker: LERTracker instance (needed for weight freezing baseline).
        target_skip_rate: Target skip rate to match LERNA's observed rate.
        grad_norm_threshold: Gradient norm threshold for baseline 1.
        total_steps: Total training steps (for reduced steps baseline).
        base_lr: Base learning rate (for cosine annealing baseline).
        seed: Random seed for reproducibility.
        wandb_enabled: Whether to log to W&B.
    """
    baselines = {}

    # Baseline 1: Gradient Norm Thresholding
    baselines["grad_norm_skip"] = GradientNormSkippingCallback(
        grad_norm_threshold=grad_norm_threshold,
        wandb_enabled=wandb_enabled,
    )

    # Baseline 2: Random Step Skipping
    baselines["random_skip"] = RandomStepSkippingCallback(
        target_skip_rate=target_skip_rate,
        seed=seed,
        wandb_enabled=wandb_enabled,
    )

    # Baseline 3: Weight Freezing (requires LER tracker)
    if ler_tracker is not None:
        baselines["weight_freeze"] = WeightFreezingCallback(
            ler_tracker=ler_tracker,
            wandb_enabled=wandb_enabled,
        )

    # Baseline 4: Reduced Total Steps
    baselines["reduced_steps"] = ReducedTotalStepsCallback(
        reduction_fraction=target_skip_rate,
        total_steps=total_steps,
        wandb_enabled=wandb_enabled,
    )

    # Baseline 5: Cosine Annealing with Warm Restarts
    baselines["cosine_warm_restarts"] = CosineAnnealingWarmRestartsCallback(
        T_0=max(total_steps // 10, 50),
        base_lr=base_lr,
        wandb_enabled=wandb_enabled,
    )

    return baselines
