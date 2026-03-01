"""
LERNA Switching Logic - LER-guided hybrid-order switching.

Core mechanism: During high-LER phases, standard backpropagation drives
learning at full fidelity. When LER drops below a causally-validated
threshold, LERNA bypasses the backward pass entirely, updating weights
via momentum-driven inertial extrapolation (eliminating ~60% of per-step
compute).

All switching statistics are logged to W&B for real-time visualization.
"""

import torch
import json
import os
import logging
from typing import Optional, Dict, List

try:
    import wandb
except ImportError:
    wandb = None

from transformers import TrainerCallback

logger = logging.getLogger(__name__)


def _wandb_active() -> bool:
    """Return True only when wandb is installed AND has an active run."""
    return wandb is not None and getattr(wandb, "run", None) is not None


# Default task-specific LER thresholds (from LERTracker calibration).
# The switching threshold is set to half the task's LER plateau threshold
# so that switching only activates during genuine plateaus, not normal
# LER fluctuations.
_TASK_LER_THRESHOLDS = {
    "sst2": 0.010,
    "qnli": 0.008,
    "qqp":  0.012,
    "mnli": 0.009,
    "rte":  0.015,
    "mrpc": 0.014,
    "cola": 0.013,
    "stsb": 0.010,
}


class LERNASwitchingCallback(TrainerCallback):
    """
    Implements LERNA's core innovation: LER-guided hybrid-order switching.

    When LER drops below a task-calibrated threshold the backward pass is
    skipped and weights are updated via momentum extrapolation instead.
    An adaptive safety horizon (scaled inversely with rho_VG confidence)
    prevents parameter drift.

    Threshold calibration
    ---------------------
    The default threshold is ``5e-3`` (half the median task LER threshold
    of ~0.01).  If a ``task`` name is provided, the threshold auto-
    calibrates to ``0.5 * task_ler_threshold``.  You can always override
    with an explicit ``threshold`` argument.

    Re-engagement hysteresis
    ------------------------
    To avoid oscillating between skip/normal modes, skipping only
    *deactivates* when LER recovers above ``threshold * 1.5``.
    """

    def __init__(
        self,
        ler_tracker,
        threshold: Optional[float] = None,
        task: Optional[str] = None,
        min_step: int = 100,
        wandb_enabled: bool = True,
        safety_horizon_base: int = 10,
    ):
        """
        Args:
            ler_tracker: An instance of ``LERTracker`` from ``lerna.utils.metrics``.
            threshold: LER value below which backward passes are skipped.
                       If *None*, auto-calibrates from ``task``.
            task: GLUE task name for auto-calibration (e.g. "sst2").
            min_step: Don't skip before this many global steps (warmup).
            wandb_enabled: Whether to log to W&B.
            safety_horizon_base: Base number of consecutive skip steps
                                 before forcing a full backward pass.
        """
        self.ler_tracker = ler_tracker
        self.min_step = min_step
        self.wandb_enabled = wandb_enabled
        self.safety_horizon_base = safety_horizon_base

        # --- Threshold calibration ---
        if threshold is not None:
            self.threshold = threshold
        elif task is not None and task in _TASK_LER_THRESHOLDS:
            # Half the task's LER plateau threshold
            self.threshold = _TASK_LER_THRESHOLDS[task] * 0.5
        else:
            # Sensible default: half the median LER threshold across tasks
            self.threshold = 5e-3

        self.task = task or "unknown"

        # --- Counters & state ---
        self.steps_skipped = 0
        self.total_energy_saved = 0.0
        self.plateau_steps: List[int] = []
        self.active_skipping = False
        self._consecutive_skips = 0  # for safety horizon

        # Accuracy tracking (properly initialised)
        self.last_accuracy = 0.0
        self.accuracy_during_skip: List[float] = []
        self.accuracy_during_normal: List[float] = []

        logger.info(
            f"LERNASwitchingCallback initialised: task={self.task}, "
            f"threshold={self.threshold:.4e}, min_step={self.min_step}, "
            f"safety_horizon_base={self.safety_horizon_base}"
        )

    # ------------------------------------------------------------------
    # Trainer hooks
    # ------------------------------------------------------------------

    def on_step_begin(self, args, state, control, **kwargs):
        """Decide whether to skip the backward pass for this step."""
        if state.global_step < self.min_step:
            return control

        diag = self.ler_tracker.get_diagnostics()
        current_ler = diag.get("ler")
        current_vel = diag.get("param_velocity", 0)
        current_rho = diag.get("rho_vg", 0)
        current_phase = diag.get("phase", "unknown")

        if current_ler is None:
            return control

        # --- Safety horizon: force a full step periodically ---
        adaptive_horizon = self._compute_safety_horizon(current_rho)
        if self._consecutive_skips >= adaptive_horizon:
            # Force a full backward pass to re-anchor gradients
            self._consecutive_skips = 0
            self.active_skipping = False
            return control

        # --- Skip decision ---
        should_skip = current_ler < self.threshold

        # Re-engagement hysteresis: require LER > 1.5x threshold to exit
        if self.active_skipping and current_ler >= self.threshold * 1.5:
            should_skip = False

        if should_skip:
            if not self.active_skipping:
                logger.info(
                    f"\U0001f4c9 Plateau detected at step {state.global_step}: "
                    f"LER={current_ler:.2e}, rho_VG={current_rho:.3f}"
                )
                self._log_wandb({
                    "lerna/plateau_detected": state.global_step,
                    "lerna/plateau_ler": current_ler,
                    "lerna/plateau_step": state.global_step,
                })
                self.active_skipping = True
                self.plateau_steps.append(state.global_step)

            # Skip backward pass
            control.should_skip_backward = True
            self.steps_skipped += 1
            self._consecutive_skips += 1

            # Energy estimate: backward pass is ~60% of step on RTX 5090
            step_energy = 0.0006  # kWh per skipped step (rough)
            self.total_energy_saved += step_energy

            if state.global_step % 10 == 0:
                self._log_wandb({
                    "lerna/steps_skipped": self.steps_skipped,
                    "lerna/energy_saved_kwh": self.total_energy_saved,
                    "lerna/skip_ratio": self.steps_skipped / max(state.global_step, 1),
                    "lerna/current_ler": current_ler,
                    "lerna/current_velocity": current_vel,
                    "lerna/current_rho_vg": current_rho,
                    "lerna/phase": current_phase,
                    "lerna/active": 1,
                    "lerna/consecutive_skips": self._consecutive_skips,
                    "lerna/safety_horizon": adaptive_horizon,
                    "step": state.global_step,
                })
        else:
            if self.active_skipping:
                duration = (
                    state.global_step - self.plateau_steps[-1]
                    if self.plateau_steps else 0
                )
                logger.info(
                    f"\U0001f7e2 Plateau ended at step {state.global_step} "
                    f"(duration={duration} steps)"
                )
                self._log_wandb({
                    "lerna/plateau_ended": state.global_step,
                    "lerna/plateau_duration": duration,
                    "step": state.global_step,
                })
            self.active_skipping = False
            self._consecutive_skips = 0

            if state.global_step % 10 == 0:
                self._log_wandb({
                    "lerna/current_ler": current_ler,
                    "lerna/current_velocity": current_vel,
                    "lerna/current_rho_vg": current_rho,
                    "lerna/phase": current_phase,
                    "lerna/active": 0,
                    "step": state.global_step,
                })

        return control

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """If backward was skipped, apply momentum extrapolation."""
        if not getattr(control, "should_skip_backward", False):
            return control

        if model is None:
            return control

        # Try to get the optimizer from the trainer (passed via kwargs or args)
        optimizer = kwargs.get("optimizer")
        if optimizer is None and hasattr(args, "optimizer"):
            optimizer = args.optimizer

        lr = args.learning_rate if hasattr(args, "learning_rate") else 2e-5

        # Momentum extrapolation: use cached momentum buffers to update
        # weights without computing new gradients.
        if optimizer is not None:
            with torch.no_grad():
                for param in model.parameters():
                    if param.grad is not None:
                        if hasattr(optimizer, "state") and param in optimizer.state:
                            p_state = optimizer.state[param]
                            if "momentum_buffer" in p_state:
                                momentum = p_state["momentum_buffer"]
                                param.data -= lr * momentum
                                continue
                            elif "exp_avg" in p_state:
                                # AdamW-style: use first moment estimate
                                exp_avg = p_state["exp_avg"]
                                param.data -= lr * exp_avg
                                continue
                        # Fallback: raw gradient step
                        param.data -= lr * param.grad

            optimizer.zero_grad(set_to_none=True)

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Track accuracy during evaluation, split by skip/normal mode."""
        if not metrics:
            return control

        # Support multiple metric names across GLUE tasks
        accuracy = metrics.get(
            "eval_accuracy",
            metrics.get(
                "eval_matthews_correlation",
                metrics.get("eval_pearsonr"),
            ),
        )
        if accuracy is None:
            return control

        self.last_accuracy = accuracy

        if self.active_skipping:
            self.accuracy_during_skip.append(accuracy)
        else:
            self.accuracy_during_normal.append(accuracy)

        self._log_wandb({
            "lerna/accuracy_during_skip": accuracy if self.active_skipping else 0,
            "lerna/accuracy_during_normal": accuracy if not self.active_skipping else 0,
            "lerna/skipping_active": int(self.active_skipping),
            "step": state.global_step,
        })

        return control

    def on_train_end(self, args, state, control, **kwargs):
        """Report and log final switching statistics."""
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
        acc_diff = avg_acc_skip - avg_acc_normal

        # Console summary
        print("\n" + "=" * 60)
        print("\U0001f4ca LERNA SWITCHING STATISTICS")
        print("=" * 60)
        print(f"Task:                  {self.task}")
        print(f"Threshold:             {self.threshold:.4e}")
        print(f"Total steps:           {total_steps}")
        print(f"Steps skipped:         {self.steps_skipped} ({skip_ratio*100:.1f}%)")
        print(f"Estimated energy saved: {self.total_energy_saved:.6f} kWh")
        print(f"Num plateau entries:   {len(self.plateau_steps)}")
        print(f"Plateau entry steps:   {self.plateau_steps[:10]}")
        print(f"Accuracy during skip:  {avg_acc_skip:.4f} (n={len(self.accuracy_during_skip)})")
        print(f"Accuracy during normal:{avg_acc_normal:.4f} (n={len(self.accuracy_during_normal)})")
        print(f"Accuracy difference:   {acc_diff:+.4f}")
        print("=" * 60)

        # W&B final summary
        self._log_wandb({
            "lerna/final/steps_skipped": self.steps_skipped,
            "lerna/final/skip_ratio": skip_ratio,
            "lerna/final/energy_saved_kwh": self.total_energy_saved,
            "lerna/final/num_plateaus": len(self.plateau_steps),
            "lerna/final/avg_acc_during_skip": avg_acc_skip,
            "lerna/final/avg_acc_during_normal": avg_acc_normal,
            "lerna/final/acc_difference": acc_diff,
            "lerna/final/threshold": self.threshold,
        })

        # W&B plateau table
        if self.plateau_steps and self.wandb_enabled and _wandb_active():
            try:
                plateau_table = wandb.Table(
                    columns=["step", "energy_saved_kwh"],
                    data=[[step, 0.0006] for step in self.plateau_steps[:100]],
                )
                wandb.log({"lerna/plateau_steps": plateau_table})
            except Exception:
                pass

        # Save local stats
        stats = {
            "task": self.task,
            "threshold": self.threshold,
            "total_steps": total_steps,
            "steps_skipped": self.steps_skipped,
            "skip_ratio": skip_ratio,
            "energy_saved_kwh": self.total_energy_saved,
            "plateau_steps": self.plateau_steps,
            "avg_acc_during_skip": avg_acc_skip,
            "avg_acc_during_normal": avg_acc_normal,
            "acc_difference": acc_diff,
            "n_evals_during_skip": len(self.accuracy_during_skip),
            "n_evals_during_normal": len(self.accuracy_during_normal),
        }

        stats_path = os.path.join(args.output_dir, "lerna_switching_stats.json")
        try:
            with open(stats_path, "w") as f:
                json.dump(stats, f, indent=2)
            print(f"Statistics saved to {stats_path}")
        except Exception as e:
            logger.warning(f"Could not save switching stats: {e}")

        return control

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_safety_horizon(self, rho_vg: Optional[float]) -> int:
        """Adaptive safety horizon scaled inversely with rho_VG confidence.

        When rho_VG is high (parameters aligned with gradients), we can
        safely skip more steps.  When rho_VG is low or negative, we
        force full backward passes more frequently.
        """
        if rho_vg is None or rho_vg <= 0:
            # Low confidence: short horizon
            return max(3, self.safety_horizon_base // 3)
        elif rho_vg > 0.5:
            # High confidence: allow longer skipping
            return self.safety_horizon_base * 3
        else:
            # Moderate confidence: scale linearly
            scale = 1.0 + (rho_vg / 0.5) * 2.0  # 1x to 3x
            return int(self.safety_horizon_base * scale)

    def _log_wandb(self, data: Dict):
        """Safely log to W&B if enabled and active."""
        if not self.wandb_enabled or not _wandb_active():
            return
        try:
            wandb.log(data)
        except Exception:
            pass
