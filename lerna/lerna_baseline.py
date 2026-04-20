"""LERNA baseline for Phase 1.2 comparison.

Uses the real LER signal to decide which steps to skip, running inside the
same Phase12Trainer snapshot-rollback harness as grad_norm / random_skip /
weight_freeze so the accuracy comparison is apples-to-apples.

True backward-pass elimination (energy savings) is Phase 2+ (LERNATrainer).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from transformers import TrainerCallback

from lerna.utils.metrics import LERTracker  # adjust import to your module


class LERNABaselineCallback(TrainerCallback):
    """Skip-selection via Learning Efficiency Ratio.

    Skips a step when:
        LER < ler_threshold  AND  |rho_VG| < rho_vg_threshold

    The rho_VG guard prevents skipping when velocity-gradient correlation is
    strong (model is still learning in a directed way).
    """

    def __init__(
        self,
        ler_threshold: float = 0.3,
        rho_vg_threshold: float = 0.5,
        warmup_steps: int = 50,
        max_skip_fraction: float = 0.6,
        name: str = "lerna",
    ):
        self.ler_threshold = ler_threshold
        self.rho_vg_threshold = rho_vg_threshold
        self.warmup_steps = warmup_steps
        self.max_skip_fraction = max_skip_fraction
        self.name = name

        self.tracker: Optional[LERTracker] = None
        self._trainer = None
        self._skip_count = 0
        self._total_decisions = 0
        self._ler_history: list[float] = []
        self._rho_vg_history: list[float] = []

    # ------------------------------------------------------------------ #
    # Hooks required by Phase12Trainer contract
    # ------------------------------------------------------------------ #
    def bind_trainer(self, trainer):
        self._trainer = trainer
        self.tracker = LERTracker()  # configure per your LERTracker signature

    def on_train_begin(self, args, state, control, **kwargs):
        if self.tracker is None:
            self.tracker = LERTracker()

    def on_step_begin(self, args, state, control, **kwargs):
        """Decide whether to skip THIS step's backward pass."""
        if self._trainer is None:
            return

        # Warmup — never skip in first N steps (LER not yet stable)
        if state.global_step < self.warmup_steps:
            self._trainer.should_skip_backward = False
            return

        # Respect max skip budget
        if self._total_decisions > 0:
            current_skip_frac = self._skip_count / self._total_decisions
            if current_skip_frac >= self.max_skip_fraction:
                self._trainer.should_skip_backward = False
                self._total_decisions += 1
                return

        diag = self.tracker.get_diagnostics() if self.tracker else {}
        ler = diag.get("ler")
        rho_vg = diag.get("rho_vg")

        skip = False
        if ler is not None and ler < self.ler_threshold:
            if rho_vg is None or abs(rho_vg) < self.rho_vg_threshold:
                skip = True

        self._trainer.should_skip_backward = skip
        self._total_decisions += 1
        if skip:
            self._skip_count += 1

        if ler is not None:
            self._ler_history.append(float(ler))
        if rho_vg is not None:
            self._rho_vg_history.append(float(rho_vg))

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        """Feed gradients into LERTracker for rho_VG computation."""
        if self.tracker is None:
            return
        model = kwargs.get("model")
        if model is None:
            return

        total_norm_sq = 0.0
        grad_snapshot = []
        for p in model.parameters():
            if p.grad is not None:
                g = p.grad.data
                total_norm_sq += g.norm(2).item() ** 2
                grad_snapshot.append(g.detach().flatten())
        grad_norm = total_norm_sq ** 0.5
        flat_grad = (
            torch.cat(grad_snapshot) if grad_snapshot else torch.zeros(1)
        )
        self.tracker.update_gradient(grad_norm=grad_norm, flat_grad=flat_grad)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Feed eval metrics into LERTracker for LER update."""
        if self.tracker is None or metrics is None:
            return
        eval_loss = metrics.get("eval_loss")
        eval_acc = metrics.get("eval_accuracy") or metrics.get("eval_matthews_correlation")
        self.tracker.update_evaluation(eval_loss=eval_loss, eval_metric=eval_acc)

    # ------------------------------------------------------------------ #
    # Activation summary (matches other baselines' contract)
    # ------------------------------------------------------------------ #
    def get_activation_summary(self) -> dict:
        skip_frac = (
            self._skip_count / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        return {
            "baseline": self.name,
            "activated": self._skip_count > 0,
            "skip_count": self._skip_count,
            "total_decisions": self._total_decisions,
            "skip_fraction": skip_frac,
            "mean_ler": float(np.mean(self._ler_history)) if self._ler_history else None,
            "mean_rho_vg": float(np.mean(self._rho_vg_history)) if self._rho_vg_history else None,
        }
