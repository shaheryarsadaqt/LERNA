"""LERNA baseline for Phase 1.2 comparison.

Reads LER signal from a shared LERTracker (fed by LERFeedCallback) to decide
which steps to skip. Runs inside the same Phase12Trainer snapshot-rollback
harness as grad_norm / random_skip / weight_freeze so accuracy is directly
comparable.

True backward-pass elimination (energy savings) is Phase 2+ (LERNATrainer).
"""
from __future__ import annotations

from typing import Optional
import numpy as np
from transformers import TrainerCallback


class LERNABaselineCallback(TrainerCallback):
    """Skip-selection via Learning Efficiency Ratio.

    Reads a shared LERTracker (populated by LERFeedCallback elsewhere in the
    runner). Skips a step when:
        LER < ler_threshold  AND  |rho_VG| < rho_vg_threshold
    """

    def __init__(
        self,
        ler_tracker,                          # REQUIRED — shared LERTracker instance
        ler_threshold: float = 0.3,
        rho_vg_threshold: float = 0.5,
        warmup_steps: int = 50,
        max_skip_fraction: float = 0.6,
        baseline_name: str = "lerna",
    ):
        self.ler_tracker = ler_tracker
        self.ler_threshold = ler_threshold
        self.rho_vg_threshold = rho_vg_threshold
        self.warmup_steps = warmup_steps
        self.max_skip_fraction = max_skip_fraction
        self.baseline_name = name

        self._trainer = None
        self._skip_count = 0
        self._total_decisions = 0
        self._ler_history: list[float] = []
        self._rho_vg_history: list[float] = []

    # ------------------------------------------------------------------ #
    def bind_trainer(self, trainer):
        self._trainer = trainer

    # ------------------------------------------------------------------ #
    def on_step_begin(self, args, state, control, **kwargs):
        if self._trainer is None:
            return

        # Warmup: never skip while LER is still noisy
        if state.global_step < self.warmup_steps:
            self._trainer.should_skip_backward = False
            return

        # Respect max skip budget
        if self._total_decisions > 0:
            if self._skip_count / self._total_decisions >= self.max_skip_fraction:
                self._trainer.should_skip_backward = False
                self._total_decisions += 1
                return

        diag = self.ler_tracker.get_diagnostics() if self.ler_tracker else {}
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

    # ------------------------------------------------------------------ #
    def get_activation_summary(self) -> dict:
        skip_frac = (
            self._skip_count / self._total_decisions
            if self._total_decisions > 0 else 0.0
        )
        return {
            "baseline": self.baseline_name,
            "activated": self._skip_count > 0,
            "skip_count": self._skip_count,
            "total_decisions": self._total_decisions,
            "skip_fraction": skip_frac,
            "mean_ler": float(np.mean(self._ler_history)) if self._ler_history else None,
            "mean_rho_vg": float(np.mean(self._rho_vg_history)) if self._rho_vg_history else None,
        }
