"""REVISED — lerna/trainers/policies.py

Changes from original v1:

    [IMP-3] GradNormSkipPolicy now has max_consecutive_skips (default=1) — forces a real backward step after each skip to refresh the gradient norm, preventing infinite skip loops
    [FIX #7] Single-source grad-norm recording (unchanged from v1)
    [FIX #6] LERNAPolicy ports real decision logic (unchanged from v1)

"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch

# Import the actual SafetyHorizon used by LERNA today (Verdict Issue #6)
from lerna.callbacks.lerna_switching import SafetyHorizon


# ---------------------------------------------------------------------------
# 1) Always false  — uniform instrumentation for non-skipping baselines
# ---------------------------------------------------------------------------

class AlwaysFalsePolicy:
    name = "always_false"

    def should_skip(self, trainer, model, inputs) -> bool:
        return False


# ---------------------------------------------------------------------------
# 2) Gradient norm thresholding (Phase 1.2 grad_norm)
#
# [FIX #7] SINGLE-SOURCE grad-norm recording. The policy NEVER reads
#          trainer._pre_clip_grad_norm itself. Only the external
#          GradientNormCaptureCallback pushes values in via record_grad_norm().
#
# [IMP-3] PROBE RULE: After `max_consecutive_skips` consecutive skips, the
#         policy forces a real backward step to refresh the gradient norm.
#         Without this, once the last recorded grad norm is below threshold,
#         the policy would skip forever because no new norms are recorded
#         on skipped steps (no backward => no gradients => no new norm).
#
#         Default: max_consecutive_skips=1. This means: skip one step, then
#         force one real step to get a fresh gradient norm. This alternating
#         pattern is the most scientifically defensible for a true-backward-
#         skipping grad-norm baseline. The effective skip rate is bounded
#         to at most 50% (the actual rate depends on threshold calibration).
#
#         Higher values (e.g., 5) allow more consecutive skips but use
#         increasingly stale gradient information, which weakens the
#         scientific claim that "low grad norm => safe to skip".
# ---------------------------------------------------------------------------

class GradNormSkipPolicy:
    name = "grad_norm"

    def __init__(
        self,
        target_skip_rate: float = 0.33,
        calibration_steps: int = 200,
        recalibrate_every: int = 500,
        min_step: int = 0,
        min_calibration_samples: int = 50,
        rolling_window_size: int = 1000,
        max_consecutive_skips: int = 1,     # [IMP-3] probe rule
    ):
        self.target_skip_rate = target_skip_rate
        self.calibration_steps = calibration_steps
        self.recalibrate_every = recalibrate_every
        self.min_step = min_step
        self.min_calibration_samples = min_calibration_samples
        self.rolling_window_size = rolling_window_size
        self.max_consecutive_skips = max_consecutive_skips  # [IMP-3]

        self._grad_norms: List[float] = []
        self._rolling: List[float] = []
        self._threshold: Optional[float] = None
        self._last_calibration_step = 0
        self._consecutive_skips: int = 0   # [IMP-3] counter

    # Called by GradientNormCaptureCallback only.
    def record_grad_norm(self, value: float) -> None:
        v = float(value)
        if v <= 0:
            return
        self._grad_norms.append(v)
        self._rolling.append(v)
        if len(self._rolling) > self.rolling_window_size:
            self._rolling = self._rolling[-self.rolling_window_size:]

    def _calibrate(self) -> bool:
        src = self._rolling if len(self._rolling) >= self.min_calibration_samples \
              else self._grad_norms
        if len(src) < self.min_calibration_samples:
            return False
        self._threshold = float(np.percentile(src, self.target_skip_rate * 100))
        return True

    def should_skip(self, trainer, model, inputs) -> bool:
        step = trainer.state.global_step
        if step < self.min_step:
            return False
        if self._threshold is None:
            if len(self._grad_norms) >= max(self.calibration_steps,
                                            self.min_calibration_samples):
                self._calibrate()
                self._last_calibration_step = step
            return False
        if self.recalibrate_every > 0 and (step - self._last_calibration_step) >= self.recalibrate_every:
            self._calibrate()
            self._last_calibration_step = step
        if not self._grad_norms:
            return False

        # [IMP-3] Probe rule: if we've skipped max_consecutive_skips times
        # in a row, force a real backward step to refresh the gradient norm.
        if self._consecutive_skips >= self.max_consecutive_skips:
            self._consecutive_skips = 0  # reset; this step will be real
            return False

        want_skip = float(self._grad_norms[-1]) < float(self._threshold)

        if want_skip:
            self._consecutive_skips += 1
        else:
            self._consecutive_skips = 0

        return want_skip


# ---------------------------------------------------------------------------
# 3) Random skipping (Phase 1.2 random_skip)
# ---------------------------------------------------------------------------

class RandomSkipPolicy:
    name = "random_skip"

    def __init__(self, target_skip_rate: float = 0.22, min_step: int = 100, seed: int = 42):
        self.target_skip_rate = target_skip_rate
        self.min_step = min_step
        self._rng = random.Random(seed)

    def should_skip(self, trainer, model, inputs) -> bool:
        if trainer.state.global_step < self.min_step:
            return False
        return self._rng.random() < self.target_skip_rate


# ---------------------------------------------------------------------------
# 4) LER plateau (Phase 1.2 weight_freeze)
# ---------------------------------------------------------------------------

class LERPlateauPolicy:
    name = "ler_plateau"

    def __init__(self, ler_tracker, threshold: float = 1e-5, min_step: int = 100):
        self.ler_tracker = ler_tracker
        self.threshold = threshold
        self.min_step = min_step

    def should_skip(self, trainer, model, inputs) -> bool:
        if trainer.state.global_step < self.min_step:
            return False
        ler = self.ler_tracker.get_diagnostics().get("ler")
        return (ler is not None) and (ler < self.threshold)


# ---------------------------------------------------------------------------
# 5) LERNA full policy for Phase 1.3 — PORTS REAL DECISION LOGIC
#    (Verdict Issue #6: do not simplify LERNA's switching condition.)
#
# Mirrors LERNASwitchingCallback.on_pre_optimizer_step / SafetyHorizon
# behavior, including:
#   - LER + rho_VG plateau detection
#   - Safety Horizon (PL-condition-based bound) when enabled
#   - Hysteresis (delegated to LERTracker via use_hysteresis at construction)
#   - Loss-history-informed PL constant estimation
#
# [IMP-6] Confirmed: LERTracker(use_hysteresis=...) is supported.
#         See lerna/utils/metrics.py LERTracker.__init__ line ~351.
#         The toggle lives in LERTracker.get_efficiency_phase() which
#         checks self.use_hysteresis. No code changes needed for this.
#
# The trainer is responsible only for executing the skip safely.
# ---------------------------------------------------------------------------

class LERNAPolicy:
    name = "lerna"

    def __init__(
        self,
        ler_tracker,
        threshold: float = 1e-5,
        min_step: int = 100,
        use_ler: bool = True,
        use_rho_vg: bool = True,
        use_safety_horizon: bool = True,
        # use_hysteresis is configured at LERTracker construction time.
        # [IMP-6] Confirmed: LERTracker.__init__ accepts use_hysteresis.
        # We accept and store it here for transparency in instrumentation/logs.
        use_hysteresis: bool = True,
        rho_vg_threshold: float = 0.1,
    ):
        self.ler_tracker = ler_tracker
        self.threshold = threshold
        self.min_step = min_step
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_safety_horizon = use_safety_horizon
        self.use_hysteresis = use_hysteresis
        self.rho_vg_threshold = rho_vg_threshold

        # Mirrors LERNASwitchingCallback internals (ported, not reimplemented)
        self.safety_horizon = SafetyHorizon() if use_safety_horizon else None
        self._consecutive_skips = 0
        self._loss_history: List[float] = []

    # External hook so a trainer/callback can feed losses for PL estimation.
    # Kept symmetric with LERNASwitchingCallback.on_log.
    def record_loss(self, value: float) -> None:
        try:
            self._loss_history.append(float(value))
        except (TypeError, ValueError):
            return

    def should_skip(self, trainer, model, inputs) -> bool:
        if trainer.state.global_step < self.min_step:
            self._consecutive_skips = 0
            return False

        diag = self.ler_tracker.get_diagnostics()
        ler = diag.get("ler")
        rho_vg = diag.get("rho_vg", 0.0) or 0.0

        # Ported from LERNASwitchingCallback.on_pre_optimizer_step:
        # Plateau detection rule.
        if self.use_ler:
            if ler is None:
                base = False
            else:
                base = ler < self.threshold
        else:
            base = rho_vg < self.rho_vg_threshold  # mirrors existing fallback

        # rho_VG gating in addition to LER (ported semantics).
        # When use_rho_vg=False we drop this check, matching the no_rho_vg ablation.
        if base and self.use_rho_vg and self.use_ler:
            # In the original code, rho_vg is recorded; thrashing rho<0 means
            # SafetyHorizon returns 0 (handled below). We don't add an extra
            # rho_vg threshold here because the existing callback doesn't either.
            pass

        # Safety horizon check (ported from LERNASwitchingCallback).
        if base and self.use_safety_horizon and self.safety_horizon is not None:
            loss_improvement = None
            if len(self._loss_history) >= 2:
                loss_improvement = abs(self._loss_history[-2] - self._loss_history[-1])
            # grad_norm not strictly needed for the PL estimate when
            # loss_improvement is the dominant signal; passing 0.0 mirrors the
            # callback's fallback path.
            grad_norm = float(trainer._pre_clip_grad_norm or 0.0)
            max_safe_skips = self.safety_horizon.compute_horizon(
                rho_vg=rho_vg if self.use_rho_vg else 0.0,
                ler=ler if ler is not None else 0.0,
                grad_norm=grad_norm,
                loss_improvement=loss_improvement,
            )
            if self._consecutive_skips >= max_safe_skips:
                base = False

        if base:
            self._consecutive_skips += 1
        else:
            self._consecutive_skips = 0
        return base
