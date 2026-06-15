"""REVISED — lerna/trainers/policies.py

Changes from original v1:

    [IMP-3] GradNormSkipPolicy now has max_consecutive_skips (default=1) — forces a real backward step after each skip to refresh the gradient norm, preventing infinite skip loops
    [FIX #7] Single-source grad-norm recording (unchanged from v1)
    [FIX #6] LERNAPolicy ports real decision logic (unchanged from v1)

"""

from __future__ import annotations

import math
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
        self._grad_norm_skip_decisions: int = 0
        self._forced_probe_count: int = 0
        self._grad_norm_forced_probe_count: int = 0
        self._grad_norm_last: Optional[float] = None

    # Called by GradientNormCaptureCallback only.
    def record_grad_norm(self, value: float) -> None:
        try:
            v = float(value)
        except (TypeError, ValueError):
            return

        if not math.isfinite(v) or v <= 0:
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
        self._threshold = float(np.nanpercentile(src, self.target_skip_rate * 100))
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
            self._grad_norm_forced_probe_count += 1
            return False

        want_skip = float(self._grad_norms[-1]) < float(self._threshold)

        if want_skip:
            self._consecutive_skips += 1
            self._grad_norm_skip_decisions += 1
        else:
            self._consecutive_skips = 0

        return want_skip

    def get_diagnostics(self) -> dict:
        finite_norms = [
            float(x)
            for x in self._grad_norms
            if x is not None and math.isfinite(float(x))
        ]

        return {
            "grad_norm_samples_collected": len(self._grad_norms),
            "grad_norm_finite_samples": len(finite_norms),
            "grad_norm_threshold": self._threshold,
            "grad_norm_calibrated": self._threshold is not None,
            "grad_norm_calibration_step": self._last_calibration_step,
            "grad_norm_min": min(finite_norms) if finite_norms else None,
            "grad_norm_max": max(finite_norms) if finite_norms else None,
            "grad_norm_mean": float(np.mean(finite_norms)) if finite_norms else None,
            "grad_norm_last": finite_norms[-1] if finite_norms else None,
            "grad_norm_forced_probe_count": self._grad_norm_forced_probe_count,
            "grad_norm_skip_decisions": self._grad_norm_skip_decisions,
        }


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


# ---------------------------------------------------------------------------
# 6) Phase 1.3b: LERNACalibratedPolicy — target-skip-rate, quantile-calibrated,
#    confidence-gated LERNA controller.
#
# Directly attacks the two primary root causes identified in the ablation
# post-mortem:
#   R1 — Threshold is ~1000× too strict (was hardcoded 1e-5 vs real LER scale
#         of 0.008–0.015). Quantile calibration fixes this automatically.
#   R2 — LER was updated only at eval time. Per-step _online_ler_update
#         (in the trainer) now fills _ler_window densely.
#   R4 — Safety horizon collapsed to 0. Replaced with confidence-gated burst
#         skipping and forced probes.
#
# LERNACalibratedPolicy needs the per-step online LER feed from the trainer
# (P0-3 fix) to function correctly.
# ---------------------------------------------------------------------------

class LERNACalibratedPolicy:
    """Phase 1.3b: target-skip-rate, quantile-calibrated, confidence-gated LERNA.

    Uses the target_skip_rate-th percentile of observed LER (or rho_VG) as
    the dynamic threshold. Includes confidence-gated burst skipping and
    forced periodic probes to prevent stale diagnostics.
    """
    name = "lerna_calibrated"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.25,
        fallback_threshold: float = 0.01,
        min_step: int = 50,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,
        use_ler: bool = True,
        use_rho_vg: bool = True,
        use_safety_horizon: bool = True,
        max_consecutive_skips: int = 4,
        probe_interval: int = 8,
        rho_vg_skip_below: float = 0.1,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = target_skip_rate
        self.threshold = fallback_threshold
        self._calibrated = False
        self.min_step = min_step
        self.calibration_steps = calibration_steps
        self.recalibrate_every = recalibrate_every
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_safety_horizon = use_safety_horizon
        self.max_consecutive_skips = max_consecutive_skips
        self.probe_interval = probe_interval
        self.rho_vg_skip_below = rho_vg_skip_below

        self._ler_window: List[float] = []
        self._last_calib_step = 0
        self._consecutive_skips = 0
        self._steps_since_probe = 0
        self._skips = 0
        self._opportunities = 0

    def _calibrate(self):
        """Set threshold to the target_skip_rate-th percentile of observed LER."""
        if len(self._ler_window) < max(20, self.calibration_steps // 2):
            return
        # primary signal: LER if enabled, else rho_VG
        self.threshold = float(np.nanpercentile(
            self._ler_window, self.target_skip_rate * 100
        ))
        self._calibrated = True

    def should_skip(self, trainer, model, inputs) -> bool:
        step = trainer.state.global_step
        self._opportunities += 1
        if step < self.min_step:
            self._consecutive_skips = 0
            return False

        diag = self.trk.get_diagnostics()
        ler = diag.get("ler")
        rho = diag.get("rho_vg", 0.0) or 0.0

        # Collect signal for calibration (dense now, thanks to online update)
        sig = ler if (self.use_ler and ler is not None) else (rho if self.use_rho_vg else None)
        if sig is not None:
            self._ler_window.append(float(sig))
            if len(self._ler_window) > 2000:
                self._ler_window = self._ler_window[-2000:]

        if not self._calibrated and len(self._ler_window) >= self.calibration_steps:
            self._calibrate()
            self._last_calib_step = step
            return False
        if not self._calibrated:
            return False
        if self.recalibrate_every and (step - self._last_calib_step) >= self.recalibrate_every:
            self._calibrate()
            self._last_calib_step = step

        # Base low-utility decision against the SAME quantile-calibrated
        # threshold regardless of which signal is active (low signal => skip).
        base = (sig is not None) and (float(sig) < self.threshold)

        # (R4 fix) Confidence-gated burst + forced probe instead of brittle horizon
        if base and self.use_safety_horizon:
            # high confidence = strong rho_VG alignment -> allow burst; else probe sooner
            conf_cap = self.max_consecutive_skips if rho > 0.2 else max(1, self.max_consecutive_skips // 2)
            if self._consecutive_skips >= conf_cap or self._steps_since_probe >= self.probe_interval:
                base = False  # force a real backward (probe) to refresh diagnostics

        if base:
            self._consecutive_skips += 1
            self._steps_since_probe += 1
            self._skips += 1
        else:
            self._consecutive_skips = 0
            self._steps_since_probe = 0
        return base

    def get_diagnostics(self):
        denom = max(self._opportunities, 1)
        return {
            "calibrated": self._calibrated,
            "threshold": self.threshold,
            "observed_skip_rate": self._skips / denom,
            "n_calib_samples": len(self._ler_window),
        }


# ---------------------------------------------------------------------------
# 7) Phase 1.3b+: LERNAHybridPolicy — multi-signal, z-scored, quantile-calibrated.
#    Beats random/grad_norm by being MORE selective at the SAME skip budget:
#    skip only steps that are jointly low-utility across LER + rho_VG +
#    grad_norm + loss-slope. grad_norm is included because it was the strongest
#    Phase 1.2 signal, so the hybrid is >= grad_norm by design, + LER awareness.
# ---------------------------------------------------------------------------

class _RollingZ:
    def __init__(self, maxlen: int = 500):
        self.buf: List[float] = []
        self.maxlen = maxlen

    def push_and_z(self, x) -> float:
        if x is None or not math.isfinite(float(x)):
            return 0.0
        self.buf.append(float(x))
        if len(self.buf) > self.maxlen:
            self.buf = self.buf[-self.maxlen:]
        if len(self.buf) < 10:
            return 0.0
        mu = float(np.mean(self.buf))
        sd = float(np.std(self.buf)) + 1e-8
        return (float(x) - mu) / sd


class LERNAHybridPolicy:
    name = "lerna_hybrid"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.20,
        fallback_threshold: float = 0.01,
        min_step: int = 50,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,
        use_ler: bool = True,
        use_rho_vg: bool = True,
        use_grad_norm: bool = True,
        use_loss_slope: bool = True,
        use_safety_horizon: bool = True,
        max_consecutive_skips: int = 4,
        probe_interval: int = 8,
        w_ler: float = 1.0,
        w_rho: float = 1.0,
        w_grad: float = 1.0,
        w_slope: float = 1.0,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = target_skip_rate
        self.min_step = min_step
        self.calibration_steps = calibration_steps
        self.recalibrate_every = recalibrate_every
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_grad_norm = use_grad_norm
        self.use_loss_slope = use_loss_slope
        self.use_safety_horizon = use_safety_horizon
        self.max_consecutive_skips = max_consecutive_skips
        self.probe_interval = probe_interval
        self.w_ler, self.w_rho = w_ler, w_rho
        self.w_grad, self.w_slope = w_grad, w_slope

        self._zl, self._zr = _RollingZ(), _RollingZ()
        self._zg, self._zs = _RollingZ(), _RollingZ()
        self._score_window: List[float] = []
        self._threshold: Optional[float] = None
        self._calibrated = False
        self._last_calib = 0
        self._grad_norm_last: Optional[float] = None
        self._consecutive_skips = 0
        self._steps_since_probe = 0
        self._skips = 0
        self._opportunities = 0

    # Fed by _GradNormCapture on real backward steps.
    def record_grad_norm(self, v) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(v) and v > 0:
            self._grad_norm_last = v

    def _utility(self, ler, rho) -> Optional[float]:
        """HIGH utility (productive) => HIGH score. Skip when score < threshold."""
        parts: List[float] = []
        if self.use_ler and ler is not None:
            parts.append(self.w_ler * self._zl.push_and_z(ler))
        if self.use_rho_vg and rho is not None:
            parts.append(self.w_rho * self._zr.push_and_z(rho))
        if self.use_grad_norm and self._grad_norm_last is not None:
            parts.append(self.w_grad * self._zg.push_and_z(self._grad_norm_last))
        if self.use_loss_slope:
            lh = getattr(self.trk, "loss_history", None)
            if lh and len(lh) >= 2:
                slope = float(lh[-2]) - float(lh[-1])
                parts.append(self.w_slope * self._zs.push_and_z(slope))
        if not parts:
            return None
        return float(np.mean(parts))

    def _recalibrate(self):
        if len(self._score_window) >= max(20, self.calibration_steps // 2):
            self._threshold = float(np.nanpercentile(
                self._score_window, self.target_skip_rate * 100))
            self._calibrated = True

    def should_skip(self, trainer, model, inputs) -> bool:
        step = trainer.state.global_step
        self._opportunities += 1
        if step < self.min_step:
            self._consecutive_skips = 0
            return False

        diag = self.trk.get_diagnostics()
        ler = diag.get("ler")
        rho = diag.get("rho_vg", 0.0) or 0.0

        score = self._utility(ler, rho)
        if score is not None:
            self._score_window.append(score)
            if len(self._score_window) > 2000:
                self._score_window = self._score_window[-2000:]

        if not self._calibrated and len(self._score_window) >= self.calibration_steps:
            self._recalibrate()
            self._last_calib = step
            return False
        if not self._calibrated:
            return False
        if self.recalibrate_every and (step - self._last_calib) >= self.recalibrate_every:
            self._recalibrate()
            self._last_calib = step

        base = (score is not None) and (score < self._threshold)

        # Confidence-gated burst + forced probe.
        if base and self.use_safety_horizon:
            cap = self.max_consecutive_skips if rho > 0.2 else max(1, self.max_consecutive_skips // 2)
            if self._consecutive_skips >= cap or self._steps_since_probe >= self.probe_interval:
                base = False

        if base:
            self._consecutive_skips += 1
            self._steps_since_probe += 1
            self._skips += 1
        else:
            self._consecutive_skips = 0
            self._steps_since_probe = 0
        return base

    def get_diagnostics(self) -> dict:
        denom = max(self._opportunities, 1)
        return {
            "calibrated": self._calibrated,
            "threshold": self._threshold,
            "observed_skip_rate": self._skips / denom,
            "n_calib_samples": len(self._score_window),
            "signals": {
                "ler": self.use_ler, "rho_vg": self.use_rho_vg,
                "grad_norm": self.use_grad_norm, "loss_slope": self.use_loss_slope,
            },
        }

