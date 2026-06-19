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

    def __init__(
        self,
        target_skip_rate: float = 0.22,
        min_step: int = 100,
        seed: int = 42,
        total_steps: Optional[int] = None,
    ):
        self.target_skip_rate = float(target_skip_rate)
        self.min_step = int(min_step)
        self.total_steps = total_steps
        self._rng = random.Random(seed)

        self._skip_set = None
        self._quota_total_steps = None
        self._quota_size = None

        self._decision_idx = -1
        self._decisions_seen = 0
        self._skip_decisions = 0

    def _build_skip_set(self, total_steps: int):
        total_steps = int(total_steps)
        eligible = list(range(self.min_step, total_steps))

        # Match the reported overall skip ratio, not only post-warmup ratio.
        k = int(round(self.target_skip_rate * total_steps))
        k = min(k, len(eligible))

        self._skip_set = set(self._rng.sample(eligible, k)) if k > 0 else set()
        self._quota_total_steps = total_steps
        self._quota_size = k

    def should_skip(self, trainer, model, inputs) -> bool:
        self._decision_idx += 1
        self._decisions_seen += 1

        if self._decision_idx < self.min_step:
            return False

        if self._skip_set is None:
            runtime_total = getattr(getattr(trainer, "state", None), "max_steps", None)
            total = runtime_total or self.total_steps
            if total is not None and int(total) > self.min_step:
                self._build_skip_set(int(total))

        if self._skip_set is not None:
            decision = self._decision_idx in self._skip_set
        else:
            decision = self._rng.random() < self.target_skip_rate

        if decision:
            self._skip_decisions += 1

        return decision

    def get_diagnostics(self):
        return {
            "target_skip_rate": self.target_skip_rate,
            "min_step": self.min_step,
            "quota_total_steps": self._quota_total_steps,
            "quota_size": self._quota_size,
            "decisions_seen": self._decisions_seen,
            "skip_decisions": self._skip_decisions,
            "realized_skip_rate": self._skip_decisions / max(1, self._decisions_seen),
        }


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
        w_ler: float = 0.5,    # weak/smooth signal — keep small
        w_rho: float = 0.0,    # rho_VG moved to SAFETY VETO, not utility
        w_grad: float = 2.0,   # strongest signal once it's the TRUE pre-clip norm
        w_slope: float = 1.5,  # only genuinely per-step signal (raw loss slope)
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
        """HIGH utility (productive) => HIGH score. Skip when score < threshold.
        rho_VG is intentionally NOT a utility term here — it is used as a
        safety veto in should_skip() instead (low/negative rho = thrashing)."""
        parts: List[float] = []
        if self.use_ler and self.w_ler > 0 and ler is not None:
            parts.append(self.w_ler * self._zl.push_and_z(ler))
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
        ler = diag.get("ler_raw", diag.get("ler"))            # per-step, not smoothed
        rho = diag.get("rho_vg_raw", diag.get("rho_vg", 0.0)) or 0.0

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

        # SAFETY VETO: never skip when velocity fights the update (thrashing),
        # because momentum extrapolation on a skipped step is then risky.
        if base and rho < 0.0:
            base = False

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



# ---------------------------------------------------------------------------
# 8) Fix 7: LERNAQuotaHybridPolicy — exact-quota, two-stage (filter -> rank).
#    Stage 1 safety vetoes (rho = SAFETY only). Stage 2 fills an exact skip
#    budget with the LOWEST predicted-harm steps using rolling PERCENTILE RANKS
#    (robust to heavy-tailed grad norms) + online quota pressure.
# ---------------------------------------------------------------------------

class _RollingRank:
    """Rolling percentile-rank estimator. Robust to heavy tails (rank-based)."""
    def __init__(self, maxlen: int = 500, warmup: int = 5):
        self.buf: List[float] = []
        self.maxlen = maxlen
        self.warmup = warmup

    def push_and_rank(self, x) -> Optional[float]:
        if x is None:
            return None
        try:
            x = float(x)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        if len(self.buf) < self.warmup:
            self.buf.append(x)
            return 0.5  # neutral until enough history
        rank = sum(1 for v in self.buf if v <= x) / len(self.buf)
        self.buf.append(x)
        if len(self.buf) > self.maxlen:
            self.buf = self.buf[-self.maxlen:]
        return rank


class LERNAQuotaHybridPolicy:
    name = "lerna_quota_hybrid"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.20,
        fallback_threshold: float = 0.01,   # accepted for ctor-compat (unused)
        min_step: int = 50,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,       # accepted for ctor-compat (unused)
        use_ler: bool = True,
        use_rho_vg: bool = True,            # accepted for ctor-compat
        use_grad_norm: bool = True,
        use_loss_slope: bool = True,
        use_safety_horizon: bool = True,
        max_consecutive_skips: int = 1,
        probe_interval: int = 8,
        total_steps: Optional[int] = None,
        rho_veto_threshold: float = -0.2,
        w_grad: float = 0.65,
        w_slope: float = 0.25,
        w_ler: float = 0.10,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = float(target_skip_rate)
        self.min_step = int(min_step)
        self.calibration_steps = int(calibration_steps)
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_grad_norm = use_grad_norm
        self.use_loss_slope = use_loss_slope
        self.use_safety_horizon = use_safety_horizon
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.probe_interval = int(probe_interval)
        self.total_steps = total_steps
        self.rho_veto_threshold = float(rho_veto_threshold)
        self.w_grad, self.w_slope, self.w_ler = w_grad, w_slope, w_ler

        self._rg, self._rs, self._rl = _RollingRank(), _RollingRank(), _RollingRank()
        self._harm_window: List[float] = []
        self._grad_norm_last: Optional[float] = None
        self._grad_norm_is_stale = False
        self._consecutive_skips = 0
        self._steps_since_probe = 0

        self._quota_total_steps: Optional[int] = None
        self._quota_size: Optional[int] = None
        self._decisions_seen = 0
        self._skip_decisions = 0
        self._safe_candidates_seen = 0

        self._warmup_veto = self._rho_veto = self._probe_veto = 0
        self._max_consec_veto = self._missing_veto = self._quota_exhausted = 0

        self._dynamic_threshold: Optional[float] = None
        self._last_harm: Optional[float] = None
        self._grad_rank_last = self._slope_rank_last = self._ler_rank_last = None
        self._rho_last: Optional[float] = None
        self._selected: List[float] = []
        self._nonselected: List[float] = []

    # Fed by the trainer on real backward steps (single source of truth).
    def record_grad_norm(self, v) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(v) and v > 0:
            self._grad_norm_last = v
            self._grad_norm_is_stale = False

    def record_loss(self, v) -> None:  # ctor-compat with LERFeedCallback
        pass

    def _mark_real(self):
        self._consecutive_skips = 0
        self._steps_since_probe = 0

    def _do_skip(self):
        self._skip_decisions += 1
        self._consecutive_skips += 1
        self._steps_since_probe += 1
        self._grad_norm_is_stale = True  # next decision's grad norm is now stale
        if self._last_harm is not None:
            self._selected.append(self._last_harm)
        return True


    def should_skip(self, trainer, model, inputs) -> bool:
        decision_idx = self._decisions_seen  # 0-based this decision
        self._decisions_seen += 1

        # Lazily fix the EXACT quota from runtime max_steps (like random_skip).
        if self._quota_size is None:
            rt = getattr(getattr(trainer, "state", None), "max_steps", None)
            total = rt or self.total_steps
            if total:
                self._quota_total_steps = int(total)
                self._quota_size = int(round(self.target_skip_rate * int(total)))

        # ---------- STAGE 1: SAFETY FILTER ----------
        # [FIX 1] Use internal decision counter, not trainer.state.global_step.
        if decision_idx < self.min_step:
            self._warmup_veto += 1
            self._mark_real()
            return False

        diag = self.trk.get_diagnostics()
        ler = diag.get("ler_raw", diag.get("ler"))
        rho = diag.get("rho_vg_raw", diag.get("rho_vg", 0.0)) or 0.0
        self._rho_last = rho

        # [FIX 3] Gate safety-specific vetoes behind use_safety_horizon.
        if self.use_safety_horizon:
            if self._consecutive_skips >= self.max_consecutive_skips:
                self._max_consec_veto += 1
                self._mark_real()
                return False
            if self._steps_since_probe >= self.probe_interval:
                self._probe_veto += 1
                self._mark_real()
                return False

        # [FIX 2] rho veto is gated by use_rho_vg for clean ablations.
        if self.use_rho_vg and rho < self.rho_veto_threshold:
            self._rho_veto += 1
            self._mark_real()
            return False

        # ---------- harm score from PERCENTILE RANKS ----------
        grad_rank = self._rg.push_and_rank(self._grad_norm_last) if self.use_grad_norm else 0.5
        slope = None
        lh = getattr(self.trk, "loss_history", None)
        if self.use_loss_slope and lh and len(lh) >= 2:
            slope = float(lh[-2]) - float(lh[-1])  # +ve = improving
        slope_rank = self._rs.push_and_rank(slope)
        ler_rank = self._rl.push_and_rank(ler) if self.use_ler else 0.5

        if grad_rank is None:  # grad norm is the dominant signal; don't skip blind
            self._missing_veto += 1
            self._mark_real()
            return False

        sr = slope_rank if slope_rank is not None else 0.5
        lr_ = ler_rank if ler_rank is not None else 0.5
        harm = self.w_grad * grad_rank + self.w_slope * sr + self.w_ler * lr_
        self._last_harm = harm
        self._grad_rank_last, self._slope_rank_last, self._ler_rank_last = grad_rank, sr, lr_
        self._safe_candidates_seen += 1

        # ---------- STAGE 2: QUOTA-AWARE RANK SELECTION ----------
        remaining_skips = (self._quota_size - self._skip_decisions) if self._quota_size is not None else None
        remaining_decisions = (max(self._quota_total_steps - self._decisions_seen, 0)
                               if self._quota_total_steps is not None else None)

        if remaining_skips is not None and remaining_skips <= 0:
            self._quota_exhausted += 1
            self._mark_real()
            self._nonselected.append(harm)
            return False

        # need a harm-history warmup for a meaningful dynamic threshold
        if len(self._harm_window) < self.calibration_steps:
            self._harm_window.append(harm)
            if len(self._harm_window) > 2000:
                self._harm_window = self._harm_window[-2000:]
            self._mark_real()
            self._nonselected.append(harm)
            return False

        # force-skip tail: must skip everything safe that remains
        if (remaining_skips is not None and remaining_decisions is not None
                and remaining_skips >= remaining_decisions):
            self._harm_window.append(harm)
            if len(self._harm_window) > 2000:
                self._harm_window = self._harm_window[-2000:]
            return self._do_skip()

        if remaining_decisions and remaining_decisions > 0 and remaining_skips is not None:
            pressure = remaining_skips / remaining_decisions
        else:
            pressure = self.target_skip_rate
        pressure = min(max(pressure, 0.0), 1.0)
        # [FIX 4] Compute threshold from PREVIOUS harm window (before appending current).
        self._dynamic_threshold = float(np.nanpercentile(self._harm_window, pressure * 100.0))

        if harm <= self._dynamic_threshold:
            self._harm_window.append(harm)
            if len(self._harm_window) > 2000:
                self._harm_window = self._harm_window[-2000:]
            return self._do_skip()
        self._harm_window.append(harm)
        if len(self._harm_window) > 2000:
            self._harm_window = self._harm_window[-2000:]
        self._mark_real()
        self._nonselected.append(harm)
        return False

    def get_diagnostics(self) -> dict:
        hw = self._harm_window
        denom = max(self._decisions_seen, 1)
        return {
            "policy_name": self.name,
            "target_skip_rate": self.target_skip_rate,
            "quota_total_steps": self._quota_total_steps,
            "quota_size": self._quota_size,
            "decisions_seen": self._decisions_seen,
            "skip_decisions": self._skip_decisions,
            "remaining_skips": (self._quota_size - self._skip_decisions) if self._quota_size else None,
            "realized_skip_rate": self._skip_decisions / denom,
            "dynamic_threshold": self._dynamic_threshold,
            "safe_candidates_seen": self._safe_candidates_seen,
            "warmup_veto_count": self._warmup_veto,
            "rho_veto_count": self._rho_veto,
            "probe_veto_count": self._probe_veto,
            "max_consecutive_veto_count": self._max_consec_veto,
            "missing_signal_veto_count": self._missing_veto,
            "quota_exhausted_count": self._quota_exhausted,
            "harm_score_min": float(np.min(hw)) if hw else None,
            "harm_score_max": float(np.max(hw)) if hw else None,
            "harm_score_mean": float(np.mean(hw)) if hw else None,
            "grad_rank_last": self._grad_rank_last,
            "slope_rank_last": self._slope_rank_last,
            "ler_rank_last": self._ler_rank_last,
            "rho_last": self._rho_last,
            "grad_norm_last": self._grad_norm_last,
            "grad_norm_is_stale": self._grad_norm_is_stale,
            "selected_skip_score_mean": float(np.mean(self._selected)) if self._selected else None,
            "non_skip_score_mean": float(np.mean(self._nonselected)) if self._nonselected else None,
        }


# ---------------------------------------------------------------------------
# 9) Fix 8: LERNAGuardedStochasticPolicy — Guarded Stratified Random LERNA
# ---------------------------------------------------------------------------

class LERNAGuardedStochasticPolicy:
    """Fix 8: Guarded Stratified Random LERNA (--policy guarded_hybrid).

    Keep random's diversity; use LERNA signals ONLY as hard vetoes that forbid
    skipping clearly-important steps. Among SAFE candidates, skip stochastically
    at budget pressure with per-window stratification (matched budget + spread).

    Signals are rolling PERCENTILE RANKS (relative to recent history) so vetoes
    are scale-free and do not bias skips toward late training.

    risk_gamma == 0.0  -> pure guarded random (strongest hypothesis)
    risk_gamma  > 0.0  -> soft, budget-preserving bias toward lower-harm steps
    """
    name = "lerna_guarded_hybrid"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.20,
        total_steps: Optional[int] = None,
        min_step: int = 50,
        seed: int = 42,
        window_size: int = 40,
        window_quota_slack: int = 1,          # [#4] headroom so cap isn't razor-tight
        max_consecutive_skips: int = 1,
        probe_interval: int = 8,
        grad_rank_veto: float = 0.80,
        loss_rank_veto: float = 0.75,
        ler_rank_veto: float = 0.80,
        rho_veto_threshold: float = -0.2,
        spike_factor: float = 0.25,
        use_grad_norm: bool = True,
        use_loss_rank: bool = True,
        use_ler: bool = True,
        use_rho_vg: bool = True,
        use_safety_horizon: bool = True,      # [#3] threaded from ablation overrides
        risk_gamma: float = 0.0,
        guard_mode: str = "on",              # "on" = full guarded logic; "off" = pure quota random (debug)
        # ctor-compat (unused) so the runner can pass shared kwargs
        fallback_threshold: float = 0.01,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = float(target_skip_rate)
        self.total_steps = total_steps
        self.min_step = int(min_step)
        self.window_size = int(window_size)
        self.window_quota_slack = int(window_quota_slack)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.probe_interval = int(probe_interval)
        self.grad_rank_veto = float(grad_rank_veto)
        self.loss_rank_veto = float(loss_rank_veto)
        self.ler_rank_veto = float(ler_rank_veto)
        self.rho_veto_threshold = float(rho_veto_threshold)
        self.spike_factor = float(spike_factor)
        self.use_grad_norm = use_grad_norm
        self.use_loss_rank = use_loss_rank
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_safety_horizon = use_safety_horizon
        self.risk_gamma = float(risk_gamma)
        self.guard_mode = guard_mode
        self.name = f"lerna_guarded_hybrid_{self.guard_mode}"

        self._off_seed = seed
        self._off_skip_set = None
        self._off_skip_set_size = None

        self._rng = random.Random(seed)
        self._rg = _RollingRank()      # grad norm
        self._rloss = _RollingRank()   # current loss
        self._rler = _RollingRank()    # LER

        self._grad_norm_last: Optional[float] = None
        self._grad_norm_is_stale = False
        self._consecutive_skips = 0
        self._steps_since_probe = 0

        self._quota_total_steps: Optional[int] = None

        # [Fix 8c] Per-run state counters MUST live in __init__ (they were
        # erroneously being reset to 0 at the top of every should_skip() call,
        # which both (a) forced di=0<min_step -> permanent warmup veto and
        # (b) raised AttributeError before the guard_mode branch was ever
        # reached -> trainer caught it and never skipped -> skip_ratio=0.0).
        self._quota_size: Optional[int] = None
        self._window_quota = 1
        self._decisions_seen = 0
        self._skip_decisions = 0
        self._cur_window = -1
        self._window_skips = 0

        # debug diagnostics for the stochastic sampler
        self._pressure_last: Optional[float] = None
        self._probability_last: Optional[float] = None
        self._random_draw_last: Optional[float] = None
        self._reached_sampling_count = 0

        self._warmup_veto = self._rho_veto = self._grad_veto = 0
        self._loss_veto = self._ler_veto = self._spike_veto = 0
        self._max_consec_veto = self._probe_veto = self._missing_veto = 0
        self._window_cap_veto = self._quota_exhausted = self._forced_tail = 0
        self._random_safe_skip = self._safe_candidates_seen = 0

        self._last_harm = None
        self._grad_rank_last = self._loss_rank_last = self._ler_rank_last = None
        self._rho_last = None
        self._selected: List[float] = []
        self._nonselected: List[float] = []

    # fed by TrueBackwardSkippingTrainer.training_step on real backward steps
    def record_grad_norm(self, v) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(v) and v > 0:
            self._grad_norm_last = v
            self._grad_norm_is_stale = False

    def record_loss(self, v) -> None:   # ctor-compat with LERFeedCallback
        pass

    def _mark_real(self):
        self._consecutive_skips = 0
        self._steps_since_probe = 0

    def _do_skip(self):
        self._skip_decisions += 1
        self._window_skips += 1
        self._consecutive_skips += 1
        self._steps_since_probe += 1
        self._grad_norm_is_stale = True
        if self._last_harm is not None:
            self._selected.append(self._last_harm)
        return True

    def should_skip(self, trainer, model, inputs) -> bool:
        di = self._decisions_seen
        self._decisions_seen += 1

        if getattr(self, "guard_mode", None) == "off":
            if self._off_skip_set is None:
                total = (
                    self._quota_total_steps
                    or getattr(getattr(trainer, "state", None), "max_steps", None)
                    or getattr(getattr(trainer, "args", None), "max_steps", None)
                    or self.total_steps
                )

                if total is None or int(total) <= self.min_step:
                    raise ValueError(
                        f"guard_mode=off could not resolve total_steps: "
                        f"state.max_steps={getattr(getattr(trainer, 'state', None), 'max_steps', None)}, "
                        f"args.max_steps={getattr(getattr(trainer, 'args', None), 'max_steps', None)}, "
                        f"self.total_steps={self.total_steps}, "
                        f"min_step={self.min_step}"
                    )

                total = int(total)
                eligible = list(range(self.min_step, total))
                k = min(int(round(self.target_skip_rate * total)), len(eligible))

                rng = random.Random(self._off_seed)
                self._off_skip_set = set(rng.sample(eligible, k)) if k > 0 else set()

                self._quota_total_steps = total
                self._quota_size = k
                self._off_skip_set_size = len(self._off_skip_set)

            decision = di in self._off_skip_set
            if decision:
                self._skip_decisions += 1
                self._random_safe_skip += 1
            return decision

        # lazy EXACT quota from runtime max_steps (same source as random_skip)
        if self._quota_size is None:
            rt = getattr(getattr(trainer, "state", None), "max_steps", None)
            total = rt or self.total_steps
            if total:
                self._quota_total_steps = int(total)
                self._quota_size = int(round(self.target_skip_rate * int(total)))
                self._window_quota = max(
                    1, int(math.ceil(self.target_skip_rate * self.window_size))
                       + self.window_quota_slack)

        if di < self.min_step:
            self._warmup_veto += 1
            self._mark_real(); return False

        diag = self.trk.get_diagnostics()
        ler = diag.get("ler_raw", diag.get("ler"))
        rho = diag.get("rho_vg_raw", diag.get("rho_vg", 0.0)) or 0.0
        self._rho_last = rho

        if self.use_safety_horizon:
            if self._consecutive_skips >= self.max_consecutive_skips:
                self._max_consec_veto += 1; self._mark_real(); return False
            if self._steps_since_probe >= self.probe_interval:
                self._probe_veto += 1; self._mark_real(); return False

        grad_rank = self._rg.push_and_rank(self._grad_norm_last) if self.use_grad_norm else 0.5
        lh = getattr(self.trk, "loss_history", None)
        cur_loss = float(lh[-1]) if lh else None
        loss_rank = (self._rloss.push_and_rank(cur_loss)
                     if (self.use_loss_rank and cur_loss is not None) else 0.5)
        ler_rank = self._rler.push_and_rank(ler) if self.use_ler else 0.5

        if grad_rank is None:
            self._missing_veto += 1; self._mark_real(); return False
        lr_ = loss_rank if loss_rank is not None else 0.5
        er_ = ler_rank if ler_rank is not None else 0.5

        harm = 0.5 * grad_rank + 0.3 * lr_ + 0.2 * er_
        self._last_harm = harm
        self._grad_rank_last, self._loss_rank_last, self._ler_rank_last = grad_rank, lr_, er_

        spike = False
        if lh and len(lh) >= 6 and cur_loss is not None:
            ema = float(np.mean(lh[-6:-1]))
            spike = cur_loss > ema * (1.0 + self.spike_factor)

        # ---- quota accounting (computed BEFORE soft vetoes so the global
        #      budget can always be met at the tail) ----
        remaining_skips = (self._quota_size - self._skip_decisions) if self._quota_size is not None else None
        if remaining_skips is not None and remaining_skips <= 0:
            self._quota_exhausted += 1; self._mark_real(); self._nonselected.append(harm); return False
        remaining_decisions = (max(self._quota_total_steps - self._decisions_seen, 0)
                               if self._quota_total_steps is not None else None)

        # forced-tail regime: not enough decisions left to hit the quota
        # otherwise. Here the matched budget takes priority over the SOFT
        # rank vetoes (grad/loss/ler) and the window cap — but NOT over the
        # genuine-danger guards (rho, spike, max_consecutive) above/below.
        forced_tail = (remaining_skips is not None and remaining_decisions is not None
                       and remaining_skips >= remaining_decisions)

        # ---- HARD danger vetoes: never skip these, even at the tail ----
        if self.use_rho_vg and rho < self.rho_veto_threshold:
            self._rho_veto += 1; self._mark_real(); self._nonselected.append(harm); return False
        if spike:
            self._spike_veto += 1; self._mark_real(); self._nonselected.append(harm); return False

        # ---- SOFT rank vetoes: the actual "guarding". Bypassed only in the
        #      forced-tail regime so matched budget is preserved. ----
        if not forced_tail:
            if self.use_grad_norm and grad_rank > self.grad_rank_veto:
                self._grad_veto += 1; self._mark_real(); self._nonselected.append(harm); return False
            if self.use_loss_rank and lr_ > self.loss_rank_veto:
                self._loss_veto += 1; self._mark_real(); self._nonselected.append(harm); return False
            if self.use_ler and er_ > self.ler_rank_veto:
                self._ler_veto += 1; self._mark_real(); self._nonselected.append(harm); return False

        self._safe_candidates_seen += 1

        # [#4] force-skip tail so exact global quota beats stratification
        if forced_tail:
            self._forced_tail += 1
            return self._do_skip()

        # window bookkeeping
        w = di // self.window_size
        if w != self._cur_window:
            self._cur_window = w
            self._window_skips = 0

        # window cap (stratification) — prevents late-training skip bursts
        if self._window_skips >= self._window_quota:
            self._window_cap_veto += 1; self._mark_real(); self._nonselected.append(harm); return False

        # ---- stochastic acceptance at budget pressure (DIVERSITY) ----
        if remaining_decisions and remaining_decisions > 0 and remaining_skips is not None:
            pressure = remaining_skips / remaining_decisions
        else:
            pressure = self.target_skip_rate
        p = pressure * (1.0 + self.risk_gamma * (0.5 - harm))   # gamma=0 -> pure guarded random
        p = min(max(p, 0.0), 1.0)

        self._reached_sampling_count += 1
        self._pressure_last = pressure
        self._probability_last = p
        draw = self._rng.random()
        self._random_draw_last = draw
        if draw < p:
            self._random_safe_skip += 1
            return self._do_skip()
        self._mark_real(); self._nonselected.append(harm); return False

    def get_diagnostics(self) -> dict:
        denom = max(self._decisions_seen, 1)
        return {
            "policy_name": self.name,
            "target_skip_rate": self.target_skip_rate,
            "quota_total_steps": self._quota_total_steps,
            "quota_size": self._quota_size,
            "window_size": self.window_size,
            "window_quota": self._window_quota,
            "current_window": self._cur_window,
            "window_skips": self._window_skips,
            "decisions_seen": self._decisions_seen,
            "skip_decisions": self._skip_decisions,
            "realized_skip_rate": self._skip_decisions / denom,
            "safe_candidates_seen": self._safe_candidates_seen,
            "warmup_veto_count": self._warmup_veto,
            "rho_veto_count": self._rho_veto,
            "grad_veto_count": self._grad_veto,
            "loss_veto_count": self._loss_veto,
            "ler_veto_count": self._ler_veto,
            "spike_veto_count": self._spike_veto,
            "max_consecutive_veto_count": self._max_consec_veto,
            "probe_veto_count": self._probe_veto,
            "missing_signal_veto_count": self._missing_veto,
            "window_cap_veto_count": self._window_cap_veto,
            "quota_exhausted_count": self._quota_exhausted,
            "forced_tail_skip_count": self._forced_tail,
            "random_safe_skip_count": self._random_safe_skip,
            "risk_gamma": self.risk_gamma,
            "guard_mode": getattr(self, "guard_mode", None),
            "off_skip_set_size": getattr(self, "_off_skip_set_size", None),
            "pressure_last": self._pressure_last,
            "probability_last": self._probability_last,
            "random_draw_last": self._random_draw_last,
            "reached_sampling_count": self._reached_sampling_count,
            "grad_rank_last": self._grad_rank_last,
            "loss_rank_last": self._loss_rank_last,
            "ler_rank_last": self._ler_rank_last,
            "rho_last": self._rho_last,
            "grad_norm_last": self._grad_norm_last,
            "grad_norm_is_stale": self._grad_norm_is_stale,
            "skipped_risk_mean": float(np.mean(self._selected)) if self._selected else None,
            "non_skipped_risk_mean": float(np.mean(self._nonselected)) if self._nonselected else None,
        }


class LERNAPhaseStratifiedPolicy:
    """Phase-Stratified Guarded Random LERNA  (--policy phase_strat).

    Thesis: step inequality is mostly PHASE inequality. Protect high-utility
    (early) updates; spend the skip budget preferentially on low-utility
    (late/plateau) phases; keep within-phase skipping RANDOM (unbiased) so we
    inherit random's strength instead of fighting it.

    Budget split across P phases by `phase_weights` (must sum to ~1). Within a
    phase, skip stochastically at that phase's local pressure. LERNA signals
    are HARD danger vetoes only (loss spike / rho thrashing), never an
    easy-step selector. Optional tiny `risk_gamma` biases within-phase choice
    toward redundant steps without breaking the matched budget.
    """
    name = "lerna_phase_strat"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.20,
        total_steps: Optional[int] = None,
        min_step: int = 50,
        seed: int = 42,
        n_phases: int = 4,
        phase_weights: Optional[List[float]] = None,
        max_consecutive_skips: int = 1,
        rho_veto_threshold: float = -0.2,
        spike_factor: float = 1.0,
        use_rho_vg: bool = True,
        use_safety_horizon: bool = True,
        risk_gamma: float = 0.0,
        fallback_threshold: float = 0.01,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,
        probe_interval: int = 8,
        use_ler: bool = True,
        use_grad_norm: bool = True,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = float(target_skip_rate)
        self.total_steps = total_steps
        self.min_step = int(min_step)
        self.n_phases = int(n_phases)
        if phase_weights is None:
            # still phase-aware, but much milder per-phase bias
            phase_weights = [0.22, 0.24, 0.26, 0.28][:self.n_phases]
        s = sum(phase_weights) or 1.0
        self.phase_weights = [w / s for w in phase_weights]
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.rho_veto_threshold = float(rho_veto_threshold)
        self.spike_factor = float(spike_factor)
        self.use_rho_vg = use_rho_vg
        self.use_safety_horizon = use_safety_horizon
        self.risk_gamma = float(risk_gamma)
        self.use_ler = use_ler
        self.use_grad_norm = use_grad_norm

        self._rng = random.Random(seed)
        self._rg = _RollingRank()

        self._quota_total_steps: Optional[int] = None
        self._quota_size: Optional[int] = None
        self._phase_bounds: List[int] = []
        self._phase_quota: List[int] = []
        self._phase_skips: List[int] = []
        self._phase_eligible: List[int] = []

        self._grad_norm_last: Optional[float] = None
        self._consecutive_skips = 0
        self._decisions_seen = 0
        self._skip_decisions = 0
        self._cur_phase = -1

        self._rho_last = None
        self._spike_veto = self._rho_veto = self._max_consec_veto = 0
        self._warmup_veto = self._quota_exhausted = self._forced_tail = 0
        self._random_safe_skip = 0
        self._selected: List[float] = []

    def record_grad_norm(self, v) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(v) and v > 0:
            self._grad_norm_last = v

    def record_loss(self, v) -> None:
        pass

    def _lazy_init(self, trainer):
        if self._quota_size is not None:
            return
        rt = getattr(getattr(trainer, "state", None), "max_steps", None)
        total = rt or self.total_steps
        if not total:
            total = getattr(getattr(trainer, "args", None), "max_steps", None)
        if not total or int(total) <= self.min_step:
            raise RuntimeError(
                "random_veto_deferral could not resolve total_steps; pass total_steps from runner"
            )
        total = int(total)
        self._quota_total_steps = total
        self._quota_size = int(round(self.target_skip_rate * total))

        eligible = list(range(self.min_step, total))
        n_elig = len(eligible)
        edges = [self.min_step + int(round(n_elig * i / self.n_phases))
                 for i in range(self.n_phases + 1)]
        self._phase_bounds = edges
        self._phase_eligible = [edges[i + 1] - edges[i] for i in range(self.n_phases)]

        raw = [self._quota_size * w for w in self.phase_weights]
        q = [int(math.floor(x)) for x in raw]
        q = [min(q[i], self._phase_eligible[i]) for i in range(self.n_phases)]
        rem = self._quota_size - sum(q)
        order = sorted(range(self.n_phases), key=lambda i: raw[i] - q[i], reverse=True)
        idx = 0
        while rem > 0 and idx < 10 * self.n_phases:
            i = order[idx % self.n_phases]
            if q[i] < self._phase_eligible[i]:
                q[i] += 1
                rem -= 1
            idx += 1
        self._phase_quota = q
        self._phase_skips = [0] * self.n_phases

    def _phase_of(self, di: int) -> int:
        for i in range(self.n_phases):
            if self._phase_bounds[i] <= di < self._phase_bounds[i + 1]:
                return i
        return self.n_phases - 1

    def _do_skip(self, ph: int) -> bool:
        self._skip_decisions += 1
        self._phase_skips[ph] += 1
        self._consecutive_skips += 1
        if self._grad_norm_last is not None:
            self._selected.append(self._grad_norm_last)
        return True

    def should_skip(self, trainer, model, inputs) -> bool:
        di = self._decisions_seen
        self._decisions_seen += 1
        self._lazy_init(trainer)

        if di < self.min_step or self._quota_size is None:
            self._warmup_veto += 1
            self._consecutive_skips = 0
            return False

        ph = self._phase_of(di)
        self._cur_phase = ph

        diag = self.trk.get_diagnostics()
        rho = diag.get("rho_vg_raw", diag.get("rho_vg", 0.0)) or 0.0
        self._rho_last = rho
        lh = getattr(self.trk, "loss_history", None)
        cur_loss = float(lh[-1]) if lh else None

        if self.use_safety_horizon and self._consecutive_skips >= self.max_consecutive_skips:
            self._max_consec_veto += 1
            self._consecutive_skips = 0
            return False
        if self.use_rho_vg and rho < self.rho_veto_threshold:
            self._rho_veto += 1
            self._consecutive_skips = 0
            return False
        if lh and len(lh) >= 6 and cur_loss is not None:
            ema = float(np.mean(lh[-6:-1]))
            if cur_loss > ema * (1.0 + self.spike_factor):
                self._spike_veto += 1
                self._consecutive_skips = 0
                return False

        q_left = self._phase_quota[ph] - self._phase_skips[ph]
        if q_left <= 0:
            self._quota_exhausted += 1
            self._consecutive_skips = 0
            return False
        decisions_left_in_phase = max(self._phase_bounds[ph + 1] - di, 1)

        if q_left >= decisions_left_in_phase:
            self._forced_tail += 1
            return self._do_skip(ph)

        pressure = q_left / decisions_left_in_phase
        if self.risk_gamma > 0.0 and self._grad_norm_last is not None:
            grad_rank = self._rg.push_and_rank(self._grad_norm_last) or 0.5
            pressure = pressure * (1.0 + self.risk_gamma * (0.5 - grad_rank))
        p = min(max(pressure, 0.0), 1.0)

        if self._rng.random() < p:
            self._random_safe_skip += 1
            return self._do_skip(ph)
        self._consecutive_skips = 0
        return False

    def get_diagnostics(self) -> dict:
        denom = max(self._decisions_seen, 1)
        return {
            "policy_name": self.name,
            "target_skip_rate": self.target_skip_rate,
            "quota_total_steps": self._quota_total_steps,
            "quota_size": self._quota_size,
            "n_phases": self.n_phases,
            "phase_weights": self.phase_weights,
            "phase_bounds": self._phase_bounds,
            "phase_quota": self._phase_quota,
            "phase_skips": self._phase_skips,
            "phase_eligible": self._phase_eligible,
            "decisions_seen": self._decisions_seen,
            "skip_decisions": self._skip_decisions,
            "realized_skip_rate": self._skip_decisions / denom,
            "current_phase": self._cur_phase,
            "warmup_veto_count": self._warmup_veto,
            "rho_veto_count": self._rho_veto,
            "spike_veto_count": self._spike_veto,
            "max_consecutive_veto_count": self._max_consec_veto,
            "quota_exhausted_count": self._quota_exhausted,
            "forced_tail_skip_count": self._forced_tail,
            "random_safe_skip_count": self._random_safe_skip,
            "risk_gamma": self.risk_gamma,
            "grad_norm_last": self._grad_norm_last,
            "skipped_grad_mean": float(np.mean(self._selected)) if self._selected else None,
        }


class LERNARandomVetoDeferralPolicy:
    """Sparse random skip with vetoes and deferred danger protection.

    This policy preserves exact global skip budget while applying sparse
    danger vetoes. Within the non-dangerous subset, skips are random.
    """
    name = "random_veto_deferral"

    def __init__(
        self,
        ler_tracker,
        target_skip_rate: float = 0.20,
        total_steps: Optional[int] = None,
        min_step: int = 50,
        seed: int = 42,
        use_loss_spike_veto: bool = False,
        spike_factor: float = 1.0,
        spike_ema_window: int = 20,
        use_rho_vg_veto: bool = False,
        rho_veto_threshold: float = -0.05,
        use_grad_norm_veto: bool = False,
        grad_rank_veto: float = 0.95,
        use_margin_veto: bool = False,
        margin_rank_floor: float = 0.10,
        use_novelty_veto: bool = False,
        novelty_rank_veto: float = 0.90,
        mem_bank_size: int = 64,
        target_veto_rate: float = 0.15,
        repay_mode: str = "spread",
        repay_protect_dangerous: bool = True,
        use_phase_protection: bool = False,
        n_phases: int = 4,
        early_protect_phases: int = 1,
        max_consecutive_skips: int = 4,
        use_safety_horizon: bool = True,
        probe_interval: int = 8,
        use_ler: bool = True,
        use_rho_vg: bool = True,
        use_grad_norm: bool = True,
        fallback_threshold: float = 0.01,
        calibration_steps: int = 60,
        recalibrate_every: int = 200,
        risk_gamma: float = 0.0,
    ):
        self.trk = ler_tracker
        self.target_skip_rate = float(target_skip_rate)
        self.total_steps = total_steps
        self.min_step = int(min_step)
        self.seed = int(seed)
        self.use_loss_spike_veto = use_loss_spike_veto
        self.spike_factor = float(spike_factor)
        self.spike_ema_window = int(spike_ema_window)
        self.use_rho_vg_veto = use_rho_vg_veto
        self.rho_veto_threshold = float(rho_veto_threshold)
        self.use_grad_norm_veto = use_grad_norm_veto
        self.grad_rank_veto = float(grad_rank_veto)
        self.use_margin_veto = use_margin_veto
        self.margin_rank_floor = float(margin_rank_floor)
        self.use_novelty_veto = use_novelty_veto
        self.novelty_rank_veto = float(novelty_rank_veto)
        self.mem_bank_size = int(mem_bank_size)
        self.target_veto_rate = float(target_veto_rate)
        self.repay_mode = repay_mode
        self.repay_protect_dangerous = repay_protect_dangerous
        self.use_phase_protection = use_phase_protection
        self.n_phases = int(n_phases)
        self.early_protect_phases = int(early_protect_phases)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.use_safety_horizon = use_safety_horizon
        self.probe_interval = int(probe_interval)
        self.use_ler = use_ler
        self.use_rho_vg = use_rho_vg
        self.use_grad_norm = use_grad_norm
        self.fallback_threshold = fallback_threshold
        self.calibration_steps = calibration_steps
        self.recalibrate_every = recalibrate_every
        self.risk_gamma = float(risk_gamma)

        self._rng = random.Random(self.seed)
        self._skip_set = None
        self._quota_total_steps: Optional[int] = None
        self._quota_size: Optional[int] = None
        self._candidate_set_size: Optional[int] = None
        self._decisions_seen = 0
        self._skip_decisions = 0
        self._accepted_random_skips = 0
        self._candidate_indices_seen: set[int] = set()
        self._deferred_pool: List[int] = []
        self._deferred_pool_peak = 0
        self._random_candidate_count = 0
        self._vetoed_skips = 0
        self._repaid_skips = 0
        self._danger_veto_counts_by_type = {
            "loss_spike": 0,
            "rho_thrash": 0,
            "grad_explosion": 0,
            "low_margin": 0,
            "novel_batch": 0,
        }
        self._rg = _RollingRank()
        self._rm = _RollingRank()
        self._rn = _RollingRank()
        self._grad_norm_last: Optional[float] = None
        self._margin_last: Optional[float] = None
        self._margin_available_count = 0
        self._margin_missing_count = 0
        self._grad_norm_is_stale = False
        self._mem: List[np.ndarray] = []
        self._consecutive_skips = 0
        self._phase_bounds: List[int] = []
        self._phase_skips: List[int] = []
        self._current_phase = -1

        self._warmup_veto = 0
        self._spike_veto = 0
        self._rho_veto = 0
        self._grad_veto = 0
        self._margin_veto = 0
        self._novelty_veto = 0
        self._max_consec_veto = 0
        self._quota_exhausted = 0
        self._forced_tail = 0
        self._random_safe_skip = 0
        self._deferred_count = 0

    def record_grad_norm(self, v) -> None:
        try:
            v = float(v)
        except (TypeError, ValueError):
            return
        if math.isfinite(v) and v > 0:
            self._grad_norm_last = v
            self._grad_norm_is_stale = False

    def record_loss(self, v) -> None:
        pass

    def _lazy_init(self, trainer):
        if self._quota_size is not None:
            return

        rt = getattr(getattr(trainer, "state", None), "max_steps", None)
        total = rt or self.total_steps or getattr(getattr(trainer, "args", None), "max_steps", None)

        if not total or int(total) <= self.min_step:
            raise RuntimeError(
                "random_veto_deferral could not resolve total_steps; pass total_steps from runner"
            )

        total = int(total)
        eligible = list(range(self.min_step, total))
        k = min(int(round(self.target_skip_rate * total)), len(eligible))

        rng = random.Random(self.seed)
        self._skip_set = set(rng.sample(eligible, k)) if k > 0 else set()

        self._quota_total_steps = total
        self._quota_size = k
        self._candidate_set_size = len(self._skip_set)

        if self._candidate_set_size != self._quota_size:
            raise RuntimeError(
                f"RVD candidate set size mismatch: candidate_set_size={self._candidate_set_size}, quota_size={self._quota_size}"
            )

        if self.use_phase_protection:
            self._phase_bounds = [
                self.min_step + int(round(len(eligible) * i / self.n_phases))
                for i in range(self.n_phases + 1)
            ]

        self._phase_skips = [0] * max(self.n_phases, 1)

    def _step_margin(self, trainer):
        logits = getattr(trainer, "_last_real_logits", None)
        if logits is None:
            logits = getattr(trainer, "_last_logits", None)
        if logits is None:
            logits = getattr(trainer, "last_logits", None)

        if logits is None:
            self._margin_missing_count += 1
            return None

        try:
            if isinstance(logits, (tuple, list)):
                logits = logits[0]

            if logits.dim() == 2 and logits.size(-1) >= 2:
                top2 = torch.topk(logits.detach().float(), 2, dim=-1).values
                margin = float((top2[:, 0] - top2[:, 1]).mean().item())
                self._margin_available_count += 1
                return margin
        except Exception:
            return None

        return None

    def _is_dangerous(self, trainer):
        dangerous = False
        danger_types = {
            "loss_spike": False,
            "rho_thrash": False,
            "grad_explosion": False,
            "low_margin": False,
            "novel_batch": False,
        }
        diag = self.trk.get_diagnostics()
        rho = diag.get("rho_vg_raw", diag.get("rho_vg", 0.0)) or 0.0
        lh = getattr(self.trk, "loss_history", None)
        cur_loss = float(lh[-1]) if lh else None
        if self.use_loss_spike_veto and cur_loss is not None and lh and len(lh) >= self.spike_ema_window + 1:
            ema = float(np.mean(lh[-self.spike_ema_window - 1:-1]))
            if cur_loss > ema * (1.0 + self.spike_factor):
                dangerous = True
                danger_types["loss_spike"] = True
        if self.use_rho_vg_veto and rho < self.rho_veto_threshold:
            dangerous = True
            danger_types["rho_thrash"] = True
        grad_rank = None
        if self.use_grad_norm_veto and self._grad_norm_last is not None:
            grad_rank = self._rg.push_and_rank(self._grad_norm_last)
            if grad_rank is not None and grad_rank > self.grad_rank_veto:
                dangerous = True
                danger_types["grad_explosion"] = True
        margin = None
        if self.use_margin_veto:
            margin = self._step_margin(trainer)
            self._margin_last = margin
            if margin is not None:
                margin_rank = self._rm.push_and_rank(margin)
                if margin_rank is not None and margin_rank < self.margin_rank_floor:
                    dangerous = True
                    danger_types["low_margin"] = True
        if self.use_novelty_veto:
            emb = getattr(trainer, "_last_cls_embedding", None)
            if emb is not None:
                try:
                    vec = np.asarray(emb, dtype=float).flatten()
                except Exception:
                    vec = None
                if vec is not None and vec.size:
                    if len(self._mem) >= self.mem_bank_size:
                        self._mem.pop(0)
                    norm = np.linalg.norm(vec)
                    if norm > 0:
                        vec = vec / norm
                        if self._mem:
                            sims = [float(np.dot(vec, m)) for m in self._mem]
                            sim = max(sims)
                        else:
                            sim = 0.0
                        novelty_rank = self._rn.push_and_rank(sim)
                        if novelty_rank is not None and novelty_rank > self.novelty_rank_veto:
                            dangerous = True
                            danger_types["novel_batch"] = True
                        self._mem.append(vec)
        return dangerous, danger_types

    def _all_vetoes_disabled(self) -> bool:
        return not (
            self.use_loss_spike_veto
            or self.use_rho_vg_veto
            or self.use_grad_norm_veto
            or self.use_margin_veto
            or self.use_novelty_veto
            or self.use_phase_protection
        )

    def _do_skip(self, idx=None):
        if self._deferred_pool:
            self._repaid_skips += 1
            self._deferred_pool.pop(0)
        self._skip_decisions += 1
        self._consecutive_skips += 1
        self._random_safe_skip += 1
        return True

    def should_skip(self, trainer, model, inputs) -> bool:
        idx = self._decisions_seen
        self._decisions_seen += 1
        self._lazy_init(trainer)

        if idx < self.min_step or self._quota_size is None:
            self._warmup_veto += 1
            self._consecutive_skips = 0
            return False

        if self._all_vetoes_disabled():
            decision = (self._skip_set is not None) and (idx in self._skip_set)

            if decision:
                self._random_candidate_count += 1
                self._candidate_indices_seen.add(idx)
                self._accepted_random_skips += 1
                return self._do_skip(idx)

            self._consecutive_skips = 0
            return False

        idx = self._decisions_seen - 1
        is_candidate = self._skip_set is not None and idx in self._skip_set

        remaining_skips = (self._quota_size - self._skip_decisions) if self._quota_size is not None else 0
        remaining_decisions = max((self._quota_total_steps - self._decisions_seen), 1) if self._quota_total_steps is not None else 1
        if remaining_skips <= 0:
            self._quota_exhausted += 1
            self._consecutive_skips = 0
            return False

        if is_candidate:
            self._random_candidate_count += 1
            self._candidate_indices_seen.add(idx)
            dangerous, danger_types = self._is_dangerous(trainer)
            if dangerous:
                for key, flagged in danger_types.items():
                    if flagged:
                        self._danger_veto_counts_by_type[key] += 1
                self._vetoed_skips += 1
                self._deferred_pool.append(idx)
                self._deferred_pool_peak = max(self._deferred_pool_peak, len(self._deferred_pool))
                self._deferred_count += 1
                self._consecutive_skips = 0
                return False

            self._accepted_random_skips += 1
            return self._do_skip(idx)

        if self._deferred_pool:
            if self.repay_mode == "asap":
                return self._do_skip(idx)

            p_repay = min(
                max(4.0 * len(self._deferred_pool) / max(remaining_decisions, 1), 0.0),
                1.0,
            )
            if self._rng.random() < p_repay:
                return self._do_skip(idx)

        tail_window = max(5, int(0.02 * self._quota_total_steps))
        if (
            remaining_skips >= remaining_decisions
            and idx >= self._quota_total_steps - tail_window
        ):
            self._forced_tail += 1
            return self._do_skip(idx)

        self._consecutive_skips = 0
        return False

    def get_diagnostics(self) -> dict:
        denom = max(self._decisions_seen, 1)
        full_backward_steps = self._decisions_seen - self._skip_decisions
        quota_ok = True
        if self._quota_total_steps is not None:
            quota_ok = (
                self._skip_decisions + full_backward_steps == self._decisions_seen
                and self._decisions_seen <= self._quota_total_steps
            )
        return {
            "policy_name": self.name,
            "target_skip_rate": self.target_skip_rate,
            "target_veto_rate": self.target_veto_rate,
            "realized_skip_rate": self._skip_decisions / denom,
            "all_vetoes_disabled": self._all_vetoes_disabled(),
            "use_loss_spike_veto": self.use_loss_spike_veto,
            "use_rho_vg_veto": self.use_rho_vg_veto,
            "use_grad_norm_veto": self.use_grad_norm_veto,
            "use_margin_veto": self.use_margin_veto,
            "margin_rank_floor": self.margin_rank_floor,
            "margin_last": self._margin_last,
            "use_novelty_veto": self.use_novelty_veto,
            "use_phase_protection": self.use_phase_protection,
            "candidate_set_size": self._candidate_set_size,
            "skip_set_contains_min_step": self.min_step in self._skip_set if self._skip_set is not None else None,
            "skip_set_sample": sorted(list(self._skip_set))[:10] if self._skip_set is not None else None,
            "random_candidate_skips": self._random_candidate_count,
            "unique_candidate_indices_seen": len(self._candidate_indices_seen),
            "accepted_random_skips": self._accepted_random_skips,
            "vetoed_skips": self._vetoed_skips,
            "veto_rate_vs_candidates": (
                self._vetoed_skips / max(1, self._random_candidate_count)
            ),
            "deferred_pool_now": len(self._deferred_pool),
            "deferred_pool_peak": self._deferred_pool_peak,
            "repaid_skips": self._repaid_skips,
            "forced_tail_skips": self._forced_tail,
            "danger_veto_counts_by_type": dict(self._danger_veto_counts_by_type),
            "phase_skip_distribution": self._phase_skips if self._phase_skips else None,
            "warmup_veto_count": self._warmup_veto,
            "quota_exhausted_count": self._quota_exhausted,
            "max_consecutive_veto_count": self._max_consec_veto,
            "full_backward_steps": full_backward_steps,
            "compute_normalized_budget": (
                full_backward_steps / self._quota_total_steps
                if self._quota_total_steps else None
            ),
            "invariant_skip_accounting_ok": (
                self._skip_decisions
                == self._accepted_random_skips + self._repaid_skips + self._forced_tail
            ),
            "invariant_candidate_count_ok": (
                self._candidate_set_size is None
                or self._random_candidate_count <= self._candidate_set_size
            ),
            "quota_total_steps": self._quota_total_steps,
            "quota_size": self._quota_size,
            "decisions_seen": self._decisions_seen,
            "skip_decisions": self._skip_decisions,
            "random_safe_skip_count": self._random_safe_skip,
            "deferred_count": self._deferred_count,
            "risk_gamma": self.risk_gamma,
        }

