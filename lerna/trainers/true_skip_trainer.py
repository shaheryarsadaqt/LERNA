"""REVISED — lerna/trainers/true_skip_trainer.py

Changes from original v1:

    [CRIT-1] Two separate flags _skip_optimizer_step / _skip_scheduler_step instead of single _skip_this_step
    [CRIT-2] Loss divided before backward for gradient accumulation
    [IMP-4] Renamed real_optimizer_steps  optimizer_step_attempts with documentation
    [IMP-7] Added global_step semantics documentation

"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Protocol

import torch
from transformers import Trainer, TrainerCallback

logger = logging.getLogger(__name__)


class _OptimizerStepWrapper:
    def __init__(self, trainer: "TrueBackwardSkippingTrainer"):
        self._trainer = trainer

    def __call__(self, *args, **kwargs):
        if self._trainer._skip_optimizer_step:
            self._trainer._skip_optimizer_step = False
            if self._trainer.instr.precision_mode == "fp16":
                scaler = self._trainer._scaler_ref
                if scaler is None:
                    accel = getattr(self._trainer, "accelerator", None)
                    if accel is not None:
                        scaler = getattr(accel, "scaler", None)
                # Clear per-optimizer states directly. This avoids calling
                # scaler.update() which asserts that inf checks were recorded
                # (not true when grads are None on skipped steps). Clearing
                # _per_optimizer_states leaves the scaler ready for the next
                # unscale_/step cycle (defaultdict will recreate READY entries).
                if scaler is not None and hasattr(scaler, "_per_optimizer_states"):
                    try:
                        scaler._per_optimizer_states.clear()
                    except Exception:
                        # Be conservative: don't let a scaler internals error
                        # crash the training loop; treat as best-effort.
                        pass
            return None
        self._trainer.instr.optimizer_step_attempts += 1
        return self._trainer._orig_opt_step(*args, **kwargs)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._trainer = None


class _GradScalerStepWrapper:
    def __init__(self, trainer: "TrueBackwardSkippingTrainer"):
        self._trainer = trainer

    def __call__(self, *args, **kwargs):
        self._trainer.instr.grad_scaler_step_calls += 1
        return self._trainer._orig_scaler_step(*args, **kwargs)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._trainer = None


class _SchedulerStepWrapper:
    def __init__(self, trainer: "TrueBackwardSkippingTrainer", policy: str):
        self._trainer = trainer
        self._policy = policy

    def __call__(self, *args, **kwargs):
        if (
            self._trainer._skip_scheduler_step
            and self._policy == SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP
        ):
            self._trainer._skip_scheduler_step = False
            return None
        self._trainer.instr.scheduler_step_calls += 1
        return self._trainer._orig_sched_step(*args, **kwargs)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        self._trainer = None


# ----------------------------------------------------------------------------
# Public protocol
# ----------------------------------------------------------------------------

class SkipPolicy(Protocol):
    name: str
    def should_skip(
        self,
        trainer: "TrueBackwardSkippingTrainer",
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
    ) -> bool: ...


class SchedulerStepPolicy:
    SKIP_ON_BACKWARD_SKIP = "skip_on_backward_skip"  # research choice A (default)
    ALWAYS_STEP = "always_step"                       # documented alternative


# Standard label set used in results.json (per Verdict Issue #9)
class ComputeSavingMechanism:
    BACKWARD_SKIPPING = "backward_skipping"
    REDUCED_TOTAL_STEPS = "reduced_total_steps"
    EARLY_STOPPING = "early_stopping"
    NONE = "none"


SKIP_UPDATE_MODE_FREEZE = "freeze"
SKIP_UPDATE_MODE_MOMENTUM = "momentum"
VALID_SKIP_UPDATE_MODES = (SKIP_UPDATE_MODE_FREEZE, SKIP_UPDATE_MODE_MOMENTUM)

_SKIP_UPDATE_MECHANISM_DESC = {
    SKIP_UPDATE_MODE_FREEZE: (
        "freeze: skipped-backward steps perform no parameter update and "
        "no optimizer-state update"
    ),
    SKIP_UPDATE_MODE_MOMENTUM: (
        "momentum: skipped-backward steps update parameters by extrapolating "
        "from stale optimizer state (SGD momentum_buffer, or bias-corrected "
        "AdamW exp_avg/exp_avg_sq); no gradient is computed and optimizer "
        "state itself is not refreshed"
    ),
}


def normalize_skip_update_mode(explicit_mode=None, legacy_use_momentum_extrap=None):
    """Single source of truth for the skipped-step parameter-update mode.

    Returns (effective_mode, used_legacy_compat).
    Raises ValueError on an invalid mode or a conflicting explicit/legacy pair.
    """
    if explicit_mode is not None and explicit_mode not in VALID_SKIP_UPDATE_MODES:
        raise ValueError(
            f"Invalid skip_update_mode={explicit_mode!r}; "
            f"expected one of {VALID_SKIP_UPDATE_MODES}."
        )
    if legacy_use_momentum_extrap is None:
        return (explicit_mode or SKIP_UPDATE_MODE_FREEZE, False)
    legacy_mode = (
        SKIP_UPDATE_MODE_MOMENTUM if legacy_use_momentum_extrap
        else SKIP_UPDATE_MODE_FREEZE
    )
    if explicit_mode is None:
        return (legacy_mode, True)
    if explicit_mode != legacy_mode:
        raise ValueError(
            f"Conflicting skip-update settings: explicit skip_update_mode="
            f"{explicit_mode!r} vs legacy use_momentum_extrap="
            f"{legacy_use_momentum_extrap} (implies {legacy_mode!r}). "
            f"Remove one of the two sources of truth."
        )
    return (explicit_mode, True)


# ----------------------------------------------------------------------------
# Instrumentation container
# ----------------------------------------------------------------------------

@dataclass
class SkipInstrumentation:
    forward_calls: int = 0
    backward_calls: int = 0
    optimizer_step_attempts: int = 0        # [IMP-4] renamed from real_optimizer_steps
    grad_scaler_step_calls: int = 0
    scheduler_step_calls: int = 0
    skipped_backward_steps: int = 0
    batches_seen: int = 0
    skipped_batches: int = 0
    precision_mode: str = "fp32"
    true_backward_skipping_enabled: bool = True
    scheduler_step_policy: str = SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP
    policy_name: str = ""
    compute_saving_mechanism: str = ComputeSavingMechanism.BACKWARD_SKIPPING
    grad_accumulation_steps: int = 1
    skip_update_mode: str = SKIP_UPDATE_MODE_FREEZE
    parameters_may_change_on_skipped_step: bool = False
    skip_update_mechanism: str = _SKIP_UPDATE_MECHANISM_DESC[SKIP_UPDATE_MODE_FREEZE]
    skip_update_mode_legacy_compat_used: bool = False

    def as_dict(self) -> Dict[str, Any]:
        d = self.__dict__.copy()
        d["skip_ratio_by_batch"] = (
            self.skipped_batches / max(self.batches_seen, 1)
        )
        denom = self.optimizer_step_attempts + self.skipped_backward_steps
        d["skip_ratio_by_optimizer_opportunity"] = (
            self.skipped_backward_steps / max(denom, 1)
        )
        # Documented invariants (booleans for quick audit)
        d["invariant_forward_eq_backward_plus_skipped"] = (
            self.forward_calls == self.backward_calls + self.skipped_backward_steps
        )
        d["invariant_opt_le_backward"] = (
            self.optimizer_step_attempts <= self.backward_calls
        )
        d["invariant_sched_le_opt"] = (
            self.scheduler_step_calls <= self.optimizer_step_attempts
        )
        # [IMP-4] Clarify semantics in output
        d["_note_optimizer_step_attempts"] = (
            "Counts non-skipped calls into optimizer.step(). Under AMP/fp16, "
            "GradScaler may still skip the real parameter update on overflow. "
            "These are attempts, not guaranteed updates."
        )
        # [IMP-7] Clarify global_step semantics
        d["_note_global_step"] = (
            "HF state.global_step counts training-loop iterations (batches), "
            "not real optimizer updates. Use optimizer_step_attempts for "
            "actual (attempted) parameter updates."
        )
        d["skip_update_mode"] = self.skip_update_mode
        d["parameters_may_change_on_skipped_step"] = (
            self.parameters_may_change_on_skipped_step
        )
        d["skip_update_mechanism"] = self.skip_update_mechanism
        d["skip_update_mode_legacy_compat_used"] = (
            self.skip_update_mode_legacy_compat_used
        )
        return d


# ----------------------------------------------------------------------------
# Internal callback for install/uninstall of wrappers (after optimizer exists)
# ----------------------------------------------------------------------------

class _SkipWrapperInstaller(TrainerCallback):
    def __init__(self, trainer: "TrueBackwardSkippingTrainer"):
        self._trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        # Set precision tag once optimizer/scaler exist
        if args.fp16:
            self._trainer.instr.precision_mode = "fp16"
        elif args.bf16:
            self._trainer.instr.precision_mode = "bf16"
        else:
            self._trainer.instr.precision_mode = "fp32"
        self._trainer.instr.grad_accumulation_steps = args.gradient_accumulation_steps
        self._trainer._install_wrappers()
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._trainer._uninstall_wrappers()
        self._trainer._dump_instrumentation()
        return control


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------

class TrueBackwardSkippingTrainer(Trainer):
    """HuggingFace Trainer subclass with AMP-safe true backward skipping.

    Public hook for subclasses:
        on_skipped_backward_step(loss, model, inputs) -> None
            Called after forward on a skipped step. Default: no-op (true freeze).
    """

    def __init__(
        self,
        *args,
        skip_policy: Optional[SkipPolicy] = None,
        scheduler_step_policy: str = SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP,
        instrumentation_path: Optional[str] = None,
        capture_logits: bool = True,
        compute_saving_mechanism: str = ComputeSavingMechanism.BACKWARD_SKIPPING,
        allow_grad_accumulation_with_skipping: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        from .policies import AlwaysFalsePolicy  # lazy
        self.skip_policy: SkipPolicy = skip_policy or AlwaysFalsePolicy()
        self.scheduler_step_policy = scheduler_step_policy
        self.instrumentation_path = instrumentation_path
        self.capture_logits = capture_logits

        # [FIX #3] enforce grad_accum == 1 for true-skipping experiments
        ga = int(getattr(self.args, "gradient_accumulation_steps", 1))
        non_trivial_policy = getattr(self.skip_policy, "name", "") != "always_false"
        if non_trivial_policy and ga != 1 and not allow_grad_accumulation_with_skipping:
            raise ValueError(
                f"TrueBackwardSkippingTrainer: gradient_accumulation_steps={ga} is "
                "unsupported for true-skipping experiments because mixed accumulation "
                "windows make optimizer-step semantics ambiguous. Set "
                "gradient_accumulation_steps=1 or pass "
                "allow_grad_accumulation_with_skipping=True at your own risk."
            )

        self.instr = SkipInstrumentation(
            scheduler_step_policy=scheduler_step_policy,
            policy_name=getattr(self.skip_policy, "name", type(self.skip_policy).__name__),
            compute_saving_mechanism=compute_saving_mechanism,
            grad_accumulation_steps=ga,
        )

        # [CRIT-1] Two independent flags for optimizer and scheduler wrappers.
        # Each wrapper consumes its own flag so they don't interfere.
        self._skip_optimizer_step: bool = False
        self._skip_scheduler_step: bool = False

        # Diagnostics
        self._last_real_logits: Optional[torch.Tensor] = None
        self._last_logits: Optional[torch.Tensor] = None
        self.last_logits: Optional[torch.Tensor] = None
        self._pre_clip_grad_norm: Optional[float] = None  # written by external callback
        self._last_grad_scale: float = 1.0

        # Wrapper bookkeeping
        self._orig_opt_step = None
        self._orig_sched_step = None
        self._orig_scaler_step = None
        self._scaler_ref = None
        self._wrappers_installed = False

        self.add_callback(_SkipWrapperInstaller(self))

    # ----------------------------------------------------- subclass hook
    def on_skipped_backward_step(self, loss, model, inputs) -> None:
        return None

    # ----------------------------------------------------- online LER update (FIX P0-3)
    def _online_ler_update(self, loss_value, logits, model, every: int = 1):
        """Dense, per-step LER feed using the TRAIN batch already computed.
        Cheap: reuses logits/loss; param-velocity uses tracker's own snapshot."""
        trk = getattr(self, "ler_tracker", None) or getattr(self, "_ler_tracker", None)
        if trk is None or logits is None:
            return
        if self.state.global_step % every != 0:
            return
        try:
            trk.update(loss=float(loss_value), logits=logits, accuracy=None, model=model)
        except Exception as exc:
            logger.debug(f"[TrueSkip] online LER update failed: {exc!r}")

    # ----------------------------------------------------- core training
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        if self.capture_logits:
            logits = getattr(outputs, "logits", None)
            if logits is None and isinstance(outputs, dict):
                logits = outputs.get("logits")
            logits = logits.detach() if logits is not None else None
            self._last_real_logits = logits
            self._last_logits = logits
            self.last_logits = logits
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # [FIX P0-3] Forward ALWAYS happens before skip decision (R3).
        # [CRIT-1] Always clear both flags at start of each step (safety net).
        self._skip_optimizer_step = False
        self._skip_scheduler_step = False

        model.train()
        inputs = self._prepare_inputs(inputs)
        self.instr.batches_seen += 1

        # [FIX P0-3] Forward ALWAYS happens first.
        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)

        logits = getattr(outputs, "logits", None)
        if logits is None and isinstance(outputs, dict):
            logits = outputs.get("logits")
        logits = logits.detach() if logits is not None else None
        self._last_real_logits = logits
        self._last_logits = logits
        self.last_logits = logits

        self.instr.forward_calls += 1
        if self.args.n_gpu > 1:
            loss = loss.mean()

        # [FIX P0-3] Feed current-step diagnostics ONLINE before deciding (R2/R3).
        self._online_ler_update(loss.detach(), self._last_real_logits, model)

        # [FIX P0-3] Decide AFTER the current forward.
        try:
            should_skip = bool(self.skip_policy.should_skip(self, model, inputs))
        except Exception as exc:
            logger.warning(f"[TrueSkip] policy.should_skip raised {exc!r}; not skipping")
            should_skip = False

        if should_skip:
            # [CRIT-1] Set BOTH flags independently.
            self._skip_optimizer_step = True
            self._skip_scheduler_step = True
            self.instr.skipped_batches += 1
            # AMP rule: no backward => no scaler inf checks => optimizer.step
            # wrapper will no-op (Verdict Issue #1, primary safety).
            self.instr.skipped_backward_steps += 1
            self.on_skipped_backward_step(loss=loss.detach(), model=model, inputs=inputs)
            try:
                self.optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            return loss.detach() / max(self.args.gradient_accumulation_steps, 1)

        # [CRIT-2] Normal step: divide loss BEFORE backward for gradient
        # accumulation (matching HF Trainer behavior). Without this, gradients
        # are scaled wrong when gradient_accumulation_steps > 1.
        ga = self.args.gradient_accumulation_steps
        if ga > 1:
            loss = loss / ga

        if getattr(self, "use_apex", False):
            from apex import amp  # type: ignore
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)
        self.instr.backward_calls += 1

        # --- SINGLE SOURCE OF TRUTH: pre-clip grad norm (before HF clips) ---
        with torch.no_grad():
            squared_norm = 0.0
            for p in model.parameters():
                if p.requires_grad and p.grad is not None:
                    pn = float(p.grad.detach().float().norm().item())
                    squared_norm += pn ** 2
            grad_norm = squared_norm ** 0.5

        scaler = getattr(
            getattr(self, "accelerator", None),
            "scaler",
            None,
        )
        grad_scale = 1.0
        if scaler is not None:
            candidate_scale = float(scaler.get_scale())
            if math.isfinite(candidate_scale) and candidate_scale > 0:
                grad_scale = candidate_scale

        self._pre_clip_grad_norm = grad_norm / grad_scale
        self._last_grad_scale = grad_scale
        pol = self.skip_policy
        if hasattr(pol, "record_grad_norm"):
            pol.record_grad_norm(self._pre_clip_grad_norm)

        # Return the un-scaled detached loss for logging consistency.
        return loss.detach()

    # --------------------------------------------------- wrapper install/restore
    def _install_wrappers(self):
        if self._wrappers_installed:
            return
        trainer = self

        # ---- optimizer.step (PRIMARY safety; consumes _skip_optimizer_step) ----
        if self.optimizer is not None:
            self._orig_opt_step = self.optimizer.step
            self.optimizer.step = _OptimizerStepWrapper(self)  # type: ignore[assignment]

        # ---- GradScaler.step (INSTRUMENTATION ONLY per Verdict Issue #1) ----
        scaler = None
        accel = getattr(self, "accelerator", None)
        if accel is not None:
            scaler = getattr(accel, "scaler", None)
        if scaler is None:
            scaler = getattr(self, "scaler", None)
        if scaler is not None and hasattr(scaler, "step"):
            self._scaler_ref = scaler
            self._orig_scaler_step = scaler.step
            scaler.step = _GradScalerStepWrapper(self)  # type: ignore[assignment]

        # ---- lr_scheduler.step (consumes _skip_scheduler_step) [CRIT-1] ----
        if self.lr_scheduler is not None:
            self._orig_sched_step = self.lr_scheduler.step
            self.lr_scheduler.step = _SchedulerStepWrapper(self, self.scheduler_step_policy)  # type: ignore[assignment]

        self._wrappers_installed = True
        logger.info(
            f"[TrueSkip] Wrappers installed (policy={self.instr.policy_name}, "
            f"scheduler_step_policy={self.scheduler_step_policy}, "
            f"precision={self.instr.precision_mode}, "
            f"compute_saving_mechanism={self.instr.compute_saving_mechanism})"
        )

    def _uninstall_wrappers(self):
        if not self._wrappers_installed:
            return
        if self._orig_opt_step is not None and self.optimizer is not None:
            self.optimizer.step = self._orig_opt_step  # type: ignore[assignment]
        if self._orig_sched_step is not None and self.lr_scheduler is not None:
            self.lr_scheduler.step = self._orig_sched_step  # type: ignore[assignment]
        if self._orig_scaler_step is not None and self._scaler_ref is not None:
            self._scaler_ref.step = self._orig_scaler_step  # type: ignore[assignment]
        self._wrappers_installed = False
        self._orig_opt_step = None
        self._orig_sched_step = None
        self._orig_scaler_step = None
        self._scaler_ref = None
        logger.info("[TrueSkip] Wrappers uninstalled.")

    def _save_optimizer_and_scheduler(self, output_dir: str):
        scheduler = self.lr_scheduler
        saved_step = None
        step_present = False
        if scheduler is not None and hasattr(scheduler, "__dict__") and "step" in scheduler.__dict__:
            saved_step = scheduler.__dict__.pop("step")
            step_present = True
        try:
            return super()._save_optimizer_and_scheduler(output_dir)
        finally:
            if step_present and scheduler is not None:
                scheduler.__dict__["step"] = saved_step

    # ----------------------------------------------------- instrumentation
    def _dump_instrumentation(self):
        if not self.instrumentation_path:
            self.instrumentation_path = os.path.join(
                self.args.output_dir, "instrumentation.json"
            )
        try:
            os.makedirs(os.path.dirname(self.instrumentation_path), exist_ok=True)
            with open(self.instrumentation_path, "w") as f:
                json.dump(self.instr.as_dict(), f, indent=2)
            logger.info(f"[TrueSkip] Instrumentation saved: {self.instrumentation_path}")
        except Exception as exc:
            logger.warning(f"[TrueSkip] Could not save instrumentation: {exc!r}")

    def get_instrumentation(self) -> Dict[str, Any]:
        return self.instr.as_dict()


# ---------------------------------------------------------------------------
# Optional subclass for LERNA momentum extrapolation (Phase 1.3)
# ---------------------------------------------------------------------------

class LERNAMomentumTrainer(TrueBackwardSkippingTrainer):
    """Momentum-aware backward-skipping trainer with explicit mode control.

    Modes (via ``skip_update_mode``):
        "freeze"   -> pure weight freeze on skipped steps (default).
        "momentum" -> extrapolate via stale AdamW/SGD moments on skipped steps.

    Legacy ``apply_momentum`` (bool):
        Accepted for backwards compatibility. Derived from ``skip_update_mode``
        as ``apply_momentum = (skip_update_mode == "momentum")``. Not a source
        of truth; ``skip_update_mode`` remains the only behavioral gate.
    """

    def __init__(
        self,
        *args,
        skip_update_mode: Optional[str] = None,
        apply_momentum: Optional[bool] = None,   # legacy compatibility only
        **kwargs,
    ):
        # Normalize BEFORE super().__init__ so conflicts fail fast.
        effective_mode, used_legacy = normalize_skip_update_mode(
            explicit_mode=skip_update_mode,
            legacy_use_momentum_extrap=apply_momentum,
        )
        super().__init__(*args, **kwargs)
        self.skip_update_mode = effective_mode
        self.skip_update_mode_legacy_compat_used = used_legacy
        # Derived compatibility boolean for old readers. Behavior is
        # gated only on ``skip_update_mode``.
        self.apply_momentum = (
            effective_mode == SKIP_UPDATE_MODE_MOMENTUM
        )
        if used_legacy:
            logger.warning(
                "LERNAMomentumTrainer: legacy 'apply_momentum=%s' normalized "
                "to skip_update_mode=%r. Migrate to skip_update_mode.",
                apply_momentum, effective_mode,
            )
        self.instr.skip_update_mode = effective_mode
        self.instr.parameters_may_change_on_skipped_step = (
            effective_mode == SKIP_UPDATE_MODE_MOMENTUM
        )
        self.instr.skip_update_mechanism = _SKIP_UPDATE_MECHANISM_DESC[effective_mode]
        self.instr.skip_update_mode_legacy_compat_used = used_legacy

    def on_skipped_backward_step(self, loss, model, inputs):
        if self.skip_update_mode != SKIP_UPDATE_MODE_MOMENTUM:
            return  # freeze: no parameter update, no optimizer-state update
        opt = self.optimizer
        if opt is None:
            return
        with torch.no_grad():
            for group in opt.param_groups:
                lr = group.get("lr", self.args.learning_rate)
                beta1, beta2 = group.get("betas", (0.9, 0.999))
                eps = group.get("eps", 1e-8)
                for param in group["params"]:
                    if not param.requires_grad or param not in opt.state:
                        continue
                    st = opt.state[param]
                    # SGD-with-momentum path (unchanged)
                    buf = st.get("momentum_buffer")
                    if buf is not None:
                        param.data.add_(buf, alpha=-lr)
                        continue
                    # AdamW path: reuse the STALE (un-refreshed) moments. On a
                    # skipped step v is not updated, so reusing v_hat is the
                    # consistent inertial extrapolation. Mirrors metrics.py:570.
                    m = st.get("exp_avg")
                    v = st.get("exp_avg_sq")
                    if m is None or v is None:
                        continue
                    step = st.get("step", 1)
                    t = float(step.item() if torch.is_tensor(step) else step)
                    t = max(t, 1.0)
                    m_hat = m / (1.0 - beta1 ** t)
                    v_hat = v / (1.0 - beta2 ** t)
                    wd = group.get("weight_decay", 0.0)
                    if wd:
                        param.data.add_(param.data, alpha=-lr * wd)
                    update = m_hat / (v_hat.sqrt() + eps)   # full AdamW direction
                    param.data.add_(update, alpha=-lr)
