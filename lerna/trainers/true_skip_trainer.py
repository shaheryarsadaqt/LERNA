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
        self._pre_clip_grad_norm: Optional[float] = None  # written by external callback

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

    # ----------------------------------------------------- core training
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
        if self.capture_logits:
            if hasattr(outputs, "logits"):
                self._last_real_logits = outputs.logits.detach()
            elif isinstance(outputs, dict) and "logits" in outputs:
                self._last_real_logits = outputs["logits"].detach()
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        # [CRIT-1] Always clear both flags at start of each step (safety net).
        self._skip_optimizer_step = False
        self._skip_scheduler_step = False

        model.train()
        inputs = self._prepare_inputs(inputs)
        self.instr.batches_seen += 1

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

        # Forward (always)
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        self.instr.forward_calls += 1
        if self.args.n_gpu > 1:
            loss = loss.mean()

        if should_skip:
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
    """apply_momentum=True   -> full LERNA (extrapolate via exp_avg / momentum_buffer)
       apply_momentum=False  -> 'no_momentum' ablation: pure weight freeze."""

    def __init__(self, *args, apply_momentum: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.apply_momentum = apply_momentum

    def on_skipped_backward_step(self, loss, model, inputs):
        if not self.apply_momentum:
            return
        opt = self.optimizer
        if opt is None:
            return
        with torch.no_grad():
            for group in opt.param_groups:
                lr = group.get("lr", self.args.learning_rate)
                for param in group["params"]:
                    if not param.requires_grad or param not in opt.state:
                        continue
                    p_state = opt.state[param]
                    if "momentum_buffer" in p_state and p_state["momentum_buffer"] is not None:
                        param.data.add_(p_state["momentum_buffer"], alpha=-lr)
                    elif "exp_avg" in p_state:
                        exp_avg = p_state["exp_avg"]
                        step = p_state.get("step", 1)
                        step_val = step.item() if hasattr(step, "item") else int(step)
                        beta1 = group.get("betas", (0.9, 0.999))[0]
                        bc = 1 - beta1 ** max(step_val, 1)
                        param.data.add_(exp_avg / bc if bc > 0 else exp_avg, alpha=-lr)
