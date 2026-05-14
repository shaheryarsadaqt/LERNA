"""Shared LERFeedCallback for Phase 1.2 + Phase 1.3.

[FIX #5] Previously this lived in scripts/run_phase1_2_simple_baselines.py and
was imported into the Phase 1.3 runner — that violates separation of concerns
and risks import side effects. It now lives here.

Responsibilities:
  * on_pre_optimizer_step: capture gradients into LERTracker (live grads only).
  * on_evaluate: push (loss, accuracy, logits) into LERTracker.
  * on_log: feed train-loss history into a LERNAPolicy (if attached) so its
    SafetyHorizon can estimate the PL constant.

[IMP-5] If logits are unavailable during on_evaluate, we SKIP the LER update
and log a warning. We do NOT feed random dummy logits. Random logits would
corrupt LERTracker's entropy computation and poison downstream skip decisions.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from transformers import TrainerCallback

logger = logging.getLogger(__name__)

# [IMP-5] Track whether we've already warned about missing logits,
# to avoid spamming the log on every eval step.
_MISSING_LOGITS_WARNED = False


class LERFeedCallback(TrainerCallback):
    def __init__(self, ler_tracker, trainer_ref=None, policy_ref=None):
        self.ler_tracker = ler_tracker
        self._trainer_ref = trainer_ref
        self._policy_ref = policy_ref  # optional LERNAPolicy for loss feed
        self._update_count = 0
        self._skipped_updates = 0  # [IMP-5] count of skipped-due-to-missing-logits
        self._model = None

    # Allow late wiring after the trainer object is built.
    def attach(self, trainer=None, policy=None):
        if trainer is not None:
            self._trainer_ref = trainer
            self._model = getattr(trainer, "model", None)
        if policy is not None:
            self._policy_ref = policy

    def on_train_begin(self, args, state, control, **kwargs):
        if "model" in kwargs and self._model is None:
            self._model = kwargs["model"]
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model", self._model)
        if model is not None and hasattr(self.ler_tracker, "capture_step_gradients"):
            self.ler_tracker.capture_step_gradients(model)
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        # [FIX #6] feed train loss into LERNAPolicy for PL/SafetyHorizon (mirrors
        # LERNASwitchingCallback.on_log) — important for ablations that keep
        # safety horizon enabled.
        if logs and "loss" in logs and self._policy_ref is not None \
           and hasattr(self._policy_ref, "record_loss"):
            self._policy_ref.record_loss(logs["loss"])
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        trainer = self._trainer_ref
        if trainer is None:
            return control
        eval_loss = metrics.get("eval_loss", 0.0)
        accuracy = metrics.get("eval_accuracy",
                    metrics.get("eval_matthews_correlation",
                    metrics.get("eval_pearson",
                    metrics.get("eval_pearsonr", None))))
        logits = getattr(trainer, "_last_real_logits", None)

        # [IMP-5] Do NOT feed random dummy logits into LERTracker. If logits
        # are missing, skip this update entirely. Random logits corrupt entropy
        # computation and poison skip decisions.
        if logits is None:
            global _MISSING_LOGITS_WARNED
            if not _MISSING_LOGITS_WARNED:
                logger.warning(
                    "LERFeedCallback: logits unavailable at eval step %d; "
                    "skipping LER update. This may happen if compute_loss "
                    "does not capture logits or if no eval batch ran. "
                    "LER-dependent policies will use stale data until logits "
                    "become available. (This warning is shown once.)",
                    state.global_step,
                )
                _MISSING_LOGITS_WARNED = True
            self._skipped_updates += 1
            return control

        model = kwargs.get("model", self._model)
        try:
            self.ler_tracker.update(
                loss=eval_loss, logits=logits, accuracy=accuracy,
                model=model, gradients=None,
            )
            self._update_count += 1
        except Exception as exc:
            logger.warning(f"LERFeedCallback: ler_tracker.update() failed: {exc!r}")
        return control
