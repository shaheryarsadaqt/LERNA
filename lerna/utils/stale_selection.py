"""Round 12 (6 Nov 2026): selection from stored per-example scores.

Two policies for TrueBackwardSkippingTrainer, used by the development round of
the reorganized report's Section 7.3. Both need every training example to carry
a stable integer id, ``lerna_uid`` (its row position after the QNLI probe
exclusions); the trainer pops it from the batch before the model sees it and
exposes it as ``trainer._lerna_batch_uids``.

StaleS1GatePolicy (M1, arm ``stale_s1_p4``)
    The LERNA gate (phase_exact_v1, P phases, cap K, history H, anneal) with the
    S1 score taken from each example's residual ||p_i - y_i|| stored at its last
    forward pass. When every example of a scheduled batch has a stored residual,
    the batch is scored and decided BEFORE any forward pass, so a skipped batch
    costs no compute at all. Otherwise (the first visit, normally the first
    epoch) the batch gets its forward pass first and the fresh S1 decides,
    exactly as in LERNA. Every forward pass refreshes the stored residuals of
    the examples it saw. Quotas, cap and feasibility forcing are unchanged, so
    the run still skips exactly Q = round(R * T) batches.

InfoBatchPolicy (arm ``infobatch``)
    InfoBatch (Qin et al., ICLR 2024) adapted to a fixed number of scheduled
    batches, so the learning-rate schedule and the step count stay those of
    full fine-tuning. At the start of each epoch the threshold is the mean of
    the stored per-example losses (losses from each example's last forward).
    Within a batch, every example whose stored loss is below the threshold is
    dropped with probability r (0.5); the kept low-loss examples get weight
    1 / (1 - r), and the loss is normalized by the scheduled batch size, which
    keeps the batch gradient unbiased (Horvitz-Thompson). Examples without a
    stored loss are always kept, so nothing is pruned in the first epoch, and
    nothing is pruned from step floor(delta * T) on (delta = 0.875: all data in
    the last 12.5% of steps, InfoBatch's annealing). Dropped examples get no
    forward and no backward pass; no step is ever skipped.
"""

from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import torch

from lerna.utils.forward_gate import ScoreQuantileGate, forward_gain_score


class StoredScoreError(RuntimeError):
    """Raised when a batch lacks the ids or labels the stored scores need."""


def _batch_uids(trainer) -> List[int]:
    uids = getattr(trainer, "_lerna_batch_uids", None)
    if uids is None:
        raise StoredScoreError(
            "batch has no lerna_uid ids; the stale-score arms need the training "
            "dataset's lerna_uid column"
        )
    return [int(u) for u in uids]


def per_example_residuals(logits, labels) -> torch.Tensor:
    """||softmax(logits)_i - onehot(labels)_i||_2 per example, as float64 on CPU.

    Same arithmetic as forward_gain_score (float32 softmax after the max shift),
    so the batch mean of these values is the S1 score of that batch.
    """
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise StoredScoreError("residuals need tensor logits and labels")
    with torch.no_grad():
        log = logits.detach().float()
        log = log - log.max(dim=-1, keepdim=True).values
        p = log.exp()
        p = p / p.sum(dim=-1, keepdim=True)
        onehot = torch.zeros_like(p).scatter_(1, labels.view(-1, 1).long(), 1.0)
        r = (p - onehot).pow(2).sum(dim=-1).sqrt()
    out = r.to("cpu", torch.float64)
    if not bool(torch.isfinite(out).all()):
        raise StoredScoreError("non-finite per-example residual")
    return out


def per_example_losses(logits, labels) -> torch.Tensor:
    """Per-example cross-entropy, as float64 on CPU."""
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise StoredScoreError("losses need tensor logits and labels")
    with torch.no_grad():
        ce = torch.nn.functional.cross_entropy(
            logits.detach().float(), labels.view(-1).long(), reduction="none"
        )
    out = ce.to("cpu", torch.float64)
    if not bool(torch.isfinite(out).all()):
        raise StoredScoreError("non-finite per-example loss")
    return out


class StaleS1GatePolicy:
    """M1: the phase-exact LERNA gate scored from stored per-example residuals."""

    name = "stale_s1_gate"

    def __init__(
        self,
        *,
        target_skip_rate: float,
        total_steps: int,
        num_examples: int,
        min_step: int = 392,
        max_consecutive_skips: int = 4,
        n_phases: int = 4,
        score_history_window: int = 64,
        anneal_fraction: float = 0.88,
        policy_seed: int = 0,
        controller_version: str = "phase_exact_v1",
    ):
        if controller_version != "phase_exact_v1":
            raise ValueError("StaleS1GatePolicy requires controller_version='phase_exact_v1'")
        if isinstance(num_examples, bool) or not isinstance(num_examples, int) or num_examples < 1:
            raise ValueError(f"num_examples must be a positive int, got {num_examples!r}")
        self.controller_version = controller_version
        self.score_source_kind = "s1_stale"
        self.num_examples = int(num_examples)
        self.total_steps = int(total_steps)
        self._residual = torch.full((self.num_examples,), float("nan"), dtype=torch.float64)
        self._pending_score: Optional[float] = None
        self.stale_decisions = 0
        self.fresh_decisions = 0
        self.forward_skips = 0
        self.stale_retains = 0
        self._gate = ScoreQuantileGate(
            rate=float(target_skip_rate),
            total_steps=int(total_steps),
            min_step=int(min_step),
            max_consecutive_skips=int(max_consecutive_skips),
            n_phases=int(n_phases),
            score_history_window=int(score_history_window),
            anneal_fraction=float(anneal_fraction),
            seed=int(policy_seed) + 1,
            score_source=self._pending_source,
            controller_version=controller_version,
        )

    # The gate pulls the score through this source; the policy sets it first.
    def _pending_source(self, decision_index, model, inputs) -> float:
        if self._pending_score is None:
            raise StoredScoreError(f"no score prepared for decision {decision_index}")
        return float(self._pending_score)

    @staticmethod
    def _decision_index(trainer) -> int:
        seen = int(trainer.instr.batches_seen)
        if seen < 1:
            raise StoredScoreError("decision requested before any batch was seen")
        return seen - 1

    def _check_uids(self, uids: Sequence[int]) -> None:
        for u in uids:
            if u < 0 or u >= self.num_examples:
                raise StoredScoreError(f"lerna_uid {u} outside [0, {self.num_examples})")

    def stored_batch_score(self, uids: Sequence[int]) -> Optional[float]:
        """Mean stored residual of the batch, or None if any example has none."""
        self._check_uids(uids)
        vals = self._residual[list(uids)]
        if bool(torch.isnan(vals).any()):
            return None
        return float(vals.mean().item())

    def pre_forward_decision(self, trainer, model, inputs) -> Optional[bool]:
        """Decide before the forward pass when every example has a stored score.

        Returns True (skip; no forward pass), False (retain; the forward and
        backward passes follow without a second decision), or None (no stored
        score for some example: run the forward pass and decide from it).
        """
        uids = _batch_uids(trainer)
        score = self.stored_batch_score(uids)
        if score is None:
            return None
        self._pending_score = score
        try:
            skip = bool(self._gate.decide(self._decision_index(trainer), trainer, inputs))
        finally:
            self._pending_score = None
        self.stale_decisions += 1
        if skip:
            self.forward_skips += 1
        else:
            self.stale_retains += 1
        return skip

    def observe_forward(self, trainer, inputs, logits) -> None:
        """Store the residuals of every example the forward pass just scored."""
        uids = _batch_uids(trainer)
        self._check_uids(uids)
        labels = inputs.get("labels") if hasattr(inputs, "get") else None
        res = per_example_residuals(logits, labels)
        if res.shape[0] != len(uids):
            raise StoredScoreError(
                f"{res.shape[0]} residuals for {len(uids)} ids"
            )
        self._residual[list(uids)] = res

    def should_skip(self, trainer, model, inputs) -> bool:
        """Decision after a forward pass (no stored score for the whole batch)."""
        labels = inputs.get("labels") if hasattr(inputs, "get") else None
        score = forward_gain_score(getattr(trainer, "_last_real_logits", None), labels)
        self._pending_score = score
        try:
            skip = bool(self._gate.decide(self._decision_index(trainer), trainer, inputs))
        finally:
            self._pending_score = None
        self.fresh_decisions += 1
        return skip

    def get_diagnostics(self) -> dict:
        d = self._gate.diagnostics()
        d["score_source"] = self.score_source_kind
        d["stale_decisions"] = self.stale_decisions
        d["fresh_decisions"] = self.fresh_decisions
        d["forward_skips"] = self.forward_skips
        d["stale_retains"] = self.stale_retains
        d["examples_with_stored_score"] = int((~torch.isnan(self._residual)).sum().item())
        d["num_examples"] = self.num_examples
        return d


class InfoBatchPolicy:
    """InfoBatch adapted to a fixed number of scheduled batches (no step is skipped)."""

    name = "infobatch_fixed_steps"

    def __init__(
        self,
        *,
        total_steps: int,
        num_examples: int,
        batches_per_epoch: int,
        prune_ratio: float = 0.5,
        anneal_fraction: float = 0.875,
        policy_seed: int = 0,
    ):
        for nm, v in (("total_steps", total_steps), ("num_examples", num_examples),
                      ("batches_per_epoch", batches_per_epoch)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{nm} must be a positive int, got {v!r}")
        for nm, v in (("prune_ratio", prune_ratio), ("anneal_fraction", anneal_fraction)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{nm} must be a finite number, got {v!r}")
        if not 0.0 <= float(prune_ratio) < 1.0:
            raise ValueError(f"prune_ratio must be in [0, 1), got {prune_ratio!r}")
        if not 0.0 <= float(anneal_fraction) <= 1.0:
            raise ValueError(f"anneal_fraction must be in [0, 1], got {anneal_fraction!r}")
        self.total_steps = int(total_steps)
        self.num_examples = int(num_examples)
        self.batches_per_epoch = int(batches_per_epoch)
        self.prune_ratio = float(prune_ratio)
        self.anneal_fraction = float(anneal_fraction)
        self.stop_prune_step = int(math.floor(self.anneal_fraction * self.total_steps + 1e-9))
        self.keep_weight = 1.0 / (1.0 - self.prune_ratio)
        self._loss = torch.full((self.num_examples,), float("nan"), dtype=torch.float64)
        self._rng = random.Random(int(policy_seed) + 2)
        self._threshold: Optional[float] = None
        self.thresholds: List[Optional[float]] = []
        self.pruned_examples = 0
        self.kept_low_loss_examples = 0
        self.batches_with_pruning = 0
        self.empty_batch_fallbacks = 0
        self.examples_seen = 0

    def _decision_index(self, trainer) -> int:
        seen = int(trainer.instr.batches_seen)
        if seen < 1:
            raise StoredScoreError("selection requested before any batch was seen")
        return seen - 1

    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        """Positions to keep in this batch and their loss weights."""
        t = self._decision_index(trainer)
        uids = _batch_uids(trainer)
        for u in uids:
            if u < 0 or u >= self.num_examples:
                raise StoredScoreError(f"lerna_uid {u} outside [0, {self.num_examples})")
        if t % self.batches_per_epoch == 0:
            known = self._loss[~torch.isnan(self._loss)]
            self._threshold = float(known.mean().item()) if known.numel() > 0 else None
            self.thresholds.append(self._threshold)
        n = len(uids)
        self.examples_seen += n
        if self._threshold is None or t >= self.stop_prune_step or self.prune_ratio == 0.0:
            return list(range(n)), [1.0] * n
        keep: List[int] = []
        weights: List[float] = []
        pruned = 0
        low_kept = 0
        for pos, u in enumerate(uids):
            s = float(self._loss[u].item())
            if math.isnan(s) or s >= self._threshold:
                keep.append(pos)
                weights.append(1.0)
            elif self._rng.random() < self.prune_ratio:
                pruned += 1
            else:
                keep.append(pos)
                weights.append(self.keep_weight)
                low_kept += 1
        if not keep:
            # Every example was low-loss and dropped (probability <= r^batch):
            # train the whole batch unweighted rather than an empty one.
            self.empty_batch_fallbacks += 1
            return list(range(n)), [1.0] * n
        self.pruned_examples += pruned
        self.kept_low_loss_examples += low_kept
        if pruned:
            self.batches_with_pruning += 1
        return keep, weights

    def observe_forward(self, trainer, inputs, logits) -> None:
        """Store the losses of the examples the forward pass just trained on."""
        uids = _batch_uids(trainer)
        labels = inputs.get("labels") if hasattr(inputs, "get") else None
        losses = per_example_losses(logits, labels)
        if losses.shape[0] != len(uids):
            raise StoredScoreError(f"{losses.shape[0]} losses for {len(uids)} ids")
        self._loss[list(uids)] = losses

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        return {
            "policy": self.name,
            "prune_ratio": self.prune_ratio,
            "anneal_fraction": self.anneal_fraction,
            "stop_prune_step": self.stop_prune_step,
            "keep_weight": self.keep_weight,
            "batches_per_epoch": self.batches_per_epoch,
            "thresholds": list(self.thresholds),
            "pruned_examples": self.pruned_examples,
            "kept_low_loss_examples": self.kept_low_loss_examples,
            "batches_with_pruning": self.batches_with_pruning,
            "empty_batch_fallbacks": self.empty_batch_fallbacks,
            "examples_seen": self.examples_seen,
            "examples_with_stored_loss": int((~torch.isnan(self._loss)).sum().item()),
            "num_examples": self.num_examples,
        }


__all__ = [
    "StaleS1GatePolicy",
    "InfoBatchPolicy",
    "StoredScoreError",
    "per_example_residuals",
    "per_example_losses",
]
