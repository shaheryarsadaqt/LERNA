"""Round 16 (7 Nov 2026): the compared methods of the final comparison, at LERNA's fixed step count.

Every policy here runs the scheduled batches of full fine-tuning: no step is skipped, so the step
count and the learning-rate schedule are those of full fine-tuning, as for InfoBatch (round 12) and
LERNA v2 (round 14). Before each forward pass a policy decides which of the batch's examples take
part in the update (DropBP instead decides which layers' backward passes run).

Cost. The batch log records the padded tokens of each training forward pass and update. Some
methods also run forward passes that train nothing: a scoring pass over the whole batch (Selective
Backprop, fresh-score importance sampling), a refresh of the hidden examples' losses (KAKURENBO) or
a monitoring pass over the excluded ones (IES). Each policy counts their padded tokens as
`extra_forward_padded_tokens`; they cost one unit per token, as any forward pass does. DropBP runs
part of each backward pass; it records, per step, how many attention and MLP blocks it dropped, and
`flops_per_token` gives the multiply-adds those blocks hold, so the backward share of every step can
be recomputed.

Weights. The subset methods (eps-greedy, UCB, OrderDP, Selective Backprop, IES) train on the plain
mean over their kept examples (each weighted n / K, the trainer dividing by the scheduled size n),
as the round-15 top-k arm does; this gives every step random keep's magnitude. Fresh-score
importance sampling keeps Horvitz-Thompson weights 1 / pi, as LERNA v2 does. KAKURENBO weights its
kept examples 1 / (1 - F), its learning-rate rule.

The methods
    RankedSubsetPolicy (eps-greedy, UCB; Raju, Daruwalla and Lipasti 2021. OrderDP; Jin et al.,
        ICLR 2026). LERNA v2's budget controller with equal scores sets how many examples each batch
        keeps (K, as random keep would); the method decides which. Scores are each example's
        cross-entropy at its last forward pass (an example never seen yet gets the mean over the
        examples seen so far).
        - eps-greedy: the round((1 - eps) K) highest exponential moving averages of the loss
          (alpha 0.8), plus the rest drawn uniformly from the remaining examples (eps 0.1).
        - UCB: the K highest EMA + c * var (c 1.0), var the exponentially weighted variance.
        - OrderDP: a uniform random candidate set of ceil(K / 0.6) examples (the paper's q/s = 0.6),
          then its K highest last losses.
    FreshImportancePolicy (Katharopoulos and Fleuret, ICML 2018). After the warmup, a forward pass
        without gradients scores the whole batch by the fresh residual ||p - y|| (the gradient-norm
        bound for cross-entropy), then LERNA v2's controller keeps example i with probability
        clip(s_i / mu, pi_min, 1) and weight 1 / pi. The scoring pass costs a third of the batch's
        full step, and the controller spends the rest of the budget on the kept examples. No
        learning-rate factor.
    SelectiveBackpropPolicy (Jiang et al. 2019). A scoring pass over the whole batch; example i is
        kept with probability CDF(loss_i)^beta, the CDF over the last 1024 losses. beta is fixed so
        the expected cost is the budget: the scoring pass costs 1/3, so the kept fraction is
        k = budget - 1/3 and beta = 1 / k - 1 (CDF values are uniform, so E[CDF^beta] = 1/(1+beta)).
        The selection runs from the first step, as in the paper.
    KakurenboPolicy (Nguyen et al., NeurIPS 2023). At the start of each epoch, the floor(F_e N)
        examples with the lowest stored loss are candidates; those the model got right with
        confidence >= tau (0.7) at their last forward pass are hidden for the epoch, the others
        moved back. F_e = 0.3 x alpha, alpha 1, 0.8, 0.6, 0.4 from 0, 30, 60, 80% of training. The
        kept examples weigh 1 / (1 - F*_e), F*_e the fraction actually hidden. A hidden example gets
        a forward pass without gradients at its scheduled visit, which refreshes its loss for the
        next epoch's choice (the paper refreshes at the epoch's end); not in the last epoch.
    IESPolicy (Yuan et al., ICLR 2025). Each example keeps its losses from its last three epochs; at
        its visit in epoch e it is mastered, and excluded from the update, when they come from
        epochs e-1, e-2, e-3 and |L_{e-1} - 2 L_{e-2} + L_{e-3}| < delta (1e-3). A mastered example
        still gets a forward pass without gradients, so its loss history continues and it comes back
        when the difference grows. Needs three earlier epochs, so it acts from the fourth epoch.
    DropBPPolicy (Woo et al., NeurIPS 2024). Every attention block and every MLP block of the
        encoder (with the norm before it) skips its backward pass at a step with probability p,
        independently, at every step from the first; the forward pass is unchanged. Uniform drop
        rates: the paper's sensitivity-based allocation is not reproduced.

At least one example is always kept (each policy counts the fallbacks).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lerna.utils.budget_selection import LernaBudgetPolicy
from lerna.utils.stale_selection import (
    StoredScoreError,
    _batch_uids,
    per_example_losses,
    per_example_residuals,
)


# --------------------------------------------------------------------------- helpers
def _decision_index(trainer) -> int:
    seen = int(trainer.instr.batches_seen)
    if seen < 1:
        raise StoredScoreError("selection requested before any batch was seen")
    return seen - 1


def _batch_shape(inputs) -> Tuple[int, int, List[int]]:
    """(rows, padded width, real length of each row) of a scheduled batch."""
    ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
    mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
    if not isinstance(ids, torch.Tensor) or ids.dim() != 2:
        raise StoredScoreError("selection needs a 2-D input_ids tensor")
    n, width = int(ids.shape[0]), int(ids.shape[1])
    if isinstance(mask, torch.Tensor) and mask.dim() == 2:
        lengths = [int(v) for v in mask.sum(dim=1).tolist()]
    else:
        lengths = [width] * n
    return n, width, lengths


def _labels(inputs) -> torch.Tensor:
    labels = inputs.get("labels") if hasattr(inputs, "get") else None
    if not isinstance(labels, torch.Tensor):
        raise StoredScoreError("the batch has no labels tensor")
    return labels


def _subset(inputs, positions: Sequence[int]) -> Tuple[dict, int]:
    """The rows `positions` of the batch with the unused padding trimmed, and their padded tokens."""
    ids = inputs["input_ids"]
    n = int(ids.shape[0])
    idx = torch.tensor([int(p) for p in positions], dtype=torch.long, device=ids.device)
    out = {}
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor) and v.dim() >= 1 and int(v.shape[0]) == n:
            out[k] = v.index_select(0, idx)
        else:
            out[k] = v
    mask = out.get("attention_mask")
    if isinstance(mask, torch.Tensor) and mask.dim() == 2 and mask.shape[1] > 0:
        longest = max(int(mask.sum(dim=1).max().item()), 1)
        width = int(mask.shape[1])
        for k in ("input_ids", "attention_mask", "token_type_ids"):
            v = out.get(k)
            if isinstance(v, torch.Tensor) and v.dim() == 2 and int(v.shape[1]) == width:
                out[k] = v[:, :longest]
    sub_ids = out["input_ids"]
    return out, int(sub_ids.shape[0]) * int(sub_ids.shape[1])


def _forward_logits(trainer, inputs) -> torch.Tensor:
    """Logits of a forward pass without gradients (the model is not changed)."""
    model = getattr(trainer, "model", None)
    if model is None:
        raise StoredScoreError("a scoring pass needs trainer.model")
    feed = {k: v for k, v in inputs.items() if k != "labels"}
    with torch.no_grad():
        out = model(**feed)
    logits = getattr(out, "logits", None)
    if logits is None and isinstance(out, dict):
        logits = out.get("logits")
    if not isinstance(logits, torch.Tensor):
        raise StoredScoreError("the scoring pass returned no logits")
    return logits.detach()


def _loss_correct_conf(logits: torch.Tensor, labels: torch.Tensor):
    """Per-example cross-entropy, correctness and confidence (max softmax), as numpy."""
    loss = per_example_losses(logits, labels).numpy()
    with torch.no_grad():
        p = torch.softmax(logits.detach().float(), dim=-1)
        conf, pred = p.max(dim=-1)
    correct = (pred.view(-1).cpu() == labels.view(-1).long().cpu()).numpy()
    return loss, correct, conf.to("cpu", torch.float64).numpy()


def _check_int(name: str, v, lo: int = 1) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < lo:
        raise ValueError(f"{name} must be an int >= {lo}, got {v!r}")
    return int(v)


def _check_float(name: str, v, lo: float, hi: float, lo_open: bool = False, hi_open: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
        raise ValueError(f"{name} must be a finite number, got {v!r}")
    x = float(v)
    if (x < lo or (lo_open and x == lo)) or (x > hi or (hi_open and x == hi)):
        raise ValueError(f"{name}={v!r} out of range")
    return x


# --------------------------------------------------------------------------- eps-greedy / UCB / OrderDP
class RankedSubsetPolicy(LernaBudgetPolicy):
    """eps-greedy, UCB and OrderDP: the controller sets K per batch, the method's score picks which."""

    name = "ranked_subset"
    SELECTIONS = ("egreedy", "ucb", "orderdp")

    def __init__(self, *, budget: float, total_steps: int, num_examples: int, batches_per_epoch: int,
                 warmup_steps: int, selection: str, pi_min: float = 0.1, window: int = 64,
                 chunks: int = 16, policy_seed: int = 0, epsilon: float = 0.1, ema_alpha: float = 0.8,
                 ucb_c: float = 1.0, orderdp_ratio: float = 0.6):
        super().__init__(budget=budget, total_steps=total_steps, num_examples=num_examples,
                         batches_per_epoch=batches_per_epoch, warmup_steps=warmup_steps,
                         use_scores=False, use_lr_weight=False, pi_min=pi_min, window=window,
                         chunks=chunks, policy_seed=policy_seed, selection=selection)
        self.epsilon = _check_float("epsilon", epsilon, 0.0, 1.0)
        self.ema_alpha = _check_float("ema_alpha", ema_alpha, 0.0, 1.0, lo_open=True)
        self.ucb_c = _check_float("ucb_c", ucb_c, 0.0, 1e6)
        self.orderdp_ratio = _check_float("orderdp_ratio", orderdp_ratio, 0.0, 1.0, lo_open=True)
        self._choice_rng = np.random.default_rng(np.random.SeedSequence([self.policy_seed, 16, 1]))
        self._last_loss = np.full(self.num_examples, np.nan, dtype=np.float64)
        self._ema = np.full(self.num_examples, np.nan, dtype=np.float64)
        self._var = np.zeros(self.num_examples, dtype=np.float64)
        self._sum = {"egreedy": 0.0, "ucb": 0.0, "orderdp": 0.0}
        self._count = 0
        self.unseen_scored = 0

    def _rank_values(self, uids: Sequence[int]) -> np.ndarray:
        u = np.asarray(uids, dtype=np.int64)
        if self.selection == "orderdp":
            v = self._last_loss[u]
        elif self.selection == "egreedy":
            v = self._ema[u]
        else:
            v = self._ema[u] + self.ucb_c * self._var[u]
        unseen = np.isnan(v)
        if unseen.any():
            prior = self._sum[self.selection] / self._count if self._count else 0.0
            v = np.where(unseen, prior, v)
            self.unseen_scored += int(unseen.sum())
        return v

    def _choose(self, t, pi, s, n, uids):
        k = max(1, min(n, int(round(float(pi.sum())))))
        if k >= n:
            return list(range(n)), [1.0] * n
        r = self._rank_values(uids)
        order = sorted(range(n), key=lambda i: (-float(r[i]), i))  # highest first, ties by position
        if self.selection == "egreedy":
            g = int(round((1.0 - self.epsilon) * k))
            keep = order[:g]
            rest = order[g:]
            extra = k - g
            if extra > 0:
                picks = self._choice_rng.choice(len(rest), size=extra, replace=False)
                keep = keep + [rest[int(j)] for j in picks]
        elif self.selection == "ucb":
            keep = order[:k]
        else:  # orderdp
            c = min(n, max(k, int(math.ceil(k / self.orderdp_ratio))))
            cand = sorted(int(j) for j in self._choice_rng.choice(n, size=c, replace=False))
            keep = sorted(cand, key=lambda i: (-float(r[i]), i))[:k]
        keep = sorted(int(i) for i in keep)
        return keep, [float(n) / float(k)] * k

    def observe_forward(self, trainer, inputs, logits) -> None:
        uids = _batch_uids(trainer)
        labels = _labels(inputs)
        super().observe_forward(trainer, inputs, logits)
        loss = per_example_losses(logits, labels).numpy()
        a = self.ema_alpha
        for u, x in zip(uids, loss.tolist()):
            old = {"egreedy": self._ema[u], "ucb": self._ema[u] + self.ucb_c * self._var[u],
                   "orderdp": self._last_loss[u]}
            if math.isnan(self._ema[u]):
                self._ema[u] = x
                self._var[u] = 0.0
                self._count += 1
            else:
                prev = self._ema[u]
                self._var[u] = (1.0 - a) * self._var[u] + a * (x - prev) ** 2
                self._ema[u] = a * x + (1.0 - a) * prev
            self._last_loss[u] = x
            new = {"egreedy": self._ema[u], "ucb": self._ema[u] + self.ucb_c * self._var[u],
                   "orderdp": self._last_loss[u]}
            for key in self._sum:
                self._sum[key] += new[key] - (0.0 if math.isnan(old[key]) else old[key])

    def get_diagnostics(self) -> dict:
        d = super().get_diagnostics()
        d.update({"policy": self.name, "method": self.selection, "epsilon": self.epsilon,
                  "ema_alpha": self.ema_alpha, "ucb_c": self.ucb_c, "orderdp_ratio": self.orderdp_ratio,
                  "unseen_scored": self.unseen_scored, "overhead_units": self.overhead_units,
                  "extra_forward_padded_tokens": 0, "extra_forward_examples": 0,
                  "examples_with_stored_loss": int((~np.isnan(self._ema)).sum())})
        return d


# --------------------------------------------------------------------------- fresh-score importance sampling
class FreshImportancePolicy(LernaBudgetPolicy):
    """Importance sampling by the fresh residual of a scoring pass, under LERNA v2's controller."""

    name = "fresh_importance_sampling"
    SELECTIONS = ("sample",)

    def __init__(self, *, budget: float, total_steps: int, num_examples: int, batches_per_epoch: int,
                 warmup_steps: int, pi_min: float = 0.1, window: int = 64, chunks: int = 16,
                 policy_seed: int = 0):
        super().__init__(budget=budget, total_steps=total_steps, num_examples=num_examples,
                         batches_per_epoch=batches_per_epoch, warmup_steps=warmup_steps,
                         use_scores=True, use_lr_weight=False, pi_min=pi_min, window=window,
                         chunks=chunks, policy_seed=policy_seed, selection="sample")
        frac = float(warmup_steps) / float(total_steps)
        # the kept examples' share after the warmup, if every later step pays a scoring pass
        kept_share = (self.budget - frac) / (1.0 - frac) - 1.0 / 3.0
        if not kept_share > self.pi_min:
            raise ValueError(f"budget {budget!r} leaves {kept_share:.3f} for the kept examples after the "
                             f"scoring passes; it must exceed pi_min={pi_min!r}")
        self.overhead_ratio = 1.0 / 3.0
        self.scoring_passes = 0
        self.extra_forward_padded_tokens = 0
        self.extra_forward_examples = 0

    def _batch_scores(self, trainer, inputs, t, uids, stored, seen, n, width):
        if t < self.warmup_steps:
            return np.ones(n, dtype=np.float64), None, None, 0
        logits = _forward_logits(trainer, inputs)
        s = per_example_residuals(logits, _labels(inputs)).numpy()
        if s.shape[0] != n:
            raise StoredScoreError(f"{s.shape[0]} scores for {n} rows")
        self.scoring_passes += 1
        self.extra_forward_padded_tokens += n * width
        self.extra_forward_examples += n
        return s, None, None, n * width

    def get_diagnostics(self) -> dict:
        d = super().get_diagnostics()
        d.update({"policy": self.name, "scores": "fresh residual of a scoring pass",
                  "scoring_passes": self.scoring_passes, "overhead_units": self.overhead_units,
                  "extra_forward_padded_tokens": self.extra_forward_padded_tokens,
                  "extra_forward_examples": self.extra_forward_examples})
        return d


# --------------------------------------------------------------------------- Selective Backprop
class SelectiveBackpropPolicy:
    """Keep example i with probability CDF(loss_i)^beta from a scoring pass (Jiang et al. 2019)."""

    name = "selective_backprop"

    def __init__(self, *, budget: float, total_steps: int, num_examples: int, batches_per_epoch: int,
                 history: int = 1024, policy_seed: int = 0):
        self.budget = _check_float("budget", budget, 0.0, 1.0)
        self.total_steps = _check_int("total_steps", total_steps)
        self.num_examples = _check_int("num_examples", num_examples)
        self.batches_per_epoch = _check_int("batches_per_epoch", batches_per_epoch)
        self.history = _check_int("history", history)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"policy_seed must be an int, got {policy_seed!r}")
        self.policy_seed = int(policy_seed)
        self.kept_share = self.budget - 1.0 / 3.0
        if not 0.0 < self.kept_share <= 2.0 / 3.0:
            raise ValueError(f"budget {budget!r} must exceed the scoring pass's 1/3")
        self.beta = 1.0 / self.kept_share - 1.0
        self._hist: deque = deque(maxlen=self.history)
        self._rng = np.random.default_rng(np.random.SeedSequence([self.policy_seed, 16, 2]))
        self.steps_seen = 0
        self.scheduled_examples = 0
        self.kept_examples = 0
        self.pruned_examples = 0
        self.empty_batch_fallbacks = 0
        self.scoring_passes = 0
        self.extra_forward_padded_tokens = 0
        self.extra_forward_examples = 0
        self.mean_select_probability_sum = 0.0

    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        t = _decision_index(trainer)
        if t != self.steps_seen:
            raise StoredScoreError(f"decision {t} out of order: expected {self.steps_seen}")
        n, width, _ = _batch_shape(inputs)
        logits = _forward_logits(trainer, inputs)
        loss = per_example_losses(logits, _labels(inputs)).numpy()
        if loss.shape[0] != n:
            raise StoredScoreError(f"{loss.shape[0]} losses for {n} rows")
        self._hist.extend(loss.tolist())
        h = np.sort(np.fromiter(self._hist, dtype=np.float64))
        cdf = np.searchsorted(h, loss, side="right") / float(len(h))
        p = np.power(cdf, self.beta)
        draws = self._rng.random(n)
        keep = [i for i in range(n) if draws[i] < p[i]]
        if not keep:
            keep = [int(np.argmax(loss))]
            self.empty_batch_fallbacks += 1
        self.scoring_passes += 1
        self.extra_forward_padded_tokens += n * width
        self.extra_forward_examples += n
        self.mean_select_probability_sum += float(p.mean())
        self.scheduled_examples += n
        self.kept_examples += len(keep)
        self.pruned_examples += n - len(keep)
        self.steps_seen += 1
        return keep, [float(n) / float(len(keep))] * len(keep)

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        return {"policy": self.name, "budget": self.budget, "beta": self.beta,
                "kept_share_target": self.kept_share, "history": self.history,
                "steps_seen": self.steps_seen, "scheduled_examples": self.scheduled_examples,
                "kept_examples": self.kept_examples, "pruned_examples": self.pruned_examples,
                "empty_batch_fallbacks": self.empty_batch_fallbacks, "scoring_passes": self.scoring_passes,
                "extra_forward_padded_tokens": self.extra_forward_padded_tokens,
                "extra_forward_examples": self.extra_forward_examples,
                "mean_select_probability": (self.mean_select_probability_sum / self.steps_seen
                                            if self.steps_seen else None),
                "total_steps": self.total_steps, "num_examples": self.num_examples,
                "batches_per_epoch": self.batches_per_epoch}


# --------------------------------------------------------------------------- KAKURENBO
class KakurenboPolicy:
    """Hide the lowest-loss, confidently right examples for an epoch (Nguyen et al., NeurIPS 2023)."""

    name = "kakurenbo_fixed_steps"
    ALPHA_STEPS = ((0.0, 1.0), (0.3, 0.8), (0.6, 0.6), (0.8, 0.4))

    def __init__(self, *, total_steps: int, num_examples: int, batches_per_epoch: int,
                 max_fraction: float = 0.3, tau: float = 0.7, policy_seed: int = 0):
        self.total_steps = _check_int("total_steps", total_steps)
        self.num_examples = _check_int("num_examples", num_examples)
        self.batches_per_epoch = _check_int("batches_per_epoch", batches_per_epoch)
        self.max_fraction = _check_float("max_fraction", max_fraction, 0.0, 1.0, hi_open=True)
        self.tau = _check_float("tau", tau, 0.0, 1.0)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"policy_seed must be an int, got {policy_seed!r}")
        self.policy_seed = int(policy_seed)
        self.last_epoch = (self.total_steps - 1) // self.batches_per_epoch
        self._loss = np.full(self.num_examples, np.nan, dtype=np.float64)
        self._correct = np.zeros(self.num_examples, dtype=bool)
        self._conf = np.full(self.num_examples, np.nan, dtype=np.float64)
        self._hidden = np.zeros(self.num_examples, dtype=bool)
        self._weight = 1.0
        self._pending: Optional[List[int]] = None
        self.epochs: List[dict] = []
        self.steps_seen = 0
        self.scheduled_examples = 0
        self.kept_examples = 0
        self.pruned_examples = 0
        self.empty_batch_fallbacks = 0
        self.extra_forward_padded_tokens = 0
        self.extra_forward_examples = 0
        self.extra_forward_passes = 0

    def _alpha(self, frac: float) -> float:
        a = 1.0
        for start, value in self.ALPHA_STEPS:
            if frac >= start - 1e-12:
                a = value
        return a

    def _start_epoch(self, e: int) -> None:
        frac = (e * self.batches_per_epoch) / float(self.total_steps)
        f_max = self.max_fraction * self._alpha(frac)
        m = int(math.floor(f_max * self.num_examples + 1e-9))
        self._hidden[:] = False
        seen = np.flatnonzero(~np.isnan(self._loss))
        candidates = moved_back = 0
        if e > 0 and m > 0 and seen.size:
            order = seen[np.lexsort((seen, self._loss[seen]))]  # lowest loss first, ties by id
            cand = order[:m]
            ok = self._correct[cand] & (self._conf[cand] >= self.tau)
            self._hidden[cand[ok]] = True
            candidates, moved_back = int(cand.size), int((~ok).sum())
        hidden = int(self._hidden.sum())
        f_star = hidden / float(self.num_examples)
        self._weight = 1.0 / (1.0 - f_star)
        self.epochs.append({"epoch": e, "fraction_of_training": frac, "max_fraction": f_max,
                            "candidates": candidates, "moved_back": moved_back, "hidden": hidden,
                            "hidden_fraction": f_star, "kept_weight": self._weight,
                            "refresh": bool(e < self.last_epoch)})

    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        t = _decision_index(trainer)
        if t != self.steps_seen:
            raise StoredScoreError(f"decision {t} out of order: expected {self.steps_seen}")
        if self._pending is not None:
            raise StoredScoreError("the previous selection was never observed by a forward pass")
        uids = _batch_uids(trainer)
        for u in uids:
            if u < 0 or u >= self.num_examples:
                raise StoredScoreError(f"lerna_uid {u} outside [0, {self.num_examples})")
        n, width, _ = _batch_shape(inputs)
        if n != len(uids):
            raise StoredScoreError(f"{n} rows for {len(uids)} ids")
        e = t // self.batches_per_epoch
        if t % self.batches_per_epoch == 0:
            self._start_epoch(e)
        hid = [i for i, u in enumerate(uids) if self._hidden[u]]
        keep = [i for i, u in enumerate(uids) if not self._hidden[u]]
        if not keep:
            j = int(np.argmax(self._loss[np.asarray(uids, dtype=np.int64)]))
            keep, hid = [j], [i for i in hid if i != j]
            self.empty_batch_fallbacks += 1
        if hid and e < self.last_epoch:
            sub, tokens = _subset(inputs, hid)
            loss, correct, conf = _loss_correct_conf(_forward_logits(trainer, sub), _labels(sub))
            hu = np.asarray([uids[i] for i in hid], dtype=np.int64)
            self._loss[hu], self._correct[hu], self._conf[hu] = loss, correct, conf
            self.extra_forward_padded_tokens += tokens
            self.extra_forward_examples += len(hid)
            self.extra_forward_passes += 1
        self._pending = [uids[i] for i in keep]
        self.scheduled_examples += n
        self.kept_examples += len(keep)
        self.pruned_examples += n - len(keep)
        self.steps_seen += 1
        return keep, [self._weight] * len(keep)

    def observe_forward(self, trainer, inputs, logits) -> None:
        if self._pending is None:
            raise StoredScoreError("forward pass observed without a selection")
        uids = _batch_uids(trainer)
        if list(uids) != self._pending:
            raise StoredScoreError("the forward pass's ids differ from the selected ones")
        loss, correct, conf = _loss_correct_conf(logits, _labels(inputs))
        u = np.asarray(uids, dtype=np.int64)
        self._loss[u], self._correct[u], self._conf[u] = loss, correct, conf
        self._pending = None

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        return {"policy": self.name, "max_fraction": self.max_fraction, "tau": self.tau,
                "alpha_steps": [list(a) for a in self.ALPHA_STEPS], "last_epoch": self.last_epoch,
                "epochs": list(self.epochs), "steps_seen": self.steps_seen,
                "scheduled_examples": self.scheduled_examples, "kept_examples": self.kept_examples,
                "pruned_examples": self.pruned_examples, "empty_batch_fallbacks": self.empty_batch_fallbacks,
                "extra_forward_padded_tokens": self.extra_forward_padded_tokens,
                "extra_forward_examples": self.extra_forward_examples,
                "extra_forward_passes": self.extra_forward_passes,
                "examples_with_stored_loss": int((~np.isnan(self._loss)).sum()),
                "total_steps": self.total_steps, "num_examples": self.num_examples,
                "batches_per_epoch": self.batches_per_epoch}


# --------------------------------------------------------------------------- IES
class IESPolicy:
    """Exclude mastered examples from the update (Yuan et al., ICLR 2025)."""

    name = "ies_fixed_steps"

    def __init__(self, *, total_steps: int, num_examples: int, batches_per_epoch: int,
                 delta: float = 1e-3, policy_seed: int = 0):
        self.total_steps = _check_int("total_steps", total_steps)
        self.num_examples = _check_int("num_examples", num_examples)
        self.batches_per_epoch = _check_int("batches_per_epoch", batches_per_epoch)
        self.delta = _check_float("delta", delta, 0.0, 1e6, lo_open=True)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"policy_seed must be an int, got {policy_seed!r}")
        self.policy_seed = int(policy_seed)
        # newest first: losses of the example's last three epochs, and those epochs
        self._loss = np.full((self.num_examples, 3), np.nan, dtype=np.float64)
        self._epoch = np.full((self.num_examples, 3), -1, dtype=np.int64)
        self._pending: Optional[Tuple[int, List[int]]] = None
        self._per_epoch: Dict[int, List[int]] = {}
        self.steps_seen = 0
        self.scheduled_examples = 0
        self.kept_examples = 0
        self.pruned_examples = 0
        self.empty_batch_fallbacks = 0
        self.reincluded = 0
        self.extra_forward_padded_tokens = 0
        self.extra_forward_examples = 0
        self.extra_forward_passes = 0
        self._was_mastered = np.zeros(self.num_examples, dtype=bool)

    def _record(self, uids: Sequence[int], losses: np.ndarray, e: int) -> None:
        u = np.asarray(uids, dtype=np.int64)
        if np.any(self._epoch[u, 0] == e):
            raise StoredScoreError(f"an example was recorded twice in epoch {e}")
        self._loss[u, 2] = self._loss[u, 1]
        self._loss[u, 1] = self._loss[u, 0]
        self._loss[u, 0] = losses
        self._epoch[u, 2] = self._epoch[u, 1]
        self._epoch[u, 1] = self._epoch[u, 0]
        self._epoch[u, 0] = e

    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        t = _decision_index(trainer)
        if t != self.steps_seen:
            raise StoredScoreError(f"decision {t} out of order: expected {self.steps_seen}")
        if self._pending is not None:
            raise StoredScoreError("the previous selection was never observed by a forward pass")
        uids = _batch_uids(trainer)
        for u in uids:
            if u < 0 or u >= self.num_examples:
                raise StoredScoreError(f"lerna_uid {u} outside [0, {self.num_examples})")
        n, width, _ = _batch_shape(inputs)
        if n != len(uids):
            raise StoredScoreError(f"{n} rows for {len(uids)} ids")
        e = t // self.batches_per_epoch
        u = np.asarray(uids, dtype=np.int64)
        consecutive = np.all(self._epoch[u] == np.array([e - 1, e - 2, e - 3]), axis=1)
        d2 = self._loss[u, 0] - 2.0 * self._loss[u, 1] + self._loss[u, 2]
        mastered = consecutive & (np.abs(np.nan_to_num(d2, nan=np.inf)) < self.delta)
        mas = [i for i in range(n) if mastered[i]]
        keep = [i for i in range(n) if not mastered[i]]
        if not keep:
            j = int(np.argmax(np.abs(d2)))
            keep, mas = [j], [i for i in mas if i != j]
            self.empty_batch_fallbacks += 1
        for i in keep:
            if self._was_mastered[uids[i]]:
                self.reincluded += 1
                self._was_mastered[uids[i]] = False
        for i in mas:
            self._was_mastered[uids[i]] = True
        if mas:
            sub, tokens = _subset(inputs, mas)
            loss = per_example_losses(_forward_logits(trainer, sub), _labels(sub)).numpy()
            self._record([uids[i] for i in mas], loss, e)
            self.extra_forward_padded_tokens += tokens
            self.extra_forward_examples += len(mas)
            self.extra_forward_passes += 1
        rec = self._per_epoch.setdefault(e, [0, 0])
        rec[0] += n
        rec[1] += len(mas)
        self._pending = (e, [uids[i] for i in keep])
        self.scheduled_examples += n
        self.kept_examples += len(keep)
        self.pruned_examples += n - len(keep)
        self.steps_seen += 1
        return keep, [float(n) / float(len(keep))] * len(keep)

    def observe_forward(self, trainer, inputs, logits) -> None:
        if self._pending is None:
            raise StoredScoreError("forward pass observed without a selection")
        e, expect = self._pending
        uids = _batch_uids(trainer)
        if list(uids) != expect:
            raise StoredScoreError("the forward pass's ids differ from the selected ones")
        self._record(uids, per_example_losses(logits, _labels(inputs)).numpy(), e)
        self._pending = None

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        epochs = {str(k): {"scheduled": v[0], "mastered": v[1],
                           "mastered_fraction": (v[1] / v[0]) if v[0] else None}
                  for k, v in sorted(self._per_epoch.items())}
        return {"policy": self.name, "delta": self.delta, "epochs": epochs, "steps_seen": self.steps_seen,
                "scheduled_examples": self.scheduled_examples, "kept_examples": self.kept_examples,
                "pruned_examples": self.pruned_examples, "empty_batch_fallbacks": self.empty_batch_fallbacks,
                "reincluded": self.reincluded,
                "extra_forward_padded_tokens": self.extra_forward_padded_tokens,
                "extra_forward_examples": self.extra_forward_examples,
                "extra_forward_passes": self.extra_forward_passes,
                "total_steps": self.total_steps, "num_examples": self.num_examples,
                "batches_per_epoch": self.batches_per_epoch}


# --------------------------------------------------------------------------- DropBP
def _encoder_layers(model) -> list:
    """The encoder layers of a ModernBERT-style classifier (each with .attn and .mlp) or, from
    round 17, of an UltraBERT classifier (each with .self_attn and .mlp)."""
    for owner in (getattr(model, "model", None), getattr(model, "base_model", None), model):
        layers = getattr(owner, "layers", None) if owner is not None else None
        if layers is not None and len(layers) > 0 and all(
                (hasattr(l, "attn") or hasattr(l, "self_attn")) and hasattr(l, "mlp") for l in layers):
            return list(layers)
    raise ValueError("DropBP needs a model whose encoder layers have .attn (or .self_attn) and .mlp blocks")


def _attention_block(layer):
    """The attention module of an encoder layer: ModernBERT's .attn, UltraBERT's .self_attn."""
    return layer.attn if hasattr(layer, "attn") else layer.self_attn


def _linear_macs(module) -> int:
    return int(sum(m.weight.numel() for m in module.modules() if isinstance(m, torch.nn.Linear)))


def _patch_dropbp(module) -> None:
    inner = module.forward

    def forward(*args, **kwargs):
        if module._lerna_dropbp and torch.is_grad_enabled():
            with torch.no_grad():
                return inner(*args, **kwargs)
        return inner(*args, **kwargs)

    module.forward = forward
    module._lerna_dropbp = False


class DropBPPolicy:
    """Drop each attention / MLP block's backward pass with probability p at every step."""

    name = "dropbp_uniform"

    def __init__(self, *, model, drop_rate: float, total_steps: int, policy_seed: int = 0):
        self.drop_rate = _check_float("drop_rate", drop_rate, 0.0, 1.0, hi_open=True)
        self.total_steps = _check_int("total_steps", total_steps)
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"policy_seed must be an int, got {policy_seed!r}")
        self.policy_seed = int(policy_seed)
        mcfg = getattr(model, "config", None)
        if mcfg is not None and getattr(mcfg, "reference_compile", None):
            raise ValueError("DropBP needs reference_compile off (the compiled MLP path is not wrapped)")
        layers = _encoder_layers(model)
        self._blocks = []  # (layer, kind, module)
        for li, layer in enumerate(layers):
            for kind in ("attn", "mlp"):
                mod = _attention_block(layer) if kind == "attn" else layer.mlp
                if hasattr(mod, "_lerna_dropbp"):
                    raise ValueError("the model already carries DropBP wrappers")
                _patch_dropbp(mod)
                self._blocks.append((li, kind, mod))
        hidden = int(getattr(mcfg, "hidden_size", 0) or 0) if mcfg is not None else 0
        # the attention scores' width: heads x head size where the configuration gives a head
        # size (UltraBERT), otherwise the hidden size (ModernBERT, as in round 16)
        head_dim = getattr(mcfg, "head_dim", None) if mcfg is not None else None
        heads = getattr(mcfg, "num_attention_heads", None) if mcfg is not None else None
        width = int(heads) * int(head_dim) if head_dim and heads else hidden
        self.flops_per_token = {
            "layers": len(layers),
            "attn_linear": _linear_macs(_attention_block(layers[0])),
            "attn_scores_per_position": 2 * width,
            "mlp": _linear_macs(layers[0].mlp),
        }
        for layer in layers:
            if (_linear_macs(_attention_block(layer)) != self.flops_per_token["attn_linear"]
                    or _linear_macs(layer.mlp) != self.flops_per_token["mlp"]):
                raise ValueError("DropBP's accounting assumes identical layers")
        self._rng = np.random.default_rng(np.random.SeedSequence([self.policy_seed, 16, 3]))
        self.drop_counts: List[List[int]] = []  # per step: [attention blocks dropped, MLP blocks dropped]
        self.steps_seen = 0
        self.pruned_examples = 0
        self.kept_examples = 0
        self.scheduled_examples = 0

    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        t = _decision_index(trainer)
        if t != self.steps_seen:
            raise StoredScoreError(f"decision {t} out of order: expected {self.steps_seen}")
        n, width, _ = _batch_shape(inputs)
        drop = self._rng.random(len(self._blocks)) < self.drop_rate
        a = m = 0
        for flag, (_, kind, mod) in zip(drop.tolist(), self._blocks):
            mod._lerna_dropbp = bool(flag)
            if flag:
                if kind == "attn":
                    a += 1
                else:
                    m += 1
        self.drop_counts.append([a, m])
        self.scheduled_examples += n
        self.kept_examples += n
        self.steps_seen += 1
        return list(range(n)), [1.0] * n

    def backward_fraction(self, step: int, width: int) -> float:
        """The share of a full backward pass that step `step` ran, for padded width `width`."""
        f = self.flops_per_token
        attn = f["attn_linear"] + f["attn_scores_per_position"] * int(width)
        full = f["layers"] * (attn + f["mlp"])
        a, m = self.drop_counts[step]
        return 1.0 - (a * attn + m * f["mlp"]) / float(full)

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    def get_diagnostics(self) -> dict:
        blocks = 2 * self.flops_per_token["layers"]
        dropped = sum(a + m for a, m in self.drop_counts)
        return {"policy": self.name, "drop_rate": self.drop_rate, "allocation": "uniform",
                "flops_per_token": dict(self.flops_per_token), "steps_seen": self.steps_seen,
                "blocks": blocks, "dropped_block_fraction": (dropped / float(blocks * self.steps_seen)
                                                              if self.steps_seen else None),
                "drop_counts": [list(x) for x in self.drop_counts],
                "scheduled_examples": self.scheduled_examples, "kept_examples": self.kept_examples,
                "pruned_examples": self.pruned_examples, "extra_forward_padded_tokens": 0,
                "extra_forward_examples": 0, "total_steps": self.total_steps}


__all__ = [
    "RankedSubsetPolicy",
    "FreshImportancePolicy",
    "SelectiveBackpropPolicy",
    "KakurenboPolicy",
    "IESPolicy",
    "DropBPPolicy",
]
