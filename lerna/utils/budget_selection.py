"""Round 14 (6 Nov 2026): LERNA v2, learning-rate-aware example selection under an exact budget.

LernaBudgetPolicy never skips a step. Before each scheduled batch's forward pass it decides which
of the batch's examples take part in that step; the others get no forward and no backward pass.

The rule
    After the warmup, example i of the batch at decision t is kept with probability

        pi_i = clip(w_t * s_i / mu_t, pi_min, 1)

    w_t   the learning-rate multiplier the update at t uses (the scheduler's lambda at step t);
          1 for the variants without the learning-rate factor.
    s_i   the example's residual ||p_i - y_i|| from its last forward pass, put on today's scale by
          the revisit ratio (below); an example never seen yet gets the first-visit mean. The
          random variants use s_i = 1 for every example.
    mu_t  the price the budget controller sets so that the run spends exactly its budget.

    Kept examples get weight 1 / pi_i and the loss is divided by the scheduled batch size
    (Horvitz-Thompson), so the batch gradient stays unbiased. During the warmup every example is
    kept. At least one example is always kept (the one with the largest pi).

Why this rule
    For SGD on an L-smooth loss, keeping example i with probability pi_i independently adds
    (L/2) eta_t^2 (1/B^2) sum_i ||g_i||^2 (1/pi_i - 1) to the expected loss of step t. Minimizing
    the sum over the run, subject to the expected compute sum_{t,i} c pi_{t,i} = C, gives
    pi_{t,i} = min(1, eta_t ||g_{t,i}|| / mu) with one constant mu for the whole run (the KKT
    conditions). ||p - y|| is the usual cheap bound on the per-example gradient norm (Katharopoulos
    and Fleuret, ICML 2018). The floor pi_min caps the weights at 1 / pi_min. With AdamW the
    derivation is a heuristic, not a theorem.

Today's scale
    A stored residual is as old as the example's last visit. Over the recent window, the ratio of
    fresh to stored residuals on revisited kept examples (each weighted 1 / pi, so the ratio is not
    biased by the selection) rescales every stored residual. The first-visit mean is the weighted
    mean fresh residual of examples seen for the first time: the model's current error on data it
    hasn't trained on.

Hard selection (round 15, selection="topk")
    A compared method, not LERNA v2: the same keep probabilities and budget controller, but the
    K = round(sum pi) examples with the largest pi are kept deterministically, and each counts
    n / K, so the loss is the plain mean over the kept examples. This is hard pruning without
    reweighting, as in dynamic EL2N pruning (Attendu and Corbeil, 2023).

The budget controller
    Units are those of the batch log: padded tokens of the forward pass plus twice those of the
    update, so 3 x kept examples x their longest length (the trainer trims the unused padding).
    Before each decision after the warmup, the controller solves for mu: the cost of this batch
    plus the projected cost of every remaining step must equal the remaining budget, which is
    budget x (units of all scheduled batches) minus the units spent. The scheduled batches' units
    are known exactly at the last step and estimated from their mean before it. The recent window
    of scheduled batches (their decision scores and widths) stands in for the future batches, the
    future multipliers w are known from the schedule, and the ratio of realized to projected units
    over the window corrects the projection for trimmed padding.

Round 16 hooks
    _batch_scores (the decision scores of a batch) and _choose (which examples to keep, given pi)
    let the compared methods in lerna/utils/compared_methods.py reuse this controller, and
    overhead_ratio charges a scoring pass to the budget: the pass's units, and those of the
    remaining steps' passes, are taken from the budget before mu is solved. LERNA v2's own
    decisions are unchanged.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from lerna.utils.stale_selection import StoredScoreError, _batch_uids, per_example_residuals


class LernaBudgetPolicy:
    """LERNA v2: keep example i with probability clip(w_t s_i / mu_t, pi_min, 1); every step runs."""

    name = "lerna_budget"
    TRAJECTORY_EVERY = 64
    # "sample": keep each example with probability pi, weight 1 / pi (LERNA v2).
    # "topk" (round 15): keep the K = round(sum pi) examples with the largest pi, deterministically,
    # each weighted n / K so the loss is the plain mean over the kept examples: hard pruning without
    # reweighting, as in dynamic EL2N pruning (Attendu and Corbeil, 2023), under the same budget.
    SELECTIONS = ("sample", "topk")

    def __init__(
        self,
        *,
        budget: float,
        total_steps: int,
        num_examples: int,
        batches_per_epoch: int,
        warmup_steps: int,
        use_scores: bool = True,
        use_lr_weight: bool = True,
        pi_min: float = 0.1,
        window: int = 64,
        chunks: int = 16,
        policy_seed: int = 0,
        selection: str = "sample",
    ):
        if selection not in self.SELECTIONS:
            raise ValueError(f"selection must be one of {self.SELECTIONS}, got {selection!r}")
        for nm, v in (("total_steps", total_steps), ("num_examples", num_examples),
                      ("batches_per_epoch", batches_per_epoch), ("window", window), ("chunks", chunks)):
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{nm} must be a positive int, got {v!r}")
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or not 0 <= warmup_steps < total_steps:
            raise ValueError(f"warmup_steps must be an int in [0, total_steps), got {warmup_steps!r}")
        for nm, v in (("budget", budget), ("pi_min", pi_min)):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
                raise ValueError(f"{nm} must be a finite number, got {v!r}")
        if not 0.0 < float(pi_min) <= 1.0:
            raise ValueError(f"pi_min must be in (0, 1], got {pi_min!r}")
        if not float(pi_min) < float(budget) <= 1.0:
            raise ValueError(f"budget must be in (pi_min, 1], got {budget!r}")
        for nm, v in (("use_scores", use_scores), ("use_lr_weight", use_lr_weight)):
            if not isinstance(v, bool):
                raise ValueError(f"{nm} must be a bool, got {v!r}")
        if isinstance(policy_seed, bool) or not isinstance(policy_seed, int):
            raise ValueError(f"policy_seed must be an int, got {policy_seed!r}")
        self.budget = float(budget)
        self.total_steps = int(total_steps)
        self.num_examples = int(num_examples)
        self.batches_per_epoch = int(batches_per_epoch)
        self.warmup_steps = int(warmup_steps)
        self.use_scores = bool(use_scores)
        self.use_lr_weight = bool(use_lr_weight)
        self.pi_min = float(pi_min)
        self.window = int(window)
        self.chunks = int(chunks)
        self.policy_seed = int(policy_seed)
        self.selection = selection
        self._rng = np.random.default_rng(np.random.SeedSequence([self.policy_seed, 14]))
        self._residual = np.full(self.num_examples, np.nan, dtype=np.float64)
        self._lr_w: Optional[np.ndarray] = None
        # windows over the most recent scheduled batches
        self._win_scores: deque = deque(maxlen=self.window)  # decision scores of each scheduled batch
        self._win_width: deque = deque(maxlen=self.window)  # padded width of each scheduled batch
        self._win_proj: deque = deque(maxlen=self.window)  # projected units at the chosen mu
        self._win_real: deque = deque(maxlen=self.window)  # realized units
        self._win_revisit: deque = deque(maxlen=self.window)  # (sum w*fresh, sum w*stored) of revisits
        self._win_first: deque = deque(maxlen=self.window)  # (sum w*fresh, sum w) of first visits
        self._win_all: deque = deque(maxlen=self.window)  # (sum w*fresh, sum w) of every kept example
        self._pending: Optional[dict] = None
        # Round 16: units a subclass spends on a batch before its selection (a scoring forward
        # pass), as a fraction of the batch's scheduled units after the warmup; 0 for LERNA v2.
        self.overhead_ratio = 0.0
        self.overhead_units = 0
        # accounting
        self.steps_seen = 0
        self.scheduled_units = 0
        self.spent_units = 0
        self.scheduled_examples = 0
        self.kept_examples = 0
        self.pruned_examples = 0
        self.pruned_during_warmup = 0
        self.empty_batch_fallbacks = 0
        self.first_visits_kept = 0
        self.revisits_kept = 0
        self.max_weight = 1.0
        self.mu_inf_steps = 0  # steps where the remaining budget only paid for the floor
        self.mu_zero_steps = 0  # steps after the warmup where every example was kept
        self._epoch: Dict[int, List[float]] = {}  # epoch -> [scheduled, kept, sum pi]
        self.trajectory: List[dict] = []
        # the last decision, for tests and debugging
        self.last_pi: Optional[np.ndarray] = None
        self.last_mu: Optional[float] = None
        self.last_scores: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ helpers
    def _decision_index(self, trainer) -> int:
        seen = int(trainer.instr.batches_seen)
        if seen < 1:
            raise StoredScoreError("selection requested before any batch was seen")
        return seen - 1

    def _lr_weight(self, trainer, t: int) -> float:
        if not self.use_lr_weight:
            return 1.0
        sched = getattr(trainer, "lr_scheduler", None)
        if sched is not None and not hasattr(sched, "lr_lambdas") and hasattr(sched, "scheduler"):
            sched = sched.scheduler  # a wrapper such as accelerate's AcceleratedScheduler
        lambdas = getattr(sched, "lr_lambdas", None)
        if not lambdas:
            raise StoredScoreError("the learning-rate factor needs a LambdaLR scheduler")
        if self._lr_w is None:
            fn = lambdas[0]
            w = np.array([float(fn(k)) for k in range(self.total_steps)], dtype=np.float64)
            if not np.all(np.isfinite(w)) or np.any(w < 0.0):
                raise StoredScoreError("the learning-rate multipliers must be finite and non-negative")
            self._lr_w = w
        last = getattr(sched, "last_epoch", None)
        if last is not None and int(last) != t:
            raise StoredScoreError(f"scheduler is at step {last}, decision {t}: the multipliers would not match")
        return float(self._lr_w[t])

    def _revisit_ratio(self) -> float:
        num = sum(a for a, _ in self._win_revisit)
        den = sum(b for _, b in self._win_revisit)
        return num / den if den > 0.0 else 1.0

    def _first_visit_prior(self) -> float:
        num = sum(a for a, _ in self._win_first)
        den = sum(b for _, b in self._win_first)
        if den > 0.0:
            return num / den
        num = sum(a for a, _ in self._win_all)
        den = sum(b for _, b in self._win_all)
        return num / den if den > 0.0 else 1.0

    def _future_chunks(self, t: int) -> Tuple[np.ndarray, np.ndarray]:
        """(counts, mean multipliers) of the steps after t, in contiguous chunks."""
        rest = self.total_steps - t - 1
        if rest <= 0:
            return np.zeros(0), np.zeros(0)
        if self.use_lr_weight:
            w = self._lr_w[t + 1:]
        else:
            w = np.ones(rest)
        parts = np.array_split(w, min(self.chunks, rest))
        return (np.array([len(p) for p in parts], dtype=np.float64),
                np.array([float(p.mean()) for p in parts], dtype=np.float64))

    def _solve_mu(self, t: int, w_t: float, s: np.ndarray, width: int, remaining: float) -> float:
        """The price mu at which this batch plus the projected rest of the run spends `remaining`.

        Returns 0.0 when even keeping everything spends no more than `remaining` (keep all) and
        math.inf when even the floor pi_min spends at least `remaining` (floor everywhere).
        """
        if self._win_scores:
            ws = np.concatenate(list(self._win_scores))
            wc = np.concatenate([np.full(len(a), 3.0 * float(L)) for a, L in zip(self._win_scores, self._win_width)])
            nwin = float(len(self._win_scores))
        else:
            ws, wc, nwin = s, np.full(len(s), 3.0 * float(width)), 1.0
        proj = float(sum(self._win_proj))
        real = float(sum(self._win_real))
        rho = real / proj if proj > 0.0 else 1.0
        counts, wbar = self._future_chunks(t)
        # products (multiplier x score) and their cost weights, current batch first
        prods = [w_t * s]
        costs = [np.full(len(s), 3.0 * float(width))]
        if len(counts):
            prods.append(np.outer(wbar, ws).ravel())
            costs.append(np.outer(counts / nwin, wc).ravel())
        P = np.concatenate(prods)
        Cw = np.concatenate(costs) * rho
        pos = P[P > 0.0]
        if pos.size == 0:
            # every multiplier x score is 0: pi is the floor everywhere whatever mu is
            return math.inf

        def spend(mu: float) -> float:
            return float(np.sum(Cw * np.clip(P / mu, self.pi_min, 1.0)))

        lo, hi = float(pos.min()), float(P.max()) / self.pi_min
        if spend(lo) <= remaining:
            return 0.0
        if spend(hi) >= remaining:
            return math.inf
        a, b = math.log(lo), math.log(hi)
        for _ in range(64):
            mid = 0.5 * (a + b)
            if spend(math.exp(mid)) > remaining:
                a = mid
            else:
                b = mid
        return math.exp(0.5 * (a + b))

    # ------------------------------------------------------------------ hooks (round 16)
    def _batch_scores(self, trainer, inputs, t: int, uids: Sequence[int], stored: np.ndarray,
                      seen: np.ndarray, n: int, width: int):
        """(decision scores, revisit ratio, first-visit prior, overhead units) of this batch.

        LERNA v2 scores from the stored residuals and spends nothing before the selection. The
        round-16 compared methods override this (fresh scores from a scoring pass).
        """
        if self.use_scores:
            ratio, prior = self._revisit_ratio(), self._first_visit_prior()
            s = np.where(seen, np.nan_to_num(stored) * ratio, prior)
        else:
            ratio, prior = None, None
            s = np.ones(n, dtype=np.float64)
        return s, ratio, prior, 0

    def _choose(self, t: int, pi: np.ndarray, s: np.ndarray, n: int,
                uids: Sequence[int]) -> Tuple[List[int], List[float]]:
        """Positions kept and their weights, given the keep probabilities pi."""
        if self.selection == "topk":
            k = max(1, min(n, int(round(float(pi.sum())))))
            # largest pi first; ties by the larger score, then by position
            order = sorted(range(n), key=lambda i: (-float(pi[i]), -float(s[i]), i))
            keep = sorted(order[:k])
            weights = [float(n) / float(k)] * k
        else:
            draws = self._rng.random(n)
            keep = [i for i in range(n) if draws[i] < pi[i]]
            if not keep:
                keep = [int(np.argmax(pi))]
                self.empty_batch_fallbacks += 1
            weights = [1.0 / float(pi[i]) for i in keep]
        return keep, weights

    # ------------------------------------------------------------------ the policy interface
    def select_examples(self, trainer, inputs) -> Tuple[List[int], List[float]]:
        """Positions to keep in this batch and their loss weights (1 / pi)."""
        t = self._decision_index(trainer)
        if t != self.steps_seen:
            raise StoredScoreError(f"decision {t} out of order: expected {self.steps_seen}")
        if t >= self.total_steps:
            raise StoredScoreError(f"decision {t} beyond the horizon of {self.total_steps}")
        if self._pending is not None:
            raise StoredScoreError("the previous selection was never observed by a forward pass")
        uids = _batch_uids(trainer)
        for u in uids:
            if u < 0 or u >= self.num_examples:
                raise StoredScoreError(f"lerna_uid {u} outside [0, {self.num_examples})")
        ids = inputs.get("input_ids") if hasattr(inputs, "get") else None
        mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
        if not isinstance(ids, torch.Tensor) or ids.dim() != 2:
            raise StoredScoreError("selection needs a 2-D input_ids tensor")
        n, width = int(ids.shape[0]), int(ids.shape[1])
        if n != len(uids):
            raise StoredScoreError(f"{n} rows for {len(uids)} ids")
        if isinstance(mask, torch.Tensor) and mask.dim() == 2:
            lengths = [int(v) for v in mask.sum(dim=1).tolist()]
        else:
            lengths = [width] * n
        self.scheduled_units += 3 * n * width
        w_t = self._lr_weight(trainer, t)
        stored = self._residual[np.asarray(uids, dtype=np.int64)]
        seen = ~np.isnan(stored)
        s, ratio, prior, overhead = self._batch_scores(trainer, inputs, t, uids, stored, seen, n, width)
        if t < self.warmup_steps:
            mu = None
            pi = np.ones(n, dtype=np.float64)
        else:
            est_total = self.scheduled_units + (self.total_steps - t - 1) * (self.scheduled_units / (t + 1))
            remaining = self.budget * est_total - self.spent_units
            if self.overhead_ratio:
                # Round 16: this batch's scoring pass and those of the remaining steps are spent
                # whatever is kept, so only the rest of the budget is left for the kept examples.
                remaining = (remaining - overhead
                             - self.overhead_ratio * (self.total_steps - t - 1) * (self.scheduled_units / (t + 1)))
            mu = self._solve_mu(t, w_t, s, width, remaining)
            if mu == 0.0:
                pi = np.ones(n, dtype=np.float64)
                self.mu_zero_steps += 1
            elif math.isinf(mu):
                pi = np.full(n, self.pi_min, dtype=np.float64)
                self.mu_inf_steps += 1
            else:
                pi = np.clip(w_t * s / mu, self.pi_min, 1.0)
        self.last_pi, self.last_mu, self.last_scores = pi.copy(), mu, np.array(s, dtype=np.float64)
        keep, weights = self._choose(t, pi, s, n, uids)
        longest = max(max(lengths[i] for i in keep), 1)
        realized = 3 * len(keep) * longest
        self.spent_units += realized
        if overhead:
            self.spent_units += overhead
            self.overhead_units += overhead
        self._win_scores.append(np.array(s, dtype=np.float64))
        self._win_width.append(width)
        self._win_proj.append(3.0 * width * float(pi.sum()))
        self._win_real.append(float(realized))
        self._pending = {"t": t, "uids": [uids[i] for i in keep], "pi": [float(pi[i]) for i in keep],
                         "seen": [bool(seen[i]) for i in keep],
                         "stored": [float(stored[i]) if seen[i] else float("nan") for i in keep]}
        self.scheduled_examples += n
        self.kept_examples += len(keep)
        self.pruned_examples += n - len(keep)
        if t < self.warmup_steps:
            self.pruned_during_warmup += n - len(keep)
        self.max_weight = max(self.max_weight, max(weights))
        e = self._epoch.setdefault(t // self.batches_per_epoch, [0, 0, 0.0])
        e[0] += n
        e[1] += len(keep)
        e[2] += float(pi.sum())
        if t % self.TRAJECTORY_EVERY == 0 or t == self.total_steps - 1:
            self.trajectory.append({
                "t": t, "mu": (None if mu is None or math.isinf(mu) else float(mu)),
                "mu_state": ("warmup" if mu is None else "inf" if math.isinf(mu) else "zero" if mu == 0.0 else "solved"),
                "lr_weight": w_t, "revisit_ratio": ratio, "first_visit_prior": prior,
                "kept": len(keep), "scheduled": n,
                "spent_over_scheduled": self.spent_units / self.scheduled_units,
            })
        self.steps_seen += 1
        return keep, weights

    def observe_forward(self, trainer, inputs, logits) -> None:
        """Store the kept examples' fresh residuals and update the scale estimates."""
        p = self._pending
        if p is None:
            raise StoredScoreError("forward pass observed without a selection")
        uids = _batch_uids(trainer)
        if list(uids) != p["uids"]:
            raise StoredScoreError("the forward pass's ids differ from the selected ones")
        labels = inputs.get("labels") if hasattr(inputs, "get") else None
        res = per_example_residuals(logits, labels).numpy()
        if res.shape[0] != len(uids):
            raise StoredScoreError(f"{res.shape[0]} residuals for {len(uids)} ids")
        rv_num = rv_den = fv_num = fv_den = al_num = al_den = 0.0
        for r, pi, seen, stored in zip(res.tolist(), p["pi"], p["seen"], p["stored"]):
            w = 1.0 / pi
            al_num += w * r
            al_den += w
            if seen:
                rv_num += w * r
                rv_den += w * stored
                self.revisits_kept += 1
            else:
                fv_num += w * r
                fv_den += w
                self.first_visits_kept += 1
        self._win_revisit.append((rv_num, rv_den))
        self._win_first.append((fv_num, fv_den))
        self._win_all.append((al_num, al_den))
        self._residual[np.asarray(uids, dtype=np.int64)] = res
        self._pending = None

    def should_skip(self, trainer, model, inputs) -> bool:
        return False

    @property
    def realized_fraction(self) -> Optional[float]:
        return self.spent_units / self.scheduled_units if self.scheduled_units else None

    def get_diagnostics(self) -> dict:
        epochs = {str(k): {"scheduled": int(v[0]), "kept": int(v[1]),
                           "kept_fraction": (v[1] / v[0]) if v[0] else None,
                           "mean_pi": (v[2] / v[0]) if v[0] else None}
                  for k, v in sorted(self._epoch.items())}
        return {
            "policy": self.name,
            "budget": self.budget,
            "use_scores": self.use_scores,
            "use_lr_weight": self.use_lr_weight,
            "selection": self.selection,
            "pi_min": self.pi_min,
            "window": self.window,
            "chunks": self.chunks,
            "warmup_steps": self.warmup_steps,
            "total_steps": self.total_steps,
            "batches_per_epoch": self.batches_per_epoch,
            "steps_seen": self.steps_seen,
            "scheduled_units": self.scheduled_units,
            "spent_units": self.spent_units,
            "realized_fraction": self.realized_fraction,
            "scheduled_examples": self.scheduled_examples,
            "kept_examples": self.kept_examples,
            "pruned_examples": self.pruned_examples,
            "pruned_during_warmup": self.pruned_during_warmup,
            "empty_batch_fallbacks": self.empty_batch_fallbacks,
            "first_visits_kept": self.first_visits_kept,
            "revisits_kept": self.revisits_kept,
            "max_weight": self.max_weight,
            "mu_inf_steps": self.mu_inf_steps,
            "mu_zero_steps": self.mu_zero_steps,
            "never_seen_examples": int(np.isnan(self._residual).sum()),
            "num_examples": self.num_examples,
            "epochs": epochs,
            "trajectory": list(self.trajectory),
        }


__all__ = ["LernaBudgetPolicy"]
