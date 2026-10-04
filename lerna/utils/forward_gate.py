"""LER v4: forward-looking LER gate.

This module defines the deficit-aware quantile gate used by the
`ScoreQuantileGatePolicy` skip policy and its uniform-random control. It is
intentionally a *pure*, CPU-testable scorer: the policy wrapper
(`ScoreQuantileGatePolicy`) is the only part that touches a live trainer.

Design (single rule, zero tuned knobs):

* 4 phase windows of equal length over `total_steps`.
* Exact per-window quota = round(rate * window_length).
* Deficit-aware skip probability: q = remaining_window_quota /
  remaining_window_steps. A step is skipped when its score is below the q-quantile
  of the last `score_history_window` (default 64) observed scores.
* Feasibility forcing: when remaining_quota / remaining_steps >= 4/5, skip is
  forced (this is the maximum sustainable skip fraction under cap=4). Forced
  decisions are counted in diagnostics.
* Annealing: after `anneal_step`, the score is replaced by i.i.d. Uniform[0,1],
  so selection becomes random while the budget is preserved.
* cap: never exceed `max_consecutive_skips` consecutive skips; if the run is at
  the cap a backward step is forced even under forcing pressure.
* min_step: never skip before `min_step` decisions.
* Ties against the quantile threshold are broken by the policy RNG.
* Non-finite score: raises `TerminalInvalidScoreError`, which absorbs all
  subsequent decisions (terminal).

Score source is pluggable so the gate (forward predicted gain) and the control
(uniform random) share identical budget mechanics. Only the score differs.
"""

from __future__ import annotations

import math
import random
from typing import Deque, Optional, Sequence

from collections import deque


class TerminalInvalidScoreError(RuntimeError):
    """Raised when a score feeding the gate is non-finite.

    The gate is terminal from this point: no later decision may override it,
    matching the strict-mode semantics of the LERNA tracker.
    """

    def __init__(self, decision_index: int, value: float):
        self.decision_index = decision_index
        self.value = value
        super().__init__(
            f"Terminal invalid score at decision {decision_index}: {value!r}"
        )


def _phase_bounds(total_steps: int, n_phases: int) -> list[tuple[int, int]]:
    bounds = []
    step = 0
    for _ in range(n_phases):
        # last window absorbs the remainder so the whole horizon is covered.
        end = total_steps if len(bounds) == n_phases - 1 else step + total_steps // n_phases
        bounds.append((step, end))
        step = end
    return bounds


class InfeasibleBudgetError(ValueError):
    """Raised when the requested skip quota exceeds the number of eligible steps.

    Mirrors RandomSkipPolicy.build_exact_random_skip_set: the quota is
    round(rate * total_steps) and is only realizable over steps with
    decision_index >= min_step. An infeasible setting must fail loudly rather
    than silently under-spend the budget.
    """


def _eligible_steps_in_window(start: int, end: int, min_step: int) -> int:
    """Count decision indices in [start, end) that are eligible (>= min_step)."""
    first_eligible = max(start, min_step)
    return max(0, end - first_eligible)


def _eligible_phase_quota(
    rate: float, total_steps: int, min_step: int, n_phases: int
) -> tuple[list[int], int, int, list[int]]:
    """Return (per_window_quota, requested_total, total_eligible, eligible_per_window).

    - requested_total = round(rate * total_steps) (the RandomSkipPolicy definition).
    - Eligible steps are those with decision_index >= min_step.
    - A window with zero eligible steps receives quota 0; its proportional share is
      absorbed by the eligible windows, so the sum of per-window quotas is exactly
      requested_total.
    - Raises InfeasibleBudgetError if requested_total exceeds total eligible steps.
    """
    requested_total = int(round(rate * total_steps))
    windows = _phase_bounds(total_steps, n_phases)
    eligible = [_eligible_steps_in_window(s, e, min_step) for (s, e) in windows]
    total_eligible = sum(eligible)

    if total_eligible <= 0:
        return [0] * n_phases, 0, 0, eligible

    if requested_total > total_eligible:
        raise InfeasibleBudgetError(
            f"Infeasible skip quota: requested {requested_total} skips "
            f"({rate:.3f} of {total_steps} steps), but only {total_eligible} steps are "
            f"eligible after min_step={min_step}. Reduce target_skip_rate, lower min_step, "
            f"or increase total_steps."
        )

    # Proportional allocation over eligible steps (largest-remainder / Hare method)
    # so that sum(per_window_quota) == requested_total exactly.
    raw = [requested_total * elig / total_eligible for elig in eligible]
    quota = [int(math.floor(r)) for r in raw]  # 0-eligible windows floor to 0
    remainder = requested_total - sum(quota)
    fracs = [raw[i] - quota[i] for i in range(n_phases)]
    for _ in range(remainder):
        # only eligible windows can absorb quota (0-eligible -> fracs -1, never chosen)
        j = max(range(n_phases), key=lambda i: (fracs[i] if eligible[i] > 0 else -1.0))
        quota[j] += 1
        fracs[j] = raw[j] - quota[j]
    return quota, requested_total, total_eligible, eligible


def _nanquantile(values: Sequence[float], q: float) -> float:
    """Deterministic quantile with the same semantics the tracker uses."""
    vals = sorted(float(v) for v in values)
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    # linear interpolation, q in [0,1]
    pos = q * (len(vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    frac = pos - lo
    return vals[lo] * (1.0 - frac) + vals[hi] * frac


class ScoreQuantileGate:
    """Pure, deterministic gate scorer.

    `score_source` is a callable(decision_index, model, inputs) -> float. The
    gate (forward predicted gain) and the control (uniform) supply different
    sources; everything else is identical, so "only the score differs" holds.
    """

    def __init__(
        self,
        *,
        rate: float,
        total_steps: int,
        min_step: int = 50,
        max_consecutive_skips: int = 4,
        n_phases: int = 4,
        score_history_window: int = 64,
        anneal_fraction: float = 0.88,
        forcing_fraction: float = 0.8,
        seed: int = 0,
        score_source,
    ):
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"rate must be in (0, 1], got {rate!r}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps!r}")
        self.rate = float(rate)
        self.total_steps = int(total_steps)
        self.min_step = int(min_step)
        self.max_consecutive_skips = int(max_consecutive_skips)
        self.n_phases = int(n_phases)
        self.score_history_window = int(score_history_window)
        self.anneal_step = int(round(self.total_steps * anneal_fraction))
        self.forcing_fraction = float(forcing_fraction)
        self.seed = int(seed)
        self._rng = random.Random(self.seed)
        self._score_source = score_source

        self._phase_windows = _phase_bounds(self.total_steps, self.n_phases)
        # Quota is defined over ELIGIBLE steps (decision_index >= min_step), matching
        # RandomSkipPolicy's build_exact_random_skip_set: requested_total =
        # round(rate * total_steps), sampled from eligible steps. A window with zero
        # eligible steps carries no quota and its share is absorbed by the eligible
        # windows (equivalently: it "passes its quota to the next" eligible window).
        self._phase_quota, self._requested_total, self._total_eligible, self._eligible_per_window = (
            _eligible_phase_quota(self.rate, self.total_steps, self.min_step, self.n_phases)
        )
        self._phase_used = [0] * self.n_phases
        self._consecutive_skips = 0
        self._terminal = False
        self._terminal_at: Optional[int] = None

        # counters used in tests / diagnostics
        self.decisions_seen = 0
        self.skipped = 0
        self.forced_skips = 0
        self.cap_blocked = 0
        self.warmup_blocked = 0
        self.anneal_skips = 0

        self._scores: Deque[float] = deque(maxlen=self.score_history_window)

    # ------------------------------------------------------------------ api
    def phase_for(self, decision_index: int) -> int:
        for i, (start, end) in enumerate(self._phase_windows):
            if start <= decision_index < end:
                return i
        return self.n_phases - 1

    def _remaining_in_phase(self, decision_index: int) -> tuple[int, int]:
        """Return (remaining_quota, remaining_steps_decision) in the phase."""
        phase = self.phase_for(decision_index)
        start, end = self._phase_windows[phase]
        remaining_steps = max(0, end - decision_index)
        remaining_quota = max(0, self._phase_quota[phase] - self._phase_used[phase])
        return remaining_quota, remaining_steps

    def note_score(self, score: float) -> None:
        """Observe a forward-computed score. Non-finite values are terminal.

        Terminal state is recorded before raising so callers can see the index
        that invalidated the run (consistent with strict-mode semantics).
        """
        if not (score == score and score != float("inf") and score != float("-inf")):
            if not self._terminal:
                self._terminal = True
                # decision_index is the last observed step index
            raise TerminalInvalidScoreError(self.decisions_seen, score)
        self._scores.append(float(score))

    def decide(self, decision_index: int, model, inputs) -> bool:
        """Run one gate decision. Returns whether to skip the backward pass."""
        self.decisions_seen += 1
        if self._terminal:
            # terminal absorbs: never skip again after a fatal score
            return False

        score = self._score_source(decision_index, model, inputs)
        if not (score == score and score != float("inf") and score != float("-inf")):
            self._terminal = True
            self._terminal_at = decision_index
            raise TerminalInvalidScoreError(decision_index, score)

        annealed = decision_index >= self.anneal_step
        if annealed:
            # annealing: score blinded by uniform; budget preserved, selection random
            score = self._rng.random()

        if decision_index < self.min_step:
            self.warmup_blocked += 1
            self._scores.append(float(score))
            return False

        # feasibility forcing: under cap-ceiling pressure, skip is forced
        remaining_quota, remaining_steps = self._remaining_in_phase(decision_index)

        # Hard invariant: never skip beyond the per-window quota. Once a phase's
        # quota is met, remaining_quota == 0 and we stop skipping in that phase.
        if remaining_quota <= 0:
            self.note_score(score)
            return False

        forcing_active = False
        if remaining_steps > 0:
            pressure = remaining_quota / remaining_steps
            forcing_active = pressure >= self.forcing_fraction

        skip = False
        if forcing_active and self._consecutive_skips < self.max_consecutive_skips:
            skip = True
            self.forced_skips += 1
        elif self._consecutive_skips >= self.max_consecutive_skips:
            # cap binds: must run a real backward to refresh
            self.cap_blocked += 1
            skip = False
        else:
            q = remaining_quota / remaining_steps if remaining_steps > 0 else 0.0
            if self._scores:
                threshold = _nanquantile(list(self._scores), q)
                if score < threshold:
                    skip = True
                elif score == threshold:
                    skip = self._rng.random() < 0.5  # tie-break
                # score > threshold -> don't skip
            # no history yet -> don't skip

        if skip:
            phase = self.phase_for(decision_index)
            self._phase_used[phase] += 1
            self.skipped += 1
            self._consecutive_skips += 1
            if annealed:
                self.anneal_skips += 1
        else:
            self._consecutive_skips = 0

        self.note_score(score)
        return skip

    def terminate_on_score(self, decision_index: int, value: float) -> None:
        self._terminal = True
        self._terminal_at = decision_index
        raise TerminalInvalidScoreError(decision_index, value)

    def phase_quotas_met(self) -> list[bool]:
        return [
            self._phase_used[i] <= self._phase_quota[i]
            and (self._phase_quota[i] - self._phase_used[i]) <= 0
            for i in range(self.n_phases)
        ]

    def diagnostics(self) -> dict:
        return {
            "rate": self.rate,
            "total_steps": self.total_steps,
            "min_step": self.min_step,
            "max_consecutive_skips": self.max_consecutive_skips,
            "n_phases": self.n_phases,
            "anneal_step": self.anneal_step,
            "phase_quota": self._phase_quota,
            "requested_total": self._requested_total,
            "total_eligible": self._total_eligible,
            "eligible_per_window": self._eligible_per_window,
            "phase_used": list(self._phase_used),
            "realized_total": self.skipped,
            "skipped": self.skipped,
            "forced_skips": self.forced_skips,
            "cap_blocked": self.cap_blocked,
            "warmup_blocked": self.warmup_blocked,
            "anneal_skips": self.anneal_skips,
            "decisions_seen": self.decisions_seen,
        "terminal": self._terminal,
        "terminal_at": self._terminal_at,
    }


# ---------------------------------------------------------------------------
# Score sources
# ---------------------------------------------------------------------------

# Upper bound from Katharopoulos & Fleuret (2018): for softmax cross-entropy,
# ||grad_w.r.t. logits|| = ||p - y||, bounded by sqrt(2) for two classes.
# mean_i ||p_i - y_i|| is exactly the proposed LER v4 score. It is bounded,
# label-noise-robust, and costs ~0 ms because the logits are already cached in
# trainer._last_real_logits (capture_logits=True).


def forward_gain_score(logits, labels) -> float:
    """mean over the batch of ||softmax(logits)_i - onehot(labels)_i||.

    The score is the bounded upper bound on the per-step gradient norm used by
    the forward-only LER gate. Returns NaN if the inputs are unusable, which
    the gate treats as terminal (strict mode).

    logits: [B, C] tensor (already detached). labels: [B] int class ids.
    """
    import torch

    if logits is None or labels is None:
        return float("nan")
    if not isinstance(logits, torch.Tensor) or not isinstance(labels, torch.Tensor):
        return float("nan")
    logits = logits.detach()
    if logits.dim() < 2 or labels.numel() == 0 or logits.shape[0] != labels.shape[0]:
        return float("nan")
    num_classes = logits.shape[1]
    with torch.no_grad():
        log = logits.float()
        log = log - log.max(dim=-1, keepdim=True).values  # stabilize
        p = log.exp()
        p = p / p.sum(dim=-1, keepdim=True)
        onehot = torch.zeros_like(p).scatter_(1, labels.view(-1, 1), 1.0)
        diff = p - onehot
        score = float(diff.pow(2).sum(dim=-1).sqrt().mean().item())
    if not math.isfinite(score):
        return float("nan")
    return score


class ScoreQuantileGatePolicy:
    """SkipPolicy plugging ScoreQuantileGate into TrueBackwardSkippingTrainer.

    `score_source` is either "gain" (forward predicted-gain bound) or "uniform"
    (i.i.d. Uniform[0,1] from `policy_seed`). Everything else — quota, phases,
    cap, forcing, annealing — is shared, so switching the source turns the
    contender into its own control.

    Reads logits from `trainer._last_real_logits` (captured when
    `online_ler_mode != "off"`, e.g. "legacy_dense" with no ler_tracker).
    """

    name = "score_quantile_gate"

    def __init__(
        self,
        *,
        target_skip_rate: float,
        total_steps: int,
        min_step: int = 50,
        max_consecutive_skips: int = 4,
        n_phases: int = 4,
        score_history_window: int = 64,
        anneal_fraction: float = 0.88,
        forcing_fraction: float = 0.8,
        policy_seed: int = 0,
        score_source: str = "gain",
    ):
        if score_source not in ("gain", "uniform"):
            raise ValueError(f"score_source must be 'gain' or 'uniform', got {score_source!r}")
        self.score_source_kind = score_source
        self._rng = random.Random(int(policy_seed))
        self._trainer_ref = None

        def _source(decision_index, model, inputs):
            if score_source == "uniform":
                return self._rng.random()
            return forward_gain_score(
                getattr(self._trainer_ref, "_last_real_logits", None),
                inputs.get("labels") if isinstance(inputs, dict) else None,
            )

        self._gate = ScoreQuantileGate(
            rate=float(target_skip_rate),
            total_steps=int(total_steps),
            min_step=int(min_step),
            max_consecutive_skips=int(max_consecutive_skips),
            n_phases=int(n_phases),
            score_history_window=int(score_history_window),
            anneal_fraction=float(anneal_fraction),
            forcing_fraction=float(forcing_fraction),
            seed=int(policy_seed),
            score_source=_source,
        )

    def should_skip(self, trainer, model, inputs) -> bool:
        self._trainer_ref = trainer
        instr = getattr(trainer, "instr", None)
        idx = (getattr(instr, "batches_seen", 0) - 1) if instr is not None else self._gate.decisions_seen
        if idx < 0:
            idx = self._gate.decisions_seen
        return bool(self._gate.decide(int(idx), trainer, inputs))

    def get_diagnostics(self) -> dict:
        d = self._gate.diagnostics()
        d["score_source"] = self.score_source_kind
        return d


# Re-exported so the matrix / harness can resolve the name string.
__all__ = ["ScoreQuantileGate", "ScoreQuantileGatePolicy", "forward_gain_score",
           "TerminalInvalidScoreError"]
