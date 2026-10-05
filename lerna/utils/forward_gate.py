"""LER v4: forward-looking LER gate.

This module defines the deficit-aware quantile gate used by the
`ScoreQuantileGatePolicy` skip policy and its uniform-random control. It is
intentionally a *pure*, CPU-testable scorer: the policy wrapper
(`ScoreQuantileGatePolicy`) is the only part that touches a live trainer.

Design (single rule, zero tuned knobs):

* 4 phase windows of equal length over `total_steps`.
* Exact per-window quota = round(rate * window_length) allocated via largest-remainder
  over ELIGIBLE steps (decision_index >= min_step), so sum(phase_quota) ==
  round(rate * total_steps) exactly.
* Deficit-aware skip probability: q = remaining_window_quota /
  remaining_window_steps. A step is skipped when its score is below the q-quantile
  of the last `score_history_window` (default 64) observed scores.
* Feasibility forcing: at each step, check whether skipping and whether a
  backward step each still leave the per-window quota reachable. If only one
  does, force that one. Otherwise the score decides. The cap (K=4 consecutive
  skips) always applies — including to forced decisions — so the quota is met
  without ever exceeding K consecutive skips, which means no leftover is
  dropped at window boundaries.
* Annealing: after `anneal_step`, selection uses Bernoulli(q) directly (not the
  score). The original score is still recorded in history; no Uniform[0,1] values
  are fed into the gain-score history, avoiding threshold skew.
* cap: never exceed `max_consecutive_skips` consecutive skips for any decision,
  forcing or not; if the run is at the cap a backward step is forced.
* min_step: never skip before `min_step` decisions. Loss rises during warmup
  and the model is far from convergence, so gain scores are high but noisy and
  not representative of the steady-state distribution. min_step should be
  >= warmup_steps to avoid skipping during the LR warmup phase, where skip
  effects differ. The smoke test uses min_step=50 with warmup_steps=12
  (warmup ends before min_step).
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

import torch


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


def max_skips_possible(r: int, k: int) -> int:
    """Exact maximum skips in r decisions under cap K (max K consecutive).

    With cap K, the densest skip pattern is K skips followed by 1 forced
    backward step, repeating every K+1 decisions. So:
        F(r) = K * floor(r / (K+1)) + min(r mod (K+1), K)
    """
    if r <= 0:
        return 0
    return k * (r // (k + 1)) + min(r % (k + 1), k)


def max_skips_possible_with_consec(r: int, k: int, c: int) -> int:
    """Exact maximum skips in r decisions under cap K, starting with c consecutive.

    Unlike max_skips_possible (which assumes c=0), this accounts for a skip run
    already in progress. When c > 0, the first `k - c` skips fill the run to the
    cap, then a forced backward resets to 0. Starting from 0, the standard
    K-skip + 1-backward pattern applies.

    Args:
        r: remaining decision steps (excluding the current step).
        k: cap (max consecutive skips).
        c: current consecutive skip count (0 <= c <= k). When c == k the cap
           fires on the next decision (forcing a backward), so the remaining
           r-1 steps start fresh from 0.
    """
    if r <= 0:
        return 0
    if c >= k:
        # Cap fires on the first of the r remaining steps: forced backward,
        # then r-1 steps unconstrained from 0 consecutive.
        return max_skips_possible(r - 1, k)
    free = k - c  # skips before hitting the cap from the current streak
    first_skips = min(free, r)
    if r <= free:
        return r
    # After first_skips, a forced backward consumes 1 step, then repeat from 0.
    return first_skips + max_skips_possible(r - first_skips - 1, k)


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
        min_step: int = 392,
        max_consecutive_skips: int = 4,
        n_phases: int = 4,
        score_history_window: int = 64,
        anneal_fraction: float = 0.88,
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
        # Hard feasibility: the requested quota must be achievable under the cap.
        # With cap K, max_skips_possible(n_elig, K) is the most skips possible
        # across all eligible steps. If the quota exceeds this, the run would
        # dead-lock (forced skips hit the cap and can never catch up). Reject
        # loudly rather than silently under-spend.
        max_achievable = max_skips_possible(self._total_eligible, self.max_consecutive_skips)
        if self._requested_total > max_achievable:
            raise InfeasibleBudgetError(
                f"Infeasible skip quota under cap: requested {self._requested_total} skips "
                f"({self.rate:.3f} of {self.total_steps}), but max achievable under "
                f"max_consecutive_skips={self.max_consecutive_skips} is {max_achievable} "
                f"over {self._total_eligible} eligible steps. Lower the skip rate, "
                f"raise the cap, or increase total_steps."
            )
        self._phase_used = [0] * self.n_phases
        self._consecutive_skips = 0
        self._longest_skip_run = 0
        self._terminal = False
        self._terminal_at: Optional[int] = None

        # counters used in tests / diagnostics
        self.decisions_seen = 0
        self.skipped = 0
        self.forced_skips = 0
        self.cap_blocked = 0
        self.warmup_blocked = 0
        self.anneal_skips = 0
        self.global_forced_skips = 0

        self._scores: Deque[float] = deque(maxlen=self.score_history_window)

        # per-window skip-position diagnostics (bug d): track where skips land
        # within each phase window to detect front-loading or back-loading bias.
        self._phase_skip_positions: list[list[int]] = [[] for _ in range(self.n_phases)]

    def _record_skip_position(self, decision_index: int, skip: bool) -> None:
        """Record the phase-relative position of skips for parity diagnostics."""
        if not skip:
            return
        phase = self.phase_for(decision_index)
        start, _ = self._phase_windows[phase]
        rel = decision_index - start
        self._phase_skip_positions[phase].append(rel)

    # ------------------------------------------------------------------ api
    def phase_for(self, decision_index: int) -> int:
        for i, (start, end) in enumerate(self._phase_windows):
            if start <= decision_index < end:
                return i
        return self.n_phases - 1

    def _remaining_in_phase(self, decision_index: int) -> tuple[int, int]:
        """Return (remaining_quota, remaining_steps) in the current phase."""
        phase = self.phase_for(decision_index)
        start, end = self._phase_windows[phase]
        remaining_steps = max(0, end - decision_index)
        remaining_quota = max(0, self._phase_quota[phase] - self._phase_used[phase])
        return remaining_quota, remaining_steps

    def _remaining_global(self, decision_index: int) -> tuple[int, int]:
        """Return (total_remaining_quota, total_remaining_eligible_steps) across
        all phases at or after the current decision index.

        This is the GLOBAL feasibility basis for per-step forcing: the sum of
        per-phase F values is NOT the same as F of the total, so we must check
        global reachability at each step to guarantee realized == requested.
        """
        total_quota = 0
        total_steps = 0
        for phase in range(self.phase_for(decision_index), self.n_phases):
            start, end = self._phase_windows[phase]
            if end <= decision_index:
                continue
            # eligible steps in this phase (from decision_index onward)
            elig_start = max(start, self.min_step)
            if end <= elig_start:
                continue
            elig_in_phase = max(0, end - max(decision_index, elig_start))
            total_steps += elig_in_phase
            total_quota += max(0, self._phase_quota[phase] - self._phase_used[phase])
        return total_quota, total_steps

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

        # B3: Do not compute score before (min_step - score_history_window).
        # S1 (||p-y||) and S2 (alignment) are NaN before the warmup window ends
        # and before the AlignmentProbe hook is attached, so computing them here
        # would terminal the run at step 0. The score is only computed once
        # decisions are eligible.
        early_return = max(self.min_step - self.score_history_window, 0)
        if decision_index < early_return:
            # B13: reset streak on non-skip
            self._consecutive_skips = 0
            return False

        score = self._score_source(decision_index, model, inputs)
        if not (score == score and score != float("inf") and score != float("-inf")):
            self._terminal = True
            self._terminal_at = decision_index
            raise TerminalInvalidScoreError(decision_index, score)

        annealed = decision_index >= self.anneal_step

        if decision_index < self.min_step:
            # Record score into history so the quantile window is full when
            # min_step is reached, but still never skip.
            self.warmup_blocked += 1
            # B13: reset streak on non-skip
            self._consecutive_skips = 0
            self.note_score(score)
            return False

        remaining_quota, remaining_steps = self._remaining_in_phase(decision_index)

        # Hard invariant: never skip beyond the per-window quota. Once a phase's
        # quota is met, remaining_quota == 0 and we stop skipping in that phase.
        if remaining_quota <= 0:
            # B13: reset streak on non-skip
            self._consecutive_skips = 0
            self.note_score(score)
            return False

        K = self.max_consecutive_skips

        # At the cap: must run a real backward step to avoid exceeding K.
        if self._consecutive_skips >= K:
            self.cap_blocked += 1
            self._consecutive_skips = 0
            self.note_score(score)
            return False

        # Exact feasibility check:
        # We check BOTH per-phase and global reachability to guarantee
        # realized == requested at end-of-run.
        #
        # Per-phase (local): if skipping now would make this phase's quota
        # unreachable, or if not skipping would, we force the other choice.
        #
        # Global (total): the sum of per-phase F values is NOT the same as
        # F(total), so we must also check global reachability. If the current
        # phase is locally infeasible (cannot meet its quota either way), we
        # fall back to the global check: force a skip if skipping is still
        # globally feasible, or force a backward if not skipping is still
        # globally feasible.
        #
        # - F(r, K) = K * floor(r/(K+1)) + min(r mod (K+1), K)
        # - F_with_consec(r, K, c) = max skips in r steps starting with c
        #   consecutive (c skips already in the current run).
        #
        # per_phase_skip_possible: can this phase still be satisfied if we skip now?
        #   remaining_quota - 1 skips in remaining_steps - 1 steps (from c+1 consec)
        per_phase_skip_possible = (
            max_skips_possible_with_consec(
                remaining_steps - 1, K, self._consecutive_skips + 1
            )
            >= remaining_quota - 1
        )
        # per_phase_real_possible: can this phase still be satisfied if we backward now?
        #   remaining_quota skips in remaining_steps - 1 steps, starting from 0 consec
        per_phase_real_possible = max_skips_possible(remaining_steps - 1, K) >= remaining_quota

        # Global check over ALL remaining phases (for cross-phase compensation).
        total_quota, total_steps = self._remaining_global(decision_index)
        # global_skip_possible: can the TOTAL quota still be met if we skip now?
        global_skip_possible = (
            max_skips_possible_with_consec(
                total_steps - 1, K, self._consecutive_skips + 1
            )
            >= total_quota - 1
        ) if total_quota > 0 else False
        # global_real_possible: can the TOTAL quota still be met if we backward now?
        global_real_possible = max_skips_possible(total_steps - 1, K) >= total_quota if total_quota > 0 else True

        # Determine if we're forced in either direction.
        # Priority:
        # 1. Cap: at the consecutive limit, must backward.
        # 2. Per-phase forcing: local check dominates when it's decisive.
        # 3. Global forcing: if per-phase is infeasible both ways, use global.
        if not per_phase_skip_possible and not per_phase_real_possible:
            # Phase is locally infeasible (cannot meet its quota either way).
            # Fall back to global reachability to minimize total deficit.
            if not global_skip_possible and not global_real_possible:
                # Globally infeasible too: cannot recover. Don't skip (conservative).
                # B13: reset streak on non-skip
                self._consecutive_skips = 0
                self.note_score(score)
                return False
            elif not global_real_possible:
                # Not skipping is globally infeasible; force skip (minimize deficit).
                self.global_forced_skips += 1
                phase = self.phase_for(decision_index)
                self._phase_used[phase] += 1
                self.skipped += 1
                self._consecutive_skips += 1
                self._longest_skip_run = max(self._longest_skip_run, self._consecutive_skips)
                self._record_skip_position(decision_index, True)
                self.note_score(score)
                return True
            else:
                # global_skip_possible is False (skipping globally infeasible):
                # force backward to preserve global quota.
                self.global_forced_skips += 1
                # B13: reset streak on non-skip
                self._consecutive_skips = 0
                self.note_score(score)
                return False
        elif not per_phase_skip_possible:
            # Skipping would make this phase's quota unreachable; force backward.
            self.forced_skips += 1
            # B13: reset streak on non-skip
            self._consecutive_skips = 0
            self.note_score(score)
            return False
        elif not per_phase_real_possible:
            # Not skipping would make this phase's quota unreachable; force skip.
            self.forced_skips += 1
            phase = self.phase_for(decision_index)
            self._phase_used[phase] += 1
            self.skipped += 1
            self._consecutive_skips += 1
            self._longest_skip_run = max(self._longest_skip_run, self._consecutive_skips)
            self._record_skip_position(decision_index, True)
            self.note_score(score)
            return True

        # Both are feasible: the score decides.
        skip = False
        if annealed:
            # (b) Annealing: use Bernoulli(q) directly. Do NOT feed Uniform[0,1]
            # into the score history (which holds gain scores in [0, sqrt2]);
            # that would skew the quantile threshold for ~window steps.
            q = remaining_quota / remaining_steps if remaining_steps > 0 else 0.0
            skip = self._rng.random() < q
            if skip:
                self.anneal_skips += 1
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
            self._longest_skip_run = max(self._longest_skip_run, self._consecutive_skips)
            self._record_skip_position(decision_index, True)
        else:
            self._consecutive_skips = 0

        self.note_score(score)  # always append the original score, not uniform
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
        # per-window skip-position diagnostics (bug d): median and spread of
        # phase-relative positions where skips land, for parity comparison.
        window_positions = []
        for positions in self._phase_skip_positions:
            if positions:
                sorted_pos = sorted(positions)
                n = len(sorted_pos)
                median_pos = sorted_pos[n // 2] if n % 2 == 1 else (
                    sorted_pos[n // 2 - 1] + sorted_pos[n // 2]
                ) / 2
                window_positions.append({
                    "skip_count": n,
                    "median_rel_position": median_pos,
                    "min_rel_position": sorted_pos[0],
                    "max_rel_position": sorted_pos[-1],
                })
            else:
                window_positions.append({"skip_count": 0, "median_rel_position": None})
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
            "anneal_start": self.anneal_step,
            "global_forced_skips": self.global_forced_skips,
            "longest_skip_run": self._longest_skip_run,
            "decisions_seen": self.decisions_seen,
            "phase_skip_positions": window_positions,
            "terminal": self._terminal,
            "terminal_at": self._terminal_at,
        }


# ---------------------------------------------------------------------------
# Score sources
# ---------------------------------------------------------------------------

# Upper bound from Katharopoulos & Fleuret (2018): for softmax cross-entropy,
# ||grad_w.r.t. logits|| = ||p - y||, bounded by sqrt(2) for two classes.
# mean_i ||p_i - y_i|| is exactly the proposed LER v4 score (S1). It is bounded,
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


# ---------------------------------------------------------------------------
# S2: last-layer alignment score
# ---------------------------------------------------------------------------
#
# Formula:  s_B = <G_B, G_P ⊘ (sqrt(v) + eps)>
#
# - G_B = (1/B) * sum_i (p_i - y_i) h_i^T   (per-batch gradient w.r.t. classifier.weight)
#   where h_i is the hidden state entering the classifier (captured via a
#   forward-pre-hook on the classifier module), p_i = softmax(logits_i),
#   y_i = one-hot label.  Shape: [num_labels, hidden_dim].
#
# - G_P is the same gradient computed on a 256-example score probe P_s, run in
#   eval mode every 64 steps in fp32.
#
# - v is AdamW's exp_avg_sq for classifier.weight, shape [num_labels, hidden_dim].
#
# The score is a first-order prediction of how much the probe loss drops if
# this batch's update is applied: it is the dot product of the batch gradient
# with the probe gradient, each coordinate normalized by sqrt(v)+eps, which is
# exactly the sign/direction of the AdamW update step on classifier.weight.
#
# Note (recorded in pre-reg): loss rescaling (the magnitude of the gradient)
# is NOT used — only direction. Under AdamW the update is approximately
# scale-invariant, so the alignment of directions is what matters.


class AlignmentProbe:
    """Capture the hidden state entering the classifier and compute S2 scores.

    Attaches a forward-pre-hook to ``model.classifier`` (the Linear layer that
    maps pooled hidden state -> logits in ModernBertForSequenceClassification).
    The hook stores the most recent input tensor so that S2 scores can be
    computed from cached logits + labels without an extra forward pass.
    """

    def __init__(self, model, hook_module_name: str = "classifier"):
        self.model = model
        self.hook_module_name = hook_module_name
        self._last_hidden: Optional[torch.Tensor] = None
        self._hook_handle = None
        self._attach()

    def _attach(self) -> None:
        module = getattr(self.model, self.hook_module_name, None)
        if module is None:
            raise AttributeError(
                f"Model has no attribute '{self.hook_module_name}'; cannot attach "
                f"forward-pre-hook for alignment score."
            )

        def _pre_hook(_module, args):
            # args[0] is the input tensor entering the Linear layer (pooled_output).
            self._last_hidden = args[0].detach().clone()

        self._hook_handle = module.register_forward_pre_hook(_pre_hook)

    def remove(self) -> None:
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    @property
    def last_hidden(self) -> Optional[torch.Tensor]:
        return self._last_hidden


def _classifier_grad_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    hidden: torch.Tensor,
) -> torch.Tensor:
    """Compute G = (1/B) * sum_i (p_i - y_i) h_i^T, the gradient of
    softmax cross-entropy loss w.r.t. the classifier weight.

    Args:
        logits: [B, C] detached logits tensor.
        labels: [B] integer class ids.
        hidden: [B, H] hidden state entering the classifier.

    Returns:
        [C, H] tensor — the mean per-example gradient w.r.t. classifier.weight.
    """
    with torch.no_grad():
        p = logits.float().softmax(dim=-1)
        onehot = torch.zeros_like(p).scatter_(1, labels.view(-1, 1), 1.0)
        diff = p - onehot  # [B, C]
        # G = (1/B) * sum_i diff_i * h_i^T  =>  [C, H]
        G = (diff.t() @ hidden.float()) / labels.shape[0]
        return G


def alignment_score(
    trainer,
    model,
    inputs,
    probe_fn,
    adam_state_key: str = "classifier.weight",
    eps: float = 1e-8,
) -> float:
    """Compute S2: the last-layer alignment score.

    s_B = <G_B, G_P ⊘ (sqrt(v) + eps)>

    where G_B is the current batch's gradient w.r.t. classifier.weight (derived
    from cached logits + hidden state), G_P is the probe gradient (recomputed by
    ``probe_fn``), and v is AdamW's exp_avg_sq for classifier.weight.

    Returns NaN if any component is unavailable (gate treats as terminal).
    """
    import torch

    logits = getattr(trainer, "_last_real_logits", None)
    if logits is None:
        logits = getattr(trainer, "_last_logits", None)
    if logits is None:
        return float("nan")

    labels = inputs.get("labels") if isinstance(inputs, dict) else None
    if labels is None:
        return float("nan")

    # Retrieve the hidden state captured by AlignmentProbe's forward-pre-hook.
    probe = getattr(trainer, "_alignment_probe", None)
    if probe is None:
        return float("nan")
    hidden = probe.last_hidden
    if hidden is None:
        return float("nan")

    with torch.no_grad():
        logits_f = logits.float().detach()
        if logits_f.shape[0] != labels.shape[0] or logits_f.shape[0] != hidden.shape[0]:
            return float("nan")

        G_B = _classifier_grad_from_logits(logits_f, labels, hidden)

        # G_P from the probe (refreshed every 64 steps by probe_fn)
        G_P = probe_fn()
        if G_P is None:
            return float("nan")
        G_P = G_P.float()

        # v: AdamW exp_avg_sq for classifier.weight
        optimizer = getattr(trainer, "optimizer", None) or getattr(trainer, "_optimizer", None)
        if optimizer is None:
            return float("nan")
        # Resolve the parameter by name from the model
        param = None
        for name, p in model.named_parameters():
            if name == adam_state_key:
                param = p
                break
        if param is None:
            return float("nan")
        state = optimizer.state.get(param, {})
        v = state.get("exp_avg_sq")
        if v is None:
            return float("nan")
        v = v.float()

        if G_B.shape != G_P.shape or G_B.shape != v.shape:
            return float("nan")

        s = _s2_alignment_core(G_B, G_P, v, eps)
        return s



def _s2_alignment_core(G_B, G_P, v, eps: float = 1e-8) -> float:
    """B12: shared S2 alignment computation used by both stage0 and stage2.

    s_B = <G_B, G_P ⊘ (sqrt(v) + eps)>

    where G_B is the current batch's gradient w.r.t. classifier.weight,
    G_P is the probe gradient, and v is AdamW's exp_avg_sq for classifier.weight.

    Returns NaN if shapes mismatch or result is non-finite.
    """
    if G_B.shape != G_P.shape or G_B.shape != v.shape:
        return float("nan")
    normalized = G_P / (v.sqrt() + eps)
    s = float(torch.dot(G_B.flatten(), normalized.flatten()).item())
    if not math.isfinite(s):
        return float("nan")
    return s


class ScoreQuantileGatePolicy:
    """SkipPolicy plugging ScoreQuantileGate into TrueBackwardSkippingTrainer.

    `score_source` is "s1" (forward predicted-gain bound, ||p-y||),
    "s2" (last-layer alignment score), or "uniform" (i.i.d. Uniform[0,1] from
    `policy_seed`). Everything else — quota, phases, cap, feasibility forcing,
    annealing — is shared, so switching the source turns the contender into
    its own control.

    Reads logits from `trainer._last_real_logits` (captured when
    `online_ler_mode != "off"`, e.g. "legacy_dense" with no ler_tracker).
    """

    name = "score_quantile_gate"

    def __init__(
        self,
        *,
        target_skip_rate: float,
        total_steps: int,
        min_step: int = 392,
        max_consecutive_skips: int = 4,
        n_phases: int = 4,
        score_history_window: int = 64,
        anneal_fraction: float = 0.88,
        policy_seed: int = 0,
        score_source: str = "s1",
        alignment_probe_fn=None,
        alignment_adam_state_key: str = "classifier.weight",
        alignment_eps: float = 1e-8,
    ):
        if score_source not in ("s1", "s2", "uniform"):
            raise ValueError(
                f"score_source must be 's1', 's2', or 'uniform', got {score_source!r}"
            )
        self.score_source_kind = score_source
        self._rng = random.Random(int(policy_seed))
        self._trainer_ref = None
        self.alignment_probe_fn = alignment_probe_fn
        self.alignment_adam_state_key = alignment_adam_state_key
        self.alignment_eps = float(alignment_eps)

        def _source(decision_index, model, inputs):
            if score_source == "uniform":
                return self._rng.random()
            elif score_source == "s1":
                return forward_gain_score(
                    getattr(self._trainer_ref, "_last_real_logits", None),
                    inputs.get("labels") if inputs is not None and hasattr(inputs, "get") else None,
                )
            else:  # s2
                return alignment_score(
                    trainer=self._trainer_ref,
                    model=model,
                    inputs=inputs,
                    probe_fn=self.alignment_probe_fn,
                    adam_state_key=self.alignment_adam_state_key,
                    eps=self.alignment_eps,
                )

        self._gate = ScoreQuantileGate(
            rate=float(target_skip_rate),
            total_steps=int(total_steps),
            min_step=int(min_step),
            max_consecutive_skips=int(max_consecutive_skips),
            n_phases=int(n_phases),
            score_history_window=int(score_history_window),
            anneal_fraction=float(anneal_fraction),
            seed=int(policy_seed) + 1,
            score_source=_source,
        )

    def setup_probe(self, trainer, model) -> None:
        """Attach the AlignmentProbe before training begins (B3).

        For score_source='s2', the forward-pre-hook on model.classifier must
        be installed before train() so that hidden states are captured from the
        very first forward pass.  Attaching lazily on the first should_skip()
        risks missing early forward passes and producing a None last_hidden
        when the first eligible decision at min_step fires.
        """
        if self.score_source_kind == "s2":
            trainer._alignment_probe = AlignmentProbe(model)

    def should_skip(self, trainer, model, inputs) -> bool:
        self._trainer_ref = trainer
        if self.score_source_kind == "s2":
            # B3: probe should already be attached via setup_probe() before train().
            # If not, attach now as a fallback.
            if not hasattr(trainer, "_alignment_probe"):
                trainer._alignment_probe = AlignmentProbe(model)
            # F8: fail hard at min_step if probe not attached or score non-finite
            instr = getattr(trainer, "instr", None)
            idx = (getattr(instr, "batches_seen", 0) - 1) if instr is not None else self._gate.decisions_seen
            if idx >= self._gate.min_step:
                probe = getattr(trainer, "_alignment_probe", None)
                assert probe is not None, (
                    "FATAL: score_source='s2' but _alignment_probe is None at "
                    f"step {idx}"
                )
                assert probe.last_hidden is not None, (
                    "FATAL: score_source='s2' but AlignmentProbe.last_hidden is None "
                    f"at step {idx} — hook not firing"
                )
                if self.alignment_probe_fn is None:
                    raise RuntimeError(
                        "FATAL: score_source='s2' but alignment_probe_fn is not set. "
                        "S2 alignment score will be NaN."
                    )
        instr = getattr(trainer, "instr", None)
        idx = (getattr(instr, "batches_seen", 0) - 1) if instr is not None else self._gate.decisions_seen
        if idx < 0:
            idx = self._gate.decisions_seen
        return bool(self._gate.decide(int(idx), trainer, inputs))

    def get_diagnostics(self) -> dict:
        d = self._gate.diagnostics()
        d["score_source"] = self.score_source_kind
        d["alignment_eps"] = self.alignment_eps
        d["alignment_adam_state_key"] = self.alignment_adam_state_key
        return d


# Re-exported so the matrix / harness can resolve the name string.
__all__ = [
    "ScoreQuantileGate",
    "ScoreQuantileGatePolicy",
    "forward_gain_score",
    "alignment_score",
    "AlignmentProbe",
    "TerminalInvalidScoreError",
    "InfeasibleBudgetError",
]
