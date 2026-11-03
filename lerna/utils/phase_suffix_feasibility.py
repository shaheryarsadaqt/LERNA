"""Exact phase-suffix feasibility checker using backward dynamic programming.

State semantics (per the reviewed reference algorithm):

  t: next scheduled minibatch index, 0 <= t < total_steps for action queries.
  c: current consecutive-skip streak.
  q: required remaining skips in the CURRENT phase.

Bit q of F[t][c] means: at step t with incoming skip streak c, we can perform
exactly q further skips in the current phase and satisfy every later fixed
phase quota. Earlier phase quotas must already be satisfied.

Within a phase:
  - Retain resets streak to zero and leaves q unchanged.
  - Skip is allowed only after warmup and below the hard cap; it increases
    streak and consumes one required skip.
  - At a phase boundary, current remaining quota must be zero, then the next
    phase starts with its full quota. Boundary feasibility depends on the
    entering streak.
  - End of horizon requires every quota and total to be exact.

Backward recurrence on bitsets:

  F[t,c] = F[t+1,0] OR (F[t+1,c+1] << 1)

Omit the skip term during warmup or at c=K, and mask to the current phase's
quota range. At a boundary, use bit zero only when the next phase can realize
its full quota with the same streak. The final horizon accepts zero remaining
quota.

This is exact: terminal/boundary states encode the exact constraints; the two
recurrence branches enumerate all legal next actions; backward induction covers
every finite suffix. Reject when the initial bit is false; commit only to a
child with a true bit. Under valid state transitions, at least one action
remains until completion. This proves scheduling feasibility, not model quality
or convergence.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


class PhaseSuffixInfeasibleError(RuntimeError):
    """Raised when no feasible completion exists from a given gate state.

    Carries the full state (t, c, q, phase) so the caller can record diagnostics
    and stop the run without silent fallback or quota/cap relaxation.
    """

    def __init__(
        self,
        *,
        t: int,
        c: int,
        q: int,
        phase: int,
        phase_quota: int,
        phase_used: int,
        message: str = "no feasible completion",
    ):
        self.t = t
        self.c = c
        self.q = q
        self.phase = phase
        self.phase_quota = phase_quota
        self.phase_used = phase_used
        super().__init__(
            f"{message} at t={t}, c={c}, q={q}, phase={phase} "
            f"(phase_quota={phase_quota}, phase_used={phase_used})"
        )


def _validate_int(value, name: str, min_v: int, max_v: int) -> int:
    """Validate an integer field before coercion. Reject booleans and fractions."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got boolean {value!r}")
    if isinstance(value, float) and value != int(value):
        raise ValueError(
            f"{name} must be an integer, got fractional {value!r}"
        )
    ival = int(value)
    if ival < min_v or ival > max_v:
        raise ValueError(
            f"{name} must be in [{min_v}, {max_v}], got {ival}"
        )
    return ival


def _validate_finite_float(value, name: str, min_v: float, max_v: float) -> float:
    """Validate a finite float in [min_v, max_v]. Reject booleans."""
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a finite number, got boolean {value!r}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite number, got {value!r}")
    if value != value or value == float("inf") or value == float("-inf"):
        raise ValueError(f"{name} must be finite, got {value!r}")
    fval = float(value)
    if fval < min_v or fval > max_v:
        raise ValueError(
            f"{name} must be in [{min_v}, {max_v}], got {fval!r}"
        )
    return fval


def phase_bounds(total_steps: int, n_phases: int) -> List[Tuple[int, int]]:
    """Equal-length phases; last absorbs the remainder so the whole horizon is covered."""
    bounds: List[Tuple[int, int]] = []
    step = 0
    for i in range(n_phases):
        end = total_steps if i == n_phases - 1 else step + total_steps // n_phases
        bounds.append((step, end))
        step = end
    return bounds


class PhaseSuffixFeasibility:
    """Exact suffix feasibility for phase-quota scheduling with hard consecutive-skip cap.

    Builds backward DP tables once on CPU. Query overhead per decision is O(P)
    (phase lookup scans phases) plus O(1) bitset lookup within the phase.
    Python big-integer bit operations scale with quota in machine words, not
    strict constant time.

    Memory is O((K+1) * sum_j N_j * (Q_j+1)) bits plus Python container overhead.
    A pre-allocation guard rejects configurations that would exceed
    MAX_ALLOWED_CELLS to avoid unbounded memory/work. No approximate fallback.
    """

    # Conservative pre-allocation guard: ~200M cells max, accounting for Python
    # container overhead and big-integer width. Fail clearly before unsafe
    # allocation; no approximate fallback. Supports QNLI-sized configs
    # (T~6.5K, P=4, Q~3.9K -> ~32M cells; P=1 -> ~128M cells) while rejecting
    # pathological cases.
    MAX_ALLOWED_CELLS = 200_000_000

    def __init__(
        self,
        total_steps: int,
        n_phases: int,
        phase_quotas: List[int],
        min_step: int,
        max_consecutive_skips: int,
    ):
        # Validate before any coercion (reject booleans/fractions).
        self.total_steps = _validate_int(total_steps, "total_steps", 1, 1 << 30)
        self.n_phases = _validate_int(n_phases, "n_phases", 1, self.total_steps)
        self.min_step = _validate_int(min_step, "min_step", 0, self.total_steps)
        self.K = _validate_int(max_consecutive_skips, "max_consecutive_skips", 1, 1 << 20)

        # Validate quotas
        if len(phase_quotas) != self.n_phases:
            raise ValueError(
                f"phase_quotas must have length {self.n_phases}, "
                f"got {len(phase_quotas)}"
            )
        self.phase_quotas = []
        for i, q in enumerate(phase_quotas):
            qi = _validate_int(q, f"phase_quotas[{i}]", 0, self.total_steps)
            self.phase_quotas.append(qi)

        self.bounds = phase_bounds(self.total_steps, self.n_phases)

        # Pre-allocation guard: estimate cells and reject unsafe configurations.
        estimated_cells = 0
        for p in range(self.n_phases):
            start, end = self.bounds[p]
            phase_len = end - start
            Q = self.phase_quotas[p]
            estimated_cells += (phase_len + 1) * (self.K + 1) * (Q + 1)
        if estimated_cells > self.MAX_ALLOWED_CELLS:
            raise ValueError(
                f"PhaseSuffixFeasibility would allocate ~{estimated_cells:,} cells, "
                f"exceeding MAX_ALLOWED_CELLS={self.MAX_ALLOWED_CELLS:,}. "
                f"Reduce T, P, K, or quotas."
            )

        # Build DP tables once. Process phases backward; self.F[p] is a list over
        # offset 0..phase_len (inclusive of boundary), each entry is a list over
        # c 0..K of int bitsets.
        self.F: List[List[List[int]]] = []
        self._build()

    # ------------------------------------------------------------------ build

    def _build(self) -> None:
        P = self.n_phases
        K = self.K

        # Process phases backward so later phases are available as boundary seeds.
        # Collect in reverse order, then reverse so self.F[0] = phase 0.
        reversed_F: List[List[List[int]]] = []

        for p in range(P - 1, -1, -1):
            start, end = self.bounds[p]
            Q = self.phase_quotas[p]
            phase_len = end - start

            # phase_F[offset][c] = bitset over q in [0, Q]
            phase_F: List[List[int]] = [
                [0] * (K + 1) for _ in range(phase_len + 1)
            ]

            # --- boundary at end of phase -------------------------------------
            if p == P - 1:
                # Last phase: at horizon end, need q == 0 (no more phases).
                for c in range(K + 1):
                    phase_F[phase_len][c] = 1  # bit 0 set
            else:
                # At boundary, current phase's q must be 0, then next phase starts
                # with its full quota. Bit 0 of F[end][c] = bit Q_next of
                # F_next[start_next][c]. reversed_F[P-1-p] holds phase p+1's table
                # (phases are built backward: P-1, P-2, ..., 0, so reversed_F[0]
                # is phase P-1, reversed_F[P-1-p] is phase p+1).
                next_Q = self.phase_quotas[p + 1]
                next_F = reversed_F[P - p - 2]  # phase p+1 table
                next_offset_0 = next_F[0]
                for c in range(K + 1):
                    phase_F[phase_len][c] = (
                        1 if ((next_offset_0[c] >> next_Q) & 1) else 0
                    )

            # --- within-phase backward recurrence -----------------------------
            mask = (1 << (Q + 1)) - 1  # valid q range [0, Q]
            for offset in range(phase_len - 1, -1, -1):
                t = start + offset
                for c in range(K + 1):
                    bitset = 0

                    # Retain: always legal; streak -> 0; q unchanged.
                    bitset |= phase_F[offset + 1][0]

                    # Skip: legal if after warmup, below cap, and q > 0.
                    if t >= self.min_step and c < K:
                        bitset |= phase_F[offset + 1][c + 1] << 1

                    phase_F[offset][c] = bitset & mask

            reversed_F.append(phase_F)

        # Reverse so self.F[0] corresponds to phase 0.
        self.F = list(reversed(reversed_F))

    # ------------------------------------------------------------------ query

    def phase_for(self, t: int) -> int:
        for p, (start, end) in enumerate(self.bounds):
            if start <= t < end:
                return p
        return self.n_phases - 1

    def is_feasible_at(self, t: int, c: int, q: int) -> bool:
        """Whether from step t with streak c and q skips still needed in the
        current phase, a feasible completion of the entire remaining horizon
        exists.

        Supports terminal t=T: only q=0 is feasible (all phases complete).
        Rejects t<0 and t>T.
        """
        if isinstance(t, bool) or isinstance(c, bool) or isinstance(q, bool):
            raise TypeError("t, c, q must be integers, not booleans")
        if not isinstance(t, int) or not isinstance(c, int) or not isinstance(q, int):
            raise TypeError("t, c, q must be integers")
        if t < 0 or t > self.total_steps:
            raise ValueError(
                f"t must be in [0, {self.total_steps}], got {t}"
            )
        if c < 0 or c > self.K:
            raise ValueError(
                f"c must be in [0, {self.K}], got {c}"
            )
        if q < 0:
            return False
        p = self.phase_for(t) if t < self.total_steps else self.n_phases - 1
        start, _ = self.bounds[p]
        Q = self.phase_quotas[p]
        if q > Q:
            return False
        # At terminal t=T, only q==0 is feasible.
        if t == self.total_steps:
            return q == 0
        offset = t - start
        return bool((self.F[p][offset][c] >> q) & 1)

    def is_action_feasible(self, t: int, c: int, q: int, action: int) -> bool:
        """Whether taking action (0=retain, 1=skip) at step t with streak c and
        q skips still needed leaves a feasible completion.

        Handles phase boundaries: crossing into a new phase requires q to be
        exactly satisfied (0) before advancing to the next phase's full quota.
        Rejects t<0, t>=T, and invalid action values.
        """
        if isinstance(t, bool) or isinstance(c, bool) or isinstance(q, bool) or isinstance(action, bool):
            raise TypeError("t, c, q, action must be integers, not booleans")
        if not isinstance(t, int) or not isinstance(c, int) or not isinstance(q, int) or not isinstance(action, int):
            raise TypeError("t, c, q, action must be integers")
        if t < 0 or t >= self.total_steps:
            raise ValueError(
                f"t must be in [0, {self.total_steps - 1}], got {t}"
            )
        if c < 0 or c > self.K:
            raise ValueError(f"c must be in [0, {self.K}], got {c}")
        if q < 0:
            return False
        p = self.phase_for(t)
        start, end = self.bounds[p]
        Q = self.phase_quotas[p]
        if q > Q:
            return False
        offset = t - start

        # Compute resulting state after the action.
        if action == 0:
            # Retain: streak -> 0; q unchanged.
            new_c = 0
            new_q = q
        elif action == 1:
            # Skip: check preconditions.
            if t < self.min_step:
                return False
            if c >= self.K:
                return False
            if q <= 0:
                return False
            new_c = c + 1
            new_q = q - 1
        else:
            raise ValueError(f"invalid action: {action}")

        next_t = t + 1
        if next_t < end:
            # Still in the same phase.
            return self.is_feasible_at(next_t, new_c, new_q)

        # Crossing a phase boundary: current phase's remaining quota must be 0.
        if new_q != 0:
            return False

        if p == self.n_phases - 1:
            # Last phase: at horizon end, q == 0 means done.
            return True

        # Next phase starts with its full quota and the resulting streak.
        return self.is_feasible_at(next_t, new_c, self.phase_quotas[p + 1])

    def get_feasible_actions(self, t: int, c: int, q: int) -> List[int]:
        """Return the list of feasible actions (0=retain, 1=skip) at this state."""
        actions: List[int] = []
        if self.is_action_feasible(t, c, q, 0):
            actions.append(0)
        if self.is_action_feasible(t, c, q, 1):
            actions.append(1)
        return actions

    def initial_feasible(self) -> bool:
        """Whether the initial configuration (step 0, streak 0, full quota Q_0)
        is feasible."""
        if self.n_phases == 0:
            return True
        Q0 = self.phase_quotas[0]
        return self.is_feasible_at(0, 0, Q0)

    def witness(self) -> Optional[List[int]]:
        """Return a feasible skip/backward schedule (list of 0/1) if one exists,
        else None. Greedy: prefer retain when both actions are feasible."""
        if not self.initial_feasible():
            return None

        P = self.n_phases
        schedule: List[int] = []
        t = 0
        c = 0
        p = 0
        q = self.phase_quotas[0]

        while t < self.total_steps:
            start, end = self.bounds[p]
            # Greedy: try retain first, then skip.
            if self.is_action_feasible(t, c, q, 0):
                schedule.append(0)
                c = 0
                # q unchanged
            elif self.is_action_feasible(t, c, q, 1):
                schedule.append(1)
                c += 1
                q -= 1
            else:
                # Should not happen if initial_feasible() returned True.
                return None

            t += 1
            if t >= end:
                # Phase boundary: q must be 0.
                if q != 0:
                    return None
                if p == P - 1:
                    break
                p += 1
                q = self.phase_quotas[p]

        return schedule
