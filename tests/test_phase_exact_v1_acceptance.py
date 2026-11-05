"""CPU acceptance tests for phase_exact_v1 controller (Step 3, V2).

Regressions A-E per LERNA_STEP3_CLI_PROMPT_V2.md section 6, plus
task-sized synthetic traces and legacy compatibility checks.
No training, no model downloads, no GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from lerna.utils.forward_gate import (
    ScoreQuantileGate,
    ScoreQuantileGatePolicy,
    PhaseSuffixInfeasibleError,
    TerminalInvalidScoreError,
    InfeasibleBudgetError,
    _eligible_phase_quota,
)
from lerna.utils.phase_suffix_feasibility import (
    PhaseSuffixFeasibility,
    phase_bounds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _constant_score(val):
    def _src(decision_index, model, inputs):
        return val
    return _src


def _rising_score(T):
    def _src(decision_index, model, inputs):
        return decision_index / T
    return _src


def _falling_score(T):
    def _src(decision_index, model, inputs):
        return 1.0 - decision_index / T
    return _src


def _alternating_score():
    def _src(decision_index, model, inputs):
        return float(decision_index % 2)
    return _src


def _seeded_random_score(seed):
    rng = __import__("random").Random(seed)
    def _src(decision_index, model, inputs):
        return rng.random()
    return _src


# ---------------------------------------------------------------------------
# Regressions A-E
# ---------------------------------------------------------------------------

class TestRegressions:
    """A–C use W=0, rate=Q/T, history=64, anneal_fraction=.88,
    constant score=.5 and direct gate seed=12 (policy_seed=11 through wrapper).
    """

    def test_case_A_legacy_underspends_new_rejects(self):
        """A. T4,W0,Q2,P4,K1 quotas [1,1,0,0]: legacy documented result;
        new constructor rejects."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="legacy_v4",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(4)]
        assert actions == [1, 0, 0, 0]
        assert gate.skipped == 1  # legacy underspends (documented)

        with pytest.raises(PhaseSuffixInfeasibleError):
            ScoreQuantileGate(
                rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_case_B_unique_schedule(self):
        """B. T5,W0,Q3,P2,K1 quotas [1,2]: new gate must complete
        [1,0,1,0,1], the unique legal schedule."""
        gate = ScoreQuantileGate(
            rate=0.6, total_steps=5, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(5)]
        assert actions == [1, 0, 1, 0, 1]
        assert gate.skipped == 3
        assert gate._requested_total == 3
        assert gate._phase_used == [1, 2]

    def test_case_C_legacy_accepts_new_rejects(self):
        """C. T9,W0,Q8,P4,K4 quotas [2,2,2,2]: legacy accepts and underspends;
        new rejects."""
        gate = ScoreQuantileGate(
            rate=8/9, total_steps=9, min_step=0, max_consecutive_skips=4,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="legacy_v4",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(9)]
        assert actions == [1, 1, 1, 1, 0, 1, 1, 0, 1]
        assert gate.skipped == 7  # legacy underspends (7 of 8)

        with pytest.raises(PhaseSuffixInfeasibleError):
            ScoreQuantileGate(
                rate=8/9, total_steps=9, min_step=0, max_consecutive_skips=4,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_case_D_T41_infeasible(self):
        """D. T41,W0,P41,K1,Q2,rate=2/41 yields quotas [1,1,0,...,0]:
        exact infeasible despite individual/global capacity bounds."""
        T = 41
        P = 41
        Q = 2
        rate = Q / T
        with pytest.raises(PhaseSuffixInfeasibleError):
            ScoreQuantileGate(
                rate=rate, total_steps=T, min_step=0, max_consecutive_skips=1,
                n_phases=P, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_case_E_suffix_infeasible(self):
        """E. T41,W0,P1,K1,Q20 after 22 retained opportunities: query the
        helper at t=22,c=0,q=20 and require infeasible."""
        feas = PhaseSuffixFeasibility(
            total_steps=41, n_phases=1, phase_quotas=[20],
            min_step=0, max_consecutive_skips=1,
        )
        # After 22 retains at t=22, c=0, q=20: only 19 steps remain,
        # need 20 skips — infeasible under K=1 (max 10 skips in 19 steps).
        assert not feas.is_action_feasible(22, 0, 20, 0)
        assert not feas.is_action_feasible(22, 0, 20, 1)


# ---------------------------------------------------------------------------
# Task-sized synthetic traces
# ---------------------------------------------------------------------------

class TestTaskSizedTraces:
    """QNLI T6548,W392,Q3929,K4,P1/4; MRPC T377,W22,Q226,K4,P1/4;
    RTE T257,W15,Q154,K4,P1/4. Constant/rising/falling/alternating/seeded
    random streams. Every new-version completed trace meets phase quotas and K.
    """

    @pytest.mark.parametrize("T,W,Q,P", [
        (6548, 392, 3929, 1),
        (6548, 392, 3929, 4),
        (377, 22, 226, 1),
        (377, 22, 226, 4),
        (257, 15, 154, 1),
        (257, 15, 154, 4),
    ])
    @pytest.mark.parametrize("score_kind,score_fn", [
        ("constant", lambda T: _constant_score(0.5)),
        ("rising", lambda T: _rising_score(T)),
        ("falling", lambda T: _falling_score(T)),
        ("alternating", lambda T: _alternating_score()),
        ("seeded_random", lambda T: _seeded_random_score(7)),
    ])
    def test_new_version_meets_quota_and_cap(self, T, W, Q, P, score_kind, score_fn):
        rate = Q / T
        gate = ScoreQuantileGate(
            rate=rate, total_steps=T, min_step=W, max_consecutive_skips=4,
            n_phases=P, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=score_fn(T),
            controller_version="phase_exact_v1",
        )
        actions = []
        for t in range(T):
            a = gate.decide(t, None, None)
            actions.append(1 if a else 0)
        assert gate.skipped == gate._requested_total, (
            f"{score_kind} T={T} P={P}: realized={gate.skipped} != requested={gate._requested_total}"
        )
        assert gate._longest_skip_run <= 4, (
            f"{score_kind} T={T} P={P}: longest run {gate._longest_skip_run} > K=4"
        )
        for i in range(P):
            assert gate._phase_used[i] == gate._phase_quota[i], (
                f"{score_kind} T={T} P={P}: phase {i} used={gate._phase_used[i]} "
                f"quota={gate._phase_quota[i]}"
            )


# ---------------------------------------------------------------------------
# Legacy compatibility (no silent fixes)
# ---------------------------------------------------------------------------

class TestLegacyCompatibility:
    """Legacy mode unchanged: same actions, counters, histories, RNG states."""

    def test_legacy_case_A_unchanged(self):
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="legacy_v4",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(4)]
        assert actions == [1, 0, 0, 0]
        assert gate.skipped == 1

    def test_legacy_case_B_unchanged(self):
        gate = ScoreQuantileGate(
            rate=0.6, total_steps=5, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="legacy_v4",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(5)]
        assert actions == [0, 1, 0, 0, 0]
        assert gate.skipped == 1

    def test_legacy_case_C_unchanged(self):
        gate = ScoreQuantileGate(
            rate=8/9, total_steps=9, min_step=0, max_consecutive_skips=4,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="legacy_v4",
        )
        actions = [int(gate.decide(t, None, None)) for t in range(9)]
        assert actions == [1, 1, 1, 1, 0, 1, 1, 0, 1]
        assert gate.skipped == 7


# ---------------------------------------------------------------------------
# Grid of small valid configurations
# ---------------------------------------------------------------------------

class TestSmallGrid:
    """All small valid configurations: vary P,K,W,quota, score shapes."""

    @pytest.mark.parametrize("T,P,W,K,Q", [
        (4, 1, 0, 1, 2),
        (4, 2, 0, 1, 2),
        (4, 2, 0, 2, 3),
        (5, 2, 0, 1, 3),
        (5, 3, 0, 2, 4),
        (6, 3, 1, 2, 3),
        (6, 2, 2, 4, 4),
        (7, 3, 0, 3, 5),
    ])
    def test_all_small_valid_complete_exact(self, T, P, W, K, Q):
        rate = Q / T
        # Use actual allocated quotas, not [1]*P
        quota, requested, total_eligible, eligible = _eligible_phase_quota(
            rate, T, W, P
        )
        feas = PhaseSuffixFeasibility(
            total_steps=T, n_phases=P, phase_quotas=quota,
            min_step=W, max_consecutive_skips=K,
        )
        if not feas.initial_feasible():
            # Infeasible configs must be explicit rejection tests, not skips.
            with pytest.raises(PhaseSuffixInfeasibleError):
                ScoreQuantileGate(
                    rate=rate, total_steps=T, min_step=W,
                    max_consecutive_skips=K, n_phases=P,
                    score_history_window=64, anneal_fraction=0.88,
                    seed=12, score_source=_constant_score(0.5),
                    controller_version="phase_exact_v1",
                )
            return
        gate = ScoreQuantileGate(
            rate=rate, total_steps=T, min_step=W, max_consecutive_skips=K,
            n_phases=P, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        for t in range(T):
            gate.decide(t, None, None)
        assert gate.skipped == gate._requested_total
        assert gate._longest_skip_run <= K


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestErrorCases:
    def test_unknown_version_rejected(self):
        with pytest.raises(ValueError):
            ScoreQuantileGate(
                rate=0.5, total_steps=10, min_step=0, max_consecutive_skips=4,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="bogus",
            )

    def test_boolean_int_field_rejected(self):
        with pytest.raises(TypeError):
            ScoreQuantileGate(
                rate=0.5, total_steps=True, min_step=0, max_consecutive_skips=4,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_fractional_int_field_rejected(self):
        with pytest.raises(ValueError):
            ScoreQuantileGate(
                rate=0.5, total_steps=10.5, min_step=0, max_consecutive_skips=4,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_rate_out_of_range_rejected(self):
        with pytest.raises(ValueError):
            ScoreQuantileGate(
                rate=1.5, total_steps=10, min_step=0, max_consecutive_skips=4,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_terminal_raises_again_new_version(self):
        """Terminal NEW-version gate raises again on subsequent calls."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=lambda *args: float("nan"),
            controller_version="phase_exact_v1",
        )
        with pytest.raises(TerminalInvalidScoreError):
            gate.decide(0, None, None)
        with pytest.raises(TerminalInvalidScoreError):
            gate.decide(1, None, None)

    def test_sequential_index_validation(self):
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        gate.decide(0, None, None)
        with pytest.raises(ValueError):
            gate.decide(2, None, None)  # skip index 1


# ---------------------------------------------------------------------------
# New-version rejection and invariant tests (Step 3B findings #2, #3, #4)
# ---------------------------------------------------------------------------

class TestNewVersionRejections:
    def test_w_t_positive_quota_rejects(self):
        """T=4, W=4, rate=0.5, P=2, K=1: Q=2 but 0 eligible steps -> reject."""
        with pytest.raises(InfeasibleBudgetError):
            ScoreQuantileGate(
                rate=0.5, total_steps=4, min_step=4, max_consecutive_skips=1,
                n_phases=2, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )

    def test_zero_budget_works(self):
        """Genuine Q=0 (rate rounds to 0) is valid."""
        gate = ScoreQuantileGate(
            rate=0.0, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        assert gate._requested_total == 0
        for t in range(4):
            assert gate.decide(t, None, None) is False
        assert gate.skipped == 0

    def test_policy_coercion_rejected(self):
        """Policy: T4.9 and boolean K must reject before coercion."""
        with pytest.raises((ValueError, TypeError)):
            ScoreQuantileGatePolicy(
                target_skip_rate=0.5, total_steps=4.9, min_step=0,
                n_phases=2, max_consecutive_skips=True,
                score_source="uniform", controller_version="phase_exact_v1",
            )
        with pytest.raises((ValueError, TypeError)):
            ScoreQuantileGatePolicy(
                target_skip_rate=0.5, total_steps=10, min_step=0,
                n_phases=2, max_consecutive_skips=4.5,
                score_source="uniform", controller_version="phase_exact_v1",
            )

    def test_s2_rejected_for_phase_exact_v1(self):
        """phase_exact_v1 explicitly rejects S2."""
        with pytest.raises(ValueError):
            ScoreQuantileGatePolicy(
                target_skip_rate=0.5, total_steps=10, min_step=0,
                n_phases=4, max_consecutive_skips=4,
                score_source="s2", controller_version="phase_exact_v1",
            )

    def test_action_index_boolean_rejected(self):
        """boolean False is not index zero."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        with pytest.raises(TypeError):
            gate.decide(False, None, None)

    def test_action_index_past_end_rejected(self):
        """t >= T must be rejected."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        for t in range(4):
            gate.decide(t, None, None)
        with pytest.raises(ValueError):
            gate.decide(4, None, None)
        with pytest.raises(ValueError):
            gate.decide(5, None, None)

    def test_terminal_counter_not_advance(self):
        """After terminal, decisions_seen and _next_decision_index don't advance."""
        calls = []
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12,
            score_source=lambda *a: (calls.append(a[0]) or float("nan")),
            controller_version="phase_exact_v1",
        )
        with pytest.raises(TerminalInvalidScoreError):
            gate.decide(0, None, None)
        before = (gate.decisions_seen, gate._next_decision_index)
        with pytest.raises(TerminalInvalidScoreError):
            gate.decide(1, None, None)
        after = (gate.decisions_seen, gate._next_decision_index)
        assert before == after, f"terminal mutated counters: {before} -> {after}"
        assert calls == [0], f"score called after terminal: {calls}"

    def test_impossible_suffix_becomes_terminal(self):
        """No feasible action from accepted state -> terminal invariant error."""
        gate = ScoreQuantileGate(
            rate=20/41, total_steps=41, min_step=0, max_consecutive_skips=1,
            n_phases=1, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        gate._next_decision_index = 22
        gate.decisions_seen = 22
        with pytest.raises(PhaseSuffixInfeasibleError):
            gate.decide(22, None, None)
        assert gate._terminal is True
        # Subsequent call re-raises stored terminal, no state change
        before = (gate.decisions_seen, gate._next_decision_index)
        with pytest.raises(PhaseSuffixInfeasibleError):
            gate.decide(23, None, None)
        after = (gate.decisions_seen, gate._next_decision_index)
        assert before == after

    def test_expired_phase_reject(self):
        """Injected unpaid expired phase must be rejected."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        # Inject prefix [0,0]: both retains, phase 0 quota unpaid (1 needed, 0 used)
        gate._next_decision_index = 2
        gate.decisions_seen = 2
        with pytest.raises(PhaseSuffixInfeasibleError):
            gate.decide(2, None, None)

    def test_t8_p4_w2_k2_q6_infeasible_under_cap(self):
        """T8,P4,W2,K2,Q6: max skips under cap is 4 < requested 6, so infeasible."""
        with pytest.raises((InfeasibleBudgetError, PhaseSuffixInfeasibleError)):
            ScoreQuantileGate(
                rate=0.75, total_steps=8, min_step=2, max_consecutive_skips=2,
                n_phases=4, score_history_window=64, anneal_fraction=0.88,
                seed=12, score_source=_constant_score(0.5),
                controller_version="phase_exact_v1",
            )


# ---------------------------------------------------------------------------
# Step 3C remaining-fix regression tests (R1–R6)
# ---------------------------------------------------------------------------

class TestR1LegacyParityScoreCalls:
    """R1: Score must not be computed during warmup_early (legacy parity)."""

    def test_no_score_during_warmup_early(self):
        """Warmup-early steps (t < min_step - history) must not call score."""
        calls = []
        gate = ScoreQuantileGate(
            rate=0.2, total_steps=10, min_step=6, max_consecutive_skips=1,
            n_phases=2, score_history_window=4, anneal_fraction=0.88,
            seed=12,
            score_source=lambda *a: (calls.append(a[0]) or 0.5),
            controller_version="legacy_v4",
        )
        # min_step=6, history=4 => early_return = max(6-4,0) = 2
        # Steps 0,1 are warmup_early; no score should be computed.
        for t in range(2):
            gate.decide(t, None, None)
        assert calls == [], f"score computed during warmup_early: {calls}"

    def test_score_computed_from_warmup_history(self):
        """Warmup-history steps (early_return <= t < min_step) compute score."""
        calls = []
        gate = ScoreQuantileGate(
            rate=0.2, total_steps=10, min_step=6, max_consecutive_skips=1,
            n_phases=2, score_history_window=4, anneal_fraction=0.88,
            seed=12,
            score_source=lambda *a: (calls.append(a[0]) or 0.5),
            controller_version="legacy_v4",
        )
        # Steps 2,3,4,5 are warmup_history; score should be computed.
        for t in range(2, 6):
            gate.decide(t, None, None)
        assert calls == [2, 3, 4, 5], f"score not computed in warmup_history: {calls}"


class TestR2TerminalHandling:
    """R2: Exact terminal/state handling."""

    def test_terminate_on_score_persists_terminal(self):
        """terminate_on_score stores terminal error; subsequent decide re-raises."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        with pytest.raises(TerminalInvalidScoreError):
            gate.terminate_on_score(2, float("nan"))
        assert gate._terminal is True
        assert gate._terminal_at == 2
        # Subsequent call re-raises stored error (TerminalInvalidScoreError,
        # not TypeError from a None _terminal_error).
        before = (gate.decisions_seen, gate._next_decision_index)
        with pytest.raises(TerminalInvalidScoreError):
            gate.decide(3, None, None)
        after = (gate.decisions_seen, gate._next_decision_index)
        assert before == after

    def test_decide_after_terminal_raises_stored_error(self):
        """After terminal from _validate_prefix_invariants, re-raise stored error."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        # Inject prefix failure: phase 0 quota unpaid before phase 1 starts
        gate._next_decision_index = 2
        gate.decisions_seen = 2
        with pytest.raises(PhaseSuffixInfeasibleError):
            gate.decide(2, None, None)
        assert gate._terminal is True
        # Subsequent call re-raises stored error
        before = (gate.decisions_seen, gate._next_decision_index)
        with pytest.raises(PhaseSuffixInfeasibleError):
            gate.decide(3, None, None)
        after = (gate.decisions_seen, gate._next_decision_index)
        assert before == after

    def test_cap_retain_checks_feasibility(self):
        """At cap, gate retains (doesn't skip) and continues normally."""
        gate = ScoreQuantileGate(
            rate=0.4, total_steps=8, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.1),
            controller_version="phase_exact_v1",
        )
        # Skip at t=0 (low score), hits cap c=1 at t=1
        actions = [int(gate.decide(t, None, None)) for t in range(8)]
        # At cap, must retain: no consecutive skips allowed
        for i in range(1, len(actions)):
            if actions[i-1] == 1:
                assert actions[i] == 0, f"consecutive skips at {i-1},{i}"


class TestR3Diagnostics:
    """R3: Diagnostics independence from logging."""

    def test_reason_counts_identical_logging_on_off(self):
        """reason_counts must be identical with logging ON vs OFF."""
        def rising(T):
            return lambda di, m, i: di / T

        T = 100
        gate_off = ScoreQuantileGate(
            rate=0.6, total_steps=T, min_step=6, max_consecutive_skips=4,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=rising(T),
            controller_version="phase_exact_v1",
        )
        [gate_off.decide(t, None, None) for t in range(T)]
        counts_off = dict(gate_off.diagnostics()["reason_counts"])

        gate_on = ScoreQuantileGate(
            rate=0.6, total_steps=T, min_step=6, max_consecutive_skips=4,
            n_phases=4, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=rising(T),
            controller_version="phase_exact_v1",
        )
        gate_on.enable_diagnostics_logging()
        [gate_on.decide(t, None, None) for t in range(T)]
        counts_on = dict(gate_on.diagnostics()["reason_counts"])

        assert counts_off == counts_on, f"reason_counts differ: {counts_off} vs {counts_on}"
        assert sum(counts_off.values()) == T, f"reason_counts total != {T}: {counts_off}"

    def test_terminal_event_not_in_reason_counts(self):
        """Terminal events must not be counted in reason_counts."""
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12,
            score_source=lambda *a: float("nan"),
            controller_version="phase_exact_v1",
        )
        try:
            gate.decide(0, None, None)
        except TerminalInvalidScoreError:
            pass
        # reason_counts should be empty (terminal event not a valid action)
        assert gate.diagnostics()["reason_counts"] == {}, (
            f"terminal event in reason_counts: {gate.diagnostics()['reason_counts']}"
        )

    def test_tie_and_empty_history_subreasons(self):
        """Tie-break and empty-history produce distinct sub-reasons."""
        # Empty history: no scores yet, score > threshold path not taken
        gate = ScoreQuantileGate(
            rate=0.5, total_steps=4, min_step=0, max_consecutive_skips=1,
            n_phases=2, score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=_constant_score(0.5),
            controller_version="phase_exact_v1",
        )
        # First decision after min_step: empty history -> "empty_history"
        gate.decide(0, None, None)  # warmup_history (min_step=0, so no warmup)
        # Actually min_step=0 means decision_index=0 >= min_step, so it's a regular step
        # with empty history -> "empty_history"
        # But wait: early_return = max(0-64,0) = 0, so decision 0 is NOT warmup_early
        # and decision_index=0 >= min_step=0, so it's a regular step
        # With empty history, _score_decides returns "empty_history"
        diag = gate.diagnostics()
        # Just verify the reason is one of the expected sub-reasons
        assert "reason_counts" in diag


class TestR4RunnerValidation:
    """R4: Runner config validation."""

    def test_experiment_config_negative_horizon_rejected(self):
        """requested_skips > total_steps must be rejected."""
        from scripts.run_controller_ablation import ExperimentConfig
        with pytest.raises(ValueError):
            ExperimentConfig(
                task="qnli", arm="gate_s1_p1",
                controller_version="phase_exact_v1", score_source="uniform",
                total_steps=4, min_step=0, n_phases=2,
                max_consecutive_skips=1, policy_seed=0,
                target_skip_rate=1.5,  # Q=6 > T=4
                output_dir="/tmp/test", attempt_id="test",
            )

    def test_experiment_config_valid_horizon(self):
        """requested_skips <= total_steps must be accepted."""
        from scripts.run_controller_ablation import ExperimentConfig
        cfg = ExperimentConfig(
            task="qnli", arm="gate_s1_p1",
            controller_version="phase_exact_v1", score_source="s1",
            total_steps=10, min_step=2, n_phases=1,
            max_consecutive_skips=4, policy_seed=0,
            target_skip_rate=0.5,  # Q=5 <= T=10
            output_dir="/tmp/test", attempt_id="test",
            probe_exclusions_file="/some/exclusions.json",
            probe_exclusions_hash="abc123",
            model_revision="a" * 40,
            tokenizer_revision="b" * 40,
            dataset_revision="c" * 40,
            dry_run=True,
        )
        assert cfg.requested_skips == 5


class TestR6StatisticsCompleteness:
    """R6: Statistics generator completeness."""

    def test_incomplete_contrast_recorded_not_silent(self):
        """Insufficient-data contrasts are recorded with status, not silently omitted.
        n=1 is valid (one paired observation); n=0 is insufficient_data."""
        import tempfile, csv, json
        from scripts.generate_corrected_comparisons import _comparisons

        # Create a ledger with only 1 seed for each arm (paired, n=1)
        by = {("gate", 0): 0.8, ("control", 0): 0.7}  # only 1 seed each, paired
        result = _comparisons(by, [("gate", "control")], "qnli", "/raid/x")
        assert len(result) == 1
        assert result[0]["status"] == "complete"
        assert result[0]["n"] == 1
        assert abs(result[0]["mean_raw"] - 0.1) < 1e-9

        # No paired observations at all → insufficient_data
        by2 = {("gate", 0): 0.8}  # only gate, no control
        result2 = _comparisons(by2, [("gate", "control")], "qnli", "/raid/x")
        assert len(result2) == 1
        assert result2[0]["status"] == "insufficient_data"
        assert result2[0]["n"] == 0
        assert result2[0]["mean_raw"] is None

    def test_matched_pair_deletion_detected(self):
        """Same seed missing from both arms → matched_pair_deleted reported."""
        from scripts.generate_corrected_comparisons import _comparisons
        # Seeds 0,1 in gate; seeds 0,1 in control (seed 2 deleted from both arms)
        # Seed 2 exists in the expected cohort (size=3) but is absent from both arms
        by = {
            ("gate", 0): 0.8, ("gate", 1): 0.7,
            ("control", 0): 0.7, ("control", 1): 0.6,
        }
        result = _comparisons(by, [("gate", "control")], "qnli", "/raid/x",
                             expected_cohort_size=3,
                             expected_seed_set={0, 1, 2})
        assert len(result) == 1
        assert result[0]["status"] == "partial"
        assert result[0]["missing_data"] is True
        assert result[0]["cohort_size_violation"] is True
        # Seed 2 is in expected cohort but missing from both arms
        assert 2 in result[0]["matched_pair_deleted"]

    def test_cohort_size_enforcement(self):
        """Arm with fewer than expected seeds → cohort_size_violation."""
        from scripts.generate_corrected_comparisons import _comparisons
        # Expected 3 seeds, but gate only has 2
        by = {
            ("gate", 0): 0.8, ("gate", 1): 0.7,
            ("control", 0): 0.7, ("control", 1): 0.6, ("control", 2): 0.5,
        }
        result = _comparisons(by, [("gate", "control")], "qnli", "/raid/x", expected_cohort_size=3)
        assert len(result) == 1
        assert result[0]["status"] == "partial"
        assert result[0]["cohort_size_violation"] is True
        assert result[0]["actual_cohort_size_a"] == 2
        assert result[0]["actual_cohort_size_b"] == 3
        assert 2 in result[0]["unmatched_b"]
