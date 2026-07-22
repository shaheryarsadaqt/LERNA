"""Deterministic tests for Piece 3 RVD repayment and quota feasibility."""

import pytest

from lerna.trainers.policies import LERNARandomVetoDeferralPolicy


class FakeTracker:
    def __init__(self):
        self.loss_history = []
        self.rho = 0.0

    def get_diagnostics(self):
        return {"rho_vg_raw": self.rho, "ler_raw": 0.0}


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class FixedRng:
    def __init__(self, values):
        self._values = list(values)

    def random(self):
        return self._values.pop(0) if self._values else 0.999


def make_policy(total_steps, skip_indices, **kwargs):
    defaults = {
        "target_skip_rate": len(skip_indices) / max(total_steps, 1),
        "total_steps": total_steps,
        "min_step": 0,
        "seed": 1234,
        "use_rho_vg_veto": True,
        "rho_veto_threshold": -0.05,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
        "max_consecutive_skips": 4,
    }
    defaults.update(kwargs)
    tracker = FakeTracker()
    policy = LERNARandomVetoDeferralPolicy(tracker, **defaults)
    policy._quota_total_steps = total_steps
    policy._quota_size = len(skip_indices)
    policy._skip_set = set(skip_indices)
    policy._candidate_set_size = len(skip_indices)
    policy._phase_skips = [0] * max(policy.n_phases, 1)
    return policy, tracker


def step(policy, tracker, trainer, dangerous=False):
    tracker.rho = -1.0 if dangerous else 0.0
    return policy.should_skip(trainer, None, None)


def assert_invariants(policy):
    diagnostics = policy.get_diagnostics()
    assert diagnostics["invariant_skip_source_decomposition_ok"]
    assert diagnostics["invariant_debt_conservation_ok"]
    assert diagnostics["invariant_debt_never_negative_ok"]
    assert diagnostics["invariant_forced_tail_no_double_count_ok"]
    assert diagnostics["invariant_repayment_single_source_ok"]


def test_repay_mode_validated():
    with pytest.raises(ValueError):
        LERNARandomVetoDeferralPolicy(FakeTracker(), repay_mode="bogus")


@pytest.mark.parametrize("repay_mode", ["asap", "spread"])
def test_valid_repay_modes_are_accepted(repay_mode):
    policy = LERNARandomVetoDeferralPolicy(
        FakeTracker(), repay_mode=repay_mode
    )
    assert policy.repay_mode == repay_mode


def test_safe_asap_repayment():
    policy, tracker = make_policy(10, {2})
    trainer = FakeTrainer(10)
    assert step(policy, tracker, trainer) is False
    assert step(policy, tracker, trainer) is False
    assert step(policy, tracker, trainer, dangerous=True) is False
    assert step(policy, tracker, trainer) is True
    diagnostics = policy.get_diagnostics()
    assert diagnostics["ordinary_debt_repayments"] == 1
    assert diagnostics["outstanding_debt"] == 0
    assert diagnostics["dangerous_repayment_deferred_count"] == 0
    assert_invariants(policy)


def test_dangerous_asap_repayment_deferred():
    policy, tracker = make_policy(10, {2})
    trainer = FakeTrainer(10)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    assert step(policy, tracker, trainer, dangerous=True) is False
    diagnostics = policy.get_diagnostics()
    assert diagnostics["dangerous_repayment_deferred_count"] == 1
    assert diagnostics["outstanding_debt"] == 1
    assert step(policy, tracker, trainer) is True
    assert policy.get_diagnostics()["outstanding_debt"] == 0
    assert_invariants(policy)


def test_protection_disabled_repays_through_danger():
    policy, tracker = make_policy(
        10, {2}, repay_protect_dangerous=False
    )
    trainer = FakeTrainer(10)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    assert step(policy, tracker, trainer, dangerous=True) is True
    diagnostics = policy.get_diagnostics()
    assert diagnostics["dangerous_repayment_deferred_count"] == 0
    assert diagnostics["ordinary_debt_repayments"] == 1
    assert_invariants(policy)


def test_spread_mode_selected_and_not_selected():
    policy, tracker = make_policy(10, {2}, repay_mode="spread")
    trainer = FakeTrainer(10)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    policy._rng = FixedRng([0.99, 0.0])
    assert step(policy, tracker, trainer) is False
    assert policy.get_diagnostics()["outstanding_debt"] == 1
    assert step(policy, tracker, trainer) is True
    assert policy.get_diagnostics()["outstanding_debt"] == 0
    assert_invariants(policy)


def test_max_consecutive_candidate_deferral_counts_total_veto():
    policy, tracker = make_policy(
        20, {2, 3}, max_consecutive_skips=1, use_safety_horizon=True
    )
    trainer = FakeTrainer(20)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    assert step(policy, tracker, trainer) is True
    assert step(policy, tracker, trainer) is False
    diagnostics = policy.get_diagnostics()
    assert diagnostics["max_consecutive_veto_count"] == 1
    assert diagnostics["vetoed_skips"] == 1
    assert diagnostics["veto_rate_vs_candidates"] == 0.5
    assert diagnostics["debt_created"] == 1
    assert diagnostics["outstanding_debt"] == 1
    assert_invariants(policy)


def test_safety_horizon_disabled_does_not_enforce_consecutive_limit():
    policy, tracker = make_policy(
        20, {2, 3}, max_consecutive_skips=1, use_safety_horizon=False
    )
    trainer = FakeTrainer(20)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    assert step(policy, tracker, trainer) is True
    assert step(policy, tracker, trainer) is True
    diagnostics = policy.get_diagnostics()
    assert diagnostics["max_consecutive_veto_count"] == 0
    assert diagnostics["vetoed_skips"] == 0
    assert diagnostics["outstanding_debt"] == 0
    assert_invariants(policy)


def test_max_consecutive_repayment_deferral():
    policy, tracker = make_policy(
        20, {2, 3}, max_consecutive_skips=1, use_safety_horizon=True
    )
    trainer = FakeTrainer(20)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    step(policy, tracker, trainer, dangerous=True)
    assert step(policy, tracker, trainer) is True
    assert step(policy, tracker, trainer) is False
    diagnostics = policy.get_diagnostics()
    assert diagnostics["max_consecutive_repayment_veto_count"] == 1
    assert diagnostics["outstanding_debt"] == 1
    assert step(policy, tracker, trainer) is True
    assert policy.get_diagnostics()["outstanding_debt"] == 0
    assert_invariants(policy)


def test_forced_quota_overrides_danger_and_consecutive():
    policy, tracker = make_policy(
        5, {0, 1, 2}, max_consecutive_skips=1, use_safety_horizon=True
    )
    trainer = FakeTrainer(5)
    assert step(policy, tracker, trainer, dangerous=True) is False
    assert step(policy, tracker, trainer, dangerous=True) is False
    assert step(policy, tracker, trainer, dangerous=True) is True
    assert step(policy, tracker, trainer, dangerous=True) is True
    assert step(policy, tracker, trainer, dangerous=True) is True
    diagnostics = policy.get_diagnostics()
    assert diagnostics["skip_decisions"] == diagnostics["quota_size"] == 3
    assert diagnostics["forced_dangerous_candidate_count"] == 1
    assert diagnostics["forced_dangerous_repayment_count"] == 2
    assert diagnostics["forced_max_consecutive_override_count"] == 2
    assert diagnostics["forced_tail_debt_repayments"] == 2
    assert diagnostics["outstanding_debt"] == 0
    assert_invariants(policy)


def test_exact_quota_zero_debt_feasible_run():
    candidates = {3, 6, 9, 12}
    policy, tracker = make_policy(30, candidates)
    trainer = FakeTrainer(30)
    for index in range(30):
        step(policy, tracker, trainer, dangerous=index in (3, 6))
    diagnostics = policy.get_diagnostics()
    assert diagnostics["skip_decisions"] == len(candidates)
    assert diagnostics["outstanding_debt"] == 0
    assert diagnostics["forced_tail_skips"] == 0
    assert_invariants(policy)


def test_no_forced_action_before_mathematically_necessary():
    policy, tracker = make_policy(10, {8, 9})
    trainer = FakeTrainer(10)
    for _ in range(8):
        assert step(policy, tracker, trainer) is False
    assert step(policy, tracker, trainer) is True
    assert step(policy, tracker, trainer) is True
    diagnostics = policy.get_diagnostics()
    assert diagnostics["forced_tail_skips"] == 0
    assert diagnostics["forced_dangerous_candidate_count"] == 0
    assert diagnostics["forced_dangerous_repayment_count"] == 0
    assert diagnostics["forced_max_consecutive_override_count"] == 0
    assert diagnostics["skip_decisions"] == 2
    assert_invariants(policy)


def test_danger_evaluated_once_for_candidate():
    policy, tracker = make_policy(10, {2})
    trainer = FakeTrainer(10)
    calls = []
    original = policy._is_dangerous

    def counted(candidate_trainer):
        calls.append(candidate_trainer)
        return original(candidate_trainer)

    policy._is_dangerous = counted
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    assert len(calls) == 1


def test_danger_evaluated_once_for_repayment_attempt():
    policy, tracker = make_policy(10, {2})
    trainer = FakeTrainer(10)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer)
    step(policy, tracker, trainer, dangerous=True)
    calls = []
    original = policy._is_dangerous

    def counted(candidate_trainer):
        calls.append(candidate_trainer)
        return original(candidate_trainer)

    policy._is_dangerous = counted
    step(policy, tracker, trainer, dangerous=True)
    assert len(calls) == 1


def test_danger_evaluated_once_for_forced_tail():
    policy, tracker = make_policy(4, {0, 1})
    trainer = FakeTrainer(4)
    step(policy, tracker, trainer, dangerous=True)
    step(policy, tracker, trainer, dangerous=True)
    calls = []
    original = policy._is_dangerous

    def counted(candidate_trainer):
        calls.append(candidate_trainer)
        return original(candidate_trainer)

    policy._is_dangerous = counted
    assert step(policy, tracker, trainer, dangerous=True) is True
    assert len(calls) == 1
    assert policy.get_diagnostics()["forced_tail_skips"] == 1
