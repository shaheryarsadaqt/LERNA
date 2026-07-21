"""Piece 2: RVD debt and mutually exclusive skip-source accounting.

Deterministic, lightweight tests for LERNARandomVetoDeferralPolicy.
No statistical assertions. No trainer/runner/validator imports.
"""

import pytest

from lerna.trainers.policies import (
    LERNARandomVetoDeferralPolicy,
    RandomSkipPolicy,
)


class FakeTracker:
    """Controllable tracker: `rho` drives the rho_vg veto deterministically."""

    def __init__(self):
        self.rho = 0.0
        self.loss_history = []

    def get_diagnostics(self):
        return {"rho_vg_raw": self.rho}


class FakeState:
    max_steps = None


class FakeArgs:
    max_steps = None


class FakeTrainer:
    state = FakeState()
    args = FakeArgs()


TOTAL_STEPS = 40
MIN_STEP = 2
SAFE_RHO = 1.0
DANGEROUS_RHO = -1.0  # below rho_veto_threshold -> veto


def make_policy(repay_mode="spread", **overrides):
    kwargs = dict(
        ler_tracker=FakeTracker(),
        target_skip_rate=0.20,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        use_rho_vg_veto=True,
        rho_veto_threshold=-0.05,
        repay_mode=repay_mode,
    )
    kwargs.update(overrides)
    return LERNARandomVetoDeferralPolicy(**kwargs)


def force_candidates(policy, candidates):
    """Initialize the policy, then pin an exact candidate set (deterministic)."""
    trainer = FakeTrainer()
    policy._lazy_init(trainer)
    policy._skip_set = set(candidates)
    policy._quota_size = len(candidates)
    policy._candidate_set_size = len(candidates)
    return trainer


def run_step(policy, trainer):
    return policy.should_skip(trainer, None, None)


def assert_invariants(policy):
    d = policy.get_diagnostics()
    assert d["invariant_skip_source_decomposition_ok"] is True
    assert d["invariant_skip_accounting_ok"] is True  # compatibility alias
    assert d["invariant_debt_conservation_ok"] is True
    assert d["invariant_debt_nonnegative_ok"] is True
    src = d["skip_source_counts"]
    assert d["skip_decisions"] == (
        src["accepted_candidate"] + src["ordinary_repayment"] + src["forced_tail"]
    )
    assert d["debt_created"] == (
        d["ordinary_debt_repayments"]
        + d["forced_tail_debt_repayments"]
        + d["outstanding_debt"]
    )
    assert d["outstanding_debt"] >= 0


def test_veto_creates_one_debt_and_no_skip():
    policy = make_policy()
    trainer = force_candidates(policy, {3})
    results = []
    for idx in range(4):  # steps 0..3; idx 3 is a vetoed candidate
        policy.trk.rho = DANGEROUS_RHO if idx == 3 else SAFE_RHO
        results.append(run_step(policy, trainer))
    assert results == [False, False, False, False]
    d = policy.get_diagnostics()
    assert d["debt_created"] == 1
    assert d["outstanding_debt"] == 1
    assert d["skip_decisions"] == 0
    assert d["skip_source_counts"] == {
        "accepted_candidate": 0,
        "ordinary_repayment": 0,
        "forced_tail": 0,
    }
    assert_invariants(policy)


def test_accepted_candidate_never_consumes_debt():
    # Regression test for the defect: acceptance must not drain debt.
    policy = make_policy()
    trainer = force_candidates(policy, {3, 4})
    result = None
    for idx in range(5):  # veto at 3, accept at 4
        policy.trk.rho = DANGEROUS_RHO if idx == 3 else SAFE_RHO
        result = run_step(policy, trainer)
    assert result is True
    d = policy.get_diagnostics()
    assert d["skip_source_counts"]["accepted_candidate"] == 1
    assert d["skip_source_counts"]["ordinary_repayment"] == 0
    assert d["skip_source_counts"]["forced_tail"] == 0
    assert d["outstanding_debt"] == 1  # debt untouched by acceptance
    assert d["debt_created"] == 1
    assert d["ordinary_debt_repayments"] == 0
    assert d["skip_decisions"] == 1
    assert_invariants(policy)


def test_ordinary_repayment_consumes_exactly_one_oldest_item():
    policy = make_policy(repay_mode="asap")
    trainer = force_candidates(policy, {3, 4})
    result = None
    for idx in range(6):  # vetoes at 3 and 4, noncandidate repayment at 5
        policy.trk.rho = DANGEROUS_RHO if idx in (3, 4) else SAFE_RHO
        result = run_step(policy, trainer)
    assert result is True
    d = policy.get_diagnostics()
    assert d["skip_source_counts"]["ordinary_repayment"] == 1
    assert d["skip_source_counts"]["accepted_candidate"] == 0
    assert d["ordinary_debt_repayments"] == 1
    assert d["debt_created"] == 2
    assert d["outstanding_debt"] == 1
    assert policy._deferred_pool == [4]  # oldest item (3) was removed
    assert_invariants(policy)


def test_forced_tail_consumes_debt_counts_only_forced_tail():
    policy = make_policy()
    trainer = force_candidates(policy, {3})
    # Create debt through the real veto path
    for idx in range(4):
        policy.trk.rho = DANGEROUS_RHO if idx == 3 else SAFE_RHO
        run_step(policy, trainer)
    d = policy.get_diagnostics()
    assert d["outstanding_debt"] == 1
    assert d["debt_created"] == 1

    # Forced tail with debt: consumes exactly one debt item
    assert policy._do_skip(7, source="forced_tail") is True
    d = policy.get_diagnostics()
    assert d["skip_source_counts"]["forced_tail"] == 1
    assert d["forced_tail_debt_repayments"] == 1
    assert d["ordinary_debt_repayments"] == 0
    assert d["outstanding_debt"] == 0
    assert d["skip_decisions"] == 1
    assert_invariants(policy)

    # Forced tail without debt: forced_tail source only, no debt counters.
    assert policy._do_skip(8, source="forced_tail") is True
    d = policy.get_diagnostics()
    assert d["skip_source_counts"]["forced_tail"] == 2
    assert d["forced_tail_debt_repayments"] == 1
    assert d["ordinary_debt_repayments"] == 0
    assert d["skip_decisions"] == 2
    assert_invariants(policy)


def test_repayment_without_debt_raises_and_debt_never_negative():
    policy = make_policy()
    force_candidates(policy, {3})
    assert policy._deferred_pool == []
    with pytest.raises(RuntimeError):
        policy._do_skip(5, source="ordinary_repayment")
    d = policy.get_diagnostics()
    # Failed call must not mutate any accounting.
    assert d["skip_decisions"] == 0
    assert d["ordinary_debt_repayments"] == 0
    assert d["outstanding_debt"] == 0
    assert d["invariant_debt_nonnegative_ok"] is True
    with pytest.raises(ValueError):
        policy._do_skip(5, source="not_a_source")
    assert policy.get_diagnostics()["skip_decisions"] == 0
    assert_invariants(policy)


def test_all_vetoes_disabled_matches_random_skip_policy_exactly():
    total, min_step, rate, seed = 200, 10, 0.20, 7
    rvd = LERNARandomVetoDeferralPolicy(
        ler_tracker=FakeTracker(),
        target_skip_rate=rate,
        total_steps=total,
        min_step=min_step,
        seed=seed,
    )  # all veto flags and phase protection default to False
    rsp = RandomSkipPolicy(
        target_skip_rate=rate, min_step=min_step, seed=seed, total_steps=total
    )
    trainer = FakeTrainer()
    for idx in range(total):
        a = rvd.should_skip(trainer, None, None)
        b = rsp.should_skip(trainer, None, None)
        assert a == b, f"decision divergence at step {idx}"
    d = rvd.get_diagnostics()
    assert d["skip_decisions"] == rsp.get_diagnostics()["skip_decisions"]
    assert d["skip_source_counts"]["ordinary_repayment"] == 0
    assert d["skip_source_counts"]["forced_tail"] == 0
    assert d["debt_created"] == 0
    assert d["outstanding_debt"] == 0
    assert_invariants(rvd)
