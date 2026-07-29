"""Tests for FixedPhaseStratifiedRandomPolicy (pure fixed temporal baseline).

These tests LOCK the CURRENT behavior of the Fixed Phase-Stratified Random
controller: fixed equal-length phases, weighted random skipping within each
phase, cumulative quota carryover, and no rho/loss-spike/LER signals.
Deterministic, lightweight, no models/datasets/GPUs, no statistical asserts.
"""

import pytest

from lerna.trainers.policies import (
    FixedPhaseStratifiedRandomPolicy,
    LERNAPhaseStratifiedPolicy,
    PhaseStratifiedGuardedRandomPolicy,
)


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class FakeTracker:
    def __init__(self):
        self.rho = 1.0
        self.loss_history = []

    def get_diagnostics(self):
        return {"rho_vg_raw": self.rho, "ler_raw": None, "ler": None}


TOTAL_STEPS = 100
MIN_STEP = 20
RATE = 0.40


def make_fixed(**overrides):
    kwargs = dict(
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        n_phases=4,
    )
    kwargs.update(overrides)
    return FixedPhaseStratifiedRandomPolicy(**kwargs)


def make_legacy(tracker=None, **overrides):
    kwargs = dict(
        ler_tracker=tracker if tracker is not None else FakeTracker(),
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        n_phases=4,
    )
    kwargs.update(overrides)
    return LERNAPhaseStratifiedPolicy(**kwargs)


def make_guarded(tracker=None, **overrides):
    kwargs = dict(
        ler_tracker=tracker if tracker is not None else FakeTracker(),
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        n_phases=4,
    )
    kwargs.update(overrides)
    return PhaseStratifiedGuardedRandomPolicy(**kwargs)


def run_horizon(policy, trainer):
    decisions = []
    for di in range(trainer.state.max_steps):
        decisions.append(policy.should_skip(trainer, None, None))
    return decisions


class _AlwaysZero:
    def random(self):
        return 0.0


# 1. Fixed equal-length phase boundaries

def test_equal_length_phase_boundaries():
    policy = make_fixed()
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._lazy_init(trainer)
    d = policy.get_diagnostics()

    assert d["phase_bounds"][0] == MIN_STEP
    assert d["phase_bounds"][-1] == TOTAL_STEPS
    assert d["phase_bounds"] == [20, 40, 60, 80, 100]

    assert d["phase_eligible"] == [20, 20, 20, 20]
    assert sum(d["phase_eligible"]) == TOTAL_STEPS - MIN_STEP

    bounds = d["phase_bounds"]
    assert d["phase_eligible"] == [
        bounds[i + 1] - bounds[i] for i in range(d["n_phases"])
    ]

    assert policy._phase_of(20) == 0
    assert policy._phase_of(39) == 0
    assert policy._phase_of(40) == 1
    assert policy._phase_of(59) == 1
    assert policy._phase_of(60) == 2
    assert policy._phase_of(79) == 2
    assert policy._phase_of(80) == 3
    assert policy._phase_of(99) == 3


# 2. Fixed default weights and largest-remainder quotas

def test_default_weights_and_phase_quotas():
    policy = make_fixed()
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._lazy_init(trainer)
    d = policy.get_diagnostics()

    assert d["phase_weights"] == pytest.approx([0.22, 0.24, 0.26, 0.28])
    assert d["quota_size"] == 40
    assert d["phase_quota"] == [9, 10, 10, 11]
    assert sum(d["phase_quota"]) == d["quota_size"]


# 3. Deterministic parity

def test_deterministic_with_same_seed():
    pol_a = make_fixed(seed=42)
    pol_b = make_fixed(seed=42)
    tr_a = FakeTrainer(TOTAL_STEPS)
    tr_b = FakeTrainer(TOTAL_STEPS)

    dec_a = run_horizon(pol_a, tr_a)
    dec_b = run_horizon(pol_b, tr_b)

    assert dec_a == dec_b
    da, db = pol_a.get_diagnostics(), pol_b.get_diagnostics()
    assert da["phase_skips"] == db["phase_skips"]
    assert da["skip_decisions"] == db["skip_decisions"]


# 4. Max-consecutive-skips enforcement

def test_max_consecutive_skips_enforcement():
    policy = make_fixed(max_consecutive_skips=1)
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._rng = _AlwaysZero()

    for _ in range(MIN_STEP):
        policy.should_skip(trainer, None, None)

    assert policy.should_skip(trainer, None, None) is True
    assert policy._consecutive_skips == 1
    assert policy.should_skip(trainer, None, None) is False

    d = policy.get_diagnostics()
    assert d["max_consecutive_veto_count"] >= 1
    assert policy._consecutive_skips == 0


# 5. Exact quota guarantee

def test_exact_quota_guarantee():
    policy = make_fixed(max_consecutive_skips=100)
    trainer = FakeTrainer(TOTAL_STEPS)
    run_horizon(policy, trainer)
    d = policy.get_diagnostics()

    assert d["skip_decisions"] == d["quota_size"]
    assert d["quota_exact"] is True


# 6. Cumulative phase debt accounting

def test_cumulative_phase_debt_accounting():
    policy = make_fixed(max_consecutive_skips=100)
    trainer = FakeTrainer(TOTAL_STEPS)
    run_horizon(policy, trainer)
    d = policy.get_diagnostics()

    assert d["phase_quota"] == [9, 10, 10, 11]
    assert d["skip_decisions"] == d["quota_size"]
    assert d["phase_debt_carried_total"] >= 0
    assert d["quota_exact"] is True


# 7. No signal veto counters present

def test_no_signal_veto_counters():
    policy = make_fixed()
    trainer = FakeTrainer(TOTAL_STEPS)
    run_horizon(policy, trainer)
    d = policy.get_diagnostics()

    assert d["rho_veto_count"] == 0
    assert d["spike_veto_count"] == 0


# 8. Quota exhaustion blocks further skips

def test_quota_exhaustion_blocks_further_skips():
    policy = make_fixed(target_skip_rate=0.01)
    trainer = FakeTrainer(100)
    run_horizon(policy, trainer)
    d = policy.get_diagnostics()

    assert d["skip_decisions"] == d["quota_size"]
    for _ in range(10):
        assert policy.should_skip(trainer, None, None) is False


# 9. Parity with legacy guarded policy ---------------------------------------

def test_parity_with_legacy_guarded_policy():
    legacy = make_legacy()
    guarded = make_guarded()
    trainer = FakeTrainer(TOTAL_STEPS)

    run_horizon(legacy, trainer)
    run_horizon(guarded, trainer)

    d_legacy = legacy.get_diagnostics()
    d_guarded = guarded.get_diagnostics()

    assert d_legacy["phase_quota"] == d_guarded["phase_quota"]
    assert d_legacy["phase_skips"] == d_guarded["phase_skips"]
    assert d_legacy["skip_decisions"] == d_guarded["skip_decisions"]
    assert d_legacy["quota_size"] == d_guarded["quota_size"]
    assert d_legacy["forced_global_tail_skip_count"] == d_guarded["forced_global_tail_skip_count"]
    assert d_legacy["forced_safety_override_count"] == d_guarded["forced_safety_override_count"]
    assert d_legacy["phase_debt_carried_total"] == d_guarded["phase_debt_carried_total"]
    d_guarded = {**d_guarded, "policy_name": d_legacy["policy_name"]}
    assert d_legacy == d_guarded
