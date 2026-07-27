"""Exact-quota tests for LERNAPhaseStratifiedPolicy.

Lightweight fake trainer/tracker objects; no trainer/runner imports.
"""

import pytest

from lerna.trainers.policies import LERNAPhaseStratifiedPolicy


class FakeTracker:
    def __init__(self):
        self.rho = 1.0
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


TOTAL_STEPS = 200
MIN_STEP = 10


def make_policy(target_skip_rate, seed=42, **overrides):
    kwargs = dict(
        ler_tracker=FakeTracker(),
        target_skip_rate=target_skip_rate,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=seed,
    )
    kwargs.update(overrides)
    return LERNAPhaseStratifiedPolicy(**kwargs)


def run_policy(policy, total_steps=TOTAL_STEPS):
    trainer = FakeTrainer()
    skip_indices = []
    for i in range(total_steps):
        if policy.should_skip(trainer, None, None):
            skip_indices.append(i)
    return skip_indices


def run_policy_with_rho(policy, rho_fn, total_steps=TOTAL_STEPS):
    trainer = FakeTrainer()
    skip_indices = []
    for i in range(total_steps):
        policy.trk.rho = rho_fn(i)
        if policy.should_skip(trainer, None, None):
            skip_indices.append(i)
    return skip_indices


def test_exact_quota_30_percent():
    policy = make_policy(0.30)
    skips = run_policy(policy)
    quota = round(0.30 * TOTAL_STEPS)
    assert len(skips) == quota
    diagnostics = policy.get_diagnostics()
    assert diagnostics["skip_decisions"] == quota
    assert diagnostics["quota_size"] == quota


def test_exact_quota_40_percent():
    policy = make_policy(0.40)
    skips = run_policy(policy)
    quota = round(0.40 * TOTAL_STEPS)
    assert len(skips) == quota
    assert policy.get_diagnostics()["realized_skip_rate"] == quota / TOTAL_STEPS


def test_same_seed_identical_indices():
    first = run_policy(make_policy(0.40, seed=7))
    second = run_policy(make_policy(0.40, seed=7))
    assert first == second


def test_different_seeds_differ_but_exact_quota():
    first = run_policy(make_policy(0.40, seed=7))
    second = run_policy(make_policy(0.40, seed=8))
    quota = round(0.40 * TOTAL_STEPS)
    assert len(first) == quota
    assert len(second) == quota
    assert first != second


def test_veto_heavy_creates_debt_and_carries_forward():
    policy = make_policy(0.40)
    dangerous_until = TOTAL_STEPS // 2

    def rho_fn(index):
        return -1.0 if index < dangerous_until else 1.0

    skips = run_policy_with_rho(policy, rho_fn)
    quota = round(0.40 * TOTAL_STEPS)
    diagnostics = policy.get_diagnostics()
    assert diagnostics["rho_veto_count"] > 0
    assert diagnostics["phase_debt_carried_total"] > 0
    assert len(skips) == quota
    assert diagnostics["skip_decisions"] == quota


def test_global_tail_enforcement_is_counted():
    policy = make_policy(0.30)
    skips = run_policy_with_rho(policy, lambda index: -1.0)
    quota = round(0.30 * TOTAL_STEPS)
    diagnostics = policy.get_diagnostics()
    assert len(skips) == quota
    assert diagnostics["forced_global_tail_skip_count"] > 0
    assert diagnostics["forced_safety_override_count"] > 0
    assert skips == list(range(TOTAL_STEPS - quota, TOTAL_STEPS))


def test_infeasible_quota_raises_value_error():
    policy = make_policy(0.30, total_steps=100, min_step=80)
    with pytest.raises(ValueError, match="Infeasible skip quota"):
        policy.should_skip(FakeTrainer(), None, None)


def test_final_diagnostics_quota_exact_true():
    policy = make_policy(0.40)
    run_policy(policy)
    diagnostics = policy.get_diagnostics()
    assert diagnostics["decisions_seen"] == TOTAL_STEPS
    assert diagnostics["quota_exact"] is True
