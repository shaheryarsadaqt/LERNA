"""Tests for LERGuidedStratifiedPolicy (Step 3B.1, no safety).

Deterministic, lightweight tests using fake trainer/tracker objects only.
No models, datasets, GPUs, Transformers Trainer, downloads, or statistical
assertions.
"""

import pytest

from lerna.trainers.policies import (
    FixedPhaseStratifiedRandomPolicy,
    LERGuidedStratifiedPolicy,
)


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class FakeLaggedTracker:
    """Fake SampledLaggedLERTracker exposing only committed diagnostics."""

    def __init__(self, mode="sampled_lagged", timing="post_decision_after_backward"):
        self.mode = mode
        self.timing = timing
        self.ler_raw = None
        self.ler = None
        self.last_update_decision = None
        self.observation_age_decisions = 0
        self.extra = {}
        self.update_calls = 0
        self.note_calls = 0

    def commit(self, ler, decision_index):
        self.ler_raw = ler
        self.ler = ler
        self.last_update_decision = decision_index
        self.observation_age_decisions = 0

    def tick(self):
        self.observation_age_decisions += 1

    def update(self, *a, **k):
        self.update_calls += 1

    def note_decision(self, *a, **k):
        self.note_calls += 1

    def get_diagnostics(self):
        d = {
            "mode": self.mode,
            "timing": self.timing,
            "ler_raw": self.ler_raw,
            "ler": self.ler,
            "last_update_decision": self.last_update_decision,
            "observation_age_decisions": self.observation_age_decisions,
        }
        d.update(self.extra)
        return d


class ScriptedRNG:
    def __init__(self, values):
        self.values = list(values)
        self.i = 0

    def random(self):
        v = self.values[min(self.i, len(self.values) - 1)]
        self.i += 1
        return v


TOTAL_STEPS = 100
MIN_STEP = 20
RATE = 0.20


def make_policy(tracker, **overrides):
    kwargs = dict(
        ler_tracker=tracker,
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        n_phases=4,
        max_consecutive_skips=4,
        probe_interval=8,
        min_ler_observations=3,
        ler_guidance_strength=1.0,
    )
    kwargs.update(overrides)
    return LERGuidedStratifiedPolicy(**kwargs)


def run_stream(policy, tracker, commits, total=TOTAL_STEPS):
    """Drive the policy; commits maps decision index -> (ler, update_decision)."""
    trainer = FakeTrainer(total)
    decisions = []
    for di in range(total):
        if di in commits:
            tracker.commit(*commits[di])
        decisions.append(bool(policy.should_skip(trainer, None, None)))
        tracker.tick()
    return decisions


def test_incompatible_or_missing_tracker_rejected():
    with pytest.raises((ValueError, RuntimeError)):
        make_policy(None)
    with pytest.raises((ValueError, RuntimeError)):
        make_policy(object())
    with pytest.raises((ValueError, RuntimeError)):
        make_policy(FakeLaggedTracker(mode="legacy"))
    with pytest.raises((ValueError, RuntimeError)):
        make_policy(FakeLaggedTracker(mode="off"))
    with pytest.raises((ValueError, RuntimeError)):
        make_policy(FakeLaggedTracker(timing="pre_decision"))
    policy = make_policy(FakeLaggedTracker())
    assert policy.name == "ler_guided_stratified"


def test_equal_seeds_and_signals_are_deterministic():
    commits = {
        0: (1.0, 0),
        5: (2.0, 5),
        25: (0.5, 25),
        45: (3.0, 45),
        70: (1.5, 70),
    }
    t1, t2 = FakeLaggedTracker(), FakeLaggedTracker()
    p1, p2 = make_policy(t1, seed=7), make_policy(t2, seed=7)
    d1 = run_stream(p1, t1, commits)
    d2 = run_stream(p2, t2, commits)
    assert d1 == d2
    g1, g2 = p1.get_diagnostics(), p2.get_diagnostics()
    for key in (
        "skip_decisions",
        "signal_observation_count",
        "ler_guided_decision_count",
        "ler_selected_skip_count",
        "probe_decision_count",
    ):
        assert g1[key] == g2[key]


def test_duplicate_last_update_decision_not_recorded_twice():
    tracker = FakeLaggedTracker()
    policy = make_policy(tracker)
    trainer = FakeTrainer(TOTAL_STEPS)
    tracker.commit(1.0, 5)
    for _ in range(6):
        policy.should_skip(trainer, None, None)
        tracker.tick()
    assert policy.get_diagnostics()["signal_observation_count"] == 1
    tracker.commit(2.0, 6)
    policy.should_skip(trainer, None, None)
    diag = policy.get_diagnostics()
    assert diag["signal_observation_count"] == 2
    assert diag["last_signal_update_decision"] == 6


def test_missing_and_stale_signals_force_probes():
    tracker = FakeLaggedTracker()
    policy = make_policy(tracker)
    decisions = run_stream(policy, tracker, commits={})
    diag = policy.get_diagnostics()
    assert diag["missing_signal_probe_count"] > 0
    assert diag["stale_signal_probe_count"] == 0
    assert diag["probe_decision_count"] == diag["missing_signal_probe_count"]
    assert diag["signal_observation_count"] == 0
    assert diag["skip_decisions"] == diag["quota_size"]
    assert diag["quota_exact"] is True
    assert sum(decisions) == diag["quota_size"]

    tracker2 = FakeLaggedTracker()
    policy2 = make_policy(tracker2, min_ler_observations=1)
    run_stream(policy2, tracker2, commits={0: (1.0, 0)})
    diag2 = policy2.get_diagnostics()
    assert diag2["stale_signal_probe_count"] > 0
    assert diag2["missing_signal_probe_count"] == 0
    assert (
        diag2["probe_decision_count"]
        == diag2["stale_signal_probe_count"] + diag2["missing_signal_probe_count"]
    )
    assert diag2["max_signal_age_observed"] >= policy2.probe_interval


def test_low_ler_skips_more_than_high_ler_under_identical_rng():
    results = {}
    for label, final_ler in (("low", 0.1), ("high", 10.0)):
        tracker = FakeLaggedTracker()
        policy = make_policy(tracker)
        trainer = FakeTrainer(TOTAL_STEPS)
        commits = {0: (1.0, 0), 1: (2.0, 1), 2: (3.0, 2), 19: (final_ler, 3)}
        for di in range(20):
            if di in commits:
                tracker.commit(*commits[di])
            policy.should_skip(trainer, None, None)
            tracker.tick()
        policy._rng = ScriptedRNG([0.2])
        results[label] = (
            bool(policy.should_skip(trainer, None, None)),
            policy.get_diagnostics()["current_ler_rank"],
        )
    low_skip, low_rank = results["low"]
    high_skip, high_rank = results["high"]
    assert low_rank < 0.5 < high_rank
    assert low_skip is True
    assert high_skip is False


def test_ler_guidance_only_after_min_observations():
    tracker = FakeLaggedTracker()
    policy = make_policy(tracker, probe_interval=1000)
    trainer = FakeTrainer(TOTAL_STEPS)
    tracker.commit(1.0, 0)
    for _ in range(30):
        policy.should_skip(trainer, None, None)
        tracker.tick()
    diag = policy.get_diagnostics()
    assert diag["signal_observation_count"] == 1
    assert diag["ler_guided_decision_count"] == 0
    assert diag["fallback_random_decision_count"] > 0
    tracker.commit(2.0, 30)
    policy.should_skip(trainer, None, None)
    tracker.tick()
    tracker.commit(3.0, 31)
    for _ in range(10):
        policy.should_skip(trainer, None, None)
        tracker.tick()
    assert policy.get_diagnostics()["ler_guided_decision_count"] > 0


def test_no_safety_inputs_affect_decisions():
    commits = {0: (1.0, 0), 10: (2.0, 10), 30: (0.5, 30), 60: (4.0, 60)}
    t1, t2 = FakeLaggedTracker(), FakeLaggedTracker()
    t1.extra = {
        "rho_vg_raw": 100.0,
        "rho_vg": 100.0,
        "loss_spike": True,
        "grad_norm": 1e9,
        "safety_veto": True,
    }
    t2.extra = {
        "rho_vg_raw": -100.0,
        "rho_vg": -100.0,
        "loss_spike": False,
        "grad_norm": 0.0,
        "safety_veto": False,
    }
    p1, p2 = make_policy(t1, seed=99), make_policy(t2, seed=99)
    d1 = run_stream(p1, t1, commits)
    d2 = run_stream(p2, t2, commits)
    assert d1 == d2
    assert p1.get_diagnostics()["safety_enabled"] is False
    assert p1.effective_config()["safety_enabled"] is False
    assert t1.update_calls == 0
    assert t1.note_calls == 0


def test_forced_quota_handling_reaches_exact_final_quota():
    commits = {
        di: (float((di % 7) + 1), di) for di in range(0, TOTAL_STEPS, 5)
    }
    tracker = FakeLaggedTracker()
    policy = make_policy(tracker)
    decisions = run_stream(policy, tracker, commits)
    diag = policy.get_diagnostics()
    assert diag["decisions_seen"] == TOTAL_STEPS
    assert diag["skip_decisions"] == diag["quota_size"] == int(RATE * TOTAL_STEPS)
    assert diag["quota_exact"] is True
    assert sum(decisions) == diag["quota_size"]
    assert not any(decisions[:MIN_STEP])
    assert (
        diag["forced_quota_decision_count"]
        == diag["forced_tail_skip_count"] + diag["forced_global_tail_skip_count"]
    )


def test_diagnostics_report_probe_guidance_and_quota_invariants():
    commits = {
        di: (float((di * 3) % 5 + 1), di) for di in range(0, TOTAL_STEPS, 9)
    }
    tracker = FakeLaggedTracker()
    policy = make_policy(tracker)
    run_stream(policy, tracker, commits)
    diag = policy.get_diagnostics()
    for key in (
        "safety_enabled",
        "probe_interval",
        "min_ler_observations",
        "ler_guidance_strength",
        "signal_observation_count",
        "last_signal_update_decision",
        "current_signal_age",
        "max_signal_age_observed",
        "current_ler",
        "current_ler_rank",
        "probe_decision_count",
        "missing_signal_probe_count",
        "stale_signal_probe_count",
        "ler_guided_decision_count",
        "ler_selected_skip_count",
        "fallback_random_decision_count",
        "fallback_random_skip_count",
        "forced_quota_decision_count",
    ):
        assert key in diag
    assert diag["safety_enabled"] is False
    assert diag["probe_interval"] == 8
    assert diag["min_ler_observations"] == 3
    assert diag["ler_guidance_strength"] == 1.0
    assert (
        diag["probe_decision_count"]
        == diag["missing_signal_probe_count"] + diag["stale_signal_probe_count"]
    )
    assert diag["skip_decisions"] == (
        diag["ler_selected_skip_count"]
        + diag["fallback_random_skip_count"]
        + diag["forced_quota_decision_count"]
    )
    assert diag["ler_selected_skip_count"] <= diag["ler_guided_decision_count"]
    assert diag["fallback_random_skip_count"] <= diag["fallback_random_decision_count"]
    assert diag["rho_veto_count"] == 0
    assert diag["spike_veto_count"] == 0
    assert diag["forced_safety_override_count"] == 0
    assert diag["quota_exact"] is True
    assert diag["phase_debt_current"] >= 0
    assert diag["phase_debt_carried_total"] >= 0
    cfg = policy.effective_config()
    for key in (
        "policy_class",
        "policy_name",
        "target_skip_rate",
        "total_steps",
        "min_step",
        "n_phases",
        "phase_weights",
        "seed",
        "max_consecutive_skips",
        "probe_interval",
        "min_ler_observations",
        "ler_guidance_strength",
        "required_tracker_mode",
        "required_tracker_timing",
        "safety_enabled",
    ):
        assert key in cfg
    assert cfg["policy_name"] == "ler_guided_stratified"
    assert cfg["required_tracker_mode"] == "sampled_lagged"
    assert cfg["required_tracker_timing"] == "post_decision_after_backward"


def test_fixed_baseline_remains_unchanged():
    assert issubclass(LERGuidedStratifiedPolicy, FixedPhaseStratifiedRandomPolicy)
    assert FixedPhaseStratifiedRandomPolicy.name == "fixed_phase_strat"
    fixed = FixedPhaseStratifiedRandomPolicy(
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
    )
    trainer = FakeTrainer(TOTAL_STEPS)
    skips = sum(
        bool(fixed.should_skip(trainer, None, None)) for _ in range(TOTAL_STEPS)
    )
    diag = fixed.get_diagnostics()
    assert skips == diag["skip_decisions"]
    assert diag["quota_exact"] is False
    for key in (
        "safety_enabled",
        "probe_decision_count",
        "ler_guided_decision_count",
    ):
        assert key not in diag
    assert not hasattr(FixedPhaseStratifiedRandomPolicy, "effective_config")