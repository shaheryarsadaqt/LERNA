"""Step 3B.2 focused deterministic tests for LERGuidedStratifiedSafetyPolicy."""

import random

import pytest

from lerna.trainers.policies import (
    LERGuidedStratifiedPolicy,
    LERGuidedStratifiedSafetyPolicy,
)


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class FakeSampledLaggedTracker:
    def __init__(self):
        self.ler_raw = None
        self.observation_age_decisions = None
        self.last_update_decision = None
        self.rho_vg_raw = None
        self.rho_vg = None
        self.loss_history = []
        self.update_calls = 0
        self.note_calls = 0

    def update(self, *args, **kwargs):
        self.update_calls += 1

    def note_decision(self, *args, **kwargs):
        self.note_calls += 1

    def get_diagnostics(self):
        return {
            "mode": "sampled_lagged",
            "timing": "post_decision_after_backward",
            "ler_raw": self.ler_raw,
            "ler": self.ler_raw,
            "observation_age_decisions": self.observation_age_decisions,
            "last_update_decision": self.last_update_decision,
            "rho_vg_raw": self.rho_vg_raw,
            "rho_vg": self.rho_vg,
        }


def make_policy(tracker=None, cls=LERGuidedStratifiedSafetyPolicy, **kwargs):
    tracker = tracker if tracker is not None else FakeSampledLaggedTracker()
    defaults = dict(
        target_skip_rate=0.20,
        total_steps=40,
        min_step=4,
        seed=42,
        n_phases=4,
    )
    defaults.update(kwargs)
    return cls(tracker, **defaults), tracker


def run_decisions(policy, trainer, n):
    return [policy.should_skip(trainer, None, None) for _ in range(n)]


def prime_guided_skip(policy, tracker):
    tracker.ler_raw = 0.1
    tracker.observation_age_decisions = 0
    tracker.last_update_decision = 3
    policy._ler_observations = [1.0, 2.0, 0.1]
    policy._last_signal_update_decision = 3
    policy._current_ler = 0.1
    policy._current_ler_rank = policy._percentile_rank(0.1)
    policy._current_signal_age = 0
    policy._rng = random.Random(1)


def test_rho_veto_blocks_skip_and_counts():
    safe_tracker = FakeSampledLaggedTracker()
    safe_tracker.rho_vg_raw = -0.5
    safe, _ = make_policy(safe_tracker)
    base_tracker = FakeSampledLaggedTracker()
    base_tracker.rho_vg_raw = -0.5
    base, _ = make_policy(base_tracker, cls=LERGuidedStratifiedPolicy)
    trainer = FakeTrainer(40)
    assert run_decisions(safe, trainer, 4) == [False] * 4
    assert run_decisions(base, trainer, 4) == [False] * 4
    prime_guided_skip(safe, safe_tracker)
    prime_guided_skip(base, base_tracker)

    assert base.should_skip(trainer, None, None) is True
    assert safe.should_skip(trainer, None, None) is False
    d = safe.get_diagnostics()
    assert d["rho_veto_count"] == 1
    assert d["spike_veto_count"] == 0
    assert d["safety_veto_count"] == 1
    assert d["forced_safety_override_count"] == 0
    assert d["skip_decisions"] == 0
    assert d["rho_last"] == -0.5


def test_rho_reads_raw_first_zero_valid_and_nonfinite_ignored():
    tracker = FakeSampledLaggedTracker()
    tracker.rho_vg_raw = 0.0
    tracker.rho_vg = -0.9
    policy, _ = make_policy(tracker)
    run_decisions(policy, FakeTrainer(40), 6)
    d = policy.get_diagnostics()
    assert d["rho_veto_count"] == 0
    assert d["rho_last"] == 0.0

    tracker2 = FakeSampledLaggedTracker()
    tracker2.rho_vg = -0.9
    policy2, _ = make_policy(tracker2)
    run_decisions(policy2, FakeTrainer(40), 6)
    assert policy2.get_diagnostics()["rho_veto_count"] == 2

    tracker3 = FakeSampledLaggedTracker()
    tracker3.rho_vg_raw = float("nan")
    policy3, _ = make_policy(tracker3)
    run_decisions(policy3, FakeTrainer(40), 6)
    d3 = policy3.get_diagnostics()
    assert d3["rho_veto_count"] == 0
    assert d3["rho_last"] is None


def test_loss_spike_veto_semantics():
    safe_tracker = FakeSampledLaggedTracker()
    safe_tracker.loss_history = [1.0] * 5 + [3.0]
    safe, _ = make_policy(safe_tracker)
    base_tracker = FakeSampledLaggedTracker()
    base_tracker.loss_history = [1.0] * 5 + [3.0]
    base, _ = make_policy(base_tracker, cls=LERGuidedStratifiedPolicy)
    trainer = FakeTrainer(40)
    assert run_decisions(safe, trainer, 4) == [False] * 4
    assert run_decisions(base, trainer, 4) == [False] * 4
    prime_guided_skip(safe, safe_tracker)
    prime_guided_skip(base, base_tracker)

    assert base.should_skip(trainer, None, None) is True
    assert safe.should_skip(trainer, None, None) is False
    d = safe.get_diagnostics()
    assert d["spike_veto_count"] == 1
    assert d["rho_veto_count"] == 0
    assert d["safety_veto_count"] == 1
    assert d["loss_last"] == 3.0
    assert d["skip_decisions"] == 0

    tracker2 = FakeSampledLaggedTracker()
    tracker2.loss_history = [1.0] * 5 + [2.0]
    policy2, _ = make_policy(tracker2)
    run_decisions(policy2, FakeTrainer(40), 6)
    assert policy2.get_diagnostics()["spike_veto_count"] == 0

    tracker3 = FakeSampledLaggedTracker()
    tracker3.loss_history = [1.0] * 5 + [float("inf")]
    policy3, _ = make_policy(tracker3)
    run_decisions(policy3, FakeTrainer(40), 6)
    assert policy3.get_diagnostics()["spike_veto_count"] == 0

    tracker4 = FakeSampledLaggedTracker()
    tracker4.loss_history = [1.0] * 4 + [10.0]
    policy4, _ = make_policy(tracker4)
    run_decisions(policy4, FakeTrainer(40), 6)
    assert policy4.get_diagnostics()["spike_veto_count"] == 0


def test_rho_priority_over_spike():
    tracker = FakeSampledLaggedTracker()
    tracker.rho_vg_raw = -0.9
    tracker.loss_history = [1.0] * 5 + [10.0]
    policy, _ = make_policy(tracker)
    run_decisions(policy, FakeTrainer(40), 6)
    d = policy.get_diagnostics()
    assert d["rho_veto_count"] == 2
    assert d["spike_veto_count"] == 0
    assert d["safety_veto_count"] == d["rho_veto_count"] + d["spike_veto_count"]


def test_disabled_safety_matches_ler_only_policy():
    def dangerous_tracker():
        t = FakeSampledLaggedTracker()
        t.rho_vg_raw = -0.9
        t.loss_history = [1.0] * 5 + [10.0]
        return t

    base, _ = make_policy(dangerous_tracker(), cls=LERGuidedStratifiedPolicy)
    safe_off, _ = make_policy(
        dangerous_tracker(),
        use_rho_vg_safety=False,
        use_loss_spike_safety=False,
    )
    base_decisions = run_decisions(base, FakeTrainer(40), 40)
    off_decisions = run_decisions(safe_off, FakeTrainer(40), 40)
    assert base_decisions == off_decisions
    db = base.get_diagnostics()
    ds = safe_off.get_diagnostics()
    assert db["skip_decisions"] == ds["skip_decisions"]
    assert ds["rho_veto_count"] == 0
    assert ds["spike_veto_count"] == 0
    assert ds["safety_veto_count"] == 0
    assert ds["forced_safety_override_count"] == 0
    assert base._rng.getstate() == safe_off._rng.getstate()


def test_forced_global_tail_overrides_safety():
    tracker = FakeSampledLaggedTracker()
    tracker.rho_vg_raw = -0.9
    policy, _ = make_policy(
        tracker, target_skip_rate=0.5, total_steps=20, min_step=2
    )
    decisions = run_decisions(policy, FakeTrainer(20), 20)
    d = policy.get_diagnostics()
    assert d["quota_size"] == 10
    assert decisions[:10] == [False] * 10
    assert decisions[10:] == [True] * 10
    assert d["skip_decisions"] == 10
    assert d["forced_safety_override_count"] == 10
    assert d["rho_veto_count"] == 8
    assert d["spike_veto_count"] == 0
    assert d["safety_veto_count"] == 8
    assert d["forced_global_tail_skip_count"] == 10
    assert d["forced_tail_skip_count"] == 0
    assert d["quota_exact"] is True


def test_exact_quota_with_safe_signals():
    tracker = FakeSampledLaggedTracker()
    tracker.rho_vg_raw = 0.5
    tracker.loss_history = [1.0] * 10
    policy, _ = make_policy(tracker)
    run_decisions(policy, FakeTrainer(40), 40)
    d = policy.get_diagnostics()
    assert d["quota_size"] == 8
    assert d["skip_decisions"] == 8
    assert d["quota_exact"] is True
    assert d["safety_veto_count"] == 0
    assert d["forced_safety_override_count"] == 0


def test_determinism_same_seed():
    def build():
        t = FakeSampledLaggedTracker()
        t.rho_vg_raw = 0.5
        t.loss_history = [1.0] * 10
        t.ler_raw = 0.01
        t.observation_age_decisions = 0
        t.last_update_decision = 0
        return make_policy(t)[0]

    p1, p2 = build(), build()
    d1 = run_decisions(p1, FakeTrainer(40), 40)
    d2 = run_decisions(p2, FakeTrainer(40), 40)
    assert d1 == d2
    assert p1.get_diagnostics() == p2.get_diagnostics()


def test_diagnostics_and_effective_config_keys():
    assert issubclass(LERGuidedStratifiedSafetyPolicy, LERGuidedStratifiedPolicy)
    policy, _ = make_policy(
        rho_veto_threshold=-0.3,
        loss_spike_factor=0.5,
        loss_spike_window=7,
    )
    d = policy.get_diagnostics()
    assert d["safety_enabled"] is True
    for key in (
        "rho_veto_count",
        "spike_veto_count",
        "safety_veto_count",
        "forced_safety_override_count",
        "rho_last",
        "loss_last",
    ):
        assert key in d
    assert d["use_rho_vg_safety"] is True
    assert d["rho_veto_threshold"] == -0.3
    assert d["use_loss_spike_safety"] is True
    assert d["loss_spike_factor"] == 0.5
    assert d["loss_spike_window"] == 7

    cfg = policy.effective_config()
    assert cfg["policy_class"] == "LERGuidedStratifiedSafetyPolicy"
    assert cfg["policy_name"] == "ler_guided_stratified_safe"
    assert cfg["safety_enabled"] is True
    assert cfg["use_rho_vg_safety"] is True
    assert cfg["rho_veto_threshold"] == -0.3
    assert cfg["use_loss_spike_safety"] is True
    assert cfg["loss_spike_factor"] == 0.5
    assert cfg["loss_spike_window"] == 7


def test_veto_consumes_no_rng_no_skip_counters_tracker_immutable():
    tracker = FakeSampledLaggedTracker()
    tracker.rho_vg_raw = -0.9
    tracker.loss_history = [1.0] * 5 + [10.0]
    snapshot = list(tracker.loss_history)
    policy, _ = make_policy(tracker)
    decisions = run_decisions(policy, FakeTrainer(40), 10)
    assert decisions == [False] * 10
    assert tracker.loss_history == snapshot
    assert tracker.update_calls == 0
    assert tracker.note_calls == 0
    assert policy._rng.getstate() == random.Random(42).getstate()
    d = policy.get_diagnostics()
    assert d["skip_decisions"] == 0
    assert d["ler_selected_skip_count"] == 0
    assert d["fallback_random_skip_count"] == 0
    assert d["forced_tail_skip_count"] == 0
    assert d["forced_global_tail_skip_count"] == 0
    assert d["forced_quota_decision_count"] == 0
    assert d["random_safe_skip_count"] == 0
    assert d["safety_veto_count"] == d["rho_veto_count"] + d["spike_veto_count"]


def test_constructor_validation():
    with pytest.raises(ValueError):
        make_policy(rho_veto_threshold=float("nan"))
    with pytest.raises(ValueError):
        make_policy(loss_spike_factor=float("inf"))
    with pytest.raises(ValueError):
        make_policy(loss_spike_factor=-0.1)
    with pytest.raises(ValueError):
        make_policy(loss_spike_window=0)