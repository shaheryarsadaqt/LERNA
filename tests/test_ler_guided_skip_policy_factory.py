"""Step 3B.3A3: focused factory tests for build_ler_guided_skip_policy.

Deterministic, lightweight tests using fake trackers only. No models,
datasets, GPUs, or statistical assertions.
"""

import pytest

from lerna.trainers.policies import (
    LERGuidedStratifiedPolicy,
    LERGuidedStratifiedSafetyPolicy,
)
from scripts.run_ablation_study import (
    LER_GUIDED_CONTROL,
    LER_GUIDED_CONTROLS,
    LER_GUIDED_SAFE_CONTROL,
    build_ler_guided_controller_config,
    build_ler_guided_skip_policy,
)


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class FakeLagTracker:
    """Fake SampledLaggedLERTracker satisfying the required diagnostics."""

    def __init__(self, mode="sampled_lagged", timing="post_decision_after_backward"):
        self.mode = mode
        self.timing = timing
        self.ler_raw = None
        self.last_update_decision = None
        self.observation_age_decisions = 0
        self.rho_vg_raw = None
        self.loss_history = []
        self.update_calls = 0
        self.note_calls = 0

    def commit(self, ler, decision_index):
        self.ler_raw = ler
        self.last_update_decision = decision_index
        self.observation_age_decisions = 0

    def tick(self):
        self.observation_age_decisions += 1

    def update(self, *a, **k):
        self.update_calls += 1

    def note_decision(self, *a, **k):
        self.note_calls += 1

    def get_diagnostics(self):
        return {
            "mode": self.mode,
            "timing": self.timing,
            "ler_raw": self.ler_raw,
            "ler": self.ler_raw,
            "last_update_decision": self.last_update_decision,
            "observation_age_decisions": self.observation_age_decisions,
            "rho_vg_raw": self.rho_vg_raw,
            "rho_vg": self.rho_vg_raw,
        }


TOTAL_STEPS = 100
MIN_STEP = 50
RATE = 0.20


def make_config(control=LER_GUIDED_CONTROL, **overrides):
    kwargs = {
        "control": control,
        "target_skip_rate": RATE,
        "total_steps": TOTAL_STEPS,
        "policy_seed": 42,
        "max_consecutive_skips": 4,
        "probe_interval": 8,
        "rho_veto_threshold": -0.2,
    }
    kwargs.update(overrides)
    return build_ler_guided_controller_config(**kwargs)


def build_policy(control=LER_GUIDED_CONTROL, config_overrides=None, tracker=None):
    tracker = tracker or FakeLagTracker()
    config = make_config(control=control, **(config_overrides or {}))
    return build_ler_guided_skip_policy(
        ler_tracker=tracker,
        ler_guided_controller=config,
    ), config, tracker


def run_stream(policy, tracker, commits, total=TOTAL_STEPS):
    trainer = FakeTrainer(total)
    decisions = []
    for di in range(total):
        if di in commits:
            tracker.commit(*commits[di])
        decisions.append(bool(policy.should_skip(trainer, None, None)))
        tracker.tick()
    return decisions


def test_factory_returns_correct_class_for_no_safety():
    policy, config, _ = build_policy(LER_GUIDED_CONTROL)
    assert type(policy) is LERGuidedStratifiedPolicy
    assert policy.name == LER_GUIDED_CONTROL
    assert policy.safety_enabled is False
    assert policy.get_diagnostics()["safety_enabled"] is False
    assert policy.effective_config()["safety_enabled"] is False
    assert policy.ler_tracker is not None


def test_factory_returns_correct_class_for_safety():
    policy, config, _ = build_policy(
        LER_GUIDED_SAFE_CONTROL,
        config_overrides={"rho_veto_threshold": -0.35},
    )
    assert type(policy) is LERGuidedStratifiedSafetyPolicy
    assert policy.name == LER_GUIDED_SAFE_CONTROL
    assert policy.safety_enabled is True
    assert policy.get_diagnostics()["safety_enabled"] is True
    assert policy.effective_config()["safety_enabled"] is True
    diag = policy.get_diagnostics()
    assert diag["rho_veto_threshold"] == -0.35
    assert diag["use_rho_vg_safety"] is True
    assert diag["use_loss_spike_safety"] is True
    assert diag["loss_spike_factor"] == 1.0
    assert diag["loss_spike_window"] == 5


@pytest.mark.parametrize("control", sorted(LER_GUIDED_CONTROLS))
def test_factory_threads_canonical_common_parameters(control):
    policy, config, _ = build_policy(control)
    assert policy.target_skip_rate == RATE
    assert policy.total_steps == TOTAL_STEPS
    assert policy.min_step == MIN_STEP
    assert policy.seed == 42
    assert policy.n_phases == 4
    assert policy.phase_weights == [0.22, 0.24, 0.26, 0.28]
    assert policy.max_consecutive_skips == 4
    assert policy.probe_interval == 8
    assert policy.min_ler_observations == 3
    assert policy.ler_guidance_strength == 1.0
    assert policy.REQUIRED_TRACKER_MODE == "sampled_lagged"
    assert policy.REQUIRED_TRACKER_TIMING == "post_decision_after_backward"
    cfg = policy.effective_config()
    assert cfg["policy_name"] == control
    assert cfg["required_tracker_mode"] == "sampled_lagged"
    assert cfg["required_tracker_timing"] == "post_decision_after_backward"


def test_factory_rejects_unknown_control():
    config = make_config()
    config["control"] = "rvd"
    with pytest.raises(ValueError, match="control"):
        build_ler_guided_skip_policy(
            ler_tracker=FakeLagTracker(),
            ler_guided_controller=config,
        )


def test_safety_enabled_is_derived_from_control_not_config_flag():
    # A malformed non-safety config that claims safety_enabled=True must still
    # produce the non-safety class; the control field is authoritative.
    config = make_config(LER_GUIDED_CONTROL)
    config["safety_enabled"] = True
    policy = build_ler_guided_skip_policy(
        ler_tracker=FakeLagTracker(),
        ler_guided_controller=config,
    )
    assert type(policy) is LERGuidedStratifiedPolicy
    assert policy.safety_enabled is False

    # A malformed safety config with no safety parameters must still map to
    # the safety class only if the control says so; missing safety keys then
    # fail loudly rather than silently mis-arming.
    config2 = make_config(LER_GUIDED_SAFE_CONTROL)
    config2.pop("use_rho_vg_safety")
    with pytest.raises(KeyError):
        build_ler_guided_skip_policy(
            ler_tracker=FakeLagTracker(),
            ler_guided_controller=config2,
        )


def test_factory_enforces_tracker_validation():
    with pytest.raises(ValueError):
        build_ler_guided_skip_policy(
            ler_tracker=None,
            ler_guided_controller=make_config(),
        )
    with pytest.raises(ValueError):
        build_ler_guided_skip_policy(
            ler_tracker=FakeLagTracker(mode="off"),
            ler_guided_controller=make_config(),
        )


def test_factory_does_not_mutate_config():
    policy, config, _ = build_policy(LER_GUIDED_CONTROL)
    assert config["phase_weights"] == [0.22, 0.24, 0.26, 0.28]
    assert policy.phase_weights == config["phase_weights"]


def test_factory_policy_reaches_exact_quota():
    commits = {
        di: (float((di % 7) + 1), di) for di in range(0, TOTAL_STEPS, 5)
    }
    policy, _, tracker = build_policy(LER_GUIDED_CONTROL)
    decisions = run_stream(policy, tracker, commits)
    diag = policy.get_diagnostics()
    assert diag["decisions_seen"] == TOTAL_STEPS
    assert diag["skip_decisions"] == diag["quota_size"] == int(RATE * TOTAL_STEPS)
    assert diag["quota_exact"] is True
    assert sum(decisions) == diag["quota_size"]
    assert not any(decisions[:MIN_STEP])


def test_factory_safety_policy_vetoes_rho_thrash():
    tracker = FakeLagTracker()
    tracker.rho_vg_raw = -0.9
    policy = build_ler_guided_skip_policy(
        ler_tracker=tracker,
        ler_guided_controller=make_config(LER_GUIDED_SAFE_CONTROL),
    )
    # Drive the full stream: after warmup, ordinary rho vetoes block skips.
    run_stream(policy, tracker, commits={})
    diag = policy.get_diagnostics()
    assert diag["rho_veto_count"] > 0
    assert diag["safety_veto_count"] == diag["rho_veto_count"]
    assert diag["rho_last"] == -0.9


def test_factory_deterministic_same_config_same_decisions():
    commits = {0: (1.0, 0), 5: (2.0, 5), 25: (0.5, 25), 45: (3.0, 45)}
    p1, _, t1 = build_policy(LER_GUIDED_CONTROL, config_overrides={"policy_seed": 7})
    p2, _, t2 = build_policy(LER_GUIDED_CONTROL, config_overrides={"policy_seed": 7})
    assert run_stream(p1, t1, commits) == run_stream(p2, t2, commits)
    assert p1.get_diagnostics() == p2.get_diagnostics()


def test_factory_policy_never_mutates_tracker():
    policy, _, tracker = build_policy(LER_GUIDED_CONTROL)
    run_stream(policy, tracker, commits={0: (1.0, 0)})
    assert tracker.update_calls == 0
    assert tracker.note_calls == 0