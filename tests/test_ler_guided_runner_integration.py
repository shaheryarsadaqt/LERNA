"""Step 3B.3B: focused integration tests for LER-guided runner wiring.

Deterministic, lightweight tests using fake trackers only. No models,
datasets, GPUs, or statistical assertions. Verifies the factory routing
through build_skip_policy, defensive config copies, and that existing
explicit arms are byte-for-byte unchanged.
"""

import pytest

from lerna.trainers.policies import (
    AlwaysFalsePolicy,
    GradNormSkipPolicy,
    LERGuidedStratifiedPolicy,
    LERGuidedStratifiedSafetyPolicy,
    LERNARandomVetoDeferralPolicy,
    RandomSkipPolicy,
)
from scripts.run_ablation_study import (
    LER_GUIDED_CONTROL,
    LER_GUIDED_CONTROLS,
    LER_GUIDED_SAFE_CONTROL,
    POLICY_MIN_STEP,
    build_ler_guided_controller_config,
    build_ler_guided_skip_policy,
    build_skip_policy,
    copy_ler_guided_config,
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
RATE = 0.20


def make_controller_cfg(**overrides):
    cfg = {
        "policy_seed": 42,
        "max_consecutive_skips": 4,
        "use_loss_spike_veto": False,
        "spike_factor": 1.0,
        "spike_ema_window": 20,
        "use_margin_veto": False,
        "margin_rank_floor": 0.20,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
    }
    cfg.update(overrides)
    return cfg


def make_ler_guided_config(control=LER_GUIDED_CONTROL, **overrides):
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


def call_build_skip_policy(control, **overrides):
    kwargs = {
        "control": control,
        "ler_tracker": FakeLagTracker(),
        "target_skip_rate": RATE,
        "total_steps": TOTAL_STEPS,
        "controller_cfg": make_controller_cfg(),
        "rho_veto_threshold": -0.2,
        "probe_interval": 8,
        "use_ler": True,
        "use_rho_vg": True,
        "use_safety_horizon": True,
        "fallback_threshold": 0.01,
        "risk_gamma": 0.0,
    }
    kwargs.update(overrides)
    return build_skip_policy(**kwargs)


@pytest.mark.parametrize("control", sorted(LER_GUIDED_CONTROLS))
def test_build_skip_policy_routes_ler_guided_controls(control):
    policy = call_build_skip_policy(
        control,
        ler_guided_controller_config=make_ler_guided_config(control),
    )
    expected = (
        LERGuidedStratifiedSafetyPolicy
        if control == LER_GUIDED_SAFE_CONTROL
        else LERGuidedStratifiedPolicy
    )
    assert type(policy) is expected
    assert policy.name == control


def test_build_skip_policy_requires_config_for_ler_guided():
    with pytest.raises(ValueError, match="ler_guided_controller_config"):
        call_build_skip_policy(LER_GUIDED_CONTROL)


@pytest.mark.parametrize(
    ("control", "config_control"),
    [
        # Safe control with a no-safety config.
        (LER_GUIDED_SAFE_CONTROL, LER_GUIDED_CONTROL),
        # No-safety control with a safe config.
        (LER_GUIDED_CONTROL, LER_GUIDED_SAFE_CONTROL),
    ],
)
def test_build_skip_policy_rejects_control_mismatch(control, config_control):
    with pytest.raises(ValueError, match="does not match controller"):
        call_build_skip_policy(
            control,
            ler_guided_controller_config=make_ler_guided_config(config_control),
        )


def test_build_skip_policy_existing_arms_unchanged():
    # full_finetune
    assert type(call_build_skip_policy("full_finetune")) is AlwaysFalsePolicy
    # exact_random
    er = call_build_skip_policy("exact_random")
    assert type(er) is RandomSkipPolicy
    assert er.target_skip_rate == RATE
    assert er.min_step == POLICY_MIN_STEP
    assert er.seed == 42
    assert er.total_steps == TOTAL_STEPS
    # grad_norm
    gn = call_build_skip_policy("grad_norm")
    assert type(gn) is GradNormSkipPolicy
    assert gn.target_skip_rate == RATE
    assert gn.min_step == POLICY_MIN_STEP
    assert gn.max_consecutive_skips == 4
    # rvd
    rvd = call_build_skip_policy("rvd")
    assert type(rvd) is LERNARandomVetoDeferralPolicy
    assert rvd.target_skip_rate == RATE
    assert rvd.min_step == POLICY_MIN_STEP
    assert rvd.seed == 42
    assert rvd.use_loss_spike_veto is False
    assert rvd.use_margin_veto is False
    assert rvd.repay_mode == "asap"


def test_build_skip_policy_random_skip_alias_still_maps_to_exact_random():
    policy = call_build_skip_policy("random_skip")
    assert type(policy) is RandomSkipPolicy
    assert policy.seed == 42


def test_copy_ler_guided_config_is_defensive():
    config = make_ler_guided_config(LER_GUIDED_SAFE_CONTROL)
    copied = copy_ler_guided_config(config)
    assert copied == config
    assert copied is not config
    assert copied["phase_weights"] is not config["phase_weights"]
    config["phase_weights"][0] = 99.0
    assert copied["phase_weights"] == [0.22, 0.24, 0.26, 0.28]
    copied["target_skip_rate"] = 0.5
    assert config["target_skip_rate"] == RATE


def test_canonical_config_after_total_steps_matches_factory_input():
    # Mirrors run_ablation_single: build the canonical config once total_steps
    # is known, then feed it to build_skip_policy.
    for control in sorted(LER_GUIDED_CONTROLS):
        config = build_ler_guided_controller_config(
            control=control,
            target_skip_rate=RATE,
            total_steps=TOTAL_STEPS,
            policy_seed=42,
            max_consecutive_skips=4,
            probe_interval=8,
            rho_veto_threshold=-0.2,
        )
        policy = build_ler_guided_skip_policy(
            ler_tracker=FakeLagTracker(),
            ler_guided_controller=config,
        )
        assert policy.total_steps == TOTAL_STEPS
        assert policy.min_step == POLICY_MIN_STEP
        assert policy.target_skip_rate == RATE
        assert policy.seed == 42
        assert policy.probe_interval == 8
        assert policy.max_consecutive_skips == 4
        assert policy.phase_weights == [0.22, 0.24, 0.26, 0.28]
        assert policy.effective_config()["policy_name"] == control


def test_ler_guided_policy_reaches_exact_quota_through_build_skip_policy():
    tracker = FakeLagTracker()
    policy = call_build_skip_policy(
        LER_GUIDED_CONTROL,
        ler_tracker=tracker,
        ler_guided_controller_config=make_ler_guided_config(),
    )
    trainer = FakeTrainer(TOTAL_STEPS)
    commits = {
        di: (float((di % 7) + 1), di) for di in range(0, TOTAL_STEPS, 5)
    }
    for di in range(TOTAL_STEPS):
        if di in commits:
            tracker.commit(*commits[di])
        policy.should_skip(trainer, None, None)
        tracker.tick()
    diag = policy.get_diagnostics()
    assert diag["decisions_seen"] == TOTAL_STEPS
    assert diag["skip_decisions"] == diag["quota_size"] == int(RATE * TOTAL_STEPS)
    assert diag["quota_exact"] is True
    assert tracker.update_calls == 0
    assert tracker.note_calls == 0