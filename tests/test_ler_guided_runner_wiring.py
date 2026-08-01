"""Step 3B.3A: canonical LER-guided runner arm and identity helpers."""

import math

import pytest

import lerna.trainers as trainers
from lerna.trainers.true_skip_trainer import (
    ONLINE_LER_MODE_SAMPLED_LAGGED,
    ONLINE_LER_TIMING_POST_DECISION,
)
from lerna.utils.run_provenance import build_scientific_fingerprint
from scripts.run_ablation_study import (
    ABLATIONS,
    DEFAULT_ABLATIONS,
    LER_GUIDED_CONTROL,
    LER_GUIDED_CONTROLS,
    LER_GUIDED_SAFE_CONTROL,
    POLICY_MIN_STEP,
    SKIPPING_CONTROLS,
    add_ler_guided_to_identity,
    build_ler_guided_controller_config,
    resolve_online_ler_config,
)


COMMON_KEYS = {
    "control",
    "policy_class",
    "policy_name",
    "target_skip_rate",
    "total_steps",
    "min_step",
    "policy_seed",
    "n_phases",
    "phase_weights",
    "max_consecutive_skips",
    "probe_interval",
    "min_ler_observations",
    "ler_guidance_strength",
    "required_tracker_mode",
    "required_tracker_timing",
    "safety_enabled",
}
SAFETY_KEYS = {
    "use_rho_vg_safety",
    "rho_veto_threshold",
    "use_loss_spike_safety",
    "loss_spike_factor",
    "loss_spike_window",
}


def build_config(control=LER_GUIDED_CONTROL, **overrides):
    kwargs = {
        "control": control,
        "target_skip_rate": 0.20,
        "total_steps": 100,
        "policy_seed": 42,
        "max_consecutive_skips": 4,
        "probe_interval": 8,
        "rho_veto_threshold": -0.2,
    }
    kwargs.update(overrides)
    return build_ler_guided_controller_config(**kwargs)


def fingerprint(control=LER_GUIDED_CONTROL, **overrides):
    config = build_config(control=control, **overrides)
    identity = {
        "task": "mrpc",
        "training_seed": 42,
        "control": control,
    }
    return build_scientific_fingerprint(
        add_ler_guided_to_identity(identity, config)
    )


def test_both_policy_classes_are_exported():
    assert trainers.LERGuidedStratifiedPolicy.__name__ == (
        "LERGuidedStratifiedPolicy"
    )
    assert trainers.LERGuidedStratifiedSafetyPolicy.__name__ == (
        "LERGuidedStratifiedSafetyPolicy"
    )
    assert "LERGuidedStratifiedPolicy" in trainers.__all__
    assert "LERGuidedStratifiedSafetyPolicy" in trainers.__all__


def test_truthful_explicit_arms_are_default_skipping_controls():
    assert LER_GUIDED_CONTROLS == {
        "ler_guided_stratified",
        "ler_guided_stratified_safe",
    }
    for control in LER_GUIDED_CONTROLS:
        assert ABLATIONS[control] == {"control": control}
        assert control in SKIPPING_CONTROLS
        assert control in DEFAULT_ABLATIONS
        assert "alias_of" not in ABLATIONS[control]


@pytest.mark.parametrize("control", sorted(LER_GUIDED_CONTROLS))
def test_auto_online_ler_is_sampled_and_post_backward(control):
    resolved = resolve_online_ler_config(
        "auto",
        effective_control=control,
        policy=None,
        parameter_sample_size=4096,
        update_interval=2,
    )
    assert resolved["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert resolved["timing"] == ONLINE_LER_TIMING_POST_DECISION
    assert resolved["enabled"] is True
    assert resolved["reason"] == "auto_signal_consuming_arm"


def test_no_safety_config_has_exact_canonical_schema():
    config = build_config()
    assert set(config) == COMMON_KEYS
    assert config == {
        "control": LER_GUIDED_CONTROL,
        "policy_class": "LERGuidedStratifiedPolicy",
        "policy_name": "ler_guided_stratified",
        "target_skip_rate": 0.20,
        "total_steps": 100,
        "min_step": POLICY_MIN_STEP,
        "policy_seed": 42,
        "n_phases": 4,
        "phase_weights": [0.22, 0.24, 0.26, 0.28],
        "max_consecutive_skips": 4,
        "probe_interval": 8,
        "min_ler_observations": 3,
        "ler_guidance_strength": 1.0,
        "required_tracker_mode": "sampled_lagged",
        "required_tracker_timing": "post_decision_after_backward",
        "safety_enabled": False,
    }
    assert not SAFETY_KEYS.intersection(config)


def test_safety_config_has_exact_canonical_schema():
    config = build_config(
        control=LER_GUIDED_SAFE_CONTROL,
        rho_veto_threshold=-0.35,
    )
    assert set(config) == COMMON_KEYS | SAFETY_KEYS
    assert config["policy_class"] == "LERGuidedStratifiedSafetyPolicy"
    assert config["policy_name"] == LER_GUIDED_SAFE_CONTROL
    assert config["safety_enabled"] is True
    assert config["use_rho_vg_safety"] is True
    assert config["rho_veto_threshold"] == -0.35
    assert config["use_loss_spike_safety"] is True
    assert config["loss_spike_factor"] == 1.0
    assert config["loss_spike_window"] == 5


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"control": "rvd"}, "control"),
        ({"target_skip_rate": -0.01}, "target_skip_rate"),
        ({"target_skip_rate": 1.01}, "target_skip_rate"),
        ({"target_skip_rate": math.nan}, "target_skip_rate"),
        ({"target_skip_rate": math.inf}, "target_skip_rate"),
        ({"total_steps": 0}, "total_steps"),
        ({"total_steps": POLICY_MIN_STEP}, "total_steps"),
        ({"total_steps": 100.5}, "total_steps"),
        ({"max_consecutive_skips": 0}, "max_consecutive_skips"),
        ({"probe_interval": 0}, "probe_interval"),
        ({"rho_veto_threshold": math.nan}, "rho_veto_threshold"),
        ({"rho_veto_threshold": math.inf}, "rho_veto_threshold"),
    ],
)
def test_invalid_canonical_inputs_fail(overrides, message):
    with pytest.raises((TypeError, ValueError), match=message):
        build_config(**overrides)


def test_infeasible_post_warmup_quota_is_rejected_without_clipping():
    with pytest.raises(ValueError, match="Infeasible skip quota"):
        build_config(total_steps=100, target_skip_rate=0.80)


def test_config_results_and_phase_weights_are_independent():
    first = build_config()
    second = build_config()
    assert first == second
    assert first is not second
    assert first["phase_weights"] is not second["phase_weights"]
    first["phase_weights"][0] = 99.0
    assert second["phase_weights"] == [0.22, 0.24, 0.26, 0.28]


def test_identity_helper_defensively_copies_both_inputs():
    identity = {"task": "mrpc", "training_seed": 42}
    config = build_config()
    identity_before = dict(identity)
    config_before = {**config, "phase_weights": list(config["phase_weights"])}

    extended = add_ler_guided_to_identity(identity, config)

    assert identity == identity_before
    assert config == config_before
    assert extended is not identity
    assert extended["ler_guided_controller"] == config
    assert extended["ler_guided_controller"] is not config
    assert extended["ler_guided_controller"]["phase_weights"] is not (
        config["phase_weights"]
    )
    config["phase_weights"][0] = 99.0
    assert extended["ler_guided_controller"]["phase_weights"] == [
        0.22,
        0.24,
        0.26,
        0.28,
    ]


def test_identical_copied_inputs_have_identical_fingerprints():
    assert fingerprint() == fingerprint()


def test_safe_and_no_safety_fingerprints_differ():
    assert fingerprint(LER_GUIDED_CONTROL) != fingerprint(
        LER_GUIDED_SAFE_CONTROL
    )


@pytest.mark.parametrize("control", sorted(LER_GUIDED_CONTROLS))
def test_probe_interval_changes_active_fingerprint(control):
    assert fingerprint(control, probe_interval=4) != fingerprint(
        control,
        probe_interval=8,
    )


def test_active_rho_threshold_changes_only_safety_fingerprint():
    assert fingerprint(
        LER_GUIDED_SAFE_CONTROL,
        rho_veto_threshold=-0.2,
    ) != fingerprint(
        LER_GUIDED_SAFE_CONTROL,
        rho_veto_threshold=-0.4,
    )
    assert fingerprint(
        LER_GUIDED_CONTROL,
        rho_veto_threshold=-0.2,
    ) == fingerprint(
        LER_GUIDED_CONTROL,
        rho_veto_threshold=-0.4,
    )