"""Step 3B.3A: canonical LER-guided runner arm and identity helpers."""

import ast
from pathlib import Path

import math

import pytest

import lerna.trainers as trainers
from lerna.trainers.true_skip_trainer import (
    ONLINE_LER_MODE_OFF,
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
    PHASE1_3_MATRIX,
    POLICY_MIN_STEP,
    SKIPPING_CONTROLS,
    add_ler_guided_to_identity,
    build_ler_guided_controller_config,
    build_phase_strat_controller_config,
    build_skip_policy,
    resolve_online_ler_config,
)


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


# ---------------------------------------------------------------------------
# Phase 1.3 Matrix Contract tests
# ---------------------------------------------------------------------------


def test_phase1_3_matrix_has_exactly_six_arms():
    assert len(PHASE1_3_MATRIX) == 6


def test_phase1_3_matrix_contains_exactly_canonical_arms():
    assert set(PHASE1_3_MATRIX) == {
        "full_finetune",
        "exact_random",
        "fixed_phase_strat",
        "phase_strat_guarded",
        "ler_guided_stratified",
        "ler_guided_stratified_safe",
    }


def test_phase1_3_matrix_ordering_is_canonical():
    assert PHASE1_3_MATRIX == [
        "full_finetune",
        "exact_random",
        "fixed_phase_strat",
        "phase_strat_guarded",
        "ler_guided_stratified",
        "ler_guided_stratified_safe",
    ]


def test_fixed_phase_strat_is_explicit_ablation():
    assert "fixed_phase_strat" in ABLATIONS
    assert ABLATIONS["fixed_phase_strat"] == {"control": "fixed_phase_strat"}
    assert "alias_of" not in ABLATIONS["fixed_phase_strat"]


def test_phase_strat_guarded_is_explicit_ablation():
    assert "phase_strat_guarded" in ABLATIONS
    assert ABLATIONS["phase_strat_guarded"] == {
        "control": "phase_strat_guarded"
    }
    assert "alias_of" not in ABLATIONS["phase_strat_guarded"]


def test_build_skip_policy_routes_fixed_phase_strat():
    from lerna.trainers.policies import FixedPhaseStratifiedRandomPolicy

    policy = build_skip_policy(
        control="fixed_phase_strat",
        ler_tracker=None,
        target_skip_rate=0.20,
        total_steps=100,
        controller_cfg={"policy_seed": 42},
        rho_veto_threshold=-0.2,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )
    assert type(policy) is FixedPhaseStratifiedRandomPolicy
    assert policy.target_skip_rate == 0.20
    assert policy.min_step == POLICY_MIN_STEP


def test_build_skip_policy_routes_phase_strat_guarded():
    from lerna.trainers.policies import PhaseStratifiedGuardedRandomPolicy

    policy = build_skip_policy(
        control="phase_strat_guarded",
        ler_tracker=FakeLagTracker(),
        target_skip_rate=0.20,
        total_steps=100,
        controller_cfg={"policy_seed": 42, "max_consecutive_skips": 4},
        rho_veto_threshold=-0.2,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )
    assert type(policy) is PhaseStratifiedGuardedRandomPolicy
    assert policy.target_skip_rate == 0.20
    assert policy.min_step == POLICY_MIN_STEP


def test_resolve_online_ler_fixed_phase_strat_is_off():
    resolved = resolve_online_ler_config(
        "auto",
        effective_control="fixed_phase_strat",
        policy=None,
        parameter_sample_size=4096,
        update_interval=1,
    )
    assert resolved["mode"] == ONLINE_LER_MODE_OFF
    assert resolved["enabled"] is False
    assert "fixed_phase_strat" in resolved["reason"]


def test_resolve_online_ler_phase_strat_guarded_is_sampled_lagged():
    resolved = resolve_online_ler_config(
        "auto",
        effective_control="phase_strat_guarded",
        policy=None,
        parameter_sample_size=4096,
        update_interval=2,
    )
    assert resolved["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert resolved["enabled"] is True
    assert resolved["reason"] == "auto_signal_consuming_arm"


def test_fixed_phase_strat_in_skipping_controls():
    assert "fixed_phase_strat" in SKIPPING_CONTROLS


def test_phase_strat_guarded_in_skipping_controls():
    assert "phase_strat_guarded" in SKIPPING_CONTROLS


def test_phase_strat_controller_config_has_required_keys():
    config = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    assert config["control"] == "fixed_phase_strat"
    assert config["controller_class"] == "FixedPhaseStratifiedRandomPolicy"
    assert config["policy_seed"] == 42
    assert config["total_steps"] == 100
    assert config["requested_quota"] == 20
    assert config["n_phases"] == 4
    assert "phase_weights" in config
    assert "phase_bounds" in config
    assert "phase_quota" in config
    assert "phase_eligible" in config
    # fixed_phase_strat does not consume max_consecutive_skips or risk_gamma;
    # they must not appear in its provenance config.
    assert "max_consecutive_skips" not in config
    assert "risk_gamma" not in config


def test_phase_strat_controller_config_guarded_has_safety():
    config = build_phase_strat_controller_config(
        control="phase_strat_guarded",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
        guarded=True,
        rho_veto_threshold=-0.2,
        spike_factor=1.0,
        use_rho_vg=True,
        use_safety_horizon=True,
        risk_gamma=0.0,
    )
    assert config["control"] == "phase_strat_guarded"
    assert config["controller_class"] == "PhaseStratifiedGuardedRandomPolicy"
    assert "guarded_safety" in config
    assert config["guarded_safety"]["use_rho_vg"] is True
    assert config["guarded_safety"]["rho_veto_threshold"] == -0.2
    assert config["guarded_safety"]["use_safety_horizon"] is True
    assert config["guarded_safety"]["spike_factor"] == 1.0


def test_phase_strat_controller_config_not_guarded_has_no_safety():
    config = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    assert "guarded_safety" not in config


def test_phase_strat_controller_config_quota_matches_build_exact_random_skip_set():
    from lerna.trainers.policies import build_exact_random_skip_set

    config = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    _, expected_quota = build_exact_random_skip_set(
        total_steps=100,
        target_skip_rate=0.20,
        min_step=POLICY_MIN_STEP,
        seed=42,
    )
    assert config["requested_quota"] == expected_quota


def test_legacy_arms_excluded_from_phase1_3_matrix():
    for legacy in ["full_lerna", "no_rho_vg", "no_ler", "no_safety",
                   "no_hysteresis", "no_momentum", "rvd", "grad_norm",
                   "random_skip", "phase_strat"]:
        assert legacy not in PHASE1_3_MATRIX


def test_random_skip_is_alias_not_in_matrix():
    assert "random_skip" in ABLATIONS
    assert ABLATIONS["random_skip"].get("alias_of") == "exact_random"
    assert "random_skip" not in PHASE1_3_MATRIX


def test_phase_strat_ambiguous_legacy_not_in_matrix():
    assert "phase_strat" not in PHASE1_3_MATRIX


def test_exact_random_still_in_matrix():
    assert "exact_random" in PHASE1_3_MATRIX


def test_full_finetune_still_in_matrix():
    assert "full_finetune" in PHASE1_3_MATRIX


def test_phase_strat_controller_config_is_defensive_copy():
    config1 = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    config2 = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    assert config1 == config2
    assert config1 is not config2
    assert config1["phase_weights"] is not config2["phase_weights"]
    config1["phase_weights"][0] = 99.0
    assert config2["phase_weights"] == [0.22, 0.24, 0.26, 0.28]


def test_phase_strat_controller_config_fingerprint_consistency():
    config = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    id1 = {"task": "mrpc", "training_seed": 42, "control": "fixed_phase_strat"}
    id1["phase_strat_controller"] = config
    id2 = dict(id1)
    id2["phase_strat_controller"] = dict(config)
    id2["phase_strat_controller"]["phase_weights"] = list(
        config["phase_weights"]
    )
    fp1 = build_scientific_fingerprint(id1)
    fp2 = build_scientific_fingerprint(id2)
    assert fp1 == fp2


def test_phase_strat_controller_config_fingerprint_differs_by_seed():
    config_a = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    config_b = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=43,
        max_consecutive_skips=4,
    )
    assert config_a != config_b


def test_phase_strat_controller_config_fingerprint_differs_by_control():
    config_fixed = build_phase_strat_controller_config(
        control="fixed_phase_strat",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
    )
    config_guarded = build_phase_strat_controller_config(
        control="phase_strat_guarded",
        target_skip_rate=0.20,
        total_steps=100,
        policy_seed=42,
        max_consecutive_skips=4,
        guarded=True,
        rho_veto_threshold=-0.2,
        spike_factor=1.0,
        use_rho_vg=True,
        use_safety_horizon=True,
        risk_gamma=0.0,
    )
    assert config_fixed != config_guarded


def test_legacy_policy_fixed_phase_strat_still_works():
    from lerna.trainers.policies import FixedPhaseStratifiedRandomPolicy

    policy = FixedPhaseStratifiedRandomPolicy(
        target_skip_rate=0.20,
        total_steps=100,
        min_step=POLICY_MIN_STEP,
        seed=42,
    )
    assert policy.name == "fixed_phase_strat"
    assert policy.target_skip_rate == 0.20


def test_legacy_policy_phase_strat_guarded_still_works():
    from lerna.trainers.policies import PhaseStratifiedGuardedRandomPolicy

    policy = PhaseStratifiedGuardedRandomPolicy(
        ler_tracker=FakeLagTracker(),
        target_skip_rate=0.20,
        total_steps=100,
        min_step=POLICY_MIN_STEP,
        seed=42,
        max_consecutive_skips=4,
        rho_veto_threshold=-0.2,
        use_rho_vg=True,
        use_safety_horizon=True,
        risk_gamma=0.0,
    )
    assert policy.name == "phase_strat_guarded"
    assert policy.target_skip_rate == 0.20


def test_exact_random_still_routes_through_build_skip_policy():
    from lerna.trainers.policies import RandomSkipPolicy

    policy = build_skip_policy(
        control="exact_random",
        ler_tracker=None,
        target_skip_rate=0.20,
        total_steps=100,
        controller_cfg={"policy_seed": 42},
        rho_veto_threshold=-0.2,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )
    assert type(policy) is RandomSkipPolicy
    assert policy.seed == 42


def test_quota_hybrid_runner_passes_recalibrate_every():
    """The runner quota_hybrid branch must pass recalibrate_every=200."""
    runner_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "run_ablation_study.py"
    )
    source = runner_path.read_text(encoding="utf-8")
    tree = ast.parse(source)

    found = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not isinstance(test, ast.Compare):
            continue
        if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
            continue
        if len(test.comparators) != 1:
            continue
        left = test.left
        right = test.comparators[0]
        is_quota_hybrid = (
            (isinstance(left, ast.Name) and left.id == "policy"
             and isinstance(right, ast.Constant) and right.value == "quota_hybrid")
            or (isinstance(left, ast.Constant) and left.value == "quota_hybrid"
                and isinstance(right, ast.Name) and right.id == "policy")
        )
        if not is_quota_hybrid:
            continue
        for stmt in node.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if not (isinstance(target, ast.Name) and target.id == "skip_policy"):
                    continue
                call = stmt.value
                if not isinstance(call, ast.Call):
                    continue
                if not (isinstance(call.func, ast.Name) and call.func.id == "LERNAQuotaHybridPolicy"):
                    continue
                for keyword in call.keywords:
                    if keyword.arg == "recalibrate_every":
                        if isinstance(keyword.value, ast.Constant) and keyword.value.value == 200:
                            found = True
                            break
            if found:
                break
        if found:
            break

    assert found, "quota_hybrid branch must pass recalibrate_every=200 to LERNAQuotaHybridPolicy"