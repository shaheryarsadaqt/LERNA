"""Piece 4: deterministic RVD runner configuration and quota tests."""

import json

import pytest

from lerna.trainers import (
    AlwaysFalsePolicy,
    LERNARandomVetoDeferralPolicy,
    RandomSkipPolicy,
)
from lerna.trainers.policies import build_exact_random_skip_set
from scripts.run_ablation_study import (
    ABLATIONS,
    DEFAULT_ABLATIONS,
    assert_fixed_budget,
    build_arg_parser,
    build_rvd_controller_config,
    build_skip_policy,
)


class FakeTracker:
    def __init__(self):
        self.loss_history = []

    def get_diagnostics(self):
        return {"rho_vg_raw": 0.0}


class StubState:
    def __init__(self, max_steps):
        self.global_step = 0
        self.max_steps = max_steps


class StubTrainer:
    def __init__(self, max_steps):
        self.state = StubState(max_steps)


BASE_CONFIG = {
    "veto_mode": "none",
    "margin_rank_floor": 0.20,
    "spike_factor": 1.0,
    "spike_ema_window": 20,
    "repay_mode": "asap",
    "repay_protect_dangerous": True,
    "policy_seed": None,
    "training_seed": 42,
    "max_consecutive_skips": 4,
}


def make_controller_config(**overrides):
    config = dict(BASE_CONFIG)
    config.update(overrides)
    return build_rvd_controller_config(**config)


def build_control(control, controller_cfg=None):
    return build_skip_policy(
        control=control,
        ler_tracker=FakeTracker(),
        target_skip_rate=0.20,
        total_steps=400,
        controller_cfg=controller_cfg or make_controller_config(),
        rho_veto_threshold=-0.05,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )


@pytest.mark.parametrize(
    ("mode", "margin", "loss_spike"),
    [
        ("none", False, False),
        ("margin", True, False),
        ("loss_spike", False, True),
    ],
)
def test_veto_mode_mapping_exactly_one(mode, margin, loss_spike):
    config = make_controller_config(veto_mode=mode)
    assert config["use_margin_veto"] is margin
    assert config["use_loss_spike_veto"] is loss_spike
    assert config["use_rho_vg_veto"] is False
    assert config["use_grad_norm_veto"] is False
    assert config["use_novelty_veto"] is False
    assert config["use_phase_protection"] is False
    assert sum(
        bool(config[key])
        for key in (
            "use_margin_veto",
            "use_loss_spike_veto",
            "use_rho_vg_veto",
            "use_grad_norm_veto",
            "use_novelty_veto",
            "use_phase_protection",
        )
    ) == (0 if mode == "none" else 1)


def test_policy_seed_defaults_to_training_seed():
    defaulted = make_controller_config(training_seed=1234)
    assert defaulted["policy_seed"] == 1234
    assert defaulted["policy_seed_defaulted_to_training_seed"] is True

    explicit = make_controller_config(training_seed=1234, policy_seed=99)
    assert explicit["policy_seed"] == 99
    assert explicit["policy_seed_defaulted_to_training_seed"] is False


def test_parser_flags_normalization():
    args = build_arg_parser().parse_args(
        [
            "--rvd-veto-mode", "loss_spike",
            "--rvd-margin-rank-floor", "0.3",
            "--rvd-spike-factor", "1.5",
            "--rvd-spike-ema-window", "12",
            "--rvd-repay-mode", "spread",
            "--no-rvd-repay-protect-dangerous",
            "--rvd-policy-seed", "77",
            "--allow-early-stopping-with-skipping",
        ]
    )
    assert args.rvd_veto_mode == "loss_spike"
    assert args.rvd_margin_rank_floor == 0.3
    assert args.rvd_spike_factor == 1.5
    assert args.rvd_spike_ema_window == 12
    assert args.rvd_repay_mode == "spread"
    assert args.rvd_repay_protect_dangerous is False
    assert args.rvd_policy_seed == 77
    assert args.allow_early_stopping_with_skipping is True


def test_explicit_arms_and_legacy_alias_construct_expected_classes():
    assert isinstance(build_control("full_finetune"), AlwaysFalsePolicy)
    exact = build_control("exact_random")
    alias = build_control("random_skip")
    rvd = build_control("rvd")
    assert isinstance(exact, RandomSkipPolicy)
    assert isinstance(alias, RandomSkipPolicy)
    assert isinstance(rvd, LERNARandomVetoDeferralPolicy)
    assert exact.seed == alias.seed == 42
    assert ABLATIONS["random_skip"]["alias_of"] == "exact_random"
    assert "random_skip" not in DEFAULT_ABLATIONS


def test_impossible_quota_raises_without_clipping():
    with pytest.raises(ValueError, match="only 50 steps are eligible"):
        build_exact_random_skip_set(
            total_steps=100,
            target_skip_rate=0.9,
            min_step=50,
            seed=42,
        )

    skip_set, quota = build_exact_random_skip_set(
        total_steps=101,
        target_skip_rate=0.2,
        min_step=50,
        seed=42,
    )
    assert quota == round(101 * 0.2)
    assert len(skip_set) == quota

    full_set, full_quota = build_exact_random_skip_set(
        total_steps=10,
        target_skip_rate=1.0,
        min_step=0,
        seed=42,
    )
    assert full_quota == 10
    assert full_set == set(range(10))


def test_early_stopping_guard_and_matched_budget_provenance():
    with pytest.raises(RuntimeError, match="Fixed-budget violation"):
        assert_fixed_budget(
            ablation_name="rvd",
            control="rvd",
            no_early_stopping=False,
            allow_early_stopping_with_skipping=False,
        )

    exploratory = assert_fixed_budget(
        ablation_name="rvd",
        control="rvd",
        no_early_stopping=False,
        allow_early_stopping_with_skipping=True,
    )
    assert exploratory["matched_budget"] is False

    fixed = assert_fixed_budget(
        ablation_name="rvd",
        control="rvd",
        no_early_stopping=True,
        allow_early_stopping_with_skipping=False,
    )
    assert fixed["matched_budget"] is True

    full_early_stop = assert_fixed_budget(
        ablation_name="full_finetune",
        control="full_finetune",
        no_early_stopping=False,
        allow_early_stopping_with_skipping=False,
    )
    assert full_early_stop["is_skipping_arm"] is False
    assert full_early_stop["matched_budget"] is False


def test_controller_config_is_json_serializable_and_names_inactive_fields():
    policy = build_control(
        "rvd",
        make_controller_config(veto_mode="margin", policy_seed=7),
    )
    encoded = json.dumps(policy.effective_config(), sort_keys=True)
    assert encoded
    config = policy.effective_config()
    assert config["veto_mode"] == "margin"
    inactive = config["inactive_compat_fields"]
    for field in (
        "probe_interval",
        "risk_gamma",
        "target_veto_rate",
        "fallback_threshold",
        "calibration_steps",
        "recalibrate_every",
        "use_ler",
        "use_rho_vg",
        "use_grad_norm",
    ):
        assert field in inactive


def test_exact_random_and_rvd_none_have_identical_decisions():
    total_steps = 400
    exact = build_control("exact_random")
    rvd = build_control("rvd", make_controller_config(veto_mode="none"))
    trainer = StubTrainer(total_steps)

    exact_decisions = []
    rvd_decisions = []
    for _ in range(total_steps):
        exact_decisions.append(exact.should_skip(trainer, None, None))
        rvd_decisions.append(rvd.should_skip(trainer, None, None))

    assert exact_decisions == rvd_decisions
    assert exact._skip_set == rvd._skip_set
    assert len(exact._skip_set) == round(total_steps * 0.20)
    assert rvd.get_diagnostics()["all_vetoes_disabled"] is True
