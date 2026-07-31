"""Step 4B.2a: pure runner resolver for online LER diagnostics configuration."""

import pytest

from lerna.trainers.true_skip_trainer import (
    ONLINE_LER_MODE_OFF,
    ONLINE_LER_MODE_LEGACY_DENSE,
    ONLINE_LER_MODE_SAMPLED_LAGGED,
    ONLINE_LER_TIMING_NONE,
    ONLINE_LER_TIMING_PRE_DECISION,
    ONLINE_LER_TIMING_POST_DECISION,
)
from lerna.utils.lagged_ler import SampledLaggedLERTracker
from lerna.utils.metrics import LERTracker
from scripts.run_ablation_study import (
    build_arg_parser,
    build_online_ler_tracker,
    resolve_online_ler_config,
)


CANONICAL_KEYS = {
    "requested_mode",
    "mode",
    "enabled",
    "timing",
    "parameter_sample_size",
    "update_interval",
    "reason",
}


def _resolve(requested_mode, **overrides):
    kwargs = {
        "effective_control": None,
        "policy": "hybrid",
        "parameter_sample_size": 4096,
        "update_interval": 10,
    }
    kwargs.update(overrides)
    return resolve_online_ler_config(requested_mode, **kwargs)


def test_auto_full_finetune_resolves_off():
    cfg = _resolve("auto", effective_control="full_finetune", policy=None)
    assert cfg["mode"] == ONLINE_LER_MODE_OFF
    assert cfg["enabled"] is False
    assert cfg["timing"] == ONLINE_LER_TIMING_NONE
    assert cfg["parameter_sample_size"] == 0
    assert cfg["update_interval"] == 0
    assert cfg["reason"] == "auto_signal_free_control:full_finetune"
    assert set(cfg) == CANONICAL_KEYS


def test_auto_exact_random_resolves_off():
    cfg = _resolve("auto", effective_control="exact_random", policy=None)
    assert cfg["mode"] == ONLINE_LER_MODE_OFF
    assert cfg["enabled"] is False
    assert cfg["timing"] == ONLINE_LER_TIMING_NONE
    assert cfg["reason"] == "auto_signal_free_control:exact_random"


def test_auto_fixed_phase_strat_resolves_off():
    cfg = _resolve("auto", effective_control=None, policy="fixed_phase_strat")
    assert cfg["mode"] == ONLINE_LER_MODE_OFF
    assert cfg["enabled"] is False
    assert cfg["timing"] == ONLINE_LER_TIMING_NONE
    assert cfg["reason"] == "auto_signal_free_policy:fixed_phase_strat"


def test_auto_signal_consuming_policy_resolves_sampled_lagged():
    cfg = _resolve("auto", effective_control=None, policy="hybrid")
    assert cfg["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert cfg["enabled"] is True
    assert cfg["timing"] == ONLINE_LER_TIMING_POST_DECISION
    assert cfg["parameter_sample_size"] == 4096
    assert cfg["update_interval"] == 10
    assert cfg["reason"] == "auto_signal_consuming_arm"


def test_auto_signal_consuming_control_resolves_sampled_lagged():
    cfg = _resolve("auto", effective_control="rvd", policy=None)
    assert cfg["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert cfg["reason"] == "auto_signal_consuming_arm"


def test_explicit_off_preserved_for_signal_consuming_policy():
    cfg = _resolve("off", effective_control=None, policy="hybrid")
    assert cfg["requested_mode"] == "off"
    assert cfg["mode"] == ONLINE_LER_MODE_OFF
    assert cfg["enabled"] is False
    assert cfg["timing"] == ONLINE_LER_TIMING_NONE
    assert cfg["parameter_sample_size"] == 0
    assert cfg["update_interval"] == 0
    assert cfg["reason"] == "explicit:off"


def test_explicit_legacy_dense_preserved_for_signal_free_control():
    cfg = _resolve(
        "legacy_dense",
        effective_control="full_finetune",
        policy=None,
        update_interval=3,
    )
    assert cfg["mode"] == ONLINE_LER_MODE_LEGACY_DENSE
    assert cfg["enabled"] is True
    assert cfg["timing"] == ONLINE_LER_TIMING_PRE_DECISION
    assert cfg["parameter_sample_size"] == 0
    assert cfg["update_interval"] == 3
    assert cfg["reason"] == "explicit:legacy_dense"


def test_explicit_sampled_lagged_preserved_for_signal_free_control():
    cfg = _resolve(
        "sampled_lagged",
        effective_control="exact_random",
        policy=None,
        parameter_sample_size=128,
        update_interval=5,
    )
    assert cfg["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert cfg["enabled"] is True
    assert cfg["timing"] == ONLINE_LER_TIMING_POST_DECISION
    assert cfg["parameter_sample_size"] == 128
    assert cfg["update_interval"] == 5
    assert cfg["reason"] == "explicit:sampled_lagged"


def test_canonical_keys_present_for_all_modes():
    for mode in ("auto", "off", "legacy_dense", "sampled_lagged"):
        cfg = _resolve(mode)
        assert set(cfg) == CANONICAL_KEYS
        assert cfg["requested_mode"] == mode
        assert isinstance(cfg["enabled"], bool)
        assert isinstance(cfg["parameter_sample_size"], int)
        assert isinstance(cfg["update_interval"], int)


def test_invalid_requested_mode_raises():
    with pytest.raises(ValueError) as excinfo:
        _resolve("dense")
    message = str(excinfo.value)
    assert "requested_mode" in message or "online_ler_mode" in message


@pytest.mark.parametrize("mode", ["legacy_dense", "sampled_lagged"])
def test_enabled_interval_below_one_raises(mode):
    with pytest.raises(ValueError, match="update_interval"):
        _resolve(mode, update_interval=0)


def test_sampled_lagged_sample_size_below_one_raises():
    with pytest.raises(ValueError, match="parameter_sample_size"):
        _resolve("sampled_lagged", parameter_sample_size=0)


def test_off_mode_ignores_invalid_budgets():
    cfg = _resolve("off", parameter_sample_size=0, update_interval=0)
    assert cfg["parameter_sample_size"] == 0
    assert cfg["update_interval"] == 0


def test_legacy_dense_canonicalizes_sample_size_to_zero():
    cfg = _resolve("legacy_dense", parameter_sample_size=999)
    assert cfg["parameter_sample_size"] == 0


def test_helper_does_not_reuse_mutated_results():
    kwargs = {
        "effective_control": "exact_random",
        "policy": "hybrid",
        "parameter_sample_size": 256,
        "update_interval": 7,
    }
    cfg = resolve_online_ler_config("sampled_lagged", **kwargs)
    assert kwargs == {
        "effective_control": "exact_random",
        "policy": "hybrid",
        "parameter_sample_size": 256,
        "update_interval": 7,
    }
    cfg["mode"] = "mutated"
    cfg2 = resolve_online_ler_config("sampled_lagged", **kwargs)
    assert cfg2["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED


def test_parser_online_ler_defaults():
    args = build_arg_parser().parse_args([])
    assert args.online_ler_mode == "auto"
    assert args.online_ler_sample_size == 4096
    assert args.online_ler_update_interval == 1


def test_parser_online_ler_explicit_values():
    args = build_arg_parser().parse_args(
        [
            "--online-ler-mode",
            "sampled_lagged",
            "--online-ler-sample-size",
            "128",
            "--online-ler-update-interval",
            "5",
        ]
    )
    assert args.online_ler_mode == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert args.online_ler_sample_size == 128
    assert args.online_ler_update_interval == 5


def test_factory_off_mode_returns_none():
    cfg = _resolve("off")
    tracker = build_online_ler_tracker(
        cfg,
        task_name="sst2",
        use_hysteresis=True,
        sample_seed=42,
    )
    assert tracker is None


def test_factory_legacy_dense_returns_ler_tracker():
    cfg = _resolve("legacy_dense")
    tracker = build_online_ler_tracker(
        cfg,
        task_name="sst2",
        use_hysteresis=False,
        sample_seed=42,
    )
    assert type(tracker) is LERTracker


def test_factory_sampled_lagged_returns_sampled_tracker():
    cfg = _resolve("sampled_lagged", parameter_sample_size=128)
    tracker = build_online_ler_tracker(
        cfg,
        task_name="qnli",
        use_hysteresis=True,
        sample_seed=1234,
    )
    assert type(tracker) is SampledLaggedLERTracker
    assert tracker.task == "qnli"
    assert tracker.window_size == 5
    assert tracker.parameter_sample_size == 128
    assert tracker.sample_seed == 1234


def test_factory_rejects_unknown_concrete_mode():
    cfg = _resolve("legacy_dense")
    cfg["mode"] = "dense"
    with pytest.raises(ValueError, match="online LER mode"):
        build_online_ler_tracker(
            cfg,
            task_name="sst2",
            use_hysteresis=True,
            sample_seed=42,
        )
