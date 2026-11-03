"""Step 4B.2b2b: runtime diagnostics metadata and artifact truthfulness."""

import copy
import json
import os
import tempfile

from lerna.trainers.true_skip_trainer import (
    ONLINE_LER_MODE_OFF,
    ONLINE_LER_MODE_LEGACY_DENSE,
    ONLINE_LER_MODE_SAMPLED_LAGGED,
    ONLINE_LER_TIMING_NONE,
    ONLINE_LER_TIMING_PRE_DECISION,
    ONLINE_LER_TIMING_POST_DECISION,
)
from scripts.run_ablation_study import (
    AblationDiagnosticsCallback,
    build_online_ler_artifact_contract,
    build_online_ler_runtime_metadata,
)


def _sampled_config(**overrides):
    config = {
        "requested_mode": "auto",
        "mode": ONLINE_LER_MODE_SAMPLED_LAGGED,
        "enabled": True,
        "timing": ONLINE_LER_TIMING_POST_DECISION,
        "parameter_sample_size": 4096,
        "update_interval": 2,
        "reason": "auto_signal_consuming_arm",
        "sample_seed": 1042,
    }
    config.update(overrides)
    return config


def _legacy_config(**overrides):
    config = {
        "requested_mode": ONLINE_LER_MODE_LEGACY_DENSE,
        "mode": ONLINE_LER_MODE_LEGACY_DENSE,
        "enabled": True,
        "timing": ONLINE_LER_TIMING_PRE_DECISION,
        "parameter_sample_size": 0,
        "update_interval": 1,
        "reason": f"explicit:{ONLINE_LER_MODE_LEGACY_DENSE}",
        "sample_seed": None,
    }
    config.update(overrides)
    return config


def _off_config(**overrides):
    config = {
        "requested_mode": ONLINE_LER_MODE_OFF,
        "mode": ONLINE_LER_MODE_OFF,
        "enabled": False,
        "timing": ONLINE_LER_TIMING_NONE,
        "parameter_sample_size": 0,
        "update_interval": 0,
        "reason": f"explicit:{ONLINE_LER_MODE_OFF}",
        "sample_seed": None,
    }
    config.update(overrides)
    return config


class _FakeTracker:
    def __init__(self, diagnostics):
        self.ler_history = [0.1, 0.2]
        self.rho_vg_history = [0.5]
        self.velocity_history = [1.0, 2.0]
        self.diagnostics = diagnostics

    def get_diagnostics(self):
        return self.diagnostics


class _FakeTrainer:
    def __init__(self, instrumentation):
        self._instrumentation = instrumentation

    def get_instrumentation(self):
        return dict(self._instrumentation)


def _build_callback(tracker, holder, output_dir, online_diagnostics):
    return AblationDiagnosticsCallback(
        ler_trk=tracker,
        model_ref=None,
        trainer_ref_holder=holder,
        greater_is_better=False,
        use_rho_vg=True,
        use_ler=True,
        use_hysteresis=True,
        use_safety_horizon=True,
        skip_update_mode="freeze",
        skip_update_mode_legacy_compat_used=False,
        ablation_name="full_lerna",
        ablation_overrides={},
        output_dir=output_dir,
        use_wandb=False,
        task_cfg=None,
        eval_ds=None,
        tokenizer=None,
        online_diagnostics=online_diagnostics,
    )


def test_sampled_runtime_metadata_records_config_and_runtime():
    config = _sampled_config()
    instrumentation = {
        "batches_seen": 12,
        "online_ler_update_attempts": 6,
        "online_ler_update_successes": 5,
        "online_ler_last_update_decision": 10,
    }
    tracker = {
        "parameter_sample_size_realized": 2048,
        "observation_age_decisions": 1,
        "n_updates": 5,
        "n_decisions": 12,
    }

    meta = build_online_ler_runtime_metadata(
        config,
        instrumentation,
        tracker_diagnostics=tracker,
    )

    assert meta["requested_mode"] == "auto"
    assert meta["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert meta["enabled"] is True
    assert meta["timing"] == ONLINE_LER_TIMING_POST_DECISION
    assert meta["parameter_sample_size"] == 4096
    assert meta["parameter_sample_size_realized"] == 2048
    assert meta["update_interval"] == 2
    assert meta["reason"] == "auto_signal_consuming_arm"
    assert meta["sample_seed"] == 1042
    assert meta["update_attempts"] == 6
    assert meta["update_successes"] == 5
    assert meta["last_update_decision"] == 10
    assert meta["observation_age_decisions"] == 1
    assert meta["n_updates"] == 5
    assert meta["n_decisions"] == 12


def test_legacy_runtime_metadata_derives_observation_age():
    config = _legacy_config()
    instrumentation = {
        "batches_seen": 10,
        "online_ler_update_attempts": 10,
        "online_ler_update_successes": 8,
        "online_ler_last_update_decision": 6,
    }
    tracker = {"n_updates": 8}

    meta = build_online_ler_runtime_metadata(
        config,
        instrumentation,
        tracker_diagnostics=tracker,
    )

    assert meta["mode"] == ONLINE_LER_MODE_LEGACY_DENSE
    assert meta["timing"] == ONLINE_LER_TIMING_PRE_DECISION
    assert meta["parameter_sample_size_realized"] == 0
    assert meta["last_update_decision"] == 6
    assert meta["observation_age_decisions"] == 3
    assert meta["n_updates"] == 8
    assert meta["n_decisions"] == 10


def test_off_runtime_metadata_truthful_zeros():
    config = _off_config()
    instrumentation = {
        "batches_seen": 25,
        "online_ler_update_attempts": 3,
        "online_ler_update_successes": 2,
        "online_ler_last_update_decision": 7,
    }

    meta = build_online_ler_runtime_metadata(config, instrumentation)

    assert meta["mode"] == ONLINE_LER_MODE_OFF
    assert meta["enabled"] is False
    assert meta["timing"] == ONLINE_LER_TIMING_NONE
    assert meta["parameter_sample_size"] == 0
    assert meta["parameter_sample_size_realized"] == 0
    assert meta["update_interval"] == 0
    assert meta["sample_seed"] is None
    assert meta["update_attempts"] == 0
    assert meta["update_successes"] == 0
    assert meta["last_update_decision"] is None
    assert meta["observation_age_decisions"] is None
    assert meta["n_updates"] == 0
    assert meta["n_decisions"] == 0


def test_runtime_helper_does_not_mutate_inputs():
    config = _sampled_config()
    instrumentation = {
        "batches_seen": 12,
        "online_ler_update_attempts": 6,
        "online_ler_update_successes": 5,
        "online_ler_last_update_decision": 10,
    }
    tracker = {
        "parameter_sample_size_realized": 2048,
        "observation_age_decisions": 1,
        "n_updates": 5,
        "n_decisions": 12,
    }
    config_before = copy.deepcopy(config)
    instrumentation_before = copy.deepcopy(instrumentation)
    tracker_before = copy.deepcopy(tracker)

    meta = build_online_ler_runtime_metadata(
        config,
        instrumentation,
        tracker_diagnostics=tracker,
    )

    assert config == config_before
    assert instrumentation == instrumentation_before
    assert tracker == tracker_before
    assert meta is not config
    assert meta is not tracker


def test_artifact_contract_includes_ler_diagnostics_for_enabled_modes():
    for config in (_legacy_config(), _sampled_config()):
        contract = build_online_ler_artifact_contract(config)
        assert contract["output_paths"] == {
            "results": "results.json",
            "instrumentation": "instrumentation.json",
            "manifest": "run_manifest.json",
            "ler_diagnostics": "ler_diagnostics.json",
        }
        assert contract["required_artifacts"] == [
            "instrumentation.json",
            "ler_diagnostics.json",
        ]


def test_artifact_contract_excludes_ler_diagnostics_for_off_mode():
    contract = build_online_ler_artifact_contract(_off_config())
    assert contract["output_paths"] == {
        "results": "results.json",
        "instrumentation": "instrumentation.json",
        "manifest": "run_manifest.json",
    }
    assert "ler_diagnostics" not in contract["output_paths"]
    assert contract["required_artifacts"] == ["instrumentation.json"]


def test_artifact_contract_returns_independent_copies():
    config = _sampled_config()
    first = build_online_ler_artifact_contract(config)
    second = build_online_ler_artifact_contract(config)

    assert first is not second
    assert first["output_paths"] is not second["output_paths"]
    assert first["required_artifacts"] is not second["required_artifacts"]

    first["output_paths"]["extra"] = "extra.json"
    first["required_artifacts"].append("extra.json")

    assert "extra" not in second["output_paths"]
    assert "extra.json" not in second["required_artifacts"]


def test_callback_save_diagnostics_writes_runtime_metadata():
    tracker = _FakeTracker(
        {
            "ler": 0.2,
            "rho_vg": 0.5,
            "parameter_sample_size_realized": 128,
            "observation_age_decisions": 1,
            "n_updates": 3,
            "n_decisions": 7,
        }
    )
    instrumentation = {
        "batches_seen": 10,
        "online_ler_update_attempts": 5,
        "online_ler_update_successes": 3,
        "online_ler_last_update_decision": 8,
    }
    holder = [_FakeTrainer(instrumentation)]

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = _build_callback(tracker, holder, tmpdir, _sampled_config())
        callback._save_diagnostics()
        with open(os.path.join(tmpdir, "ler_diagnostics.json")) as handle:
            payload = json.load(handle)

    runtime = payload["online_diagnostics_runtime"]
    assert runtime["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED
    assert runtime["parameter_sample_size_realized"] == 128
    assert runtime["update_attempts"] == 5
    assert runtime["update_successes"] == 3
    assert runtime["last_update_decision"] == 8
    assert runtime["observation_age_decisions"] == 1
    assert runtime["n_updates"] == 3
    assert runtime["n_decisions"] == 7
    assert payload["ler_history"] == [0.1, 0.2]
    assert payload["rho_vg_history"] == [0.5]
    assert payload["velocity_history"] == [1.0, 2.0]
    assert payload["ablation_name"] == "full_lerna"
    assert payload["skip_update_mode"] == "freeze"


def test_callback_save_diagnostics_does_not_mutate_tracker_diagnostics():
    shared_diag = {"ler": 0.2, "n_updates": 3, "n_decisions": 7}
    tracker = _FakeTracker(shared_diag)
    holder = [None]
    config = _legacy_config()
    config_before = copy.deepcopy(config)
    diag_before = copy.deepcopy(shared_diag)

    with tempfile.TemporaryDirectory() as tmpdir:
        callback = _build_callback(tracker, holder, tmpdir, config)
        callback._save_diagnostics()
        with open(os.path.join(tmpdir, "ler_diagnostics.json")) as handle:
            payload = json.load(handle)

    assert shared_diag == diag_before
    assert config == config_before
    assert payload["online_diagnostics_runtime"]["n_updates"] == 3
    assert payload["online_diagnostics_runtime"]["n_decisions"] == 7
