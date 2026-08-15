"""Runner integration tests for persisted Phase 1.3 operations."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from lerna.utils.phase1_3_matrix import PHASE1_3_CANONICAL_ARMS
from scripts import run_ablation_study as runner

RATES = [0.30, 0.40]


def _facts():
    return {
        "task": "mrpc",
        "num_epochs": 3,
        "max_samples_requested": None,
        "max_samples_effective": 25000,
        "train_samples_realized": 3668,
        "eval_samples_realized": 408,
        "train_dataset_fingerprint": "mrpc-train",
        "eval_dataset_fingerprint": "mrpc-eval",
        "total_steps": 575,
        "per_device_train_batch_size": 8,
        "gradient_accumulation_steps": 1,
        "effective_n_gpu": 1,
    }


def _bundle(root: Path):
    plan = runner.build_phase1_3_matrix_plan(
        tasks=["mrpc"],
        seeds=[runner.PILOT_SEED],
        target_skip_rates=RATES,
        model_name=runner.MODELS["ettin"],
        model_revision=runner.ETTIN_REVISION,
        base_output_dir=str(root),
        data_facts_provider=lambda task: _facts(),
        git_sha="a" * 40,
        scheduler_step_policy="skip_on_backward_skip",
        max_consecutive_skips=4,
        probe_interval=8,
        rho_veto_threshold=-0.2,
        risk_gamma=0.0,
        online_ler_mode="auto",
        online_ler_parameter_sample_size=4096,
        online_ler_update_interval=1,
        use_rho_vg=True,
        use_safety_horizon=True,
        provenance_classification=runner.CLASSIFICATION_PILOT_NON_CLAIM,
    )
    environment = {
        "hardware_config": {"max_samples": 25000},
        "locked": True,
    }
    envelope = {
        "matrix_kind": "pilot",
        "git_sha": "a" * 40,
        "plan_sha256": "b" * 64,
        "environment_sha256": "c" * 64,
    }
    return {
        "root": root,
        "envelope": envelope,
        "plan": plan,
        "environment": environment,
        "dimensions": {
            "tasks": ["mrpc"],
            "seeds": [runner.PILOT_SEED],
            "target_skip_rates": RATES,
        },
    }


def _argv(action, root, *extra, pilot=True):
    argv = [
        "run_ablation_study.py",
        "--mode",
        "phase1_3",
        "--phase1-3-action",
        action,
        "--model",
        "ettin",
        "--output-dir",
        str(root),
    ]
    if pilot:
        argv.append("--pilot")
    argv.extend(extra)
    return argv


def _install_common(monkeypatch, bundle):
    clean = mock.Mock(
        return_value={"commit_sha": "a" * 40, "status_entries": []}
    )
    environment = mock.Mock(return_value=dict(bundle["environment"]))
    tokenizer = mock.Mock(
        side_effect=AssertionError("persisted actions must not load a tokenizer")
    )
    monkeypatch.setattr(runner, "detect_device_profile", lambda: "server")
    monkeypatch.setattr(
        runner,
        "get_training_config",
        lambda profile: {
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 16,
            "gradient_accumulation_steps": 1,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 0,
            "max_samples": 25000,
        },
    )
    monkeypatch.setattr(runner, "require_clean_git_state", clean)
    monkeypatch.setattr(runner, "load_phase1_3_plan", lambda root: bundle)
    monkeypatch.setattr(runner, "collect_phase1_3_environment", environment)
    monkeypatch.setattr(runner, "assert_environment_matches", lambda *args: None)
    monkeypatch.setattr(runner, "load_tokenizer", tokenizer)
    return clean, environment, tokenizer


def test_run_consumes_persisted_plan_in_exact_order(monkeypatch, tmp_path):
    bundle = _bundle(tmp_path / "pilot")
    _, _, tokenizer = _install_common(monkeypatch, bundle)
    fresh = mock.Mock()
    run_cell = mock.Mock(return_value={"ok": True})
    freeze = mock.Mock(
        return_value={"n_valid_runs": 12, "completed_matrix_valid": True}
    )
    progress = [
        {
            "cell": cell,
            "state": "pending",
            "completed": None,
            "failed_attempts": [],
        }
        for cell in bundle["plan"]
    ]
    monkeypatch.setattr(runner, "assert_fresh_execution", fresh)
    monkeypatch.setattr(runner, "scan_phase1_3_progress", lambda *args, **kwargs: progress)
    monkeypatch.setattr(runner, "run_ablation_single", run_cell)
    monkeypatch.setattr(runner, "_validate_and_freeze_strict_matrix", freeze)
    monkeypatch.setattr(sys, "argv", _argv("run", bundle["root"]))

    runner.main()

    fresh.assert_called_once_with(
        bundle["plan"], base_output_dir=str(bundle["root"])
    )
    assert run_cell.call_count == 12
    assert [
        (
            call.kwargs["task_name"],
            call.kwargs["seed"],
            call.kwargs["target_skip_rate"],
            call.kwargs["ablation_name"],
        )
        for call in run_cell.call_args_list
    ] == [
        ("mrpc", runner.PILOT_SEED, rate, arm)
        for rate in RATES
        for arm in PHASE1_3_CANONICAL_ARMS
    ]
    assert all(
        call.kwargs["planned_cell"] is cell
        for call, cell in zip(run_cell.call_args_list, bundle["plan"])
    )
    assert all(call.kwargs["skip_update_mode"] == "freeze" for call in run_cell.call_args_list)
    assert all(call.kwargs["max_consecutive_skips"] == 4 for call in run_cell.call_args_list)
    assert all(call.kwargs["probe_interval"] == 8 for call in run_cell.call_args_list)
    assert all(call.kwargs["rho_veto_threshold"] == -0.2 for call in run_cell.call_args_list)
    assert all(call.kwargs["risk_gamma"] == 0.0 for call in run_cell.call_args_list)
    assert all(
        call.kwargs["online_ler_parameter_sample_size"] == 4096
        for call in run_cell.call_args_list
    )
    assert all(
        call.kwargs["online_ler_update_interval"] == 1
        for call in run_cell.call_args_list
    )
    assert all(
        call.kwargs["provenance_classification"] == runner.CLASSIFICATION_PILOT_NON_CLAIM
        for call in run_cell.call_args_list
    )
    tokenizer.assert_not_called()
    freeze.assert_called_once_with(bundle)


def test_resume_recovers_then_skips_only_verified_completion(
    monkeypatch, tmp_path
):
    bundle = _bundle(tmp_path / "pilot")
    _install_common(monkeypatch, bundle)
    recover = mock.Mock(return_value=["attempt-001"])
    run_cell = mock.Mock(return_value={"ok": True})
    freeze = mock.Mock(
        return_value={"n_valid_runs": 12, "completed_matrix_valid": True}
    )
    progress = [
        {
            "cell": cell,
            "state": "completed" if index == 0 else "pending",
            "completed": {"attempt": 1} if index == 0 else None,
            "failed_attempts": [],
        }
        for index, cell in enumerate(bundle["plan"])
    ]
    monkeypatch.setattr(runner, "recover_stale_running_attempts", recover)
    monkeypatch.setattr(runner, "scan_phase1_3_progress", lambda *args, **kwargs: progress)
    monkeypatch.setattr(runner, "run_ablation_single", run_cell)
    monkeypatch.setattr(runner, "_validate_and_freeze_strict_matrix", freeze)
    monkeypatch.setattr(
        sys,
        "argv",
        _argv(
            "resume",
            bundle["root"],
            "--recover-stale-running-after-hours",
            "12",
        ),
    )

    runner.main()

    recover.assert_called_once_with(
        bundle["plan"],
        base_output_dir=str(bundle["root"]),
        stale_after_hours=12.0,
    )
    assert run_cell.call_count == 11
    assert run_cell.call_args_list[0].kwargs["planned_cell"] is bundle["plan"][1]
    freeze.assert_called_once_with(bundle)


def test_validate_action_has_no_training_or_planning_side_effects(
    monkeypatch, tmp_path
):
    bundle = _bundle(tmp_path / "pilot")
    _, _, tokenizer = _install_common(monkeypatch, bundle)
    run_cell = mock.Mock(side_effect=AssertionError("validate must not train"))
    scan = mock.Mock(side_effect=AssertionError("validate uses completed validator"))
    freeze = mock.Mock(
        return_value={"n_valid_runs": 12, "completed_matrix_valid": True}
    )
    monkeypatch.setattr(runner, "run_ablation_single", run_cell)
    monkeypatch.setattr(runner, "scan_phase1_3_progress", scan)
    monkeypatch.setattr(runner, "_validate_and_freeze_strict_matrix", freeze)
    monkeypatch.setattr(sys, "argv", _argv("validate", bundle["root"]))

    runner.main()

    tokenizer.assert_not_called()
    run_cell.assert_not_called()
    scan.assert_not_called()
    freeze.assert_called_once_with(bundle)


def test_pilot_flag_must_match_persisted_plan(monkeypatch, capsys, tmp_path):
    bundle = _bundle(tmp_path / "pilot")
    _install_common(monkeypatch, bundle)
    monkeypatch.setattr(
        sys, "argv", _argv("validate", bundle["root"], pilot=False)
    )

    with pytest.raises(SystemExit):
        runner.main()

    assert "--pilot must be present" in capsys.readouterr().err


def test_power_evidence_embeds_raw_authoritative_copy():
    callback = SimpleNamespace(
        energy_measurement_source="nvidia-smi",
        energy_valid=True,
        energy_invalid_reason="ok",
        _gpu_name="NVIDIA V100-SXM2-32GB",
        gpu_index=0,
        gpu_selector="0",
        sample_interval_s=1.0,
        _nvidia_smi_query_count=3,
        _nvidia_smi_success_count=3,
        cumulative_kwh=0.002,
        _power_samples=[
            {
                "timestamp": 1.0,
                "power_w": 200.0,
                "measurement_source": "nvidia-smi",
            }
        ],
        step_energies=[{"step": 1, "step_kwh": 0.001}],
    )

    evidence = runner.build_power_evidence(callback)

    assert evidence["authoritative_copy"] == "results.json"
    assert evidence["energy_valid"] is True
    assert evidence["raw_samples"] == callback._power_samples
    assert evidence["raw_samples"] is not callback._power_samples
    assert evidence["per_step_energy"] is not callback.step_energies
