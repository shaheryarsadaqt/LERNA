"""Dependency-light tests for Phase 1.3 persisted operational controls."""

from __future__ import annotations

import importlib.util
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lerna"
    / "utils"
    / "phase1_3_operations.py"
)
SPEC = importlib.util.spec_from_file_location(
    "phase1_3_operations_under_test", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
operations = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(operations)


def _cell(arm="full_finetune", fingerprint="0123456789abcdef"):
    return {
        "arm": arm,
        "task": "mrpc",
        "training_seed": 7,
        "model_id": "jhu-clsp/ettin-encoder-150m",
        "model_revision": "45d08642849e5c5701b162671ac811b7654bfd9f",
        "fingerprint": fingerprint,
        "identity_inputs": {"task": "mrpc", "seed": 7},
        "online_diagnostics": {"enabled": False},
    }


def _persist(root: Path, plan=None, environment=None):
    plan = [_cell()] if plan is None else plan
    environment = {"locked": True} if environment is None else environment
    operations.persist_phase1_3_plan(
        base_output_dir=root,
        plan=plan,
        tasks=["mrpc"],
        seeds=[7],
        target_skip_rates=[0.30, 0.40],
        matrix_kind="pilot",
        git_sha="a" * 40,
        environment=environment,
    )
    return operations.load_phase1_3_plan(root)


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_frozen_mrpc_facts_are_exact_and_type_strict():
    facts = {
        "task": "mrpc",
        "train_samples_realized": operations.MRPC_TRAIN_SAMPLES,
        "eval_samples_realized": operations.MRPC_VALIDATION_SAMPLES,
        "total_steps": operations.MRPC_TOTAL_STEPS,
    }
    operations.require_frozen_mrpc_facts(facts)

    drifted = dict(facts, total_steps=574)
    with pytest.raises(operations.Phase13OperationalError, match="total_steps"):
        operations.require_frozen_mrpc_facts(drifted)

    drifted = dict(facts, train_samples_realized=3668.0)
    with pytest.raises(
        operations.Phase13OperationalError, match="train_samples_realized"
    ):
        operations.require_frozen_mrpc_facts(drifted)


def test_plan_persistence_is_atomic_fresh_and_hash_verified(tmp_path):
    root = tmp_path / "matrix"
    bundle = _persist(root)

    assert bundle["envelope"]["matrix_kind"] == "pilot"
    assert bundle["envelope"]["plan_sha256"] == operations.canonical_sha256(
        bundle["plan"]
    )
    assert bundle["envelope"]["environment_sha256"] == (
        operations.canonical_sha256(bundle["environment"])
    )
    assert sorted(path.name for path in root.iterdir()) == [
        operations.ENVIRONMENT_FILENAME,
        operations.PLAN_FILENAME,
        operations.PLAN_CHECKSUM_FILENAME,
    ]
    with pytest.raises(operations.Phase13OperationalError, match="absent"):
        _persist(root)

    plan_path = root / operations.PLAN_FILENAME
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["plan"][0]["task"] = "sst2"
    _write_json(plan_path, payload)
    with pytest.raises(
        operations.Phase13OperationalError, match="checksum mismatch"
    ):
        operations.load_phase1_3_plan(root)


def test_clean_git_gate_includes_untracked_and_exact_sha(monkeypatch, tmp_path):
    calls = []

    def clean(command, cwd=None):
        calls.append(command)
        return "b" * 40 if command[1:3] == ["rev-parse", "HEAD"] else ""

    monkeypatch.setattr(operations, "_run_command", clean)
    assert operations.require_clean_git_state(
        tmp_path, expected_sha="b" * 40
    )["commit_sha"] == "b" * 40
    assert any("--untracked-files=all" in command for command in calls)

    def dirty(command, cwd=None):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return "b" * 40
        return "?? local.txt"

    monkeypatch.setattr(operations, "_run_command", dirty)
    with pytest.raises(operations.Phase13OperationalError, match="untracked"):
        operations.require_clean_git_state(tmp_path)


def test_environment_lock_enforces_offline_v100_and_exact_equality(
    monkeypatch, tmp_path
):
    for key, value in operations._REQUIRED_OFFLINE_ENV.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setattr(
        operations,
        "_package_versions",
        lambda: {"torch": "2.5.1"},
    )
    monkeypatch.setattr(
        operations,
        "_requirements_hashes",
        lambda root: {"requirements/x.txt": "c" * 64},
    )
    monkeypatch.setattr(
        operations,
        "_selected_gpu_evidence",
        lambda: {
            "selector": "0",
            "name": "NVIDIA V100-SXM2-32GB",
            "uuid": "GPU-1",
            "memory_total_mib": 32510,
            "driver_version": "555.1",
        },
    )
    monkeypatch.setattr(
        operations,
        "_ettin_snapshot_evidence",
        lambda *args, **kwargs: {
            "snapshot_path": "/raid/hf_cache/snapshot",
            "file_count": 5,
            "inventory_sha256": "d" * 64,
            "entries": [],
        },
    )
    monkeypatch.setattr(
        operations,
        "_cache_tree_evidence",
        lambda path, label: {
            "path": str(path),
            "file_count": 2,
            "total_bytes": 100,
            "inventory_sha256": "f" * 64,
        },
    )
    hardware = {
        "per_device_train_batch_size": 8,
        "per_device_eval_batch_size": 16,
        "gradient_accumulation_steps": 1,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "max_samples": 25000,
    }
    evidence = operations.collect_phase1_3_environment(
        repo_root=tmp_path,
        profile="server",
        model_id="jhu-clsp/ettin-encoder-150m",
        model_revision="e" * 40,
        hardware_config=hardware,
    )
    operations.assert_environment_matches(evidence, dict(evidence))
    drifted = json.loads(json.dumps(evidence))
    drifted["gpu"]["uuid"] = "GPU-2"
    with pytest.raises(operations.Phase13OperationalError, match="does not match"):
        operations.assert_environment_matches(evidence, drifted)

    monkeypatch.setenv("HF_HUB_OFFLINE", "0")
    with pytest.raises(operations.Phase13OperationalError, match="offline"):
        operations.collect_phase1_3_environment(
            repo_root=tmp_path,
            profile="server",
            model_id="jhu-clsp/ettin-encoder-150m",
            model_revision="e" * 40,
            hardware_config=hardware,
        )


def test_progress_scanner_skips_only_verified_completion(monkeypatch, tmp_path):
    root = tmp_path / "matrix"
    bundle = _persist(root)
    cell = bundle["plan"][0]
    attempt = root / cell["arm"] / cell["fingerprint"] / "attempt-001"
    _write_json(
        attempt / "run_manifest.json",
        {
            "status": "completed",
            "fingerprint": cell["fingerprint"],
            "identity_inputs": cell["identity_inputs"],
            "attempt": 1,
        },
    )
    verified = {
        "attempt": 1,
        "attempt_dir": attempt,
        "manifest": {"status": "completed"},
        "results": {"task": "mrpc"},
    }
    reader = lambda observed_cell, observed_dir, observed_num: verified
    monkeypatch.setattr(operations, "_read_completed_attempt", reader)

    progress = operations.scan_phase1_3_progress(
        bundle["plan"], base_output_dir=root
    )
    assert progress[0]["state"] == "completed"
    assert progress[0]["completed"] is verified

    foreign = root / cell["arm"] / "foreign"
    foreign.mkdir(parents=True)
    with pytest.raises(operations.Phase13OperationalError, match="foreign"):
        operations.scan_phase1_3_progress(
            bundle["plan"], base_output_dir=root
        )


def test_progress_rejects_running_and_fresh_execution_rejects_cells(tmp_path):
    root = tmp_path / "matrix"
    bundle = _persist(root)
    cell = bundle["plan"][0]
    attempt = root / cell["arm"] / cell["fingerprint"] / "attempt-001"
    _write_json(
        attempt / "run_manifest.json",
        {
            "status": "running",
            "start_time_utc": datetime.now(timezone.utc).isoformat(),
            "fingerprint": cell["fingerprint"],
            "identity_inputs": cell["identity_inputs"],
            "attempt": 1,
        },
    )
    with pytest.raises(operations.Phase13OperationalError, match="stale recovery"):
        operations.scan_phase1_3_progress(
            bundle["plan"], base_output_dir=root
        )
    with pytest.raises(operations.Phase13OperationalError, match="existing cell"):
        operations.assert_fresh_execution(
            bundle["plan"], base_output_dir=root
        )


def test_stale_recovery_is_explicit_age_gated_and_preserves_attempt(
    monkeypatch, tmp_path
):
    root = tmp_path / "matrix"
    bundle = _persist(root)
    cell = bundle["plan"][0]
    attempt = root / cell["arm"] / cell["fingerprint"] / "attempt-001"
    now = datetime(2026, 8, 14, tzinfo=timezone.utc)
    _write_json(
        attempt / "run_manifest.json",
        {
            "status": "running",
            "start_time_utc": (now - timedelta(hours=13)).isoformat(),
            "fingerprint": cell["fingerprint"],
            "identity_inputs": cell["identity_inputs"],
            "attempt": 1,
        },
    )
    finalized = []
    fake_provenance = SimpleNamespace(
        finalize_manifest_failed=lambda path, exc: finalized.append((path, exc))
    )
    monkeypatch.setattr(
        operations, "_provenance_module", lambda: fake_provenance
    )
    recovered = operations.recover_stale_running_attempts(
        bundle["plan"],
        base_output_dir=root,
        stale_after_hours=12,
        now=now,
    )
    assert recovered == [str(attempt)]
    assert finalized[0][0] == str(attempt)
    assert attempt.is_dir()

    finalized.clear()
    _write_json(
        attempt / "run_manifest.json",
        {
            "status": "running",
            "start_time_utc": (now - timedelta(hours=1)).isoformat(),
            "fingerprint": cell["fingerprint"],
            "identity_inputs": cell["identity_inputs"],
            "attempt": 1,
        },
    )
    with pytest.raises(operations.Phase13OperationalError, match="only 1.00h"):
        operations.recover_stale_running_attempts(
            bundle["plan"],
            base_output_dir=root,
            stale_after_hours=12,
            now=now,
        )
    assert finalized == []


def test_validation_freeze_hashes_results_and_power_evidence_immutably(tmp_path):
    root = tmp_path / "matrix"
    bundle = _persist(root)
    cell = bundle["plan"][0]
    attempt = root / cell["arm"] / cell["fingerprint"] / "attempt-001"
    power = {
        "authoritative_copy": "results.json",
        "energy_valid": True,
        "raw_samples": [{"power_w": 200.0, "timestamp": 1.0}],
    }
    _write_json(attempt / "results.json", {"power_evidence": power})
    _write_json(attempt / "run_manifest.json", {"status": "completed"})
    valid_runs = [
        {
            "cell_id": ("mrpc", 7, 0.30, "full_finetune"),
            "attempt_num": 1,
            "results_path": str(attempt / "results.json"),
        }
    ]
    report = operations.freeze_matrix_validation(
        base_output_dir=root,
        bundle=bundle,
        valid_runs=valid_runs,
    )
    assert report["completed_matrix_valid"] is True
    assert report["runs"][0]["power_evidence_sha256"] == (
        operations.canonical_sha256(power)
    )
    assert report["runs"][0]["attempt_dir"] == str(
        attempt.relative_to(root)
    )
    operations.freeze_matrix_validation(
        base_output_dir=root,
        bundle=bundle,
        valid_runs=valid_runs,
    )

    validation_path = root / operations.VALIDATION_FILENAME
    _write_json(validation_path, {"tampered": True})
    with pytest.raises(operations.Phase13OperationalError, match="overwrite"):
        operations.freeze_matrix_validation(
            base_output_dir=root,
            bundle=bundle,
            valid_runs=valid_runs,
        )
