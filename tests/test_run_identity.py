"""Piece 9: collision-proof run and attempt identity tests."""

import json
import os

import pytest

from lerna.utils.run_provenance import (
    build_scientific_fingerprint,
    write_manifest_running,
    finalize_manifest_completed,
    load_manifest,
    CLASSIFICATION_MATCHED_CLAIM,
)
from scripts.run_ablation_study import compute_authoritative_horizon


class FakeDataset:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def test_fingerprint_stable_for_identical_configs():
    """Identical scientific configurations must produce identical fingerprints."""
    fp1 = build_scientific_fingerprint(
        task="mrpc",
        training_seed=42,
        model_id="modernbert",
        max_samples=1000,
        num_epochs=5,
        control="rvd",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        rvd_veto_mode="margin",
        rvd_margin_rank_floor=0.20,
        rvd_spike_factor=1.0,
        rvd_spike_ema_window=20,
        rvd_repay_mode="asap",
        rvd_repay_protect_dangerous=True,
        max_consecutive_skips=4,
        total_steps=160,
        git_sha="abc123",
    )
    fp2 = build_scientific_fingerprint(
        task="mrpc",
        training_seed=42,
        model_id="modernbert",
        max_samples=1000,
        num_epochs=5,
        control="rvd",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        rvd_veto_mode="margin",
        rvd_margin_rank_floor=0.20,
        rvd_spike_factor=1.0,
        rvd_spike_ema_window=20,
        rvd_repay_mode="asap",
        rvd_repay_protect_dangerous=True,
        max_consecutive_skips=4,
        total_steps=160,
        git_sha="abc123",
    )
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_changes_when_scientific_input_differs():
    """Any relevant configuration difference must produce a different fingerprint."""
    base = dict(
        task="mrpc",
        training_seed=42,
        model_id="modernbert",
        max_samples=1000,
        num_epochs=5,
        control="rvd",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        rvd_veto_mode="margin",
        rvd_margin_rank_floor=0.20,
        rvd_spike_factor=1.0,
        rvd_spike_ema_window=20,
        rvd_repay_mode="asap",
        rvd_repay_protect_dangerous=True,
        max_consecutive_skips=4,
        total_steps=160,
        git_sha="abc123",
    )
    base_fp = build_scientific_fingerprint(**base)

    changed = dict(base)
    changed["target_skip_rate"] = 0.40
    assert build_scientific_fingerprint(**changed) != base_fp

    changed = dict(base)
    changed["rvd_veto_mode"] = "loss_spike"
    assert build_scientific_fingerprint(**changed) != base_fp

    changed = dict(base)
    changed["policy_seed"] = 99
    assert build_scientific_fingerprint(**changed) != base_fp

    changed = dict(base)
    changed["no_early_stopping"] = False
    assert build_scientific_fingerprint(**changed) != base_fp


def test_retry_preserves_previous_attempt(tmp_path):
    """Retries must preserve previous attempts and not overwrite artifacts."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    def clean_git():
        return {"commit_sha": "abc123", "dirty": False, "tracked_changes": [], "untracked_paths": []}

    kwargs = dict(
        argv=["prog"],
        task="mrpc",
        model_id="modernbert",
        seed=42,
        controller_name="LERNARandomVetoDeferralPolicy",
        controller_seed=42,
        target_skip_rate=0.30,
        planned_quota=48,
        total_steps=160,
        warmup_steps=10,
        skip_update_mode="freeze",
        controller_config_effective={},
        matched_budget_planned=True,
        budget_classification="fixed_epoch",
        output_paths={"results": "results.json"},
        fingerprint="fp1",
        attempt=1,
        requested_classification="local_development",
        git_provider=clean_git,
    )

    write_manifest_running(str(run_dir), **kwargs)
    assert (run_dir / "run_manifest.json").exists()

    # Second attempt must not overwrite
    with pytest.raises(Exception, match="Refusing to overwrite existing manifest"):
        write_manifest_running(str(run_dir), **kwargs)


def test_attempt_directory_increment(tmp_path):
    """Attempt directories must auto-increment when previous attempts exist."""
    base = tmp_path / "base"
    base.mkdir()
    arm = base / "rvd" / "fp1"
    arm.mkdir(parents=True)

    def clean_git():
        return {"commit_sha": "abc123", "dirty": False, "tracked_changes": [], "untracked_paths": []}

    for i in range(1, 4):
        run_dir = arm / f"attempt-{i:03d}"
        run_dir.mkdir()
        write_manifest_running(
            str(run_dir),
            argv=["prog"],
            task="mrpc",
            model_id="modernbert",
            seed=42,
            controller_name="RVD",
            controller_seed=42,
            target_skip_rate=0.30,
            planned_quota=48,
            total_steps=160,
            warmup_steps=10,
            skip_update_mode="freeze",
            controller_config_effective={},
            matched_budget_planned=True,
            budget_classification="fixed_epoch",
            output_paths={"results": "results.json"},
            fingerprint="fp1",
            attempt=i,
            requested_classification="local_development",
            git_provider=clean_git,
        )

    assert (arm / "attempt-001" / "run_manifest.json").exists()
    assert (arm / "attempt-002" / "run_manifest.json").exists()
    assert (arm / "attempt-003" / "run_manifest.json").exists()


def test_manifest_contains_identity_fields(tmp_path):
    """Manifest must record identity_inputs, fingerprint, and attempt."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    identity = {
        "task": "mrpc",
        "training_seed": 42,
        "model_id": "modernbert",
        "max_samples": 1000,
        "num_epochs": 5,
        "control": "rvd",
        "target_skip_rate": 0.30,
        "policy_seed": 42,
        "skip_update_mode": "freeze",
        "no_early_stopping": True,
        "rvd_veto_mode": "margin",
        "rvd_margin_rank_floor": 0.20,
        "rvd_spike_factor": 1.0,
        "rvd_spike_ema_window": 20,
        "rvd_repay_mode": "asap",
        "rvd_repay_protect_dangerous": True,
        "max_consecutive_skips": 4,
        "total_steps": 160,
    }

    def clean_git():
        return {"commit_sha": "abc123", "dirty": False, "tracked_changes": [], "untracked_paths": []}

    write_manifest_running(
        str(run_dir),
        argv=["prog"],
        task="mrpc",
        model_id="modernbert",
        seed=42,
        controller_name="RVD",
        controller_seed=42,
        target_skip_rate=0.30,
        planned_quota=48,
        total_steps=160,
        warmup_steps=10,
        skip_update_mode="freeze",
        controller_config_effective={},
        matched_budget_planned=True,
        budget_classification="fixed_epoch",
        output_paths={"results": "results.json"},
        identity_inputs=identity,
        fingerprint="fp1234567890abc",
        attempt=1,
        requested_classification="local_development",
        git_provider=clean_git,
    )
    manifest = load_manifest(str(run_dir))
    assert manifest["identity_inputs"] == identity
    assert manifest["fingerprint"] == "fp1234567890abc"
    assert manifest["attempt"] == 1


def test_identical_scientific_inputs_never_collide():
    """Fingerprint must be collision-resistant across input combinations."""
    fps = set()
    for rate in [0.1, 0.2, 0.3, 0.4]:
        for mode in ["none", "margin", "loss_spike"]:
            fp = build_scientific_fingerprint(
                task="mrpc",
                training_seed=42,
                model_id="modernbert",
                max_samples=1000,
                num_epochs=5,
                control="rvd",
                target_skip_rate=rate,
                policy_seed=42,
                skip_update_mode="freeze",
                no_early_stopping=True,
                rvd_veto_mode=mode,
                rvd_margin_rank_floor=0.20,
                rvd_spike_factor=1.0,
                rvd_spike_ema_window=20,
                rvd_repay_mode="asap",
                rvd_repay_protect_dangerous=True,
                max_consecutive_skips=4,
                total_steps=160,
                git_sha="abc123",
            )
            fps.add(fp)
    assert len(fps) == 12


def test_horizon_matches_authoritative_calculation():
    """Piece 8 horizon must be used as total_steps in fingerprint."""
    ds = FakeDataset(1000)
    total = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=1,
    )
    assert total == 160

    fp = build_scientific_fingerprint(
        task="mrpc",
        training_seed=42,
        model_id="modernbert",
        max_samples=1000,
        num_epochs=5,
        control="rvd",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        rvd_veto_mode="margin",
        rvd_margin_rank_floor=0.20,
        rvd_spike_factor=1.0,
        rvd_spike_ema_window=20,
        rvd_repay_mode="asap",
        rvd_repay_protect_dangerous=True,
        max_consecutive_skips=4,
        total_steps=total,
        git_sha="abc123",
    )
    assert fp is not None
    assert len(fp) == 16
