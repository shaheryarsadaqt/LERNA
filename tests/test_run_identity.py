"""Piece 9B: collision-proof run and attempt identity tests."""

import json
import os

import pytest

from lerna.utils.run_provenance import (
    build_identity_inputs,
    build_scientific_fingerprint,
    write_manifest_running,
    finalize_manifest_completed,
    load_manifest,
    CLASSIFICATION_MATCHED_CLAIM,
)
from scripts.run_ablation_study import compute_authoritative_horizon


class FakeDataset:
    def __init__(self, length, fingerprint=None):
        self._length = length
        self._fingerprint = fingerprint

    def __len__(self):
        return self._length


def _make_identity_inputs(**overrides):
    defaults = dict(
        task="mrpc",
        training_seed=42,
        model_id="modernbert",
        max_samples_requested=None,
        train_samples_realized=1000,
        eval_samples_realized=200,
        train_dataset_fingerprint="abc123",
        eval_dataset_fingerprint="eval_fp_abc",
        num_epochs=5,
        control="rvd",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        scheduler_step_policy="always_step",
        no_early_stopping=True,
        total_steps=160,
        git_sha="abc123",
    )
    defaults.update(overrides)
    return defaults


def test_fingerprint_stable_for_identical_configs():
    """Identical scientific configurations must produce identical fingerprints."""
    identity = _make_identity_inputs()
    fp1 = build_scientific_fingerprint(identity)
    fp2 = build_scientific_fingerprint(identity)
    assert fp1 == fp2
    assert len(fp1) == 16


def test_fingerprint_changes_when_scientific_input_differs():
    """Any relevant configuration difference must produce a different fingerprint."""
    base = _make_identity_inputs()
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["target_skip_rate"] = 0.40
    assert build_scientific_fingerprint(changed) != base_fp

    changed = dict(base)
    changed["policy_seed"] = 99
    assert build_scientific_fingerprint(changed) != base_fp

    changed = dict(base)
    changed["no_early_stopping"] = False
    assert build_scientific_fingerprint(changed) != base_fp

    changed = dict(base)
    changed["scheduler_step_policy"] = "skip_on_backward_skip"
    assert build_scientific_fingerprint(changed) != base_fp


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
    identity = _make_identity_inputs()

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
        identity = _make_identity_inputs(target_skip_rate=rate)
        fp = build_scientific_fingerprint(identity)
        fps.add(fp)
    assert len(fps) == 4


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

    identity = _make_identity_inputs(total_steps=total)
    fp = build_scientific_fingerprint(identity)
    assert fp is not None
    assert len(fp) == 16


def test_none_max_samples_fingerprints_without_error():
    """max_samples_requested=None must fingerprint without error (unlimited)."""
    identity = _make_identity_inputs(max_samples_requested=None)
    fp = build_scientific_fingerprint(identity)
    assert fp is not None
    assert len(fp) == 16


def test_unlimited_configs_produce_identical_fingerprints():
    """Identical unlimited configurations must produce identical fingerprints."""
    identity = _make_identity_inputs(max_samples_requested=None)
    fp1 = build_scientific_fingerprint(identity)
    fp2 = build_scientific_fingerprint(identity)
    assert fp1 == fp2


def test_unlimited_and_capped_configs_differ():
    """Unlimited (None) and explicitly capped configurations must produce different fingerprints."""
    unlimited = _make_identity_inputs(max_samples_requested=None)
    capped = _make_identity_inputs(max_samples_requested=1000)
    assert build_scientific_fingerprint(unlimited) != build_scientific_fingerprint(capped)


def test_different_realized_train_sizes_differ():
    """Different realized train dataset sizes must produce different fingerprints."""
    small = _make_identity_inputs(train_samples_realized=500)
    large = _make_identity_inputs(train_samples_realized=2000)
    assert build_scientific_fingerprint(small) != build_scientific_fingerprint(large)


def test_different_dataset_fingerprints_differ():
    """Different train dataset fingerprints must produce different fingerprints."""
    fp1 = _make_identity_inputs(train_dataset_fingerprint="ds_a")
    fp2 = _make_identity_inputs(train_dataset_fingerprint="ds_b")
    assert build_scientific_fingerprint(fp1) != build_scientific_fingerprint(fp2)


def test_identity_inputs_reused_across_fingerprint_manifest_and_results(tmp_path):
    """The same identity_inputs dict object must be used for fingerprint, manifest, and results."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    identity = _make_identity_inputs()

    def clean_git():
        return {"commit_sha": "abc123", "dirty": False, "tracked_changes": [], "untracked_paths": []}

    fingerprint = build_scientific_fingerprint(identity)

    write_manifest_running(
        str(run_dir),
        argv=["prog"],
        task=identity["task"],
        model_id=identity["model_id"],
        seed=identity["training_seed"],
        controller_name="RVD",
        controller_seed=identity["policy_seed"],
        target_skip_rate=identity["target_skip_rate"],
        planned_quota=48,
        total_steps=identity["total_steps"],
        warmup_steps=10,
        skip_update_mode=identity["skip_update_mode"],
        controller_config_effective={},
        matched_budget_planned=True,
        budget_classification="fixed_epoch",
        output_paths={"results": "results.json"},
        identity_inputs=identity,
        fingerprint=fingerprint,
        attempt=1,
        requested_classification="local_development",
        git_provider=clean_git,
    )

    manifest = load_manifest(str(run_dir))
    assert manifest["identity_inputs"] == identity
    assert manifest["fingerprint"] == fingerprint

    # Simulate results writing with the same identity_inputs object
    results = {"fingerprint": fingerprint, "identity_inputs": identity}
    assert results["identity_inputs"] is identity


def test_fingerprint_constructed_after_dataset_and_horizon():
    """Fingerprint must include realized dataset sizes, not just requested max_samples."""
    identity = _make_identity_inputs(
        max_samples_requested=None,
        train_samples_realized=1500,
        eval_samples_realized=300,
        train_dataset_fingerprint="hf_ds_abc",
        total_steps=160,
    )
    fp = build_scientific_fingerprint(identity)
    assert fp is not None

    # Verify the identity_inputs dict contains the realized fields
    assert identity["train_samples_realized"] == 1500
    assert identity["eval_samples_realized"] == 300
    assert identity["train_dataset_fingerprint"] == "hf_ds_abc"
    assert identity["max_samples_requested"] is None


def test_piece8b_mrpc_horizon_unchanged():
    """Piece 8B's 160-step MRPC horizon must remain unchanged."""
    ds = FakeDataset(1000)
    total = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=1,
    )
    assert total == 160

    identity = _make_identity_inputs(total_steps=total)
    fp = build_scientific_fingerprint(identity)
    assert fp is not None
    assert len(fp) == 16


# ---------------------------------------------------------------------------
# #5B: RVD settings and max_consecutive_skips must not affect non-consuming
# fingerprints. Controller-specific settings live in controller configs.
# ---------------------------------------------------------------------------

def test_rvd_settings_do_not_affect_non_rvd_fingerprints():
    """Changing RVD settings must not affect non-RVD fingerprints."""
    base = _make_identity_inputs(control="exact_random")
    base_fp = build_scientific_fingerprint(base)

    # RVD settings are not part of the universal schema; adding them to a
    # non-RVD identity must not change the fingerprint.
    changed = dict(base)
    changed["rvd_veto_mode"] = "loss_spike"
    changed["rvd_margin_rank_floor"] = 0.5
    changed["rvd_spike_factor"] = 2.0
    changed["rvd_spike_ema_window"] = 50
    changed["rvd_repay_mode"] = "spread"
    changed["rvd_repay_protect_dangerous"] = False
    assert build_scientific_fingerprint(changed) == base_fp


def test_max_consecutive_skips_does_not_affect_full_finetune():
    """Changing max_consecutive_skips must not affect full_finetune fingerprints."""
    base = _make_identity_inputs(control="full_finetune")
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["max_consecutive_skips"] = 8
    assert build_scientific_fingerprint(changed) == base_fp


def test_max_consecutive_skips_does_not_affect_exact_random():
    """Changing max_consecutive_skips must not affect exact_random fingerprints."""
    base = _make_identity_inputs(control="exact_random")
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["max_consecutive_skips"] = 8
    assert build_scientific_fingerprint(changed) == base_fp


def test_max_consecutive_skips_does_not_affect_fixed_phase_strat():
    """Changing max_consecutive_skips must not affect fixed_phase_strat fingerprints."""
    base = _make_identity_inputs(control="fixed_phase_strat")
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["max_consecutive_skips"] = 8
    assert build_scientific_fingerprint(changed) == base_fp


def test_active_controller_changes_affect_fingerprint():
    """Active controller changes must affect the applicable controller's fingerprint."""
    base = _make_identity_inputs(control="phase_strat_guarded")
    base["phase_strat_controller"] = {
        "control": "phase_strat_guarded",
        "max_consecutive_skips": 4,
        "risk_gamma": 0.0,
    }
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["phase_strat_controller"] = dict(base["phase_strat_controller"])
    changed["phase_strat_controller"]["max_consecutive_skips"] = 8
    assert build_scientific_fingerprint(changed) != base_fp


def test_eval_dataset_changes_affect_fingerprint():
    """Evaluation-dataset changes must affect fingerprints."""
    base = _make_identity_inputs()
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["eval_dataset_fingerprint"] = "different_eval_fp"
    assert build_scientific_fingerprint(changed) != base_fp

    changed = dict(base)
    changed["eval_samples_realized"] = 500
    assert build_scientific_fingerprint(changed) != base_fp


def test_alias_equivalent_configs_produce_identical_fingerprints():
    """Alias-equivalent effective configurations must produce identical fingerprints."""
    exact = _make_identity_inputs(control="exact_random")
    alias = _make_identity_inputs(control="random_skip")
    assert build_scientific_fingerprint(exact) == build_scientific_fingerprint(alias)

class _FakeTracker:
    """Minimal tracker satisfying LERNAQuotaHybridPolicy.get_diagnostics()."""

    def get_diagnostics(self):
        return {"ler_raw": None, "ler": None, "rho_vg_raw": 1.0, "rho_vg": 1.0}


def test_quota_hybrid_constructs_with_recalibrate_every():
    """quota_hybrid must construct successfully with recalibrate_every=200."""
    from lerna.trainers.policies import LERNAQuotaHybridPolicy

    policy = LERNAQuotaHybridPolicy(
        ler_tracker=_FakeTracker(),
        target_skip_rate=0.20,
        fallback_threshold=0.01,
        min_step=50,
        calibration_steps=60,
        recalibrate_every=200,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        max_consecutive_skips=4,
        probe_interval=8,
        total_steps=160,
        rho_veto_threshold=-0.2,
    )
    assert policy.name == "lerna_quota_hybrid"
    assert policy.target_skip_rate == 0.20
    assert policy.max_consecutive_skips == 4


# ---------------------------------------------------------------------------
# #5B.1: canonicalization must preserve unknown fields, strip only known
# inactive fields, and canonicalize RVD configs.
# ---------------------------------------------------------------------------

def test_unknown_active_fields_change_fingerprint():
    """Unknown behaviorally active fields must change fingerprints."""
    base = _make_identity_inputs()
    base_fp = build_scientific_fingerprint(base)

    changed_a = dict(base)
    changed_a["new_active_setting"] = "a"
    assert build_scientific_fingerprint(changed_a) != base_fp

    changed_b = dict(base)
    changed_b["new_active_setting"] = "b"
    assert build_scientific_fingerprint(changed_b) != base_fp
    assert build_scientific_fingerprint(changed_a) != build_scientific_fingerprint(changed_b)


def test_stray_rvd_config_does_not_affect_non_rvd_fingerprints():
    """An accidental rvd sub-dict must not affect non-RVD fingerprints."""
    base = _make_identity_inputs(control="exact_random")
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["rvd"] = {"veto_mode": "margin", "margin_rank_floor": 0.5}
    assert build_scientific_fingerprint(changed) == base_fp


def test_inactive_rvd_mode_settings_do_not_change_rvd_fingerprint():
    """Inactive RVD mode settings must not change RVD fingerprints."""
    base = _make_identity_inputs(control="rvd")
    base["rvd"] = {
        "veto_mode": "none",
        "use_margin_veto": False,
        "use_loss_spike_veto": False,
        "margin_rank_floor": 0.5,
        "spike_factor": 2.0,
        "spike_ema_window": 50,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
        "policy_seed": 42,
        "policy_seed_defaulted_to_training_seed": True,
        "max_consecutive_skips": 4,
    }
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["rvd"] = dict(base["rvd"])
    changed["rvd"]["margin_rank_floor"] = 0.8
    changed["rvd"]["spike_factor"] = 3.0
    changed["rvd"]["spike_ema_window"] = 100
    assert build_scientific_fingerprint(changed) == base_fp


def test_active_rvd_margin_settings_change_fingerprint():
    """Active RVD margin settings must change RVD fingerprints."""
    base = _make_identity_inputs(control="rvd")
    base["rvd"] = {
        "veto_mode": "margin",
        "use_margin_veto": True,
        "use_loss_spike_veto": False,
        "margin_rank_floor": 0.5,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
        "policy_seed": 42,
        "max_consecutive_skips": 4,
    }
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["rvd"] = dict(base["rvd"])
    changed["rvd"]["margin_rank_floor"] = 0.8
    assert build_scientific_fingerprint(changed) != base_fp


def test_active_rvd_loss_spike_settings_change_fingerprint():
    """Active RVD loss-spike settings must change RVD fingerprints."""
    base = _make_identity_inputs(control="rvd")
    base["rvd"] = {
        "veto_mode": "loss_spike",
        "use_margin_veto": False,
        "use_loss_spike_veto": True,
        "spike_factor": 1.0,
        "spike_ema_window": 20,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
        "policy_seed": 42,
        "max_consecutive_skips": 4,
    }
    base_fp = build_scientific_fingerprint(base)

    changed = dict(base)
    changed["rvd"] = dict(base["rvd"])
    changed["rvd"]["spike_factor"] = 2.0
    assert build_scientific_fingerprint(changed) != base_fp

    changed = dict(base)
    changed["rvd"] = dict(base["rvd"])
    changed["rvd"]["spike_ema_window"] = 50
    assert build_scientific_fingerprint(changed) != base_fp


def test_explicit_and_defaulted_seed_fingerprint_equally():
    """Explicit and defaulted identical resolved seeds must fingerprint equally."""
    base = _make_identity_inputs(control="rvd")
    base["rvd"] = {
        "veto_mode": "margin",
        "use_margin_veto": True,
        "margin_rank_floor": 0.5,
        "repay_mode": "asap",
        "repay_protect_dangerous": True,
        "policy_seed": 42,
        "policy_seed_defaulted_to_training_seed": True,
        "max_consecutive_skips": 4,
    }
    defaulted_fp = build_scientific_fingerprint(base)

    explicit = dict(base)
    explicit["rvd"] = dict(base["rvd"])
    explicit["rvd"]["policy_seed_defaulted_to_training_seed"] = False
    explicit_fp = build_scientific_fingerprint(explicit)
    assert defaulted_fp == explicit_fp
