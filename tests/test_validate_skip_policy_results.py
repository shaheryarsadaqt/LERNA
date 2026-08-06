"""Piece 5: fixture-based tests for authoritative local result validation.

No trainer/runner/W&B imports. Fixtures mirror the results.json shape written
by scripts/run_ablation_study.py after Pieces 2 and 3.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "validate_skip_policy_results",
    REPO_ROOT / "scripts" / "validate_skip_policy_results.py",
)
vspr = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = vspr
_SPEC.loader.exec_module(vspr)

TOTAL = 200
QUOTA = 30
RATE = QUOTA / TOTAL  # 0.15


def _instrumentation(
    skipped=QUOTA,
    batches=TOTAL,
    mode="freeze",
    scheduler_policy="skip_on_backward_skip",
):
    backward = batches - skipped
    scheduler_calls = batches if scheduler_policy == "always_step" else backward
    instrumentation = {
        "forward_calls": batches,
        "backward_calls": backward,
        "optimizer_step_attempts": backward,
        "scheduler_step_calls": scheduler_calls,
        "skipped_backward_steps": skipped,
        "batches_seen": batches,
        "skipped_batches": skipped,
        "skip_ratio_by_batch": skipped / max(batches, 1),
        "skip_update_mode": mode,
        "scheduler_step_policy": scheduler_policy,
        "scheduler_step_opportunities": batches,
        "invariant_forward_eq_backward_plus_skipped": True,
        "invariant_opt_le_backward": True,
        "invariant_sched_le_opportunities": True,
        "invariant_scheduler_policy_consistent": True,
    }
    if scheduler_policy == "skip_on_backward_skip":
        instrumentation["invariant_sched_le_opt"] = True
    return instrumentation


def _run_config(
    mode="freeze",
    matched=True,
    scheduler_policy="skip_on_backward_skip",
    control=None,
    controller_config=None,
):
    config = {
        "policy": "random_veto_deferral",
        "target_skip_rate": RATE,
        "no_early_stopping": True,
        "allow_early_stopping_with_skipping": False,
        "matched_budget": matched,
        "skip_update_mode": mode,
        "scheduler_step_policy": scheduler_policy,
    }
    if control is not None:
        config["control"] = control
    if controller_config is not None:
        config["controller_config"] = controller_config
    return config


def _phase1_3_identity(control, training_seed=42):
    return {
        "task": "mrpc",
        "training_seed": training_seed,
        "model_id": "modernbert",
        "max_samples_requested": None,
        "train_samples_realized": TOTAL,
        "eval_samples_realized": TOTAL // 2,
        "train_dataset_fingerprint": "abc123",
        "eval_dataset_fingerprint": "eval_fp_abc",
        "num_epochs": 3,
        "control": control,
        "target_skip_rate": RATE,
        "policy_seed": training_seed,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "no_early_stopping": True,
        "total_steps": TOTAL,
        "git_sha": "abc123",
    }


def _phase1_3_controller_config(control, policy_class, is_skipping_arm):
    config = {
        "control": control,
        "policy_class": policy_class,
        "is_skipping_arm": is_skipping_arm,
        "arm": control,
        "target_skip_rate": RATE,
        "policy_seed": 42,
        "min_step": 50,
        "configured_total_steps": TOTAL,
        "matched_budget": True,
        "is_skipping_arm": is_skipping_arm,
        "allow_early_stopping_with_skipping": False,
        "early_stopping_active": False,
        "num_epochs": 3,
    }
    if control == "full_finetune":
        config["compute_saving_mechanism"] = "none"
    else:
        config["compute_saving_mechanism"] = "backward_skipping"
    return config


def _phase1_3_online_diagnostics(control):
    config = {
        "requested_mode": "off",
        "mode": "off",
        "enabled": False,
        "timing": "none",
        "parameter_sample_size": 0,
        "update_interval": 0,
        "reason": "control does not use online LER",
        "sample_seed": None,
    }
    runtime = {
        "requested_mode": "off",
        "mode": "off",
        "enabled": False,
        "timing": "none",
        "parameter_sample_size": 0,
        "update_interval": 0,
        "reason": "control does not use online LER",
        "sample_seed": None,
        "parameter_sample_size_realized": 0,
        "update_attempts": 0,
        "update_successes": 0,
        "n_updates": 0,
        "n_decisions": 0,
        "last_update_decision": None,
        "observation_age_decisions": None,
    }
    return config, runtime


def random_results():
    """Valid completed exact-random run."""
    return {
        "eval_metrics": {"eval_accuracy": 0.9},
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "true_skip_instrumentation": _instrumentation(),
        "policy_diagnostics": {
            "policy_name": "random_skip",
            "target_skip_rate": RATE,
            "quota_total_steps": TOTAL,
            "quota_size": QUOTA,
            "requested_quota": QUOTA,
            "decisions_seen": TOTAL,
            "skip_decisions": QUOTA,
            "realized_skip_rate": RATE,
        },
        "run_config": {**_run_config(), "policy": "random"},
    }


def rvd_results():
    """Valid completed RVD run claiming exact quota."""
    return {
        "eval_metrics": {"eval_accuracy": 0.9},
        "skip_update_mode": "freeze",
        "true_skip_instrumentation": _instrumentation(),
        "policy_diagnostics": {
            "policy_name": "random_veto_deferral",
            "target_skip_rate": RATE,
            "target_veto_rate": 0.15,
            "quota_total_steps": TOTAL,
            "quota_size": QUOTA,
            "decisions_seen": TOTAL,
            "skip_decisions": QUOTA,
            "realized_skip_rate": RATE,
            "veto_rate_vs_candidates": 0.10,
            "outstanding_debt": 0,
            "deferred_pool_now": 0,
            "debt_created": 4,
            "ordinary_debt_repayments": 3,
            "forced_tail_debt_repayments": 1,
            "skip_source_counts": {
                "accepted_candidate": 26,
                "ordinary_repayment": 3,
                "forced_tail": 1,
            },
            "invariant_skip_accounting_ok": True,
            "invariant_skip_source_decomposition_ok": True,
            "invariant_debt_conservation_ok": True,
            "invariant_debt_nonnegative_ok": True,
            "invariant_candidate_count_ok": True,
            "invariant_debt_never_negative_ok": True,
            "invariant_forced_tail_no_double_count_ok": True,
            "invariant_repayment_single_source_ok": True,
        },
        "run_config": _run_config(),
    }


def full_finetune_results():
    """Valid fixed-epoch full fine-tuning baseline with no policy diagnostics."""
    identity = _phase1_3_identity("full_finetune")
    controller_config = _phase1_3_controller_config(
        "full_finetune", "AlwaysFalsePolicy", False
    )
    controller_config.update(
        {
            "requested_quota": None,
            "runtime_quota_total_steps": None,
        }
    )
    online_cfg, online_rt = _phase1_3_online_diagnostics("full_finetune")
    identity_inputs = {**identity, "online_diagnostics": online_cfg}
    return {
        "eval_metrics": {"eval_accuracy": 0.9},
        "policy_name": "always_false",
        "skip_update_mode": "freeze",
        "true_skip_instrumentation": _instrumentation(skipped=0),
        "policy_diagnostics": {},
        "ablation": "full_finetune",
        "identity_inputs": identity_inputs,
        "controller_config": {
            **controller_config,
            "online_diagnostics": online_cfg,
        },
        "online_diagnostics": online_rt,
        "fingerprint": vspr.build_scientific_fingerprint(identity_inputs),
        "run_config": {
            **_run_config(
                control="full_finetune",
                controller_config={
                    **controller_config,
                    "online_diagnostics": online_cfg,
                },
            ),
            "target_skip_rate": RATE,
            "rvd_policy_seed": 42,
            "online_diagnostics": online_cfg,
        },
    }


def write_results(tmp_path, data):
    p = tmp_path / "results.json"
    p.write_text(json.dumps(data))
    return p


def fields(report, severity=None):
    return [
        f.field for f in report.findings
        if severity is None or f.severity == severity
    ]


def test_valid_exact_random_passes(tmp_path):
    report = vspr.validate_results(write_results(tmp_path, random_results()))
    assert report.ok, [f.to_dict() for f in report.errors]
    assert report.protocol_complete is True
    assert report.valid_for_matched_budget is True


def test_valid_always_step_artifact_passes(tmp_path):
    data = random_results()
    data["scheduler_step_policy"] = "always_step"
    data["run_config"]["scheduler_step_policy"] = "always_step"
    data["true_skip_instrumentation"] = _instrumentation(
        scheduler_policy="always_step"
    )
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok, [finding.to_dict() for finding in report.errors]


def test_scheduler_policy_mismatch_fails(tmp_path):
    data = random_results()
    data["scheduler_step_policy"] = "always_step"
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "scheduler_step_policy_consistency" in fields(report, "error")


def test_always_step_count_above_opportunities_fails(tmp_path):
    data = random_results()
    data["scheduler_step_policy"] = "always_step"
    data["run_config"]["scheduler_step_policy"] = "always_step"
    data["true_skip_instrumentation"] = _instrumentation(
        scheduler_policy="always_step"
    )
    data["true_skip_instrumentation"]["scheduler_step_calls"] = TOTAL + 1
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert (
        "true_skip_instrumentation.scheduler_step_calls"
        in fields(report, "error")
    )


def test_legacy_scheduler_artifact_defaults_to_skip_policy(tmp_path):
    data = random_results()
    del data["scheduler_step_policy"]
    del data["run_config"]["scheduler_step_policy"]
    instrumentation = data["true_skip_instrumentation"]
    del instrumentation["scheduler_step_policy"]
    del instrumentation["invariant_scheduler_policy_consistent"]
    del instrumentation["invariant_sched_le_opportunities"]
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok, [finding.to_dict() for finding in report.errors]


def test_valid_rvd_passes(tmp_path):
    report = vspr.validate_results(write_results(tmp_path, rvd_results()))
    assert report.ok, [f.to_dict() for f in report.errors]


def test_valid_full_finetune_passes(tmp_path):
    report = vspr.validate_results(
        write_results(tmp_path, full_finetune_results())
    )
    assert report.ok, [f.to_dict() for f in report.errors]


def test_nonzero_outstanding_debt_fails(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["outstanding_debt"] = 2
    data["policy_diagnostics"]["deferred_pool_now"] = 2
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "policy_diagnostics.outstanding_debt" in fields(report, "error")


def test_count_mismatch_fails(tmp_path):
    data = rvd_results()
    data["true_skip_instrumentation"]["skipped_backward_steps"] = QUOTA - 1
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "skip_decisions_vs_skipped_backward_steps" in fields(report, "error")


def test_legacy_invariant_key_not_required_when_absent(tmp_path):
    data = rvd_results()
    assert "invariant_quota_decomposition_ok" not in data["policy_diagnostics"]
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok
    assert not any("invariant_quota_decomposition_ok" in f for f in fields(report))


def test_legacy_invariant_key_checked_only_when_present(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["invariant_quota_decomposition_ok"] = False
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("policy_diagnostics.invariant_quota_decomposition_ok"
            in fields(report, "error"))


def test_early_stopping_rejects_matched_budget(tmp_path):
    data = rvd_results()
    data["run_config"]["no_early_stopping"] = False
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "run_config.matched_budget" in fields(report, "error")


def test_research_override_rejects_matched_budget(tmp_path):
    data = rvd_results()
    data["run_config"]["allow_early_stopping_with_skipping"] = True
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("run_config.allow_early_stopping_with_skipping"
            in fields(report, "error"))


def test_momentum_mode_rejected_for_matched_run(tmp_path):
    data = rvd_results()
    data["skip_update_mode"] = "momentum"
    data["run_config"]["skip_update_mode"] = "momentum"
    data["true_skip_instrumentation"]["skip_update_mode"] = "momentum"
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "skip_update_mode" in fields(report, "error")


def test_momentum_mode_allowed_as_historical_comparison(tmp_path):
    data = rvd_results()
    data["skip_update_mode"] = "momentum"
    data["run_config"]["skip_update_mode"] = "momentum"
    data["true_skip_instrumentation"]["skip_update_mode"] = "momentum"
    data["run_config"]["historical_momentum_comparison"] = True
    report = vspr.validate_results(write_results(tmp_path, data))
    assert "skip_update_mode" not in fields(report, "error")


def test_incomplete_run_rejects_matched_budget(tmp_path):
    data = rvd_results()
    diag = data["policy_diagnostics"]
    diag["decisions_seen"] = 150
    diag["skip_decisions"] = 23
    diag["realized_skip_rate"] = 23 / 150
    diag["skip_source_counts"] = {
        "accepted_candidate": 23, "ordinary_repayment": 0, "forced_tail": 0,
    }
    data["true_skip_instrumentation"] = _instrumentation(skipped=23, batches=150)
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "run_status" in fields(report, "error")


def test_interrupted_run_rejects_matched_budget(tmp_path):
    data = rvd_results()
    data["interrupted"] = True
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "run_status" in fields(report, "error")


def test_high_veto_rate_is_descriptive_not_failure(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["veto_rate_vs_candidates"] = 0.90  # >> target 0.15
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok
    infos = [f for f in report.findings if f.severity == "info"]
    assert any(f.field == "policy_diagnostics.veto_rate_vs_candidates"
               for f in infos)


def test_missing_required_artifact_fails(tmp_path):
    path = write_results(tmp_path, rvd_results())
    report = vspr.validate_results(
        path, required_artifacts=["ler_diagnostics.json"]
    )
    assert not report.ok
    assert "artifact:ler_diagnostics.json" in fields(report, "error")


def test_findings_are_structured(tmp_path):
    data = rvd_results()
    data["true_skip_instrumentation"]["skipped_backward_steps"] = QUOTA - 1
    report = vspr.validate_results(write_results(tmp_path, data))
    payload = report.to_dict()
    assert payload["ok"] is False
    for finding in payload["findings"]:
        assert set(finding) == {"severity", "field", "expected", "actual", "message"}


def test_exact_quota_requires_exact_horizon_equality(tmp_path):
    data = random_results()
    data["policy_diagnostics"].update({
        "decisions_seen": TOTAL + 1,
        "realized_skip_rate": QUOTA / (TOTAL + 1),
    })
    data["true_skip_instrumentation"] = _instrumentation(
        skipped=QUOTA, batches=TOTAL + 1
    )
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "policy_diagnostics.decisions_seen" in fields(report, "error")
    assert "run_status" in fields(report, "error")


def test_quota_rounding_does_not_fail_rate_comparison(tmp_path):
    data = random_results()
    total = 10
    quota = 3
    target = 1 / 3
    data["policy_diagnostics"].update({
        "target_skip_rate": target,
        "quota_total_steps": total,
        "quota_size": quota,
        "requested_quota": quota,
        "decisions_seen": total,
        "skip_decisions": quota,
        "realized_skip_rate": quota / total,
    })
    data["true_skip_instrumentation"] = _instrumentation(
        skipped=quota, batches=total
    )
    data["run_config"]["target_skip_rate"] = target
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok, [f.to_dict() for f in report.errors]


def test_completed_rvd_requires_every_emitted_invariant(tmp_path):
    data = rvd_results()
    del data["policy_diagnostics"]["invariant_candidate_count_ok"]
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("policy_diagnostics.invariant_candidate_count_ok"
            in fields(report, "error"))


def test_matched_run_requires_trainer_invariants(tmp_path):
    data = random_results()
    instrumentation = data["true_skip_instrumentation"]
    del instrumentation["invariant_scheduler_policy_consistent"]
    del instrumentation["invariant_sched_le_opt"]
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert (
        "true_skip_instrumentation.invariant_scheduler_policy_consistent"
        in fields(report, "error")
    )


def test_malformed_source_count_is_structured_error(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["skip_source_counts"][
        "ordinary_repayment"
    ] = "not-an-int"
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("policy_diagnostics.skip_source_counts.ordinary_repayment"
            in fields(report, "error"))


def test_non_mapping_source_counts_is_structured_error(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["skip_source_counts"] = 7
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "policy_diagnostics.skip_source_counts" in fields(report, "error")


def test_debt_fields_must_agree(tmp_path):
    data = rvd_results()
    data["policy_diagnostics"]["deferred_pool_now"] = 1
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "policy_diagnostics.deferred_pool_now" in fields(report, "error")


def test_nonintegral_count_is_rejected(tmp_path):
    data = random_results()
    data["policy_diagnostics"]["skip_decisions"] = QUOTA - 0.5
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "policy_diagnostics.skip_decisions" in fields(report, "error")


def test_full_finetune_rejects_any_skip(tmp_path):
    data = full_finetune_results()
    data["true_skip_instrumentation"] = _instrumentation(skipped=1)
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("true_skip_instrumentation.skipped_backward_steps"
            in fields(report, "error"))


def test_cli_emits_one_json_document_and_uses_exit_status(tmp_path):
    valid_path = write_results(tmp_path, rvd_results())
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "validate_skip_policy_results.py"),
        str(valid_path),
    ]
    valid = subprocess.run(command, capture_output=True, text=True, check=False)
    assert valid.returncode == 0
    assert json.loads(valid.stdout)["valid_for_matched_budget"] is True

    invalid_data = rvd_results()
    invalid_data["interrupted"] = True
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid_data))
    invalid = subprocess.run(
        [*command[:-1], str(invalid_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert invalid.returncode == 1
    assert json.loads(invalid.stdout)["valid_for_matched_budget"] is False


def test_unmatched_incomplete_run_is_explicitly_not_matched_valid(tmp_path):
    data = random_results()
    data["run_config"]["matched_budget"] = False
    data["policy_diagnostics"].update({
        "decisions_seen": TOTAL - 10,
        "skip_decisions": QUOTA - 2,
        "realized_skip_rate": (QUOTA - 2) / (TOTAL - 10),
    })
    data["true_skip_instrumentation"] = _instrumentation(
        skipped=QUOTA - 2, batches=TOTAL - 10
    )
    report = vspr.validate_results(write_results(tmp_path, data))
    assert report.ok
    assert report.protocol_complete is False
    assert report.valid_for_matched_budget is False
    assert "quota_protocol_complete" in fields(report, "warning")


PHASE_STRAT_CONTROLS = ("fixed_phase_strat", "phase_strat_guarded")
N_PHASES = 4
PHASE_QUOTA = [8, 8, 7, 7]  # sums to QUOTA
PHASE_ELIGIBLE = [38, 37, 37, 38]  # sums to TOTAL - min_step
PHASE_BOUNDS = [50, 88, 125, 162, 200]
PHASE_WEIGHTS = [0.25, 0.25, 0.25, 0.25]
_REMOVE = object()
COMMON_PHASE_FIELDS = (
    "control",
    "controller_class",
    "policy_name",
    "target_skip_rate",
    "total_steps",
    "min_step",
    "policy_seed",
    "n_phases",
    "phase_weights",
    "phase_bounds",
    "phase_quota",
    "phase_eligible",
    "requested_quota",
)


def _phase_strat_controller(control):
    """Valid synthetic phase_strat_controller payload for a phase arm."""
    config = {
        "control": control,
        "controller_class": vspr.PHASE1_3_POLICY_CLASSES[control],
        "policy_name": control,
        "target_skip_rate": RATE,
        "total_steps": TOTAL,
        "min_step": 50,
        "policy_seed": 42,
        "n_phases": N_PHASES,
        "phase_weights": list(PHASE_WEIGHTS),
        "phase_bounds": list(PHASE_BOUNDS),
        "phase_quota": list(PHASE_QUOTA),
        "phase_eligible": list(PHASE_ELIGIBLE),
        "requested_quota": QUOTA,
    }
    if control == "phase_strat_guarded":
        config["risk_gamma"] = 0.5
        config["max_consecutive_skips"] = 3
        config["guarded_safety"] = {
            "use_rho_vg": True,
            "rho_veto_threshold": -0.2,
            "use_safety_horizon": True,
            "spike_factor": 3.0,
        }
    return config


def _phase_controller_inputs(control="fixed_phase_strat"):
    identity = _phase1_3_identity(control)
    identity["phase_strat_controller"] = _phase_strat_controller(control)
    controller_config = _phase1_3_controller_config(
        control,
        vspr.PHASE1_3_POLICY_CLASSES[control],
        True,
    )
    controller_config["phase_strat_controller"] = _phase_strat_controller(
        control
    )
    data = {
        "ablation": control,
        "identity_inputs": identity,
        "controller_config": controller_config,
        "fingerprint": vspr.build_scientific_fingerprint(identity),
    }
    run_config = _run_config(
        control=control,
        controller_config=json.loads(json.dumps(controller_config)),
    )
    run_config["phase_strat_controller"] = _phase_strat_controller(control)
    report = vspr.ValidationReport(path="synthetic")
    return report, data, run_config


def _phase_copies(data, run_config):
    return [
        data["identity_inputs"]["phase_strat_controller"],
        data["controller_config"]["phase_strat_controller"],
        run_config["controller_config"]["phase_strat_controller"],
        run_config["phase_strat_controller"],
    ]


def _attach_stray_phase_config(data, run_config):
    stray = _phase_strat_controller("fixed_phase_strat")
    data["identity_inputs"]["phase_strat_controller"] = json.loads(
        json.dumps(stray)
    )
    data["fingerprint"] = vspr.build_scientific_fingerprint(
        data["identity_inputs"]
    )
    data["controller_config"]["phase_strat_controller"] = json.loads(
        json.dumps(stray)
    )
    run_config["controller_config"]["phase_strat_controller"] = json.loads(
        json.dumps(stray)
    )
    run_config["phase_strat_controller"] = json.loads(json.dumps(stray))


def _assert_phase_field_error(report, key):
    target = "phase_strat_controller." + key
    error_fields = fields(report, "error")
    assert any(target in f for f in error_fields), (target, error_fields)


@pytest.mark.parametrize("control", PHASE_STRAT_CONTROLS)
def test_phase_controller_valid_configs_have_no_errors(control):
    report, data, run_config = _phase_controller_inputs(control)
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    assert not report.errors, [f.to_dict() for f in report.errors]


@pytest.mark.parametrize(
    "mutate, field",
    [
        (
            lambda d, rc: d["identity_inputs"].pop("phase_strat_controller"),
            "identity_inputs.phase_strat_controller",
        ),
        (
            lambda d, rc: d["identity_inputs"].__setitem__(
                "phase_strat_controller", 7
            ),
            "identity_inputs.phase_strat_controller",
        ),
        (
            lambda d, rc: d["controller_config"].pop(
                "phase_strat_controller"
            ),
            "controller_config.phase_strat_controller",
        ),
        (
            lambda d, rc: d["controller_config"].__setitem__(
                "phase_strat_controller", "bad"
            ),
            "controller_config.phase_strat_controller",
        ),
        (
            lambda d, rc: rc.pop("phase_strat_controller"),
            "run_config.phase_strat_controller",
        ),
        (
            lambda d, rc: rc.__setitem__("phase_strat_controller", []),
            "run_config.phase_strat_controller",
        ),
    ],
)
def test_phase_controller_missing_or_malformed_sources(mutate, field):
    report, data, run_config = _phase_controller_inputs()
    mutate(data, run_config)
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    assert field in fields(report, "error")


def test_phase_controller_copies_must_match():
    report, data, run_config = _phase_controller_inputs()
    run_config["phase_strat_controller"]["phase_weights"] = [
        0.4,
        0.2,
        0.2,
        0.2,
    ]
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    assert "phase_strat_controller_equality" in fields(report, "error")


@pytest.mark.parametrize("key", COMMON_PHASE_FIELDS)
def test_phase_controller_missing_common_field_is_rejected(key):
    report, data, run_config = _phase_controller_inputs()
    for entry in _phase_copies(data, run_config):
        entry.pop(key)
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("control", "full_finetune"),
        ("policy_name", "wrong_policy"),
        ("controller_class", "WrongController"),
    ],
)
def test_phase_controller_wrong_identity_strings(key, value):
    report, data, run_config = _phase_controller_inputs()
    for entry in _phase_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("target_skip_rate", RATE + 0.05),
        ("total_steps", TOTAL + 1),
        ("policy_seed", 43),
    ],
)
def test_phase_controller_identity_disagreement(key, value):
    report, data, run_config = _phase_controller_inputs()
    for entry in _phase_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("n_phases", "4"),
        ("n_phases", True),
        ("n_phases", 4.5),
        ("n_phases", 0),
        ("total_steps", True),
        ("min_step", 1.5),
        ("policy_seed", 1.5),
        ("requested_quota", True),
    ],
)
def test_phase_controller_integer_fields_are_validated(key, value):
    report, data, run_config = _phase_controller_inputs()
    for entry in _phase_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("phase_quota", "not-a-list"),
        ("phase_quota", [8, 8, 14]),
        ("phase_quota", [8, 8, 7, -7]),
        ("phase_quota", [8, 8, 7, 6.5]),
        ("phase_eligible", {"phase": 200}),
        ("phase_eligible", [75, 75]),
        ("phase_eligible", [38, 37, 37, -38]),
        ("phase_eligible", [38, 37, 37, 37.5]),
        ("phase_bounds", "not-a-list"),
        ("phase_bounds", [50, 100, 150, 200]),
        ("phase_weights", 7),
        ("phase_weights", [0.5, 0.5]),
        ("phase_weights", [0.25, 0.25, 0.25, -0.25]),
        ("phase_weights", [0.25, 0.25, 0.25, float("nan")]),
        ("phase_weights", [0.25, 0.25, 0.25, float("inf")]),
    ],
)
def test_phase_controller_list_fields_are_validated(key, value):
    report, data, run_config = _phase_controller_inputs()
    for entry in _phase_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("guarded_safety", {"use_rho_vg": True}),
        ("risk_gamma", 0.5),
        ("max_consecutive_skips", 3),
    ],
)
def test_fixed_phase_rejects_guarded_only_fields(key, value):
    report, data, run_config = _phase_controller_inputs("fixed_phase_strat")
    for entry in _phase_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "path, value, key",
    [
        (("risk_gamma",), _REMOVE, "risk_gamma"),
        (("risk_gamma",), "high", "risk_gamma"),
        (("risk_gamma",), -0.5, "risk_gamma"),
        (("risk_gamma",), float("nan"), "risk_gamma"),
        (("max_consecutive_skips",), _REMOVE, "max_consecutive_skips"),
        (("max_consecutive_skips",), 0, "max_consecutive_skips"),
        (("max_consecutive_skips",), True, "max_consecutive_skips"),
        (("max_consecutive_skips",), 2.5, "max_consecutive_skips"),
        (("guarded_safety",), _REMOVE, "guarded_safety"),
        (("guarded_safety",), 7, "guarded_safety"),
        (
            ("guarded_safety", "use_rho_vg"),
            "yes",
            "guarded_safety.use_rho_vg",
        ),
        (
            ("guarded_safety", "use_safety_horizon"),
            1,
            "guarded_safety.use_safety_horizon",
        ),
        (
            ("guarded_safety", "rho_veto_threshold"),
            float("nan"),
            "guarded_safety.rho_veto_threshold",
        ),
        (
            ("guarded_safety", "spike_factor"),
            -1.0,
            "guarded_safety.spike_factor",
        ),
        (
            ("guarded_safety", "spike_factor"),
            float("inf"),
            "guarded_safety.spike_factor",
        ),
    ],
)
def test_guarded_phase_controller_field_errors(path, value, key):
    report, data, run_config = _phase_controller_inputs("phase_strat_guarded")
    for entry in _phase_copies(data, run_config):
        target = entry
        for part in path[:-1]:
            target = target[part]
        if value is _REMOVE:
            target.pop(path[-1])
        else:
            target[path[-1]] = value
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    _assert_phase_field_error(report, key)


@pytest.mark.parametrize(
    "control",
    sorted(set(vspr.PHASE1_3_CONTROLS) - set(PHASE_STRAT_CONTROLS)),
)
def test_non_phase_controls_reject_stray_phase_config(control):
    report, data, run_config = _phase1_3_base_inputs(control)
    _attach_stray_phase_config(data, run_config)
    vspr._check_phase1_3_phase_controller(report, data, run_config)
    error_fields = fields(report, "error")
    assert any(
        "phase_strat_controller" in field for field in error_fields
    ), error_fields


def test_stray_phase_config_fails_validate_results(tmp_path):
    data = full_finetune_results()
    _attach_stray_phase_config(data, data["run_config"])
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    error_fields = fields(report, "error")
    assert any(
        "phase_strat_controller" in field for field in error_fields
    ), error_fields


def _phase1_3_base_inputs(control="full_finetune"):
    identity = _phase1_3_identity(control)
    controller_config = _phase1_3_controller_config(
        control,
        vspr.PHASE1_3_POLICY_CLASSES[control],
        control != "full_finetune",
    )
    data = {
        "ablation": control,
        "identity_inputs": identity,
        "controller_config": controller_config,
        "fingerprint": vspr.build_scientific_fingerprint(identity),
    }
    run_config = _run_config(
        control=control,
        controller_config=json.loads(json.dumps(controller_config)),
    )
    report = vspr.ValidationReport(path="synthetic")
    return report, data, run_config


@pytest.mark.parametrize("control", sorted(vspr.PHASE1_3_CONTROLS))
def test_phase1_3_valid_base_identity_has_no_errors(control):
    report, data, run_config = _phase1_3_base_inputs(control)
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert not report.errors, [f.to_dict() for f in report.errors]


def test_phase1_3_noncanonical_control_returns_without_findings():
    report, data, run_config = _phase1_3_base_inputs()
    for key in ("control", "arm"):
        data["controller_config"][key] = "not_a_phase1_3_control"
        run_config["controller_config"][key] = "not_a_phase1_3_control"
    data["ablation"] = "not_a_phase1_3_control"
    run_config["control"] = "not_a_phase1_3_control"
    data["identity_inputs"]["control"] = "not_a_phase1_3_control"
    data["fingerprint"] = vspr.build_scientific_fingerprint(
        data["identity_inputs"]
    )
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert not report.findings


@pytest.mark.parametrize(
    "mutate, field",
    [
        (lambda d, rc: d.pop("identity_inputs"), "identity_inputs"),
        (lambda d, rc: d.__setitem__("identity_inputs", 7), "identity_inputs"),
        (lambda d, rc: d.pop("controller_config"), "controller_config"),
        (
            lambda d, rc: d.__setitem__("controller_config", "bad"),
            "controller_config",
        ),
        (
            lambda d, rc: rc.pop("controller_config"),
            "run_config.controller_config",
        ),
        (
            lambda d, rc: rc.__setitem__("controller_config", []),
            "run_config.controller_config",
        ),
    ],
)
def test_phase1_3_missing_or_malformed_sections(mutate, field):
    report, data, run_config = _phase1_3_base_inputs()
    mutate(data, run_config)
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert field in fields(report, "error")


@pytest.mark.parametrize(
    "location",
    ["ablation", "identity_control", "controller_control", "controller_arm"],
)
def test_phase1_3_control_disagreement(location):
    report, data, run_config = _phase1_3_base_inputs("full_finetune")
    other = next(
        c for c in sorted(vspr.PHASE1_3_CONTROLS) if c != "full_finetune"
    )
    if location == "ablation":
        data["ablation"] = other
    elif location == "identity_control":
        data["identity_inputs"]["control"] = other
        data["fingerprint"] = vspr.build_scientific_fingerprint(
            data["identity_inputs"]
        )
    elif location == "controller_control":
        data["controller_config"]["control"] = other
        run_config["controller_config"]["control"] = other
    else:
        data["controller_config"]["arm"] = other
        run_config["controller_config"]["arm"] = other
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert "phase1_3_control_identity" in fields(report, "error")


@pytest.mark.parametrize(
    "key, value, field",
    [
        ("arm_alias_of", "full_finetune", "controller_config.arm_alias_of"),
        ("policy_class", "WrongPolicy", "controller_config.policy_class"),
        ("is_skipping_arm", True, "controller_config.is_skipping_arm"),
    ],
)
def test_phase1_3_controller_field_errors(key, value, field):
    report, data, run_config = _phase1_3_base_inputs("full_finetune")
    data["controller_config"][key] = value
    run_config["controller_config"][key] = value
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert field in fields(report, "error")


def test_phase1_3_controller_copies_must_match():
    report, data, run_config = _phase1_3_base_inputs()
    run_config["controller_config"]["min_step"] = 51
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert "controller_config_equality" in fields(report, "error")


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda d: d.pop("fingerprint"),
        lambda d: d.__setitem__("fingerprint", 7),
        lambda d: d.__setitem__("fingerprint", "0" * 64),
        lambda d: d["identity_inputs"].__setitem__("training_seed", 43),
    ],
)
def test_phase1_3_fingerprint_errors(corrupt):
    report, data, run_config = _phase1_3_base_inputs()
    corrupt(data)
    vspr._check_phase1_3_base_identity(report, data, run_config)
    assert "fingerprint" in fields(report, "error")


def test_phase1_3_tampered_fingerprint_fails_validate_results(tmp_path):
    data = full_finetune_results()
    data["fingerprint"] = "0" * len(data["fingerprint"])
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "fingerprint" in fields(report, "error")


LER_CONTROLS = ("ler_guided_stratified", "ler_guided_stratified_safe")
LER_SAFE_CONTROL = "ler_guided_stratified_safe"
NON_LER_CONTROLS = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
)
LER_COMMON_FIELDS = (
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
)
LER_SAFE_ONLY_FIELDS = (
    "use_rho_vg_safety",
    "rho_veto_threshold",
    "use_loss_spike_safety",
    "loss_spike_factor",
    "loss_spike_window",
)


def _ler_guided_controller(control):
    """Valid synthetic ler_guided_controller payload for a LER arm."""
    config = {
        "control": control,
        "policy_class": vspr.PHASE1_3_POLICY_CLASSES[control],
        "policy_name": control,
        "target_skip_rate": RATE,
        "total_steps": TOTAL,
        "min_step": 50,
        "policy_seed": 42,
        "n_phases": N_PHASES,
        "phase_weights": list(PHASE_WEIGHTS),
        "max_consecutive_skips": 3,
        "probe_interval": 10,
        "min_ler_observations": 5,
        "ler_guidance_strength": 0.5,
        "required_tracker_mode": "sampled_lagged",
        "required_tracker_timing": "post_decision_after_backward",
        "safety_enabled": control == LER_SAFE_CONTROL,
    }
    if control == LER_SAFE_CONTROL:
        config["use_rho_vg_safety"] = True
        config["rho_veto_threshold"] = -0.2
        config["use_loss_spike_safety"] = True
        config["loss_spike_factor"] = 3.0
        config["loss_spike_window"] = 10
    return config


def _ler_controller_inputs(control="ler_guided_stratified"):
    identity = _phase1_3_identity(control)
    identity["ler_guided_controller"] = _ler_guided_controller(control)
    controller_config = _phase1_3_controller_config(
        control,
        vspr.PHASE1_3_POLICY_CLASSES[control],
        True,
    )
    controller_config["ler_guided_controller"] = _ler_guided_controller(
        control
    )
    data = {
        "ablation": control,
        "identity_inputs": identity,
        "controller_config": controller_config,
        "fingerprint": vspr.build_scientific_fingerprint(identity),
    }
    run_config = _run_config(
        control=control,
        controller_config=json.loads(json.dumps(controller_config)),
    )
    run_config["ler_guided_controller"] = _ler_guided_controller(control)
    report = vspr.ValidationReport(path="synthetic")
    return report, data, run_config


def _ler_copies(data, run_config):
    return [
        data["identity_inputs"]["ler_guided_controller"],
        data["controller_config"]["ler_guided_controller"],
        run_config["controller_config"]["ler_guided_controller"],
        run_config["ler_guided_controller"],
    ]


def _attach_stray_ler_config(data, run_config):
    stray = _ler_guided_controller("ler_guided_stratified")
    data["identity_inputs"]["ler_guided_controller"] = json.loads(
        json.dumps(stray)
    )
    data["fingerprint"] = vspr.build_scientific_fingerprint(
        data["identity_inputs"]
    )
    data["controller_config"]["ler_guided_controller"] = json.loads(
        json.dumps(stray)
    )
    run_config["controller_config"]["ler_guided_controller"] = json.loads(
        json.dumps(stray)
    )
    run_config["ler_guided_controller"] = json.loads(json.dumps(stray))


def _assert_ler_field_error(report, key):
    target = "ler_guided_controller." + key
    error_fields = fields(report, "error")
    assert any(target in f for f in error_fields), (target, error_fields)


@pytest.mark.parametrize("control", LER_CONTROLS)
def test_ler_controller_valid_configs_have_no_errors(control):
    report, data, run_config = _ler_controller_inputs(control)
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    assert not report.errors, [f.to_dict() for f in report.errors]


@pytest.mark.parametrize(
    "mutate, field",
    [
        (
            lambda d, rc: d["identity_inputs"].pop("ler_guided_controller"),
            "identity_inputs.ler_guided_controller",
        ),
        (
            lambda d, rc: d["identity_inputs"].__setitem__(
                "ler_guided_controller", 7
            ),
            "identity_inputs.ler_guided_controller",
        ),
        (
            lambda d, rc: d["controller_config"].pop(
                "ler_guided_controller"
            ),
            "controller_config.ler_guided_controller",
        ),
        (
            lambda d, rc: d["controller_config"].__setitem__(
                "ler_guided_controller", "bad"
            ),
            "controller_config.ler_guided_controller",
        ),
        (
            lambda d, rc: rc.pop("ler_guided_controller"),
            "run_config.ler_guided_controller",
        ),
        (
            lambda d, rc: rc.__setitem__("ler_guided_controller", []),
            "run_config.ler_guided_controller",
        ),
    ],
)
def test_ler_controller_missing_or_malformed_sources(mutate, field):
    report, data, run_config = _ler_controller_inputs()
    mutate(data, run_config)
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    assert field in fields(report, "error")


def test_ler_controller_copies_must_match():
    report, data, run_config = _ler_controller_inputs()
    run_config["ler_guided_controller"]["probe_interval"] = 11
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    assert "ler_guided_controller_equality" in fields(report, "error")


@pytest.mark.parametrize("key", LER_COMMON_FIELDS)
def test_ler_controller_missing_common_field_is_rejected(key):
    report, data, run_config = _ler_controller_inputs()
    for entry in _ler_copies(data, run_config):
        entry.pop(key)
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("control", "full_finetune"),
        ("policy_name", "wrong_policy"),
        ("policy_class", "WrongPolicy"),
    ],
)
def test_ler_controller_wrong_identity_strings(key, value):
    report, data, run_config = _ler_controller_inputs()
    for entry in _ler_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("target_skip_rate", RATE + 0.05),
        ("total_steps", TOTAL + 1),
        ("policy_seed", 43),
    ],
)
def test_ler_controller_identity_disagreement(key, value):
    report, data, run_config = _ler_controller_inputs()
    for entry in _ler_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("total_steps", "100"),
        ("total_steps", True),
        ("total_steps", 100.5),
        ("total_steps", 0),
        ("min_step", "50"),
        ("min_step", True),
        ("min_step", 50.5),
        ("min_step", -1),
        ("policy_seed", "42"),
        ("policy_seed", True),
        ("policy_seed", 42.5),
        ("n_phases", "4"),
        ("n_phases", True),
        ("n_phases", 4.5),
        ("n_phases", 0),
        ("max_consecutive_skips", "3"),
        ("max_consecutive_skips", True),
        ("max_consecutive_skips", 2.5),
        ("max_consecutive_skips", 0),
        ("probe_interval", "10"),
        ("probe_interval", True),
        ("probe_interval", 10.5),
        ("probe_interval", 0),
        ("min_ler_observations", "5"),
        ("min_ler_observations", True),
        ("min_ler_observations", 5.5),
        ("min_ler_observations", 0),
    ],
)
def test_ler_controller_integer_fields_are_validated(key, value):
    report, data, run_config = _ler_controller_inputs()
    if key in {"total_steps", "policy_seed"}:
        data["identity_inputs"][key] = value
    for entry in _ler_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "value",
    [
        7,
        [0.5, 0.5],
        [0.25, 0.25, 0.25, -0.25],
        [0.25, 0.25, 0.25, float("nan")],
        [0.25, 0.25, 0.25, float("inf")],
    ],
)
def test_ler_controller_phase_weights_are_validated(value):
    report, data, run_config = _ler_controller_inputs()
    for entry in _ler_copies(data, run_config):
        entry["phase_weights"] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, "phase_weights")


@pytest.mark.parametrize(
    "key, value",
    [
        ("ler_guidance_strength", "high"),
        ("ler_guidance_strength", -0.5),
        ("ler_guidance_strength", float("nan")),
        ("ler_guidance_strength", float("inf")),
        ("required_tracker_mode", "dense_immediate"),
        ("required_tracker_mode", 7),
        ("required_tracker_timing", "pre_decision_before_backward"),
        ("required_tracker_timing", None),
    ],
)
def test_ler_controller_guidance_and_tracker_are_validated(key, value):
    report, data, run_config = _ler_controller_inputs()
    for entry in _ler_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "control, value",
    [
        ("ler_guided_stratified", "no"),
        ("ler_guided_stratified", 1),
        ("ler_guided_stratified", True),
        ("ler_guided_stratified_safe", 0),
        ("ler_guided_stratified_safe", False),
    ],
)
def test_ler_controller_safety_enabled_is_validated(control, value):
    report, data, run_config = _ler_controller_inputs(control)
    for entry in _ler_copies(data, run_config):
        entry["safety_enabled"] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, "safety_enabled")


@pytest.mark.parametrize("key", LER_SAFE_ONLY_FIELDS)
def test_non_safe_ler_arm_rejects_safety_only_fields(key):
    report, data, run_config = _ler_controller_inputs("ler_guided_stratified")
    value = _ler_guided_controller(LER_SAFE_CONTROL)[key]
    for entry in _ler_copies(data, run_config):
        entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize(
    "key, value",
    [
        ("use_rho_vg_safety", _REMOVE),
        ("use_rho_vg_safety", False),
        ("use_rho_vg_safety", "yes"),
        ("use_rho_vg_safety", 1),
        ("use_loss_spike_safety", _REMOVE),
        ("use_loss_spike_safety", False),
        ("use_loss_spike_safety", "yes"),
        ("use_loss_spike_safety", 1),
        ("rho_veto_threshold", _REMOVE),
        ("rho_veto_threshold", "low"),
        ("rho_veto_threshold", True),
        ("rho_veto_threshold", float("nan")),
        ("rho_veto_threshold", float("inf")),
        ("loss_spike_factor", _REMOVE),
        ("loss_spike_factor", "high"),
        ("loss_spike_factor", True),
        ("loss_spike_factor", -1.0),
        ("loss_spike_factor", float("nan")),
        ("loss_spike_factor", float("inf")),
        ("loss_spike_window", _REMOVE),
        ("loss_spike_window", "10"),
        ("loss_spike_window", True),
        ("loss_spike_window", 2.5),
        ("loss_spike_window", 0),
    ],
)
def test_safe_ler_arm_safety_field_errors(key, value):
    report, data, run_config = _ler_controller_inputs(LER_SAFE_CONTROL)
    for entry in _ler_copies(data, run_config):
        if value is _REMOVE:
            entry.pop(key)
        else:
            entry[key] = value
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    _assert_ler_field_error(report, key)


@pytest.mark.parametrize("control", NON_LER_CONTROLS)
def test_non_ler_controls_reject_stray_ler_config(control):
    report, data, run_config = _phase1_3_base_inputs(control)
    _attach_stray_ler_config(data, run_config)
    vspr._check_phase1_3_ler_controller(report, data, run_config)
    error_fields = fields(report, "error")
    assert any(
        "ler_guided_controller" in field for field in error_fields
    ), error_fields


def test_stray_ler_config_fails_validate_results(tmp_path):
    data = full_finetune_results()
    _attach_stray_ler_config(data, data["run_config"])
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    error_fields = fields(report, "error")
    assert any(
        "ler_guided_controller" in field for field in error_fields
    ), error_fields


OD_CONTROLS = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
)
OD_OFF_CONTROLS = OD_CONTROLS[:3]
OD_SAMPLED_CONTROLS = OD_CONTROLS[3:]
OD_CONFIG_FIELDS = (
    "requested_mode",
    "mode",
    "enabled",
    "timing",
    "parameter_sample_size",
    "update_interval",
    "reason",
    "sample_seed",
)
OD_RUNTIME_ONLY_FIELDS = (
    "parameter_sample_size_realized",
    "update_attempts",
    "update_successes",
    "n_updates",
    "n_decisions",
    "last_update_decision",
    "observation_age_decisions",
)
OD_RUNTIME_FIELDS = OD_CONFIG_FIELDS + OD_RUNTIME_ONLY_FIELDS
OD_COUNTER_FIELDS = (
    "parameter_sample_size_realized",
    "update_attempts",
    "update_successes",
    "n_updates",
    "n_decisions",
)
OD_OPTIONAL_RUNTIME_FIELDS = (
    "last_update_decision",
    "observation_age_decisions",
)


def _online_diagnostics_payload(control):
    """Valid online-diagnostics config and runtime payloads for any arm."""
    if control in OD_OFF_CONTROLS:
        config = {
            "requested_mode": "auto",
            "mode": "off",
            "enabled": False,
            "timing": "none",
            "parameter_sample_size": 0,
            "update_interval": 0,
            "reason": "arm does not use online LER diagnostics",
            "sample_seed": None,
        }
        runtime = {
            **config,
            "parameter_sample_size_realized": 0,
            "update_attempts": 0,
            "update_successes": 0,
            "n_updates": 0,
            "n_decisions": 0,
            "last_update_decision": None,
            "observation_age_decisions": None,
        }
    else:
        config = {
            "requested_mode": "auto",
            "mode": "sampled_lagged",
            "enabled": True,
            "timing": "post_decision_after_backward",
            "parameter_sample_size": 4096,
            "update_interval": 1,
            "reason": "arm uses online LER diagnostics",
            "sample_seed": 42,
        }
        runtime = {
            **config,
            "parameter_sample_size_realized": 4096,
            "update_attempts": 150,
            "update_successes": 150,
            "n_updates": 150,
            "n_decisions": 150,
            "last_update_decision": 149,
            "observation_age_decisions": 1,
        }
    return config, runtime


def _online_diag_inputs(control="full_finetune"):
    report, data, run_config = _phase1_3_base_inputs(control)
    config, runtime = _online_diagnostics_payload(control)
    data["identity_inputs"]["online_diagnostics"] = json.loads(
        json.dumps(config)
    )
    data["fingerprint"] = vspr.build_scientific_fingerprint(
        data["identity_inputs"]
    )
    data["controller_config"]["online_diagnostics"] = json.loads(
        json.dumps(config)
    )
    run_config["controller_config"]["online_diagnostics"] = json.loads(
        json.dumps(config)
    )
    run_config["online_diagnostics"] = json.loads(json.dumps(config))
    data["online_diagnostics"] = json.loads(json.dumps(runtime))
    return report, data, run_config


def _online_diag_copies(data, run_config):
    return [
        data["identity_inputs"]["online_diagnostics"],
        data["controller_config"]["online_diagnostics"],
        run_config["controller_config"]["online_diagnostics"],
        run_config["online_diagnostics"],
    ]


def _set_online_diag_everywhere(data, run_config, key, value):
    for entry in _online_diag_copies(data, run_config):
        entry[key] = value
    data["online_diagnostics"][key] = value


@pytest.mark.parametrize("control", OD_CONTROLS)
def test_online_diagnostics_valid_payloads_have_no_errors(control):
    report, data, run_config = _online_diag_inputs(control)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert not report.errors, [f.to_dict() for f in report.errors]


def test_online_diagnostics_sampled_zero_activity_is_valid():
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    runtime = data["online_diagnostics"]
    runtime["update_attempts"] = 0
    runtime["update_successes"] = 0
    runtime["n_updates"] = 0
    runtime["n_decisions"] = 0
    runtime["last_update_decision"] = None
    runtime["observation_age_decisions"] = None
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert not report.errors, [f.to_dict() for f in report.errors]


@pytest.mark.parametrize(
    "mutate, field",
    [
        (
            lambda d, rc: d["identity_inputs"].pop("online_diagnostics"),
            "identity_inputs.online_diagnostics",
        ),
        (
            lambda d, rc: d["identity_inputs"].__setitem__(
                "online_diagnostics", 7
            ),
            "identity_inputs.online_diagnostics",
        ),
        (
            lambda d, rc: d["controller_config"].pop("online_diagnostics"),
            "controller_config.online_diagnostics",
        ),
        (
            lambda d, rc: d["controller_config"].__setitem__(
                "online_diagnostics", "bad"
            ),
            "controller_config.online_diagnostics",
        ),
        (
            lambda d, rc: rc.pop("online_diagnostics"),
            "run_config.online_diagnostics",
        ),
        (
            lambda d, rc: rc.__setitem__("online_diagnostics", []),
            "run_config.online_diagnostics",
        ),
    ],
)
def test_online_diagnostics_missing_or_malformed_sources(mutate, field):
    report, data, run_config = _online_diag_inputs()
    mutate(data, run_config)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert field in fields(report, "error")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("online_diagnostics"),
        lambda d: d.__setitem__("online_diagnostics", 7),
    ],
)
def test_online_diagnostics_missing_or_malformed_runtime(mutate):
    report, data, run_config = _online_diag_inputs()
    mutate(data)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics" in fields(report, "error")


@pytest.mark.parametrize(
    "key, value",
    [
        ("reason", "a different reason"),
        ("update_interval", 0.0),
        ("enabled", 0),
    ],
)
def test_online_diagnostics_copies_must_match(key, value):
    report, data, run_config = _online_diag_inputs()
    run_config["online_diagnostics"][key] = value
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics_config_equality" in fields(report, "error")


@pytest.mark.parametrize("key", OD_CONFIG_FIELDS)
def test_online_diagnostics_missing_config_field_is_rejected(key):
    report, data, run_config = _online_diag_inputs()
    for entry in _online_diag_copies(data, run_config):
        entry.pop(key)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics." + key
        in fields(report, "error")
    )


@pytest.mark.parametrize("key", ("requested_mode", "reason"))
@pytest.mark.parametrize("value", ("", 7))
def test_online_diagnostics_string_config_fields_are_validated(key, value):
    report, data, run_config = _online_diag_inputs()
    _set_online_diag_everywhere(data, run_config, key, value)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics." + key
        in fields(report, "error")
    )


@pytest.mark.parametrize("control", OD_CONTROLS)
@pytest.mark.parametrize(
    "key, off_value, sampled_value",
    [
        ("mode", "sampled_lagged", "off"),
        ("enabled", True, False),
        ("enabled", "yes", 1),
        ("timing", "post_decision_after_backward", "none"),
    ],
)
def test_online_diagnostics_arm_config_values_are_validated(
    control, key, off_value, sampled_value
):
    report, data, run_config = _online_diag_inputs(control)
    value = off_value if control in OD_OFF_CONTROLS else sampled_value
    _set_online_diag_everywhere(data, run_config, key, value)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics." + key
        in fields(report, "error")
    )


@pytest.mark.parametrize("key", ("parameter_sample_size", "update_interval"))
@pytest.mark.parametrize(
    "control, value",
    [
        ("full_finetune", "0"),
        ("full_finetune", True),
        ("full_finetune", 0.0),
        ("full_finetune", 0.5),
        ("full_finetune", 1),
        ("full_finetune", -1),
        ("ler_guided_stratified", "1"),
        ("ler_guided_stratified", True),
        ("ler_guided_stratified", 1.0),
        ("ler_guided_stratified", 1.5),
        ("ler_guided_stratified", 0),
        ("ler_guided_stratified", -1),
    ],
)
def test_online_diagnostics_integer_config_fields_are_validated(
    key, control, value
):
    report, data, run_config = _online_diag_inputs(control)
    _set_online_diag_everywhere(data, run_config, key, value)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics." + key
        in fields(report, "error")
    )


@pytest.mark.parametrize("value", (0, 42, "42", True, False, 42.0, 42.5))
def test_online_diagnostics_off_sample_seed_must_be_none(value):
    report, data, run_config = _online_diag_inputs("full_finetune")
    _set_online_diag_everywhere(data, run_config, "sample_seed", value)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics.sample_seed"
        in fields(report, "error")
    )


@pytest.mark.parametrize("value", (None, "42", True, False, 42.0, 42.5))
def test_online_diagnostics_sampled_sample_seed_is_validated(value):
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    _set_online_diag_everywhere(data, run_config, "sample_seed", value)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert (
        "identity_inputs.online_diagnostics.sample_seed"
        in fields(report, "error")
    )


@pytest.mark.parametrize("key", OD_RUNTIME_FIELDS)
def test_online_diagnostics_missing_runtime_field_is_rejected(key):
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    data["online_diagnostics"].pop(key)
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


@pytest.mark.parametrize(
    "control, key, value",
    [
        ("full_finetune", "requested_mode", "manual"),
        ("full_finetune", "mode", "sampled_lagged"),
        ("full_finetune", "enabled", 0),
        ("full_finetune", "timing", "post_decision_after_backward"),
        ("full_finetune", "parameter_sample_size", 0.0),
        ("full_finetune", "update_interval", 0.0),
        ("full_finetune", "reason", "a different reason"),
        ("ler_guided_stratified", "enabled", 1),
        ("ler_guided_stratified", "parameter_sample_size", 4096.0),
        ("ler_guided_stratified", "update_interval", True),
        ("ler_guided_stratified", "sample_seed", 42.0),
        ("ler_guided_stratified", "sample_seed", 43),
    ],
)
def test_online_diagnostics_runtime_must_copy_config(control, key, value):
    report, data, run_config = _online_diag_inputs(control)
    data["online_diagnostics"][key] = value
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


@pytest.mark.parametrize("key", OD_COUNTER_FIELDS)
@pytest.mark.parametrize("value", (-1, "1", True, 1.0, 1.5))
def test_online_diagnostics_runtime_counters_are_validated(key, value):
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    data["online_diagnostics"][key] = value
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


@pytest.mark.parametrize("key", OD_OPTIONAL_RUNTIME_FIELDS)
@pytest.mark.parametrize("value", (None, 0, 7))
def test_online_diagnostics_optional_fields_accept_valid_values(key, value):
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    data["online_diagnostics"][key] = value
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key not in fields(report, "error")


@pytest.mark.parametrize("key", OD_OPTIONAL_RUNTIME_FIELDS)
@pytest.mark.parametrize("value", (-1, "0", True, 1.0, 1.5))
def test_online_diagnostics_optional_fields_reject_invalid_values(key, value):
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    data["online_diagnostics"][key] = value
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


@pytest.mark.parametrize("key", OD_COUNTER_FIELDS)
def test_online_diagnostics_off_counters_must_be_zero(key):
    report, data, run_config = _online_diag_inputs("full_finetune")
    data["online_diagnostics"][key] = 1
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


@pytest.mark.parametrize("key", OD_OPTIONAL_RUNTIME_FIELDS)
def test_online_diagnostics_off_optional_fields_must_be_none(key):
    report, data, run_config = _online_diag_inputs("full_finetune")
    data["online_diagnostics"][key] = 0
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics." + key in fields(report, "error")


def test_online_diagnostics_successes_cannot_exceed_attempts():
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    runtime = data["online_diagnostics"]
    runtime["update_attempts"] = 10
    runtime["update_successes"] = 11
    runtime["n_updates"] = 11
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics.update_successes" in fields(report, "error")


def test_online_diagnostics_n_updates_must_equal_successes():
    report, data, run_config = _online_diag_inputs("ler_guided_stratified")
    runtime = data["online_diagnostics"]
    runtime["update_attempts"] = 150
    runtime["update_successes"] = 150
    runtime["n_updates"] = 149
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert "online_diagnostics.n_updates" in fields(report, "error")


def test_online_diagnostics_noncanonical_control_returns_without_findings():
    report, data, run_config = _online_diag_inputs("full_finetune")
    for key in ("control", "arm"):
        data["controller_config"][key] = "random_veto_deferral"
        run_config["controller_config"][key] = "random_veto_deferral"
    data["ablation"] = "random_veto_deferral"
    run_config["control"] = "random_veto_deferral"
    data["identity_inputs"]["control"] = "random_veto_deferral"
    data["fingerprint"] = vspr.build_scientific_fingerprint(
        data["identity_inputs"]
    )
    data["online_diagnostics"]["update_attempts"] = 1
    vspr._check_phase1_3_online_diagnostics(report, data, run_config)
    assert not report.findings


def test_online_diagnostics_valid_full_finetune_passes_validate_results(
    tmp_path,
):
    report = vspr.validate_results(
        write_results(tmp_path, full_finetune_results())
    )
    assert report.ok, [f.to_dict() for f in report.errors]


def test_online_diagnostics_nonzero_attempts_fails_validate_results(tmp_path):
    data = full_finetune_results()
    assert data["online_diagnostics"]["update_attempts"] == 0
    data["online_diagnostics"]["update_attempts"] = 1
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert "online_diagnostics.update_attempts" in fields(report, "error")


FIXED_BUDGET_CONTROLS = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
)
FIXED_BUDGET_SKIPPING_CONTROLS = FIXED_BUDGET_CONTROLS[1:]


def _phase1_3_budget_inputs(control):
    """Valid A9 fixed-budget inputs for one canonical Phase 1.3 control."""
    report, data, run_config = _phase1_3_base_inputs(control)
    controller_config = data["controller_config"]
    if control == "full_finetune":
        controller_config["requested_quota"] = None
        controller_config["runtime_quota_total_steps"] = None
        diag = {}
        instr = _instrumentation(skipped=0)
    else:
        controller_config["requested_quota"] = QUOTA
        controller_config["runtime_quota_total_steps"] = TOTAL
        diag = {
            "target_skip_rate": RATE,
            "quota_total_steps": TOTAL,
            "quota_size": QUOTA,
            "decisions_seen": TOTAL,
            "skip_decisions": QUOTA,
        }
        if control == "exact_random":
            diag["seed"] = 42
            diag["requested_quota"] = QUOTA
        else:
            diag["quota_exact"] = True
        instr = _instrumentation(skipped=QUOTA)
    run_config["controller_config"] = json.loads(
        json.dumps(controller_config)
    )
    run_config["rvd_policy_seed"] = 42
    data["skip_update_mode"] = "freeze"
    data["forward_calls"] = instr["forward_calls"]
    data["backward_calls"] = instr["backward_calls"]
    data["skipped_backward_steps"] = instr["skipped_backward_steps"]
    return report, data, run_config, diag, instr


@pytest.mark.parametrize("control", FIXED_BUDGET_CONTROLS)
def test_phase1_3_fixed_budget_valid_controls_have_no_errors(control):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(control)
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert not report.errors, [f.to_dict() for f in report.errors]


def test_phase1_3_fixed_budget_noncanonical_control_returns_without_findings():
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(
        "exact_random"
    )
    run_config["control"] = "random_veto_deferral"
    diag["quota_size"] = QUOTA + 5
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert not report.findings


FIXED_BUDGET_FLAGS = (
    ("identity_inputs", "no_early_stopping", True),
    ("run_config", "no_early_stopping", True),
    ("controller_config", "early_stopping_active", False),
    ("controller_config", "allow_early_stopping_with_skipping", False),
    ("run_config", "allow_early_stopping_with_skipping", False),
    ("controller_config", "matched_budget", True),
    ("run_config", "matched_budget", True),
)
FIXED_BUDGET_FLAG_CASES = ("missing", "contradictory", "integer")
FREEZE_MODE_SOURCES = (
    "identity_inputs.skip_update_mode",
    "skip_update_mode",
    "run_config.skip_update_mode",
    "true_skip_instrumentation.skip_update_mode",
)


def _budget_flag_targets(data, run_config, source):
    if source == "identity_inputs":
        return [data["identity_inputs"]]
    if source == "run_config":
        return [run_config]
    return [data["controller_config"], run_config["controller_config"]]


def _apply_budget_flag(data, run_config, source, key, value):
    for entry in _budget_flag_targets(data, run_config, source):
        if value is _REMOVE:
            entry.pop(key)
        else:
            entry[key] = value


def _freeze_mode_target(data, run_config, instr, field):
    return {
        "identity_inputs.skip_update_mode": data["identity_inputs"],
        "skip_update_mode": data,
        "run_config.skip_update_mode": run_config,
        "true_skip_instrumentation.skip_update_mode": instr,
    }[field]


@pytest.mark.parametrize("case", FIXED_BUDGET_FLAG_CASES)
@pytest.mark.parametrize("source, key, expected", FIXED_BUDGET_FLAGS)
def test_phase1_3_fixed_budget_flag_errors(source, key, expected, case):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(
        "exact_random"
    )
    value = {
        "missing": _REMOVE,
        "contradictory": not expected,
        "integer": int(expected),
    }[case]
    _apply_budget_flag(data, run_config, source, key, value)
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert source + "." + key in fields(report, "error")


@pytest.mark.parametrize("value", (_REMOVE, False))
def test_phase1_3_fixed_budget_early_stopped_valid_values(value):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(
        "exact_random"
    )
    if value is _REMOVE:
        data.pop("early_stopped", None)
    else:
        data["early_stopped"] = value
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert "early_stopped" not in fields(report, "error")


@pytest.mark.parametrize("value", (True, 0, None))
def test_phase1_3_fixed_budget_early_stopped_invalid_values(value):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(
        "exact_random"
    )
    data["early_stopped"] = value
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert "early_stopped" in fields(report, "error")


@pytest.mark.parametrize("case", FIXED_BUDGET_FLAG_CASES)
@pytest.mark.parametrize(
    "control, expected",
    [("full_finetune", False), ("exact_random", True)],
)
def test_phase1_3_fixed_budget_is_skipping_arm_errors(
    control, expected, case
):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(control)
    value = {
        "missing": _REMOVE,
        "contradictory": not expected,
        "integer": int(expected),
    }[case]
    _apply_budget_flag(
        data, run_config, "controller_config", "is_skipping_arm", value
    )
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert "controller_config.is_skipping_arm" in fields(report, "error")


@pytest.mark.parametrize("value", (_REMOVE, "momentum", None))
@pytest.mark.parametrize("field", FREEZE_MODE_SOURCES)
def test_phase1_3_fixed_budget_freeze_mode_source_errors(field, value):
    report, data, run_config, diag, instr = _phase1_3_budget_inputs(
        "exact_random"
    )
    target = _freeze_mode_target(data, run_config, instr, field)
    if value is _REMOVE:
        target.pop("skip_update_mode")
    else:
        target["skip_update_mode"] = value
    vspr._check_phase1_3_fixed_budget(report, data, run_config, diag, instr)
    assert field in fields(report, "error")
