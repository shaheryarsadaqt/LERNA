"""Piece 5: fixture-based tests for authoritative local result validation.

No trainer/runner/W&B imports. Fixtures mirror the results.json shape written
by scripts/run_ablation_study.py after Pieces 2 and 3.
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

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


def _instrumentation(skipped=QUOTA, batches=TOTAL, mode="freeze"):
    backward = batches - skipped
    return {
        "forward_calls": batches,
        "backward_calls": backward,
        "optimizer_step_attempts": backward,
        "scheduler_step_calls": backward,
        "skipped_backward_steps": skipped,
        "batches_seen": batches,
        "skipped_batches": skipped,
        "skip_ratio_by_batch": skipped / max(batches, 1),
        "skip_update_mode": mode,
        "invariant_forward_eq_backward_plus_skipped": True,
        "invariant_opt_le_backward": True,
        "invariant_sched_le_opt": True,
    }


def _run_config(mode="freeze", matched=True):
    return {
        "policy": "random_veto_deferral",
        "target_skip_rate": RATE,
        "no_early_stopping": True,
        "allow_early_stopping_with_skipping": False,
        "matched_budget": matched,
        "skip_update_mode": mode,
    }


def random_results():
    """Valid completed exact-random run."""
    return {
        "eval_metrics": {"eval_accuracy": 0.9},
        "skip_update_mode": "freeze",
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
    return {
        "eval_metrics": {"eval_accuracy": 0.9},
        "policy_name": "always_false",
        "skip_update_mode": "freeze",
        "true_skip_instrumentation": _instrumentation(skipped=0),
        "policy_diagnostics": {},
        "run_config": {
            **_run_config(),
            "control": "full_finetune",
            "target_skip_rate": RATE,
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
    del data["true_skip_instrumentation"]["invariant_sched_le_opt"]
    report = vspr.validate_results(write_results(tmp_path, data))
    assert not report.ok
    assert ("true_skip_instrumentation.invariant_sched_le_opt"
            in fields(report, "error"))


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
