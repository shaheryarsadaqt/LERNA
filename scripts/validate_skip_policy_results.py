#!/usr/bin/env python3
"""Authoritative local validation of skip-policy run results (Piece 5).

Validates a run's results.json against the policy and trainer invariants
actually emitted after Pieces 2 and 3. Produces structured findings
(severity, field, expected, actual, message) and exits nonzero on
correctness failures. Observed veto rate is reported descriptively only.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

VALID_SKIP_UPDATE_MODES = ("freeze", "momentum")
RVD_POLICY_NAME = "random_veto_deferral"
LEGACY_QUOTA_INVARIANT_KEY = "invariant_quota_decomposition_ok"

# Boolean invariants emitted by the policies after Pieces 2 and 3.
POLICY_INVARIANT_KEYS = (
    "invariant_skip_accounting_ok",
    "invariant_skip_source_decomposition_ok",
    "invariant_debt_conservation_ok",
    "invariant_debt_nonnegative_ok",
    "invariant_candidate_count_ok",
    "invariant_debt_never_negative_ok",
    "invariant_forced_tail_no_double_count_ok",
    "invariant_repayment_single_source_ok",
)

# Boolean invariants emitted by TrueBackwardSkippingTrainer instrumentation.
INSTRUMENTATION_INVARIANT_KEYS = (
    "invariant_forward_eq_backward_plus_skipped",
    "invariant_opt_le_backward",
    "invariant_sched_le_opt",
)

_MISSING = object()


@dataclass
class Finding:
    severity: str
    field: str
    expected: Any
    actual: Any
    message: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ValidationReport:
    path: str
    findings: List[Finding] = field(default_factory=list)
    protocol_complete: Optional[bool] = None
    matched_budget_claimed: Optional[bool] = None

    @property
    def errors(self) -> List[Finding]:
        return [f for f in self.findings if f.severity == SEVERITY_ERROR]

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def valid_for_matched_budget(self) -> bool:
        return (
            self.ok
            and self.matched_budget_claimed is True
            and self.protocol_complete is True
        )

    def add(self, severity: str, field_name: str, expected: Any, actual: Any,
            message: str) -> None:
        self.findings.append(
            Finding(severity=severity, field=field_name, expected=expected,
                    actual=actual, message=message)
        )

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "ok": self.ok,
            "protocol_complete": self.protocol_complete,
            "matched_budget_claimed": self.matched_budget_claimed,
            "valid_for_matched_budget": self.valid_for_matched_budget,
            "n_errors": len(self.errors),
            "findings": [f.to_dict() for f in self.findings],
        }


def _as_float(value) -> Optional[float]:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _as_int(value) -> Optional[int]:
    parsed = _as_float(value)
    if parsed is None or not parsed.is_integer():
        return None
    return int(parsed)


def _check_existing_boolean_invariants(report: ValidationReport, diag: dict,
                                       instr: dict) -> None:
    """Check only invariant keys that actually exist in the emitted output."""
    # Legacy key: checked ONLY if that exact key exists.
    if LEGACY_QUOTA_INVARIANT_KEY in diag:
        if diag[LEGACY_QUOTA_INVARIANT_KEY] is not True:
            report.add(SEVERITY_ERROR, f"policy_diagnostics.{LEGACY_QUOTA_INVARIANT_KEY}",
                       True, diag[LEGACY_QUOTA_INVARIANT_KEY],
                       "legacy quota decomposition invariant present but false")

    for key in POLICY_INVARIANT_KEYS:
        if key in diag and diag[key] is not True:
            report.add(SEVERITY_ERROR, f"policy_diagnostics.{key}", True, diag[key],
                       f"policy invariant {key} is false")

    for key in INSTRUMENTATION_INVARIANT_KEYS:
        if key in instr and instr[key] is not True:
            report.add(SEVERITY_ERROR, f"true_skip_instrumentation.{key}", True,
                       instr[key], f"instrumentation invariant {key} is false")


def _require_true_invariants(report: ValidationReport, payload: dict,
                              prefix: str, keys) -> None:
    # False values are reported by _check_existing_boolean_invariants().
    for key in keys:
        if key not in payload:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}", True, None,
                       f"required invariant {key} is missing")


def _check_agreement(report: ValidationReport, diag: dict, instr: dict) -> None:
    count_fields = (
        (diag, "policy_diagnostics", "skip_decisions"),
        (diag, "policy_diagnostics", "decisions_seen"),
        (diag, "policy_diagnostics", "quota_size"),
        (diag, "policy_diagnostics", "quota_total_steps"),
        (instr, "true_skip_instrumentation", "skipped_backward_steps"),
        (instr, "true_skip_instrumentation", "skipped_batches"),
        (instr, "true_skip_instrumentation", "batches_seen"),
        (instr, "true_skip_instrumentation", "forward_calls"),
        (instr, "true_skip_instrumentation", "backward_calls"),
    )
    for payload, prefix, key in count_fields:
        if key in payload:
            parsed = _as_int(payload[key])
            if parsed is None or parsed < 0:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                           "nonnegative integer", payload[key],
                           f"{key} is malformed")

    skip_decisions = _as_int(diag.get("skip_decisions"))
    decisions_seen = _as_int(diag.get("decisions_seen"))
    skipped_backward = _as_int(instr.get("skipped_backward_steps"))
    skipped_batches = _as_int(instr.get("skipped_batches"))
    batches_seen = _as_int(instr.get("batches_seen"))
    forward_calls = _as_int(instr.get("forward_calls"))

    if skip_decisions is not None and skipped_backward is not None:
        if skip_decisions != skipped_backward:
            report.add(SEVERITY_ERROR, "skip_decisions_vs_skipped_backward_steps",
                       skip_decisions, skipped_backward,
                       "policy skip_decisions disagrees with trainer "
                       "skipped_backward_steps")
    if skipped_backward is not None and skipped_batches is not None:
        if skipped_backward != skipped_batches:
            report.add(SEVERITY_ERROR, "skipped_backward_steps_vs_skipped_batches",
                       skipped_backward, skipped_batches,
                       "trainer skipped-backward and skipped-batch counts disagree")
    if decisions_seen is not None and batches_seen is not None:
        if decisions_seen != batches_seen:
            report.add(SEVERITY_ERROR, "decisions_seen_vs_batches_seen",
                       decisions_seen, batches_seen,
                       "policy decisions and trainer batches disagree")
    if decisions_seen is not None and forward_calls is not None:
        if decisions_seen != forward_calls:
            report.add(SEVERITY_ERROR, "decisions_seen_vs_forward_calls",
                       decisions_seen, forward_calls,
                       "policy decisions and trainer forward calls disagree")

    reported_rate = _as_float(diag.get("realized_skip_rate"))
    if skip_decisions is not None and decisions_seen is not None:
        recomputed = skip_decisions / max(decisions_seen, 1)
        if reported_rate is not None and abs(recomputed - reported_rate) > 1e-9:
            report.add(SEVERITY_ERROR, "policy_diagnostics.realized_skip_rate",
                       recomputed, reported_rate,
                       "reported realized_skip_rate does not equal "
                       "skip_decisions / decisions_seen")
    else:
        recomputed = reported_rate

    batch_rate = _as_float(instr.get("skip_ratio_by_batch"))
    if batch_rate is None and batches_seen is not None and skipped_batches is not None:
        batch_rate = skipped_batches / max(batches_seen, 1)
    if batch_rate is not None and recomputed is not None:
        if abs(batch_rate - recomputed) > 1e-9:
            report.add(SEVERITY_ERROR, "instrumentation_batch_skip_rate",
                       recomputed, batch_rate,
                       "instrumentation batch skip rate disagrees with policy "
                       "realized skip rate")


def _check_exact_quota_plan(report: ValidationReport, diag: dict,
                             completed: bool, count_tolerance: int) -> None:
    quota_size = _as_int(diag.get("quota_size"))
    if quota_size is None:
        return

    quota_total = _as_int(diag.get("quota_total_steps"))
    decisions_seen = _as_int(diag.get("decisions_seen"))
    skip_decisions = _as_int(diag.get("skip_decisions"))
    target_rate = _as_float(diag.get("target_skip_rate"))

    if quota_size < 0:
        report.add(SEVERITY_ERROR, "policy_diagnostics.quota_size", ">= 0",
                   quota_size, "quota_size must be nonnegative")
    if quota_total is None or quota_total <= 0:
        report.add(SEVERITY_ERROR, "policy_diagnostics.quota_total_steps",
                   "positive integer", diag.get("quota_total_steps"),
                   "exact-quota diagnostics require a positive horizon")
        return
    if target_rate is not None:
        expected_quota = round(target_rate * quota_total)
        if quota_size != expected_quota:
            report.add(SEVERITY_ERROR, "policy_diagnostics.quota_size",
                       expected_quota, quota_size,
                       "quota_size does not match round(target_skip_rate * "
                       "quota_total_steps)")
    if decisions_seen is not None and decisions_seen > quota_total:
        report.add(SEVERITY_ERROR, "policy_diagnostics.decisions_seen",
                   f"<= {quota_total}", decisions_seen,
                   "policy decision count exceeded the configured horizon")
    if completed and skip_decisions is not None:
        if abs(skip_decisions - quota_size) > count_tolerance:
            report.add(SEVERITY_ERROR, "quota_size_agreement", quota_size,
                       skip_decisions,
                       "completed exact-quota run did not realize the planned "
                       "integer quota")


def _check_rvd_exact_quota(report: ValidationReport, diag: dict,
                           instr: dict) -> None:
    """Validate all emitted RVD invariants and closed source accounting."""
    _require_true_invariants(
        report, diag, "policy_diagnostics", POLICY_INVARIANT_KEYS
    )

    outstanding = _as_int(diag.get("outstanding_debt"))
    deferred_now = _as_int(diag.get("deferred_pool_now"))
    if outstanding is None:
        report.add(SEVERITY_ERROR, "policy_diagnostics.outstanding_debt", 0,
                   diag.get("outstanding_debt"),
                   "outstanding debt missing or non-integral in completed RVD "
                   "diagnostics")
    elif outstanding != 0:
        report.add(SEVERITY_ERROR, "policy_diagnostics.outstanding_debt", 0,
                   outstanding, "completed RVD run must have zero outstanding debt")
    if deferred_now is not None and outstanding is not None:
        if deferred_now != outstanding:
            report.add(SEVERITY_ERROR, "policy_diagnostics.deferred_pool_now",
                       outstanding, deferred_now,
                       "deferred_pool_now disagrees with outstanding_debt")

    source_counts = diag.get("skip_source_counts")
    expected_sources = {
        "accepted_candidate", "ordinary_repayment", "forced_tail"
    }
    parsed_counts = {}
    if not isinstance(source_counts, dict):
        report.add(SEVERITY_ERROR, "policy_diagnostics.skip_source_counts",
                   "dict of source counts", source_counts,
                   "skip_source_counts missing from completed RVD diagnostics")
    else:
        if set(source_counts) != expected_sources:
            report.add(SEVERITY_ERROR, "policy_diagnostics.skip_source_counts.keys",
                       sorted(expected_sources), sorted(map(str, source_counts)),
                       "skip_source_counts must use the closed Piece 2 source set")
        for name, value in source_counts.items():
            parsed = _as_int(value)
            if parsed is None or parsed < 0:
                report.add(SEVERITY_ERROR,
                           f"policy_diagnostics.skip_source_counts.{name}",
                           "nonnegative integer", value,
                           "skip-source count is malformed")
            else:
                parsed_counts[name] = parsed

    if (isinstance(source_counts, dict)
            and len(parsed_counts) == len(source_counts)):
        total_sources = sum(parsed_counts.values())
        skipped_backward = _as_int(instr.get("skipped_backward_steps"))
        skip_decisions = _as_int(diag.get("skip_decisions"))
        if skipped_backward is not None and total_sources != skipped_backward:
            report.add(SEVERITY_ERROR, "skip_source_counts_sum", skipped_backward,
                       total_sources,
                       "sum of skip-source counts must equal trainer "
                       "skipped_backward_steps")
        if skip_decisions is not None and total_sources != skip_decisions:
            report.add(SEVERITY_ERROR, "skip_source_counts_vs_skip_decisions",
                       skip_decisions, total_sources,
                       "sum of skip-source counts must equal policy skip_decisions")


def _check_veto_rate_descriptive(report: ValidationReport, diag: dict) -> None:
    """Requirement 4: veto rate is descriptive; no cap is enforced."""
    veto_rate = _as_float(diag.get("veto_rate_vs_candidates"))
    if veto_rate is None:
        return
    target = _as_float(diag.get("target_veto_rate"))
    report.add(SEVERITY_INFO, "policy_diagnostics.veto_rate_vs_candidates",
               "descriptive only (no enforced cap)", veto_rate,
               f"observed veto rate {veto_rate:.4f}"
               + (f" (target_veto_rate={target:.4f} is not enforced by any "
                  "current policy)" if target is not None else ""))


def _check_skip_update_mode(report: ValidationReport, data: dict, run_config: dict,
                            instr: dict, matched: bool,
                            allow_historical_momentum: bool) -> None:
    sources = {
        "skip_update_mode": data.get("skip_update_mode", _MISSING),
        "run_config.skip_update_mode": run_config.get("skip_update_mode", _MISSING),
        "true_skip_instrumentation.skip_update_mode":
            instr.get("skip_update_mode", _MISSING),
    }
    present = {k: v for k, v in sources.items() if v is not _MISSING}
    if not present:
        report.add(SEVERITY_ERROR, "skip_update_mode",
                   f"one of {VALID_SKIP_UPDATE_MODES}", None,
                   "skip_update_mode missing from results")
        return

    valid_present = {}
    for field_name, mode in present.items():
        if not isinstance(mode, str) or mode not in VALID_SKIP_UPDATE_MODES:
            report.add(SEVERITY_ERROR, field_name,
                       f"one of {VALID_SKIP_UPDATE_MODES}", mode,
                       "invalid skip_update_mode value")
        else:
            valid_present[field_name] = mode

    modes = set(valid_present.values())
    if len(modes) > 1:
        report.add(SEVERITY_ERROR, "skip_update_mode_consistency",
                   "single consistent mode", dict(valid_present),
                   "skip_update_mode disagrees across results, run_config and "
                   "instrumentation")
    if not valid_present:
        return

    effective = next(iter(valid_present.values()))
    historical = (
        allow_historical_momentum
        or data.get("historical_momentum_comparison") is True
        or run_config.get("historical_momentum_comparison") is True
    )
    if matched and effective == "momentum" and not historical:
        report.add(SEVERITY_ERROR, "skip_update_mode", "freeze", effective,
                   "matched Phase 1.3 runs must use freeze unless explicitly "
                   "classified as a historical momentum comparison")


def _check_matched_budget(report: ValidationReport, data: dict, run_config: dict,
                          diag: dict, instr: dict, completed: bool,
                          interrupted: bool, rate_tolerance: float,
                          count_tolerance: int,
                          require_policy_diagnostics: bool = True) -> None:
    """Requirement 6: reject invalid matched-budget classification."""
    early_stopping_occurred = (
        run_config.get("no_early_stopping") is False
        or run_config.get("early_stopping_active") is True
        or data.get("early_stopped") is True
    )
    if early_stopping_occurred:
        report.add(SEVERITY_ERROR, "run_config.matched_budget", False, True,
                   "matched-budget classification rejected: early stopping "
                   "occurred or was active")

    if run_config.get("allow_early_stopping_with_skipping") is True:
        report.add(SEVERITY_ERROR,
                   "run_config.allow_early_stopping_with_skipping", False, True,
                   "matched-budget classification rejected: research override "
                   "was used")

    if interrupted:
        report.add(SEVERITY_ERROR, "run_status", "completed", "interrupted",
                   "matched-budget classification rejected: run was interrupted")
    elif not completed:
        report.add(SEVERITY_ERROR, "run_status", "completed", "incomplete",
                   "matched-budget classification rejected: run did not complete "
                   "the planned decision total")

    requested_quota = _as_int(diag.get("quota_size"))
    realized_count = _as_int(instr.get("skipped_backward_steps",
                                       diag.get("skip_decisions")))
    if requested_quota is not None and realized_count is not None:
        if abs(realized_count - requested_quota) > count_tolerance:
            report.add(SEVERITY_ERROR, "requested_vs_realized_quota",
                       requested_quota, realized_count,
                       "matched-budget classification rejected: realized skip "
                       "count differs from requested integer quota")
    elif requested_quota is None:
        # Non-quota legacy policies have no exact integer budget to compare.
        target_rate = _as_float(run_config.get(
            "target_skip_rate", diag.get("target_skip_rate")
        ))
        realized_rate = _as_float(diag.get("realized_skip_rate"))
        if target_rate is not None and realized_rate is not None:
            if abs(realized_rate - target_rate) > rate_tolerance:
                report.add(SEVERITY_ERROR, "requested_vs_realized_skip_rate",
                           f"within {rate_tolerance} of {target_rate:.6f}",
                           realized_rate,
                           "matched-budget classification rejected: requested "
                           "and realized rates differ beyond tolerance")

    artifact_fields = ["true_skip_instrumentation", "eval_metrics", "run_config"]
    if require_policy_diagnostics:
        artifact_fields.insert(0, "policy_diagnostics")
    for artifact_field in artifact_fields:
        value = data.get(artifact_field)
        if not value:
            report.add(SEVERITY_ERROR, artifact_field, "present and non-empty",
                       value,
                       "matched-budget classification rejected: required "
                       f"artifact '{artifact_field}' is missing")

    _require_true_invariants(
        report, instr, "true_skip_instrumentation",
        INSTRUMENTATION_INVARIANT_KEYS,
    )


def _check_full_finetune(report: ValidationReport, instr: dict) -> None:
    skipped = _as_int(instr.get("skipped_backward_steps"))
    skipped_batches = _as_int(instr.get("skipped_batches"))
    forward = _as_int(instr.get("forward_calls"))
    backward = _as_int(instr.get("backward_calls"))

    if skipped != 0:
        report.add(SEVERITY_ERROR, "true_skip_instrumentation.skipped_backward_steps",
                   0, instr.get("skipped_backward_steps"),
                   "full_finetune must not skip backward passes")
    if skipped_batches != 0:
        report.add(SEVERITY_ERROR, "true_skip_instrumentation.skipped_batches", 0,
                   instr.get("skipped_batches"),
                   "full_finetune must not skip batches")
    if forward is not None and backward is not None and forward != backward:
        report.add(SEVERITY_ERROR, "full_finetune_forward_vs_backward", forward,
                   backward,
                   "full_finetune must execute backward for every forward pass")


def validate_results(path: Path, *, rate_tolerance: float = 0.005,
                     count_tolerance: int = 0,
                     allow_historical_momentum: bool = False,
                     required_artifacts: Optional[List[str]] = None
                     ) -> ValidationReport:
    path = Path(path)
    report = ValidationReport(path=str(path))

    if rate_tolerance < 0 or not math.isfinite(rate_tolerance):
        report.add(SEVERITY_ERROR, "rate_tolerance", "finite and >= 0",
                   rate_tolerance, "rate tolerance is invalid")
        return report
    if count_tolerance < 0:
        report.add(SEVERITY_ERROR, "count_tolerance", ">= 0", count_tolerance,
                   "count tolerance is invalid")
        return report
    if not path.exists():
        report.add(SEVERITY_ERROR, "results_json", "existing file", None,
                   f"results.json not found: {path}")
        return report

    for name in (required_artifacts or []):
        sibling = path.parent / name
        if not sibling.exists():
            report.add(SEVERITY_ERROR, f"artifact:{name}", "existing file", None,
                       f"required artifact missing: {sibling}")

    try:
        with path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        report.add(SEVERITY_ERROR, "results_json", "valid JSON", str(exc),
                   "results.json is unreadable or truncated (interrupted run)")
        return report
    if not isinstance(data, dict):
        report.add(SEVERITY_ERROR, "results_json", "JSON object",
                   type(data).__name__, "results.json root must be an object")
        return report

    raw_diag = data.get("policy_diagnostics")
    raw_instr = data.get("true_skip_instrumentation")
    raw_config = data.get("run_config")
    diag = raw_diag if isinstance(raw_diag, dict) else {}
    instr = raw_instr if isinstance(raw_instr, dict) else {}
    run_config = raw_config if isinstance(raw_config, dict) else {}
    for field_name, raw_value in (
        ("policy_diagnostics", raw_diag),
        ("true_skip_instrumentation", raw_instr),
        ("run_config", raw_config),
    ):
        if raw_value is not None and not isinstance(raw_value, dict):
            report.add(SEVERITY_ERROR, field_name, "JSON object", raw_value,
                       f"{field_name} must be an object")

    policy_name = diag.get("policy_name") or data.get("policy_name")
    control = run_config.get("control")
    is_full_finetune = control == "full_finetune" or policy_name == "always_false"
    if not diag and not is_full_finetune:
        report.add(SEVERITY_ERROR, "policy_diagnostics", "present and non-empty",
                   raw_diag, "policy_diagnostics missing from skip-policy result")
        return report

    interrupted = bool(
        data.get("interrupted") is True
        or "error" in data
        or "eval_metrics" not in data
    )
    decisions_seen = _as_int(diag.get("decisions_seen"))
    quota_total = _as_int(diag.get("quota_total_steps"))
    claims_exact_quota = _as_int(diag.get("quota_size")) is not None
    if claims_exact_quota:
        completed = (
            not interrupted
            and decisions_seen is not None
            and quota_total is not None
            and decisions_seen == quota_total
        )
    else:
        completed = not interrupted

    matched_raw = run_config.get("matched_budget", _MISSING)
    if matched_raw is not _MISSING and not isinstance(matched_raw, bool):
        report.add(SEVERITY_ERROR, "run_config.matched_budget", "boolean",
                   matched_raw, "matched_budget must be a boolean")
    matched = matched_raw is True
    report.protocol_complete = completed
    report.matched_budget_claimed = matched if matched_raw is not _MISSING else None
    if claims_exact_quota and not completed:
        report.add(
            SEVERITY_WARNING, "quota_protocol_complete", True, False,
            "exact-quota protocol did not complete its planned horizon",
        )

    _check_existing_boolean_invariants(report, diag, instr)
    _check_agreement(report, diag, instr)
    _check_veto_rate_descriptive(report, diag)
    _check_skip_update_mode(report, data, run_config, instr, matched,
                            allow_historical_momentum)
    _check_exact_quota_plan(report, diag, completed, count_tolerance)

    if is_full_finetune:
        _check_full_finetune(report, instr)
    if policy_name == RVD_POLICY_NAME and completed and claims_exact_quota:
        _check_rvd_exact_quota(report, diag, instr)

    if matched:
        _check_matched_budget(
            report, data, run_config, diag, instr, completed, interrupted,
            rate_tolerance, count_tolerance,
            require_policy_diagnostics=not is_full_finetune,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate skip-policy diagnostics in a run's results.json."
    )
    parser.add_argument("results_json", type=Path, help="Path to results.json")
    parser.add_argument("--rate-tolerance", "--tolerance", dest="rate_tolerance",
                        type=float, default=0.005,
                        help="Allowed deviation for realized skip rates")
    parser.add_argument("--count-tolerance", type=int, default=0,
                        help="Allowed absolute deviation for skip counts")
    parser.add_argument("--allow-historical-momentum", action="store_true",
                        help="Classify this run as a historical momentum "
                             "comparison (permits skip_update_mode=momentum)")
    parser.add_argument("--require-artifact", action="append", default=[],
                        help="Sibling artifact filename that must exist "
                             "(repeatable)")
    args = parser.parse_args()

    report = validate_results(
        args.results_json,
        rate_tolerance=args.rate_tolerance,
        count_tolerance=args.count_tolerance,
        allow_historical_momentum=args.allow_historical_momentum,
        required_artifacts=args.require_artifact,
    )
    print(json.dumps(report.to_dict(), indent=2, default=str))
    if not report.ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
