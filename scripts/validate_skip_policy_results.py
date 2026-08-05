#!/usr/bin/env python3
"""Authoritative local validation of skip-policy run results (Piece 5).

Validates a run's results.json against the policy and trainer invariants
actually emitted after Pieces 2 and 3. Produces structured findings
(severity, field, expected, actual, message) and exits nonzero on
correctness failures. Observed veto rate is reported descriptively only.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional

_PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "lerna_run_provenance",
    Path(__file__).resolve().parents[1] / "lerna" / "utils" / "run_provenance.py",
)
if _PROVENANCE_SPEC is None or _PROVENANCE_SPEC.loader is None:
    raise ImportError("could not load lerna.utils.run_provenance")
_PROVENANCE_MODULE = importlib.util.module_from_spec(_PROVENANCE_SPEC)
_PROVENANCE_SPEC.loader.exec_module(_PROVENANCE_MODULE)
build_scientific_fingerprint = _PROVENANCE_MODULE.build_scientific_fingerprint

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"
SEVERITY_INFO = "info"

VALID_SKIP_UPDATE_MODES = ("freeze", "momentum")
SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP = "skip_on_backward_skip"
SCHEDULER_POLICY_ALWAYS_STEP = "always_step"
VALID_SCHEDULER_STEP_POLICIES = (
    SCHEDULER_POLICY_ALWAYS_STEP,
    SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP,
)
RVD_POLICY_NAME = "random_veto_deferral"

PHASE1_3_CONTROLS = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
)

PHASE1_3_POLICY_CLASSES = {
    "full_finetune": "AlwaysFalsePolicy",
    "exact_random": "RandomSkipPolicy",
    "fixed_phase_strat": "FixedPhaseStratifiedRandomPolicy",
    "phase_strat_guarded": "PhaseStratifiedGuardedRandomPolicy",
    "ler_guided_stratified": "LERGuidedStratifiedPolicy",
    "ler_guided_stratified_safe": "LERGuidedStratifiedSafetyPolicy",
}
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

# Common invariants emitted by TrueBackwardSkippingTrainer instrumentation.
INSTRUMENTATION_COMMON_INVARIANT_KEYS = (
    "invariant_forward_eq_backward_plus_skipped",
    "invariant_opt_le_backward",
)
LEGACY_SCHEDULER_INVARIANT_KEY = "invariant_sched_le_opt"
SCHEDULER_INVARIANT_KEY = "invariant_scheduler_policy_consistent"
SCHEDULER_OPPORTUNITY_INVARIANT_KEY = "invariant_sched_le_opportunities"

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


def _resolve_scheduler_step_policy(report: ValidationReport, data: dict,
                                   run_config: dict, instr: dict) -> str:
    """Resolve explicit policy, defaulting legacy artifacts to trainer behavior."""
    sources = {
        "scheduler_step_policy": data.get("scheduler_step_policy", _MISSING),
        "run_config.scheduler_step_policy": run_config.get(
            "scheduler_step_policy", _MISSING
        ),
        "true_skip_instrumentation.scheduler_step_policy": instr.get(
            "scheduler_step_policy", _MISSING
        ),
    }
    present = {key: value for key, value in sources.items() if value is not _MISSING}
    if not present:
        return SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP

    valid = {}
    for field_name, value in present.items():
        if value not in VALID_SCHEDULER_STEP_POLICIES:
            report.add(
                SEVERITY_ERROR,
                field_name,
                f"one of {VALID_SCHEDULER_STEP_POLICIES}",
                value,
                "invalid scheduler_step_policy value",
            )
        else:
            valid[field_name] = value
    if len(set(valid.values())) > 1:
        report.add(
            SEVERITY_ERROR,
            "scheduler_step_policy_consistency",
            "one consistent scheduler policy",
            valid,
            "scheduler_step_policy disagrees across result artifacts",
        )
    return next(iter(valid.values()), SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP)


def _check_scheduler_step_counts(report: ValidationReport, instr: dict,
                                 scheduler_policy: str) -> None:
    scheduler_steps = _as_int(instr.get("scheduler_step_calls"))
    optimizer_attempts = _as_int(instr.get("optimizer_step_attempts"))
    if scheduler_steps is None or optimizer_attempts is None:
        return

    if scheduler_policy == SCHEDULER_POLICY_ALWAYS_STEP:
        skipped = _as_int(instr.get("skipped_backward_steps"))
        if skipped is None:
            report.add(
                SEVERITY_ERROR,
                "true_skip_instrumentation.skipped_backward_steps",
                "integer",
                instr.get("skipped_backward_steps"),
                "always_step validation requires skipped backward count",
            )
            return
        bound = optimizer_attempts + skipped
        bound_description = "optimizer attempts plus skipped backward steps"
    else:
        bound = optimizer_attempts
        bound_description = "optimizer step attempts"

    if scheduler_steps > bound:
        report.add(
            SEVERITY_ERROR,
            "true_skip_instrumentation.scheduler_step_calls",
            f"<= {bound}",
            scheduler_steps,
            f"scheduler calls exceed {bound_description}",
        )


def _require_scheduler_invariant(report: ValidationReport, instr: dict,
                                 scheduler_policy: str) -> None:
    if SCHEDULER_INVARIANT_KEY in instr:
        if instr[SCHEDULER_INVARIANT_KEY] is not True:
            report.add(
                SEVERITY_ERROR,
                f"true_skip_instrumentation.{SCHEDULER_INVARIANT_KEY}",
                True,
                instr[SCHEDULER_INVARIANT_KEY],
                "policy-aware scheduler invariant is false",
            )
        return
    if (
        scheduler_policy == SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP
        and instr.get(LEGACY_SCHEDULER_INVARIANT_KEY) is True
    ):
        return
    report.add(
        SEVERITY_ERROR,
        f"true_skip_instrumentation.{SCHEDULER_INVARIANT_KEY}",
        True,
        None,
        "required policy-aware scheduler invariant is missing",
    )


def _check_existing_boolean_invariants(report: ValidationReport, diag: dict,
                                       instr: dict,
                                       scheduler_policy: str) -> None:
    """Check only invariant keys that actually exist in the emitted output."""
    if LEGACY_QUOTA_INVARIANT_KEY in diag:
        if diag[LEGACY_QUOTA_INVARIANT_KEY] is not True:
            report.add(SEVERITY_ERROR, f"policy_diagnostics.{LEGACY_QUOTA_INVARIANT_KEY}",
                       True, diag[LEGACY_QUOTA_INVARIANT_KEY],
                       "legacy quota decomposition invariant present but false")

    for key in POLICY_INVARIANT_KEYS:
        if key in diag and diag[key] is not True:
            report.add(SEVERITY_ERROR, f"policy_diagnostics.{key}", True, diag[key],
                       f"policy invariant {key} is false")

    scheduler_keys = (
        SCHEDULER_INVARIANT_KEY,
        SCHEDULER_OPPORTUNITY_INVARIANT_KEY,
    )
    for key in (*INSTRUMENTATION_COMMON_INVARIANT_KEYS, *scheduler_keys):
        if key in instr and instr[key] is not True:
            report.add(SEVERITY_ERROR, f"true_skip_instrumentation.{key}", True,
                       instr[key], f"instrumentation invariant {key} is false")

    if (
        scheduler_policy == SCHEDULER_POLICY_SKIP_ON_BACKWARD_SKIP
        and LEGACY_SCHEDULER_INVARIANT_KEY in instr
        and instr[LEGACY_SCHEDULER_INVARIANT_KEY] is not True
    ):
        report.add(
            SEVERITY_ERROR,
            f"true_skip_instrumentation.{LEGACY_SCHEDULER_INVARIANT_KEY}",
            True,
            instr[LEGACY_SCHEDULER_INVARIANT_KEY],
            "legacy scheduler invariant is false",
        )


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
        INSTRUMENTATION_COMMON_INVARIANT_KEYS,
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


def _check_phase1_3_base_identity(report: ValidationReport, data: dict,
                                  run_config: dict) -> None:
    control = run_config.get("control")
    if control not in PHASE1_3_CONTROLS:
        return

    raw_identity = data.get("identity_inputs")
    identity_inputs = raw_identity if isinstance(raw_identity, dict) else None
    if identity_inputs is None:
        report.add(SEVERITY_ERROR, "identity_inputs", "JSON object",
                   raw_identity,
                   "identity_inputs missing or malformed for canonical "
                   "Phase 1.3 control")

    raw_top_cc = data.get("controller_config")
    top_cc = raw_top_cc if isinstance(raw_top_cc, dict) else None
    if top_cc is None:
        report.add(SEVERITY_ERROR, "controller_config", "JSON object",
                   raw_top_cc,
                   "top-level controller_config missing or malformed for "
                   "canonical Phase 1.3 control")

    raw_rc_cc = run_config.get("controller_config")
    rc_cc = raw_rc_cc if isinstance(raw_rc_cc, dict) else None
    if rc_cc is None:
        report.add(SEVERITY_ERROR, "run_config.controller_config",
                   "JSON object", raw_rc_cc,
                   "run_config.controller_config missing or malformed for "
                   "canonical Phase 1.3 control")

    control_sources = {
        "ablation": data.get("ablation"),
        "run_config.control": control,
    }
    if identity_inputs is not None:
        control_sources["identity_inputs.control"] = identity_inputs.get(
            "control"
        )
    if top_cc is not None:
        control_sources["controller_config.control"] = top_cc.get("control")
        control_sources["controller_config.arm"] = top_cc.get("arm")
    if any(value != control for value in control_sources.values()):
        report.add(SEVERITY_ERROR, "phase1_3_control_identity", control,
                   control_sources,
                   "control identity disagrees across ablation, run_config, "
                   "identity_inputs and controller_config")

    if top_cc is not None:
        alias = top_cc.get("arm_alias_of")
        if alias is not None:
            report.add(SEVERITY_ERROR, "controller_config.arm_alias_of", None,
                       alias,
                       "canonical Phase 1.3 arm must not alias another arm")
        expected_policy_class = PHASE1_3_POLICY_CLASSES[control]
        if top_cc.get("policy_class") != expected_policy_class:
            report.add(SEVERITY_ERROR, "controller_config.policy_class",
                       expected_policy_class, top_cc.get("policy_class"),
                       f"unexpected policy_class for control '{control}'")
        expected_skipping = control != "full_finetune"
        if top_cc.get("is_skipping_arm") is not expected_skipping:
            report.add(SEVERITY_ERROR, "controller_config.is_skipping_arm",
                       expected_skipping, top_cc.get("is_skipping_arm"),
                       "is_skipping_arm must be false only for full_finetune "
                       "and true for every other canonical control")

    if top_cc is not None and rc_cc is not None and top_cc != rc_cc:
        report.add(SEVERITY_ERROR, "controller_config_equality",
                   "controller_config == run_config.controller_config",
                   {"controller_config": top_cc,
                    "run_config.controller_config": rc_cc},
                   "top-level controller_config is not deeply equal to "
                   "run_config.controller_config")

    fingerprint = data.get("fingerprint")
    if not isinstance(fingerprint, str):
        report.add(SEVERITY_ERROR, "fingerprint", "string", fingerprint,
                   "top-level fingerprint missing or not a string")
    if identity_inputs is not None:
        try:
            recomputed = build_scientific_fingerprint(identity_inputs)
        except Exception as exc:  # noqa: BLE001
            report.add(SEVERITY_ERROR, "fingerprint_recompute",
                       "recomputable scientific fingerprint", repr(exc),
                       "identity_inputs could not be fingerprinted")
        else:
            if isinstance(fingerprint, str) and recomputed != fingerprint:
                report.add(SEVERITY_ERROR, "fingerprint", recomputed,
                           fingerprint,
                           "top-level fingerprint does not match the "
                            "fingerprint recomputed from identity_inputs")


_PHASE_STRAT_CONTROLS = ("fixed_phase_strat", "phase_strat_guarded")

_PHASE_STRAT_CONTROLLER_CLASSES = {
    "fixed_phase_strat": "FixedPhaseStratifiedRandomPolicy",
    "phase_strat_guarded": "PhaseStratifiedGuardedRandomPolicy",
}

_PHASE_STRAT_REQUIRED_FIELDS = (
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

_PHASE_STRAT_INT_FIELDS = ("total_steps", "min_step", "policy_seed",
                           "n_phases", "requested_quota")

_PHASE_STRAT_LIST_FIELDS = {
    "phase_weights": 0,
    "phase_bounds": 1,
    "phase_quota": 0,
    "phase_eligible": 0,
}


def _check_phase1_3_phase_controller(report: ValidationReport, data: dict,
                                     run_config: dict) -> None:
    """Validate phase-controller provenance for canonical Phase 1.3 arms."""
    control = run_config.get("control")
    if control not in PHASE1_3_CONTROLS:
        return

    raw_identity = data.get("identity_inputs")
    identity_inputs = raw_identity if isinstance(raw_identity, dict) else None
    raw_top_cc = data.get("controller_config")
    top_cc = raw_top_cc if isinstance(raw_top_cc, dict) else None

    sources = {
        "identity_inputs.phase_strat_controller": (
            identity_inputs.get("phase_strat_controller", _MISSING)
            if identity_inputs is not None else _MISSING
        ),
        "controller_config.phase_strat_controller": (
            top_cc.get("phase_strat_controller", _MISSING)
            if top_cc is not None else _MISSING
        ),
        "run_config.phase_strat_controller": run_config.get(
            "phase_strat_controller", _MISSING
        ),
    }

    if control not in _PHASE_STRAT_CONTROLS:
        for field_name, value in sources.items():
            if value is not _MISSING:
                report.add(SEVERITY_ERROR, field_name, "absent", value,
                           "stray phase_strat_controller config on "
                           f"non-phase control '{control}'")
        return

    configs = {}
    missing_config = False
    for field_name, value in sources.items():
        if value is _MISSING or not isinstance(value, dict):
            report.add(SEVERITY_ERROR, field_name, "JSON object",
                       None if value is _MISSING else value,
                       "phase_strat_controller config missing or malformed "
                       f"for phase control '{control}'")
            missing_config = True
        else:
            configs[field_name] = value
    if missing_config:
        return

    values = list(configs.values())
    if any(cfg != values[0] for cfg in values[1:]):
        report.add(SEVERITY_ERROR, "phase_strat_controller_equality",
                   "deeply equal phase_strat_controller configs",
                   configs,
                   "phase_strat_controller disagrees across "
                   "identity_inputs, controller_config and run_config")

    prefix = "identity_inputs.phase_strat_controller"
    cfg = configs[prefix]

    for key in _PHASE_STRAT_REQUIRED_FIELDS:
        if key not in cfg:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}", "present", None,
                       f"required phase controller field {key} is missing")

    if "control" in cfg and cfg["control"] != control:
        report.add(SEVERITY_ERROR, f"{prefix}.control", control,
                   cfg["control"],
                   "phase controller control disagrees with the "
                   "effective control")
    if "policy_name" in cfg and cfg["policy_name"] != control:
        report.add(SEVERITY_ERROR, f"{prefix}.policy_name", control,
                   cfg["policy_name"],
                   "phase controller policy_name disagrees with the "
                   "effective control")

    expected_class = _PHASE_STRAT_CONTROLLER_CLASSES[control]
    if ("controller_class" in cfg
            and cfg["controller_class"] != expected_class):
        report.add(SEVERITY_ERROR, f"{prefix}.controller_class",
                   expected_class, cfg["controller_class"],
                   f"unexpected phase controller class for '{control}'")

    if "target_skip_rate" in cfg:
        target_rate = _as_float(cfg["target_skip_rate"])
        if target_rate is None:
            report.add(SEVERITY_ERROR, f"{prefix}.target_skip_rate",
                       "finite number", cfg["target_skip_rate"],
                       "phase controller target_skip_rate is malformed")
        else:
            identity_rate = _as_float(identity_inputs.get("target_skip_rate"))
            if identity_rate != target_rate:
                report.add(SEVERITY_ERROR, f"{prefix}.target_skip_rate",
                           identity_inputs.get("target_skip_rate"),
                           target_rate,
                           "phase controller target_skip_rate disagrees "
                           "with identity_inputs.target_skip_rate")

    parsed_ints = {}
    for key in _PHASE_STRAT_INT_FIELDS:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, str) or isinstance(value, bool):
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       "integer (not boolean)", value,
                       f"phase controller {key} is malformed")
            continue
        parsed = _as_int(value)
        if parsed is None:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       "integer (not boolean)", value,
                       f"phase controller {key} is malformed")
        else:
            parsed_ints[key] = parsed

    for key in ("total_steps", "policy_seed"):
        if key not in parsed_ints:
            continue
        if _as_int(identity_inputs.get(key)) != parsed_ints[key]:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       identity_inputs.get(key), parsed_ints[key],
                       f"phase controller {key} disagrees with "
                       f"identity_inputs.{key}")

    n_phases = parsed_ints.get("n_phases")
    if n_phases is not None and n_phases < 1:
        report.add(SEVERITY_ERROR, f"{prefix}.n_phases", ">= 1", n_phases,
                   "phase controller must declare at least one phase")

    for key, extra in _PHASE_STRAT_LIST_FIELDS.items():
        if key not in cfg:
            continue
        value = cfg[key]
        if not isinstance(value, list):
            report.add(SEVERITY_ERROR, f"{prefix}.{key}", "list", value,
                       f"phase controller {key} must be a list")
            continue
        if (n_phases is not None and n_phases >= 1
                and len(value) != n_phases + extra):
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       f"length {n_phases + extra}", len(value),
                       f"phase controller {key} has the wrong length")
        if key in ("phase_quota", "phase_eligible"):
            for index, entry in enumerate(value):
                parsed = _as_int(entry)
                if parsed is None or parsed < 0:
                    report.add(SEVERITY_ERROR, f"{prefix}.{key}[{index}]",
                               "nonnegative integer", entry,
                               f"phase controller {key} entry is malformed")
        elif key == "phase_weights":
            for index, entry in enumerate(value):
                parsed = _as_float(entry)
                if parsed is None or parsed < 0:
                    report.add(SEVERITY_ERROR, f"{prefix}.{key}[{index}]",
                               "finite nonnegative number", entry,
                               f"phase controller {key} entry is malformed")

    if control == "fixed_phase_strat":
        for key in ("guarded_safety", "risk_gamma", "max_consecutive_skips"):
            if key in cfg:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}", "absent",
                           cfg[key],
                           f"fixed_phase_strat must not carry {key}")
        return

    risk_gamma = _as_float(cfg.get("risk_gamma"))
    if risk_gamma is None or risk_gamma < 0:
        report.add(SEVERITY_ERROR, f"{prefix}.risk_gamma",
                   "finite nonnegative number", cfg.get("risk_gamma"),
                   "guarded phase controller risk_gamma is malformed")

    max_skips = _as_int(cfg.get("max_consecutive_skips"))
    if max_skips is None or max_skips < 1:
        report.add(SEVERITY_ERROR, f"{prefix}.max_consecutive_skips",
                   "integer >= 1", cfg.get("max_consecutive_skips"),
                   "guarded phase controller max_consecutive_skips is "
                   "malformed")

    raw_safety = cfg.get("guarded_safety")
    if not isinstance(raw_safety, dict):
        report.add(SEVERITY_ERROR, f"{prefix}.guarded_safety",
                   "JSON object", raw_safety,
                   "guarded phase controller guarded_safety missing or "
                   "malformed")
        return

    safety_prefix = f"{prefix}.guarded_safety"
    for key in ("use_rho_vg", "use_safety_horizon"):
        if not isinstance(raw_safety.get(key), bool):
            report.add(SEVERITY_ERROR, f"{safety_prefix}.{key}", "boolean",
                       raw_safety.get(key),
                       f"guarded_safety {key} must be a boolean")
    if _as_float(raw_safety.get("rho_veto_threshold")) is None:
        report.add(SEVERITY_ERROR, f"{safety_prefix}.rho_veto_threshold",
                   "finite number", raw_safety.get("rho_veto_threshold"),
                   "guarded_safety rho_veto_threshold is malformed")
    spike_factor = _as_float(raw_safety.get("spike_factor"))
    if spike_factor is None or spike_factor < 0:
        report.add(SEVERITY_ERROR, f"{safety_prefix}.spike_factor",
                   "finite nonnegative number",
                   raw_safety.get("spike_factor"),
                    "guarded_safety spike_factor is malformed")


_LER_GUIDED_CONTROLS = ("ler_guided_stratified", "ler_guided_stratified_safe")

_LER_GUIDED_CONTROLLER_CLASSES = {
    "ler_guided_stratified": "LERGuidedStratifiedPolicy",
    "ler_guided_stratified_safe": "LERGuidedStratifiedSafetyPolicy",
}

_LER_GUIDED_REQUIRED_FIELDS = (
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

_LER_GUIDED_INT_FIELDS = ("total_steps", "min_step", "policy_seed", "n_phases",
                          "max_consecutive_skips", "probe_interval",
                          "min_ler_observations")

_LER_GUIDED_INT_MINIMUMS = {
    "total_steps": 1,
    "min_step": 0,
    "n_phases": 1,
    "max_consecutive_skips": 1,
    "probe_interval": 1,
    "min_ler_observations": 1,
}

_LER_GUIDED_SAFETY_ONLY_FIELDS = (
    "use_rho_vg_safety",
    "rho_veto_threshold",
    "use_loss_spike_safety",
    "loss_spike_factor",
    "loss_spike_window",
)

_LER_REQUIRED_TRACKER_MODE = "sampled_lagged"
_LER_REQUIRED_TRACKER_TIMING = "post_decision_after_backward"


def _check_phase1_3_ler_controller(report: ValidationReport, data: dict,
                                   run_config: dict) -> None:
    """Validate LER-guided controller provenance for canonical Phase 1.3 arms."""
    control = run_config.get("control")
    if control not in PHASE1_3_CONTROLS:
        return

    raw_identity = data.get("identity_inputs")
    identity_inputs = raw_identity if isinstance(raw_identity, dict) else None
    raw_top_cc = data.get("controller_config")
    top_cc = raw_top_cc if isinstance(raw_top_cc, dict) else None

    sources = {
        "identity_inputs.ler_guided_controller": (
            identity_inputs.get("ler_guided_controller", _MISSING)
            if identity_inputs is not None else _MISSING
        ),
        "controller_config.ler_guided_controller": (
            top_cc.get("ler_guided_controller", _MISSING)
            if top_cc is not None else _MISSING
        ),
        "run_config.ler_guided_controller": run_config.get(
            "ler_guided_controller", _MISSING
        ),
    }

    if control not in _LER_GUIDED_CONTROLS:
        for field_name, value in sources.items():
            if value is not _MISSING:
                report.add(SEVERITY_ERROR, field_name, "absent", value,
                           "stray ler_guided_controller config on "
                           f"non-LER control '{control}'")
        return

    configs = {}
    missing_config = False
    for field_name, value in sources.items():
        if value is _MISSING or not isinstance(value, dict):
            report.add(SEVERITY_ERROR, field_name, "JSON object",
                       None if value is _MISSING else value,
                       "ler_guided_controller config missing or malformed "
                       f"for LER control '{control}'")
            missing_config = True
        else:
            configs[field_name] = value
    if missing_config:
        return

    values = list(configs.values())
    if any(cfg != values[0] for cfg in values[1:]):
        report.add(SEVERITY_ERROR, "ler_guided_controller_equality",
                   "deeply equal ler_guided_controller configs",
                   configs,
                   "ler_guided_controller disagrees across "
                   "identity_inputs, controller_config and run_config")

    prefix = "identity_inputs.ler_guided_controller"
    cfg = configs[prefix]

    for key in _LER_GUIDED_REQUIRED_FIELDS:
        if key not in cfg:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}", "present", None,
                       f"required LER controller field {key} is missing")

    if "control" in cfg and cfg["control"] != control:
        report.add(SEVERITY_ERROR, f"{prefix}.control", control,
                   cfg["control"],
                   "LER controller control disagrees with the "
                   "effective control")
    if "policy_name" in cfg and cfg["policy_name"] != control:
        report.add(SEVERITY_ERROR, f"{prefix}.policy_name", control,
                   cfg["policy_name"],
                   "LER controller policy_name disagrees with the "
                   "effective control")

    expected_class = _LER_GUIDED_CONTROLLER_CLASSES[control]
    if "policy_class" in cfg and cfg["policy_class"] != expected_class:
        report.add(SEVERITY_ERROR, f"{prefix}.policy_class",
                   expected_class, cfg["policy_class"],
                   f"unexpected LER controller class for '{control}'")

    if "target_skip_rate" in cfg:
        target_rate = _as_float(cfg["target_skip_rate"])
        if target_rate is None or not 0.0 <= target_rate <= 1.0:
            report.add(SEVERITY_ERROR, f"{prefix}.target_skip_rate",
                       "finite number in [0, 1]", cfg["target_skip_rate"],
                       "LER controller target_skip_rate is malformed")
        else:
            identity_rate = _as_float(identity_inputs.get("target_skip_rate"))
            if identity_rate != target_rate:
                report.add(SEVERITY_ERROR, f"{prefix}.target_skip_rate",
                           identity_inputs.get("target_skip_rate"),
                           target_rate,
                           "LER controller target_skip_rate disagrees "
                           "with identity_inputs.target_skip_rate")

    parsed_ints = {}
    for key in _LER_GUIDED_INT_FIELDS:
        if key not in cfg:
            continue
        value = cfg[key]
        if isinstance(value, str) or isinstance(value, bool):
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       "integer (not boolean)", value,
                       f"LER controller {key} is malformed")
            continue
        parsed = _as_int(value)
        if parsed is None:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       "integer (not boolean)", value,
                       f"LER controller {key} is malformed")
        else:
            parsed_ints[key] = parsed

    for key in ("total_steps", "policy_seed"):
        if key not in parsed_ints:
            continue
        if _as_int(identity_inputs.get(key)) != parsed_ints[key]:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       identity_inputs.get(key), parsed_ints[key],
                       f"LER controller {key} disagrees with "
                       f"identity_inputs.{key}")

    for key, minimum in _LER_GUIDED_INT_MINIMUMS.items():
        if key in parsed_ints and parsed_ints[key] < minimum:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                       f">= {minimum}", parsed_ints[key],
                       f"LER controller {key} is out of range")

    n_phases = parsed_ints.get("n_phases")
    if "phase_weights" in cfg:
        weights = cfg["phase_weights"]
        if not isinstance(weights, list):
            report.add(SEVERITY_ERROR, f"{prefix}.phase_weights", "list",
                       weights, "LER controller phase_weights must be a list")
        else:
            if (n_phases is not None and n_phases >= 1
                    and len(weights) != n_phases):
                report.add(SEVERITY_ERROR, f"{prefix}.phase_weights",
                           f"length {n_phases}", len(weights),
                           "LER controller phase_weights has the wrong "
                           "length")
            for index, entry in enumerate(weights):
                parsed = _as_float(entry)
                if parsed is None or parsed < 0:
                    report.add(SEVERITY_ERROR,
                               f"{prefix}.phase_weights[{index}]",
                               "finite nonnegative number", entry,
                               "LER controller phase_weights entry is "
                               "malformed")

    if "ler_guidance_strength" in cfg:
        strength = _as_float(cfg["ler_guidance_strength"])
        if strength is None or strength < 0:
            report.add(SEVERITY_ERROR, f"{prefix}.ler_guidance_strength",
                       "finite nonnegative number",
                       cfg["ler_guidance_strength"],
                       "LER controller ler_guidance_strength is malformed")

    if ("required_tracker_mode" in cfg
            and cfg["required_tracker_mode"] != _LER_REQUIRED_TRACKER_MODE):
        report.add(SEVERITY_ERROR, f"{prefix}.required_tracker_mode",
                   _LER_REQUIRED_TRACKER_MODE, cfg["required_tracker_mode"],
                   "LER controller required_tracker_mode is not the "
                   "canonical tracker mode")
    if ("required_tracker_timing" in cfg
            and cfg["required_tracker_timing"] != _LER_REQUIRED_TRACKER_TIMING):
        report.add(SEVERITY_ERROR, f"{prefix}.required_tracker_timing",
                   _LER_REQUIRED_TRACKER_TIMING,
                   cfg["required_tracker_timing"],
                   "LER controller required_tracker_timing is not the "
                   "canonical tracker timing")

    expected_safety = control == "ler_guided_stratified_safe"
    if "safety_enabled" in cfg:
        safety_enabled = cfg["safety_enabled"]
        if not isinstance(safety_enabled, bool):
            report.add(SEVERITY_ERROR, f"{prefix}.safety_enabled", "boolean",
                       safety_enabled,
                       "LER controller safety_enabled must be a boolean")
        elif safety_enabled is not expected_safety:
            report.add(SEVERITY_ERROR, f"{prefix}.safety_enabled",
                       expected_safety, safety_enabled,
                       "LER controller safety_enabled disagrees with the "
                       f"effective control '{control}'")

    if not expected_safety:
        for key in _LER_GUIDED_SAFETY_ONLY_FIELDS:
            if key in cfg:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}", "absent",
                           cfg[key],
                           f"ler_guided_stratified must not carry {key}")
        return

    for key in ("use_rho_vg_safety", "use_loss_spike_safety"):
        if cfg.get(key) is not True:
            report.add(SEVERITY_ERROR, f"{prefix}.{key}", True,
                       cfg.get(key),
                       f"safe LER controller {key} must be boolean true")

    if _as_float(cfg.get("rho_veto_threshold")) is None:
        report.add(SEVERITY_ERROR, f"{prefix}.rho_veto_threshold",
                   "finite number", cfg.get("rho_veto_threshold"),
                   "safe LER controller rho_veto_threshold is malformed")

    spike_factor = _as_float(cfg.get("loss_spike_factor"))
    if spike_factor is None or spike_factor < 0:
        report.add(SEVERITY_ERROR, f"{prefix}.loss_spike_factor",
                   "finite nonnegative number", cfg.get("loss_spike_factor"),
                   "safe LER controller loss_spike_factor is malformed")

    spike_window = cfg.get("loss_spike_window")
    if isinstance(spike_window, str) or isinstance(spike_window, bool):
        report.add(SEVERITY_ERROR, f"{prefix}.loss_spike_window",
                   "integer >= 1 (not boolean)", spike_window,
                   "safe LER controller loss_spike_window is malformed")
    else:
        parsed_window = _as_int(spike_window)
        if parsed_window is None or parsed_window < 1:
            report.add(SEVERITY_ERROR, f"{prefix}.loss_spike_window",
                       "integer >= 1 (not boolean)", spike_window,
                       "safe LER controller loss_spike_window is malformed")


_PHASE1_3_ONLINE_MODE_OFF = "off"
_PHASE1_3_ONLINE_MODE_SAMPLED_LAGGED = "sampled_lagged"
_PHASE1_3_ONLINE_TIMING_NONE = "none"
_PHASE1_3_ONLINE_TIMING_POST_DECISION = "post_decision_after_backward"

_PHASE1_3_ONLINE_EXPECTED = {
    "full_finetune": (_PHASE1_3_ONLINE_MODE_OFF, False,
                      _PHASE1_3_ONLINE_TIMING_NONE),
    "exact_random": (_PHASE1_3_ONLINE_MODE_OFF, False,
                     _PHASE1_3_ONLINE_TIMING_NONE),
    "fixed_phase_strat": (_PHASE1_3_ONLINE_MODE_OFF, False,
                          _PHASE1_3_ONLINE_TIMING_NONE),
    "phase_strat_guarded": (_PHASE1_3_ONLINE_MODE_SAMPLED_LAGGED, True,
                            _PHASE1_3_ONLINE_TIMING_POST_DECISION),
    "ler_guided_stratified": (_PHASE1_3_ONLINE_MODE_SAMPLED_LAGGED, True,
                              _PHASE1_3_ONLINE_TIMING_POST_DECISION),
    "ler_guided_stratified_safe": (_PHASE1_3_ONLINE_MODE_SAMPLED_LAGGED,
                                   True,
                                   _PHASE1_3_ONLINE_TIMING_POST_DECISION),
}

_PHASE1_3_ONLINE_CONFIG_FIELDS = (
    "requested_mode",
    "mode",
    "enabled",
    "timing",
    "parameter_sample_size",
    "update_interval",
    "reason",
    "sample_seed",
)

_PHASE1_3_ONLINE_RUNTIME_COUNTER_FIELDS = (
    "parameter_sample_size_realized",
    "update_attempts",
    "update_successes",
    "n_updates",
    "n_decisions",
)

_PHASE1_3_ONLINE_RUNTIME_OPTIONAL_FIELDS = (
    "last_update_decision",
    "observation_age_decisions",
)

_PHASE1_3_ONLINE_RUNTIME_FIELDS = (
    _PHASE1_3_ONLINE_CONFIG_FIELDS
    + _PHASE1_3_ONLINE_RUNTIME_COUNTER_FIELDS
    + _PHASE1_3_ONLINE_RUNTIME_OPTIONAL_FIELDS
)


def _check_phase1_3_online_diagnostics(report: ValidationReport, data: dict,
                                       run_config: dict) -> None:
    """Validate online-diagnostics provenance and runtime state."""
    control = run_config.get("control")
    if control not in PHASE1_3_CONTROLS:
        return

    expected_mode, expected_enabled, expected_timing = (
        _PHASE1_3_ONLINE_EXPECTED[control]
    )
    is_off = expected_mode == _PHASE1_3_ONLINE_MODE_OFF

    def _strict_int(value) -> Optional[int]:
        return value if type(value) is int else None

    def _same_json_value(left, right) -> bool:
        if type(left) is not type(right):
            return False
        if isinstance(left, dict):
            return (
                left.keys() == right.keys()
                and all(
                    _same_json_value(left[key], right[key]) for key in left
                )
            )
        if isinstance(left, list):
            return (
                len(left) == len(right)
                and all(
                    _same_json_value(left[i], right[i])
                    for i in range(len(left))
                )
            )
        return left == right

    raw_identity = data.get("identity_inputs")
    identity_inputs = raw_identity if isinstance(raw_identity, dict) else None
    raw_top_cc = data.get("controller_config")
    top_cc = raw_top_cc if isinstance(raw_top_cc, dict) else None

    sources = {
        "identity_inputs.online_diagnostics": (
            identity_inputs.get("online_diagnostics", _MISSING)
            if identity_inputs is not None else _MISSING
        ),
        "controller_config.online_diagnostics": (
            top_cc.get("online_diagnostics", _MISSING)
            if top_cc is not None else _MISSING
        ),
        "run_config.online_diagnostics": run_config.get(
            "online_diagnostics", _MISSING
        ),
    }

    configs = {}
    for field_name, value in sources.items():
        if value is _MISSING or not isinstance(value, dict):
            report.add(SEVERITY_ERROR, field_name, "JSON object",
                       None if value is _MISSING else value,
                       "online_diagnostics config missing or malformed for "
                       f"canonical Phase 1.3 control '{control}'")
        else:
            configs[field_name] = value

    values = list(configs.values())
    if len(values) > 1 and any(
        not _same_json_value(cfg, values[0]) for cfg in values[1:]
    ):
        report.add(SEVERITY_ERROR, "online_diagnostics_config_equality",
                   "deeply equal online_diagnostics configs", configs,
                   "online_diagnostics disagrees across identity_inputs, "
                   "controller_config and run_config")

    prefix = "identity_inputs.online_diagnostics"
    cfg = configs.get(prefix)

    if cfg is not None:
        for key in _PHASE1_3_ONLINE_CONFIG_FIELDS:
            if key not in cfg:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}", "present",
                           None,
                           f"required online diagnostics field {key} is "
                           "missing")

        for key in ("requested_mode", "reason"):
            if key in cfg:
                value = cfg[key]
                if not isinstance(value, str) or not value:
                    report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                               "nonempty string", value,
                               f"online diagnostics {key} must be a "
                               "nonempty string")

        if "enabled" in cfg:
            enabled = cfg["enabled"]
            if not isinstance(enabled, bool):
                report.add(SEVERITY_ERROR, f"{prefix}.enabled", "boolean",
                           enabled,
                           "online diagnostics enabled must be a boolean")
            elif enabled is not expected_enabled:
                report.add(SEVERITY_ERROR, f"{prefix}.enabled",
                           expected_enabled, enabled,
                           "online diagnostics enabled disagrees with the "
                           f"effective control '{control}'")

        if "mode" in cfg and cfg["mode"] != expected_mode:
            report.add(SEVERITY_ERROR, f"{prefix}.mode", expected_mode,
                       cfg["mode"],
                       "online diagnostics mode is not the canonical mode "
                       f"for control '{control}'")
        if "timing" in cfg and cfg["timing"] != expected_timing:
            report.add(SEVERITY_ERROR, f"{prefix}.timing", expected_timing,
                       cfg["timing"],
                       "online diagnostics timing is not the canonical "
                       f"timing for control '{control}'")

        for key in ("parameter_sample_size", "update_interval"):
            if key not in cfg:
                continue
            parsed = _strict_int(cfg[key])
            if parsed is None:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}",
                           "integer (not boolean or string)", cfg[key],
                           f"online diagnostics {key} is malformed")
            elif is_off and parsed != 0:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}", 0, parsed,
                           f"off-mode online diagnostics {key} must be "
                           "zero")
            elif not is_off and parsed < 1:
                report.add(SEVERITY_ERROR, f"{prefix}.{key}", ">= 1",
                           parsed,
                           f"sampled-lagged online diagnostics {key} must "
                           "be at least 1")

        if "sample_seed" in cfg:
            seed = cfg["sample_seed"]
            if is_off:
                if seed is not None:
                    report.add(SEVERITY_ERROR, f"{prefix}.sample_seed",
                               None, seed,
                               "off-mode online diagnostics sample_seed "
                               "must be None")
            elif _strict_int(seed) is None:
                report.add(SEVERITY_ERROR, f"{prefix}.sample_seed",
                           "integer (not boolean or string)", seed,
                           "sampled-lagged online diagnostics sample_seed "
                           "must be an integer")

    rt_prefix = "online_diagnostics"
    raw_runtime = data.get("online_diagnostics", _MISSING)
    if raw_runtime is _MISSING or not isinstance(raw_runtime, dict):
        report.add(SEVERITY_ERROR, rt_prefix, "JSON object",
                   None if raw_runtime is _MISSING else raw_runtime,
                   "top-level online_diagnostics missing or malformed for "
                   f"canonical Phase 1.3 control '{control}'")
        return
    runtime = raw_runtime

    for key in _PHASE1_3_ONLINE_RUNTIME_FIELDS:
        if key not in runtime:
            report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}", "present",
                       None,
                       f"required runtime online diagnostics field {key} "
                       "is missing")

    if cfg is not None:
        for key in _PHASE1_3_ONLINE_CONFIG_FIELDS:
            if (key in runtime and key in cfg
                    and not _same_json_value(runtime[key], cfg[key])):
                report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}", cfg[key],
                           runtime[key],
                           f"runtime online diagnostics {key} disagrees "
                           "with the provenance config")

    counters = {}
    for key in _PHASE1_3_ONLINE_RUNTIME_COUNTER_FIELDS:
        if key not in runtime:
            continue
        parsed = _strict_int(runtime[key])
        if parsed is None or parsed < 0:
            report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}",
                       "nonnegative integer (not boolean or string)",
                       runtime[key],
                       f"runtime online diagnostics {key} is malformed")
        else:
            counters[key] = parsed

    optionals = {}
    for key in _PHASE1_3_ONLINE_RUNTIME_OPTIONAL_FIELDS:
        if key not in runtime:
            continue
        value = runtime[key]
        if value is None:
            optionals[key] = None
            continue
        parsed = _strict_int(value)
        if parsed is None or parsed < 0:
            report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}",
                       "None or nonnegative integer", value,
                       f"runtime online diagnostics {key} is malformed")
        else:
            optionals[key] = parsed

    if is_off:
        for key in _PHASE1_3_ONLINE_RUNTIME_COUNTER_FIELDS:
            if key in counters and counters[key] != 0:
                report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}", 0,
                           counters[key],
                           f"off-mode online diagnostics {key} must be "
                           "zero")
        for key in _PHASE1_3_ONLINE_RUNTIME_OPTIONAL_FIELDS:
            if optionals.get(key) is not None:
                report.add(SEVERITY_ERROR, f"{rt_prefix}.{key}", None,
                           optionals[key],
                           f"off-mode online diagnostics {key} must be "
                           "None")
        return

    attempts = counters.get("update_attempts")
    successes = counters.get("update_successes")
    if (attempts is not None and successes is not None
            and successes > attempts):
        report.add(SEVERITY_ERROR, f"{rt_prefix}.update_successes",
                   f"<= {attempts}", successes,
                   "online diagnostics update_successes exceeds "
                   "update_attempts")
    n_updates = counters.get("n_updates")
    if (successes is not None and n_updates is not None
            and n_updates != successes):
        report.add(SEVERITY_ERROR, f"{rt_prefix}.n_updates", successes,
                   n_updates,
                   "online diagnostics n_updates must equal "
                   "update_successes")


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

    _check_phase1_3_base_identity(report, data, run_config)
    _check_phase1_3_phase_controller(report, data, run_config)
    _check_phase1_3_ler_controller(report, data, run_config)
    _check_phase1_3_online_diagnostics(report, data, run_config)

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

    scheduler_policy = _resolve_scheduler_step_policy(
        report, data, run_config, instr
    )
    _check_existing_boolean_invariants(
        report, diag, instr, scheduler_policy
    )
    _check_scheduler_step_counts(report, instr, scheduler_policy)
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
        _require_scheduler_invariant(report, instr, scheduler_policy)
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
