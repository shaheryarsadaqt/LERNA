"""Read-only validation for a completed Phase 1.3 six-arm matrix."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

_THIS_DIR = Path(__file__).resolve().parent
_MATRIX_PATH = _THIS_DIR / "phase1_3_matrix.py"
_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "lerna_phase1_3_matrix_for_completed",
    _MATRIX_PATH,
)
if _MATRIX_SPEC is None or _MATRIX_SPEC.loader is None:
    raise ImportError("could not load lerna.utils.phase1_3_matrix")
_MATRIX_MODULE = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(_MATRIX_MODULE)

PHASE1_3_CANONICAL_ARMS = _MATRIX_MODULE.PHASE1_3_CANONICAL_ARMS
MatrixPlanError = _MATRIX_MODULE.MatrixPlanError
validate_phase1_3_matrix_plan = _MATRIX_MODULE.validate_phase1_3_matrix_plan
_cell_ref = _MATRIX_MODULE._cell_ref
_is_nonempty_str = _MATRIX_MODULE._is_nonempty_str
_is_int = _MATRIX_MODULE._is_int
_is_float = _MATRIX_MODULE._is_float
_strict_equal = _MATRIX_MODULE._strict_equal
_add_error = _MATRIX_MODULE._add_error
_FINGERPRINT_PATTERN = _MATRIX_MODULE._FINGERPRINT_PATTERN
build_scientific_fingerprint = _MATRIX_MODULE.build_scientific_fingerprint

_PROVENANCE_PATH = _THIS_DIR / "run_provenance.py"
_PROVENANCE_SPEC = importlib.util.spec_from_file_location(
    "lerna_run_provenance_for_completed",
    _PROVENANCE_PATH,
)
if _PROVENANCE_SPEC is None or _PROVENANCE_SPEC.loader is None:
    raise ImportError("could not load lerna.utils.run_provenance")
_PROVENANCE_MODULE = importlib.util.module_from_spec(_PROVENANCE_SPEC)
_PROVENANCE_SPEC.loader.exec_module(_PROVENANCE_MODULE)
verify_completed_manifest = _PROVENANCE_MODULE.verify_completed_manifest

_VALIDATOR_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_skip_policy_results.py"
_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_skip_policy_results_for_completed_matrix",
    _VALIDATOR_PATH,
)
if _VALIDATOR_SPEC is None or _VALIDATOR_SPEC.loader is None:
    raise ImportError("could not load validate_skip_policy_results")
_VALIDATOR_MODULE = importlib.util.module_from_spec(_VALIDATOR_SPEC)
sys.modules[_VALIDATOR_SPEC.name] = _VALIDATOR_MODULE
_VALIDATOR_SPEC.loader.exec_module(_VALIDATOR_MODULE)
validate_skip_results = _VALIDATOR_MODULE.validate_results

_ATTEMPT_PATTERN = re.compile(r"attempt-(\d+)\Z")


class CompletedMatrixError(ValueError):
    """Raised once with every structured completed-matrix validation finding."""

    def __init__(self, findings: list[dict[str, Any]]):
        self.findings = list(findings)
        count = sum(finding.get("severity") == "error" for finding in findings)
        super().__init__(
            f"Phase 1.3 completed matrix validation failed with {count} error(s)"
        )


def validate_phase1_3_completed_matrix(
    plan,
    *,
    tasks,
    seeds,
    target_skip_rates,
    minimum_seed_count,
    base_output_dir,
) -> dict:
    findings: list[dict[str, Any]] = []
    valid_runs: list[dict[str, Any]] = []

    base = os.fspath(base_output_dir) if base_output_dir is not None else None
    if not isinstance(base, str) or not base:
        _add_error(findings, "base_output_dir", None, "base_output_dir must resolve to a non-empty string path")
        raise CompletedMatrixError(findings)

    try:
        validate_phase1_3_matrix_plan(
            plan,
            tasks=tasks,
            seeds=seeds,
            target_skip_rates=target_skip_rates,
            minimum_seed_count=minimum_seed_count,
            base_output_dir=base_output_dir,
        )
    except MatrixPlanError as exc:
        findings.extend(exc.findings)

    plan_map: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    for cell in plan:
        cell_id = _cell_ref(cell)
        if cell_id is not None and isinstance(cell, dict):
            plan_map[cell_id] = cell

    expected_dirs: set[str] = set()
    for cell_id, cell in plan_map.items():
        arm = cell.get("arm")
        fingerprint = cell.get("fingerprint")
        if _is_nonempty_str(arm) and _is_nonempty_str(fingerprint):
            expected_dirs.add(os.path.normpath(os.path.join(base, arm, fingerprint)))

    if os.path.isdir(base):
        for arm_name in os.listdir(base):
            arm_path = os.path.join(base, arm_name)
            if not os.path.isdir(arm_path):
                continue
            for fp_name in os.listdir(arm_path):
                fp_path = os.path.join(arm_path, fp_name)
                if not os.path.isdir(fp_path):
                    continue
                if _ATTEMPT_PATTERN.match(fp_name):
                    _add_error(findings, "filesystem", None, f"malformed attempt path (missing fingerprint): {fp_path}")
                    continue
                for attempt_name in os.listdir(fp_path):
                    if not _ATTEMPT_PATTERN.match(attempt_name):
                        _add_error(findings, "filesystem", None, f"malformed attempt path: {os.path.join(fp_path, attempt_name)}")
                if os.path.normpath(fp_path) not in expected_dirs:
                    _add_error(findings, "filesystem", None, f"unexpected cell directory: {fp_path}")

    for cell_id, cell in plan_map.items():
        arm = cell.get("arm")
        fingerprint = cell.get("fingerprint")
        if not _is_nonempty_str(arm) or not _FINGERPRINT_PATTERN.fullmatch(fingerprint or ""):
            continue

        cell_dir = os.path.join(base, arm, fingerprint)
        if not os.path.isdir(cell_dir):
            _add_error(findings, "cell_directory", cell_id, f"missing cell directory: {cell_dir}")
            continue

        attempt_entries: list[tuple[str, int, str]] = []
        for name in os.listdir(cell_dir):
            path = os.path.join(cell_dir, name)
            if os.path.isdir(path):
                m = _ATTEMPT_PATTERN.match(name)
                if m:
                    attempt_entries.append((name, int(m.group(1)), path))

        if not attempt_entries:
            _add_error(findings, "attempts", cell_id, "no attempt directories found")
            continue

        attempt_entries.sort(key=lambda entry: entry[1])

        valid_attempts: list[tuple[str, str, dict[str, Any]]] = []
        running_attempts: list[str] = []
        failed_attempts: list[str] = []

        for attempt_name, attempt_num, attempt_path in attempt_entries:
            manifest_file = os.path.join(attempt_path, "run_manifest.json")
            if not os.path.isfile(manifest_file):
                _add_error(findings, "manifest", cell_id, f"missing run_manifest.json in {attempt_name}")
                continue

            try:
                with open(manifest_file, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                _add_error(findings, "manifest", cell_id, f"unreadable run_manifest.json in {attempt_name}: {exc}")
                continue

            if not isinstance(manifest, dict):
                _add_error(findings, "manifest", cell_id, f"run_manifest.json is not an object in {attempt_name}")
                continue

            status = manifest.get("status")
            if status == "running":
                running_attempts.append(attempt_name)
                _add_error(findings, "manifest.status", cell_id, f"running attempt found: {attempt_name}")
                continue
            if status == "failed":
                failed_attempts.append(attempt_name)
                continue
            if status != "completed":
                _add_error(findings, "manifest.status", cell_id, f"unexpected manifest status {status!r} in {attempt_name}")
                continue

            verification = verify_completed_manifest(attempt_path)
            if not verification["ok"]:
                for err in verification["errors"]:
                    _add_error(
                        findings,
                        f"manifest.verification.{err['field']}",
                        cell_id,
                        f"{attempt_name}: {err['message']}",
                    )
                continue

            manifest_fingerprint = manifest.get("fingerprint")
            if manifest_fingerprint != fingerprint:
                _add_error(
                    findings,
                    "manifest.fingerprint",
                    cell_id,
                    f"{attempt_name}: fingerprint drift {manifest_fingerprint!r} != {fingerprint!r}",
                )
                continue

            manifest_identity = manifest.get("identity_inputs")
            plan_identity = cell.get("identity_inputs")
            if not _strict_equal(manifest_identity, plan_identity):
                _add_error(findings, "manifest.identity_inputs", cell_id, f"{attempt_name}: identity drift")
                continue

            manifest_run = manifest.get("run") or {}
            if manifest_run.get("task") != cell.get("task"):
                _add_error(findings, "manifest.run.task", cell_id, f"{attempt_name}: task drift")
                continue
            if manifest_run.get("seed") != cell.get("training_seed"):
                _add_error(findings, "manifest.run.seed", cell_id, f"{attempt_name}: training_seed drift")
                continue
            if manifest_run.get("target_skip_rate") != cell.get("target_skip_rate"):
                _add_error(findings, "manifest.run.target_skip_rate", cell_id, f"{attempt_name}: target_skip_rate drift")
                continue
            if manifest_run.get("controller_name") != cell.get("arm"):
                _add_error(findings, "manifest.run.controller_name", cell_id, f"{attempt_name}: arm drift")
                continue

            manifest_attempt = manifest.get("attempt")
            if manifest_attempt != attempt_num:
                _add_error(
                    findings,
                    "manifest.attempt",
                    cell_id,
                    f"{attempt_name}: manifest attempt {manifest_attempt!r} != directory attempt {attempt_num}",
                )
                continue

            results_file = os.path.join(attempt_path, "results.json")
            if not os.path.isfile(results_file):
                _add_error(findings, "results.json", cell_id, f"{attempt_name}: missing results.json")
                continue

            try:
                with open(results_file, "r", encoding="utf-8") as handle:
                    results = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                _add_error(findings, "results.json", cell_id, f"{attempt_name}: unreadable results.json: {exc}")
                continue

            if not isinstance(results, dict):
                _add_error(findings, "results.json", cell_id, f"{attempt_name}: results.json is not an object")
                continue

            if results.get("fingerprint") != fingerprint:
                _add_error(
                    findings,
                    "results.json.fingerprint",
                    cell_id,
                    f"{attempt_name}: results fingerprint drift",
                )
                continue
            if results.get("attempt") != attempt_num:
                _add_error(
                    findings,
                    "results.json.attempt",
                    cell_id,
                    f"{attempt_name}: results attempt drift",
                )
                continue

            results_identity = results.get("identity_inputs")
            if not _strict_equal(results_identity, plan_identity):
                _add_error(findings, "results.json.identity_inputs", cell_id, f"{attempt_name}: results identity drift")
                continue

            try:
                validation_report = validate_skip_results(
                    Path(results_file),
                    required_artifacts=["instrumentation.json"],
                )
            except Exception as exc:
                _add_error(findings, "results.json.validation", cell_id, f"{attempt_name}: validator raised {type(exc).__name__}: {exc}")
                continue

            if not validation_report.ok:
                for err in validation_report.findings:
                    _add_error(
                        findings,
                        f"results.json.validation.{err.field}",
                        cell_id,
                        f"{attempt_name}: {err.message}",
                    )
                continue

            valid_attempts.append((attempt_name, results_file, manifest))

        if len(valid_attempts) > 1:
            for attempt_name, _, _ in valid_attempts:
                _add_error(findings, "attempts", cell_id, f"duplicate valid completed attempt: {attempt_name}")
        elif len(valid_attempts) == 1:
            attempt_name, results_file, manifest = valid_attempts[0]
            valid_runs.append({
                "cell": cell,
                "cell_id": cell_id,
                "attempt": attempt_name,
                "attempt_num": int(_ATTEMPT_PATTERN.match(attempt_name).group(1)),
                "results_path": results_file,
                "manifest": manifest,
            })

    plan_order = [_cell_ref(cell) for cell in plan]
    valid_runs.sort(key=lambda run: plan_order.index(run["cell_id"]) if run["cell_id"] in plan_order else len(plan_order))

    missing_cells = [cell_id for cell_id in plan_map if cell_id not in {run["cell_id"] for run in valid_runs}]
    for cell_id in missing_cells:
        _add_error(findings, "cell_completion", cell_id, "no valid completed attempt found")

    if findings:
        raise CompletedMatrixError(findings)

    return {
        "valid_runs": valid_runs,
        "findings": findings,
    }
