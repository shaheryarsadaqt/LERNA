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

_ATTEMPT_PATTERN = re.compile(r"attempt-(\d{3})\Z")


class CompletedMatrixError(ValueError):
    """Raised once with every structured completed-matrix validation finding."""

    def __init__(self, findings: list[dict[str, Any]]):
        self.findings = list(findings)
        count = sum(finding.get("severity") == "error" for finding in findings)
        super().__init__(
            f"Phase 1.3 completed matrix validation failed with {count} error(s)"
        )


def _required_artifacts_for_cell(cell: dict[str, Any]) -> list[str]:
    online = cell.get("online_diagnostics") or {}
    if isinstance(online, dict) and online.get("enabled"):
        return ["instrumentation.json", "ler_diagnostics.json"]
    return ["instrumentation.json"]


def _expected_output_paths_for_cell(cell: dict[str, Any]) -> dict[str, str]:
    online = cell.get("online_diagnostics") or {}
    output_paths = {
        "results": "results.json",
        "instrumentation": "instrumentation.json",
        "manifest": "run_manifest.json",
    }
    if isinstance(online, dict) and online.get("enabled"):
        output_paths["ler_diagnostics"] = "ler_diagnostics.json"
    return output_paths


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
    except (TypeError, ValueError) as exc:
        _add_error(findings, "plan", None, f"plan validation failed: {exc}")

    plan_cells: list[tuple[int, Any, tuple[Any, Any, Any, Any] | None]] = []
    plan_map: dict[tuple[Any, Any, Any, Any], dict[str, Any]] = {}
    try:
        for index, cell in enumerate(plan):
            cell_id = _cell_ref(cell)
            plan_cells.append((index, cell, cell_id))
            if cell_id is not None and isinstance(cell, dict):
                plan_map[cell_id] = cell
    except TypeError as exc:
        _add_error(findings, "plan", None, f"plan is not iterable: {exc}")

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
                    attempt_num = int(m.group(1))
                    if attempt_num < 1:
                        _add_error(findings, "attempts", cell_id, f"non-positive attempt number: {name}")
                        continue
                    attempt_entries.append((name, attempt_num, path))

        if not attempt_entries:
            _add_error(findings, "attempts", cell_id, "no attempt directories found")
            continue

        attempt_entries.sort(key=lambda entry: entry[1])

        valid_attempts: list[tuple[str, str, dict[str, Any]]] = []
        running_attempts: list[str] = []
        failed_attempts: list[str] = []

        for attempt_name, attempt_num, attempt_path in attempt_entries:
            attempt_findings: list[dict[str, Any]] = []

            manifest_file = os.path.join(attempt_path, "run_manifest.json")
            if not os.path.isfile(manifest_file):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest",
                    "cell": cell_id,
                    "message": f"missing run_manifest.json in {attempt_name}",
                })
                findings.extend(attempt_findings)
                continue

            try:
                with open(manifest_file, "r", encoding="utf-8") as handle:
                    manifest = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest",
                    "cell": cell_id,
                    "message": f"unreadable run_manifest.json in {attempt_name}: {exc}",
                })
                findings.extend(attempt_findings)
                continue

            if not isinstance(manifest, dict):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest",
                    "cell": cell_id,
                    "message": f"run_manifest.json is not an object in {attempt_name}",
                })
                findings.extend(attempt_findings)
                continue

            status = manifest.get("status")
            if status == "running":
                running_attempts.append(attempt_name)
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.status",
                    "cell": cell_id,
                    "message": f"running attempt found: {attempt_name}",
                })
                findings.extend(attempt_findings)
                continue
            if status == "failed":
                failed_attempts.append(attempt_name)
                continue
            if status != "completed":
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.status",
                    "cell": cell_id,
                    "message": f"unexpected manifest status {status!r} in {attempt_name}",
                })
                findings.extend(attempt_findings)
                continue

            verification = verify_completed_manifest(attempt_path)
            if not verification["ok"]:
                for err in verification["errors"]:
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"manifest.verification.{err['field']}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: {err['message']}",
                    })

            manifest_fingerprint = manifest.get("fingerprint")
            if manifest_fingerprint != fingerprint:
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.fingerprint",
                    "cell": cell_id,
                    "message": f"{attempt_name}: fingerprint drift {manifest_fingerprint!r} != {fingerprint!r}",
                })

            manifest_identity = manifest.get("identity_inputs")
            plan_identity = cell.get("identity_inputs")
            if not _strict_equal(manifest_identity, plan_identity):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.identity_inputs",
                    "cell": cell_id,
                    "message": f"{attempt_name}: identity drift",
                })

            manifest_run = manifest.get("run") or {}
            if manifest_run.get("task") != cell.get("task"):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run.task",
                    "cell": cell_id,
                    "message": f"{attempt_name}: task drift",
                })
            if manifest_run.get("seed") != cell.get("training_seed"):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run.seed",
                    "cell": cell_id,
                    "message": f"{attempt_name}: training_seed drift",
                })
            if manifest_run.get("target_skip_rate") != cell.get("target_skip_rate"):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run.target_skip_rate",
                    "cell": cell_id,
                    "message": f"{attempt_name}: target_skip_rate drift",
                })

            expected_policy_class = cell.get("controller_config", {}).get("policy_class")
            if expected_policy_class is not None and manifest_run.get("controller_name") != expected_policy_class:
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run.controller_name",
                    "cell": cell_id,
                    "message": f"{attempt_name}: controller_name drift {manifest_run.get('controller_name')!r} != {expected_policy_class!r}",
                })

            if manifest_run.get("controller_seed") != cell.get("policy_seed"):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run.controller_seed",
                    "cell": cell_id,
                    "message": f"{attempt_name}: policy_seed drift",
                })

            manifest_attempt = manifest.get("attempt")
            if manifest_attempt != attempt_num:
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.attempt",
                    "cell": cell_id,
                    "message": f"{attempt_name}: manifest attempt {manifest_attempt!r} != directory attempt {attempt_num}",
                })

            expected_output_paths = _expected_output_paths_for_cell(cell)
            manifest_output_paths = manifest.get("output_paths") or {}
            for logical_name, expected_filename in expected_output_paths.items():
                actual_filename = manifest_output_paths.get(logical_name)
                if actual_filename != expected_filename:
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"manifest.output_paths.{logical_name}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: expected {expected_filename!r}, got {actual_filename!r}",
                    })

            results_file = os.path.join(attempt_path, "results.json")
            if not os.path.isfile(results_file):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json",
                    "cell": cell_id,
                    "message": f"{attempt_name}: missing results.json",
                })
                findings.extend(attempt_findings)
                continue

            try:
                with open(results_file, "r", encoding="utf-8") as handle:
                    results = json.load(handle)
            except (json.JSONDecodeError, OSError) as exc:
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json",
                    "cell": cell_id,
                    "message": f"{attempt_name}: unreadable results.json: {exc}",
                })
                findings.extend(attempt_findings)
                continue

            if not isinstance(results, dict):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results.json is not an object",
                })
                findings.extend(attempt_findings)
                continue

            if results.get("fingerprint") != fingerprint:
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.fingerprint",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results fingerprint drift",
                })

            if results.get("attempt") != attempt_num:
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.attempt",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results attempt drift",
                })

            results_identity = results.get("identity_inputs")
            if not _strict_equal(results_identity, plan_identity):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.identity_inputs",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results identity drift",
                })

            required_artifacts = _required_artifacts_for_cell(cell)
            try:
                validation_report = validate_skip_results(
                    Path(results_file),
                    required_artifacts=required_artifacts,
                )
            except Exception as exc:
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.validation",
                    "cell": cell_id,
                    "message": f"{attempt_name}: validator raised {type(exc).__name__}: {exc}",
                })
                findings.extend(attempt_findings)
                continue

            if not validation_report.ok:
                for err in validation_report.findings:
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"results.json.validation.{err.field}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: {err.message}",
                    })

            if validation_report.ok and not validation_report.valid_for_matched_budget:
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.valid_for_matched_budget",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results are not valid_for_matched_budget",
                })

            if attempt_findings:
                findings.extend(attempt_findings)
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
