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
_RUNTIME_ONLY_CONTROLLER_KEYS = frozenset(
    {
        "policy_effective_config",
        "runtime_quota_total_steps",
    }
)


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

    try:
        base = os.fspath(base_output_dir) if base_output_dir is not None else None
    except TypeError as exc:
        _add_error(findings, "base_output_dir", None, f"base_output_dir is not a path: {exc}")
        raise CompletedMatrixError(findings)

    if not isinstance(base, str) or not base:
        _add_error(findings, "base_output_dir", None, "base_output_dir must resolve to a non-empty string path")
        raise CompletedMatrixError(findings)

    plan_valid = True
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
        plan_valid = False
    except (TypeError, ValueError) as exc:
        _add_error(findings, "plan", None, f"plan validation failed: {exc}")
        plan_valid = False

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
        plan_valid = False

    if not plan_valid:
        if findings:
            raise CompletedMatrixError(findings)
        return {"valid_runs": [], "findings": findings}

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

            if manifest.get("provenance_classification") != "matched_claim":
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.provenance_classification",
                    "cell": cell_id,
                    "message": f"{attempt_name}: provenance classification is not matched_claim",
                })

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

            manifest_controller = manifest.get("controller_config_effective")
            if not isinstance(manifest_controller, dict):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.controller_config_effective",
                    "cell": cell_id,
                    "message": f"{attempt_name}: controller_config_effective is not an object",
                })
            else:
                planned_controller = cell.get("controller_config") or {}
                if not isinstance(planned_controller, dict):
                    planned_controller = {}
                manifest_keys = set(manifest_controller.keys())
                planned_keys = set(planned_controller.keys())
                unexpected_keys = manifest_keys - planned_keys - _RUNTIME_ONLY_CONTROLLER_KEYS
                for key in sorted(unexpected_keys):
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"manifest.controller_config_effective.{key}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: unexpected controller field {key!r}",
                    })
                for key, planned_value in planned_controller.items():
                    if key not in manifest_controller:
                        attempt_findings.append({
                            "severity": "error",
                            "field": f"manifest.controller_config_effective.{key}",
                            "cell": cell_id,
                            "message": f"{attempt_name}: missing controller field {key!r}",
                        })
                        continue
                    actual_value = manifest_controller[key]
                    if not _strict_equal(planned_value, actual_value):
                        attempt_findings.append({
                            "severity": "error",
                            "field": f"manifest.controller_config_effective.{key}",
                            "cell": cell_id,
                            "message": f"{attempt_name}: controller field {key} drift {actual_value!r} != {planned_value!r}",
                        })

            manifest_run = manifest.get("run")
            if not isinstance(manifest_run, dict):
                manifest_run = {}
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.run",
                    "cell": cell_id,
                    "message": f"{attempt_name}: manifest run is not an object",
                })

            run_fields = (
                ("task", cell.get("task")),
                ("seed", cell.get("training_seed")),
                ("target_skip_rate", cell.get("target_skip_rate")),
                ("controller_name", cell.get("controller_config", {}).get("policy_class")),
                ("controller_seed", cell.get("policy_seed")),
                ("model_id", cell.get("model_id")),
                ("model_revision", cell.get("model_revision")),
                ("planned_quota", cell.get("requested_quota")),
                ("total_steps", cell.get("total_steps")),
                ("skip_update_mode", cell.get("skip_update_mode")),
                ("matched_budget_planned", cell.get("matched_budget")),
            )
            for manifest_key, planned_value in run_fields:
                if manifest_key not in manifest_run:
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"manifest.run.{manifest_key}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: missing run field {manifest_key!r}",
                    })
                    continue
                if not _strict_equal(manifest_run[manifest_key], planned_value):
                    attempt_findings.append({
                        "severity": "error",
                        "field": f"manifest.run.{manifest_key}",
                        "cell": cell_id,
                        "message": f"{attempt_name}: {manifest_key} drift {manifest_run[manifest_key]!r} != {planned_value!r}",
                    })

            manifest_attempt = manifest.get("attempt")
            if not _strict_equal(manifest_attempt, attempt_num):
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.attempt",
                    "cell": cell_id,
                    "message": f"{attempt_name}: manifest attempt {manifest_attempt!r} != directory attempt {attempt_num}",
                })

            expected_output_paths = _expected_output_paths_for_cell(cell)
            manifest_output_paths = manifest.get("output_paths")
            if not isinstance(manifest_output_paths, dict):
                manifest_output_paths = {}
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.output_paths",
                    "cell": cell_id,
                    "message": f"{attempt_name}: output_paths is not an object",
                })
            if manifest_output_paths != expected_output_paths:
                attempt_findings.append({
                    "severity": "error",
                    "field": "manifest.output_paths",
                    "cell": cell_id,
                    "message": f"{attempt_name}: output_paths mismatch {manifest_output_paths!r} != {expected_output_paths!r}",
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

            if not _strict_equal(results.get("attempt"), attempt_num):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.attempt",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results attempt drift",
                })

            if not _strict_equal(results.get("task"), cell.get("task")):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.task",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results task drift",
                })

            if not _strict_equal(results.get("seed"), cell.get("training_seed")):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.seed",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results seed drift",
                })

            if not _strict_equal(results.get("ablation"), cell.get("arm")):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.ablation",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results ablation drift",
                })

            if not _strict_equal(results.get("model_revision"), cell.get("model_revision")):
                attempt_findings.append({
                    "severity": "error",
                    "field": "results.json.model_revision",
                    "cell": cell_id,
                    "message": f"{attempt_name}: results model_revision drift",
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

    plan_order = [cell_id for _, _, cell_id in plan_cells]
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
