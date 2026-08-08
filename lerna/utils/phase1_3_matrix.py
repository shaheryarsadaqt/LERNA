"""Dependency-light validation for the Phase 1.3 six-arm matrix plan."""

from __future__ import annotations

import importlib.util
import math
import os
import re
from itertools import pairwise
from pathlib import Path
from typing import Any

try:
    from .run_provenance import build_scientific_fingerprint
except ImportError:
    _PROVENANCE_SPEC = importlib.util.spec_from_file_location(
        "lerna_run_provenance_for_matrix",
        Path(__file__).with_name("run_provenance.py"),
    )
    if _PROVENANCE_SPEC is None or _PROVENANCE_SPEC.loader is None:
        raise ImportError("could not load lerna.utils.run_provenance")
    _PROVENANCE_MODULE = importlib.util.module_from_spec(_PROVENANCE_SPEC)
    _PROVENANCE_SPEC.loader.exec_module(_PROVENANCE_MODULE)
    build_scientific_fingerprint = (
        _PROVENANCE_MODULE.build_scientific_fingerprint
    )


PHASE1_3_CANONICAL_ARMS: tuple[str, ...] = (
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
)
STRICT_TARGET_SKIP_RATES: tuple[float, float] = (0.30, 0.40)
POLICY_MIN_STEP: int = 50

PHASE1_3_POLICY_CLASSES = {
    "full_finetune": "AlwaysFalsePolicy",
    "exact_random": "RandomSkipPolicy",
    "fixed_phase_strat": "FixedPhaseStratifiedRandomPolicy",
    "phase_strat_guarded": "PhaseStratifiedGuardedRandomPolicy",
    "ler_guided_stratified": "LERGuidedStratifiedPolicy",
    "ler_guided_stratified_safe": "LERGuidedStratifiedSafetyPolicy",
}

_OFFLINE_DIAGNOSTIC_ARMS = frozenset(
    {"full_finetune", "exact_random", "fixed_phase_strat"}
)
_PHASE_ARMS = frozenset({"fixed_phase_strat", "phase_strat_guarded"})
_LER_ARMS = frozenset(
    {"ler_guided_stratified", "ler_guided_stratified_safe"}
)
_FINGERPRINT_PATTERN = re.compile(r"[0-9a-f]{16}\Z")

PLANNED_CELL_REQUIRED_FIELDS = frozenset(
    {
        "arm",
        "control",
        "task",
        "training_seed",
        "policy_seed",
        "model_id",
        "target_skip_rate",
        "num_epochs",
        "total_steps",
        "min_step",
        "requested_quota",
        "planned_skips",
        "is_skipping_arm",
        "matched_budget",
        "no_early_stopping",
        "skip_update_mode",
        "scheduler_step_policy",
        "online_diagnostics",
        "controller_config",
        "identity_inputs",
        "fingerprint",
        "planned_arm_dir",
    }
)

_IDENTITY_REQUIRED_FIELDS = frozenset(
    {
        "task",
        "training_seed",
        "model_id",
        "max_samples_requested",
        "train_samples_realized",
        "eval_samples_realized",
        "train_dataset_fingerprint",
        "eval_dataset_fingerprint",
        "num_epochs",
        "control",
        "target_skip_rate",
        "policy_seed",
        "skip_update_mode",
        "scheduler_step_policy",
        "no_early_stopping",
        "total_steps",
        "git_sha",
        "online_diagnostics",
    }
)

_CONTROLLER_REQUIRED_FIELDS = frozenset(
    {
        "arm",
        "arm_alias_of",
        "control",
        "policy_class",
        "compute_saving_mechanism",
        "policy_seed",
        "target_skip_rate",
        "min_step",
        "configured_total_steps",
        "requested_quota",
        "matched_budget",
        "is_skipping_arm",
        "allow_early_stopping_with_skipping",
        "early_stopping_active",
        "num_epochs",
        "online_diagnostics",
    }
)

_ONLINE_DIAGNOSTIC_FIELDS = frozenset(
    {
        "requested_mode",
        "mode",
        "enabled",
        "timing",
        "parameter_sample_size",
        "update_interval",
        "reason",
        "sample_seed",
    }
)

_PHASE_CONTROLLER_FIELDS = frozenset(
    {
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
    }
)

_LER_CONTROLLER_FIELDS = frozenset(
    {
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
    }
)

_LER_SAFETY_FIELDS = frozenset(
    {
        "use_rho_vg_safety",
        "rho_veto_threshold",
        "use_loss_spike_safety",
        "loss_spike_factor",
        "loss_spike_window",
    }
)

_MISSING = object()


class MatrixPlanError(ValueError):
    """Raised once with every structured matrix-plan validation finding."""

    def __init__(self, findings: list[dict[str, Any]]):
        self.findings = list(findings)
        count = sum(finding.get("severity") == "error" for finding in findings)
        super().__init__(
            f"Phase 1.3 matrix plan validation failed with {count} error(s)"
        )


def _add_error(
    findings: list[dict[str, Any]],
    field: str,
    cell: tuple[Any, Any, Any, Any] | None,
    message: str,
) -> None:
    findings.append(
        {
            "severity": "error",
            "field": field,
            "cell": cell,
            "message": message,
        }
    )


def _is_int(value: Any) -> bool:
    return type(value) is int


def _is_float(value: Any) -> bool:
    return type(value) is float and math.isfinite(value)


def _is_nonempty_str(value: Any) -> bool:
    return type(value) is str and bool(value)


def _strict_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return left.keys() == right.keys() and all(
            _strict_equal(left[key], right[key]) for key in left
        )
    if type(left) is list:
        return len(left) == len(right) and all(
            _strict_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


def _has_duplicates(values: list[Any]) -> bool:
    return any(value in values[:index] for index, value in enumerate(values))


def _cell_ref(cell: Any) -> tuple[Any, Any, Any, Any] | None:
    if type(cell) is not dict:
        return None
    values = (
        cell.get("task"),
        cell.get("training_seed"),
        cell.get("target_skip_rate"),
        cell.get("arm"),
    )
    if not (
        _is_nonempty_str(values[0])
        and _is_int(values[1])
        and _is_float(values[2])
        and _is_nonempty_str(values[3])
    ):
        return None
    return values


def _require_fields(
    value: dict[str, Any],
    required: frozenset[str],
    findings: list[dict[str, Any]],
    cell: tuple[Any, Any, Any, Any] | None,
    prefix: str = "",
) -> None:
    for key in sorted(required):
        if key not in value:
            _add_error(
                findings,
                prefix + key,
                cell,
                f"required field {prefix + key!r} is missing",
            )


def _expect(
    value: dict[str, Any],
    key: str,
    expected: Any,
    findings: list[dict[str, Any]],
    cell: tuple[Any, Any, Any, Any] | None,
    prefix: str = "",
) -> None:
    actual = value.get(key, _MISSING)
    if actual is _MISSING:
        return
    if actual != expected or type(actual) is not type(expected):
        _add_error(
            findings,
            prefix + key,
            cell,
            f"expected {expected!r}, got {actual!r}",
        )


def _validate_dimensions(
    *,
    tasks: Any,
    seeds: Any,
    target_skip_rates: Any,
    minimum_seed_count: Any,
    base_output_dir: Any,
    findings: list[dict[str, Any]],
) -> tuple[bool, str | None]:
    dimensions_valid = True

    if type(tasks) is not list:
        _add_error(findings, "tasks", None, "tasks must be a list")
        dimensions_valid = False
    else:
        if not tasks:
            _add_error(findings, "tasks", None, "tasks must not be empty")
            dimensions_valid = False
        for index, task in enumerate(tasks):
            if not _is_nonempty_str(task):
                _add_error(
                    findings,
                    f"tasks[{index}]",
                    None,
                    "task must be a non-empty string",
                )
                dimensions_valid = False
        if _has_duplicates(tasks):
            _add_error(findings, "tasks", None, "duplicate tasks are not allowed")
            dimensions_valid = False

    if type(seeds) is not list:
        _add_error(findings, "seeds", None, "seeds must be a list")
        dimensions_valid = False
    else:
        if not seeds:
            _add_error(findings, "seeds", None, "seeds must not be empty")
            dimensions_valid = False
        for index, seed in enumerate(seeds):
            if not _is_int(seed):
                _add_error(
                    findings,
                    f"seeds[{index}]",
                    None,
                    "seed must be an integer; bools and floats are invalid",
                )
                dimensions_valid = False
        if _has_duplicates(seeds):
            _add_error(findings, "seeds", None, "duplicate seeds are not allowed")
            dimensions_valid = False

    if type(target_skip_rates) is not list:
        _add_error(
            findings,
            "target_skip_rates",
            None,
            "target_skip_rates must be a list",
        )
        dimensions_valid = False
    else:
        if not target_skip_rates:
            _add_error(
                findings,
                "target_skip_rates",
                None,
                "target_skip_rates must not be empty",
            )
            dimensions_valid = False
        for index, rate in enumerate(target_skip_rates):
            if not _is_float(rate):
                _add_error(
                    findings,
                    f"target_skip_rates[{index}]",
                    None,
                    "target skip rate must be a finite float",
                )
                dimensions_valid = False
            elif not 0.0 <= rate <= 1.0:
                _add_error(
                    findings,
                    f"target_skip_rates[{index}]",
                    None,
                    "target skip rate must be in [0, 1]",
                )
                dimensions_valid = False
        if _has_duplicates(target_skip_rates):
            _add_error(
                findings,
                "target_skip_rates",
                None,
                "duplicate target skip rates are not allowed",
            )
            dimensions_valid = False

    if not _is_int(minimum_seed_count) or minimum_seed_count < 1:
        _add_error(
            findings,
            "minimum_seed_count",
            None,
            "minimum_seed_count must be an integer >= 1",
        )
        dimensions_valid = False
    elif type(seeds) is list:
        unique_valid_seeds = []
        for seed in seeds:
            if _is_int(seed) and seed not in unique_valid_seeds:
                unique_valid_seeds.append(seed)
        if len(unique_valid_seeds) < minimum_seed_count:
            _add_error(
                findings,
                "seeds",
                None,
                f"at least {minimum_seed_count} unique seeds are required",
            )

    try:
        normalized_output_dir = os.fspath(base_output_dir)
    except TypeError:
        normalized_output_dir = None
    if type(normalized_output_dir) is not str or not normalized_output_dir:
        _add_error(
            findings,
            "base_output_dir",
            None,
            "base_output_dir must resolve to a non-empty string path",
        )
        dimensions_valid = False
        normalized_output_dir = None

    return dimensions_valid, normalized_output_dir


def _validate_cell_schema(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    _require_fields(cell, PLANNED_CELL_REQUIRED_FIELDS, findings, cell_id)

    string_fields = (
        "arm",
        "control",
        "task",
        "model_id",
        "skip_update_mode",
        "scheduler_step_policy",
        "fingerprint",
        "planned_arm_dir",
    )
    integer_fields = (
        "training_seed",
        "policy_seed",
        "num_epochs",
        "total_steps",
        "min_step",
        "planned_skips",
    )
    boolean_fields = (
        "is_skipping_arm",
        "matched_budget",
        "no_early_stopping",
    )

    for field in string_fields:
        if field in cell and not _is_nonempty_str(cell[field]):
            _add_error(
                findings,
                field,
                cell_id,
                "value must be a non-empty string",
            )
    for field in integer_fields:
        if field in cell and not _is_int(cell[field]):
            _add_error(
                findings,
                field,
                cell_id,
                "value must be an integer; bools and floats are invalid",
            )
    for field in boolean_fields:
        if field in cell and type(cell[field]) is not bool:
            _add_error(findings, field, cell_id, "value must be a boolean")

    rate = cell.get("target_skip_rate", _MISSING)
    if rate is not _MISSING and not _is_float(rate):
        _add_error(
            findings,
            "target_skip_rate",
            cell_id,
            "value must be a finite float",
        )

    quota = cell.get("requested_quota", _MISSING)
    if quota is not _MISSING and quota is not None and not _is_int(quota):
        _add_error(
            findings,
            "requested_quota",
            cell_id,
            "value must be null or an integer; bools and floats are invalid",
        )

    for field in ("online_diagnostics", "controller_config", "identity_inputs"):
        if field in cell and type(cell[field]) is not dict:
            _add_error(findings, field, cell_id, "value must be an object")

    fingerprint = cell.get("fingerprint")
    if type(fingerprint) is str and not _FINGERPRINT_PATTERN.fullmatch(fingerprint):
        _add_error(
            findings,
            "fingerprint",
            cell_id,
            "fingerprint must be 16 lowercase hexadecimal characters",
        )


def _validate_online_diagnostics(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    online = cell.get("online_diagnostics")
    if type(online) is not dict:
        return

    _require_fields(
        online,
        _ONLINE_DIAGNOSTIC_FIELDS,
        findings,
        cell_id,
        "online_diagnostics.",
    )

    for key in ("requested_mode", "mode", "timing", "reason"):
        if key in online and not _is_nonempty_str(online[key]):
            _add_error(
                findings,
                "online_diagnostics." + key,
                cell_id,
                "value must be a non-empty string",
            )
    if "enabled" in online and type(online["enabled"]) is not bool:
        _add_error(
            findings,
            "online_diagnostics.enabled",
            cell_id,
            "value must be a boolean",
        )
    for key in ("parameter_sample_size", "update_interval"):
        if key in online and not _is_int(online[key]):
            _add_error(
                findings,
                "online_diagnostics." + key,
                cell_id,
                "value must be an integer; bools and floats are invalid",
            )

    arm = cell.get("arm")
    if arm not in PHASE1_3_CANONICAL_ARMS:
        return
    offline = arm in _OFFLINE_DIAGNOSTIC_ARMS
    expected_mode = "off" if offline else "sampled_lagged"
    expected_timing = "none" if offline else "post_decision_after_backward"
    expected_enabled = not offline

    _expect(
        online,
        "mode",
        expected_mode,
        findings,
        cell_id,
        "online_diagnostics.",
    )
    _expect(
        online,
        "timing",
        expected_timing,
        findings,
        cell_id,
        "online_diagnostics.",
    )
    _expect(
        online,
        "enabled",
        expected_enabled,
        findings,
        cell_id,
        "online_diagnostics.",
    )

    requested_mode = online.get("requested_mode")
    if type(requested_mode) is str and requested_mode not in {"auto", expected_mode}:
        _add_error(
            findings,
            "online_diagnostics.requested_mode",
            cell_id,
            f"expected 'auto' or {expected_mode!r}, got {requested_mode!r}",
        )

    if offline:
        _expect(
            online,
            "parameter_sample_size",
            0,
            findings,
            cell_id,
            "online_diagnostics.",
        )
        _expect(
            online,
            "update_interval",
            0,
            findings,
            cell_id,
            "online_diagnostics.",
        )
        _expect(
            online,
            "sample_seed",
            None,
            findings,
            cell_id,
            "online_diagnostics.",
        )
    else:
        for key in ("parameter_sample_size", "update_interval"):
            value = online.get(key)
            if _is_int(value) and value < 1:
                _add_error(
                    findings,
                    "online_diagnostics." + key,
                    cell_id,
                    "sampled_lagged diagnostics require an integer >= 1",
                )
        sample_seed = online.get("sample_seed", _MISSING)
        if not _is_int(sample_seed):
            _add_error(
                findings,
                "online_diagnostics.sample_seed",
                cell_id,
                "sampled_lagged diagnostics require an integer sample_seed",
            )
        elif sample_seed != cell.get("training_seed"):
            _add_error(
                findings,
                "online_diagnostics.sample_seed",
                cell_id,
                "sample_seed must equal training_seed",
            )

    identity = cell.get("identity_inputs")
    if type(identity) is dict and not _strict_equal(
        identity.get("online_diagnostics", _MISSING), online
    ):
        _add_error(
            findings,
            "identity_inputs.online_diagnostics",
            cell_id,
            "identity online diagnostics must equal the planned cell copy",
        )
    controller = cell.get("controller_config")
    if type(controller) is dict and not _strict_equal(
        controller.get("online_diagnostics", _MISSING), online
    ):
        _add_error(
            findings,
            "controller_config.online_diagnostics",
            cell_id,
            "controller online diagnostics must equal the planned cell copy",
        )


def _validate_identity(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    identity = cell.get("identity_inputs")
    if type(identity) is not dict:
        return

    _require_fields(
        identity,
        _IDENTITY_REQUIRED_FIELDS,
        findings,
        cell_id,
        "identity_inputs.",
    )

    integer_fields = (
        "training_seed",
        "train_samples_realized",
        "eval_samples_realized",
        "num_epochs",
        "policy_seed",
        "total_steps",
    )
    for key in integer_fields:
        if key in identity and not _is_int(identity[key]):
            _add_error(
                findings,
                "identity_inputs." + key,
                cell_id,
                "value must be an integer; bools and floats are invalid",
            )
    if "max_samples_requested" in identity:
        cap = identity["max_samples_requested"]
        if cap is not None and not _is_int(cap):
            _add_error(
                findings,
                "identity_inputs.max_samples_requested",
                cell_id,
                "value must be null or an integer",
            )
    for key in ("task", "model_id", "control", "skip_update_mode", "git_sha"):
        if key in identity and not _is_nonempty_str(identity[key]):
            _add_error(
                findings,
                "identity_inputs." + key,
                cell_id,
                "value must be a non-empty string",
            )
    if "scheduler_step_policy" in identity and not _is_nonempty_str(
        identity["scheduler_step_policy"]
    ):
        _add_error(
            findings,
            "identity_inputs.scheduler_step_policy",
            cell_id,
            "value must be a non-empty string",
        )
    for key in ("train_dataset_fingerprint", "eval_dataset_fingerprint"):
        if key in identity:
            value = identity[key]
            if value is not None and not _is_nonempty_str(value):
                _add_error(
                    findings,
                    "identity_inputs." + key,
                    cell_id,
                    "value must be null or a non-empty string",
                )
    if "target_skip_rate" in identity and not _is_float(
        identity["target_skip_rate"]
    ):
        _add_error(
            findings,
            "identity_inputs.target_skip_rate",
            cell_id,
            "value must be a finite float",
        )
    if "no_early_stopping" in identity and type(
        identity["no_early_stopping"]
    ) is not bool:
        _add_error(
            findings,
            "identity_inputs.no_early_stopping",
            cell_id,
            "value must be a boolean",
        )

    mirrored_fields = (
        "task",
        "training_seed",
        "model_id",
        "num_epochs",
        "control",
        "target_skip_rate",
        "policy_seed",
        "skip_update_mode",
        "scheduler_step_policy",
        "no_early_stopping",
        "total_steps",
    )
    for key in mirrored_fields:
        if (
            key in identity
            and key in cell
            and (
                identity[key] != cell[key]
                or type(identity[key]) is not type(cell[key])
            )
        ):
            _add_error(
                findings,
                "identity_inputs." + key,
                cell_id,
                f"identity value must equal planned cell {key}",
            )


def _validate_phase_controller(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    arm = cell.get("arm")
    controller = cell.get("controller_config")
    identity = cell.get("identity_inputs")
    controller_phase = (
        controller.get("phase_strat_controller", _MISSING)
        if type(controller) is dict
        else _MISSING
    )
    identity_phase = (
        identity.get("phase_strat_controller", _MISSING)
        if type(identity) is dict
        else _MISSING
    )

    if arm not in _PHASE_ARMS:
        if controller_phase is not _MISSING:
            _add_error(
                findings,
                "controller_config.phase_strat_controller",
                cell_id,
                "phase controller provenance is not applicable to this arm",
            )
        if identity_phase is not _MISSING:
            _add_error(
                findings,
                "identity_inputs.phase_strat_controller",
                cell_id,
                "phase controller provenance is not applicable to this arm",
            )
        return

    if controller_phase is _MISSING:
        _add_error(
            findings,
            "controller_config.phase_strat_controller",
            cell_id,
            "phase arm requires phase controller provenance",
        )
    elif type(controller_phase) is not dict:
        _add_error(
            findings,
            "controller_config.phase_strat_controller",
            cell_id,
            "phase controller provenance must be an object",
        )
    if identity_phase is _MISSING:
        _add_error(
            findings,
            "identity_inputs.phase_strat_controller",
            cell_id,
            "phase arm identity requires phase controller provenance",
        )
    elif type(identity_phase) is not dict:
        _add_error(
            findings,
            "identity_inputs.phase_strat_controller",
            cell_id,
            "phase controller provenance must be an object",
        )
    if type(controller_phase) is not dict or type(identity_phase) is not dict:
        return
    if not _strict_equal(controller_phase, identity_phase):
        _add_error(
            findings,
            "phase_strat_controller",
            cell_id,
            "controller and identity phase provenance copies must match",
        )

    prefix = "controller_config.phase_strat_controller."
    _require_fields(
        controller_phase,
        _PHASE_CONTROLLER_FIELDS,
        findings,
        cell_id,
        prefix,
    )
    if "policy_class" in controller_phase:
        _add_error(
            findings,
            prefix + "policy_class",
            cell_id,
            "phase controller provenance must use controller_class",
        )

    expected_class = PHASE1_3_POLICY_CLASSES.get(arm)
    expected_values = {
        "control": arm,
        "controller_class": expected_class,
        "policy_name": arm,
        "target_skip_rate": cell.get("target_skip_rate"),
        "total_steps": cell.get("total_steps"),
        "min_step": cell.get("min_step"),
        "policy_seed": cell.get("policy_seed"),
        "requested_quota": cell.get("requested_quota"),
    }
    for key, expected in expected_values.items():
        _expect(controller_phase, key, expected, findings, cell_id, prefix)

    n_phases = controller_phase.get("n_phases")
    if not _is_int(n_phases) or n_phases < 1:
        _add_error(
            findings,
            prefix + "n_phases",
            cell_id,
            "n_phases must be an integer >= 1",
        )
        return

    weights = controller_phase.get("phase_weights")
    if type(weights) is not list or len(weights) != n_phases:
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase_weights must be a list with n_phases entries",
        )
    elif not all(_is_float(weight) and weight >= 0.0 for weight in weights):
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase weights must be finite non-negative floats",
        )
    elif not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase weights must sum to 1",
        )

    bounds = controller_phase.get("phase_bounds")
    eligible = controller_phase.get("phase_eligible")
    quota = controller_phase.get("phase_quota")
    valid_bounds = (
        type(bounds) is list
        and len(bounds) == n_phases + 1
        and all(_is_int(bound) for bound in bounds)
    )
    if not valid_bounds:
        _add_error(
            findings,
            prefix + "phase_bounds",
            cell_id,
            "phase_bounds must contain n_phases + 1 integers",
        )
    else:
        if bounds[0] != cell.get("min_step") or bounds[-1] != cell.get(
            "total_steps"
        ):
            _add_error(
                findings,
                prefix + "phase_bounds",
                cell_id,
                "phase bounds must span min_step through total_steps",
            )
        if any(right < left for left, right in pairwise(bounds)):
            _add_error(
                findings,
                prefix + "phase_bounds",
                cell_id,
                "phase bounds must be non-decreasing",
            )

    if type(eligible) is not list or len(eligible) != n_phases or not all(
        _is_int(count) and count >= 0 for count in eligible
    ):
        _add_error(
            findings,
            prefix + "phase_eligible",
            cell_id,
            "phase_eligible must contain n_phases non-negative integers",
        )
    elif valid_bounds and eligible != [
        bounds[index + 1] - bounds[index] for index in range(n_phases)
    ]:
        _add_error(
            findings,
            prefix + "phase_eligible",
            cell_id,
            "phase_eligible must match phase-bound widths",
        )

    if type(quota) is not list or len(quota) != n_phases or not all(
        _is_int(count) and count >= 0 for count in quota
    ):
        _add_error(
            findings,
            prefix + "phase_quota",
            cell_id,
            "phase_quota must contain n_phases non-negative integers",
        )
    else:
        if sum(quota) != cell.get("requested_quota"):
            _add_error(
                findings,
                prefix + "phase_quota",
                cell_id,
                "phase quotas must sum to requested_quota",
            )
        if (
            type(eligible) is list
            and len(eligible) == n_phases
            and all(_is_int(count) for count in eligible)
            and any(
                phase_quota > capacity
                for phase_quota, capacity in zip(quota, eligible)
            )
        ):
            _add_error(
                findings,
                prefix + "phase_quota",
                cell_id,
                "phase quota cannot exceed phase eligibility",
            )

    guarded_fields = {"max_consecutive_skips", "risk_gamma", "guarded_safety"}
    if arm == "phase_strat_guarded":
        _require_fields(
            controller_phase,
            frozenset(guarded_fields),
            findings,
            cell_id,
            prefix,
        )
        max_skips = controller_phase.get("max_consecutive_skips")
        if not _is_int(max_skips) or max_skips < 1:
            _add_error(
                findings,
                prefix + "max_consecutive_skips",
                cell_id,
                "max_consecutive_skips must be an integer >= 1",
            )
        risk_gamma = controller_phase.get("risk_gamma")
        if not _is_float(risk_gamma):
            _add_error(
                findings,
                prefix + "risk_gamma",
                cell_id,
                "risk_gamma must be a finite float",
            )
        safety = controller_phase.get("guarded_safety")
        if type(safety) is not dict:
            _add_error(
                findings,
                prefix + "guarded_safety",
                cell_id,
                "guarded_safety must be an object",
            )
        else:
            safety_required = frozenset(
                {
                    "use_rho_vg",
                    "rho_veto_threshold",
                    "use_safety_horizon",
                    "spike_factor",
                }
            )
            _require_fields(
                safety,
                safety_required,
                findings,
                cell_id,
                prefix + "guarded_safety.",
            )
            for key in ("use_rho_vg", "use_safety_horizon"):
                if key in safety and type(safety[key]) is not bool:
                    _add_error(
                        findings,
                        prefix + "guarded_safety." + key,
                        cell_id,
                        "value must be a boolean",
                    )
            for key in ("rho_veto_threshold", "spike_factor"):
                if key in safety and not _is_float(safety[key]):
                    _add_error(
                        findings,
                        prefix + "guarded_safety." + key,
                        cell_id,
                        "value must be a finite float",
                    )
    else:
        for key in sorted(guarded_fields):
            if key in controller_phase:
                _add_error(
                    findings,
                    prefix + key,
                    cell_id,
                    "guarded-only field is not applicable to fixed_phase_strat",
                )


def _validate_ler_controller(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    arm = cell.get("arm")
    controller = cell.get("controller_config")
    identity = cell.get("identity_inputs")
    controller_ler = (
        controller.get("ler_guided_controller", _MISSING)
        if type(controller) is dict
        else _MISSING
    )
    identity_ler = (
        identity.get("ler_guided_controller", _MISSING)
        if type(identity) is dict
        else _MISSING
    )

    if arm not in _LER_ARMS:
        if controller_ler is not _MISSING:
            _add_error(
                findings,
                "controller_config.ler_guided_controller",
                cell_id,
                "LER controller provenance is not applicable to this arm",
            )
        if identity_ler is not _MISSING:
            _add_error(
                findings,
                "identity_inputs.ler_guided_controller",
                cell_id,
                "LER controller provenance is not applicable to this arm",
            )
        return

    if controller_ler is _MISSING:
        _add_error(
            findings,
            "controller_config.ler_guided_controller",
            cell_id,
            "LER arm requires LER controller provenance",
        )
    elif type(controller_ler) is not dict:
        _add_error(
            findings,
            "controller_config.ler_guided_controller",
            cell_id,
            "LER controller provenance must be an object",
        )
    if identity_ler is _MISSING:
        _add_error(
            findings,
            "identity_inputs.ler_guided_controller",
            cell_id,
            "LER arm identity requires LER controller provenance",
        )
    elif type(identity_ler) is not dict:
        _add_error(
            findings,
            "identity_inputs.ler_guided_controller",
            cell_id,
            "LER controller provenance must be an object",
        )
    if type(controller_ler) is not dict or type(identity_ler) is not dict:
        return
    if not _strict_equal(controller_ler, identity_ler):
        _add_error(
            findings,
            "ler_guided_controller",
            cell_id,
            "controller and identity LER provenance copies must match",
        )

    prefix = "controller_config.ler_guided_controller."
    _require_fields(
        controller_ler,
        _LER_CONTROLLER_FIELDS,
        findings,
        cell_id,
        prefix,
    )
    if "controller_class" in controller_ler:
        _add_error(
            findings,
            prefix + "controller_class",
            cell_id,
            "LER controller provenance must use policy_class",
        )

    expected_class = PHASE1_3_POLICY_CLASSES.get(arm)
    expected_values = {
        "control": arm,
        "policy_class": expected_class,
        "policy_name": arm,
        "target_skip_rate": cell.get("target_skip_rate"),
        "total_steps": cell.get("total_steps"),
        "min_step": cell.get("min_step"),
        "policy_seed": cell.get("policy_seed"),
        "required_tracker_mode": "sampled_lagged",
        "required_tracker_timing": "post_decision_after_backward",
        "safety_enabled": arm == "ler_guided_stratified_safe",
    }
    for key, expected in expected_values.items():
        _expect(controller_ler, key, expected, findings, cell_id, prefix)

    for key in ("n_phases", "max_consecutive_skips", "probe_interval", "min_ler_observations"):
        value = controller_ler.get(key)
        if not _is_int(value) or value < 1:
            _add_error(
                findings,
                prefix + key,
                cell_id,
                "value must be an integer >= 1",
            )
    strength = controller_ler.get("ler_guidance_strength")
    if not _is_float(strength):
        _add_error(
            findings,
            prefix + "ler_guidance_strength",
            cell_id,
            "ler_guidance_strength must be a finite float",
        )

    n_phases = controller_ler.get("n_phases")
    weights = controller_ler.get("phase_weights")
    if not _is_int(n_phases) or type(weights) is not list or len(weights) != n_phases:
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase_weights must be a list with n_phases entries",
        )
    elif not all(_is_float(weight) and weight >= 0.0 for weight in weights):
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase weights must be finite non-negative floats",
        )
    elif not math.isclose(sum(weights), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _add_error(
            findings,
            prefix + "phase_weights",
            cell_id,
            "phase weights must sum to 1",
        )

    if arm == "ler_guided_stratified_safe":
        _require_fields(
            controller_ler,
            _LER_SAFETY_FIELDS,
            findings,
            cell_id,
            prefix,
        )
        _expect(
            controller_ler,
            "use_rho_vg_safety",
            True,
            findings,
            cell_id,
            prefix,
        )
        _expect(
            controller_ler,
            "use_loss_spike_safety",
            True,
            findings,
            cell_id,
            prefix,
        )
        for key in ("rho_veto_threshold", "loss_spike_factor"):
            if key in controller_ler and not _is_float(controller_ler[key]):
                _add_error(
                    findings,
                    prefix + key,
                    cell_id,
                    "value must be a finite float",
                )
        window = controller_ler.get("loss_spike_window")
        if not _is_int(window) or window < 1:
            _add_error(
                findings,
                prefix + "loss_spike_window",
                cell_id,
                "loss_spike_window must be an integer >= 1",
            )
    else:
        for key in sorted(_LER_SAFETY_FIELDS):
            if key in controller_ler:
                _add_error(
                    findings,
                    prefix + key,
                    cell_id,
                    "safety field is not applicable to the non-safe LER arm",
                )


def _validate_controller(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    controller = cell.get("controller_config")
    if type(controller) is not dict:
        return

    _require_fields(
        controller,
        _CONTROLLER_REQUIRED_FIELDS,
        findings,
        cell_id,
        "controller_config.",
    )
    arm = cell.get("arm")
    skipping = arm != "full_finetune"
    expected_values = {
        "arm": arm,
        "arm_alias_of": None,
        "control": arm,
        "policy_class": PHASE1_3_POLICY_CLASSES.get(arm),
        "compute_saving_mechanism": "backward_skipping" if skipping else "none",
        "policy_seed": cell.get("policy_seed"),
        "target_skip_rate": cell.get("target_skip_rate"),
        "min_step": cell.get("min_step"),
        "configured_total_steps": cell.get("total_steps"),
        "requested_quota": cell.get("requested_quota"),
        "matched_budget": True,
        "is_skipping_arm": skipping,
        "allow_early_stopping_with_skipping": False,
        "early_stopping_active": False,
        "num_epochs": cell.get("num_epochs"),
    }
    for key, expected in expected_values.items():
        _expect(controller, key, expected, findings, cell_id, "controller_config.")

    if "rvd" in controller:
        _add_error(
            findings,
            "controller_config.rvd",
            cell_id,
            "RVD provenance is not part of the canonical Phase 1.3 matrix",
        )

    _validate_phase_controller(cell, findings, cell_id)
    _validate_ler_controller(cell, findings, cell_id)


def _validate_cell_contract(
    cell: dict[str, Any],
    findings: list[dict[str, Any]],
    cell_id: tuple[Any, Any, Any, Any] | None,
) -> None:
    arm = cell.get("arm")
    control = cell.get("control")
    if arm not in PHASE1_3_CANONICAL_ARMS:
        _add_error(
            findings,
            "arm",
            cell_id,
            "arm must be one of the six canonical controls; aliases are invalid",
        )
    if control != arm or type(control) is not type(arm):
        _add_error(findings, "control", cell_id, "control must exactly equal arm")

    _expect(cell, "skip_update_mode", "freeze", findings, cell_id)
    _expect(
        cell,
        "scheduler_step_policy",
        "skip_on_backward_skip",
        findings,
        cell_id,
    )
    _expect(cell, "matched_budget", True, findings, cell_id)
    _expect(cell, "no_early_stopping", True, findings, cell_id)
    _expect(cell, "min_step", POLICY_MIN_STEP, findings, cell_id)

    num_epochs = cell.get("num_epochs")
    if _is_int(num_epochs) and num_epochs < 1:
        _add_error(findings, "num_epochs", cell_id, "num_epochs must be >= 1")
    total_steps = cell.get("total_steps")
    min_step = cell.get("min_step")
    if _is_int(total_steps) and _is_int(min_step) and total_steps <= min_step:
        _add_error(
            findings,
            "total_steps",
            cell_id,
            "total_steps must be greater than min_step",
        )

    if arm == "full_finetune":
        _expect(cell, "requested_quota", None, findings, cell_id)
        _expect(cell, "planned_skips", 0, findings, cell_id)
        _expect(cell, "is_skipping_arm", False, findings, cell_id)
    elif arm in PHASE1_3_CANONICAL_ARMS:
        _expect(cell, "is_skipping_arm", True, findings, cell_id)
        rate = cell.get("target_skip_rate")
        if _is_float(rate) and _is_int(total_steps):
            exact_quota = round(rate * total_steps)
            _expect(cell, "requested_quota", exact_quota, findings, cell_id)
            _expect(cell, "planned_skips", exact_quota, findings, cell_id)
            if _is_int(min_step) and exact_quota > total_steps - min_step:
                _add_error(
                    findings,
                    "requested_quota",
                    cell_id,
                    "exact quota is infeasible after min_step and must not be clipped",
                )

    _validate_online_diagnostics(cell, findings, cell_id)
    _validate_identity(cell, findings, cell_id)
    _validate_controller(cell, findings, cell_id)

    identity = cell.get("identity_inputs")
    fingerprint = cell.get("fingerprint")
    if type(identity) is dict and type(fingerprint) is str:
        try:
            recomputed = build_scientific_fingerprint(identity)
        except (TypeError, ValueError) as exc:
            _add_error(
                findings,
                "identity_inputs",
                cell_id,
                f"identity inputs are not fingerprintable: {exc}",
            )
        else:
            if recomputed != fingerprint:
                _add_error(
                    findings,
                    "fingerprint",
                    cell_id,
                    f"stored fingerprint does not match recomputed {recomputed!r}",
                )


def _validate_order_and_cross_product(
    plan: list[Any],
    *,
    tasks: list[str],
    seeds: list[int],
    target_skip_rates: list[float],
    findings: list[dict[str, Any]],
) -> None:
    expected_order = [
        (task, seed, rate, arm)
        for task in tasks
        for seed in seeds
        for rate in target_skip_rates
        for arm in PHASE1_3_CANONICAL_ARMS
    ]
    if len(plan) != len(expected_order):
        _add_error(
            findings,
            "plan.length",
            None,
            f"expected {len(expected_order)} cells, got {len(plan)}",
        )

    actual_keys = []
    for index, cell in enumerate(plan):
        actual = _cell_ref(cell)
        if actual is not None:
            actual_keys.append(actual)
        if index < len(expected_order) and actual != expected_order[index]:
            _add_error(
                findings,
                "plan.order",
                actual,
                f"cell {index} must be {expected_order[index]!r}",
            )

    counts: dict[tuple[Any, Any, Any, Any], int] = {}
    for key in actual_keys:
        counts[key] = counts.get(key, 0) + 1
    for key, count in counts.items():
        if count > 1:
            _add_error(
                findings,
                "plan.cells",
                key,
                f"duplicate planned cell appears {count} times",
            )

    expected_keys = set(expected_order)
    actual_key_set = set(actual_keys)
    for key in expected_order:
        if key not in actual_key_set:
            _add_error(findings, "plan.cells", key, "required planned cell is missing")
    for key in sorted(actual_key_set - expected_keys, key=repr):
        _add_error(
            findings,
            "plan.cells",
            key,
            "planned cell is outside the requested cross-product",
        )


def _validate_pairing(
    plan: list[Any],
    *,
    tasks: list[str],
    seeds: list[int],
    target_skip_rates: list[float],
    findings: list[dict[str, Any]],
) -> None:
    cells_by_key: dict[tuple[Any, Any, Any, Any], list[dict[str, Any]]] = {}
    for cell in plan:
        key = _cell_ref(cell)
        if key is not None and type(cell) is dict:
            cells_by_key.setdefault(key, []).append(cell)

    paired_fields = (
        "task",
        "model_id",
        "total_steps",
        "num_epochs",
        "training_seed",
        "policy_seed",
        "target_skip_rate",
    )
    for task in tasks:
        for seed in seeds:
            for rate in target_skip_rates:
                group = []
                for arm in PHASE1_3_CANONICAL_ARMS:
                    matches = cells_by_key.get((task, seed, rate, arm), [])
                    if len(matches) == 1:
                        group.append(matches[0])
                if len(group) != len(PHASE1_3_CANONICAL_ARMS):
                    continue
                baseline = group[0]
                for cell in group[1:]:
                    for field in paired_fields:
                        if cell.get(field, _MISSING) != baseline.get(field, _MISSING):
                            _add_error(
                                findings,
                                "paired_group." + field,
                                _cell_ref(cell),
                                f"all six arms must share paired field {field!r}",
                            )


def _validate_uniqueness_and_paths(
    plan: list[Any],
    *,
    base_output_dir: str,
    findings: list[dict[str, Any]],
) -> None:
    fingerprints: dict[str, list[tuple[Any, Any, Any, Any] | None]] = {}
    paths: dict[str, list[tuple[Any, Any, Any, Any] | None]] = {}
    for cell in plan:
        if type(cell) is not dict:
            continue
        cell_id = _cell_ref(cell)
        fingerprint = cell.get("fingerprint")
        if type(fingerprint) is str:
            fingerprints.setdefault(fingerprint, []).append(cell_id)
        planned_path = cell.get("planned_arm_dir")
        if type(planned_path) is str:
            paths.setdefault(planned_path, []).append(cell_id)
        arm = cell.get("arm")
        if type(arm) is str and type(fingerprint) is str and type(planned_path) is str:
            expected = os.path.join(base_output_dir, arm, fingerprint)
            if planned_path != expected:
                _add_error(
                    findings,
                    "planned_arm_dir",
                    cell_id,
                    f"expected planned arm directory {expected!r}",
                )

    for fingerprint, cells in fingerprints.items():
        if len(cells) > 1:
            for cell_id in cells:
                _add_error(
                    findings,
                    "fingerprint",
                    cell_id,
                    f"fingerprint {fingerprint!r} is not unique",
                )
    for path, cells in paths.items():
        if len(cells) > 1:
            for cell_id in cells:
                _add_error(
                    findings,
                    "planned_arm_dir",
                    cell_id,
                    f"planned arm directory {path!r} is not unique",
                )


def validate_phase1_3_matrix_plan(
    plan: list[dict],
    *,
    tasks: list[str],
    seeds: list[int],
    target_skip_rates: list[float],
    minimum_seed_count: int,
    base_output_dir: os.PathLike[str] | str,
) -> list[dict[str, Any]]:
    """Validate a complete Phase 1.3 plan without filesystem writes.

    Every detected error is collected into one ``MatrixPlanError``. A valid
    plan returns the (normally empty) structured findings list.
    """
    findings: list[dict[str, Any]] = []
    dimensions_valid, normalized_output_dir = _validate_dimensions(
        tasks=tasks,
        seeds=seeds,
        target_skip_rates=target_skip_rates,
        minimum_seed_count=minimum_seed_count,
        base_output_dir=base_output_dir,
        findings=findings,
    )

    if type(plan) is not list:
        _add_error(findings, "plan", None, "plan must be a list")
    else:
        for index, cell in enumerate(plan):
            if type(cell) is not dict:
                _add_error(
                    findings,
                    f"plan[{index}]",
                    None,
                    "planned cell must be an object",
                )
                continue
            cell_id = _cell_ref(cell)
            _validate_cell_schema(cell, findings, cell_id)
            _validate_cell_contract(cell, findings, cell_id)

        if dimensions_valid:
            _validate_order_and_cross_product(
                plan,
                tasks=tasks,
                seeds=seeds,
                target_skip_rates=target_skip_rates,
                findings=findings,
            )
            _validate_pairing(
                plan,
                tasks=tasks,
                seeds=seeds,
                target_skip_rates=target_skip_rates,
                findings=findings,
            )
        if normalized_output_dir is not None:
            _validate_uniqueness_and_paths(
                plan,
                base_output_dir=normalized_output_dir,
                findings=findings,
            )

    if any(finding["severity"] == "error" for finding in findings):
        raise MatrixPlanError(findings)
    return findings
