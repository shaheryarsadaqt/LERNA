"""Authoritative local run provenance for LERNA Phase 1.3.

Each run owns one atomic ``run_manifest.json`` with the closed lifecycle
``running -> completed | failed``. Matched-claim completion requires approval
from the Piece 5 result validator.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, List, Optional

MANIFEST_SCHEMA_VERSION = 1
MANIFEST_FILENAME = "run_manifest.json"

CLASSIFICATION_MATCHED_CLAIM = "matched_claim"
CLASSIFICATION_LOCAL_DEVELOPMENT = "local_development"

DEFAULT_ARTIFACT_FILENAMES = (
    "results.json",
    "instrumentation.json",
    "ler_diagnostics.json",
)

_SECRET_KEY_PATTERN = re.compile(
    r"(?:api[_-]?key|access[_-]?key|token|secret|password|credential)",
    re.IGNORECASE,
)
_SECRET_ASSIGNMENT_PATTERN = re.compile(
    r"(?i)(api[_-]?key|access[_-]?key|token|secret|password|credential)"
    r"(\s*[=:]\s*)([^\s,;]+)"
)
_URL_CREDENTIAL_PATTERN = re.compile(r"(://[^:/\s]+:)([^@/\s]+)(@)")


class ProvenanceError(RuntimeError):
    """Raised when a provenance invariant is violated."""


def collect_git_state(repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Return commit, tracked-tree state, and untracked paths conservatively."""
    cwd = repo_root or os.getcwd()
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        tracked = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).splitlines()
        return {
            "commit_sha": sha,
            "dirty": bool(tracked),
            "tracked_changes": tracked,
            "untracked_paths": untracked,
        }
    except (OSError, subprocess.SubprocessError):
        return {
            "commit_sha": None,
            "dirty": True,
            "tracked_changes": ["git_state_unavailable"],
            "untracked_paths": [],
        }


def collect_environment_versions() -> Dict[str, Any]:
    """Return interpreter, framework, and device versions without raising."""
    versions: Dict[str, Any] = {"python": sys.version.split()[0]}
    try:
        import torch

        versions["torch"] = str(torch.__version__)
        versions["cuda"] = str(getattr(torch.version, "cuda", None))
        versions["device"] = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        )
    except Exception:
        versions.update(
            {"torch": "unavailable", "cuda": "unavailable", "device": "unavailable"}
        )
    try:
        import transformers

        versions["transformers"] = str(transformers.__version__)
    except Exception:
        versions["transformers"] = "unavailable"
    return versions


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _redact_text(value: str) -> str:
    redacted = _SECRET_ASSIGNMENT_PATTERN.sub(r"\1\2<redacted>", value)
    return _URL_CREDENTIAL_PATTERN.sub(r"\1<redacted>\3", redacted)


def _sanitize(obj: Any) -> Any:
    """Recursively redact secret values and coerce data to JSON-safe types."""
    if isinstance(obj, dict):
        sanitized = {}
        for key, value in obj.items():
            text_key = str(key)
            sanitized[text_key] = (
                "<redacted>" if _SECRET_KEY_PATTERN.search(text_key) else _sanitize(value)
            )
        return sanitized
    if isinstance(obj, (list, tuple)):
        return [_sanitize(value) for value in obj]
    if isinstance(obj, str):
        return _redact_text(obj)
    if isinstance(obj, (int, float, bool)) or obj is None:
        return obj
    return _redact_text(str(obj))


def _sanitize_argv(argv: Iterable[str]) -> List[str]:
    values = [str(value) for value in argv]
    sanitized: List[str] = []
    redact_next = False
    for value in values:
        if redact_next:
            sanitized.append("<redacted>")
            redact_next = False
            continue
        if "=" in value:
            option, raw_value = value.split("=", 1)
            if _SECRET_KEY_PATTERN.search(option):
                sanitized.append(f"{option}=<redacted>")
                continue
            value = f"{option}={raw_value}"
        sanitized.append(_redact_text(value))
        if value.startswith("-") and _SECRET_KEY_PATTERN.search(value):
            redact_next = True
    return sanitized


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(
        prefix=".run_manifest_", suffix=".tmp", dir=directory
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def manifest_path(run_dir: str) -> str:
    return os.path.join(run_dir, MANIFEST_FILENAME)


def load_manifest(run_dir: str) -> Dict[str, Any]:
    with open(manifest_path(run_dir), "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ProvenanceError("run manifest root must be a JSON object")
    return data


def _duration_seconds(manifest: Dict[str, Any], end_iso: str) -> Optional[float]:
    try:
        start = datetime.fromisoformat(manifest["start_time_utc"])
        end = datetime.fromisoformat(end_iso)
        return max((end - start).total_seconds(), 0.0)
    except (KeyError, TypeError, ValueError):
        return None


def _require_running(manifest: Dict[str, Any], target_status: str) -> None:
    current = manifest.get("status")
    if current != "running":
        raise ProvenanceError(
            f"Invalid manifest transition {current!r} -> {target_status!r}; "
            "only a running manifest may become terminal"
        )


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_check(run_dir: str, filename: str) -> Dict[str, Any]:
    path = os.path.join(run_dir, filename)
    exists = os.path.isfile(path)
    return {
        "path": filename,
        "exists": exists,
        "size_bytes": os.path.getsize(path) if exists else None,
        "sha256": _sha256_file(path) if exists else None,
    }


def build_identity_inputs(
    *,
    task: str,
    training_seed: int,
    model_id: str,
    max_samples_requested: Optional[int],
    train_samples_realized: int,
    eval_samples_realized: int,
    train_dataset_fingerprint: Optional[str],
    eval_dataset_fingerprint: Optional[str],
    num_epochs: int,
    control: str,
    target_skip_rate: float,
    policy_seed: int,
    skip_update_mode: str,
    no_early_stopping: bool,
    total_steps: int,
    git_sha: str,
    scheduler_step_policy: str = "skip_on_backward_skip",
) -> Dict[str, Any]:
    """Build the canonical identity dictionary for a scientific run.

    None serializes deterministically as JSON null, truthfully representing
    an unlimited requested cap. Realized dataset size and the Hugging Face
    dataset fingerprint identify the actual data used.

    Controller-specific settings (RVD configuration, max_consecutive_skips,
    etc.) are represented only in their controller configuration
    sub-dictionaries, which callers attach after this function returns.
    """
    return {
        "task": str(task),
        "training_seed": int(training_seed),
        "model_id": str(model_id),
        "max_samples_requested": max_samples_requested,
        "train_samples_realized": int(train_samples_realized),
        "eval_samples_realized": int(eval_samples_realized),
        "train_dataset_fingerprint": train_dataset_fingerprint,
        "eval_dataset_fingerprint": eval_dataset_fingerprint,
        "num_epochs": int(num_epochs),
        "control": str(control),
        "target_skip_rate": float(target_skip_rate),
        "policy_seed": int(policy_seed),
        "skip_update_mode": str(skip_update_mode),
        "scheduler_step_policy": str(scheduler_step_policy),
        "no_early_stopping": bool(no_early_stopping),
        "total_steps": int(total_steps),
        "git_sha": str(git_sha),
    }


_BASE_IDENTITY_KEYS = frozenset(
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
    }
)

_CONTROLLER_CONFIG_SUBDICTS = frozenset(
    {
        "online_diagnostics",
        "rvd",
        "ler_guided_controller",
        "phase_strat_controller",
    }
)

_MAX_CONSECUTIVE_SKIPS_APPLICABLE = frozenset(
    {
        "grad_norm",
        "quota_hybrid",
        "guarded_hybrid",
        "phase_strat",
        "phase_strat_guarded",
    }
)

_RVD_ACTIVE_MARGIN_FIELDS = frozenset({"margin_rank_floor"})
_RVD_ACTIVE_LOSS_SPIKE_FIELDS = frozenset({"spike_factor", "spike_ema_window"})
_LER_GUIDED_CONTROLS = frozenset(
    {"ler_guided_stratified", "ler_guided_stratified_safe"}
)


def _canonicalize_rvd_config(
    rvd_config: Dict[str, Any],
    training_seed: int,
) -> Dict[str, Any]:
    """Strip behaviorally inactive fields from an RVD controller config."""
    veto_mode = rvd_config.get("veto_mode", "none")
    canonical = dict(rvd_config)
    if veto_mode != "margin":
        for key in _RVD_ACTIVE_MARGIN_FIELDS:
            canonical.pop(key, None)
    if veto_mode != "loss_spike":
        for key in _RVD_ACTIVE_LOSS_SPIKE_FIELDS:
            canonical.pop(key, None)
    policy_seed = canonical.get("policy_seed")
    if policy_seed is not None and policy_seed == training_seed:
        canonical.pop("policy_seed_defaulted_to_training_seed", None)
    return canonical


def _canonicalize_identity_inputs(identity_inputs: Dict[str, Any]) -> Dict[str, Any]:
    """Canonicalize identity inputs for fingerprinting.

    Preserves unknown fields by default, strips only known inactive fields,
    and normalizes controller-specific configuration.
    """
    control = identity_inputs.get("control")
    if control == "random_skip":
        control = "exact_random"

    canonical = dict(identity_inputs)
    canonical["control"] = control

    if control != "rvd":
        for key in list(canonical.keys()):
            if key.startswith("rvd_") and key not in _BASE_IDENTITY_KEYS:
                del canonical[key]

    if "max_consecutive_skips" in canonical:
        applicable = control in {"grad_norm"} | _MAX_CONSECUTIVE_SKIPS_APPLICABLE
        if not applicable:
            del canonical["max_consecutive_skips"]
        elif control == "phase_strat_guarded" and "phase_strat_controller" in canonical:
            del canonical["max_consecutive_skips"]

    for key in list(canonical.keys()):
        if key not in _CONTROLLER_CONFIG_SUBDICTS:
            continue
        if key == "rvd":
            if control != "rvd":
                del canonical[key]
            else:
                training_seed = canonical.get("training_seed")
                canonical[key] = _canonicalize_rvd_config(canonical[key], training_seed)
        elif key == "phase_strat_controller":
            if control not in {"fixed_phase_strat", "phase_strat_guarded"}:
                del canonical[key]
        elif key == "ler_guided_controller":
            if control not in _LER_GUIDED_CONTROLS:
                del canonical[key]
        elif key == "online_diagnostics":
            continue

    return canonical


def build_scientific_fingerprint(identity_inputs: Dict[str, Any]) -> str:
    """Build a deterministic collision-resistant fingerprint for a scientific run.

    The fingerprint covers every configuration dimension that can affect
    experimental outcomes. Identical configurations produce identical fingerprints;
    any relevant difference produces a different fingerprint.

    Args:
        identity_inputs: Canonical identity dictionary produced by
            ``build_identity_inputs()``, optionally extended with controller
            configuration sub-dictionaries.
    """
    canonical = _canonicalize_identity_inputs(identity_inputs)
    serialized = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def _validation_summary(status: Dict[str, Any]) -> Dict[str, Any]:
    """Keep validation decisions without copying arbitrary result values."""
    summary = {
        key: status.get(key)
        for key in (
            "ok",
            "protocol_complete",
            "matched_budget_claimed",
            "valid_for_matched_budget",
            "n_errors",
        )
    }
    findings = status.get("findings")
    if isinstance(findings, list):
        summary["findings"] = [
            {
                "severity": finding.get("severity"),
                "field": finding.get("field"),
            }
            for finding in findings
            if isinstance(finding, dict)
        ]
    return _sanitize(summary)


def classify_run(
    *,
    git_dirty: bool,
    matched_budget_planned: bool,
    skip_update_mode: Optional[str],
    requested_classification: str = CLASSIFICATION_MATCHED_CLAIM,
) -> str:
    """Validate whether a run may request matched-claim provenance."""
    if requested_classification == CLASSIFICATION_LOCAL_DEVELOPMENT:
        return CLASSIFICATION_LOCAL_DEVELOPMENT
    if requested_classification != CLASSIFICATION_MATCHED_CLAIM:
        raise ProvenanceError(
            f"Unknown provenance classification: {requested_classification!r}"
        )
    if git_dirty:
        raise ProvenanceError(
            "Refusing matched_claim: tracked git state is dirty or unavailable"
        )
    if not matched_budget_planned:
        raise ProvenanceError(
            "Refusing matched_claim: the run is configured as unmatched "
            "(for example, early stopping is active)"
        )
    if skip_update_mode != "freeze":
        raise ProvenanceError(
            "Refusing matched_claim: authoritative Phase 1.3 requires "
            "skip_update_mode='freeze'"
        )
    return CLASSIFICATION_MATCHED_CLAIM


def write_manifest_running(
    run_dir: str,
    *,
    argv: List[str],
    task: str,
    model_id: str,
    seed: int,
    controller_name: str,
    controller_seed: Optional[int],
    target_skip_rate: Optional[float],
    planned_quota: Optional[int],
    total_steps: Optional[int],
    warmup_steps: Optional[int],
    skip_update_mode: Optional[str],
    controller_config_effective: Optional[Dict[str, Any]],
    matched_budget_planned: bool,
    budget_classification: str,
    output_paths: Dict[str, str],
    requested_classification: str = CLASSIFICATION_MATCHED_CLAIM,
    repo_root: Optional[str] = None,
    git_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    version_provider: Callable[[], Dict[str, Any]] = collect_environment_versions,
    identity_inputs: Optional[Dict[str, Any]] = None,
    fingerprint: Optional[str] = None,
    attempt: Optional[int] = None,
) -> Dict[str, Any]:
    """Create the unique initial running manifest atomically."""
    os.makedirs(run_dir, exist_ok=True)
    target = manifest_path(run_dir)
    if os.path.exists(target):
        raise ProvenanceError(
            f"Refusing to overwrite existing manifest: {MANIFEST_FILENAME}"
        )
    stale = [
        name for name in DEFAULT_ARTIFACT_FILENAMES
        if os.path.exists(os.path.join(run_dir, name))
    ]
    if stale:
        raise ProvenanceError(
            "Refusing to start in a run directory containing canonical artifacts: "
            + ", ".join(sorted(stale))
        )

    git_state = git_provider() if git_provider is not None else collect_git_state(repo_root)
    classification = classify_run(
        git_dirty=bool(git_state.get("dirty", True)),
        matched_budget_planned=bool(matched_budget_planned),
        skip_update_mode=skip_update_mode,
        requested_classification=requested_classification,
    )
    manifest: Dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "status": "running",
        "start_time_utc": _utc_now_iso(),
        "provenance_classification": classification,
        "git": _sanitize(git_state),
        "command": {"argv": _sanitize_argv(argv)},
        "environment": _sanitize(version_provider()),
        "run": {
            "task": task,
            "model_id": model_id,
            "seed": seed,
            "controller_name": controller_name,
            "controller_seed": controller_seed,
            "target_skip_rate": target_skip_rate,
            "planned_quota": planned_quota,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "skip_update_mode": skip_update_mode,
            "matched_budget_planned": bool(matched_budget_planned),
            "budget_classification": budget_classification,
        },
        "controller_config_effective": _sanitize(
            controller_config_effective or {}
        ),
        "output_paths": _sanitize(output_paths),
    }
    if identity_inputs is not None:
        manifest["identity_inputs"] = _sanitize(identity_inputs)
    if fingerprint is not None:
        manifest["fingerprint"] = fingerprint
    if attempt is not None:
        manifest["attempt"] = attempt
    _atomic_write_json(target, manifest)
    return manifest


def finalize_manifest_completed(
    run_dir: str,
    *,
    realized_skips: Optional[int] = None,
    realized_skip_rate: Optional[float] = None,
    validation_status: Dict[str, Any],
    artifact_filenames: Iterable[str] = DEFAULT_ARTIFACT_FILENAMES,
) -> Dict[str, Any]:
    """Transition running to completed after authoritative validation."""
    manifest = load_manifest(run_dir)
    _require_running(manifest, "completed")
    validation = _validation_summary(validation_status)
    if (
        manifest.get("provenance_classification") == CLASSIFICATION_MATCHED_CLAIM
        and validation.get("valid_for_matched_budget") is not True
    ):
        raise ProvenanceError(
            "Refusing matched_claim completion: Piece 5 validation did not "
            "approve valid_for_matched_budget"
        )

    end_iso = _utc_now_iso()
    manifest.update(
        {
            "status": "completed",
            "end_time_utc": end_iso,
            "duration_seconds": _duration_seconds(manifest, end_iso),
            "artifacts": {
                name: _artifact_check(run_dir, name)
                for name in artifact_filenames
            },
            "realized": {
                "skipped_steps": realized_skips,
                "skip_rate": realized_skip_rate,
            },
            "validation": validation,
        }
    )
    _atomic_write_json(manifest_path(run_dir), manifest)
    verification = verify_completed_manifest(run_dir)
    if not verification["ok"]:
        raise ProvenanceError(
            "Completed manifest failed integrity verification: "
            + "; ".join(
                f"{entry['field']}: {entry['message']}"
                for entry in verification["errors"]
            )
        )
    return manifest


def finalize_manifest_failed(run_dir: str, exc: BaseException) -> Dict[str, Any]:
    """Transition running to failed without serializing exception secrets."""
    manifest = load_manifest(run_dir)
    _require_running(manifest, "failed")
    end_iso = _utc_now_iso()
    manifest.update(
        {
            "status": "failed",
            "end_time_utc": end_iso,
            "duration_seconds": _duration_seconds(manifest, end_iso),
            "error": {
                "type": type(exc).__name__,
                "message": "run failed; inspect local logs for details",
            },
            "artifacts": {
                name: _artifact_check(run_dir, name)
                for name in DEFAULT_ARTIFACT_FILENAMES
            },
        }
    )
    _atomic_write_json(manifest_path(run_dir), manifest)
    return manifest


_COMPLETED_REQUIRED_FIELDS = (
    "end_time_utc",
    "duration_seconds",
    "artifacts",
    "realized",
    "validation",
)
_ARTIFACT_METADATA_KEYS = ("path", "exists", "size_bytes", "sha256")
_SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _verify_artifacts(run_dir: str, manifest: Dict[str, Any],
                      errors: List[Dict[str, str]]) -> None:
    """Verify artifact metadata schema, existence, size, and recomputed SHA-256.

    The manifest itself (``run_manifest.json``) is never hashed: hashing it
    would capture the preceding running manifest, not the final completed one.
    """
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict):
        errors.append({
            "field": "artifacts",
            "message": "artifacts must be an object",
        })
        return

    raw_output_paths = manifest.get("output_paths")
    output_paths = raw_output_paths if isinstance(raw_output_paths, dict) else {}
    output_names = {
        name for name in output_paths.values()
        if isinstance(name, str) and name != MANIFEST_FILENAME
    }
    expected = set(DEFAULT_ARTIFACT_FILENAMES) | output_names
    required = {"results.json", "instrumentation.json"} | output_names

    for name in sorted(expected):
        entry = raw_artifacts.get(name)
        if not isinstance(entry, dict):
            if name in required:
                errors.append({
                    "field": f"artifacts.{name}",
                    "message": "required artifact metadata entry is missing "
                               "or not an object",
                })
            continue
        for key in _ARTIFACT_METADATA_KEYS:
            if key not in entry:
                errors.append({
                    "field": f"artifacts.{name}.{key}",
                    "message": f"artifact metadata key {key!r} is missing",
                })
        if entry.get("path") != name:
            errors.append({
                "field": f"artifacts.{name}.path",
                "message": f"artifact path does not match key {name!r}",
            })
        sha256 = entry.get("sha256")
        if sha256 is not None and not _SHA256_HEX_PATTERN.match(str(sha256)):
            errors.append({
                "field": f"artifacts.{name}.sha256",
                "message": "sha256 must be a 64-character lowercase hex digest",
            })
        actual_path = os.path.join(run_dir, name)
        exists = os.path.isfile(actual_path)
        if entry.get("exists") is not exists:
            errors.append({
                "field": f"artifacts.{name}.exists",
                "message": f"recorded exists={entry.get('exists')!r} but "
                           f"actual is {exists}",
            })
        if exists:
            actual_size = os.path.getsize(actual_path)
            if entry.get("size_bytes") != actual_size:
                errors.append({
                    "field": f"artifacts.{name}.size_bytes",
                    "message": f"recorded size {entry.get('size_bytes')!r} "
                               f"does not match actual {actual_size}",
                })
            actual_sha = _sha256_file(actual_path)
            if sha256 != actual_sha:
                errors.append({
                    "field": f"artifacts.{name}.sha256",
                    "message": "recorded sha256 does not match recomputed digest",
                })
        elif name in required:
            errors.append({
                "field": f"artifacts.{name}.exists",
                "message": f"required artifact {name!r} does not exist",
            })

    for name in sorted(raw_artifacts):
        if name not in expected:
            errors.append({
                "field": f"artifacts.{name}",
                "message": f"unexpected artifact entry {name!r}",
            })


def _verify_results_identity(run_dir: str, manifest: Dict[str, Any],
                             errors: List[Dict[str, str]]) -> None:
    """Verify results.json identity, fingerprint, and attempt match the manifest."""
    results_path = os.path.join(run_dir, "results.json")
    if not os.path.isfile(results_path):
        errors.append({
            "field": "results.json",
            "message": "results.json is missing",
        })
        return
    try:
        with open(results_path, encoding="utf-8") as handle:
            results = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        errors.append({
            "field": "results.json",
            "message": f"results.json is unreadable: {exc}",
        })
        return
    if not isinstance(results, dict):
        errors.append({
            "field": "results.json",
            "message": "results.json root must be an object",
        })
        return

    manifest_identity = manifest.get("identity_inputs")
    results_identity = results.get("identity_inputs")
    if _sanitize(results_identity) != manifest_identity:
        errors.append({
            "field": "results.json.identity_inputs",
            "message": "results.json identity_inputs does not match the manifest",
        })
    for key in ("fingerprint", "attempt"):
        if manifest.get(key) != results.get(key):
            errors.append({
                "field": f"results.json.{key}",
                "message": f"results.json {key} does not match the manifest",
            })


def verify_completed_manifest(run_dir: str) -> Dict[str, Any]:
    """Verify a completed manifest and its artifacts for scientific integrity.

    Dependency-light: no torch or framework imports. Returns
    ``{"ok": bool, "errors": [{"field": str, "message": str}, ...]}``.
    """
    errors: List[Dict[str, str]] = []
    try:
        manifest = load_manifest(run_dir)
    except (OSError, json.JSONDecodeError, ProvenanceError) as exc:
        return {
            "ok": False,
            "errors": [{"field": "manifest", "message": str(exc)}],
        }

    if manifest.get("status") != "completed":
        errors.append({
            "field": "status",
            "message": f"expected 'completed', got {manifest.get('status')!r}",
        })
    for field in _COMPLETED_REQUIRED_FIELDS:
        if field not in manifest:
            errors.append({
                "field": field,
                "message": f"completed manifest is missing required field {field!r}",
            })

    _verify_artifacts(run_dir, manifest, errors)
    _verify_results_identity(run_dir, manifest, errors)
    return {"ok": not errors, "errors": errors}
