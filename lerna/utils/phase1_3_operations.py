"""Operational controls for claim-ready Phase 1.3 matrices.

This module is dependency-light. It persists an immutable matrix plan and
execution environment, classifies existing attempts without creating new
ones, supports explicit stale-running recovery, and freezes deterministic
whole-matrix validation evidence.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PLAN_SCHEMA_VERSION = 1
ENVIRONMENT_SCHEMA_VERSION = 1
VALIDATION_SCHEMA_VERSION = 1
PLAN_FILENAME = "matrix_plan.json"
PLAN_CHECKSUM_FILENAME = "matrix_plan.sha256"
ENVIRONMENT_FILENAME = "matrix_environment.json"
VALIDATION_FILENAME = "matrix_validation.json"
PILOT_SEED = 7
MRPC_TRAIN_SAMPLES = 3668
MRPC_VALIDATION_SAMPLES = 408
MRPC_TOTAL_STEPS = 575
PRODUCTION_SEEDS = (
    42,
    123,
    456,
    789,
    1024,
    2025,
    4096,
    8192,
    16384,
    32768,
)
_REQUIRED_OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
}
_ATTEMPT_PREFIX = "attempt-"
_PROVENANCE_CACHE = None
_RESULTS_VALIDATOR_CACHE = None
_REQUIRED_PACKAGES = (
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "numpy",
    "accelerate",
    "scipy",
    "scikit-learn",
    "pyarrow",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
)
_ALLOWED_ROOT_FILES = {
    PLAN_FILENAME,
    PLAN_CHECKSUM_FILENAME,
    ENVIRONMENT_FILENAME,
    VALIDATION_FILENAME,
}


class Phase13OperationalError(RuntimeError):
    """Raised when a persisted Phase 1.3 operational invariant fails."""


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise Phase13OperationalError(
            f"value is not canonical JSON: {exc}"
        ) from exc
    return payload.encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of one canonical JSON value."""
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ) + "\n"
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def _write_immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        existing = _load_json(path)
        if _canonical_json_bytes(existing) != _canonical_json_bytes(value):
            raise Phase13OperationalError(
                f"refusing to overwrite non-identical evidence: {path}"
            )
        return
    _atomic_write_json(path, value)


def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise Phase13OperationalError(
            f"could not read JSON evidence {path}: {exc}"
        ) from exc


def _run_command(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(
            command,
            cwd=None if cwd is None else str(cwd),
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise Phase13OperationalError(
            f"command failed: {' '.join(command)}: {exc}"
        ) from exc


def _validate_root_structure(root: Path) -> None:
    for child in root.iterdir():
        if child.is_dir():
            continue
        if child.name not in _ALLOWED_ROOT_FILES:
            raise Phase13OperationalError(
                f"unexpected root file: {child}"
            )


def validate_power_evidence(results: dict[str, Any]) -> None:
    power = results.get("power_evidence")
    if not isinstance(power, dict):
        raise Phase13OperationalError("results.json lacks power_evidence object")
    if power.get("authoritative_copy") != "results.json":
        raise Phase13OperationalError(
            f"power evidence authority drift: {power.get('authoritative_copy')!r}"
        )
    if not isinstance(power.get("measurement_source"), str):
        raise Phase13OperationalError("power evidence missing measurement_source")
    if not isinstance(power.get("energy_valid"), bool):
        raise Phase13OperationalError("power evidence energy_valid is not bool")
    if not isinstance(power.get("energy_invalid_reason"), str):
        raise Phase13OperationalError("power evidence energy_invalid_reason is not str")
    if not isinstance(power.get("gpu_name"), str) or not power.get("gpu_name"):
        raise Phase13OperationalError("power evidence missing gpu_name")
    gpu_index = power.get("gpu_index")
    if not isinstance(gpu_index, int) or isinstance(gpu_index, bool):
        raise Phase13OperationalError("power evidence gpu_index is not int")
    if not isinstance(power.get("gpu_selector"), str) or not power.get("gpu_selector"):
        raise Phase13OperationalError("power evidence missing gpu_selector")
    sample_interval = power.get("sample_interval_s")
    if not isinstance(sample_interval, (int, float)) or isinstance(sample_interval, bool) or sample_interval <= 0:
        raise Phase13OperationalError("power evidence sample_interval_s is not positive number")
    query_count = power.get("nvidia_smi_query_count")
    if not isinstance(query_count, int) or isinstance(query_count, bool) or query_count < 0:
        raise Phase13OperationalError("power evidence nvidia_smi_query_count is not non-negative int")
    success_count = power.get("nvidia_smi_success_count")
    if not isinstance(success_count, int) or isinstance(success_count, bool) or success_count < 0:
        raise Phase13OperationalError("power evidence nvidia_smi_success_count is not non-negative int")
    if success_count > query_count:
        raise Phase13OperationalError(
            f"power evidence success_count {success_count} > query_count {query_count}"
        )
    total_energy = power.get("total_energy_kwh")
    if not isinstance(total_energy, (int, float)) or isinstance(total_energy, bool) or total_energy < 0:
        raise Phase13OperationalError("power evidence total_energy_kwh is not non-negative number")
    raw_samples = power.get("raw_samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise Phase13OperationalError("power evidence raw_samples is not a non-empty list")
    for index, sample in enumerate(raw_samples):
        if not isinstance(sample, dict):
            raise Phase13OperationalError(f"power evidence raw_samples[{index}] is not an object")
        if not isinstance(sample.get("timestamp"), (int, float)) or isinstance(sample.get("timestamp"), bool):
            raise Phase13OperationalError(f"power evidence raw_samples[{index}] missing numeric timestamp")
        if not isinstance(sample.get("power_w"), (int, float)) or isinstance(sample.get("power_w"), bool):
            raise Phase13OperationalError(f"power evidence raw_samples[{index}] missing numeric power_w")
    per_step = power.get("per_step_energy")
    if not isinstance(per_step, list):
        raise Phase13OperationalError("power evidence per_step_energy is not a list")
    for index, step in enumerate(per_step):
        if not isinstance(step, dict):
            raise Phase13OperationalError(f"power evidence per_step_energy[{index}] is not an object")
        if not isinstance(step.get("step"), int) or isinstance(step.get("step"), bool):
            raise Phase13OperationalError(f"power evidence per_step_energy[{index}] missing int step")
        if not isinstance(step.get("step_kwh"), (int, float)) or isinstance(step.get("step_kwh"), bool):
            raise Phase13OperationalError(f"power evidence per_step_energy[{index}] missing numeric step_kwh")


def require_clean_git_state(
    repo_root: str | os.PathLike[str],
    *,
    expected_sha: str | None = None,
) -> dict[str, Any]:
    """Require the exact commit and a tree clean of tracked and untracked files."""
    root = Path(repo_root).resolve()
    sha = _run_command(["git", "rev-parse", "HEAD"], cwd=root)
    status = _run_command(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
    )
    entries = status.splitlines() if status else []
    if entries:
        raise Phase13OperationalError(
            "claim-ready Phase 1.3 requires a fully clean Git tree, including "
            f"untracked files; found {entries!r}"
        )
    if expected_sha is not None and sha != expected_sha:
        raise Phase13OperationalError(
            f"Git commit drift: current {sha!r} != planned {expected_sha!r}"
        )
    return {"commit_sha": sha, "status_entries": []}


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in _REQUIRED_PACKAGES:
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            raise Phase13OperationalError(
                f"required package {distribution!r} is not installed"
            )
    cuda_version = os.environ.get("CUDA_VERSION")
    if cuda_version:
        versions["cuda"] = cuda_version
    cudnn_version = os.environ.get("CUDNN_VERSION")
    if cudnn_version:
        versions["cudnn"] = cudnn_version
    return versions


def _requirements_hashes(repo_root: Path) -> dict[str, str]:
    candidates = [repo_root / "setup.py"]
    requirements_dir = repo_root / "requirements"
    if requirements_dir.is_dir():
        candidates.extend(sorted(requirements_dir.glob("*.txt")))
    return {
        str(path.relative_to(repo_root)): file_sha256(path)
        for path in candidates
        if path.is_file()
    }


def _selected_gpu_evidence() -> dict[str, Any]:
    selector = os.environ.get("LERNA_NVIDIA_SMI_GPU", "0")
    output = _run_command(
        [
            "nvidia-smi",
            "-i",
            selector,
            "--query-gpu=name,uuid,memory.total,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = [row.strip() for row in output.splitlines() if row.strip()]
    if len(rows) != 1:
        raise Phase13OperationalError(
            f"expected exactly one selected GPU row, got {rows!r}"
        )
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 4:
        raise Phase13OperationalError(
            f"unexpected nvidia-smi identity row: {rows[0]!r}"
        )
    name, uuid, memory_text, driver = parts
    try:
        memory_mib = int(float(memory_text))
    except ValueError as exc:
        raise Phase13OperationalError(
            f"invalid GPU memory value: {memory_text!r}"
        ) from exc
    if "V100" not in name.upper() or not 31_000 <= memory_mib <= 33_500:
        raise Phase13OperationalError(
            "production profile requires one selected NVIDIA V100 32 GB; "
            f"got name={name!r}, memory_mib={memory_mib}"
        )
    return {
        "selector": selector,
        "name": name,
        "uuid": uuid,
        "memory_total_mib": memory_mib,
        "driver_version": driver,
    }


def _cache_tree_evidence(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_dir():
        raise Phase13OperationalError(
            f"required offline {label} cache is missing: {path}"
        )
    entries: list[dict[str, Any]] = []
    total_bytes = 0
    for item in sorted(path.rglob("*")):
        if item.is_dir():
            continue
        if item.is_symlink() and not item.exists():
            raise Phase13OperationalError(f"broken {label} cache symlink: {item}")
        resolved = item.resolve()
        size = resolved.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": str(item.relative_to(path)),
                "link_target": os.readlink(item) if item.is_symlink() else None,
                "resolved_name": resolved.name,
                "size": size,
                "content_sha256": _content_sha256(resolved),
            }
        )
    if not entries:
        raise Phase13OperationalError(
            f"required offline {label} cache is empty: {path}"
        )
    return {
        "path": str(path),
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "inventory_sha256": canonical_sha256(entries),
    }


def _ettin_snapshot_evidence(
    hf_home: Path,
    *,
    model_id: str,
    model_revision: str,
) -> dict[str, Any]:
    cache_name = f"models--{model_id.replace('/', '--')}"
    candidates = (
        hf_home / cache_name / "snapshots" / model_revision,
        hf_home / "hub" / cache_name / "snapshots" / model_revision,
    )
    snapshot = next((path for path in candidates if path.is_dir()), None)
    if snapshot is None:
        raise Phase13OperationalError(
            f"pinned Ettin snapshot is missing under HF_HOME={str(hf_home)!r}"
        )
    entries: list[dict[str, Any]] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_dir():
            continue
        if path.is_symlink() and not path.exists():
            raise Phase13OperationalError(f"broken cache symlink: {path}")
        resolved = path.resolve()
        entries.append(
            {
                "path": str(path.relative_to(snapshot)),
                "link_target": os.readlink(path) if path.is_symlink() else None,
                "resolved_name": resolved.name,
                "size": resolved.stat().st_size,
                "content_sha256": _content_sha256(resolved),
            }
        )
    if not entries:
        raise Phase13OperationalError(f"pinned Ettin snapshot is empty: {snapshot}")
    return {
        "snapshot_path": str(snapshot),
        "file_count": len(entries),
        "inventory_sha256": canonical_sha256(entries),
        "entries": entries,
    }


def collect_phase1_3_environment(
    *,
    repo_root: str | os.PathLike[str],
    profile: str,
    model_id: str,
    model_revision: str,
    hardware_config: dict[str, Any],
) -> dict[str, Any]:
    """Collect and enforce the frozen V100/Ettin offline environment."""
    offline = {key: os.environ.get(key) for key in _REQUIRED_OFFLINE_ENV}
    if offline != _REQUIRED_OFFLINE_ENV:
        raise Phase13OperationalError(
            f"offline environment mismatch: {offline!r} != "
            f"{_REQUIRED_OFFLINE_ENV!r}"
        )
    hf_home_text = os.environ.get("HF_HOME")
    if not hf_home_text:
        raise Phase13OperationalError("HF_HOME must be set for claim-ready runs")
    if profile != "server":
        raise Phase13OperationalError(
            f"production profile must resolve to 'server', got {profile!r}"
        )
    fp16 = hardware_config.get("fp16")
    bf16 = hardware_config.get("bf16")
    if fp16 is not True or bf16 is not False:
        raise Phase13OperationalError(
            f"V100 execution requires fp16=True and bf16=False, got "
            f"fp16={fp16!r}, bf16={bf16!r}"
        )
    root = Path(repo_root).resolve()
    hf_home = Path(hf_home_text).resolve()
    dataset_cache = Path(
        os.environ.get("HF_DATASETS_CACHE", str(hf_home / "datasets"))
    ).resolve()
    metric_module_cache = Path(
        os.environ.get("HF_MODULES_CACHE", str(hf_home / "modules"))
    ).resolve()
    return {
        "schema_version": ENVIRONMENT_SCHEMA_VERSION,
        "python": sys.version.split()[0],
        "packages": _package_versions(),
        "requirements_sha256": _requirements_hashes(root),
        "profile": profile,
        "hardware_config": {
            key: hardware_config.get(key)
            for key in (
                "per_device_train_batch_size",
                "per_device_eval_batch_size",
                "gradient_accumulation_steps",
                "fp16",
                "bf16",
                "gradient_checkpointing",
                "dataloader_num_workers",
                "max_samples",
            )
        },
        "offline_environment": offline,
        "hf_home": str(hf_home),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_snapshot": _ettin_snapshot_evidence(
            hf_home,
            model_id=model_id,
            model_revision=model_revision,
        ),
        "dataset_cache": _cache_tree_evidence(
            dataset_cache, label="dataset"
        ),
        "metric_module_cache": _cache_tree_evidence(
            metric_module_cache, label="metric module"
        ),
        "gpu": _selected_gpu_evidence(),
    }


def assert_environment_matches(
    planned: dict[str, Any],
    current: dict[str, Any],
) -> None:
    """Require type-strict equality for the persisted execution environment."""
    if _canonical_json_bytes(planned) != _canonical_json_bytes(current):
        raise Phase13OperationalError(
            "current dependency/cache/GPU environment does not match "
            "matrix_environment.json"
        )


def require_frozen_mrpc_facts(facts: dict[str, Any]) -> None:
    """Require the audited full-MRPC sizes and V100 training horizon."""
    expected = {
        "task": "mrpc",
        "train_samples_realized": MRPC_TRAIN_SAMPLES,
        "eval_samples_realized": MRPC_VALIDATION_SAMPLES,
        "total_steps": MRPC_TOTAL_STEPS,
    }
    for key, value in expected.items():
        if type(facts.get(key)) is not type(value) or facts.get(key) != value:
            raise Phase13OperationalError(
                f"frozen MRPC fact {key} drift: {facts.get(key)!r} != {value!r}"
            )


def persist_phase1_3_plan(
    *,
    base_output_dir: str | os.PathLike[str],
    plan: list[dict[str, Any]],
    tasks: list[str],
    seeds: list[int],
    target_skip_rates: list[float],
    matrix_kind: str,
    git_sha: str,
    environment: dict[str, Any],
) -> dict[str, Any]:
    """Atomically create a fresh output root with immutable plan evidence."""
    root = Path(base_output_dir)
    if root.exists():
        raise Phase13OperationalError(
            f"planning requires an absent fresh output root: {root}"
        )
    if matrix_kind not in {"pilot", "production"}:
        raise Phase13OperationalError(
            f"invalid matrix kind: {matrix_kind!r}"
        )
    parent = root.parent
    parent.mkdir(parents=True, exist_ok=True)
    environment_sha = canonical_sha256(environment)
    envelope = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "created_time_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_kind": matrix_kind,
        "git_sha": git_sha,
        "dimensions": {
            "tasks": list(tasks),
            "seeds": list(seeds),
            "target_skip_rates": list(target_skip_rates),
        },
        "plan_sha256": canonical_sha256(plan),
        "environment_sha256": environment_sha,
        "plan": plan,
    }
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.planning-", dir=str(parent))
    )
    try:
        _atomic_write_json(staging / ENVIRONMENT_FILENAME, environment)
        plan_path = staging / PLAN_FILENAME
        _atomic_write_json(plan_path, envelope)
        _atomic_write_text(
            staging / PLAN_CHECKSUM_FILENAME,
            f"{file_sha256(plan_path)}  {PLAN_FILENAME}\n",
        )
        if root.exists():
            raise Phase13OperationalError(
                f"output root appeared during planning: {root}"
            )
        os.replace(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return envelope


def load_phase1_3_plan(
    base_output_dir: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load and hash-verify the persisted plan and environment."""
    root = Path(base_output_dir)
    plan_path = root / PLAN_FILENAME
    checksum_path = root / PLAN_CHECKSUM_FILENAME
    try:
        checksum_parts = checksum_path.read_text(encoding="utf-8").split()
    except OSError as exc:
        raise Phase13OperationalError(
            f"could not read matrix plan checksum: {exc}"
        ) from exc
    try:
        actual_plan_file_sha = file_sha256(plan_path)
    except OSError as exc:
        raise Phase13OperationalError(
            f"could not hash matrix_plan.json: {exc}"
        ) from exc
    if checksum_parts != [actual_plan_file_sha, PLAN_FILENAME]:
        raise Phase13OperationalError("matrix_plan.json file checksum mismatch")
    envelope = _load_json(plan_path)
    environment = _load_json(root / ENVIRONMENT_FILENAME)
    if not isinstance(envelope, dict):
        raise Phase13OperationalError("matrix_plan.json must contain an object")
    if envelope.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise Phase13OperationalError("unsupported matrix plan schema")
    plan = envelope.get("plan")
    if not isinstance(plan, list):
        raise Phase13OperationalError("matrix_plan.json plan must be a list")
    if envelope.get("plan_sha256") != canonical_sha256(plan):
        raise Phase13OperationalError("matrix_plan.json plan hash mismatch")
    if envelope.get("environment_sha256") != canonical_sha256(environment):
        raise Phase13OperationalError("matrix environment hash mismatch")
    dimensions = envelope.get("dimensions")
    if not isinstance(dimensions, dict):
        raise Phase13OperationalError("matrix plan dimensions must be an object")
    return {
        "root": root,
        "envelope": envelope,
        "plan": plan,
        "environment": environment,
        "dimensions": dimensions,
        "plan_file_sha256": checksum_parts[0],
    }


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Phase13OperationalError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _provenance_module():
    global _PROVENANCE_CACHE
    if _PROVENANCE_CACHE is None:
        _PROVENANCE_CACHE = _load_module(
            "lerna_phase1_3_operations_provenance",
            Path(__file__).resolve().parent / "run_provenance.py",
        )
    return _PROVENANCE_CACHE


def _results_validator_module():
    global _RESULTS_VALIDATOR_CACHE
    if _RESULTS_VALIDATOR_CACHE is None:
        _RESULTS_VALIDATOR_CACHE = _load_module(
            "lerna_phase1_3_operations_results_validator",
            Path(__file__).resolve().parents[2]
            / "scripts"
            / "validate_skip_policy_results.py",
        )
    return _RESULTS_VALIDATOR_CACHE


def _expected_output_paths(cell: dict[str, Any]) -> dict[str, str]:
    paths = {
        "results": "results.json",
        "instrumentation": "instrumentation.json",
        "manifest": "run_manifest.json",
    }
    if cell["online_diagnostics"]["enabled"]:
        paths["ler_diagnostics"] = "ler_diagnostics.json"
    return paths


def _required_artifacts(cell: dict[str, Any]) -> list[str]:
    artifacts = ["instrumentation.json"]
    if cell["online_diagnostics"]["enabled"]:
        artifacts.append("ler_diagnostics.json")
    return artifacts


def _assert_attempt_manifest_binding(
    cell: dict[str, Any],
    manifest: dict[str, Any],
    attempt_num: int,
    attempt_dir: Path,
) -> None:
    expected = {
        "fingerprint": cell["fingerprint"],
        "identity_inputs": cell["identity_inputs"],
        "attempt": attempt_num,
    }
    for key, value in expected.items():
        if type(manifest.get(key)) is not type(value) or manifest.get(key) != value:
            raise Phase13OperationalError(
                f"attempt manifest {key} drift in {attempt_dir}"
            )


def _read_completed_attempt(
    cell: dict[str, Any],
    attempt_dir: Path,
    attempt_num: int,
) -> dict[str, Any]:
    provenance = _provenance_module()
    verification = provenance.verify_completed_manifest(str(attempt_dir))
    if verification.get("ok") is not True:
        raise Phase13OperationalError(
            f"completed manifest integrity failed in {attempt_dir}: "
            f"{verification.get('errors')!r}"
        )
    manifest = _load_json(attempt_dir / "run_manifest.json")
    results = _load_json(attempt_dir / "results.json")
    if not isinstance(manifest, dict) or not isinstance(results, dict):
        raise Phase13OperationalError(
            f"completed evidence is not object-shaped: {attempt_dir}"
        )
    expected_manifest_values = {
        "status": "completed",
        "fingerprint": cell["fingerprint"],
        "identity_inputs": cell["identity_inputs"],
        "attempt": attempt_num,
        "output_paths": _expected_output_paths(cell),
    }
    expected_classification = cell.get(
        "provenance_classification", "matched_claim"
    )
    if expected_classification in ("matched_claim", "pilot_non_claim"):
        expected_manifest_values["provenance_classification"] = expected_classification
    for key, expected in expected_manifest_values.items():
        if type(manifest.get(key)) is not type(expected) or manifest.get(key) != expected:
            raise Phase13OperationalError(
                f"manifest {key} drift in {attempt_dir}"
            )
    expected_results = {
        "task": cell["task"],
        "seed": cell["training_seed"],
        "ablation": cell["arm"],
        "model": cell["model_id"],
        "model_revision": cell.get("model_revision"),
        "fingerprint": cell["fingerprint"],
        "identity_inputs": cell["identity_inputs"],
        "attempt": attempt_num,
    }
    for key, expected in expected_results.items():
        if type(results.get(key)) is not type(expected) or results.get(key) != expected:
            raise Phase13OperationalError(
                f"results {key} drift in {attempt_dir}"
            )
    validation = manifest.get("validation")
    if not isinstance(validation, dict) or validation.get(
        "valid_for_matched_budget"
    ) is not True:
        raise Phase13OperationalError(
            f"completed attempt lacks matched-budget approval: {attempt_dir}"
        )
    report = _results_validator_module().validate_results(
        attempt_dir / "results.json",
        required_artifacts=_required_artifacts(cell),
    )
    if report.ok is not True or report.valid_for_matched_budget is not True:
        raise Phase13OperationalError(
            f"Piece 5 validation failed for completed attempt: {attempt_dir}"
        )
    validate_power_evidence(results)
    return {
        "attempt": attempt_num,
        "attempt_dir": attempt_dir,
        "manifest": manifest,
        "results": results,
    }


def _attempt_number(name: str) -> int | None:
    if not name.startswith(_ATTEMPT_PREFIX):
        return None
    suffix = name[len(_ATTEMPT_PREFIX) :]
    if len(suffix) != 3 or not suffix.isdigit():
        return None
    value = int(suffix)
    return value if value >= 1 else None


def scan_phase1_3_progress(
    plan: list[dict[str, Any]],
    *,
    base_output_dir: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Classify every planned cell without modifying attempts."""
    root = Path(base_output_dir)
    _validate_root_structure(root)
    expected_fingerprints = {
        (cell["arm"], cell["fingerprint"]) for cell in plan
    }
    for child in root.iterdir():
        if child.is_dir():
            if child.name not in {cell["arm"] for cell in plan}:
                raise Phase13OperationalError(
                    f"unexpected directory in matrix root: {child}"
                )
            for fingerprint_dir in child.iterdir():
                if not fingerprint_dir.is_dir():
                    raise Phase13OperationalError(
                        f"unexpected non-directory in arm path: {fingerprint_dir}"
                    )
                if (child.name, fingerprint_dir.name) not in expected_fingerprints:
                    raise Phase13OperationalError(
                        f"foreign cell directory in matrix root: {fingerprint_dir}"
                    )
        else:
            if child.name not in _ALLOWED_ROOT_FILES:
                raise Phase13OperationalError(
                    f"unexpected root file: {child}"
                )

    progress: list[dict[str, Any]] = []
    for cell in plan:
        cell_dir = root / cell["arm"] / cell["fingerprint"]
        failed: list[Path] = []
        running: list[Path] = []
        completed: list[dict[str, Any]] = []
        if cell_dir.exists():
            if not cell_dir.is_dir():
                raise Phase13OperationalError(
                    f"planned cell path is not a directory: {cell_dir}"
                )
            for child in sorted(cell_dir.iterdir()):
                if not child.is_dir():
                    raise Phase13OperationalError(
                        f"unexpected file in cell directory: {child}"
                    )
                attempt_num = _attempt_number(child.name)
                if attempt_num is None:
                    raise Phase13OperationalError(
                        f"noncanonical attempt directory: {child}"
                    )
                manifest = _load_json(child / "run_manifest.json")
                if not isinstance(manifest, dict):
                    raise Phase13OperationalError(
                        f"attempt manifest is not an object: {child}"
                    )
                _assert_attempt_manifest_binding(
                    cell, manifest, attempt_num, child
                )
                status = manifest.get("status")
                if status == "failed":
                    failed.append(child)
                elif status == "running":
                    running.append(child)
                elif status == "completed":
                    completed.append(
                        _read_completed_attempt(cell, child, attempt_num)
                    )
                else:
                    raise Phase13OperationalError(
                        f"unknown attempt status {status!r}: {child}"
                    )
        if running:
            raise Phase13OperationalError(
                "running attempts require explicit stale recovery: "
                + ", ".join(str(path) for path in running)
            )
        if len(completed) > 1:
            raise Phase13OperationalError(
                f"duplicate valid completed attempts for {cell['fingerprint']}"
            )
        progress.append(
            {
                "cell": cell,
                "state": "completed" if completed else "pending",
                "completed": completed[0] if completed else None,
                "failed_attempts": failed,
            }
        )
    return progress


def assert_fresh_execution(
    plan: list[dict[str, Any]],
    *,
    base_output_dir: str | os.PathLike[str],
) -> None:
    root = Path(base_output_dir)
    for cell in plan:
        cell_dir = root / cell["arm"] / cell["fingerprint"]
        if cell_dir.exists():
            raise Phase13OperationalError(
                f"fresh execution refuses existing cell path: {cell_dir}"
            )


def recover_stale_running_attempts(
    plan: list[dict[str, Any]],
    *,
    base_output_dir: str | os.PathLike[str],
    stale_after_hours: float,
    now: datetime | None = None,
) -> list[str]:
    """Mark only explicitly old running attempts failed, preserving directories."""
    if not isinstance(stale_after_hours, (int, float)) or isinstance(
        stale_after_hours, bool
    ) or stale_after_hours <= 0:
        raise Phase13OperationalError("stale_after_hours must be positive")
    current = now or datetime.now(timezone.utc)
    if current.tzinfo is None:
        raise Phase13OperationalError("recovery clock must be timezone-aware")
    root = Path(base_output_dir)

    expected_fingerprints = {
        (cell["arm"], cell["fingerprint"]) for cell in plan
    }
    blockers: list[str] = []
    for child in root.iterdir():
        if child.is_dir():
            if child.name not in {cell["arm"] for cell in plan}:
                blockers.append(f"unexpected matrix root directory: {child}")
            for fingerprint_dir in child.iterdir():
                if not fingerprint_dir.is_dir():
                    blockers.append(
                        f"unexpected non-directory in arm path: {fingerprint_dir}"
                    )
                    continue
                if (child.name, fingerprint_dir.name) not in expected_fingerprints:
                    blockers.append(
                        f"foreign cell directory in matrix root: {fingerprint_dir}"
                    )
                cell_dir = child / fingerprint_dir.name
                for attempt_dir in sorted(cell_dir.iterdir()):
                    if not attempt_dir.is_dir():
                        blockers.append(f"unexpected file in cell directory: {attempt_dir}")
                        continue
                    if _attempt_number(attempt_dir.name) is None:
                        blockers.append(
                            f"noncanonical attempt directory: {attempt_dir}"
                        )
                        continue
                    manifest = _load_json(attempt_dir / "run_manifest.json")
                    if not isinstance(manifest, dict):
                        blockers.append(f"malformed manifest {attempt_dir}")
                        continue
                    cell = next(
                        (
                            cell
                            for cell in plan
                            if cell["arm"] == child.name
                            and cell["fingerprint"] == fingerprint_dir.name
                        ),
                        None,
                    )
                    if cell is None:
                        blockers.append(
                            f"foreign cell without plan binding: {attempt_dir}"
                        )
                        continue
                    _assert_attempt_manifest_binding(
                        cell,
                        manifest,
                        _attempt_number(attempt_dir.name),
                        attempt_dir,
                    )
                    status = manifest.get("status")
                    if status == "running":
                        started = manifest.get("start_time_utc")
                        try:
                            start_time = datetime.fromisoformat(started)
                        except (TypeError, ValueError):
                            blockers.append(
                                f"invalid running start_time_utc {attempt_dir}"
                            )
                            continue
                        if start_time.tzinfo is None:
                            blockers.append(
                                f"naive running start_time_utc {attempt_dir}"
                            )
                            continue
                        age_hours = (current - start_time).total_seconds() / 3600.0
                        if age_hours < stale_after_hours:
                            blockers.append(
                                f"running attempt is only {age_hours:.2f}h old: {attempt_dir}"
                            )
        else:
            if child.name not in _ALLOWED_ROOT_FILES:
                blockers.append(f"unexpected root file: {child}")
    if blockers:
        raise Phase13OperationalError(
            "stale recovery refused: " + "; ".join(blockers)
        )

    candidates: list[Path] = []
    for cell in plan:
        cell_dir = root / cell["arm"] / cell["fingerprint"]
        if not cell_dir.is_dir():
            continue
        for attempt_dir in sorted(cell_dir.iterdir()):
            if not attempt_dir.is_dir():
                continue
            if _attempt_number(attempt_dir.name) is None:
                continue
            manifest = _load_json(attempt_dir / "run_manifest.json")
            if not isinstance(manifest, dict):
                continue
            if manifest.get("status") != "running":
                continue
            _assert_attempt_manifest_binding(
                cell,
                manifest,
                _attempt_number(attempt_dir.name),
                attempt_dir,
            )
            started = manifest.get("start_time_utc")
            try:
                start_time = datetime.fromisoformat(started)
            except (TypeError, ValueError):
                continue
            if start_time.tzinfo is None:
                continue
            age_hours = (current - start_time).total_seconds() / 3600.0
            if age_hours >= stale_after_hours:
                candidates.append(attempt_dir)
    if not candidates:
        return []
    provenance = _provenance_module()
    for attempt_dir in candidates:
        provenance.finalize_manifest_failed(
            str(attempt_dir),
            RuntimeError("stale running attempt recovered before resume"),
        )
    return [str(path) for path in candidates]


def _power_evidence_sha(results: dict[str, Any]) -> str:
    power = results.get("power_evidence")
    if not isinstance(power, dict):
        raise Phase13OperationalError("completed results lack power_evidence")
    return canonical_sha256(power)


def freeze_matrix_validation(
    *,
    base_output_dir: str | os.PathLike[str],
    bundle: dict[str, Any],
    valid_runs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Write or re-verify one deterministic completed-matrix report."""
    root = Path(base_output_dir)
    runs: list[dict[str, Any]] = []
    for run in valid_runs:
        attempt_dir = Path(run["results_path"]).parent
        results = _load_json(attempt_dir / "results.json")
        manifest = _load_json(attempt_dir / "run_manifest.json")
        validate_power_evidence(results)
        if bundle["envelope"]["matrix_kind"] == "production":
            classification = manifest.get("provenance_classification")
            if classification == "pilot_non_claim":
                raise Phase13OperationalError(
                    "production matrix validation rejects pilot evidence: "
                    f"{attempt_dir}"
                )
        relative_attempt = str(attempt_dir.relative_to(root))
        runs.append(
            {
                "cell_id": list(run["cell_id"]),
                "attempt": run["attempt_num"],
                "attempt_dir": relative_attempt,
                "manifest_sha256": file_sha256(
                    attempt_dir / "run_manifest.json"
                ),
                "results_sha256": file_sha256(attempt_dir / "results.json"),
                "power_evidence_sha256": _power_evidence_sha(results),
                "energy_valid": results["power_evidence"].get("energy_valid"),
            }
        )
    report = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "status": "completed",
        "completed_matrix_valid": True,
        "matrix_kind": bundle["envelope"]["matrix_kind"],
        "git_sha": bundle["envelope"]["git_sha"],
        "plan_sha256": bundle["envelope"]["plan_sha256"],
        "plan_file_sha256": bundle["plan_file_sha256"],
        "environment_sha256": bundle["envelope"]["environment_sha256"],
        "n_cells": len(bundle["plan"]),
        "n_valid_runs": len(valid_runs),
        "runs": runs,
    }
    if report["n_cells"] != report["n_valid_runs"]:
        raise Phase13OperationalError(
            "cannot freeze incomplete completed-matrix evidence"
        )
    _write_immutable_json(root / VALIDATION_FILENAME, report)
    return report
