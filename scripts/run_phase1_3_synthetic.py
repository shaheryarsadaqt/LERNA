#!/usr/bin/env python3
"""Run the deterministic offline Phase 1.3 synthetic protocol on CPU."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_DISABLED"] = "true"

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lerna.trainers import (
    AlwaysFalsePolicy,
    ComputeSavingMechanism,
    FixedPhaseStratifiedRandomPolicy,
    LERGuidedStratifiedPolicy,
    LERGuidedStratifiedSafetyPolicy,
    PhaseStratifiedGuardedRandomPolicy,
    RandomSkipPolicy,
    TrueBackwardSkippingTrainer,
)
from lerna.utils.phase1_3_completed_matrix import (
    validate_phase1_3_completed_matrix,
)
from lerna.utils.phase1_3_matrix import (
    PHASE1_3_CANONICAL_ARMS,
    POLICY_MIN_STEP,
    STRICT_TARGET_SKIP_RATES,
    validate_phase1_3_matrix_plan,
)
from lerna.utils.run_provenance import (
    CLASSIFICATION_MATCHED_CLAIM,
    collect_git_state,
    finalize_manifest_completed,
    finalize_manifest_failed,
    verify_completed_manifest,
    write_manifest_running,
)
from scripts.run_ablation_study import (
    AblationTrainer,
    assert_phase1_3_runtime_matches_plan,
    build_online_ler_artifact_contract,
    build_online_ler_runtime_metadata,
    build_online_ler_tracker,
    build_phase1_3_matrix_plan,
    build_phase_strat_controller_config,
    plan_phase1_3_cell,
)
from scripts.validate_skip_policy_results import validate_results

SYNTHETIC_TASK = "phase1_3_synthetic"
SYNTHETIC_MODEL_ID = "tiny-linear-cpu"
SYNTHETIC_MODEL_REVISION = "synthetic-cpu-local-v1"
SYNTHETIC_TOTAL_STEPS = 100
SYNTHETIC_NUM_EPOCHS = 1
SYNTHETIC_TRAIN_SIZE = 100
SYNTHETIC_EVAL_SIZE = 32
SYNTHETIC_WIDTH = 8
SYNTHETIC_CLASSES = 2
SYNTHETIC_DATA_SEED = 20260301
DEFAULT_TRAINING_SEED = 7
MAX_CONSECUTIVE_SKIPS = 4
PROBE_INTERVAL = 8
RHO_VETO_THRESHOLD = -0.2
RISK_GAMMA = 0.0
ONLINE_LER_PARAMETER_SAMPLE_SIZE = 64
ONLINE_LER_UPDATE_INTERVAL = 1
SCHEDULER_STEP_POLICY = "skip_on_backward_skip"


class SyntheticDataset(Dataset):
    """Deterministic local classification data with no external assets."""

    def __init__(self, *, size: int, width: int, seed: int):
        generator = torch.Generator().manual_seed(seed)
        inputs = torch.randn(size, width, generator=generator)
        separator = torch.linspace(-0.8, 0.8, width)
        logits = inputs @ separator
        labels = (logits > 0).to(dtype=torch.long)
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.inputs[index],
            "labels": self.labels[index],
        }


class TinyClassifier(nn.Module):
    """Small CPU model that exercises real forward and backward operations."""

    def __init__(self, *, width: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(width, num_labels)

    def forward(self, input_ids=None, labels=None, **_kwargs):
        logits = self.classifier(input_ids)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def _collate(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
    }


def _compute_metrics(prediction) -> dict[str, float]:
    predictions = prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted = np.asarray(predictions).argmax(axis=-1)
    labels = np.asarray(prediction.label_ids)
    return {"accuracy": float((predicted == labels).mean())}


def _dataset_fingerprint(dataset: SyntheticDataset) -> str:
    digest = hashlib.sha256()
    digest.update(dataset.inputs.detach().cpu().numpy().tobytes())
    digest.update(dataset.labels.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    return value


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_json_ready(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def _resolve_git_sha() -> str:
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()
    if len(sha) != 40 or any(ch not in "0123456789abcdef" for ch in sha):
        raise RuntimeError(f"could not resolve a full Git SHA: {sha!r}")
    return sha


def _require_claim_ready_checkout(git_sha: str) -> None:
    state = collect_git_state(str(REPO_ROOT))
    if state.get("commit_sha") != git_sha:
        raise RuntimeError("Git provenance SHA changed during synthetic preflight")
    if state.get("dirty"):
        raise RuntimeError(
            "The synthetic matched-claim protocol requires a clean tracked tree"
        )


def _require_fresh_output(base_output_dir: Path) -> None:
    if base_output_dir.exists() and any(base_output_dir.iterdir()):
        raise RuntimeError(
            f"synthetic output directory must be absent or empty: {base_output_dir}"
        )


def _set_training_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _data_facts(
    train_dataset: SyntheticDataset,
    eval_dataset: SyntheticDataset,
) -> dict[str, Any]:
    return {
        "task": SYNTHETIC_TASK,
        "num_epochs": SYNTHETIC_NUM_EPOCHS,
        "max_samples_requested": None,
        "max_samples_effective": None,
        "train_samples_realized": len(train_dataset),
        "eval_samples_realized": len(eval_dataset),
        "train_dataset_fingerprint": _dataset_fingerprint(train_dataset),
        "eval_dataset_fingerprint": _dataset_fingerprint(eval_dataset),
        "total_steps": SYNTHETIC_TOTAL_STEPS,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
        "effective_n_gpu": 1,
    }


def _plan_matrix(
    *,
    base_output_dir: str,
    training_seed: int,
    facts: dict[str, Any],
    git_sha: str,
) -> list[dict[str, Any]]:
    plan = build_phase1_3_matrix_plan(
        tasks=[SYNTHETIC_TASK],
        seeds=[training_seed],
        target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
        model_name=SYNTHETIC_MODEL_ID,
        model_revision=SYNTHETIC_MODEL_REVISION,
        base_output_dir=base_output_dir,
        data_facts_provider=lambda task: dict(facts),
        git_sha=git_sha,
        scheduler_step_policy=SCHEDULER_STEP_POLICY,
        max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
        probe_interval=PROBE_INTERVAL,
        rho_veto_threshold=RHO_VETO_THRESHOLD,
        risk_gamma=RISK_GAMMA,
        online_ler_mode="auto",
        online_ler_parameter_sample_size=ONLINE_LER_PARAMETER_SAMPLE_SIZE,
        online_ler_update_interval=ONLINE_LER_UPDATE_INTERVAL,
        use_rho_vg=True,
        use_safety_horizon=True,
        provenance_classification="matched_claim",
    )
    findings = validate_phase1_3_matrix_plan(
        plan,
        tasks=[SYNTHETIC_TASK],
        seeds=[training_seed],
        target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
        minimum_seed_count=1,
        base_output_dir=base_output_dir,
    )
    if findings:
        raise RuntimeError(f"unexpected synthetic plan findings: {findings}")
    if len(plan) != 12:
        raise RuntimeError(f"synthetic matrix must contain 12 cells, got {len(plan)}")
    return plan


def _build_policy(cell: dict[str, Any], tracker):
    arm = cell["arm"]
    common = {
        "target_skip_rate": cell["target_skip_rate"],
        "total_steps": cell["total_steps"],
        "min_step": cell["min_step"],
        "seed": cell["policy_seed"],
    }
    if arm == "full_finetune":
        return AlwaysFalsePolicy()
    if arm == "exact_random":
        return RandomSkipPolicy(**common)
    if arm == "fixed_phase_strat":
        return FixedPhaseStratifiedRandomPolicy(
            **common,
            max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
        )
    if arm == "phase_strat_guarded":
        if tracker is None:
            raise RuntimeError("phase_strat_guarded requires a sampled LER tracker")
        return PhaseStratifiedGuardedRandomPolicy(
            tracker,
            **common,
            max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
            rho_veto_threshold=RHO_VETO_THRESHOLD,
            spike_factor=1.0,
            use_rho_vg=True,
            use_safety_horizon=True,
            risk_gamma=RISK_GAMMA,
            probe_interval=PROBE_INTERVAL,
        )
    if arm == "ler_guided_stratified":
        if tracker is None:
            raise RuntimeError("ler_guided_stratified requires a sampled LER tracker")
        return LERGuidedStratifiedPolicy(
            tracker,
            **common,
            max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
            probe_interval=PROBE_INTERVAL,
            min_ler_observations=3,
            ler_guidance_strength=1.0,
        )
    if arm == "ler_guided_stratified_safe":
        if tracker is None:
            raise RuntimeError(
                "ler_guided_stratified_safe requires a sampled LER tracker"
            )
        return LERGuidedStratifiedSafetyPolicy(
            tracker,
            **common,
            max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
            probe_interval=PROBE_INTERVAL,
            min_ler_observations=3,
            ler_guidance_strength=1.0,
            use_rho_vg_safety=True,
            rho_veto_threshold=RHO_VETO_THRESHOLD,
            use_loss_spike_safety=True,
            loss_spike_factor=1.0,
            loss_spike_window=5,
        )
    raise ValueError(f"unsupported synthetic arm: {arm!r}")


def _canonical_ler_config(cell: dict[str, Any], policy) -> dict[str, Any]:
    effective = dict(policy.effective_config())
    effective["control"] = cell["arm"]
    effective["policy_seed"] = effective.pop("seed")
    return effective


def _runtime_controller(cell: dict[str, Any], policy) -> dict[str, Any]:
    arm = cell["arm"]
    skipping = cell["is_skipping_arm"]
    target_skip_rate = float(getattr(policy, "target_skip_rate", cell["target_skip_rate"]))
    total_steps = int(getattr(policy, "total_steps", cell["total_steps"]) or cell["total_steps"])
    min_step = int(getattr(policy, "min_step", cell["min_step"]))
    requested_quota = round(target_skip_rate * total_steps) if skipping else None
    controller = {
        "arm": arm,
        "arm_alias_of": None,
        "control": arm,
        "policy_class": type(policy).__name__,
        "compute_saving_mechanism": (
            ComputeSavingMechanism.BACKWARD_SKIPPING
            if skipping
            else ComputeSavingMechanism.NONE
        ),
        "policy_seed": int(cell["policy_seed"]),
        "target_skip_rate": target_skip_rate,
        "min_step": min_step,
        "configured_total_steps": total_steps,
        "requested_quota": requested_quota,
        "matched_budget": True,
        "is_skipping_arm": skipping,
        "allow_early_stopping_with_skipping": False,
        "early_stopping_active": False,
        "num_epochs": int(cell["num_epochs"]),
        "online_diagnostics": dict(cell["online_diagnostics"]),
        "policy_effective_config": (
            dict(policy.effective_config())
            if hasattr(policy, "effective_config")
            else {}
        ),
    }
    if arm in ("fixed_phase_strat", "phase_strat_guarded"):
        phase_kwargs = {
            "control": arm,
            "target_skip_rate": target_skip_rate,
            "total_steps": total_steps,
            "policy_seed": int(cell["policy_seed"]),
            "max_consecutive_skips": int(policy.max_consecutive_skips),
            "n_phases": int(policy.n_phases),
            "phase_weights": list(policy.phase_weights),
        }
        if arm == "phase_strat_guarded":
            phase_kwargs.update(
                guarded=True,
                rho_veto_threshold=float(policy.rho_veto_threshold),
                spike_factor=float(policy.spike_factor),
                use_rho_vg=bool(policy.use_rho_vg),
                use_safety_horizon=bool(policy.use_safety_horizon),
                risk_gamma=float(policy.risk_gamma),
            )
        controller["phase_strat_controller"] = (
            build_phase_strat_controller_config(**phase_kwargs)
        )
    elif arm in ("ler_guided_stratified", "ler_guided_stratified_safe"):
        controller["ler_guided_controller"] = _canonical_ler_config(cell, policy)
    return controller


def _assert_tracker_contract(cell: dict[str, Any], tracker) -> None:
    expected = cell["online_diagnostics"]
    if not expected["enabled"]:
        if tracker is not None:
            raise RuntimeError(f"offline arm {cell['arm']} constructed a tracker")
        return
    if tracker is None:
        raise RuntimeError(f"online arm {cell['arm']} did not construct a tracker")
    diagnostics = tracker.get_diagnostics()
    checks = {
        "mode": diagnostics.get("mode"),
        "timing": diagnostics.get("timing"),
        "parameter_sample_size": tracker.parameter_sample_size,
        "sample_seed": tracker.sample_seed,
    }
    expected_checks = {
        "mode": expected["mode"],
        "timing": expected["timing"],
        "parameter_sample_size": expected["parameter_sample_size"],
        "sample_seed": expected["sample_seed"],
    }
    if checks != expected_checks:
        raise RuntimeError(
            f"tracker configuration drift for {cell['arm']}: "
            f"{checks!r} != {expected_checks!r}"
        )


def _runtime_cell(
    *,
    planned_cell: dict[str, Any],
    facts: dict[str, Any],
    git_sha: str,
    base_output_dir: str,
    controller: dict[str, Any],
) -> dict[str, Any]:
    runtime = plan_phase1_3_cell(
        task_name=planned_cell["task"],
        training_seed=planned_cell["training_seed"],
        policy_seed=planned_cell["policy_seed"],
        ablation_name=planned_cell["arm"],
        target_skip_rate=planned_cell["target_skip_rate"],
        model_name=planned_cell["model_id"],
        model_revision=planned_cell.get("model_revision"),
        data_facts=facts,
        git_sha=git_sha,
        base_output_dir=base_output_dir,
        scheduler_step_policy=SCHEDULER_STEP_POLICY,
        max_consecutive_skips=MAX_CONSECUTIVE_SKIPS,
        probe_interval=PROBE_INTERVAL,
        rho_veto_threshold=RHO_VETO_THRESHOLD,
        risk_gamma=RISK_GAMMA,
        online_ler_mode="auto",
        online_ler_parameter_sample_size=ONLINE_LER_PARAMETER_SAMPLE_SIZE,
        online_ler_update_interval=ONLINE_LER_UPDATE_INTERVAL,
        use_rho_vg=True,
        use_safety_horizon=True,
        provenance_classification=planned_cell.get("provenance_classification", "matched_claim"),
    )
    runtime["controller_config"] = controller
    return runtime


def _allocate_attempt(planned_arm_dir: str) -> tuple[int, Path]:
    arm_dir = Path(planned_arm_dir)
    arm_dir.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        attempt_dir = arm_dir / f"attempt-{attempt:03d}"
        try:
            attempt_dir.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        return attempt, attempt_dir


def _training_arguments(attempt_dir: Path, seed: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(attempt_dir),
        max_steps=SYNTHETIC_TOTAL_STEPS,
        num_train_epochs=SYNTHETIC_NUM_EPOCHS,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        weight_decay=0.0,
        warmup_steps=0,
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        use_cpu=True,
        fp16=False,
        bf16=False,
        disable_tqdm=True,
        seed=seed,
        data_seed=seed,
    )


def _assert_execution_contract(
    cell: dict[str, Any],
    instrumentation: dict[str, Any],
    policy_diagnostics: dict[str, Any],
    trainer: TrueBackwardSkippingTrainer,
) -> None:
    total_steps = cell["total_steps"]
    quota = cell["requested_quota"] or 0
    expected_backward = total_steps - quota
    expected_counts = {
        "batches_seen": total_steps,
        "forward_calls": total_steps,
        "backward_calls": expected_backward,
        "skipped_backward_steps": quota,
        "skipped_batches": quota,
        "optimizer_step_attempts": expected_backward,
        "scheduler_step_calls": expected_backward,
    }
    for key, expected in expected_counts.items():
        actual = instrumentation.get(key)
        if type(actual) is not int or actual != expected:
            raise RuntimeError(
                f"{cell['arm']} execution count {key}={actual!r}, expected {expected}"
            )
    if trainer.state.global_step != total_steps:
        raise RuntimeError(
            f"{cell['arm']} global_step={trainer.state.global_step}, "
            f"expected {total_steps}"
        )
    for key in (
        "invariant_forward_eq_backward_plus_skipped",
        "invariant_opt_le_backward",
        "invariant_sched_le_opportunities",
        "invariant_scheduler_policy_consistent",
        "invariant_sched_le_opt",
    ):
        if instrumentation.get(key) is not True:
            raise RuntimeError(f"{cell['arm']} failed instrumentation invariant {key}")
    if instrumentation.get("skip_update_mode") != "freeze":
        raise RuntimeError(f"{cell['arm']} did not execute freeze-mode skipping")
    if instrumentation.get("scheduler_step_policy") != SCHEDULER_STEP_POLICY:
        raise RuntimeError(f"{cell['arm']} scheduler policy drifted")
    if cell["is_skipping_arm"]:
        policy_counts = {
            "quota_total_steps": total_steps,
            "quota_size": quota,
            "decisions_seen": total_steps,
            "skip_decisions": quota,
        }
        for key, expected in policy_counts.items():
            if policy_diagnostics.get(key) != expected:
                raise RuntimeError(
                    f"{cell['arm']} policy count {key}="
                    f"{policy_diagnostics.get(key)!r}, expected {expected}"
                )


def _results_controller(
    controller: dict[str, Any],
    policy,
    policy_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    realized = copy.deepcopy(controller)
    realized["policy_effective_config"] = (
        dict(policy.effective_config())
        if hasattr(policy, "effective_config")
        else {}
    )
    runtime_total = policy_diagnostics.get("quota_total_steps")
    realized["runtime_quota_total_steps"] = runtime_total
    runtime_quota = policy_diagnostics.get("quota_size")
    if runtime_quota is not None:
        realized["requested_quota"] = runtime_quota
    return realized


def _build_results(
    *,
    cell: dict[str, Any],
    attempt: int,
    instrumentation: dict[str, Any],
    policy_diagnostics: dict[str, Any],
    controller: dict[str, Any],
    online_runtime: dict[str, Any],
    tracker_diagnostics: dict[str, Any],
    train_result,
    eval_metrics: dict[str, Any],
    runtime_seconds: float,
    git_sha: str,
) -> dict[str, Any]:
    run_config = {
        "policy": cell["arm"],
        "control": cell["arm"],
        "target_skip_rate": cell["target_skip_rate"],
        "no_early_stopping": True,
        "allow_early_stopping_with_skipping": False,
        "matched_budget": True,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": SCHEDULER_STEP_POLICY,
        "controller_config": copy.deepcopy(controller),
        "rvd_policy_seed": cell["policy_seed"],
        "online_diagnostics": copy.deepcopy(cell["online_diagnostics"]),
    }
    if "phase_strat_controller" in controller:
        run_config["phase_strat_controller"] = copy.deepcopy(
            controller["phase_strat_controller"]
        )
    if "ler_guided_controller" in controller:
        run_config["ler_guided_controller"] = copy.deepcopy(
            controller["ler_guided_controller"]
        )
    return {
        "task": cell["task"],
        "seed": cell["training_seed"],
        "ablation": cell["arm"],
        "model": cell["model_id"],
        "model_revision": cell.get("model_revision"),
        "profile": "synthetic_cpu",
        "eval_metrics": _json_ready(eval_metrics),
        "train_loss": float(train_result.training_loss),
        "train_runtime_s": float(runtime_seconds),
        "energy_kwh": None,
        "power_avg_watts": None,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": SCHEDULER_STEP_POLICY,
        "true_skip_instrumentation": copy.deepcopy(instrumentation),
        "policy_diagnostics": copy.deepcopy(policy_diagnostics),
        "run_config": run_config,
        "identity_inputs": copy.deepcopy(cell["identity_inputs"]),
        "controller_config": copy.deepcopy(controller),
        "online_diagnostics": copy.deepcopy(online_runtime),
        "ler_final": copy.deepcopy(tracker_diagnostics),
        "fingerprint": cell["fingerprint"],
        "attempt": attempt,
        "forward_calls": instrumentation["forward_calls"],
        "backward_calls": instrumentation["backward_calls"],
        "skipped_backward_steps": instrumentation["skipped_backward_steps"],
        "policy_name": instrumentation.get("policy_name"),
        "compute_saving_mechanism": controller["compute_saving_mechanism"],
        "code_git_sha": git_sha,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "synthetic_protocol": True,
        "provenance_classification": cell.get("provenance_classification", "matched_claim"),
    }


def _finalize_failed_attempt(attempt_dir: Path, exc: BaseException) -> None:
    manifest_path = attempt_dir / "run_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if manifest.get("status") == "running":
            finalize_manifest_failed(str(attempt_dir), exc)
    except (OSError, json.JSONDecodeError, RuntimeError, TypeError, ValueError):
        return


def _execute_cell(
    *,
    cell: dict[str, Any],
    facts: dict[str, Any],
    train_dataset: SyntheticDataset,
    eval_dataset: SyntheticDataset,
    git_sha: str,
    base_output_dir: str,
) -> dict[str, Any]:
    online = cell["online_diagnostics"]
    tracker = build_online_ler_tracker(
        online,
        task_name=cell["task"],
        use_hysteresis=True,
        sample_seed=cell["training_seed"],
    )
    _assert_tracker_contract(cell, tracker)
    policy = _build_policy(cell, tracker)
    controller = _runtime_controller(cell, policy)
    runtime_cell = _runtime_cell(
        planned_cell=cell,
        facts=facts,
        git_sha=git_sha,
        base_output_dir=base_output_dir,
        controller=controller,
    )
    assert_phase1_3_runtime_matches_plan(cell, runtime_cell)

    attempt, attempt_dir = _allocate_attempt(cell["planned_arm_dir"])
    artifact_contract = build_online_ler_artifact_contract(online)
    output_paths = artifact_contract["output_paths"]
    write_manifest_running(
        str(attempt_dir),
        argv=list(sys.argv),
        task=cell["task"],
        model_id=cell["model_id"],
        model_revision=cell.get("model_revision"),
        seed=cell["training_seed"],
        controller_name=type(policy).__name__,
        controller_seed=cell["policy_seed"],
        target_skip_rate=cell["target_skip_rate"],
        planned_quota=cell["requested_quota"],
        total_steps=cell["total_steps"],
        warmup_steps=0,
        skip_update_mode="freeze",
        controller_config_effective=controller,
        matched_budget_planned=True,
        budget_classification="fixed_epoch",
        output_paths=output_paths,
        requested_classification=CLASSIFICATION_MATCHED_CLAIM,
        repo_root=str(REPO_ROOT),
        identity_inputs=cell["identity_inputs"],
        fingerprint=cell["fingerprint"],
        attempt=attempt,
    )

    trainer = None
    model = None
    try:
        _set_training_seed(cell["training_seed"])
        model = TinyClassifier(
            width=SYNTHETIC_WIDTH,
            num_labels=SYNTHETIC_CLASSES,
        )
        training_args = _training_arguments(attempt_dir, cell["training_seed"])
        trainer = AblationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=_collate,
            compute_metrics=_compute_metrics,
            ler_tracker=tracker,
            skip_policy=policy,
            skip_update_mode="freeze",
            apply_momentum=None,
            scheduler_step_policy=SCHEDULER_STEP_POLICY,
            instrumentation_path=str(attempt_dir / "instrumentation.json"),
            capture_logits=bool(online["enabled"]),
            online_ler_mode=online["mode"],
            online_ler_enabled=bool(online["enabled"]),
            online_ler_update_interval=online["update_interval"],
            compute_saving_mechanism=controller["compute_saving_mechanism"],
        )

        started = time.monotonic()
        train_result = trainer.train()
        eval_metrics = trainer.evaluate()
        runtime_seconds = time.monotonic() - started

        instrumentation = trainer.get_instrumentation()
        policy_diagnostics = (
            policy.get_diagnostics() if hasattr(policy, "get_diagnostics") else {}
        )
        _assert_execution_contract(
            cell,
            instrumentation,
            policy_diagnostics,
            trainer,
        )
        tracker_diagnostics = tracker.get_diagnostics() if tracker is not None else {}
        online_runtime = build_online_ler_runtime_metadata(
            online,
            instrumentation,
            tracker_diagnostics=tracker_diagnostics,
        )
        realized_controller = _results_controller(
            controller,
            policy,
            policy_diagnostics,
        )
        results = _build_results(
            cell=cell,
            attempt=attempt,
            instrumentation=instrumentation,
            policy_diagnostics=policy_diagnostics,
            controller=realized_controller,
            online_runtime=online_runtime,
            tracker_diagnostics=tracker_diagnostics,
            train_result=train_result,
            eval_metrics=eval_metrics,
            runtime_seconds=runtime_seconds,
            git_sha=git_sha,
        )

        _atomic_write_json(attempt_dir / "instrumentation.json", instrumentation)
        if online["enabled"]:
            _atomic_write_json(
                attempt_dir / "ler_diagnostics.json",
                {
                    "online_diagnostics": online_runtime,
                    "tracker_diagnostics": tracker_diagnostics,
                    "policy_diagnostics": policy_diagnostics,
                },
            )
        results_path = attempt_dir / "results.json"
        _atomic_write_json(results_path, results)

        report = validate_results(
            results_path,
            required_artifacts=artifact_contract["required_artifacts"],
        )
        if not report.ok or not report.valid_for_matched_budget:
            details = "; ".join(
                f"{finding.field}: {finding.message}"
                for finding in report.findings
            )
            raise RuntimeError(
                f"Piece 5 rejected {cell['arm']} at rate "
                f"{cell['target_skip_rate']}: {details}"
            )

        artifact_filenames = ["results.json", *artifact_contract["required_artifacts"]]
        finalize_manifest_completed(
            str(attempt_dir),
            realized_skips=instrumentation["skipped_backward_steps"],
            realized_skip_rate=(
                instrumentation["skipped_backward_steps"] / cell["total_steps"]
            ),
            validation_status=report.to_dict(),
            artifact_filenames=artifact_filenames,
        )
        verification = verify_completed_manifest(str(attempt_dir))
        if not verification["ok"]:
            raise RuntimeError(
                f"completed manifest verification failed for {cell['arm']}: "
                f"{verification['errors']}"
            )
        return {
            "cell_id": [
                cell["task"],
                cell["training_seed"],
                cell["target_skip_rate"],
                cell["arm"],
            ],
            "attempt": attempt,
            "attempt_dir": str(attempt_dir),
            "backward_calls": instrumentation["backward_calls"],
            "skipped_backward_steps": instrumentation["skipped_backward_steps"],
            "valid_for_matched_budget": report.valid_for_matched_budget,
        }
    except Exception as exc:
        _finalize_failed_attempt(attempt_dir, exc)
        raise
    finally:
        del trainer
        del model
        gc.collect()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the offline 12-cell Phase 1.3 synthetic CPU protocol"
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Fresh directory for matrix and per-attempt artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_TRAINING_SEED,
        help="Single paired synthetic training and policy seed",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if torch.cuda.is_available():
        raise RuntimeError("synthetic Phase 1.3 execution must remain CPU-only")
    torch.set_num_threads(1)

    git_sha = _resolve_git_sha()
    _require_claim_ready_checkout(git_sha)
    base_output_dir = Path(args.output_dir).expanduser().resolve()

    train_dataset = SyntheticDataset(
        size=SYNTHETIC_TRAIN_SIZE,
        width=SYNTHETIC_WIDTH,
        seed=SYNTHETIC_DATA_SEED,
    )
    eval_dataset = SyntheticDataset(
        size=SYNTHETIC_EVAL_SIZE,
        width=SYNTHETIC_WIDTH,
        seed=SYNTHETIC_DATA_SEED + 1,
    )
    facts = _data_facts(train_dataset, eval_dataset)
    if facts["total_steps"] <= POLICY_MIN_STEP:
        raise RuntimeError("synthetic horizon must exceed POLICY_MIN_STEP")

    plan = _plan_matrix(
        base_output_dir=str(base_output_dir),
        training_seed=int(args.seed),
        facts=facts,
        git_sha=git_sha,
    )
    _require_fresh_output(base_output_dir)

    base_output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(base_output_dir / "matrix_plan.json", plan)
    _atomic_write_json(
        base_output_dir / "matrix_validation.json",
        {
            "status": "planned",
            "plan_valid": True,
            "n_cells": len(plan),
            "git_sha": git_sha,
        },
    )

    completed_cells: list[dict[str, Any]] = []
    try:
        for index, cell in enumerate(plan, start=1):
            print(
                f"[{index:02d}/{len(plan):02d}] {cell['arm']} "
                f"rate={cell['target_skip_rate']:.2f} "
                f"seed={cell['training_seed']}"
            )
            completed_cells.append(
                _execute_cell(
                    cell=cell,
                    facts=facts,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    git_sha=git_sha,
                    base_output_dir=str(base_output_dir),
                )
            )
    except Exception as exc:
        _atomic_write_json(
            base_output_dir / "matrix_validation.json",
            {
                "status": "failed",
                "plan_valid": True,
                "n_cells": len(plan),
                "n_completed_cells": len(completed_cells),
                "error_type": type(exc).__name__,
                "git_sha": git_sha,
            },
        )
        raise

    completed = validate_phase1_3_completed_matrix(
        plan,
        tasks=[SYNTHETIC_TASK],
        seeds=[int(args.seed)],
        target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
        minimum_seed_count=1,
        base_output_dir=str(base_output_dir),
    )
    final_report = {
        "status": "completed",
        "plan_valid": True,
        "completed_matrix_valid": True,
        "n_cells": len(plan),
        "n_valid_runs": len(completed["valid_runs"]),
        "git_sha": git_sha,
        "task": SYNTHETIC_TASK,
        "seed": int(args.seed),
        "target_skip_rates": list(STRICT_TARGET_SKIP_RATES),
        "arms": list(PHASE1_3_CANONICAL_ARMS),
        "runs": completed_cells,
    }
    _atomic_write_json(base_output_dir / "matrix_validation.json", final_report)
    print(json.dumps(final_report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
