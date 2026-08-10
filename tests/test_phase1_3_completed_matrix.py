"""Dependency-light tests for the Phase 1.3 completed-matrix validator (6D-3)."""

import copy
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Load the module under test ─────────────────────────────────────────────
_COMPLETED_SPEC = importlib.util.spec_from_file_location(
    "phase1_3_completed_matrix",
    REPO_ROOT / "lerna" / "utils" / "phase1_3_completed_matrix.py",
)
cm = importlib.util.module_from_spec(_COMPLETED_SPEC)
_COMPLETED_SPEC.loader.exec_module(cm)

validate_phase1_3_completed_matrix = cm.validate_phase1_3_completed_matrix
CompletedMatrixError = cm.CompletedMatrixError
PHASE1_3_CANONICAL_ARMS = cm.PHASE1_3_CANONICAL_ARMS

# ── Load the matrix-plan module for building fixtures ──────────────────────
_MATRIX_SPEC = importlib.util.spec_from_file_location(
    "phase1_3_matrix",
    REPO_ROOT / "lerna" / "utils" / "phase1_3_matrix.py",
)
matrix = importlib.util.module_from_spec(_MATRIX_SPEC)
_MATRIX_SPEC.loader.exec_module(matrix)

validate_phase1_3_matrix_plan = matrix.validate_phase1_3_matrix_plan
build_scientific_fingerprint = matrix.build_scientific_fingerprint
MatrixPlanError = matrix.MatrixPlanError
POLICY_MIN_STEP = matrix.POLICY_MIN_STEP
STRICT_TARGET_SKIP_RATES = matrix.STRICT_TARGET_SKIP_RATES

# ── Load run_provenance for creating manifest fixtures ─────────────────────
_PROV_SPEC = importlib.util.spec_from_file_location(
    "run_provenance",
    REPO_ROOT / "lerna" / "utils" / "run_provenance.py",
)
provenance = importlib.util.module_from_spec(_PROV_SPEC)
_PROV_SPEC.loader.exec_module(provenance)
write_manifest_running = provenance.write_manifest_running
finalize_manifest_completed = provenance.finalize_manifest_completed
verify_completed_manifest = provenance.verify_completed_manifest
FAKE_GIT_CLEAN = lambda: {
    "commit_sha": "abc123",
    "dirty": False,
    "tracked_changes": [],
    "untracked_paths": [".kilo/settings.json"],
}
FAKE_VERSIONS = lambda: {
    "python": "3.11.0",
    "torch": "2.4.0",
    "transformers": "4.44.0",
    "cuda": "12.1",
    "device": "MockGPU",
}

# ── Load the Piece 5 validator for results.json fixtures ───────────────────
_VAL_SPEC = importlib.util.spec_from_file_location(
    "validate_skip_policy_results",
    REPO_ROOT / "scripts" / "validate_skip_policy_results.py",
)
vspr = importlib.util.module_from_spec(_VAL_SPEC)
sys.modules[_VAL_SPEC.name] = vspr
_VAL_SPEC.loader.exec_module(vspr)

# ── Constants ──────────────────────────────────────────────────────────────
TASK = "synthetic_task"
SEED = 7
RATES = [0.30, 0.40]
MODEL_ID = "synthetic-model"
EPOCHS = 3
TOTAL_STEPS = 200
GIT_SHA = "0123abcd4567ef890123abcd4567ef8901234567"
FINGERPRINT_RE = re.compile(r"[0-9a-f]{16}\Z")

POLICY_CLASSES = {
    "full_finetune": "AlwaysFalsePolicy",
    "exact_random": "RandomSkipPolicy",
    "fixed_phase_strat": "FixedPhaseStratifiedRandomPolicy",
    "phase_strat_guarded": "PhaseStratifiedGuardedRandomPolicy",
    "ler_guided_stratified": "LERGuidedStratifiedPolicy",
    "ler_guided_stratified_safe": "LERGuidedStratifiedSafetyPolicy",
}
OFFLINE_ARMS = ("full_finetune", "exact_random", "fixed_phase_strat")
PHASE_ARMS = ("fixed_phase_strat", "phase_strat_guarded")
LER_ARMS = ("ler_guided_stratified", "ler_guided_stratified_safe")

FULL, RANDOM, FIXED, GUARDED, LER, LER_SAFE = range(6)

BOUND_MANIFEST_RUN_FIELDS = (
    "task",
    "seed",
    "target_skip_rate",
    "controller_name",
    "controller_seed",
    "model_id",
    "planned_quota",
    "total_steps",
    "skip_update_mode",
    "matched_budget_planned",
)

# ── Import guard ───────────────────────────────────────────────────────────
_IMPORT_GUARD_SCRIPT = """
import importlib.util
import sys

BANNED = (
    "torch",
    "transformers",
    "datasets",
    "evaluate",
    "scripts.run_ablation_study",
)


class _BannedImportGuard:
    def find_spec(self, fullname, path=None, target=None):
        for banned in BANNED:
            if fullname == banned or fullname.startswith(banned + "."):
                raise ImportError("banned scientific import: " + fullname)
        return None


sys.meta_path.insert(0, _BannedImportGuard())
spec = importlib.util.spec_from_file_location(
    "phase1_3_completed_matrix_guarded",
    sys.argv[1],
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.validate_phase1_3_completed_matrix)
print("GUARDED_LOAD_OK")
"""


# ── Fixture helpers ────────────────────────────────────────────────────────

def _online_diag(arm, training_seed):
    if arm in OFFLINE_ARMS:
        return {
            "requested_mode": "auto",
            "mode": "off",
            "enabled": False,
            "timing": "none",
            "parameter_sample_size": 0,
            "update_interval": 0,
            "reason": "offline diagnostics arm",
            "sample_seed": None,
        }
    return {
        "requested_mode": "auto",
        "mode": "sampled_lagged",
        "enabled": True,
        "timing": "post_decision_after_backward",
        "parameter_sample_size": 64,
        "update_interval": 10,
        "reason": "online diagnostics arm",
        "sample_seed": training_seed,
    }


def _phase_controller(arm, rate, quota, total_steps, policy_seed=SEED):
    mid = (POLICY_MIN_STEP + total_steps) // 2
    config = {
        "control": arm,
        "controller_class": POLICY_CLASSES[arm],
        "policy_name": arm,
        "target_skip_rate": rate,
        "total_steps": total_steps,
        "min_step": POLICY_MIN_STEP,
        "policy_seed": policy_seed,
        "n_phases": 2,
        "phase_weights": [0.5, 0.5],
        "phase_bounds": [POLICY_MIN_STEP, mid, total_steps],
        "phase_eligible": [mid - POLICY_MIN_STEP, total_steps - mid],
        "phase_quota": [quota - quota // 2, quota // 2],
        "requested_quota": quota,
    }
    if arm == "phase_strat_guarded":
        config.update(
            {
                "max_consecutive_skips": 3,
                "risk_gamma": 1.5,
                "guarded_safety": {
                    "use_rho_vg": True,
                    "rho_veto_threshold": 0.5,
                    "use_safety_horizon": True,
                    "spike_factor": 2.0,
                },
            }
        )
    return config


def _ler_controller(arm, rate, total_steps, policy_seed=SEED):
    config = {
        "control": arm,
        "policy_class": POLICY_CLASSES[arm],
        "policy_name": arm,
        "target_skip_rate": rate,
        "total_steps": total_steps,
        "min_step": POLICY_MIN_STEP,
        "policy_seed": policy_seed,
        "n_phases": 2,
        "phase_weights": [0.5, 0.5],
        "max_consecutive_skips": 3,
        "probe_interval": 5,
        "min_ler_observations": 4,
        "ler_guidance_strength": 1.0,
        "required_tracker_mode": "sampled_lagged",
        "required_tracker_timing": "post_decision_after_backward",
        "safety_enabled": arm == "ler_guided_stratified_safe",
    }
    if arm == "ler_guided_stratified_safe":
        config.update(
            {
                "use_rho_vg_safety": True,
                "rho_veto_threshold": 0.5,
                "use_loss_spike_safety": True,
                "loss_spike_factor": 2.0,
                "loss_spike_window": 5,
            }
        )
    return config


def _build_cell(
    arm,
    rate,
    base_output_dir="fixture",
    total_steps=TOTAL_STEPS,
    task=TASK,
    seed=SEED,
):
    skipping = arm != "full_finetune"
    quota = round(rate * total_steps) if skipping else None
    online = _online_diag(arm, seed)
    controller = {
        "arm": arm,
        "arm_alias_of": None,
        "control": arm,
        "policy_class": POLICY_CLASSES[arm],
        "compute_saving_mechanism": "backward_skipping" if skipping else "none",
        "policy_seed": seed,
        "target_skip_rate": rate,
        "min_step": POLICY_MIN_STEP,
        "configured_total_steps": total_steps,
        "requested_quota": quota,
        "matched_budget": True,
        "is_skipping_arm": skipping,
        "allow_early_stopping_with_skipping": False,
        "early_stopping_active": False,
        "num_epochs": EPOCHS,
        "online_diagnostics": copy.deepcopy(online),
    }
    identity = {
        "task": task,
        "training_seed": seed,
        "model_id": MODEL_ID,
        "max_samples_requested": None,
        "train_samples_realized": 1000,
        "eval_samples_realized": 200,
        "train_dataset_fingerprint": "synthetic-train-data",
        "eval_dataset_fingerprint": "synthetic-eval-data",
        "num_epochs": EPOCHS,
        "control": arm,
        "target_skip_rate": rate,
        "policy_seed": seed,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "no_early_stopping": True,
        "total_steps": total_steps,
        "git_sha": GIT_SHA,
        "online_diagnostics": copy.deepcopy(online),
    }
    if arm in PHASE_ARMS:
        phase = _phase_controller(arm, rate, quota, total_steps, seed)
        controller["phase_strat_controller"] = copy.deepcopy(phase)
        identity["phase_strat_controller"] = copy.deepcopy(phase)
    if arm in LER_ARMS:
        ler = _ler_controller(arm, rate, total_steps, seed)
        controller["ler_guided_controller"] = copy.deepcopy(ler)
        identity["ler_guided_controller"] = copy.deepcopy(ler)
    fingerprint = build_scientific_fingerprint(identity)
    return {
        "arm": arm,
        "control": arm,
        "task": task,
        "training_seed": seed,
        "policy_seed": seed,
        "model_id": MODEL_ID,
        "target_skip_rate": rate,
        "num_epochs": EPOCHS,
        "total_steps": total_steps,
        "min_step": POLICY_MIN_STEP,
        "requested_quota": quota,
        "planned_skips": quota if skipping else 0,
        "is_skipping_arm": skipping,
        "matched_budget": True,
        "no_early_stopping": True,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "online_diagnostics": online,
        "controller_config": controller,
        "identity_inputs": identity,
        "fingerprint": fingerprint,
        "planned_arm_dir": os.path.join(base_output_dir, arm, fingerprint),
    }


def _build_plan(rates=None, base_output_dir="fixture", tasks=None, seeds=None):
    rates = RATES if rates is None else rates
    tasks = [TASK] if tasks is None else tasks
    seeds = [SEED] if seeds is None else seeds
    return [
        _build_cell(arm, rate, base_output_dir, task=task, seed=seed)
        for task in tasks
        for seed in seeds
        for rate in rates
        for arm in PHASE1_3_CANONICAL_ARMS
    ]


def _results_data(cell, attempt_num=1, *, tamper=None):
    """Build a results.json payload that passes Piece 5 validation."""
    arm = cell["arm"]
    rate = cell["target_skip_rate"]
    total_steps = cell["total_steps"]
    quota = round(rate * total_steps)
    skipping = arm != "full_finetune"
    online = cell["online_diagnostics"]
    is_online = arm not in OFFLINE_ARMS
    training_seed = cell["training_seed"]

    backward = total_steps - quota if skipping else total_steps
    scheduler_calls = backward

    instr = {
        "forward_calls": total_steps,
        "backward_calls": backward,
        "optimizer_step_attempts": backward,
        "scheduler_step_calls": scheduler_calls,
        "skipped_backward_steps": quota if skipping else 0,
        "batches_seen": total_steps,
        "skipped_batches": quota if skipping else 0,
        "skip_ratio_by_batch": (quota / total_steps) if skipping else 0.0,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "scheduler_step_opportunities": total_steps,
        "invariant_forward_eq_backward_plus_skipped": True,
        "invariant_opt_le_backward": True,
        "invariant_sched_le_opportunities": True,
        "invariant_scheduler_policy_consistent": True,
        "invariant_sched_le_opt": True,
    }

    if skipping:
        diag = {
            "policy_name": arm,
            "target_skip_rate": rate,
            "quota_total_steps": total_steps,
            "quota_size": quota,
            "decisions_seen": total_steps,
            "skip_decisions": quota,
            "realized_skip_rate": rate,
            "requested_quota": quota,
            "seed": training_seed if arm == "exact_random" else None,
        }
        if arm in ("fixed_phase_strat", "phase_strat_guarded",
                   "ler_guided_stratified", "ler_guided_stratified_safe"):
            diag["quota_exact"] = True
    else:
        diag = {}

    # Strip None seeds for non-exact_random arms
    if arm != "exact_random" and "seed" in diag:
        del diag["seed"]

    cc = copy.deepcopy(cell["controller_config"])
    cc["policy_effective_config"] = {
        "policy_class": POLICY_CLASSES[arm],
        "configured_total_steps": total_steps,
    }
    cc["runtime_quota_total_steps"] = total_steps if skipping else None
    run_config = {
        "policy": arm,
        "target_skip_rate": rate,
        "no_early_stopping": True,
        "allow_early_stopping_with_skipping": False,
        "matched_budget": True,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "control": arm,
        "controller_config": copy.deepcopy(cc),
        "rvd_policy_seed": training_seed,
        "online_diagnostics": copy.deepcopy(online),
    }
    if arm in PHASE_ARMS:
        run_config["phase_strat_controller"] = copy.deepcopy(
            cc["phase_strat_controller"]
        )
    if arm in LER_ARMS:
        run_config["ler_guided_controller"] = copy.deepcopy(
            cc["ler_guided_controller"]
        )

    online_rt = {
        **copy.deepcopy(online),
        "parameter_sample_size_realized": 0,
        "update_attempts": 0,
        "update_successes": 0,
        "n_updates": 0,
        "n_decisions": 0,
        "last_update_decision": None,
        "observation_age_decisions": None,
    }
    if is_online:
        online_rt["parameter_sample_size_realized"] = 64
        online_rt["update_attempts"] = total_steps
        online_rt["update_successes"] = total_steps
        online_rt["n_updates"] = total_steps
        online_rt["n_decisions"] = total_steps
        online_rt["last_update_decision"] = total_steps - 1
        online_rt["observation_age_decisions"] = 1

    identity = cell["identity_inputs"]
    fingerprint = cell["fingerprint"]

    data = {
        "task": cell["task"],
        "seed": cell["training_seed"],
        "eval_metrics": {"eval_accuracy": 0.9},
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "true_skip_instrumentation": instr,
        "policy_diagnostics": diag,
        "run_config": run_config,
        "ablation": arm,
        "identity_inputs": copy.deepcopy(identity),
        "controller_config": copy.deepcopy(cc),
        "online_diagnostics": online_rt,
        "fingerprint": fingerprint,
        "attempt": attempt_num,
        "forward_calls": total_steps,
        "backward_calls": backward,
        "skipped_backward_steps": quota if skipping else 0,
    }

    if tamper:
        tamper(data)

    return data


def _manifest_output_paths(cell):
    """Expected output_paths for a cell."""
    online = cell.get("online_diagnostics") or {}
    paths = {
        "results": "results.json",
        "instrumentation": "instrumentation.json",
        "manifest": "run_manifest.json",
    }
    if isinstance(online, dict) and online.get("enabled"):
        paths["ler_diagnostics"] = "ler_diagnostics.json"
    return paths


def _create_cell_fixture(
    base_dir,
    cell,
    attempt_num=1,
    *,
    status="completed",
    provenance_classification="matched_claim",
    tamper_manifest=None,
    tamper_results=None,
    extra_output_paths=None,
    missing_artifact=None,
):
    """Create a physical attempt directory with valid manifest and artifacts."""
    arm = cell["arm"]
    fingerprint = cell["fingerprint"]
    cell_dir = os.path.join(base_dir, arm, fingerprint)
    attempt_dir = os.path.join(cell_dir, f"attempt-{attempt_num:03d}")
    os.makedirs(attempt_dir, exist_ok=True)

    skipping = arm != "full_finetune"
    rate = cell["target_skip_rate"]
    total_steps = cell["total_steps"]
    quota = round(rate * total_steps)
    is_online = arm not in OFFLINE_ARMS

    # Build the results payload first so the running manifest can reference
    # the (possibly tampered) fingerprint and identity while still passing
    # completion integrity verification.
    results = _results_data(cell, attempt_num, tamper=tamper_results)

    output_paths = _manifest_output_paths(cell)
    if extra_output_paths:
        output_paths.update(extra_output_paths)

    identity = cell["identity_inputs"]
    fingerprint_val = cell["fingerprint"]
    if isinstance(results, dict):
        fingerprint_val = results.get("fingerprint", fingerprint_val)
        if isinstance(results.get("identity_inputs"), dict):
            identity = results["identity_inputs"]
    controller_cc = copy.deepcopy(cell["controller_config"])
    controller_cc["policy_effective_config"] = {
        "policy_class": POLICY_CLASSES[arm],
        "configured_total_steps": total_steps,
    }

    # Write the running manifest first: write_manifest_running refuses to
    # start in a directory that already contains canonical artifacts.
    write_manifest_running(
        attempt_dir,
        argv=["run_ablation_study.py"],
        task=cell["task"],
        model_id=MODEL_ID,
        seed=cell["training_seed"],
        controller_name=POLICY_CLASSES[arm],
        controller_seed=cell["policy_seed"],
        target_skip_rate=rate,
        planned_quota=quota if skipping else None,
        total_steps=total_steps,
        warmup_steps=10,
        skip_update_mode="freeze",
        controller_config_effective=controller_cc,
        matched_budget_planned=True,
        budget_classification="fixed_epoch",
        output_paths=output_paths,
        git_provider=FAKE_GIT_CLEAN,
        version_provider=FAKE_VERSIONS,
        identity_inputs=identity,
        fingerprint=fingerprint_val,
        attempt=attempt_num,
    )

    # Write results.json
    results_path = os.path.join(attempt_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f)

    # Write instrumentation.json
    instr_path = os.path.join(attempt_dir, "instrumentation.json")
    with open(instr_path, "w", encoding="utf-8") as f:
        json.dump({"batches": total_steps}, f)

    # Write ler_diagnostics.json for online cells
    if is_online:
        ler_path = os.path.join(attempt_dir, "ler_diagnostics.json")
        with open(ler_path, "w", encoding="utf-8") as f:
            json.dump({"ler_estimate": 0.05}, f)

    if status == "running":
        return  # Don't finalize

    validation_status = {
        "valid_for_matched_budget": True,
        "ok": True,
        "protocol_complete": True,
        "matched_budget_claimed": True,
        "n_errors": 0,
        "findings": [],
    }

    artifact_filenames = ["results.json", "instrumentation.json"]
    if is_online:
        artifact_filenames.append("ler_diagnostics.json")

    if status == "completed":
        manifest = finalize_manifest_completed(
            attempt_dir,
            realized_skips=quota if skipping else 0,
            realized_skip_rate=rate if skipping else 0.0,
            validation_status=validation_status,
            artifact_filenames=artifact_filenames,
        )
    elif status == "failed":
        manifest = provenance.finalize_manifest_failed(
            attempt_dir, RuntimeError("simulated failure")
        )

    if tamper_manifest:
        manifest_path = os.path.join(attempt_dir, "run_manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        tamper_manifest(manifest)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)


def _create_full_fixture(base_dir, plan, *, attempt_count=1, **kwargs):
    """Create a complete filesystem fixture for a plan."""
    for cell in plan:
        for attempt_num in range(1, attempt_count + 1):
            _create_cell_fixture(base_dir, cell, attempt_num, **kwargs)


def _findings_from(plan, **overrides):
    """Call validate_phase1_3_completed_matrix and return findings."""
    kwargs = {
        "tasks": [TASK],
        "seeds": [SEED],
        "target_skip_rates": list(RATES),
        "minimum_seed_count": 1,
        "base_output_dir": "fixture",
    }
    kwargs.update(overrides)
    try:
        result = validate_phase1_3_completed_matrix(plan, **kwargs)
        return result.get("findings", [])
    except CompletedMatrixError as exc:
        return exc.findings


def _error_fields(findings):
    """Return set of field names from error-severity findings."""
    return {f["field"] for f in findings if f.get("severity") == "error"}


def _error_messages(findings, field):
    return [f["message"] for f in findings if f.get("field") == field]


def _has_field_prefix(findings, prefix):
    return any(
        f.get("severity") == "error" and f.get("field", "").startswith(prefix)
        for f in findings
    )


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _attempt_dir(base, cell, attempt_num=1):
    return os.path.join(
        base,
        cell["arm"],
        cell["fingerprint"],
        f"attempt-{attempt_num:03d}",
    )


def _manifest_path(base, cell, attempt_num=1):
    return os.path.join(_attempt_dir(base, cell, attempt_num), "run_manifest.json")


def _results_path(base, cell, attempt_num=1):
    return os.path.join(_attempt_dir(base, cell, attempt_num), "results.json")


# ── Test class ─────────────────────────────────────────────────────────────

class Phase13CompletedMatrixValidatorTests(unittest.TestCase):
    """6D-3: completed-matrix validator tests."""

    # ── 0. Import guard ────────────────────────────────────────────────────

    def test_import_guard_bans_scientific_dependencies(self):
        completed_path = REPO_ROOT / "lerna" / "utils" / "phase1_3_completed_matrix.py"
        proc = subprocess.run(
            [sys.executable, "-c", _IMPORT_GUARD_SCRIPT, str(completed_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("GUARDED_LOAD_OK", proc.stdout)

    # ── 1. Valid 12-cell end-to-end ────────────────────────────────────────

    def test_valid_12_cell_matrix_returns_runs_in_plan_order(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            result = validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir=base,
            )

            self.assertEqual(len(result["valid_runs"]), 12)
            self.assertEqual(result["findings"], [])

            # Verify runs are in exact plan order
            plan_order = [
                (cell["task"], cell["training_seed"],
                 cell["target_skip_rate"], cell["arm"])
                for cell in plan
            ]
            run_order = [
                (run["cell"]["task"], run["cell"]["training_seed"],
                 run["cell"]["target_skip_rate"], run["cell"]["arm"])
                for run in result["valid_runs"]
            ]
            self.assertEqual(run_order, plan_order)

            # Each run has attempt-001
            for run in result["valid_runs"]:
                self.assertEqual(run["attempt"], "attempt-001")
                self.assertEqual(run["attempt_num"], 1)

    # ── 2. Validation performs no writes ───────────────────────────────────

    def test_validation_performs_no_writes(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            before = set()
            for root, dirs, files in os.walk(tmp):
                for name in files:
                    before.add(os.path.relpath(os.path.join(root, name), tmp))

            validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir=base,
            )

            after = set()
            for root, dirs, files in os.walk(tmp):
                for name in files:
                    after.add(os.path.relpath(os.path.join(root, name), tmp))

            self.assertEqual(after, before)

    # ── 3. CompletedMatrixError structure ──────────────────────────────────

    def test_error_carries_structured_findings(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Don't create any directories - all cells will be missing
            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIsInstance(exc, ValueError)
                self.assertIsInstance(exc.findings, list)
                self.assertGreaterEqual(len(exc.findings), 12)
                for finding in exc.findings:
                    self.assertEqual(
                        set(finding), {"severity", "field", "cell", "message"}
                    )
                    self.assertEqual(finding["severity"], "error")

    # ── 4. Failed historical attempts allowed ──────────────────────────────

    def test_failed_historical_attempt_accepted_with_valid_completion(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Create attempt-001 (failed) and attempt-002 (completed)
            for cell in plan:
                _create_cell_fixture(base, cell, 1, status="failed")
                _create_cell_fixture(base, cell, 2, status="completed")

            result = validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir=base,
            )

            self.assertEqual(len(result["valid_runs"]), 12)
            self.assertEqual(result["findings"], [])
            for run in result["valid_runs"]:
                self.assertEqual(run["attempt"], "attempt-002")

    # ── 5. Running attempts rejected ───────────────────────────────────────

    def test_running_attempt_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                _create_cell_fixture(base, cell, 1, status="running")

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("manifest.status",
                              _error_fields(exc.findings))

    # ── 6. Missing attempts ────────────────────────────────────────────────

    def test_missing_attempts_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Create cell directories but no attempt directories
            for cell in plan:
                arm = cell["arm"]
                fingerprint = cell["fingerprint"]
                os.makedirs(os.path.join(base, arm, fingerprint), exist_ok=True)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("attempts", _error_fields(exc.findings))
                self.assertIn("cell_completion", _error_fields(exc.findings))

    # ── 7. Duplicate valid completions rejected ────────────────────────────

    def test_duplicate_valid_completions_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                _create_cell_fixture(base, cell, 1, status="completed")
                _create_cell_fixture(base, cell, 2, status="completed")

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("attempts", _error_fields(exc.findings))
                msgs = _error_messages(exc.findings, "attempts")
                self.assertTrue(
                    any("duplicate valid completed attempt" in m for m in msgs)
                )

    # ── 8. Noncanonical attempts rejected ──────────────────────────────────

    def test_noncanonical_attempt_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            arm = plan[FULL]["arm"]
            fingerprint = plan[FULL]["fingerprint"]
            # Create a noncanonical attempt dir
            os.makedirs(os.path.join(base, arm, fingerprint, "attempt-abc"), exist_ok=True)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("filesystem", _error_fields(exc.findings))

    # ── 9. Missing cell directories ────────────────────────────────────────

    def test_missing_cell_directory_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Don't create any directories
            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("cell_directory", _error_fields(exc.findings))

    # ── 10. Unexpected cell directories ────────────────────────────────────

    def test_unexpected_cell_directory_detected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            # Add an unexpected directory
            os.makedirs(os.path.join(base, "bogus_arm", "deadbeef12345678",
                                     "attempt-001"), exist_ok=True)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                fs_errors = [f for f in exc.findings
                             if f.get("field") == "filesystem"]
                self.assertTrue(
                    any("unexpected cell directory" in f["message"]
                        for f in fs_errors)
                )

    # ── 11. Invalid plan raises CompletedMatrixError ───────────────────────

    def test_invalid_plan_raises_aggregated_error(self):
        plan = _build_plan()
        plan[FULL]["skip_update_mode"] = "unfreeze"
        try:
            validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir="fixture",
            )
            self.fail("expected CompletedMatrixError")
        except CompletedMatrixError as exc:
            self.assertIn("skip_update_mode", _error_fields(exc.findings))

    # ── 12. Invalid base_output_dir ────────────────────────────────────────

    def test_invalid_base_output_dir_raises_error(self):
        plan = _build_plan()
        for bad_dir in (None, "", 123, [], {}):
            with self.subTest(bad_dir=bad_dir):
                try:
                    validate_phase1_3_completed_matrix(
                        plan,
                        tasks=[TASK],
                        seeds=[SEED],
                        target_skip_rates=list(RATES),
                        minimum_seed_count=1,
                        base_output_dir=bad_dir,
                    )
                    self.fail("expected CompletedMatrixError")
                except CompletedMatrixError as exc:
                    self.assertIn("base_output_dir", _error_fields(exc.findings))

    # ── 13. Malformed attempt paths ────────────────────────────────────────

    def test_malformed_attempt_directory_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Create a valid cell for one arm, but with a malformed attempt number
            cell = plan[FULL]
            arm = cell["arm"]
            fingerprint = cell["fingerprint"]
            os.makedirs(os.path.join(base, arm, fingerprint, "attempt-000"), exist_ok=True)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("attempts", _error_fields(exc.findings))

    # ── 14. matched_claim provenance required ──────────────────────────────

    def test_non_matched_claim_provenance_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["provenance_classification"] = "local_development"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.provenance_classification",
                    _error_fields(exc.findings),
                )

    def test_malformed_manifest_and_results_objects_are_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            malformed_manifest_values = (
                (plan[0], "run"),
                (plan[1], "controller_config_effective"),
                (plan[2], "output_paths"),
            )
            for cell, key in malformed_manifest_values:
                manifest_path = _manifest_path(base, cell)
                manifest = _read_json(manifest_path)
                manifest[key] = []
                _write_json(manifest_path, manifest)

            _write_json(_results_path(base, plan[3]), [])

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            self.assertIn("manifest.run", fields)
            self.assertIn("manifest.controller_config_effective", fields)
            self.assertIn("manifest.output_paths", fields)
            self.assertIn("results.json", fields)

    # ── 15. Online cells require ler_diagnostics.json ──────────────────────

    def test_online_cell_requires_ler_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Create all cells, but remove ler_diagnostics.json from LER cell
            _create_full_fixture(base, plan)
            ler_cell = plan[LER]
            ler_dir = os.path.join(base, ler_cell["arm"], ler_cell["fingerprint"],
                                   "attempt-001")
            ler_path = os.path.join(ler_dir, "ler_diagnostics.json")
            if os.path.exists(ler_path):
                os.unlink(ler_path)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                fields = _error_fields(exc.findings)
                self.assertTrue(
                    _has_field_prefix(exc.findings, "results.json.validation")
                    or "manifest.verification.artifacts.ler_diagnostics.json.exists"
                    in fields,
                    msg=f"ler-diagnostics finding missing from {fields}",
                )

    # ── 16. Offline cells do not require ler_diagnostics.json ──────────────

    def test_offline_cell_no_ler_diagnostics_ok(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)
            # Remove ler_diagnostics.json from offline cells (they shouldn't have it)
            for cell in plan:
                if cell["arm"] in OFFLINE_ARMS:
                    cell_dir = os.path.join(base, cell["arm"], cell["fingerprint"],
                                            "attempt-001")
                    ler_path = os.path.join(cell_dir, "ler_diagnostics.json")
                    if os.path.exists(ler_path):
                        os.unlink(ler_path)

            result = validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir=base,
            )
            self.assertEqual(len(result["valid_runs"]), 12)

    # ── 17. Matched-budget tampering is rejected by Piece 5 ────────────────

    def test_matched_budget_tampering_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(data):
                    data["run_config"]["matched_budget"] = False
                _create_cell_fixture(base, cell, 1, tamper_results=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertTrue(
                    _has_field_prefix(exc.findings, "results.json.validation"),
                    msg=f"Piece 5 validation findings missing from "
                        f"{_error_fields(exc.findings)}",
                )

    # ── 18. Manifest identity drift ────────────────────────────────────────

    def test_manifest_identity_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["identity_inputs"] = copy.deepcopy(m["identity_inputs"])
                    m["identity_inputs"]["task"] = "wrong_task"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.identity_inputs",
                    _error_fields(exc.findings),
                )

    # ── 19. Manifest fingerprint drift ─────────────────────────────────────

    def test_manifest_fingerprint_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["fingerprint"] = "deadbeef12345678"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.fingerprint",
                    _error_fields(exc.findings),
                )

    # ── 20. Controller drift ───────────────────────────────────────────────

    def test_controller_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    if "controller_config_effective" in m:
                        m["controller_config_effective"]["policy_seed"] = 9999
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.controller_config_effective.policy_seed",
                    _error_fields(exc.findings),
                )

    # ── 21. Run field drift ────────────────────────────────────────────────

    def test_every_bound_manifest_run_field_requires_presence(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            manifest_path = _manifest_path(base, plan[FULL])
            manifest = _read_json(manifest_path)
            for key in BOUND_MANIFEST_RUN_FIELDS:
                manifest["run"].pop(key)
            _write_json(manifest_path, manifest)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            for key in BOUND_MANIFEST_RUN_FIELDS:
                with self.subTest(key=key):
                    self.assertIn(f"manifest.run.{key}", fields)

    def test_every_bound_manifest_run_field_rejects_value_drift(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            manifest_path = _manifest_path(base, plan[RANDOM])
            manifest = _read_json(manifest_path)
            for key in BOUND_MANIFEST_RUN_FIELDS:
                manifest["run"][key] = {"drift": key}
            _write_json(manifest_path, manifest)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            for key in BOUND_MANIFEST_RUN_FIELDS:
                with self.subTest(key=key):
                    self.assertIn(f"manifest.run.{key}", fields)

    def test_manifest_run_numeric_type_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            manifest_path = _manifest_path(base, plan[RANDOM])
            manifest = _read_json(manifest_path)
            manifest["run"]["seed"] = float(manifest["run"]["seed"])
            manifest["run"]["controller_seed"] = float(
                manifest["run"]["controller_seed"]
            )
            manifest["run"]["planned_quota"] = float(
                manifest["run"]["planned_quota"]
            )
            manifest["run"]["total_steps"] = float(
                manifest["run"]["total_steps"]
            )
            manifest["run"]["matched_budget_planned"] = 1
            _write_json(manifest_path, manifest)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            for key in (
                "seed",
                "controller_seed",
                "planned_quota",
                "total_steps",
                "matched_budget_planned",
            ):
                with self.subTest(key=key):
                    self.assertIn(f"manifest.run.{key}", fields)

    # ── 22. Attempt bound ──────────────────────────────────────────────────

    def test_attempt_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["attempt"] = 1.0
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.attempt",
                    _error_fields(exc.findings),
                )

    # ── 23. Output paths mismatch ──────────────────────────────────────────

    def test_output_paths_mismatch_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["output_paths"]["results"] = "other.json"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.output_paths",
                    _error_fields(exc.findings),
                )

    # ── 24. Results fingerprint drift ──────────────────────────────────────

    def test_results_fingerprint_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(data):
                    data["fingerprint"] = "deadbeef12345678"
                _create_cell_fixture(base, cell, 1, tamper_results=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "results.json.fingerprint",
                    _error_fields(exc.findings),
                )

    # ── 25. Results task/seed/ablation drift ───────────────────────────────

    def test_results_task_seed_ablation_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(data):
                    data["task"] = "wrong_task"
                    data["seed"] = 999
                    data["ablation"] = "wrong_arm"
                _create_cell_fixture(base, cell, 1, tamper_results=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                fields = _error_fields(exc.findings)
                self.assertIn("results.json.task", fields)
                self.assertIn("results.json.seed", fields)
                self.assertIn("results.json.ablation", fields)

    def test_results_grouping_and_attempt_fields_require_presence(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            results_path = _results_path(base, plan[FULL])
            results = _read_json(results_path)
            for key in ("task", "seed", "ablation", "attempt"):
                results.pop(key)
            _write_json(results_path, results)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            for key in ("task", "seed", "ablation", "attempt"):
                with self.subTest(key=key):
                    self.assertIn(f"results.json.{key}", fields)

    def test_results_seed_and_attempt_type_drift_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            results_path = _results_path(base, plan[FULL])
            results = _read_json(results_path)
            results["seed"] = float(results["seed"])
            results["attempt"] = float(results["attempt"])
            _write_json(results_path, results)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            fields = _error_fields(raised.exception.findings)
            self.assertIn("results.json.seed", fields)
            self.assertIn("results.json.attempt", fields)

    # ── 26. Missing None fields rejected ───────────────────────────────────

    def test_missing_none_fields_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # full_finetune plans both fields explicitly as None.
            for cell in plan:
                if cell["arm"] == "full_finetune":
                    def tamper(m):
                        m["controller_config_effective"].pop("arm_alias_of")
                        m["controller_config_effective"].pop("requested_quota")
                    _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)
                else:
                    _create_cell_fixture(base, cell, 1)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                fields = _error_fields(exc.findings)
                self.assertIn(
                    "manifest.controller_config_effective.arm_alias_of",
                    fields,
                )
                self.assertIn(
                    "manifest.controller_config_effective.requested_quota",
                    fields,
                )

    # ── 27. Only policy_effective_config and runtime_quota_total_steps ─────

    def test_extra_controller_keys_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    if "controller_config_effective" in m:
                        m["controller_config_effective"]["bogus_field"] = "evil"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                fields = _error_fields(exc.findings)
                self.assertTrue(
                    any("bogus_field" in f for f in fields),
                    msg=f"unexpected key not found in {fields}",
                )

    def test_documented_runtime_controller_extras_are_accepted(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                self.assertNotIn(
                    "policy_effective_config", cell["controller_config"]
                )
                self.assertNotIn(
                    "runtime_quota_total_steps", cell["controller_config"]
                )
            _create_full_fixture(base, plan)

            for cell in plan:
                manifest_path = _manifest_path(base, cell)
                manifest = _read_json(manifest_path)
                self.assertIn(
                    "policy_effective_config",
                    manifest["controller_config_effective"],
                )
                manifest["controller_config_effective"][
                    "runtime_quota_total_steps"
                ] = cell["total_steps"] if cell["is_skipping_arm"] else None
                _write_json(manifest_path, manifest)

            result = validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir=base,
            )
            self.assertEqual(len(result["valid_runs"]), 12)
            self.assertEqual(result["findings"], [])

    def test_every_planned_controller_field_requires_presence(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)
            expected_fields = set()

            for cell in plan:
                manifest_path = _manifest_path(base, cell)
                manifest = _read_json(manifest_path)
                for key in cell["controller_config"]:
                    expected_fields.add(
                        f"manifest.controller_config_effective.{key}"
                    )
                    manifest["controller_config_effective"].pop(key)
                _write_json(manifest_path, manifest)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            self.assertTrue(
                expected_fields <= _error_fields(raised.exception.findings)
            )

    def test_every_planned_controller_field_rejects_value_drift(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)
            expected_fields = set()

            for cell in plan:
                manifest_path = _manifest_path(base, cell)
                manifest = _read_json(manifest_path)
                for key in cell["controller_config"]:
                    expected_fields.add(
                        f"manifest.controller_config_effective.{key}"
                    )
                    manifest["controller_config_effective"][key] = {
                        "drift": key
                    }
                _write_json(manifest_path, manifest)

            with self.assertRaises(CompletedMatrixError) as raised:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )

            self.assertTrue(
                expected_fields <= _error_fields(raised.exception.findings)
            )

    # ── 28. Hash tampering ─────────────────────────────────────────────────

    def test_hash_tampering_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)

            # Tamper with results.json content
            for cell in plan:
                cell_dir = os.path.join(base, cell["arm"], cell["fingerprint"],
                                        "attempt-001")
                results_path = os.path.join(cell_dir, "results.json")
                with open(results_path, "a", encoding="utf-8") as f:
                    f.write(" ")

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.verification.artifacts.results.json.sha256",
                    _error_fields(exc.findings),
                )

    # ── 29. Multiple independent corruptions collected into one exception ──

    def test_multiple_corruptions_collected_into_one_exception(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            # Create only some cells, leave others missing
            for i, cell in enumerate(plan):
                if i % 2 == 0:
                    _create_cell_fixture(base, cell, 1)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                # Should have both cell_directory errors and cell_completion errors
                fields = _error_fields(exc.findings)
                self.assertIn("cell_directory", fields)
                self.assertIn("cell_completion", fields)

    # ── 30. Extra output paths on manifest rejected ────────────────────────

    def test_extra_manifest_output_paths_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["output_paths"]["extra"] = "extra.json"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn(
                    "manifest.output_paths",
                    _error_fields(exc.findings),
                )

    # ── 31. Mixed attempts (one valid, one invalid) ────────────────────────

    def test_mixed_attempts_fails_when_no_valid_completion(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                _create_cell_fixture(base, cell, 1, status="failed")
                _create_cell_fixture(base, cell, 2, status="running")

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("cell_completion", _error_fields(exc.findings))

    # ── 32. Interrupted manifest rejected ──────────────────────────────────

    def test_interrupted_manifest_rejected(self):
        """An interrupted (non-completed, non-failed, non-running) status."""
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            for cell in plan:
                def tamper(m):
                    m["status"] = "interrupted"
                _create_cell_fixture(base, cell, 1, tamper_manifest=tamper)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("manifest.status", _error_fields(exc.findings))

    # ── 33. Missing manifest in attempt ────────────────────────────────────

    def test_missing_manifest_in_attempt_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            _create_full_fixture(base, plan)
            # Delete one manifest
            cell = plan[FULL]
            manifest_path = os.path.join(base, cell["arm"], cell["fingerprint"],
                                         "attempt-001", "run_manifest.json")
            os.unlink(manifest_path)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("manifest", _error_fields(exc.findings))

    # ── 34. Non-positive attempt number ────────────────────────────────────

    def test_non_positive_attempt_number_rejected(self):
        with tempfile.TemporaryDirectory(prefix="lerna-completed-") as tmp:
            base = os.path.join(tmp, "output")
            plan = _build_plan(base_output_dir=base)
            cell = plan[FULL]
            arm = cell["arm"]
            fingerprint = cell["fingerprint"]
            os.makedirs(os.path.join(base, arm, fingerprint, "attempt-000"), exist_ok=True)

            try:
                validate_phase1_3_completed_matrix(
                    plan,
                    tasks=[TASK],
                    seeds=[SEED],
                    target_skip_rates=list(RATES),
                    minimum_seed_count=1,
                    base_output_dir=base,
                )
                self.fail("expected CompletedMatrixError")
            except CompletedMatrixError as exc:
                self.assertIn("attempts", _error_fields(exc.findings))

    # ── 35. Noncanonical arm in plan ───────────────────────────────────────

    def test_noncanonical_arm_rejected(self):
        plan = _build_plan()
        plan[FULL]["arm"] = "bogus_arm"
        plan[FULL]["control"] = "bogus_arm"
        plan[FULL]["controller_config"]["arm"] = "bogus_arm"
        plan[FULL]["controller_config"]["control"] = "bogus_arm"
        plan[FULL]["identity_inputs"]["control"] = "bogus_arm"
        plan[FULL]["fingerprint"] = build_scientific_fingerprint(
            plan[FULL]["identity_inputs"]
        )
        try:
            validate_phase1_3_completed_matrix(
                plan,
                tasks=[TASK],
                seeds=[SEED],
                target_skip_rates=list(RATES),
                minimum_seed_count=1,
                base_output_dir="fixture",
            )
            self.fail("expected CompletedMatrixError")
        except CompletedMatrixError as exc:
            self.assertIn("arm", _error_fields(exc.findings))


if __name__ == "__main__":
    unittest.main()