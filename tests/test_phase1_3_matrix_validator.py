"""Dependency-light tests for the pure Phase 1.3 matrix-plan validator (6C-3)."""

import copy
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MATRIX_MODULE_PATH = REPO_ROOT / "lerna" / "utils" / "phase1_3_matrix.py"
_SPEC = importlib.util.spec_from_file_location(
    "phase1_3_matrix",
    MATRIX_MODULE_PATH,
)
matrix = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(matrix)

PHASE1_3_CANONICAL_ARMS = matrix.PHASE1_3_CANONICAL_ARMS
STRICT_TARGET_SKIP_RATES = matrix.STRICT_TARGET_SKIP_RATES
POLICY_MIN_STEP = matrix.POLICY_MIN_STEP
PLANNED_CELL_REQUIRED_FIELDS = matrix.PLANNED_CELL_REQUIRED_FIELDS
MatrixPlanError = matrix.MatrixPlanError
validate_phase1_3_matrix_plan = matrix.validate_phase1_3_matrix_plan
build_scientific_fingerprint = matrix.build_scientific_fingerprint

TASK = "synthetic_task"
SEED = 7
RATES = [0.30, 0.40]
MODEL_ID = "synthetic-model"
EPOCHS = 3
TOTAL_STEPS = 200
GIT_SHA = "0123abcd4567ef890123abcd4567ef8901234567"
BASE_DIR = os.path.join("synthetic-results", "phase1_3")
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
LER_SAFETY_VALUES = {
    "use_rho_vg_safety": True,
    "rho_veto_threshold": 0.5,
    "use_loss_spike_safety": True,
    "loss_spike_factor": 2.0,
    "loss_spike_window": 5,
}
IDENTITY_FIELDS = (
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
)
CONTROLLER_FIELDS = (
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
)
ONLINE_DIAG_FIELDS = (
    "requested_mode",
    "mode",
    "enabled",
    "timing",
    "parameter_sample_size",
    "update_interval",
    "reason",
    "sample_seed",
)

# Arm index offsets inside one (task, seed, rate) six-arm group.
FULL, RANDOM, FIXED, GUARDED, LER, LER_SAFE = range(6)

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
    "phase1_3_matrix_guarded",
    sys.argv[1],
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.validate_phase1_3_matrix_plan)
assert callable(module.build_scientific_fingerprint)
print("GUARDED_LOAD_OK")
"""


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
        config.update(copy.deepcopy(LER_SAFETY_VALUES))
    return config


def _build_cell(
    arm,
    rate,
    base_output_dir=BASE_DIR,
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


def _build_plan(rates=None, base_output_dir=BASE_DIR, tasks=None, seeds=None):
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


def _kwargs(overrides):
    kwargs = {
        "tasks": [TASK],
        "seeds": [SEED],
        "target_skip_rates": list(RATES),
        "minimum_seed_count": 1,
        "base_output_dir": BASE_DIR,
    }
    kwargs.update(overrides)
    return kwargs


def _validate(plan, **overrides):
    return validate_phase1_3_matrix_plan(plan, **_kwargs(overrides))


def _findings(plan, **overrides):
    try:
        validate_phase1_3_matrix_plan(plan, **_kwargs(overrides))
    except MatrixPlanError as exc:
        return exc.findings
    raise AssertionError("expected MatrixPlanError, but validation passed")


def _fields(findings):
    return {finding["field"] for finding in findings}


def _messages(findings, field):
    return [f["message"] for f in findings if f["field"] == field]


def _refresh(cell, base_output_dir=BASE_DIR):
    cell["fingerprint"] = build_scientific_fingerprint(cell["identity_inputs"])
    cell["planned_arm_dir"] = os.path.join(
        base_output_dir, cell["arm"], cell["fingerprint"]
    )
    return cell


def _set_online(cell, key, value):
    for target in (
        cell["online_diagnostics"],
        cell["identity_inputs"]["online_diagnostics"],
        cell["controller_config"]["online_diagnostics"],
    ):
        target[key] = copy.deepcopy(value)
    return _refresh(cell)


def _del_online(cell, key):
    for target in (
        cell["online_diagnostics"],
        cell["identity_inputs"]["online_diagnostics"],
        cell["controller_config"]["online_diagnostics"],
    ):
        del target[key]
    return _refresh(cell)


def _set_sub(cell, sub_key, key, value):
    for owner in (cell["controller_config"], cell["identity_inputs"]):
        owner.setdefault(sub_key, {})[key] = copy.deepcopy(value)
    return _refresh(cell)


def _del_sub(cell, sub_key, key):
    for owner in (cell["controller_config"], cell["identity_inputs"]):
        del owner[sub_key][key]
    return _refresh(cell)


class Phase13MatrixValidatorTests(unittest.TestCase):
    # 0. Literal public-constant contract.
    def test_public_constants_literal_values(self):
        self.assertEqual(
            PHASE1_3_CANONICAL_ARMS,
            (
                "full_finetune",
                "exact_random",
                "fixed_phase_strat",
                "phase_strat_guarded",
                "ler_guided_stratified",
                "ler_guided_stratified_safe",
            ),
        )
        self.assertEqual(STRICT_TARGET_SKIP_RATES, (0.30, 0.40))
        self.assertEqual(POLICY_MIN_STEP, 50)

    # 1. Complete valid 12-cell plan.
    def test_valid_plan_returns_empty_findings(self):
        self.assertEqual(tuple(RATES), STRICT_TARGET_SKIP_RATES)
        plan = _build_plan()
        self.assertEqual(len(plan), 12)
        self.assertEqual(_validate(plan), [])

    # 2. Validation performs no filesystem writes.
    def test_validation_performs_no_filesystem_writes(self):
        with tempfile.TemporaryDirectory(prefix="lerna-matrix-plan-") as tmp:
            base = os.path.join(tmp, "planned_phase1_3_output")
            plan = _build_plan(base_output_dir=base)
            self.assertEqual(_validate(plan, base_output_dir=base), [])
            self.assertFalse(os.path.exists(base))
            for cell in plan:
                self.assertFalse(os.path.exists(cell["planned_arm_dir"]))
            self.assertEqual(os.listdir(tmp), [])

    # 3. MatrixPlanError structure.
    def test_error_carries_structured_findings(self):
        plan = _build_plan()
        plan[FULL]["skip_update_mode"] = "unfreeze"
        plan[RANDOM]["matched_budget"] = False
        try:
            _validate(plan, minimum_seed_count=0)
        except MatrixPlanError as exc:
            self.assertIsInstance(exc, ValueError)
            self.assertIsInstance(exc.findings, list)
            self.assertGreaterEqual(len(exc.findings), 3)
            for finding in exc.findings:
                self.assertEqual(
                    set(finding), {"severity", "field", "cell", "message"}
                )
                self.assertEqual(finding["severity"], "error")
        else:
            self.fail("expected MatrixPlanError")

    # 4. Malformed plans, cells, missing fields, and strict types.
    def test_reject_non_list_plans_and_non_dict_cells(self):
        for bad_plan in ({}, None, "plan", tuple(_build_plan())):
            with self.subTest(plan=type(bad_plan).__name__):
                self.assertIn("plan", _fields(_findings(bad_plan)))
        for bad_cell in (None, 42, "cell", ["oops"]):
            with self.subTest(cell=type(bad_cell).__name__):
                plan = _build_plan()
                plan[3] = bad_cell
                self.assertIn("plan[3]", _fields(_findings(plan)))

    def test_reject_missing_required_cell_fields(self):
        for field in sorted(PLANNED_CELL_REQUIRED_FIELDS):
            with self.subTest(field=field):
                plan = _build_plan()
                del plan[RANDOM][field]
                findings = _findings(plan)
                self.assertIn(
                    f"required field {field!r} is missing",
                    _messages(findings, field),
                )

    def test_reject_strict_type_violations(self):
        cases = [
            ("task", 123),
            ("model_id", ""),
            ("skip_update_mode", None),
            ("scheduler_step_policy", 7),
            ("is_skipping_arm", 1),
            ("matched_budget", "true"),
            ("no_early_stopping", 0),
            ("online_diagnostics", []),
            ("controller_config", "config"),
            ("identity_inputs", 9),
            ("planned_arm_dir", ""),
            ("fingerprint", 12345),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                plan = _build_plan()
                plan[FULL][field] = value
                self.assertIn(field, _fields(_findings(plan)))

    # 5. Integer strictness for every integer cell field.
    def test_integer_fields_reject_bool_and_floats(self):
        integer_fields = (
            "training_seed",
            "policy_seed",
            "num_epochs",
            "total_steps",
            "min_step",
            "planned_skips",
        )
        for field in integer_fields:
            reference = _build_cell("exact_random", RATES[0])[field]
            for value in (True, float(reference), float(reference) + 0.5):
                with self.subTest(field=field, value=value):
                    plan = _build_plan()
                    plan[RANDOM][field] = value
                    self.assertIn(field, _fields(_findings(plan)))

    def test_requested_quota_accepts_only_none_or_int(self):
        quota = _build_cell("exact_random", RATES[0])["requested_quota"]
        for value in (True, float(quota), quota + 0.5, "60"):
            with self.subTest(value=value):
                plan = _build_plan()
                plan[RANDOM]["requested_quota"] = value
                findings = _findings(plan)
                self.assertIn(
                    "value must be null or an integer; bools and floats are invalid",
                    _messages(findings, "requested_quota"),
                )

    def test_requested_quota_none_rejected_on_skipping_arm(self):
        quota = round(RATES[0] * TOTAL_STEPS)
        for offset, arm in zip(
            (RANDOM, FIXED, GUARDED, LER, LER_SAFE), PHASE1_3_CANONICAL_ARMS[1:]
        ):
            with self.subTest(arm=arm):
                plan = _build_plan()
                plan[offset]["requested_quota"] = None
                findings = _findings(plan)
                self.assertIn(
                    f"expected {quota!r}, got None",
                    _messages(findings, "requested_quota"),
                )

    # 6. Rates and fingerprint formats.
    def test_reject_non_finite_and_non_float_rates(self):
        for value in (float("nan"), float("inf"), "0.3", 1, True):
            with self.subTest(value=value):
                plan = _build_plan()
                plan[RANDOM]["target_skip_rate"] = value
                self.assertIn("target_skip_rate", _fields(_findings(plan)))
        for value in (float("nan"), "0.4", 2.0, -0.1):
            with self.subTest(dimension_rate=value):
                findings = _findings(
                    _build_plan(), target_skip_rates=[RATES[0], value]
                )
                self.assertIn("target_skip_rates[1]", _fields(findings))

    def test_fingerprint_must_be_16_lowercase_hex(self):
        for cell in _build_plan():
            self.assertTrue(FINGERPRINT_RE.fullmatch(cell["fingerprint"]))
        bad_fingerprints = (
            "ABCDEF0123456789",
            "0123456789abcde",
            "0123456789abcdef0",
            "0123456789abcdeg",
        )
        for value in bad_fingerprints:
            with self.subTest(fingerprint=value):
                plan = _build_plan()
                plan[FULL]["fingerprint"] = value
                findings = _findings(plan)
                self.assertIn(
                    "fingerprint must be 16 lowercase hexadecimal characters",
                    _messages(findings, "fingerprint"),
                )

    # 7. Dimension rules.
    def test_dimension_duplicates_types_and_seed_count(self):
        cases = [
            ({"tasks": [TASK, TASK]}, "tasks"),
            ({"tasks": TASK}, "tasks"),
            ({"tasks": [""]}, "tasks[0]"),
            ({"seeds": [SEED, SEED]}, "seeds"),
            ({"seeds": {SEED}}, "seeds"),
            ({"seeds": [True]}, "seeds[0]"),
            ({"seeds": [7.0]}, "seeds[0]"),
            ({"target_skip_rates": [RATES[0], RATES[0]]}, "target_skip_rates"),
            ({"target_skip_rates": tuple(RATES)}, "target_skip_rates"),
            ({"minimum_seed_count": 0}, "minimum_seed_count"),
            ({"minimum_seed_count": True}, "minimum_seed_count"),
            ({"base_output_dir": ""}, "base_output_dir"),
        ]
        for overrides, field in cases:
            with self.subTest(field=field):
                findings = _findings(_build_plan(), **overrides)
                self.assertIn(field, _fields(findings))
        findings = _findings(_build_plan(), minimum_seed_count=2)
        self.assertIn(
            "at least 2 unique seeds are required", _messages(findings, "seeds")
        )

    # 8. Exact global ordering.
    def test_enforce_exact_global_cell_order(self):
        swaps = [
            (FULL, RANDOM),
            (FIXED, GUARDED),
            (LER_SAFE, 6),
        ]
        for left, right in swaps:
            with self.subTest(swap=(left, right)):
                plan = _build_plan()
                plan[left], plan[right] = plan[right], plan[left]
                self.assertIn("plan.order", _fields(_findings(plan)))
        reversed_rates = _build_plan(rates=[RATES[1], RATES[0]])
        self.assertIn("plan.order", _fields(_findings(reversed_rates)))

    def test_task_and_seed_group_order_enforced(self):
        tasks = [TASK, "synthetic_task_b"]
        seeds = [SEED, SEED + 4]
        overrides = {"tasks": tasks, "seeds": seeds, "minimum_seed_count": 2}
        group = len(RATES) * len(PHASE1_3_CANONICAL_ARMS)
        task_block = group * len(seeds)

        plan = _build_plan(tasks=tasks, seeds=seeds)
        self.assertEqual(len(plan), 2 * task_block)
        self.assertEqual(_validate(plan, **overrides), [])

        swapped_tasks = plan[task_block:] + plan[:task_block]
        findings = _findings(swapped_tasks, **overrides)
        self.assertEqual(_fields(findings), {"plan.order"})

        swapped_seeds = (
            plan[group:task_block] + plan[:group] + plan[task_block:]
        )
        findings = _findings(swapped_seeds, **overrides)
        self.assertEqual(_fields(findings), {"plan.order"})

    # 9. Duplicate, missing, unexpected, and wrong-count cells.
    def test_reject_duplicate_missing_unexpected_cells(self):
        plan = _build_plan()
        plan.append(copy.deepcopy(plan[FULL]))
        findings = _findings(plan)
        self.assertIn("plan.length", _fields(findings))
        self.assertTrue(
            any(
                "duplicate planned cell" in m
                for m in _messages(findings, "plan.cells")
            )
        )

        plan = _build_plan()
        del plan[LER_SAFE]
        findings = _findings(plan)
        self.assertIn("plan.length", _fields(findings))
        self.assertIn(
            "required planned cell is missing", _messages(findings, "plan.cells")
        )

        plan = _build_plan()
        stray = copy.deepcopy(plan[RANDOM])
        stray["task"] = "unexpected_task"
        stray["identity_inputs"]["task"] = "unexpected_task"
        _refresh(stray)
        plan[RANDOM] = stray
        findings = _findings(plan)
        self.assertIn(
            "planned cell is outside the requested cross-product",
            _messages(findings, "plan.cells"),
        )
        self.assertIn(
            "required planned cell is missing", _messages(findings, "plan.cells")
        )

    # 10. Paired fields across each six-arm group.
    def test_enforce_paired_fields_across_six_arm_group(self):
        paired_cases = [
            ("model_id", "other-model"),
            ("total_steps", TOTAL_STEPS + 100),
            ("num_epochs", EPOCHS + 1),
            ("policy_seed", SEED + 1),
        ]
        for field, value in paired_cases:
            with self.subTest(field=field):
                plan = _build_plan()
                plan[RANDOM][field] = value
                self.assertIn(
                    "paired_group." + field, _fields(_findings(plan))
                )
        for field, value in (
            ("task", "other_task"),
            ("training_seed", SEED + 1),
            ("target_skip_rate", 0.5),
        ):
            with self.subTest(key_field=field):
                plan = _build_plan()
                plan[RANDOM][field] = value
                findings = _findings(plan)
                self.assertTrue(
                    {"plan.cells", "plan.order"} & _fields(findings)
                )

    # 11. Execution contract constants.
    def test_enforce_execution_contract(self):
        cases = [
            ("skip_update_mode", "unfreeze"),
            ("matched_budget", False),
            ("no_early_stopping", False),
            ("scheduler_step_policy", "always_step"),
            ("min_step", POLICY_MIN_STEP + 1),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                plan = _build_plan()
                plan[GUARDED][field] = value
                self.assertIn(field, _fields(_findings(plan)))
        plan = _build_plan()
        plan[FULL]["total_steps"] = POLICY_MIN_STEP
        findings = _findings(plan)
        self.assertIn(
            "total_steps must be greater than min_step",
            _messages(findings, "total_steps"),
        )

    # 12. full_finetune contract.
    def test_full_finetune_contract(self):
        cases = [
            ("requested_quota", 0),
            ("planned_skips", 5),
            ("is_skipping_arm", True),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                plan = _build_plan()
                plan[FULL][field] = value
                self.assertIn(field, _fields(_findings(plan)))

    # 13. Exact quotas for the five skipping arms and infeasibility.
    def test_exact_quota_for_all_skipping_arms(self):
        for offset, arm in zip(
            (RANDOM, FIXED, GUARDED, LER, LER_SAFE), PHASE1_3_CANONICAL_ARMS[1:]
        ):
            expected = round(RATES[0] * TOTAL_STEPS)
            with self.subTest(arm=arm, field="requested_quota"):
                plan = _build_plan()
                plan[offset]["requested_quota"] = expected + 1
                findings = _findings(plan)
                self.assertIn(
                    f"expected {expected}, got {expected + 1}",
                    _messages(findings, "requested_quota"),
                )
            with self.subTest(arm=arm, field="planned_skips"):
                plan = _build_plan()
                plan[offset]["planned_skips"] = expected - 1
                self.assertIn("planned_skips", _fields(_findings(plan)))

    def test_infeasible_quota_rejected_without_clipping(self):
        infeasible_rates = [0.30, 0.90]
        plan = _build_plan(rates=infeasible_rates)
        findings = _findings(plan, target_skip_rates=infeasible_rates)
        self.assertIn(
            "exact quota is infeasible after min_step and must not be clipped",
            _messages(findings, "requested_quota"),
        )

    # 14. Online diagnostics contract.
    def test_online_diagnostics_canonical_values(self):
        offline_cases = [
            ("mode", "sampled_lagged"),
            ("enabled", True),
            ("timing", "post_decision_after_backward"),
            ("parameter_sample_size", 8),
            ("update_interval", 4),
            ("sample_seed", SEED),
            ("requested_mode", "sampled_lagged"),
        ]
        for key, value in offline_cases:
            with self.subTest(arm="exact_random", key=key):
                plan = _build_plan()
                _set_online(plan[RANDOM], key, value)
                self.assertIn(
                    "online_diagnostics." + key, _fields(_findings(plan))
                )
        online_cases = [
            ("mode", "off"),
            ("enabled", False),
            ("timing", "none"),
            ("parameter_sample_size", 0),
            ("update_interval", 0),
            ("sample_seed", SEED + 1),
            ("sample_seed", None),
            ("requested_mode", "off"),
        ]
        for key, value in online_cases:
            with self.subTest(arm="ler_guided_stratified", key=key):
                plan = _build_plan()
                _set_online(plan[LER], key, value)
                self.assertIn(
                    "online_diagnostics." + key, _fields(_findings(plan))
                )

    def test_online_diagnostics_copy_equality(self):
        plan = _build_plan()
        plan[LER]["online_diagnostics"]["reason"] = "diverged copy"
        fields = _fields(_findings(plan))
        self.assertIn("identity_inputs.online_diagnostics", fields)
        self.assertIn("controller_config.online_diagnostics", fields)

    def test_online_diagnostics_required_field_deletion(self):
        for offset, arm in ((RANDOM, "exact_random"), (LER, "ler_guided_stratified")):
            for field in ONLINE_DIAG_FIELDS:
                with self.subTest(arm=arm, field=field, copies="all"):
                    plan = _build_plan()
                    _del_online(plan[offset], field)
                    findings = _findings(plan)
                    self.assertIn(
                        f"required field 'online_diagnostics.{field}' is missing",
                        _messages(findings, "online_diagnostics." + field),
                    )
        for field in ONLINE_DIAG_FIELDS:
            with self.subTest(field=field, copies="identity"):
                plan = _build_plan()
                del plan[LER]["identity_inputs"]["online_diagnostics"][field]
                _refresh(plan[LER])
                self.assertIn(
                    "identity_inputs.online_diagnostics",
                    _fields(_findings(plan)),
                )
            with self.subTest(field=field, copies="controller"):
                plan = _build_plan()
                del plan[LER]["controller_config"]["online_diagnostics"][field]
                self.assertIn(
                    "controller_config.online_diagnostics",
                    _fields(_findings(plan)),
                )

    # 15. Outer controller mirroring and RVD rejection.
    def test_outer_controller_mirroring(self):
        cases = [
            ("arm", "full_finetune"),
            ("arm_alias_of", "random_skip"),
            ("control", "full_finetune"),
            ("policy_class", "WrongPolicy"),
            ("compute_saving_mechanism", "none"),
            ("policy_seed", SEED + 1),
            ("target_skip_rate", 0.5),
            ("min_step", POLICY_MIN_STEP - 1),
            ("configured_total_steps", TOTAL_STEPS + 10),
            ("requested_quota", round(RATES[0] * TOTAL_STEPS) + 1),
            ("matched_budget", False),
            ("is_skipping_arm", False),
            ("allow_early_stopping_with_skipping", True),
            ("early_stopping_active", True),
            ("num_epochs", EPOCHS + 1),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                plan = _build_plan()
                plan[RANDOM]["controller_config"][key] = value
                self.assertIn(
                    "controller_config." + key, _fields(_findings(plan))
                )
        for key in CONTROLLER_FIELDS:
            with self.subTest(missing=key):
                plan = _build_plan()
                del plan[RANDOM]["controller_config"][key]
                self.assertIn(
                    "controller_config." + key, _fields(_findings(plan))
                )

    def test_reject_rvd_provenance(self):
        plan = _build_plan()
        plan[RANDOM]["controller_config"]["rvd"] = {"veto_mode": "none"}
        findings = _findings(plan)
        self.assertIn(
            "RVD provenance is not part of the canonical Phase 1.3 matrix",
            _messages(findings, "controller_config.rvd"),
        )

    # 16. Phase controller contract.
    def test_phase_controller_presence_and_copy_equality(self):
        plan = _build_plan()
        del plan[FIXED]["controller_config"]["phase_strat_controller"]
        self.assertIn(
            "controller_config.phase_strat_controller",
            _fields(_findings(plan)),
        )

        plan = _build_plan()
        del plan[FIXED]["identity_inputs"]["phase_strat_controller"]
        _refresh(plan[FIXED])
        self.assertIn(
            "identity_inputs.phase_strat_controller",
            _fields(_findings(plan)),
        )

        plan = _build_plan()
        controller_copy = plan[FIXED]["controller_config"]["phase_strat_controller"]
        controller_copy["phase_weights"] = [0.4, 0.6]
        findings = _findings(plan)
        self.assertIn(
            "controller and identity phase provenance copies must match",
            _messages(findings, "phase_strat_controller"),
        )

    def test_phase_controller_values_and_arrays(self):
        prefix = "controller_config.phase_strat_controller."
        mid = (POLICY_MIN_STEP + TOTAL_STEPS) // 2
        quota = round(RATES[0] * TOTAL_STEPS)
        cases = [
            ("controller_class", "WrongController"),
            ("policy_class", POLICY_CLASSES["fixed_phase_strat"]),
            ("control", "exact_random"),
            ("policy_name", "exact_random"),
            ("policy_seed", SEED + 1),
            ("target_skip_rate", 0.5),
            ("total_steps", TOTAL_STEPS + 1),
            ("min_step", POLICY_MIN_STEP - 1),
            ("requested_quota", quota + 1),
            ("n_phases", 0),
            ("phase_weights", [0.6, 0.6]),
            ("phase_weights", [0.5, "0.5"]),
            ("phase_bounds", [0, mid, TOTAL_STEPS]),
            ("phase_bounds", [POLICY_MIN_STEP, POLICY_MIN_STEP - 10, TOTAL_STEPS]),
            ("phase_eligible", [mid - POLICY_MIN_STEP - 5, TOTAL_STEPS - mid + 5]),
            ("phase_quota", [quota - quota // 2, quota // 2 + 1]),
        ]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                plan = _build_plan()
                _set_sub(plan[FIXED], "phase_strat_controller", key, value)
                self.assertIn(prefix + key, _fields(_findings(plan)))

        big_quota = round(RATES[1] * TOTAL_STEPS)
        plan = _build_plan()
        _set_sub(
            plan[6 + FIXED],
            "phase_strat_controller",
            "phase_quota",
            [big_quota - 4, 4],
        )
        findings = _findings(plan)
        self.assertIn(
            "phase quota cannot exceed phase eligibility",
            _messages(findings, prefix + "phase_quota"),
        )

        for key in ("policy_seed", "phase_bounds", "requested_quota"):
            with self.subTest(missing=key):
                plan = _build_plan()
                _del_sub(plan[FIXED], "phase_strat_controller", key)
                self.assertIn(prefix + key, _fields(_findings(plan)))

    def test_phase_controller_guarded_only_fields(self):
        prefix = "controller_config.phase_strat_controller."
        guarded_cases = [
            ("max_consecutive_skips", 0),
            ("risk_gamma", 2),
            ("guarded_safety", {"use_rho_vg": True}),
            ("guarded_safety", "not-a-dict"),
        ]
        for key, value in guarded_cases:
            with self.subTest(guarded=key, value=value):
                plan = _build_plan()
                _set_sub(plan[GUARDED], "phase_strat_controller", key, value)
                fields = _fields(_findings(plan))
                self.assertTrue(
                    any(field.startswith(prefix + key) for field in fields)
                )
        plan = _build_plan()
        _del_sub(plan[GUARDED], "phase_strat_controller", "guarded_safety")
        self.assertIn(prefix + "guarded_safety", _fields(_findings(plan)))

        for key, value in (
            ("max_consecutive_skips", 3),
            ("risk_gamma", 1.5),
            ("guarded_safety", {"use_rho_vg": True}),
        ):
            with self.subTest(fixed_gets=key):
                plan = _build_plan()
                _set_sub(plan[FIXED], "phase_strat_controller", key, value)
                findings = _findings(plan)
                self.assertIn(
                    "guarded-only field is not applicable to fixed_phase_strat",
                    _messages(findings, prefix + key),
                )

    # 17. LER controller contract.
    def test_ler_controller_presence_and_copy_equality(self):
        plan = _build_plan()
        del plan[LER]["controller_config"]["ler_guided_controller"]
        self.assertIn(
            "controller_config.ler_guided_controller", _fields(_findings(plan))
        )

        plan = _build_plan()
        del plan[LER]["identity_inputs"]["ler_guided_controller"]
        _refresh(plan[LER])
        self.assertIn(
            "identity_inputs.ler_guided_controller", _fields(_findings(plan))
        )

        plan = _build_plan()
        plan[LER]["controller_config"]["ler_guided_controller"][
            "phase_weights"
        ] = [0.4, 0.6]
        findings = _findings(plan)
        self.assertIn(
            "controller and identity LER provenance copies must match",
            _messages(findings, "ler_guided_controller"),
        )

    def test_ler_controller_values_trackers_and_safety(self):
        prefix = "controller_config.ler_guided_controller."
        cases = [
            ("policy_class", "WrongPolicy"),
            ("controller_class", POLICY_CLASSES["ler_guided_stratified"]),
            ("control", "exact_random"),
            ("policy_name", "exact_random"),
            ("policy_seed", SEED + 1),
            ("target_skip_rate", 0.5),
            ("total_steps", TOTAL_STEPS + 1),
            ("min_step", POLICY_MIN_STEP - 1),
            ("required_tracker_mode", "off"),
            ("required_tracker_timing", "none"),
            ("safety_enabled", True),
            ("n_phases", 0),
            ("max_consecutive_skips", 0),
            ("probe_interval", 0),
            ("min_ler_observations", 0),
            ("ler_guidance_strength", "strong"),
            ("phase_weights", [0.7, 0.7]),
        ]
        for key, value in cases:
            with self.subTest(key=key):
                plan = _build_plan()
                _set_sub(plan[LER], "ler_guided_controller", key, value)
                self.assertIn(prefix + key, _fields(_findings(plan)))

        for key in ("policy_class", "required_tracker_mode", "phase_weights"):
            with self.subTest(missing=key):
                plan = _build_plan()
                _del_sub(plan[LER], "ler_guided_controller", key)
                self.assertIn(prefix + key, _fields(_findings(plan)))

        safe_cases = [
            ("safety_enabled", False),
            ("use_rho_vg_safety", False),
            ("use_loss_spike_safety", False),
            ("rho_veto_threshold", "high"),
            ("loss_spike_factor", 2),
            ("loss_spike_window", 0),
        ]
        for key, value in safe_cases:
            with self.subTest(safe=key):
                plan = _build_plan()
                _set_sub(plan[LER_SAFE], "ler_guided_controller", key, value)
                self.assertIn(prefix + key, _fields(_findings(plan)))
        plan = _build_plan()
        _del_sub(plan[LER_SAFE], "ler_guided_controller", "rho_veto_threshold")
        self.assertIn(
            prefix + "rho_veto_threshold", _fields(_findings(plan))
        )

    # 18. Inapplicable subconfigs and safety-only fields.
    def test_reject_subconfigs_on_inapplicable_arms(self):
        plan = _build_plan()
        phase = _phase_controller(
            "fixed_phase_strat", RATES[0], round(RATES[0] * TOTAL_STEPS), TOTAL_STEPS
        )
        for owner in (
            plan[RANDOM]["controller_config"],
            plan[RANDOM]["identity_inputs"],
        ):
            owner["phase_strat_controller"] = copy.deepcopy(phase)
        _refresh(plan[RANDOM])
        fields = _fields(_findings(plan))
        self.assertIn("controller_config.phase_strat_controller", fields)
        self.assertIn("identity_inputs.phase_strat_controller", fields)

        plan = _build_plan()
        ler = _ler_controller("ler_guided_stratified", RATES[0], TOTAL_STEPS)
        for owner in (
            plan[FULL]["controller_config"],
            plan[FULL]["identity_inputs"],
        ):
            owner["ler_guided_controller"] = copy.deepcopy(ler)
        _refresh(plan[FULL])
        fields = _fields(_findings(plan))
        self.assertIn("controller_config.ler_guided_controller", fields)
        self.assertIn("identity_inputs.ler_guided_controller", fields)

    def test_reject_safety_fields_on_non_safe_ler_arm(self):
        prefix = "controller_config.ler_guided_controller."
        for key, value in sorted(LER_SAFETY_VALUES.items()):
            with self.subTest(safety_field=key):
                plan = _build_plan()
                _set_sub(plan[LER], "ler_guided_controller", key, value)
                findings = _findings(plan)
                self.assertIn(
                    "safety field is not applicable to the non-safe LER arm",
                    _messages(findings, prefix + key),
                )

    # 19. Identity requirements and identity/cell equality.
    def test_identity_required_fields(self):
        for field in IDENTITY_FIELDS:
            with self.subTest(field=field):
                plan = _build_plan()
                del plan[RANDOM]["identity_inputs"][field]
                _refresh(plan[RANDOM])
                findings = _findings(plan)
                self.assertIn(
                    f"required field 'identity_inputs.{field}' is missing",
                    _messages(findings, "identity_inputs." + field),
                )

    def test_identity_must_mirror_planned_cell(self):
        cases = [
            ("task", "other_task"),
            ("training_seed", SEED + 1),
            ("model_id", "other-model"),
            ("num_epochs", EPOCHS + 1),
            ("control", "full_finetune"),
            ("target_skip_rate", 0.5),
            ("policy_seed", SEED + 1),
            ("skip_update_mode", "unfreeze"),
            ("scheduler_step_policy", "always_step"),
            ("no_early_stopping", False),
            ("total_steps", TOTAL_STEPS + 1),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                plan = _build_plan()
                plan[RANDOM]["identity_inputs"][field] = value
                _refresh(plan[RANDOM])
                findings = _findings(plan)
                self.assertIn(
                    f"identity value must equal planned cell {field}",
                    _messages(findings, "identity_inputs." + field),
                )
        for field, value in (
            ("git_sha", ""),
            ("max_samples_requested", 12.5),
            ("train_samples_realized", 10.0),
            ("train_dataset_fingerprint", 7),
        ):
            with self.subTest(typed_field=field):
                plan = _build_plan()
                plan[RANDOM]["identity_inputs"][field] = value
                _refresh(plan[RANDOM])
                self.assertIn(
                    "identity_inputs." + field, _fields(_findings(plan))
                )

    # 20. Fingerprint recomputation, uniqueness, and planned paths.
    def test_reject_recomputed_fingerprint_mismatch(self):
        plan = _build_plan()
        plan[RANDOM]["fingerprint"] = "0" * 16
        plan[RANDOM]["planned_arm_dir"] = os.path.join(
            BASE_DIR, plan[RANDOM]["arm"], "0" * 16
        )
        findings = _findings(plan)
        self.assertTrue(
            any(
                "does not match recomputed" in message
                for message in _messages(findings, "fingerprint")
            )
        )

    def test_reject_duplicate_fingerprints(self):
        plan = _build_plan()
        plan[RANDOM]["fingerprint"] = plan[FULL]["fingerprint"]
        plan[RANDOM]["planned_arm_dir"] = os.path.join(
            BASE_DIR, plan[RANDOM]["arm"], plan[FULL]["fingerprint"]
        )
        findings = _findings(plan)
        self.assertTrue(
            any(
                "is not unique" in message
                for message in _messages(findings, "fingerprint")
            )
        )

    def test_reject_incorrect_and_duplicate_planned_paths(self):
        plan = _build_plan()
        plan[FIXED]["planned_arm_dir"] = os.path.join(
            BASE_DIR, "elsewhere", plan[FIXED]["fingerprint"]
        )
        findings = _findings(plan)
        self.assertTrue(
            any(
                "expected planned arm directory" in message
                for message in _messages(findings, "planned_arm_dir")
            )
        )

        plan = _build_plan()
        plan[GUARDED]["planned_arm_dir"] = plan[FIXED]["planned_arm_dir"]
        findings = _findings(plan)
        self.assertTrue(
            any(
                "is not unique" in message
                for message in _messages(findings, "planned_arm_dir")
            )
        )

    # 21. Multiple independent corruptions collected in one error.
    def test_multiple_corruptions_collected_into_one_error(self):
        plan = _build_plan()
        plan[FULL]["skip_update_mode"] = "unfreeze"
        plan[RANDOM]["requested_quota"] = round(RATES[0] * TOTAL_STEPS) + 1
        plan[FIXED]["fingerprint"] = "ABCDEF0123456789"
        try:
            _validate(plan)
        except MatrixPlanError as exc:
            fields = _fields(exc.findings)
            self.assertIn("skip_update_mode", fields)
            self.assertIn("requested_quota", fields)
            self.assertIn("fingerprint", fields)
            corrupted_cells = {
                finding["cell"][3]
                for finding in exc.findings
                if finding["cell"] is not None
            }
            self.assertGreaterEqual(len(corrupted_cells), 3)
        else:
            self.fail("expected one MatrixPlanError collecting all findings")

    # 22. First validator load cannot import the scientific stack.
    def test_first_validator_load_cannot_import_scientific_stack(self):
        result = subprocess.run(
            [sys.executable, "-c", _IMPORT_GUARD_SCRIPT, str(MATRIX_MODULE_PATH)],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("GUARDED_LOAD_OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
