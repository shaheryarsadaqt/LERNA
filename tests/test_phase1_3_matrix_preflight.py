"""6C-5: dependency-controlled tests for Phase 1.3 plan construction."""

import copy
import itertools
import os
from unittest import mock

import pytest
from lerna.utils.phase1_3_matrix import (
    PHASE1_3_CANONICAL_ARMS,
    PLANNED_CELL_REQUIRED_FIELDS,
    validate_phase1_3_matrix_plan,
)
from lerna.utils.run_provenance import build_scientific_fingerprint
from scripts import run_ablation_study as runner

TASK = "mrpc"
SEED = 7
MODEL_ID = "synthetic-model"
TOTAL_STEPS = 200
RATES = (0.30, 0.40)

COMMON_CONTROLLER_FIELDS = {
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
LER_SAFETY_FIELDS = {
    "use_rho_vg_safety",
    "rho_veto_threshold",
    "use_loss_spike_safety",
    "loss_spike_factor",
    "loss_spike_window",
}
CELL_KWARGS = {
    "scheduler_step_policy": "skip_on_backward_skip",
    "max_consecutive_skips": 4,
    "probe_interval": 8,
    "rho_veto_threshold": -0.2,
    "risk_gamma": 0.0,
    "online_ler_mode": "auto",
    "online_ler_parameter_sample_size": 32,
    "online_ler_update_interval": 2,
    "use_rho_vg": True,
    "use_safety_horizon": True,
}


class FakeDataset:
    def __init__(self, size, fingerprint):
        self._size = size
        self._fingerprint = fingerprint

    def __len__(self):
        return self._size


def _hardware_config(max_samples=2000):
    return {
        "per_device_train_batch_size": 32,
        "per_device_eval_batch_size": 64,
        "gradient_accumulation_steps": 1,
        "fp16": True,
        "bf16": False,
        "gradient_checkpointing": False,
        "dataloader_num_workers": 0,
        "max_samples": max_samples,
    }


def _data_facts(task=TASK, total_steps=TOTAL_STEPS, num_epochs=5):
    return {
        "task": task,
        "num_epochs": num_epochs,
        "max_samples_requested": None,
        "max_samples_effective": None,
        "train_samples_realized": 1000,
        "eval_samples_realized": 200,
        "train_dataset_fingerprint": f"{task}-train-fp",
        "eval_dataset_fingerprint": f"{task}-eval-fp",
        "total_steps": total_steps,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "effective_n_gpu": 1,
    }


def _plan_cell(
    arm,
    rate=RATES[0],
    *,
    task=TASK,
    seed=SEED,
    facts=None,
    base_output_dir="planned",
    **overrides,
):
    kwargs = dict(CELL_KWARGS)
    kwargs.update(overrides)
    return runner.plan_phase1_3_cell(
        task_name=task,
        training_seed=seed,
        policy_seed=seed,
        ablation_name=arm,
        target_skip_rate=rate,
        model_name=MODEL_ID,
        data_facts=_data_facts(task) if facts is None else facts,
        git_sha="abc123",
        base_output_dir=base_output_dir,
        **kwargs,
    )


@pytest.mark.parametrize(
    "max_samples,expected_requested,expected_effective",
    [(None, None, 2000), (128, 128, 128)],
)
@pytest.mark.parametrize(
    "cuda_available,visible_gpus",
    [(False, 0), (True, 1), (True, 4)],
)
def test_resolve_task_data_facts_exact_inputs_and_copying(
    monkeypatch,
    tmp_path,
    max_samples,
    expected_requested,
    expected_effective,
    cuda_available,
    visible_gpus,
):
    monkeypatch.chdir(tmp_path)
    original_hw = _hardware_config()
    original_snapshot = copy.deepcopy(original_hw)
    tokenizer = object()
    train_ds = FakeDataset(1000, "train-fp")
    eval_ds = FakeDataset(200, "eval-fp")
    load_task = mock.Mock(return_value=(train_ds, eval_ds, {"num_labels": 2}))
    device_count = mock.Mock(return_value=visible_gpus)

    monkeypatch.setattr(runner, "get_training_config", lambda profile: original_hw)
    monkeypatch.setattr(runner, "load_glue_task", load_task)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(runner.torch.cuda, "device_count", device_count)

    facts = runner.resolve_task_data_facts(
        TASK,
        tokenizer,
        max_samples,
        "server",
    )

    assert original_hw == original_snapshot
    load_task.assert_called_once_with(
        TASK,
        tokenizer,
        max_length=128,
        max_samples=expected_effective,
    )
    assert device_count.call_count == int(cuda_available)
    assert facts == {
        "task": TASK,
        "num_epochs": 5,
        "max_samples_requested": expected_requested,
        "max_samples_effective": expected_effective,
        "train_samples_realized": 1000,
        "eval_samples_realized": 200,
        "train_dataset_fingerprint": "train-fp",
        "eval_dataset_fingerprint": "eval-fp",
        "total_steps": 160,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "effective_n_gpu": 1,
    }
    assert set(facts) == set(_data_facts())
    assert all(
        type(facts[field]) is int
        for field in (
            "num_epochs",
            "train_samples_realized",
            "eval_samples_realized",
            "total_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "effective_n_gpu",
        )
    )
    assert list(tmp_path.iterdir()) == []


def test_resolve_task_data_facts_uses_epoch_fallback(monkeypatch):
    config = _hardware_config(max_samples=None)
    train_ds = FakeDataset(1000, "train-fp")
    eval_ds = FakeDataset(200, "eval-fp")
    load_task = mock.Mock(return_value=(train_ds, eval_ds, {}))
    monkeypatch.setattr(runner, "get_training_config", lambda profile: config)
    monkeypatch.setattr(runner, "load_glue_task", load_task)
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)

    facts = runner.resolve_task_data_facts(
        "synthetic_task_without_override",
        object(),
        None,
        "cpu",
    )

    assert facts["num_epochs"] == 3
    assert facts["total_steps"] == 96
    assert facts["effective_n_gpu"] == 1


@pytest.mark.parametrize(
    "arm,rate",
    itertools.product(PHASE1_3_CANONICAL_ARMS, RATES),
)
def test_plan_phase1_3_cell_complete_canonical_projection(
    tmp_path,
    arm,
    rate,
):
    base_output_dir = tmp_path / "planned"
    cell = _plan_cell(
        arm,
        rate,
        base_output_dir=base_output_dir,
    )

    assert set(cell) == PLANNED_CELL_REQUIRED_FIELDS
    assert cell["arm"] == cell["control"] == arm
    assert cell["task"] == TASK
    assert cell["training_seed"] == cell["policy_seed"] == SEED
    assert cell["model_id"] == MODEL_ID
    assert cell["target_skip_rate"] == rate
    assert cell["num_epochs"] == 5
    assert cell["total_steps"] == TOTAL_STEPS
    assert cell["min_step"] == runner.POLICY_MIN_STEP
    assert cell["matched_budget"] is True
    assert cell["no_early_stopping"] is True
    assert cell["skip_update_mode"] == "freeze"
    assert cell["scheduler_step_policy"] == "skip_on_backward_skip"
    assert type(cell["target_skip_rate"]) is float
    assert type(cell["is_skipping_arm"]) is bool
    assert type(cell["matched_budget"]) is bool
    assert type(cell["no_early_stopping"]) is bool
    for field in (
        "training_seed",
        "policy_seed",
        "num_epochs",
        "total_steps",
        "min_step",
        "planned_skips",
    ):
        assert type(cell[field]) is int
    assert all(
        type(cell[field]) is dict
        for field in ("online_diagnostics", "controller_config", "identity_inputs")
    )

    controller = cell["controller_config"]
    identity = cell["identity_inputs"]
    assert COMMON_CONTROLLER_FIELDS <= set(controller)
    assert controller["arm_alias_of"] is None
    assert controller["control"] == arm
    assert controller["policy_seed"] == SEED
    assert controller["configured_total_steps"] == TOTAL_STEPS
    assert controller["num_epochs"] == 5
    assert identity["online_diagnostics"] == cell["online_diagnostics"]
    assert controller["online_diagnostics"] == cell["online_diagnostics"]

    skipping = arm != "full_finetune"
    expected_quota = round(rate * TOTAL_STEPS) if skipping else None
    assert cell["is_skipping_arm"] is skipping
    assert cell["requested_quota"] == expected_quota
    assert cell["planned_skips"] == (expected_quota if skipping else 0)
    assert controller["requested_quota"] == expected_quota
    assert controller["compute_saving_mechanism"] == (
        "backward_skipping" if skipping else "none"
    )

    online = cell["online_diagnostics"]
    offline = arm in runner.ONLINE_LER_SIGNAL_FREE_CONTROLS
    assert online["mode"] == ("off" if offline else "sampled_lagged")
    assert online["enabled"] is (not offline)
    assert online["timing"] == (
        "none" if offline else "post_decision_after_backward"
    )
    assert online["sample_seed"] == (None if offline else SEED)

    phase_expected = arm in {"fixed_phase_strat", "phase_strat_guarded"}
    ler_expected = arm in {
        "ler_guided_stratified",
        "ler_guided_stratified_safe",
    }
    assert ("phase_strat_controller" in controller) is phase_expected
    assert ("phase_strat_controller" in identity) is phase_expected
    assert ("ler_guided_controller" in controller) is ler_expected
    assert ("ler_guided_controller" in identity) is ler_expected
    if phase_expected:
        phase = controller["phase_strat_controller"]
        assert phase["control"] == arm
        assert "controller_class" in phase
        assert "policy_class" not in phase
        assert phase["requested_quota"] == expected_quota
    if ler_expected:
        ler = controller["ler_guided_controller"]
        assert ler["control"] == arm
        assert "policy_class" in ler
        assert "controller_class" not in ler
        if arm == "ler_guided_stratified_safe":
            assert LER_SAFETY_FIELDS <= set(ler)
            assert ler["safety_enabled"] is True
        else:
            assert not (LER_SAFETY_FIELDS & set(ler))
            assert ler["safety_enabled"] is False

    assert cell["fingerprint"] == build_scientific_fingerprint(identity)
    assert cell["planned_arm_dir"] == os.path.join(
        base_output_dir,
        arm,
        cell["fingerprint"],
    )
    assert not base_output_dir.exists()
    assert not os.path.exists(cell["planned_arm_dir"])


def test_plan_phase1_3_cell_provenance_copies_are_independent():
    online_cell = _plan_cell("ler_guided_stratified")
    online_cell["online_diagnostics"]["reason"] = "mutated"
    assert online_cell["identity_inputs"]["online_diagnostics"]["reason"] != "mutated"
    assert online_cell["controller_config"]["online_diagnostics"]["reason"] != "mutated"

    phase_cell = _plan_cell("fixed_phase_strat")
    phase_controller = phase_cell["controller_config"]["phase_strat_controller"]
    phase_identity = phase_cell["identity_inputs"]["phase_strat_controller"]
    assert phase_controller is not phase_identity
    phase_controller["phase_weights"][0] = 99.0
    assert phase_identity["phase_weights"][0] != 99.0

    ler_cell = _plan_cell("ler_guided_stratified_safe")
    ler_controller = ler_cell["controller_config"]["ler_guided_controller"]
    ler_identity = ler_cell["identity_inputs"]["ler_guided_controller"]
    assert ler_controller is not ler_identity
    ler_controller["phase_weights"][0] = 99.0
    assert ler_identity["phase_weights"][0] != 99.0


@pytest.mark.parametrize("arm", ["random_skip", "phase_strat", "rvd", "full_lerna"])
def test_plan_phase1_3_cell_rejects_noncanonical_arms(arm):
    with pytest.raises(ValueError, match="Unknown Phase 1.3 arm"):
        _plan_cell(arm)


@pytest.mark.parametrize("rate", [float("nan"), float("inf"), -0.1, 1.1])
def test_plan_phase1_3_cell_rejects_invalid_rates(rate):
    with pytest.raises(ValueError, match="finite and in"):
        _plan_cell("exact_random", rate)


def test_plan_phase1_3_cell_rejects_mismatches_and_infeasible_budget():
    mismatched = _data_facts("other_task")
    with pytest.raises(ValueError, match="does not match"):
        _plan_cell("exact_random", facts=mismatched)
    with pytest.raises(ValueError, match="scheduler_step_policy"):
        _plan_cell(
            "exact_random",
            scheduler_step_policy="always_step",
        )
    with pytest.raises(ValueError, match="greater than POLICY_MIN_STEP"):
        _plan_cell(
            "full_finetune",
            facts=_data_facts(total_steps=runner.POLICY_MIN_STEP),
        )
    with pytest.raises(ValueError, match="never clipped"):
        _plan_cell(
            "exact_random",
            0.90,
            facts=_data_facts(total_steps=60),
        )


def test_build_phase1_3_matrix_plan_complete_valid_no_write(tmp_path):
    tasks = ["task_a", "task_b"]
    seeds = [7, 11]
    rates = [0.30, 0.40]
    facts_by_task = {task: _data_facts(task) for task in tasks}
    facts_before = copy.deepcopy(facts_by_task)
    provider_calls = []

    def provider(task):
        provider_calls.append(task)
        return facts_by_task[task]

    base_output_dir = tmp_path / "planned"
    plan = runner.build_phase1_3_matrix_plan(
        tasks=tasks,
        seeds=seeds,
        target_skip_rates=rates,
        model_name=MODEL_ID,
        base_output_dir=base_output_dir,
        data_facts_provider=provider,
        git_sha="abc123",
        **CELL_KWARGS,
    )

    expected_order = [
        (task, seed, rate, arm)
        for task in tasks
        for seed in seeds
        for rate in rates
        for arm in PHASE1_3_CANONICAL_ARMS
    ]
    assert provider_calls == tasks
    assert facts_by_task == facts_before
    assert len(plan) == len(expected_order) == 48
    assert [
        (
            cell["task"],
            cell["training_seed"],
            cell["target_skip_rate"],
            cell["arm"],
        )
        for cell in plan
    ] == expected_order
    assert all(cell["policy_seed"] == cell["training_seed"] for cell in plan)
    assert sum(cell["arm"] == "full_finetune" for cell in plan) == 8
    assert len({cell["fingerprint"] for cell in plan}) == 48
    assert len({cell["planned_arm_dir"] for cell in plan}) == 48
    assert validate_phase1_3_matrix_plan(
        plan,
        tasks=tasks,
        seeds=seeds,
        target_skip_rates=rates,
        minimum_seed_count=2,
        base_output_dir=base_output_dir,
    ) == []
    assert not base_output_dir.exists()
    assert all(not os.path.exists(cell["planned_arm_dir"]) for cell in plan)
