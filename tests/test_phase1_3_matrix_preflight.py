"""Phase 1.3 plan construction and strict preflight integration tests."""

import copy
import itertools
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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
        model_revision=None,
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
        model_revision=None,
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


STRICT_SEEDS = tuple(range(10))
STRICT_RATE_ARGS = ("--target-skip-rates", "0.30", "0.40")


def _strict_argv(*extra, output_dir=None, tasks=("synthetic",)):
    argv = [
        "run_ablation_study.py",
        "--mode",
        "phase1_3",
        "--tasks",
        *tasks,
        "--seeds",
        *(str(seed) for seed in STRICT_SEEDS),
        *STRICT_RATE_ARGS,
    ]
    if output_dir is not None:
        argv.extend(("--output-dir", str(output_dir)))
    argv.extend(extra)
    return argv


def _strict_data_facts(task):
    facts = _data_facts(task=task, total_steps=200, num_epochs=3)
    facts.update(
        max_samples_requested=2000,
        max_samples_effective=2000,
        train_dataset_fingerprint=f"{task}-strict-train",
        eval_dataset_fingerprint=f"{task}-strict-eval",
    )
    return facts


def test_parser_rate_arguments_are_structurally_mutually_exclusive():
    parser = runner.build_arg_parser()
    defaults = parser.parse_args([])
    assert defaults.target_skip_rate is None
    assert defaults.target_skip_rates is None
    assert defaults.scheduler_step_policy == "always_step"

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--target-skip-rate",
                "0.20",
                "--target-skip-rates",
                "0.30",
                "0.40",
            ]
        )


@pytest.mark.parametrize(
    "argv,error_text",
    [
        (
            ["run_ablation_study.py", "--mode", "phase1_3", "--seeds",
             *(str(seed) for seed in STRICT_SEEDS)],
            "requires --target-skip-rates 0.30 0.40",
        ),
        (
            ["run_ablation_study.py", "--mode", "phase1_3", "--seeds",
             *(str(seed) for seed in STRICT_SEEDS), "--target-skip-rate", "0.30"],
            "requires --target-skip-rates",
        ),
        (
            ["run_ablation_study.py", "--mode", "phase1_3", "--seeds",
             *(str(seed) for seed in STRICT_SEEDS), "--target-skip-rates",
             "0.40", "0.30"],
            "exact order",
        ),
        (
            ["run_ablation_study.py", "--mode", "phase1_3",
             *STRICT_RATE_ARGS],
            "requires explicit --seeds",
        ),
        (
            ["run_ablation_study.py", "--mode", "phase1_3", "--seeds",
             *(str(seed) for seed in range(9)), *STRICT_RATE_ARGS],
            "at least 10 unique seeds",
        ),
        (
            ["run_ablation_study.py", "--mode", "phase1_3", "--seeds",
             "0", "1", "2", "3", "4", "5", "6", "7", "8", "8",
             *STRICT_RATE_ARGS],
            "duplicate seeds",
        ),
        (_strict_argv("--tasks", "synthetic", "synthetic"), "duplicate tasks"),
        (_strict_argv("--ablations", "exact_random"), "rejects --ablations"),
        (
            _strict_argv("--allow-early-stopping-with-skipping"),
            "forbids early-stopping overrides",
        ),
        (
            _strict_argv("--skip-update-mode", "momentum"),
            "requires --skip-update-mode freeze",
        ),
        (
            _strict_argv("--policy", "calibrated"),
            "rejects nondefault legacy --policy",
        ),
        (
            _strict_argv("--rvd-policy-seed", "99"),
            "rejects --rvd-policy-seed",
        ),
        (
            _strict_argv("--online-ler-mode", "off"),
            "requires --online-ler-mode auto",
        ),
        (
            _strict_argv(
                "--provenance-classification", "local_development"
            ),
            "requires matched_claim provenance",
        ),
    ],
)
def test_strict_cli_rejects_invalid_matrix_configuration(
    monkeypatch,
    capsys,
    argv,
    error_text,
):
    profile = mock.Mock(side_effect=AssertionError("profile must not resolve"))
    monkeypatch.setattr(runner, "detect_device_profile", profile)
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        runner.main()

    assert error_text in capsys.readouterr().err
    profile.assert_not_called()


def test_target_rate_list_is_rejected_by_legacy_modes(monkeypatch, capsys):
    profile = mock.Mock(side_effect=AssertionError("profile must not resolve"))
    monkeypatch.setattr(runner, "detect_device_profile", profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ablation_study.py",
            "--mode",
            "smoke",
            "--target-skip-rates",
            "0.30",
            "0.40",
        ],
    )

    with pytest.raises(SystemExit):
        runner.main()

    assert "only valid with --mode phase1_3" in capsys.readouterr().err
    profile.assert_not_called()


def test_strict_main_validates_complete_plan_before_any_run(
    monkeypatch,
    tmp_path,
):
    output_dir = tmp_path / "strict-output"
    tasks = ("synthetic_a", "synthetic_b")
    events = []
    tokenizer = object()
    load_tokenizer = mock.Mock(
        side_effect=lambda model_name: events.append("load_tokenizer") or tokenizer
    )
    model_loader = mock.Mock(
        side_effect=AssertionError("model weights must not load in preflight")
    )
    provider_calls = []

    def resolve_facts(task, observed_tokenizer, max_samples, profile):
        assert observed_tokenizer is tokenizer
        assert max_samples == 2000
        assert profile == "cpu"
        provider_calls.append(task)
        events.append(f"facts:{task}")
        return _strict_data_facts(task)

    def validate(plan, **kwargs):
        assert not output_dir.exists()
        events.append("validate")
        assert kwargs == {
            "tasks": list(tasks),
            "seeds": list(STRICT_SEEDS),
            "target_skip_rates": [0.30, 0.40],
            "minimum_seed_count": 10,
            "base_output_dir": str(output_dir),
        }
        return validate_phase1_3_matrix_plan(plan, **kwargs)

    run_calls = []

    def run_cell(**kwargs):
        assert events[-1] in {"validate", "wandb_finish", "run"}
        assert "validate" in events
        assert not output_dir.exists()
        events.append("run")
        run_calls.append(kwargs)
        return {"ablation": kwargs["ablation_name"]}

    finish_wandb = mock.Mock(
        side_effect=lambda: events.append("wandb_finish")
    )
    monkeypatch.setattr(runner, "detect_device_profile", lambda: "cpu")
    monkeypatch.setattr(runner, "_resolve_git_sha", lambda: "abc123")
    monkeypatch.setattr(runner, "resolve_task_data_facts", resolve_facts)
    monkeypatch.setattr(runner, "validate_phase1_3_matrix_plan", validate)
    monkeypatch.setattr(runner, "run_ablation_single", run_cell)
    monkeypatch.setattr(runner, "_ensure_wandb_finished", finish_wandb)
    monkeypatch.setattr(
        "lerna.utils.model_loader.load_tokenizer", load_tokenizer
    )
    monkeypatch.setattr(
        "lerna.utils.model_loader.load_model_and_tokenizer", model_loader
    )
    monkeypatch.setattr(
        sys,
        "argv",
        _strict_argv(
            "--wandb",
            output_dir=output_dir,
            tasks=tasks,
        ),
    )

    runner.main()

    assert events[:4] == [
        "load_tokenizer",
        "facts:synthetic_a",
        "facts:synthetic_b",
        "validate",
    ]
    assert events[4] == "wandb_finish"
    assert provider_calls == list(tasks)
    load_tokenizer.assert_called_once()
    model_loader.assert_not_called()
    assert len(run_calls) == 240
    expected_order = [
        (task, seed, rate, arm)
        for task in tasks
        for seed in STRICT_SEEDS
        for rate in RATES
        for arm in PHASE1_3_CANONICAL_ARMS
    ]
    assert [
        (
            call["task_name"],
            call["seed"],
            call["target_skip_rate"],
            call["ablation_name"],
        )
        for call in run_calls
    ] == expected_order
    assert all(call["planned_cell"] is not None for call in run_calls)
    assert all(call["no_early_stopping"] is True for call in run_calls)
    assert all(call["skip_update_mode"] == "freeze" for call in run_calls)
    assert all(
        call["scheduler_step_policy"] == "skip_on_backward_skip"
        for call in run_calls
    )
    assert all(
        call["allow_early_stopping_with_skipping"] is False
        for call in run_calls
    )
    assert output_dir.joinpath("ablation_summary.json").is_file()


@pytest.mark.parametrize(
    "argv,expected_rate,expected_calls",
    [
        (
            ["run_ablation_study.py", "--mode", "smoke"],
            0.20,
            len(PHASE1_3_CANONICAL_ARMS),
        ),
        (
            [
                "run_ablation_study.py",
                "--mode",
                "custom",
                "--tasks",
                "sst2",
                "--seeds",
                "9",
                "--ablations",
                "exact_random",
                "--target-skip-rate",
                "0.35",
            ],
            0.35,
            1,
        ),
    ],
)
def test_legacy_main_preserves_scalar_workflows(
    monkeypatch,
    tmp_path,
    argv,
    expected_rate,
    expected_calls,
):
    output_dir = tmp_path / "legacy-output"
    argv = [*argv, "--output-dir", str(output_dir)]
    load_tokenizer = mock.Mock(
        side_effect=AssertionError("legacy main must not preflight")
    )
    run_calls = []

    def run_cell(**kwargs):
        run_calls.append(kwargs)
        return {"ablation": kwargs["ablation_name"]}

    monkeypatch.setattr(runner, "detect_device_profile", lambda: "cpu")
    monkeypatch.setattr(runner, "run_ablation_single", run_cell)
    monkeypatch.setattr(
        "lerna.utils.model_loader.load_tokenizer", load_tokenizer
    )
    monkeypatch.setattr(sys, "argv", argv)

    runner.main()

    load_tokenizer.assert_not_called()
    assert len(run_calls) == expected_calls
    assert all(call["target_skip_rate"] == expected_rate for call in run_calls)
    assert all(call["planned_cell"] is None for call in run_calls)
    assert all(call["no_early_stopping"] is False for call in run_calls)
    assert all(call["skip_update_mode"] is None for call in run_calls)
    assert all(
        call["scheduler_step_policy"] == "always_step"
        for call in run_calls
    )


def _set_nested(mapping, path, value):
    target = mapping
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value


@pytest.mark.parametrize(
    "path,value,error_path",
    [
        (("training_seed",), float(SEED), "training_seed"),
        (("total_steps",), TOTAL_STEPS + 1, "total_steps"),
        (("requested_quota",), 1, "requested_quota"),
        (("fingerprint",), "0" * 16, "fingerprint"),
        (("planned_arm_dir",), "wrong", "planned_arm_dir"),
        (("identity_inputs", "task"), "other", "identity_inputs.task"),
        (("online_diagnostics", "mode"), "off", "online_diagnostics.mode"),
        (
            ("controller_config", "policy_seed"),
            SEED + 1,
            "controller_config.policy_seed",
        ),
    ],
)
def test_runtime_match_rejects_every_shared_projection_drift(
    path,
    value,
    error_path,
):
    planned = _plan_cell("ler_guided_stratified_safe")
    runtime = copy.deepcopy(planned)
    runtime["controller_config"]["policy_effective_config"] = {
        "runtime_only": True
    }
    _set_nested(runtime, path, value)

    with pytest.raises(ValueError, match=error_path):
        runner.assert_phase1_3_runtime_matches_plan(planned, runtime)


def test_runtime_match_accepts_only_runtime_controller_extensions():
    planned = _plan_cell("ler_guided_stratified_safe")
    runtime = copy.deepcopy(planned)
    runtime["controller_config"]["policy_effective_config"] = {
        "runtime_only": True
    }
    runner.assert_phase1_3_runtime_matches_plan(planned, runtime)

    runtime = copy.deepcopy(planned)
    runtime["identity_inputs"]["runtime_only"] = True
    with pytest.raises(ValueError, match="unexpected runtime field"):
        runner.assert_phase1_3_runtime_matches_plan(planned, runtime)


class _RuntimeGatePassed(Exception):
    pass


class _FakeRuntimeModel:
    class Config:
        use_cache = True

    config = Config()


def _runtime_facts(task):
    return {
        "task": task,
        "num_epochs": 3,
        "max_samples_requested": 2000,
        "max_samples_effective": 2000,
        "train_samples_realized": 1000,
        "eval_samples_realized": 1000,
        "train_dataset_fingerprint": "runtime-dataset-fp",
        "eval_dataset_fingerprint": "runtime-dataset-fp",
        "total_steps": 96,
        "per_device_train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "effective_n_gpu": 1,
    }


def _install_runtime_gate_fakes(monkeypatch, task):
    hw_config = _hardware_config(max_samples=2000)
    hw_config.update(fp16=False, bf16=False)
    dataset = FakeDataset(1000, "runtime-dataset-fp")
    load_model = mock.Mock(return_value=(_FakeRuntimeModel(), object()))
    mkdir = mock.Mock()
    manifest = mock.Mock()
    power = mock.Mock()
    trainer = mock.Mock()
    finish_wandb = mock.Mock()
    wandb = SimpleNamespace(init=mock.Mock(), Settings=mock.Mock())

    monkeypatch.setitem(runner.GLUE_TASK_CONFIG, task, {"num_labels": 2})
    monkeypatch.setattr(
        runner, "get_training_config", lambda profile: dict(hw_config)
    )
    monkeypatch.setattr(
        runner,
        "load_glue_task",
        lambda *args, **kwargs: (dataset, dataset, {}),
    )
    monkeypatch.setattr(runner, "_resolve_git_sha", lambda: "abc123")
    monkeypatch.setattr(runner.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        "lerna.utils.model_loader.load_model_and_tokenizer", load_model
    )
    monkeypatch.setattr(runner.os, "makedirs", mkdir)
    monkeypatch.setattr(runner, "write_manifest_running", manifest)
    monkeypatch.setattr(runner, "PowerTelemetryCallback", power)
    monkeypatch.setattr(runner, "AblationTrainer", trainer)
    monkeypatch.setattr(runner, "_ensure_wandb_finished", finish_wandb)
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    return {
        "load_model": load_model,
        "mkdir": mkdir,
        "manifest": manifest,
        "power": power,
        "trainer": trainer,
        "finish_wandb": finish_wandb,
        "wandb": wandb,
    }


@pytest.mark.parametrize("arm", PHASE1_3_CANONICAL_ARMS)
def test_all_arms_match_runtime_before_any_side_effect(
    monkeypatch,
    tmp_path,
    arm,
):
    task = "synthetic_runtime"
    output_dir = tmp_path / "runtime-output"
    planned = _plan_cell(
        arm,
        task=task,
        seed=17,
        facts=_runtime_facts(task),
        base_output_dir=str(output_dir),
        online_ler_parameter_sample_size=4096,
        online_ler_update_interval=1,
    )
    side_effects = _install_runtime_gate_fakes(monkeypatch, task)
    real_assert = runner.assert_phase1_3_runtime_matches_plan

    def stop_after_valid_match(planned_cell, runtime_cell):
        real_assert(planned_cell, runtime_cell)
        raise _RuntimeGatePassed

    monkeypatch.setattr(
        runner,
        "assert_phase1_3_runtime_matches_plan",
        stop_after_valid_match,
    )

    with pytest.raises(_RuntimeGatePassed):
        runner.run_ablation_single(
            task_name=task,
            seed=17,
            ablation_name=arm,
            ablation_overrides={"control": arm},
            model_name=MODEL_ID,
            profile="cpu",
            base_output_dir=str(output_dir),
            use_wandb=True,
            max_samples_override=2000,
            no_early_stopping=True,
            target_skip_rate=RATES[0],
            skip_update_mode="freeze",
            scheduler_step_policy="skip_on_backward_skip",
            online_ler_mode="auto",
            planned_cell=planned,
        )

    side_effects["load_model"].assert_called_once()
    for name in (
        "mkdir",
        "manifest",
        "power",
        "trainer",
        "finish_wandb",
    ):
        side_effects[name].assert_not_called()
    side_effects["wandb"].init.assert_not_called()
    assert not output_dir.exists()


def test_runtime_fingerprint_mismatch_aborts_before_side_effects(
    monkeypatch,
    tmp_path,
):
    task = "synthetic_runtime"
    output_dir = tmp_path / "runtime-output"
    planned = _plan_cell(
        "exact_random",
        task=task,
        seed=17,
        facts=_runtime_facts(task),
        base_output_dir=str(output_dir),
        online_ler_parameter_sample_size=4096,
        online_ler_update_interval=1,
    )
    planned["fingerprint"] = "0" * 16
    side_effects = _install_runtime_gate_fakes(monkeypatch, task)

    with pytest.raises(ValueError, match="fingerprint"):
        runner.run_ablation_single(
            task_name=task,
            seed=17,
            ablation_name="exact_random",
            ablation_overrides={"control": "exact_random"},
            model_name=MODEL_ID,
            profile="cpu",
            base_output_dir=str(output_dir),
            use_wandb=True,
            max_samples_override=2000,
            no_early_stopping=True,
            target_skip_rate=RATES[0],
            skip_update_mode="freeze",
            scheduler_step_policy="skip_on_backward_skip",
            online_ler_mode="auto",
            planned_cell=planned,
        )

    for name in (
        "mkdir",
        "manifest",
        "power",
        "trainer",
        "finish_wandb",
    ):
        side_effects[name].assert_not_called()
    side_effects["wandb"].init.assert_not_called()
    assert not output_dir.exists()


def test_source_order_freezes_preflight_and_runtime_boundaries():
    source = Path(runner.__file__).read_text(encoding="utf-8")
    main_start = source.index("def main():")
    main_source = source[main_start:]
    assert main_source.index("load_tokenizer(model_name)") < main_source.index(
        "validate_phase1_3_matrix_plan("
    ) < main_source.index("run_ablation_single(")

    run_start = source.index("def run_ablation_single(")
    run_end = source.index("\ndef build_arg_parser()", run_start)
    run_source = source[run_start:run_end]
    gate = run_source.index("assert_phase1_3_runtime_matches_plan(")
    wandb_init = run_source.index("wandb.init(")
    allocation = run_source.index("os.makedirs(arm_dir")
    power_callback = run_source.index("PowerTelemetryCallback(")
    trainer = run_source.index("trainer = AblationTrainer(")
    manifest = run_source.index("write_manifest_running(")
    assert gate < wandb_init < allocation < power_callback < trainer < manifest
