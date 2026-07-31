"""Focused tests for deterministic sampled lagged LER diagnostics."""

import inspect
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lerna.trainers import AlwaysFalsePolicy, TrueBackwardSkippingTrainer

from lerna.utils.lagged_ler import (
    SAMPLED_LAGGED_MODE,
    SAMPLED_LAGGED_TIMING,
    SampledLaggedLERTracker,
)


class TinyModel(nn.Module):
    def __init__(self, width=8):
        super().__init__()
        self.hidden = nn.Linear(width, width)
        self.classifier = nn.Linear(width, 2)

    def forward(self, inputs):
        return self.classifier(torch.tanh(self.hidden(inputs)))


CANONICAL_KEYS = {
    "mode",
    "timing",
    "parameter_sample_size_requested",
    "parameter_sample_size_realized",
    "sampled_tensor_count",
    "n_updates",
    "n_decisions",
    "last_update_decision",
    "observation_age_decisions",
    "ler",
    "ler_raw",
    "rho_vg",
    "rho_vg_raw",
    "param_velocity",
    "n_velocity_samples",
    "n_rho_vg_samples",
    "full_parameter_snapshots",
    "full_vector_concatenation",
}


def _logits():
    return torch.tensor([[1.0, -1.0], [-0.5, 0.5]])


def test_sampling_is_deterministic_and_uses_one_global_budget():
    model_a = TinyModel()
    model_b = TinyModel()
    model_b.load_state_dict(model_a.state_dict())
    tracker_a = SampledLaggedLERTracker(parameter_sample_size=11, sample_seed=7)
    tracker_b = SampledLaggedLERTracker(parameter_sample_size=11, sample_seed=7)

    tracker_a.update(loss=1.0, logits=_logits(), model=model_a, decision_index=0)
    tracker_b.update(loss=1.0, logits=_logits(), model=model_b, decision_index=0)

    assert [name for name, _ in tracker_a._sample_plan] == [
        name for name, _ in tracker_b._sample_plan
    ]
    for name in tracker_a._sample_index:
        assert torch.equal(tracker_a._sample_index[name], tracker_b._sample_index[name])
        assert tracker_a._sample_index[name].device.type == "cpu"

    diagnostics = tracker_a.get_diagnostics()
    assert diagnostics["parameter_sample_size_realized"] == 11
    assert sum(value.numel() for value in tracker_a._prev_samples.values()) == 11
    assert diagnostics["sampled_tensor_count"] > 1
    assert diagnostics["full_parameter_snapshots"] is False
    assert diagnostics["full_vector_concatenation"] is False


def test_sampled_source_avoids_dense_permutations_and_vector_concatenation():
    source = inspect.getsource(__import__("lerna.utils.lagged_ler", fromlist=["*"]))
    assert "randperm" not in source
    assert "torch.cat" not in source
    assert "state_dict" not in source


def test_velocity_is_rms_and_independent_of_sample_budget():
    measured = []
    for budget in (7, 31):
        model = TinyModel()
        tracker = SampledLaggedLERTracker(
            parameter_sample_size=budget,
            sample_seed=3,
        )
        tracker.update(loss=1.0, logits=_logits(), model=model, decision_index=0)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.01)
        tracker.update(loss=0.8, logits=_logits(), model=model, decision_index=1)
        measured.append(tracker.get_diagnostics()["param_velocity"])

    assert measured[0] == pytest.approx(0.01, rel=1e-5)
    assert measured[1] == pytest.approx(0.01, rel=1e-5)


def test_partial_adam_state_falls_back_per_parameter_to_live_gradient():
    class SplitModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.adam_parameter = nn.Parameter(torch.ones(8))
            self.gradient_parameter = nn.Parameter(torch.ones(8))

    model = SplitModel()
    optimizer = torch.optim.Adam([model.adam_parameter], lr=1e-2)
    model.adam_parameter.grad = torch.ones_like(model.adam_parameter)
    model.gradient_parameter.grad = torch.full_like(model.gradient_parameter, 2.0)
    optimizer.step()

    tracker = SampledLaggedLERTracker(parameter_sample_size=8, sample_seed=2)
    tracker.update(loss=1.0, logits=_logits(), model=model, decision_index=0)
    state_lookup = {
        id(model.adam_parameter): (
            optimizer.param_groups[0],
            optimizer.state[model.adam_parameter],
        )
    }

    adam_indices = tracker._sample_index["adam_parameter"].to(
        model.adam_parameter.device
    )
    gradient_indices = tracker._sample_index["gradient_parameter"].to(
        model.gradient_parameter.device
    )
    adam_signal = tracker._sampled_signal(
        model.adam_parameter,
        adam_indices,
        state_lookup,
    )
    gradient_signal = tracker._sampled_signal(
        model.gradient_parameter,
        gradient_indices,
        state_lookup,
    )

    assert adam_signal is not None
    assert gradient_signal is not None
    expected_gradient = model.gradient_parameter.grad.index_select(
        0,
        gradient_indices,
    ).float()
    assert torch.equal(gradient_signal, expected_gradient)


def test_completed_updates_produce_velocity_rho_and_ler():
    torch.manual_seed(9)
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    tracker = SampledLaggedLERTracker(parameter_sample_size=16, sample_seed=4)
    inputs = torch.randn(4, 8)
    labels = torch.tensor([0, 1, 0, 1])

    optimizer.zero_grad()
    first_logits = model(inputs)
    first_loss = nn.functional.cross_entropy(first_logits, labels)
    first_loss.backward()
    tracker.note_decision(0)
    tracker.update(
        loss=float(first_loss.detach()),
        logits=first_logits.detach(),
        model=model,
        optimizer=optimizer,
        decision_index=0,
    )
    optimizer.step()

    optimizer.zero_grad()
    second_logits = model(inputs)
    second_loss = nn.functional.cross_entropy(second_logits, labels)
    second_loss.backward()
    tracker.note_decision(1)
    tracker.update(
        loss=float(second_loss.detach()),
        logits=second_logits.detach(),
        model=model,
        optimizer=optimizer,
        decision_index=1,
    )

    diagnostics = tracker.get_diagnostics()
    assert diagnostics["param_velocity"] > 0.0
    assert diagnostics["rho_vg_raw"] is not None
    assert diagnostics["ler_raw"] is not None
    assert diagnostics["ler"] is not None
    assert diagnostics["n_updates"] == 2
    assert diagnostics["last_update_decision"] == 1


def test_diagnostics_schema_and_decision_age_are_lagged():
    model = TinyModel()
    tracker = SampledLaggedLERTracker(parameter_sample_size=9)
    tracker.note_decision(4)
    tracker.update(loss=1.0, logits=_logits(), model=model, decision_index=4)
    before_histories = (
        list(tracker.loss_history),
        list(tracker.entropy_history),
        list(tracker.velocity_history),
        list(tracker.rho_vg_history),
        list(tracker.ler_history),
    )

    tracker.note_decision(5)
    tracker.note_decision(6)
    diagnostics = tracker.get_diagnostics()

    assert set(diagnostics) == CANONICAL_KEYS
    assert diagnostics["mode"] == SAMPLED_LAGGED_MODE
    assert diagnostics["timing"] == SAMPLED_LAGGED_TIMING
    assert diagnostics["last_update_decision"] == 4
    assert diagnostics["observation_age_decisions"] == 2
    assert before_histories == (
        tracker.loss_history,
        tracker.entropy_history,
        tracker.velocity_history,
        tracker.rho_vg_history,
        tracker.ler_history,
    )


def test_invalid_tracker_configuration_raises():
    with pytest.raises(ValueError, match="parameter_sample_size"):
        SampledLaggedLERTracker(parameter_sample_size=0)
    with pytest.raises(ValueError, match="window_size"):
        SampledLaggedLERTracker(window_size=1)


class TrainerTinyDataset(Dataset):
    def __init__(self, size=8, width=4):
        generator = torch.Generator().manual_seed(13)
        self.inputs = torch.randn(size, width, generator=generator)
        self.labels = torch.randint(0, 2, (size,), generator=generator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.inputs[index], "labels": self.labels[index]}


class TrainerTinyModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.classifier = nn.Linear(width, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.classifier(input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def _trainer_collate(batch):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
    }


class CountingTracker:
    def __init__(self):
        self.calls = 0

    def update(self, **kwargs):
        self.calls += 1


class ObservingSkipPolicy:
    name = "observing_skip"

    def __init__(self, tracker, skip_decisions=()):
        self.tracker = tracker
        self.skip_decisions = set(skip_decisions)
        self.decisions = 0
        self.seen_update_counts = []
        self.seen_observation_ages = []
        self.seen_histories = []

    def _histories(self):
        return (
            list(self.tracker.loss_history),
            list(self.tracker.entropy_history),
            list(self.tracker.velocity_history),
            list(self.tracker.rho_vg_history),
            list(self.tracker.ler_history),
        )

    def should_skip(self, trainer, model, inputs):
        self.seen_update_counts.append(self.tracker.n_updates)
        self.seen_observation_ages.append(
            self.tracker.get_diagnostics()["observation_age_decisions"]
        )
        self.seen_histories.append(self._histories())
        should_skip = self.decisions in self.skip_decisions
        self.decisions += 1
        return should_skip


class LegacyObservingPolicy:
    name = "legacy_observing"

    def __init__(self, tracker):
        self.tracker = tracker
        self.seen_call_counts = []

    def should_skip(self, trainer, model, inputs):
        self.seen_call_counts.append(self.tracker.calls)
        return False


def _build_trainer(tmpdir, *, policy, tracker, **kwargs):
    args = TrainingArguments(
        output_dir=tmpdir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=True,
        fp16=False,
        bf16=False,
        seed=19,
    )
    trainer = TrueBackwardSkippingTrainer(
        model=TrainerTinyModel(),
        args=args,
        train_dataset=TrainerTinyDataset(),
        data_collator=_trainer_collate,
        skip_policy=policy,
        instrumentation_path=os.path.join(tmpdir, "instrumentation.json"),
        **kwargs,
    )
    trainer._ler_tracker = tracker
    return trainer


def test_sampled_mode_policy_sees_lagged_observations_and_skips_do_not_update():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SampledLaggedLERTracker(parameter_sample_size=8, sample_seed=1)
        policy = ObservingSkipPolicy(tracker, skip_decisions={1})
        trainer = _build_trainer(
            tmpdir,
            policy=policy,
            tracker=tracker,
            online_ler_mode="sampled_lagged",
            online_ler_update_interval=1,
        )
        trainer.train()

        assert policy.seen_update_counts == [0, 1, 1, 2]
        assert policy.seen_observation_ages == [1, 1, 2, 1]
        assert policy.seen_histories[2] == policy.seen_histories[1]

        diagnostics = trainer.get_instrumentation()
        assert tracker.n_updates == diagnostics["backward_calls"] == 3
        assert len(tracker.loss_history) == 3
        assert len(tracker.entropy_history) == 3
        assert diagnostics["online_ler_mode"] == "sampled_lagged"
        assert diagnostics["online_ler_update_timing"] == (
            "post_decision_after_backward"
        )
        assert diagnostics["online_ler_update_attempts"] == 3
        assert diagnostics["online_ler_update_successes"] == 3
        assert diagnostics["online_ler_last_update_decision"] == 3
        assert diagnostics["skipped_backward_steps"] == 1
        assert diagnostics["invariant_forward_eq_backward_plus_skipped"] is True


def test_sampled_interval_uses_zero_based_completed_backward_cadence():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = SampledLaggedLERTracker(parameter_sample_size=8, sample_seed=3)
        policy = ObservingSkipPolicy(tracker)
        trainer = _build_trainer(
            tmpdir,
            policy=policy,
            tracker=tracker,
            online_ler_mode="sampled_lagged",
            online_ler_update_interval=2,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert policy.seen_update_counts == [0, 1, 1, 2]
        assert diagnostics["backward_calls"] == 4
        assert tracker.n_updates == 2
        assert diagnostics["online_ler_update_attempts"] == 2
        assert diagnostics["online_ler_update_successes"] == 2
        assert diagnostics["online_ler_last_update_decision"] == 2


def test_legacy_dense_still_updates_before_each_decision():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CountingTracker()
        policy = LegacyObservingPolicy(tracker)
        trainer = _build_trainer(
            tmpdir,
            policy=policy,
            tracker=tracker,
            online_ler_mode="legacy_dense",
            online_ler_update_interval=1,
        )
        trainer.train()

        assert policy.seen_call_counts == [1, 2, 3, 4]
        diagnostics = trainer.get_instrumentation()
        assert diagnostics["online_ler_mode"] == "legacy_dense"
        assert diagnostics["online_ler_update_timing"] == "pre_decision"
        assert diagnostics["online_ler_update_attempts"] == 4
        assert diagnostics["online_ler_update_successes"] == 4
        assert diagnostics["online_ler_last_update_decision"] == 3
        assert diagnostics["invariant_forward_eq_backward_plus_skipped"] is True


def test_off_mode_and_invalid_mode_contract():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CountingTracker()
        trainer = _build_trainer(
            tmpdir,
            policy=AlwaysFalsePolicy(),
            tracker=tracker,
            online_ler_mode="off",
            online_ler_enabled=True,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert tracker.calls == 0
        assert diagnostics["online_ler_mode"] == "off"
        assert diagnostics["online_ler_update_timing"] == "none"
        assert diagnostics["online_ler_enabled"] is False
        assert diagnostics["online_ler_update_interval"] == 0
        assert diagnostics["online_ler_update_attempts"] == 0
        assert diagnostics["online_ler_update_successes"] == 0
        assert diagnostics["online_ler_last_update_decision"] is None
        assert diagnostics["capture_logits_enabled"] is False

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="online_ler_mode"):
            _build_trainer(
                tmpdir,
                policy=AlwaysFalsePolicy(),
                tracker=None,
                online_ler_mode="bogus",
            )
