"""Focused tests for deterministic sampled lagged LER diagnostics."""

import inspect

import pytest
import torch
import torch.nn as nn

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
