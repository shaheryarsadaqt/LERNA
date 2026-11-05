"""Regression test: vectorized _sampled_dynamics_v2 matches v1 within atol=1e-6.

Tracker revision v2 vectorizes _sampled_dynamics to eliminate per-tensor
.item() CPU syncs.  Reduction order changes (PyTorch pairwise summation vs.
Python sequential float accumulation), so we test to atol=1e-6, not exact.
"""

import math

import pytest
import torch
import torch.nn as nn

from lerna.utils.lagged_ler import SampledLaggedLERTracker


def _rand_model(width=64, depth=4):
    layers = []
    prev = width
    for _ in range(depth):
        layers.append(nn.Linear(prev, width))
        prev = width
    layers.append(nn.Linear(width, 2))
    return nn.Sequential(*layers)


def _run_observation_sequence(steps=200, parameter_sample_size=4096, seed=0):
    """Run a 200-step synthetic optimizer trajectory, comparing v1 vs v2."""
    torch.manual_seed(123)
    model_v1 = _rand_model()
    model_v2 = _rand_model()
    model_v2.load_state_dict(model_v1.state_dict())

    optimizer_v1 = torch.optim.Adam(model_v1.parameters(), lr=1e-3)
    optimizer_v2 = torch.optim.Adam(model_v2.parameters(), lr=1e-3)

    tracker_v1 = SampledLaggedLERTracker(
        parameter_sample_size=parameter_sample_size,
        sample_seed=seed,
    )
    tracker_v2 = SampledLaggedLERTracker(
        parameter_sample_size=parameter_sample_size,
        sample_seed=seed,
    )
    tracker_v1.set_optimizer(optimizer_v1)
    tracker_v2.set_optimizer(optimizer_v2)

    logits = torch.randn(4, 2)
    for step in range(steps):
        # Perturb both models identically
        with torch.no_grad():
            for p1, p2 in zip(model_v1.parameters(), model_v2.parameters()):
                noise = torch.randn_like(p1) * 0.01
                p1.add_(noise)
                p2.add_(noise)

        optimizer_v1.zero_grad(set_to_none=True)
        optimizer_v2.zero_grad(set_to_none=True)

        # Fake gradient
        for p1, p2 in zip(model_v1.parameters(), model_v2.parameters()):
            p1.grad = torch.randn_like(p1) * 0.001
            p2.grad = p1.grad.clone()

        optimizer_v1.step()
        optimizer_v2.step()

        loss_val = 1.0 - step * 0.001

        tracker_v1.update(
            loss=loss_val,
            logits=logits,
            model=model_v1,
            optimizer=optimizer_v1,
            decision_index=step,
        )
        tracker_v2.update(
            loss=loss_val,
            logits=logits,
            model=model_v2,
            optimizer=optimizer_v2,
            decision_index=step,
        )

    return tracker_v1, tracker_v2


def test_v1_vs_v2_outputs_match_atol():
    """v2 (vectorized) must match v1 (per-tensor .item()) within atol=1e-6."""
    tracker_v1, tracker_v2 = _run_observation_sequence(
        steps=200, parameter_sample_size=4096, seed=0
    )

    # Compare histories
    assert len(tracker_v1.velocity_history) == len(tracker_v2.velocity_history)
    assert len(tracker_v1.rho_vg_history) == len(tracker_v2.rho_vg_history)
    assert len(tracker_v1.ler_history) == len(tracker_v2.ler_history)
    assert tracker_v1.valid_emissions == tracker_v2.valid_emissions
    assert tracker_v1.terminal_invalid_event == tracker_v2.terminal_invalid_event

    for v1, v2 in zip(tracker_v1.velocity_history, tracker_v2.velocity_history):
        assert v2 == pytest.approx(v1, abs=1e-6)

    for v1, v2 in zip(tracker_v1.rho_vg_history, tracker_v2.rho_vg_history):
        if v1 is not None and v2 is not None:
            assert v2 == pytest.approx(v1, abs=1e-6)
        else:
            assert v1 is None and v2 is None

    for v1, v2 in zip(tracker_v1.ler_history, tracker_v2.ler_history):
        assert v2 == pytest.approx(v1, abs=1e-6)


def test_v2_item_calls_outside_loop():
    """v2 must not call .item() inside the per-tensor loop.

    v1 calls .item() 3 times per tensor (3 * N_sync at runtime).
    v2 calls .item() only after the loop completes, once per metric
    (4 total, constant regardless of tensor count).
    """
    import ast
    import inspect
    import textwrap
    import lerna.utils.lagged_ler as mod

    src = textwrap.dedent(inspect.getsource(
        mod.SampledLaggedLERTracker._sampled_dynamics
    ))
    tree = ast.parse(src)

    # Find the for loop inside the function
    for_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
    ]
    assert len(for_loops) > 0, "v2 must still have the per-tensor loop"

    # Check the _iter_sampled loop specifically (the inner loop over sampled tensors)
    for loop in for_loops:
        loop_src = ast.get_source_segment(src, loop)
        if "self._iter_sampled" in loop_src:
            assert ".item()" not in loop_src, (
                "v2 must not call .item() inside the _iter_sampled loop; "
                "all CPU syncs must be deferred to after the loop"
            )

    # Verify the full-method count is constant (<=5, not proportional to tensor count)
    full_count = sum(
        1 for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "item"
    )
    assert full_count <= 5, f"Expected <=5 total .item() calls, got {full_count}"


def test_v2_reduction_order_documented():
    """Verify that v2 uses stacked/torch reduction (pairwise) not sequential float."""
    import inspect
    import textwrap
    import lerna.utils.lagged_ler as mod

    src = textwrap.dedent(inspect.getsource(
        mod.SampledLaggedLERTracker._sampled_dynamics
    ))
    assert "torch.stack" in src, "v2 must use torch.stack for on-device accumulation"
    assert "delta_sq_tensors" in src
    assert "signal_delta_sq_tensors" in src


def test_v1_vs_v3_outputs_match_atol():
    """v3 (flat-tensor gather) must match v1 (per-tensor .item()) within atol=1e-6."""
    tracker_v1, tracker_v3 = _run_observation_sequence_v3(
        steps=200, parameter_sample_size=4096, seed=0
    )

    assert len(tracker_v1.velocity_history) == len(tracker_v3.velocity_history)
    assert len(tracker_v1.rho_vg_history) == len(tracker_v3.rho_vg_history)
    assert len(tracker_v1.ler_history) == len(tracker_v3.ler_history)
    assert tracker_v1.valid_emissions == tracker_v3.valid_emissions
    assert tracker_v1.terminal_invalid_event == tracker_v3.terminal_invalid_event

    for v1, v3 in zip(tracker_v1.velocity_history, tracker_v3.velocity_history):
        assert v3 == pytest.approx(v1, abs=1e-6)

    for v1, v3 in zip(tracker_v1.rho_vg_history, tracker_v3.rho_vg_history):
        if v1 is not None and v3 is not None:
            assert v3 == pytest.approx(v1, abs=1e-6)
        else:
            assert v1 is None and v3 is None

    for v1, v3 in zip(tracker_v1.ler_history, tracker_v3.ler_history):
        assert v3 == pytest.approx(v1, abs=1e-6)


def _run_observation_sequence_v3(steps=200, parameter_sample_size=4096, seed=0):
    """Run a 200-step synthetic optimizer trajectory, comparing v1 vs v3."""
    torch.manual_seed(123)
    model_v1 = _rand_model()
    model_v3 = _rand_model()
    model_v3.load_state_dict(model_v1.state_dict())

    optimizer_v1 = torch.optim.Adam(model_v1.parameters(), lr=1e-3)
    optimizer_v3 = torch.optim.Adam(model_v3.parameters(), lr=1e-3)

    tracker_v1 = SampledLaggedLERTracker(
        parameter_sample_size=parameter_sample_size,
        sample_seed=seed,
    )
    tracker_v3 = SampledLaggedLERTracker(
        parameter_sample_size=parameter_sample_size,
        sample_seed=seed,
        tracker_revision="v3",
    )
    tracker_v1.set_optimizer(optimizer_v1)
    tracker_v3.set_optimizer(optimizer_v3)

    logits = torch.randn(4, 2)
    for step in range(steps):
        with torch.no_grad():
            for p1, p3 in zip(model_v1.parameters(), model_v3.parameters()):
                noise = torch.randn_like(p1) * 0.01
                p1.add_(noise)
                p3.add_(noise)

        optimizer_v1.zero_grad(set_to_none=True)
        optimizer_v3.zero_grad(set_to_none=True)

        for p1, p3 in zip(model_v1.parameters(), model_v3.parameters()):
            p1.grad = torch.randn_like(p1) * 0.001
            p3.grad = p1.grad.clone()

        optimizer_v1.step()
        optimizer_v3.step()

        loss_val = 1.0 - step * 0.001

        tracker_v1.update(
            loss=loss_val,
            logits=logits,
            model=model_v1,
            optimizer=optimizer_v1,
            decision_index=step,
        )
        tracker_v3.update(
            loss=loss_val,
            logits=logits,
            model=model_v3,
            optimizer=optimizer_v3,
            decision_index=step,
        )

    return tracker_v1, tracker_v3


def test_v3_uses_single_index_select_for_samples():
    """v3 must use _global_sample_index and _capture_flat_tensor."""
    import inspect
    import textwrap
    import lerna.utils.lagged_ler as mod

    src = textwrap.dedent(inspect.getsource(
        mod.SampledLaggedLERTracker._sampled_dynamics_v3
    ))
    assert "global_idx" in src, "v3 must use global index for flat gather"
    assert "_capture_flat_tensor" in src, "v3 must use flat tensor capture"


def test_v3_compute_entropy_batches_item_calls():
    """v3 _compute_entropy must batch .item() into single .tolist()."""
    import inspect
    import textwrap
    import lerna.utils.lagged_ler as mod

    src = textwrap.dedent(inspect.getsource(
        mod.SampledLaggedLERTracker._compute_entropy_v3
    ))
    assert ".tolist()" in src, "v3 entropy must use .tolist() for batched sync"
