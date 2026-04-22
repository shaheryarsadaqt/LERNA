"""Test per-group learning rate in momentum extrapolation.

Two param groups with different lr; assert each group moved by its own lr.
"""
import pytest
import torch
import torch.nn as nn
from lerna.utils.momentum_extrapolation import apply_momentum_extrapolation


def test_per_group_lr():
    """Verify per-parameter-group learning rates are respected."""
    # Model with two layers
    layer1 = nn.Linear(8, 16)
    layer2 = nn.Linear(16, 4)
    
    # Create param groups with different learning rates
    param_groups = [
        {'params': [layer1.weight], 'lr': 0.1},
        {'params': [layer2.weight], 'lr': 0.01},
    ]
    
    optimizer = torch.optim.SGD(param_groups, momentum=0.9)
    
    # Forward/backward to populate momentum buffer
    x = torch.randn(4, 8)
    h = layer1(x)
    y = layer2(h)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    # Save parameters before extrapolation
    w1_before = layer1.weight.data.clone()
    w2_before = layer2.weight.data.clone()
    
    # Get momentum buffers
    m1 = optimizer.state[layer1.weight]['momentum_buffer'].clone()
    m2 = optimizer.state[layer2.weight]['momentum_buffer'].clone()
    
    # Apply momentum extrapolation
    apply_momentum_extrapolation(optimizer)
    
    # Verify: layer1 moved by lr=0.1, layer2 by lr=0.01
    w1_after = layer1.weight.data
    w2_after = layer2.weight.data
    
    delta1 = (w1_before - w1_after).abs()
    delta2 = (w2_before - w2_after).abs()
    
    # Layer 1 should have moved approximately 0.1 * momentum
    expected_delta1 = 0.1 * m1.abs()
    expected_delta2 = 0.01 * m2.abs()
    
    # Check approximately
    assert torch.allclose(delta1, expected_delta1, rtol=1e-5), \
        f"Layer 1 LR not applied correctly: delta1={delta1.mean().item()}, expected={expected_delta1.mean().item()}"
    assert torch.allclose(delta2, expected_delta2, rtol=1e-5), \
        f"Layer 2 LR not applied correctly: delta2={delta2.mean().item()}, expected={expected_delta2.mean().item()}"


def test_per_group_lr_adam():
    """Verify per-parameter-group learning rates work for Adam."""
    layer1 = nn.Linear(8, 16)
    layer2 = nn.Linear(16, 4)
    
    param_groups = [
        {'params': [layer1.weight], 'lr': 0.1, 'betas': (0.9, 0.999), 'eps': 1e-8},
        {'params': [layer2.weight], 'lr': 0.01, 'betas': (0.9, 0.999), 'eps': 1e-8},
    ]
    
    optimizer = torch.optim.Adam(param_groups)
    
    # Forward/backward to populate state
    x = torch.randn(4, 8)
    h = layer1(x)
    y = layer2(h)
    loss = y.sum()
    loss.backward()
    optimizer.step()
    
    # Save parameters before extrapolation
    w1_before = layer1.weight.data.clone()
    w2_before = layer2.weight.data.clone()
    
    # Apply momentum extrapolation
    apply_momentum_extrapolation(optimizer, adam_use_second_moment=True)
    
    # Both should have moved (not zero)
    delta1 = (w1_before - layer1.weight.data).abs().max().item()
    delta2 = (w2_before - layer2.weight.data).abs().max().item()
    
    assert delta1 > 0, "Layer 1 should have moved"
    assert delta2 > 0, "Layer 2 should have moved"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])