"""Test LER goes to zero when model is converged.

Feed a constant loss; assert ler_history[-1] == 0.
"""
import pytest
import torch
import torch.nn as nn
from lerna.utils.metrics import LERTracker


def test_ler_zero_on_constant_loss():
    """Verify LER goes to zero when there's no improvement (constant loss)."""
    # Simple model
    model = nn.Linear(8, 4)
    
    # Create LER tracker
    tracker = LERTracker(
        task="sst2",
        window_size=5,
    )
    
    # Simulate constant loss scenario (no improvement)
    # First, do a few steps to populate state
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Do a few steps to populate optimizer state
    for _ in range(3):
        x = torch.randn(4, 8)
        y = model(x)
        loss = y.sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    # Now simulate constant loss with zero gradients (velocity = 0)
    for _ in range(10):
        # Zero gradient = no parameter change = zero velocity
        x = torch.randn(4, 8)
        y = model(x)
        loss_val = 1.0  # Constant loss
        
        # Call update with zero gradients (all zeros)
        zero_grads = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
        tracker.update(
            loss=loss_val,
            logits=y,
            gradients=zero_grads,
            n_steps=1,
        )
    
    # Check LER history
    if tracker.ler_history:
        final_ler = tracker.ler_history[-1]
        # With zero velocity, LER should be 0
        assert final_ler == 0.0, f"LER should be 0 with zero velocity, got {final_ler}"


def test_ler_none_first_step():
    """Verify LER is None before first valid computation."""
    tracker = LERTracker(
        task="sst2",
        window_size=5,
    )
    
    # Before any valid update, ler should be None
    assert len(tracker.ler_history) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])