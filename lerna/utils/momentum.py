"""Single source of truth for gradient-free momentum extrapolation."""
from __future__ import annotations
import torch
from typing import Any

__all__ = ["apply_momentum_extrapolation"]


def apply_momentum_extrapolation(optimizer: torch.optim.Optimizer) -> int:
    """Apply θ ← θ − lr · m̂ using the optimizer's own state.

    Semantics:
        * SGD  : uses ``momentum_buffer`` directly (already the update direction).
        * Adam : uses ``exp_avg`` with the bias-corrected first moment
                 m̂ = exp_avg / (1 - β₁^t). We do NOT add the v̂ scaling —
                 this is an intentional coarse extrapolation; see paper §X.
        * Per-parameter-group learning rate is respected (layer-wise LR,
          LLRD, LoRA/base split, etc.).

    Returns the number of parameters updated.
    """
    n = 0
    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = group["lr"]
            beta1 = group.get("betas", (0.9, 0.999))[0]
            for p in group["params"]:
                if not p.requires_grad:
                    continue
                state = optimizer.state.get(p)
                if not state:
                    continue
                if "momentum_buffer" in state and state["momentum_buffer"] is not None:
                    p.data.add_(state["momentum_buffer"], alpha=-lr)
                    n += 1
                elif "exp_avg" in state:
                    step = int(state.get("step", 1))
                    bc = 1.0 - beta1 ** max(step, 1)
                    p.data.add_(state["exp_avg"], alpha=-lr / bc)
                    n += 1
    return n