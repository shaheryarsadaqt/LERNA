"""Single source of truth for LERNA momentum extrapolation.

Used by:
  - callbacks.lerna_switching.LERNASwitchingCallback.on_step_end
  - callbacks.simple_baselines.*
  - scripts.run_phase1_2_simple_baselines
  - LERNATrainer / Phase12Trainer training_step

Applies the correct update for each optimizer family:

SGD:   θ ← θ − lr · momentum_buffer
Adam:  θ ← θ − lr · bias_corrected(m_t) / (sqrt(bias_corrected(v_t)) + eps)

If you use the SGD form on Adam state you get SGD-with-momentum, not Adam.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable
import torch


def apply_momentum_extrapolation(
    optimizer: torch.optim.Optimizer,
    *,
    eps: float = 1e-8,
    adam_use_second_moment: bool = True,
) -> Dict[str, int]:
    """Apply a momentum-only parameter update without requiring fresh gradients.

    Respects per-param-group learning rates (fixes the per-group LR bug).
    Applies proper Adam update including v_t scaling and bias correction.
    """
    stats = {"sgd_updated": 0, "adam_updated": 0, "skipped": 0}

    with torch.no_grad():
        for group in optimizer.param_groups:
            lr = group.get("lr", 0.0)
            if lr == 0.0:
                continue
            # Adam hyperparameters live on the group, not globally.
            betas = group.get("betas", (0.9, 0.999))
            beta1, beta2 = betas
            group_eps = group.get("eps", eps)

            for param in group["params"]:
                if not param.requires_grad:
                    stats["skipped"] += 1
                    continue
                p_state = optimizer.state.get(param, None)
                if not p_state:
                    stats["skipped"] += 1
                    continue

                # --- SGD branch ---
                if "momentum_buffer" in p_state and "exp_avg" not in p_state:
                    param.data.add_(p_state["momentum_buffer"], alpha=-lr)
                    stats["sgd_updated"] += 1
                    continue

                # --- Adam / AdamW branch ---
                if "exp_avg" in p_state:
                    step = int(p_state.get("step", 0))
                    if step <= 0:
                        stats["skipped"] += 1
                        continue

                    # Torch stores 'step' as a 0-d tensor on newer versions
                    if torch.is_tensor(p_state["step"]):
                        step = int(p_state["step"].item())

                    exp_avg = p_state["exp_avg"]
                    bc1 = 1.0 - beta1 ** step
                    m_hat = exp_avg / bc1

                    if adam_use_second_moment and "exp_avg_sq" in p_state:
                        exp_avg_sq = p_state["exp_avg_sq"]
                        bc2 = 1.0 - beta2 ** step
                        v_hat = exp_avg_sq / bc2
                        denom = v_hat.sqrt().add_(group_eps)
                        param.data.addcdiv_(m_hat, denom, value=-lr)
                    else:
                        # Degenerate 'momentum-proxy' update (kept for ablation only).
                        param.data.add_(m_hat, alpha=-lr)

                    stats["adam_updated"] += 1
                    continue

                stats["skipped"] += 1

    return stats