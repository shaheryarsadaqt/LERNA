"""Post-load model sanity checks."""

import torch


def assert_layernorm_trained(
    model: torch.nn.Module,
    weight_mean_min: float = 0.5,
    weight_std_max: float = 0.8,
    log_first_n: int = 3,
) -> None:
    """Raise if any LayerNorm.weight looks randomly initialized."""
    checked = 0
    for name, p in model.named_parameters():
        if "LayerNorm.weight" not in name:
            continue
        t = p.detach().float()
        mean = t.mean().item()
        std = t.std().item()
        if checked < log_first_n:
            print(f"  [SANITY] {name}: mean={mean:.4f}, std={std:.4f}")
        if abs(mean) < weight_mean_min or std > weight_std_max:
            raise RuntimeError(
                f"LayerNorm parameter {name} looks randomly initialized "
                f"(mean={mean:.4f}, std={std:.4f}). Likely cause: "
                f"load_best_model_at_end loaded a checkpoint with legacy "
                f"gamma/beta keys."
            )
        checked += 1
    if checked == 0:
        print("  [SANITY] No LayerNorm.weight parameters found (skipped).")