"""Checkpoint compatibility utilities.

Handles legacy state_dict naming that arises when checkpoints are saved
with one transformers version and loaded with another.
"""

from typing import Dict, List, Tuple
import torch


_LAYERNORM_RENAMES = [
    ("LayerNorm.gamma", "LayerNorm.weight"),
    ("LayerNorm.beta",  "LayerNorm.bias"),
]


def remap_legacy_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename legacy LayerNorm keys (gamma/beta) to modern (weight/bias)."""
    out = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in _LAYERNORM_RENAMES:
            new_k = new_k.replace(old, new)
        out[new_k] = v
    return out


def safe_load_state_dict(
    model: torch.nn.Module,
    state_dict: Dict[str, torch.Tensor],
    strict: bool = False,
    forbid_missing_substrings: Tuple[str, ...] = ("LayerNorm",),
) -> Tuple[List[str], List[str]]:
    """Load with legacy-key remap and assert no critical tensors went missing."""
    state_dict = remap_legacy_keys(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)

    forbidden_missing = [
        k for k in missing if any(s in k for s in forbid_missing_substrings)
    ]
    if forbidden_missing:
        raise RuntimeError(
            f"Critical tensors missing after load: {forbidden_missing[:5]}\n"
            f"Total missing={len(missing)}, unexpected={len(unexpected)}."
        )
    return missing, unexpected