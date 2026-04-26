"""Post-load model sanity checks."""
from typing import Optional
import torch
from transformers import AutoModel


def assert_layernorm_trained(
    model: torch.nn.Module,
    base_model_name: str,
    abs_tol: float = 0.15,
    log_first_n: int = 3,
) -> None:
    """Verify LayerNorm.weight tensors are close to their pretrained reference.

    Catches the gamma/beta key corruption that occurs when transformers/PyTorch
    version mismatches cause LayerNorm parameters to silently mis-load.

    Args:
        model: the (post-load) model to validate
        base_model_name: HF model id used to initialize `model` (e.g. "roberta-base")
        abs_tol: max allowed absolute deviation in mean from reference
        log_first_n: how many passing checks to print
    """
    ref_sd = {
        k: v for k, v in AutoModel.from_pretrained(base_model_name).state_dict().items()
        if "LayerNorm.weight" in k
    }
    cur_sd = dict(model.named_parameters())

    logged = 0
    for name, p in cur_sd.items():
        if "LayerNorm.weight" not in name:
            continue
        # Strip task-head prefix (e.g. "roberta." in RobertaForSequenceClassification)
        key = name.split(".", 1)[1] if "." in name and name.split(".", 1)[0] not in ref_sd else name
        if key not in ref_sd:
            continue
        cur_mean = p.detach().float().mean().item()
        ref_mean = ref_sd[key].mean().item()
        delta = abs(cur_mean - ref_mean)
        if delta > abs_tol:
            raise RuntimeError(
                f"LayerNorm sanity check failed: {name} mean={cur_mean:.4f} "
                f"deviates from pretrained reference {ref_mean:.4f} by {delta:.4f} "
                f"(tol={abs_tol}). Likely gamma/beta key corruption from "
                f"safetensors save/load mismatch."
            )
        if logged < log_first_n:
            print(f"  [SANITY] {name}: mean={cur_mean:.4f} (ref {ref_mean:.4f}, Δ={delta:.4f}) ✓")
            logged += 1

    if logged == 0:
        raise RuntimeError("No LayerNorm.weight parameters found — model architecture unexpected.")