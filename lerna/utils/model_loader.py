"""Model loading utilities for LERNA experiments.

Handles architecture-specific quirks (ModernBERT compile flags,
attention implementations, etc.) so experiment scripts stay clean.
"""
from __future__ import annotations

from typing import Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer


_MODERNBERT_PREFIXES = ("answerdotai/ModernBERT",)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    problem_type: Optional[str] = None,
    device_map: Optional[str] = None,
) -> Tuple:
    """Load model and tokenizer with architecture-aware defaults.

    ModernBERT: reference_compile=False, attn_implementation="sdpa"
    Others: standard AutoModel loading
    """
    is_modernbert = any(model_name.startswith(p) for p in _MODERNBERT_PREFIXES)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model_kwargs = dict(num_labels=num_labels)
    if problem_type is not None:
        model_kwargs["problem_type"] = problem_type
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    if is_modernbert:
        model_kwargs["reference_compile"] = False
        model_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )

    return model, tokenizer


MODELS = {
    "roberta": "roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta": "microsoft/deberta-v3-base",
}
