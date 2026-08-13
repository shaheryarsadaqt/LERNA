"""Model loading utilities for LERNA experiments.

Handles architecture-specific quirks (ModernBERT/Ettin compile flags,
attention implementations, etc.) so experiment scripts stay clean.
"""
from __future__ import annotations

from typing import Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer


_MODERNBERT_PREFIXES = ("answerdotai/ModernBERT",)
_ETTIN_PREFIXES = ("jhu-clsp/ettin-encoder-150m",)
ETTIN_MODEL_ID = "jhu-clsp/ettin-encoder-150m"
ETTIN_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"


def load_tokenizer(model_name: str, *, revision: Optional[str] = None, local_files_only: bool = False):
    """Load tokenizer assets without model weights for CPU-side planning."""
    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision
    if local_files_only:
        kwargs["local_files_only"] = local_files_only
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


def validate_ettin_revision(model_name: str, model_revision: Optional[str]) -> Optional[str]:
    """Validate and return the immutable Ettin revision for production runs.

    Returns the validated revision (or None for non-Ettin models). Raises
    ValueError for missing, uppercase, malformed, non-40-character, or
    incorrect revisions so production Ettin runs can never bind to the
    wrong scientific base.
    """
    if model_name != ETTIN_MODEL_ID:
        return None
    if not model_revision:
        raise ValueError(
            f"Ettin model requires an explicit revision; "
            f"expected {ETTIN_REVISION!r}"
        )
    if model_revision != model_revision.lower():
        raise ValueError(
            f"Ettin revision must be lowercase SHA; got {model_revision!r}"
        )
    if len(model_revision) != 40 or any(
        ch not in "0123456789abcdef" for ch in model_revision
    ):
        raise ValueError(
            f"Ettin revision must be a 40-character lowercase hex SHA; "
            f"got {model_revision!r}"
        )
    if model_revision != ETTIN_REVISION:
        raise ValueError(
            f"Ettin revision mismatch: expected {ETTIN_REVISION!r}, "
            f"got {model_revision!r}"
        )
    return model_revision


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int = 2,
    problem_type: Optional[str] = None,
    device_map: Optional[str] = None,
    revision: Optional[str] = None,
    local_files_only: bool = False,
) -> Tuple:
    """Load model and tokenizer with architecture-aware defaults.

    ModernBERT/Ettin: reference_compile=False, attn_implementation="sdpa"
    Others: standard AutoModel loading
    """
    is_modernbert = any(model_name.startswith(p) for p in _MODERNBERT_PREFIXES)
    is_ettin = any(model_name.startswith(p) for p in _ETTIN_PREFIXES)

    tokenizer = load_tokenizer(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )

    model_kwargs = dict(num_labels=num_labels)
    if problem_type is not None:
        model_kwargs["problem_type"] = problem_type
    if device_map is not None:
        model_kwargs["device_map"] = device_map
    if is_modernbert or is_ettin:
        model_kwargs["reference_compile"] = False
        model_kwargs["attn_implementation"] = "sdpa"
    if is_ettin:
        if revision is not None:
            model_kwargs["revision"] = revision
        if local_files_only:
            model_kwargs["local_files_only"] = local_files_only

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, **model_kwargs
    )

    return model, tokenizer


MODELS = {
    "roberta": "roberta-base",
    "modernbert": "answerdotai/ModernBERT-base",
    "deberta": "microsoft/deberta-v3-base",
    "ettin": ETTIN_MODEL_ID,
}