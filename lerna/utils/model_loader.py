"""Model loading utilities for LERNA experiments.

Handles architecture-specific quirks (ModernBERT/Ettin compile flags,
attention implementations, etc.) so experiment scripts stay clean.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Optional, Tuple

from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from .ettin_constants import ETTIN_MODEL_ID, ETTIN_REVISION, validate_ettin_revision
except ImportError:
    _ETTIN_SPEC = importlib.util.spec_from_file_location(
        "lerna_model_loader_ettin_constants",
        Path(__file__).with_name("ettin_constants.py"),
    )
    if _ETTIN_SPEC is None or _ETTIN_SPEC.loader is None:
        raise ImportError("could not load lerna.utils.ettin_constants")
    _ETTIN_MODULE = importlib.util.module_from_spec(_ETTIN_SPEC)
    _ETTIN_SPEC.loader.exec_module(_ETTIN_MODULE)
    ETTIN_MODEL_ID = _ETTIN_MODULE.ETTIN_MODEL_ID
    ETTIN_REVISION = _ETTIN_MODULE.ETTIN_REVISION
    validate_ettin_revision = _ETTIN_MODULE.validate_ettin_revision

_MODERNBERT_PREFIXES = ("answerdotai/ModernBERT",)
_ETTIN_PREFIXES = ("jhu-clsp/ettin-encoder-150m",)


def load_tokenizer(model_name: str, *, revision: Optional[str] = None, local_files_only: bool = False):
    """Load tokenizer assets without model weights for CPU-side planning."""
    kwargs = {}
    if revision is not None:
        kwargs["revision"] = revision
    if local_files_only:
        kwargs["local_files_only"] = local_files_only
    return AutoTokenizer.from_pretrained(model_name, **kwargs)


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