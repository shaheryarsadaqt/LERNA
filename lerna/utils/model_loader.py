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
# Round 17: UltraBERT runs on LERNA's own implementation (lerna/utils/ultrabert_model.py),
# because transformers 4.56 has none.
_ULTRABERT_PREFIXES = ("dmis-lab/UltraBERT-",)


def is_ettin_model(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _ETTIN_PREFIXES)


def is_ultrabert_model(model_name: str) -> bool:
    return isinstance(model_name, str) and any(model_name.startswith(p) for p in _ULTRABERT_PREFIXES)


def _ultrabert_module():
    try:
        from . import ultrabert_model
    except ImportError:
        spec = importlib.util.spec_from_file_location(
            "lerna_model_loader_ultrabert_model", Path(__file__).with_name("ultrabert_model.py"))
        if spec is None or spec.loader is None:
            raise ImportError("could not load lerna.utils.ultrabert_model")
        ultrabert_model = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ultrabert_model)
    return ultrabert_model


def _require_pinned(model_name: str, revision: Optional[str]) -> str:
    if (not isinstance(revision, str) or len(revision) != 40
            or any(ch not in "0123456789abcdef" for ch in revision)):
        raise ValueError(f"{model_name} requires an explicit 40-character lowercase hex revision; got {revision!r}")
    return revision


def _accepts_reference_compile() -> bool:
    """Whether the installed transformers still has ModernBertConfig.reference_compile.

    transformers deprecated this option during v5 and later removed it. Versions
    without it raise TypeError in the model constructor if it is passed (seen with
    5.17.0). from_pretrained hands a keyword to the config only when the config has
    that attribute, so the same test is used here.
    """
    try:
        from transformers import ModernBertConfig
    except Exception:
        return False
    try:
        return hasattr(ModernBertConfig(), "reference_compile")
    except Exception:
        return False


def load_tokenizer(model_name: str, *, revision: Optional[str] = None, local_files_only: bool = False):
    """Load tokenizer assets without model weights for CPU-side planning."""
    if is_ultrabert_model(model_name):
        # always from the local Hub cache, at a pinned revision
        return _ultrabert_module().load_ultrabert_tokenizer(model_name, revision=_require_pinned(model_name, revision))
    if is_ettin_model(model_name):
        validated_revision = validate_ettin_revision(model_name, revision)
        if validated_revision is None:
            raise ValueError(
                f"Ettin model requires an explicit revision; "
                f"expected {ETTIN_REVISION!r}"
            )
        revision = validated_revision
        local_files_only = True
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

    ModernBERT/Ettin: attn_implementation="eager", plus reference_compile=False
    when the installed transformers still has that option
    Others: standard AutoModel loading
    """
    if is_ultrabert_model(model_name):
        # Round 17: LERNA's UltraBERT implementation, eager attention, FP32, from the local
        # Hub cache at a pinned revision; only the classifier is newly initialized.
        if device_map is not None:
            raise ValueError("UltraBERT is loaded without device_map; the trainer places it")
        pinned = _require_pinned(model_name, revision)
        ub = _ultrabert_module()
        tokenizer = ub.load_ultrabert_tokenizer(model_name, revision=pinned)
        model = ub.load_ultrabert_for_sequence_classification(
            model_name, revision=pinned, num_labels=num_labels, problem_type=problem_type)
        return model, tokenizer

    is_modernbert = any(model_name.startswith(p) for p in _MODERNBERT_PREFIXES)
    is_ettin = is_ettin_model(model_name)

    if is_ettin:
        validated_revision = validate_ettin_revision(model_name, revision)
        if validated_revision is None:
            raise ValueError(
                f"Ettin model requires an explicit revision; "
                f"expected {ETTIN_REVISION!r}"
            )
        revision = validated_revision
        local_files_only = True

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
        if _accepts_reference_compile():
            model_kwargs["reference_compile"] = False
        model_kwargs["attn_implementation"] = "eager"
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
    "ultrabert": "dmis-lab/UltraBERT-150M",
}