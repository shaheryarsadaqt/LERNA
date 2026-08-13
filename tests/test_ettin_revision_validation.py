"""Mutation-sensitive tests for the shared Ettin revision validator."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Mock transformers (the only dependency model_loader needs) before import.
transformers_mock = ModuleType("transformers")
transformers_mock.AutoModelForSequenceClassification = mock.MagicMock()
transformers_mock.AutoTokenizer = mock.MagicMock()
sys.modules["transformers"] = transformers_mock

_LOADER_PATH = REPO_ROOT / "lerna" / "utils" / "model_loader.py"
_LOADER_SPEC = importlib.util.spec_from_file_location(
    "lerna_model_loader_for_ettin_revision",
    _LOADER_PATH,
)
model_loader = importlib.util.module_from_spec(_LOADER_SPEC)
_LOADER_SPEC.loader.exec_module(model_loader)

ETTIN_MODEL_ID = model_loader.ETTIN_MODEL_ID
ETTIN_REVISION = model_loader.ETTIN_REVISION
validate_ettin_revision = model_loader.validate_ettin_revision
MODELS = model_loader.MODELS


def test_ettin_exact_revision_accepted():
    assert (
        validate_ettin_revision(ETTIN_MODEL_ID, ETTIN_REVISION) == ETTIN_REVISION
    )


def test_ettin_missing_revision_rejected():
    with pytest.raises(ValueError, match="requires an explicit revision"):
        validate_ettin_revision(ETTIN_MODEL_ID, None)


def test_ettin_uppercase_revision_rejected():
    with pytest.raises(ValueError, match="must be lowercase SHA"):
        validate_ettin_revision(ETTIN_MODEL_ID, ETTIN_REVISION.upper())


def test_ettin_malformed_revision_rejected():
    with pytest.raises(ValueError, match="40-character lowercase hex SHA"):
        validate_ettin_revision(ETTIN_MODEL_ID, "short")


def test_ettin_wrong_revision_rejected():
    with pytest.raises(ValueError, match="revision mismatch"):
        validate_ettin_revision(ETTIN_MODEL_ID, "a" * 40)


def test_legacy_model_without_revision_returns_none():
    assert validate_ettin_revision("roberta-base", None) is None


def test_ettin_parser_selection_maps_to_model_id():
    assert ETTIN_MODEL_ID == "jhu-clsp/ettin-encoder-150m"
    assert MODELS["ettin"] == ETTIN_MODEL_ID
    assert len(ETTIN_REVISION) == 40
    assert all(ch in "0123456789abcdef" for ch in ETTIN_REVISION)