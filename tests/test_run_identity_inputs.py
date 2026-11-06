"""Lightweight identity and fingerprint tests that do not require torch."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_PROVENANCE_PATH = REPO_ROOT / "lerna" / "utils" / "run_provenance.py"
_SPEC = importlib.util.spec_from_file_location(
    "lerna_run_provenance_for_identity_inputs",
    _PROVENANCE_PATH,
)
run_provenance = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_provenance)

build_identity_inputs = run_provenance.build_identity_inputs
build_scientific_fingerprint = run_provenance.build_scientific_fingerprint


def test_build_identity_inputs_includes_model_revision_when_provided():
    identity = build_identity_inputs(
        task="mrpc",
        training_seed=42,
        model_id="jhu-clsp/ettin-encoder-150m",
        model_revision="45d08642849e5c5701b162671ac811b7654bfd9f",
        max_samples_requested=None,
        train_samples_realized=1000,
        eval_samples_realized=200,
        train_dataset_fingerprint="abc",
        eval_dataset_fingerprint="def",
        num_epochs=3,
        control="full_finetune",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        total_steps=100,
        git_sha="abc123",
    )
    assert identity["model_revision"] == "45d08642849e5c5701b162671ac811b7654bfd9f"
    assert identity["model_id"] == "jhu-clsp/ettin-encoder-150m"


def test_build_identity_inputs_omits_model_revision_when_none():
    identity = build_identity_inputs(
        task="synthetic_task",
        training_seed=7,
        model_id="tiny-linear-cpu",
        model_revision=None,
        max_samples_requested=None,
        train_samples_realized=100,
        eval_samples_realized=32,
        train_dataset_fingerprint="train",
        eval_dataset_fingerprint="eval",
        num_epochs=1,
        control="full_finetune",
        target_skip_rate=0.0,
        policy_seed=7,
        skip_update_mode="freeze",
        no_early_stopping=True,
        total_steps=100,
        git_sha="def456",
    )
    assert identity["model_revision"] is None


def test_fingerprint_changes_when_model_revision_changes():
    base = build_identity_inputs(
        task="mrpc",
        training_seed=42,
        model_id="jhu-clsp/ettin-encoder-150m",
        model_revision=None,
        max_samples_requested=None,
        train_samples_realized=1000,
        eval_samples_realized=200,
        train_dataset_fingerprint="abc",
        eval_dataset_fingerprint="def",
        num_epochs=3,
        control="full_finetune",
        target_skip_rate=0.30,
        policy_seed=42,
        skip_update_mode="freeze",
        no_early_stopping=True,
        total_steps=100,
        git_sha="abc123",
    )
    changed = dict(base)
    changed["model_revision"] = "45d08642849e5c5701b162671ac811b7654bfd9f"
    assert build_scientific_fingerprint(changed) != build_scientific_fingerprint(base)


def test_legacy_loader_call_does_not_gain_revision_kwargs():
    import sys
    from types import ModuleType
    from unittest import mock

    _original_transformers = sys.modules.get("transformers")
    transformers_mock = ModuleType("transformers")
    transformers_mock.AutoModelForSequenceClassification = mock.MagicMock()
    transformers_mock.AutoTokenizer = mock.MagicMock()
    sys.modules["transformers"] = transformers_mock

    try:
        _LOADER_PATH = REPO_ROOT / "lerna" / "utils" / "model_loader.py"
        _LOADER_SPEC = importlib.util.spec_from_file_location(
            "lerna_model_loader_for_identity_inputs",
            _LOADER_PATH,
        )
        model_loader = importlib.util.module_from_spec(_LOADER_SPEC)
        _LOADER_SPEC.loader.exec_module(model_loader)

        with mock.patch.object(
            model_loader.AutoTokenizer,
            "from_pretrained",
            return_value=object(),
        ):
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=object(),
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer("roberta-base", num_labels=2)
        _, kwargs = model_from_pretrained.call_args
        assert "revision" not in kwargs
        assert "local_files_only" not in kwargs
    finally:
        if _original_transformers is not None:
            sys.modules["transformers"] = _original_transformers
        else:
            del sys.modules["transformers"]
