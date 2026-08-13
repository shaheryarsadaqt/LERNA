"""Dependency-light tests for the tokenizer-only model loader (6C-1T)."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest import mock

import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]

_original_transformers = sys.modules.get("transformers")
transformers_mock = ModuleType("transformers")
transformers_mock.AutoModelForSequenceClassification = mock.MagicMock()
transformers_mock.AutoTokenizer = mock.MagicMock()
sys.modules["transformers"] = transformers_mock

try:
    _LOADER_PATH = REPO_ROOT / "lerna" / "utils" / "model_loader.py"
    _LOADER_SPEC = importlib.util.spec_from_file_location(
        "lerna_model_loader_for_loader_tests",
        _LOADER_PATH,
    )
    model_loader = importlib.util.module_from_spec(_LOADER_SPEC)
    _LOADER_SPEC.loader.exec_module(model_loader)
finally:
    if _original_transformers is not None:
        sys.modules["transformers"] = _original_transformers
    else:
        del sys.modules["transformers"]


class TokenizerOnlyLoaderTests(unittest.TestCase):
    """6C-1T: load_tokenizer loads tokenizer assets without model weights."""

    def test_load_tokenizer_delegates_to_auto_tokenizer(self):
        fake_tokenizer = object()
        with mock.patch.object(
            model_loader.AutoTokenizer,
            "from_pretrained",
            return_value=fake_tokenizer,
        ) as from_pretrained:
            result = model_loader.load_tokenizer("some/model")
        from_pretrained.assert_called_once_with("some/model")
        self.assertIs(result, fake_tokenizer)

    def test_load_tokenizer_does_not_load_model_weights(self):
        with mock.patch.object(
            model_loader.AutoTokenizer,
            "from_pretrained",
            return_value=object(),
        ) as from_pretrained:
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
            ) as model_from_pretrained:
                model_loader.load_tokenizer("some/model")
        from_pretrained.assert_called_once()
        model_from_pretrained.assert_not_called()

    def test_load_model_and_tokenizer_uses_load_tokenizer(self):
        fake_tokenizer = object()
        fake_model = object()
        with mock.patch.object(
            model_loader,
            "load_tokenizer",
            return_value=fake_tokenizer,
        ) as load_tokenizer:
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=fake_model,
            ) as model_from_pretrained:
                model, tokenizer = model_loader.load_model_and_tokenizer(
                    "some/model", num_labels=3
                )
        load_tokenizer.assert_called_once_with(
            "some/model", revision=None, local_files_only=False
        )
        model_from_pretrained.assert_called_once_with(
            "some/model", num_labels=3
        )
        self.assertIs(model, fake_model)
        self.assertIs(tokenizer, fake_tokenizer)

    def test_load_model_and_tokenizer_modernbert_kwargs(self):
        with mock.patch.object(
            model_loader,
            "load_tokenizer",
            return_value=object(),
        ):
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer(
                    "answerdotai/ModernBERT-base", num_labels=2
                )
        _, kwargs = model_from_pretrained.call_args
        self.assertEqual(kwargs["num_labels"], 2)
        self.assertIs(kwargs["reference_compile"], False)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")

    def test_load_model_and_tokenizer_ettin_kwargs(self):
        with mock.patch.object(
            model_loader,
            "load_tokenizer",
            return_value=object(),
        ):
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer(
                    model_loader.ETTIN_MODEL_ID,
                    num_labels=2,
                    revision=model_loader.ETTIN_REVISION,
                    local_files_only=True,
                )
        _, kwargs = model_from_pretrained.call_args
        self.assertEqual(kwargs["num_labels"], 2)
        self.assertIs(kwargs["reference_compile"], False)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(kwargs["revision"], model_loader.ETTIN_REVISION)
        self.assertIs(kwargs["local_files_only"], True)

    def test_load_model_and_tokenizer_legacy_does_not_gain_ettin_kwargs(self):
        with mock.patch.object(
            model_loader,
            "load_tokenizer",
            return_value=object(),
        ):
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer(
                    "roberta-base", num_labels=2
                )
        _, kwargs = model_from_pretrained.call_args
        self.assertNotIn("revision", kwargs)
        self.assertNotIn("local_files_only", kwargs)

    def test_load_tokenizer_receives_same_revision_and_local_only(self):
        with mock.patch.object(
            model_loader.AutoTokenizer,
            "from_pretrained",
            return_value=object(),
        ) as tokenizer_from_pretrained:
            with mock.patch.object(
                model_loader.AutoModelForSequenceClassification,
                "from_pretrained",
                return_value=object(),
            ):
                model_loader.load_model_and_tokenizer(
                    model_loader.ETTIN_MODEL_ID,
                    num_labels=2,
                    revision=model_loader.ETTIN_REVISION,
                    local_files_only=True,
                )
        tokenizer_from_pretrained.assert_called_once_with(
            model_loader.ETTIN_MODEL_ID,
            revision=model_loader.ETTIN_REVISION,
            local_files_only=True,
        )

    def test_ettin_model_registry_entry(self):
        self.assertEqual(
            model_loader.MODELS["ettin"],
            model_loader.ETTIN_MODEL_ID,
        )

    def test_ettin_revision_constant_is_pinned_sha(self):
        self.assertEqual(len(model_loader.ETTIN_REVISION), 40)
        self.assertTrue(
            all(ch in "0123456789abcdef" for ch in model_loader.ETTIN_REVISION)
        )

    def test_load_model_and_tokenizer_ettin_requires_exact_revision(self):
        with mock.patch.object(
            model_loader.AutoModelForSequenceClassification,
            "from_pretrained",
            return_value=object(),
        ) as model_from_pretrained:
            with mock.patch.object(
                model_loader.AutoTokenizer,
                "from_pretrained",
                return_value=object(),
            ):
                with self.assertRaises(ValueError) as ctx:
                    model_loader.load_model_and_tokenizer(
                        model_loader.ETTIN_MODEL_ID,
                        num_labels=2,
                        revision=None,
                        local_files_only=False,
                    )
                self.assertIn("requires an explicit revision", str(ctx.exception))
        model_from_pretrained.assert_not_called()

    def test_load_tokenizer_ettin_requires_exact_revision(self):
        with mock.patch.object(
            model_loader.AutoTokenizer,
            "from_pretrained",
            return_value=object(),
        ):
            with self.assertRaises(ValueError) as ctx:
                model_loader.load_tokenizer(
                    model_loader.ETTIN_MODEL_ID,
                    revision=None,
                    local_files_only=False,
                )
            self.assertIn("requires an explicit revision", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()