"""Dependency-light tests for the tokenizer-only model loader (6C-1T)."""

import unittest
from unittest import mock

from lerna.utils import model_loader


class TokenizerOnlyLoaderTests(unittest.TestCase):
    """6C-1T: load_tokenizer loads tokenizer assets without model weights."""

    def test_load_tokenizer_delegates_to_auto_tokenizer(self):
        fake_tokenizer = object()
        with mock.patch(
            "lerna.utils.model_loader.AutoTokenizer.from_pretrained",
            return_value=fake_tokenizer,
        ) as from_pretrained:
            result = model_loader.load_tokenizer("some/model")
        from_pretrained.assert_called_once_with("some/model")
        self.assertIs(result, fake_tokenizer)

    def test_load_tokenizer_does_not_load_model_weights(self):
        with mock.patch(
            "lerna.utils.model_loader.AutoTokenizer.from_pretrained",
            return_value=object(),
        ) as from_pretrained:
            with mock.patch(
                "lerna.utils.model_loader.AutoModelForSequenceClassification.from_pretrained"
            ) as model_from_pretrained:
                model_loader.load_tokenizer("some/model")
        from_pretrained.assert_called_once()
        model_from_pretrained.assert_not_called()

    def test_load_model_and_tokenizer_uses_load_tokenizer(self):
        fake_tokenizer = object()
        fake_model = object()
        with mock.patch(
            "lerna.utils.model_loader.load_tokenizer",
            return_value=fake_tokenizer,
        ) as load_tokenizer:
            with mock.patch(
                "lerna.utils.model_loader.AutoModelForSequenceClassification.from_pretrained",
                return_value=fake_model,
            ) as model_from_pretrained:
                model, tokenizer = model_loader.load_model_and_tokenizer(
                    "some/model", num_labels=3
                )
        load_tokenizer.assert_called_once_with("some/model")
        model_from_pretrained.assert_called_once_with(
            "some/model", num_labels=3
        )
        self.assertIs(model, fake_model)
        self.assertIs(tokenizer, fake_tokenizer)

    def test_load_model_and_tokenizer_modernbert_kwargs(self):
        with mock.patch(
            "lerna.utils.model_loader.load_tokenizer",
            return_value=object(),
        ):
            with mock.patch(
                "lerna.utils.model_loader.AutoModelForSequenceClassification.from_pretrained"
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer(
                    "answerdotai/ModernBERT-base", num_labels=2
                )
        _, kwargs = model_from_pretrained.call_args
        self.assertEqual(kwargs["num_labels"], 2)
        self.assertIs(kwargs["reference_compile"], False)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")

    def test_load_model_and_tokenizer_passes_problem_type_and_device_map(self):
        with mock.patch(
            "lerna.utils.model_loader.load_tokenizer",
            return_value=object(),
        ):
            with mock.patch(
                "lerna.utils.model_loader.AutoModelForSequenceClassification.from_pretrained"
            ) as model_from_pretrained:
                model_loader.load_model_and_tokenizer(
                    "some/model",
                    num_labels=2,
                    problem_type="single_label_classification",
                    device_map="cpu",
                )
        _, kwargs = model_from_pretrained.call_args
        self.assertEqual(kwargs["problem_type"], "single_label_classification")
        self.assertEqual(kwargs["device_map"], "cpu")


if __name__ == "__main__":
    unittest.main()
