"""Tests for optional Step 4A online diagnostics."""

import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lerna.trainers import AlwaysFalsePolicy, RandomSkipPolicy, TrueBackwardSkippingTrainer


class TinyDataset(Dataset):
    def __init__(self, size=8, width=4):
        generator = torch.Generator().manual_seed(13)
        self.inputs = torch.randn(size, width, generator=generator)
        self.labels = torch.randint(0, 2, (size,), generator=generator)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {"input_ids": self.inputs[index], "labels": self.labels[index]}


class TinyModel(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.classifier = nn.Linear(width, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        logits = self.classifier(input_ids)
        loss = nn.functional.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}


def collate(batch):
    return {
        "input_ids": torch.stack([row["input_ids"] for row in batch]),
        "labels": torch.stack([row["labels"] for row in batch]),
    }


class CountingTracker:
    def __init__(self, fail=False):
        self.calls = 0
        self.fail = fail

    def update(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise RuntimeError("expected tracker failure")


class GradNormPolicy:
    name = "grad_norm_consumer"

    def __init__(self):
        self.norms = []

    def should_skip(self, trainer, model, inputs):
        return False

    def record_grad_norm(self, value):
        self.norms.append(value)


def build_trainer(tmpdir, *, policy=None, tracker=None, **kwargs):
    args = TrainingArguments(
        output_dir=tmpdir,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=True,
        fp16=False,
        bf16=False,
    )
    trainer = TrueBackwardSkippingTrainer(
        model=TinyModel(),
        args=args,
        train_dataset=TinyDataset(),
        data_collator=collate,
        skip_policy=policy or AlwaysFalsePolicy(),
        instrumentation_path=os.path.join(tmpdir, "instrumentation.json"),
        **kwargs,
    )
    trainer._ler_tracker = tracker
    return trainer


def test_disabled_mode_does_not_call_tracker_or_retain_logits():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CountingTracker(fail=True)
        trainer = build_trainer(
            tmpdir,
            tracker=tracker,
            online_ler_enabled=False,
            capture_logits=False,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert tracker.calls == 0
        assert trainer._last_real_logits is None
        assert trainer._last_logits is None
        assert trainer.last_logits is None
        assert trainer._pre_clip_grad_norm is None
        assert diagnostics["online_ler_enabled"] is False
        assert diagnostics["online_ler_update_interval"] == 0
        assert diagnostics["online_ler_update_attempts"] == 0
        assert diagnostics["online_ler_update_successes"] == 0
        assert diagnostics["capture_logits_enabled"] is False
        assert diagnostics["grad_norm_capture_enabled"] is False


def test_enabled_mode_updates_at_configured_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CountingTracker()
        trainer = build_trainer(
            tmpdir,
            tracker=tracker,
            online_ler_enabled=True,
            online_ler_update_interval=2,
            capture_logits=True,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert diagnostics["batches_seen"] == 4
        assert tracker.calls == 2
        assert diagnostics["online_ler_update_attempts"] == 2
        assert diagnostics["online_ler_update_successes"] == 2
        assert diagnostics["capture_logits_enabled"] is True
        assert trainer._last_real_logits is not None


def test_failed_tracker_updates_are_attempted_but_not_successful():
    with tempfile.TemporaryDirectory() as tmpdir:
        tracker = CountingTracker(fail=True)
        trainer = build_trainer(
            tmpdir,
            tracker=tracker,
            online_ler_enabled=True,
            online_ler_update_interval=1,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert tracker.calls == diagnostics["batches_seen"]
        assert diagnostics["online_ler_update_attempts"] == tracker.calls
        assert diagnostics["online_ler_update_successes"] == 0


def test_grad_norm_scan_runs_only_for_consuming_policy():
    with tempfile.TemporaryDirectory() as tmpdir:
        policy = GradNormPolicy()
        trainer = build_trainer(
            tmpdir,
            policy=policy,
            online_ler_enabled=False,
            capture_logits=False,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert diagnostics["grad_norm_capture_enabled"] is True
        assert len(policy.norms) == diagnostics["backward_calls"]
        assert policy.norms
        assert all(value >= 0.0 for value in policy.norms)
        assert trainer._pre_clip_grad_norm == pytest.approx(policy.norms[-1])


def test_disabled_diagnostics_preserve_skip_accounting_invariants():
    with tempfile.TemporaryDirectory() as tmpdir:
        trainer = build_trainer(
            tmpdir,
            policy=RandomSkipPolicy(target_skip_rate=0.5, min_step=0, seed=7),
            online_ler_enabled=False,
            capture_logits=False,
        )
        trainer.train()

        diagnostics = trainer.get_instrumentation()
        assert diagnostics["forward_calls"] == diagnostics["batches_seen"]
        assert diagnostics["invariant_forward_eq_backward_plus_skipped"] is True
        assert diagnostics["invariant_opt_le_backward"] is True
        assert diagnostics["invariant_scheduler_policy_consistent"] is True
        assert diagnostics["online_ler_update_attempts"] == 0
        assert diagnostics["online_ler_update_successes"] == 0


def test_enabled_mode_rejects_nonpositive_interval():
    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="online_ler_update_interval"):
            build_trainer(
                tmpdir,
                online_ler_enabled=True,
                online_ler_update_interval=0,
            )
