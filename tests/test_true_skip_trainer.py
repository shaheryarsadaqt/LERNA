"""REVISED — tests/test_true_skip_trainer.py

Changes from original v1:

    [CRIT-1] Test that scheduler does NOT advance on skipped steps (the primary scheduler-flag-bug test)
    [IMP-4] Uses optimizer_step_attempts instead of optimizer_step_calls/real_optimizer_steps
    All inequality invariants per [FIX #4]

"""
import os
import tempfile

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lerna.trainers import TrueBackwardSkippingTrainer
from lerna.trainers.policies import AlwaysFalsePolicy, RandomSkipPolicy
from lerna.trainers.true_skip_trainer import _OptimizerStepWrapper


class _TinyDS(Dataset):
    def __init__(self, n=16, d=8, c=2):
        self.x = torch.randn(n, d)
        self.y = torch.randint(0, c, (n,))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"input_ids": self.x[i], "labels": self.y[i]}


class _TinyModel(nn.Module):
    def __init__(self, d=8, c=2):
        super().__init__()
        self.fc = nn.Linear(d, c)

    def forward(self, input_ids=None, labels=None, **kw):
        logits = self.fc(input_ids)
        loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return {"loss": loss, "logits": logits}


def _collate(b):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in b]),
        "labels": torch.stack([x["labels"] for x in b]),
    }


def _run(policy, td):
    args = TrainingArguments(
        output_dir=td,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        use_cpu=True,
        fp16=False,
        bf16=False,
    )
    t = TrueBackwardSkippingTrainer(
        model=_TinyModel(),
        args=args,
        train_dataset=_TinyDS(),
        data_collator=_collate,
        skip_policy=policy,
        instrumentation_path=os.path.join(td, "instrumentation.json"),
    )
    t.train()
    return t.get_instrumentation()


def _check_universal_invariants(i):
    assert i["forward_calls"] == i["batches_seen"]
    assert i["forward_calls"] == i["backward_calls"] + i["skipped_backward_steps"]
    assert i["optimizer_step_attempts"] <= i["backward_calls"]
    assert i["scheduler_step_calls"] <= i["optimizer_step_attempts"]
    assert i["grad_scaler_step_calls"] <= i["optimizer_step_attempts"]


def test_always_false():
    with tempfile.TemporaryDirectory() as td:
        i = _run(AlwaysFalsePolicy(), td)
        _check_universal_invariants(i)
        assert i["skipped_backward_steps"] == 0
        assert i["backward_calls"] == i["forward_calls"]
        assert os.path.exists(os.path.join(td, "instrumentation.json"))


def test_full_skip():
    with tempfile.TemporaryDirectory() as td:
        i = _run(RandomSkipPolicy(target_skip_rate=1.0, min_step=0, seed=0), td)
        _check_universal_invariants(i)
        assert i["skipped_backward_steps"] == i["forward_calls"]
        assert i["backward_calls"] == 0
        assert i["optimizer_step_attempts"] == 0
        assert i["scheduler_step_calls"] == 0
        assert i["skip_ratio_by_batch"] == 1.0


def test_partial_skip():
    with tempfile.TemporaryDirectory() as td:
        i = _run(RandomSkipPolicy(target_skip_rate=0.5, min_step=0, seed=123), td)
        _check_universal_invariants(i)
        assert 0 < i["skipped_backward_steps"] < i["forward_calls"]


def test_scheduler_does_not_advance_on_skip():
    """[CRIT-1] The primary scheduler-flag-bug regression test.

    With full skipping (rate=1.0), the scheduler must NOT advance at all.
    If the two-flag fix is broken and the optimizer wrapper clears a flag
    the scheduler needs, this test will fail because scheduler_step_calls > 0.
    """
    with tempfile.TemporaryDirectory() as td:
        i = _run(RandomSkipPolicy(target_skip_rate=1.0, min_step=0, seed=0), td)
        assert i["scheduler_step_calls"] == 0, (
            f"Scheduler advanced {i['scheduler_step_calls']} times on fully-skipped "
            f"run. This indicates the two-flag fix (CRIT-1) is broken."
        )
        assert i["optimizer_step_attempts"] == 0


def test_fp16_skip_step_updates_scaler_state():
    class FakeScaler:
        def __init__(self):
            # Simulate internal per-optimizer state mapping used by GradScaler
            self._per_optimizer_states = {"dummy_opt": object()}

        # Provide clearable mapping; wrapper should clear this dict.

    class FakeTrainer:
        def __init__(self):
            self._skip_optimizer_step = True
            self._scaler_ref = FakeScaler()
            self.optimizer = None
            self.instr = type("I", (), {"precision_mode": "fp16", "optimizer_step_attempts": 0})()
            self._orig_opt_step = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("should not be called"))

    fake_trainer = FakeTrainer()
    wrapper = _OptimizerStepWrapper(fake_trainer)
    result = wrapper()
    assert result is None
    assert fake_trainer._skip_optimizer_step is False
    assert fake_trainer._scaler_ref._per_optimizer_states == {}
    assert fake_trainer.instr.optimizer_step_attempts == 0


def test_scheduler_advances_on_real_steps():
    """Verify scheduler DOES advance on non-skipped steps."""
    with tempfile.TemporaryDirectory() as td:
        i = _run(AlwaysFalsePolicy(), td)
        assert i["scheduler_step_calls"] > 0
        assert i["optimizer_step_attempts"] > 0


def test_grad_accum_guard():
    """[FIX #3] non-trivial policy + grad_accum>1 must raise."""
    import pytest

    with tempfile.TemporaryDirectory() as td:
        args = TrainingArguments(
            output_dir=td,
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            save_strategy="no",
            report_to="none",
            remove_unused_columns=False,
            use_cpu=True,
        )
        with pytest.raises(ValueError):
            TrueBackwardSkippingTrainer(
                model=_TinyModel(),
                args=args,
                train_dataset=_TinyDS(),
                data_collator=_collate,
                skip_policy=RandomSkipPolicy(target_skip_rate=0.5),
            )
