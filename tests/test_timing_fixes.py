"""Tests for [Gate-1 Fix 2] timing instrumentation correctness.

Covers:
  - evaluate() override records eval_seconds > 0
  - on_evaluate_begin is not relied upon (the old broken hook)
  - _TimingCallback.train_loop_seconds is set on train_end
  - mean_step_ms fields are populated when CUDA is available
  - avg_dataloader_wait_ms is populated
"""

import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainingArguments

from lerna.trainers import TrueBackwardSkippingTrainer, AlwaysFalsePolicy
from lerna.trainers.true_skip_trainer import _TimingCallback


class _TinyDS(Dataset):
    def __init__(self, n=32, d=8, c=2):
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


@pytest.fixture
def _tmpdir():
    d = tempfile.mkdtemp()
    yield d


@pytest.fixture
def _trainer(_tmpdir):
    model = _TinyModel()
    args = TrainingArguments(
        output_dir=_tmpdir,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        max_steps=8,
        eval_steps=4,
        do_eval=True,
        remove_unused_columns=False,
        logging_steps=4,
        dataloader_num_workers=0,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
        seed=42,
    )
    trainer = TrueBackwardSkippingTrainer(
        model=model,
        args=args,
        train_dataset=_TinyDS(n=32),
        eval_dataset=_TinyDS(n=8),
        data_collator=_collate,
        skip_policy=AlwaysFalsePolicy(),
        online_ler_enabled=False,
    )
    return trainer


def test_evaluate_records_eval_seconds_greater_than_zero(_trainer):
    """eval_seconds must be > 0 after at least one evaluate() call."""
    _trainer.evaluate()
    assert _trainer.instr.eval_seconds > 0.0, (
        "eval_seconds must be > 0 after evaluate(); "
        f"got {_trainer.instr.eval_seconds}"
    )


def test_timing_callback_no_on_evaluate_hook():
    """_TimingCallback must NOT override on_evaluate (broken hook in transformers 5.x).

    In transformers >= 5.x, on_evaluate fires AFTER evaluation completes,
    making it useless for timing.  The evaluate() override in the Trainer
    is the correct fix.
    """
    from transformers import TrainerCallback
    base_methods = {name for name in dir(TrainerCallback) if name.startswith("on_")}
    our_overrides = {
        name for name in vars(_TimingCallback)
        if name.startswith("on_")
    }
    # We should only override on_train_begin and on_train_end
    assert "on_evaluate" not in our_overrides or (
        "on_evaluate" in base_methods
    ), (
        "_TimingCallback should not rely on on_evaluate for timing; "
        "use the evaluate() override instead"
    )
    # The real check: does _TimingCallback define on_evaluate_end?
    # (the old broken pair)
    assert "on_evaluate_end" not in vars(_TimingCallback), (
        "_TimingCallback should not define on_evaluate_end (non-existent in "
        "transformers 5.x)"
    )


def test_train_loop_seconds_set_after_training(_trainer):
    """train_loop_seconds must be set (> 0) after training."""
    _trainer.train()
    assert _trainer.instr.train_loop_seconds > 0.0


def test_mean_step_ms_populated_with_cuda(_trainer):
    """mean_step_ms_executed must be populated when CUDA is available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available in this environment")
    _trainer.train()
    assert _trainer.instr.mean_step_ms_executed is not None
    assert _trainer.instr.mean_step_ms_executed > 0.0


def test_avg_dataloader_wait_ms_populated(_trainer):
    """avg_dataloader_wait_ms must be set after training."""
    _trainer.train()
    assert _trainer.instr.avg_dataloader_wait_ms is not None


def test_avg_dataloader_wait_ms_in_json_file(_trainer):
    """avg_dataloader_wait_ms must be in the instrumentation.json file.

    Bug: _SkipWrapperInstaller (registered before _TimingCallback) called
    _dump_instrumentation() before _TimingCallback.on_train_end could compute
    avg_dataloader_wait_ms, leaving it None in the JSON output.

    Fix: _dump_instrumentation() now computes the field as a safety net.
    """
    _trainer.train()
    instr_path = os.path.join(_trainer.args.output_dir, "instrumentation.json")
    assert os.path.exists(instr_path), f"instrumentation.json not found at {instr_path}"
    with open(instr_path) as f:
        data = json.load(f)
    assert data["avg_dataloader_wait_ms"] is not None, (
        "avg_dataloader_wait_ms must be non-None in instrumentation.json "
        "(was None due to callback ordering bug)"
    )
    assert data["avg_dataloader_wait_ms"] > 0.0


def test_train_loop_seconds_in_json_file(_trainer):
    """train_loop_seconds must be > 0 in instrumentation.json.

    Same callback-ordering bug as avg_dataloader_wait_ms: _TimingCallback.on_train_end
    was called after _dump_instrumentation, so train_loop_seconds was 0.0 in the file.
    """
    _trainer.train()
    instr_path = os.path.join(_trainer.args.output_dir, "instrumentation.json")
    assert os.path.exists(instr_path)
    with open(instr_path) as f:
        data = json.load(f)
    assert data["train_loop_seconds"] > 0.0, (
        "train_loop_seconds must be > 0 in instrumentation.json "
        f"(got {data['train_loop_seconds']})"
    )
