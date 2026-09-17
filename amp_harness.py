from __future__ import annotations

import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

ROOT = Path(os.environ.get('LERNA_ROOT', Path(__file__).resolve().parent)).resolve()
OUT = Path(os.environ.get('LERNA_AMP_OUT', Path(__file__).resolve().parent / 'amp_evidence')).resolve()
EXPECTED_BRANCH = os.environ.get(
    'LERNA_EXPECTED_BRANCH', 'candidate/terminal-invalid-observation-0f55d012'
)
EXPECTED_HEAD = os.environ.get(
    'LERNA_EXPECTED_HEAD', 'a048dc99645a2ff228e60aa4d3917bb6acf3dbc9'
)
sys.path.insert(0, str(ROOT))

from lerna.trainers.policies import AlwaysFalsePolicy  # noqa: E402
from lerna.trainers.true_skip_trainer import (  # noqa: E402
    ObservationFailureError,
    SkipPolicy,
    TrueBackwardSkippingTrainer,
)
from lerna.utils.lagged_ler import SampledLaggedLERTracker  # noqa: E402


def source_identity():
    trainer_source = Path(inspect.getfile(TrueBackwardSkippingTrainer)).resolve()
    if trainer_source.parent.parent.parent != ROOT:
        raise AssertionError(
            f'imported trainer is outside active checkout: {trainer_source}'
        )
    branch = subprocess.check_output(
        ['git', '-C', str(ROOT), 'symbolic-ref', '--short', 'HEAD'],
        text=True,
    ).strip()
    head = subprocess.check_output(
        ['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
        text=True,
    ).strip()
    if branch != EXPECTED_BRANCH:
        raise AssertionError(f'branch mismatch: expected {EXPECTED_BRANCH}, got {branch}')
    if head != EXPECTED_HEAD:
        raise AssertionError(f'HEAD mismatch: expected {EXPECTED_HEAD}, got {head}')
    source_hash = hashlib.sha256(trainer_source.read_bytes()).hexdigest()
    return {
        'root': str(ROOT),
        'trainer_source': str(trainer_source),
        'trainer_source_sha256': source_hash,
        'branch': branch,
        'head': head,
        'output': str(OUT),
    }


class TinyDataset(Dataset):
    def __init__(self, size=4):
        self.inputs = torch.tensor(
            [[0.2, -0.1], [0.4, 0.3], [-0.5, 0.2], [0.1, 0.6]],
            dtype=torch.float32,
        )[:size]
        self.labels = torch.tensor([0, 1, 1, 0], dtype=torch.long)[:size]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {'input_ids': self.inputs[index], 'labels': self.labels[index]}


class TinyModel(torch.nn.Module):
    def __init__(self, nan_on_forward=None, overflow_on_forward=None):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.nan_on_forward = nan_on_forward
        self.overflow_on_forward = overflow_on_forward
        self.forward_index = 0
        self.forward_observations = []

    def forward(self, input_ids=None, labels=None, **kwargs):
        index = self.forward_index
        self.forward_index += 1
        logits = self.linear(input_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if self.nan_on_forward is not None and index == self.nan_on_forward:
            loss = loss * torch.tensor(float('nan'), device=loss.device)
        if self.overflow_on_forward is not None and index == self.overflow_on_forward:
            loss = loss * 1e39
        self.forward_observations.append({
            'index': index,
            'input_device': str(input_ids.device),
            'logits_dtype': str(logits.dtype),
            'loss_dtype': str(loss.dtype),
            'autocast_enabled': bool(torch.is_autocast_enabled()),
            'autocast_gpu_dtype': str(torch.get_autocast_gpu_dtype()),
        })
        return {'loss': loss, 'logits': logits}


class ScriptedPolicy(SkipPolicy):
    name = 'amp_scripted'

    def __init__(self, skip_after_real_steps=None):
        self.skip_after_real_steps = skip_after_real_steps
        self.calls = 0
        self.skipped = []
        self.snapshot_before_skip = None
        self.trainer = None

    def should_skip(self, trainer, model, inputs):
        self.calls += 1
        should_skip = (
            self.skip_after_real_steps is not None
            and self.calls > self.skip_after_real_steps
        )
        if should_skip:
            self.skipped.append(self.calls - 1)
            self.snapshot_before_skip = snapshot(trainer)
        return should_skip


class ActionRecorder(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.events = []
        self.scaler = None

    def on_train_begin(self, args, state, control, optimizer=None, lr_scheduler=None, **kwargs):
        self.scaler = getattr(self.trainer.accelerator, 'scaler', None)
        if optimizer is not None:
            original = optimizer.step
            def optimizer_step(*a, **kw):
                self.events.append('optimizer.step')
                return original(*a, **kw)
            optimizer.step = optimizer_step
        if lr_scheduler is not None:
            original_scheduler = lr_scheduler.step
            def scheduler_step(*a, **kw):
                self.events.append('scheduler.step')
                return original_scheduler(*a, **kw)
            lr_scheduler.step = scheduler_step
        if self.scaler is not None:
            original_scaler_step = self.scaler.step
            original_scaler_update = self.scaler.update
            def scaler_step(*a, **kw):
                self.events.append('scaler.step')
                return original_scaler_step(*a, **kw)
            def scaler_update(*a, **kw):
                self.events.append('scaler.update')
                return original_scaler_update(*a, **kw)
            self.scaler.step = scaler_step
            self.scaler.update = scaler_update

    def on_step_end(self, args, state, control, **kwargs):
        self.events.append('trainer.step_end')


def digest_tensor(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def state_summary(trainer):
    model = {
        name: {'shape': list(value.shape), 'dtype': str(value.dtype), 'sha256': digest_tensor(value)}
        for name, value in trainer.model.state_dict().items()
    }
    optimizer = {}
    for parameter, state in trainer.optimizer.state.items():
        optimizer[str(id(parameter))] = {
            key: (digest_tensor(value) if torch.is_tensor(value) else value)
            for key, value in state.items()
        }
    scheduler = None
    if trainer.lr_scheduler is not None:
        scheduler = {
            key: value
            for key, value in trainer.lr_scheduler.state_dict().items()
            if not callable(value)
        }
    scaler = trainer.accelerator.scaler.state_dict() if getattr(trainer.accelerator, 'scaler', None) is not None else None
    return {'model': model, 'optimizer': optimizer, 'scheduler': scheduler, 'scaler': scaler}


def snapshot(trainer):
    return copy.deepcopy(state_summary(trainer))


def same_state(before, after, key):
    return before[key] == after[key]


def make_args(path, max_steps):
    return TrainingArguments(
        output_dir=str(path),
        max_steps=max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-2,
        warmup_steps=0,
        logging_steps=1,
        save_strategy='no',
        report_to='none',
        remove_unused_columns=False,
        use_cpu=False,
        fp16=True,
        bf16=False,
        dataloader_num_workers=0,
        disable_tqdm=True,
        seed=123,
    )


def build(path, model, policy, max_steps, tracker=None):
    trainer = TrueBackwardSkippingTrainer(
        model=model,
        args=make_args(path, max_steps),
        train_dataset=TinyDataset(),
        skip_policy=policy,
        instrumentation_path=str(path / 'instrumentation.json'),
        online_ler_mode='sampled_lagged' if tracker is not None else 'off',
        online_ler_update_interval=1,
        capture_logits=tracker is not None,
    )
    if tracker is not None:
        trainer.ler_tracker = tracker
    policy.trainer = trainer
    recorder = ActionRecorder(trainer)
    trainer.callback_handler.callbacks.insert(0, recorder)
    return trainer, recorder


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False))


def finite_case():
    path = OUT / 'finite'
    path.mkdir()
    model = TinyModel().cuda()
    trainer, recorder = build(path, model, AlwaysFalsePolicy(), 3)
    trainer.train()
    scaler = recorder.scaler
    result = {
        'case': 'finite_multi_step_amp',
        'state': state_summary(trainer),
        'events': recorder.events,
        'instrumentation': trainer.get_instrumentation(),
        'forward_observations': model.forward_observations,
        'scaler_scale': float(scaler.get_scale()),
        'scaler_state': scaler.state_dict(),
    }
    assert len(model.forward_observations) == 3
    assert all(item['input_device'].startswith('cuda') for item in model.forward_observations)
    assert all(item['autocast_enabled'] for item in model.forward_observations)
    assert all(item['logits_dtype'] == 'torch.float16' for item in model.forward_observations)
    assert recorder.events.count('optimizer.step') == 3
    assert recorder.events.count('scaler.step') == 3
    assert recorder.events.count('scaler.update') == 3
    assert trainer.instr.scheduler_step_calls == 2
    write_json(path / 'result.json', result)


def skip_case():
    path = OUT / 'backward_skip'
    path.mkdir()
    model = TinyModel().cuda()
    policy = ScriptedPolicy(skip_after_real_steps=1)
    trainer, recorder = build(path, model, policy, 2)
    trainer.train()
    after = snapshot(trainer)
    before = policy.snapshot_before_skip
    result = {
        'case': 'true_backward_skip_amp',
        'before_skip': before,
        'after_run': after,
        'events': recorder.events,
        'instrumentation': trainer.get_instrumentation(),
        'forward_observations': model.forward_observations,
        'scaler_state_after': recorder.scaler.state_dict(),
        'scaler_scale_after': float(recorder.scaler.get_scale()),
        'comparisons': {key: same_state(before, after, key) for key in ('model', 'optimizer', 'scheduler', 'scaler')},
    }
    assert before is not None
    assert result['comparisons']['model']
    assert result['comparisons']['optimizer']
    assert result['comparisons']['scheduler'], (
        f"scheduler changed: before={before['scheduler']}, after={after['scheduler']}"
    )
    assert trainer.instr.optimizer_step_attempts == 1
    assert recorder.events.count('scaler.step') == 1
    assert recorder.events.count('scaler.update') == 1
    assert trainer.instr.scheduler_step_calls == 0
    assert trainer.instr.skipped_backward_steps == 1
    write_json(path / 'result.json', result)


def overflow_recovery_case():
    path = OUT / 'overflow_recovery'
    path.mkdir()
    model = TinyModel(overflow_on_forward=0).cuda()
    initial_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    trainer, recorder = build(path, model, AlwaysFalsePolicy(), 2)
    trainer.train()
    result = {
        'case': 'amp_overflow_then_finite_recovery',
        'events': recorder.events,
        'instrumentation': trainer.get_instrumentation(),
        'forward_observations': model.forward_observations,
        'scaler_scale_after': float(recorder.scaler.get_scale()),
        'scaler_state_after': recorder.scaler.state_dict(),
        'optimizer_state_steps': [
            int(state['step'])
            for state in trainer.optimizer.state.values()
            if 'step' in state
        ],
        'model_changed_after_recovery': any(
            not torch.equal(initial_model[name], value)
            for name, value in model.state_dict().items()
        ),
    }
    assert len(model.forward_observations) == 2
    assert recorder.events.count('scaler.step') == 2
    assert recorder.events.count('scaler.update') == 2
    assert trainer.instr.grad_scaler_step_calls == 2
    assert trainer.instr.optimizer_step_attempts == 2
    assert result['scaler_scale_after'] < 65536.0
    assert result['optimizer_state_steps'] == [1, 1]
    assert result['model_changed_after_recovery']
    write_json(path / 'result.json', result)


class TerminalCaptureTrainer(TrueBackwardSkippingTrainer):
    def __init__(self, *args, capture_path=None, **kwargs):
        self.capture_path = capture_path
        self.before_rejection = None
        super().__init__(*args, **kwargs)

    def _dump_ler_diagnostics(self, valid_prefix, terminal_event):
        self.before_rejection = snapshot(self)
        return super()._dump_ler_diagnostics(valid_prefix, terminal_event)


def terminal_case():
    path = OUT / 'terminal_rejection'
    path.mkdir()
    model = TinyModel(nan_on_forward=1).cuda()
    policy = AlwaysFalsePolicy()
    tracker = SampledLaggedLERTracker(parameter_sample_size=2, sample_seed=7)
    trainer = TerminalCaptureTrainer(
        model=model,
        args=make_args(path, 2),
        train_dataset=TinyDataset(),
        skip_policy=policy,
        instrumentation_path=str(path / 'instrumentation.json'),
        online_ler_mode='sampled_lagged',
        online_ler_update_interval=1,
        capture_logits=True,
    )
    trainer.ler_tracker = tracker
    recorder = ActionRecorder(trainer)
    trainer.callback_handler.callbacks.insert(0, recorder)
    error = None
    try:
        trainer.train()
    except ObservationFailureError as exc:
        error = exc
    if error is None:
        raise AssertionError('terminal case did not raise ObservationFailureError')
    after = snapshot(trainer)
    result = {
        'case': 'terminal_rejection_after_backward_before_pending_actions',
        'error_type': type(error).__name__,
        'cause_type': type(error.__cause__).__name__ if error.__cause__ else None,
        'before_rejection': trainer.before_rejection,
        'after_termination': after,
        'events': recorder.events,
        'instrumentation': trainer.get_instrumentation(),
        'forward_observations': model.forward_observations,
        'scaler_state_after': recorder.scaler.state_dict(),
        'scaler_scale_after': float(recorder.scaler.get_scale()),
        'comparisons': {key: same_state(trainer.before_rejection, after, key) for key in ('model', 'optimizer', 'scheduler', 'scaler')},
        'terminal_event': getattr(error, 'tracker_event', None),
    }
    assert result['cause_type'] == 'TerminalInvalidObservation'
    assert result['comparisons']['model']
    assert result['comparisons']['optimizer']
    assert result['comparisons']['scheduler']
    assert result['comparisons']['scaler']
    assert recorder.events.count('scaler.step') == 1
    assert recorder.events.count('scaler.update') == 1
    assert recorder.events.count('optimizer.step') == 1
    assert trainer.instr.scheduler_step_calls == 0
    assert trainer.instr.backward_calls == 2
    assert trainer.instr.optimizer_step_attempts == 1
    assert trainer.instr.grad_scaler_step_calls == 1
    write_json(path / 'result.json', result)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    identity = source_identity()
    write_json(OUT / 'source_identity.json', identity)
    print(json.dumps(identity, sort_keys=True), flush=True)
    for name, function in (
        ('finite', finite_case),
        ('backward_skip', skip_case),
        ('terminal_rejection', terminal_case),
        ('overflow_recovery', overflow_recovery_case),
    ):
        print(f'BEGIN {name}', flush=True)
        function()
        print(f'PASS {name}', flush=True)
    print('ALL AMP CASES PASSED', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
