"""Source-verified AMP contract checks; synthetic fault injection, not a benchmark.

CUDA: python amp_harness.py --repo . --output-dir ../amp-check --device cuda
CPU:  same with --device cpu (requires torch.amp.GradScaler CPU support).
Every output directory must be new. CPU checks never certify CUDA execution.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import platform
import subprocess
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    parser.add_argument('--expected-trainer-sha256')
    return parser.parse_args()


CLI = parse_args() if __name__ == '__main__' else None
if CLI is not None:
    sys.path.insert(0, str(CLI.repo.resolve()))

import torch
import transformers
import accelerate
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments, set_seed
from lerna.trainers.policies import AlwaysFalsePolicy, LERGuidedStratifiedPolicy
from lerna.trainers.true_skip_trainer import ObservationFailureError, TrueBackwardSkippingTrainer
from lerna.utils.lagged_ler import SampledLaggedLERTracker

CASES = ('finite', 'sampled', 'backward_skip', 'overflow_recovery', 'terminal_rejection', 'tracker_error')


def strict_value(value):
    if isinstance(value, float) and not math.isfinite(value):
        return {'kind': 'nonfinite', 'value': 'NaN' if math.isnan(value) else ('+Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {str(k): strict_value(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (tuple, list)):
        return [strict_value(v) for v in value]
    return value


def write_json(path, value):
    path.write_text(json.dumps(strict_value(value), indent=2, allow_nan=False) + '\n')


def fingerprint(value):
    if torch.is_tensor(value):
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        return {'sha256': hashlib.sha256(raw).hexdigest(), 'shape': list(value.shape), 'dtype': str(value.dtype)}
    if isinstance(value, dict):
        return {str(k): fingerprint(v) for k, v in value.items() if not callable(v)}
    if isinstance(value, (list, tuple)):
        return [fingerprint(v) for v in value]
    return strict_value(value)


def snapshot(trainer):
    scaler = trainer.accelerator.scaler
    return {
        'model': fingerprint(trainer.model.state_dict()),
        'optimizer': fingerprint(trainer.optimizer.state_dict()),
        'scheduler': fingerprint(trainer.lr_scheduler.state_dict()),
        'scaler': fingerprint(scaler.state_dict()),
    }


class TinyDataset(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        return {'input_ids': torch.tensor([0.1 * (index + 1), -0.2]), 'labels': torch.tensor(index % 2)}


class TinyModel(torch.nn.Module):
    def __init__(self, injections=None, terminal_decision=None):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.injections = injections or {}
        self.terminal_decision = terminal_decision
        self.forward_count = 0
        self.observations = []
        self.linear.weight.register_hook(self.inject_gradient)

    def inject_gradient(self, grad):
        value = self.injections.get(self.forward_count - 1)
        if value is not None:
            grad = grad.clone()
            grad.flatten()[0] = value
        return grad

    def forward(self, input_ids=None, labels=None, **kwargs):
        decision = self.forward_count
        self.forward_count += 1
        logits = self.linear(input_ids)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        if decision == self.terminal_decision:
            loss = loss * float('nan')
        self.observations.append({
            'decision': decision, 'device': str(input_ids.device),
            'logits_dtype': str(logits.dtype), 'loss': float(loss.detach()),
            'cuda_autocast': bool(torch.is_autocast_enabled()),
        })
        return {'loss': loss, 'logits': logits}


class ScriptedPolicy:
    name = 'amp_contract_scripted'

    def __init__(self, skips):
        self.skips = set(skips)

    def should_skip(self, trainer, model, inputs):
        return trainer.instr.batches_seen - 1 in self.skips


class BrokenTracker(SampledLaggedLERTracker):
    def update(self, **kwargs):
        if self.n_updates == 1:
            raise RuntimeError('injected required tracker failure')
        return super().update(**kwargs)


class InspectTrainer(TrueBackwardSkippingTrainer):
    def __init__(self, *args, **kwargs):
        self.gradient_checks = {}
        self.before_rejection = None
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs, num_items_in_batch=None):
        result = super().training_step(model, inputs, num_items_in_batch)
        gradients = [p.grad.detach() for p in model.parameters() if p.grad is not None]
        scale = float(self.accelerator.scaler.get_scale())
        finite = all(bool(torch.isfinite(g).all()) for g in gradients) if gradients else None
        squared = sum(float(g.double().square().sum()) for g in gradients)
        self.gradient_checks[self.instr.batches_seen - 1] = {
            'gradients_present': bool(gradients), 'grad_finite': finite,
            'scaled_grad_norm': math.sqrt(squared) if gradients else None,
            'unscaled_grad_norm': math.sqrt(squared) / scale if gradients else None,
            'scale_before_step': scale,
        }
        return result

    def _fail_observation(self, exc, tracker, decision_index):
        self.before_rejection = snapshot(self)
        return super()._fail_observation(exc, tracker, decision_index)


class StepTrace(TrainerCallback):
    def __init__(self, trainer):
        self.trainer = trainer
        self.rows = []
        self.before = None
        self.original_methods = None

    def on_train_begin(self, args, state, control, **kwargs):
        t = self.trainer
        self.original_methods = (t.optimizer.step, t.lr_scheduler.step, t.accelerator.scaler.step)

    def on_step_begin(self, args, state, control, **kwargs):
        self.before = snapshot(self.trainer)
        self.before_counters = self.trainer.get_instrumentation()

    def on_step_end(self, args, state, control, **kwargs):
        t = self.trainer
        after = snapshot(t)
        counts = t.get_instrumentation()
        decision = counts['batches_seen'] - 1
        delta = {k: counts[k] - self.before_counters[k] for k in (
            'optimizer_step_attempts', 'optimizer_step_successes',
            'optimizer_steps_skipped_amp', 'scheduler_step_calls', 'skipped_backward_steps')}
        self.rows.append({
            'decision': decision, 'deltas': delta,
            'gradient': t.gradient_checks[decision], 'before': self.before, 'after': after,
            'unchanged': {k: self.before[k] == after[k] for k in self.before},
        })


def run_case(root: Path, device: str, case: str):
    path = root / case
    path.mkdir(parents=True, exist_ok=False)
    set_seed(123)  # Before model construction, identically for all cases.
    terminal = case in ('terminal_rejection', 'tracker_error')
    injections = {1: float('inf'), 3: float('nan')} if case == 'overflow_recovery' else {}
    skips = {1, 3} if case == 'backward_skip' else ({4} if injections else set())
    steps = 6 if injections else (4 if case in ('sampled', 'backward_skip') else 3)
    tracker = None
    if case in ('sampled', 'terminal_rejection', 'tracker_error'):
        cls = BrokenTracker if case == 'tracker_error' else SampledLaggedLERTracker
        tracker = cls(parameter_sample_size=4, sample_seed=7)
    policy = ScriptedPolicy(skips) if skips else AlwaysFalsePolicy()
    if case == 'tracker_error':
        policy = LERGuidedStratifiedPolicy(tracker, total_steps=steps, target_skip_rate=0.0, min_step=0)
    model = TinyModel(injections, 1 if case == 'terminal_rejection' else None)
    args = TrainingArguments(
        output_dir=str(path), max_steps=steps, per_device_train_batch_size=1,
        gradient_accumulation_steps=1, learning_rate=1e-2, warmup_steps=0,
        save_strategy='no', report_to='none', logging_strategy='no',
        remove_unused_columns=False, use_cpu=device == 'cpu', fp16=device == 'cuda',
        dataloader_num_workers=0, disable_tqdm=True, seed=123,
    )
    t = InspectTrainer(
        model=model, args=args, train_dataset=TinyDataset(), skip_policy=policy,
        online_ler_mode='sampled_lagged' if tracker is not None else 'off',
        capture_logits=tracker is not None,
    )
    if tracker is not None:
        t.ler_tracker = tracker
    # A low starting scale isolates injected overflow from startup calibration.
    if device == 'cuda':
        t.accelerator.scaler = torch.cuda.amp.GradScaler(init_scale=128, growth_interval=100)
    else:
        t.accelerator.scaler = torch.amp.GradScaler('cpu', init_scale=128, growth_interval=100)
    trace = StepTrace(t)
    t.callback_handler.callbacks.insert(0, trace)
    error = None
    try:
        t.train()
    except Exception as exc:
        error = exc
    counts = t.get_instrumentation()
    failures = []

    def check(condition, message):
        if not condition:
            failures.append(message)

    check(not t._wrappers_installed and t._optimizer_success_hook is None, 'wrappers/hooks not restored')
    check(trace.original_methods == (t.optimizer.step, t.lr_scheduler.step, t.accelerator.scaler.step), 'method bindings not restored')
    check(counts['optimizer_step_attempts'] == counts['optimizer_step_successes'] + counts['optimizer_steps_skipped_amp'], 'attempt/update/overflow counts disagree')
    check(counts['scheduler_step_calls'] == counts['optimizer_step_successes'], 'scheduler advanced without a successful optimizer step')
    if terminal:
        check(isinstance(error, ObservationFailureError), 'expected terminal observation error')
        check(counts['run_invalidated'], 'terminal attempt not invalidated')
        check(counts['backward_calls'] == 2 and counts['optimizer_step_successes'] == 1, 'terminal finite prefix wrong')
        check(t.before_rejection == snapshot(t), 'pending action changed state after rejection')
        check(counts['online_ler_update_attempts'] == 2 and counts['online_ler_update_successes'] == 1, 'tracker attempts/successes wrong')
        check(not getattr(error, 'evidence_write_failures', []), 'terminal evidence not fully exported')
        forwards = model.forward_count
        try:
            t.train()
        except RuntimeError as exc:
            check('invalidated' in str(exc), 'invalidated reuse raised wrong error')
        else:
            check(False, 'invalidated trainer was reused')
        check(model.forward_count == forwards, 'reuse performed another forward')
    else:
        check(error is None, f'unexpected training exception: {error!r}')
        check(not counts['run_invalidated'], 'finite/overflow attempt invalidated')
        check(model.forward_count == steps, 'incorrect forward horizon')
        check(counts['skipped_backward_steps'] == len(skips), 'intentional skip count wrong')
        check(counts['optimizer_steps_skipped_amp'] == len(injections), 'AMP suppression count wrong')
        check(counts['optimizer_step_successes'] == steps - len(skips) - len(injections), 'successful step count wrong')
    for row in trace.rows:
        decision = row['decision']
        if decision in skips:
            check(all(row['unchanged'].values()), f'D{decision}: intentional skip changed state')
            check(row['gradient']['grad_finite'] is None, f'D{decision}: empty gradients misreported as finite')
        elif decision in injections:
            check(row['gradient']['grad_finite'] is False, f'D{decision}: injected nonfinite gradient missed')
            check(all(row['unchanged'][k] for k in ('model', 'optimizer', 'scheduler')), f'D{decision}: overflow changed protected state')
            check(row['after']['scaler']['scale'] < row['before']['scaler']['scale'], f'D{decision}: scaler did not back off')
        else:
            check(row['deltas']['optimizer_step_successes'] == 1, f'D{decision}: finite recovery did not update')
            check(not row['unchanged']['model'], f'D{decision}: expected nonzero parameter update')
    if case == 'sampled':
        check(counts['online_ler_update_successes'] == steps == tracker.n_updates, 'sampled observations not committed')
        check(tracker.valid_emissions > 0, 'no valid sampled emissions')
    if device == 'cuda':
        check(all(row['cuda_autocast'] and row['logits_dtype'] == 'torch.float16' for row in model.observations), 'native CUDA fp16 autocast not exercised')
    result = {
        'case': case, 'validation_status': 'passed' if not failures else 'failed',
        'run_status': 'expected_terminal_rejection' if terminal and isinstance(error, ObservationFailureError) else ('failed' if error else 'completed'),
        'device': device, 'native_cuda_amp_tested': device == 'cuda',
        'synthetic_fault_injection': strict_value(injections),
        'error_type': type(error).__name__ if error else None,
        'cause_type': type(error.__cause__).__name__ if error and error.__cause__ else None,
        'failures': failures, 'instrumentation': counts, 'steps': trace.rows,
        'forward_observations': model.observations,
        'tracker': tracker.get_diagnostics() if tracker is not None else None,
        'before_rejection': t.before_rejection, 'final_state': snapshot(t),
    }
    write_json(path / 'result.json', result)
    if failures:
        raise AssertionError(f'{case}: ' + '; '.join(failures))
    return result


def main(args):
    repo = args.repo.resolve()
    imported = Path(inspect.getfile(TrueBackwardSkippingTrainer)).resolve()
    if imported != repo / 'lerna/trainers/true_skip_trainer.py':
        raise RuntimeError(f'Wrong trainer imported: {imported}')
    source_hash = hashlib.sha256(imported.read_bytes()).hexdigest()
    if args.expected_trainer_sha256 and source_hash != args.expected_trainer_sha256:
        raise RuntimeError('Imported trainer SHA-256 differs from expected source')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no native AMP validation performed')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    metadata = {
        'trainer_file': str(imported), 'trainer_sha256': source_hash,
        'torch': torch.__version__, 'transformers': transformers.__version__,
        'accelerate': accelerate.__version__, 'python': platform.python_version(),
        'device': args.device, 'native_cuda_amp_tested': args.device == 'cuda',
        'gpu_name': torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        'cuda_runtime': torch.version.cuda, 'seed': 123,
        'purpose': 'Synthetic correctness checks with explicit fault injection; not timing or MRPC evidence',
    }
    for command, key in ((['git', 'rev-parse', 'HEAD'], 'local_git_head'), (['git', 'status', '--porcelain'], 'git_status')):
        response = subprocess.run(command, cwd=repo, capture_output=True, text=True)
        metadata[key] = response.stdout.strip() if response.returncode == 0 else None
    write_json(output / 'metadata.json', metadata)
    results = {}
    for case in CASES:
        try:
            run_case(output, args.device, case)
            results[case] = 'passed'
        except Exception as exc:
            results[case] = f'failed: {exc}'
        print(f'{case}: {results[case]}', flush=True)
    passed = all(v == 'passed' for v in results.values())
    write_json(output / 'summary.json', {'passed': passed, 'cases': results, **metadata})
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main(CLI))
