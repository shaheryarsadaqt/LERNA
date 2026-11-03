"""Integration test for terminal invalid observation contract.

Exercises the complete failure path:
1. Tracker pre-clamp validation rejects non-finite loss
2. Trainer flushes instrumentation + ler_diagnostics before raising ObservationFailureError
3. Exception chain preserved (ObservationFailureError -> TerminalInvalidObservation)
4. No optimizer/scaler/scheduler action after terminal invalidity
5. Failed manifest published via finalize_manifest_failed
"""

import os
import json
import tempfile
import shutil
import torch
import torch.nn as nn
from unittest.mock import patch, MagicMock

from lerna.utils.lagged_ler import SampledLaggedLERTracker, TerminalInvalidObservation
from lerna.trainers.true_skip_trainer import (
    TrueBackwardSkippingTrainer,
    ObservationFailureError,
    ONLINE_LER_MODE_SAMPLED_LAGGED,
)
from lerna.trainers.policies import LERGuidedStratifiedPolicy
from lerna.utils.run_provenance import finalize_manifest_failed, verify_completed_manifest


class TerminalLossModel(nn.Module):
    """Model that returns NaN loss on forward pass after a threshold."""
    def __init__(self, fail_at_step=1):
        super().__init__()
        self.linear = nn.Linear(10, 2)
        self.step = 0
        self.fail_at_step = fail_at_step

    def forward(self, input_ids, labels=None, **kwargs):
        self.step += 1
        logits = self.linear(input_ids)
        loss = None
        if labels is not None:
            if self.step >= self.fail_at_step:
                # Return NaN loss that requires grad (multiply parameter by NaN)
                loss = self.linear.weight.sum() * float('nan')
            else:
                loss = torch.nn.functional.cross_entropy(logits, labels)
        # Return dict instead of custom object to avoid DataParallel gather issues
        return {'loss': loss, 'logits': logits}


def test_tracker_terminal_invalid_observation():
    """Tracker rejects non-finite loss with tagged operands and valid prefix."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tracker = SampledLaggedLERTracker(
        task='test', window_size=5, parameter_sample_size=100, sample_seed=42
    )
    tracker.set_optimizer(optimizer)

    logits = torch.randn(4, 2)
    try:
        tracker.update(loss=float('nan'), logits=logits, model=model, optimizer=optimizer, decision_index=0)
        assert False, "Expected TerminalInvalidObservation"
    except TerminalInvalidObservation as e:
        assert e.event_type == "nonfinite_loss"
        assert e.decision_index == 0
        assert 'loss' in e.operands
        assert e.operands['loss']['kind'] == 'nonfinite'
        assert e.operands['loss']['value'] == 'NaN'
        # Valid prefix has finite histories
        assert 'ler_raw_history' in e.valid_prefix
        assert 'loss_history' in e.valid_prefix
        assert all(isinstance(v, (int, float)) and not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')))
                       for v in e.valid_prefix['loss_history'])
        # Tracker recorded terminal event
        assert tracker.terminal_invalid_event is not None
        assert tracker.terminal_invalid_event['event_type'] == 'nonfinite_loss'
        assert tracker.invalid_emission_events == 1


def test_trainer_terminal_failure_path():
    """Trainer flushes instrumentation and diagnostics, raises ObservationFailureError."""
    temp_dir = tempfile.mkdtemp()
    try:
        model = TerminalLossModel(fail_at_step=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        from transformers import TrainingArguments
        args = TrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=2,
            max_steps=10,
            logging_steps=1,
            report_to='none',
            save_strategy='no',
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
        )

        # Create a minimal dataset
        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, idx):
                return {
                    'input_ids': torch.randn(10),
                    'labels': torch.tensor(idx % 2),
                }

        trainer = TrueBackwardSkippingTrainer(
            model=model,
            args=args,
            train_dataset=DummyDataset(),
            optimizers=(optimizer, None),
            online_ler_mode=ONLINE_LER_MODE_SAMPLED_LAGGED,
            online_ler_update_interval=1,
            capture_logits=True,
        )

        # Attach tracker
        tracker = SampledLaggedLERTracker(
            task='test', window_size=5, parameter_sample_size=100, sample_seed=42
        )
        tracker.set_optimizer(optimizer)
        trainer.ler_tracker = tracker

        # Run training - should raise ObservationFailureError
        try:
            trainer.train()
            assert False, "Expected ObservationFailureError"
        except ObservationFailureError as e:
            # Verify exception chain
            assert isinstance(e.__cause__, TerminalInvalidObservation)
            assert e.tracker_event is not None
            assert e.tracker_event['event_type'] == 'nonfinite_loss'
            assert e.valid_prefix is not None
            assert 'ler_raw_history' in e.valid_prefix

            # Verify instrumentation persisted
            instr_path = os.path.join(temp_dir, 'instrumentation.json')
            assert os.path.exists(instr_path), "instrumentation.json not written"
            with open(instr_path) as f:
                instr = json.load(f)
            assert 'batches_seen' in instr

            # Verify ler_diagnostics persisted with terminal event
            diag_path = os.path.join(temp_dir, 'ler_diagnostics.json')
            assert os.path.exists(diag_path), "ler_diagnostics.json not written"
            with open(diag_path) as f:
                diag = json.load(f)
            assert diag['terminal_invalid_event'] is not None
            assert diag['terminal_invalid_event']['event_type'] == 'nonfinite_loss'
            assert diag['terminal_invalid_event']['operands']['loss']['kind'] == 'nonfinite'
            assert diag['terminal_invalid_event']['operands']['loss']['value'] == 'NaN'
            assert diag['valid_prefix'] is not None
            # Finite histories in valid prefix
            for key in ['ler_raw_history', 'ler_history', 'loss_history', 'entropy_history']:
                for v in diag['valid_prefix'].get(key, []):
                    assert isinstance(v, (int, float))
                    assert not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')))

            # Verify final accepted values preserved (not null)
            assert diag['ler'] is not None or diag['ler_raw_history'] == []  # warmup may have no emissions
            assert diag['ler_raw'] is not None or diag['ler_raw_history'] == []

            # Verify optimizer/scheduler NOT called after terminal event by checking
            # that instrumentation shows no completed steps (batches_seen should be 0 or 1)
            # The terminal event happens in pre-decision update before backward/optimizer
            assert instr['batches_seen'] <= 1
        except Exception as e:
            assert False, f"Wrong exception type: {type(e).__name__}: {e}"

    finally:
        shutil.rmtree(temp_dir)


def test_policy_terminal_guard():
    """LERGuidedStratifiedPolicy raises on terminal invalid observation.

    Returning False would select real backward and continue training, which
    violates the terminal contract. The policy must raise so the trainer
    marks the run invalidated and flushes evidence before propagating.
    """
    from lerna.trainers.policies import TerminalInvalidPolicyError

    model = nn.Linear(10, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    tracker = SampledLaggedLERTracker(
        task='test', window_size=5, parameter_sample_size=100, sample_seed=42
    )
    tracker.set_optimizer(optimizer)

    # Inject terminal event
    logits = torch.randn(4, 2)
    try:
        tracker.update(loss=float('nan'), logits=logits, model=model, optimizer=optimizer, decision_index=0)
    except TerminalInvalidObservation:
        pass

    policy = LERGuidedStratifiedPolicy(
        ler_tracker=tracker,
        min_step=0,
        target_skip_rate=0.2,
        total_steps=100,
        probe_interval=8,
        ler_guidance_strength=1.0,
    )

    # Create mock trainer
    mock_trainer = MagicMock()
    mock_trainer.state.global_step = 1
    mock_trainer._pre_clip_grad_norm = 0.5

    # Policy must raise TerminalInvalidPolicyError, not return False
    try:
        policy.should_skip(mock_trainer, model, {'input_ids': torch.randn(2, 10), 'labels': torch.tensor([0, 1])})
        assert False, "Expected TerminalInvalidPolicyError"
    except TerminalInvalidPolicyError as e:
        assert e.policy_name == 'ler_guided_stratified'
        assert e.decision_index == 0
        assert e.terminal_event is not None
        assert e.terminal_event['event_type'] == 'nonfinite_loss'
    assert policy._terminal_invalid_observed is True


def test_failed_manifest_publication():
    """Failed manifest is published with correct error info."""
    from lerna.utils.run_provenance import write_manifest_running, DEFAULT_ARTIFACT_FILENAMES, CLASSIFICATION_LOCAL_DEVELOPMENT
    temp_dir = tempfile.mkdtemp()
    try:
        # Simulate a terminal failure exception
        model = nn.Linear(10, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tracker = SampledLaggedLERTracker(task='test', window_size=5, parameter_sample_size=100, sample_seed=42)
        tracker.set_optimizer(optimizer)
        logits = torch.randn(4, 2)
        try:
            tracker.update(loss=float('nan'), logits=logits, model=model, optimizer=optimizer, decision_index=0)
        except TerminalInvalidObservation as tio:
            ofe = ObservationFailureError(
                tracker_event=tracker.terminal_invalid_event,
                valid_prefix=tio.valid_prefix,
                original_exception=tio,
            )
            ofe.__cause__ = tio

            # Create a running manifest first (required by finalize_manifest_failed)
            # Use local_development classification to avoid git dirty check
            write_manifest_running(
                run_dir=temp_dir,
                argv=['test'],
                task='test',
                model_id='test-model',
                model_revision='rev1',
                seed=42,
                controller_name='test_controller',
                controller_seed=42,
                target_skip_rate=0.2,
                planned_quota=20,
                total_steps=100,
                warmup_steps=10,
                skip_update_mode='freeze',
                controller_config_effective={'policy_class': 'LERGuidedStratifiedPolicy'},
                matched_budget_planned=True,
                budget_classification='matched_budget',
                output_paths={
                    'results': 'results.json',
                    'instrumentation': 'instrumentation.json',
                    'manifest': 'run_manifest.json',
                    'ler_diagnostics': 'ler_diagnostics.json',
                },
                requested_classification=CLASSIFICATION_LOCAL_DEVELOPMENT,
                attempt=1,
            )

            # Call finalize_manifest_failed
            finalize_manifest_failed(temp_dir, ofe)

            # Verify failed manifest exists
            manifest_path = os.path.join(temp_dir, 'run_manifest.json')
            assert os.path.exists(manifest_path), "Failed manifest not written"

            with open(manifest_path) as f:
                manifest = json.load(f)

            assert manifest['status'] == 'failed'
            assert 'error' in manifest
            assert manifest['error']['type'] == 'ObservationFailureError'
            # Error message is intentionally generic (security - no exception secrets serialized)
            assert manifest['error']['message'] == 'run failed; inspect local logs for details'
            assert manifest['provenance_classification'] is not None

            # Verify verification fails (not completed)
            result = verify_completed_manifest(temp_dir)
            assert result['ok'] is False
    finally:
        shutil.rmtree(temp_dir)


def test_no_pending_action_after_terminal():
    """Verify no optimizer/scaler/scheduler action after terminal invalidity.

    For sampled_lagged timing, the observation happens POST backward pass,
    so backward_calls may be > 0. The key invariant is that no optimizer/
    scaler/scheduler step occurs after the terminal invalid observation.
    """
    temp_dir = tempfile.mkdtemp()
    try:
        model = TerminalLossModel(fail_at_step=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        from transformers import TrainingArguments
        args = TrainingArguments(
            output_dir=temp_dir,
            per_device_train_batch_size=2,
            max_steps=10,
            logging_steps=1,
            report_to='none',
            save_strategy='no',
            remove_unused_columns=False,
            gradient_accumulation_steps=1,
        )

        class DummyDataset(torch.utils.data.Dataset):
            def __len__(self): return 4
            def __getitem__(self, idx):
                return {'input_ids': torch.randn(10), 'labels': torch.tensor(idx % 2)}

        trainer = TrueBackwardSkippingTrainer(
            model=model,
            args=args,
            train_dataset=DummyDataset(),
            optimizers=(optimizer, None),
            online_ler_mode=ONLINE_LER_MODE_SAMPLED_LAGGED,
            online_ler_update_interval=1,
            capture_logits=True,
        )

        tracker = SampledLaggedLERTracker(task='test', window_size=5, parameter_sample_size=100, sample_seed=42)
        tracker.set_optimizer(optimizer)
        trainer.ler_tracker = tracker

        try:
            trainer.train()
        except ObservationFailureError:
            pass

        # For sampled_lagged timing, terminal invalidity is detected in
        # _sampled_lagged_update which runs AFTER backward pass.
        # So backward_calls may be 1 (the step that produced the NaN loss).
        # But no optimizer/scaler/scheduler action should occur after.
        instr_path = os.path.join(temp_dir, 'instrumentation.json')
        assert os.path.exists(instr_path), "instrumentation.json not written"
        with open(instr_path) as f:
            instr = json.load(f)
        # batches_seen <= 1 because we only processed one batch before terminal event
        assert instr['batches_seen'] <= 1, f"Expected batches_seen <= 1, got {instr['batches_seen']}"
        # No optimizer step attempted after terminal event
        assert instr['optimizer_step_attempts'] == 0, f"Expected optimizer_step_attempts == 0, got {instr['optimizer_step_attempts']}"
        # No scaler step after terminal event
        assert instr['grad_scaler_step_calls'] == 0, f"Expected grad_scaler_step_calls == 0, got {instr['grad_scaler_step_calls']}"
        # No scheduler step after terminal event
        assert instr['scheduler_step_calls'] == 0, f"Expected scheduler_step_calls == 0, got {instr['scheduler_step_calls']}"
        # Run should be marked invalidated
        assert instr['run_invalidated'] is True, f"Expected run_invalidated == True, got {instr['run_invalidated']}"

        # Also verify ler_diagnostics has terminal event with proper operands
        diag_path = os.path.join(temp_dir, 'ler_diagnostics.json')
        assert os.path.exists(diag_path), "ler_diagnostics.json not written"
        with open(diag_path) as f:
            diag = json.load(f)
        assert diag['terminal_invalid_event'] is not None
        assert diag['terminal_invalid_event']['event_type'] == 'nonfinite_loss'
        assert diag['terminal_invalid_event']['operands']['loss']['kind'] == 'nonfinite'
        assert diag['terminal_invalid_event']['operands']['loss']['value'] == 'NaN'
        assert diag['valid_prefix'] is not None
        # Valid prefix should have finite histories (empty in this case since it fails on first update)
        for key in ['ler_raw_history', 'ler_history', 'loss_history', 'entropy_history']:
            for v in diag['valid_prefix'].get(key, []):
                assert isinstance(v, (int, float))
                assert not (isinstance(v, float) and (v != v or v == float('inf') or v == float('-inf')))

    finally:
        shutil.rmtree(temp_dir)


def test_validator_rejects_terminal_event():
    """Signal integrity validator rejects terminal invalid event."""
    from lerna.utils.phase1_3_completed_matrix import _check_signal_integrity

    temp_dir = tempfile.mkdtemp()
    try:
        attempt_dir = os.path.join(temp_dir, 'attempt-001')
        os.makedirs(attempt_dir)

        # Write ler_diagnostics.json WITH terminal event
        diag = {
            'mode': 'sampled_lagged',
            'timing': 'post_decision_after_backward',
            'parameter_sample_size_requested': 4096,
            'parameter_sample_size_realized': 4096,
            'sampled_tensor_count': 12,
            'n_updates': 1,
            'n_decisions': 10,
            'last_update_decision': 0,
            'observation_age_decisions': 10,
            'ler': None,
            'ler_raw': None,
            'rho_vg': 0.0,
            'rho_vg_raw': 0.0,
            'param_velocity': 0.0,
            'n_velocity_samples': 0,
            'n_rho_vg_samples': 0,
            'full_parameter_snapshots': False,
            'full_vector_concatenation': False,
            'observation_attempts': 1,
            'warmup_updates': 0,
            'valid_emissions': 0,
            'invalid_emission_events': 1,
            'terminal_invalid_event': {
                'event_type': 'nonfinite_loss',
                'decision_index': 0,
                'observation_attempt': 1,
                'operands': {'loss': {'kind': 'nonfinite', 'value': 'NaN'}}
            },
            'last_valid_emission_decision': None,
            'ler_raw_history': [],
            'ler_history': [],
            'velocity_history': [],
            'rho_vg_history': [],
            'loss_history': [],
            'entropy_history': [],
        }
        with open(os.path.join(attempt_dir, 'ler_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        findings = []
        result = _check_signal_integrity(attempt_dir, 'test_cell', 'attempt-001', findings)
        assert result is False, "Expected signal integrity check to fail"
        assert len(findings) == 1
        assert findings[0]['severity'] == 'error'
        assert 'TERMINAL INVALID OBSERVATION' in findings[0]['message']
        assert 'nonfinite_loss' in findings[0]['message']

    finally:
        shutil.rmtree(temp_dir)


def test_schema_validator_rejects_terminal_event():
    """Schema validator rejects terminal invalid event via invalid_emission_events > 0."""
    from lerna.utils.phase1_3_completed_matrix import _validate_ler_diagnostics_schema

    temp_dir = tempfile.mkdtemp()
    try:
        attempt_dir = os.path.join(temp_dir, 'attempt-001')
        os.makedirs(attempt_dir)

        # Write ler_diagnostics.json WITH terminal event
        diag = {
            'mode': 'sampled_lagged',
            'timing': 'post_decision_after_backward',
            'parameter_sample_size_requested': 4096,
            'parameter_sample_size_realized': 4096,
            'sampled_tensor_count': 12,
            'n_updates': 1,
            'n_decisions': 10,
            'last_update_decision': 0,
            'observation_age_decisions': 10,
            'ler': None,
            'ler_raw': None,
            'rho_vg': 0.0,
            'rho_vg_raw': 0.0,
            'param_velocity': 0.0,
            'n_velocity_samples': 0,
            'n_rho_vg_samples': 0,
            'full_parameter_snapshots': False,
            'full_vector_concatenation': False,
            'observation_attempts': 1,
            'warmup_updates': 0,
            'valid_emissions': 0,
            'invalid_emission_events': 1,  # This should trigger rejection
            'terminal_invalid_event': {
                'event_type': 'nonfinite_loss',
                'decision_index': 0,
                'observation_attempt': 1,
                'operands': {'loss': {'kind': 'nonfinite', 'value': 'NaN'}}
            },
            'last_valid_emission_decision': None,
            'ler_raw_history': [],
            'ler_history': [],
            'velocity_history': [],
            'rho_vg_history': [],
            'loss_history': [],
            'entropy_history': [],
        }
        with open(os.path.join(attempt_dir, 'ler_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        findings = []
        result = _validate_ler_diagnostics_schema(attempt_dir, 'test_cell', 'attempt-001', findings)
        assert result is False, "Expected schema validation to fail"
        assert len(findings) >= 1
        # Find the invalid_emission_events error
        invalid_events_findings = [f for f in findings if 'invalid_emission_events' in f['field']]
        assert len(invalid_events_findings) == 1
        assert invalid_events_findings[0]['severity'] == 'error'
        assert 'invalid_emission_events = 1 > 0' in invalid_events_findings[0]['message']

    finally:
        shutil.rmtree(temp_dir)


def test_validator_accepts_valid_diagnostics():
    """Signal integrity validator accepts valid diagnostics."""
    from lerna.utils.phase1_3_completed_matrix import _check_signal_integrity, _validate_ler_diagnostics_schema

    temp_dir = tempfile.mkdtemp()
    try:
        attempt_dir = os.path.join(temp_dir, 'attempt-001')
        os.makedirs(attempt_dir)

        # Valid diagnostics: no terminal event, finite histories, correct counters
        diag = {
            'mode': 'sampled_lagged',
            'timing': 'post_decision_after_backward',
            'parameter_sample_size_requested': 4096,
            'parameter_sample_size_realized': 4096,
            'sampled_tensor_count': 12,
            'n_updates': 5,
            'n_decisions': 50,
            'last_update_decision': 45,
            'observation_age_decisions': 5,
            'ler': 0.011,
            'ler_raw': 0.0125,
            'rho_vg': 0.01,
            'rho_vg_raw': 0.01,
            'param_velocity': 0.001,
            'n_velocity_samples': 5,
            'n_rho_vg_samples': 5,
            'full_parameter_snapshots': False,
            'full_vector_concatenation': False,
            'observation_attempts': 5,
            'warmup_updates': 2,
            'valid_emissions': 3,
            'invalid_emission_events': 0,
            'terminal_invalid_event': None,
            'last_valid_emission_decision': 45,
            'ler_raw_history': [0.01, 0.011, 0.0125],
            'ler_history': [0.01, 0.0105, 0.011],
            'velocity_history': [0.001] * 5,
            'rho_vg_history': [0.01] * 5,
            'loss_history': [0.5, 0.48, 0.47, 0.46, 0.45],
            'entropy_history': [1.0, 1.1, 1.05, 1.08, 1.06],
        }
        with open(os.path.join(attempt_dir, 'ler_diagnostics.json'), 'w') as f:
            json.dump(diag, f)

        # Test signal integrity (should pass - no terminal event)
        findings = []
        result = _check_signal_integrity(attempt_dir, 'test_cell', 'attempt-001', findings)
        assert result is True, f"Expected signal integrity check to pass, got findings: {findings}"
        assert len(findings) == 0

        # Test full schema validation (should pass)
        findings = []
        result = _validate_ler_diagnostics_schema(attempt_dir, 'test_cell', 'attempt-001', findings)
        assert result is True, f"Expected schema validation to pass, got findings: {findings}"
        assert len(findings) == 0

    finally:
        shutil.rmtree(temp_dir)


if __name__ == '__main__':
    import sys
    import pytest
    sys.exit(pytest.main([__file__, '-v']))