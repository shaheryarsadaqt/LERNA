"""F8 integration test: S2 score_source fails hard at step 392.

If score_source == "s2" and the probe isn't attached or the score is non-finite
at min_step, the policy must raise (not silently degrade).
"""
import pytest
import torch
import torch.nn as nn
from unittest import mock

from lerna.utils.forward_gate import (
    ScoreQuantileGate,
    ScoreQuantileGatePolicy,
    TerminalInvalidScoreError,
    InfeasibleBudgetError,
    AlignmentProbe,
)


def test_s2_score_source_requires_probe_fn():
    """Policy with score_source='s2' but no probe_fn should raise at min_step."""
    class _FakeInputs:
        def get(self, key, default=None):
            if key == "labels":
                return torch.tensor([0, 1])
            return default

    policy = ScoreQuantileGatePolicy(
        target_skip_rate=0.60,
        total_steps=6548,
        min_step=392,
        score_source="s2",
        alignment_probe_fn=None,  # F8: probe_fn is None
    )

    trainer = mock.Mock()
    trainer.instr = mock.Mock(batches_seen=393)
    trainer._alignment_probe = mock.Mock()
    trainer._alignment_probe.last_hidden = None

    inputs = _FakeInputs()
    model = mock.Mock()

    with pytest.raises((RuntimeError, ValueError, AssertionError, AttributeError)) as exc_info:
        policy.should_skip(trainer, model, inputs)
    assert "AlignmentProbe" in str(exc_info.value) or "probe_fn" in str(exc_info.value).lower()


def test_s2_nonfinite_score_raises():
    """Non-finite S2 score should trigger TerminalInvalidScoreError."""
    gate = ScoreQuantileGate(
        rate=0.60, total_steps=6548, min_step=392, n_phases=4,
        max_consecutive_skips=4, seed=7,
        score_source=lambda i, m, x: float("nan"),
    )
    with pytest.raises(TerminalInvalidScoreError):
        for i in range(393):
            gate.decide(i, None, None)


def test_s2_score_finite_with_attached_probe():
    """When probe is attached and score returns finite, gate should work."""
    policy = ScoreQuantileGatePolicy(
        target_skip_rate=0.60,
        total_steps=6548,
        min_step=392,
        score_source="s2",
        alignment_probe_fn=mock.Mock(return_value=0.5),
    )
    trainer = mock.Mock()
    trainer.instr = mock.Mock(batches_seen=393)
    probe = mock.Mock()
    probe.last_hidden = torch.randn(4, 768)
    trainer._alignment_probe = probe
    trainer._last_real_logits = torch.randn(4, 2)
    policy._trainer_ref = trainer

    with mock.patch("lerna.utils.forward_gate.alignment_score", return_value=0.5):
        # Should not raise (probe attached, score finite)
        result = policy.should_skip(trainer, mock.Mock(), {"labels": torch.tensor([0, 1])})
        assert isinstance(result, bool)
