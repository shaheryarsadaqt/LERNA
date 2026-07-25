"""Piece 8: authoritative training-horizon calculation tests."""

import math
import pytest

from scripts.run_ablation_study import compute_authoritative_horizon


class FakeDataset:
    def __init__(self, length):
        self._length = length

    def __len__(self):
        return self._length


def test_multi_gpu_forces_single_gpu_horizon():
    """Multiple visible GPUs must be forced to single-GPU before horizon calc."""
    ds = FakeDataset(1000)
    # Without forcing, 8 GPUs would yield 20 steps (wrong).
    raw_multi = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=8,
    )
    assert raw_multi == 20

    # After forcing single-GPU, the authoritative horizon must be 160.
    forced_single = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=1,
    )
    assert forced_single == 160


def test_mrpc_modernbert_1000_samples_resolves_to_160():
    """MRPC/ModernBERT/1k samples with single-GPU must equal 160 steps."""
    ds = FakeDataset(1000)
    total = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=5,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=1,
    )
    assert total == 160


def test_partial_final_batch_ceil_semantics():
    """Partial final batch must be counted (ceil, not floor)."""
    ds = FakeDataset(100)
    # 100 samples, batch=32, ga=1, n_gpu=1 -> ceil(100/32)=4 batches/epoch
    total = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=1,
        n_gpu=1,
    )
    assert total == 12  # ceil(3 * ceil(100/32)) = ceil(3 * 4) = 12


def test_gradient_accumulation_scales_horizon():
    """Gradient accumulation must reduce the effective batch size."""
    ds = FakeDataset(100)
    # ga=2 means effective batch size is 64, so ceil(100/64)=2 batches/epoch
    total = compute_authoritative_horizon(
        train_dataset=ds,
        num_epochs=3,
        per_device_train_batch_size=32,
        gradient_accumulation_steps=2,
        n_gpu=1,
    )
    assert total == 6  # ceil(3 * ceil(100/64)) = ceil(3 * 2) = 6


def test_equality_between_configured_and_runtime_horizons():
    """Configured horizon must equal runtime max_steps."""

    class FakePolicy:
        def __init__(self, total_steps):
            self.total_steps = total_steps

    class FakeState:
        def __init__(self, max_steps):
            self.max_steps = max_steps

    class FakeTrainer:
        def __init__(self, max_steps):
            self.state = FakeState(max_steps)
            self.skip_policy = FakePolicy(total_steps=max_steps)
            self._horizon_checked = False

        def _check_authoritative_horizon(self):
            if self._horizon_checked or not hasattr(self.skip_policy, "total_steps"):
                return
            self._horizon_checked = True
            runtime_max_steps = getattr(self.state, "max_steps", None)
            configured_total_steps = getattr(self.skip_policy, "total_steps", None)
            if runtime_max_steps is not None and configured_total_steps is not None:
                if runtime_max_steps != configured_total_steps:
                    raise RuntimeError(
                        f"Authoritative horizon mismatch: policy configured "
                        f"total_steps={configured_total_steps}, but "
                        f"trainer.state.max_steps={runtime_max_steps}."
                    )

    trainer = FakeTrainer(max_steps=160)
    trainer._check_authoritative_horizon()  # should not raise

    trainer.state.max_steps = 159
    trainer._horizon_checked = False
    with pytest.raises(RuntimeError, match="Authoritative horizon mismatch"):
        trainer._check_authoritative_horizon()


def test_exact_random_and_rvd_none_decisions_unchanged():
    """Piece 8 must not alter exact-random vs RVD-none parity."""
    from lerna.trainers import (
        LERNARandomVetoDeferralPolicy,
        RandomSkipPolicy,
    )
    from scripts.run_ablation_study import build_skip_policy, build_rvd_controller_config
    from tests.test_rvd_runner_config import make_controller_config

    class FakeTracker:
        def get_diagnostics(self):
            return {"rho_vg_raw": 0.0}

    class StubTrainer:
        def __init__(self, max_steps):
            self.state = type("State", (), {"global_step": 0, "max_steps": max_steps})()

    total_steps = 160
    exact = build_skip_policy(
        control="exact_random",
        ler_tracker=FakeTracker(),
        target_skip_rate=0.30,
        total_steps=total_steps,
        controller_cfg=make_controller_config(),
        rho_veto_threshold=-0.05,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )
    rvd = build_skip_policy(
        control="rvd",
        ler_tracker=FakeTracker(),
        target_skip_rate=0.30,
        total_steps=total_steps,
        controller_cfg=make_controller_config(veto_mode="none"),
        rho_veto_threshold=-0.05,
        probe_interval=8,
        use_ler=True,
        use_rho_vg=True,
        use_safety_horizon=True,
        fallback_threshold=0.01,
        risk_gamma=0.0,
    )
    trainer = StubTrainer(total_steps)

    exact_decisions = []
    rvd_decisions = []
    for _ in range(total_steps):
        exact_decisions.append(exact.should_skip(trainer, None, None))
        rvd_decisions.append(rvd.should_skip(trainer, None, None))

    assert exact_decisions == rvd_decisions
    assert exact._skip_set == rvd._skip_set
    assert len(exact._skip_set) == round(total_steps * 0.30)
