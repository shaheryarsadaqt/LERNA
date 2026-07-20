"""REVISED — tests/test_true_skip_trainer.py

Changes from original v1:

    [CRIT-1] Test that scheduler does NOT advance on skipped steps (the primary scheduler-flag-bug test)
    [IMP-4] Uses optimizer_step_attempts instead of optimizer_step_calls/real_optimizer_steps
    All inequality invariants per [FIX #4]

"""
import copy
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import TrainerCallback, TrainingArguments

from lerna.trainers import (
    TrueBackwardSkippingTrainer, LERNAMomentumTrainer,
    normalize_skip_update_mode, AlwaysFalsePolicy, RandomSkipPolicy, SkipPolicy,
)
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


def _tiny_args(td, epochs=2, seed=7):
    return TrainingArguments(
        output_dir=td,
        num_train_epochs=epochs,
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
        seed=seed,        # identical dataloader shuffling across paired runs
    )


class _ScriptedSkipPolicy(SkipPolicy):
    """Deterministic: first  decisions are real, all later ones skip.
    Guarantees optimizer state exists before the first skipped step."""
    name = "scripted"

    def __init__(self, real_steps=2):
        self.real_steps = real_steps
        self._calls = 0

    def should_skip(self, *args, **kwargs):
        self._calls += 1
        return self._calls > self.real_steps


class _WeightsSnapshotUntilFirstSkip(TrainerCallback):
    """Snapshot of weights at the end of the last step before any skip."""
    def __init__(self):
        self.trainer = None
        self.snapshot = None
        self.opt_state_snapshot = None
        self.real_steps_before_first_skip = 0

    def on_step_end(self, args, state, control, model=None, **kwargs):
        t = self.trainer
        if t is not None and t.instr.skipped_backward_steps == 0:
            self.snapshot = {
                n: p.detach().clone() for n, p in model.named_parameters()
            }
            self.real_steps_before_first_skip = t.instr.optimizer_step_attempts
            if t.optimizer is not None:
                self.opt_state_snapshot = {}
                for group in t.optimizer.param_groups:
                    for p in group["params"]:
                        if p in t.optimizer.state:
                            self.opt_state_snapshot[id(p)] = {
                                k: (v.clone() if torch.is_tensor(v) else copy.deepcopy(v))
                                for k, v in t.optimizer.state[p].items()
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


def test_momentum_extrapolation_weights_frozen_when_disabled():
    """[FIX P2-2] With apply_momentum=False, weights must be byte-identical before/after a skipped step."""
    from lerna.trainers import LERNAMomentumTrainer

    class NoMomentumTrainer(LERNAMomentumTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, apply_momentum=False, **kwargs)

    with tempfile.TemporaryDirectory() as td:
        model = _TinyModel()
        w0 = model.fc.weight.data.clone()
        b0 = model.fc.bias.data.clone()

        args = TrainingArguments(
            output_dir=td,
            num_train_epochs=1,
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
        trainer = NoMomentumTrainer(
            model=model,
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
            skip_policy=RandomSkipPolicy(target_skip_rate=1.0, min_step=0, seed=0),
            instrumentation_path=os.path.join(td, "instrumentation.json"),
        )
        trainer.train()
        i = trainer.get_instrumentation()

        w1 = model.fc.weight.data.clone()
        b1 = model.fc.bias.data.clone()

        assert torch.equal(w0, w1), (
            f"Weights changed on fully-skipped run with apply_momentum=False"
        )

def test_momentum_extrapolation_preserves_skip_accounting():
    """Verify momentum-mode skipping preserves forward/backward accounting."""
    with tempfile.TemporaryDirectory() as td:
        model = _TinyModel()
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
        trainer = LERNAMomentumTrainer(
            model=model,
            args=args,
            train_dataset=_TinyDS(),
            data_collator=_collate,
            skip_policy=RandomSkipPolicy(target_skip_rate=0.5, min_step=0, seed=42),
            apply_momentum=True,
            instrumentation_path=os.path.join(td, "instrumentation.json"),
        )
        trainer.train()
        i = trainer.get_instrumentation()
        assert i["skipped_backward_steps"] > 0, (
            "No steps skipped; cannot verify momentum extrapolation."
        )
        assert i["forward_calls"] == i["backward_calls"] + i["skipped_backward_steps"]




def test_default_mode_normalizes_to_freeze():
    assert normalize_skip_update_mode() == ("freeze", False)
    assert normalize_skip_update_mode(explicit_mode="momentum") == ("momentum", False)
    assert normalize_skip_update_mode(legacy_use_momentum_extrap=True) == ("momentum", True)
    assert normalize_skip_update_mode(legacy_use_momentum_extrap=False) == ("freeze", True)
    with pytest.raises(ValueError):
        normalize_skip_update_mode(explicit_mode="freeze", legacy_use_momentum_extrap=True)
    with pytest.raises(ValueError):
        normalize_skip_update_mode(explicit_mode="bogus")


def test_trainer_default_is_freeze():
    with tempfile.TemporaryDirectory() as td:
        t = LERNAMomentumTrainer(
            model=_TinyModel(), args=_tiny_args(td),
            train_dataset=_TinyDS(), data_collator=_collate,
            skip_policy=AlwaysFalsePolicy(),
            instrumentation_path=os.path.join(td, "instrumentation.json"),
        )
        assert t.skip_update_mode == "freeze"
        assert t.apply_momentum is False
        assert t.instr.skip_update_mode == "freeze"
        assert t.instr.parameters_may_change_on_skipped_step is False
        assert "freeze" in t.instr.skip_update_mechanism


def test_conflicting_explicit_and_legacy_rejected():
    with tempfile.TemporaryDirectory() as td:
        with pytest.raises(ValueError):
            LERNAMomentumTrainer(
                model=_TinyModel(), args=_tiny_args(td),
                train_dataset=_TinyDS(), data_collator=_collate,
                skip_policy=AlwaysFalsePolicy(),
                skip_update_mode="freeze", apply_momentum=True,
                instrumentation_path=os.path.join(td, "instrumentation.json"),
            )


def _run_mode(mode, model, ds, td):
    snap_cb = _WeightsSnapshotUntilFirstSkip()
    trainer = LERNAMomentumTrainer(
        model=model, args=_tiny_args(td),
        train_dataset=ds, data_collator=_collate,
        skip_policy=_ScriptedSkipPolicy(real_steps=2),
        skip_update_mode=mode,
        instrumentation_path=os.path.join(td, "instrumentation.json"),
        callbacks=[snap_cb],
    )
    snap_cb.trainer = trainer
    trainer.train()
    return trainer, snap_cb


def test_freeze_and_momentum_paired_runs():
    """Freeze and momentum runs start from IDENTICAL init weights, data,
    optimizer config, and policy decisions."""
    torch.manual_seed(1234)
    ref_model = _TinyModel()
    ref_ds = _TinyDS()

    model_f = copy.deepcopy(ref_model)
    model_m = copy.deepcopy(ref_model)
    ds_f = copy.deepcopy(ref_ds)
    ds_m = copy.deepcopy(ref_ds)

    with tempfile.TemporaryDirectory() as td_f, tempfile.TemporaryDirectory() as td_m:
        t_f, snap_f = _run_mode("freeze", model_f, ds_f, td_f)
        t_m, snap_m = _run_mode("momentum", model_m, ds_m, td_m)

        i_f, i_m = t_f.get_instrumentation(), t_m.get_instrumentation()

        # Real optimizer step happened BEFORE the first skip.
        for i in (i_f, i_m):
            assert i["optimizer_step_attempts"] >= 1
            assert i["skipped_backward_steps"] >= 1
        assert snap_f.real_steps_before_first_skip >= 1
        assert snap_m.real_steps_before_first_skip >= 1

        # Equal policy decisions: same scripted policy, same number of steps.
        assert i_f["forward_calls"] == i_m["forward_calls"]
        assert i_f["backward_calls"] == i_m["backward_calls"]
        assert i_f["skipped_backward_steps"] == i_m["skipped_backward_steps"]

        # Derived boolean matches mode.
        assert t_f.apply_momentum is False
        assert t_m.apply_momentum is True

        # (2) freeze: byte-identical across all skipped steps.
        assert i_f["skip_update_mode"] == "freeze"
        assert i_f["parameters_may_change_on_skipped_step"] is False
        for n, p in model_f.named_parameters():
            assert torch.equal(snap_f.snapshot[n], p.detach()), (
                f"{n} changed on a skipped step in freeze mode"
            )
        assert snap_f.opt_state_snapshot is not None, (
            "optimizer state snapshot was not captured"
        )
        assert len(snap_f.opt_state_snapshot) > 0, (
            "optimizer state snapshot is empty"
        )
        opt = t_f.optimizer
        live_ids = {
            id(p)
            for group in opt.param_groups
            for p in group["params"]
            if p in opt.state
        }
        assert set(snap_f.opt_state_snapshot.keys()) == live_ids, (
            "optimizer state parameter ID set changed on skipped step"
        )
        for param_id, snap_state in snap_f.opt_state_snapshot.items():
            live_state = opt.state[next(p for group in opt.param_groups for p in group["params"] if id(p) == param_id)]
            assert set(snap_state.keys()) == set(live_state.keys()), (
                "optimizer state keys changed on skipped step"
            )
            for k, snap_v in snap_state.items():
                live_v = live_state[k]
                if torch.is_tensor(snap_v):
                    assert torch.equal(snap_v, live_v), (
                        f"optimizer state '{k}' changed on skipped step"
                    )
                else:
                    assert snap_v == live_v

        # (4) momentum: weights change on skipped steps with valid opt state.
        assert i_m["skip_update_mode"] == "momentum"
        assert i_m["parameters_may_change_on_skipped_step"] is True
        assert any(
            not torch.equal(snap_m.snapshot[n], p.detach())
            for n, p in model_m.named_parameters()
        ), "momentum mode did not change weights on skipped steps"

        # Runs diverge overall.
        total_diff = sum(
            (pf - pm).abs().sum().item()
            for pf, pm in zip(model_f.parameters(), model_m.parameters())
        )
        assert total_diff > 0


class _DeltaProbeTrainer(LERNAMomentumTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.delta_checks = []

    def on_skipped_backward_step(self, loss, model, inputs):
        opt = self.optimizer
        expected = {}
        if opt is not None:
            with torch.no_grad():
                for group in opt.param_groups:
                    lr = group.get("lr", self.args.learning_rate)
                    beta1, beta2 = group.get("betas", (0.9, 0.999))
                    eps = group.get("eps", 1e-8)
                    wd = group.get("weight_decay", 0.0)
                    for p in group["params"]:
                        if not p.requires_grad or p not in opt.state:
                            continue
                        st = opt.state[p]
                        base = p.detach().clone()
                        buf = st.get("momentum_buffer")
                        if buf is not None:
                            expected[id(p)] = base.add(buf, alpha=-lr)
                            continue
                        m, v = st.get("exp_avg"), st.get("exp_avg_sq")
                        if m is None or v is None:
                            continue
                        step = st.get("step", 1)
                        tt = float(step.item() if torch.is_tensor(step) else step)
                        tt = max(tt, 1.0)
                        m_hat = m / (1.0 - beta1 ** tt)
                        v_hat = v / (1.0 - beta2 ** tt)
                        exp = base.clone()
                        if wd:
                            exp.add_(exp, alpha=-lr * wd)
                        exp.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)
                        expected[id(p)] = exp
        super().on_skipped_backward_step(loss, model, inputs)
        if opt is not None:
            with torch.no_grad():
                for group in opt.param_groups:
                    for p in group["params"]:
                        if id(p) in expected:
                            self.delta_checks.append((
                                p.detach().clone(),
                                expected[id(p)].clone(),
                            ))


def test_momentum_extrapolation_applies_exact_delta():
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(1234)
        model = _TinyModel()
        trainer = _DeltaProbeTrainer(
            model=model, args=_tiny_args(td),
            train_dataset=_TinyDS(), data_collator=_collate,
            skip_policy=_ScriptedSkipPolicy(real_steps=2),
            skip_update_mode="momentum",
            instrumentation_path=os.path.join(td, "instrumentation.json"),
        )
        trainer.train()
        i = trainer.get_instrumentation()
        assert i["skipped_backward_steps"] >= 1
        assert len(trainer.delta_checks) > 0, (
            "no optimizer-state-backed parameters were compared"
        )
        for idx, (actual, expected) in enumerate(trainer.delta_checks):
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-7,
                msg=f"delta check {idx} failed",
            )


def test_legacy_apply_momentum_false_maps_to_freeze():
    with tempfile.TemporaryDirectory() as td:
        torch.manual_seed(1234)
        model = _TinyModel()
        snap_cb = _WeightsSnapshotUntilFirstSkip()
        trainer = LERNAMomentumTrainer(
            model=model, args=_tiny_args(td),
            train_dataset=_TinyDS(), data_collator=_collate,
            skip_policy=_ScriptedSkipPolicy(real_steps=2),
            apply_momentum=False,   # legacy kwarg
            instrumentation_path=os.path.join(td, "instrumentation.json"),
            callbacks=[snap_cb],
        )
        snap_cb.trainer = trainer
        assert trainer.skip_update_mode == "freeze"
        assert trainer.skip_update_mode_legacy_compat_used is True
        trainer.train()
        i = trainer.get_instrumentation()
        assert i["optimizer_step_attempts"] >= 1
        assert i["skipped_backward_steps"] >= 1
        assert i["skip_update_mode_legacy_compat_used"] is True
        for n, p in model.named_parameters():
            assert torch.equal(snap_cb.snapshot[n], p.detach())
