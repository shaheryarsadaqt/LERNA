"""Step 5 runner extensions (5 November 2026).

Covers:
- the max-loss and inverted-S1 score sources and their gated arms;
- the plain equal-compute arm (A9);
- warmup as a ratio of each arm's own horizon;
- the device option;
- the evaluator's per-example loss;
- the per-batch log and its compute summary;
- the runtime block.
All CPU-only.
"""

import argparse
import hashlib
import math
import sys
from pathlib import Path
from unittest import mock

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerna.utils.forward_gate import (  # noqa: E402
    ScoreQuantileGatePolicy,
    forward_gain_score,
    forward_loss_score,
)
from scripts.run_controller_ablation import (  # noqa: E402
    ARMS,
    BATCH_LOG_COLUMNS,
    GATE_ARMS,
    PLAIN_ARMS,
    ExperimentConfig,
    _batch_log_summary,
    _evaluate_model,
    _recipe_overrides,
    _write_batch_log,
    run_execution,
)


def _cfg(arm, total_steps=16, **kw):
    phases, source = GATE_ARMS.get(arm, (1, "uniform"))
    d = dict(
        task="qnli", arm=arm, controller_version="legacy_v4", score_source=source,
        total_steps=total_steps, min_step=0, n_phases=phases, max_consecutive_skips=4,
        policy_seed=0, target_skip_rate=0.5, output_dir="/tmp/test_step5_ext",
        attempt_id="t1", capture_logits=True, lr_warmup_steps=0,
        probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
    )
    d.update(kw)
    return ExperimentConfig(**d)


# --------------------------------------------------------------------------- scores

def test_forward_loss_score_matches_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(6, 2)
    labels = torch.tensor([0, 1, 1, 0, 1, 0])
    expected = float(torch.nn.functional.cross_entropy(logits, labels))
    assert forward_loss_score(logits, labels) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("logits,labels", [
    (None, torch.tensor([0])),
    (torch.randn(2, 2), None),
    (torch.randn(3, 2), torch.tensor([0, 1])),
    (torch.randn(2), torch.tensor([0, 1])),
    ([[0.0, 1.0]], torch.tensor([0])),
])
def test_forward_loss_score_unusable_inputs_give_nan(logits, labels):
    assert math.isnan(forward_loss_score(logits, labels))


def _policy_source(kind):
    pol = ScoreQuantileGatePolicy(target_skip_rate=0.5, total_steps=16, min_step=0,
                                  n_phases=1, policy_seed=0, score_source=kind)
    return pol


@pytest.mark.parametrize("kind", ["loss", "s1_inverted"])
def test_policy_sources_use_the_cached_logits(kind):
    torch.manual_seed(1)
    logits = torch.randn(4, 2)
    labels = torch.tensor([0, 1, 1, 0])
    pol = _policy_source(kind)
    pol._trainer_ref = type("T", (), {"_last_real_logits": logits})()
    got = pol._gate._score_source(0, None, {"labels": labels})
    if kind == "loss":
        assert got == pytest.approx(forward_loss_score(logits, labels), rel=1e-6)
    else:
        assert got == pytest.approx(-forward_gain_score(logits, labels), rel=1e-6)
        assert got < 0
    assert pol.get_diagnostics()["score_source"] == kind


def test_policy_inverted_source_keeps_nan():
    pol = _policy_source("s1_inverted")
    pol._trainer_ref = type("T", (), {"_last_real_logits": None})()
    assert math.isnan(pol._gate._score_source(0, None, {"labels": torch.tensor([0])}))


def test_policy_rejects_unknown_source():
    with pytest.raises(ValueError, match="score_source must be"):
        _policy_source("max_grad")


@pytest.mark.parametrize("kind", ["loss", "s1_inverted"])
def test_phase_exact_accepts_new_sources(kind):
    ScoreQuantileGatePolicy(target_skip_rate=0.5, total_steps=16, min_step=0, n_phases=4,
                            policy_seed=0, score_source=kind, controller_version="phase_exact_v1")


# --------------------------------------------------------------------------- config

def test_arm_lists_are_consistent():
    assert set(GATE_ARMS) | set(PLAIN_ARMS) == set(ARMS)
    assert not set(GATE_ARMS) & set(PLAIN_ARMS)
    assert GATE_ARMS["gate_loss_p4"] == (4, "loss")
    assert GATE_ARMS["gate_s1inv_p1"] == (1, "s1_inverted")


@pytest.mark.parametrize("arm", ["gate_loss_p1", "gate_loss_p4", "gate_s1inv_p1", "gate_s1inv_p4"])
def test_new_gate_arms_validate_phases_and_source(arm):
    phases, source = GATE_ARMS[arm]
    _cfg(arm)
    with pytest.raises(ValueError, match="requires n_phases"):
        _cfg(arm, n_phases=2 if phases != 2 else 3)
    with pytest.raises(ValueError, match="requires score_source"):
        _cfg(arm, score_source="s1")


def test_step5_qnli_accounting():
    cfg = _cfg("equal_compute", total_steps=6548, target_skip_rate=0.60)
    assert cfg.requested_skips == 3929
    assert cfg.reduced_steps == 2619
    assert cfg.equal_compute_steps == 3928
    # same compute as a gated arm, to within one forward pass
    gated_units = 6548 + 2 * (6548 - 3929)
    assert 3 * cfg.equal_compute_steps <= gated_units < 3 * (cfg.equal_compute_steps + 1)
    assert cfg.controller_version == "legacy_v4" and cfg.score_source == "uniform"


def test_equal_compute_rejects_empty_horizon():
    with pytest.raises(ValueError, match="equal_compute_steps"):
        _cfg("equal_compute", total_steps=2, target_skip_rate=1.0)


def test_warmup_ratio_gives_the_archived_counts():
    cfg = _cfg("full_finetune", total_steps=6548, target_skip_rate=0.60, lr_warmup_ratio=0.06)
    assert cfg.warmup_steps_for(6548) == 392
    assert cfg.warmup_steps_for(2619) == 157
    assert cfg.warmup_steps_for(3928) == 235
    assert cfg.warmup_steps_for(5000) == 300  # exact products are not rounded down by float error
    plain = _cfg("full_finetune", lr_warmup_steps=12)
    assert plain.warmup_steps_for(6548) == 12


@pytest.mark.parametrize("bad", [-0.1, 1.0, float("nan"), True, "0.06"])
def test_warmup_ratio_rejects_bad_values(bad):
    with pytest.raises(ValueError, match="lr_warmup_ratio"):
        _cfg("full_finetune", lr_warmup_ratio=bad)


def test_device_validation():
    assert _cfg("full_finetune", device="cuda").device == "cuda"
    with pytest.raises(ValueError, match="device"):
        _cfg("full_finetune", device="gpu")


def test_recipe_overrides_only_pass_given_options():
    ns = argparse.Namespace(lr_schedule_shape=None, lr_warmup_steps=None, lr_warmup_ratio=None, device=None)
    assert _recipe_overrides(ns) == {}
    ns = argparse.Namespace(lr_schedule_shape="linear", lr_warmup_steps=None, lr_warmup_ratio=0.06, device="cuda")
    assert _recipe_overrides(ns) == {"lr_schedule_shape": "linear", "lr_warmup_ratio": 0.06, "device": "cuda"}


# --------------------------------------------------------------------------- execution (CPU fixtures)

@pytest.mark.parametrize("arm", ["gate_loss_p1", "gate_s1inv_p1", "gate_loss_p4", "gate_s1inv_p4"])
def test_new_gate_arms_execute(arm):
    result = run_execution(_cfg(arm, total_steps=16))
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["forwards"] == 16
    assert r["skipped"] == 8 and r["backwards"] == 8
    log = result["batch_log"]
    assert len(log) == 16
    assert sum(row[4] for row in log) == 8
    assert [row[0] for row in log] == list(range(16))
    s = r["batch_log_summary"]
    assert s["batches"] == 16 and s["skips"] == 8 and s["updates"] == 8


def test_equal_compute_arm_executes_at_its_horizon():
    cfg = _cfg("equal_compute", total_steps=12, target_skip_rate=0.5)
    assert cfg.equal_compute_steps == 12 - (2 * 6 + 2) // 3  # 8
    result = run_execution(cfg)
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["effective_total"] == 8 and r["forwards"] == 8 and r["backwards"] == 8
    assert r["skipped"] == 0
    assert len(result["batch_log"]) == 8


def test_linear_decay_with_warmup_ratio_ends_at_zero():
    cfg = _cfg("reduced_steps", total_steps=16, target_skip_rate=0.5,
               lr_schedule_shape="linear", lr_warmup_ratio=0.25)
    result = run_execution(cfg)
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["effective_total"] == 8
    assert r["lr_warmup_steps"] == 2 and r["lr_warmup_ratio"] == 0.25
    assert r["final_learning_rate"] == pytest.approx(0.0, abs=1e-12)


def test_runtime_block_records_interpreter_and_optimizer():
    result = run_execution(_cfg("full_finetune", total_steps=4))
    assert result["status"] == "OK", result.get("reason")
    rt = result["runtime"]
    assert rt["executable"] == sys.executable
    assert rt["versions"]["torch"] == torch.__version__
    assert rt["device"] == "cpu" and rt["model_device"] == "cpu"
    assert rt["optimizer"]["betas"] == [0.9, 0.999]
    assert rt["optimizer"]["eps"] == pytest.approx(1e-8)
    assert rt["layernorms_with_bias"] == 0
    assert rt["parameters"] > 0


def test_cuda_needs_exactly_one_visible_gpu():
    cfg = _cfg("full_finetune", total_steps=4, device="cuda")
    with mock.patch("torch.cuda.is_available", return_value=True), \
            mock.patch("torch.cuda.device_count", return_value=2):
        result = run_execution(cfg)
    assert result["status"] == "FAILED"
    assert result["failure_stage"] == "device"
    assert "exactly one visible GPU" in result["reason"]


# --------------------------------------------------------------------------- evaluator

class _LinearProbe(torch.nn.Module):
    """Deterministic stand-in: logits = W x, loss = batch-mean cross-entropy."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(3)
        self.lin = torch.nn.Linear(4, 2)

    def forward(self, input_ids, labels=None, **kw):
        logits = self.lin(input_ids.float())
        loss = torch.nn.functional.cross_entropy(logits, labels.view(-1)) if labels is not None else None
        return type("O", (), {"loss": loss, "logits": logits})()


def test_evaluator_loss_is_the_per_example_mean():
    torch.manual_seed(4)
    xs = torch.randn(7, 4)
    ys = torch.tensor([0, 1, 1, 0, 1, 0, 1])
    data = [{"input_ids": xs[i], "labels": ys[i]} for i in range(7)]
    model = _LinearProbe()
    cfg = _cfg("full_finetune", total_steps=4, eval_batch_size=3)  # batches of 3, 3, 1
    loss, acc, f1 = _evaluate_model(None, model, data, cfg)
    with torch.no_grad():
        logits = model.lin(xs)
        expected = float(torch.nn.functional.cross_entropy(logits, ys))
        expected_acc = float((logits.argmax(-1) == ys).float().mean())
    assert loss == pytest.approx(expected, rel=1e-6)
    assert acc == pytest.approx(expected_acc)
    assert f1 is None


# --------------------------------------------------------------------------- batch log

def test_batch_log_summary_and_csv(tmp_path):
    rows = [[0, 50, 64, 2, 0, 0.7], [1, 40, 64, 2, 1, 0.6], [2, 60, 96, 3, 0, 0.5]]
    s = _batch_log_summary(rows)
    assert s["columns"] == BATCH_LOG_COLUMNS
    assert s["padded_tokens_all"] == 224 and s["padded_tokens_updated"] == 160
    assert s["real_tokens_all"] == 150 and s["real_tokens_updated"] == 110
    assert s["compute_units_padded"] == 224 + 320
    assert s["compute_fraction_within_arm"] == pytest.approx((224 + 320) / 672)
    assert _batch_log_summary([]) is None
    result = {"batch_log": rows}
    _write_batch_log(result, tmp_path)
    assert "batch_log" not in result
    data = (tmp_path / "batch_log.csv").read_bytes()
    assert result["batch_log_file"] == {"file": "batch_log.csv", "rows": 3,
                                        "sha256": hashlib.sha256(data).hexdigest()}
    lines = data.decode().splitlines()
    assert lines[0] == ",".join(BATCH_LOG_COLUMNS)
    assert lines[2] == "1,40,64,2,1,0.6"
