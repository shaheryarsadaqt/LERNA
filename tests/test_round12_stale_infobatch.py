"""Round 12 runner extensions (6 November 2026).

Covers:
- M1, the phase-exact gate scored from stored per-example residuals
  (arm stale_s1_p4), including skips decided before any forward pass;
- InfoBatch at a fixed step count (arm infobatch);
- the plain-fraction arms of the accuracy-compute reference curve;
- the extended per-batch log and its compute summary.
All CPU-only.
"""

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerna.trainers.true_skip_trainer import SkipInstrumentation  # noqa: E402
from lerna.utils.forward_gate import forward_gain_score  # noqa: E402
from lerna.utils.stale_selection import (  # noqa: E402
    InfoBatchPolicy,
    StaleS1GatePolicy,
    StoredScoreError,
    per_example_losses,
    per_example_residuals,
)
from scripts.run_controller_ablation import (  # noqa: E402
    ARMS,
    BATCH_LOG_COLUMNS,
    GATE_ARMS,
    PLAIN_ARMS,
    PLAIN_FRACTIONS,
    STORED_SCORE_ARMS,
    ExperimentConfig,
    _batch_log_summary,
    _recipe_overrides,
    _write_batch_log,
    run_dry_run,
    run_execution,
)


def _cfg(arm, total_steps=64, **kw):
    phases, source = GATE_ARMS.get(arm, (1, "uniform"))
    d = dict(
        task="qnli", arm=arm,
        controller_version="phase_exact_v1" if arm in GATE_ARMS else "legacy_v4",
        score_source=source, total_steps=total_steps, min_step=4, n_phases=phases,
        max_consecutive_skips=4, policy_seed=0, target_skip_rate=0.5,
        output_dir="/tmp/test_round12", attempt_id="t12", capture_logits=True,
        lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None,
        dry_run=True, train_batch_size=8,
    )
    d.update(kw)
    return ExperimentConfig(**d)


class _Instr:
    def __init__(self):
        self.batches_seen = 0


class _FakeTrainer:
    """The trainer attributes the stored-score policies read."""

    def __init__(self):
        self.instr = _Instr()
        self._lerna_batch_uids = None
        self._last_real_logits = None


def _logits_for(probs_true, labels):
    """Two-class logits whose true-class probability is probs_true."""
    rows = []
    for p, y in zip(probs_true, labels):
        z = math.log(p / (1.0 - p))
        rows.append([0.0, z] if y == 1 else [z, 0.0])
    return torch.tensor(rows, dtype=torch.float32)


# --------------------------------------------------------------------------- arm lists and horizons

def test_round12_arm_lists():
    assert {"stale_s1_p4", "infobatch", "plain_030", "plain_050", "plain_070"} <= set(ARMS)
    assert GATE_ARMS["stale_s1_p4"] == (4, "s1_stale")
    assert set(PLAIN_FRACTIONS) <= set(PLAIN_ARMS) and "infobatch" in PLAIN_ARMS
    # round 14 adds the LERNA v2 arms to the stored-score arms
    assert {"stale_s1_p4", "infobatch"} <= set(STORED_SCORE_ARMS)
    assert set(GATE_ARMS) | set(PLAIN_ARMS) == set(ARMS)
    assert not set(GATE_ARMS) & set(PLAIN_ARMS)


@pytest.mark.parametrize("T, expected", [(6548, (1964, 3274, 4584)), (4210, (1263, 2105, 2947))])
def test_plain_fraction_steps(T, expected):
    got = tuple(_cfg(a, total_steps=T).plain_fraction_steps for a in ("plain_030", "plain_050", "plain_070"))
    assert got == expected
    for a, steps in zip(("plain_030", "plain_050", "plain_070"), expected):
        c = _cfg(a, total_steps=T)
        assert c.arm_horizon() == steps
        assert run_dry_run(c)["result"]["total_steps"] == steps


def test_arm_horizon_of_the_other_arms():
    c = _cfg("infobatch", total_steps=6548, target_skip_rate=0.6)
    assert c.arm_horizon() == 6548
    assert _cfg("reduced_steps", total_steps=6548, target_skip_rate=0.6).arm_horizon() == 2619
    assert _cfg("equal_compute", total_steps=6548, target_skip_rate=0.6).arm_horizon() == 3928
    assert _cfg("stale_s1_p4", total_steps=6548, target_skip_rate=0.6, min_step=392).arm_horizon() == 6548


def test_stale_arm_always_uses_the_phase_exact_controller():
    assert _cfg("stale_s1_p4", controller_version="legacy_v4").controller_version == "phase_exact_v1"
    result = run_execution(_cfg("stale_s1_p4", total_steps=32, controller_version="legacy_v4"))
    assert result["status"] == "OK", result.get("reason")
    assert result["config"]["controller_version"] == "phase_exact_v1"


@pytest.mark.parametrize("field, bad", [("infobatch_prune_ratio", 1.0), ("infobatch_prune_ratio", -0.1),
                                        ("infobatch_prune_ratio", float("nan")),
                                        ("infobatch_anneal_fraction", 1.5), ("infobatch_prune_ratio", True)])
def test_infobatch_options_are_validated(field, bad):
    with pytest.raises(ValueError):
        _cfg("infobatch", **{field: bad})


def test_stale_arm_plans_like_the_gate():
    r = run_dry_run(_cfg("stale_s1_p4", total_steps=6548, target_skip_rate=0.6, min_step=392))
    assert r["status"] == "OK", r.get("reason")
    assert r["result"]["phase_quota"] == [794, 1045, 1045, 1045]
    assert r["result"]["realized_total"] == 3929


def test_recipe_overrides_pass_infobatch_options():
    import argparse
    ns = argparse.Namespace(lr_schedule_shape=None, lr_warmup_steps=None, lr_warmup_ratio=None,
                            device=None, infobatch_prune_ratio=0.3, infobatch_anneal_fraction=0.9)
    assert _recipe_overrides(ns) == {"infobatch_prune_ratio": 0.3, "infobatch_anneal_fraction": 0.9}


# --------------------------------------------------------------------------- per-example scores

def test_residual_mean_is_s1():
    torch.manual_seed(5)
    logits = torch.randn(9, 2)
    labels = torch.tensor([0, 1, 1, 0, 1, 0, 0, 1, 1])
    r = per_example_residuals(logits, labels)
    assert r.dtype == torch.float64 and r.shape == (9,)
    assert float(r.mean()) == pytest.approx(forward_gain_score(logits, labels), rel=1e-6)


def test_per_example_losses_match_cross_entropy():
    torch.manual_seed(6)
    logits = torch.randn(5, 2)
    labels = torch.tensor([0, 1, 0, 1, 1])
    want = torch.nn.functional.cross_entropy(logits, labels, reduction="none").double()
    assert torch.allclose(per_example_losses(logits, labels), want)


def test_scores_need_tensors():
    with pytest.raises(StoredScoreError):
        per_example_residuals(None, torch.tensor([0]))
    with pytest.raises(StoredScoreError):
        per_example_losses(torch.zeros(1, 2), None)


# --------------------------------------------------------------------------- M1 policy (unit)

def _stale_policy(**kw):
    d = dict(target_skip_rate=0.5, total_steps=16, num_examples=8, min_step=0,
             max_consecutive_skips=4, n_phases=1, score_history_window=4,
             anneal_fraction=1.0, policy_seed=0)
    d.update(kw)
    return StaleS1GatePolicy(**d)


def test_stale_policy_decides_before_forward_only_with_every_score_stored():
    pol = _stale_policy()
    tr = _FakeTrainer()
    labels = torch.tensor([0, 1, 0, 1])
    tr._lerna_batch_uids = [0, 1, 2, 3]
    tr.instr.batches_seen = 1
    assert pol.pre_forward_decision(tr, None, {"labels": labels}) is None  # nothing stored yet
    logits = _logits_for([0.9, 0.8, 0.7, 0.6], [0, 1, 0, 1])
    pol.observe_forward(tr, {"labels": labels}, logits)
    tr._last_real_logits = logits
    assert pol.should_skip(tr, None, {"labels": labels}) in (True, False)  # decision 0, fresh S1
    assert pol.fresh_decisions == 1
    expected = float(per_example_residuals(logits, labels).mean())
    assert pol.stored_batch_score([0, 1, 2, 3]) == pytest.approx(expected)
    assert pol.stored_batch_score([0, 1, 2, 4]) is None  # example 4 has no score
    tr.instr.batches_seen = 2
    out = pol.pre_forward_decision(tr, None, {"labels": labels})
    assert out in (True, False)
    assert pol.stale_decisions == 1 and pol.forward_skips + pol.stale_retains == 1


def test_stale_policy_rejects_missing_or_bad_ids():
    pol = _stale_policy()
    tr = _FakeTrainer()
    tr.instr.batches_seen = 1
    with pytest.raises(StoredScoreError, match="lerna_uid"):
        pol.pre_forward_decision(tr, None, {"labels": torch.tensor([0])})
    tr._lerna_batch_uids = [8]
    with pytest.raises(StoredScoreError, match="outside"):
        pol.pre_forward_decision(tr, None, {"labels": torch.tensor([0])})


def test_stale_policy_requires_phase_exact():
    with pytest.raises(ValueError):
        _stale_policy(controller_version="legacy_v4")
    with pytest.raises(ValueError):
        _stale_policy(num_examples=0)


# --------------------------------------------------------------------------- InfoBatch policy (unit)

def test_infobatch_prunes_only_known_low_loss_examples_after_the_first_epoch():
    pol = InfoBatchPolicy(total_steps=8, num_examples=4, batches_per_epoch=1,
                          prune_ratio=0.5, anneal_fraction=0.75, policy_seed=1)
    tr = _FakeTrainer()
    tr._lerna_batch_uids = [0, 1, 2, 3]
    labels = torch.tensor([0, 1, 0, 1])
    tr.instr.batches_seen = 1
    keep, w = pol.select_examples(tr, {"labels": labels})
    assert keep == [0, 1, 2, 3] and w == [1.0] * 4  # no stored losses: no threshold
    # losses: examples 0 and 1 low, 2 and 3 high
    pol.observe_forward(tr, {"labels": labels}, _logits_for([0.99, 0.99, 0.2, 0.2], [0, 1, 0, 1]))
    kept_low = pruned = 0
    for step in range(2, 7):  # steps 1..5 (< floor(0.75 * 8) = 6) may prune
        tr.instr.batches_seen = step
        keep, w = pol.select_examples(tr, {"labels": labels})
        assert 2 in keep and 3 in keep  # above-threshold examples are always kept
        for pos, wt in zip(keep, w):
            assert wt == (2.0 if pos in (0, 1) else 1.0)
        kept_low += sum(1 for k in keep if k in (0, 1))
        pruned += 2 - sum(1 for k in keep if k in (0, 1))
    assert pruned == pol.pruned_examples and kept_low == pol.kept_low_loss_examples
    assert pruned > 0 and kept_low > 0
    tr.instr.batches_seen = 7  # step 6 >= stop: no pruning
    assert pol.select_examples(tr, {"labels": labels}) == ([0, 1, 2, 3], [1.0] * 4)
    assert pol.thresholds[0] is None and pol.thresholds[1] == pytest.approx(
        float(per_example_losses(_logits_for([0.99, 0.99, 0.2, 0.2], [0, 1, 0, 1]), labels).mean()))


def test_infobatch_never_returns_an_empty_batch():
    pol = InfoBatchPolicy(total_steps=100, num_examples=3, batches_per_epoch=1,
                          prune_ratio=0.99, anneal_fraction=1.0, policy_seed=0)
    tr = _FakeTrainer()
    tr._lerna_batch_uids = [0, 1, 2]
    labels = torch.tensor([0, 1, 0])
    tr.instr.batches_seen = 1
    pol.select_examples(tr, {"labels": labels})
    pol.observe_forward(tr, {"labels": labels}, _logits_for([0.99, 0.99, 0.01], [0, 1, 0]))
    for step in range(2, 60):
        tr.instr.batches_seen = step
        keep, w = pol.select_examples(tr, {"labels": labels})
        assert keep and len(keep) == len(w)


@pytest.mark.parametrize("kw", [dict(prune_ratio=1.0), dict(anneal_fraction=2.0),
                                dict(batches_per_epoch=0), dict(num_examples=True)])
def test_infobatch_policy_validates(kw):
    d = dict(total_steps=10, num_examples=4, batches_per_epoch=1)
    d.update(kw)
    with pytest.raises(ValueError):
        InfoBatchPolicy(**d)


# --------------------------------------------------------------------------- execution (CPU fixtures)

def test_stale_arm_executes_with_exact_quotas_and_forward_free_skips():
    result = run_execution(_cfg("stale_s1_p4", total_steps=64))
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["skipped"] == 32 and r["backwards"] == 32
    assert r["forward_skipped"] > 0
    assert r["forwards"] + r["forward_skipped"] == 64
    sel = r["selection"]
    assert sel["fresh_decisions"] == 8  # the first epoch: 64 examples in batches of 8
    assert sel["stale_decisions"] == 56 and sel["forward_skips"] == r["forward_skipped"]
    log = result["batch_log"]
    assert len(log) == 64 and all(len(row) == len(BATCH_LOG_COLUMNS) for row in log)
    free = [row for row in log if row[6] == 0]
    assert len(free) == r["forward_skipped"]
    assert all(row[4] == 1 and math.isnan(row[5]) for row in free)
    assert all(row[6] == 1 for row in log[:8])  # no forward-free skip in the first epoch
    s = r["batch_log_summary"]
    assert s["forward_free_skips"] == r["forward_skipped"]
    assert s["compute_fraction_within_arm"] < 2.0 / 3.0  # cheaper than the fresh-score gate


def test_stale_arm_costs_less_than_the_fresh_gate():
    stale = run_execution(_cfg("stale_s1_p4", total_steps=64))["result"]["batch_log_summary"]
    fresh = run_execution(_cfg("gate_s1_p4", total_steps=64))["result"]["batch_log_summary"]
    assert fresh["compute_fraction_within_arm"] == pytest.approx(2.0 / 3.0)
    assert stale["compute_units_padded"] < fresh["compute_units_padded"]


def test_infobatch_arm_executes_and_drops_examples_after_the_first_epoch():
    result = run_execution(_cfg("infobatch", total_steps=64))
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["skipped"] == 0 and r["backwards"] == 64 and r["forwards"] == 64
    sel = r["selection"]
    assert r["examples_dropped_before_forward"] == sel["pruned_examples"] > 0
    assert sel["stop_prune_step"] == 56 and sel["batches_per_epoch"] == 8
    log = result["batch_log"]
    assert all(row[3] == row[7] == 8 for row in log[:8])  # first epoch: whole batches
    assert all(row[3] == row[7] for row in log[56:])  # annealing tail: whole batches
    assert sum(row[7] - row[3] for row in log) == sel["pruned_examples"]
    assert r["final_learning_rate"] is not None


def test_plain_fraction_arm_executes_at_its_horizon():
    result = run_execution(_cfg("plain_050", total_steps=40))
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    assert r["effective_total"] == 20 and r["forwards"] == 20 and r["backwards"] == 20


def test_existing_arm_rows_keep_their_first_six_fields():
    result = run_execution(_cfg("gate_s1_p4", total_steps=32))
    assert result["status"] == "OK", result.get("reason")
    for row in result["batch_log"]:
        assert row[6] == 1 and row[7] == row[3] and row[8] == row[2]


def test_policy_failure_before_forward_is_rejected(monkeypatch):
    def boom(self, trainer, model, inputs):
        raise RuntimeError("stored score unavailable")
    monkeypatch.setattr(StaleS1GatePolicy, "pre_forward_decision", boom)
    result = run_execution(_cfg("stale_s1_p4", total_steps=16))
    assert result["status"] == "REJECTED"
    assert result["reason"] == "policy_decision_failure"


def test_stale_arm_without_ids_is_rejected(monkeypatch):
    import scripts.run_controller_ablation as rca
    monkeypatch.setattr(rca, "STORED_SCORE_ARMS", ())  # the loader then adds no lerna_uid
    result = run_execution(_cfg("stale_s1_p4", total_steps=16))
    assert result["status"] == "REJECTED"
    assert result["reason"] == "policy_decision_failure"


# --------------------------------------------------------------------------- trainer helpers

def _bare_trainer():
    from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer
    tr = TrueBackwardSkippingTrainer.__new__(TrueBackwardSkippingTrainer)
    tr.instr = SkipInstrumentation()
    tr._lerna_batch_uids = [10, 11, 12]
    tr._lerna_example_weights = None
    tr._lerna_scheduled_batch = None
    tr.capture_logits = True
    return tr


def test_example_selection_keeps_rows_and_trims_padding():
    tr = _bare_trainer()
    inputs = {
        "input_ids": torch.tensor([[5, 6, 7, 1, 1], [5, 6, 1, 1, 1], [5, 6, 7, 8, 9]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0], [1, 1, 1, 1, 1]]),
        "labels": torch.tensor([0, 1, 1]),
    }
    out = tr._apply_example_selection(inputs, [0, 1], [2.0, 1.0])
    assert out["input_ids"].tolist() == [[5, 6, 7], [5, 6, 1]]
    assert out["attention_mask"].tolist() == [[1, 1, 1], [1, 1, 0]]
    assert out["labels"].tolist() == [0, 1]
    assert tr._lerna_batch_uids == [10, 11]
    assert tr._lerna_scheduled_batch == (3, 15)
    assert tr._lerna_example_weights.tolist() == [2.0, 1.0]
    assert tr.instr.examples_dropped_before_forward == 1


@pytest.mark.parametrize("keep, weights", [([], []), ([0, 0], [1.0, 1.0]), ([3], [1.0]), ([0], [1.0, 2.0])])
def test_example_selection_rejects_bad_selections(keep, weights):
    tr = _bare_trainer()
    inputs = {"input_ids": torch.zeros(3, 4, dtype=torch.long), "labels": torch.tensor([0, 1, 0])}
    with pytest.raises(ValueError):
        tr._apply_example_selection(inputs, keep, weights)


class _Lin(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(2)
        self.lin = torch.nn.Linear(3, 2)

    def forward(self, input_ids, labels=None, **kw):
        logits = self.lin(input_ids.float())
        loss = torch.nn.functional.cross_entropy(logits, labels) if labels is not None else None
        return type("O", (), {"loss": loss, "logits": logits})()


def test_weighted_loss_is_horvitz_thompson_over_the_scheduled_batch():
    tr = _bare_trainer()
    model = _Lin()
    x = torch.randn(4, 3)
    y = torch.tensor([0, 1, 1, 0])
    tr._lerna_example_weights = torch.tensor([2.0, 1.0, 2.0, 1.0])
    tr._lerna_scheduled_batch = (6, 24)  # 2 of 6 scheduled examples were dropped
    loss = tr.compute_loss(model, {"input_ids": x, "labels": y})
    per = torch.nn.functional.cross_entropy(model.lin(x), y, reduction="none")
    assert float(loss) == pytest.approx(float((per * torch.tensor([2.0, 1.0, 2.0, 1.0])).sum() / 6), rel=1e-6)
    tr._lerna_example_weights = torch.ones(4)
    tr._lerna_scheduled_batch = (4, 16)
    assert float(tr.compute_loss(model, {"input_ids": x, "labels": y})) == pytest.approx(float(per.mean()), rel=1e-6)


def test_instrumentation_invariant_counts_forward_free_skips():
    ins = SkipInstrumentation()
    ins.forward_calls, ins.backward_calls, ins.skipped_backward_steps = 5, 4, 3
    ins.forward_skipped_batches = 2
    assert ins.as_dict()["invariant_forward_eq_backward_plus_skipped"] is True


# --------------------------------------------------------------------------- batch log

def test_batch_log_summary_with_forward_free_skips_and_dropped_examples(tmp_path):
    rows = [
        [0, 50, 64, 2, 0, 0.7, 1, 2, 64],            # ordinary update
        [1, 40, 64, 2, 1, float("nan"), 0, 2, 64],   # skip with no forward pass
        [2, 30, 32, 1, 0, 0.5, 1, 2, 64],            # InfoBatch kept 1 of 2 examples
    ]
    s = _batch_log_summary(rows)
    assert s["padded_tokens_forward"] == 96 and s["padded_tokens_updated"] == 96
    assert s["compute_units_padded"] == 96 + 2 * 96
    assert s["scheduled_padded_tokens"] == 192
    assert s["compute_fraction_within_arm"] == pytest.approx(288 / 576)
    assert s["forward_free_skips"] == 1
    result = {"batch_log": rows}
    _write_batch_log(result, tmp_path)
    lines = (tmp_path / "batch_log.csv").read_text().splitlines()
    assert lines[0] == ",".join(BATCH_LOG_COLUMNS)
    assert lines[2] == "1,40,64,2,1,nan,0,2,64"
