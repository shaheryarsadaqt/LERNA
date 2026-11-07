"""Round 15 runner extensions (6 November 2026).

Covers:
- the hard top-k selection of LernaBudgetPolicy (a compared method: dynamic EL2N pruning under the
  same exact budget): it keeps the K = round(sum pi) highest-pi examples, weights them n / K, is
  deterministic, and spends the budget;
- that the sampling selection (LERNA v2's own rule) is unchanged by the new option;
- the four round-15 arms of the runner, executed end to end on the CPU fixtures.
All CPU-only.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerna.utils.budget_selection import LernaBudgetPolicy  # noqa: E402
from scripts.run_controller_ablation import (  # noqa: E402
    ARMS,
    EXAMPLE_SELECTION_ARMS,
    LERNA2_ARMS,
    LERNA2_TOPK_ARMS,
    PLAIN_ARMS,
    STORED_SCORE_ARMS,
    ExperimentConfig,
    run_dry_run,
    run_execution,
)


def _linear_lambda(warm, T):
    def f(k):
        if k < warm:
            return k / max(1, warm)
        return max(0.0, (T - k) / max(1, T - warm))
    return f


class _Instr:
    def __init__(self):
        self.batches_seen = 0


class _Trainer:
    def __init__(self, T, warm):
        self.instr = _Instr()
        self._lerna_batch_uids = None
        p = torch.nn.Parameter(torch.zeros(1))
        self.opt = torch.optim.SGD([p], lr=1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, _linear_lambda(warm, T))

    def step(self):
        self.opt.step()
        self.lr_scheduler.step()


def _logits(residuals):
    rows = []
    for r in residuals:
        pt = 1.0 - float(r) / math.sqrt(2.0)
        rows.append([0.0, math.log(pt / (1.0 - pt))])
    return torch.tensor(rows, dtype=torch.float32)


def _drive(pol, N, B, T, warm, seed=0, record=None):
    rng = np.random.default_rng(seed)
    tr = _Trainer(T, warm)
    length = rng.integers(5, 40, size=N)
    hard = rng.uniform(0.02, 0.9, size=N)
    t = 0
    while t < T:
        order = rng.permutation(N)
        for b in range(0, N, B):
            if t >= T:
                break
            uids = [int(u) for u in order[b:b + B]]
            n, width = len(uids), int(length[uids].max())
            mask = torch.zeros((n, width), dtype=torch.long)
            for i, u in enumerate(uids):
                mask[i, :length[u]] = 1
            tr.instr.batches_seen += 1
            tr._lerna_batch_uids = list(uids)
            keep, weights = pol.select_examples(tr, {"input_ids": torch.ones((n, width), dtype=torch.long),
                                                     "attention_mask": mask})
            if record is not None:
                record.append({"t": t, "keep": list(keep), "weights": list(weights), "n": n,
                               "pi": pol.last_pi.copy(), "scores": pol.last_scores.copy()})
            kept = [uids[k] for k in keep]
            tr._lerna_batch_uids = kept
            res = np.clip(hard[kept] * (1.0 - 0.8 * t / T), 1e-3, 0.99)
            pol.observe_forward(tr, {"labels": torch.ones(len(kept), dtype=torch.long)}, _logits(res))
            tr.step()
            t += 1


def _policy(budget=0.35, N=800, B=16, T=300, warm=18, **kw):
    return LernaBudgetPolicy(budget=budget, total_steps=T, num_examples=N,
                             batches_per_epoch=math.ceil(N / B), warmup_steps=warm, **kw)


def test_selection_is_validated():
    with pytest.raises(ValueError):
        _policy(selection="hard")
    with pytest.raises(ValueError):
        _policy(selection=None)
    assert LernaBudgetPolicy.SELECTIONS == ("sample", "topk")


def test_topk_keeps_the_k_highest_and_weights_them_n_over_k():
    rec = []
    pol = _policy(selection="topk", use_lr_weight=False)
    _drive(pol, 800, 16, 300, 18, record=rec)
    for r in rec:
        k = max(1, min(r["n"], int(round(float(r["pi"].sum())))))
        assert len(r["keep"]) == k
        assert r["weights"] == [pytest.approx(r["n"] / k)] * k
        kept_pi = sorted(r["pi"][i] for i in r["keep"])
        dropped_pi = [r["pi"][i] for i in range(r["n"]) if i not in set(r["keep"])]
        if dropped_pi:
            assert kept_pi[0] >= max(dropped_pi) - 1e-12
    for r in rec[:18]:
        assert r["keep"] == list(range(16)) and r["weights"] == [1.0] * 16
    assert pol.get_diagnostics()["selection"] == "topk"
    assert pol.get_diagnostics()["empty_batch_fallbacks"] == 0


@pytest.mark.parametrize("budget", [0.35, 0.6])
@pytest.mark.parametrize("use_lr", [False, True])
def test_topk_spends_its_budget(budget, use_lr):
    pol = _policy(budget=budget, selection="topk", use_lr_weight=use_lr)
    _drive(pol, 800, 16, 300, 18)
    assert pol.realized_fraction == pytest.approx(budget, abs=0.006)


def test_topk_does_not_depend_on_the_policy_seed():
    def run(seed):
        rec = []
        _drive(_policy(selection="topk", use_lr_weight=False, policy_seed=seed), 800, 16, 300, 18, record=rec)
        return [tuple(r["keep"]) for r in rec]
    assert run(3) == run(4)


def test_sampling_is_unchanged_by_the_new_option():
    def run(**kw):
        rec = []
        _drive(_policy(policy_seed=5, **kw), 800, 16, 300, 18, record=rec)
        return [(tuple(r["keep"]), tuple(r["weights"])) for r in rec]
    assert run() == run(selection="sample")
    assert run() != run(selection="topk")


def test_round15_arm_lists():
    new = {"randkeep_b060", "randkeep_b083", "el2n_topk_b035", "el2n_topk_b060"}
    assert new <= set(ARMS) and new <= set(LERNA2_ARMS) and new <= set(PLAIN_ARMS)
    assert new <= set(STORED_SCORE_ARMS) and new <= set(EXAMPLE_SELECTION_ARMS)
    assert set(LERNA2_TOPK_ARMS) == {"el2n_topk_b035", "el2n_topk_b060"}
    assert LERNA2_ARMS["randkeep_b060"] == (0.60, False, False)
    assert LERNA2_ARMS["randkeep_b083"] == (0.83, False, False)
    assert LERNA2_ARMS["el2n_topk_b035"] == (0.35, True, False)
    assert LERNA2_ARMS["el2n_topk_b060"] == (0.60, True, False)


def _cfg(arm, total_steps=64, **kw):
    d = dict(
        task="qnli", arm=arm, controller_version="legacy_v4", score_source="uniform",
        total_steps=total_steps, min_step=4, n_phases=1, max_consecutive_skips=4, policy_seed=0,
        target_skip_rate=0.5, output_dir="/tmp/test_round15", attempt_id="t15", capture_logits=True,
        lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
        train_batch_size=8, lr_schedule_shape="linear", lr_warmup_ratio=0.06,
    )
    d.update(kw)
    return ExperimentConfig(**d)


def test_topk_dry_run_describes_the_arm():
    r = run_dry_run(_cfg("el2n_topk_b035", total_steps=6548))
    assert r["status"] == "OK" and "hard top-k" in r["result"]["note"] and "LERNA v2)" not in r["result"]["note"]
    r = run_dry_run(_cfg("randkeep_b060", total_steps=6548))
    assert "LERNA v2" in r["result"]["note"]


@pytest.mark.parametrize("arm", ["el2n_topk_b035", "el2n_topk_b060"])
def test_topk_arm_executes_with_hard_selection(arm):
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    sel = result["result"]["selection"]
    assert sel["selection"] == "topk" and sel["use_scores"] is True and sel["use_lr_weight"] is False
    assert sel["empty_batch_fallbacks"] == 0
    assert result["result"]["examples_dropped_before_forward"] == sel["pruned_examples"] > 0


@pytest.mark.parametrize("arm", ["randkeep_b060", "randkeep_b083", "lerna2_b035"])
def test_sampling_arms_record_their_selection(arm):
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    assert result["result"]["selection"]["selection"] == "sample"
