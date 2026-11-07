"""Round 14 runner extensions (6 November 2026): LERNA v2.

Covers:
- LernaBudgetPolicy, the learning-rate-aware example selection under an exact compute budget:
  the warmup, the keep rule and its weights, the budget controller, the random and no-LR
  variants, the scale estimates, the strict failure modes and reproducibility;
- the six round-14 arms of the runner, executed end to end on the CPU fixtures.
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
from lerna.utils.stale_selection import StoredScoreError  # noqa: E402
from scripts.run_controller_ablation import (  # noqa: E402
    ARMS,
    BATCH_LOG_COLUMNS,
    EXAMPLE_SELECTION_ARMS,
    GATE_ARMS,
    LERNA2_ARMS,
    LERNA2_MIN_BUDGET,
    PLAIN_ARMS,
    STORED_SCORE_ARMS,
    ExperimentConfig,
    run_dry_run,
    run_execution,
)


# --------------------------------------------------------------------------- a synthetic driver

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
    """The trainer attributes LernaBudgetPolicy reads, with a real LambdaLR."""

    def __init__(self, T, warm, scheduler=True):
        self.instr = _Instr()
        self._lerna_batch_uids = None
        p = torch.nn.Parameter(torch.zeros(1))
        self.opt = torch.optim.SGD([p], lr=1.0)
        self.lr_scheduler = (torch.optim.lr_scheduler.LambdaLR(self.opt, _linear_lambda(warm, T))
                             if scheduler else None)

    def step(self):
        self.opt.step()
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()


def _logits(residuals):
    """Two-class logits for label 1 whose residual ||p - y|| is the given value."""
    rows = []
    for r in residuals:
        pt = 1.0 - float(r) / math.sqrt(2.0)
        rows.append([0.0, math.log(pt / (1.0 - pt))])
    return torch.tensor(rows, dtype=torch.float32)


def _drive(pol, N, B, T, warm, seed=0, scheduler=True, record=None, residual_fn=None, make_trainer=None):
    """Feed T scheduled batches of B examples (epochs of a random order) through the policy."""
    rng = np.random.default_rng(seed)
    tr = (make_trainer or _Trainer)(T, warm, scheduler=scheduler)
    length = rng.integers(5, 40, size=N)
    hard = rng.uniform(0.02, 0.9, size=N)
    full_units = 0
    t = 0
    while t < T:
        order = rng.permutation(N)
        for b in range(0, N, B):
            if t >= T:
                break
            uids = [int(u) for u in order[b:b + B]]
            n, width = len(uids), int(length[uids].max())
            full_units += 3 * n * width
            mask = torch.zeros((n, width), dtype=torch.long)
            for i, u in enumerate(uids):
                mask[i, :length[u]] = 1
            tr.instr.batches_seen += 1
            tr._lerna_batch_uids = list(uids)
            keep, weights = pol.select_examples(tr, {"input_ids": torch.ones((n, width), dtype=torch.long),
                                                     "attention_mask": mask})
            if record is not None:
                record.append({"t": t, "uids": list(uids), "keep": list(keep), "weights": list(weights),
                               "pi": pol.last_pi.copy(), "scores": pol.last_scores.copy()})
            kept = [uids[k] for k in keep]
            tr._lerna_batch_uids = kept
            res = (residual_fn(kept, t) if residual_fn is not None
                   else np.clip(hard[kept] * (1.0 - 0.8 * t / T), 1e-3, 0.99))
            pol.observe_forward(tr, {"labels": torch.ones(len(kept), dtype=torch.long)}, _logits(res))
            tr.step()
            t += 1
    return full_units


def _policy(budget=0.4, N=400, B=16, T=100, warm=6, **kw):
    return LernaBudgetPolicy(budget=budget, total_steps=T, num_examples=N,
                             batches_per_epoch=math.ceil(N / B), warmup_steps=warm, **kw)


# --------------------------------------------------------------------------- the policy

@pytest.mark.parametrize("kw", [
    {"budget": 0.0}, {"budget": 1.2}, {"budget": 0.05}, {"budget": float("nan")}, {"budget": True},
    {"pi_min": 0.0}, {"pi_min": 1.5}, {"warmup_steps": -1}, {"warmup_steps": 100}, {"use_scores": 1},
    {"use_lr_weight": "yes"}, {"window": 0}, {"chunks": 0}, {"total_steps": 0}, {"policy_seed": 1.5},
])
def test_policy_validates(kw):
    args = dict(budget=0.4, total_steps=100, num_examples=400, batches_per_epoch=25, warmup_steps=6)
    args.update(kw)
    with pytest.raises(ValueError):
        LernaBudgetPolicy(**args)


def test_every_example_is_kept_with_weight_one_during_the_warmup():
    rec = []
    pol = _policy(T=60, warm=12)
    _drive(pol, 400, 16, 60, 12, record=rec)
    for r in rec[:12]:
        assert r["keep"] == list(range(16)) and r["weights"] == [1.0] * 16
    assert pol.get_diagnostics()["pruned_during_warmup"] == 0
    assert any(len(r["keep"]) < 16 for r in rec[12:])


@pytest.mark.parametrize("budget", [0.3, 0.4, 0.6, 0.83])
@pytest.mark.parametrize("use_scores, use_lr", [(True, True), (True, False), (False, True), (False, False)])
def test_the_run_spends_its_budget(budget, use_scores, use_lr):
    N, B, T, warm = 800, 16, 300, 18
    pol = _policy(budget=budget, N=N, B=B, T=T, warm=warm, use_scores=use_scores, use_lr_weight=use_lr)
    full = _drive(pol, N, B, T, warm)
    assert pol.scheduled_units == full
    assert pol.realized_fraction == pytest.approx(budget, abs=0.006)
    d = pol.get_diagnostics()
    assert d["kept_examples"] + d["pruned_examples"] == d["scheduled_examples"] == 300 * 16
    assert sum(v["scheduled"] for v in d["epochs"].values()) == d["scheduled_examples"]


def test_keep_probabilities_follow_the_rule_and_weights_are_their_inverses():
    rec = []
    pol = _policy(budget=0.4, N=400, B=16, T=120, warm=6)
    _drive(pol, 400, 16, 120, 6, record=rec)
    lam = _linear_lambda(6, 120)
    solved = 0
    for r in rec[6:]:
        pi, s = r["pi"], r["scores"]
        assert np.all(pi >= 0.1 - 1e-12) and np.all(pi <= 1.0)
        for k, w in zip(r["keep"], r["weights"]):
            assert w == pytest.approx(1.0 / pi[k])
        # pi is non-decreasing in the score within a batch
        order = np.argsort(s, kind="stable")
        assert np.all(np.diff(pi[order]) >= -1e-12)
        inner = (pi > 0.1 + 1e-9) & (pi < 1.0 - 1e-9)
        if inner.sum() >= 2:
            # between the floor and 1, pi is proportional to w_t * s (one mu per decision)
            ratio = pi[inner] / (lam(r["t"]) * s[inner])
            assert np.allclose(ratio, ratio[0], rtol=1e-9)
            solved += 1
    assert solved > 20


def test_the_random_variants_give_every_example_of_a_batch_the_same_chance():
    for use_lr in (True, False):
        rec = []
        pol = _policy(budget=0.4, use_scores=False, use_lr_weight=use_lr)
        _drive(pol, 400, 16, 100, 6, record=rec)
        for r in rec:
            assert np.all(r["scores"] == 1.0)
            assert np.all(r["pi"] == r["pi"][0])


def test_the_learning_rate_factor_moves_compute_to_the_early_steps():
    def kept_halves(use_lr):
        rec = []
        pol = _policy(budget=0.4, N=800, B=16, T=300, warm=18, use_scores=False, use_lr_weight=use_lr)
        _drive(pol, 800, 16, 300, 18, record=rec)
        first = sum(len(r["keep"]) for r in rec[18:159])
        second = sum(len(r["keep"]) for r in rec[159:])
        return first, second
    f_lr, s_lr = kept_halves(True)
    f_no, s_no = kept_halves(False)
    assert f_lr > 1.5 * s_lr
    assert abs(f_no - s_no) < 0.25 * (f_no + s_no)


def test_scores_steer_the_choice_toward_hard_examples():
    rec = []
    pol = _policy(budget=0.4, N=400, B=16, T=100, warm=6)
    _drive(pol, 400, 16, 100, 6, record=rec)
    # in the second epoch (all examples seen), kept examples have higher scores than dropped ones
    kept_s, drop_s = [], []
    for r in rec[25:]:
        ks = set(r["keep"])
        for i, s in enumerate(r["scores"]):
            (kept_s if i in ks else drop_s).append(s)
    assert np.mean(kept_s) > 1.2 * np.mean(drop_s)


def test_scale_estimates():
    pol = _policy()
    assert pol._revisit_ratio() == 1.0 and pol._first_visit_prior() == 1.0
    pol._win_revisit.extend([(2.0, 4.0), (1.0, 4.0)])
    assert pol._revisit_ratio() == pytest.approx(3.0 / 8.0)
    pol._win_all.append((3.0, 6.0))
    assert pol._first_visit_prior() == pytest.approx(0.5)  # no first visits yet: every kept example
    pol._win_first.append((1.0, 4.0))
    assert pol._first_visit_prior() == pytest.approx(0.25)


def test_stored_scores_are_rescaled_and_unseen_examples_get_the_first_visit_mean():
    pol = _policy(budget=0.6, N=40, B=8, T=20, warm=2)
    tr = _Trainer(20, 2)
    uids = list(range(8))
    mask = torch.ones((8, 4), dtype=torch.long)
    tr.instr.batches_seen = 1
    tr._lerna_batch_uids = uids
    keep, _ = pol.select_examples(tr, {"input_ids": torch.ones((8, 4), dtype=torch.long), "attention_mask": mask})
    tr._lerna_batch_uids = [uids[k] for k in keep]
    pol.observe_forward(tr, {"labels": torch.ones(8, dtype=torch.long)}, _logits([0.4] * 8))
    tr.step()
    # decision 1: four stored examples (0.4, rescaled by a known ratio) and four new ones
    pol._win_revisit.append((1.0, 2.0))  # ratio 0.5
    tr.instr.batches_seen = 2
    tr._lerna_batch_uids = [0, 1, 2, 3, 20, 21, 22, 23]
    pol.select_examples(tr, {"input_ids": torch.ones((8, 4), dtype=torch.long), "attention_mask": mask})
    assert np.allclose(pol.last_scores[:4], 0.4 * 0.5)
    assert np.allclose(pol.last_scores[4:], 0.4)  # first-visit mean of decision 0


def test_strict_failures():
    pol = _policy(T=20, warm=2, N=40, B=8)
    tr = _Trainer(20, 2)
    inp = {"input_ids": torch.ones((8, 4), dtype=torch.long), "attention_mask": torch.ones((8, 4), dtype=torch.long)}
    tr.instr.batches_seen = 1
    tr._lerna_batch_uids = None
    with pytest.raises(StoredScoreError):
        pol.select_examples(tr, inp)  # no ids
    tr._lerna_batch_uids = list(range(7))
    with pytest.raises(StoredScoreError):
        pol.select_examples(tr, inp)  # 8 rows for 7 ids
    tr._lerna_batch_uids = [0, 1, 2, 3, 4, 5, 6, 99]
    with pytest.raises(StoredScoreError):
        pol.select_examples(tr, inp)  # id out of range
    with pytest.raises(StoredScoreError):
        pol.observe_forward(tr, {"labels": torch.ones(8, dtype=torch.long)}, _logits([0.3] * 8))  # no selection
    tr._lerna_batch_uids = list(range(8))
    keep, _ = pol.select_examples(tr, inp)
    with pytest.raises(StoredScoreError):
        pol.select_examples(tr, inp)  # same decision twice
    tr._lerna_batch_uids = [keep[0] + 1] + [keep[k] for k in range(1, len(keep))]
    with pytest.raises(StoredScoreError):
        pol.observe_forward(tr, {"labels": torch.ones(len(keep), dtype=torch.long)}, _logits([0.3] * len(keep)))


def test_a_scheduler_out_of_step_is_rejected():
    pol = _policy(T=20, warm=0, N=40, B=8)
    tr = _Trainer(20, 0)
    tr.lr_scheduler.step()  # one step ahead of the decisions
    tr.instr.batches_seen = 1
    tr._lerna_batch_uids = list(range(8))
    inp = {"input_ids": torch.ones((8, 4), dtype=torch.long), "attention_mask": torch.ones((8, 4), dtype=torch.long)}
    with pytest.raises(StoredScoreError):
        pol.select_examples(tr, inp)


def test_a_wrapped_scheduler_is_unwrapped():
    class _Wrapper:  # like accelerate's AcceleratedScheduler
        def __init__(self, inner):
            self.scheduler = inner

        def step(self):
            self.scheduler.step()

    class _WrappedTrainer(_Trainer):
        def __init__(self, T, warm, scheduler=True):
            super().__init__(T, warm, scheduler)
            self.lr_scheduler = _Wrapper(self.lr_scheduler)

    rec_plain, rec_wrapped = [], []
    _drive(_policy(), 400, 16, 100, 6, record=rec_plain)
    _drive(_policy(), 400, 16, 100, 6, record=rec_wrapped, make_trainer=_WrappedTrainer)
    assert [r["keep"] for r in rec_plain] == [r["keep"] for r in rec_wrapped]
    assert any(len(r["keep"]) < 16 for r in rec_wrapped)


def test_the_no_lr_variants_need_no_scheduler_and_the_lr_variants_do():
    pol = _policy(use_lr_weight=False)
    _drive(pol, 400, 16, 100, 6, scheduler=False)
    assert pol.realized_fraction == pytest.approx(0.4, abs=0.02)
    pol = _policy(use_lr_weight=True)
    with pytest.raises(StoredScoreError):
        _drive(pol, 400, 16, 100, 6, scheduler=False)


def test_every_batch_keeps_at_least_one_example():
    rec = []
    pol = _policy(budget=0.12, pi_min=0.01, N=400, B=4, T=200, warm=0)
    _drive(pol, 400, 4, 200, 0, record=rec)
    assert all(len(r["keep"]) >= 1 for r in rec)
    assert pol.get_diagnostics()["empty_batch_fallbacks"] > 0


def test_selection_is_reproducible_and_depends_on_the_seed():
    def run(seed):
        rec = []
        pol = _policy(policy_seed=seed)
        _drive(pol, 400, 16, 100, 6, record=rec)
        return [tuple(r["keep"]) for r in rec]
    assert run(3) == run(3)
    assert run(3) != run(4)


def test_diagnostics():
    pol = _policy()
    _drive(pol, 400, 16, 100, 6)
    d = pol.get_diagnostics()
    for k in ("budget", "use_scores", "use_lr_weight", "pi_min", "warmup_steps", "realized_fraction",
              "scheduled_units", "spent_units", "pruned_examples", "kept_examples", "empty_batch_fallbacks",
              "never_seen_examples", "epochs", "trajectory", "first_visits_kept", "revisits_kept", "max_weight"):
        assert k in d
    assert d["steps_seen"] == 100 and d["max_weight"] <= 10.0 + 1e-9
    assert d["first_visits_kept"] + d["revisits_kept"] == d["kept_examples"]
    assert d["trajectory"][0]["mu_state"] == "warmup" and d["trajectory"][-1]["t"] == 99
    assert pol.should_skip(None, None, None) is False


# --------------------------------------------------------------------------- the runner arms

def _cfg(arm, total_steps=64, **kw):
    d = dict(
        task="qnli", arm=arm, controller_version="legacy_v4", score_source="uniform",
        total_steps=total_steps, min_step=4, n_phases=1, max_consecutive_skips=4, policy_seed=0,
        target_skip_rate=0.5, output_dir="/tmp/test_round14", attempt_id="t14", capture_logits=True,
        lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
        train_batch_size=8, lr_schedule_shape="linear", lr_warmup_ratio=0.06,
    )
    d.update(kw)
    return ExperimentConfig(**d)


def test_round14_arm_lists():
    # round 15 adds randkeep_b060/b083 and el2n_topk_b035/b060 to these eight
    assert {"lerna2_b035", "lerna2_nolr_b035", "randkeep_lr_b035", "randkeep_b035",
            "lerna2_b040", "randkeep_b040", "lerna2_b060", "lerna2_b083"} <= set(LERNA2_ARMS)
    assert set(LERNA2_ARMS) <= set(ARMS) and set(LERNA2_ARMS) <= set(PLAIN_ARMS)
    assert set(LERNA2_ARMS) <= set(STORED_SCORE_ARMS) and set(LERNA2_ARMS) <= set(EXAMPLE_SELECTION_ARMS)
    assert "infobatch" in EXAMPLE_SELECTION_ARMS and "stale_s1_p4" not in EXAMPLE_SELECTION_ARMS
    # the 2 x 2 factorial at the main budget, 35%
    assert LERNA2_ARMS["lerna2_b035"] == (0.35, True, True)
    assert LERNA2_ARMS["lerna2_nolr_b035"] == (0.35, True, False)
    assert LERNA2_ARMS["randkeep_lr_b035"] == (0.35, False, True)
    assert LERNA2_ARMS["randkeep_b035"] == (0.35, False, False)
    # the original 40% test, and the method at 60% and 83%
    assert LERNA2_ARMS["lerna2_b040"] == (0.40, True, True)
    assert LERNA2_ARMS["randkeep_b040"] == (0.40, False, False)
    assert LERNA2_ARMS["lerna2_b060"] == (0.60, True, True)
    assert LERNA2_ARMS["lerna2_b083"] == (0.83, True, True)
    assert LERNA2_MIN_BUDGET == 0.35
    assert set(GATE_ARMS) | set(PLAIN_ARMS) == set(ARMS) and not set(GATE_ARMS) & set(PLAIN_ARMS)


@pytest.mark.parametrize("field, bad", [("lerna2_pi_min", 0.0), ("lerna2_pi_min", 0.5), ("lerna2_pi_min", 0.35),
                                        ("lerna2_pi_min", True),
                                        ("lerna2_pi_min", float("inf")), ("lerna2_window", 0),
                                        ("lerna2_chunks", 1.5), ("lerna2_window", True)])
def test_lerna2_options_are_validated(field, bad):
    with pytest.raises(ValueError):
        _cfg("lerna2_b040", **{field: bad})


def test_lerna2_dry_run_describes_the_arm():
    r = run_dry_run(_cfg("lerna2_b040", total_steps=6548))
    assert r["status"] == "OK" and r["result"]["total_steps"] == 6548
    assert "LERNA v2" in r["result"]["note"]


@pytest.mark.parametrize("arm", sorted(LERNA2_ARMS))
def test_lerna2_arm_executes_every_step_and_spends_its_budget(arm):
    result = run_execution(_cfg(arm, total_steps=64))
    assert result["status"] == "OK", result.get("reason")
    r = result["result"]
    budget, use_scores, use_lr = LERNA2_ARMS[arm]
    assert r["skipped"] == 0 and r["backwards"] == 64 and r["forwards"] == 64 and r["forward_skipped"] == 0
    assert r["optimizer_updates"] == 64 and r["scheduler_advances"] == 64
    sel = r["selection"]
    assert sel["policy"] == "lerna_budget" and sel["budget"] == budget
    assert sel["use_scores"] is use_scores and sel["use_lr_weight"] is use_lr
    assert r["examples_dropped_before_forward"] == sel["pruned_examples"] > 0
    assert sel["warmup_steps"] == r["lr_warmup_steps"] == 3 and sel["pruned_during_warmup"] == 0
    assert sel["realized_fraction"] == pytest.approx(budget, abs=0.03)
    log = result["batch_log"]
    assert len(log) == 64 and all(len(row) == len(BATCH_LOG_COLUMNS) for row in log)
    assert all(row[3] == row[7] and row[2] == row[8] for row in log[:3])  # whole batches in the warmup
    assert all(1 <= row[3] <= row[7] and row[6] == 1 and row[4] == 0 for row in log)
    assert sum(row[7] - row[3] for row in log) == sel["pruned_examples"]
    s = r["batch_log_summary"]
    # the policy's own accounting is the batch log's, unit for unit
    assert s["compute_units_padded"] == sel["spent_units"]
    assert 3 * s["scheduled_padded_tokens"] == sel["scheduled_units"]
    assert s["compute_fraction_within_arm"] == pytest.approx(sel["realized_fraction"], abs=1e-12)
    assert r["final_learning_rate"] is not None and abs(r["final_learning_rate"]) < 1e-12


def test_lerna2_scheduled_batches_are_those_of_full_fine_tuning():
    full = run_execution(_cfg("full_finetune", total_steps=64))
    mine = run_execution(_cfg("lerna2_b040", total_steps=64))
    assert full["status"] == mine["status"] == "OK"
    assert [row[8] for row in mine["batch_log"]] == [row[2] for row in full["batch_log"]]
    assert [row[7] for row in mine["batch_log"]] == [row[3] for row in full["batch_log"]]
