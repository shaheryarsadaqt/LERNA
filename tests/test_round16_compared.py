"""Round 16 runner extensions (7 November 2026): the compared methods of the final comparison.

Covers lerna/utils/compared_methods.py (eps-greedy, UCB, OrderDP, fresh-score importance sampling,
Selective Backprop, KAKURENBO, IES, DropBP), the LERNA v2 arms without the learning-rate factor at
40/60/83%, and the round-16 arms of the runner executed end to end on the CPU fixtures.
All CPU-only.
"""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scripts.run_controller_ablation as runner  # noqa: E402
from lerna.utils.budget_selection import LernaBudgetPolicy  # noqa: E402
from lerna.utils.compared_methods import (  # noqa: E402
    DropBPPolicy,
    FreshImportancePolicy,
    IESPolicy,
    KakurenboPolicy,
    RankedSubsetPolicy,
    SelectiveBackpropPolicy,
)
from lerna.utils.stale_selection import StoredScoreError  # noqa: E402
from scripts.run_controller_ablation import (  # noqa: E402
    ARMS,
    COMPARED_ARMS,
    CONTROLLED_COMPARED_ARMS,
    DROPBP_ARMS,
    EXAMPLE_SELECTION_ARMS,
    FRESH_IS_ARMS,
    LERNA2_ARMS,
    PLAIN_ARMS,
    RANKED_ARMS,
    SB_ARMS,
    STORED_SCORE_ARMS,
    ExperimentConfig,
    run_dry_run,
    run_execution,
)


# --------------------------------------------------------------------------- fixtures
def _linear_lambda(warm, T):
    def f(k):
        if k < warm:
            return k / max(1, warm)
        return max(0.0, (T - k) / max(1, T - warm))
    return f


class _Instr:
    def __init__(self):
        self.batches_seen = 0


class _HardnessModel(torch.nn.Module):
    """Logits from a per-example hardness table: input_ids[:, 0] carries the example id."""

    def __init__(self, hard, T, decay=0.8):
        super().__init__()
        self.hard = np.asarray(hard, dtype=np.float64)
        self.T = T
        self.decay = decay
        self.t = 0
        self.calls = 0
        self.rows = 0

    def residuals(self, uids):
        return np.clip(self.hard[np.asarray(uids)] * (1.0 - self.decay * self.t / self.T), 1e-3, 0.99)

    def forward(self, input_ids, attention_mask=None, **kw):
        self.calls += 1
        self.rows += int(input_ids.shape[0])
        uids = [int(u) for u in input_ids[:, 0].tolist()]
        return SimpleNamespace(logits=_logits(self.residuals(uids)))


class _Trainer:
    def __init__(self, T, warm, model=None):
        self.instr = _Instr()
        self._lerna_batch_uids = None
        p = torch.nn.Parameter(torch.zeros(1))
        self.opt = torch.optim.SGD([p], lr=1.0)
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.opt, _linear_lambda(warm, T))
        self.model = model

    def step(self):
        self.opt.step()
        self.lr_scheduler.step()


def _logits(residuals):
    """Two-class logits whose label-1 residual ||p - y|| is the given value."""
    rows = []
    for r in residuals:
        pt = 1.0 - float(r) / math.sqrt(2.0)
        rows.append([0.0, math.log(pt / (1.0 - pt))])
    return torch.tensor(rows, dtype=torch.float32)


def _drive(pol, N, B, T, warm, seed=0, record=None, hard=None, decay=0.8):
    """Run a policy over T scheduled batches of a shuffled N-example dataset (epochs of ceil(N/B))."""
    rng = np.random.default_rng(seed)
    length = rng.integers(5, 40, size=N)
    if hard is None:
        hard = rng.uniform(0.02, 0.9, size=N)
    model = _HardnessModel(hard, T, decay)
    tr = _Trainer(T, warm, model=model)
    t = 0
    while t < T:
        order = rng.permutation(N)
        for b in range(0, N, B):
            if t >= T:
                break
            uids = [int(u) for u in order[b:b + B]]
            n, width = len(uids), int(length[uids].max())
            mask = torch.zeros((n, width), dtype=torch.long)
            ids = torch.ones((n, width), dtype=torch.long)
            for i, u in enumerate(uids):
                mask[i, :length[u]] = 1
                ids[i, 0] = u
            labels = torch.ones(n, dtype=torch.long)
            tr.instr.batches_seen += 1
            tr._lerna_batch_uids = list(uids)
            model.t = t
            inputs = {"input_ids": ids, "attention_mask": mask, "labels": labels}
            calls_before = model.calls
            keep, weights = pol.select_examples(tr, inputs)
            if record is not None:
                record.append({"t": t, "keep": list(keep), "weights": list(weights), "n": n, "uids": uids,
                               "width": width, "lengths": [int(length[u]) for u in uids],
                               "extra_calls": model.calls - calls_before,
                               "pi": None if getattr(pol, "last_pi", None) is None else pol.last_pi.copy()})
            kept = [uids[k] for k in keep]
            tr._lerna_batch_uids = kept
            if hasattr(pol, "observe_forward"):
                pol.observe_forward(tr, {"labels": torch.ones(len(kept), dtype=torch.long)},
                                    _logits(model.residuals(kept)))
            tr.step()
            t += 1
    return model


# --------------------------------------------------------------------------- arm lists
def test_round16_arm_lists():
    new = set(COMPARED_ARMS) | {"lerna2_nolr_b040", "lerna2_nolr_b060", "lerna2_nolr_b083"}
    assert new <= set(ARMS) and new <= set(PLAIN_ARMS) and new <= set(EXAMPLE_SELECTION_ARMS)
    assert set(COMPARED_ARMS) == {"egreedy_b035", "ucb_b035", "orderdp_b035", "isfresh_b060", "isfresh_b083",
                                  "sb_b060", "sb_b083", "kakurenbo", "ies", "dropbp_p050", "dropbp_p0875"}
    assert RANKED_ARMS == {"egreedy_b035": (0.35, "egreedy"), "ucb_b035": (0.35, "ucb"),
                           "orderdp_b035": (0.35, "orderdp")}
    assert FRESH_IS_ARMS == {"isfresh_b060": 0.60, "isfresh_b083": 0.83}
    assert SB_ARMS == {"sb_b060": 0.60, "sb_b083": 0.83}
    assert DROPBP_ARMS == {"dropbp_p050": 0.5, "dropbp_p0875": 0.875}
    assert set(CONTROLLED_COMPARED_ARMS) == set(RANKED_ARMS) | set(FRESH_IS_ARMS)
    for arm in ("lerna2_nolr_b040", "lerna2_nolr_b060", "lerna2_nolr_b083"):
        assert LERNA2_ARMS[arm] == (float(arm[-3:]) / 100.0, True, False)
    # the stored-score arms carry lerna_uid; Selective Backprop and DropBP don't need it
    assert set(RANKED_ARMS) | set(FRESH_IS_ARMS) | {"kakurenbo", "ies"} <= set(STORED_SCORE_ARMS)
    assert not (set(SB_ARMS) | set(DROPBP_ARMS)) & set(STORED_SCORE_ARMS)
    assert len(ARMS) == len(set(ARMS))


# --------------------------------------------------------------------------- eps-greedy / UCB / OrderDP
def _ranked(selection, budget=0.35, N=800, B=16, T=300, warm=18, **kw):
    return RankedSubsetPolicy(budget=budget, total_steps=T, num_examples=N, batches_per_epoch=math.ceil(N / B),
                              warmup_steps=warm, selection=selection, **kw)


def test_ranked_selection_is_validated():
    with pytest.raises(ValueError):
        _ranked("sample")
    with pytest.raises(ValueError):
        _ranked("topk")
    with pytest.raises(ValueError):
        _ranked("egreedy", epsilon=1.5)
    with pytest.raises(ValueError):
        _ranked("orderdp", orderdp_ratio=0.0)
    assert RankedSubsetPolicy.SELECTIONS == ("egreedy", "ucb", "orderdp")


@pytest.mark.parametrize("selection", ["egreedy", "ucb", "orderdp"])
def test_ranked_keeps_the_uniform_controllers_k_and_spends_the_budget(selection):
    rec = []
    pol = _ranked(selection)
    _drive(pol, 800, 16, 300, 18, record=rec)
    for r in rec:
        k = max(1, min(r["n"], int(round(float(r["pi"].sum())))))
        assert len(r["keep"]) == k and len(set(r["keep"])) == k
        assert r["weights"] == [pytest.approx(r["n"] / k)] * k
        assert np.allclose(r["pi"], r["pi"][0])  # equal scores: one keep probability per batch
    for r in rec[:18]:
        assert r["keep"] == list(range(16)) and r["weights"] == [1.0] * 16
    d = pol.get_diagnostics()
    assert d["pruned_during_warmup"] == 0 and d["use_scores"] is False and d["use_lr_weight"] is False
    assert d["method"] == selection and d["selection"] == selection
    assert pol.realized_fraction == pytest.approx(0.35, abs=0.006)


def test_ucb_and_orderdp_keep_the_highest_ranked():
    for selection in ("ucb", "orderdp", "egreedy"):
        rec = []
        pol = _ranked(selection)
        # record the rank values just before each choice
        ranks = []
        orig = pol._rank_values

        def spy(uids, _orig=orig):
            v = _orig(uids)
            ranks.append(v.copy())
            return v
        pol._rank_values = spy
        _drive(pol, 800, 16, 300, 18, record=rec)
        chosen = [r for r in rec if len(r["keep"]) < r["n"]]
        assert len(chosen) == len(ranks) > 100
        for r, v in zip(chosen, ranks):
            k = len(r["keep"])
            kept = sorted(v[i] for i in r["keep"])
            dropped = [v[i] for i in range(r["n"]) if i not in set(r["keep"])]
            if selection == "ucb":
                assert kept[0] >= max(dropped) - 1e-12
            elif selection == "egreedy":
                g = int(round(0.9 * k))
                top = sorted(range(r["n"]), key=lambda i: (-v[i], i))[:g]
                assert set(top) <= set(r["keep"])
            else:
                # OrderDP: a candidate set of ceil(k / 0.6), its k highest kept, so at most
                # n - ceil(k / 0.6) examples outrank the lowest kept one
                c = min(r["n"], max(k, math.ceil(k / 0.6)))
                assert sum(1 for x in dropped if x > kept[0] + 1e-12) <= r["n"] - c


def test_ranked_ema_and_variance_follow_raju_et_al():
    pol = _ranked("ucb", N=4, B=4, T=40, warm=0)
    tr = _Trainer(40, 0)
    seq = [0.5, 0.2, 0.9]
    ema = var = None
    for x in seq:
        tr.instr.batches_seen += 1
        tr._lerna_batch_uids = [0, 1, 2, 3]
        inputs = {"input_ids": torch.ones((4, 5), dtype=torch.long), "attention_mask": torch.ones((4, 5), dtype=torch.long)}
        keep, _ = pol.select_examples(tr, inputs)
        tr._lerna_batch_uids = [[0, 1, 2, 3][k] for k in keep]
        losses = {u: x for u in tr._lerna_batch_uids}
        # logits whose cross-entropy for label 1 is x
        lg = torch.tensor([[0.0, math.log(math.exp(-x) / (1 - math.exp(-x)))] for _ in tr._lerna_batch_uids])
        pol.observe_forward(tr, {"labels": torch.ones(len(keep), dtype=torch.long)}, lg)
        if 0 in losses:
            if ema is None:
                ema, var = x, 0.0
            else:
                var = 0.2 * var + 0.8 * (x - ema) ** 2
                ema = 0.8 * x + 0.2 * ema
        tr.step()
    assert pol._ema[0] == pytest.approx(ema, rel=1e-5)
    assert pol._var[0] == pytest.approx(var, rel=1e-4)


def test_ranked_is_reproducible_and_seed_dependent_where_random():
    def run(selection, seed):
        rec = []
        _drive(_ranked(selection, policy_seed=seed), 800, 16, 300, 18, record=rec)
        return [tuple(r["keep"]) for r in rec]
    assert run("orderdp", 3) == run("orderdp", 3)
    assert run("orderdp", 3) != run("orderdp", 4)
    assert run("ucb", 3) == run("ucb", 4)  # deterministic


# --------------------------------------------------------------------------- fresh-score importance sampling
def _fresh(budget=0.6, N=800, B=16, T=300, warm=18, **kw):
    return FreshImportancePolicy(budget=budget, total_steps=T, num_examples=N, batches_per_epoch=math.ceil(N / B),
                                 warmup_steps=warm, **kw)


@pytest.mark.parametrize("budget", [0.6, 0.83])
def test_fresh_importance_spends_the_budget_including_its_scoring_passes(budget):
    rec = []
    pol = _fresh(budget)
    model = _drive(pol, 800, 16, 300, 18, record=rec)
    d = pol.get_diagnostics()
    assert d["scoring_passes"] == 300 - 18 == model.calls
    assert d["extra_forward_padded_tokens"] == sum(r["n"] * r["width"] for r in rec[18:])
    assert d["overhead_units"] == d["extra_forward_padded_tokens"]
    assert all(r["extra_calls"] == (0 if r["t"] < 18 else 1) for r in rec)
    assert d["pruned_during_warmup"] == 0 and d["use_lr_weight"] is False
    assert pol.realized_fraction == pytest.approx(budget, abs=0.006)
    kept_units = sum(3 * len(r["keep"]) * max(r["lengths"][i] for i in r["keep"]) for r in rec)
    sched = sum(3 * r["n"] * r["width"] for r in rec)
    assert (kept_units + d["extra_forward_padded_tokens"]) / sched == pytest.approx(pol.realized_fraction, abs=1e-12)


def test_fresh_importance_samples_by_the_fresh_residual():
    rec = []
    pol = _fresh(0.6)
    _drive(pol, 800, 16, 300, 18, record=rec)
    # after the warmup, pi is proportional to the fresh residual (until clipped), weights 1 / pi
    for r in rec[18:]:
        pi = r["pi"]
        assert all(w == pytest.approx(1.0 / pi[i]) for i, w in zip(r["keep"], r["weights"]))
    later = [r for r in rec if r["t"] >= 18]
    s = np.concatenate([pol_scores for pol_scores in [np.asarray(x["pi"]) for x in later]])
    assert s.min() >= 0.1 - 1e-12 and s.max() <= 1.0


def test_fresh_importance_rejects_a_budget_its_scoring_cannot_afford():
    with pytest.raises(ValueError):
        _fresh(0.40)


# --------------------------------------------------------------------------- Selective Backprop
def _sb(budget=0.6, N=800, B=16, T=300, **kw):
    return SelectiveBackpropPolicy(budget=budget, total_steps=T, num_examples=N,
                                   batches_per_epoch=math.ceil(N / B), **kw)


def test_selective_backprop_beta_matches_the_budget():
    assert _sb(0.6).beta == pytest.approx(1.0 / (0.6 - 1.0 / 3.0) - 1.0)
    assert _sb(0.83).beta == pytest.approx(1.0 / (0.83 - 1.0 / 3.0) - 1.0)
    with pytest.raises(ValueError):
        _sb(0.30)


@pytest.mark.parametrize("budget", [0.6, 0.83])
def test_selective_backprop_keeps_the_expected_share_by_loss(budget):
    # losses that do not drift: the CDF over the recent history is then uniform, so the kept
    # share is 1 / (1 + beta) = budget - 1/3 (falling losses make it smaller: the run's measured
    # compute is what the round reports)
    rec = []
    pol = _sb(budget)
    model = _drive(pol, 800, 16, 300, 0, record=rec, decay=0.0)
    d = pol.get_diagnostics()
    assert d["scoring_passes"] == 300 == model.calls
    assert d["extra_forward_padded_tokens"] == sum(r["n"] * r["width"] for r in rec)
    kept_share = d["kept_examples"] / d["scheduled_examples"]
    assert kept_share == pytest.approx(budget - 1.0 / 3.0, abs=0.03)
    for r in rec:
        assert r["weights"] == [pytest.approx(r["n"] / len(r["keep"]))] * len(r["keep"])
    # the kept examples are the harder ones on average
    hard_kept, hard_all = [], []
    for r in rec[50:]:
        res = model.hard[np.asarray(r["uids"])]
        hard_kept += [res[i] for i in r["keep"]]
        hard_all += list(res)
    assert np.mean(hard_kept) > np.mean(hard_all) * 1.2


def test_selective_backprop_history_is_bounded_and_seeded():
    def run(seed):
        rec = []
        _drive(_sb(0.6, policy_seed=seed), 800, 16, 120, 0, record=rec)
        return [tuple(r["keep"]) for r in rec]
    assert run(1) == run(1) and run(1) != run(2)
    pol = _sb(0.6, history=50)
    _drive(pol, 800, 16, 40, 0)
    assert len(pol._hist) == 50


# --------------------------------------------------------------------------- KAKURENBO
def _kak(N=800, B=16, T=500, **kw):
    return KakurenboPolicy(total_steps=T, num_examples=N, batches_per_epoch=math.ceil(N / B), **kw)


def test_kakurenbo_hides_the_lowest_loss_confident_examples_from_the_second_epoch():
    rec = []
    pol = _kak()
    _drive(pol, 800, 16, 500, 0, record=rec)
    d = pol.get_diagnostics()
    ep = d["epochs"]
    assert [e["epoch"] for e in ep] == list(range(10))
    assert ep[0]["hidden"] == 0 and all(e["hidden"] > 0 for e in ep[1:])
    # F_e = 0.3 x alpha by the fraction of training at the epoch's start
    for e in ep:
        frac = e["fraction_of_training"]
        alpha = 1.0 if frac < 0.3 else 0.8 if frac < 0.6 else 0.6 if frac < 0.8 else 0.4
        assert e["max_fraction"] == pytest.approx(0.3 * alpha)
        assert e["hidden"] <= math.floor(e["max_fraction"] * 800 + 1e-9)
        assert e["kept_weight"] == pytest.approx(1.0 / (1.0 - e["hidden"] / 800.0))
    # every kept example in epoch e carries epoch e's weight; refresh passes run before the last epoch only
    bpe = 50
    for r in rec:
        e = r["t"] // bpe
        assert r["weights"] == [pytest.approx(ep[e]["kept_weight"])] * len(r["keep"])
        hidden_here = r["n"] - len(r["keep"])
        assert r["extra_calls"] == (1 if hidden_here and e < 9 else 0)
    assert d["pruned_examples"] == sum(r["n"] - len(r["keep"]) for r in rec)


def test_kakurenbo_moves_back_wrong_or_unsure_examples():
    pol = _kak(N=10, B=10, T=20)
    tr = _Trainer(20, 0)
    pol._loss[:] = np.linspace(0.01, 0.5, 10)
    pol._correct[:] = True
    pol._conf[:] = 0.95
    pol._correct[0] = False  # the lowest-loss example was wrong: moved back
    pol._conf[1] = 0.6       # unsure: moved back
    pol._start_epoch(1)
    e = pol.epochs[-1]
    assert e["candidates"] == 3 and e["moved_back"] == 2 and e["hidden"] == 1
    assert pol._hidden.tolist() == [False, False, True] + [False] * 7


# --------------------------------------------------------------------------- IES
def _ies(N=64, B=16, T=4 * 8, **kw):
    return IESPolicy(total_steps=T, num_examples=N, batches_per_epoch=math.ceil(N / B), **kw)


def test_ies_needs_three_earlier_epochs_and_excludes_mastered_examples():
    N, B = 64, 16
    pol = _ies(N, B, T=4 * 8)
    tr = _Trainer(32, 0)
    rng = np.random.default_rng(0)
    kept_by_epoch = {}
    for t in range(32):
        e = t // 4
        b = t % 4
        uids = list(range(b * B, (b + 1) * B))
        tr.instr.batches_seen += 1
        tr._lerna_batch_uids = list(uids)
        # examples 0..31 have a loss that stops changing; 32..63 keep changing
        def loss_of(u, e=e):
            return 0.3 if u < 32 else 0.3 + 0.1 * (e % 2)
        model_logits = {u: loss_of(u) for u in uids}

        class _M(torch.nn.Module):
            def forward(self, input_ids, attention_mask=None, **kw):
                us = [int(x) for x in input_ids[:, 0].tolist()]
                return SimpleNamespace(logits=torch.tensor(
                    [[0.0, math.log(math.exp(-model_logits[u]) / (1 - math.exp(-model_logits[u])))] for u in us]))
        tr.model = _M()
        ids = torch.ones((B, 6), dtype=torch.long)
        ids[:, 0] = torch.tensor(uids)
        keep, w = pol.select_examples(tr, {"input_ids": ids, "attention_mask": torch.ones((B, 6), dtype=torch.long),
                                           "labels": torch.ones(B, dtype=torch.long)})
        kept = [uids[k] for k in keep]
        kept_by_epoch.setdefault(e, []).extend(kept)
        tr._lerna_batch_uids = kept
        lg = tr.model(input_ids=ids[keep]).logits
        pol.observe_forward(tr, {"labels": torch.ones(len(kept), dtype=torch.long)}, lg)
        assert w == [pytest.approx(B / len(keep))] * len(keep)
        tr.step()
    for e in range(3):
        assert sorted(kept_by_epoch[e]) == list(range(N))
    # from the fourth epoch the mastered examples are excluded; the two batches made only of mastered
    # examples each keep one (the first, every second difference being 0), so a batch is never empty
    for e in range(3, 8):
        assert sorted(kept_by_epoch[e]) == [0, 16] + list(range(32, 64))
    d = pol.get_diagnostics()
    assert d["epochs"]["3"]["mastered"] == 30 and d["epochs"]["2"]["mastered"] == 0
    assert d["extra_forward_examples"] == 30 * 5 and d["empty_batch_fallbacks"] == 2 * 5
    assert d["reincluded"] == 0


def test_ies_brings_back_an_example_whose_loss_moves_again():
    pol = _ies(N=4, B=4, T=40)
    pol._loss[0] = [0.3, 0.3, 0.3]
    pol._epoch[0] = [2, 1, 0]
    tr = _Trainer(40, 0)
    tr.model = None
    # at epoch 3 example 0 is mastered (excluded); after a jump it comes back
    pol._was_mastered[0] = True
    pol._loss[0] = [0.9, 0.3, 0.3]
    pol._epoch[0] = [3, 2, 1]
    pol.steps_seen = 16
    tr.instr.batches_seen = 17
    tr._lerna_batch_uids = [0, 1, 2, 3]
    keep, _ = pol.select_examples(tr, {"input_ids": torch.ones((4, 3), dtype=torch.long),
                                       "attention_mask": torch.ones((4, 3), dtype=torch.long),
                                       "labels": torch.ones(4, dtype=torch.long)})
    assert 0 in keep and pol.reincluded == 1


# --------------------------------------------------------------------------- DropBP
def _tiny_modernbert(layers=3):
    from transformers import ModernBertConfig, ModernBertForSequenceClassification
    kw = dict(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=layers, num_attention_heads=4,
              pad_token_id=0, bos_token_id=1, eos_token_id=2, cls_token_id=1, sep_token_id=2, num_labels=2,
              max_position_embeddings=64)
    try:
        cfg = ModernBertConfig(**kw, reference_compile=False)
    except TypeError:
        cfg = ModernBertConfig(**kw)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    return ModernBertForSequenceClassification(cfg)


def _grads(model, ids, mask, labels):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    return float(out.loss), {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}


def test_dropbp_leaves_the_forward_pass_and_undropped_runs_unchanged():
    model = _tiny_modernbert()
    model.train()
    ids = torch.randint(3, 64, (4, 10))
    mask = torch.ones_like(ids)
    mask[0, 7:] = 0
    labels = torch.tensor([0, 1, 1, 0])
    l0, g0 = _grads(model, ids, mask, labels)
    pol = DropBPPolicy(model=model, drop_rate=0.5, total_steps=10)
    l1, g1 = _grads(model, ids, mask, labels)
    assert l0 == l1 and all(torch.equal(g0[n], g1[n]) for n in g0)
    for li, kind, mod in pol._blocks:
        mod._lerna_dropbp = (li == 1 and kind == "attn") or (li == 2 and kind == "mlp")
    l2, g2 = _grads(model, ids, mask, labels)
    assert l2 == l0
    assert sorted(n for n in g2 if g2[n] is None) == sorted([
        "model.layers.1.attn.Wqkv.weight", "model.layers.1.attn.Wo.weight", "model.layers.1.attn_norm.weight",
        "model.layers.2.mlp.Wi.weight", "model.layers.2.mlp.Wo.weight", "model.layers.2.mlp_norm.weight"])
    with torch.no_grad():
        out = model(input_ids=ids, attention_mask=mask)
    assert out.logits.shape == (4, 2)
    with pytest.raises(ValueError):
        DropBPPolicy(model=model, drop_rate=0.5, total_steps=10)  # already wrapped


def test_dropbp_drop_rate_and_backward_share():
    model = _tiny_modernbert()
    pol = DropBPPolicy(model=model, drop_rate=0.875, total_steps=2000, policy_seed=4)
    tr = _Trainer(2000, 0)
    for t in range(2000):
        tr.instr.batches_seen += 1
        keep, w = pol.select_examples(tr, {"input_ids": torch.ones((4, 9), dtype=torch.long),
                                           "attention_mask": torch.ones((4, 9), dtype=torch.long)})
        assert keep == [0, 1, 2, 3] and w == [1.0] * 4
    d = pol.get_diagnostics()
    assert d["dropped_block_fraction"] == pytest.approx(0.875, abs=0.01)
    f = d["flops_per_token"]
    assert f == {"layers": 3, "attn_linear": 32 * 96 + 32 * 32, "attn_scores_per_position": 64,
                 "mlp": 32 * 96 + 48 * 32}
    a, m = pol.drop_counts[7]
    attn = f["attn_linear"] + 64 * 9
    assert pol.backward_fraction(7, 9) == pytest.approx(1 - (a * attn + m * f["mlp"]) / (3 * (attn + f["mlp"])))
    shares = [pol.backward_fraction(t, 9) for t in range(2000)]
    assert np.mean(shares) == pytest.approx(0.125, abs=0.01)
    with pytest.raises(ValueError):
        DropBPPolicy(model=torch.nn.Linear(2, 2), drop_rate=0.5, total_steps=10)


# --------------------------------------------------------------------------- the runner, end to end
def _cfg(arm, total_steps=64, **kw):
    d = dict(
        task="qnli", arm=arm, controller_version="legacy_v4", score_source="uniform",
        total_steps=total_steps, min_step=4, n_phases=1, max_consecutive_skips=4, policy_seed=0,
        target_skip_rate=0.5, output_dir="/tmp/test_round16", attempt_id="t16", capture_logits=True,
        lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
        train_batch_size=8, lr_schedule_shape="linear", lr_warmup_ratio=0.06,
    )
    d.update(kw)
    return ExperimentConfig(**d)


class _LayeredTiny(torch.nn.Module):
    """A CPU fixture model with encoder layers that have .attn and .mlp blocks (for DropBP)."""

    class _Layer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = torch.nn.Linear(8, 8)
            self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 12), torch.nn.ReLU(), torch.nn.Linear(12, 8))

        def forward(self, h):
            h = h + self.attn(h)
            return h + self.mlp(h)

    def __init__(self, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([self._Layer(), self._Layer()])
        self.classifier = torch.nn.Linear(8, 2)

    def forward(self, input_ids, labels=None, **kwargs):
        h = input_ids.float()
        for layer in self.model.layers:
            h = layer(h)
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels.view(-1))
        return SimpleNamespace(loss=loss, logits=logits)


@pytest.mark.parametrize("arm", [a for a in COMPARED_ARMS if a not in DROPBP_ARMS])
def test_compared_arm_executes_and_accounts_its_compute(arm):
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    res = result["result"]
    sel = res["selection"]
    assert sel["steps_seen"] == 64
    assert res["examples_dropped_before_forward"] == sel["pruned_examples"]
    ca = res["compute_accounting"]
    log = result["batch_log"]
    assert ca["forward_padded_tokens"] == sum(r[2] for r in log)
    assert ca["extra_forward_padded_tokens"] == sel["extra_forward_padded_tokens"]
    assert ca["compute_units"] == ca["forward_padded_tokens"] + ca["extra_forward_padded_tokens"] + 2 * sum(r[2] for r in log)
    if arm in CONTROLLED_COMPARED_ARMS:
        assert sel["pruned_during_warmup"] == 0
        assert ca["compute_fraction_within_arm"] == pytest.approx(sel["realized_fraction"], abs=1e-12)
    if arm in SB_ARMS or arm in FRESH_IS_ARMS:
        assert sel["extra_forward_padded_tokens"] > 0


@pytest.mark.parametrize("arm", ["dropbp_p050", "dropbp_p0875"])
def test_dropbp_arm_executes(arm, monkeypatch):
    monkeypatch.setattr(runner, "_load_model", lambda cfg, model_loader=None: _LayeredTiny(cfg.policy_seed))
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    res = result["result"]
    sel = res["selection"]
    assert sel["steps_seen"] == 64 and len(sel["drop_counts"]) == 64 and sel["pruned_examples"] == 0
    assert sel["dropped_block_fraction"] == pytest.approx(DROPBP_ARMS[arm], abs=0.15)
    ca = res["compute_accounting"]
    assert ca["extra_forward_padded_tokens"] == 0
    assert ca["backward_token_equivalents"] < ca["forward_padded_tokens"]


def test_dropbp_on_the_tiny_fixture_model_fails_cleanly():
    result = run_execution(_cfg("dropbp_p050"))
    assert result["status"] == "FAILED" and result["failure_stage"] == "policy_construction"


@pytest.mark.parametrize("arm", ["lerna2_nolr_b040", "lerna2_nolr_b060", "lerna2_nolr_b083"])
def test_lerna2_without_the_lr_factor_at_the_other_budgets(arm):
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    sel = result["result"]["selection"]
    assert sel["use_scores"] is True and sel["use_lr_weight"] is False and sel["selection"] == "sample"
    assert sel["budget"] == LERNA2_ARMS[arm][0]
    assert "compute_accounting" not in result["result"]


@pytest.mark.parametrize("arm", COMPARED_ARMS)
def test_dry_run_describes_each_compared_arm(arm):
    r = run_dry_run(_cfg(arm, total_steps=6548))
    assert r["status"] == "OK" and r["result"]["note"].endswith(runner.COMPARED_NOTES[arm])
    assert "LERNA v2)" not in r["result"]["note"]
