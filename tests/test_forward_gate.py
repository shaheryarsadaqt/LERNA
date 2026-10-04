"""CPU tests for lerna.utils.forward_gate.

Pure-Python, no trainer, no GPU. Run: pytest tests/test_forward_gate.py
"""

import math
import random
import pytest

from lerna.utils.forward_gate import (
    InfeasibleBudgetError,
    ScoreQuantileGate,
    ScoreQuantileGatePolicy,
    TerminalInvalidScoreError,
    _eligible_phase_quota,
    forward_gain_score,
)


def _list_score_source(scores):
    return lambda decision_index, model, inputs: float(scores[decision_index])


# ---------------------------------------------------------------------------
# 1) exact per-window quota
# ---------------------------------------------------------------------------

def test_quotas_computed_exactly():
    g = ScoreQuantileGate(rate=0.3, total_steps=1000, n_phases=4, seed=7,
                          score_source=lambda i, m, x: 0.5)
    d = g.diagnostics()
    # requested_total = round(0.3 * 1000) = 300, allocated over eligible steps
    # (decision_index >= min_step=50). Window 0 has 200 eligible (250 - 50 warmup),
    # windows 1-3 have 250 each → largest-remainder: [63, 79, 79, 79].
    assert d["phase_quota"] == [63, 79, 79, 79]
    assert d["requested_total"] == 300
    assert d["total_eligible"] == 950
    assert d["eligible_per_window"] == [200, 250, 250, 250]


# ---------------------------------------------------------------------------
# 2) no skips before min_step
# ---------------------------------------------------------------------------

def test_no_skip_before_min_step():
    scores = [0.0] * 20
    g = ScoreQuantileGate(rate=0.3, total_steps=20, min_step=5, n_phases=4,
                          seed=1, score_source=_list_score_source(scores))
    skips = [g.decide(i, None, None) for i in range(20)]
    assert all(s is False for s in skips[:5])
    assert g.warmup_blocked == 5


# ---------------------------------------------------------------------------
# 3) per-window quota never exceeded, met exactly
# ---------------------------------------------------------------------------

def test_per_window_quota_never_exceeded_and_met():
    scores = [1000 - i for i in range(1000)]  # always low -> always wants skip
    g = ScoreQuantileGate(rate=0.3, total_steps=1000, min_step=5, n_phases=4,
                          seed=1, score_source=_list_score_source(scores))
    for i in range(1000):
        g.decide(i, None, None)
    d = g.diagnostics()
    for used, quota in zip(d["phase_used"], d["phase_quota"]):
        assert used <= quota            # never overspend
        assert used == quota            # met exactly (cap/forcing catch up)


# ---------------------------------------------------------------------------
# 4) feasibility forcing fires under pressure and still meets quota
# ---------------------------------------------------------------------------

def test_feasibility_forcing_fires():
    scores = [0.5] * 1000
    g = ScoreQuantileGate(rate=0.6, total_steps=1000, min_step=5, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    for i in range(1000):
        g.decide(i, None, None)
    d = g.diagnostics()
    assert g.forced_skips > 0                  # forcing engaged
    for used, quota in zip(d["phase_used"], d["phase_quota"]):
        assert used <= quota
        assert used == quota


# ---------------------------------------------------------------------------
# 5) cap of 4 respected
# ---------------------------------------------------------------------------

def test_cap_never_exceeded():
    scores = [1000 - i for i in range(1000)]  # always low -> always wants skip
    g = ScoreQuantileGate(rate=0.3, total_steps=1000, min_step=5, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    skips = [g.decide(i, None, None) for i in range(1000)]
    longest = cur = 0
    for s in skips:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    assert longest <= 4
    assert g.cap_blocked > 0


# ---------------------------------------------------------------------------
# 6) annealing ignores the score
# ---------------------------------------------------------------------------

def test_annealing_blinds_score():
    # With annealing active from step 0, the score is discarded and replaced by
    # uniform RNG. Two gates, identical seed, opposite score streams (0.0 vs 10.0)
    # must therefore make IDENTICAL decisions.
    total = 4000
    zeros = [0.0] * total
    big = [10.0] * total
    A = ScoreQuantileGate(rate=0.3, total_steps=total, min_step=50, n_phases=4,
                          max_consecutive_skips=4, anneal_fraction=0.0, seed=3,
                          score_source=_list_score_source(zeros))
    B = ScoreQuantileGate(rate=0.3, total_steps=total, min_step=50, n_phases=4,
                          max_consecutive_skips=4, anneal_fraction=0.0, seed=3,
                          score_source=_list_score_source(big))
    vA = [A.decide(i, None, None) for i in range(total)]
    vB = [B.decide(i, None, None) for i in range(total)]
    assert vA == vB                          # score fully blinded
    assert A.diagnostics()["anneal_skips"] > 0


# ---------------------------------------------------------------------------
# 7) non-finite score is terminal and stays terminal
# ---------------------------------------------------------------------------

def test_nonfinite_score_terminal():
    def bad_source(idx, m, x):
        if idx == 60:
            return float("nan")
        return 0.4
    g = ScoreQuantileGate(rate=0.3, total_steps=200, min_step=5, n_phases=4,
                          seed=1, score_source=bad_source)
    fired = []
    for i in range(200):
        try:
            g.decide(i, None, None)
        except TerminalInvalidScoreError as e:
            fired.append((i, e.decision_index))
            break
    assert len(fired) == 1 and fired[0][0] == 60
    d = g.diagnostics()
    assert d["terminal"] is True
    assert d["terminal_at"] == 60
    for i in range(61, 200):
        assert g.decide(i, None, None) is False   # absorbed


# ---------------------------------------------------------------------------
# 8) uniform control realizes the target rate in aggregate
# ---------------------------------------------------------------------------

def test_uniform_control_realized_rate():
    rng = random.Random(123)
    src = lambda idx, m, x: rng.random()
    g = ScoreQuantileGate(rate=0.45, total_steps=6548, min_step=50, n_phases=4,
                          max_consecutive_skips=4, seed=9, score_source=src)
    for i in range(6548):
        g.decide(i, None, None)
    d = g.diagnostics()
    realized = d["skipped"] / d["decisions_seen"]
    assert abs(realized - 0.45) < 0.02


# ---------------------------------------------------------------------------
# 9) determinism: same seed + same scores => identical skip vector
# ---------------------------------------------------------------------------

def test_deterministic_for_same_seed_and_scores():
    scores = [float(i % 100) for i in range(6548)]
    g1 = ScoreQuantileGate(rate=0.6, total_steps=6548, min_step=50, n_phases=4,
                           max_consecutive_skips=4, seed=42,
                           score_source=_list_score_source(scores))
    g2 = ScoreQuantileGate(rate=0.6, total_steps=6548, min_step=50, n_phases=4,
                           max_consecutive_skips=4, seed=42,
                           score_source=_list_score_source(scores))
    v1 = [g1.decide(i, None, None) for i in range(6548)]
    v2 = [g2.decide(i, None, None) for i in range(6548)]
    assert v1 == v2
    assert g1.diagnostics()["skipped"] == g2.diagnostics()["skipped"]


# ---------------------------------------------------------------------------
# 10) gate vs uniform-control: identical quota machinery, different selection
# ---------------------------------------------------------------------------

def test_different_score_source_differs_only_in_selection():
    total = 6548
    zeros = [0.0] * total
    rng = random.Random(7)
    uniform = lambda idx, m, x: rng.random()
    g_gate = ScoreQuantileGate(rate=0.3, total_steps=total, min_step=50, n_phases=4,
                               max_consecutive_skips=4, seed=11,
                               score_source=_list_score_source(zeros))
    g_ctrl = ScoreQuantileGate(rate=0.3, total_steps=total, min_step=50, n_phases=4,
                               max_consecutive_skips=4, seed=11, score_source=uniform)
    setA = [i for i in range(total) if g_gate.decide(i, None, None)]
    setB = [i for i in range(total) if g_ctrl.decide(i, None, None)]
    d_gate = g_gate.diagnostics()
    d_ctrl = g_ctrl.diagnostics()
    # identical machinery, identical quotas, identical per-window usage
    assert d_gate["phase_quota"] == d_ctrl["phase_quota"]
    assert d_gate["phase_used"] == d_ctrl["phase_used"]
    # different selections (different index sets), both within quota
    assert set(setA) != set(setB)
    assert d_gate["skipped"] <= sum(d_gate["phase_quota"])
    assert d_ctrl["skipped"] <= sum(d_ctrl["phase_quota"])


# ---------------------------------------------------------------------------
# 11) forward_gain_score bounds and shape
# ---------------------------------------------------------------------------

def test_forward_gain_score_bounds():
    import torch
    torch.manual_seed(0)
    # 2-class logits
    logits = torch.randn(32, 2)
    labels = torch.randint(0, 2, (32,))
    s = forward_gain_score(logits, labels)
    assert 0.0 <= s <= math.sqrt(2) + 1e-6
    # correct & confident => small score
    conf = torch.tensor([[10.0, -10.0]] * 4)
    lab = torch.tensor([0, 0, 0, 0])
    assert forward_gain_score(conf, lab) < 0.05
    # missing inputs -> nan (terminal upstream)
    assert math.isnan(forward_gain_score(None, labels))
    assert math.isnan(forward_gain_score(logits, None))


# ---------------------------------------------------------------------------
# 12) policy reads trainer logits with no trainer coupling in the scorer
# ---------------------------------------------------------------------------

class _FakeInstr:
    batches_seen = 1

class _FakeTrainer:
    def __init__(self, logits, labels):
        self._last_real_logits = logits
        self.instr = _FakeInstr()

def test_policy_should_skip_reads_trainer_logits():
    import torch
    logits = torch.randn(8, 2)
    labels = torch.randint(0, 2, (8,))
    tr = _FakeTrainer(logits, labels)
    pol = ScoreQuantileGatePolicy(
        target_skip_rate=0.3, total_steps=200, min_step=0,
        n_phases=4, policy_seed=5, score_source="gain",
    )
    # should not crash; produces a bool
    out = pol.should_skip(tr, None, {"labels": labels})
    assert isinstance(out, bool)
    diag = pol.get_diagnostics()
    assert diag["score_source"] == "gain"
    assert diag["phase_quota"] == [15, 15, 15, 15]  # round(0.3*50)
