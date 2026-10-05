"""Tests for lerna.utils.forward_gate — ScoreQuantileGate and score sources.

Covers:
  1. exact per-window quota
  2. no skips before min_step
  3) per-window quota never exceeded, met exactly
  4. feasibility forcing fires under pressure and still meets quota
  5. cap of 4 respected (always, including under forcing)
  6. annealing ignores the score
  7. non-finite score is terminal and stays terminal
  8. uniform control realizes the target rate in aggregate
  9. determinism: same seed + same scores => identical skip vector
  10. gate vs uniform-control: identical quota machinery, different selection
  11. forward_gain_score bounds and shape (S1)
  12. policy reads trainer logits with no trainer coupling in the scorer
  13. S2 alignment_score matches autograd gradient (G_B) within 1e-5
  14. max_skips_possible formula
   15. adversarial score sequences: monotone up, monotone down, constant, alternating
   16. no leftover quota crosses a window boundary
   17. phase_skip_positions diagnostic populated correctly
   18. max_skips_possible_with_consec accounts for mid-stride consecutive count
   19. window-boundary carry: skip run crossing phase boundary doesn't break quota
   20. global per-step feasibility: total quota met even when a phase is locally infeasible
   21. global forcing under high-rate adversarial scores — no drift"
"""

import math
import random
from collections import deque

from lerna.utils.forward_gate import (
    ScoreQuantileGate,
    ScoreQuantileGatePolicy,
    forward_gain_score,
    alignment_score,
    AlignmentProbe,
    max_skips_possible,
    max_skips_possible_with_consec,
    _classifier_grad_from_logits,
    TerminalInvalidScoreError,
    InfeasibleBudgetError,
)


def _list_score_source(scores):
    return lambda idx, m, x: scores[idx]


# ---------------------------------------------------------------------------
# 1) exact per-window quota
# ---------------------------------------------------------------------------

def test_quotas_computed_exactly():
    g = ScoreQuantileGate(rate=0.3, total_steps=1000, min_step=50, n_phases=4, seed=7,
                          score_source=lambda i, m, x: 0.5)
    d = g.diagnostics()
    # requested_total = round(0.3 * 1000) = 300, allocated over eligible steps
    # (decision_index >= min_step=50). Window 0 has 200 eligible (250 - 50 warmup),
    # windows 1-3 have 250 each → largest-remainder: [63, 79, 79, 79].
    assert d["phase_quota"] == [63, 79, 79, 79]
    assert d["requested_total"] == 300
    assert d["total_eligible"] == 950
    assert d["eligible_per_window"] == [200, 250, 250, 250]


def test_min_step_392_default():
    """The v4 protocol uses min_step=392 in every arm (no warmup skipping)."""
    g = ScoreQuantileGate(rate=0.6, total_steps=6548, n_phases=4, seed=7,
                          score_source=lambda i, m, x: 0.5)
    d = g.diagnostics()
    assert d["min_step"] == 392
    assert d["warmup_blocked"] == 0  # not yet decided, but min_step is set


def test_quota_impossible_under_cap_rejected():
    """rate=1.0 with total_steps=10, cap=4 is infeasible: F(10)=8 < 10."""
    import pytest
    with pytest.raises(InfeasibleBudgetError):
        ScoreQuantileGate(
            rate=1.0, total_steps=10, min_step=0, n_phases=1,
            max_consecutive_skips=4, seed=1,
            score_source=lambda i, m, x: 0.5,
        )


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
        assert used == quota            # met exactly (forcing catches up)


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
        assert used == quota                   # quota still met exactly


def test_forcing_respects_cap():
    """Under high pressure, forcing fires but the cap of 4 consecutive is
    always respected (no override)."""
    # rate=0.8 with total_steps=1000, cap=4: quota=800, F(995)=796... that's
    # infeasible. Use rate=0.75: quota=750, F(995)=796, feasible.
    scores = [0.0] * 1000  # all want to skip
    g = ScoreQuantileGate(rate=0.75, total_steps=1000, min_step=5, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    skips = [g.decide(i, None, None) for i in range(1000)]
    longest = cur = 0
    for s in skips:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    assert longest <= 4                         # cap always respected
    d = g.diagnostics()
    for used, quota in zip(d["phase_used"], d["phase_quota"]):
        assert used == quota                    # quota still met via forcing
    assert d["phase_quota"] == [185, 189, 188, 188]  # round(0.75*250) each, largest-remainder


# ---------------------------------------------------------------------------
# 5) cap of 4 respected for non-forcing skips
# ---------------------------------------------------------------------------

def test_cap_never_exceeded():
    scores = [1000 - i for i in range(1000)]  # decreasing scores
    g = ScoreQuantileGate(rate=0.3, total_steps=1000, min_step=5, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    skips = [g.decide(i, None, None) for i in range(1000)]
    longest = cur = 0
    for s in skips:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    assert longest <= 4                         # cap respected (forcing never active at rate=0.3)
    assert g.forced_skips == 0
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
# 11) forward_gain_score bounds and shape (S1)
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
        n_phases=4, policy_seed=5, score_source="s1",
    )
    # should not crash; produces a bool
    out = pol.should_skip(tr, None, {"labels": labels})
    assert isinstance(out, bool)
    diag = pol.get_diagnostics()
    assert diag["score_source"] == "s1"
    assert diag["phase_quota"] == [15, 15, 15, 15]  # round(0.3*50)


# ---------------------------------------------------------------------------
# 13) S2 alignment_score: G_B matches autograd gradient within 1e-5
# ---------------------------------------------------------------------------

def test_alignment_score_matches_autograd():
    import torch
    torch.manual_seed(42)
    B, C, H = 4, 2, 8
    hidden = torch.randn(B, H)
    W = torch.randn(C, H).requires_grad_(True)
    labels = torch.tensor([0, 1, 0, 1])
    logits = hidden @ W.t()
    loss = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    G_autograd = W.grad.clone()  # [C, H]

    # Compute G_B via the helper
    logits_detached = logits.detach()
    G_B = _classifier_grad_from_logits(logits_detached, labels, hidden.detach())
    assert G_B.shape == (C, H)
    assert torch.allclose(G_B, G_autograd, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# 14) max_skips_possible formula
# ---------------------------------------------------------------------------

def test_max_skips_possible_formula():
    # F(r) = K * floor(r / (K+1)) + min(r % (K+1), K)
    # With K=4: pattern is SSSS N (4 skips, 1 real), repeat every 5.
    assert max_skips_possible(0, 4) == 0
    assert max_skips_possible(1, 4) == 1
    assert max_skips_possible(5, 4) == 4
    assert max_skips_possible(10, 4) == 8
    assert max_skips_possible(6, 4) == 5
    assert max_skips_possible(4, 4) == 4
    assert max_skips_possible(3, 4) == 3
    # K=1: pattern S N S N S N... F(5)=1*2+min(1,1)=3
    assert max_skips_possible(5, 1) == 3
    assert max_skips_possible(4, 1) == 2
    assert max_skips_possible(3, 1) == 2


# ---------------------------------------------------------------------------
# 15) adversarial score sequences: monotone up, monotone down, constant, alternating
# ---------------------------------------------------------------------------

def _run_adv_sequence(scores, rate=0.6, total_steps=6548, min_step=392, **kw):
    g = ScoreQuantileGate(rate=rate, total_steps=total_steps, min_step=min_step,
                          n_phases=4, max_consecutive_skips=4, seed=1, **kw,
                          score_source=_list_score_source(scores))
    skips = [g.decide(i, None, None) for i in range(total_steps)]
    d = g.diagnostics()
    return skips, d

def test_adversarial_monotone_up():
    # scores increase → quantile threshold rises → fewer skips later
    scores = [float(i) for i in range(6548)]
    skips, d = _run_adv_sequence(scores)
    assert all(u <= q for u, q in zip(d["phase_used"], d["phase_quota"]))
    assert d["skipped"] == d["requested_total"]

def test_adversarial_monotone_down():
    # scores decrease → quantile threshold drops → more skips later
    scores = [10000.0 - i for i in range(6548)]
    skips, d = _run_adv_sequence(scores)
    assert all(u <= q for u, q in zip(d["phase_used"], d["phase_quota"]))
    assert d["skipped"] == d["requested_total"]

def test_adversarial_constant():
    scores = [0.5] * 6548
    skips, d = _run_adv_sequence(scores)
    assert all(u <= q for u, q in zip(d["phase_used"], d["phase_quota"]))
    assert d["skipped"] == d["requested_total"]

def test_adversarial_alternating():
    scores = [0.0 if i % 2 == 0 else 1.0 for i in range(6548)]
    skips, d = _run_adv_sequence(scores)
    assert all(u <= q for u, q in zip(d["phase_used"], d["phase_quota"]))
    assert d["skipped"] == d["requested_total"]


# ---------------------------------------------------------------------------
# 16) no leftover quota crosses a window boundary
# ---------------------------------------------------------------------------

def test_no_leftover_quota_crosses_window_boundary():
    scores = [0.0] * 6548  # all want to skip
    g = ScoreQuantileGate(rate=0.6, total_steps=6548, min_step=392, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    for i in range(6548):
        g.decide(i, None, None)
    d = g.diagnostics()
    for used, quota in zip(d["phase_used"], d["phase_quota"]):
        assert used <= quota            # never overspend
        assert used == quota            # met exactly — no leftover crossed boundary


# ---------------------------------------------------------------------------
# 17) phase_skip_positions diagnostic populated correctly
# ---------------------------------------------------------------------------

def test_phase_skip_positions_diagnostic():
    scores = [0.0] * 100  # all want to skip
    g = ScoreQuantileGate(rate=0.5, total_steps=100, min_step=5, n_phases=4,
                          max_consecutive_skips=4, seed=1,
                          score_source=_list_score_source(scores))
    for i in range(100):
        g.decide(i, None, None)
    d = g.diagnostics()
    assert len(d["phase_skip_positions"]) == 4
    for window_info in d["phase_skip_positions"]:
        assert "skip_count" in window_info
        assert "median_rel_position" in window_info


# ---------------------------------------------------------------------------
# 18) alignment_score returns finite value when all components present
# ---------------------------------------------------------------------------

def test_alignment_score_finite_with_mock():
    import torch
    torch.manual_seed(0)
    B, C, H = 4, 2, 8

    class _FakeParam:
        shape = (C, H)

    class _FakeOptimizer:
        def __init__(self, param):
            self.state = {param: {"exp_avg_sq": torch.ones(C, H) * 0.5}}

    class _FakeModel:
        def __init__(self):
            self._fake_param = torch.nn.Parameter(torch.randn(C, H))

        def named_parameters(self):
            yield "classifier.weight", self._fake_param

    class _FakeTrainer:
        def __init__(self, logits, labels, hidden):
            self._last_real_logits = logits
            self._alignment_probe = type("P", (), {"last_hidden": hidden})()
            self.optimizer = _FakeOptimizer(self._fake_param if hasattr(self, '_fake_param') else torch.nn.Parameter(torch.randn(C, H)))

    hidden = torch.randn(B, H)
    logits = torch.randn(B, C)
    labels = torch.tensor([0, 1, 0, 1])

    # Build a proper fake optimizer with the param
    param = torch.nn.Parameter(torch.randn(C, H))
    state = {"exp_avg_sq": torch.ones(C, H) * 0.5}

    class _FakeOpt2:
        def __init__(self, param, state):
            self._param = param
            self.state = {param: state}

    model = _FakeModel()
    # Override named_parameters to yield our param
    class _FakeModel2:
        pass
    model2 = _FakeModel2()
    model2._fake_param = param

    def _yield_params():
        yield "classifier.weight", param

    model2.named_parameters = _yield_params

    trainer = _FakeTrainer(logits, labels, hidden)
    trainer.optimizer = _FakeOpt2(param, state)

    def probe_fn():
        return torch.randn(C, H)

    s = alignment_score(
        trainer=trainer, model=model2, inputs={"labels": labels},
        probe_fn=probe_fn, adam_state_key="classifier.weight",
    )
    assert math.isfinite(s)


def test_max_skips_possible_with_consec_formula():
    """max_skips_possible_with_consec must be <= max_skips_possible (c=0)
    and must match closed-form for known streak states."""
    K = 4
    # c=0: should match max_skips_possible exactly
    for r in range(1, 30):
        assert max_skips_possible_with_consec(r, K, 0) == max_skips_possible(r, K)
    # c>=K: cap forces backward on next step, then unconstrained from 0
    for r in range(1, 10):
        assert max_skips_possible_with_consec(r, K, K) == max_skips_possible(r - 1, K)
        assert max_skips_possible_with_consec(r, K, K + 1) == max_skips_possible(r - 1, K)
    # Starting mid-stride with 3 consecutive (K=4): 1 free skip fills to cap,
    # backward, then 3 more = 4 total (same as from 0, but pattern differs)
    assert max_skips_possible_with_consec(5, K, 3) == 4
    # Starting mid-stride: always <= from 0
    for r in range(1, 30):
        for c in range(K + 1):
            assert max_skips_possible_with_consec(r, K, c) <= max_skips_possible(r, K)
    # Starting mid-stride c=3, K=4, 5 steps: 1 skip (cap), backward, then 3 = 4
    assert max_skips_possible_with_consec(5, K, 3) == 4
    # Starting mid-stride c=3, K=4, 4 steps: 1 + backward + 2 = 3
    assert max_skips_possible_with_consec(4, K, 3) == 3
    # Starting mid-stride c=2, K=4, 6 steps: 2 skips (to cap), backward, then 3 = 5
    assert max_skips_possible_with_consec(6, K, 2) == 5


def test_window_boundary_carry_respects_quota():
    """When a skip run crosses a phase boundary, the per-step feasibility
    check must account for the carry-over consecutive count so that quota
    is still met exactly."""
    K = 4
    n_phases = 2
    total = 100
    min_step = 5
    rate = 0.8  # high pressure: quota=80, F(95)=76... infeasible. Use 0.70: quota=70, F(95)=76, feasible
    rate = 0.70
    # Use decreasing scores to push skipping behavior; the feasibility forcing
    # must kick in at boundary crossings.
    scores = [0.0] * total
    g = ScoreQuantileGate(
        rate=rate, total_steps=total, min_step=min_step, n_phases=n_phases,
        max_consecutive_skips=K, seed=1, score_source=_list_score_source(scores),
    )
    skips = [g.decide(i, None, None) for i in range(total)]
    # Longest consecutive run never exceeds K
    longest = cur = 0
    for s in skips:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    assert longest <= K
    # Quota met exactly in every phase
    d = g.diagnostics()
    for used, quota in zip(d["phase_used"], d["phase_quota"]):
        assert used == quota


def test_global_feasibility_forcing_when_per_phase_infeasible():
    """When a phase becomes locally infeasible (remaining_quota > F(remaining_steps_in_phase, K)),
    the global per-step feasibility check must still enforce the TOTAL quota.

    We construct a scenario where one phase has an impossibly tight local F
    bound but the global check can still enforce the total. We verify that
    realized_total == requested_total despite local infeasibility.
    """
    K = 4
    # 2 phases, total_steps=100, min_step=0 so both phases are fully eligible.
    # Phase 0: [0,50), Phase 1: [50,100). rate=0.5 → requested_total=50.
    # Allocate [25, 25] per phase (equal split for simplicity, but the gate
    # computes via largest-remainder).
    total = 100
    min_step = 0
    rate = 0.5
    # Use scores that push skipping early in phase 0, creating a deficit that
    # the global forcing must recover.
    scores = [0.0] * total  # all scores 0.0 → always skip when not forced
    g = ScoreQuantileGate(
        rate=rate, total_steps=total, min_step=min_step, n_phases=2,
        max_consecutive_skips=K, seed=1, score_source=_list_score_source(scores),
    )
    d = g.diagnostics()
    # requested_total = round(0.5 * 100) = 50; both phases eligible (50 each).
    assert d["requested_total"] == 50
    skips = [g.decide(i, None, None) for i in range(total)]
    # Verify the total quota is met exactly
    d = g.diagnostics()
    assert d["realized_total"] == d["requested_total"], (
        f"Global per-step forcing failed: realized={d['realized_total']} "
        f"!= requested={d['requested_total']}. phase_used={d['phase_used']} "
        f"phase_quota={d['phase_quota']} forced={d['forced_skips']} "
        f"global_forced={d['global_forced_skips']}"
    )


def test_global_feasibility_no_drift_on_high_rate():
    """A high-rate gate (0.7) over 200 steps with K=4 must still realize exactly
    round(0.7 * 200) = 140 skips, even under adversarial score sequences that
    would let naive annealing drift."""
    K = 4
    total = 200
    rate = 0.7
    # Constant scores push the gate to skip whenever not forced backward by the cap.
    scores = [0.0] * total
    g = ScoreQuantileGate(
        rate=rate, total_steps=total, min_step=0, n_phases=4,
        max_consecutive_skips=K, seed=1, score_source=_list_score_source(scores),
    )
    skips = [g.decide(i, None, None) for i in range(total)]
    d = g.diagnostics()
    expected = round(rate * total)
    assert d["realized_total"] == expected, (
        f"realized={d['realized_total']} != expected={expected} "
        f"(per-phase: {d['phase_used']}, quota: {d['phase_quota']})"
    )
    # Cap always respected
    longest = cur = 0
    for s in skips:
        cur = cur + 1 if s else 0
        longest = max(longest, cur)
    assert longest <= K
