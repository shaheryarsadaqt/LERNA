"""Characterization tests for LERNAPhaseStratifiedPolicy (--policy phase_strat).

These tests LOCK the CURRENT behavior of the Phase-Stratified controller,
including behaviors that may be undesirable (veto-induced quota shortfall,
no cross-phase quota carryover, and LER non-influence). They are intentional
characterization of what the code does today, NOT an endorsement of the
design. Do not "fix" the policy to change these outcomes without updating
these tests deliberately.

Deterministic, lightweight, no models/datasets/GPUs, no statistical asserts.
"""

import pytest

from lerna.trainers.policies import LERNAPhaseStratifiedPolicy


class FakeTracker:
    """Controllable tracker: `rho` drives the rho veto; `loss_history` drives
    the spike veto; `ler`/`ler_raw` exist only to prove LER non-influence."""

    def __init__(self, ler=None, ler_raw=None):
        self.rho = 1.0  # safe by default (above rho_veto_threshold=-0.2)
        self.loss_history = []
        self.ler = ler
        self.ler_raw = ler_raw

    def get_diagnostics(self):
        return {"rho_vg_raw": self.rho, "ler_raw": self.ler_raw, "ler": self.ler}


class FakeState:
    def __init__(self, max_steps):
        self.max_steps = max_steps


class FakeTrainer:
    def __init__(self, max_steps):
        self.state = FakeState(max_steps)


class AlwaysSkipRng:
    """Deterministic RNG: random() == 0.0 always accepts any pressure > 0."""

    def random(self):
        return 0.0


TOTAL_STEPS = 100
MIN_STEP = 20
RATE = 0.40  # quota_size = 40, unambiguous largest-remainder split
SAFE_RHO = 1.0
DANGEROUS_RHO = -1.0  # below default rho_veto_threshold=-0.2


def make_policy(tracker=None, **overrides):
    kwargs = dict(
        ler_tracker=tracker if tracker is not None else FakeTracker(),
        target_skip_rate=RATE,
        total_steps=TOTAL_STEPS,
        min_step=MIN_STEP,
        seed=123,
        n_phases=4,
    )
    kwargs.update(overrides)
    return LERNAPhaseStratifiedPolicy(**kwargs)


def run_horizon(policy, trainer, rho_fn=None):
    decisions = []
    for di in range(trainer.state.max_steps):
        if rho_fn is not None:
            policy.trk.rho = rho_fn(di)
        decisions.append(policy.should_skip(trainer, None, None))
    return decisions


# 1. Fixed equal-length phase boundaries -----------------------------------

def test_equal_length_phase_boundaries_cover_post_min_step_horizon():
    policy = make_policy()
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._lazy_init(trainer)  # lazy init is the only setup step needed
    d = policy.get_diagnostics()

    # Phases cover exactly the post-min_step eligible horizon.
    assert d["phase_bounds"][0] == MIN_STEP
    assert d["phase_bounds"][-1] == TOTAL_STEPS
    assert d["phase_bounds"] == [20, 40, 60, 80, 100]

    # Divisible horizon (80 / 4) -> four equal-length phases.
    assert d["phase_eligible"] == [20, 20, 20, 20]
    assert sum(d["phase_eligible"]) == TOTAL_STEPS - MIN_STEP

    # phase_eligible matches the boundaries exactly.
    bounds = d["phase_bounds"]
    assert d["phase_eligible"] == [
        bounds[i + 1] - bounds[i] for i in range(d["n_phases"])
    ]

    # Boundaries depend only on step position (internal mapping locked).
    assert policy._phase_of(20) == 0
    assert policy._phase_of(39) == 0
    assert policy._phase_of(40) == 1
    assert policy._phase_of(59) == 1
    assert policy._phase_of(60) == 2
    assert policy._phase_of(79) == 2
    assert policy._phase_of(80) == 3
    assert policy._phase_of(99) == 3


# 2. Fixed default quota weights --------------------------------------------

def test_default_weights_and_largest_remainder_phase_quotas():
    policy = make_policy()
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._lazy_init(trainer)
    d = policy.get_diagnostics()

    # Current default normalized weights.
    assert d["phase_weights"] == pytest.approx([0.22, 0.24, 0.26, 0.28])

    # Global integer quota: round(0.40 * 100) = 40.
    assert d["quota_size"] == 40

    # Largest-remainder allocation on raw = [8.8, 9.6, 10.4, 11.2]:
    # floors [8, 9, 10, 11], two remainder units go to phases 0 and 1.
    assert d["phase_quota"] == [9, 10, 10, 11]
    assert sum(d["phase_quota"]) == d["quota_size"]


# 3. Deterministic parity ----------------------------------------------------

def test_identical_config_and_seed_give_identical_decisions():
    pol_a = make_policy(seed=7)
    pol_b = make_policy(seed=7)
    tr_a = FakeTrainer(TOTAL_STEPS)
    tr_b = FakeTrainer(TOTAL_STEPS)

    dec_a = run_horizon(pol_a, tr_a)
    dec_b = run_horizon(pol_b, tr_b)

    assert dec_a == dec_b
    da, db = pol_a.get_diagnostics(), pol_b.get_diagnostics()
    assert da["phase_skips"] == db["phase_skips"]
    assert da["skip_decisions"] == db["skip_decisions"]


# 4. Rho veto behavior --------------------------------------------------------

def test_rho_veto_blocks_skip_increments_counter_and_resets_consecutive():
    # Large max_consecutive_skips so the safety-horizon veto never interferes.
    policy = make_policy(max_consecutive_skips=100)
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._rng = AlwaysSkipRng()  # the RNG alone would always skip

    for _ in range(MIN_STEP):  # warmup steps
        policy.should_skip(trainer, None, None)

    # First eligible step skips (pressure > 0, rng accepts).
    assert policy.should_skip(trainer, None, None) is True
    assert policy._consecutive_skips == 1

    # Dangerous rho on the next eligible step: vetoed despite the RNG.
    policy.trk.rho = DANGEROUS_RHO
    assert policy.should_skip(trainer, None, None) is False
    d = policy.get_diagnostics()
    assert d["rho_veto_count"] == 1
    assert policy._consecutive_skips == 0  # consecutive-skip state resets


# 5. Loss-spike veto behavior -------------------------------------------------

def test_loss_spike_veto_blocks_skip_and_increments_counter():
    policy = make_policy(max_consecutive_skips=100)
    trainer = FakeTrainer(TOTAL_STEPS)
    policy._rng = AlwaysSkipRng()

    for _ in range(MIN_STEP):
        policy.should_skip(trainer, None, None)

    # Current loss 5.0 > previous-window mean 1.0 * (1 + spike_factor=1.0).
    policy.trk.loss_history = [1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    assert policy.should_skip(trainer, None, None) is False
    d = policy.get_diagnostics()
    assert d["spike_veto_count"] == 1
    assert d["skip_decisions"] == 0


# 6. Veto-induced quota shortfall ---------------------------------------------

def test_permanent_veto_causes_forced_global_tail_override_current_behavior():
    # CHARACTERIZATION: when safety vetoes suppress skips, the missed quota is
    # recovered in a forced-global-tail override at the end of the horizon.
    # The policy meets the exact global quota by overriding danger vetoes when
    # every remaining decision must skip. Locked current behavior, not an
    # endorsement of the design.
    policy = make_policy()
    trainer = FakeTrainer(TOTAL_STEPS)

    decisions = run_horizon(policy, trainer, rho_fn=lambda di: DANGEROUS_RHO)

    d = policy.get_diagnostics()
    # No skips during the main eligible horizon — every eligible step is vetoed.
    assert not any(decisions[:TOTAL_STEPS - d["quota_size"]])
    assert d["quota_size"] == 40
    # Exact global quota is still met via forced-tail overrides.
    assert d["skip_decisions"] == d["quota_size"]
    # Veto count matches the pre-tail eligible steps (not the tail itself).
    assert d["rho_veto_count"] == TOTAL_STEPS - MIN_STEP - d["quota_size"]
    assert d["forced_global_tail_skip_count"] == d["quota_size"]
    assert d["forced_safety_override_count"] == d["quota_size"]
    assert d["realized_skip_rate"] == d["target_skip_rate"]


# 7. Current phase carryover behavior ----------------------------------------

def test_missed_phase_quota_is_carried_forward_current_behavior():
    # CHARACTERIZATION: quota missed in a vetoed phase IS transferred to
    # later phases via the cumulative phase-quota accounting; later phases
    # absorb the debt and exceed their own configured quotas. Locked current
    # behavior, not an endorsement of the design.
    # use_safety_horizon=False removes the consecutive-skip cap so safe
    # phases can realize their exact quotas via the forced-tail path.
    policy = make_policy(use_safety_horizon=False)
    trainer = FakeTrainer(TOTAL_STEPS)

    # Phase 0 covers decisions [20, 40): make it all dangerous, rest safe.
    def rho_fn(di):
        return DANGEROUS_RHO if 20 <= di < 40 else SAFE_RHO

    run_horizon(policy, trainer, rho_fn=rho_fn)

    d = policy.get_diagnostics()
    assert d["phase_quota"] == [9, 10, 10, 11]
    # First phase realized zero skips; its quota vanished.
    assert d["phase_skips"][0] == 0
    # Phase-0 debt (9) is carried forward and realized in phase 1.
    assert d["phase_skips"][1] == d["phase_quota"][0] + d["phase_quota"][1]  # 9 + 10 = 19
    # Later safe phases meet exactly their own quotas (forced tail + quota cap).
    assert d["phase_skips"][2] == d["phase_quota"][2]
    assert d["phase_skips"][3] == d["phase_quota"][3]
    # Final skips exactly meet the global quota.
    assert d["skip_decisions"] == d["quota_size"]
    assert d["phase_debt_carried_total"] == d["phase_quota"][0]


# 8. LER non-influence ---------------------------------------------------------

def test_ler_values_do_not_influence_decisions_with_risk_gamma_zero():
    # CHARACTERIZATION: the controller reads only rho and loss history from
    # the tracker; ler / ler_raw are never consulted. Locked current
    # behavior, not an endorsement of the design.
    trk_a = FakeTracker(ler=None, ler_raw=None)
    trk_b = FakeTracker(ler=1234.5, ler_raw=-999.9)
    trk_a.loss_history = [1.0, 0.9, 0.8]
    trk_b.loss_history = [1.0, 0.9, 0.8]

    pol_a = make_policy(tracker=trk_a, seed=99, risk_gamma=0.0)
    pol_b = make_policy(tracker=trk_b, seed=99, risk_gamma=0.0)

    dec_a = run_horizon(pol_a, FakeTrainer(TOTAL_STEPS))
    dec_b = run_horizon(pol_b, FakeTrainer(TOTAL_STEPS))

    assert dec_a == dec_b
    assert pol_a.get_diagnostics()["phase_skips"] == (
        pol_b.get_diagnostics()["phase_skips"]
    )


# 9. use_ler / no_ler parity ---------------------------------------------------

def test_use_ler_flag_has_no_effect_on_phase_stratified_behavior():
    # CHARACTERIZATION: the no_ler ablation (use_ler=False) is a no-op for
    # this policy — the flag is stored but never read in should_skip().
    # Locked current behavior, not an endorsement of the design.
    pol_on = make_policy(seed=11, use_ler=True)
    pol_off = make_policy(seed=11, use_ler=False)

    dec_on = run_horizon(pol_on, FakeTrainer(TOTAL_STEPS))
    dec_off = run_horizon(pol_off, FakeTrainer(TOTAL_STEPS))

    assert dec_on == dec_off
    d_on, d_off = pol_on.get_diagnostics(), pol_off.get_diagnostics()
    assert d_on["phase_skips"] == d_off["phase_skips"]
    assert d_on["skip_decisions"] == d_off["skip_decisions"]