# LERNA Phase 3 / Phase 1.3 Amendment — Pre-Registration v4

**Status**: Binding pre-registration for Phase 3 forward gate + MRFC smoke;
Stage-2 GPU confirmatory runs contingent on supervisor countersignature (see
`docs/supervisor_signoff_v4.md`).
**Date**: 2026-10-04
**Derived from**: V3 (`FULL_EXPERIMENT_PREREGISTATION_V3.md`); QNLI confirmatory
design (4 arms × 5 seeds × rate 0.30) is unchanged from V3 and inherited.

---

## 1. SHA Registry

**Two distinct SHAs govern V4:**

| Field | Value | Purpose |
|-------|-------|---------|
| `v3_retrospective_sha` | `93ec6e488a6d79f63d021d0f517cf81fdd892801` | Frozen SHA for V3 QNLI confirmatory results (read-only). All 20/20 QNLI cells at rate 0.30 were validated at this commit. |
| `v4_prereg_sha` | `c618959a8c0e0b689585ffb3a648a5cea261cd1a` | SHA of the V4 code commit containing this pre-registration + `lerna/utils/forward_gate.py` + `tests/test_forward_gate.py` + `scripts/smoke_forward_gate_mrpc.py` + `scripts/validate_skip_policy_results.py` (SHA enforcement). |

The V4 code does **not** exist at `93ec6e4` — it is committed separately and
additively. No V4 artifact rebases or modifies the V3 frozen block.

---

## 2. Supervis or Hard Gates (V4 additions)

Both gates are answered in `docs/supervisor_signoff_v4.md`:

1. **LER v4 same line or new method?** → Same line. The v4 forward score
   (Katharopoulos & Fleuret, 2018 gradient-norm upper bound) is already used in
   v3 backward LER. V4 swaps the estimator (forward vs backward), not the
   method. The contribution is a pre-registered comparison under identical
   strict M2c protocol. Answer: **same line**.

2. **Two-sided null acceptable?** → Yes, with pre-registration fidelity.
   The primary test is a two-sided α = 0.05 paired t-test (df = n−1). Separately,
   a pre-registered TOST equivalence bound (two one-sided tests, α = 0.05 each) is
   reported as a secondary analysis to bound the effect size in the event of a null.
   A null is a publishable non-trivial outcome (Kaddour et al., NeurIPS 2023),
   provided the upper bound on effect at n seeds is reported (Path B wording
   mandatory). Answer: **yes with conditions**.

**Physical supervisor signature required** before any Stage-2 GPU cell runs.
The stand-in answers bind procedure only.

---

## 3. Phase 3 Forward Gate Design

### 3.1 Module
`lerna/utils/forward_gate.py` — pure, CPU-testable. Two exported components:

- `ScoreQuantileGate`: deterministic gate scorer. 4 equal phase windows over
  `total_steps`. Per-window quota allocated via largest-remainder (Hare method)
  over **eligible** steps (`decision_index >= min_step`), so `sum(phase_quota) ==
  round(rate * total_steps)` exactly. Forcing engages at pressure ≥ 0.8 (cap
  ceiling 4/5). Annealing at `round(0.88 * total_steps)` replaces score with
  i.i.d. Uniform[0,1] (budget preserved, selection randomized). Non-finite score
  → `TerminalInvalidScoreError` (absorbing).

- `ScoreQuantileGatePolicy`: `SkipPolicy` wrapper plugging `ScoreQuantileGate`
  into `TrueBackwardSkippingTrainer`. `score_source` ∈ {gain, uniform}.

### 3.2 Score Source
`forward_gain_score(logits, labels)` = mean over batch of
`||softmax(logits)_i − onehot(labels)_i||`, bounded by √2 (K&F 2018). Reads
`trainer._last_real_logits` (captured when `capture_logits=True`).

### 3.3 Wiring (Phase 3 matrix)
`phase1_3_matrix.py:64-65`:
- `full_finetune` → `AlwaysFalsePolicy`
- `exact_random` → `RandomSkipPolicy`
- `ler_guided_stratified` → `LERGuidedStratifiedPolicy`

`ScoreQuantileGatePolicy` is the Phase 3 contender/control; it is exercised
only via the MRFC smoke (mechanics check) and the Stage-2 confirmatory cells.

---

## 4. MRFC Smoke Test (Phase 3 mechanics validation)

**Script**: `scripts/smoke_forward_gate_mrpc.py`
**Status**: PASS (2026-10-04)

| Parameter | Value |
|-----------|-------|
| Task | MRPC (GLUE) |
| Model | `jhu-clsp/ettin-encoder-150m` @ `45d08642849e5c5701b162671ac811b7654bfd9f` |
| Total steps | 200 |
| Rate | 0.45 (round(0.45 × 200) = 90) |
| min_step | 50 |
| n_phases | 4 |
| max_consecutive_skips | 4 |
| seed | 11 |
| warmup_steps | 12 |
| Scheduler | `always_step` (pure f-of-step LR contract) |
| Precision | fp32 (no AMP) |
| CUDA device | 0 (V100-SXM2-32GB) |

**Arms (2):**
- `gate`: `ScoreQuantileGatePolicy(score_source="gain")`
- `control`: `ScoreQuantileGatePolicy(score_source="uniform")`

**Invariants verified:**
- `realized_total == requested_total` (90 == 90) both arms
- `sum(phase_used) == requested_total` (0+30+30+30 = 90)
- Per-window skip counts identical: `[0, 30, 30, 30]` == `[0, 30, 30, 30]`
- `scheduler_step_calls == batches_seen == 200` (always_step contract)
- LR curves identical: `max|diff| = 0.00e+00`
- Phase 0 quota 0 (min_step-gated; share transferred to windows 1-3)

**Overhead**: gate +53.7 ms/step over control (forward_gain_score cost, expected).

**Output**: `/raid/smoke_forward_gate_mrpc/` (smoke_summary.json with provenance).

---

## 5. Phase 2 / Stage 2 GPU Confirmatory Runs (PENDING)

| Phase | Status | Gate |
|-------|--------|------|
| Stage 0 | NO-GO (supervisor decision) | Path B |
| Stage 1 | Pending supervisor countersignature | Q1 + Q2 answered (stand-in) |
| Stage 2 | Pending supervisor countersignature | same-line + two-sided TOST |

Stage 2 selects between **rate 0.45** (reachable under the v4 schedule with cap=4)
and **rate 0.60** (may be unreachable under the cap constraint — deferred if so).
The MRFC smoke already validates mechanics at rate 0.45. Stage 2 confirmatory
cells run at the Stage 1-selected rate, not the V3 QNLI 0.30 rate.

No Stage-2 GPU cell shall begin until `docs/supervisor_signoff_v4.md` bears the
real supervisor's signature.

---

## 6. SHA Enforcement Implementation

`scripts/validate_skip_policy_results.py` now accepts `--expected-sha`. When
set, `_check_provenance_sha` enforces:

1. `provenance.v3_retrospective_sha` must equal the V3 SHA (`93ec6e4...`
   when validating QNLI confirmatory results).
2. `provenance.v4_prereg_sha` must equal the V4 code commit SHA when validating
   V4 smoke / Phase 2 confirmatory results.
3. `provenance.tree_dirty` must be `false` for the checked field.

Either SHA mismatch → `SEVERITY_ERROR` (nonzero exit).

**Validation usage:**
```
# Validate V3 QNLI confirmatory results (retrospective SHA):
python scripts/validate_skip_policy_results.py <results.json> \
    --expected-sha 93ec6e488a6d79f63d021d0f517cf81fdd892801

# Validate V4 smoke / Phase 2 results (V4 code SHA):
python scripts/validate_skip_policy_results.py <results.json> \
    --expected-sha <v4_prereg_sha>
```
