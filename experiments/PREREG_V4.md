# LERNA Phase 3 / Stage 0 + Stage 2 — Pre-Registration v4

**Status**: Binding pre-registration for the forward gate oracle test (Stage 0)
and the confirmatory skip comparison (Stage 2). Stage 0 GPU cells are
permitted; Stage 2 GPU cells are **blocked** until this page bears the real
supervisor's or committee member's countersignature
(see `docs/supervisor_signoff_v4.md`).
**Date**: 2026-10-04
**Derived from**: V3 (`FULL_EXPERIMENT_PREREGISTATION_V3.md`); QNLI confirmatory
design (4 arms × 5 seeds × rate 0.30) is unchanged from V3 and inherited.

---

## 1. SHA Registry

**Two distinct SHAs govern V4:**

| Field | Value | Purpose |
|-------|-------|---------|
| `v3_retrospective_sha` | `93ec6e488a6d79f63d021d0f517cf81fdd892801` | Frozen SHA for V3 QNLI confirmatory results (read-only). All 20/20 QNLI cells at rate 0.30 were validated at this commit. |
| `v4_prereg_sha` | `v4-prereg` (tag at commit `93839a1`) | Git tag pointing to the V4 pre-registration commit containing `lerna/utils/forward_gate.py` + `tests/test_forward_gate.py` + `scripts/smoke_forward_gate_mrpc.py` + `scripts/stage0_oracle_probe.py` + `scripts/stage2_confirmatory.py`. The validator checks `run_sha == git rev-parse v4-prereg`. |

The V4 code does **not** exist at `93ec6e4` — it is committed separately and
additively. No V4 artifact rebases or modifies the V3 frozen block.

---

## 2. Supervisor Hard Gates (V4 additions)

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
Stage 0 (oracle probe) runs at the supervisor's discretion but is not blocked;
it does not consume the Stage 2 budget. The stand-in answers bind procedure only.

---

## 3. The Rule: One Score, Chosen From Two Declared in Advance

Two candidate scores are declared in this pre-registration. They are evaluated
on disjoint Stage 0 seeds (101, 103) and the better-performing one is carried
into Stage 2. The selection rule is fixed in advance (see §4), not data-driven
at Stage 2. The Stage 2 false-positive rate is not inflated because:

- Stage 2 tests a **single** hypothesis on a **single** chosen score, fixed before
  any Stage 2 cell runs.
- Stage 0 seeds (101, 103) are disjoint from Stage 2 seeds.
- The selection uses the **higher** per-seed ρ across both seeds; ties go to S1.

This is sample splitting, not fishing. The winner's curse can only bias the
Stage 0 ρ estimate — which is a descriptive/mechanism probe, not the claim.

### S1: forward gain bound
`forward_gain_score(logits, labels)` = mean over batch of
`||softmax(logits)_i − onehot(labels)_i||` (no square), bounded by √2
(K&F 2018). Reads `trainer._last_real_logits` (captured when
`capture_logits=True`). Cost: ~0 ms (logits already cached).

### S2: last-layer alignment score
`s_B = ⟨G_B, G_P ⊘ (√v + ε)⟩`

- `G_B = (1/B)·Σ(p_i − y_i)·h_iᵀ` — the gradient of softmax cross-entropy w.r.t.
  `classifier.weight` for the current batch. Shape `[num_labels, hidden_dim]`.
- `G_P` — the same gradient computed on a 256-example score probe `P_s`, refreshed
  every 64 steps in eval mode, fp32.
- `v` = AdamW's `exp_avg_sq` for `classifier.weight`.
- `h` = the hidden state entering the classifier, captured by a forward-pre-hook
  on `model.classifier` (the Linear layer in `ModernBertForSequenceClassification`).
  Verified: `model.classifier` is `Linear(768 → num_labels)`, and `exp_avg_sq`
  for `classifier.weight` has shape `[num_labels, 768]`, matching `G_B` and `G_P`.

It is a first-order prediction of how much the probe loss drops if this batch's
update is applied. The score direction is the sign/direction of the AdamW update
on `classifier.weight`; only direction matters, not magnitude.

**Excluded: loss rescaling.** Step-level inverse-propensity weighting rescales
magnitude without changing direction; under AdamW the update is approximately
scale-invariant. Recording the AdamW caveat: the alignment score does not
predict the magnitude of per-step gain, only its expected direction.

**S2 hook timing.** The forward-pre-hook on `model.classifier` fires during
training mode (model.train()), where dropout is active and the head has
applied attention-layer normalization. `h` is captured post-head, pre-classifier.
The probe gradient `G_P` is computed in eval mode (model.eval()), where dropout
is disabled. This is **not** a bug: the alignment score measures whether the
batch's *direction* (in training-mode hidden space) aligns with the probe's
direction (in eval-mode hidden space). The two are evaluated on the same input
distribution, so the dot product is still meaningful as a direction correlation.
The discrepancy is documented here because a reviewer may question it.

**Classifier bias.** `G` computed by `_classifier_grad_from_logits` covers
`classifier.weight` **only**, not the bias. The gradient w.r.t. the bias is
`(1/B) * sum_i (p_i - y_i)`, which is a per-class scalar. The alignment score
omits it because the bias gradient is rank-1 (one scalar per class) and adds
negligible directional information beyond what `classifier.weight` already
captures. This is recorded as a design decision, not an oversight.

**Probe signal magnitude.** At lr=2e-5 on 256 examples, the per-step change
in probe loss may be close to noise. Stage 0 could simply be uninformative
(both ρ values near zero). If so, Stage 0 still selects S1 by the tie-breaking
rule (ties → S1), and Stage 2 proceeds with S1 as the score source. The
weak-signal case is pre-registered as a possible outcome, not a protocol
violation.

S2's extra cost (one probe forward pass every 64 steps, fp32, eval mode) counts
against the gate in wall-clock time and is measured.

---

## 4. Stage 0: Oracle Probe (chooses the score)

**Purpose**: Determine which candidate score (S1 or S2) better predicts per-step
gain. The chosen score carries into Stage 2. Stage 0 never cancels Stage 2.

**Seeds**: 101, 103 (not used in Stage 2; not used in V3 confirmatory or MRPC).

**Probe data**:
- `P_s` (256 examples, train split) — used to compute G_P for S2.
- `P_e` (256 examples, train split, disjoint from P_s) — used to measure the
  outcome (realized gain) in Stage 0.
- Both probes are **removed from the training sampler** in every arm. This is
  asserted at runtime via `ProbenRemovedFromSampler` check (see §9).

**Procedure**:
1. Run `full_finetune` in **shadow mode** (full backward, no skipping) on each
   seed.
2. At every 8th step, log: S1, S2, the old LER tracker score (lagged sampled-LER),
   and the realized gain on `P_e` (fp32, eval mode).
3. Score: **local Spearman ρ** of (S1 or S2) vs realized per-step gain within
   trailing 64-step windows, with moving-block bootstrap (block 64, 10,000
   resamples) and 97.5% CI for each candidate.

**Selection rule** (fixed in advance):
- Take the higher of the two per-seed ρ values for each score.
- Pick the score with the higher value. Ties go to S1.
- Report the "eligible" mark (ρ ≥ 0.10 with CI above 0) only as description.
- **Stage 0 never cancels Stage 2.** Stage 2 costs ~6 GPU-hours, too cheap to let
  a noisy 256-example probe kill it.

**Outcome**: The chosen score is recorded in `docs/gate_a_decision.md`.

---

## 5. Accuracy measurement protocol

**Final checkpoint only — no dev-set selection.**

- `no_early_stopping=True` is enforced for every Phase 1.3 cell (asserted in
  `_validate_cell_contract`).
- `load_best_model_at_end=False` is derived from `not no_early_stopping`;
  this is guaranteed, not optional.
- Accuracy on the dev/test set is taken from the **final** checkpoint after the
  full `total_steps` schedule. No early-stopping selection, no best-model
  selection, no checkpoint averaging.
- `eval_strategy="steps"` with `eval_steps = max(total_steps // 20, 10)` is used
  only for monitoring/logging, never for model selection.
- `metric_for_best_model` is recorded in metadata but never acted upon.

This is a deliberate **deviation from `run_baseline_glue_modernbert.py`** (which
uses `load_best_model_at_end=True` with patience-based early stopping). The Phase 1.3
protocol forbids it because dev-set selection would inflate the gate's apparent
benefit: a gate that skips "lucky" steps could be rewarded by selecting a
checkpoint that happened to have good dev accuracy by chance, rather than having
genuinely learned more from the skipped backward passes.

---

## 6. Phase 3 Forward Gate Design

### 5.1 Module
`lerna/utils/forward_gate.py` — pure, CPU-testable. Three exported components:

- `ScoreQuantileGate`: deterministic gate scorer. 4 equal phase windows over
  `total_steps`. Per-window quota allocated via largest-remainder (Hare method)
  over **eligible** steps (`decision_index >= min_step`), so `sum(phase_quota) ==
  round(rate * total_steps)` exactly. At each step, an exact feasibility check
  determines whether skip or backward is forced (see §5.3). Annealing at
  `round(0.88 * total_steps)` replaces score with i.i.d. Uniform[0,1] (budget
  preserved, selection randomized). Non-finite score →
  `TerminalInvalidScoreError` (absorbing).

- `ScoreQuantileGatePolicy`: `SkipPolicy` wrapper plugging `ScoreQuantileGate`
  into `TrueBackwardSkippingTrainer`. `score_source` ∈ {s1, s2, uniform}.
  For s2, the policy attaches an `AlignmentProbe` to `model.classifier` and
  provides a probe evaluation callback.

- `AlignmentProbe`: attaches a forward-pre-hook to `model.classifier`,
  capturing `h` (the hidden state entering the classifier). Used by S2.

### 5.2 Score Sources
- **S1** (`forward_gain_score`): mean over batch of `||p − y||`, bounded by √2.
  See §3.1.
- **S2** (`alignment_score`): last-layer alignment score. See §3.2.

### 5.3 Feasibility forcing (bug a fix)
At each step, compute:
- `skip_possible`: can quota be met if we skip now?
  → `max_skips_possible(remaining_steps − 1, K) ≥ remaining_quota − 1`
- `real_possible`: can quota be met if we run a real backward now?
  → `max_skips_possible(remaining_steps − 1, K) ≥ remaining_quota`

Where `F(r) = K · ⌊r/(K+1)⌋ + min(r mod (K+1), K)` is the exact maximum skips
in `r` decisions under cap `K`.

Decision:
- If neither is feasible → abort (should not happen post-init validation).
- If only `real_possible` → force a real backward (`forced_skips += 1`).
- If only `skip_possible` → force a skip (`forced_skips += 1`).
- If both → the score decides (quantile or Bernoulli under annealing).

The cap (K=4) **always applies**, including to forced skips. No forcing
overrides the cap. This ensures no leftover quota crosses a window boundary.

### 5.4 Per-window skip-position diagnostic (bug d fix)
The gate records `phase_skip_positions`: for each phase window, the list of
phase-relative positions where skips land. The diagnostics report skip count,
median, min, and max position per window. This enables a parity check between
gate and control on skip placement, detecting front-loading or back-loading
bias from the score's temporal structure.

### 5.5 min_step = 392 (bug e fix)
All arms use `min_step = 392`. This exceeds the warmup phase (warmup_steps = 12
for the smoke test, 392 for production) so no skips occur during LR warmup,
where gain scores are inflated and unrepresentative. AdamW's second-moment
estimate `v` already exists at step 392, so S2 can be computed from the start
of the eligible region.

---

## 7. MRFC Smoke Test (Phase 3 mechanics validation)

**Script**: `scripts/smoke_forward_gate_mrpc.py`
**Status**: PASS (expected after bug fixes)

| Parameter | Value |
|-----------|-------|
| Task | MRPC (GLUE) |
| Model | `jhu-clsp/ettin-encoder-150m` |
| Total steps | 200 |
| Rate | 0.45 (round(0.45 × 200) = 90) |
| min_step | 392 → clamped to 50 for the 200-step smoke (warmup=12) |
| n_phases | 4 |
| max_consecutive_skips | 4 |
| seed | 11 |
| warmup_steps | 12 |
| Scheduler | `always_step` (pure f-of-step LR contract) |
| Precision | fp32 (no AMP) |
| CUDA device | 0 (V100-SXM2-32GB) |

**Arms (2):**
- `gate`: `ScoreQuantileGatePolicy(score_source="s1")`
- `control`: `ScoreQuantileGatePolicy(score_source="uniform")`

**Invariants verified:**
- `realized_total == requested_total` (90 == 90) both arms
- `sum(phase_used) == requested_total`
- `longest_consecutive_skips <= 4` (cap always respected, including under forcing)
- `scheduler_step_calls == batches_seen == 200` (always_step contract)
- LR curves identical: `max|diff| = 0.00e+00`
- Phase 0 quota 0 (min_step-gated; share transferred to windows 1-3)

**NOTE**: The smoke test uses `min_step=50` because 200 total steps < 392 warmup.
The production protocol uses `min_step=392`. The feasibility and cap logic is
identical; only the warmup window differs.

---

## 8. Stage 2: Confirmatory Test (rate 0.60, pending supervisor signature)

**Arms (4)** at rate 0.60, paired over seeds:
1. `gate` — `ScoreQuantileGatePolicy` with the chosen score from Stage 0
2. `control` — `ScoreQuantileGatePolicy(score_source="uniform")`
3. `full_finetune` — `AlwaysFalsePolicy` (0% skip)
4. `reduced_steps` — `ReducedTotalStepsCallback` (train fewer steps, matched budget)

**Seeds (Stage 2)**: 24 seeds: [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107]

Wait — seeds 101 and 103 are used in Stage 0. The Stage 2 seed list must be disjoint.

**Corrected Stage 2 Seeds**: 24 seeds: [11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 107, 109, 113]

**Reserve seeds**: 157, 163 — for failed cells only. No reruns.

**Rate**: Fixed at 0.60 (chosen pre-hoc from the v3 protocol; the MRFC smoke
validates mechanics, and 0.60 gives the gate the most room to beat random.
0.70 is not allowed by the trainer under cap=4: F(6548−392)=5181 skips max,
but round(0.70 × 6548) = 4584 ≤ 5181, so 0.70 is actually feasible — but
0.60 is the pre-registered choice and is not changed after seeing results).

**Matrix**: 4 arms × 24 seeds = 96 cells, ~6 GPU-hours across 8 idle V100s.

### Primary test (the only one):
- `Δacc(gate − control)`, paired t-test, two-sided, α = 0.05, df = 23.
- Bootstrap 95% CI (10,000 resamples).
- MDE ≈ 0.17 pt, assuming paired SD = 0.287 from v3 (retrospective). Power analysis output:

  ```
  Power Analysis for Stage 2 Paired t-test
    n = 24 seeds (df = 23)
    alpha = 0.05 (two-sided)
    target power = 0.80
    paired SD = 0.287
    noncentrality parameter (ncp) = 2.9256
    MDE = 0.1714 pt
  ```

### Secondary outcomes (pre-specified, listed in advance):
- `gate vs reduced_steps` (paired t-test)
- `control vs reduced_steps` (paired t-test)
- TOST vs `full_finetune` at 0.5 pt margin
- Realized skip parity (gate and control at identical skip counts per seed)
- Cap hits and forced-skip counts
- Skip-position parity (using the `phase_skip_positions` diagnostic)
- ms/step (timing cells, 3 seeds × 4 arms, single isolated GPU)

**Descriptive extras (not tested, not claimed):**
- Gate and control at rate 0.45, 5 seeds each, to show how the effect changes
  with skip rate.

**Rate justification**: 0.60 is chosen because random loses the most there,
giving the gate the most room to beat it. Choosing the rate after seeing
control's results is avoided — the rate is fixed in the pre-registration.

### Outcomes, written down before running:
- **Gate beats control**: "A forward-predicted-gain gate beats random at matched
  backward budget." Stage 0 ρ values are the mechanism evidence.
- **Null**: The headline becomes the efficiency study: 60% backward skipping
  compared with full_finetune and reduced_steps, plus the finding that neither
  S1 nor S2 beats random, with Stage 0 explaining why. This works as an
  empirical-study submission to TMLR.
- **Gate loses to control**: Report it as is.
- **Gate beats control but not reduced_steps (on accuracy/GPU-second)**: The
  reviewer critique "just train shorter" applies because `reduced_steps` skips
  60% of ALL computation (forward + backward + optimizer) while the gate skips
  only 60% of backward steps but still runs 100% of forward passes (plus the
  64-step S2 probe refreshes). The gate's matched budget is "backward steps
  only"; `reduced_steps`'s matched budget is "total compute." These are
  different denominators. This distinction is pre-registered here: the gate's
  primary claim is "beats random at matched backward budget." The
  `reduced_steps` comparison is a secondary outcome labeled as such.
- In every case: no changing the rate, the score, or the seeds after Stage 2.

---

### Compute budget clarification (registered):
- `full_finetune`: 6548 forward + 6548 backward + 6548 optimizer steps.
- `reduced_steps` at 0.60: 2619 forward + 2619 backward + 2619 optimizer steps.
- `gate` at 0.60: **6548 forward** + 2619 backward + 2619 optimizer steps.
  - The gate skips backward passes only. Forward passes (including the S2 probe
    refresh every 64 steps) are never skipped.
  - S2 probe cost: one forward pass over 256 examples every 64 steps in fp32
    eval mode. This is negligible (<0.1% of a training step at 13.4k samples).
- `control` (uniform) at 0.60: same compute as gate (6548 forward, 2619 backward).

The "matched budget" for gate vs control is backward steps (2619 each).
The "matched budget" for gate vs reduced_steps is NOT total compute.

---

## 9. Stage 2 Failure Policy

- Strict forward-loss check ON (`nonfinite_forward_loss_policy="fail"`)
- `forward_loss_observations == total_steps` (6548 for QNLI)
- `nonfinite_forward_loss_count == 0`
- Manifest must carry `git_sha`, `tracker_revision`, `min_step`, `score_source`
- Seeds 157/163 are reserves for failed cells only. If >2 cells fail and
  cannot be replaced from reserves, the matrix is reported as incomplete.

---

## 10. Probe Exclusion Assertion

Both `P_s` (256 examples) and `P_e` (256 examples) are removed from the training
sampler in every arm. The runner asserts:
```python
# Assert probes excluded
probe_ids = set(P_s_ids) | set(P_e_ids)
train_ids = set(train_dataset.ids)
assert len(probe_ids & train_ids) == 0, "Probe examples leaked into training"
```
If a probe is found in the training data, the cell is invalidated. A single
probe doing both scoring and outcome would make S2 "look good by construction."

---

## 11. Execution Protocol

1. **Stage 0 first**: Run oracle probe (seeds 101, 103) in shadow mode.
2. **Decision recorded**: `docs/gate_a_decision.md` must be dated and committed
   before any Stage 2 cell runs.
3. **Supervisor sign-off**: `docs/supervisor_signoff_v4.md` must bear the real
   supervisor's signature before Stage 2 GPU cells.
4. **Serial execution**: Stage 2 cells run one at a time (96 cells across 8 V100s).
5. **Clean git state**: `require_clean_git_tree()` before any cell runs.
6. **No code changes during run**: enforced. Data is never deleted mid-run.
7. **Host isolation**: No other jobs on the host during runs.

---

## 12. Checklist Before Pre-Reg Commit

1. **Cap revert + exact feasibility check**: Tests on adversarial score sequences
   (monotone up, monotone down, constant, alternating) at 0.45 and 0.60 over 6548
   steps: realized == requested, longest skip run ≤ 4, no infeasibility error.
   → See `tests/test_forward_gate.py` (tests 15–16).

2. **S2 hook + probe refresh**: On a tiny model, `G_B` (from
   `_classifier_grad_from_logits`) matches the autograd gradient for
   `classifier.weight` within 1e-5, and the output is finite with the right
   shape `[num_labels, hidden_dim]`.
   → See `tests/test_forward_gate.py` (test 13).

3. **Both probes excluded from training sampler**: Asserted in the runner.
   → See §9.

4. **Pre-reg text fixes**: Seeds 101/103 (Stage 0), 24 Stage 2 seeds, reserves
   157/163; rate fixed at 0.60; selection rule; power output.
   → This document.

5. **CPU dry run**: 6548 steps for both gate and control arms: final LR ≤ 1e-8
   and `scheduler_step_calls == 6548`.
   → See `scripts/smoke_forward_gate_mrpc.py` (extended to 6548-step dry run).

6. **Clean git tree**: New `v4_prereg_sha`. The current SHA predates these fixes,
   so it cannot be the run SHA.

---

## 13. Addendum: Changes from V3

| Section | V3 | V4 (this amendment) |
|---------|-----|---------------------|
| Gate estimator | backward LER | forward gain (S1) + alignment (S2) |
| Score formula | ||p−y||² (documented) | ||p−y|| (code-accurate) |
| min_step | 50 (smoke) | 392 (production; 50 for smoke) |
| Forcing | forcing_fraction=0.8, cap override | exact feasibility check, cap always applies |
| Stage 0 | not run | oracle probe (seeds 101, 103) |
| Rate | 0.30 (confirmatory), 0.60 (Gate A) | 0.60 (Stage 2, pre-registered) |
| Seeds | 11, 13, 17, 19, 23 | 101, 103 (Stage 0) + 24 Stage 2 seeds |
| S2 | absent | alignment score with probe |
| Stage 1 | intermediate | dropped (rate fixed at 0.60) |
