# LERNA Final Close-Out Roadmap (pre-registration v4, last one)

**Author:** Muhammad Sheheryar
**Supervisor:** Prof. Chen Qing Cai
**Date:** 2026-10-02
**Status:** ready for Day 10 supervisor meeting
**Commitment:** v4 is the final pre-registration. No v5.

---

## 0. Read this first — three places this document departs from the draft brief

The draft brief for this roadmap asserted three things that the frozen run data contradicts. Because this
document is a pre-registration and a thesis evidence base, it records the verified numbers instead. The
departures are listed here so the supervisor can see them explicitly rather than discover them at defense.

| # | Brief asserted | Frozen data shows | Where it lands |
|---|---|---|---|
| 1 | "every LER arm slower than full fine-tuning" | **False.** `ler_guided_stratified` 14.38 min vs `full_finetune` 15.46 min. LER skipping *is* faster than full fine-tuning. It is only slower than `exact_random` (12.95 min). | §1.3 — a real but much weaker efficiency claim |
| 2 | TOST passes, so non-inferiority is established | TOST passes, **but** `lgs` vs `full_finetune` is significantly *worse*: Δ = −0.212 pt, t(4) = −3.072, **p = 0.037**, CI95 [−0.404, −0.020] pt. | §1.2 — both tests must be reported, not just the passing one |
| 3 | run provenance matches frozen SHA `d4383a8` | All 20 cells ran at `93ec6e4`; `gate3_summary.json` carries `"matrix_invalidated": true`. The `d4383a8`→`93ec6e4` diff is a **disk-threshold constant only** (500→100 GB), non-scientific, but the SHA in pre-reg v3 is wrong. | §3.6 — disclosed in v4 rather than silently corrected |

A fourth item is a build-cost correction, not a data correction: `lgs_shuffled` **requires new code**, not
configuration only. See §3.4.

---

## 1. Data verdict (no hedging)

### 1.1 What is measured and clean

QNLI, `ettin-encoder-150m`, fp16 + eager attention, 2 epochs, batch 32, lr 2e-5, warmup 0.06,
max_grad_norm 1.0, skip rate 0.30, seeds `[11, 13, 17, 19, 23]`, 6548 steps/cell.

**20/20 cells PASS M2c strict.** Zero NaN events. Zero non-finite forward losses. 6548 total steps per cell.
All 20 cells at identical code SHA `93ec6e4`. No failed seeds. This is a clean matrix and the only
confirmatory-grade block in the project.

### 1.2 The accuracy results

Per-arm accuracy at 0.30 skip (mean of 5 seeds, accuracy in percentage points):

| Arm | Mean acc | SD | vs `full_finetune` (paired) | p |
|---|---|---|---|---|
| `full_finetune` | **93.267** | 0.268 | — | — |
| `exact_random` | 93.114 | 0.112 | −0.154 pt | 0.202 |
| `phase_strat_guarded` | 92.986 | 0.249 | −0.281 pt | — |
| `ler_guided_stratified` | 93.055 | 0.258 | **−0.212 pt** | **0.037** |

Paired tests against `ler_guided_stratified`, n = 5, df = 4:

| Comparison | Δ (pt) | t(4) | p | 95% CI (pt) | TOST @ 0.5 pt |
|---|---|---|---|---|---|
| lgs − `full_finetune` | −0.2123 | −3.072 | **0.0372** | [−0.404, −0.020] | PASS |
| lgs − `exact_random` | −0.0586 | −0.596 | 0.583 | [−0.332, +0.214] | PASS |
| lgs − `phase_strat_guarded` | +0.0696 | — | — | — | PASS |

**These two results must be reported together and neither may be omitted.**

1. **Non-inferiority holds.** At a 0.5 pt margin every LER arm is non-inferior to every other arm. The
   30% budget cut costs less than half a point.
2. **LER guidance is not better than random.** lgs − exact_random = −0.059 pt, p = 0.583. The LER signal
   does not beat a random permutation of the same budget.
3. **And lgs is detectably worse than full fine-tuning.** p = 0.037 with the CI excluding zero. This is the
   single most important number in the project and it is *not* in pre-reg v3. Reporting only the TOST pass
   would be selective reporting; a committee member who runs the paired t-test will find this.

The prior "better result" definition — lgs ≥ +0.5 pt over full fine-tuning, TOST p < 0.05, across three
tasks — is **retired in writing**. It was incoherent (TOST tests equivalence, not superiority) and it
contradicts the data. The thesis is **Path B**. Path A survives only as a pre-registered gate on QNLI (§3.3).

### 1.3 Wall-clock: the efficiency claim is real but narrow

Mean serial training runtime per cell, one V100, no concurrency:

| Arm | Runtime | vs full fine-tuning | vs exact_random |
|---|---|---|---|
| `full_finetune` | 15.46 min | — | +19.4% |
| `phase_strat_guarded` | 15.27 min | −1.2% | +17.9% |
| `ler_guided_stratified` | 14.38 min | **−7.0%** | +11.0% |
| `exact_random` | 12.95 min | **−16.2%** | — |

Per-step: ~90 ms random-skip vs ~108 ms LER-skip vs ~140 ms full.

**The claim that survives:** backward skipping at 30% is ~7% faster in wall-clock than full fine-tuning when
the budget is spent by LER guidance, and ~16% faster when spent at random. The tracker inference overhead
consumes roughly half the available saving — LER recovers only 44% of what random skipping achieves.

**The claim that does not survive:** "LER-guided skipping is faster" is false against `exact_random`. Any
"X% faster than fine-tuning" claim must name its baseline, and the honest version is the 7% above.

### 1.4 The structural objection that `er` cannot answer

`exact_random` at matched budget is the control that matters. It already establishes that *skipping 30% of
backward steps costs ~0.15 pt versus not skipping*. `lgs` does not beat it. The defensible reading is that
the budget reduction is nearly free but that **LER does not identify which steps are worth keeping**. That
is a negative result about the signal, not about the mechanism, and §3.3 is the test that settles it.

`er` vs `reduced_steps` must not be used as the headline control. Both spend an identical budget; comparing
them only re-measures the cost of skipping.

### 1.5 Exploratory blocks — reported separately, never merged

- QNLI v3 @ 0.30, seeds 11/13/17/19/23 — **confirmatory**, the block analysed above.
- MRPC blocks (`gate2_mrpc_v3`, 8-cell smoke at seeds 42/43) — **exploratory**. Smoke-scale only; not a
  120-cell matrix. Reported with the label and never pooled with QNLI.
- `qnli_exploratory_seed42`, `ablation_pilot*` — **exploratory**, diagnostic only.

Seeds 42–8192 remain exploratory by pre-registration and are excluded from all confirmatory statistics.

---

## 2. Final headline

**Outcome pending Stage 0.** The v3 QNLI 0.30 data are retrospective only — two confounds bind them
(cap 4 vs no cap on the skip arms; LR schedule artifact where skip arms stepped ~4584/6548 vs
full fine-tuning every step — the Kaddour failure mode). They are not Path B evidence until Stage 0
is run. The retrospective, with both confounds stated, is written in `docs/chapter_why_ler_failed.md`
(status: draft, outcome undecided).

> **Branch A — if the Stage 0 oracle gate PASSES.** The forward gain proxy predicts per-step
> contribution above a within-seed permutation + step-index/loss-stratified partial correlation:
> local ρ > 0 with a bootstrap CI excluding 0 on both seed 11 and seed 13.
>
> **Branch B — if the oracle gate FAILS.** Reported as "undetectable at n = 5" (not "no signal
> exists"), consistent with Kaddour. No further attempt is made to make LER win.

Both branches are defensible theses; neither claims superiority. Stage 0 is the only decision
point; it has not been run.

---

## 3. Pre-registration v4 (pasteable)

### 3.1 Frozen claim

The thesis claim is **non-inferiority under a matched backward-step budget**, plus a pre-registered test of
whether the LER signal is informative at all. No superiority claim is made in any arm, task, or rate.

### 3.2 Arms

Five arms, all at matched budget except `full_finetune` (0% skip, rate-independent):

`full_finetune` · `exact_random` · `ler_guided_stratified` (`lgs`) · `lgs_shuffled` · `phase_strat_guarded`

`lgs_shuffled` is the within-seed permutation control: identical LER machinery, identical realized skip
count, per-seed deterministic permutation of the LER ranking. It isolates *"is the ranking informative"*
from *"does the tracker machinery cost anything"*.

### 3.3 Gate A — the oracle test (decides Branch A vs Branch B)

`lgs_shuffled` must beat all three, on QNLI at skip 0.60, paired over 5 seeds:

1. **Partial Spearman** ρ(LER score, per-step gain | step index, current loss) ≥ 0.20, bootstrap 95% CI
   excluding 0.
2. **Paired t-test** `lgs` > `lgs_shuffled`, two-sided, p < 0.05.
3. **Split-half reliability** r ≥ 0.30 across probe halves within seed.

**Failure is reported as "undetectable", not "no signal exists."** With n = 5 seeds the test has limited
power; a failure bounds the effect from above, it does not prove absence. This wording is mandatory in the
thesis and the defense.

### 3.4 Build cost — correction to the brief

`lgs_shuffled` **requires a small code change**, not configuration only. `VALID_ONLINE_LER_MODES` in
`lerna/trainers/true_skip_trainer.py:30-33` is exactly `{off, legacy_dense, sampled_lagged}`. There is no
permutation mode; the string `shuffled` does not appear anywhere in the repository.

**Required work:** one new constant `ONLINE_LER_MODE_SHUFFLED`, one branch in the existing mode dispatch, a
deterministic per-seed permutation seeded from the existing `policy_seed`, and a test asserting the realized
skip count is identical to `lgs` at the same seed and rate. No change to the skip-policy logic, the tracker,
or the budget accounting. Estimate: **1 day including tests.** It is additive and does not touch the frozen
QNLI 0.30 block.

### 3.5 Matrix

| Task | Rate | Seeds | Cells |
|---|---|---|---|
| QNLI | 0.30 | 11, 13, 17, 19, 23 (frozen) + 5 new | 4 skip arms × 10 × 1 + full × 10 = 50 |
| QNLI | 0.60 | 5 new | 4 skip arms × 5 + full × 5 = 25 |
| MRPC | 0.30 | 30 new | 4 skip arms × 30 + full × 30 = 150 |
| Gate A oracle probe | 0.60 | 5 | 5 |
| Serial timing cells | 0.30 | 5 | 5 |

**Total ≈ 235 accuracy cells.** MRPC at 0.60 is **dropped**: the MRPC dev set is 408 examples, which cannot
resolve a 0.60-skip effect with a useful confidence interval, and pre-registering it would be
pre-registering a known failure. 0.60 is retained on QNLI only, because Gate A requires it.

### 3.6 Provenance disclosure (mandatory)

The QNLI 0.30 confirmatory block ran at code SHA **`93ec6e4`**, not the `d4383a8` recorded in pre-reg v3.
`gate3_summary.json` carries `"matrix_invalidated": true` for this reason.

`git diff d4383a8 93ec6e4` touches three files and the only functional change is one constant: the preflight
disk threshold, 500 GB → 100 GB. The remainder is a test file and one documentation line. **No model,
data, hyperparameter, seed, or selection logic differs.** The block is therefore scientifically valid, but
v4 records the true SHA and the reason rather than leaving the incorrect one in place. v3 is not edited
retroactively; v4 supersedes it.

**Additive change log (post-freeze, value-preserving, does not alter frozen cell SHAs):**

| Date | File | Blob (worktree) | Nature | Effect on frozen block |
|---|---|---|---|---|
| 2026-10-04 | `lerna/utils/lagged_ler.py` | `99952d7` (HEAD `0dae3bc` = `480cf78`) | Wire `_compute_entropy_v3` into the v3 update path (`update`), replacing the per-call `_compute_entropy` only on `tracker_revision == "v3"`. | None. v3 entropy is bit-identical to the v2 formula (verified `< 1e-9`); this only removes redundant CPU syncs. The frozen QNLI 0.30 cells' manifests retain `git_sha = 93ec6e4` (immutable commit); this edit is uncommitted atop `0dae3bc` and is additive logic on a branch the frozen cells do not regress.

### 3.7 Explicitly excluded: `lgs_ipw`

**`lgs_ipw` is excluded by design and will not be run.** Step-level inverse-propensity reweighting multiplies
an executed step's gradient by 1/p_t, which scales magnitude without changing direction. Under AdamW the
update is approximately scale-invariant to such a factor, so step-level IPW is close to a no-op. The
unbiased-estimator argument that motivates IPW does not transfer from within-batch sample reweighting
(K-ABENA-style) to step-level skipping. This exclusion is recorded now so that no reviewer or committee
member raises it as an omission.

### 3.8 Statistical protocol

Paired two-sided t-test, α = 0.05, seeds as paired units. TOST non-inferiority at a 0.5 pt margin, one-sided
95% lower bound on Δ. M2c strict on every cell: `forwarded_loss_observations == total_steps`,
`nonfinite_forward_loss_count == 0`, policy `fail`. Arms with duplicate accuracies across seeds are flagged
by the runner; seed 11/13 at 0.9323 in the frozen block was verified as coincidence (train loss 0.2337 vs
0.2317, eval loss 0.1864 vs 0.1882, 5093 vs 5463 correct) and is documented, not re-run.

---

## 4. Eight-week plan

**Day 10 supervisor meeting must fix the defense date in writing.** If the supervisor does not grant 8 weeks,
Weeks 4–8 collapse to the 4-week writing-only fallback. **No experiments change either way** — the
experiment set is frozen by §3 and does not respond to schedule pressure.

**Blocker to clear before Week 1:** `/new_raid` is 100% full (63 GB free of 15 TB). The runner's preflight
requires 500 GB free and will abort. Either relocate the output root to `/raid` (1.1 TB free) or reclaim
the 152 GB held by `gate3_qnli_v3` (101 GB) and `gate3_qnli_confirmatory` (51 GB). **This is a Day 1 task.**

| Week | Experiments | Writing |
|---|---|---|
| 1 | Clear disk blocker. Build `lgs_shuffled` + test (§3.4). Freeze runner SHA. | Methods chapter, full draft |
| 2 | QNLI 0.60 confirmatory + Gate A oracle cells. | Methods chapter, full draft |
| 3 | MRPC 30 seeds, serial. **Experiments hard-stop Friday of Week 3.** | Introduction, Related Work |
| 4 | — | Results written from frozen tables only |
| 5 | — | Results, Limitations |
| 6 | — | Discussion, Conclusion |
| 7 | — | Full draft assembled, supervisor review |
| 8 | — | Revisions, formatting, defense rehearsal |

Compute load is ~40 GPU-hours total against 8 idle V100s. The constraint is writing, not compute.

**4-week fallback:** Weeks 4–8 compress to Results, Limitations, Related Work (shortened), Discussion,
Conclusion. No new experiments. MRPC 30 seeds and Gate A either complete in Week 3 or are reported as
not-run with the reason stated.

---

## 5. Supervisor one-pager

**The ask.** Approve Path B as the thesis framing: a budget-controlled non-inferiority study with a
pre-registered test of whether the LER signal is informative. Approve the 8-week schedule and fix the
defense date in writing today.

**What the data says, in three sentences.** At a matched 30% backward-step budget on QNLI, skipping is
non-inferior to full fine-tuning within 0.5 pt, but it is also detectably *worse* than full fine-tuning
(−0.21 pt, p = 0.037), and LER guidance does not beat random skipping at the same budget (−0.06 pt,
p = 0.583). Skipping saves ~7% wall-clock under LER and ~16% under random; the tracker overhead eats about
half the saving.

**What is not claimed.** No superiority anywhere. No "LER is faster" claim. The old target of +0.5 pt over
full fine-tuning is retired in writing because the data rules it out and the test used for it was the
wrong test.

**The one open question, asked directly.** *Does the HIT Shenzhen master's rubric require a proposed method
that outperforms baselines, or does it accept a rigorous study with a negative or null result?* This is the
single question whose answer determines whether Path B is acceptable. It is asked at this meeting, not
discovered at defense.

**Escalation.** If the answer is that a win is required, escalate to Prof. Zhang (graduate committee) before
committing to a framing. **Author's note: this contact has not been verified — confirm the name and role
before relying on it.**

**Why this meeting is Day 10.** The analysis was already frozen and the matrices already run. The delay is
not analysis; it is that the honest framing needed a supervisor decision before it could be written. Every
week this conversation is deferred is a week of writing that cannot be recovered.

---

## 6. Defense script

**Q1. Your method doesn't beat full fine-tuning. Why should I believe it works?**
It is not claimed to. The claim is non-inferiority under a matched budget: at 30% fewer backward steps the
accuracy loss is 0.21 pt, inside a pre-registered 0.5 pt margin. Skipping is nearly free. Whether the LER
ranking is *informative* is a separate question I test separately and report whatever it returns.

**Q2. Isn't the gain just from training less?**
That is exactly what the `exact_random` arm controls, and I report it: skipping 30% of backward steps costs
0.15 pt versus not skipping. That cost is stated, not hidden. The interesting result is that LER guidance
does not improve on it (−0.06 pt, p = 0.583).

**Q3. Your TOST passes but your t-test is significant at p = 0.037. Which is it?**
Both, and they answer different questions. The t-test asks whether the arms differ at all; they do, and LER
skipping is slightly worse. The TOST asks whether the difference is small enough to be acceptable against a
0.5 pt margin; it is. Non-inferior within a stated margin is compatible with a real but small degradation,
and I report both rather than only the one that supports the thesis.

**Q4. What is `lgs_shuffled` and why does it need new code?**
It is the within-seed permutation control: the same tracker, the same realized skip count, with the LER
ranking shuffled. It separates "is the ranking informative" from "does the tracker cost anything." It needs
a fourth `online_ler_mode` constant and one dispatch branch — about a day including tests. The existing mode
set is `off`, `legacy_dense`, `sampled_lagged`; there was no permutation mode to configure.

**Q5. Why did you exclude inverse-propensity weighting?**
Because at step level it rescales the gradient without changing its direction, and AdamW is approximately
scale-invariant to that factor, so it would be close to a no-op. The unbiasedness argument for IPW applies
to within-batch sample reweighting, not to step-level skipping. I record the exclusion so it is not raised as
an oversight.

**Q6. Why is 0.60 run on QNLI only?**
Because Gate A needs a high-skip regime to have enough skipped steps for a permutation control to mean
anything, and QNLI is the only task with the sample size to resolve it. MRPC's dev set is 408 examples;
pre-registering a 0.60 MRPC confirmatory test would be pre-registering a test that cannot resolve its own
effect. MRPC 0.30 at 30 seeds gives the precision MRPC can actually support.

---

## 7. Publishability verdict

Blunt, with a stated probability and the one objection that kills each.

| Venue | Probability | The objection that kills it |
|---|---|---|
| **Negative-results workshop** (ICBINB-style, ICLR/NeurIPS 2027) | **65%** | "Random ≈ shorter training is already known; where is the new insight?" The paper must carry the answer: the budget-controlled framework plus the oracle test showing *why* the signal fails, not just that it did. |
| **Efficient-NLP / SustaiNLP-type workshop** (ACL/EMNLP 2027) | **55%** | "Two tasks and one 150M encoder does not generalize." The paper must self-label as a controlled case study, never as a method contribution. |
| **TMLR → MLRC reproducibility track** | **20%** (stretch) | "This is not a reproduction of a published paper." It can only be framed as a reproducibility-of-a-claim study, which is a genuine stretch. Marked as such. |
| **Main-track method paper** (ACL/EMNLP/NeurIPS) | **5%** | Everything above, compounded. **Not publishable, and written down here so nobody argues for it.** |

**Thesis:** defensible under a rigor rubric. The named risk is not statistical — it is a supervisor or
committee who requires a win. No plan available to me produces a win, and the correct response to that is
to change the framing or the venue, not to re-run until a seed passes.

**What would move this up one tier:** a third task, or one decoder-only model, to show the null is not an
artifact of a single encoder-only setup. Neither is done before the thesis. Stated so the limitation is
visible rather than discovered.

---

## Appendix — `gate_a_decision.md` template

Fill on completion of the Gate A run. **The thesis headline branch is not selectable until this file is
dated.**

```markdown
# Gate A Decision — LER informativeness oracle test

Date:            YYYY-MM-DD
Runner SHA:      <commit>
QNLI skip rate:  0.60
Seeds:           11, 13, 17, 19, 23
Cells:           lgs, lgs_shuffled, paired within seed

## Test 1 — partial Spearman rho(LER, gain | step index, current loss)
  estimate: ____   bootstrap 95% CI: [____, ____]   threshold: >= 0.20, CI excludes 0
  result:  PASS / FAIL

## Test 2 — paired t-test, lgs > lgs_shuffled
  delta: ____ pt   t(4): ____   p: ____              threshold: p < 0.05
  result:  PASS / FAIL

## Test 3 — split-half reliability
  r: ____                                              threshold: >= 0.30
  result:  PASS / FAIL

## Realized skip count identical to lgs at every seed
  verified: YES / NO

## DECISION
  Branch A (signal detected)  /  Branch B (undetectable)

Branch B is reported as "undetectable at n=5", NOT as "no signal exists".
No further attempt is made to make LER win. See §2 of the roadmap.
```
