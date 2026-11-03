# Why the old LER failed — retrospective (Path B framing)

**Status:** draft, outcome undecided. This is a retrospective reading of the frozen v3 confirmatory
block (QNLI `@0.30`, code SHA `93ec6e4`, 20/20 cells PASS M2c); it is **not** a locked thesis verdict.
The forward gate's informativeness is to be resolved by Stage 0 (oracle probe), the outcome of which
is not yet known (see open items). RHO-loss / UDS are out of scope for v4 and are
not introduced here. The numbers below are the frozen v3 outcomes, reinterpreted with their
confounds stated rather than presented as v4 evidence.

---

## 1. The frozen v3 block (context)

QNLI, `ettin-encoder-150m`, fp16 + eager attention, 2 epochs, batch 32, lr 2e-5, warmup 0.06,
max_grad_norm 1.0, **30% backward-step budget**, seeds 11, 13, 17, 19, 23, 6548 steps/cell, all at
code SHA `93ec6e4`. 20/20 cells PASS M2c (zero NaN, zero non-finite forward losses). Mean accuracy
(pp, paired over 5 seeds):

| Arm | Mean acc | SD |
|---|---|---|
| `full_finetune` | 93.267 | 0.268 |
| `exact_random` | 93.114 | 0.112 |
| `phase_strat_guarded` | 92.986 | 0.249 |
| `ler_guided_stratified` (lgs) | 93.055 | 0.258 |

Two paired comparisons, stated exactly as they were run:

| Comparison | Δ (pt) | t(4) | p | 95% CI (pt) | TOST @ 0.5 pt |
|---|---|---|---|---|---|
| lgs − `full_finetune` | −0.212 | −3.072 | **0.037** | [−0.404, −0.020] | PASS |
| lgs − `exact_random` | −0.059 | −0.596 | 0.583 | [−0.332, +0.214] | PASS |

Both must be reported; selecting only the passing TOST would be selective reporting.

## 2. Two confounds that prevent reading this as v4 evidence

**Confound 1 — cap 4 vs. no cap on the skip arms.** The v3 `ler_guided_stratified` policy ran with
`max_consecutive_skips=4` (cap 4), whereas `exact_random` had no such binding cap. At a 0.30 budget
over a 6548-step horizon the cap is rarely binding, but it changes the *distribution* of skips
across consecutive steps, so "lgs − random" is a test of "capped stratified guidance vs uncapped
random," not "signal vs noise." This is why the −0.059 pt / p = 0.583 figure is a retrospective
null, not a v4 result: the two arms were not configured comparably for a signal-vs-noise test.

**Confound 2 — learning-rate schedule artifact (the Kaddour failure mode).** The skip arms' learning
rate **never fully decayed**: the scheduler stepped ~4584 of 6548 times (≈70%), while `full_finetune`'s
scheduler stepped every step. The −0.212 pt / p = 0.037 gap between lgs and full fine-tuning is
therefore consistent with a **schedule artifact** — the skip arms trained on a flatter LR tail while
full fine-tuning ran the full decay — rather than evidence about the LER signal itself. This is
precisely the failure mode Kaddour et al. (NeurIPS 2023) warn about (batch-selection gain vanishing
when the baseline LR has decayed differently). It does **not** establish that LER guidance is worse;
it establishes that the v3 comparison was not schedule-controlled.

Neither confound is correctable without re-running, and v3 is frozen. They are stated so the v4
protocol (§3.6 of the close-out roadmap) fixes both before the data are used as evidence.

## 3. What survives from v3 (retrospectively)

1. **Mechanical viability.** Skipping 30% of backward steps is non-inferior to full fine-tuning
   within a 0.5 pt TOST margin in every pairwise comparison. The *budget* is nearly free.
2. **Efficiency, weak.** LER-guided skipping is ~7% faster than full fine-tuning (14.38 min vs 15.46
   min) but ~16% slower than random skipping at the same budget; the tracker overhead consumes
   roughly half the available saving (LER recovers ~44% of random's gain). The claim that survives
   v4 is "backward skipping at 30% is non-inferior within 0.5 pt and saves ~7% under LER," scoped to
   QNLI, with the schedule and cap confounds noted.
3. **The signal question is unanswered.** Because of the two confounds, "is the forward gain proxy
   informative" cannot be answered from v3. It is answered (or not) by Stage 0.

## 4. Why the signal likely fails anyway (Kaddour) and how v4 tests it differently

Under AdamW at a fixed learning rate, per-step loss reduction is dominated by the second-moment
normalization and the LR schedule, not by the identity of the skipped step. The forward gain proxy
(`||softmax(logits) − onehot||`, the Katharopoulos & Fleuret 2018 bound) is a near-static read of
parameter-velocity direction, so permuting the ranking within a seed (`exact_random`) reproduces
the v3 budget–accuracy trade-off — the null reading, modulo the cap/schedule confounds above.

v4 tests the signal *on its own terms* rather than via budget vs full fine-tuning:
- **Stage 0 oracle probe** (§5 below): local Spearman of forward gain vs realized per-step gain in
  trailing 64-step windows, moving-block bootstrap (block 64, 10k resamples), with old LER logged in
  shadow mode using the *same* statistic. The oracle gate asks whether the forward score predicts
  contribution above a within-seed permutation and a step-index/loss-stratified partial correlation.
- Only if that local ρ ≥ 0.20 with a bootstrap CI excluding 0 on both seed 11 and seed 13
  does the signal claim hold. (ρ > 0 is too weak: over ~800 points it passes at ρ≈0.07,
  which is statistically significant but explains <1% of variance and cannot move accuracy.)
  Otherwise Path B.

## 5. Headline (until Stage 0 decides)

The v3 data show backward skipping at 30% is non-inferior to full fine-tuning within 0.5 pt, saves
~7% wall-clock under LER (vs ~16% under random), and that the LER ranking is not detectably better
than a permutation of itself — **but** both comparisons were confounded (cap 4 vs no cap; LR schedule
artifact), so none of this is v4 evidence. Path B is the default assumption; it becomes the thesis
headline only if Stage 0 fails. This chapter stays at "draft, outcome undecided" until the Stage 0
oracle probe is run and dated in `docs/gate_a_decision.md`.
