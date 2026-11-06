# LERNA v4 Deviations Log
# Analysis freeze: 2026-10-09. All numbers below are final. Nothing gets re-run.

## 1. Tag history (v4-prereg-r3 moved 4 times)
- 168a820 — original r3 tag
- 30bca95 — after AlwaysFalsePolicy diagnostics fix (validator)
- ac1a952 — dereference annotated tag in _check_provenance
- b91878a — after Stage 0 NaN fix + eager attention (B17)
The tag moved because the provenance check requires HEAD == v4-prereg-r3.
All Stage 2 cells ran at commit 30bca95 (results.json provenance field).
The training path (lerna/ and scripts/stage2_confirmatory.py) was UNCHANGED
between a6c6d13 (first launch) and 30bca95 (re-run). Only AlwaysFalsePolicy
diagnostics changed (added decisions_seen counter, omitted quota keys when zero).
See: git diff a6c6d13..30bca95 -- lerna/ scripts/stage2_confirmatory.py

## 2. Discarded runs
- Earlier Stage 2 run: Delta = +0.16 pt, p = 0.012. Thrown away after B17
  eager-attention fix. Both runs pointed the same direction.
- Earlier Stage 0: S2 rho = -0.28. Thrown away after S2 sign fix.
  Both Stage 0 versions picked S1, so the primary is unaffected either way.

## 3. S2 sign change (CRITICAL — theoretical inversion)
_s2_alignment_core in lerna/utils/forward_gate.py returns -s (line 802).
The raw inner product s_raw = <G_B, G_P / sqrt(v)> predicts the probe loss
DROP with the SAME sign (verified on a tiny model: s_raw=+195.8, actual
P_s drop=+0.0039, same sign). AdamW update dW = -lr * G_B/sqrt(v), so the
probe loss change is approximately -lr * s_raw — meaning s_raw > 0 predicts
the probe loss INCREASES (negative drop).
The negation was applied empirically after Stage 0 data showed rho went from
-0.28 to +0.100. This is overfitting to the P_s/P_e split. The theoretical
sign is s_raw, not -s_raw. S2 was not selected either way (S1 won both
Stage 0 seeds), so the primary is unaffected, but this must be disclosed.

## 4. Stage 0 probe losses computed in train mode
The pre-registration says eval mode. compute_gp_from_logits and
compute_probe_loss in scripts/stage0_oracle_probe.py no longer call
model.eval()/model.train() (they were removed to fix a NaN bug where the
mode transitions corrupted ModernBERT state on CUDA). The forward passes
now run in train mode under torch.no_grad(). ModernBERT uses LayerNorm
(not BatchNorm), so train vs eval is deterministic here. Deviation from
pre-reg, disclosed.

## 5. Late harness check (GPU nondeterminism)
control/11 gave 5050 correct in the first run (manifest at a6c6d13) and
5063 in the re-run (results at 30bca95). Same code, same seed, different
result. This is the nondeterminism floor: 13 of 5463 dev examples.
gate/11 matched exactly across runs. No McNemar test is possible because
no per-example predictions were saved.

## 6. Dynamic padding
Training batches pad to the longest in batch (cap 512), not max_length.
padding="max_length" is prohibited in production. This is a pre-registered
deviation from the original plan.

## 7. No checkpoints or per-example predictions saved
Each cell saved only: instrumentation.json, manifest.json, results.json.
results.json contains aggregate eval_accuracy and eval_loss only.
No per-example predictions, no logits, no model checkpoints. Any
per-example re-analysis would require re-running the cells.

## 8. Pairing limitation
The paired t-test catches run-to-run noise but cannot run a McNemar test
because per-example predictions were not saved.

## 9. Winner's curse
An effect of +0.14 pt close to the 0.17 pt MDE probably overstates the
true effect size.

## 10. Open questions
- PREREG_V4.md last changed 2026-10-05 09:53 UTC. First Stage 2 manifest
  2026-10-06 07:42 UTC. Pre-reg was frozen before launch. Audit item 1 PASSES.
- PREREG_V4.md names jhu-clsp/ettin-encoder-150m at revision 45d0864.
- No new Stage 2 seeds, rates, or scores. No extra SST-2 seeds.
  No reruns of the primary.

## 11. Phase 1.2 addendum (2026-10-09)
- Phase 1.2 "full" baseline matrix launched 2026-10-09 at commit d3397aa
  (6 baselines x 4 GLUE tasks x 5 seeds = 120 cells, Ettin-150M at 45d0864).
- Pre-registered as an addendum; the V4 primary (QNLI gate vs control) is untouched.
- Full 720-cell GLUE matrix (8 tasks x 6 baselines x 2 rates x 5-10 seeds) is
  future work, not completed. Orchestrator and 7 analysis scripts archived and reusable.
- `v4-prereg-r3` moved to d3397aa to satisfy the provenance check for this addendum.
  The V4 Stage 2 cells ran at 30bca95 and retain that SHA in their results.json files.

## 12. E3 reduced_steps budget defect and rerun (2026-10-22)
- **Defect**: the E3 (MRPC/RTE) `reduced_steps` arms ran **2619** optimizer
  steps — QNLI's hardcoded `REDUCED_TOTAL_STEPS` — because
  `scripts/stage2_confirmatory.py` resolved the reduced-step count from
  `REDUCED_TOTAL_STEPS` whenever `--reduced-fraction` was not supplied. On
  MRPC (`total_steps=377`) and RTE (`total_steps=258`) this trained 7x and 10x
  longer than the full-finetune arm. Those two rows were therefore **not
  compute-matched**, and their apparent accuracy advantage was over-training,
  not a compute-saving result.
  Archived (not deleted) at `/raid/stage2_e3/{mrpc,rte}/reduced_steps/`.
- **Fix**: commit `5f2c161` on branch `fix/e3-reduced-steps-budget` adds an
  absolute `--reduced-steps` override (taking precedence over
  `--reduced-fraction`) and resolves the reduced-step count in a single place
  (`_resolve_reduced_steps`) so `TrainingArguments.max_steps`, `min_step`, and
  the scheduler-step assertion cannot disagree. `int(total_steps*0.40)` cannot
  express the required budgets exactly (377*0.40 -> 150, not 151).
- **Rerun**: commit `7ef4d9d`, 24 cells (12 seeds x {MRPC, RTE}), output
  `/raid/stage2_e3_fixed/{mrpc,rte}/reduced_steps/`.
  MRPC = **151** steps, RTE = **103** steps (= the gate/control backward
  budget, 377 - round(0.60*377) and 258 - round(0.60*258)).
  All 24 cells: `validation_passed=true`, `backward_calls == forward_calls ==
  expected`, `skipped_backward_steps == 0`, `tree_dirty=false`,
  `provenance.post_hoc=true`.
- **Effect of the correction** (accuracy, budget-matched):

  | Task | Arm | defective (2619 steps) | corrected (matched budget) |
  |---|---|---|---|
  | MRPC | reduced_steps | 0.8597 +/- 0.0129 | **0.7625 +/- 0.0237** |
  | RTE | reduced_steps | 0.7428 +/- 0.0208 | **0.5912 +/- 0.0197** |

  The corrected reduced_steps arms are no longer above the gate/control arms
  (MRPC gate 0.7637, control 0.7676; RTE gate 0.6014, control 0.6038).
  Paired: MRPC vs control -0.51 pt (p=0.39), RTE vs control -1.26 pt (p=0.11).
  The earlier "reduced_steps wins" claim does not survive budget matching.
- **Provenance caveat**: E3 is a post-hoc extension and is **not** covered by
  `PREREG_V4.md`; the rerun is stamped `post_hoc=true` and records both its own
  HEAD sha and `prereg_tag_sha` in every results.json.
- **Provenance bug found and fixed**: `provenance.tree_dirty` was computed as
  `not _check_lerna_in_checkout()`, i.e. it reported whether the `lerna`
  package was importable from the checkout, not whether the git tree was
  clean. Every results.json written before commit `7ef4d9d` therefore carries
  a meaningless `tree_dirty` value and must not be cited as evidence of a
  clean tree. Separately, results.json was written **before** the validator
  ran, so single-cell artifacts never recorded `validation_passed`; the file is
  now rewritten after validation.
- **Unchanged**: the QNLI V4 primary (gate vs control, 24 seeds, rate 0.60)
  was not re-run and is unaffected.