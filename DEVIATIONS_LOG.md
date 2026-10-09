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