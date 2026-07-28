# LERNA: Learning Efficiency Ratio Navigation and Adaptation

Temporal compute control for transformer fine-tuning through true backward skipping.

> **Research status:** LERNA is an active master's thesis research prototype.  
> The current controller evidence is limited to MRPC over three seeds. Broader claims require additional experiments.

## Overview

Standard transformer fine-tuning executes a forward pass, backward pass, optimizer update, and learning-rate scheduler update for every training batch.

LERNA investigates **Training Step Inequality**: the hypothesis that training steps can differ in their marginal learning value and may not all require the same backward-computation budget.

The current implementation provides:

- operation-level true backward skipping;
- exact-quota random skipping;
- matched-budget controller evaluation;
- Phase-Stratified guarded random skipping;
- Random-Veto Deferral (RVD);
- realized skip-rate and optimizer-operation instrumentation;
- run manifests and result-validation checks.

## Current Research Status

| Phase | Purpose | Status |
|---|---|---|
| Phase 1.1 | Diagnose late-stage fine-tuning inefficiency | Diagnostic runs completed; aggregate analysis remains limited |
| Phase 1.2 | Evaluate simple compute-saving baselines | Partially completed across six reported GLUE tasks |
| Phase 1.3 | Evaluate controllers under matched backward-skip budgets | MRPC study completed for seeds 42–44 at 30% and 40% targets |

Phase 1.3 is the primary controller study in the current midterm report.

## True Backward Skipping

Every training batch still receives a forward pass. After the forward pass, the controller chooses between a full update and a skipped update.

| Operation | Full update | Skipped update |
|---|---:|---:|
| Forward pass | Yes | Yes |
| Backward pass | Yes | No |
| Optimizer step | Yes | No |
| Scheduler step | Yes | No |
| Parameter update | Yes | No |

The authoritative Phase 1.3 mode is `freeze`: parameters and optimizer state do not change on skipped steps.

Backward-call reduction is the primary directly attributable efficiency measurement. Runtime and energy depend on hardware, utilization, diagnostic overhead, and measurement conditions and are therefore treated as secondary measurements.

## Phase 1.3 Controllers

### Full Fine-Tuning

The no-skip reference. Every batch receives a complete backward and optimizer update.

### Exact Random Skip

Selects an exact integer quota of steps uniformly at random after the warm-up period. It does not condition on batch properties and is the strongest current MRPC baseline.

### GradNorm Skip

Uses the most recently observed real pre-clip gradient norm as a lagged signal. Because skipped steps do not produce a new gradient norm, forced probe steps refresh the signal.

The current GradNorm results are off-budget and should be interpreted as a cautionary diagnostic rather than a matched-budget competitor.

### Phase-Stratified LERNA

Divides training into fixed temporal strata and distributes the skip quota increasingly toward later strata. Selection within each stratum remains randomized, with rho and loss-spike signals acting as safety vetoes.

The current implementation demonstrates stable phase-weighted skipping, but it does not yet establish that the LER signal itself improves performance over exact random skipping.

### Random-Veto Deferral

RVD starts from exact random skip proposals. A risk signal may veto a proposed skip and defer the skipped quota to another step.

The evaluated variants are:

- `RVD-Margin`: vetoes low-margin proposed skips;
- `RVD-Loss-Spike`: vetoes proposed skips during large training-loss increases.

With all vetoes disabled, RVD reproduces exact random skipping.

## MRPC Midterm Results

Results are descriptive means and population-style standard deviations over seeds 42, 43, and 44. The sample size is too small for strong statistical superiority claims.

| Controller | Target | Accuracy | Accuracy SD | F1 | F1 SD | Realized skip |
|---|---:|---:|---:|---:|---:|---:|
| Full fine-tuning | 0% | 0.8587 | 0.0145 | Not recorded | — | 0.0% |
| Exact Random Skip | 30% | 0.8791 | 0.0153 | 0.9121 | 0.0121 | 29.9% |
| Phase-Stratified LERNA | 30% | 0.8611 | 0.0110 | 0.9013 | 0.0102 | 29.7% |
| RVD-Margin | 30% | 0.8701 | 0.0072 | 0.9061 | 0.0065 | 29.9% |
| GradNorm Skip | 30% | 0.8007 | 0.0356 | 0.8673 | 0.0167 | ~62.0% |
| Exact Random Skip | 40% | 0.8685 | 0.0179 | 0.9070 | 0.0098 | 40.0% |
| Phase-Stratified LERNA | 40% | 0.8644 | 0.0101 | 0.9042 | 0.0066 | 39.7% |
| RVD-Margin | 40% | 0.8431 | 0.0208 | 0.8930 | 0.0086 | 40.0% |
| RVD-Loss-Spike | 40% | 0.8587 | 0.0162 | 0.9012 | 0.0082 | 40.0% |
| GradNorm Skip | 40% | 0.7868 | 0.0571 | 0.8560 | 0.0381 | ~67.7% |

The current evidence supports the following interpretation:

- exact random skip is the strongest mean-performance baseline;
- Phase-Stratified LERNA remains close to full fine-tuning while skipping approximately 40% of backward calls;
- RVD-Margin has low variance at 30% but degrades at 40%;
- RVD-Loss-Spike is safer than RVD-Margin at 40% but remains below exact random;
- GradNorm does not respect the requested budget.

## Running a Fixed-Budget MRPC Study

Install the package and development dependencies:

```bash
python -m pip install -e ".[dev]"
```

Example Phase-Stratified comparison at a 40% target:

```bash
python scripts/run_ablation_study.py \
  --mode custom \
  --tasks mrpc \
  --seeds 42 43 44 \
  --ablations full_finetune exact_random full_lerna grad_norm \
  --policy phase_strat \
  --target-skip-rate 0.40 \
  --no-early-stopping \
  --skip-update-mode freeze \
  --model modernbert \
  --unlimited \
  --output-dir experiments/phase1_3_mrpc_t040
```

`--no-early-stopping` is required for fixed-horizon matched-budget comparisons. Matched-claim runs also require a clean tracked Git worktree.

Each successful run produces:

- `results.json`
- `instrumentation.json`
- `ler_diagnostics.json`
- `run_manifest.json`

## Instrumentation

The true-skipping trainer records:

- forward calls;
- backward calls;
- skipped backward steps;
- optimizer-step attempts;
- scheduler-step calls;
- realized skip rate;
- skip-update mode;
- policy-specific diagnostics;
- runtime and power telemetry.

Core accounting checks include:

```text
forward_calls = backward_calls + skipped_backward_steps
optimizer_step_attempts <= backward_calls
scheduler_step_calls <= optimizer_step_attempts
```

## Claim Boundaries

The current repository does **not** establish that:

- LERNA surpasses exact random skipping;
- the controller generalizes beyond MRPC;
- backward-call reduction equals proportional energy savings;
- momentum extrapolation is superior to freeze-style skipping;
- LER and rho diagnostics have constant-time overhead;
- theoretical PL convergence guarantees have been empirically validated;
- results have been validated on instruction tuning, generation tasks, or 70B models.

Momentum extrapolation remains available as a historical or exploratory implementation mode. It is not part of the authoritative Phase 1.3 matched-budget claim.

## Current Limitations

- Phase 1.3 controller evidence covers MRPC only.
- The current study uses only three seeds.
- Exact random remains the strongest baseline.
- Phase-Stratified LERNA does not yet isolate the contribution of LER.
- RVD veto signals are not robust across budgets.
- GradNorm is off-budget.
- Online LER and rho diagnostics add model-scale overhead.
- Raw Phase 1.3 result artifacts and deterministic table-generation scripts are not yet included in the repository.

## Next Work

The immediate research priorities are:

1. reproduce the MRPC results from a clean checkout;
2. expand MRPC evaluation to at least ten paired seeds;
3. separate controller RNG from model and dataloader RNG;
4. isolate LER through targeted controller ablations;
5. improve RVD veto calibration and repayment behavior;
6. extend matched-budget evaluation to additional GLUE tasks;
7. measure runtime and energy with diagnostic overhead reported explicitly;
8. provide seed-level artifacts and reproducible table-generation scripts.

## Repository Structure

```text
lerna/trainers/true_skip_trainer.py   True backward-skipping trainer
lerna/trainers/policies.py            Skip and controller policies
lerna/utils/metrics.py                LER and rho diagnostics
lerna/utils/run_provenance.py         Run manifests and provenance
scripts/run_ablation_study.py         Phase 1.3 experiment runner
scripts/validate_skip_policy_results.py
tests/                                Trainer, RVD, provenance, and validator tests
```

## Citation Status

A final archival citation will be added after the thesis results, experimental artifacts, and publication draft are complete.