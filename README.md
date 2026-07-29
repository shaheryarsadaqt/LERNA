# LERNA: Learning Efficiency Ratio Navigation and Adaptation

A research framework for temporal backward-compute control during transformer fine-tuning.

> **Current status:** Active master's thesis research prototype.  
> The current matched-budget controller evidence is limited to MRPC over three training seeds.

## Research Question

Standard transformer fine-tuning gives every minibatch a forward pass, backward pass, optimizer update, and learning-rate scheduler update.

LERNA investigates **Training Step Inequality**: the hypothesis that training steps can differ in their marginal learning value and may not all require the same backward-computation budget.

The current repository focuses on:

- true backward skipping;
- exact-quota controller evaluation;
- temporal skip-budget allocation;
- random-preserving risk vetoes;
- operation-level instrumentation;
- reproducible run provenance.

## Research Status

| Phase | Research purpose | Current status |
|---|---|---|
| Phase 1.1 | Diagnose late-stage fine-tuning inefficiency | Diagnostic runs completed; aggregate analysis remains limited |
| Phase 1.2 | Evaluate simple compute-saving baselines | Partially completed across six reported GLUE tasks |
| Phase 1.3 | Evaluate controllers under matched backward-skip budgets | MRPC study completed over seeds 42–44 at 30% and 40% targets |
| Phase 1.3b | Develop genuinely LER-guided quota allocation | Planned |

Phase 1.3 establishes the true-skipping framework and matched-budget evaluation protocol. It does not yet establish that LER-guided selection improves over exact random skipping.

## True Backward Skipping

Every training batch receives a forward pass. After the forward pass, a controller chooses between a full update and a skipped update.

| Operation | Full update | Skipped update |
|---|---:|---:|
| Forward pass | Yes | Yes |
| Backward pass | Yes | No |
| Optimizer step | Yes | No |
| Scheduler step | Yes | No |
| Parameter update in authoritative Phase 1.3 mode | Yes | No |

The authoritative Phase 1.3 skip-update mode is `freeze`. Parameters and optimizer state remain unchanged on skipped steps.

The trainer records whether backward, optimizer, and scheduler operations were actually executed. This distinguishes true backward skipping from callback-level simulated skipping.

## Current Controllers

### Full Fine-Tuning

The no-skip reference. Every training batch receives a complete backward pass and optimizer update.

### Exact Random Skip

Selects an exact integer quota of skipped steps uniformly at random after the warm-up period.

Exact random does not condition on batch properties or learning-dynamics signals. It is the strongest current MRPC baseline.

### GradNorm Skip

Uses the most recently observed real pre-clip gradient norm as a lagged signal. Because skipped steps do not produce a new gradient norm, forced probe steps are used to refresh the signal.

The current GradNorm runs do not respect the requested 30% and 40% budgets. GradNorm is therefore treated as an off-budget diagnostic baseline rather than a matched-budget competitor.

### Fixed Phase-Stratified Random Skip

A planned clean temporal baseline that will:

- divide training into predetermined equal-length phases;
- assign a fixed skip quota to each phase;
- select skipped steps uniformly at random within each phase;
- use no LER, rho, loss-spike, margin, or gradient-norm signal.

This controller will isolate whether temporal quota allocation alone adds value beyond exact global random skipping.

### Phase-Stratified Guarded Random Skip

This is the accurate name for the controller currently implemented as:

```text
LERNAPhaseStratifiedPolicy
```

The current code identifier is retained temporarily for historical compatibility and will be renamed only after characterization tests lock its existing behavior.

The controller:

- divides training into four predetermined equal-length phases;
- uses fixed phase-quota weights;
- selects steps randomly within each phase;
- optionally prevents consecutive skip bursts;
- uses rho-based and loss-spike safety vetoes.

It is a **fixed-schedule temporal controller with safety vetoes**. It does not use LER to determine phase boundaries, phase quotas, or skip decisions and is not currently classified as a genuinely LER-driven controller.

### Random-Veto Deferral

Random-Veto Deferral, or RVD, starts from exact random skip proposals. A risk signal may veto a proposed skip and defer the skipped quota to another step.

The current evaluated variants are:

- `RVD-Margin`: vetoes proposed skips on low-margin batches;
- `RVD-Loss-Spike`: vetoes proposed skips during large training-loss increases.

When all vetoes are disabled, RVD reproduces exact random skipping.

### LER-Guided Stratified Controller

This is planned Phase 1.3b work and is not part of the current validated MRPC evidence.

The intended controller will:

- compute LER from scheduled real-update probe steps;
- use lagged LER to allocate the next window's skip quota;
- retain random selection within each quota window;
- preserve an exact global backward-skip budget;
- evaluate safety vetoes as a separate ablation.

This separation allows the contribution of LER to be measured against an otherwise equivalent fixed temporal controller.

## Current MRPC Evidence

The current results are descriptive means over training seeds 42, 43, and 44. Three seeds are insufficient for a statistical-superiority claim.

| Controller | Target skip | Mean accuracy | Mean F1 | Realized skip |
|---|---:|---:|---:|---:|
| Full fine-tuning | 0% | 0.8587 | Not recorded | 0.0% |
| Exact Random Skip | 30% | 0.8791 | 0.9121 | 29.9% |
| Phase-Stratified Guarded Random Skip | 30% | 0.8611 | 0.9013 | 29.7% |
| RVD-Margin | 30% | 0.8701 | 0.9061 | 29.9% |
| GradNorm Skip | 30% | 0.8007 | 0.8673 | approximately 62.0% |
| Exact Random Skip | 40% | 0.8685 | 0.9070 | 40.0% |
| Phase-Stratified Guarded Random Skip | 40% | 0.8644 | 0.9042 | 39.7% |
| RVD-Margin | 40% | 0.8431 | 0.8930 | 40.0% |
| RVD-Loss-Spike | 40% | 0.8587 | 0.9012 | 40.0% |
| GradNorm Skip | 40% | 0.7868 | 0.8560 | approximately 67.7% |

The current evidence supports the following limited interpretation:

- exact random has the highest mean MRPC performance;
- Phase-Stratified Guarded Random Skip remains close to full fine-tuning while skipping approximately 40% of backward calls;
- RVD-Margin has low variance at 30% but degrades at 40%;
- RVD-Loss-Spike is safer than RVD-Margin at 40% but remains below exact random;
- GradNorm is off-budget and unstable.

These results do not establish statistical superiority or broad task generalization.

## Instrumentation

The true-skipping trainer records:

- forward calls;
- backward calls;
- skipped backward steps;
- optimizer-step attempts;
- scheduler-step calls;
- realized skip rate;
- skip-update mode;
- controller diagnostics;
- runtime and power telemetry.

Core accounting checks include:

```text
forward_calls = backward_calls + skipped_backward_steps
optimizer_step_attempts <= backward_calls
scheduler_step_calls <= optimizer_step_attempts
```

Backward-call reduction is the primary directly attributable efficiency measurement.

Runtime and energy depend on:

- hardware;
- GPU utilization;
- data loading;
- diagnostic overhead;
- evaluation frequency;
- thermal and contention conditions.

Backward-call reduction must therefore not be interpreted automatically as an equal percentage of runtime or energy savings.

## Diagnostic Cost

The current LER and rho implementation performs model-scale parameter operations, including parameter snapshots and vector comparisons.

Its cost scales with model size and is not considered constant-time overhead.

The next implementation stage will:

- make online diagnostics optional;
- measure diagnostic overhead explicitly;
- reduce per-step parameter processing;
- investigate sampled parameters, layer summaries, optimizer-state norms, and lower-frequency probes.

## Claim Boundaries

The current repository does **not** establish that:

- LERNA surpasses exact random skipping;
- LER-guided control is validated;
- results generalize beyond MRPC;
- backward-call reduction produces proportional runtime or energy savings;
- momentum extrapolation is superior to freeze-style skipping;
- LER or rho diagnostics have `O(1)` cost;
- theoretical convergence guarantees have been empirically validated;
- results have been validated on generative tasks or models up to 70B parameters;
- the current three-seed results demonstrate statistical superiority.

Momentum extrapolation remains historical or exploratory functionality. It is not part of the authoritative freeze-mode Phase 1.3 matched-budget claim.

## Current Limitations

- Phase 1.3 controller evidence is MRPC-only.
- The current study uses three seeds.
- Exact random remains the strongest baseline.
- The existing Phase-Stratified controller is not LER-driven.
- RVD veto signals are not robust across budgets.
- GradNorm does not respect the requested budget.
- Online diagnostics currently add model-scale overhead.
- A complete seed-level Phase 1.3 evidence package is not yet included.
- Dedicated tests for the existing Phase-Stratified policy are still required.

## Immediate Research Plan

1. Add characterization tests for the existing Phase-Stratified Guarded Random Skip controller.
2. Implement a pure Fixed Phase-Stratified Random baseline without safety signals.
3. Make diagnostics optional and measure their overhead.
4. Reduce the cost of LER and rho computation.
5. Repair the experiment runner, ablations, and scientific fingerprints.
6. Implement LER-Guided Stratified control as a separate Phase 1.3b policy.
7. Run at least ten paired MRPC seeds at 30% and 40%.
8. Generate statistical tables directly from validated manifests.
9. Preserve seed-level artifacts for every reported aggregate.
10. Extend matched-budget evaluation beyond MRPC only after the MRPC protocol is reproducible.

## Repository Structure

```text
lerna/trainers/true_skip_trainer.py
    True backward-skipping trainer and operation instrumentation

lerna/trainers/policies.py
    Random, GradNorm, Phase-Stratified, RVD, and experimental policies

lerna/utils/metrics.py
    LER, rho, and supporting diagnostic calculations

lerna/utils/run_provenance.py
    Run manifests, fingerprints, and provenance classification

scripts/run_ablation_study.py
    Phase 1.3 controller experiment runner

scripts/validate_skip_policy_results.py
    Local validation of run artifacts and matched-budget invariants

tests/
    Trainer, RVD, provenance, validator, and horizon tests
```

## Reproducibility Status

Python source compilation currently passes.

Before additional GPU experiments, the complete test suite must be run in an environment containing the development and scientific dependencies, including `pytest` and `scipy`.

Raw seed-level artifacts and deterministic Phase 1.3 table-generation scripts will be added as part of the reproducibility package.

## Citation Status

A final archival citation will be added after the thesis results, seed-level artifacts, reproducibility package, and publication draft are complete.