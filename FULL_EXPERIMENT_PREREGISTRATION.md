# LERNA Phase 1.3 Full Experiment — Pre-Registration

**Status**: Pre-registered after Gate 2 pilot re-run validation
**Date**: 2026-09-25
**Git SHA**: 64f16f0cb90a73a28ce396ed7623c362efef4e07

---

## 1. Executive Summary

This document pre-registers the complete Phase 1.3 ablation matrix design before execution. The 12-cell pilot (seed 7, MRPC, fp16+eager) has been re-run on the hardened codebase (Gate 2) with all M2c fields exported, M3 validation wired, eager attention in fingerprint, AMP accounting active, and timing breakdown captured. All 12 cells validated.

---

## 2. Experimental Design

### 2.1 Arms (6)
| Arm | Description | Skipping? |
|-----|-------------|-----------|
| `full_finetune` | Baseline: no skipping | No |
| `exact_random` | Uniform random backward skipping | Yes |
| `fixed_phase_strat` | Phase-stratified fixed schedule | Yes |
| `phase_strat_guarded` | Phase-stratified with guardrails | Yes |
| `ler_guided_stratified` | LER-guided stratified skipping | Yes |
| `ler_guided_stratified_safe` | LER-guided with safety fallback | Yes |

### 2.2 Skip Rates (2)
- `0.30` (30% target backward skip)
- `0.40` (40% target backward skip)

### 2.3 Seeds (Target: 5 paired-complete)
Ordered list: **11, 13, 17, 19, 23, 29, 31, 37, 41, 43**
- A seed counts only if **all 12 cells** (6 arms × 2 rates) validate
- Failed seeds are replaced by the next in the list
- **No re-running failed cells with code changes mid-matrix**

### 2.4 Tasks (2)
| Task | Dev Set Size | Epochs | Est. Steps/Cell | Est. Time/Cell | Total Cells |
|------|--------------|--------|-----------------|----------------|-------------|
| MRPC | 408 | 5 | 575 | ~2 min | 60 (5 seeds × 12) |
| QNLI | 5,463 | 2 | ~6,500 | ~25-30 min | 60 (5 seeds × 12) |

**Total**: 120 cells, ~28-33 GPU-hours on V100 32GB

### 2.5 Primary Outcomes (all first-class)
For each cell:
1. **eval_f1** (macro for MRPC, accuracy for QNLI)
2. **eval_accuracy**
3. **train_loop_seconds** (from instrumentation)
4. **energy_joules** (from nvidia-smi telemetry)
5. **backward_calls_executed** (true_skip_instrumentation.backward_calls)
6. **optimizer_steps_skipped_amp** (AMP overflow skips)
7. **mean_step_time_executed_ms**, **mean_step_time_skipped_ms**

### 2.6 Analysis Plan
- **Paired comparison**: Each skipping arm vs `exact_random` within the same seed
- **Bootstrap 95% CI** over seeds (10,000 resamples)
- **Reported per task and per skip rate** — no pooling across tasks
- **Primary hypothesis**: LER-guided arms achieve non-inferior F1 with ≥10% wall-clock reduction vs `exact_random`

### 2.7 Precision Regime
- **fp16 + attn_implementation="eager"** for ALL arms
- Documented as off-recipe deviation: Ettin trained with bf16-AMP; bf16 unavailable on V100
- Report AMP-skip counts per arm to show overflows did not differ systematically

### 2.8 Failure Policy
- Strict forward-loss check remains ON (`NonfiniteForwardLossError` terminates attempt)
- Failed cells preserved with full evidence
- Matrix reports: attempted / failed / completed / validated
- Non-zero exit if matrix incomplete

---

## 3. Gate 1 Implementation Status (Verified)

| Requirement | Implementation | Verified |
|-------------|----------------|----------|
| M2c fields exported | `nonfinite_forward_loss_policy`, `forward_loss_observations`, `nonfinite_forward_loss_count`, `first_nonfinite_forward_loss_step`, `first_nonfinite_forward_loss_value` in `SkipInstrumentation` | ✅ Pilot re-run |
| M3 validation wired | `check_forward_loss_numerics()` in `freeze_matrix_validation()` | ✅ Pilot re-run |
| AMP skip counter | `optimizer_steps_skipped_amp`, `scaler_scale_final`, `scaler_scale_min` via `_GradScalerStepWrapper` | ✅ Pilot re-run |
| Identity inputs extended | `attn_implementation`, `fp16`, `torch_dtype`, `max_grad_norm`, `learning_rate`, `warmup_ratio`, `weight_decay` in `build_identity_inputs()` | ✅ Fingerprint includes all |
| Scripted summaries only | `scripts/summarize_matrix.py` generates all tables from results.json + matrix_validation.json | ✅ Used for pilot |
| Timing breakdown | `train_loop_seconds`, `eval_seconds`, `setup_seconds`, `telemetry_overhead_seconds`, `mean_step_time_executed`, `mean_step_time_skipped`, `dataloader_wait_seconds` | ✅ Pilot re-run |

---

## 4. Pilot Re-Run Evidence (Gate 2)

**Output directory**: `/new_raid/nanhangproj/experiments/ablation_pilot_v2_rerun/`

| Metric | Value |
|--------|-------|
| Git SHA | 64f16f0cb90a73a28ce396ed7623c362efef4e07 |
| Plan SHA-256 | 33375e56da90d2b238f7dc3d4cc5a9652245e71d75f2bcec5ed2382b9e6159ee |
| Environment SHA-256 | 8418069a06ed60198e3a2cde1d7c2646ec863d4dddeb6c816d2f14393f88bc55 |
| Cells validated | 12/12 |
| M2c: forward_loss_observations | 575 (all cells) |
| M2c: nonfinite_forward_loss_count | 0 (all cells) |
| M2c: policy | "fail" (all cells) |
| AMP skips | 0 (all cells) |
| Scaler scale tracked | Yes (2048-4096 range) |

---

## 5. Execution Order

1. **MRPC first** (cheap; confirms multi-seed stability, ~2.5 GPU-hrs)
2. **QNLI second** (main statistical power, ~25-30 GPU-hrs)
3. Each task: run all seeds sequentially; replace failed seeds from ordered list
4. All intermediate reports via `scripts/summarize_matrix.py` only

---

## 6. Approval Checklist

- [ ] Design pre-registered (this document)
- [ ] GPU allocation confirmed (V100 32GB, ~30 GPU-hrs)
- [ ] Data cached offline (MRPC, QNLI)
- [ ] Conda environment `lerna` frozen
- [ ] Git state clean at launch
- [ ] Matrix plan persisted before any cell execution

---

## 7. Deviations from Standard Recipe (Documented)

| Deviation | Justification |
|-----------|---------------|
| fp16 (not bf16) | V100 lacks bf16 support |
| attn_implementation="eager" | fp16+sdpa caused NaN at step 9 in pilot; eager resolves |
| Fixed 5 epochs for MRPC | Matches pilot; QNLI uses 2 epochs for step budget |
| Single-GPU forced | Avoids NCCL/DataParallel instability on DGX |

---

*This pre-registration is immutable once the first production cell starts. Any changes require a new Git SHA and new pre-registration.*