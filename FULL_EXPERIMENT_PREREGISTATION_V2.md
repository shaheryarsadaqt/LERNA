# LERNA Phase 1.3 Full Experiment — Pre-Registration Amendment v2

**Status**: Amended pre-registration for QNLI confirmatory run
**Date**: 2026-09-27
**Previous version**: v1 (2026-09-25, SHA 64f16f0cb90a73a28ce396ed7623c362efef4e07)
**Frozen SHA**: 87e114bb9ec62dabe6cec8f49acb7524af91b09b (Phase 2 smoke validated)

This amendment replaces Section 3–5 of v1 for the QNLI confirmatory component.
The MRPC 120-cell production matrix (Section 2 of v1) remains unchanged.

---

## 1. Scope of This Amendment

v1 pre-registered a 120-cell QNLI matrix: 6 arms × 2 rates × 10 seeds (5 paired-complete).
Analysis of the pilot and the seed-42 exploratory run revealed:

1. **Seed-42 run was deviant**: seeds 42...8192 do not match the pre-registered list and the
   code SHA (6bfd2b2) does not match the pre-registered SHA (64f16f0). It is archived as
   `experiments/qnli_exploratory_seed42/` and cited only as exploratory.
2. **Six arms at two rates** is overpowered for a focused confirmatory claim and exceeds
   the time budget. This amendment reduces to 4 arms × 1 rate.
3. **Tracker v3** (flat-tensor gather, batched entropy) is introduced as a code change
   requiring its own pre-registration. The v1 plan used tracker v2.

The M2c, M3, AMP, and provenance protocols from v1 remain in force unchanged.

---

## 2. QNLI Confirmatory Design

### 2.1 Arms (4)
| Arm | Description | Skipping? | Signal |
|-----|-------------|-----------|--------|
| `full_finetune` | Baseline: no skipping | No | N/A |
| `exact_random` | Uniform random backward skipping | Yes | None |
| `ler_guided_stratified` | LER-guided stratified skipping | Yes | Online LER (tracker v3) |
| `phase_strat_guarded` | Phase-stratified with guardrails | Yes | None (rule-based) |

### 2.2 Skip Rate
- `0.30` (30% target backward skip) — only rate for confirmatory QNLI

### 2.3 Seeds (Target: 5 paired-complete)
Ordered list: **11, 13, 17, 19, 23**
Reserves: **29, 31**

- A seed counts only if **all 4 cells** (4 arms × 0.30) validate
- Replace first failed seed from reserves in order
- If >2 seeds replaced → stop and report
- Seeds 42, 123, 456, 789, 1024, 2025, 4096, 8192 are explicitly **not** on the confirmatory list

### 2.4 Task Configuration
| Parameter | Value |
|-----------|-------|
| Task | QNLI (glue/qnli) |
| Dev set | 5,463 samples → 1,087 eval |
| Epochs | 2 |
| Batch size | 32 (effective, single-GPU) |
| Total steps/cell | ~6,548 |
| Est. time/cell | ~15 min (serial, single V100) |

**Total**: 4 arms × 5 seeds = **20 cells**, ~3.3 GPU-hours

### 2.5 Precision & Model
| Parameter | Value |
|-----------|-------|
| Model | `jhu-clsp/ettin-encoder-150m` @ `45d08642849e5c5701b162671ac811b7654bfd9f` |
| torch_dtype | float16 |
| attn_implementation | eager |
| max_grad_norm | 1.0 |
| learning_rate | 2e-5 |
| warmup_ratio | 0.06 |
| fp16 | True |
| online_ler_update_interval | 1 (NOT changed — v3 reduces per-call overhead, not call frequency) |

---

## 3. Primary Outcome

**Metric**: `eval_accuracy` (QNLI uses accuracy, not F1)

**Primary hypothesis (two-sided)**:
H0: Δacc(lgs − exact_random) = 0
H1: Δacc(lgs − exact_random) ≠ 0
α = 0.05, paired per-seed differences, t-test and bootstrap 95% CI (10,000 resamples)

### Decision rules (written before execution):
- CI excludes 0 → **directional finding** (LER advantage if Δ > 0, disadvantage if Δ < 0)
- CI includes 0 → **no detectable advantage** (MDE ≈ 0.6–0.8 pt based on seed-42 exploratory CI [−0.33, +0.04])

---

## 4. Secondary Outcomes (Pre-Specified)

### 4.1 Non-inferiority TOST
- **Margin**: 0.5 pt (0.005 in accuracy)
- LER vs exact_random: two one-sided tests (TOST), α = 0.05 each
- LER vs full_finetune: same TOST
- `phase_strat_guarded` vs `exact_random`: same TOST
- TOST passes if both one-sided tests reject → **equivalence claimed**

### 4.2 Efficiency outcomes
- `backward_calls` (by construction, reported not claimed — LER arms execute fewer)
- `train_loop_seconds` and `mean_step_ms_executed` from serial single-GPU runs only
- `gpu_busy` = Σgpu_ms / Σwall_ms from CUDA events (exact_random and ler_guided_stratified, 200 steps)
- Energy: coverage reported, not a primary or secondary outcome

---

## 5. Tracker v3 Specification

**Revision**: v3 (flat-tensor gather)
**Code change**: `lerna/utils/lagged_ler.py:_sampled_dynamics_v3`

Key changes from v2:
- `_global_sample_index`: single precomputed index tensor at plan time
- `_capture_flat_tensor`: `torch.cat` of flattened params/exp_avg/exp_avg_sq per family
- One `index_select` per family instead of per-tensor (184 → 3 kernel launches)
- `_compute_entropy_v3`: single `.tolist()` for std+mean (batched CPU sync)
- One `.item()` per metric (3 total: velocity, dot_sum, signal_sq_sum)

**Validation**: v1 vs v3 match within atol=1e-6 on 200-step synthetic trajectory
(test_sampled_dynamics_v1_v2.py)

---

## 6. Execution Protocol

1. **Serial execution**: One cell at a time on single V100. Parallel cells on one host
   contaminate timing (host-bound loop).
2. **Clean git state**: `require_clean_git_state()` enforced before any cell runs.
3. **Plan persisted first**: `matrix_plan.json` written before execution.
4. **Retry safety**: retry-safe layout (`attempt-NNN`, `os.makedirs(exist_ok=False)`).
   On infra failure (OOM, env), fix env only, resume unattempted cells, same SHA.
5. **fp16 NaN**: Strict fail, seed replaced per list. Never switch precision mid-matrix.
6. **Env vars**: `TRANSFORMERS_CACHE` and `HF_ENDPOINT` unset in runner entrypoint.

---

## 7. Failure Policy

- Strict forward-loss check ON (`nonfinite_forward_loss_policy="fail"`)
- `forward_loss_observations == total_steps` (6548 for QNLI)
- `nonfinite_forward_loss_count == 0`
- `power_evidence.coverage >= 0.9` for `energy_valid=True`
- Manifest must carry `git_sha`, `tracker_revision`, `nonfinite_forward_loss_policy`
- Non-zero exit if matrix incomplete

---

## 8. Reproducibility Fields

Every cell manifest must record:
- `git_sha`: full commit SHA of frozen code (91e8bf9...)
- `git_dirty`: false
- `tracker_revision`: "v3"
- `nonfinite_forward_loss_policy`: "fail"
- `identity_inputs`: all canonical fields per v1
- `fingerprint`: SHA-256 of identity_inputs

---

## 9. Changes from v1

| Section | v1 | v2 (this amendment) |
|---------|-----|---------------------|
| QNLI Arms | 6 arms (all Phase 1.3 canonical) | 4 arms (drop fixed_phase_strat, ler_guided_stratified_safe) |
| QNLI Skip rates | 0.30, 0.40 | 0.30 only |
| QNLI Seeds | 10 (11..43) | 5 (11,13,17,19,23) + 2 reserves (29,31) |
| QNLI Cells | 120 | 20 |
| Tracker | v2 | v3 (flat-tensor gather) |
| Decision rule | Equivalence claim | Two-sided t-test + TOST (proper non-inferiority) |
| Seed-42 run | Included in pre-reg | Archived as exploratory (`qnli_exploratory_seed42/`) |

---

*Amendment v1 → v2 is binding once the first confirmatory QNLI cell starts at SHA 91e8bf9.
Any code change = new plan, never a retry.*
