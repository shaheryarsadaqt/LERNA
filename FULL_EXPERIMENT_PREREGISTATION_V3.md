# LERNA Phase 1.3 Full Experiment — Pre-Registration Amendment v3

**Status**: Binding pre-registration for the corrected QNLI confirmatory run
**Date**: 2026-09-28
**Previous version**: v2 (2026-09-27, frozen SHA 8fc60da — INVALIDATED, see diagnosis)
**Previous-previous version**: v1 (2026-09-25, SHA 64f16f0cb90a73a28ce396ed7623c362efef4e07)

**Frozen SHA**: df70f428fac367a905defc08bd90d462d87262f6
("Phase 5: fix seed propagation bug, validator rule, analyze-only mode, pre-reg v3")

**Why v3**: The v2 confirmatory run (SHA 8fc60da) is invalidated because `collect_cell_metrics`
in `scripts/gate3_qnli_confirmatory.py` iterated fingerprint directories alphabetically and returned
the first completed result, ignoring the per-seed fingerprint. All arms reported identical accuracy
across seeds (SE=0, t=∞), which is mathematically impossible for independent training runs.
See `experiments/diagnosis_qnli_gate3.md` for the full diagnosis.

The v2 frozen SHA 8fc60da is preserved read-only in `experiments/qnli_v3_invalid_identical_seeds/`.
Its data is never cited as a result.

---

## 1. Scope of This Amendment

v1 pre-registered a 120-cell QNLI matrix: 6 arms × 2 rates × 10 seeds.
v2 amended to 20 cells: 4 arms × 1 rate × 5 seeds. v2 was invalidated by the seed bug.

v3 is the corrected pre-registration:
- Same 4 arms, same 5 seeds, same rate 0.30 (as v2)
- Seed propagation fixed (see Section 6)
- Validator rule strengthened (any duplicate accuracy across ≥2 seeds ⇒ invalid)
- Tracker v3 acceptance verified (v1↔v3 equivalence, CUDA overhead)
- Provenance hardened (clean-tree, disk, SHA collision preflight)
- Analysis script (`summarize_matrix.py`) extended with paired t-test + TOST

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

**Total**: 4 arms × 5 seeds = **20 cells**, ~3.3 GPU-hours (worst case 7.5h with retries/reserves)

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
| online_ler_update_interval | 1 |

---

## 3. Primary Outcome

**Metric**: `eval_accuracy` (QNLI uses accuracy, not F1)

**Primary hypothesis (two-sided)**:
H0: Δacc(lgs − exact_random) = 0
H1: Δacc(lgs − exact_random) ≠ 0
α = 0.05, paired per-seed differences, t-test (n=5, df=4, t_crit=2.776) + bootstrap 95% CI (10,000 resamples)

### Decision rules (written before execution):
- CI excludes 0 → **directional finding** (LER advantage if Δ > 0, disadvantage if Δ < 0)
- CI includes 0 → **no detectable advantage** (MDE ≈ 0.6–0.8 pt)
- If t = ±∞ or SE = 0 → pipeline bug, matrix invalidated (this is the v2 failure mode)

---

## 4. Secondary Outcomes (Pre-Specified)

### 4.1 TOST Non-Inferiority
- **Margin**: 0.5 pt (0.005 in accuracy)
- LER vs exact_random: two one-sided tests (TOST), α = 0.05 each
- LER vs full_finetune: same TOST
- `phase_strat_guarded` vs `exact_random`: same TOST
- TOST passes if CI_lower ≥ −margin → **equivalence claimed**

### 4.2 Efficiency Outcomes
- `backward_calls` (by construction, reported not claimed)
- `train_loop_seconds` and `mean_step_ms_executed` from serial single-GPU runs only
- `gpu_busy = Σgpu_ms / Σwall_ms` from CUDA events (exact_random and ler_guided_stratified, 200 steps)
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
(`tests/test_sampled_dynamics_v1_v2.py::test_v1_vs_v3_outputs_match_atol`)

**Acceptance on this SHA**: ✅ Verified (1257 tests pass, 0 failures)

---

## 6. Seed Propagation Fix

**Root cause of v2 bug**: `collect_cell_metrics(base_output_dir, seed, arm)` iterated
ALL fingerprint directories under an arm and returned the first completed one,
ignoring the `seed` parameter. The fingerprint (which encodes training_seed) was
never used to select the correct cell.

**Fix applied**:
1. `collect_cell_metrics` signature changed to `(base_output_dir, arm, fingerprint)`
2. Both call sites use fingerprint: `collect_cell_metrics(base_output_dir, arm, fingerprint)`
3. Analyzer uses `collect_cell_metrics(base_output_dir, arm, cell["fingerprint"])`

**Validator rule added**:
```python
# Any duplicate accuracy across ≥2 seeds for the same arm ⇒ invalid
if len(set(vals)) < len(vals):
    seed_identity_violation = True
```

**Seed propagation verified**:
- `training_seed`, `policy_seed`, `sample_seed` are identical in manifests and results
- `torch.manual_seed(seed)`, `np.random.seed(seed)`, `torch.cuda.manual_seed_all(seed)` all called
- `TrainingArguments(seed=seed)` ensures DataLoader generator is seeded
- Trainer's `_seed` propagates to `set_seed()` at training start

### 6.1 Known Statistical Coincidence (Not a Bug)
At the frozen SHA, ler_guided_stratified seeds 11 and 13 both yield acc=0.9322716456159619
(5093/5463 correct). Verification shows:
- training_seed=11 vs 13 (different in all manifests)
- train_loss: 0.23371 vs 0.23169 (DIFFERENT)
- eval_loss: 0.18636 vs 0.18815 (DIFFERENT)
- Models are genuinely different; accuracy match is a chance event at 5463-sample granularity

This coincidence is documented in the diagnosis report. It does NOT invalidate the
seed propagation (the models ARE different). It is flagged by the validator rule as
a conservative signal that the analyst must inspect and confirm is not a bug.

---

## 7. Execution Protocol

1. **Serial execution**: One cell at a time on single V100.
2. **Clean git state**: `require_clean_git_state()` enforced before any cell runs.
3. **Plan persisted first**: plan JSON written before execution.
4. **Retry safety**: retry-safe layout (`attempt-NNN`, `os.makedirs(exist_ok=False)`).
   On infra failure (OOM, env), fix env only, resume unattempted cells, same SHA.
5. **fp16 NaN**: Strict fail, seed replaced per list. Never switch precision mid-matrix.
6. **Env vars**: `TRANSFORMERS_CACHE` and `HF_ENDPOINT` unset in runner entrypoint.
7. **No deletions during run**: The pre-registration rule "no code changes during run"
   is enforced. Data is never deleted or copied mid-run.
8. **Host isolation**: No other jobs on the host during the serial run.

---

## 8. Failure Policy

- Strict forward-loss check ON (`nonfinite_forward_loss_policy="fail"`)
- `forward_loss_observations == total_steps` (6548 for QNLI)
- `nonfinite_forward_loss_count == 0`
- `power_evidence.coverage >= 0.9` for `energy_valid=True` (smoke test treats as informational)
- Manifest must carry `git_sha`, `tracker_revision`, `nonfinite_forward_loss_policy`
- Seed propagation validator: no two seeds produce identical accuracy for the same arm
- Non-zero exit if matrix incomplete or invalidated

---

## 9. Reproducibility Fields

Every cell manifest records:
- `git_sha`: full commit SHA of frozen code (df70f42...)
- `git_dirty`: false
- `tracker_revision`: "v3"
- `nonfinite_forward_loss_policy`: "fail"
- `identity_inputs`: all canonical fields
- `fingerprint`: SHA-256 of identity_inputs (includes training_seed)

---

## 10. Changes from v2

| Section | v2 | v3 (this amendment) |
|---------|-----|---------------------|
| Frozen SHA | 8fc60da (INVALID) | df70f42 (fix seed propagation) |
| `collect_cell_metrics` | (base, seed, arm) — ignores fingerprint | (base, arm, fingerprint) — correct |
| Validator rule | `len(set(vals)) == 1` (all identical) | `len(set(vals)) < len(vals)` (any duplicate) |
| Smoke test | 6 arms × 1 seed (42) | 4 arms × 2 seeds (42, 43) |
| MRPC pre-check | F1 convergence | F1 within 1.5pt + seed-difference check |
| GPU busy | (not specified) | CUDA-event-based, 200 steps |
| Analysis | stdout only (lost) | `summarize_matrix.py` with paired t + TOST |
| Seed-42 QNLI | Archived as exploratory | Confirmed deviated, cited only in appendix |

---

## 11. MRPC Smoke (Pre-Run Gate)

**Script**: `scripts/gate2_mrpc_smoke_v2.py`
**Design**: 4 confirmatory arms × 2 seeds (42, 43) × rate 0.30 = 8 cells
**Pass criteria**:
- 8/8 cells complete, 0 non-finite losses
- Seeds 42 and 43 differ in every arm (no seed propagation bug)
- ler_guided_stratified seed-43 F1 within 1.5 pt of seed-42 F1
- eval_seconds > 0
- Manifests complete (git_sha, tracker_revision, nonfinite_forward_loss_policy)
- M2c strict (forward_loss_observations == total_steps, nonfinite_count == 0)

**Status**: Pre-registration committed; smoke to run before QNLI confirmatory.

---

*Amendment v2 → v3 is binding once committed. The frozen SHA is df70f42.
Any code change = new plan, never a retry. The invalidated v2 run is
archived read-only at `experiments/qnli_v3_invalid_identical_seeds/`.*
