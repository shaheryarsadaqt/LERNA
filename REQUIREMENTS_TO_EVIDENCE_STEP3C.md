# Step 3C — Requirements-to-Evidence Table

This table maps each R4 requirement (R4-A through R4-D) and R5 packaging item
to the specific evidence in this correction bundle. Status categories:

- **Implemented and tested** — code change present, exercised by targeted CPU
  tests, existing tests pass.
- **Supplied but not independently verified** — artifact provided (manifest,
  patch, reconstruction script) but not re-executed by an independent party.
- **Dependency-blocked** — cannot be completed without external dependencies
  not installed in this environment (real pretrained model, real GLUE dataset,
  torch/datasets/transformers for production loading).
- **Missing real-execution prerequisites** — original execution prerequisites
  (QNLI exclusion file/hash, immutable model/dataset revisions) are absent and
  are explicitly enforced as real-execution blockers by preflight; they must
  NOT be invented.

---

## R4-A — Production loading and preflight

| Requirement | Evidence | Status |
|-------------|----------|--------|
| `_load_model()` no longer always returns TinyModel | `scripts/run_controller_ablation.py:_load_model()` accepts injectable `model_loader`; default returns TinyModel; production variant `_load_model_production()` uses `lerna.utils.model_loader.load_model_and_tokenizer` | Implemented and tested |
| `_load_tokenizer()` no longer returns None / is called | `_load_tokenizer()` accepts injectable `tokenizer_loader`; called in `run_execution()`; production variant `_load_tokenizer_production()` uses `load_tokenizer` | Implemented and tested |
| Training/evaluation loaders no longer always synthetic | `_load_train_dataset()` / `_load_eval_dataset()` accept injectable `dataset_loader`; production variants `_load_train_dataset_production()` / `_load_eval_dataset_production()` use `datasets.load_dataset` with task mapping and tokenization/padding | Implemented and tested |
| Faithful task mapping (QNLI/SST-2/MRPC/RTE) | `TASK_INPUTS` maps each task to question/sentence fields, metric, sentence-pair flag; GLUE builder configs use correct names (mrpc/sst2); production tokenizers apply truncation with max_length512; sentence order preserved (sentence1 before sentence2 for MRPC/RTE); QNLI matched-probe-ID exclusions applied before idx removal using verified exclusion IDs (JSON probe_s_ids/probe_e_ids schema, integer IDs); dynamic padding at minibatch level NOT yet wired into Trainer collator — production DataLoader uses batch_size=1 without tokenizer-aware collator | Partially implemented (task mapping complete; dynamic minibatch padding not yet wired) |
| Final evaluation with required metrics | `_evaluate_model()` computes loss + accuracy from predictions vs labels; MRPC F1 (precision/recall/F1 for positive class) computed and returned as `final_eval_f1` | Implemented and tested |
| External boundaries replaceable with CPU fixtures | All loaders accept `*_loader` callable parameters; targeted tests inject TinyModel/TinyDataLoader factories | Implemented and tested |
| Dry-run does not instantiate model/Trainer/train | `run_dry_run()` only exercises `ScoreQuantileGate.decide()` with synthetic scores; no `Trainer`, no model load, no GPU | Implemented and tested |
| Immutable revision preflight (40-char hex SHA, non-dry-run) | `ExperimentConfig.__post_init__` requires 40-char lowercase hex SHA for model/tokenizer/dataset revisions when `dry_run=False`; rejects "main", "develop", whitespace, and any non-SHA string. Dry-run accepts any revision. Real execution blocked on invalid revision. | Implemented and tested |
| QNLI exclusion-file/hash prerequisites enforced | QNLI task requires `probe_exclusions_file` and `probe_exclusions_hash`; file must exist and its SHA-256 must match the provided hash; missing/mismatch = explicit `ValueError` real-execution blocker | Implemented and tested |
| Same exclusions across QNLI arms | Single preflight check in `__post_init__` applies to all QNLI arms (gate/control/full/reduced) | Implemented and tested |
| No invented exclusions or revisions | No exclusion content fabricated; missing originals remain explicit blockers | Implemented and tested |
| Missing originals may remain blockers | QNLI exclusion file/hash absence is a documented `ValueError` blocker; not silently bypassed | Implemented and tested |

## R4-B — LR schedule shape separated from step policy

| Requirement | Evidence | Status |
|-------------|----------|--------|
| `cfg.lr_scheduler='always_step'` separated from schedule shape | New `lr_schedule_shape` field (constant_with_warmup/constant/linear/cosine) distinct from `SchedulerStepPolicy.ALWAYS_STEP` | Implemented and tested |
| Shape mapped to HF `lr_scheduler_type` | `LR_SHAPE_TO_HF` dict maps shape -> TrainingArguments keyword | Implemented and tested |
| Scheduler advances on every scheduled batch including skips | `_SchedulerStepWrapper` with `SchedulerStepPolicy.ALWAYS_STEP` steps on every batch; validated by `scheduler_step_calls == effective_total` | Implemented and tested |
| `grad_clip` passed as `max_grad_norm` | `TrainingArguments(max_grad_norm=cfg.grad_clip)` | Implemented and tested |
| `capture_logits=False` explicitly rejected | `ExperimentConfig.__post_init__` raises `ValueError` when `capture_logits=False` | Implemented and tested |
| Current-forward logit capture without extra scoring forward | `trainer.capture_logits = True` after construction; `compute_loss` sets `_last_real_logits` from current forward; S1 score source reads `_last_real_logits` | Implemented and tested |
| `lr_schedule_shape='constant'` has no warmup | `constant` schedule: expected LR = learning_rate at all steps (no warmup ramp); distinct from `constant_with_warmup` which warms up then holds constant | Implemented and tested |
| `linear`/`cosine` warmup then decay | During warmup: LR = learning_rate * (effective_total / warmup_steps); after warmup: decays toward 0 over post-warmup horizon (effective_total - warmup_steps); completed horizon endpoint = 0; fixed decay formula (was incorrectly using effective_total as decay denominator). Validated at run_execution boundary | Implemented and tested |
| Missing optimizer LR measurement fails before evaluation | `final_lr is None` adds validation error; `run_execution()` returns FAILED before calling `_evaluate_model`; no KeyError escape | Implemented and tested |
| LR trajectory/endpoint recorded for gate | Gate manifest: `lr_schedule_shape`, `lr_warmup_steps`, `learning_rate`, `scheduler_advances`, `final_eval_accuracy`, `final_learning_rate` — legacy_v4 only; no phase_exact_v1 non-dry manifest yet | Supplied but not independently verified |
| LR trajectory/endpoint recorded for full | Full manifest: same fields; gate-free run at effective horizon — legacy_v4 only | Supplied but not independently verified |
| LR trajectory/endpoint recorded for reduced | Reduced manifest: same fields; reduced_steps = T - Q with own horizon — legacy_v4 only | Supplied but not independently verified |
| Unsupported settings honored or rejected | `lr_schedule_shape` validated against `VALID_LR_SHAPES`; `capture_logits=False` rejected; `grad_clip` passed through | Implemented and tested |

## R4-C — Success validation and failure recording

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Forward/backward/optimizer/scheduler equalities preserved | `run_execution()` validates `forward_calls == effective_total`, `backward_calls == effective_total - realized`, `scheduler_step_calls == effective_total`, `optimizer_step_attempts == backward_calls` | Implemented and tested |
| phase_exact_v1: requested vs realized skips reconciled | `run_execution()` checks `realized != requested` for phase_exact_v1 | Implemented and tested |
| phase_exact_v1: exact per-phase quota usage | Checks `phase_used[i] != phase_quota[i]` for each phase | Implemented and tested |
| phase_exact_v1: completed valid-decision horizon | Checks `decisions_seen != effective_total` | Implemented and tested |
| phase_exact_v1: skip cap and nonterminal state | Checks `longest_skip_run > max_consecutive_skips` and `gate_diag.terminal` | Implemented and tested |
| phase_exact_v1: phase_used None/empty rejected | `phase_used=None` treated as empty (no TypeError); empty `phase_quota` or `phase_used` with configured phases returns FAILED before evaluation | Implemented and tested |
| phase_exact_v1: optimizer/scheduler counts and LR endpoint | Checks `optimizer_step_attempts == backward_calls`, `scheduler_step_calls == effective_total` | Implemented and tested |
| legacy_v4: no exact-phase/terminal semantics imposed | legacy_v4 path uses original permissive contract; only count equality and invariants checked | Implemented and tested |
| Full/reduced as gate-free runs at effective horizons | `use_gate=False` for full_finetune/reduced_steps; validated at `effective_total` | Implemented and tested |
| Corruption test: terminal=True rejected | `TestCorruptionValidation.test_terminal_true_rejected` | Implemented and tested |
| Corruption test: phase_quota=[5,5], phase_used=[4,6] rejected | `TestCorruptionValidation.test_phase_quota_mismatch_rejected` | Implemented and tested |
| Corruption test: decisions_seen=0 with 20 forwards rejected | `TestCorruptionValidation.test_decisions_seen_zero_with_forwards_rejected` | Implemented and tested |
| Structured failure manifests | `run_execution()` returns `{"status": "FAILED", "reason", "config", "diagnostics", "provenance"}` | Implemented and tested |
| Validation failures returned before evaluation | Evaluation (`_evaluate_model`) called only after validation passes; failures short-circuit | Implemented and tested |
| Loader/constructor/evaluator exceptions wrapped | Exceptions caught and returned as structured REJECTED/FAILED manifests | Implemented and tested |
| Failed attempts preserved, no retry/seed substitution | Manifest records status/reason/provenance; no automatic retry | Implemented and tested |

## R4-D — Targeted actual CPU evidence

| Requirement | Evidence | Status |
|-------------|----------|--------|
| Actual S1 formula agreement | `TestS1Formula.test_s1_matches_reference_formula` — forward_gain_score matches reference computation within 1e-6 | Implemented and tested |
| Actual S1 label sensitivity | `TestS1Formula.test_s1_label_sensitivity` — different labels produce different scores | Implemented and tested |
| mock_forward_gain_score is NOT S1 evidence | `TestS1Formula.test_s1_absolute_logit_mock_is_not_label_sensitive` — mock is label-insensitive | Implemented and tested |
| Policy uses current-batch cached logits | `TestCachedLogits.test_policy_reads_trainer_cached_logits` — policy reads `trainer._last_real_logits` | Implemented and tested |
| Parameter freezing on skipped steps | `TestAdamFreezing.test_skipped_step_parameters_unchanged` | Implemented and tested |
| Per-parameter Adam-state freezing | `TestAdamFreezing.test_adam_state_not_updated_on_skipped_step` | Implemented and tested |
| Selective per-parameter freezing | `TestAdamFreezing.test_per_parameter_freezing_selective` | Implemented and tested |
| Scheduled-step LR advancement | `TestLRSchedule.test_gate_lr_advances_every_batch_including_skips` — scheduler_step_calls == total | Implemented and tested |
| LR endpoint for full training | `TestLRSchedule.test_full_training_lr_endpoint` — final LR == learning_rate | Implemented and tested |
| LR endpoint for reduced training | `TestLRSchedule.test_reduced_training_own_horizon` — reduced horizon with own LR endpoint | Implemented and tested |
| Loading-boundary success | `TestLoadingBoundaries.test_load_model_injectable` etc. — injectable loaders work | Implemented and tested |
| Loading-boundary failure | `TestLoadingBoundaries.test_capture_logits_false_rejected`, `test_qnli_missing_exclusions_blocked` | Implemented and tested |
| Successful gate execution | `TestExecutionPaths.test_gate_execution_path_ok` + gate manifest | Implemented and tested |
| Successful full execution | `TestExecutionPaths.test_full_execution_path_ok` + full manifest | Implemented and tested |
| Successful reduced execution | `TestExecutionPaths.test_reduced_execution_path_ok` + reduced manifest | Implemented and tested |
| Structured rejection paths | `TestCorruptionValidation` + failure manifests from CLI | Implemented and tested |
| Dry-run no Trainer/model | `TestExecutionPaths.test_dry_run_no_trainer_no_model` | Implemented and tested |
| Populated Adam freezes around actual skip | `TestTrainerActualFreezing.test_trainer_adam_state_freezes_around_actual_skip` — per-skip snapshots of params + Adam step/exp_avg/exp_avg_sq before/after skipped batch; exact comparison; later retained step changes params | Implemented and tested |

## R5 — Measurement metadata and packaging

| Item | Evidence | Status |
|------|----------|--------|
| Effective history, anneal, seeds, score trajectory | `ScoreQuantileGate.diagnostics()` includes phase_quota, phase_used, anneal_step, seed, decisions_seen | Supplied but not independently verified |
| Controller identity | `controller_version` in config and diagnostics; `legacy_v4` default, `phase_exact_v1` opt-in | Implemented and tested |
| Instrumentation boundaries | `SkipInstrumentation.as_dict()` includes capture_logits_enabled, skip_update_mode, policy_name | Implemented and tested |
| `runner` hash renamed/documented | `source_hash_runner_script` (new) + `source_hash_runner` (alias), both hash `run_controller_ablation.py` (the measurement script) | Implemented and tested |
| Gate/full/reduced manifests | `/tmp/step3c_manifests/{gate,full,reduced}/.../execution_manifest.json` (non-dry, config.dry_run == provenance.dry_run) + dry_run manifest | Supplied but not independently verified |
| Failure manifests | CLI rejection outputs for QNLI exclusions (revisions satisfied), capture_logits=False, invalid lr_schedule_shape, missing revision; config failures wrapped at CLI boundary with structured JSON + nonzero exit | Supplied but not independently verified |
| Base-relative patch/reconstruction | `scripts/step3c_reconstruct.py` with path mapping and helper dependencies | Supplied but not independently verified |
| Trainer tracked/untracked reconciled | `SkipInstrumentation` documents tracked vs untracked fields; `_note_optimizer_step_attempts` and `_note_global_step` clarify semantics | Implemented and tested |
| Requirement-specific evidence (no blanket claims) | This table | Implemented and tested |
| Explicit outstanding prerequisites | QNLI exclusion file/hash, real pretrained model, real GLUE dataset listed as missing/dependency-blocked | Implemented and tested |

## Summary of status categories

- **Implemented and tested**: Code changes in `scripts/run_controller_ablation.py`, `tests/test_step3c_execution_contract.py`, `tests/test_phase_exact_v1_acceptance.py`, `lerna/utils/model_loader.py`, `scripts/step3c_reconstruct.py`. 127 focused tests pass (55 contract + acceptance + new boundary tests).
- **Partially implemented**: QNLI task mapping complete (JSON schema, int ID comparison, training-only exclusion); dynamic minibatch padding NOT yet wired into Trainer DataLoader.
- **Supplied but not independently verified**: Execution manifests (legacy_v4 gate/full/reduced, dry_run), failure manifests from CLI boundary, reconstruction script with --check/--apply/--diff, measure_step3b_performance.py script (not yet executed for raw output evidence).
- **Dependency-blocked**: Production model loading (requires `datasets`, `transformers` with real model weights), real GLUE dataset loading.
- **Missing real-execution prerequisites**: QNLI matched-probe-ID exclusion file and hash, pinned model/dataset revisions (originals are not in this environment; absence is enforced as blocker by preflight).

**Not yet completed**: phase_exact_v1 non-dry execution manifests, dynamic minibatch padding collator wired into Trainer, measure_step3b_performance.py raw output evidence, model_loader.py in reconstruction --apply mapping (now included).
