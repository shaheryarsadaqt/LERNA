# V100 (fp16/AMP) Smoke Tests for TrueBackwardSkippingTrainer

## Global step semantics [IMP-7]

HF `state.global_step` counts training-loop iterations (batches/opportunities),
NOT real optimizer updates. After enabling backward skipping, `global_step`
may advance on steps where no parameter update occurred. Real optimizer update
attempts are tracked separately as `optimizer_step_attempts` in instrumentation.

## Expectations on every skipped baseline run

  * skipped_backward_steps > 0
  * forward_calls == batches_seen
  * forward_calls == backward_calls + skipped_backward_steps
  * optimizer_step_attempts <= backward_calls  (AMP overflow may skip the real update) [IMP-4]
  * scheduler_step_calls <= optimizer_step_attempts  (policy A: skip_on_backward_skip) [CRIT-1]
  * grad_scaler_step_calls <= optimizer_step_attempts
  * On a fully-skipped run (rate=1.0): scheduler_step_calls == 0 [CRIT-1]
  * compute_saving_mechanism set correctly:
        grad_norm / random_skip / weight_freeze -> "backward_skipping"
        reduced_steps                            -> "reduced_total_steps"
        early_stop_p*                            -> "early_stopping"
  * NO "No inf checks were recorded for this optimizer." crash

## Phase 1.2 smoke
```bash
python scripts/run_phase1_2_simple_baselines.py --baselines grad_norm     --tasks sst2 --seeds 43 --max-samples 2000
python scripts/run_phase1_2_simple_baselines.py --baselines random_skip   --tasks sst2 --seeds 43 --max-samples 2000
python scripts/run_phase1_2_simple_baselines.py --baselines weight_freeze --tasks sst2 --seeds 43 --max-samples 2000
python scripts/run_phase1_2_simple_baselines.py --baselines reduced_steps --tasks sst2 --seeds 43 --max-samples 2000   # mechanism=reduced_total_steps
python scripts/run_phase1_2_simple_baselines.py --baselines early_stop_p3 --tasks sst2 --seeds 43 --max-samples 2000   # mechanism=early_stopping
```