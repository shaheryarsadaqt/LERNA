# Waste Quantification Bug Resolution Log

## Problem Statement

**Issue**: The waste ratio was showing as `0` (or `[0, 0]` in CI) even when training clearly reached a plateau. This happened for runs with 3 epochs and 10 epochs.

**Symptoms**:
- SST-2 10-epoch run (1960 steps): `waste_ratio = 0`, `CI = [0, 0]`
- SST-2 3-epoch run (588 steps): `waste_ratio = 0`, `CI = [0, 0]`
- Zero samples passed the plateau check

---

## Root Cause Analysis

### Bug #1: Unit Mismatch Between `record_step` Calls and Plateau Thresholds

**Location**: `scripts/run_baseline_glue.py` - `_LERNADiagnosticCallback.on_step_end`

**The Problem**:
The `_last_loss` variable was only updated inside `on_log` (line 1883), which fires every `logging_steps` interval. For a 1960-step run with `logging_steps = max(eval_steps // 5, 1) = max(98//5, 1) = 19`, the loss was only updated 19 steps apart.

```python
# OLD CODE (before fix)
def on_step_end(self, args, state, control, model=None, **kwargs):
    # ... code ...
    
    if self._last_loss is not None:  # DEDUP: _last_loss only set in on_log
        self.waste_quantifier.record_step(
            loss=self._last_loss,
            grad_norm=global_grad_norm,
            step_time=step_time,
        )
```

**Result**: `record_step` fired only ~103 times (1960 / 19) instead of 1960 times.

**The Plateau Gate** (`scripts/run_baseline_glue.py:518-521`):
```python
if (self._plateau_step is None
        and self._steps_since_ema_improvement >= self.plateau_patience  # 50
        and self._total_steps_seen >= self.min_steps_before_plateau):   # 100
    self._plateau_step = self._total_steps_seen - self.plateau_patience
```

The `_total_steps_seen` was in post-dedup units (max ~103) but the threshold `min_steps_before_plateau=100` was set assuming training-step units (max 1960). **The gate practically never opened** — by the time `_total_steps_seen` cleared 100, training was already done.

---

## Fix #1: Feed Every Training Step with Live Loss

**Solution**: Use `_last_real_loss` set by `CapturingTrainer.compute_loss` on every training step, and remove the dedup.

### Changes Made:

**File**: `scripts/run_baseline_glue.py`

1. **Removed dedup logic** from `on_step_end`:
```python
# BEFORE: _last_loss was only updated in on_log (stale between log events)
# AFTER: Read fresh per-step loss directly from trainer
def on_step_end(self, args, state, control, model=None, **kwargs):
    trainer = self._trainer if self._trainer is not None else (self._trainer_holder[0] if self._trainer_holder else None)
    fresh_loss = getattr(trainer, "_last_real_loss", None) if trainer is not None else None

    if fresh_loss is None:
        return  # Can't proceed without fresh loss

    self.waste_quantifier.record_step(
        loss=fresh_loss,
        grad_norm=global_grad_norm,
        step_time=step_time,
    )
    # ... rest of code ...
```

2. **Removed DEBUG prints** that were slowing down training:
   - Removed `[DBG step_end] step=...` print
   - Removed `[DBG compute_loss] set _last_real_loss=...` print
   - Kept only the `WARNING: fresh_loss=None` print

### Result After Fix #1:
- `_total_steps_seen` ≈ 1960 at end (matches total_steps)
- `loss_history` length ≈ 1960

---

## Bug #2: Wrong Loss Type for Plateau Detection

**Issue**: Train-loss plateau ≠ compute waste. Eval-loss plateau = compute waste.

The train loss can keep decreasing even when the model has stopped improving on the held-out evaluation set. This is because:
1. Train loss includes the training data the model has memorized
2. Eval loss reflects true generalization

**Solution**: Add eval-loss based plateau detection.

### Changes Made:

**File**: `lerna/utils/metrics.py` - `WasteQuantifier` class

1. **Added new fields**:
```python
# Eval-loss based plateau detection (primary - more accurate)
self.eval_loss_history = []
self._best_eval_loss = None
self._evals_since_improvement = 0
self._plateau_eval_step = None
self.plateau_eval_patience = plateau_eval_patience  # 3 evals
self.plateau_eval_min_improvement = plateau_eval_min_improvement  # 0.5%
```

2. **Added `record_eval()` method**:
```python
def record_eval(self, eval_loss, current_step):
    """Record eval loss and detect plateau based on eval performance."""
    self.eval_loss_history.append((current_step, eval_loss))
    
    if self._best_eval_loss is None:
        self._best_eval_loss = eval_loss
        self._evals_since_improvement = 0
    elif eval_loss < self._best_eval_loss * (1 - self.plateau_eval_min_improvement):
        self._best_eval_loss = eval_loss
        self._evals_since_improvement = 0
    else:
        self._evals_since_improvement += 1
    
    if (self._plateau_eval_step is None
            and self._evals_since_improvement >= self.plateau_eval_patience):
        idx = len(self.eval_loss_history) - self.plateau_eval_patience
        if idx >= 0:
            self._plateau_eval_step = self.eval_loss_history[idx][0]
```

3. **Updated `compute_waste_metrics()` to prefer eval plateau**:
```python
# Prefer eval-loss plateau, fallback to train-loss
plateau_step = (self._plateau_eval_step if self._plateau_eval_step is not None 
               else self._plateau_step)
```

**File**: `scripts/run_baseline_glue.py` - `on_evaluate` callback

4. **Called `record_eval()` in `on_evaluate`**:
```python
def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
    if metrics is None:
        return control

    eval_loss = metrics.get("eval_loss")
    accuracy = metrics.get("eval_accuracy", metrics.get("eval_matthews_correlation", 0))

    # Record eval loss for plateau detection (more accurate than train loss)
    if eval_loss is not None:
        self.waste_quantifier.record_eval(float(eval_loss), state.global_step)

    # ... rest of code ...
```

**Bug #2.1**: Initial condition was `if eval_loss > 0:`, which filtered out valid 0.0 eval losses.
- **Fix**: Changed to `if eval_loss is not None:`

**Bug #2.2**: `record_eval()` was called AFTER the `_train_ended` check, which was blocking training evals.
- **Fix**: Moved `record_eval()` call BEFORE the `_train_ended` check.

---

## Additional Fixes Applied

### 3. Added Comprehensive Utilities

**New File**: `lerna/utils/checkpoint_compat.py`
- Handles legacy LayerNorm key naming (gamma/beta → weight/bias)
- Safe state_dict loading with critical tensor verification

**New File**: `lerna/utils/sanity.py`
- `assert_layernorm_trained()` - Verifies LayerNorm params look trained after `load_best_model_at_end`
- Detects randomly initialized parameters that indicate corrupted loads

### 4. Fixed metrics.py Issues

**File**: `lerna/utils/metrics.py`

1. **L2-norm velocity with fp64 precision**:
```python
def _compute_param_velocity(self, model):
    """L2 norm of parameter update vector. Uses fp64 to avoid underflow."""
    delta = param.data.detach().double() - self._prev_params[name].double()
    total_delta_sq += delta.pow(2).sum().item()
    velocity = total_delta_sq ** 0.5  # Not divided by sqrt(numel)
```

2. **Recalibrated ler_threshold values** for new L2-norm velocity:
```python
"sst2": {"ler_threshold": 1e2,   "entropy_weight": 1.0},  # Was 0.01
"qnli": {"ler_threshold": 8e1,   "entropy_weight": 1.1},
# ... etc
```

3. **Lower min_phase_duration** for faster phase transitions:
```python
min_phase_duration: int = 3  # Was 20
```

4. **Removed DEBUG prints** that were slowing training

### 5. Fixed simple_baselines.py Issues

**File**: `lerna/callbacks/simple_baselines.py`

1. **Handle non-accuracy metrics** (CoLA uses matthews_correlation, STS-B uses pearson):
```python
metric_value = (
    metrics.get("eval_accuracy")
    or metrics.get("eval_matthews_correlation")
    or metrics.get("eval_pearson")
    or metrics.get("eval_pearsonr")
)
```

2. **Use task-calibrated threshold** from LERTracker:
```python
task_threshold = self.ler_tracker.task_calibration.get(
    self.ler_tracker.task, {}
).get("ler_threshold", self.threshold)

if current_ler < task_threshold:  # Use task_threshold, not self.threshold
```

3. **Fixed SGDR T_i growth** for cosine annealing:
```python
T_i = max(int(T_i * self.T_mult), T_i + 1)  # Guarantee growth even if T_mult==1
```

4. **Deferred energy measurement** to on_step_end for accurate GPU power sampling

### 6. Fixed run_phase1_2_simple_baselines.py Issues

**File**: `scripts/run_phase1_2_simple_baselines.py`

1. **Rewrote skip path** for true backward-pass skipping with grad accumulation support
2. **Added on_step_end LER sampling** (every 20 steps instead of only at eval)
3. **Removed dummy logits fallback** - now fails gracefully if no logits available
4. **Removed duplicate callback registration** (already added in Phase12Trainer.__init__)

---

## Final Results

### Before Fixes:
```
SST-2 10-epoch run:
  Waste ratio: 0 (95% CI: [0, 0])
  GSNR: 0.0277
  Total steps seen: ~103 (of 1960)
```

### After Fixes:
```
SST-2 10-epoch run:
  Waste ratio: 0.602 (95% CI: [0.580, 0.623])
  GSNR: 0.0277
  Total steps seen: 1960
  Plateau step: 780 (from eval-loss detection)
  Wasted steps: 1180
  Phase: fine_tuning (5 transitions)
```

---

## Key Takeaways

1. **Unit mismatch bugs are subtle** - The `_last_loss` dedup looked correct but had a 19x impact on the step counter
2. **Train loss is not a good proxy for generalization** - Eval loss plateau is the correct signal for compute waste
3. **DEBUG prints hurt performance** - Removed ~10k log lines per run
4. **Floating point precision matters** - Switched to fp64 for velocity computation to avoid underflow
5. **Task-specific calibration is essential** - Threshold values needed recalibration after changing velocity metric

---

## Files Modified

| File | Changes |
|------|---------|
| `lerna/utils/metrics.py` | L2-velocity, fp64, recalibrated thresholds, eval plateau detection |
| `lerna/utils/checkpoint_compat.py` | New - legacy key remapping |
| `lerna/utils/sanity.py` | New - LayerNorm sanity checks |
| `lerna/callbacks/simple_baselines.py` | Multi-metric, task thresholds, energy deferral |
| `scripts/run_baseline_glue.py` | Live loss feed, eval plateau, sanity check, safe_load |
| `scripts/run_phase1_2_simple_baselines.py` | Skip path rewrite, on_step_end LER, safe_load |

---

## Git Commits

| Commit | Description |
|--------|-------------|
| `4574780` | Add eval-loss plateau detection |
| `66fb8d6` | Fix syntax error in Wilson score interval |
| `6d73ef1` | Fix record_eval condition (not None check) |
| `35d07e2` | Call record_eval before _train_ended check |
| `2f8d4c9` | Comprehensive fixes (checkpoints, velocity, thresholds, etc.) |