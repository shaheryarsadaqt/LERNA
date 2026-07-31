#!/usr/bin/env python3
"""
LERNA Ablation Study: Disabling individual LERNA components to quantify their contribution.

Ablation configs:
    full_lerna       — control (all components enabled)
    no_rho_vg        — disable rho_VG velocity-gradient correlation
    no_ler           — disable LER tracking (plateau detection falls back)
    no_safety        — disable safety horizon (unbounded momentum extrapolation)
    no_hysteresis   — disable phase-detection hysteresis
    no_momentum     — disable momentum extrapolation (full backprop always)

Usage:
    python scripts/run_ablation_study.py --mode smoke
    python scripts/run_ablation_study.py --mode full --tasks sst2 qnli --seeds 42 43 44
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

import sys
import json
import time
import argparse
import gc
import math
from datetime import datetime, timedelta
from pathlib import Path

import torch
try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
from datasets import load_dataset
import evaluate
import numpy as np

from lerna.callbacks.efficiency_callback import PowerTelemetryCallback
from lerna.callbacks.ler_feed import LERFeedCallback
from lerna.utils.lagged_ler import SampledLaggedLERTracker
from lerna.utils.metrics import LERTracker
from lerna.trainers import (
    LERNAMomentumTrainer, ComputeSavingMechanism, LERNAPolicy,
    LERNACalibratedPolicy, LERNAHybridPolicy, LERNAQuotaHybridPolicy,
    LERNAGuardedStochasticPolicy, LERNAPhaseStratifiedPolicy,
    PhaseStratifiedGuardedRandomPolicy, FixedPhaseStratifiedRandomPolicy,
    LERNARandomVetoDeferralPolicy,
    AlwaysFalsePolicy, RandomSkipPolicy, GradNormSkipPolicy,
    SchedulerStepPolicy, normalize_skip_update_mode,
)
from lerna.trainers.policies import build_exact_random_skip_set
from lerna.trainers.true_skip_trainer import (
    ONLINE_LER_MODE_OFF,
    ONLINE_LER_MODE_LEGACY_DENSE,
    ONLINE_LER_MODE_SAMPLED_LAGGED,
    VALID_ONLINE_LER_MODES,
    ONLINE_LER_TIMING_NONE,
    ONLINE_LER_TIMING_PRE_DECISION,
    ONLINE_LER_TIMING_POST_DECISION,
)
from lerna.utils.run_provenance import (
    CLASSIFICATION_LOCAL_DEVELOPMENT,
    CLASSIFICATION_MATCHED_CLAIM,
    build_identity_inputs,
    build_scientific_fingerprint,
    finalize_manifest_completed,
    finalize_manifest_failed,
    write_manifest_running,
)
from transformers import TrainerCallback

try:
    from scripts.validate_skip_policy_results import (
        validate_results as validate_skip_results,
    )
except ModuleNotFoundError:
    from validate_skip_policy_results import (
        validate_results as validate_skip_results,
    )

try:
    from scripts.run_baseline_glue import (
        MODEL_NAME,
        GLUE_TASK_CONFIG,
        TASK_HP_OVERRIDES,
        GLUE_TASKS,
        SEEDS,
        detect_device_profile,
        get_training_config,
        load_glue_task,
        build_compute_metrics,
        _ensure_wandb_finished,
        run_single_experiment,
    )
except ModuleNotFoundError:
    from run_baseline_glue import (
        MODEL_NAME,
        GLUE_TASK_CONFIG,
        TASK_HP_OVERRIDES,
        GLUE_TASKS,
        SEEDS,
        detect_device_profile,
        get_training_config,
        load_glue_task,
        build_compute_metrics,
        _ensure_wandb_finished,
        run_single_experiment,
    )

ABLATIONS = {
    "full_lerna":       {},
    "no_rho_vg":        {"use_rho_vg": False},
    "no_ler":           {"use_ler": False},
    "no_safety":        {"use_safety_horizon": False},
    "no_hysteresis":    {"use_hysteresis": False},
    "no_momentum":      {"use_momentum_extrap": False},
    "full_finetune":    {"control": "full_finetune"},
    "exact_random":     {"control": "exact_random"},
    "rvd":              {"control": "rvd"},
    "grad_norm":        {"control": "grad_norm"},
    # Compatibility alias for old invocations; excluded from default matrices.
    "random_skip":      {"control": "exact_random", "alias_of": "exact_random"},
}

POLICY_MIN_STEP = 50
SKIPPING_CONTROLS = {"exact_random", "random_skip", "rvd", "grad_norm"}
ONLINE_LER_MODE_AUTO = "auto"
ONLINE_LER_SIGNAL_FREE_CONTROLS = ("full_finetune", "exact_random")
ONLINE_LER_SIGNAL_FREE_POLICY = "fixed_phase_strat"
DEFAULT_ABLATIONS = [
    name for name, config in ABLATIONS.items() if "alias_of" not in config
]
ABLATION_GLUE_TASKS = [t for t in GLUE_TASKS if t != "rte_modernbert_2e5"]


def build_rvd_controller_config(
    *,
    veto_mode: str,
    margin_rank_floor: float,
    spike_factor: float,
    spike_ema_window: int,
    repay_mode: str,
    repay_protect_dangerous: bool,
    policy_seed,
    training_seed: int,
    max_consecutive_skips: int,
) -> dict:
    """Normalize the supported RVD controller modes without hidden vetoes."""
    if veto_mode not in ("none", "margin", "loss_spike"):
        raise ValueError(f"Unknown RVD veto mode: {veto_mode!r}")
    if repay_mode not in ("asap", "spread"):
        raise ValueError(f"Unknown RVD repay mode: {repay_mode!r}")
    margin_rank_floor = float(margin_rank_floor)
    spike_factor = float(spike_factor)
    spike_ema_window = int(spike_ema_window)
    max_consecutive_skips = int(max_consecutive_skips)
    if not 0.0 <= margin_rank_floor <= 1.0:
        raise ValueError("rvd_margin_rank_floor must be in [0, 1]")
    if not math.isfinite(spike_factor) or spike_factor < 0.0:
        raise ValueError("rvd_spike_factor must be finite and >= 0")
    if spike_ema_window < 1:
        raise ValueError("rvd_spike_ema_window must be >= 1")
    if max_consecutive_skips < 0:
        raise ValueError("max_consecutive_skips must be >= 0")

    return {
        "veto_mode": veto_mode,
        "use_margin_veto": veto_mode == "margin",
        "use_loss_spike_veto": veto_mode == "loss_spike",
        "use_rho_vg_veto": False,
        "use_grad_norm_veto": False,
        "use_novelty_veto": False,
        "use_phase_protection": False,
        "margin_rank_floor": margin_rank_floor,
        "spike_factor": spike_factor,
        "spike_ema_window": spike_ema_window,
        "repay_mode": repay_mode,
        "repay_protect_dangerous": bool(repay_protect_dangerous),
        "policy_seed": int(training_seed if policy_seed is None else policy_seed),
        "policy_seed_defaulted_to_training_seed": policy_seed is None,
        "max_consecutive_skips": max_consecutive_skips,
    }


def resolve_online_ler_config(
    requested_mode,
    *,
    effective_control,
    policy,
    parameter_sample_size,
    update_interval,
):
    """Resolve the canonical online LER diagnostics configuration for one arm."""
    if requested_mode == ONLINE_LER_MODE_AUTO:
        if effective_control in ONLINE_LER_SIGNAL_FREE_CONTROLS:
            mode = ONLINE_LER_MODE_OFF
            reason = f"auto_signal_free_control:{effective_control}"
        elif effective_control is None and policy == ONLINE_LER_SIGNAL_FREE_POLICY:
            mode = ONLINE_LER_MODE_OFF
            reason = f"auto_signal_free_policy:{ONLINE_LER_SIGNAL_FREE_POLICY}"
        else:
            mode = ONLINE_LER_MODE_SAMPLED_LAGGED
            reason = "auto_signal_consuming_arm"
    elif requested_mode in VALID_ONLINE_LER_MODES:
        mode = requested_mode
        reason = f"explicit:{requested_mode}"
    else:
        raise ValueError(
            f"Invalid requested_mode={requested_mode!r} for online_ler_mode; "
            f"expected {ONLINE_LER_MODE_AUTO!r} or one of {VALID_ONLINE_LER_MODES}."
        )

    enabled = mode != ONLINE_LER_MODE_OFF
    if enabled:
        resolved_interval = int(update_interval)
        if resolved_interval < 1:
            raise ValueError(
                "update_interval must be >= 1 when online LER diagnostics "
                f"are enabled; got {update_interval!r}"
            )
    else:
        resolved_interval = 0

    if mode == ONLINE_LER_MODE_SAMPLED_LAGGED:
        resolved_sample_size = int(parameter_sample_size)
        if resolved_sample_size < 1:
            raise ValueError(
                "parameter_sample_size must be >= 1 for "
                f"{ONLINE_LER_MODE_SAMPLED_LAGGED!r} mode; "
                f"got {parameter_sample_size!r}"
            )
    else:
        resolved_sample_size = 0

    timing = {
        ONLINE_LER_MODE_OFF: ONLINE_LER_TIMING_NONE,
        ONLINE_LER_MODE_LEGACY_DENSE: ONLINE_LER_TIMING_PRE_DECISION,
        ONLINE_LER_MODE_SAMPLED_LAGGED: ONLINE_LER_TIMING_POST_DECISION,
    }[mode]

    return {
        "requested_mode": requested_mode,
        "mode": mode,
        "enabled": enabled,
        "timing": timing,
        "parameter_sample_size": resolved_sample_size,
        "update_interval": resolved_interval,
        "reason": reason,
    }


def build_online_ler_provenance_config(
    resolved_config,
    *,
    sample_seed,
):
    """Extend a resolved online LER config with its provenance sample seed."""
    provenance = dict(resolved_config)
    if provenance["mode"] == ONLINE_LER_MODE_SAMPLED_LAGGED:
        provenance["sample_seed"] = int(sample_seed)
    else:
        provenance["sample_seed"] = None
    return provenance


def add_online_ler_to_identity(
    identity_inputs,
    online_diagnostics,
):
    """Return new identity inputs extended with the online diagnostics config."""
    extended = dict(identity_inputs)
    extended["online_diagnostics"] = dict(online_diagnostics)
    return extended


def build_online_ler_tracker(
    online_diagnostics,
    *,
    task_name,
    use_hysteresis,
    sample_seed,
):
    """Construct the runtime tracker for one resolved online LER config."""
    mode = online_diagnostics["mode"]
    if mode == ONLINE_LER_MODE_OFF:
        return None
    if mode == ONLINE_LER_MODE_LEGACY_DENSE:
        return LERTracker(
            task=task_name,
            window_size=5,
            use_hysteresis=use_hysteresis,
        )
    if mode == ONLINE_LER_MODE_SAMPLED_LAGGED:
        return SampledLaggedLERTracker(
            task=task_name,
            window_size=5,
            parameter_sample_size=online_diagnostics["parameter_sample_size"],
            sample_seed=sample_seed,
        )
    raise ValueError(f"Unknown online LER mode: {mode!r}")


def build_online_ler_runtime_metadata(
    online_diagnostics,
    instrumentation,
    tracker_diagnostics=None,
):
    """Combine configured online LER metadata with realized runtime counters."""
    config = online_diagnostics or {}
    instr = instrumentation or {}
    tracker = tracker_diagnostics or {}
    metadata = {
        "requested_mode": config.get("requested_mode"),
        "mode": config.get("mode"),
        "enabled": config.get("enabled"),
        "timing": config.get("timing"),
        "parameter_sample_size": config.get("parameter_sample_size"),
        "update_interval": config.get("update_interval"),
        "reason": config.get("reason"),
        "sample_seed": config.get("sample_seed"),
    }

    if not config.get("enabled"):
        metadata.update(
            {
                "parameter_sample_size_realized": 0,
                "update_attempts": 0,
                "update_successes": 0,
                "last_update_decision": None,
                "observation_age_decisions": None,
                "n_updates": 0,
                "n_decisions": 0,
            }
        )
        return metadata

    update_attempts = instr.get("online_ler_update_attempts", 0)
    update_successes = instr.get("online_ler_update_successes", 0)
    last_update_decision = instr.get("online_ler_last_update_decision")
    realized_sample_size = tracker.get("parameter_sample_size_realized")
    if realized_sample_size is None:
        realized_sample_size = 0

    observation_age = tracker.get("observation_age_decisions")
    if observation_age is None and last_update_decision is not None:
        batches_seen = instr.get("batches_seen", 0)
        observation_age = max((batches_seen - 1) - last_update_decision, 0)

    n_updates = tracker.get("n_updates")
    if n_updates is None:
        n_updates = update_successes
    n_decisions = tracker.get("n_decisions")
    if n_decisions is None:
        n_decisions = (
            instr.get("batches_seen", 0)
            if config.get("mode") == ONLINE_LER_MODE_LEGACY_DENSE
            else 0
        )

    metadata.update(
        {
            "parameter_sample_size_realized": realized_sample_size,
            "update_attempts": update_attempts,
            "update_successes": update_successes,
            "last_update_decision": last_update_decision,
            "observation_age_decisions": observation_age,
            "n_updates": n_updates,
            "n_decisions": n_decisions,
        }
    )
    return metadata


def build_online_ler_artifact_contract(online_diagnostics):
    """Build the truthful artifact contract for one resolved online LER config."""
    output_paths = {
        "results": "results.json",
        "instrumentation": "instrumentation.json",
        "manifest": "run_manifest.json",
    }
    required_artifacts = ["instrumentation.json"]
    if online_diagnostics["enabled"]:
        output_paths["ler_diagnostics"] = "ler_diagnostics.json"
        required_artifacts.append("ler_diagnostics.json")
    return {
        "output_paths": output_paths,
        "required_artifacts": required_artifacts,
    }


def compute_authoritative_horizon(
    train_dataset,
    num_epochs: int,
    per_device_train_batch_size: int,
    gradient_accumulation_steps: int,
    n_gpu: int = 1,
) -> int:
    """Compute the authoritative training horizon matching HF Trainer 4.48.

    Must be called AFTER forcing single-GPU if multi-GPU is detected.
    Matches Trainer._inner_training_loop step calculation exactly:
      microbatches = ceil(dataset_size / (batch_size * gpu_count))
      updates_per_epoch = max(microbatches // gradient_accumulation_steps, 1)
      max_steps = ceil(num_epochs * updates_per_epoch)
    """
    if n_gpu < 1:
        n_gpu = 1
    if per_device_train_batch_size < 1 or gradient_accumulation_steps < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be >= 1")
    microbatch_size = per_device_train_batch_size * n_gpu
    microbatches = math.ceil(len(train_dataset) / microbatch_size)
    updates_per_epoch = max(microbatches // gradient_accumulation_steps, 1)
    return math.ceil(num_epochs * updates_per_epoch)


def assert_fixed_budget(
    *,
    ablation_name: str,
    control,
    no_early_stopping: bool,
    allow_early_stopping_with_skipping: bool,
) -> dict:
    """Reject early stopping for skipping arms and report budget provenance."""
    normalized_control = "exact_random" if control == "random_skip" else control
    is_skipping_arm = (
        normalized_control in SKIPPING_CONTROLS
        or normalized_control is None
    )
    early_stopping_active = not no_early_stopping
    if (
        is_skipping_arm
        and early_stopping_active
        and not allow_early_stopping_with_skipping
    ):
        raise RuntimeError(
            f"Fixed-budget violation: arm {ablation_name!r} skips backward "
            "steps while early stopping is active. Use --no-early-stopping "
            "for controller comparisons. The explicit "
            "--allow-early-stopping-with-skipping override creates an "
            "unmatched exploratory run."
        )
    return {
        "is_skipping_arm": is_skipping_arm,
        "early_stopping_active": early_stopping_active,
        # Any early-stopped arm is not a fixed-horizon matched-budget run.
        "matched_budget": not early_stopping_active,
    }


def build_skip_policy(
    *,
    control: str,
    ler_tracker,
    target_skip_rate: float,
    total_steps: int,
    controller_cfg: dict,
    rho_veto_threshold: float,
    probe_interval: int,
    use_ler: bool,
    use_rho_vg: bool,
    use_safety_horizon: bool,
    fallback_threshold: float,
    risk_gamma: float,
):
    """Construct one explicit baseline/RVD control arm."""
    control = "exact_random" if control == "random_skip" else control
    if control == "full_finetune":
        return AlwaysFalsePolicy()
    if control == "exact_random":
        return RandomSkipPolicy(
            target_skip_rate=target_skip_rate,
            min_step=POLICY_MIN_STEP,
            seed=controller_cfg["policy_seed"],
            total_steps=total_steps,
        )
    if control == "grad_norm":
        return GradNormSkipPolicy(
            target_skip_rate=target_skip_rate,
            min_step=POLICY_MIN_STEP,
            calibration_steps=60,
            recalibrate_every=200,
            max_consecutive_skips=controller_cfg["max_consecutive_skips"],
        )
    if control == "rvd":
        return LERNARandomVetoDeferralPolicy(
            ler_tracker=ler_tracker,
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            min_step=POLICY_MIN_STEP,
            seed=controller_cfg["policy_seed"],
            use_loss_spike_veto=controller_cfg["use_loss_spike_veto"],
            spike_factor=controller_cfg["spike_factor"],
            spike_ema_window=controller_cfg["spike_ema_window"],
            use_rho_vg_veto=False,
            use_grad_norm_veto=False,
            use_margin_veto=controller_cfg["use_margin_veto"],
            margin_rank_floor=controller_cfg["margin_rank_floor"],
            use_novelty_veto=False,
            use_phase_protection=False,
            repay_mode=controller_cfg["repay_mode"],
            repay_protect_dangerous=controller_cfg["repay_protect_dangerous"],
            rho_veto_threshold=rho_veto_threshold,
            max_consecutive_skips=controller_cfg["max_consecutive_skips"],
            probe_interval=probe_interval,
            use_ler=use_ler,
            use_rho_vg=use_rho_vg,
            use_safety_horizon=use_safety_horizon,
            fallback_threshold=fallback_threshold,
            calibration_steps=60,
            recalibrate_every=200,
            risk_gamma=risk_gamma,
        )
    raise ValueError(f"Unsupported explicit control arm: {control!r}")


class _GradNormCapture(TrainerCallback):
    """Feed pre-clip grad norm to policies exposing record_grad_norm()."""
    def __init__(self, policy):
        self._policy = policy

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(self._policy, "record_grad_norm"):
            return control
        sq = 0.0
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                sq += float(p.grad.detach().float().norm().item()) ** 2
        if sq > 0:
            self._policy.record_grad_norm(sq ** 0.5)
        return control


class AblationTrainer(LERNAMomentumTrainer):
    """Momentum trainer with an optional online LER tracker."""

    def __init__(self, *args, ler_tracker=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._ler_tracker = ler_tracker


class AblationDiagnosticsCallback:
    def __init__(
        self,
        ler_trk,
        model_ref,
        trainer_ref_holder,
        greater_is_better,
        use_rho_vg,
        use_ler,
        use_hysteresis,
        use_safety_horizon,
        skip_update_mode,
        skip_update_mode_legacy_compat_used,
        ablation_name,
        ablation_overrides,
        output_dir,
        use_wandb,
        task_cfg,
        eval_ds,
        tokenizer,
        online_diagnostics,
    ):
        self.ler_tracker = ler_trk
        self._model = model_ref
        self._trainer_holder = trainer_ref_holder
        self._greater_is_better = greater_is_better
        self.use_rho_vg = use_rho_vg
        self.use_ler = use_ler
        self.use_hysteresis = use_hysteresis
        self.use_safety_horizon = use_safety_horizon
        self.skip_update_mode = skip_update_mode
        self.skip_update_mode_legacy_compat_used = skip_update_mode_legacy_compat_used
        self.ablation_name = ablation_name
        self.ablation_overrides = ablation_overrides
        self.output_dir = output_dir
        self.use_wandb = use_wandb
        self._task_cfg = task_cfg
        self._eval_ds = eval_ds
        self._tokenizer = tokenizer
        self.online_diagnostics = dict(online_diagnostics)
        self._last_loss = None
        self._step_count = 0

    def on_init_end(self, args, state, control, **kwargs):
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        return control

    def on_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_step_end(self, args, state, control, **kwargs):
        return control

    def on_substep_end(self, args, state, control, **kwargs):
        return control

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_predict(self, args, state, control, **kwargs):
        return control

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        return control

    def on_prediction_step(self, args, state, control, **kwargs):
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        opt = kwargs.get("optimizer", None)
        if opt is not None and hasattr(self.ler_tracker, "set_optimizer"):
            self.ler_tracker.set_optimizer(opt)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._save_diagnostics()
        return control

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            self._last_loss = logs.get("loss", self._last_loss)
        return control

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None:
            return control

        accuracy = metrics.get(
            "eval_accuracy",
            metrics.get("eval_matthews_correlation",
                        metrics.get("eval_pearson", 0)),
        )
        eval_loss = metrics.get("eval_loss", 0)

        # [CLEAN CHANNEL] Do NOT call ler_tracker.update() here. Read-only log.
        diag = self.ler_tracker.get_diagnostics()
        ler_val = diag.get("ler")
        rho_val = diag.get("rho_vg")
        phase = diag.get("phase", "?")
        ler_str = f"{ler_val:.2e}" if ler_val is not None else "N/A"
        rho_str = f"{rho_val:.4f}" if rho_val is not None else "N/A"
        print(
            f"  [ABL step={state.global_step}] "
            f"LER={ler_str} | rho_VG={rho_str} | phase={phase} | "
            f"eval_loss={eval_loss:.4f} | acc={accuracy:.3f}"
        )

        if self.use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({
                        "lerna/ler": ler_val,
                        "lerna/rho_vg": rho_val,
                        "lerna/phase": phase,
                        "lerna/eval_accuracy": accuracy,
                        "lerna/eval_loss": eval_loss,
                        "ablation/ablation_name": self.ablation_name,
                    }, commit=False)
            except Exception:
                pass
        return control

    def _save_diagnostics(self):
        diag_path = os.path.join(self.output_dir, "ler_diagnostics.json")
        tracker_diagnostics = dict(self.ler_tracker.get_diagnostics())
        trainer = self._trainer_holder[0]
        instrumentation = (
            trainer.get_instrumentation() if trainer is not None else None
        ) or {}
        final = dict(tracker_diagnostics)
        final["ler_history"] = self.ler_tracker.ler_history
        final["rho_vg_history"] = self.ler_tracker.rho_vg_history
        final["velocity_history"] = self.ler_tracker.velocity_history
        final["ablation_name"] = self.ablation_name
        final["ablation_overrides"] = self.ablation_overrides
        final["skip_update_mode"] = self.skip_update_mode
        final["skip_update_mode_legacy_compat_used"] = self.skip_update_mode_legacy_compat_used
        final["online_diagnostics_runtime"] = build_online_ler_runtime_metadata(
            self.online_diagnostics,
            instrumentation,
            tracker_diagnostics=tracker_diagnostics,
        )
        with open(diag_path, "w") as f:
            json.dump(final, f, indent=2, default=str)
        print(f"  LER diagnostics saved: {diag_path}")


def run_ablation_single(
    task_name: str,
    seed: int,
    ablation_name: str,
    ablation_overrides: dict,
    model_name: str,
    profile: str,
    base_output_dir: str,
    use_wandb: bool = False,
    max_samples_override=None,
    run_idx: int = 0,
    total_runs: int = 0,
    wandb_project: str = "lerna-ablation",
    wandb_group: str = None,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 5,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    init_from_mnli: bool = False,
    no_early_stopping: bool = False,
    target_skip_rate: float = 0.20,
    max_consecutive_skips: int = 4,
    probe_interval: int = 8,
    policy: str = "hybrid",
    rho_veto_threshold: float = -0.2,
    risk_gamma: float = 0.0,
    guard_mode: str = "on",
    skip_update_mode: str = None,
    scheduler_step_policy: str = SchedulerStepPolicy.ALWAYS_STEP,
    allow_early_stopping_with_skipping: bool = False,
    rvd_veto_mode: str = "none",
    rvd_margin_rank_floor: float = 0.20,
    rvd_spike_factor: float = 1.0,
    rvd_spike_ema_window: int = 20,
    rvd_repay_mode: str = "asap",
    rvd_repay_protect_dangerous: bool = True,
    rvd_policy_seed=None,
    provenance_classification: str = CLASSIFICATION_MATCHED_CLAIM,
    online_ler_mode=ONLINE_LER_MODE_AUTO,
    online_ler_parameter_sample_size=4096,
    online_ler_update_interval=1,
):
    """Run a single experiment with a specific ablation config."""

    control = ablation_overrides.get("control")
    effective_control = "exact_random" if control == "random_skip" else control
    budget_state = assert_fixed_budget(
        ablation_name=ablation_name,
        control=effective_control,
        no_early_stopping=no_early_stopping,
        allow_early_stopping_with_skipping=allow_early_stopping_with_skipping,
    )
    if not 0.0 <= float(target_skip_rate) <= 1.0:
        raise ValueError(
            f"target_skip_rate must be in [0, 1], got {target_skip_rate!r}"
        )
    scheduler_step_policy = SchedulerStepPolicy.validate(
        scheduler_step_policy
    )
    controller_cfg = build_rvd_controller_config(
        veto_mode=rvd_veto_mode,
        margin_rank_floor=rvd_margin_rank_floor,
        spike_factor=rvd_spike_factor,
        spike_ema_window=rvd_spike_ema_window,
        repay_mode=rvd_repay_mode,
        repay_protect_dangerous=rvd_repay_protect_dangerous,
        policy_seed=rvd_policy_seed,
        training_seed=seed,
        max_consecutive_skips=max_consecutive_skips,
    )

    resolved_online_ler = resolve_online_ler_config(
        online_ler_mode,
        effective_control=effective_control,
        policy=policy,
        parameter_sample_size=online_ler_parameter_sample_size,
        update_interval=online_ler_update_interval,
    )
    online_diagnostics = build_online_ler_provenance_config(
        resolved_online_ler,
        sample_seed=seed,
    )

    task_hp = TASK_HP_OVERRIDES.get(task_name, {})
    lr = task_hp.get("learning_rate", 2e-5)
    num_epochs = task_hp.get("num_epochs", num_epochs)
    warmup_ratio = task_hp.get("warmup_ratio", warmup_ratio)
    early_stopping_patience = task_hp.get("early_stopping_patience", early_stopping_patience)
    metric_for_best_model = task_hp.get("metric_for_best_model", metric_for_best_model)
    greater_is_better = task_hp.get("greater_is_better", greater_is_better)
    init_from_mnli = task_hp.get("init_from_mnli", init_from_mnli)

    hw_cfg = get_training_config(profile)
    if max_samples_override is not None:
        hw_cfg["max_samples"] = max_samples_override

    # [Phase 1.3 Piece 1] Explicit skipped-step update mode.
    # Legacy 'use_momentum_extrap' overrides are supported ONLY as a logged
    # compatibility path; conflicts with an explicit CLI mode are rejected.
    legacy_momentum_flag = ablation_overrides.get("use_momentum_extrap", None)
    effective_skip_update_mode, skip_mode_legacy_compat_used = (
        normalize_skip_update_mode(
            explicit_mode=skip_update_mode,
            legacy_use_momentum_extrap=legacy_momentum_flag,
        )
    )
    if skip_mode_legacy_compat_used:
        print(
            f"  [compat] legacy ablation override "
            f"'use_momentum_extrap={legacy_momentum_flag}' normalized to "
            f"skip_update_mode='{effective_skip_update_mode}'"
        )

    if wandb_group is None:
        wandb_group = f"ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    print(f"\n{'='*60}")
    print(f"  Ablation [{ablation_name}]: {task_name} | seed={seed} | lr={lr}")
    print(f"  Overrides: {ablation_overrides}")
    print(f"  Skip-update mode: {effective_skip_update_mode}"
          + ("  (legacy use_momentum_extrap compat)" if skip_mode_legacy_compat_used else ""))
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from lerna.utils.model_loader import load_model_and_tokenizer
    cfg = GLUE_TASK_CONFIG[task_name]

    mnli_checkpoint_dir = os.path.join(base_output_dir, "mnli_finetuned")
    if init_from_mnli and os.path.exists(mnli_checkpoint_dir):
        from transformers import AutoConfig
        mnli_model, _ = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])
        model, _ = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])
        encoder_state = {k: v for k, v in mnli_model.state_dict().items()
                        if "classifier" not in k and "pooler" not in k}
        model.load_state_dict(encoder_state, strict=False)
        del mnli_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])

    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"])
    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    # Detect multi-GPU early so the authoritative horizon uses forced single-GPU.
    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_n_gpu = 1 if visible_gpus > 1 else max(1, visible_gpus)
    if visible_gpus > 1:
        print(f"  [Ablation] Multiple GPUs detected ({visible_gpus}); forcing single-GPU training.")

    total_steps = compute_authoritative_horizon(
        train_dataset=train_ds,
        num_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        n_gpu=effective_n_gpu,
    )
    eval_steps = max(total_steps // 20, 10)

    # [Piece 9] Deterministic scientific fingerprint for collision-proof identity.
    git_sha = "unknown"
    try:
        import subprocess
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
        ).decode().strip()
    except Exception:
        git_sha = "unknown"

    # [Piece 9B] Build canonical identity after dataset loading and horizon
    # calculation. This single dictionary is reused for fingerprint, manifest,
    # and results to avoid identity drift.
    max_samples_requested = (
        None if max_samples_override is None else int(max_samples_override)
    )
    identity_inputs = build_identity_inputs(
        task=task_name,
        training_seed=seed,
        model_id=model_name,
        max_samples_requested=max_samples_requested,
        train_samples_realized=len(train_ds),
        eval_samples_realized=len(eval_ds),
        train_dataset_fingerprint=getattr(train_ds, "_fingerprint", None),
        num_epochs=num_epochs,
        control=effective_control or policy,
        target_skip_rate=target_skip_rate,
        policy_seed=controller_cfg["policy_seed"],
        skip_update_mode=effective_skip_update_mode,
        scheduler_step_policy=scheduler_step_policy,
        no_early_stopping=no_early_stopping,
        rvd_veto_mode=rvd_veto_mode,
        rvd_margin_rank_floor=rvd_margin_rank_floor,
        rvd_spike_factor=rvd_spike_factor,
        rvd_spike_ema_window=rvd_spike_ema_window,
        rvd_repay_mode=rvd_repay_mode,
        rvd_repay_protect_dangerous=rvd_repay_protect_dangerous,
        max_consecutive_skips=max_consecutive_skips,
        total_steps=total_steps,
        git_sha=git_sha,
    )
    identity_inputs = add_online_ler_to_identity(
        identity_inputs,
        online_diagnostics,
    )
    fingerprint = build_scientific_fingerprint(identity_inputs)

    # Define run_id from task, seed, arm, and fingerprint.
    run_id = f"{task_name}_s{seed}_{ablation_name}_{fingerprint}"

    if use_wandb:
        import wandb
        _ensure_wandb_finished()
        wandb.init(
            project=wandb_project,
            name=run_id,
            group=wandb_group,
            job_type=f"ablation-{ablation_name}",
            tags=[task_name, f"ablation-{ablation_name}", f"seed-{seed}", model_name.split("/")[-1]],
            reinit=True,
            settings=wandb.Settings(init_timeout=120),
            config={
                "task": task_name,
                "seed": seed,
                "ablation": ablation_name,
                "ablation_overrides": ablation_overrides,
                "learning_rate": lr,
                "scheduler_step_policy": scheduler_step_policy,
                "model": MODEL_NAME,
                "profile": profile,
                "max_samples": hw_cfg["max_samples"],
                "run_index": run_idx,
                "total_runs": total_runs,
            },
        )

    # [Piece 9] Retry-safe directory layout:
    #   <base>/<ablation>/<fingerprint>/attempt-<N>/
    # Previous attempts are preserved; never deleted or overwritten.
    # Every attempt gets a fresh directory via atomic creation.
    arm_dir = os.path.join(base_output_dir, ablation_name, fingerprint)
    os.makedirs(arm_dir, exist_ok=True)
    attempt = 1
    while True:
        run_dir = os.path.join(arm_dir, f"attempt-{attempt:03d}")
        try:
            os.makedirs(run_dir, exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        break
    output_dir = run_dir

    print(f"\n{'='*60}")
    print(f"  Ablation [{ablation_name}]: {task_name} | seed={seed} | lr={lr}")
    print(f"  Overrides: {ablation_overrides}")
    print(f"  Skip-update mode: {effective_skip_update_mode}"
          + ("  (legacy use_momentum_extrap compat)" if skip_mode_legacy_compat_used else ""))
    print(f"  Profile: {profile} | Output: {output_dir}")
    print(f"{'='*60}")

    quota_control = effective_control
    if quota_control is None and policy == "random_veto_deferral":
        quota_control = "rvd"
    requested_quota = None
    if quota_control in ("exact_random", "rvd"):
        try:
            _, requested_quota = build_exact_random_skip_set(
                total_steps=total_steps,
                target_skip_rate=target_skip_rate,
                min_step=POLICY_MIN_STEP,
                seed=controller_cfg["policy_seed"],
            )
        except ValueError as exc:
            raise ValueError(
                f"Invalid exact quota for arm {ablation_name!r}: {exc}"
            ) from exc

    use_rho_vg = ablation_overrides.get("use_rho_vg", True)
    use_ler = ablation_overrides.get("use_ler", True)
    use_safety_horizon = ablation_overrides.get("use_safety_horizon", True)
    use_hysteresis = ablation_overrides.get("use_hysteresis", True)

    online_ler_enabled = online_diagnostics["enabled"]

    ler_tracker = build_online_ler_tracker(
        online_diagnostics,
        task_name=task_name,
        use_hysteresis=use_hysteresis,
        sample_seed=online_diagnostics["sample_seed"],
    )

    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
    )

    # Signal-consuming policies use their task calibration; signal-free arms
    # never read this compatibility fallback.
    task_cal = (
        getattr(ler_tracker, "task_calibration", {}).get(task_name, {})
        if ler_tracker is not None
        else {}
    )
    base_thr = task_cal.get("ler_threshold", 0.01)

    if effective_control is not None:
        skip_policy = build_skip_policy(
            control=effective_control,
            ler_tracker=ler_tracker,
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            controller_cfg=controller_cfg,
            rho_veto_threshold=rho_veto_threshold,
            probe_interval=probe_interval,
            use_ler=use_ler,
            use_rho_vg=use_rho_vg,
            use_safety_horizon=use_safety_horizon,
            fallback_threshold=base_thr,
            risk_gamma=risk_gamma,
        )
    else:
        if policy == "guarded_hybrid":
            skip_policy = LERNAGuardedStochasticPolicy(
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                min_step=50,
                seed=seed,
                max_consecutive_skips=max_consecutive_skips,
                probe_interval=probe_interval,
                rho_veto_threshold=rho_veto_threshold,
                use_ler=use_ler,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,   # [#3]
                risk_gamma=risk_gamma,                   # [#1] from func param
                guard_mode=guard_mode,                   # [Fix 8c] on=guarded, off=pure quota random
            )
        elif policy == "quota_hybrid":
            skip_policy = LERNAQuotaHybridPolicy(
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                fallback_threshold=base_thr,
                min_step=50,
                calibration_steps=60,
                recalibr_every=200,
                use_ler=use_ler,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                max_consecutive_skips=max_consecutive_skips,
                probe_interval=probe_interval,
                total_steps=total_steps,
                rho_veto_threshold=rho_veto_threshold,
            )
        elif policy == "phase_strat":
            skip_policy = LERNAPhaseStratifiedPolicy(
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                min_step=50,
                seed=seed,
                max_consecutive_skips=max_consecutive_skips,
                rho_veto_threshold=rho_veto_threshold,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                risk_gamma=risk_gamma,
            )
        elif policy == "phase_strat_guarded":
            # Explicit name for the guarded controller; identical behavior.
            skip_policy = PhaseStratifiedGuardedRandomPolicy(
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                min_step=50,
                seed=seed,
                max_consecutive_skips=max_consecutive_skips,
                rho_veto_threshold=rho_veto_threshold,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                risk_gamma=risk_gamma,
            )
        elif policy == "fixed_phase_strat":
            # Pure temporal baseline: no tracker, vetoes, or LERNA signals.
            skip_policy = FixedPhaseStratifiedRandomPolicy(
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                min_step=50,
                seed=seed,
            )
        elif policy == "random_veto_deferral":
            skip_policy = build_skip_policy(
                control="rvd",
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                controller_cfg=controller_cfg,
                rho_veto_threshold=rho_veto_threshold,
                probe_interval=probe_interval,
                use_ler=use_ler,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                fallback_threshold=base_thr,
                risk_gamma=risk_gamma,
            )
        else:
            PolicyCls = LERNAHybridPolicy if policy == "hybrid" else LERNACalibratedPolicy
            skip_policy = PolicyCls(
                ler_tracker=ler_tracker,
                target_skip_rate=target_skip_rate,
                fallback_threshold=base_thr,
                min_step=50,
                calibration_steps=60,
                recalibrate_every=200,
                use_ler=use_ler,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                max_consecutive_skips=max_consecutive_skips,
                probe_interval=probe_interval,
            )

    policy_effective_config = (
        skip_policy.effective_config()
        if hasattr(skip_policy, "effective_config")
        else {}
    )
    compute_saving_mechanism = (
        ComputeSavingMechanism.NONE
        if effective_control == "full_finetune"
        else ComputeSavingMechanism.BACKWARD_SKIPPING
    )
    controller_config_effective = {
        "arm": ablation_name,
        "arm_alias_of": ablation_overrides.get("alias_of"),
        "control": effective_control or policy,
        "policy_class": type(skip_policy).__name__,
        "compute_saving_mechanism": compute_saving_mechanism,
        "policy_seed": controller_cfg["policy_seed"],
        "target_skip_rate": target_skip_rate,
        "min_step": POLICY_MIN_STEP,
        "configured_total_steps": total_steps,
        "requested_quota": requested_quota,
        "matched_budget": budget_state["matched_budget"],
        "is_skipping_arm": budget_state["is_skipping_arm"],
        "allow_early_stopping_with_skipping": (
            allow_early_stopping_with_skipping
        ),
        "early_stopping_active": budget_state["early_stopping_active"],
        "num_epochs": num_epochs,
        "policy_effective_config": policy_effective_config,
        "online_diagnostics": online_diagnostics,
    }
    if quota_control == "rvd":
        controller_config_effective["rvd"] = dict(controller_cfg)
    print(
        "  Controller config: "
        + json.dumps(controller_config_effective, sort_keys=True, default=str)
    )

    trainer_holder = [None]
    ler_feed_callback = None
    diag_callback = None
    if online_ler_enabled:
        ler_feed_callback = LERFeedCallback(
            ler_tracker=ler_tracker,
            policy_ref=skip_policy,
        )
        diag_callback = AblationDiagnosticsCallback(
            ler_trk=ler_tracker,
            model_ref=model,
            trainer_ref_holder=trainer_holder,
            greater_is_better=greater_is_better,
            use_rho_vg=use_rho_vg,
            use_ler=use_ler,
            use_hysteresis=use_hysteresis,
            use_safety_horizon=use_safety_horizon,
            skip_update_mode=effective_skip_update_mode,
            skip_update_mode_legacy_compat_used=skip_mode_legacy_compat_used,
            ablation_name=ablation_name,
            ablation_overrides=ablation_overrides,
            output_dir=output_dir,
            use_wandb=use_wandb,
            task_cfg=task_cfg,
            eval_ds=eval_ds,
            tokenizer=tokenizer,
            online_diagnostics=online_diagnostics,
        )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=hw_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        fp16=hw_cfg["fp16"],
        bf16=hw_cfg["bf16"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=not no_early_stopping,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=max(eval_steps // 5, 1),
        report_to="wandb" if use_wandb else "none",
        run_name=run_id if use_wandb else None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
    )

    # Force single-GPU to avoid unstable NCCL/DataParallel on DGX multi-GPU
    if torch.cuda.is_available() and training_args._n_gpu > 1:
        training_args._n_gpu = 1

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics(task_name)

    callbacks = [power_callback]
    if online_ler_enabled:
        callbacks.extend([ler_feed_callback, diag_callback])
    if not no_early_stopping:
        callbacks.insert(
            0,
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
        )

    trainer = AblationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        ler_tracker=ler_tracker,
        skip_policy=skip_policy,
        skip_update_mode=effective_skip_update_mode,
        scheduler_step_policy=scheduler_step_policy,
        apply_momentum=legacy_momentum_flag,  # None when CLI path; preserves legacy provenance
        compute_saving_mechanism=compute_saving_mechanism,
        instrumentation_path=os.path.join(output_dir, "instrumentation.json"),
        capture_logits=online_diagnostics["enabled"],
        online_ler_mode=online_diagnostics["mode"],
        online_ler_enabled=online_diagnostics["enabled"],
        online_ler_update_interval=online_diagnostics["update_interval"],
        callbacks=callbacks,
    )
    if ler_feed_callback is not None:
        ler_feed_callback.attach(trainer=trainer)
    trainer_holder[0] = trainer

    # [Piece 8] Authoritative horizon validation is performed at runtime by
    # TrueBackwardSkippingTrainer._check_authoritative_horizon() during the
    # first training_step(), after HF has initialized trainer.state.max_steps.
    # The pre-train state is 0 in Transformers 4.48 and must not be used here.

    # Pre-clip grad norm is now fed from inside TrueBackwardSkippingTrainer.training_step
    # (single, correct source). The old _GradNormCapture read POST-clip grads (~1.0) and is removed.

    artifact_contract = build_online_ler_artifact_contract(online_diagnostics)
    output_paths = artifact_contract["output_paths"]

    write_manifest_running(
        output_dir,
        argv=list(sys.argv),
        task=task_name,
        model_id=model_name,
        seed=seed,
        controller_name=type(skip_policy).__name__,
        controller_seed=controller_cfg["policy_seed"],
        target_skip_rate=target_skip_rate,
        planned_quota=requested_quota,
        total_steps=total_steps,
        warmup_steps=training_args.get_warmup_steps(total_steps),
        skip_update_mode=effective_skip_update_mode,
        controller_config_effective=controller_config_effective,
        matched_budget_planned=budget_state["matched_budget"],
        budget_classification=(
            "fixed_epoch"
            if budget_state["matched_budget"]
            else "early_stopping_exploratory"
        ),
        output_paths=output_paths,
        requested_classification=provenance_classification,
        repo_root=str(Path(__file__).resolve().parents[1]),
        identity_inputs=identity_inputs,
        fingerprint=fingerprint,
        attempt=attempt,
    )

    try:
        start_time = time.time()
        print(
            f"\n  Starting ablation [{ablation_name}]: "
            f"{total_steps} steps, eval every {eval_steps}"
        )
        train_result = trainer.train()
        total_time = time.time() - start_time

        print(f"\n  Evaluating best model...")
        eval_result = trainer.evaluate()

        avg_power = (
            float(np.mean([s["power_w"] for s in power_callback._power_samples]))
            if power_callback._power_samples
            else 0
        )

        instrumentation = trainer.get_instrumentation()
        policy_diagnostics = (
            skip_policy.get_diagnostics()
            if hasattr(skip_policy, "get_diagnostics")
            else {}
        )
        if hasattr(skip_policy, "effective_config"):
            controller_config_effective["policy_effective_config"] = (
                skip_policy.effective_config()
            )
        runtime_quota = policy_diagnostics.get("quota_size")
        if runtime_quota is not None:
            controller_config_effective["requested_quota"] = runtime_quota
        controller_config_effective["runtime_quota_total_steps"] = (
            policy_diagnostics.get("quota_total_steps")
        )

        tracker_diagnostics = (
            ler_tracker.get_diagnostics() if ler_tracker is not None else None
        )
        ler_final = (
            tracker_diagnostics
            if tracker_diagnostics is not None
            else {"enabled": False, "mode": "off", "n_updates": 0}
        )
        online_diagnostics_runtime = build_online_ler_runtime_metadata(
            online_diagnostics,
            instrumentation,
            tracker_diagnostics=tracker_diagnostics,
        )

        results = {
            "task": task_name,
            "seed": seed,
            "ablation": ablation_name,
            "ablation_overrides": ablation_overrides,
            "learning_rate": lr,
            "profile": profile,
            "model": MODEL_NAME,
            "train_runtime_s": total_time,
            "train_loss": train_result.training_loss,
            "eval_metrics": eval_result,
            "energy_kwh": power_callback.cumulative_kwh,
            "power_avg_watts": avg_power,
            "ler_final": ler_final,
            "online_diagnostics": online_diagnostics_runtime,
            "true_skip_instrumentation": instrumentation,
            "policy_diagnostics": policy_diagnostics,
            "controller_config": controller_config_effective,
            "timestamp": datetime.now().isoformat(),
            "hw_config": {k: v for k, v in hw_cfg.items() if k != "max_samples"},
        }
        _instr = instrumentation or {}
        results["skip_ratio"] = _instr.get("skip_ratio_by_batch")
        results["backward_calls"] = _instr.get("backward_calls")
        results["skipped_backward_steps"] = _instr.get("skipped_backward_steps")
        results["forward_calls"] = _instr.get("forward_calls")
        results["policy_name"] = _instr.get("policy_name") or getattr(
            skip_policy, "name", None
        )
        results["skip_update_mode"] = effective_skip_update_mode
        results["skip_update_mode_legacy_compat_used"] = (
            skip_mode_legacy_compat_used
        )
        results["scheduler_step_policy"] = scheduler_step_policy
        results["fingerprint"] = fingerprint
        results["attempt"] = attempt
        results["identity_inputs"] = identity_inputs

        try:
            import subprocess

            git_sha = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__)
            ).decode().strip()
        except Exception:
            git_sha = "unknown"

        results["code_git_sha"] = git_sha
        results["run_config"] = {
            "policy": policy,
            "control": effective_control,
            "target_skip_rate": target_skip_rate,
            "max_consecutive_skips": max_consecutive_skips,
            "probe_interval": probe_interval,
            "guard_mode": guard_mode,
            "risk_gamma": risk_gamma,
            "no_early_stopping": no_early_stopping,
            "num_epochs": num_epochs,
            "skip_update_mode": effective_skip_update_mode,
            "scheduler_step_policy": scheduler_step_policy,
            "skip_update_mode_legacy_compat_used": skip_mode_legacy_compat_used,
            "controller_config": controller_config_effective,
            "allow_early_stopping_with_skipping": (
                allow_early_stopping_with_skipping
            ),
            "matched_budget": budget_state["matched_budget"],
            "rvd_veto_mode": rvd_veto_mode,
            "rvd_margin_rank_floor": rvd_margin_rank_floor,
            "rvd_spike_factor": rvd_spike_factor,
            "rvd_spike_ema_window": rvd_spike_ema_window,
            "rvd_repay_mode": rvd_repay_mode,
            "rvd_repay_protect_dangerous": rvd_repay_protect_dangerous,
            "rvd_policy_seed": controller_cfg["policy_seed"],
            "rvd_policy_seed_defaulted_to_training_seed": (
                controller_cfg["policy_seed_defaulted_to_training_seed"]
            ),
            "online_diagnostics": dict(online_diagnostics),
        }

        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        if use_wandb:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.summary.update({
                        "final/eval_accuracy": eval_result.get(
                            "eval_accuracy",
                            eval_result.get(
                                "eval_matthews_correlation",
                                eval_result.get("eval_pearsonr"),
                            ),
                        ),
                        "final/eval_loss": eval_result.get("eval_loss"),
                        "final/energy_kwh": power_callback.cumulative_kwh,
                        "final/runtime_s": total_time,
                        "final/ler": ler_final.get("ler"),
                        "final/rho_vg": ler_final.get("rho_vg"),
                        "final/steps_skipped": instrumentation[
                            "skipped_backward_steps"
                        ],
                        "final/skip_ratio": instrumentation["skip_ratio_by_batch"],
                        "ablation/overrides": ablation_overrides,
                    })
            except Exception:
                pass

        print(f"\n  Ablation [{ablation_name}] Results:")
        print(f"  Eval metrics: {eval_result}")
        print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
        print(f"  Time: {total_time:.1f}s")
        print(f"  Saved: {results_path}")

        del model, trainer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if use_wandb:
            _ensure_wandb_finished()

        validation_report = validate_skip_results(
            Path(results_path),
            required_artifacts=artifact_contract["required_artifacts"],
        )
        finalize_manifest_completed(
            output_dir,
            realized_skips=instrumentation.get("skipped_backward_steps"),
            realized_skip_rate=instrumentation.get("skip_ratio_by_batch"),
            validation_status=validation_report.to_dict(),
        )
        return results
    except BaseException as exc:
        try:
            finalize_manifest_failed(output_dir, exc)
        except Exception as provenance_exc:
            print(
                "  [provenance] Failed to write terminal failure manifest: "
                f"{type(provenance_exc).__name__}"
            )
        raise


def build_arg_parser():
    parser = argparse.ArgumentParser(description="LERNA Ablation Study")
    parser.add_argument("--mode", choices=["smoke", "full", "custom"], default="smoke")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Ablation names to run (default: all)")
    parser.add_argument("--output-dir", default="./experiments/ablation")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerna-ablation")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--model", default="modernbert", choices=["roberta", "modernbert", "deberta"],
                        help="Model to use for ablation study")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unlimited", action="store_true")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Run full fixed epochs so arms are compute-comparable")
    parser.add_argument(
        "--allow-early-stopping-with-skipping",
        action="store_true",
        help="Allow an unmatched exploratory skipping run with early stopping",
    )
    parser.add_argument(
        "--policy",
        choices=[
            "calibrated",
            "hybrid",
            "quota_hybrid",
            "guarded_hybrid",
            "phase_strat",
            "phase_strat_guarded",
            "fixed_phase_strat",
            "random_veto_deferral",
        ],
        default="hybrid",
    )
    parser.add_argument("--rho-veto-threshold", type=float, default=-0.2)
    parser.add_argument("--risk-gamma", type=float, default=0.0)
    parser.add_argument("--guard-mode", choices=["on", "off"], default="on",
                        help="on=full guarded stochastic LERNA; off=pure exact-quota random (debug parity check)")
    parser.add_argument("--target-skip-rate", type=float, default=0.20)
    parser.add_argument("--max-consecutive-skips", type=int, default=4)
    parser.add_argument("--probe-interval", type=int, default=8)
    parser.add_argument(
        "--rvd-veto-mode",
        choices=["none", "margin", "loss_spike"],
        default="none",
    )
    parser.add_argument("--rvd-margin-rank-floor", type=float, default=0.20)
    parser.add_argument("--rvd-spike-factor", type=float, default=1.0)
    parser.add_argument("--rvd-spike-ema-window", type=int, default=20)
    parser.add_argument(
        "--rvd-repay-mode", choices=["asap", "spread"], default="asap"
    )
    parser.add_argument(
        "--rvd-repay-protect-dangerous",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--rvd-policy-seed",
        type=int,
        default=None,
        help="RVD/exact-random policy seed; defaults to the training seed",
    )
    parser.add_argument(
        "--skip-update-mode",
        choices=["freeze", "momentum"],
        default=None,
        help="Parameter behavior on skipped-backward steps. "
             "'freeze' (effective default): no parameter update and no "
             "optimizer-state update on skipped steps. "
             "'momentum': LERNAMomentumTrainer extrapolation from stale "
             "optimizer state. Omitting the flag resolves to 'freeze'.",
    )
    parser.add_argument(
        "--scheduler-step-policy",
        choices=SchedulerStepPolicy.VALID,
        default=SchedulerStepPolicy.ALWAYS_STEP,
        help=(
            "Learning-rate scheduler behavior on skipped-backward steps. "
             "The matched Phase 1.3 default advances over every training batch."
        ),
    )
    parser.add_argument(
        "--online-ler-mode",
        choices=[
            ONLINE_LER_MODE_AUTO,
            ONLINE_LER_MODE_OFF,
            ONLINE_LER_MODE_LEGACY_DENSE,
            ONLINE_LER_MODE_SAMPLED_LAGGED,
        ],
        default=ONLINE_LER_MODE_AUTO,
        help=(
            "Online LER diagnostics tracker mode. 'auto' resolves per arm: "
            "signal-free arms get 'off'; signal-consuming arms get "
            "'sampled_lagged'."
        ),
    )
    parser.add_argument(
        "--online-ler-sample-size",
        type=int,
        default=4096,
        help="Parameter sample size for sampled_lagged online LER tracking",
    )
    parser.add_argument(
        "--online-ler-update-interval",
        type=int,
        default=1,
        help="Update interval in real backward steps for online LER tracking",
    )
    parser.add_argument(
        "--provenance-classification",
        choices=[
            CLASSIFICATION_MATCHED_CLAIM,
            CLASSIFICATION_LOCAL_DEVELOPMENT,
        ],
        default=CLASSIFICATION_MATCHED_CLAIM,
        help=(
            "matched_claim requires a clean tracked tree, fixed budget, "
            "freeze mode, and successful Piece 5 validation; use "
            "local_development explicitly for exploratory runs"
        ),
    )
    return parser


def main():
    args = build_arg_parser().parse_args()

    profile = detect_device_profile()

    if args.mode == "smoke":
        tasks = ["sst2"]
        seeds = [42]
        ablations_to_run = ["full_lerna", "no_ler", "no_safety"]
    elif args.mode == "full":
        tasks = ABLATION_GLUE_TASKS
        seeds = SEEDS
        ablations_to_run = list(DEFAULT_ABLATIONS)
    else:
        tasks = args.tasks or ["sst2"]
        seeds = args.seeds or [42]
        ablations_to_run = args.ablations or list(DEFAULT_ABLATIONS)

    if args.tasks:
        tasks = args.tasks
    if args.seeds:
        seeds = args.seeds
    if args.ablations:
        ablations_to_run = args.ablations

    effective_max_samples = args.max_samples
    if effective_max_samples is None and not args.unlimited:
        effective_max_samples = 2000 if profile != "server" else 25000

    from lerna.utils.model_loader import MODELS
    model_name = MODELS[args.model]
    wandb_group = args.wandb_group or f"ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    total_runs = len(tasks) * len(seeds) * len(ablations_to_run)
    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  LERNA Ablation Study")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Ablations: {ablations_to_run}")
    print(f"  Total runs: {total_runs}")
    print(f"  Max samples/task: {effective_max_samples or 'unlimited'}")
    print(f"  ═══════════════════════════════════════════════════════\n")

    if args.wandb:
        _ensure_wandb_finished()

    all_results = []
    run_idx = 0
    overall_start = time.time()

    for task in tasks:
        for seed in seeds:
            for ablation_name in ablations_to_run:
                run_idx += 1

                if run_idx > 1:
                    elapsed = time.time() - overall_start
                    avg_per_run = elapsed / (run_idx - 1)
                    remaining = (total_runs - run_idx + 1) * avg_per_run
                    print(f"\n  ═══ Run {run_idx}/{total_runs} | ETA: {timedelta(seconds=int(remaining))} ═══")
                else:
                    print(f"\n  ═══ Run {run_idx}/{total_runs} ═══")

                task_hp = TASK_HP_OVERRIDES.get(task, {})
                try:
                    result = run_ablation_single(
                        task_name=task,
                        seed=seed,
                        ablation_name=ablation_name,
                        ablation_overrides=ABLATIONS[ablation_name],
                        model_name=model_name,
                        profile=profile,
                        base_output_dir=args.output_dir,
                        use_wandb=args.wandb,
                        max_samples_override=effective_max_samples,
                        run_idx=run_idx,
                        total_runs=total_runs,
                        wandb_project=args.wandb_project,
                        wandb_group=wandb_group,
                        num_epochs=task_hp.get("num_epochs", 3),
                        warmup_ratio=task_hp.get("warmup_ratio", 0.1),
                        early_stopping_patience=task_hp.get("early_stopping_patience", 5),
                        metric_for_best_model=task_hp.get("metric_for_best_model", "eval_loss"),
                        greater_is_better=task_hp.get("greater_is_better", False),
                        init_from_mnli=task_hp.get("init_from_mnli", False),
                        no_early_stopping=args.no_early_stopping,
                        target_skip_rate=args.target_skip_rate,
                        max_consecutive_skips=args.max_consecutive_skips,
                        probe_interval=args.probe_interval,
                        policy=args.policy,
                        rho_veto_threshold=args.rho_veto_threshold,
                        risk_gamma=args.risk_gamma,
                        guard_mode=args.guard_mode,
                        skip_update_mode=args.skip_update_mode,
                        scheduler_step_policy=args.scheduler_step_policy,
                        allow_early_stopping_with_skipping=(
                            args.allow_early_stopping_with_skipping
                        ),
                        rvd_veto_mode=args.rvd_veto_mode,
                        rvd_margin_rank_floor=args.rvd_margin_rank_floor,
                        rvd_spike_factor=args.rvd_spike_factor,
                        rvd_spike_ema_window=args.rvd_spike_ema_window,
                        rvd_repay_mode=args.rvd_repay_mode,
                        rvd_repay_protect_dangerous=(
                            args.rvd_repay_protect_dangerous
                        ),
                        rvd_policy_seed=args.rvd_policy_seed,
                        provenance_classification=args.provenance_classification,
                        online_ler_mode=args.online_ler_mode,
                        online_ler_parameter_sample_size=(
                            args.online_ler_sample_size
                        ),
                        online_ler_update_interval=(
                            args.online_ler_update_interval
                        ),
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  FAILED: {task} seed={seed} ablation={ablation_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "task": task, "seed": seed,
                        "ablation": ablation_name, "error": str(e)})
                    if args.wandb:
                        _ensure_wandb_finished()

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - overall_start
    successful = [r for r in all_results if "error" not in r]

    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY COMPLETE: {len(all_results)} runs")
    print(f"  Summary: {summary_path}")
    print(f"  Total wall time: {timedelta(seconds=int(total_elapsed))}")

    if successful:
        print(f"\n  {'Ablation':<15} {'Runs':>5} {'Avg Acc':>10} {'Std':>8} {'Avg kWh':>10} {'Avg LER':>10} {'Skip%':>8}")
        print(f"  {'-'*80}")
        for ablab in ablations_to_run:
            ab_results = [r for r in successful if r.get("ablation") == ablab]
            if not ab_results:
                continue
            accs = [r.get("eval_metrics", {}).get("eval_accuracy",
                r.get("eval_metrics", {}).get("eval_matthews_correlation",
                r.get("eval_metrics", {}).get("eval_pearson",
                r.get("eval_metrics", {}).get("eval_f1", 0)))) for r in ab_results]
            kwhs = [r.get("energy_kwh", 0) for r in ab_results]
            lers = [r.get("ler_final", {}).get("ler") for r in ab_results if r.get("ler_final", {}).get("ler") is not None]
            skip_ratios = [r.get("true_skip_instrumentation", {}).get("skip_ratio_by_batch", 0) for r in ab_results]
            print(
                f"  {ablab:<15} {len(ab_results):>5} "
                f"{np.mean(accs):>10.4f} {np.std(accs):>8.4f} "
                f"{np.mean(kwhs):>10.6f} "
                f"{np.mean(lers) if lers else 0:.2e} "
                f"{np.mean(skip_ratios) * 100:>7.1f}%"
            )

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
