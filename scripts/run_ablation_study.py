#!/usr/bin/env python3
"""
LERNA Phase 1.3 ablation runner.

PHASE1_3_MATRIX is the authoritative six-arm matched-budget matrix. The six
canonical arm names are:

    full_finetune
    exact_random
    fixed_phase_strat
    phase_strat_guarded
    ler_guided_stratified
    ler_guided_stratified_safe

random_skip is a compatibility alias for exact_random and is excluded from
the default matrices.

full_lerna, the no_* arms (no_rho_vg, no_ler, no_safety, no_hysteresis,
no_momentum), rvd, and grad_norm are legacy/exploratory arms; they are not
aliases and are not matrix defaults. phase_strat is an ambiguous legacy
policy name; prefer the explicit matrix arm names above.

Usage:
    python scripts/run_ablation_study.py --mode smoke --no-early-stopping --skip-update-mode freeze
    python scripts/run_ablation_study.py --mode full --tasks sst2 qnli --seeds 42 43 44 --no-early-stopping --skip-update-mode freeze
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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

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
    LERGuidedStratifiedPolicy, LERGuidedStratifiedSafetyPolicy,
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
    CLASSIFICATION_PILOT_NON_CLAIM,
    build_identity_inputs,
    build_scientific_fingerprint,
    finalize_manifest_completed,
    finalize_manifest_failed,
    write_manifest_running,
)
from lerna.utils.phase1_3_completed_matrix import (
    validate_phase1_3_completed_matrix,
)
from lerna.utils.phase1_3_operations import (
    PILOT_SEED,
    PRODUCTION_SEEDS,
    Phase13OperationalError,
    assert_environment_matches,
    assert_fresh_execution,
    collect_phase1_3_environment,
    freeze_matrix_validation,
    load_phase1_3_plan,
    persist_phase1_3_plan,
    recover_stale_running_attempts,
    require_clean_git_state,
    require_frozen_mrpc_facts,
    scan_phase1_3_progress,
)
from lerna.utils.phase1_3_matrix import (
    PHASE1_3_CANONICAL_ARMS,
    PHASE1_3_POLICY_CLASSES,
    PLANNED_CELL_REQUIRED_FIELDS,
    STRICT_TARGET_SKIP_RATES,
    validate_phase1_3_matrix_plan,
)
from lerna.utils.model_loader import (
    ETTIN_MODEL_ID,
    ETTIN_REVISION,
    load_model_and_tokenizer,
    load_tokenizer,
    MODELS,
    validate_ettin_revision,
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
    "ler_guided_stratified": {"control": "ler_guided_stratified"},
    "ler_guided_stratified_safe": {"control": "ler_guided_stratified_safe"},
    "fixed_phase_strat": {"control": "fixed_phase_strat"},
    "phase_strat_guarded": {"control": "phase_strat_guarded"},
    # Compatibility alias for old invocations; excluded from default matrices.
    "random_skip":      {"control": "exact_random", "alias_of": "exact_random"},
}

PHASE1_3_MATRIX = [
    "full_finetune",
    "exact_random",
    "fixed_phase_strat",
    "phase_strat_guarded",
    "ler_guided_stratified",
    "ler_guided_stratified_safe",
]

POLICY_MIN_STEP = 50
LER_GUIDED_CONTROL = "ler_guided_stratified"
LER_GUIDED_SAFE_CONTROL = "ler_guided_stratified_safe"
LER_GUIDED_CONTROLS = {LER_GUIDED_CONTROL, LER_GUIDED_SAFE_CONTROL}
SKIPPING_CONTROLS = (
    {"exact_random", "fixed_phase_strat", "phase_strat_guarded",
     "rvd", "grad_norm"}
    | LER_GUIDED_CONTROLS
)
ONLINE_LER_MODE_AUTO = "auto"
ONLINE_LER_SIGNAL_FREE_CONTROLS = (
    "full_finetune", "exact_random", "fixed_phase_strat"
)
ONLINE_LER_SIGNAL_FREE_POLICY = "fixed_phase_strat"
DEFAULT_ABLATIONS = list(PHASE1_3_MATRIX)
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


def canonicalize_rvd_identity(rvd_config: dict, training_seed: int) -> dict:
    """Return an RVD identity containing only behaviorally active fields."""
    veto_mode = rvd_config.get("veto_mode", "none")
    canonical = dict(rvd_config)
    if veto_mode != "margin":
        canonical.pop("margin_rank_floor", None)
    if veto_mode != "loss_spike":
        canonical.pop("spike_factor", None)
        canonical.pop("spike_ema_window", None)
    policy_seed = canonical.get("policy_seed")
    if policy_seed is not None and policy_seed == training_seed:
        canonical.pop("policy_seed_defaulted_to_training_seed", None)
    return canonical


def build_ler_guided_controller_config(
    *,
    control: str,
    target_skip_rate: float,
    total_steps: int,
    policy_seed: int,
    max_consecutive_skips: int,
    probe_interval: int,
    rho_veto_threshold: float,
) -> dict:
    """Build canonical scientific configuration for a LER-guided arm."""
    if control not in LER_GUIDED_CONTROLS:
        raise ValueError(
            f"Unknown LER-guided control: {control!r}; "
            f"expected one of {sorted(LER_GUIDED_CONTROLS)}"
        )

    target_skip_rate = float(target_skip_rate)
    if not math.isfinite(target_skip_rate) or not 0.0 <= target_skip_rate <= 1.0:
        raise ValueError("target_skip_rate must be finite and in [0, 1]")

    total_steps_int = int(total_steps)
    if total_steps_int <= POLICY_MIN_STEP or total_steps_int != total_steps:
        raise ValueError(
            f"total_steps must be an integer greater than POLICY_MIN_STEP="
            f"{POLICY_MIN_STEP}; got {total_steps!r}"
        )

    max_consecutive_skips = int(max_consecutive_skips)
    if max_consecutive_skips < 1:
        raise ValueError("max_consecutive_skips must be >= 1")
    probe_interval = int(probe_interval)
    if probe_interval < 1:
        raise ValueError("probe_interval must be >= 1")
    rho_veto_threshold = float(rho_veto_threshold)
    if not math.isfinite(rho_veto_threshold):
        raise ValueError("rho_veto_threshold must be finite")

    requested_quota = int(round(target_skip_rate * total_steps_int))
    eligible_count = total_steps_int - POLICY_MIN_STEP
    if requested_quota > eligible_count:
        raise ValueError(
            f"Infeasible skip quota: requested {requested_quota} skips after "
            f"min_step={POLICY_MIN_STEP}, but only {eligible_count} decisions "
            "are eligible; quotas are never clipped"
        )

    safety_enabled = control == LER_GUIDED_SAFE_CONTROL
    config = {
        "control": control,
        "policy_class": (
            "LERGuidedStratifiedSafetyPolicy"
            if safety_enabled
            else "LERGuidedStratifiedPolicy"
        ),
        "policy_name": control,
        "target_skip_rate": target_skip_rate,
        "total_steps": total_steps_int,
        "min_step": POLICY_MIN_STEP,
        "policy_seed": int(policy_seed),
        "n_phases": 4,
        "phase_weights": [0.22, 0.24, 0.26, 0.28],
        "max_consecutive_skips": max_consecutive_skips,
        "probe_interval": probe_interval,
        "min_ler_observations": 3,
        "ler_guidance_strength": 1.0,
        "required_tracker_mode": "sampled_lagged",
        "required_tracker_timing": "post_decision_after_backward",
        "safety_enabled": safety_enabled,
    }
    if safety_enabled:
        config.update(
            {
                "use_rho_vg_safety": True,
                "rho_veto_threshold": rho_veto_threshold,
                "use_loss_spike_safety": True,
                "loss_spike_factor": 1.0,
                "loss_spike_window": 5,
            }
        )
    return config


def build_phase_strat_controller_config(
    *,
    control: str,
    target_skip_rate: float,
    total_steps: int,
    policy_seed: int,
    max_consecutive_skips: int,
    n_phases: int = 4,
    phase_weights: Optional[List[float]] = None,
    rho_veto_threshold: float = -0.2,
    spike_factor: float = 1.0,
    use_rho_vg: bool = True,
    use_safety_horizon: bool = True,
    risk_gamma: float = 0.0,
    guarded: bool = False,
) -> dict:
    """Build canonical provenance configuration for a phase-stratified arm."""
    if control not in ("fixed_phase_strat", "phase_strat_guarded"):
        raise ValueError(
            f"Unknown phase-stratified control: {control!r}; "
            "expected 'fixed_phase_strat' or 'phase_strat_guarded'"
        )

    target_skip_rate = float(target_skip_rate)
    if not math.isfinite(target_skip_rate) or not 0.0 <= target_skip_rate <= 1.0:
        raise ValueError("target_skip_rate must be finite and in [0, 1]")

    total_steps_int = int(total_steps)
    if total_steps_int <= POLICY_MIN_STEP or total_steps_int != total_steps:
        raise ValueError(
            f"total_steps must be an integer greater than POLICY_MIN_STEP="
            f"{POLICY_MIN_STEP}; got {total_steps!r}"
        )

    max_consecutive_skips = int(max_consecutive_skips)
    if max_consecutive_skips < 1:
        raise ValueError("max_consecutive_skips must be >= 1")

    if phase_weights is None:
        phase_weights = [0.22, 0.24, 0.26, 0.28][:n_phases]
    s = sum(phase_weights) or 1.0
    normalized_weights = [w / s for w in phase_weights]

    requested_quota = int(round(target_skip_rate * total_steps_int))
    eligible_count = total_steps_int - POLICY_MIN_STEP
    if requested_quota > eligible_count:
        raise ValueError(
            f"Infeasible skip quota: requested {requested_quota} skips after "
            f"min_step={POLICY_MIN_STEP}, but only {eligible_count} decisions "
            "are eligible; quotas are never clipped"
        )

    eligible = list(range(POLICY_MIN_STEP, total_steps_int))
    n_elig = len(eligible)
    edges = [
        POLICY_MIN_STEP + int(round(n_elig * i / n_phases))
        for i in range(n_phases + 1)
    ]
    phase_bounds = edges
    phase_eligible = [edges[i + 1] - edges[i] for i in range(n_phases)]

    raw = [requested_quota * w for w in normalized_weights]
    q = [int(math.floor(x)) for x in raw]
    q = [min(q[i], phase_eligible[i]) for i in range(n_phases)]
    rem = requested_quota - sum(q)
    order = sorted(
        range(n_phases), key=lambda i: raw[i] - q[i], reverse=True
    )
    idx = 0
    while rem > 0 and idx < 10 * n_phases:
        i = order[idx % n_phases]
        if q[i] < phase_eligible[i]:
            q[i] += 1
            rem -= 1
        idx += 1
    phase_quota = q

    config = {
        "control": control,
        "controller_class": (
            "PhaseStratifiedGuardedRandomPolicy"
            if guarded
            else "FixedPhaseStratifiedRandomPolicy"
        ),
        "policy_name": control,
        "target_skip_rate": target_skip_rate,
        "total_steps": total_steps_int,
        "min_step": POLICY_MIN_STEP,
        "policy_seed": int(policy_seed),
        "n_phases": n_phases,
        "phase_weights": normalized_weights,
        "phase_bounds": phase_bounds,
        "phase_quota": phase_quota,
        "phase_eligible": phase_eligible,
        "requested_quota": requested_quota,
    }
    if guarded:
        # phase_strat_guarded consumes max_consecutive_skips and risk_gamma;
        # fixed_phase_strat does not (its policy uses hardcoded defaults).
        config["max_consecutive_skips"] = max_consecutive_skips
        config["risk_gamma"] = float(risk_gamma)
        config.update(
            {
                "guarded_safety": {
                    "use_rho_vg": bool(use_rho_vg),
                    "rho_veto_threshold": float(rho_veto_threshold),
                    "use_safety_horizon": bool(use_safety_horizon),
                    "spike_factor": float(spike_factor),
                }
            }
        )
    return config


def add_ler_guided_to_identity(identity_inputs, controller_config) -> dict:
    """Return identity inputs with a defensive LER-guided config copy."""
    extended = dict(identity_inputs)
    copied_config = dict(controller_config)
    copied_config["phase_weights"] = list(controller_config["phase_weights"])
    extended["ler_guided_controller"] = copied_config
    return extended


def copy_ler_guided_config(controller_config) -> dict:
    """Return a defensively copied LER-guided controller config.

    Fresh dict and a fresh phase_weights list so later mutation of either
    cannot leak into shared provenance/results structures.
    """
    copied = dict(controller_config)
    copied["phase_weights"] = list(controller_config["phase_weights"])
    return copied


def build_ler_guided_skip_policy(
    *,
    ler_tracker,
    ler_guided_controller: dict,
):
    """Construct the LER-guided policy from its canonical controller config.

    The control field is the authoritative arm identity; safety_enabled is
    derived from it so a malformed config cannot claim the wrong arm. The
    policy constructor validates the tracker and parameters itself (single
    guard); this factory only maps the canonical config onto the classes.
    """
    control = ler_guided_controller["control"]
    if control not in LER_GUIDED_CONTROLS:
        raise ValueError(
            f"Unknown LER-guided control: {control!r}; "
            f"expected one of {sorted(LER_GUIDED_CONTROLS)}"
        )
    safety_enabled = control == LER_GUIDED_SAFE_CONTROL
    PolicyCls = (
        LERGuidedStratifiedSafetyPolicy
        if safety_enabled
        else LERGuidedStratifiedPolicy
    )
    kwargs = {
        "target_skip_rate": ler_guided_controller["target_skip_rate"],
        "total_steps": ler_guided_controller["total_steps"],
        "min_step": ler_guided_controller["min_step"],
        "seed": ler_guided_controller["policy_seed"],
        "n_phases": ler_guided_controller["n_phases"],
        "phase_weights": ler_guided_controller["phase_weights"],
        "max_consecutive_skips": ler_guided_controller["max_consecutive_skips"],
        "probe_interval": ler_guided_controller["probe_interval"],
        "min_ler_observations": ler_guided_controller["min_ler_observations"],
        "ler_guidance_strength": ler_guided_controller["ler_guidance_strength"],
    }
    if safety_enabled:
        kwargs.update(
            {
                "use_rho_vg_safety": ler_guided_controller["use_rho_vg_safety"],
                "rho_veto_threshold": ler_guided_controller["rho_veto_threshold"],
                "use_loss_spike_safety": ler_guided_controller[
                    "use_loss_spike_safety"
                ],
                "loss_spike_factor": ler_guided_controller["loss_spike_factor"],
                "loss_spike_window": ler_guided_controller["loss_spike_window"],
            }
        )
    return PolicyCls(ler_tracker=ler_tracker, **kwargs)


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


def build_power_evidence(power_callback):
    """Return the authoritative raw power evidence embedded in results.json."""
    return {
        "authoritative_copy": "results.json",
        "measurement_source": power_callback.energy_measurement_source,
        "energy_valid": bool(power_callback.energy_valid),
        "energy_invalid_reason": power_callback.energy_invalid_reason,
        "gpu_name": power_callback._gpu_name,
        "gpu_index": int(power_callback.gpu_index),
        "gpu_selector": str(power_callback.gpu_selector),
        "sample_interval_s": float(power_callback.sample_interval_s),
        "nvidia_smi_query_count": int(
            power_callback._nvidia_smi_query_count
        ),
        "nvidia_smi_success_count": int(
            power_callback._nvidia_smi_success_count
        ),
        "total_energy_kwh": float(power_callback.cumulative_kwh),
        "raw_samples": [dict(sample) for sample in power_callback._power_samples],
        "per_step_energy": [
            dict(sample) for sample in power_callback.step_energies
        ],
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


def resolve_task_data_facts(
    task_name,
    tokenizer,
    max_samples,
    profile,
) -> dict:
    """Resolve task and horizon facts without constructing training state."""
    hw_cfg = dict(get_training_config(profile))
    if max_samples is not None:
        hw_cfg["max_samples"] = max_samples

    task_hp = TASK_HP_OVERRIDES.get(task_name, {})
    num_epochs = task_hp.get("num_epochs", 3)
    train_ds, eval_ds, _ = load_glue_task(
        task_name,
        tokenizer,
        max_length=128,
        max_samples=hw_cfg["max_samples"],
    )

    visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_n_gpu = 1 if visible_gpus > 1 else max(1, visible_gpus)
    total_steps = compute_authoritative_horizon(
        train_dataset=train_ds,
        num_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        n_gpu=effective_n_gpu,
    )

    max_samples_effective = hw_cfg["max_samples"]
    return {
        "task": str(task_name),
        "num_epochs": int(num_epochs),
        "max_samples_requested": (
            None if max_samples is None else int(max_samples)
        ),
        "max_samples_effective": (
            None
            if max_samples_effective is None
            else int(max_samples_effective)
        ),
        "train_samples_realized": len(train_ds),
        "eval_samples_realized": len(eval_ds),
        "train_dataset_fingerprint": getattr(train_ds, "_fingerprint", None),
        "eval_dataset_fingerprint": getattr(eval_ds, "_fingerprint", None),
        "total_steps": int(total_steps),
        "per_device_train_batch_size": int(
            hw_cfg["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            hw_cfg["gradient_accumulation_steps"]
        ),
        "effective_n_gpu": int(effective_n_gpu),
    }


def plan_phase1_3_cell(
    *,
    task_name,
    training_seed,
    policy_seed,
    ablation_name,
    target_skip_rate,
    model_name,
    model_revision,
    data_facts,
    git_sha,
    base_output_dir,
    scheduler_step_policy,
    max_consecutive_skips,
    probe_interval,
    rho_veto_threshold,
    risk_gamma,
    online_ler_mode,
    online_ler_parameter_sample_size,
    online_ler_update_interval,
    use_rho_vg=True,
    use_safety_horizon=True,
) -> dict:
    """Build one canonical Phase 1.3 cell without execution side effects."""
    if ablation_name not in PHASE1_3_CANONICAL_ARMS:
        raise ValueError(
            f"Unknown Phase 1.3 arm: {ablation_name!r}; expected one of "
            f"{list(PHASE1_3_CANONICAL_ARMS)}"
        )
    if data_facts.get("task") != task_name:
        raise ValueError(
            f"data_facts task {data_facts.get('task')!r} does not match "
            f"planned task {task_name!r}"
        )

    target_skip_rate = float(target_skip_rate)
    if not math.isfinite(target_skip_rate) or not 0.0 <= target_skip_rate <= 1.0:
        raise ValueError("target_skip_rate must be finite and in [0, 1]")
    if scheduler_step_policy != "skip_on_backward_skip":
        raise ValueError(
            "Phase 1.3 planning requires "
            "scheduler_step_policy='skip_on_backward_skip'"
        )

    training_seed = int(training_seed)
    policy_seed = int(policy_seed)
    num_epochs = int(data_facts["num_epochs"])
    total_steps = int(data_facts["total_steps"])
    if total_steps <= POLICY_MIN_STEP:
        raise ValueError(
            f"total_steps must be greater than POLICY_MIN_STEP="
            f"{POLICY_MIN_STEP}; got {total_steps}"
        )

    is_skipping_arm = ablation_name != "full_finetune"
    if is_skipping_arm:
        requested_quota = round(target_skip_rate * total_steps)
        eligible_count = total_steps - POLICY_MIN_STEP
        if requested_quota > eligible_count:
            raise ValueError(
                f"Infeasible skip quota: requested {requested_quota} skips "
                f"after min_step={POLICY_MIN_STEP}, but only {eligible_count} "
                "decisions are eligible; quotas are never clipped"
            )
        planned_skips = requested_quota
        compute_saving_mechanism = "backward_skipping"
    else:
        requested_quota = None
        planned_skips = 0
        compute_saving_mechanism = "none"

    resolved_online_ler = resolve_online_ler_config(
        online_ler_mode,
        effective_control=ablation_name,
        policy=None,
        parameter_sample_size=online_ler_parameter_sample_size,
        update_interval=online_ler_update_interval,
    )
    online_diagnostics = build_online_ler_provenance_config(
        resolved_online_ler,
        sample_seed=training_seed,
    )
    expected_online_mode = (
        ONLINE_LER_MODE_OFF
        if ablation_name in ONLINE_LER_SIGNAL_FREE_CONTROLS
        else ONLINE_LER_MODE_SAMPLED_LAGGED
    )
    if online_diagnostics["mode"] != expected_online_mode:
        raise ValueError(
            f"Arm {ablation_name!r} requires online LER mode "
            f"{expected_online_mode!r}; got {online_diagnostics['mode']!r}"
        )

    identity_inputs = build_identity_inputs(
        task=task_name,
        training_seed=training_seed,
        model_id=model_name,
        model_revision=model_revision,
        max_samples_requested=data_facts["max_samples_requested"],
        train_samples_realized=data_facts["train_samples_realized"],
        eval_samples_realized=data_facts["eval_samples_realized"],
        train_dataset_fingerprint=data_facts["train_dataset_fingerprint"],
        eval_dataset_fingerprint=data_facts["eval_dataset_fingerprint"],
        num_epochs=num_epochs,
        control=ablation_name,
        target_skip_rate=target_skip_rate,
        policy_seed=policy_seed,
        skip_update_mode="freeze",
        scheduler_step_policy=scheduler_step_policy,
        no_early_stopping=True,
        total_steps=total_steps,
        git_sha=git_sha,
    )
    identity_inputs = add_online_ler_to_identity(
        identity_inputs,
        online_diagnostics,
    )

    controller_config = {
        "arm": ablation_name,
        "arm_alias_of": None,
        "control": ablation_name,
        "policy_class": PHASE1_3_POLICY_CLASSES[ablation_name],
        "compute_saving_mechanism": compute_saving_mechanism,
        "policy_seed": policy_seed,
        "target_skip_rate": target_skip_rate,
        "min_step": POLICY_MIN_STEP,
        "configured_total_steps": total_steps,
        "requested_quota": requested_quota,
        "matched_budget": True,
        "is_skipping_arm": is_skipping_arm,
        "allow_early_stopping_with_skipping": False,
        "early_stopping_active": False,
        "num_epochs": num_epochs,
        "online_diagnostics": dict(online_diagnostics),
    }

    if ablation_name in ("fixed_phase_strat", "phase_strat_guarded"):
        phase_kwargs = {
            "control": ablation_name,
            "target_skip_rate": target_skip_rate,
            "total_steps": total_steps,
            "policy_seed": policy_seed,
            "max_consecutive_skips": max_consecutive_skips,
        }
        if ablation_name == "phase_strat_guarded":
            phase_kwargs.update(
                guarded=True,
                rho_veto_threshold=rho_veto_threshold,
                spike_factor=1.0,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                risk_gamma=risk_gamma,
            )
        identity_inputs["phase_strat_controller"] = (
            build_phase_strat_controller_config(**phase_kwargs)
        )
        controller_config["phase_strat_controller"] = (
            build_phase_strat_controller_config(**phase_kwargs)
        )
    elif ablation_name in LER_GUIDED_CONTROLS:
        ler_config = build_ler_guided_controller_config(
            control=ablation_name,
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            policy_seed=policy_seed,
            max_consecutive_skips=max_consecutive_skips,
            probe_interval=probe_interval,
            rho_veto_threshold=rho_veto_threshold,
        )
        identity_inputs = add_ler_guided_to_identity(
            identity_inputs,
            ler_config,
        )
        controller_config["ler_guided_controller"] = (
            copy_ler_guided_config(ler_config)
        )

    fingerprint = build_scientific_fingerprint(identity_inputs)
    planned_arm_dir = os.path.join(
        base_output_dir,
        ablation_name,
        fingerprint,
    )
    return {
        "arm": ablation_name,
        "control": ablation_name,
        "task": str(task_name),
        "training_seed": training_seed,
        "policy_seed": policy_seed,
        "model_id": str(model_name),
        "model_revision": model_revision,
        "target_skip_rate": target_skip_rate,
        "num_epochs": num_epochs,
        "total_steps": total_steps,
        "min_step": POLICY_MIN_STEP,
        "requested_quota": requested_quota,
        "planned_skips": planned_skips,
        "is_skipping_arm": is_skipping_arm,
        "matched_budget": True,
        "no_early_stopping": True,
        "skip_update_mode": "freeze",
        "scheduler_step_policy": "skip_on_backward_skip",
        "online_diagnostics": dict(online_diagnostics),
        "controller_config": controller_config,
        "identity_inputs": identity_inputs,
        "fingerprint": fingerprint,
        "planned_arm_dir": planned_arm_dir,
        "max_consecutive_skips": int(max_consecutive_skips),
        "probe_interval": int(probe_interval),
        "rho_veto_threshold": float(rho_veto_threshold),
        "risk_gamma": float(risk_gamma),
        "online_ler_parameter_sample_size": int(online_ler_parameter_sample_size),
        "online_ler_update_interval": int(online_ler_update_interval),
        "use_rho_vg": bool(use_rho_vg),
        "use_safety_horizon": bool(use_safety_horizon),
    }


def build_phase1_3_matrix_plan(
    *,
    tasks,
    seeds,
    target_skip_rates,
    model_name,
    model_revision=None,
    base_output_dir,
    data_facts_provider,
    git_sha,
    **cell_kwargs,
) -> list[dict]:
    """Build the ordered Phase 1.3 matrix without validating or executing."""
    data_facts_by_task = {}
    plan = []
    for task in tasks:
        if task not in data_facts_by_task:
            data_facts_by_task[task] = data_facts_provider(task)
        data_facts = data_facts_by_task[task]
        for training_seed in seeds:
            for target_skip_rate in target_skip_rates:
                for arm in PHASE1_3_CANONICAL_ARMS:
                    plan.append(
                        plan_phase1_3_cell(
                            task_name=task,
                            training_seed=training_seed,
                            policy_seed=training_seed,
                            ablation_name=arm,
                            target_skip_rate=target_skip_rate,
                            model_name=model_name,
                            model_revision=model_revision,
                            data_facts=data_facts,
                            git_sha=git_sha,
                            base_output_dir=base_output_dir,
                            **cell_kwargs,
                        )
                    )
    return plan


def _resolve_git_sha() -> str:
    """Resolve the current checkout SHA without changing repository state."""
    try:
        import subprocess

        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _collect_planned_runtime_mismatches(
    planned,
    runtime,
    *,
    path: str,
    mismatches: list[str],
    allow_runtime_extra_keys: bool = False,
) -> None:
    """Collect type-strict differences in one planned/runtime value."""
    if type(planned) is not type(runtime):
        mismatches.append(
            f"{path}: planned type {type(planned).__name__} != "
            f"runtime type {type(runtime).__name__}"
        )
        return
    if type(planned) is dict:
        if not allow_runtime_extra_keys:
            for key in runtime.keys() - planned.keys():
                child_path = f"{path}.{key}" if path else str(key)
                mismatches.append(f"{child_path}: unexpected runtime field")
        for key, planned_value in planned.items():
            child_path = f"{path}.{key}" if path else str(key)
            if key not in runtime:
                mismatches.append(f"{child_path}: missing from runtime")
                continue
            _collect_planned_runtime_mismatches(
                planned_value,
                runtime[key],
                path=child_path,
                mismatches=mismatches,
            )
        return
    if type(planned) is list:
        if len(planned) != len(runtime):
            mismatches.append(
                f"{path}: planned length {len(planned)} != "
                f"runtime length {len(runtime)}"
            )
            return
        for index, (planned_value, runtime_value) in enumerate(
            zip(planned, runtime)
        ):
            _collect_planned_runtime_mismatches(
                planned_value,
                runtime_value,
                path=f"{path}[{index}]",
                mismatches=mismatches,
            )
        return
    if planned != runtime:
        mismatches.append(f"{path}: planned {planned!r} != runtime {runtime!r}")


def assert_phase1_3_runtime_matches_plan(
    planned_cell: dict,
    runtime_cell: dict,
) -> None:
    """Abort a strict cell before side effects when runtime identity drifts."""
    if type(planned_cell) is not dict or type(runtime_cell) is not dict:
        raise TypeError("planned_cell and runtime_cell must be dictionaries")

    mismatches = []
    for field in planned_cell.keys() - PLANNED_CELL_REQUIRED_FIELDS:
        mismatches.append(f"{field}: unexpected planned-cell field")
    for field in runtime_cell.keys() - PLANNED_CELL_REQUIRED_FIELDS:
        mismatches.append(f"{field}: unexpected runtime-cell field")
    for field in sorted(PLANNED_CELL_REQUIRED_FIELDS):
        if field not in planned_cell:
            mismatches.append(f"{field}: missing from planned cell")
            continue
        if field not in runtime_cell:
            mismatches.append(f"{field}: missing from runtime cell")
            continue
        _collect_planned_runtime_mismatches(
            planned_cell[field],
            runtime_cell[field],
            path=field,
            mismatches=mismatches,
            allow_runtime_extra_keys=(field == "controller_config"),
        )

    if mismatches:
        details = "; ".join(mismatches[:20])
        if len(mismatches) > 20:
            details += f"; ... and {len(mismatches) - 20} more"
        raise ValueError(f"Planned/runtime mismatch: {details}")


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
    ler_guided_controller_config=None,
):
    """Construct one explicit baseline/RVD/LER-guided control arm."""
    control = "exact_random" if control == "random_skip" else control
    if control == "full_finetune":
        return AlwaysFalsePolicy()
    if control in LER_GUIDED_CONTROLS:
        if ler_guided_controller_config is None:
            raise ValueError(
                f"LER-guided control {control!r} requires "
                "ler_guided_controller_config"
            )
        if ler_guided_controller_config.get("control") != control:
            raise ValueError(
                f"LER-guided control {control!r} does not match controller "
                f"config control {ler_guided_controller_config.get('control')!r}"
            )
        return build_ler_guided_skip_policy(
            ler_tracker=ler_tracker,
            ler_guided_controller=ler_guided_controller_config,
        )
    if control == "exact_random":
        return RandomSkipPolicy(
            target_skip_rate=target_skip_rate,
            min_step=POLICY_MIN_STEP,
            seed=controller_cfg["policy_seed"],
            total_steps=total_steps,
        )
    if control == "fixed_phase_strat":
        return FixedPhaseStratifiedRandomPolicy(
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            min_step=POLICY_MIN_STEP,
            seed=controller_cfg["policy_seed"],
        )
    if control == "phase_strat_guarded":
        return PhaseStratifiedGuardedRandomPolicy(
            ler_tracker=ler_tracker,
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            min_step=POLICY_MIN_STEP,
            seed=controller_cfg["policy_seed"],
            max_consecutive_skips=controller_cfg["max_consecutive_skips"],
            rho_veto_threshold=rho_veto_threshold,
            use_rho_vg=use_rho_vg,
            use_safety_horizon=use_safety_horizon,
            risk_gamma=risk_gamma,
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
    planned_cell=None,
    model_revision=None,
):
    """Run a single experiment with a specific ablation config."""

    control = ablation_overrides.get("control")
    effective_control = "exact_random" if control == "random_skip" else control
    if planned_cell is not None and model_revision is None:
        model_revision = planned_cell.get("model_revision")

    model_revision = validate_ettin_revision(model_name, model_revision)
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

    use_rho_vg = ablation_overrides.get("use_rho_vg", True)
    use_ler = ablation_overrides.get("use_ler", True)
    use_safety_horizon = ablation_overrides.get("use_safety_horizon", True)
    use_hysteresis = ablation_overrides.get("use_hysteresis", True)

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
        mnli_model, _ = load_model_and_tokenizer(
            model_name,
            num_labels=cfg["num_labels"],
            revision=model_revision,
            local_files_only=bool(model_revision),
        )
        model, _ = load_model_and_tokenizer(
            model_name,
            num_labels=cfg["num_labels"],
            revision=model_revision,
            local_files_only=bool(model_revision),
        )
        encoder_state = {k: v for k, v in mnli_model.state_dict().items()
                        if "classifier" not in k and "pooler" not in k}
        model.load_state_dict(encoder_state, strict=False)
        del mnli_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        model, tokenizer = load_model_and_tokenizer(
            model_name,
            num_labels=cfg["num_labels"],
            revision=model_revision,
            local_files_only=bool(model_revision),
        )

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

    quota_control = effective_control
    if quota_control is None and policy == "random_veto_deferral":
        quota_control = "rvd"
    requested_quota = None
    if quota_control in (
        "exact_random", "rvd", "fixed_phase_strat", "phase_strat_guarded"
    ) or quota_control in LER_GUIDED_CONTROLS:
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

    # [Piece 9] Deterministic scientific fingerprint for collision-proof identity.
    git_sha = _resolve_git_sha()

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
        model_revision=model_revision,
        max_samples_requested=max_samples_requested,
        train_samples_realized=len(train_ds),
        eval_samples_realized=len(eval_ds),
        train_dataset_fingerprint=getattr(train_ds, "_fingerprint", None),
        eval_dataset_fingerprint=getattr(eval_ds, "_fingerprint", None),
        num_epochs=num_epochs,
        control=effective_control or policy,
        target_skip_rate=target_skip_rate,
        policy_seed=controller_cfg["policy_seed"],
        skip_update_mode=effective_skip_update_mode,
        scheduler_step_policy=scheduler_step_policy,
        no_early_stopping=no_early_stopping,
        total_steps=total_steps,
        git_sha=git_sha,
    )
    identity_inputs = add_online_ler_to_identity(
        identity_inputs,
        online_diagnostics,
    )
    # RVD configuration is included only when the effective control is rvd
    # or the legacy policy is random_veto_deferral (which resolves to rvd).
    if effective_control == "rvd" or (
        effective_control is None and policy == "random_veto_deferral"
    ):
        identity_inputs["rvd"] = canonicalize_rvd_identity(controller_cfg, seed)
    # max_consecutive_skips is represented only in controller configurations
    # where it changes behavior. grad_norm consumes it but has no dedicated
    # controller config, so it is recorded here.
    if effective_control == "grad_norm":
        identity_inputs["max_consecutive_skips"] = int(max_consecutive_skips)
    # Legacy policies that consume max_consecutive_skips.
    if effective_control is None and policy in (
        "quota_hybrid", "guarded_hybrid", "phase_strat", "phase_strat_guarded"
    ):
        identity_inputs["max_consecutive_skips"] = int(max_consecutive_skips)
    if effective_control in LER_GUIDED_CONTROLS:
        ler_guided_controller_config = build_ler_guided_controller_config(
            control=effective_control,
            target_skip_rate=target_skip_rate,
            total_steps=total_steps,
            policy_seed=controller_cfg["policy_seed"],
            max_consecutive_skips=max_consecutive_skips,
            probe_interval=probe_interval,
            rho_veto_threshold=rho_veto_threshold,
        )
        identity_inputs = add_ler_guided_to_identity(
            identity_inputs,
            ler_guided_controller_config,
        )
    else:
        ler_guided_controller_config = None
    if effective_control == "fixed_phase_strat":
        phase_strat_controller_config = (
            build_phase_strat_controller_config(
                control="fixed_phase_strat",
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                policy_seed=controller_cfg["policy_seed"],
                max_consecutive_skips=max_consecutive_skips,
            )
        )
        identity_inputs["phase_strat_controller"] = (
            phase_strat_controller_config
        )
    elif effective_control == "phase_strat_guarded":
        phase_strat_controller_config = (
            build_phase_strat_controller_config(
                control="phase_strat_guarded",
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                policy_seed=controller_cfg["policy_seed"],
                max_consecutive_skips=max_consecutive_skips,
                guarded=True,
                rho_veto_threshold=rho_veto_threshold,
                spike_factor=1.0,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                risk_gamma=risk_gamma,
            )
        )
        identity_inputs["phase_strat_controller"] = (
            phase_strat_controller_config
        )
    else:
        phase_strat_controller_config = None
    fingerprint = build_scientific_fingerprint(identity_inputs)

    # Define run_id from task, seed, arm, and fingerprint.
    run_id = f"{task_name}_s{seed}_{ablation_name}_{fingerprint}"

    online_ler_enabled = online_diagnostics["enabled"]

    ler_tracker = build_online_ler_tracker(
        online_diagnostics,
        task_name=task_name,
        use_hysteresis=use_hysteresis,
        sample_seed=online_diagnostics["sample_seed"],
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
            ler_guided_controller_config=ler_guided_controller_config,
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
                recalibrate_every=200,
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
    if effective_control in LER_GUIDED_CONTROLS:
        controller_config_effective["ler_guided_controller"] = (
            copy_ler_guided_config(ler_guided_controller_config)
        )
    if effective_control == "fixed_phase_strat":
        controller_config_effective["phase_strat_controller"] = (
            build_phase_strat_controller_config(
                control="fixed_phase_strat",
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                policy_seed=controller_cfg["policy_seed"],
                max_consecutive_skips=max_consecutive_skips,
            )
        )
    if effective_control == "phase_strat_guarded":
        controller_config_effective["phase_strat_controller"] = (
            build_phase_strat_controller_config(
                control="phase_strat_guarded",
                target_skip_rate=target_skip_rate,
                total_steps=total_steps,
                policy_seed=controller_cfg["policy_seed"],
                max_consecutive_skips=max_consecutive_skips,
                guarded=True,
                rho_veto_threshold=rho_veto_threshold,
                spike_factor=1.0,
                use_rho_vg=use_rho_vg,
                use_safety_horizon=use_safety_horizon,
                risk_gamma=risk_gamma,
            )
        )
    arm_dir = os.path.join(base_output_dir, ablation_name, fingerprint)
    if planned_cell is not None:
        runtime_cell = {
            "arm": ablation_name,
            "control": effective_control,
            "task": str(task_name),
            "training_seed": int(seed),
            "policy_seed": int(controller_cfg["policy_seed"]),
            "model_id": str(model_name),
            "model_revision": model_revision,
            "target_skip_rate": float(target_skip_rate),
            "num_epochs": int(num_epochs),
            "total_steps": int(total_steps),
            "min_step": POLICY_MIN_STEP,
            "requested_quota": requested_quota,
            "planned_skips": (
                int(requested_quota) if requested_quota is not None else 0
            ),
            "is_skipping_arm": budget_state["is_skipping_arm"],
            "matched_budget": budget_state["matched_budget"],
            "no_early_stopping": bool(no_early_stopping),
            "skip_update_mode": effective_skip_update_mode,
            "scheduler_step_policy": scheduler_step_policy,
            "online_diagnostics": dict(online_diagnostics),
            "controller_config": controller_config_effective,
            "identity_inputs": identity_inputs,
            "fingerprint": fingerprint,
            "planned_arm_dir": arm_dir,
            "max_consecutive_skips": int(max_consecutive_skips),
            "probe_interval": int(probe_interval),
            "rho_veto_threshold": float(rho_veto_threshold),
            "risk_gamma": float(risk_gamma),
            "online_ler_parameter_sample_size": int(online_ler_parameter_sample_size),
            "online_ler_update_interval": int(online_ler_update_interval),
            "use_rho_vg": bool(use_rho_vg),
            "use_safety_horizon": bool(use_safety_horizon),
        }
        assert_phase1_3_runtime_matches_plan(planned_cell, runtime_cell)

    print(
        "  Controller config: "
        + json.dumps(controller_config_effective, sort_keys=True, default=str)
    )

    if use_wandb:
        import wandb

        _ensure_wandb_finished()
        wandb.init(
            project=wandb_project,
            name=run_id,
            group=wandb_group,
            job_type=f"ablation-{ablation_name}",
            tags=[
                task_name,
                f"ablation-{ablation_name}",
                f"seed-{seed}",
                model_name.split("/")[-1],
            ],
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

    # Retry-safe layout: <base>/<arm>/<fingerprint>/attempt-<N>/.
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
    print(
        f"  Skip-update mode: {effective_skip_update_mode}"
        + (
            "  (legacy use_momentum_extrap compat)"
            if skip_mode_legacy_compat_used
            else ""
        )
    )
    print(f"  Profile: {profile} | Output: {output_dir}")
    print(f"{'='*60}")

    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        gpu_index=int(os.environ.get("LERNA_NVIDIA_SMI_GPU", "0")),
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
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
        model_revision=model_revision,
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
            "model": model_name,
            "model_revision": model_revision,
            "train_runtime_s": total_time,
            "train_loss": train_result.training_loss,
            "eval_metrics": eval_result,
            "energy_kwh": power_callback.cumulative_kwh,
            "power_avg_watts": avg_power,
            "power_evidence": build_power_evidence(power_callback),
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
        if effective_control in LER_GUIDED_CONTROLS:
            results["run_config"]["ler_guided_controller"] = (
                copy_ler_guided_config(ler_guided_controller_config)
            )
        if effective_control == "fixed_phase_strat":
            results["run_config"]["phase_strat_controller"] = (
                build_phase_strat_controller_config(
                    control="fixed_phase_strat",
                    target_skip_rate=target_skip_rate,
                    total_steps=total_steps,
                    policy_seed=controller_cfg["policy_seed"],
                    max_consecutive_skips=max_consecutive_skips,
                )
            )
        if effective_control == "phase_strat_guarded":
            results["run_config"]["phase_strat_controller"] = (
                build_phase_strat_controller_config(
                    control="phase_strat_guarded",
                    target_skip_rate=target_skip_rate,
                    total_steps=total_steps,
                    policy_seed=controller_cfg["policy_seed"],
                    max_consecutive_skips=max_consecutive_skips,
                    guarded=True,
                    rho_veto_threshold=rho_veto_threshold,
                    spike_factor=1.0,
                    use_rho_vg=use_rho_vg,
                    use_safety_horizon=use_safety_horizon,
                    risk_gamma=risk_gamma,
                )
            )

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
    parser.add_argument(
        "--mode",
        choices=["smoke", "full", "custom", "phase1_3"],
        default="smoke",
    )
    parser.add_argument(
        "--phase1-3-action",
        choices=["plan", "run", "resume", "validate"],
        default=None,
        help=(
            "Explicit strict-matrix action. Planning persists immutable "
            "evidence; all other actions consume it."
        ),
    )
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Use the frozen one-seed, 12-cell seed-7 non-claim pilot.",
    )
    parser.add_argument(
        "--recover-stale-running-after-hours",
        type=float,
        default=None,
        help=(
            "Resume only: mark running attempts older than this threshold "
            "failed before retrying."
        ),
    )
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Ablation names to run (default: all)")
    parser.add_argument("--output-dir", default="./experiments/ablation")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerna-ablation")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--model", default="modernbert", choices=["roberta", "modernbert", "deberta", "ettin"],
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
    target_rate_group = parser.add_mutually_exclusive_group()
    target_rate_group.add_argument(
        "--target-skip-rate",
        type=float,
        default=None,
        help="Scalar rate for legacy smoke/full/custom workflows",
    )
    target_rate_group.add_argument(
        "--target-skip-rates",
        nargs="+",
        type=float,
        default=None,
        help="Ordered rate list required by strict Phase 1.3 mode",
    )
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


def _validate_and_freeze_strict_matrix(bundle):
    dimensions = bundle["dimensions"]
    result = validate_phase1_3_completed_matrix(
        bundle["plan"],
        tasks=dimensions["tasks"],
        seeds=dimensions["seeds"],
        target_skip_rates=dimensions["target_skip_rates"],
        minimum_seed_count=(
            1 if bundle["envelope"]["matrix_kind"] == "pilot" else 10
        ),
        base_output_dir=str(bundle["root"]),
    )
    return freeze_matrix_validation(
        base_output_dir=str(bundle["root"]),
        bundle=bundle,
        valid_runs=result["valid_runs"],
    )


def _main_phase1_3(args, parser):
    action = args.phase1_3_action
    if action is None:
        parser.error("--mode phase1_3 requires --phase1-3-action")
    if args.model != "ettin":
        parser.error("--mode phase1_3 requires --model ettin")
    if args.ablations is not None:
        parser.error(
            "--mode phase1_3 uses the canonical six arms and rejects "
            "--ablations"
        )
    if args.allow_early_stopping_with_skipping:
        parser.error("--mode phase1_3 forbids early-stopping overrides")
    if args.skip_update_mode not in (None, "freeze"):
        parser.error("--mode phase1_3 requires --skip-update-mode freeze")
    if args.policy != "hybrid":
        parser.error(
            "--mode phase1_3 rejects nondefault legacy --policy values"
        )
    if args.rvd_policy_seed is not None:
        parser.error(
            "--mode phase1_3 pairs policy_seed with training_seed and "
            "rejects --rvd-policy-seed"
        )
    if args.online_ler_mode != ONLINE_LER_MODE_AUTO:
        parser.error(
            "--mode phase1_3 requires --online-ler-mode auto for the "
            "canonical per-arm diagnostic modes"
        )
    if args.provenance_classification != CLASSIFICATION_MATCHED_CLAIM:
        parser.error("--mode phase1_3 requires matched_claim provenance")
    if (
        args.recover_stale_running_after_hours is not None
        and action != "resume"
    ):
        parser.error(
            "--recover-stale-running-after-hours is only valid with "
            "--phase1-3-action resume"
        )

    if action == "plan":
        if args.target_skip_rate is not None:
            parser.error(
                "Phase 1.3 planning requires --target-skip-rates, not "
                "--target-skip-rate"
            )
        if args.target_skip_rates != list(STRICT_TARGET_SKIP_RATES):
            parser.error(
                "Phase 1.3 planning requires --target-skip-rates 0.30 0.40 "
                "in that exact order"
            )
        tasks = list(args.tasks or ["mrpc"])
        if tasks != ["mrpc"]:
            parser.error("Phase 1.3 production planning is frozen to MRPC")
        if args.seeds is None:
            parser.error("Phase 1.3 planning requires explicit --seeds")
        seeds = list(args.seeds)
        if args.pilot:
            if seeds != [PILOT_SEED]:
                parser.error("the pilot requires exactly --seeds 7")
            minimum_seed_count = 1
            matrix_kind = "pilot"
        else:
            if seeds != list(PRODUCTION_SEEDS):
                parser.error(
                    "the production matrix requires the frozen ten seeds in "
                    "their registered order"
                )
            minimum_seed_count = 10
            matrix_kind = "production"
        if args.max_samples is not None or not args.unlimited:
            parser.error(
                "Phase 1.3 planning requires --unlimited and rejects "
                "--max-samples"
            )
    else:
        if any(
            value is not None
            for value in (
                args.tasks,
                args.seeds,
                args.target_skip_rate,
                args.target_skip_rates,
                args.max_samples,
            )
        ) or args.unlimited:
            parser.error(
                "run/resume/validate consume persisted dimensions and reject "
                "task, seed, rate, and sample overrides"
            )
        if args.max_consecutive_skips != 4:
            parser.error(
                "run/resume/validate consume persisted max_consecutive_skips"
            )
        if args.probe_interval != 8:
            parser.error(
                "run/resume/validate consume persisted probe_interval"
            )
        if args.rho_veto_threshold != -0.2:
            parser.error(
                "run/resume/validate consume persisted rho_veto_threshold"
            )
        if args.risk_gamma != 0.0:
            parser.error(
                "run/resume/validate consume persisted risk_gamma"
            )
        if args.online_ler_sample_size != 4096:
            parser.error(
                "run/resume/validate consume persisted online_ler_parameter_sample_size"
            )
        if args.online_ler_update_interval != 1:
            parser.error(
                "run/resume/validate consume persisted online_ler_update_interval"
            )
        tasks = None
        seeds = None
        minimum_seed_count = None
        matrix_kind = None

    if action in {"plan", "validate"} and args.wandb:
        parser.error("plan and validate actions do not initialize W&B")

    repo_root = Path(__file__).resolve().parents[1]
    profile = detect_device_profile()
    model_name = MODELS["ettin"]
    model_revision = ETTIN_REVISION
    git_state = require_clean_git_state(repo_root)

    if action == "plan":
        hw_config = dict(get_training_config(profile))
        tokenizer = load_tokenizer(
            model_name,
            revision=model_revision,
            local_files_only=True,
        )

        resolved_facts = {}

        def data_facts_provider(task):
            facts = resolve_task_data_facts(task, tokenizer, None, profile)
            resolved_facts[task] = facts
            return facts

        matrix_plan = build_phase1_3_matrix_plan(
            tasks=tasks,
            seeds=seeds,
            target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
            model_name=model_name,
            model_revision=model_revision,
            base_output_dir=args.output_dir,
            data_facts_provider=data_facts_provider,
            git_sha=git_state["commit_sha"],
            scheduler_step_policy=SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP,
            max_consecutive_skips=args.max_consecutive_skips,
            probe_interval=args.probe_interval,
            rho_veto_threshold=args.rho_veto_threshold,
            risk_gamma=args.risk_gamma,
            online_ler_mode=ONLINE_LER_MODE_AUTO,
            online_ler_parameter_sample_size=args.online_ler_sample_size,
            online_ler_update_interval=args.online_ler_update_interval,
            use_rho_vg=True,
            use_safety_horizon=True,
        )
        validate_phase1_3_matrix_plan(
            matrix_plan,
            tasks=tasks,
            seeds=seeds,
            target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
            minimum_seed_count=minimum_seed_count,
            base_output_dir=args.output_dir,
        )
        require_frozen_mrpc_facts(resolved_facts["mrpc"])
        metric_probe = build_compute_metrics("mrpc")
        del metric_probe, tokenizer
        environment = collect_phase1_3_environment(
            repo_root=repo_root,
            profile=profile,
            model_id=model_name,
            model_revision=model_revision,
            hardware_config=hw_config,
        )
        envelope = persist_phase1_3_plan(
            base_output_dir=args.output_dir,
            plan=matrix_plan,
            tasks=tasks,
            seeds=seeds,
            target_skip_rates=list(STRICT_TARGET_SKIP_RATES),
            matrix_kind=matrix_kind,
            git_sha=git_state["commit_sha"],
            environment=environment,
        )
        print(
            f"Persisted {len(matrix_plan)} {matrix_kind} cells to "
            f"{os.path.join(args.output_dir, 'matrix_plan.json')}"
        )
        print(f"Plan SHA-256: {envelope['plan_sha256']}")
        return

    bundle = load_phase1_3_plan(args.output_dir)
    dimensions = bundle["dimensions"]
    expected_pilot = bundle["envelope"]["matrix_kind"] == "pilot"
    if args.pilot is not expected_pilot:
        parser.error(
            "--pilot must be present exactly when consuming a pilot plan"
        )
    require_clean_git_state(
        repo_root,
        expected_sha=bundle["envelope"]["git_sha"],
    )
    plan_models = {
        (cell.get("model_id"), cell.get("model_revision"))
        for cell in bundle["plan"]
    }
    if plan_models != {(model_name, model_revision)}:
        raise Phase13OperationalError(
            f"persisted plan model identity drift: {plan_models!r}"
        )
    hw_config = dict(get_training_config(profile))
    planned_hw = bundle["environment"].get("hardware_config")
    if isinstance(planned_hw, dict):
        hw_config["max_samples"] = planned_hw.get("max_samples")
    current_environment = collect_phase1_3_environment(
        repo_root=repo_root,
        profile=profile,
        model_id=model_name,
        model_revision=model_revision,
        hardware_config=hw_config,
    )
    assert_environment_matches(bundle["environment"], current_environment)
    minimum_seed_count = 1 if expected_pilot else 10
    validate_phase1_3_matrix_plan(
        bundle["plan"],
        tasks=dimensions["tasks"],
        seeds=dimensions["seeds"],
        target_skip_rates=dimensions["target_skip_rates"],
        minimum_seed_count=minimum_seed_count,
        base_output_dir=args.output_dir,
    )

    if action == "validate":
        report = _validate_and_freeze_strict_matrix(bundle)
        print(
            f"Validated {report['n_valid_runs']} completed cells; "
            f"evidence: {os.path.join(args.output_dir, 'matrix_validation.json')}"
        )
        return

    if action == "run":
        assert_fresh_execution(
            bundle["plan"], base_output_dir=args.output_dir
        )
    elif args.recover_stale_running_after_hours is not None:
        recovered = recover_stale_running_attempts(
            bundle["plan"],
            base_output_dir=args.output_dir,
            stale_after_hours=args.recover_stale_running_after_hours,
        )
        print(f"Recovered {len(recovered)} stale running attempts")

    progress = scan_phase1_3_progress(
        bundle["plan"], base_output_dir=args.output_dir
    )
    pending = [item for item in progress if item["state"] == "pending"]
    print(
        f"Phase 1.3 {action}: {len(pending)} pending, "
        f"{len(progress) - len(pending)} verified completed"
    )
    wandb_group = args.wandb_group or (
        f"phase1-3-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    overall_start = time.time()
    total_runs = len(bundle["plan"])
    plan_index = {
        cell["fingerprint"]: index
        for index, cell in enumerate(bundle["plan"], start=1)
    }
    for execution_index, item in enumerate(pending, start=1):
        cell = item["cell"]
        run_idx = plan_index[cell["fingerprint"]]
        if execution_index > 1:
            elapsed = time.time() - overall_start
            remaining = len(pending) - execution_index + 1
            eta = timedelta(
                seconds=int((elapsed / (execution_index - 1)) * remaining)
            )
            print(
                f"\n  === Matrix cell {run_idx}/{total_runs} | ETA: {eta} ==="
            )
        else:
            print(f"\n  === Matrix cell {run_idx}/{total_runs} ===")
        task = cell["task"]
        task_hp = TASK_HP_OVERRIDES.get(task, {})
        cell_classification = (
            CLASSIFICATION_PILOT_NON_CLAIM
            if expected_pilot
            else CLASSIFICATION_MATCHED_CLAIM
        )
        run_ablation_single(
            task_name=task,
            seed=cell["training_seed"],
            ablation_name=cell["arm"],
            ablation_overrides=ABLATIONS[cell["arm"]],
            model_name=cell["model_id"],
            profile=profile,
            base_output_dir=args.output_dir,
            use_wandb=args.wandb,
            max_samples_override=cell["identity_inputs"].get(
                "max_samples_requested"
            ),
            run_idx=run_idx,
            total_runs=total_runs,
            wandb_project=args.wandb_project,
            wandb_group=wandb_group,
            num_epochs=task_hp.get("num_epochs", 3),
            warmup_ratio=task_hp.get("warmup_ratio", 0.1),
            early_stopping_patience=task_hp.get(
                "early_stopping_patience", 5
            ),
            metric_for_best_model=task_hp.get(
                "metric_for_best_model", "eval_loss"
            ),
            greater_is_better=task_hp.get("greater_is_better", False),
            init_from_mnli=task_hp.get("init_from_mnli", False),
            no_early_stopping=True,
            target_skip_rate=cell["target_skip_rate"],
            max_consecutive_skips=cell["max_consecutive_skips"],
            probe_interval=cell["probe_interval"],
            policy=args.policy,
            rho_veto_threshold=cell["rho_veto_threshold"],
            risk_gamma=cell["risk_gamma"],
            guard_mode=args.guard_mode,
            skip_update_mode="freeze",
            scheduler_step_policy=(
                SchedulerStepPolicy.SKIP_ON_BACKWARD_SKIP
            ),
            allow_early_stopping_with_skipping=False,
            rvd_veto_mode=args.rvd_veto_mode,
            rvd_margin_rank_floor=args.rvd_margin_rank_floor,
            rvd_spike_factor=args.rvd_spike_factor,
            rvd_spike_ema_window=args.rvd_spike_ema_window,
            rvd_repay_mode=args.rvd_repay_mode,
            rvd_repay_protect_dangerous=args.rvd_repay_protect_dangerous,
            rvd_policy_seed=None,
            provenance_classification=cell_classification,
            online_ler_mode=ONLINE_LER_MODE_AUTO,
            online_ler_parameter_sample_size=cell["online_ler_parameter_sample_size"],
            online_ler_update_interval=cell["online_ler_update_interval"],
            planned_cell=cell,
            model_revision=cell["model_revision"],
        )

    report = _validate_and_freeze_strict_matrix(bundle)
    print(
        f"Phase 1.3 matrix complete: {report['n_valid_runs']} validated cells"
    )
    print(
        f"Frozen evidence: {os.path.join(args.output_dir, 'matrix_validation.json')}"
    )


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.mode == "phase1_3":
        return _main_phase1_3(args, parser)
    if args.phase1_3_action is not None or args.pilot:
        parser.error(
            "--phase1-3-action and --pilot require --mode phase1_3"
        )
    if args.recover_stale_running_after_hours is not None:
        parser.error(
            "--recover-stale-running-after-hours requires --mode phase1_3"
        )
    if args.target_skip_rates is not None:
        parser.error("--target-skip-rates is only valid with --mode phase1_3")

    profile = detect_device_profile()

    if args.mode == "smoke":
        tasks = ["sst2"]
        seeds = [42]
        ablations_to_run = list(PHASE1_3_MATRIX)
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

    target_skip_rates = None
    legacy_target_skip_rate = (
        0.20 if args.target_skip_rate is None else args.target_skip_rate
    )
    effective_no_early_stopping = args.no_early_stopping
    effective_skip_update_mode = args.skip_update_mode
    effective_scheduler_step_policy = args.scheduler_step_policy
    effective_allow_early_stopping = (
        args.allow_early_stopping_with_skipping
    )

    effective_max_samples = args.max_samples
    if effective_max_samples is None and not args.unlimited:
        effective_max_samples = 2000 if profile != "server" else 25000

    from lerna.utils.model_loader import MODELS
    model_name = MODELS[args.model]
    wandb_group = args.wandb_group or (
        f"ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )

    model_revision = None
    if args.model == "ettin":
        model_revision = ETTIN_REVISION

    run_specs = [
        (task, seed, ablation_name, legacy_target_skip_rate, None)
        for task in tasks
        for seed in seeds
        for ablation_name in ablations_to_run
    ]

    total_runs = len(run_specs)
    print("\n  ═══════════════════════════════════════════════════════")
    print("  LERNA Ablation Study")
    print("  ═══════════════════════════════════════════════════════")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Ablations: {ablations_to_run}")
    if target_skip_rates is not None:
        print(f"  Target skip rates: {target_skip_rates}")
    print(f"  Total runs: {total_runs}")
    print(f"  Max samples/task: {effective_max_samples or 'unlimited'}")
    print("  ═══════════════════════════════════════════════════════\n")

    if args.wandb:
        _ensure_wandb_finished()

    all_results = []
    overall_start = time.time()

    for run_idx, (
        task,
        seed,
        ablation_name,
        run_target_skip_rate,
        planned_cell,
    ) in enumerate(run_specs, start=1):
        if run_idx > 1:
            elapsed = time.time() - overall_start
            avg_per_run = elapsed / (run_idx - 1)
            remaining = (total_runs - run_idx + 1) * avg_per_run
            print(
                f"\n  ═══ Run {run_idx}/{total_runs} | "
                f"ETA: {timedelta(seconds=int(remaining))} ═══"
            )
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
                early_stopping_patience=task_hp.get(
                    "early_stopping_patience", 5
                ),
                metric_for_best_model=task_hp.get(
                    "metric_for_best_model", "eval_loss"
                ),
                greater_is_better=task_hp.get("greater_is_better", False),
                init_from_mnli=task_hp.get("init_from_mnli", False),
                no_early_stopping=effective_no_early_stopping,
                target_skip_rate=run_target_skip_rate,
                max_consecutive_skips=args.max_consecutive_skips,
                probe_interval=args.probe_interval,
                policy=args.policy,
                rho_veto_threshold=args.rho_veto_threshold,
                risk_gamma=args.risk_gamma,
                guard_mode=args.guard_mode,
                skip_update_mode=effective_skip_update_mode,
                scheduler_step_policy=effective_scheduler_step_policy,
                allow_early_stopping_with_skipping=(
                    effective_allow_early_stopping
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
                online_ler_parameter_sample_size=args.online_ler_sample_size,
                online_ler_update_interval=args.online_ler_update_interval,
                planned_cell=planned_cell,
                model_revision=(
                    planned_cell.get("model_revision")
                    if planned_cell
                    else model_revision
                ),
            )
            all_results.append(result)
        except Exception as exc:
            print(
                f"  FAILED: {task} seed={seed} "
                f"ablation={ablation_name}: {exc}"
            )
            import traceback

            traceback.print_exc()
            all_results.append(
                {
                    "task": task,
                    "seed": seed,
                    "ablation": ablation_name,
                    "error": str(exc),
                }
            )
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
