#!/usr/bin/env python3
"""
LERNA Step 3B — Configurable controller ablation runner with safe dry-run mode
and CPU/mock execution-path verification.

CPU-only. Implements the experiment execution path (configuration, validation,
scheduling, mock score-source invocation) without launching real-model training,
GPU work, model/data downloads, or dependency installation. Actual execution
remains deferred to Step 4 after review.

Per STEP3B_CLI_CORRECTION_PROMPT.md section 5:
  * One explicit validated configuration drives dry-run and later execution:
    task/model/tokenizer IDs and revisions; dataset/splits and revision; probe
    exclusions/file hash; arm; controller/version; P,T,Q/rate; policy warmup;
    separate LR warmup; K; history window; anneal; seeds; precision/attention;
    optimizer/scheduler; output identity; logging.
  * Supports four gate/control P1/P4 configurations plus full and matched-count
    reduced references. Reduced steps = T - round(rate*T), with its own completed
    LR schedule; no skip gate for full/reduced arms.
  * Task-aware inputs: QNLI question/sentence, SST-2 sentence, MRPC/RTE sentence
    pairs. Explicit accuracy and MRPC F1 if supported.
  * Preserves final-checkpoint evaluation and matched QNLI exclusions. The
    original probe-ID file must come from the research repo; otherwise the exact
    blocker is recorded.
  * Explicit historical-compatible first recipe: always_step, FP32, eager,
    train batch32/eval64, accumulation1, max_length512, LR2e-5,
    weight_decay0.01, clip1.0, freeze on skips.
  * Preserves current-forward cached logits:
    capture_logits=True / online_ler_mode='legacy_dense' / no added tracker.
    No extra scoring forward.
  * Dry-run/config validation does NOT instantiate Trainer/load models/download/
    allocate CUDA. Synthetic scheduling tests are labeled as such; score_source
    is NOT ignored — CPU policy mocks/tiny logits test actual S1/uniform
    invocation paths without a pretrained model.
  * Later actual execution requires explicit invocation. Fresh runs only;
    reject untested resume. Fail once, preserve diagnostics and failed status;
    no automatic retry/seed replacement.
  * Refuse overwrite BEFORE writing manifests or artifacts. Allocate unique
    version/config/seed/attempt paths. Config identity matches actual execution;
    hash helper/gate/runner and relevant execution sources. Failed Git commands
    are treated as unknown/error, not clean.
  * Validation of counts, phase quotas, cap, horizon, terminal status, scheduler
    and LR endpoint must succeed before a completion status is written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from transformers import TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerna.utils.forward_gate import (
    ScoreQuantileGate,
    ScoreQuantileGatePolicy,
    PhaseSuffixInfeasibleError,
    TerminalInvalidScoreError,
    InfeasibleBudgetError,
    forward_gain_score,
)
from lerna.trainers.true_skip_trainer import (
    TrueBackwardSkippingTrainer,
    SchedulerStepPolicy,
    SkipPolicyDecisionError,
)

try:
    from lerna.utils.model_loader import load_model_and_tokenizer, load_tokenizer
    _MODEL_LOADER_AVAILABLE = True
except ImportError:
    _MODEL_LOADER_AVAILABLE = False


# --------------------------------------------------------------------------- #
# Provenance helpers
# --------------------------------------------------------------------------- #

def _git_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode != 0:
            return "unknown:git-failed"
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown:exception"


def _tree_status() -> str:
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode != 0:
            return "unknown:git-failed"
        return "clean" if r.stdout.strip() == "" else "dirty"
    except Exception:
        return "unknown:exception"


def _v4_prereg_sha() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "v4-prereg-r3"],
            capture_output=True, text=True,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        if r.returncode != 0:
            return "unknown:git-failed"
        sha = r.stdout.strip()
        return sha if sha and len(sha) == 40 else "unknown"
    except Exception:
        return "unknown:exception"


def _source_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except Exception:
        return "unknown"


def _config_hash(cfg: dict) -> str:
    payload = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Experiment configuration
# --------------------------------------------------------------------------- #

TASK_INPUTS = {
    "qnli": {"question_field": "question", "sentence_field": "sentence",
             "sentence_pair": True, "metric": "accuracy", "exclusions": "matched probe IDs"},
    "sst-2": {"question_field": None, "sentence_field": "sentence",
              "sentence_pair": False, "metric": "accuracy", "exclusions": None},
    "mrpc": {"question_field": "sentence1", "sentence_field": "sentence2",
             "sentence_pair": True, "metric": "f1", "exclusions": None},
    "rte": {"question_field": "sentence1", "sentence_field": "sentence2",
            "sentence_pair": True, "metric": "accuracy", "exclusions": None},
}

ARMS = ["gate_s1_p1", "gate_s1_p4", "control_uniform_p1", "control_uniform_p4",
        "full_finetune", "reduced_steps"]


def _glue_config_for_task(task: str) -> Optional[str]:
    """Map task name to the correct GLUE builder config name.

    GLUE uses ``mrpc`` for MRPC and ``sst2`` for SST-2 (not ``sst-2``).
    Other tasks use their task name as the config.
    """
    glue_config_map = {
        "mrpc": "mrpc",
        "sst-2": "sst2",
    }
    return glue_config_map.get(task, task)


@dataclass
class ExperimentConfig:
    """One explicit validated configuration driving dry-run and later execution."""

    # Task / data
    task: str = "qnli"
    model_id: str = "jhu-clsp/ettin-encoder-150m"
    model_revision: str = "main"
    tokenizer_id: str = "jhu-clsp/ettin-encoder-150m"
    tokenizer_revision: str = "main"
    dataset_id: str = "glue"
    dataset_revision: str = "main"
    splits: list[str] = field(default_factory=lambda: ["train", "validation"])
    probe_exclusions_file: Optional[str] = None
    probe_exclusions_hash: Optional[str] = None

    # Arm / controller
    arm: str = "gate_s1_p1"
    controller_version: str = "legacy_v4"
    score_source: str = "s1"
    total_steps: int = 6548
    min_step: int = 392
    n_phases: int = 1
    max_consecutive_skips: int = 4
    score_history_window: int = 64
    anneal_fraction: float = 0.88
    policy_seed: int = 0

    # Skip rate / quota
    target_skip_rate: float = 0.60

    # LR / training recipe (historical-compatible first recipe)
    lr_warmup_steps: int = 12
    lr_scheduler: str = "always_step"
    lr_schedule_shape: str = "constant_with_warmup"
    precision: str = "fp32"
    attention: str = "eager"
    train_batch_size: int = 32
    eval_batch_size: int = 64
    gradient_accumulation: int = 1
    max_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    freeze_on_skips: bool = True

    # Logits capture (historical working)
    capture_logits: bool = True
    online_ler_mode: str = "legacy_dense"
    ler_tracker: bool = False

    # Seeds / output / logging
    data_seed: int = 0  # data order seed for TinyDataLoader mock only
    output_dir: str = "/tmp/lerna_step3b_runner"
    attempt_id: Optional[str] = None
    dry_run: bool = False

    def __post_init__(self):
        if self.task not in TASK_INPUTS:
            raise ValueError(f"unknown task: {self.task!r}")
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm: {self.arm!r}")
        # Validate skip rate: negative rates and out-of-range rates rejected
        # before any rounding can hide them. Accept ints (0, 1) as well as
        # floats; bools are rejected.
        if isinstance(self.target_skip_rate, bool):
            raise ValueError(
                f"target_skip_rate must be a float or int, not bool (got {self.target_skip_rate!r})"
            )
        if not isinstance(self.target_skip_rate, (int, float)):
            raise ValueError(
                f"target_skip_rate must be a float or int (got {type(self.target_skip_rate).__name__})"
            )
        if not 0.0 <= float(self.target_skip_rate) <= 1.0:
            raise ValueError(
                f"target_skip_rate must be in [0, 1], got {self.target_skip_rate!r}"
            )
        # phase_exact_v1 does not support s2 score source.
        if (
            self.controller_version == "phase_exact_v1"
            and self.score_source == "s2"
        ):
            raise ValueError(
                "score_source='s2' is not supported for controller_version="
                "'phase_exact_v1'. Use 's1' or 'uniform'."
            )
        if self.arm in ("full_finetune", "reduced_steps"):
            # Full/reduced arms do not use a skip gate
            self.controller_version = "legacy_v4"
            self.score_source = "uniform"
        else:
            # Arm consistency: gate/control arms must match n_phases and
            # score_source. P1 = 1 phase, P4 = 4 phases; s1 = gain score,
            # uniform = random control.
            ARM_PHASES = {
                "gate_s1_p1": 1, "gate_s1_p4": 4,
                "control_uniform_p1": 1, "control_uniform_p4": 4,
            }
            ARM_SOURCE = {
                "gate_s1_p1": "s1", "gate_s1_p4": "s1",
                "control_uniform_p1": "uniform", "control_uniform_p4": "uniform",
            }
            expected_phases = ARM_PHASES[self.arm]
            expected_source = ARM_SOURCE[self.arm]
            if self.n_phases != expected_phases:
                raise ValueError(
                    f"arm={self.arm!r} requires n_phases={expected_phases}, "
                    f"got {self.n_phases}"
                )
            if self.score_source != expected_source:
                raise ValueError(
                    f"arm={self.arm!r} requires score_source={expected_source!r}, "
                    f"got {self.score_source!r}"
                )
        # Validate total_steps is an integer
        if not isinstance(self.total_steps, int):
            raise ValueError(
                f"total_steps must be an int (got {type(self.total_steps).__name__})"
            )
        # Reject configurations that would produce a negative horizon
        # (requested skips exceed total steps). This must be caught here
        # before any output path is allocated or config manifest is written.
        if self.requested_skips > self.total_steps:
            raise ValueError(
                f"requested_skips={self.requested_skips} "
                f"({self.target_skip_rate:.3f} of {self.total_steps}) "
                f"exceeds total_steps={self.total_steps}; "
                f"reduced_steps would be negative. Lower the skip rate "
                f"or increase total_steps."
            )
        # Reject zero reduced horizons (all steps skipped leaves nothing to train).
        if self.arm == "reduced_steps" and self.reduced_steps <= 0:
            raise ValueError(
                f"reduced_steps={self.reduced_steps} <= 0 for "
                f"target_skip_rate={self.target_skip_rate}, "
                f"total_steps={self.total_steps}. Lower the skip rate."
            )
        # --- LR schedule shape validation (R4-B: separate from step policy) ---
        VALID_LR_SHAPES = ("constant_with_warmup", "constant", "linear", "cosine")
        if self.lr_schedule_shape not in VALID_LR_SHAPES:
            raise ValueError(
                f"lr_schedule_shape={self.lr_schedule_shape!r} not in "
                f"{VALID_LR_SHAPES}"
            )
        # --- LR value validation (R4-B) ---
        # Reject learning_rate values that are not positive finite numbers or
        # are implausibly large for fine-tuning (e.g., 123.0). Measurement
        # records the actual LR; validation must reject it before training.
        if not isinstance(self.learning_rate, (int, float)) or not math.isfinite(
            float(self.learning_rate)
        ):
            raise ValueError(
                f"learning_rate={self.learning_rate!r} must be a finite "
                f"positive number. Real execution blocked."
            )
        if float(self.learning_rate) <= 0.0:
            raise ValueError(
                f"learning_rate={self.learning_rate!r} must be > 0. "
                f"Real execution blocked."
            )
        if float(self.learning_rate) > 1.0:
            raise ValueError(
                f"learning_rate={self.learning_rate!r} exceeds maximum "
                f"1.0 for fine-tuning. Real execution blocked."
            )
        # --- capture_logits must be honored, never silently overridden ---
        if not self.capture_logits:
            raise ValueError(
                "capture_logits=False is not supported; current-forward cached "
                "logits are required for S1 scoring without an extra scoring "
                "forward. Set capture_logits=True."
            )
        if self.attempt_id is None:
            self.attempt_id = str(uuid.uuid4())[:8]

        # --- Preflight: enforce immutable revisions ---
        # cfg.model_revision / cfg.tokenizer_revision / cfg.dataset_revision
        # must be pinned to an actual immutable revision identifier before
        # any loading/Trainer construction. Revisions are never invented;
        # callers supply them. An immutable revision is a 40-character
        # lowercase hex SHA-1 (git commit hash). Branch names ("main",
        # "develop"), tags, short hashes, and any other non-SHA strings
        # are explicitly rejected because they are mutable or ambiguous.
        # Dry-run does not load real models, so revision format is not
        # enforced in dry-run mode (real-execution blockers only apply
        # when dry_run=False).
        _SHA_RE = r"^[0-9a-f]{40}$"
        if not self.dry_run:
            for field_name, value in (
                ("model_revision", self.model_revision),
                ("tokenizer_revision", self.tokenizer_revision),
                ("dataset_revision", self.dataset_revision),
            ):
                if value is None or not isinstance(value, str):
                    raise ValueError(
                        f"{field_name} is {value!r}; revision must be a 40-character "
                        f"lowercase hex SHA (immutable revision prerequisite). "
                        f"Real execution blocked."
                    )
                if value.strip() != value:
                    raise ValueError(
                        f"{field_name} is {value!r}; revision must not contain "
                        f"leading/trailing whitespace. Real execution blocked."
                    )
                import re as _re
                if not _re.match(_SHA_RE, value):
                    raise ValueError(
                        f"{field_name} is {value!r}; revision must be a 40-character "
                        f"lowercase hex SHA (immutable commit hash). Branch names, "
                        f"tags, short hashes, and any other non-SHA strings are "
                        f"mutable or ambiguous and cannot establish an immutable "
                        f"revision prerequisite. Real execution blocked."
                    )

        # --- Preflight: reject empty or None model/tokenizer/dataset IDs ---
        # Applied in both dry-run and non-dry-run modes so that invalid IDs
        # are caught before any loader is invoked.
        for field_name, value in (
            ("model_id", self.model_id),
            ("tokenizer_id", self.tokenizer_id),
            ("dataset_id", self.dataset_id),
        ):
            if value is None or not isinstance(value, str):
                raise ValueError(
                    f"{field_name} is {value!r}; must be a non-empty string. "
                    f"Real execution blocked."
                )
            if value.strip() != value:
                raise ValueError(
                    f"{field_name} is {value!r}; must not contain "
                    f"leading/trailing whitespace."
                )
            if len(value.strip()) == 0:
                raise ValueError(
                    f"{field_name} is empty; must be a non-empty model/dataset "
                    f"identifier. Real execution blocked."
                )

        # --- Preflight: QNLI exclusion-file/hash prerequisites ---
        # QNLI arms apply the same verified matched-probe-ID exclusions across
        # all QNLI arms. Missing original exclusion file/hash is an explicit
        # real-execution blocker; exclusions/revisions are never invented.
        # Dry-run does not load real data, so exclusions are not required.
        if self.task == "qnli" and not self.dry_run:
            if self.probe_exclusions_file is None:
                raise ValueError(
                    "probe_exclusions_file is None for qnli; original "
                    "QNLI matched-probe-ID exclusion file prerequisite is "
                    "missing. Real execution blocked."
                )
            if self.probe_exclusions_hash is None:
                raise ValueError(
                    "probe_exclusions_hash is None for qnli; original "
                    "QNLI exclusion-file hash prerequisite is missing. "
                    "Real execution blocked."
                )
            exclusions_path = Path(self.probe_exclusions_file)
            if not exclusions_path.is_file():
                raise ValueError(
                    f"QNLI probe_exclusions_file does not exist: "
                    f"{self.probe_exclusions_file!r}. Original exclusion "
                    f"file prerequisite is missing. Real execution blocked."
                )
            actual_hash = _source_hash(exclusions_path)
            if actual_hash != self.probe_exclusions_hash:
                raise ValueError(
                    f"QNLI probe_exclusions_file hash mismatch: file "
                    f"{self.probe_exclusions_file!r} hashes to "
                    f"{actual_hash}, expected {self.probe_exclusions_hash!r}. "
                    f"Real execution blocked."
                )
            # Read and validate exclusion contents as a JSON probe object:
            # {"probe_s_ids": [int, ...], "probe_e_ids": [int, ...]}
            # Probe IDs are matched-probe positions in the unshuffled training
            # split. They are compared to integer GLUE idx values. The same
            # exclusion set is applied to training across all QNLI arms.
            # Validation data is left intact (not excluded).
            try:
                exclusion_text = exclusions_path.read_text(encoding="utf-8")
                probe_obj = json.loads(exclusion_text)
            except (OSError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"QNLI probe_exclusions_file cannot be read or parsed: "
                    f"{self.probe_exclusions_file!r}: {exc}. "
                    f"Real execution blocked."
                ) from exc
            if not isinstance(probe_obj, dict):
                raise ValueError(
                    f"QNLI probe_exclusions_file must contain a JSON object "
                    f"with 'probe_s_ids' and 'probe_e_ids' keys; got "
                    f"{type(probe_obj).__name__}. Real execution blocked."
                )
            probe_s_ids = probe_obj.get("probe_s_ids")
            probe_e_ids = probe_obj.get("probe_e_ids")
            if not isinstance(probe_s_ids, list) or not isinstance(probe_e_ids, list):
                raise ValueError(
                    f"QNLI probe_exclusions_file must contain list-valued "
                    f"'probe_s_ids' and 'probe_e_ids'. Real execution blocked."
                )
            for key, id_list in (("probe_s_ids", probe_s_ids), ("probe_e_ids", probe_e_ids)):
                for pid in id_list:
                    if isinstance(pid, bool) or not isinstance(pid, int) or pid < 0:
                        raise ValueError(
                            f"QNLI {key} must contain non-negative integers; "
                            f"got {pid!r}. Real execution blocked."
                        )
                if len(id_list) == 0:
                    raise ValueError(
                        f"QNLI {key} must not be empty; real execution blocked."
                    )
            # The two probe sets must be disjoint (a probe can't be both s and e).
            probe_s_set = set(probe_s_ids)
            probe_e_set = set(probe_e_ids)
            if not probe_s_set.isdisjoint(probe_e_set):
                raise ValueError(
                    "QNLI probe_s_ids and probe_e_ids must be disjoint; "
                    f"overlap: {sorted(probe_s_set & probe_e_set)}. "
                    "Real execution blocked."
                )
            # Store combined probe IDs for the production loader to apply.
            self._qnli_exclusion_probe_ids = probe_s_set | probe_e_set

    @property
    def requested_skips(self) -> int:
        return int(round(self.target_skip_rate * self.total_steps))

    @property
    def reduced_steps(self) -> int:
        return self.total_steps - self.requested_skips

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Output path allocation (refuse overwrite)
# --------------------------------------------------------------------------- #

def allocate_output_dir(cfg: ExperimentConfig) -> Path:
    """Allocate a unique version/config/seed/attempt path. Refuse overwrite."""
    base = Path(cfg.output_dir)
    # Config identity hash
    cfg_hash = _config_hash(cfg.to_dict())
    out = base / cfg.task / cfg.arm / f"v{cfg_hash}" / f"seed{cfg.policy_seed}" / f"attempt{cfg.attempt_id}"
    if out.exists():
        raise FileExistsError(
            f"Output path already exists: {out}. Refusing to overwrite. "
            f"Use a different attempt_id, seed, or config."
        )
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# CPU mock score sources (no pretrained model)
# --------------------------------------------------------------------------- #

class TinyLogits:
    """Tiny logits tensor for CPU mock testing of S1 score path."""

    def __init__(self, batch: int = 4, classes: int = 2, seed: int = 0):
        rng = random.Random(seed)
        self.data = [[rng.random() for _ in range(classes)] for _ in range(batch)]
        self.detached = True

    def detach(self):
        return self

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]) if self.data else 0)

    def dim(self):
        return len(self.shape)

    def float(self):
        return self

    def max(self, dim=-1, keepdim=True):
        class _Max:
            values = [[max(row) for row in self.data]]
        return _Max()

    def exp(self):
        import math
        return TinyLogits.from_data(
            [[math.exp(v) for v in row] for row in self.data]
        )

    @classmethod
    def from_data(cls, data):
        obj = cls.__new__(cls)
        obj.data = data
        obj.detached = True
        return obj

    def sum(self, dim=-1, keepdim=True):
        class _Sum:
            values = [[sum(row)] for row in self.data]
        return _Sum()

    def __sub__(self, other):
        return TinyLogits.from_data(
            [[a - b for a, b in zip(ra, rb)] for ra, rb in zip(self.data, other.data)]
        )

    def pow(self, p):
        return TinyLogits.from_data(
            [[v ** p for v in row] for row in self.data]
        )

    def sqrt(self):
        import math
        return TinyLogits.from_data(
            [[math.sqrt(v) for v in row] for row in self.data]
        )

    def mean(self):
        total = sum(v for row in self.data for v in row)
        count = sum(len(row) for row in self.data)
        return total / count if count else 0.0

    def item(self):
        return self.mean()


def mock_forward_gain_score(logits, labels) -> float:
    """CPU mock of forward_gain_score using TinyLogits (no torch).

    Returns a deterministic finite score in (0, 1) based on the logits data,
    suitable for testing the S1 invocation path without a pretrained model.
    """
    if logits is None or labels is None:
        return float("nan")
    try:
        total = 0.0
        count = 0
        for row in logits.data:
            for v in row:
                total += abs(v)
                count += 1
        score = (total / count) if count else 0.5
        # Map to (0, 1) range
        score = score / (1.0 + score)
        if not (score == score and score != float("inf") and score != float("-inf")):
            return float("nan")
        return score
    except Exception:
        return float("nan")


def mock_uniform_score(rng: random.Random):
    def _src(decision_index, model, inputs):
        return rng.random()
    return _src


# --------------------------------------------------------------------------- #
# Dry-run validation (no Trainer, no model, no GPU)
# --------------------------------------------------------------------------- #

def run_dry_run(cfg: ExperimentConfig) -> dict:
    """Validate configuration without training.

    Does NOT instantiate Trainer, load models, download data, or allocate CUDA.
    Tests the actual gate/policy wiring with CPU mock score sources.
    """
    # --- Validate config before any execution ---
    if cfg.arm in ("full_finetune", "reduced_steps"):
        # No gate for full/reduced arms
        return _dry_run_reference_arm(cfg)

    # Build the gate directly (bypasses policy's torch-dependent _source closure)
    gate = ScoreQuantileGate(
        rate=cfg.target_skip_rate,
        total_steps=cfg.total_steps,
        min_step=cfg.min_step,
        max_consecutive_skips=cfg.max_consecutive_skips,
        n_phases=cfg.n_phases,
        score_history_window=cfg.score_history_window,
        anneal_fraction=cfg.anneal_fraction,
        seed=cfg.policy_seed + 1,
        score_source=_make_score_source(cfg),
        controller_version=cfg.controller_version,
    )

    # Run full-horizon decide() sweep with synthetic scores
    actions = []
    t0 = time.perf_counter()
    try:
        for t in range(cfg.total_steps):
            a = gate.decide(t, None, None)
            actions.append(1 if a else 0)
    except PhaseSuffixInfeasibleError as e:
        return {
            "status": "REJECTED",
            "reason": "infeasible_suffix",
            "config": cfg.to_dict(),
            "error": str(e),
            "provenance": _provenance(dry_run=True),
        }

    wall = time.perf_counter() - t0
    diag = gate.diagnostics()

    # --- Validation before completion status ---
    requested = diag["requested_total"]
    realized = diag["realized_total"]
    errors = []

    if cfg.controller_version == "phase_exact_v1":
        # Validate diagnostic presence and container types before indexing.
        phase_used = diag.get("phase_used", None)
        phase_quota = diag.get("phase_quota", None)

        if phase_used is None or phase_quota is None:
            errors.append(
                "phase_exact_v1: phase_used or phase_quota missing from "
                "diagnostics; structured failure before evaluation."
            )
        elif not isinstance(phase_used, list) or not isinstance(
            phase_quota, list
        ):
            errors.append(
                "phase_exact_v1: phase_used and phase_quota must be lists; "
                f"got {type(phase_used).__name__} and "
                f"{type(phase_quota).__name__}."
            )
        else:
            # Validate nonnegative integer values.
            for name, arr in (("phase_used", phase_used), ("phase_quota", phase_quota)):
                for i, v in enumerate(arr):
                    if not isinstance(v, int) or isinstance(v, bool):
                        errors.append(
                            f"phase_exact_v1: {name}[{i}] is not an integer: "
                            f"{v!r} ({type(v).__name__})."
                        )
                    elif v < 0:
                        errors.append(
                            f"phase_exact_v1: {name}[{i}] is negative: {v}."
                        )
            # Validate equal lengths.
            if len(phase_used) != len(phase_quota):
                errors.append(
                    f"phase_exact_v1: phase_used length ({len(phase_used)}) "
                    f"!= phase_quota length ({len(phase_quota)})."
                )
            # Validate configured phase count.
            expected_n_phases = cfg.n_phases
            if len(phase_quota) != expected_n_phases:
                errors.append(
                    f"phase_exact_v1: phase_quota length ({len(phase_quota)}) "
                    f"!= configured n_phases ({expected_n_phases})."
                )
            # Validate totals: sum of phase_quota must equal total requested skips (Q),
            # not total_steps (T). phase_quota allocates skips across phases.
            total_quota = sum(phase_quota)
            if total_quota != cfg.requested_skips:
                errors.append(
                    f"phase_exact_v1: sum(phase_quota)={total_quota} "
                    f"!= requested_skips={cfg.requested_skips} (Q). "
                    f"phase_quota allocates skips across phases, not training steps."
                )
            # Validate per-phase usage matches quota exactly.
            for i in range(len(phase_quota)):
                if phase_used[i] != phase_quota[i]:
                    errors.append(
                        f"phase {i} used={phase_used[i]} != quota={phase_quota[i]}"
                    )
        # Requested vs realized skips must match exactly.
        if realized != requested:
            errors.append(
                f"realized={realized} != requested={requested}"
            )

    # General phase_used vs phase_quota validation (all controller versions):
    # a phase cannot use more skips than its quota allows.
    phase_used = diag.get("phase_used", None)
    phase_quota = diag.get("phase_quota", None)
    if phase_used is not None and phase_quota is not None:
        if isinstance(phase_used, list) and isinstance(phase_quota, list):
            for i in range(min(len(phase_used), len(phase_quota))):
                pu = phase_used[i]
                pq = phase_quota[i]
                if not isinstance(pu, int) or isinstance(pu, bool) or not isinstance(pq, int) or isinstance(pq, bool):
                    # Type issues already reported above; skip comparison.
                    continue
                if pu > pq:
                    errors.append(
                        f"phase {i} used={pu} > quota={pq}"
                    )

    if diag["longest_skip_run"] > cfg.max_consecutive_skips:
        errors.append(f"longest_skip_run={diag['longest_skip_run']} > K={cfg.max_consecutive_skips}")

    if errors:
        return {
            "status": "FAILED",
            "reason": "; ".join(errors),
            "config": cfg.to_dict(),
            "diagnostics": diag,
            "provenance": _provenance(dry_run=True),
        }

    return {
        "status": "OK",
        "config": cfg.to_dict(),
        "config_hash": _config_hash(cfg.to_dict()),
        "result": {
            "realized_total": realized,
            "requested_total": requested,
            "phase_used": list(diag["phase_used"]),
            "phase_quota": list(diag["phase_quota"]),
            "eligible_per_window": list(diag["eligible_per_window"]),
            "longest_skip_run": diag["longest_skip_run"],
            "decisions_seen": diag["decisions_seen"],
            "suffix_force_skip": diag["suffix_force_skip"],
            "suffix_force_retain": diag["suffix_force_retain"],
            "wall_seconds": wall,
        },
        "diagnostics": diag,
        "provenance": _provenance(dry_run=True),
    }

# --------------------------------------------------------------------------- #
# Mock execution path components (tiny CPU states, no pretrained model)
# --------------------------------------------------------------------------- #

class TinyInstr:
    """Mock instruction counter for the Trainer."""
    def __init__(self):
        self.batches_seen = 0


class TinyModel(torch.nn.Module):
    """Tiny CPU model with a classifier layer (extends nn.Module so
    named_parameters() and submodule attribute access work natively).

    Returns dict-like outputs with .loss and .logits so HF Trainer's
    compute_loss() can consume them directly.
    """

    def __init__(self, hidden: int = 8, classes: int = 2, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.classifier = torch.nn.Linear(hidden, classes)
        self._hidden = hidden

    def forward(self, input_ids, labels=None, **kwargs):
        """Produce logits and loss from a batch of input ids and labels."""
        import torch as _torch
        h = _torch.randn(len(input_ids), self._hidden)
        logits = self.classifier(h)
        loss = None
        if labels is not None:
            labels = labels.view(-1) if labels.dim() > 1 else labels
            if labels.dim() == 0:
                labels = labels.unsqueeze(0)
            loss = _torch.nn.functional.cross_entropy(logits, labels)
        return type("Outputs", (), {"loss": loss, "logits": logits})()


class TinyTrainer:
    """Mock Trainer exposing the interfaces the policy/score sources consume:
    optimizer, scheduler, instr.batches_seen, _last_real_logits,
    _alignment_probe.
    """

    def __init__(self, model, lr: float = 2e-5):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1)
        self.instr = TinyInstr()
        self._last_real_logits = None
        self._alignment_probe = None
        self.optimizer_step_calls = 0
        self.scheduler_step_calls = 0

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()
        self.optimizer_step_calls += 1

    def scheduler_step(self):
        self.scheduler.step()
        self.scheduler_step_calls += 1


class TinyDataLoader:
    """Mock data loader yielding tiny CPU batches (dicts with input_ids and
    labels as torch tensors). Each item is a single example; HF Trainer's
    DataLoader collates them into batches.
    """

    def __init__(self, num_examples: int, hidden: int = 8, classes: int = 2,
                 seed: int = 0):
        self.num_examples = num_examples
        self.hidden = hidden
        self.classes = classes
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        for _ in range(self.num_examples):
            input_ids = [rng.random() for _ in range(self.hidden)]
            label = rng.randint(0, self.classes - 1)
            yield {
                "input_ids": _to_tensor(input_ids),
                "labels": _to_tensor(label, dtype=torch.long),
            }


def _to_tensor(data, dtype=None):
    """Convert a nested list or flat list to a torch tensor."""
    t = torch.tensor(data, dtype=dtype)
    return t


def _dynamic_padding_collator(features, pad_token_id: int = 0):
    """Dynamic padding collator: pads input_ids to max length in batch.

    For GLUE sentence-pair tasks (MRPC, RTE), examples have variable-length
    tokenized sequences. This collator pads input_ids to the longest sequence
    in the batch (not to a fixed max_length). Labels are stacked as-is with
    no padding — one label per example, shape (batch,).
    No max_length truncation is applied here — truncation happens in the
    tokenizer (max_length=cfg.max_length) before the DataLoader sees the data.

    input_ids are padded with pad_token_id; attention_mask and token_type_ids
    are padded with 0. Labels are never padded.

    Args:
        features: list of feature dicts from the dataset.
        pad_token_id: tokenizer's pad token id used for input_ids padding.
            Defaults to 0 for backward compatibility with CPU mocks that
            don't have a real tokenizer.
    """
    if not features:
        return {}
    batch = {}
    keys = features[0].keys()
    # Compute max length across all 1D sequence tensors (input_ids, attention_mask, etc.)
    seq_lens = []
    for f in features:
        for v in f.values():
            if isinstance(v, torch.Tensor) and v.ndim >= 1:
                seq_lens.append(v.shape[0])
    max_seq_len = max(seq_lens) if seq_lens else 0
    for key in keys:
        values = [f[key] for f in features]
        if not isinstance(values[0], torch.Tensor):
            batch[key] = values
            continue
        ndim = values[0].ndim
        if ndim == 0:
            # Scalar labels (class labels) — stack as-is, no padding.
            batch[key] = torch.stack(values, dim=0)
        elif ndim == 1:
            if key == "labels":
                # Labels are stacked as-is — one label per example, shape (batch,).
                batch[key] = torch.stack(values, dim=0)
            else:
                # Sequences (input_ids, attention_mask, token_type_ids) — pad to max_seq_len.
                pad_val = pad_token_id if key == "input_ids" else 0
                padded = []
                for v in values:
                    if v.shape[0] < max_seq_len:
                        pad_len = max_seq_len - v.shape[0]
                        padding = torch.full((pad_len,), pad_val, dtype=v.dtype, device=v.device)
                        v = torch.cat([v, padding], dim=0)
                    padded.append(v)
                batch[key] = torch.stack(padded, dim=0)
        else:
            batch[key] = torch.stack(values, dim=0)
    return batch


# --------------------------------------------------------------------------- #
# Real callable execution path
# --------------------------------------------------------------------------- #

def run_execution(cfg: ExperimentConfig) -> dict:
    """Execute the controller sweep through TrueBackwardSkippingTrainer with
    mocked external loading (TinyModel, TinyDataLoader) and tiny CPU states.

    Real pretrained training is NOT invoked; no GPU, no model downloads, no
    dependency installation. This is a distinct code path from dry-run
    validation: provenance records dry_run=false and execution_mode =
    "mocked_loading_cpu".

    The execution wires ScoreQuantileGatePolicy.should_skip() into the
    TrueBackwardSkippingTrainer training loop:
    forward pass -> policy decision -> (skip or backward+optimizer+scheduler)
    -> final evaluation.

    With SchedulerStepPolicy.ALWAYS_STEP:
      forwards  = T (every scheduled batch gets a forward pass)
      backwards = T - Q (only non-skipped batches get backward)
      scheduler = T (scheduler advances on every batch)
    """
    # --- Validate config before any execution ---
    # Determine effective total steps: full_finetune uses all, reduced_steps
    # uses reduced total, gate/control arms use total_steps with skip gate.
    if cfg.arm == "full_finetune":
        effective_total = cfg.total_steps
        skip_policy_obj = None  # AlwaysFalsePolicy default
        use_gate = False
    elif cfg.arm == "reduced_steps":
        effective_total = cfg.reduced_steps
        skip_policy_obj = None  # AlwaysFalsePolicy default
        use_gate = False
    else:
        effective_total = cfg.total_steps
        skip_policy_obj = None  # set below
        use_gate = True

    # --- External loading boundaries (injectable for CPU tests) ---
    # Loader, constructor, and evaluator exceptions are caught below and
    # returned as structured FAILED manifests before evaluation (R4-C).
    # Production loading failures do NOT fall back to synthetic TinyModel/
    # TinyDataLoader — they produce a stage-specific FAILED manifest.
    # --- Seed HF global randomness before any model/data loads ---
    import numpy as np
    import torch
    # Single run seed: cfg.policy_seed drives torch/np seeding, TrainingArguments.seed,
    # and the policy. data_seed is used only by the TinyDataLoader mock.
    seed = cfg.policy_seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Record which config field supplied the seed for manifest provenance
    seed_source = "cfg.policy_seed"
    try:
        if cfg.dry_run:
            model = _load_model(cfg)
        else:
            model = _load_model_production(cfg)
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"model loading failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "model_loading",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }
    try:
        if cfg.dry_run:
            tokenizer = _load_tokenizer(cfg)
        else:
            tokenizer = _load_tokenizer_production(cfg)
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"tokenizer loading failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "tokenizer_loading",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }
# Use the tokenizer's pad_token_id for padding (instead of hardcoded 0)
    # This ensures the collator respects the tokenizer's actual padding token
    _padding_kwargs = {}
    if tokenizer is not None and hasattr(tokenizer, 'pad_token_id'):
        _pad_token_id = tokenizer.pad_token_id
        if _pad_token_id is None:
            return {
                "status": "FAILED",
                "reason": "tokenizer pad_token_id is None; cannot pad input_ids. "
                          "Real execution blocked.",
                "config": cfg.to_dict(),
                "failure_stage": "tokenizer_pad_id",
                "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
            }
        _padding_kwargs['pad_token_id'] = _pad_token_id

    # B1: reject tokenizers that pad on the left or any side other than "right".
    # The dynamic padding collator only pads on the right; a left-padding
    # tokenizer would write pad tokens at the start of each sequence and break
    # attention-mask alignment.
    if tokenizer is not None and hasattr(tokenizer, 'padding_side'):
        if tokenizer.padding_side != "right":
            return {
                "status": "FAILED",
                "reason": f"tokenizer padding_side={tokenizer.padding_side!r}; only 'right' is supported. "
                          "Real execution blocked.",
                "config": cfg.to_dict(),
                "failure_stage": "tokenizer_padding_side",
                "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
            }

    
    try:
        if cfg.dry_run:
            train_dataset = _load_train_dataset(cfg, effective_total)
        else:
            train_dataset = _load_train_dataset_production(cfg, effective_total)
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"train dataset loading failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "train_dataset_loading",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }
    try:
        if cfg.dry_run:
            eval_dataset = _load_eval_dataset(cfg)
        else:
            eval_dataset = _load_eval_dataset_production(cfg)
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"eval dataset loading failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "eval_dataset_loading",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    class _TinyDataset(torch.utils.data.Dataset):
        def __init__(self, batches):
            self.batches = batches
        def __len__(self):
            return len(self.batches)
        def __getitem__(self, idx):
            return self.batches[idx]

    if isinstance(train_dataset, list):
        train_dataset = _TinyDataset(train_dataset)

    # TrainingArguments for CPU-only execution; no eval, no saving, no logging.
    # Values come from config so provenance matches execution.
    #
    # R4-B: lr_schedule_shape (schedule shape: constant_with_warmup, constant,
    # linear, cosine) is separate from SchedulerStepPolicy.ALWAYS_STEP (which
    # only controls whether the scheduler steps on skipped batches). The shape
    # is mapped to HF lr_scheduler_type; the step policy is passed separately
    # to TrueBackwardSkippingTrainer.
    LR_SHAPE_TO_HF = {
        "constant_with_warmup": "constant_with_warmup",
        "constant": "constant",
        "linear": "linear",
        "cosine": "cosine",
    }
    lr_scheduler_type = LR_SHAPE_TO_HF[cfg.lr_schedule_shape]

    # --- Build TrainingArguments (wrapped for structured failure) ---
    try:
        training_args = TrainingArguments(
            output_dir=cfg.output_dir,
            max_steps=effective_total,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.train_batch_size,
            gradient_accumulation_steps=cfg.gradient_accumulation,
            warmup_steps=cfg.lr_warmup_steps,
            weight_decay=cfg.weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            max_grad_norm=cfg.grad_clip,
            optim="adamw_torch",
            eval_strategy="no",
            save_strategy="no",
            logging_strategy="no",
            report_to=[],
            disable_tqdm=True,
            use_cpu=True,
            seed=cfg.policy_seed,
            fp16=False,
            bf16=False,
        )
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"training arguments construction failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "training_arguments",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    # --- Build the real policy as the skip policy ---
    from lerna.trainers.policies import AlwaysFalsePolicy
    try:
        if use_gate:
            policy = ScoreQuantileGatePolicy(
                target_skip_rate=cfg.target_skip_rate,
                total_steps=effective_total,
                min_step=cfg.min_step,
                max_consecutive_skips=cfg.max_consecutive_skips,
                n_phases=cfg.n_phases,
                score_history_window=cfg.score_history_window,
                anneal_fraction=cfg.anneal_fraction,
                policy_seed=cfg.policy_seed,
                score_source=cfg.score_source,
                controller_version=cfg.controller_version,
            )
            # Attach alignment probe before training (B3) when s2 is used.
            if cfg.score_source == "s2":
                policy.setup_probe(None, model)  # trainer attached by should_skip()
        else:
            policy = AlwaysFalsePolicy()
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"policy construction failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "policy_construction",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    # --- Instantiate TrueBackwardSkippingTrainer with mocked loading ---
    try:
        trainer = TrueBackwardSkippingTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            skip_policy=policy,
            scheduler_step_policy=SchedulerStepPolicy.ALWAYS_STEP,
            capture_logits=True,
            online_ler_enabled=(cfg.online_ler_mode != "off"),
            allow_grad_accumulation_with_skipping=True,
            data_collator=lambda features, _pd=_padding_kwargs.get('pad_token_id', 0): _dynamic_padding_collator(features, pad_token_id=_pd),
        )
        # capture_logits is coupled to online_ler_enabled in the constructor;
        # enable it independently so _last_real_logits is populated for S1 scoring.
        trainer.capture_logits = True
        trainer.instr.capture_logits_enabled = True
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"trainer construction failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "trainer_construction",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    # --- Execution loop via trainer.train() ---
    t0 = time.perf_counter()
    try:
        trainer.train()
    except (PhaseSuffixInfeasibleError, TerminalInvalidScoreError) as e:
        return {
            "status": "REJECTED",
            "reason": "terminal",
            "config": cfg.to_dict(),
            "error": str(e),
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    except SkipPolicyDecisionError as e:
        return {
            "status": "REJECTED",
            "reason": "policy_decision_failure",
            "config": cfg.to_dict(),
            "error": str(e),
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"training failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "training",
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }
    wall = time.perf_counter() - t0
    instr = trainer.instr

    # --- Validation before completion status ---
    if use_gate:
        gate_diag = policy.get_diagnostics()
        requested = cfg.requested_skips
        realized = instr.skipped_batches
        errors = []

        # R4-C: phase_exact_v1 exact-state reconciliation.
        if cfg.controller_version == "phase_exact_v1":
            # Requested vs realized skips must match exactly.
            if realized != requested:
                errors.append(
                    f"realized_skips={realized} != requested_skips={requested}"
                )
            # Exact per-phase quota usage.
            phase_used = gate_diag.get("phase_used", None)
            phase_quota = gate_diag.get("phase_quota", None)

            # Validate diagnostic presence and container types before indexing.
            if phase_used is None or phase_quota is None:
                errors.append(
                    "phase_exact_v1: phase_used or phase_quota missing from "
                    "diagnostics; structured failure before evaluation."
                )
            elif not isinstance(phase_used, list) or not isinstance(
                phase_quota, list
            ):
                errors.append(
                    "phase_exact_v1: phase_used and phase_quota must be lists; "
                    f"got {type(phase_used).__name__} and "
                    f"{type(phase_quota).__name__}."
                )
            else:
                # Validate nonnegative integer values.
                for name, arr in (("phase_used", phase_used), ("phase_quota", phase_quota)):
                    for i, v in enumerate(arr):
                        if not isinstance(v, int) or isinstance(v, bool):
                            errors.append(
                                f"phase_exact_v1: {name}[{i}] is not an integer: "
                                f"{v!r} ({type(v).__name__})."
                            )
                        elif v < 0:
                            errors.append(
                                f"phase_exact_v1: {name}[{i}] is negative: {v}."
                            )
                # Validate equal lengths.
                if len(phase_used) != len(phase_quota):
                    errors.append(
                        f"phase_exact_v1: phase_used length ({len(phase_used)}) "
                        f"!= phase_quota length ({len(phase_quota)})."
                    )
                # Validate configured phase count.
                expected_n_phases = cfg.n_phases
                if len(phase_quota) != expected_n_phases:
                    errors.append(
                        f"phase_exact_v1: phase_quota length ({len(phase_quota)}) "
                        f"!= configured n_phases ({expected_n_phases})."
                    )
                # Validate totals: sum of phase_quota must equal total requested skips (Q),
                # not the effective training horizon (T).
                total_quota = sum(phase_quota)
                if total_quota != requested:
                    errors.append(
                        f"phase_exact_v1: sum(phase_quota)={total_quota} "
                        f"!= requested_skips={requested} (Q). "
                        f"phase_quota allocates skips across phases, not training steps."
                    )
                # Validate per-phase usage matches quota exactly.
                for i in range(len(phase_quota)):
                    if phase_used[i] != phase_quota[i]:
                        errors.append(
                            f"phase {i} used={phase_used[i]} != quota={phase_quota[i]}"
                        )
            # Completed valid-decision horizon: decisions_seen must equal
            # effective_total (every scheduled batch was decided).
            decisions_seen = gate_diag.get("decisions_seen", 0)
            if decisions_seen != effective_total:
                errors.append(
                    f"decisions_seen={decisions_seen} != effective_total={effective_total}"
                )
            # Nonterminal state: gate must not be terminal at completion.
            if gate_diag.get("terminal", False):
                errors.append(
                    f"gate is terminal at decision {gate_diag.get('terminal_at', 'unknown')}"
                )
            # Optimizer/scheduler counts and LR endpoint.
            if instr.optimizer_step_attempts != instr.backward_calls:
                errors.append(
                    f"optimizer_step_attempts={instr.optimizer_step_attempts} != "
                    f"backward_calls={instr.backward_calls}"
                )
            if instr.scheduler_step_calls != effective_total:
                errors.append(
                    f"scheduler_step_calls={instr.scheduler_step_calls} != "
                    f"effective_total={effective_total}"
                )
        else:
            # legacy_v4: realized may under-spend (preserved behavior).
            if realized != requested:
                errors.append(
                    f"realized_skips={realized} != requested_skips={requested}"
                )

        longest_skip_run = gate_diag.get("longest_skip_run", 0)
        if longest_skip_run > cfg.max_consecutive_skips:
            errors.append(f"longest_skip_run={longest_skip_run} > K={cfg.max_consecutive_skips}")
    else:
        gate_diag = None
        requested = 0
        realized = 0
        errors = []
        longest_skip_run = 0

    # --- EXACT completion validation (not upper bounds) ---
    # Forwards must equal effective_total exactly.
    if instr.forward_calls != effective_total:
        errors.append(
            f"forward_calls={instr.forward_calls} != effective_total={effective_total}"
        )
    # Backwards must equal effective_total - realized exactly.
    expected_backward = effective_total - realized
    if instr.backward_calls != expected_backward:
        errors.append(
            f"backward_calls={instr.backward_calls} != "
            f"effective_total - realized={expected_backward}"
        )
    # Scheduler advances must equal effective_total exactly (ALWAYS_STEP).
    if instr.scheduler_step_calls != effective_total:
        errors.append(
            f"scheduler_step_calls={instr.scheduler_step_calls} != "
            f"effective_total={effective_total}"
        )
    # Optimizer updates must equal backward_calls exactly.
    if instr.optimizer_step_attempts != instr.backward_calls:
        errors.append(
            f"optimizer_step_attempts={instr.optimizer_step_attempts} != "
            f"backward_calls={instr.backward_calls}"
        )
    # Invariants (must also hold exactly).
    if instr.forward_calls != instr.backward_calls + instr.skipped_backward_steps:
        errors.append(
            f"forward_calls={instr.forward_calls} != "
            f"backward_calls={instr.backward_calls} + skipped={instr.skipped_backward_steps}"
        )
    if instr.optimizer_step_attempts > instr.backward_calls:
        errors.append(
            f"optimizer_step_attempts={instr.optimizer_step_attempts} > "
            f"backward_calls={instr.backward_calls}"
        )
    if instr.scheduler_step_calls > instr.optimizer_step_attempts + instr.skipped_backward_steps:
        errors.append(
            f"scheduler_step_calls={instr.scheduler_step_calls} > opportunities="
            f"{instr.optimizer_step_attempts + instr.skipped_backward_steps}"
        )

    # --- LR endpoint validation (R4-B) ---
    # Validate the measured final LR matches the expected schedule endpoint
    # derived from schedule shape, warmup, and effective arm horizon.
    try:
        final_lr = float(trainer.optimizer.param_groups[0]["lr"])
    except Exception:
        final_lr = None
    if final_lr is None:
        errors.append(
            "final LR is unavailable (optimizer has no param_groups). "
            "Real execution blocked."
        )
    elif not math.isfinite(final_lr):
        errors.append(
            f"final LR {final_lr} is not finite. Real execution blocked."
        )
    else:
        lr_schedule_shape = cfg.lr_schedule_shape
        learning_rate = cfg.learning_rate
        warmup_steps = cfg.lr_warmup_steps
        if lr_schedule_shape in ("constant_with_warmup", "constant", "linear", "cosine"):
            if lr_schedule_shape == "constant":
                # Constant schedule: no warmup, LR stays at peak throughout.
                expected_lr = learning_rate
                tol = max(abs(expected_lr) * 1e-9, 1e-12)
            else:
                # Schedules with warmup: LR ramps from 0 to peak over
                # warmup_steps. Validation uses a formula, not HF's scheduler.
                warmup_progress = min(effective_total / max(warmup_steps, 1), 1.0)
                warmup_lr = learning_rate * warmup_progress
                if warmup_progress < 1.0:
                    # Still in warmup: expect warmup progress * peak.
                    expected_lr = warmup_lr
                    tol = max(abs(expected_lr) * 1e-9, 1e-12)
                elif lr_schedule_shape == "constant_with_warmup":
                    expected_lr = learning_rate
                    tol = max(abs(expected_lr) * 1e-9, 1e-12)
                elif lr_schedule_shape == "cosine":
                    # HF cosine: at T == W warmup just completed, LR at peak.
                    if effective_total <= warmup_steps:
                        expected_lr = learning_rate
                    else:
                        decay_horizon = max(effective_total - warmup_steps, 1)
                        decay_progress = min(
                            (effective_total - warmup_steps) / max(decay_horizon, 1),
                            1.0,
                        )
                        # cosine decay factor computed from the formula
                        import math as _m
                        expected_lr = learning_rate * 0.5 * (1.0 + _m.cos(_m.pi * decay_progress))
                    tol = max(abs(expected_lr) * 1e-9, 1e-12)
                else:  # linear
                    if effective_total <= warmup_steps:
                        expected_lr = 0.0
                    else:
                        decay_horizon = max(effective_total - warmup_steps, 1)
                        decay_progress = min(
                            (effective_total - warmup_steps) / max(decay_horizon, 1),
                            1.0,
                        )
                        expected_lr = learning_rate * (1.0 - decay_progress)
                    tol = max(abs(expected_lr) * 1e-9, 1e-12)
            if abs(final_lr - expected_lr) > tol:
                errors.append(
                    f"final LR {final_lr} != expected {expected_lr} for "
                    f"lr_schedule_shape={lr_schedule_shape!r} "
                    f"(warmup_steps={warmup_steps}, effective_total={effective_total})"
                )

    # --- Return validation failures BEFORE evaluation (R4-C) ---
    if errors:
        instr_diag = instr.as_dict()
        if gate_diag:
            instr_diag["longest_skip_run"] = gate_diag.get("longest_skip_run", 0)
            instr_diag["phase_used"] = gate_diag.get("phase_used", [])
            instr_diag["phase_quota"] = gate_diag.get("phase_quota", [])
            instr_diag["decisions_seen"] = gate_diag.get("decisions_seen", 0)
            instr_diag["terminal"] = gate_diag.get("terminal", False)
        return {
            "status": "FAILED",
            "reason": "; ".join(errors),
            "config": cfg.to_dict(),
            "diagnostics": instr_diag,
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    # --- Final evaluation: actual task accuracy on eval dataset ---
    _pad_token_id = _padding_kwargs.get('pad_token_id', 0)
    try:
        final_eval_loss, final_eval_acc, final_eval_f1 = _evaluate_model(
            trainer, model, eval_dataset, cfg, pad_token_id=_pad_token_id
        )
    except RuntimeError as e:
        if "eval_data_loader construction failed" in str(e):
            return {
                "status": "FAILED",
                "reason": str(e),
                "config": cfg.to_dict(),
                "failure_stage": "eval_data_loader_construction",
                "diagnostics": instr.as_dict(),
                "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
            }
        raise
    except Exception as e:
        return {
            "status": "FAILED",
            "reason": f"evaluation failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "failure_stage": "evaluation",
            "diagnostics": instr.as_dict(),
            "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
        }

    return {
        "status": "OK",
        "config": cfg.to_dict(),
        "config_hash": _config_hash(cfg.to_dict()),
        "result": {
            "arm": cfg.arm,
            "realized_total": realized,
            "requested_total": requested,
            "effective_total": effective_total,
            "forwards": instr.forward_calls,
            "backwards": instr.backward_calls,
            "optimizer_updates": instr.optimizer_step_attempts,
            "scheduler_advances": instr.scheduler_step_calls,
            "longest_skip_run": longest_skip_run,
            "skipped": instr.skipped_batches,
            "retained": instr.backward_calls,
            "wall_seconds": wall,
            "final_eval_loss": final_eval_loss,
            "final_eval_accuracy": final_eval_acc,
            "final_eval_f1": final_eval_f1,
            # R4-B: LR schedule shape and endpoint info
            "lr_schedule_shape": cfg.lr_schedule_shape,
            "lr_scheduler_step_policy": "always_step",
            "grad_clip_max_grad_norm": cfg.grad_clip,
            "lr_warmup_steps": cfg.lr_warmup_steps,
            "learning_rate": cfg.learning_rate,
            # Measured final LR from the optimizer after training completes.
            "final_learning_rate": (
                float(trainer.optimizer.param_groups[0]["lr"])
                if trainer.optimizer is not None
                else None
            ),
            # Tokenizer loading boundary result (None in CPU mock path;
            # production tokenizer loaded via _load_tokenizer_production).
            "tokenizer_loaded": tokenizer is not None,
        },
        "diagnostics": instr.as_dict(),
        "provenance": _provenance(dry_run=cfg.dry_run, seed=cfg.policy_seed, seed_source=seed_source),
    }


# --------------------------------------------------------------------------- #
# Injectable external loading boundaries (CPU-only mocks in tests)
# --------------------------------------------------------------------------- #

def _load_model(cfg: ExperimentConfig, model_loader=None):
    """Load the model for execution.

    If ``model_loader`` is provided (CPU tests pass TinyModel factories here),
    it is called as ``model_loader(cfg)`` and its result returned. Otherwise
    the ordinary loading path returns a TinyModel CPU fixture — no pretrained
    model is downloaded or instantiated in this step.
    """
    if model_loader is not None:
        return model_loader(cfg)
    return TinyModel(seed=cfg.policy_seed)


def _load_tokenizer(cfg: ExperimentConfig, tokenizer_loader=None):
    """Load the tokenizer for execution.

    If ``tokenizer_loader`` is provided it is called as
    ``tokenizer_loader(cfg)``. Otherwise returns None — tokenization is
    handled by TinyModel/TinyDataLoader in CPU tests. Production loading
    would instantiate from cfg.tokenizer_id / cfg.tokenizer_revision.
    """
    if tokenizer_loader is not None:
        return tokenizer_loader(cfg)
    return None


def _load_model_production(cfg: ExperimentConfig):
    """Production model loading via the approved training primitives.

    Uses ``lerna.utils.model_loader.load_model_and_tokenizer`` with
    architecture-aware defaults (ModernBERT/Ettin eager attention, reference
    compile disabled). Requires ``_MODEL_LOADER_AVAILABLE`` and immutable
    revision preflight (already enforced in ExperimentConfig.__post_init__).

    Not called by default execution — only when a test or caller explicitly
    requests production loading. No pretrained weights are downloaded in this
    step; callers must supply local_files_only-compatible revisions or inject
    their own loader.
    """
    if not _MODEL_LOADER_AVAILABLE:
        raise ImportError(
            "lerna.utils.model_loader is not available; cannot perform "
            "production model loading."
        )
    model, tokenizer = load_model_and_tokenizer(
        cfg.model_id,
        num_labels=2,
        problem_type=None,
        revision=cfg.model_revision,
        local_files_only=True,
    )
    return model


def _load_tokenizer_production(cfg: ExperimentConfig):
    """Production tokenizer loading via the approved training primitives.

    Uses ``lerna.utils.model_loader.load_tokenizer`` with revision enforcement.
    Not called by default execution — only when a test or caller explicitly
    requests production loading.
    """
    if not _MODEL_LOADER_AVAILABLE:
        raise ImportError(
            "lerna.utils.model_loader is not available; cannot perform "
            "production tokenizer loading."
        )
    return load_tokenizer(
        cfg.tokenizer_id,
        revision=cfg.tokenizer_revision,
        local_files_only=True,
    )


def _load_train_dataset_production(cfg: ExperimentConfig, num_examples: int):
    """Production training dataset loading with task mapping and tokenization.

    Maps task to GLUE config, loads the dataset with the configured dataset_id
    and revision/split, applies tokenization with max_length512 truncation,
    applies QNLI matched-probe-ID exclusions before idx removal (training
    split only), removes ``idx`` and other non-model inputs, and preserves
    approved sentence order. Returns a torch Dataset with dynamic padding
    deferred to the minibatch collator. Not called by default execution — only
    when a test or caller explicitly requests production loading. Real-dataset
    training is NOT performed in this step.
    """
    from datasets import load_dataset

    task_info = TASK_INPUTS[cfg.task]
    glue_config = _glue_config_for_task(cfg.task)

    ds = load_dataset(
        cfg.dataset_id,
        glue_config,
        split=cfg.splits[0] if cfg.splits else "train",
        revision=cfg.dataset_revision,
    )

    # Capture idx values before removal for QNLI matched-probe-ID exclusions.
    # Exclusions apply to the training split only; validation is left intact.
    # A non-dry QNLI config must always carry the stored exclusion set; the
    # loader must fail, not skip, when it is missing or empty.
    if cfg.task == "qnli" and not cfg.dry_run:
        exclusion_set = getattr(cfg, "_qnli_exclusion_probe_ids", None)
        if not exclusion_set:
            raise ValueError(
                "QNLI training config has no stored exclusion probe IDs; "
                "the preflight did not store them. Real execution blocked."
            )
        n = len(ds)
        # Reject any probe ID >= training split size (determined at load time).
        for pid in exclusion_set:
            if pid >= n:
                raise ValueError(
                    f"QNLI probe id={pid} exceeds training split size {n}; "
                    f"real execution blocked."
                )
        if "idx" not in ds.column_names:
            raise ValueError(
                "QNLI training split missing 'idx' column; cannot apply "
                "matched-probe-ID exclusions. Real execution blocked."
            )
        idx_values = list(ds["idx"])
        # Require idx == row position (0-based) so probe IDs map to rows.
        for i, v in enumerate(idx_values):
            if int(v) != i:
                raise ValueError(
                    f"QNLI idx[{i}]={int(v)} != row position {i}; cannot "
                    "safely map probe IDs to rows. Real execution blocked."
                )
        keep_indices = [
            i for i, v in enumerate(idx_values) if int(v) not in exclusion_set
        ]
        removed = n - len(keep_indices)
        if removed != len(exclusion_set):
            raise ValueError(
                f"QNLI exclusions removed {removed} rows but expected "
                f"{len(exclusion_set)} probe IDs. Real execution blocked."
            )
        ds = ds.select(keep_indices)

    tokenizer = _load_tokenizer_production(cfg)

    def tokenize_fn(examples):
        if task_info.get("sentence_pair"):
            return tokenizer(
                examples[task_info["question_field"]],
                examples[task_info["sentence_field"]],
                truncation=True,
                max_length=cfg.max_length,
            )
        return tokenizer(
            examples[task_info["sentence_field"]],
            truncation=True,
            max_length=cfg.max_length,
        )

    # Keep the label column; only remove the text fields being tokenized.
    text_fields = []
    if task_info.get("sentence_pair"):
        text_fields = [task_info["question_field"], task_info["sentence_field"]]
    elif task_info.get("sentence_field"):
        text_fields = [task_info["sentence_field"]]
    # Remove idx and other non-model inputs.
    remove_cols = [c for c in ds.column_names if c in text_fields or c == "idx"]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    # GLUE datasets use "label"; downstream evaluator expects "labels".
    if "label" in ds.column_names and "labels" not in ds.column_names:
        ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    return ds


def _load_eval_dataset_production(cfg: ExperimentConfig):
    """Production evaluation dataset loading with task mapping and tokenization.

    Mirrors _load_train_dataset_production for the validation split.
    QNLI matched-probe-ID exclusions are NOT applied to validation data.
    """
    from datasets import load_dataset

    task_info = TASK_INPUTS[cfg.task]
    glue_config = _glue_config_for_task(cfg.task)

    ds = load_dataset(
        cfg.dataset_id,
        glue_config,
        split=cfg.splits[-1] if cfg.splits else "validation",
        revision=cfg.dataset_revision,
    )

    tokenizer = _load_tokenizer_production(cfg)

    def tokenize_fn(examples):
        if task_info.get("sentence_pair"):
            return tokenizer(
                examples[task_info["question_field"]],
                examples[task_info["sentence_field"]],
                truncation=True,
                max_length=cfg.max_length,
            )
        return tokenizer(
            examples[task_info["sentence_field"]],
            truncation=True,
            max_length=cfg.max_length,
        )

    # Keep the label column; only remove the text fields being tokenized.
    text_fields = []
    if task_info.get("sentence_pair"):
        text_fields = [task_info["question_field"], task_info["sentence_field"]]
    elif task_info.get("sentence_field"):
        text_fields = [task_info["sentence_field"]]
    remove_cols = [c for c in ds.column_names if c in text_fields or c == "idx"]
    ds = ds.map(tokenize_fn, batched=True, remove_columns=remove_cols)
    if "label" in ds.column_names and "labels" not in ds.column_names:
        ds = ds.rename_column("label", "labels")
    ds.set_format("torch")

    return ds


def _load_train_dataset(cfg: ExperimentConfig, num_examples: int,
                        dataset_loader=None):
    """Load the training dataset for execution.

    If ``dataset_loader`` is provided it is called as
    ``dataset_loader(cfg, num_examples)`` and its result returned.
    Otherwise returns a list of TinyDataLoader CPU batches — no real dataset
    is downloaded or instantiated in this step.
    """
    if dataset_loader is not None:
        return dataset_loader(cfg, num_examples)
    return list(TinyDataLoader(num_examples=num_examples, seed=cfg.data_seed))


def _load_eval_dataset(cfg: ExperimentConfig, dataset_loader=None):
    """Load the evaluation dataset for execution.

    If ``dataset_loader`` is provided it is called as
    ``dataset_loader(cfg)``. Otherwise returns a small list of TinyDataLoader
    CPU batches — no real dataset is downloaded in this step.
    """
    if dataset_loader is not None:
        return dataset_loader(cfg)
    return list(TinyDataLoader(num_examples=min(8, max(1, cfg.total_steps // 10)),
                                seed=cfg.data_seed + 1))


def _evaluate_model(trainer, model, eval_dataset, cfg: ExperimentConfig, pad_token_id: int = 0):
    """Run actual task evaluation: accuracy on the eval dataset.

    Returns (eval_loss, eval_accuracy, eval_f1). F1 is computed for MRPC;
    None for other tasks. In CPU tests this computes accuracy from the
    TinyModel's forward pass on the eval batches.

    Uses _dynamic_padding_collator with the same pad_token_id as training.
    """
    if not eval_dataset:
        return 0.0, 0.0, None
    correct = 0
    total = 0
    total_loss = 0.0
    tp = fp = fn = 0
    model.eval()
    try:
        eval_loader = DataLoader(eval_dataset, batch_size=cfg.eval_batch_size, shuffle=False,
                                 collate_fn=lambda features, _pd=pad_token_id: _dynamic_padding_collator(features, pad_token_id=_pd))
    except Exception as e:
        raise RuntimeError(
            f"eval_data_loader construction failed: {type(e).__name__}: {e}"
        )
    with torch.no_grad():
        for batch in eval_loader:
            outputs = model(**batch)
            logits = outputs.logits
            # GLUE datasets use "label"; some fixtures use "labels".
            labels = batch.get("labels", batch.get("label"))
            if logits is not None and labels is not None:
                preds = logits.argmax(dim=-1)
                correct += int((preds == labels).sum().item())
                total += int(labels.numel())
                if outputs.loss is not None:
                    total_loss += float(outputs.loss.item())
                # MRPC F1: positive class is label 1
                if cfg.task == "mrpc":
                    for p, l in zip(preds.tolist(), labels.tolist()):
                        if p == 1 and l == 1:
                            tp += 1
                        elif p == 1 and l == 0:
                            fp += 1
                        elif p == 0 and l == 1:
                            fn += 1
    model.train()
    accuracy = correct / total if total > 0 else 0.0
    avg_loss = total_loss / max(1, len(eval_dataset))
    f1 = None
    if cfg.task == "mrpc":
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
    return avg_loss, accuracy, f1


def _make_score_source(cfg: ExperimentConfig):
    """Create the score source for the given config. Does NOT load a model."""
    if cfg.score_source == "uniform":
        rng = random.Random(cfg.policy_seed + 777)
        return mock_uniform_score(rng)
    elif cfg.score_source == "s1":
        # CPU mock: tiny logits + labels, no pretrained model
        logits = TinyLogits(batch=4, classes=2, seed=cfg.policy_seed + 31337)
        labels = [0, 1, 0, 1]
        call_count = [0]

        def _src(decision_index, model, inputs):
            call_count[0] += 1
            return mock_forward_gain_score(logits, labels)

        return _src
    else:
        raise ValueError(f"unsupported score_source for dry-run: {cfg.score_source}")


def _dry_run_reference_arm(cfg: ExperimentConfig) -> dict:
    """Dry-run for full_finetune and reduced_steps arms (no skip gate)."""
    return {
        "status": "OK",
        "config": cfg.to_dict(),
        "config_hash": _config_hash(cfg.to_dict()),
        "result": {
            "arm": cfg.arm,
            "total_steps": cfg.total_steps if cfg.arm == "full_finetune" else cfg.reduced_steps,
            "skip_gate": None,
            "note": f"{cfg.arm} arm does not use a skip gate",
        },
        "provenance": _provenance(),
    }


def _provenance(dry_run: bool = True, execution_mode: Optional[str] = None,
                seed: Optional[int] = None, seed_source: Optional[str] = None) -> dict:
    prov = {
        "resolved_code_sha": _git_sha(),
        "v4_prereg_sha": _v4_prereg_sha(),
        "tree_status": _tree_status(),
        "source_hash_forward_gate": _source_hash(
            Path(__file__).resolve().parents[1] / "lerna" / "utils" / "forward_gate.py"
        ),
        "source_hash_helper": _source_hash(
            Path(__file__).resolve().parents[1] / "lerna" / "utils" / "phase_suffix_feasibility.py"
        ),
        # R5: "source_hash_runner" hashes this measurement script (run_controller_ablation.py),
        # not an external ablation runner. Documented to avoid misreading.
        "source_hash_runner_script": _source_hash(Path(__file__).resolve()),
        "source_hash_runner": _source_hash(Path(__file__).resolve()),
        "scheduler_step_policy": "always_step",
        "dry_run": dry_run,
        "execution_mode": execution_mode
        or ("mock_fixtures_cpu" if dry_run else "production_loaders"),
    }
    if seed is not None:
        prov["seed"] = seed
    if seed_source is not None:
        prov["seed_source"] = seed_source
    prov["note"] = (
        "dry-run validation only; no training, no model download, no GPU"
        if dry_run
        else "loaded through the production loader functions"
    )
    return prov


# --------------------------------------------------------------------------- #
# CPU mock test: S1 and uniform invocation paths
# --------------------------------------------------------------------------- #

def test_cpu_score_paths() -> dict:
    """Test actual S1/uniform invocation paths with CPU mocks, no pretrained model."""
    results = {}

    # Test S1 mock
    logits = TinyLogits(batch=4, classes=2, seed=42)
    labels = [0, 1, 0, 1]
    s1_score = mock_forward_gain_score(logits, labels)
    results["s1_mock_score"] = s1_score
    results["s1_mock_finite"] = s1_score == s1_score and s1_score != float("inf")

    # Test uniform mock
    rng = random.Random(123)
    uniform_src = mock_uniform_score(rng)
    draws = [uniform_src(t, None, None) for t in range(10)]
    results["uniform_draws"] = draws
    results["uniform_in_range"] = all(0.0 <= d <= 1.0 for d in draws)

    # Test gate with S1 mock score source
    gate = ScoreQuantileGate(
        rate=0.6, total_steps=20, min_step=0, max_consecutive_skips=4,
        n_phases=2, score_history_window=64, anneal_fraction=0.88,
        seed=12, score_source=lambda t, m, x: mock_forward_gain_score(logits, labels),
        controller_version="phase_exact_v1",
    )
    actions = []
    for t in range(20):
        actions.append(1 if gate.decide(t, None, None) else 0)
    results["s1_gate_realized"] = sum(actions)
    results["s1_gate_requested"] = gate._requested_total
    results["s1_gate_exact"] = sum(actions) == gate._requested_total

    # Test gate with uniform mock score source
    rng2 = random.Random(99)
    gate2 = ScoreQuantileGate(
        rate=0.6, total_steps=20, min_step=0, max_consecutive_skips=4,
        n_phases=2, score_history_window=64, anneal_fraction=0.88,
        seed=12, score_source=mock_uniform_score(rng2),
        controller_version="phase_exact_v1",
    )
    actions2 = []
    for t in range(20):
        actions2.append(1 if gate2.decide(t, None, None) else 0)
    results["uniform_gate_realized"] = sum(actions2)
    results["uniform_gate_requested"] = gate2._requested_total
    results["uniform_gate_exact"] = sum(actions2) == gate2._requested_total

    results["all_passed"] = (
        results["s1_mock_finite"]
        and results["uniform_in_range"]
        and results["s1_gate_exact"]
        and results["uniform_gate_exact"]
    )
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> int:
    parser = argparse.ArgumentParser(
        description="LERNA Step 3B configurable controller ablation runner "
                    "with safe dry-run mode and CPU/mock execution-path tests"
    )
    parser.add_argument("--task", default="qnli", choices=list(TASK_INPUTS.keys()))
    parser.add_argument("--arm", default="gate_s1_p1", choices=ARMS)
    parser.add_argument("--controller-version", default="legacy_v4",
                        choices=["legacy_v4", "phase_exact_v1"])
    parser.add_argument("--score-source", default="s1", choices=["s1", "uniform"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration without training")
    parser.add_argument("--cpu-mock-test", action="store_true",
                        help="Run CPU mock score-path tests (no model)")
    parser.add_argument("--skip-rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--min-step", type=int, default=392)
    parser.add_argument("--n-phases", type=int, default=1)
    parser.add_argument("--max-consecutive-skips", type=int, default=4)
    parser.add_argument("--policy-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="/tmp/lerna_step3b_runner")
    parser.add_argument("--attempt-id", default=None)
    # R4-A: prerequisite inputs through CLI/config-file route
    parser.add_argument("--model-id", default=None,
                        help="Model identifier (e.g. jhu-clsp/ettin-encoder-150m)")
    parser.add_argument("--tokenizer-id", default=None,
                        help="Tokenizer identifier")
    parser.add_argument("--dataset-id", default=None,
                        help="Dataset identifier (e.g. glue)")
    parser.add_argument("--model-revision", default=None,
                        help="Immutable model revision (40-char hex SHA)")
    parser.add_argument("--tokenizer-revision", default=None,
                        help="Immutable tokenizer revision (40-char hex SHA)")
    parser.add_argument("--dataset-revision", default=None,
                        help="Immutable dataset revision (40-char hex SHA)")
    parser.add_argument("--probe-exclusions-file", default=None,
                        help="QNLI matched-probe-ID exclusion file path")
    parser.add_argument("--probe-exclusions-hash", default=None,
                        help="SHA-256 of the QNLI exclusion file")
    args = parser.parse_args()

    if args.cpu_mock_test:
        result = test_cpu_score_paths()
        print(json.dumps(result, indent=2))
        return 0 if result["all_passed"] else 1

    if args.dry_run:
        raw_cfg = {
            "task": args.task,
            "arm": args.arm,
            "controller_version": args.controller_version,
            "score_source": args.score_source,
            "total_steps": args.total_steps,
            "min_step": args.min_step,
            "n_phases": args.n_phases,
            "max_consecutive_skips": args.max_consecutive_skips,
            "policy_seed": args.policy_seed,
            "target_skip_rate": args.skip_rate,
            "output_dir": args.output_dir,
            "attempt_id": args.attempt_id,
            "dry_run": True,
        }
        if args.model_id is not None:
            raw_cfg["model_id"] = args.model_id
        if args.tokenizer_id is not None:
            raw_cfg["tokenizer_id"] = args.tokenizer_id
        if args.dataset_id is not None:
            raw_cfg["dataset_id"] = args.dataset_id
        if args.model_revision is not None:
            raw_cfg["model_revision"] = args.model_revision
        if args.tokenizer_revision is not None:
            raw_cfg["tokenizer_revision"] = args.tokenizer_revision
        if args.dataset_revision is not None:
            raw_cfg["dataset_revision"] = args.dataset_revision
        if args.probe_exclusions_file is not None:
            raw_cfg["probe_exclusions_file"] = args.probe_exclusions_file
        if args.probe_exclusions_hash is not None:
            raw_cfg["probe_exclusions_hash"] = args.probe_exclusions_hash
        try:
            cfg = ExperimentConfig(**raw_cfg)
        except (ValueError, TypeError) as e:
            manifest = {
                "status": "REJECTED",
                "reason": f"config validation failed: {type(e).__name__}: {e}",
                "config": raw_cfg,
                "provenance": _provenance(dry_run=True),
            }
            print(json.dumps(manifest, indent=2, default=str))
            return 1
        try:
            out_dir = allocate_output_dir(cfg)
        except FileExistsError as e:
            manifest = {
                "status": "REJECTED",
                "reason": "output_path_exists",
                "config": cfg.to_dict(),
                "error": str(e),
                "provenance": _provenance(),
            }
            print(json.dumps(manifest, indent=2, default=str))
            return 1

        result = run_dry_run(cfg)
        result["output_dir"] = str(out_dir)
        (out_dir / "dry_run_manifest.json").write_text(
            json.dumps(result, indent=2, default=str)
        )
        print(json.dumps(result, indent=2, default=str))
        return 0 if result["status"] == "OK" else 1

    # Non-dry-run: execute the controller sweep with CPU mock score sources.
    # Real pretrained training is NOT invoked; execution uses mocked loading
    # boundaries and tiny CPU states. Full config validation runs before any
    # output is written.
    raw_cfg = {
        "task": args.task,
        "arm": args.arm,
        "controller_version": args.controller_version,
        "score_source": args.score_source,
        "total_steps": args.total_steps,
        "min_step": args.min_step,
        "n_phases": args.n_phases,
        "max_consecutive_skips": args.max_consecutive_skips,
        "policy_seed": args.policy_seed,
        "target_skip_rate": args.skip_rate,
        "output_dir": args.output_dir,
        "attempt_id": args.attempt_id,
        "dry_run": False,
    }
    if args.model_id is not None:
        raw_cfg["model_id"] = args.model_id
    if args.tokenizer_id is not None:
        raw_cfg["tokenizer_id"] = args.tokenizer_id
    if args.dataset_id is not None:
        raw_cfg["dataset_id"] = args.dataset_id
    if args.model_revision is not None:
        raw_cfg["model_revision"] = args.model_revision
    if args.tokenizer_revision is not None:
        raw_cfg["tokenizer_revision"] = args.tokenizer_revision
    if args.dataset_revision is not None:
        raw_cfg["dataset_revision"] = args.dataset_revision
    if args.probe_exclusions_file is not None:
        raw_cfg["probe_exclusions_file"] = args.probe_exclusions_file
    if args.probe_exclusions_hash is not None:
        raw_cfg["probe_exclusions_hash"] = args.probe_exclusions_hash
    try:
        cfg = ExperimentConfig(**raw_cfg)
    except (ValueError, TypeError) as e:
        manifest = {
            "status": "REJECTED",
            "reason": f"config validation failed: {type(e).__name__}: {e}",
            "config": raw_cfg,
            "provenance": _provenance(dry_run=False),
        }
        print(json.dumps(manifest, indent=2, default=str))
        return 1

    # Validate gate construction for ALL gate arms (P1/P4, s1/uniform).
    # This catches infeasible quotas, invalid indices, and W=T Q>0 rejection
    # before any output is written.
    try:
        if cfg.arm in ("gate_s1_p1", "gate_s1_p4", "control_uniform_p1", "control_uniform_p4"):
            ScoreQuantileGate(
                rate=cfg.target_skip_rate,
                total_steps=cfg.total_steps,
                min_step=cfg.min_step,
                max_consecutive_skips=args.max_consecutive_skips,
                n_phases=cfg.n_phases,
                score_history_window=cfg.score_history_window,
                anneal_fraction=cfg.anneal_fraction,
                seed=cfg.policy_seed + 1,
                score_source=_make_score_source(cfg),
                controller_version=cfg.controller_version,
            )
        out_dir = allocate_output_dir(cfg)
    except (ValueError, TypeError, InfeasibleBudgetError, PhaseSuffixInfeasibleError, FileExistsError) as e:
        manifest = {
            "status": "REJECTED",
            "reason": f"config validation failed: {type(e).__name__}: {e}",
            "config": cfg.to_dict(),
            "provenance": _provenance(dry_run=False),
        }
        print(json.dumps(manifest, indent=2, default=str))
        return 1

    # Run the actual controller sweep (mock score source, no pretrained model).
    result = run_execution(cfg)
    result["output_dir"] = str(out_dir)
    (out_dir / "execution_manifest.json").write_text(
        json.dumps(result, indent=2, default=str)
    )
    print(json.dumps(result, indent=2, default=str))
    return 0 if result["status"] == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
