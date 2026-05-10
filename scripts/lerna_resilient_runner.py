#!/usr/bin/env python3
"""
LERNA Universal Resilient Research Runner
==========================================
Production-grade experiment orchestrator for ALL phases of LERNA research.
Works with ANY experiment script that follows the HF Trainer pattern:
  - Phase 1.1: run_baseline_glue.py (baseline GLUE benchmarks)
  - Phase 1.2: run_phase1_2_simple_baselines.py, run_baseline_comparison.py
  - Phase 1.3: run_ablation_study.py (ablation study)
  - Phase 2:   run_full_experiment.py / run_full_experiment_v2.py
  - Phase 3:   run_research_sweep.py, lerna_research_complete.py
  - Hardware:  benchmark_rtx5090.py
Architecture:
  ┌──────────────────────────────────────────────┐
  │            ResearchRunner (engine)            │
  │  ┌─────────────┐  ┌──────────────────────┐   │
  │  │ RunRegistry  │  │ DiskManager          │   │
  │  │ (manifest)   │  │ (cleanup/guard)      │   │
  │  └─────────────┘  └──────────────────────┘   │
  │  ┌─────────────────────────────────────────┐  │
  │  │ PhaseAdapter (one per experiment type)   │  │
  │  │  - builds run matrix                    │  │
  │  │  - generates run_key                    │  │
  │  │  - calls the actual training function   │  │
  │  │  - locates results.json                 │  │
  │  └─────────────────────────────────────────┘  │
  └──────────────────────────────────────────────┘
Safety guarantees:
  NEVER DELETES: results.json, ler_diagnostics.json, power/ telemetry,
                 WandB cloud data, HuggingFace model cache, ablation_summary.json
  ONLY DELETES:  checkpoint-*/ dirs, *.bin/*.safetensors model files,
                 trainer_state.json, WandB LOCAL sync dirs (after sync)
Usage:
  # Phase 1.3 — resume ablations
  python lerna_resilient_runner.py --phase ablation \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations \\
    --wandb --wandb-group phase1_3_ablations \\
    --resume --cleanup --cleanup-wandb
  # Phase 1.1 — resume baselines
  python lerna_resilient_runner.py --phase baseline \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_1/baselines \\
    --wandb --resume --cleanup
  # Phase 2 — full LERNA experiments
  python lerna_resilient_runner.py --phase full_experiment \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase2/full \\
    --wandb --resume --cleanup
  # Any phase — check status only
  python lerna_resilient_runner.py --phase ablation --status \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations
  # Any phase — dry run (preview what would execute)
  python lerna_resilient_runner.py --phase ablation --dry-run --resume \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations
"""
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"
import sys
import json
import glob
import shutil
import time
import gc
import argparse
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Any, Set
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
import numpy as np
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 1: RUN REGISTRY — crash-safe manifest tracking
# ════════════════════════════════════════════════════════════════════════════════
class RunRegistry:
    """
    Persistent ledger tracking every run's status (completed/failed/pending).
    Uses atomic writes (write-to-tmp + os.replace) so a crash mid-write
    never corrupts the manifest.
    """
    def __init__(self, output_dir: str, phase: str):
        os.makedirs(output_dir, exist_ok=True)
        self.manifest_path = os.path.join(output_dir, f"run_manifest_{phase}.json")
        self.manifest = self._load()
    def _load(self) -> dict:
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                backup = self.manifest_path + f".bak.{int(time.time())}"
                shutil.copy2(self.manifest_path, backup)
                print(f"  [registry] Corrupt manifest backed up to {backup}")
        return {
            "runs": {},
            "meta": {
                "created": datetime.now().isoformat(),
                "phase": self.manifest_path,
            },
        }
    def _save(self):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=2, default=str)
        os.replace(tmp, self.manifest_path)
    def is_completed(self, run_key: str) -> bool:
        return self.manifest["runs"].get(run_key, {}).get("status") == "completed"
    def is_failed(self, run_key: str) -> bool:
        return self.manifest["runs"].get(run_key, {}).get("status") == "failed"
    def mark_completed(self, run_key: str, results_path: str,
                       wandb_synced: bool = False, extra: dict = None):
        entry = {
            "status": "completed",
            "results_path": results_path,
            "wandb_synced": wandb_synced,
            "completed_at": datetime.now().isoformat(),
        }
        if extra:
            entry.update(extra)
        self.manifest["runs"][run_key] = entry
        self._save()
    def mark_failed(self, run_key: str, error: str):
        self.manifest["runs"][run_key] = {
            "status": "failed",
            "error": error[:500],
            "failed_at": datetime.now().isoformat(),
        }
        self._save()
    def bootstrap_from_key(self, run_key: str, results_path: str):
        if run_key not in self.manifest["runs"]:
            self.manifest["runs"][run_key] = {
                "status": "completed",
                "results_path": results_path,
                "wandb_synced": True,
                "completed_at": "pre-patch-bootstrap",
            }
    def save_if_dirty(self):
        self._save()
    def count_by_status(self) -> dict:
        counts = {"completed": 0, "failed": 0, "pending": 0}
        for entry in self.manifest["runs"].values():
            s = entry.get("status", "pending")
            counts[s] = counts.get(s, 0) + 1
        return counts
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 2: DISK MANAGER — space monitoring + surgical cleanup
# ════════════════════════════════════════════════════════════════════════════════
class DiskManager:
    """
    Handles all disk-space operations:
      - Pre-run space checks (abort threshold)
      - Post-run checkpoint cleanup
      - WandB local cache cleanup
      - Bulk cleanup of old completed runs
    """
    PROTECTED_FILES = {
        "results.json",
        "ler_diagnostics.json",
        "all_predictions.jsonl",
        "config.json",
    }
    PROTECTED_DIRS = {"power"}
    CHECKPOINT_PATTERNS = ["checkpoint-*"]
    JUNK_FILE_PATTERNS = ["*.bin", "*.safetensors", "training_args.bin"]
    JUNK_FILES = ["trainer_state.json", "optimizer.pt", "scheduler.pt",
                  "rng_state.pth", "scaler.pt"]
    @staticmethod
    def get_free_gb(path: str = ".") -> float:
        try:
            stat = os.statvfs(path)
            return (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        except OSError:
            return float("inf")
    @staticmethod
    def dir_size_bytes(path: str) -> int:
        total = 0
        for dp, _, fns in os.walk(path):
            for f in fns:
                try:
                    total += os.path.getsize(os.path.join(dp, f))
                except OSError:
                    pass
        return total
    @classmethod
    def check_space(cls, path: str, min_gb: float) -> bool:
        free = cls.get_free_gb(path)
        if free < min_gb:
            print(f"\n  [disk] WARNING: {free:.1f} GB free < {min_gb:.1f} GB minimum")
            return False
        return True
    @classmethod
    def cleanup_run(cls, run_dir: str) -> int:
        """
        Clean training intermediates from a single run directory.
        Returns bytes freed.
        """
        if not os.path.isdir(run_dir):
            return 0
        freed = 0
        for pattern in cls.CHECKPOINT_PATTERNS:
            for ckpt in glob.glob(os.path.join(run_dir, pattern)):
                if os.path.isdir(ckpt):
                    size = cls.dir_size_bytes(ckpt)
                    shutil.rmtree(ckpt, ignore_errors=True)
                    freed += size
        for pattern in cls.JUNK_FILE_PATTERNS:
            for fpath in glob.glob(os.path.join(run_dir, pattern)):
                basename = os.path.basename(fpath)
                if basename in cls.PROTECTED_FILES:
                    continue
                try:
                    freed += os.path.getsize(fpath)
                    os.remove(fpath)
                except OSError:
                    pass
        for fname in cls.JUNK_FILES:
            fpath = os.path.join(run_dir, fname)
            if os.path.exists(fpath):
                try:
                    freed += os.path.getsize(fpath)
                    os.remove(fpath)
                except OSError:
                    pass
        return freed
    @classmethod
    def cleanup_wandb_local(cls, base_dir: str = ".") -> int:
        """Clean WandB local sync dirs. Safe because data is already in WandB cloud."""
        wandb_dir = os.path.join(base_dir, "wandb")
        if not os.path.isdir(wandb_dir):
            return 0
        freed = 0
        for entry in os.listdir(wandb_dir):
            full = os.path.join(wandb_dir, entry)
            if os.path.isdir(full) and (
                entry.startswith("run-") or entry.startswith("offline-run-")
            ):
                size = cls.dir_size_bytes(full)
                shutil.rmtree(full, ignore_errors=True)
                freed += size
        return freed
    @classmethod
    def cleanup_torch_cache(cls) -> None:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 3: PHASE ADAPTERS — one per experiment type
# ════════════════════════════════════════════════════════════════════════════════
class PhaseAdapter(ABC):
    """
    Abstract adapter that bridges the universal runner to a specific
    experiment script. Each phase implements:
      - run_key(): how to name a run
      - build_matrix(): generate all (task, seed, ...) combos
      - find_results(): where results.json lives for a given run
      - execute_run(): call the actual training function
      - validate_results(): check if results.json is valid (not partial)
      - get_run_dir(): filesystem path for a run's output
    """
    @abstractmethod
    def run_key(self, **kwargs) -> str:
        ...
    @abstractmethod
    def build_matrix(self, args) -> List[dict]:
        ...
    @abstractmethod
    def find_results_path(self, run_params: dict, output_dir: str) -> str:
        ...
    @abstractmethod
    def get_run_dir(self, run_params: dict, output_dir: str) -> str:
        ...
    @abstractmethod
    def execute_run(self, run_params: dict, args, run_idx: int,
                    total_runs: int) -> dict:
        ...
    def validate_results(self, results_path: str) -> bool:
        if not os.path.exists(results_path):
            return False
        try:
            with open(results_path, "r") as f:
                data = json.load(f)
            return ("eval_metrics" in data or "test_metrics" in data) and "error" not in data
        except (json.JSONDecodeError, IOError):
            return False
    def scan_completed(self, output_dir: str, matrix: List[dict]) -> Set[str]:
        completed = set()
        for params in matrix:
            rpath = self.find_results_path(params, output_dir)
            if self.validate_results(rpath):
                completed.add(self.run_key(**params))
        return completed
# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTER: Phase 1.3 Ablation Study
# ──────────────────────────────────────────────────────────────────────────────
class AblationAdapter(PhaseAdapter):
    ABLATIONS = {
        "full_lerna":    {},
        "no_rho_vg":     {"use_rho_vg": False},
        "no_ler":        {"use_ler": False},
        "no_safety":     {"use_safety_horizon": False},
        "no_hysteresis": {"use_hysteresis": False},
        "no_momentum":   {"use_momentum_extrap": False},
    }
    def __init__(self):
        try:
            from scripts.run_ablation_study import run_ablation_single
            from scripts.run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        except ModuleNotFoundError:
            from run_ablation_study import run_ablation_single
            from run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        self._run_fn = run_ablation_single
        self._task_hp = TASK_HP_OVERRIDES
        self._detect_profile = detect_device_profile
        self._ensure_wandb = _ensure_wandb_finished
    def run_key(self, task: str, seed: int, ablation: str, **kw) -> str:
        return f"{task}_s{seed}_ab-{ablation}"
    def build_matrix(self, args) -> List[dict]:
        ablations = args.ablations or list(self.ABLATIONS.keys())
        matrix = []
        for task in args.tasks:
            for seed in args.seeds:
                for ab in ablations:
                    matrix.append({"task": task, "seed": seed, "ablation": ab})
        return matrix
    def find_results_path(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, p["ablation"], key, "results.json")
    def get_run_dir(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, p["ablation"], key)
    def execute_run(self, p: dict, args, run_idx: int, total_runs: int) -> dict:
        from lerna.utils.model_loader import MODELS
        model_name = MODELS[args.model]
        profile = self._detect_profile()
        effective_max = args.max_samples
        if effective_max is None and not args.unlimited:
            effective_max = 2000 if profile != "server" else 25000
        task_hp = self._task_hp.get(p["task"], {})
        return self._run_fn(
            task_name=p["task"],
            seed=p["seed"],
            lr=task_hp.get("learning_rate", args.lr),
            profile=profile,
            base_output_dir=args.output_dir,
            use_wandb=args.wandb,
            max_samples_override=effective_max,
            run_idx=run_idx,
            total_runs=total_runs,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            num_epochs=task_hp.get("num_epochs", 3),
            warmup_ratio=task_hp.get("warmup_ratio", 0.1),
            early_stopping_patience=task_hp.get("early_stopping_patience", 5),
            metric_for_best_model=task_hp.get("metric_for_best_model", "eval_loss"),
            greater_is_better=task_hp.get("greater_is_better", False),
            init_from_mnli=task_hp.get("init_from_mnli", False),
        )
# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTER: Phase 1.1 Baseline GLUE
# ──────────────────────────────────────────────────────────────────────────────
class BaselineAdapter(PhaseAdapter):
    def __init__(self):
        try:
            from scripts.run_baseline_glue import (
                run_single_experiment, TASK_HP_OVERRIDES,
                detect_device_profile, _ensure_wandb_finished,
                get_training_config,
            )
        except ModuleNotFoundError:
            from run_baseline_glue import (
                run_single_experiment, TASK_HP_OVERRIDES,
                detect_device_profile, _ensure_wandb_finished,
                get_training_config,
            )
        self._run_fn = run_single_experiment
        self._task_hp = TASK_HP_OVERRIDES
        self._detect_profile = detect_device_profile
        self._ensure_wandb = _ensure_wandb_finished
        self._get_config = get_training_config
    def run_key(self, task: str, seed: int, **kw) -> str:
        return f"{task}_s{seed}"
    def build_matrix(self, args) -> List[dict]:
        matrix = []
        for task in args.tasks:
            for seed in args.seeds:
                matrix.append({"task": task, "seed": seed})
        return matrix
    def find_results_path(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, key, "results.json")
    def get_run_dir(self, p: dict, output_dir: str) -> str:
        return os.path.join(output_dir, self.run_key(**p))
    def execute_run(self, p: dict, args, run_idx: int, total_runs: int) -> dict:
        from lerna.utils.model_loader import MODELS
        model_name = MODELS[args.model]
        profile = self._detect_profile()
        effective_max = args.max_samples
        if effective_max is None and not args.unlimited:
            effective_max = 2000 if profile != "server" else 25000
        task_hp = self._task_hp.get(p["task"], {})
        return self._run_fn(
            task_name=p["task"],
            seed=p["seed"],
            lr=task_hp.get("learning_rate", args.lr),
            profile=profile,
            base_output_dir=args.output_dir,
            use_wandb=args.wandb,
            max_samples_override=effective_max,
            run_idx=run_idx,
            total_runs=total_runs,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
            num_epochs=task_hp.get("num_epochs", 3),
            warmup_ratio=task_hp.get("warmup_ratio", 0.1),
            early_stopping_patience=task_hp.get("early_stopping_patience", 5),
            metric_for_best_model=task_hp.get("metric_for_best_model", "eval_loss"),
            greater_is_better=task_hp.get("greater_is_better", False),
            init_from_mnli=task_hp.get("init_from_mnli", False),
        )
# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTER: Phase 2 Full LERNA Experiment
# ──────────────────────────────────────────────────────────────────────────────
class FullExperimentAdapter(PhaseAdapter):
    def __init__(self):
        try:
            from scripts.run_full_experiment_v2 import run_subprocess
            from scripts.run_baseline_glue import TASK_HP_OVERRIDES
        except ModuleNotFoundError:
            from run_full_experiment_v2 import run_subprocess
            from run_baseline_glue import TASK_HP_OVERRIDES
        self._run_fn = run_subprocess
        self._task_hp = TASK_HP_OVERRIDES
        self._detect_profile = detect_device_profile
        self._ensure_wandb = _ensure_wandb_finished
    def run_key(self, task: str, seed: int, **kw) -> str:
        return f"{task}_s{seed}_lerna"
    def build_matrix(self, args) -> List[dict]:
        matrix = []
        for task in args.tasks:
            for seed in args.seeds:
                matrix.append({"task": task, "seed": seed})
        return matrix
    def find_results_path(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, key, "results.json")
    def get_run_dir(self, p: dict, output_dir: str) -> str:
        return os.path.join(output_dir, self.run_key(**p))
    def execute_run(self, p: dict, args, run_idx: int, total_runs: int) -> dict:
        from lerna.utils.model_loader import MODELS
        from scripts.run_full_experiment_v2 import TASK_LR
        lr = TASK_LR.get(p["task"], 2e-5)
        return self._run_fn(
            task=p["task"],
            seed=p["seed"],
            lr=lr,
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            wandb_project=args.wandb_project or "lerna-full",
            wandb_group=args.wandb_group,
            max_samples=args.max_samples if args.max_samples else None,
            script="scripts/run_baseline_glue.py",
            logger=None,
            idx=run_idx,
            total=total_runs,
            timeout=args.timeout or 7200,
        )
# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTER: Hyperparameter Sweep
# ──────────────────────────────────────────────────────────────────────────────
class SweepAdapter(PhaseAdapter):
    def __init__(self):
        try:
            from scripts.run_research_sweep import run_experiment as run_single_sweep
            from scripts.run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        except ModuleNotFoundError:
            from run_research_sweep import run_experiment as run_single_sweep
            from run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        self._run_fn = run_single_sweep
        self._task_hp = TASK_HP_OVERRIDES
        self._detect_profile = detect_device_profile
        self._ensure_wandb = _ensure_wandb_finished
    def run_key(self, task: str, seed: int, **kw) -> str:
        return f"{task}_s{seed}_sweep"
    def build_matrix(self, args) -> List[dict]:
        matrix = []
        for task in args.tasks:
            for seed in args.seeds:
                matrix.append({"task": task, "seed": seed})
        return matrix
    def find_results_path(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, key, "results.json")
    def get_run_dir(self, p: dict, output_dir: str) -> str:
        return os.path.join(output_dir, self.run_key(**p))
    def execute_run(self, p: dict, args, run_idx: int, total_runs: int) -> dict:
        from lerna.utils.model_loader import MODELS
        from scripts.run_research_sweep import ExperimentConfig
        model_name = MODELS[args.model]
        profile = self._detect_profile()
        effective_max = args.max_samples
        if effective_max is None and not args.unlimited:
            effective_max = 2000 if profile != "server" else 25000
        config = ExperimentConfig(
            model_name=model_name,
            task=p["task"],
            seed=p["seed"],
            output_dir=args.output_dir,
            use_wandb=args.wandb,
            # Add other defaults
        )
        return self._run_fn(config)
# ──────────────────────────────────────────────────────────────────────────────
#  ADAPTER: Phase 1.2 Simple Baselines
# ──────────────────────────────────────────────────────────────────────────────
class SimpleBaselineAdapter(PhaseAdapter):
    def __init__(self):
        try:
            from scripts.run_phase1_2_simple_baselines import run_single_baseline_experiment as run_single_baseline
            from scripts.run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        except ModuleNotFoundError:
            from run_phase1_2_simple_baselines import run_single_baseline_experiment as run_single_baseline
            from run_baseline_glue import (
                TASK_HP_OVERRIDES, detect_device_profile,
                _ensure_wandb_finished,
            )
        self._run_fn = run_single_baseline
        self._task_hp = TASK_HP_OVERRIDES
        self._detect_profile = detect_device_profile
        self._ensure_wandb = _ensure_wandb_finished
    def run_key(self, task: str, seed: int, **kw) -> str:
        return f"{task}_s{seed}_simple"
    def build_matrix(self, args) -> List[dict]:
        matrix = []
        for task in args.tasks:
            for seed in args.seeds:
                matrix.append({"task": task, "seed": seed})
        return matrix
    def find_results_path(self, p: dict, output_dir: str) -> str:
        key = self.run_key(**p)
        return os.path.join(output_dir, key, "results.json")
    def get_run_dir(self, p: dict, output_dir: str) -> str:
        return os.path.join(output_dir, self.run_key(**p))
    def execute_run(self, p: dict, args, run_idx: int, total_runs: int) -> dict:
        from lerna.utils.model_loader import MODELS
        model_name = MODELS[args.model]
        profile = self._detect_profile()
        effective_max = args.max_samples
        if effective_max is None and not args.unlimited:
            effective_max = 2000 if profile != "server" else 25000
        return self._run_fn(
            task_name=p["task"],
            seed=p["seed"],
            model_name=model_name,
            profile=profile,
            base_output_dir=args.output_dir,
            use_wandb=args.wandb,
            max_samples_override=effective_max,
            run_idx=run_idx,
            total_runs=total_runs,
            wandb_project=args.wandb_project,
            wandb_group=args.wandb_group,
        )
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 4: ADAPTER REGISTRY
# ════════════════════════════════════════════════════════════════════════════════
PHASE_ADAPTERS = {
    "ablation":        AblationAdapter,
    "baseline":        BaselineAdapter,
    "full_experiment": FullExperimentAdapter,
    "sweep":           SweepAdapter,
    "simple_baseline": SimpleBaselineAdapter,
}
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 5: RESEARCH RUNNER ENGINE — the universal orchestrator
# ════════════════════════════════════════════════════════════════════════════════
class ResearchRunner:
    """
    Universal experiment orchestrator. Phase-agnostic.
    All phase-specific logic is delegated to the PhaseAdapter.
    """
    def __init__(self, adapter: PhaseAdapter, args):
        self.adapter = adapter
        self.args = args
        self.registry = RunRegistry(args.output_dir, args.phase)
        self.disk = DiskManager()
    def bootstrap_existing(self, matrix: List[dict]):
        """Detect runs completed before this system was installed."""
        count = 0
        for params in matrix:
            key = self.adapter.run_key(**params)
            if self.registry.is_completed(key):
                continue
            rpath = self.adapter.find_results_path(params, self.args.output_dir)
            if self.adapter.validate_results(rpath):
                self.registry.bootstrap_from_key(key, rpath)
                count += 1
        if count > 0:
            self.registry.save_if_dirty()
            print(f"  [registry] Bootstrapped {count} previously completed runs")
    def classify_runs(self, matrix: List[dict]):
        """Split matrix into to_run and to_skip."""
        to_run, to_skip = [], []
        for params in matrix:
            key = self.adapter.run_key(**params)
            if self.registry.is_completed(key):
                to_skip.append((params, "completed"))
            elif self.registry.is_failed(key) and not self.args.retry_failed:
                to_skip.append((params, "failed"))
            else:
                to_run.append(params)
        return to_run, to_skip
    def print_status(self, matrix, to_run, to_skip):
        """Print status report and exit."""
        print(f"\n  ═══════════════════════════════════════════════════════")
        print(f"  {self.args.phase.upper()} EXPERIMENT STATUS")
        print(f"  ═══════════════════════════════════════════════════════")
        print(f"  Total matrix:  {len(matrix)} runs")
        print(f"  Completed:     {sum(1 for _, r in to_skip if r == 'completed')}")
        print(f"  Failed:        {sum(1 for _, r in to_skip if r == 'failed')}")
        print(f"  Remaining:     {len(to_run)}")
        print(f"  Disk free:     {self.disk.get_free_gb(self.args.output_dir):.1f} GB")
        print(f"  Manifest:      {self.registry.manifest_path}")
        print(f"  ═══════════════════════════════════════════════════════")
        if to_run:
            print(f"\n  Remaining runs:")
            for i, p in enumerate(to_run, 1):
                key = self.adapter.run_key(**p)
                print(f"    {i:>3}. {key}")
        if to_skip:
            print(f"\n  Skipped runs:")
            for p, reason in to_skip:
                key = self.adapter.run_key(**p)
                print(f"    - {key}: {reason}")
    def print_plan(self, matrix, to_run, to_skip):
        """Print execution plan."""
        a = self.args
        print(f"\n  ═══════════════════════════════════════════════════════")
        print(f"  LERNA Resilient Runner — {a.phase.upper()}")
        print(f"  ═══════════════════════════════════════════════════════")
        print(f"  Tasks:         {a.tasks}")
        print(f"  Seeds:         {a.seeds}")
        if hasattr(a, 'ablations') and a.ablations:
            print(f"  Ablations:     {a.ablations}")
        print(f"  Total matrix:  {len(matrix)} runs")
        print(f"  Skipping:      {len(to_skip)} (done/failed)")
        print(f"  To execute:    {len(to_run)} runs")
        print(f"  Resume:        {'ON' if a.resume else 'OFF'}")
        print(f"  Cleanup:       {'ON' if a.cleanup else 'OFF'}")
        print(f"  Cleanup WandB: {'ON' if a.cleanup_wandb else 'OFF'}")
        print(f"  Min disk:      {a.min_disk_gb:.1f} GB")
        print(f"  Disk free:     {self.disk.get_free_gb(a.output_dir):.1f} GB")
        print(f"  Manifest:      {self.registry.manifest_path}")
        print(f"  ═══════════════════════════════════════════════════════")
    def cleanup_old_completed(self, to_skip):
        """Reclaim disk from previously completed runs."""
        print(f"\n  [cleanup] Scanning old completed runs for leftover checkpoints...")
        total_freed = 0
        for params, reason in to_skip:
            if reason == "completed":
                run_dir = self.adapter.get_run_dir(params, self.args.output_dir)
                freed = self.disk.cleanup_run(run_dir)
                total_freed += freed
        if total_freed > 0:
            print(f"  [cleanup] Freed {total_freed / (1024**3):.2f} GB from old runs")
        else:
            print(f"  [cleanup] No leftover checkpoints found")
        return total_freed
    def execute(self, to_run, to_skip, total_matrix_size):
        """Run the experiment loop with all resilience features."""
        a = self.args
        try:
            from scripts.run_baseline_glue import _ensure_wandb_finished
        except ModuleNotFoundError:
            from run_baseline_glue import _ensure_wandb_finished
        if a.wandb:
            _ensure_wandb_finished()
        all_results = []
        run_idx = 0
        overall_start = time.time()
        succeeded = 0
        failed = 0
        total_freed = 0
        skip_count = len(to_skip)
        for params in to_run:
            run_idx += 1
            key = self.adapter.run_key(**params)
            run_dir = self.adapter.get_run_dir(params, a.output_dir)
            # ── disk guard ──
            if not self.disk.check_space(a.output_dir, a.min_disk_gb):
                print(f"\n  PAUSED: Disk below {a.min_disk_gb} GB.")
                print(f"  Completed {succeeded} this session. Re-run with --resume to continue.")
                break
            # ── ETA ──
            if run_idx > 1:
                elapsed = time.time() - overall_start
                avg = elapsed / (run_idx - 1)
                remaining = (len(to_run) - run_idx + 1) * avg
                eta = f"ETA: {timedelta(seconds=int(remaining))}"
            else:
                eta = "ETA: calculating..."
            print(f"\n  ═══ Run {run_idx}/{len(to_run)} "
                  f"(overall {skip_count + run_idx}/{total_matrix_size}) | {eta} ═══")
            print(f"  -> {key}")
            print(f"  -> Disk: {self.disk.get_free_gb(a.output_dir):.1f} GB free")
            try:
                result = self.adapter.execute_run(
                    params, a, skip_count + run_idx, total_matrix_size
                )
                all_results.append(result)
                rpath = self.adapter.find_results_path(params, a.output_dir)
                self.registry.mark_completed(key, rpath, wandb_synced=a.wandb)
                succeeded += 1
                # ── post-run cleanup ──
                if a.cleanup:
                    freed = self.disk.cleanup_run(run_dir)
                    total_freed += freed
                    if freed > 0:
                        print(f"  [cleanup] Freed {freed / (1024**2):.0f} MB (checkpoints)")
                if a.cleanup_wandb:
                    freed = self.disk.cleanup_wandb_local(os.getcwd())
                    total_freed += freed
                    if freed > 0:
                        print(f"  [cleanup] Freed {freed / (1024**2):.0f} MB (wandb local)")
                self.disk.cleanup_torch_cache()
            except Exception as e:
                print(f"  FAILED: {key}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"error": str(e), **params})
                self.registry.mark_failed(key, str(e))
                failed += 1
                if a.wandb:
                    _ensure_wandb_finished()
                self.disk.cleanup_torch_cache()
        # ── merge into summary ──
        summary_path = os.path.join(a.output_dir, f"{a.phase}_summary.json")
        existing = []
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r") as f:
                    existing = json.load(f)
            except (json.JSONDecodeError, TypeError):
                pass
        merged = existing + all_results
        os.makedirs(a.output_dir, exist_ok=True)
        tmp = summary_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        os.replace(tmp, summary_path)
        total_elapsed = time.time() - overall_start
        print(f"\n{'='*64}")
        print(f"  SESSION COMPLETE — {a.phase.upper()}")
        print(f"{'='*64}")
        print(f"  This session:   {succeeded} succeeded, {failed} failed")
        print(f"  Overall:        {skip_count + succeeded}/{total_matrix_size} completed")
        print(f"  Disk freed:     {total_freed / (1024**3):.2f} GB")
        print(f"  Disk free now:  {self.disk.get_free_gb(a.output_dir):.1f} GB")
        print(f"  Wall time:      {timedelta(seconds=int(total_elapsed))}")
        print(f"  Summary:        {summary_path}")
        print(f"  Manifest:       {self.registry.manifest_path}")
        remaining = total_matrix_size - skip_count - succeeded
        if remaining > 0:
            print(f"\n  {remaining} runs remaining — re-run with --resume to continue")
        else:
            print(f"\n  ALL RUNS COMPLETE!")
        # ── per-ablation summary (if ablation phase) ──
        ok = [r for r in all_results if "error" not in r]
        if ok and "ablation" in ok[0]:
            ablations_seen = sorted(set(r["ablation"] for r in ok))
            print(f"\n  {'Ablation':<15} {'Runs':>5} {'Avg Acc':>10} {'Std':>8} {'Avg kWh':>10}")
            print(f"  {'-'*50}")
            for ab in ablations_seen:
                ab_r = [r for r in ok if r.get("ablation") == ab]
                accs = [
                    r.get("eval_metrics", {}).get("eval_accuracy",
                        r.get("eval_metrics", {}).get("eval_matthews_correlation",
                        r.get("eval_metrics", {}).get("eval_pearson", 0)))
                    for r in ab_r
                ]
                kwhs = [r.get("energy_kwh", 0) for r in ab_r]
                print(f"  {ab:<15} {len(ab_r):>5} "
                      f"{np.mean(accs):>10.4f} {np.std(accs):>8.4f} "
                      f"{np.mean(kwhs):>10.6f}")
        print(f"{'='*64}")
        return all_results
# ════════════════════════════════════════════════════════════════════════════════
#  SECTION 6: CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════════
def main():
    try:
        from scripts.run_baseline_glue import GLUE_TASKS, SEEDS
    except ModuleNotFoundError:
        from run_baseline_glue import GLUE_TASKS, SEEDS
    parser = argparse.ArgumentParser(
        description="LERNA Universal Resilient Research Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Phase 1.3 ablation — resume + cleanup
  python lerna_resilient_runner.py --phase ablation \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations \\
    --wandb --wandb-group phase1_3_ablations \\
    --resume --cleanup --cleanup-wandb
  # Phase 1.1 baseline — resume + cleanup
  python lerna_resilient_runner.py --phase baseline \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_1/baselines \\
    --wandb --resume --cleanup
  # Phase 2 — full LERNA experiments
  python lerna_resilient_runner.py --phase full_experiment \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase2/full \\
    --wandb --resume --cleanup
  # Any phase — check status only
  python lerna_resilient_runner.py --phase ablation --status \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations
  # Dry run — see what would execute
  python lerna_resilient_runner.py --phase ablation --dry-run --resume \\
    --tasks sst2 mrpc rte qnli --seeds 42 123 456 789 1024 \\
    --output-dir ./experiments/phase1_3/ablations
        """,
    )
    # Phase selection
    parser.add_argument(
        "--phase", required=True,
        choices=list(PHASE_ADAPTERS.keys()),
        help="Which experiment phase to run",
    )
    # Experiment matrix
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Ablation names (only for --phase ablation)")
    parser.add_argument("--output-dir", required=True)
    # Model & data
    parser.add_argument("--model", default="modernbert",
                        choices=["roberta", "modernbert", "deberta"])
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Default learning rate (default: 2e-5)")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unlimited", action="store_true")
    # WandB
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerna-ablation")
    parser.add_argument("--wandb-group", default=None)
    # Resilience flags
    parser.add_argument("--resume", action="store_true",
                        help="Skip completed runs (auto-detected from results.json)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Delete checkpoints after each successful run")
    parser.add_argument("--cleanup-wandb", action="store_true",
                        help="Delete WandB local sync dirs after each run")
    parser.add_argument("--cleanup-before-resume", action="store_true",
                        help="Clean checkpoints from OLD completed runs before starting")
    parser.add_argument("--min-disk-gb", type=float, default=5.0,
                        help="Pause execution if free disk drops below this (GB)")
    parser.add_argument("--retry-failed", action="store_true",
                        help="Re-attempt previously failed runs")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would run without executing")
    parser.add_argument("--status", action="store_true",
                        help="Print completion status and exit")
    parser.add_argument("--timeout", type=int, default=7200,
                        help="Timeout per run in seconds (default: 7200)")
    args = parser.parse_args()
    # Default wandb group
    if args.wandb_group is None:
        args.wandb_group = f"{args.phase}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    # ── Initialize adapter ──
    print(f"\n  Loading {args.phase} adapter...")
    adapter = PHASE_ADAPTERS[args.phase]()
    # ── Build runner ──
    runner = ResearchRunner(adapter, args)
    # ── Build matrix ──
    matrix = adapter.build_matrix(args)
    # ── Bootstrap old results ──
    runner.bootstrap_existing(matrix)
    # ── Classify ──
    if args.resume:
        to_run, to_skip = runner.classify_runs(matrix)
    else:
        to_run = matrix
        to_skip = []
    # ── Status mode ──
    if args.status:
        # Force resume classification for status
        to_run_s, to_skip_s = runner.classify_runs(matrix)
        runner.print_status(matrix, to_run_s, to_skip_s)
        return
    # ── Print plan ──
    runner.print_plan(matrix, to_run, to_skip)
    if not args.resume and to_skip:
        # Re-classify to show what WOULD be skipped
        _, would_skip = runner.classify_runs(matrix)
        if would_skip:
            print(f"\n  [!] Resume OFF — {len(would_skip)} completed runs will re-execute.")
            print(f"      Add --resume to skip them.")
    # ── Dry run ──
    if args.dry_run:
        print(f"\n  DRY RUN — would execute {len(to_run)} runs:")
        for i, p in enumerate(to_run, 1):
            key = adapter.run_key(**p)
            print(f"    {i:>3}. {key}")
        return
    if not to_run:
        print(f"\n  All {len(matrix)} runs already completed! Nothing to do.")
        return
    # ── Pre-cleanup ──
    if args.cleanup_before_resume and args.resume:
        runner.cleanup_old_completed(to_skip)
    # ── Execute ──
    runner.execute(to_run, to_skip, len(matrix))
if __name__ == "__main__":
    main()