#!/usr/bin/env python3
"""
LERNA Phase 1.2: Ettin-150m Manifest Launcher

Generates a Cartesian matrix of 10 baselines × 8 GLUE tasks × 10 seeds = 800 runs.
Each manifest entry requires a verifiable results.json to be marked complete.

Usage:
  python scripts/run_phase1_2_ettin.py --mode generate   # write manifest only
  python scripts/run_phase1_2_ettin.py --mode validate   # verify existing results
  python scripts/run_phase1_2_ettin.py --mode run        # execute missing runs
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

if not os.environ.get("HF_HUB_OFFLINE") and not os.environ.get("TRANSFORMERS_OFFLINE"):
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from run_baseline_glue import TASK_HP_OVERRIDES

TASKS = ("sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb")
SEEDS = range(42, 52)
OUTPUT = Path("experiments/phase1_2_ettin150m_2026")
MANIFEST = OUTPUT / "run_manifest.json"

BASELINES = [
    "full_finetune",
    "grad_norm",
    "random_skip",
    "weight_freeze",
    "reduced_steps",
    "cosine_restarts",
    "early_stop_p3",
    "early_stop_p5",
    "early_stop_p10",
    "early_stop_p20",
]

DEFAULT_BASELINES = [
    "full_finetune",
    "grad_norm",
    "random_skip",
    "weight_freeze",
    "reduced_steps",
    "cosine_restarts",
    "early_stop_p3",
    "early_stop_p5",
    "early_stop_p10",
    "early_stop_p20",
]

DEFAULT_TASKS = TASKS
DEFAULT_SEEDS = list(SEEDS)

EXPECTED_MODEL = "jhu-clsp/ettin-encoder-150m"
EXPECTED_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
EXPECTED_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"


def build_matrix(baselines=None, tasks=None, seeds=None):
    baselines = baselines or DEFAULT_BASELINES
    tasks = tasks or DEFAULT_TASKS
    seeds = seeds or DEFAULT_SEEDS
    entries = []
    for baseline in baselines:
        for task in tasks:
            for seed in seeds:
                entries.append({
                    "baseline": baseline,
                    "task": task,
                    "seed": seed,
                    "status": "pending",
                    "results_path": None,
                })
    return entries


def validate_results_json(path: Path, entry: dict) -> tuple[bool, str]:
    try:
        data = json.loads(path.read_text())
    except Exception as exc:
        return False, f"unreadable_results_json: {exc}"

    baseline = entry["baseline"]
    es_enabled = baseline.startswith("early_stop_p")
    es_patience = int(baseline.removeprefix("early_stop_p")) if es_enabled else None

    required = {
        "phase": "1.2",
        "model_name": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "dataset_scope": data.get("dataset_scope", "full GLUE"),
        "baseline": baseline,
        "task": entry["task"],
        "seed": entry["seed"],
        "primary_metric": None,
        "train_runtime_s": None,
        "forward_calls": None,
        "backward_calls": None,
        "training_device": None,
        "cuda_available": None,
        "energy_kwh": None,
        "energy_valid": None,
        "energy_invalid_reason": None,
    }
    for key, expected in required.items():
        actual = data.get(key)
        if expected is not None and actual != expected:
            return False, f"mismatch_{key}: {actual!r} != {expected!r}"
        if expected is None and actual is None:
            return False, f"missing_{key}"
    if data.get("early_stopping_enabled") != es_enabled:
        return False, f"mismatch_early_stopping_enabled: {data.get(chr(34)+'early_stopping_enabled'+chr(34))!r} != {es_enabled}"
    if data.get("early_stopping_patience") != es_patience:
        return False, f"mismatch_early_stopping_patience: {data.get(chr(34)+'early_stopping_patience'+chr(34))!r} != {es_patience}"
    return True, "ok"


def generate(args):
    output_dir = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "run_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = args.baselines or DEFAULT_BASELINES
    tasks = args.tasks or DEFAULT_TASKS
    seeds = args.seeds or DEFAULT_SEEDS

    entries = build_matrix(baselines=baselines, tasks=tasks, seeds=seeds)
    manifest = {
        "phase": "1.2-ettin",
        "model_name": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(),
        "total_runs": len(entries),
        "max_samples": None,
        "dataset_scope": "full GLUE",
        "wandb": "disabled",
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Generated manifest: {manifest_path}")
    print(f"Total runs: {len(entries)}")


def validate(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    completed = 0
    failed = 0
    for entry in entries:
        results_path = entry.get("results_path")
        if not results_path:
            entry["status"] = "pending"
            continue
        path = Path(results_path)
        if not path.exists():
            entry["status"] = "pending"
            continue
        ok, reason = validate_results_json(path, entry)
        if ok:
            entry["status"] = "completed"
            completed += 1
        else:
            entry["status"] = "invalid"
            failed += 1
            print(f"  INVALID: {entry['baseline']}/{entry['task']}/s{entry['seed']}: {reason}")
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Validation complete: {completed} completed, {failed} invalid, {len(entries) - completed - failed} pending")


def run(args):
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"Manifest not found: {manifest_path}")
        sys.exit(1)
    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    pending = [e for e in entries if e.get("status") != "completed"]
    print(f"Pending runs: {len(pending)}")
    for entry in pending:
        print(f"  {entry['baseline']}/{entry['task']}/s{entry['seed']}")


def execute(args):
    import subprocess
    import torch
    from run_phase1_2_simple_baselines import (
        run_single_baseline_experiment,
        detect_device_profile,
        release_cuda_memory,
    )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Refusing to run official Phase 1.2 on CPU."
        )

    profile = args.profile or detect_device_profile()
    max_samples = args.max_samples
    if max_samples is None and not args.unlimited:
        max_samples = None if profile == "server" else 2000

    gpu_selector = os.environ.get("LERNA_NVIDIA_SMI_GPU", "0")
    gpu_index = 0
    try:
        r = subprocess.run(
            ["nvidia-smi", "-i", gpu_selector,
             "--query-gpu=power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if r.returncode == 0:
            print(f"  Preflight GPU power read: {r.stdout.strip()}")
        else:
            print(f"  Preflight warning: nvidia-smi returned {r.returncode}")
    except Exception as exc:
        print(f"  Preflight warning: nvidia-smi query failed: {exc}")

    output_dir = Path(args.output)
    manifest_path = Path(args.manifest) if args.manifest else output_dir / "run_manifest.json"
    output_dir.mkdir(parents=True, exist_ok=True)

    baselines = args.baselines or DEFAULT_BASELINES
    tasks = args.tasks or DEFAULT_TASKS
    seeds = args.seeds or DEFAULT_SEEDS
    entries = build_matrix(baselines=baselines, tasks=tasks, seeds=seeds)
    manifest = {
        "phase": "1.2-ettin",
        "model_name": EXPECTED_MODEL,
        "model_revision": EXPECTED_REVISION,
        "output_dir": str(output_dir),
        "created_at": datetime.now().isoformat(),
        "total_runs": len(entries),
        "max_samples": None,
        "dataset_scope": "full GLUE",
        "wandb": "disabled",
        "entries": entries,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))

    succeeded = failed = 0
    for entry in entries:
        baseline = entry["baseline"]
        task = entry["task"]
        seed = entry["seed"]
        try:
            lr = TASK_HP_OVERRIDES.get(task, {}).get("learning_rate", 2e-5)
            result = run_single_baseline_experiment(
                baseline_name=baseline,
                task_name=task,
                seed=seed,
                profile=profile,
                base_output_dir=str(output_dir),
                max_samples_override=max_samples,
                model_name=EXPECTED_MODEL,
                model_revision=EXPECTED_REVISION,
                unlimited=getattr(args, "unlimited", False),
            )
            entry["status"] = "completed"
            entry["results_path"] = str(output_dir / f"{baseline}/{task}_s{seed}_lr{lr:.0e}" / "results.json")
            succeeded += 1
        except Exception as exc:
            import traceback
            entry["status"] = "failed"
            failed += 1
            print(f"  FAILED: {baseline}/{task}/s{seed}: {exc}")
            traceback.print_exc()
        finally:
            release_cuda_memory()

    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"Session complete: {succeeded} succeeded, {failed} failed")


def main():
    ap = argparse.ArgumentParser(description="LERNA Phase 1.2 Ettin-150m manifest launcher")
    ap.add_argument("--mode", choices=["generate", "validate", "run", "execute"], default="generate")
    ap.add_argument("--output-dir", default=str(OUTPUT), dest="output")
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--reset-manifest", action="store_true", help="Force regenerate manifest")
    ap.add_argument("--initialize-only", action="store_true", help="Generate manifest and exit")
    ap.add_argument("--baselines", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--profile", choices=["cpu", "laptop", "server"], default=None)
    ap.add_argument("--unlimited", action="store_true")
    args = ap.parse_args()


    if args.reset_manifest:
        generate(args)

    if args.initialize_only:
        return

    if args.mode == "run" or args.mode == "execute":
        execute(args)
    elif args.mode == "generate":
        generate(args)
    elif args.mode == "validate":
        validate(args)


if __name__ == "__main__":
    main()
