#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MISSING = ROOT / "missing_phase1_2_anylr.txt"
DEFAULT_OUTPUT = ROOT / "experiments" / "phase1_2_baselines"
DEFAULT_MANIFEST = DEFAULT_OUTPUT / "run_manifest_phase1_2_missing.json"
LINE_RE = re.compile(r"^MISSING\s+(?P<baseline>\S+)\s+(?P<task>\S+)\s+seed=(?P<seed>\d+)\s*$")


def run_key(run):
    return f"{run['baseline']}/{run['task']}/seed={run['seed']}"


def load_missing(path: Path):
    runs, seen = [], set()
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = LINE_RE.match(line)
        if not m:
            raise ValueError(f"Invalid missing-run line {line_no}: {raw!r}")
        run = {"baseline": m.group("baseline"), "task": m.group("task"), "seed": int(m.group("seed"))}
        key = run_key(run)
        if key not in seen:
            runs.append(run)
            seen.add(key)
    return runs


def load_manifest(path: Path):
    if not path.exists():
        return {"runs": {}, "created_at": datetime.now().isoformat()}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        backup.write_text(path.read_text())
        return {"runs": {}, "created_at": datetime.now().isoformat()}


def save_manifest(path: Path, manifest):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(tmp, path)


def free_gb(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / (1024 ** 3)


def find_existing_result(output_dir: Path, run):
    pattern = output_dir / run["baseline"] / f"{run['task']}_s{run['seed']}_lr*" / "results.json"
    for candidate in sorted(glob.glob(str(pattern))):
        path = Path(candidate)
        try:
            data = json.loads(path.read_text())
        except Exception:
            continue
        if data.get("baseline") == run["baseline"] and data.get("task") == run["task"] and int(data.get("seed", -1)) == run["seed"]:
            return path
    return None


def classify_runs(runs, output_dir, manifest, retry_failed):
    pending, skipped = [], []
    for run in runs:
        key = run_key(run)
        entry = manifest.get("runs", {}).get(key, {})
        if entry.get("status") == "completed":
            skipped.append((run, "manifest-completed"))
            continue
        if entry.get("status") == "failed" and not retry_failed:
            skipped.append((run, "manifest-failed"))
            continue
        existing = find_existing_result(output_dir, run)
        if existing:
            manifest.setdefault("runs", {})[key] = {
                "status": "completed",
                "results_path": str(existing),
                "completed_at": "detected-existing-results",
            }
            skipped.append((run, "existing-results"))
            continue
        pending.append(run)
    return pending, skipped


def append_line(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(text.rstrip() + "\n")


def cleanup_wandb_local(root: Path):
    wandb_dir = root / "wandb"
    if not wandb_dir.exists():
        return
    for path in wandb_dir.glob("run-*"):
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def execute(args, pending, skipped, manifest):
    sys.path.insert(0, str(ROOT))
    from scripts.run_phase1_2_simple_baselines import (
        cleanup_phase1_2_run_dir,
        detect_device_profile,
        release_cuda_memory,
        run_single_baseline_experiment,
        _ensure_wandb_finished,
    )

    profile = args.profile or detect_device_profile()
    max_samples = args.max_samples
    if max_samples is None and not args.unlimited:
        max_samples = None if profile == "server" else 2000

    total = len(pending) + len(skipped)
    start = time.time()
    succeeded = failed = 0

    if args.wandb:
        _ensure_wandb_finished()

    for idx, run in enumerate(pending, 1):
        if free_gb(args.output_dir) < args.min_disk_gb:
            print(f"Disk guard paused: {free_gb(args.output_dir):.1f} GB free < {args.min_disk_gb:.1f} GB")
            break

        key = run_key(run)
        elapsed = time.time() - start
        eta = "calculating" if idx == 1 else str(timedelta(seconds=int((len(pending) - idx + 1) * elapsed / (idx - 1))))

        print("=" * 72, flush=True)
        print(f"Run {idx}/{len(pending)} | overall {len(skipped) + idx}/{total} | ETA {eta}", flush=True)
        print(f"{key} | profile={profile} | max_samples={max_samples or 'unlimited'}", flush=True)
        print(f"Disk free: {free_gb(args.output_dir):.1f} GB", flush=True)
        print("=" * 72, flush=True)

        result_path = None
        try:
            result = run_single_baseline_experiment(
                baseline_name=run["baseline"],
                task_name=run["task"],
                seed=run["seed"],
                profile=profile,
                base_output_dir=str(args.output_dir),
                use_wandb=args.wandb,
                max_samples_override=max_samples,
                run_idx=len(skipped) + idx,
                total_runs=total,
                wandb_project=args.wandb_project,
                wandb_group=args.wandb_group,
                target_skip_rate=args.target_skip_rate,
                cleanup_artifacts=not args.no_cleanup_artifacts,
                keep_power_samples=args.keep_power_samples,
                keep_debug_artifacts=args.keep_debug_artifacts,
            )
            result_path = find_existing_result(args.output_dir, run)
            manifest.setdefault("runs", {})[key] = {
                "status": "completed",
                "results_path": str(result_path) if result_path else None,
                "completed_at": datetime.now().isoformat(),
                "primary_metric": result.get("primary_metric"),
                "energy_kwh": result.get("energy_kwh"),
            }
            save_manifest(args.manifest, manifest)
            append_line(args.completed_log, f"DONE {run['baseline']} {run['task']} seed={run['seed']} path={result_path}")
            succeeded += 1

        except Exception as exc:
            traceback.print_exc()
            result_path = find_existing_result(args.output_dir, run)
            if not args.no_cleanup_artifacts:
                candidates = sorted((args.output_dir / run["baseline"]).glob(f"{run['task']}_s{run['seed']}_lr*"))
                run_dir = result_path.parent if result_path else (candidates[-1] if candidates else None)
                if run_dir:
                    cleanup_phase1_2_run_dir(
                        str(run_dir),
                        keep_power_samples=args.keep_power_samples,
                        keep_debug_artifacts=args.keep_debug_artifacts,
                    )
            release_cuda_memory()
            manifest.setdefault("runs", {})[key] = {
                "status": "failed",
                "error": str(exc),
                "failed_at": datetime.now().isoformat(),
                "results_path": str(result_path) if result_path else None,
            }
            save_manifest(args.manifest, manifest)
            append_line(args.failed_log, f"FAILED {run['baseline']} {run['task']} seed={run['seed']} error={exc}")
            failed += 1
            if args.wandb:
                _ensure_wandb_finished()

        finally:
            if args.cleanup_wandb_local:
                cleanup_wandb_local(ROOT)
            release_cuda_memory()

    print("=" * 72)
    print(f"Session complete: {succeeded} succeeded, {failed} failed")
    print(f"Manifest: {args.manifest}")
    print(f"Disk free: {free_gb(args.output_dir):.1f} GB")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--missing-file", type=Path, default=DEFAULT_MISSING)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--completed-log", type=Path, default=ROOT / "completed_phase1_2_resume.txt")
    p.add_argument("--failed-log", type=Path, default=ROOT / "failed_phase1_2_resume.txt")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="lerna-phase1.2")
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--target-skip-rate", type=float, default=0.33)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--unlimited", action="store_true")
    p.add_argument("--profile", choices=["cpu", "laptop", "server"], default=None)
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--no-cleanup-artifacts", action="store_true")
    p.add_argument("--keep-power-samples", action="store_true")
    p.add_argument("--keep-debug-artifacts", action="store_true")
    p.add_argument("--cleanup-wandb-local", action="store_true")
    p.add_argument("--min-disk-gb", type=float, default=30.0)
    p.add_argument("--status", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    args.output_dir = args.output_dir.resolve()
    args.manifest = args.manifest.resolve()
    if args.wandb_group is None:
        args.wandb_group = f"phase1.2-resume-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    runs = load_missing(args.missing_file)
    manifest = load_manifest(args.manifest)
    pending, skipped = classify_runs(runs, args.output_dir, manifest, args.retry_failed)
    save_manifest(args.manifest, manifest)

    print(f"Missing-list runs: {len(runs)}")
    print(f"Skipped: {len(skipped)}")
    print(f"Pending: {len(pending)}")
    print(f"Output: {args.output_dir}")
    print(f"Manifest: {args.manifest}")
    print(f"Disk free: {free_gb(args.output_dir):.1f} GB")

    if args.status or args.dry_run:
        for i, run in enumerate(pending, 1):
            print(f"{i:04d}. {run_key(run)}")
        return

    execute(args, pending, skipped, manifest)


if __name__ == "__main__":
    main()
