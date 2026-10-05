"""Launcher for Stage 2: dispatch cells across 8 GPUs.

Runs one cell per GPU, 8 cells in parallel. Each cell writes to its own output
directory. The launcher collects completion status and writes a batch summary.

Usage:
    python scripts/run_stage2_multi_gpu.py \
        --score s1 \
        --seeds 11 13 17 19 23 ... \
        --rate 0.60 \
        --total-steps 6548 \
        --model jhu-clsp/ettin-encoder-150m@45d0864 \
        --n-gpus 8 \
        --output-dir /raid/stage2_confirmatory/
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ARMS = ["gate", "control", "full_finetune", "reduced_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", choices=["s1", "s2"], default="s1")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-rev", type=str, required=True)
    parser.add_argument("--n-gpus", type=int, default=8)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--probe-file", type=str,
                        default="experiments/probe_indices_v4.json")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Build all cell assignments
    cells = []
    for arm in ARMS:
        for seed in args.seeds:
            cells.append((arm, seed))

    print(f"Total cells: {len(cells)} across {args.n_gpus} GPUs")

    completed = []
    failed = []

    def run_one_cell(cell_idx, arm, seed):
        gpu = cell_idx % args.n_gpus
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        cmd = [
            sys.executable, "scripts/stage2_confirmatory.py",
            "--score", args.score,
            "--seeds", str(seed),
            "--rate", str(args.rate),
            "--total-steps", str(args.total_steps),
            "--model", args.model,
            "--model-rev", args.model_rev,
            "--output-dir", args.output_dir,
            "--probe-file", args.probe_file,
            "--single-cell",
            "--arm", arm,
            "--single-seed", str(seed),
            "--gpu", str(gpu),
        ]
        cwd = str(Path(__file__).resolve().parents[1])
        t0 = time.perf_counter()
        result = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=cwd)
        wall = time.perf_counter() - t0
        return cell_idx, arm, seed, gpu, result.returncode, wall, result.stdout, result.stderr

    with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
        futures = []
        for i, (arm, seed) in enumerate(cells):
            futures.append(executor.submit(run_one_cell, i, arm, seed))

        for future in as_completed(futures):
            cell_idx, arm, seed, gpu, rc, wall, stdout, stderr = future.result()
            status = "SUCCESS" if rc == 0 else "FAILED"
            print(f"[{status}] arm={arm} seed={seed} gpu={gpu} wall={wall:.1f}s")
            if rc == 0:
                completed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall})
            else:
                failed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall,
                               "stderr": stderr[-500:]})

    summary = {
        "total_cells": len(cells),
        "completed": len(completed),
        "failed": len(failed),
        "completed_list": completed,
        "failed_list": failed,
        "n_gpus": args.n_gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output / "launcher_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(completed)}/{len(cells)} cells completed")


if __name__ == "__main__":
    main()
