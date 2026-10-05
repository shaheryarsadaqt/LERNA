"""Launcher for Stage 2: dispatch cells across 8 GPUs.

Runs one cell per GPU, 8 cells in parallel. Each cell writes to its own output
directory. The launcher collects completion status and writes a batch summary.

B6: GPUs are dynamically assigned via a free-GPU queue (queue.Queue) so that
      cells of unequal wall-time never collide on the same GPU.
B7: Reserve-seed swap in all 4 arms — when a cell crashes, the replacement
      seed is substituted across all arms (including already-completed ones),
      which are re-run. This is handled in stage2_confirmatory.py main().
The launcher mirrors the serial-path retry logic: retry-once on same GPU,
then fall back to a reserve seed.

Usage:
    python scripts/run_stage2_multi_gpu.py \
        --score s1 \
        --seeds 11 13 17 19 23 ... \
        --rate 0.60 \
        --total-steps 6548 \
        --model jhu-clsp/ettin-encoder-150m \
        --model-rev 45d0864 \
        --n-gpus 8 \
        --output-dir /raid/stage2_confirmatory/
"""
import argparse
import json
import os
import queue
import subprocess
import sys
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from typing import Optional


ARMS = ["gate", "control", "full_finetune", "reduced_steps"]
RESERVE_SEEDS = [157, 163]  # B7: reserve seeds for crashes


class GPUPool:
    """B6: Dynamic GPU allocation to prevent collisions when cells have
    unequal wall-time. Callers acquire() a GPU id at start and release() it
    when done."""

    def __init__(self, n_gpus: int):
        self._q: queue.Queue = queue.Queue()
        for i in range(n_gpus):
            self._q.put(i)

    def acquire(self, timeout: float = 1.0) -> int:
        return self._q.get(timeout=timeout)

    def release(self, gpu: int) -> None:
        self._q.put(gpu)


class CrashLog:
    """B6/B7: Crash log shared across all cell workers."""

    def __init__(self):
        self.entries: list = []

    def add(self, entry: dict) -> None:
        self.entries.append(entry)


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
    crash_log = CrashLog()
    active_seeds = list(args.seeds)
    reserve_seeds = list(RESERVE_SEEDS)
    pool = GPUPool(args.n_gpus)

    # B7: Track per-arm seed state. When a seed is replaced, all arms must
    # re-run with the new seed.
    cell_results: dict[tuple[str, int], dict] = {}

    def run_one_cell(cell_idx, arm, seed, attempt=0, reserve_swapped=False):
        """Run a single cell with GPU pool assignment and retry-once.

        B6: GPU is acquired from the pool (not cell_idx % 8).
        B7: If the cell crashes twice, swap in a reserve seed for ALL arms.
        """
        gpu = pool.acquire()
        try:
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
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    env=env, cwd=cwd)
            wall = time.perf_counter() - t0
            return cell_idx, arm, seed, gpu, result.returncode, wall, result.stdout, result.stderr, None
        except Exception as e:
            return cell_idx, arm, seed, gpu, -1, 0.0, "", str(e), traceback.format_exc()
        finally:
            pool.release(gpu)

    with ThreadPoolExecutor(max_workers=args.n_gpus) as executor:
        futures = []
        for i, (arm, seed) in enumerate(cells):
            futures.append(executor.submit(run_one_cell, i, arm, seed))

        while futures:
            future = futures.pop(0)
            cell_idx, arm, seed, gpu, rc, wall, stdout, stderr, tb = future.result()

            key = (arm, seed)
            status = "SUCCESS" if rc == 0 else "FAILED"
            print(f"[{status}] arm={arm} seed={seed} gpu={gpu} wall={wall:.1f}s")

            if rc == 0:
                completed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall})
                cell_results[key] = {"status": "completed", "wall": wall}
            else:
                # B6: retry-once on same GPU with same seed
                if not crash_log.entries or all(
                    e.get("key") != key for e in crash_log.entries
                ):
                    print(f"  Retrying arm={arm} seed={seed} (attempt 2)")
                    retry_future = executor.submit(run_one_cell, cell_idx, arm, seed,
                                                   attempt=1)
                    retry_result = retry_future.result()
                    r_rc = retry_result[4]
                    if r_rc == 0:
                        completed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall})
                        cell_results[key] = {"status": "completed", "wall": wall}
                    else:
                        # B7: swap reserve seed in ALL arms
                        if reserve_seeds:
                            new_seed = reserve_seeds.pop(0)
                            crash_log.add({
                                "arm": arm, "orig_seed": seed,
                                "replacement_seed": new_seed,
                                "error": stderr[-500:] or tb,
                                "attempts": 2,
                            })
                            # Remove old cell results for this arm and re-queue with new seed
                            for old_key in list(cell_results.keys()):
                                if old_key[0] == arm:
                                    del cell_results[old_key]
                            # Re-queue ALL arms with the new seed (B7)
                            for a in ARMS:
                                if (a, new_seed) not in cell_results:
                                    new_i = len(futures) + len(completed)
                                    futures.append(executor.submit(
                                        run_one_cell, new_i, a, new_seed
                                    ))
                        else:
                            failed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall,
                                           "stderr": stderr[-500:]})
                else:
                    failed.append({"arm": arm, "seed": seed, "gpu": gpu, "wall": wall,
                                   "stderr": stderr[-500:]})

    summary = {
        "total_cells": len(cells),
        "completed": len(completed),
        "failed": len(failed),
        "completed_list": completed,
        "failed_list": failed,
        "crash_log": crash_log.entries,
        "n_gpus": args.n_gpus,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output / "launcher_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: {len(completed)}/{len(cells)} cells completed")


if __name__ == "__main__":
    main()
