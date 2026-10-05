"""Timing cells for Stage 2: ms/step per arm on an isolated GPU.

Runs 3 seeds per arm (gate, control, full_finetune, reduced_steps) = 12 cells.
Alternates arm order to avoid scheduling bias. Takes ms/step from the trainer's
instrumented step timer, not from total wall time.

B14: alternation uses cell index (not seed % 2, which is always 1 for odd seeds).
B14: f-strings guard against None from instrumentation keys.

Usage:
    python scripts/run_timing_cells.py \
        --score s1 \
        --seeds 11 13 17 \
        --rate 0.60 \
        --model jhu-clsp/ettin-encoder-150m \
        --model-rev 45d0864 \
        --output-dir /raid/stage2_timing/
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ARMS = ["gate", "control", "full_finetune", "reduced_steps"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--score", choices=["s1", "s2"], default="s1")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 13, 17])
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--model-rev", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--probe-file", type=str,
                        default="experiments/probe_indices_v4.json")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(repo_root / "scripts"))

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    from stage2_confirmatory import run_cell, _load_probe_data, _check_provenance

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # B5: provenance check before any training
    probe_data = _load_probe_data(str(repo_root / "experiments" / "probe_indices_v4.json"))
    run_sha = _check_provenance(probe_ids_sha=probe_data.get("ids_sha256"))

    # B14: Build cell list with alternating arm order per cell index
    cells = []
    cell_idx = 0
    for seed in args.seeds:
        arm_order = ARMS[:]
        # B14: use cell_idx % 2, not seed % 2 (all seeds are odd → seed % 2 == 1 always)
        order = arm_order if cell_idx % 2 == 0 else list(reversed(arm_order))
        for arm in order:
            cells.append((arm, seed))
            cell_idx += 1

    results = []
    for arm, seed in cells:
        print(f"\n=== timing: arm={arm} seed={seed} ===")
        r = run_cell(
            arm, seed, args.rate, args.total_steps,
            args.score, args.model, args.model_rev,
            output,
            probe_data=probe_data, run_sha=run_sha,
        )
        instr_path = output / arm / str(seed) / "instrumentation.json"
        if instr_path.exists():
            with open(instr_path) as f:
                instr = json.load(f)
            r["mean_step_ms_executed"] = instr.get("mean_step_ms_executed")
            r["mean_step_ms_skipped"] = instr.get("mean_step_ms_skipped")
            r["mean_step_ms"] = (
                instr.get("mean_step_ms_executed") or
                instr.get("mean_step_ms_skipped")
            )
        results.append(r)
        ms_val = r.get("mean_step_ms")
        ms_str = f"{ms_val:.2f}" if ms_val is not None else "N/A"
        acc = r.get("eval_metrics", {}).get("eval_accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        sched = r.get("true_skip_instrumentation", {}).get("scheduler_step_calls", "N/A")
        print(f"  ms/step={ms_str} "
              f"final_acc={acc_str} "
              f"scheduler_calls={sched}")

    # Write timing summary
    summary = {
        "stage": "timing_cells",
        "score_source": args.score,
        "seeds": args.seeds,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(output / "timing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nTiming summary written to {output / 'timing_summary.json'}")


if __name__ == "__main__":
    main()
