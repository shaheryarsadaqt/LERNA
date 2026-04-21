"""Compute energy delta between paired LERNA and baseline runs.

Usage:
    python scripts/compute_paired_energy.py \
        --lerna-dir runs/lerna/ --baseline-dir runs/baseline/ \
        --out reports/energy_paired.json
"""
import json
import argparse
import glob
import os
import numpy as np
from scipy import stats


def load_run_summaries(d):
    """Load run results from each run directory.
    
    Supports two formats:
    1. run_summary.json with:
       {
           "seed": int,
           "task": str,
           "total_energy_j": float,
           "wall_time_s": float,
           "eval_metric": float,
       }
    2. train_results.json (from run_baseline_glue.py) with:
       {
           "seed": int,
           "task": str,
           "energy_kwh": float,
           "train_runtime_s": float,
           "eval_metrics": {"eval_accuracy": float, ...},
       }
    """
    summaries = []
    
    # Try run_summary.json first (preferred format)
    for p in glob.glob(os.path.join(d, "**/run_summary.json"), recursive=True):
        try:
            with open(p) as f:
                data = json.load(f)
                # Normalize to common format
                summary = {
                    "seed": data.get("seed"),
                    "task": data.get("task"),
                    "total_energy_j": data.get("total_energy_j", data.get("energy_kwh", 0) * 3.6e6),
                    "wall_time_s": data.get("wall_time_s", data.get("train_runtime_s", 0)),
                    "eval_metric": data.get("eval_metric") or _extract_eval_metric(data),
                }
                if summary["seed"] and summary["task"]:
                    summaries.append(summary)
        except Exception as e:
            print(f"Warning: Could not load {p}: {e}")
    
    # If no run_summary.json found, try train_results.json
    if not summaries:
        for p in glob.glob(os.path.join(d, "**/train_results.json"), recursive=True):
            try:
                with open(p) as f:
                    data = json.load(f)
                    summary = {
                        "seed": data.get("seed"),
                        "task": data.get("task"),
                        "total_energy_j": data.get("energy_kwh", 0) * 3.6e6,  # kWh -> J
                        "wall_time_s": data.get("train_runtime_s", 0),
                        "eval_metric": _extract_eval_metric(data),
                    }
                    if summary["seed"] and summary["task"]:
                        summaries.append(summary)
            except Exception as e:
                print(f"Warning: Could not load {p}: {e}")
    
    return summaries


def _extract_eval_metric(data):
    """Extract the primary eval metric from eval_metrics dict."""
    em = data.get("eval_metrics", {})
    if not em:
        return 0.0
    # Priority: accuracy > matthews_correlation > pearsonr
    return em.get("eval_accuracy",
           em.get("eval_matthews_correlation",
           em.get("eval_pearsonr", 0.0)))


def pair_runs(lerna_runs, baseline_runs):
    """Pair LERNA runs with baseline runs by (task, seed)."""
    key = lambda r: (r["task"], r["seed"])
    b = {key(r): r for r in baseline_runs}
    pairs = [(l, b[key(l)]) for l in lerna_runs if key(l) in b]
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Compute paired energy delta")
    parser.add_argument("--lerna-dir", required=True, help="Directory containing LERNA run summaries")
    parser.add_argument("--baseline-dir", required=True, help="Directory containing baseline run summaries")
    parser.add_argument("--out", required=True, help="Output JSON file path")
    args = parser.parse_args()

    lerna_runs = load_run_summaries(args.lerna_dir)
    baseline_runs = load_run_summaries(args.baseline_dir)

    if not lerna_runs:
        raise SystemExit(f"No LERNA runs found in {args.lerna_dir}")
    if not baseline_runs:
        raise SystemExit(f"No baseline runs found in {args.baseline_dir}")

    pairs = pair_runs(lerna_runs, baseline_runs)
    if not pairs:
        raise SystemExit("No matched (task,seed) pairs found between LERNA and baseline runs.")

    # Relative energy savings per pair (signed: positive = LERNA saved energy)
    rel = np.array([
        (b["total_energy_j"] - l["total_energy_j"]) / b["total_energy_j"]
        for l, b in pairs
    ])
    
    # Metric delta (LERNA - baseline)
    d_metric = np.array([
        l["eval_metric"] - b["eval_metric"]
        for l, b in pairs
    ])

    # Paired bootstrap CI (10_000 resamples)
    rng = np.random.default_rng(0)
    n = len(rel)
    boot_e = np.empty(10_000)
    boot_m = np.empty(10_000)
    for i in range(10_000):
        idx = rng.integers(0, n, n)
        boot_e[i] = rel[idx].mean()
        boot_m[i] = d_metric[idx].mean()
    ci_e = np.percentile(boot_e, [2.5, 97.5])
    ci_m = np.percentile(boot_m, [2.5, 97.5])

    # Paired t-test on energy
    t, p = stats.ttest_rel(
        np.array([b["total_energy_j"] for _, b in pairs]),
        np.array([l["total_energy_j"] for l, _ in pairs])
    )

    out = {
        "n_pairs": n,
        "energy_saving_mean": float(rel.mean()),
        "energy_saving_ci95": [float(ci_e[0]), float(ci_e[1])],
        "metric_delta_mean": float(d_metric.mean()),
        "metric_delta_ci95": [float(ci_m[0]), float(ci_m[1])],
        "paired_t": float(t),
        "p_value": float(p),
        "pairs": [
            {
                "task": l["task"],
                "seed": l["seed"],
                "lerna_energy_j": l["total_energy_j"],
                "baseline_energy_j": b["total_energy_j"],
                "energy_saving_pct": float((b["total_energy_j"] - l["total_energy_j"]) / b["total_energy_j"] * 100),
                "metric_delta": float(l["eval_metric"] - b["eval_metric"]),
            }
            for l, b in pairs
        ]
    }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    
    print(json.dumps(out, indent=2))
    print(f"\nResults written to {args.out}")


if __name__ == "__main__":
    main()