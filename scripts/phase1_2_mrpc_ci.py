#!/usr/bin/env python3
"""Compute confidence intervals for MRPC 10-seed waste values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture


def load_wastes_from_wandb(run_path: str) -> list[float]:
    """Load waste values from a W&B run folder or results JSON."""
    path = Path(run_path)
    if path.is_dir():
        results = path / "results.json"
        if results.exists():
            with open(results) as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [r.get("waste_metrics", {}).get("waste_ratio", 0.0) for r in data]
                elif isinstance(data, dict):
                    return [data.get("waste_metrics", {}).get("waste_ratio", 0.0)]
    elif path.suffix == ".json":
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                return [r.get("waste_metrics", {}).get("waste_ratio", 0.0) for r in data]
    raise ValueError(f"Could not parse waste values from {run_path}")


def compute_t_ci(values: list[float], alpha: float = 0.95) -> tuple[float, float, float]:
    """Compute mean with t-distribution CI."""
    x = np.asarray(values)
    n = len(x)
    mean = x.mean()
    se = stats.sem(x)
    ci = stats.t.interval(alpha, n - 1, loc=mean, scale=se)
    return float(mean), float(ci[0]), float(ci[1])


def report_bimodal(values: list[float], task_name: str = "MRPC") -> str:
    """Report with GMM bimodality detection."""
    x = np.array(values).reshape(-1, 1)
    if len(x) < 4:
        return f"{task_name}: n<4, cannot fit mixture"

    gmm1 = GaussianMixture(1, random_state=0).fit(x)
    gmm2 = GaussianMixture(2, random_state=0).fit(x)
    if gmm2.bic(x) < gmm1.bic(x) - 4:
        means = gmm2.means_.flatten()
        weights = gmm2.weights_
        return (
            f"BIMODAL — "
            f"{weights[0]*100:.0f}% at {means[0]:.2%}, "
            f"{weights[1]*100:.0f}% at {means[1]:.2%}"
        )
    return f"unimodal: {x.mean():.2%} ± {x.std():.2%}"


def main():
    parser = argparse.ArgumentParser(description="MRPC waste CI + bimodal report")
    parser.add_argument("wastes", nargs="*", type=float, help="Waste ratio values (0-1)")
    parser.add_argument("--input-json", type=str, help="Path to results.json from a run")
    parser.add_argument("--alpha", type=float, default=0.95, help="CI level (default 0.95)")
    args = parser.parse_args()

    if args.input_json:
        wastes = load_wastes_from_wandb(args.input_json)
    elif args.wastes:
        wastes = args.wastes
    else:
        print("Provide --wastes ... or --input-json")
        return

    wastes = [v for v in wastes if v is not None and not np.isnan(v)]
    if not wastes:
        print("No valid waste values found.")
        return

    n = len(wastes)
    mean, ci_low, ci_high = compute_t_ci(wastes, args.alpha)

    print(f"\n=== MRPC Waste Analysis (n={n}) ===")
    print(f"Values: {[f'{v:.3f}' for v in sorted(wastes)]}")
    print(f"Mean:   {mean:.3f}")
    print(f"95% CI: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"Bimodal: {report_bimodal(wastes)}")

    print(f"\nASCII histogram:")
    for i in range(0, 101, 10):
        bar = "".join("█" if int(v * 100) >= i else " " for v in sorted(wastes))
        print(f"  {i:3d}% |{bar}|")


if __name__ == "__main__":
    main()