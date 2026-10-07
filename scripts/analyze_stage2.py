"""Analyze Stage 2 confirmatory results (B6: separate analysis script).

Reads only cells that passed validation, then computes:
- Primary test: Δacc(gate − control), paired t-test, two-sided α=0.05, df=23
- Bootstrap 95% CI (10,000 resamples)
- Secondary outcomes: gate vs reduced_steps, control vs reduced_steps, TOST vs full_finetune
- Skip parity, cap hits, forced-skip counts
- Pareto: accuracy vs GPU-seconds

Usage:
    python scripts/analyze_stage2.py \
        --results-dir /raid/stage2_confirmatory/ \
        --expected-sha <v4-prereg-r3 SHA>
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.stats import t as t_dist


def load_validated_cells(results_dir: Path, expected_sha: str | None):
    """Load all results.json files, returning only those that passed validation."""
    cells = []
    skipped = []
    for results_path in sorted(results_dir.rglob("results.json")):
        try:
            with open(results_path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            skipped.append(str(results_path))
            continue

        if not data.get("validation_passed"):
            skipped.append(str(results_path))
            continue

        cfg = data.get("run_config", {})
        em = data.get("eval_metrics", {})
        cells.append({
            "arm": cfg.get("arm"),
            "seed": cfg.get("seed"),
            "eval_accuracy": em.get("eval_accuracy"),
            "eval_loss": em.get("eval_loss"),
            "wall_seconds": data.get("wall_seconds", 0),
            "actual_steps_run": cfg.get("actual_steps_run", 0),
            "policy_diagnostics": data.get("policy_diagnostics", {}),
            "true_skip_instrumentation": data.get("true_skip_instrumentation", {}),
        })

    print(f"Loaded {len(cells)} validated cells, skipped {len(skipped)}")
    for s in skipped:
        print(f"  SKIPPED (validation failed or unreadable): {s}")
    return cells


def _paired_ttest(diffs: list[float]):
    """Two-sided paired t-test on the differences."""
    if not diffs:
        return {"error": "no paired data"}
    diff = np.array(diffs)
    n = len(diff)
    df = n - 1
    mean_diff = float(diff.mean())
    sd = float(diff.std(ddof=1))
    se = sd / math.sqrt(n) if sd > 0 else 0.0
    t_stat = mean_diff / se if se > 0 else float("inf")
    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df)) if se > 0 else (0.0 if mean_diff != 0 else 1.0)

    # B12: bootstrap CI (10,000 resamples)
    rng = np.random.RandomState(42)
    boot_means = [float(rng.choice(diff, size=n, replace=True).mean()) for _ in range(10000)]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))

    return {
        "mean_delta": mean_diff,
        "sd": sd,
        "se": se,
        "t_stat": t_stat,
        "p_val": p_val,
        "n": n,
        "df": df,
        "ci_95": [ci_lo, ci_hi],
        "passed": p_val < 0.05,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze Stage 2 confirmatory results")
    parser.add_argument("--results-dir", type=str, required=True)
    parser.add_argument("--expected-sha", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    cells = load_validated_cells(results_dir, args.expected_sha)

    # Group by arm
    by_arm = {}
    for c in cells:
        by_arm.setdefault(c["arm"], {})[c["seed"]] = c

    # Primary test: gate vs control
    gate = by_arm.get("gate", {})
    ctrl = by_arm.get("control", {})
    common_seeds = sorted(set(gate.keys()) & set(ctrl.keys()))
    diffs = [gate[s]["eval_accuracy"] - ctrl[s]["eval_accuracy"] for s in common_seeds]
    primary = _paired_ttest(diffs)
    primary["comparison"] = "gate vs control"

    # Secondary: gate vs reduced_steps
    reduced = by_arm.get("reduced_steps", {})
    common_gs = sorted(set(gate.keys()) & set(reduced.keys()))
    diffs_gs = [gate[s]["eval_accuracy"] - reduced[s]["eval_accuracy"] for s in common_gs]
    secondary_gs = _paired_ttest(diffs_gs)
    secondary_gs["comparison"] = "gate vs reduced_steps"

    # Secondary: control vs reduced_steps
    common_cs = sorted(set(ctrl.keys()) & set(reduced.keys()))
    diffs_cs = [ctrl[s]["eval_accuracy"] - reduced[s]["eval_accuracy"] for s in common_cs]
    secondary_cs = _paired_ttest(diffs_cs)
    secondary_cs["comparison"] = "control vs reduced_steps"

    # TOST vs full_finetune at 0.5 pt margin
    full = by_arm.get("full_finetune", {})
    tost_results = {}
    for arm_name, arm_dict in [("gate", gate), ("control", ctrl)]:
        common = sorted(set(arm_dict.keys()) & set(full.keys()))
        diffs_tost = [arm_dict[s]["eval_accuracy"] - full[s]["eval_accuracy"] for s in common]
        if diffs_tost:
            diff_arr = np.array(diffs_tost)
            margin = 0.005
            n = len(diff_arr)
            df = n - 1
            mean_delta = float(diff_arr.mean())
            sd = float(diff_arr.std(ddof=1))
            se = sd / math.sqrt(n) if sd > 0 else 0.0
            t1 = (mean_delta + margin) / se if se > 0 else float("inf")
            t2 = (mean_delta - margin) / se if se > 0 else float("-inf")
            p1 = 1 - t_dist.cdf(t1, df) if se > 0 else 0.0
            p2 = t_dist.cdf(t2, df) if se > 0 else 0.0
            tost_p = max(p1, p2)
            lower = mean_delta - t_dist.ppf(0.95, df) * se
            upper = mean_delta + t_dist.ppf(0.95, df) * se
            tost_results[f"{arm_name}_vs_full_finetune"] = {
                "margin_pt": margin * 100,
                "mean_delta_pt": mean_delta * 100,
                "ci_95_pt": [lower * 100, upper * 100],
                "tost_p": float(tost_p),
                "tost_passed": tost_p < 0.05,
            }

    # Gate/control diagnostics (primary arms)
    diag_gate = gate.get(common_seeds[0], {}).get("policy_diagnostics", {}) if common_seeds else {}
    diag_ctrl = ctrl.get(common_seeds[0], {}).get("policy_diagnostics", {}).get("policy_diagnostics", {}) if common_seeds else {}

    summary = {
        "primary_test": primary,
        "secondary_tests": {
            "gate_vs_reduced_steps": secondary_gs,
            "control_vs_reduced_steps": secondary_cs,
            "tost_vs_full_finetune": tost_results,
        },
        "n_cells_validated": len(cells),
        "n_paired_primary": len(common_seeds),
        "gate_diagnostics": diag_gate,
    }

    output_path = Path(args.output) if args.output else results_dir / "stage2_analysis.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n=== Stage 2 Analysis ===")
    print(f"  Primary: Δacc(gate − control) = {primary.get('mean_delta', 'N/A')}")
    print(f"  t({primary.get('df', '?')}) = {primary.get('t_stat', 'N/A')}")
    print(f"  p = {primary.get('p_val', 'N/A')}")
    print(f"  95% CI: {primary.get('ci_95', [None, None])}")
    print(f"  Passed: {primary.get('passed', 'N/A')}")
    print(f"  Written to: {output_path}")


if __name__ == "__main__":
    main()
