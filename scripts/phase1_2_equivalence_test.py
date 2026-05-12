#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2 — Step 3/5: TOST Equivalence Testing
=============================================================================

A non-significant paired t-test ≠ "the two methods are equivalent".
To make the strong claim "LERNA is meaningfully different from this baseline"
on accuracy AND/OR to support a "baseline approaches LERNA" claim, we use
TOST (Two One-Sided Tests; Schuirmann 1987, Lakens 2017).

For each (baseline, task) pair we test:
    H0:   |mean(LERNA) − mean(baseline)| ≥ δ_margin   (not equivalent)
    H1:   |mean(LERNA) − mean(baseline)|  < δ_margin   (equivalent)

If TOST p < α   → REJECT H0  → statistically equivalent within margin δ.
If TOST p ≥ α   → CANNOT claim equivalence (could be different OR n too small).

Defaults:
    δ_margin (accuracy) = 0.005   (0.5 percentage points, the strategy target)
    δ_margin (energy)   = log(1.05) ≈ 0.0488   (5 % multiplicative)
    α                    = 0.05

Combined with significance_analysis.py this yields the classical 2×2 matrix
that publications expect:

                      |  signif. diff?   not signif. diff?
    -----------------+-----------------------------------------
    equivalent       |    trivial diff     practically same
    not equivalent   |    meaningful diff  inconclusive (need more n)

Outputs:
    equivalence_results.csv     equivalence_results.json
    equivalence_summary.txt

Usage:
    python scripts/phase1_2_equivalence_test.py \
        --input experiments/phase1_2_analysis/aggregated_long.csv \
        --output-dir experiments/phase1_2_analysis \
        --margin-accuracy 0.005 --margin-energy-pct 5.0

Author: LERNA Research Team
Phase:  1.2 — Analysis pipeline step 3/5
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phase1_2_tost")


def tost_paired(x: np.ndarray, y: np.ndarray, margin: float,
                alpha: float = 0.05) -> tuple[float, float, float, bool]:
    """Two one-sided t-tests for paired data.

    Returns (p_lower, p_upper, p_max, equivalent).
    `p_max` is the p-value of the *whole* TOST (max of the two one-sided p's);
    `equivalent` = p_max < alpha.
    """
    diff = x - y
    n = len(diff)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), False
    mean_d = diff.mean()
    sd_d = diff.std(ddof=1)
    if sd_d == 0:
        equiv = abs(mean_d) < margin
        return (0.0 if equiv else 1.0,
                0.0 if equiv else 1.0,
                0.0 if equiv else 1.0,
                equiv)
    se = sd_d / math.sqrt(n)
    df = n - 1
    t_lower = (mean_d + margin) / se     # H0_L: diff <= -margin
    t_upper = (mean_d - margin) / se     # H0_U: diff >=  margin
    p_lower = 1 - stats.t.cdf(t_lower, df)
    p_upper = stats.t.cdf(t_upper, df)
    p_max = max(p_lower, p_upper)
    return float(p_lower), float(p_upper), float(p_max), bool(p_max < alpha)


# =============================================================================
@dataclass
class EquivRow:
    baseline: str
    task: str
    n_seeds: int
    metric_delta: float
    metric_margin: float
    tost_p_lower_metric: float
    tost_p_upper_metric: float
    tost_p_metric: float
    equivalent_metric: bool
    energy_log_ratio: float
    energy_margin_log: float
    tost_p_lower_energy: float
    tost_p_upper_energy: float
    tost_p_energy: float
    equivalent_energy: bool


def _align(df: pd.DataFrame, baseline: str, task: str, ref: str):
    a = df[(df.baseline == ref) & (df.task == task)].set_index("seed")
    b = df[(df.baseline == baseline) & (df.task == task)].set_index("seed")
    common = sorted(set(a.index) & set(b.index))
    if len(common) < 2:
        return None
    return (a.loc[common, "primary_metric"].to_numpy(),
            b.loc[common, "primary_metric"].to_numpy(),
            a.loc[common, "energy_kwh"].to_numpy(),
            b.loc[common, "energy_kwh"].to_numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="experiments/phase1_2_analysis/aggregated_long.csv")
    ap.add_argument("--output-dir", default="experiments/phase1_2_analysis")
    ap.add_argument("--reference-baseline", default="lerna")
    ap.add_argument("--margin-accuracy", type=float, default=0.005,
                    help="Equivalence margin for primary metric (default ±0.005)")
    ap.add_argument("--margin-energy-pct", type=float, default=5.0,
                    help="Equivalence margin for energy as multiplicative %% (default ±5%%)")
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    margin_energy_log = math.log(1 + args.margin_energy_pct / 100.0)
    log.info(f"  margins: metric={args.margin_accuracy}, energy=±{args.margin_energy_pct}% "
             f"(log≈{margin_energy_log:.4f})")

    baselines = sorted([b for b in df.baseline.unique() if b != args.reference_baseline])
    tasks = sorted(df.task.unique())

    rows: list[EquivRow] = []
    for b in baselines:
        for t in tasks:
            aligned = _align(df, b, t, args.reference_baseline)
            if aligned is None:
                continue
            le_m, ba_m, le_e, ba_e = aligned
            pL, pU, pM, eqM = tost_paired(le_m, ba_m, args.margin_accuracy, args.alpha)

            log_le = np.log(np.maximum(le_e, 1e-12))
            log_ba = np.log(np.maximum(ba_e, 1e-12))
            pLe, pUe, pMe, eqE = tost_paired(log_le, log_ba, margin_energy_log, args.alpha)

            rows.append(EquivRow(
                baseline=b, task=t, n_seeds=len(le_m),
                metric_delta=float(np.mean(le_m - ba_m)),
                metric_margin=args.margin_accuracy,
                tost_p_lower_metric=pL, tost_p_upper_metric=pU,
                tost_p_metric=pM, equivalent_metric=eqM,
                energy_log_ratio=float(np.mean(log_ba - log_le)),
                energy_margin_log=margin_energy_log,
                tost_p_lower_energy=pLe, tost_p_upper_energy=pUe,
                tost_p_energy=pMe, equivalent_energy=eqE,
            ))

    if not rows:
        log.error("  no comparisons produced")
        return

    res_df = pd.DataFrame([asdict(r) for r in rows])
    res_df.to_csv(out_dir / "equivalence_results.csv", index=False)
    (out_dir / "equivalence_results.json").write_text(
        json.dumps({"phase": "1.2-tost",
                    "reference_baseline": args.reference_baseline,
                    "alpha": args.alpha,
                    "results": [asdict(r) for r in rows]}, indent=2))

    lines = []
    lines.append("=" * 100)
    lines.append(f"  LERNA Phase 1.2 — TOST Equivalence (α={args.alpha}, "
                 f"metric margin=±{args.margin_accuracy}, "
                 f"energy margin=±{args.margin_energy_pct}%)")
    lines.append("=" * 100)
    lines.append(f"{'baseline':<18}{'task':<7}{'n':>3}  "
                 f"{'Δmetric':>9}{'p_tost':>9}{'eqv?':>6}   "
                 f"{'logΔE':>8}{'p_tost':>9}{'eqv?':>6}")
    lines.append("-" * 100)
    for r in sorted(rows, key=lambda x: (x.task, x.baseline)):
        lines.append(
            f"{r.baseline:<18}{r.task:<7}{r.n_seeds:>3}  "
            f"{r.metric_delta:>+9.4f}{r.tost_p_metric:>9.4f}"
            f"{'  YES' if r.equivalent_metric else '   no':>6}   "
            f"{r.energy_log_ratio:>+8.4f}{r.tost_p_energy:>9.4f}"
            f"{'  YES' if r.equivalent_energy else '   no':>6}"
        )
    lines.append("-" * 100)
    lines.append("  Δmetric = LERNA − baseline. logΔE = log(baseline/LERNA energy).")
    lines.append("  YES = equivalent within margin at α=0.05.")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "equivalence_summary.txt").write_text(txt)
    log.info("  DONE")


if __name__ == "__main__":
    main()
