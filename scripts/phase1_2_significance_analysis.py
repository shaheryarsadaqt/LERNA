#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2 — Step 2/5: Publication-Grade Significance Analysis
=============================================================================

For every (baseline, task) pair vs (lerna, task) computes, on per-seed paired
data:

    • Paired t-test                              (parametric)
    • Wilcoxon signed-rank (Pratt zero handling) (non-parametric)
    • Cohen's dz (paired)                        (effect size, raw)
    • Hedges' g_av (paired, small-sample corr.)  (effect size, unbiased)
    • Bias-corrected & accelerated (BCa) bootstrap 95% CI for the paired delta
    • Bootstrap CI for the ratio of energy use (baseline / lerna)
    • Post-hoc statistical power (paired)

Then performs family-wise multiple-comparison correction across all
(baseline, task) tests:

    • Holm–Bonferroni (controls FWER)
    • Benjamini–Hochberg FDR (controls FDR)
    • Storey q-values (optional, via statsmodels)

Outputs:
    significance_results.csv       — one row per (baseline, task)
    significance_results.json      — same, with metadata
    significance_summary.txt       — human-readable table

Usage:
    python scripts/phase1_2_significance_analysis.py \
        --input experiments/phase1_2_analysis/aggregated_long.csv \
        --output-dir experiments/phase1_2_analysis \
        --reference-baseline lerna \
        --n-bootstrap 10000

Author: LERNA Research Team
Phase:  1.2 — Analysis pipeline step 2/5
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import stats

try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phase1_2_sig")


# =============================================================================
# Statistical primitives
# =============================================================================

def cohens_dz_paired(diff: np.ndarray) -> float:
    """Cohen's d for paired samples (signed, uses sd of differences)."""
    if len(diff) < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else 0.0


def hedges_gav_paired(x: np.ndarray, y: np.ndarray) -> float:
    """Hedges' g_av for paired data with small-sample correction (Lakens 2013).

    g_av = (mean(x) - mean(y)) / pooled_sd · J(df)
    where pooled_sd = sqrt((s_x^2 + s_y^2) / 2) and J is Hedges' correction.
    """
    n = len(x)
    if n < 2 or n != len(y):
        return float("nan")
    sx, sy = x.std(ddof=1), y.std(ddof=1)
    pooled = math.sqrt((sx ** 2 + sy ** 2) / 2.0)
    if pooled == 0:
        return 0.0
    d_av = (x.mean() - y.mean()) / pooled
    df = n - 1
    # Hedges' small-sample correction J(df) ≈ 1 − 3 / (4·df − 1)
    j = 1 - (3 / (4 * df - 1)) if df > 0 else 1.0
    return float(d_av * j)


def bca_bootstrap_ci(values: np.ndarray,
                     stat_fn=np.mean,
                     n_boot: int = 10_000,
                     alpha: float = 0.05,
                     rng_seed: int = 0) -> tuple[float, float, float]:
    """Bias-corrected & accelerated bootstrap CI (Efron 1987)."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n < 2:
        v = stat_fn(values) if n == 1 else float("nan")
        return float(v), float(v), float(v)

    rng = np.random.default_rng(rng_seed)
    theta_hat = stat_fn(values)

    # bootstrap replicates
    boot = np.array([stat_fn(values[rng.integers(0, n, n)]) for _ in range(n_boot)])

    # bias-correction z0
    prop_lt = np.mean(boot < theta_hat)
    # clip to avoid ±inf from ppf(0) / ppf(1)
    prop_lt = np.clip(prop_lt, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_lt)

    # acceleration via jackknife
    jack = np.array([stat_fn(np.delete(values, i)) for i in range(n)])
    jack_mean = jack.mean()
    num = np.sum((jack_mean - jack) ** 3)
    den = 6.0 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = num / den if den != 0 else 0.0

    z_alpha_lo = stats.norm.ppf(alpha / 2)
    z_alpha_hi = stats.norm.ppf(1 - alpha / 2)

    def adjust(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    lo = np.percentile(boot, 100 * adjust(z_alpha_lo))
    hi = np.percentile(boot, 100 * adjust(z_alpha_hi))
    return float(theta_hat), float(lo), float(hi)


def post_hoc_power_paired(diff: np.ndarray, alpha: float = 0.05) -> float:
    """Post-hoc power for paired t-test at observed effect size & n."""
    n = len(diff)
    if n < 2:
        return float("nan")
    sd = diff.std(ddof=1)
    if sd == 0:
        return 1.0
    dz = abs(diff.mean()) / sd
    ncp = dz * math.sqrt(n)
    df = n - 1
    t_crit = stats.t.ppf(1 - alpha / 2, df)
    # two-sided power via noncentral t
    power = (1 - stats.nct.cdf(t_crit, df, ncp)
             + stats.nct.cdf(-t_crit, df, ncp))
    return float(np.clip(power, 0.0, 1.0))


# =============================================================================
# Per-pair test
# =============================================================================

@dataclass
class PairedTestRow:
    baseline: str
    task: str
    n_seeds: int
    lerna_mean: float
    baseline_mean: float
    delta_mean: float
    delta_ci_low: float
    delta_ci_high: float
    paired_t_stat: float
    paired_t_pvalue: float
    wilcoxon_stat: float
    wilcoxon_pvalue: float
    cohens_dz: float
    hedges_gav: float
    post_hoc_power: float
    energy_lerna_mean: float
    energy_baseline_mean: float
    energy_ratio_mean: float
    energy_ratio_ci_low: float
    energy_ratio_ci_high: float
    # filled in after FWER / FDR correction
    p_holm: float = float("nan")
    p_bh_fdr: float = float("nan")
    significant_holm: bool = False
    significant_bh: bool = False


def paired_test(lerna_metric: np.ndarray, base_metric: np.ndarray,
                lerna_energy: np.ndarray, base_energy: np.ndarray,
                baseline: str, task: str,
                n_boot: int, rng_seed: int) -> PairedTestRow:
    n = len(lerna_metric)
    diff = lerna_metric - base_metric

    # Paired t
    if n >= 2:
        t_stat, t_p = stats.ttest_rel(lerna_metric, base_metric)
    else:
        t_stat, t_p = float("nan"), float("nan")

    # Wilcoxon (handle all-zeros)
    if n >= 2 and not np.allclose(diff, 0):
        try:
            w_stat, w_p = stats.wilcoxon(lerna_metric, base_metric,
                                         zero_method="pratt", alternative="two-sided")
        except ValueError:
            w_stat, w_p = float("nan"), float("nan")
    else:
        w_stat, w_p = float("nan"), 1.0

    # Effect sizes
    dz = cohens_dz_paired(diff)
    g_av = hedges_gav_paired(lerna_metric, base_metric)

    # BCa CI for paired delta
    _, delta_lo, delta_hi = bca_bootstrap_ci(diff, np.mean,
                                             n_boot=n_boot, rng_seed=rng_seed)

    # Energy ratio (per-seed log-ratio for stability, then exp)
    safe_le = np.maximum(lerna_energy, 1e-12)
    safe_be = np.maximum(base_energy, 1e-12)
    log_ratio = np.log(safe_be / safe_le)
    lr_mean, lr_lo, lr_hi = bca_bootstrap_ci(log_ratio, np.mean,
                                             n_boot=n_boot, rng_seed=rng_seed + 1)

    return PairedTestRow(
        baseline=baseline,
        task=task,
        n_seeds=int(n),
        lerna_mean=float(np.mean(lerna_metric)),
        baseline_mean=float(np.mean(base_metric)),
        delta_mean=float(np.mean(diff)),
        delta_ci_low=delta_lo,
        delta_ci_high=delta_hi,
        paired_t_stat=float(t_stat),
        paired_t_pvalue=float(t_p),
        wilcoxon_stat=float(w_stat),
        wilcoxon_pvalue=float(w_p),
        cohens_dz=float(dz),
        hedges_gav=float(g_av),
        post_hoc_power=post_hoc_power_paired(diff),
        energy_lerna_mean=float(np.mean(lerna_energy)),
        energy_baseline_mean=float(np.mean(base_energy)),
        energy_ratio_mean=float(math.exp(lr_mean)),
        energy_ratio_ci_low=float(math.exp(lr_lo)),
        energy_ratio_ci_high=float(math.exp(lr_hi)),
    )


# =============================================================================
# Driver
# =============================================================================

def _align_seeds(df: pd.DataFrame, ref: str, baseline: str, task: str
                 ) -> tuple[np.ndarray, ...] | None:
    sub_ref = df[(df.baseline == ref) & (df.task == task)].set_index("seed")
    sub_b = df[(df.baseline == baseline) & (df.task == task)].set_index("seed")
    common = sorted(set(sub_ref.index) & set(sub_b.index))
    if len(common) < 2:
        return None
    sub_ref = sub_ref.loc[common]
    sub_b = sub_b.loc[common]
    return (sub_ref.primary_metric.to_numpy(),
            sub_b.primary_metric.to_numpy(),
            sub_ref.energy_kwh.to_numpy(),
            sub_b.energy_kwh.to_numpy())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="experiments/phase1_2_analysis/aggregated_long.csv")
    ap.add_argument("--output-dir", default="experiments/phase1_2_analysis")
    ap.add_argument("--reference-baseline", default="lerna",
                    help="Baseline that all others are compared against")
    ap.add_argument("--n-bootstrap", type=int, default=10_000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--rng-seed", type=int, default=20260101)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    if args.reference_baseline not in df.baseline.unique():
        log.error(f"  reference baseline '{args.reference_baseline}' not in data")
        return

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baselines = sorted([b for b in df.baseline.unique() if b != args.reference_baseline])
    tasks = sorted(df.task.unique())

    log.info(f"  reference: {args.reference_baseline}")
    log.info(f"  baselines: {baselines}")
    log.info(f"  tasks:     {tasks}")
    log.info(f"  bootstrap replicates: {args.n_bootstrap:,}")

    rows: list[PairedTestRow] = []
    for b in baselines:
        for t in tasks:
            aligned = _align_seeds(df, args.reference_baseline, b, t)
            if aligned is None:
                log.warning(f"  skipping ({b}, {t}) — < 2 paired seeds")
                continue
            le_m, ba_m, le_e, ba_e = aligned
            rows.append(paired_test(le_m, ba_m, le_e, ba_e,
                                    b, t, args.n_bootstrap, args.rng_seed))

    if not rows:
        log.error("  no paired comparisons produced — aborting")
        return

    # ---- Multiple-comparison correction across the whole family ----
    pvals = np.array([r.paired_t_pvalue for r in rows])
    if HAS_STATSMODELS:
        rej_holm, p_holm, _, _ = multipletests(pvals, alpha=args.alpha, method="holm")
        rej_bh, p_bh, _, _ = multipletests(pvals, alpha=args.alpha, method="fdr_bh")
    else:
        # Manual Holm + BH if statsmodels unavailable
        m = len(pvals)
        order = np.argsort(pvals)
        ranks = np.empty(m, dtype=int)
        ranks[order] = np.arange(1, m + 1)
        p_holm = np.minimum.accumulate(np.sort(pvals * (m - np.arange(m))))[ranks - 1]
        p_holm = np.clip(p_holm, 0, 1)
        p_bh = np.minimum.accumulate(np.sort(pvals * m / np.arange(1, m + 1))[::-1])[::-1][ranks - 1]
        p_bh = np.clip(p_bh, 0, 1)
        rej_holm = p_holm < args.alpha
        rej_bh = p_bh < args.alpha

    for i, r in enumerate(rows):
        r.p_holm = float(p_holm[i])
        r.p_bh_fdr = float(p_bh[i])
        r.significant_holm = bool(rej_holm[i])
        r.significant_bh = bool(rej_bh[i])

    res_df = pd.DataFrame([asdict(r) for r in rows])
    csv_path = out_dir / "significance_results.csv"
    res_df.to_csv(csv_path, index=False)
    log.info(f"  wrote: {csv_path}")

    payload = {
        "phase": "1.2-significance",
        "reference_baseline": args.reference_baseline,
        "alpha": args.alpha,
        "n_bootstrap": args.n_bootstrap,
        "n_comparisons": len(rows),
        "results": [asdict(r) for r in rows],
    }
    json_path = out_dir / "significance_results.json"
    json_path.write_text(json.dumps(payload, indent=2))
    log.info(f"  wrote: {json_path}")

    # ---- Human-readable summary ----
    lines = []
    lines.append("=" * 110)
    lines.append("  LERNA Phase 1.2 — Paired Significance Analysis (vs LERNA)")
    lines.append("=" * 110)
    lines.append(f"{'baseline':<18}{'task':<7}{'n':>3}  {'Δmean':>8}  "
                 f"{'95% BCa CI':>22}  {'dz':>6}  {'g_av':>6}  "
                 f"{'p_t':>8}  {'p_holm':>8}  {'p_BH':>8}  {'pow':>5}  sig")
    lines.append("-" * 110)
    for r in sorted(rows, key=lambda x: (x.task, x.baseline)):
        sig = ("***" if r.significant_holm
               else "**" if r.significant_bh
               else "")
        lines.append(
            f"{r.baseline:<18}{r.task:<7}{r.n_seeds:>3}  "
            f"{r.delta_mean:>+8.4f}  "
            f"[{r.delta_ci_low:>+7.4f},{r.delta_ci_high:>+7.4f}]  "
            f"{r.cohens_dz:>+6.2f}  {r.hedges_gav:>+6.2f}  "
            f"{r.paired_t_pvalue:>8.4f}  {r.p_holm:>8.4f}  {r.p_bh_fdr:>8.4f}  "
            f"{r.post_hoc_power:>5.2f}  {sig}"
        )
    lines.append("-" * 110)
    lines.append("  Δmean = LERNA − baseline. Positive → LERNA wins.")
    lines.append("  *** = significant after Holm-Bonferroni (FWER-controlled)")
    lines.append("  **  = significant after BH-FDR only")
    txt_path = out_dir / "significance_summary.txt"
    txt = "\n".join(lines)
    txt_path.write_text(txt)
    print(txt)
    log.info(f"  wrote: {txt_path}")


if __name__ == "__main__":
    main()
