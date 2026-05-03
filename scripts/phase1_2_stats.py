"""Paired significance tests for Phase 1.2 baselines vs LERNA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture


@dataclass
class PairedTestResult:
    baseline: str
    task: str
    lerna_mean: float
    baseline_mean: float
    delta: float
    t_stat: float
    p_value: float
    ci_low: float
    ci_high: float
    significant: bool  # p < 0.05

    def __str__(self) -> str:
        flag = "***" if self.significant else "   "
        return (
            f"{flag} {self.baseline:<18} {self.task:<6} "
            f"Δ={self.delta:+.4f}  "
            f"95%CI=[{self.ci_low:+.4f},{self.ci_high:+.4f}]  "
            f"p={self.p_value:.4f}"
        )


def paired_t_test(
    lerna_scores: Sequence[float],
    baseline_scores: Sequence[float],
    baseline_name: str,
    task: str,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> PairedTestResult:
    """Paired t-test + bootstrap CI on the per-seed difference."""
    lerna = np.asarray(lerna_scores, dtype=float)
    base = np.asarray(baseline_scores, dtype=float)
    if lerna.shape != base.shape:
        raise ValueError("Seed arrays must be same length and order")

    diff = lerna - base
    t, p = stats.ttest_rel(lerna, base)

    rng = np.random.default_rng(0)
    n = len(diff)
    boot_means = np.array([
        diff[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)
    ])
    ci_low, ci_high = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])

    return PairedTestResult(
        baseline=baseline_name,
        task=task,
        lerna_mean=float(lerna.mean()),
        baseline_mean=float(base.mean()),
        delta=float(diff.mean()),
        t_stat=float(t),
        p_value=float(p),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        significant=bool(p < alpha),
    )


def run_all_paired_tests(results_by_baseline: dict[str, dict[str, list[float]]]):
    """results_by_baseline['grad_norm']['mrpc'] = [seed42_score, seed43_score, ...]"""
    lerna_scores = results_by_baseline["lerna"]
    out: list[PairedTestResult] = []
    for baseline, task_scores in results_by_baseline.items():
        if baseline == "lerna":
            continue
        for task, scores in task_scores.items():
            if task not in lerna_scores:
                continue
            out.append(
                paired_t_test(lerna_scores[task], scores, baseline, task)
            )
    return out


def report_bimodal_waste(task_wastes: list[float], task_name: str) -> str:
    """Report waste with GMM-based bimodality detection.

    Mean ± std is misleading for a bimodal distribution.
    This function fits 1- and 2-component Gaussian Mixture Models
    and uses BIC to decide whether the distribution is bimodal.
    """
    x = np.array(task_wastes).reshape(-1, 1)
    if len(x) < 4:
        return f"{task_name}: n<4, cannot fit mixture"

    gmm1 = GaussianMixture(1, random_state=0).fit(x)
    gmm2 = GaussianMixture(2, random_state=0).fit(x)
    if gmm2.bic(x) < gmm1.bic(x) - 4:
        means = gmm2.means_.flatten()
        weights = gmm2.weights_
        return (
            f"{task_name}: BIMODAL — "
            f"{weights[0]*100:.0f}% at {means[0]:.2%}, "
            f"{weights[1]*100:.0f}% at {means[1]:.2%}"
        )
    return f"{task_name}: {x.mean():.2%} ± {x.std():.2%} (unimodal)"


def summarize_waste_by_task(results_by_baseline: dict[str, dict[str, list[float]]]):
    """Print bimodal-aware waste summary for each task and baseline."""
    baselines = list(results_by_baseline.keys())
    all_tasks = set()
    for task_scores in results_by_baseline.values():
        all_tasks.update(task_scores.keys())

    for task in sorted(all_tasks):
        print(f"\n=== {task} ===")
        for baseline in baselines:
            if task not in results_by_baseline[baseline]:
                continue
            wastes = results_by_baseline[baseline][task]
            if not wastes:
                continue
            report = report_bimodal_waste(wastes, baseline)
            print(f"  {report}")


@dataclass
class WasteReport:
    raw_mean: float
    raw_std: float
    calibrated_mean: float
    calibrated_std: float
    n_total: int
    n_hit_floor: int
    pct_hit_floor: float

    def __str__(self) -> str:
        floor_note = f" ({self.pct_hit_floor:.0%} hit floor, excluded)" if self.n_hit_floor > 0 else ""
        return (
            f"raw={self.raw_mean:.1%}±{self.raw_std:.1%}  "
            f"calibrated={self.calibrated_mean:.1%}±{self.calibrated_std:.1%}{floor_note}"
        )


def waste_report(
    wastes: list[float],
    hit_floor_flags: list[bool] | None = None,
) -> WasteReport:
    """Two-metric waste reporting: raw mean and calibrated mean (excl. floor hits).

    Args:
        wastes: List of waste ratio values (0-1) per seed.
        hit_floor_flags: Optional list of bool flags for detector_hit_floor.
                        If None, uses all values for raw and calibrated.
    """
    wastes = np.asarray(wastes, dtype=float)
    n_total = len(wastes)

    raw_mean = float(wastes.mean())
    raw_std = float(wastes.std(ddof=1)) if n_total > 1 else 0.0

    if hit_floor_flags is None:
        return WasteReport(
            raw_mean=raw_mean, raw_std=raw_std,
            calibrated_mean=raw_mean, calibrated_std=raw_std,
            n_total=n_total, n_hit_floor=0, pct_hit_floor=0.0,
        )

    floor_flags = np.asarray(hit_floor_flags, dtype=bool)
    n_hit = int(floor_flags.sum())
    pct_hit = float(n_hit / n_total) if n_total > 0 else 0.0

    if n_hit == n_total or n_total - n_hit < 2:
        calibrated_mean = float(wastes[~floor_flags].mean()) if n_total - n_hit > 0 else raw_mean
        calibrated_std = 0.0
    else:
        calib_wastes = wastes[~floor_flags]
        calibrated_mean = float(calib_wastes.mean())
        calibrated_std = float(calib_wastes.std(ddof=1))

    return WasteReport(
        raw_mean=raw_mean, raw_std=raw_std,
        calibrated_mean=calibrated_mean, calibrated_std=calibrated_std,
        n_total=n_total, n_hit_floor=n_hit, pct_hit_floor=pct_hit,
    )


def report_task_waste(task_name: str, wastes: list[float], hit_floor_flags: list[bool] | None = None) -> str:
    """Single-task waste report with raw + calibrated + bimodal framing."""
    report = waste_report(wastes, hit_floor_flags)
    bimodal = report_bimodal_waste(wastes, task_name)
    return f"{bimodal}\n  raw={report.raw_mean:.1%}±{report.raw_std:.1%} | calibrated={report.calibrated_mean:.1%}±{report.calibrated_std:.1%} ({report.n_total - report.n_hit_floor}/{report.n_total} seeds)"
