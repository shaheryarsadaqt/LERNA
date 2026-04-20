"""Paired significance tests for Phase 1.2 baselines vs LERNA."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np
from scipy import stats


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
