#!/usr/bin/env python3
"""Bootstrap 95% CIs for Phase 1.2 per-baseline metrics."""
import numpy as np

def bootstrap_ci(scores, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0):
    scores = np.asarray(scores, dtype=float)
    if len(scores) < 2:
        mean_val = float(scores.mean() if len(scores) else 0.0)
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    n = len(scores)
    boot = np.array([scores[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    mean = float(scores.mean())
    ci_low, ci_high = np.percentile(boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return mean, float(ci_low), float(ci_high)
