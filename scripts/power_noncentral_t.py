"""Power analysis for Stage 2 paired t-test.

Computes the minimum detectable effect (MDE) for a paired t-test given:
  - n = 24 seeds (df = 23)
  - alpha = 0.05 (two-sided)
  - power = 0.80
  - paired SD = 0.287 (from V3 confirmatory, retrospective)

Uses the noncentral t-distribution to compute MDE exactly.

Usage:
    python scripts/power_noncentral_t.py
    python scripts/power_noncentral_t.py --n 24 --alpha 0.05 --power 0.80 --sd 0.287
"""
import argparse
import math

try:
    from scipy.stats import nct, t as t_dist
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def cohen_d_to_delta_d(df: int, alpha: float, power: float) -> float:
    """Solve for the noncentrality parameter that gives the target power.

    For a two-sided test: power = P(|T| > t_{1-alpha/2} | ncp = delta).
    We solve delta such that the probability of |T| exceeding the critical
    value equals power. This is done by bisection on the noncentral t CDF.
    """
    if not HAS_SCIPY:
        # Fallback: use normal approximation
        from statistics import NormalDist
        z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
        z_beta = NormalDist().inv_cdf(power)
        # Noncentrality parameter for paired t (approximate)
        ncp = z_alpha + z_beta
        return ncp
    else:
        t_crit = t_dist.ppf(1 - alpha / 2, df)

        def power_at_ncp(ncp):
            # Two-sided: P(T > t_crit | ncp) + P(T < -t_crit | ncp)
            return 1.0 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

        # Bisect for ncp
        lo, hi = 0.0, 100.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if power_at_ncp(mid) < power:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2


def main():
    parser = argparse.ArgumentParser(description="Power analysis for Stage 2 paired t-test")
    parser.add_argument("--n", type=int, default=24, help="Number of paired seeds")
    parser.add_argument("--alpha", type=float, default=0.05, help="Two-sided alpha")
    parser.add_argument("--power", type=float, default=0.80, help="Target power")
    parser.add_argument("--sd", type=float, default=0.287, help="Paired SD (from V3)")
    args = parser.parse_args()

    df = args.n - 1
    ncp = cohen_d_to_delta_d(df, args.alpha, args.power)

    # MDE = ncp * sd / sqrt(n)
    mde = ncp * args.sd / math.sqrt(args.n)

    print(f"Power Analysis for Stage 2 Paired t-test")
    print(f"  n = {args.n} seeds (df = {df})")
    print(f"  alpha = {args.alpha} (two-sided)")
    print(f"  target power = {args.power}")
    print(f"  paired SD = {args.sd}")
    print(f"  noncentrality parameter (ncp) = {ncp:.4f}")
    print(f"  Cohen's dz = {ncp:.4f} (since dz = t/np.sqrt(n))")
    print(f"  MDE = {mde:.4f} pt = {mde*100:.2f} pt")
    print(f"  MDE = {mde*100:.2f} percentage points")

    if not HAS_SCIPY:
        print(f"  (scipy not available; using normal approximation)")

    # Also compute for V3 parameters (n=5) for comparison
    df3 = 4
    ncp3 = cohen_d_to_delta_d(df3, args.alpha, args.power)
    mde3 = ncp3 * args.sd / math.sqrt(5)
    print(f"\n  For comparison (n=5, df=4):")
    print(f"    ncp = {ncp3:.4f}")
    print(f"    MDE = {mde3:.4f} pt = {mde3*100:.2f} pt")


if __name__ == "__main__":
    main()
