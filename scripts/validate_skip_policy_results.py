#!/usr/bin/env python3
"""Validate skip-policy diagnostics in a run's results.json."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_results(path: Path, tolerance: float = 0.005) -> bool:
    if not path.exists():
        raise FileNotFoundError(f"results.json not found: {path}")
    with path.open() as f:
        data = json.load(f)

    diag = data.get("policy_diagnostics", {})
    if not diag:
        raise ValueError("policy_diagnostics missing from results.json")

    target_skip_rate = float(data.get("run_config", {}).get("target_skip_rate", diag.get("target_skip_rate", 0.0)))
    target_veto_rate = float(diag.get("target_veto_rate", diag.get("run_config", {}).get("target_veto_rate", 0.0)))
    realized = float(diag.get("realized_skip_rate", 0.0))
    veto_rate_vs_candidates = float(diag.get("veto_rate_vs_candidates", 0.0))
    deferred_pool_now = int(diag.get("deferred_pool_now", 0))
    invariant_ok = bool(diag.get("invariant_quota_decomposition_ok", False))

    lower = target_skip_rate - tolerance
    upper = target_skip_rate + tolerance
    checks = [
        ("realized_skip_rate", lower <= realized <= upper,
         f"realized_skip_rate={realized:.4f}, expected≈{target_skip_rate:.4f}±{tolerance:.4f}"),
        ("veto_rate_vs_candidates", veto_rate_vs_candidates < target_veto_rate,
         f"veto_rate_vs_candidates={veto_rate_vs_candidates:.4f}, target_veto_rate={target_veto_rate:.4f}"),
        ("deferred_pool_now", deferred_pool_now == 0,
         f"deferred_pool_now={deferred_pool_now}, expected 0"),
        ("invariant_quota_decomposition_ok", invariant_ok,
         f"invariant_quota_decomposition_ok={invariant_ok}"),
    ]

    all_ok = True
    print(f"Validating {path}")
    print(f"  target_skip_rate = {target_skip_rate:.4f}")
    print(f"  target_veto_rate = {target_veto_rate:.4f}")
    print(f"  realized_skip_rate = {realized:.4f}")
    print(f"  veto_rate_vs_candidates = {veto_rate_vs_candidates:.4f}")
    print(f"  deferred_pool_now = {deferred_pool_now}")
    print(f"  invariant_quota_decomposition_ok = {invariant_ok}")

    for name, passed, msg in checks:
        print(f"  {name}: {'PASS' if passed else 'FAIL'} ({msg})")
        if not passed:
            all_ok = False

    return all_ok


def main():
    parser = argparse.ArgumentParser(description="Validate skip policy diagnostics in results.json.")
    parser.add_argument("results_json", type=Path, help="Path to a results.json file")
    parser.add_argument("--tolerance", type=float, default=0.005,
                        help="Allowed deviation for realized skip rate")
    args = parser.parse_args()

    ok = validate_results(args.results_json, tolerance=args.tolerance)
    if not ok:
        raise SystemExit(1)
    print("All validation checks passed.")


if __name__ == "__main__":
    main()
