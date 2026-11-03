#!/usr/bin/env python3
"""Standalone analysis for Gate 3 QNLI confirmatory run.

This script reads cached cell results (by fingerprint) from the
GATE3_OUTPUT_DIR and produces a paired t-test and TOST non-inferiority
analysis on Δacc(lgs − exact_random).

Unlike the full gate3_qnli_confirmatory.py runner, this script bypasses all
preflight checks (clean git, disk space, foreign-SHA) and only operates on
already-completed cells.  It is suitable for re-analyzing a frozen experiment
after the runner has completed.
"""

import os
import math
import json
import glob
from pathlib import Path

CONFIRMATORY_ARMS = [
    "full_finetune",
    "exact_random",
    "ler_guided_stratified",
    "phase_strat_guarded",
]
CONFIRMATORY_SEEDS = [11, 13, 17, 19, 23]

BASE_OUTPUT_DIR = os.environ.get(
    "GATE3_OUTPUT_DIR",
    "/new_raid/nanhangproj/experiments/gate3_qnli_confirmatory",
)


def find_completed_attempt(base_output_dir, arm, fingerprint):
    arm_dir = Path(base_output_dir) / arm / fingerprint
    if not arm_dir.exists():
        return None
    for attempt_dir in sorted(arm_dir.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt-"):
            continue
        rpath = attempt_dir / "results.json"
        ipath = attempt_dir / "instrumentation.json"
        if rpath.exists() and ipath.exists():
            return attempt_dir.name, rpath
    return None


def collect_cell_metrics(base_output_dir, arm, fingerprint):
    completed = find_completed_attempt(base_output_dir, arm, fingerprint)
    if completed is None:
        return None
    _, rpath = completed
    d = json.load(open(rpath))
    instr = d.get("true_skip_instrumentation", {})
    return {
        "acc": d["eval_metrics"]["eval_accuracy"],
        "f1": d["eval_metrics"].get("eval_f1", None),
        "mean_step_ms_executed": instr.get("mean_step_ms_executed"),
        "mean_step_ms_skipped": instr.get("mean_step_ms_skipped"),
        "train_loop_seconds": instr.get("train_loop_seconds"),
        "eval_seconds": instr.get("eval_seconds"),
        "avg_dataloader_wait_ms": instr.get("avg_dataloader_wait_ms"),
        "forward_calls": instr.get("forward_calls", d.get("forward_calls", 0)),
        "backward_calls": instr.get("backward_calls", d.get("backward_calls", 0)),
        "skipped_backward_steps": instr.get(
            "skipped_backward_steps", d.get("skipped_backward_steps", 0)
        ),
        "energy_kwh": d.get("energy_kwh"),
        "energy_valid": d.get("power_evidence", {}).get("energy_valid", False),
        "skip_ratio": d.get("skip_ratio"),
        "seed": d.get("seed"),
        "fingerprint": d.get("fingerprint"),
    }


def discover_cells(base_output_dir, arm):
    """Discover all valid cells for an arm by scanning manifests."""
    cells = {}
    arm_dir = Path(base_output_dir) / arm
    if not arm_dir.exists():
        return cells
    for fp_dir in sorted(arm_dir.iterdir()):
        if not fp_dir.is_dir() or len(fp_dir.name) != 16:
            continue
        manifest = glob.glob(str(fp_dir / "attempt-001" / "run_manifest.json"))
        if not manifest:
            continue
        with open(manifest[0]) as f:
            m = json.load(f)
        seed = m.get("identity_inputs", {}).get("training_seed")
        if seed is None:
            continue
        metrics = collect_cell_metrics(base_output_dir, arm, fp_dir.name)
        if metrics is None:
            continue
        cells[seed] = {"fingerprint": fp_dir.name, **metrics}
    return cells


def run_analysis(base_output_dir=BASE_OUTPUT_DIR):
    t_crit = 2.776  # t(4, 0.025) for n=5, two-sided alpha=0.05

    all_cells = {}
    for arm in CONFIRMATORY_ARMS:
        all_cells[arm] = discover_cells(base_output_dir, arm)

    print("=" * 70)
    print("GATE 3 ANALYSIS (standalone, fingerprint-filtered)")
    print("=" * 70)

    print("\n--- Cell Discovery ---")
    for arm in CONFIRMATORY_ARMS:
        seeds_found = sorted(all_cells[arm].keys())
        expected = set(CONFIRMATORY_SEEDS)
        missing = expected - set(seeds_found)
        extra = set(seeds_found) - expected
        print(f"  {arm}: seeds={seeds_found} missing={missing or 'none'} extra={extra or 'none'}")

    print("\n--- Accuracy by Arm x Seed ---")
    print(f"  {'Arm':25s} | " + " | ".join(f"s{s:2d}" for s in CONFIRMATORY_SEEDS))
    print("  " + "-" * 75)
    for arm in CONFIRMATORY_ARMS:
        vals = []
        for s in CONFIRMATORY_SEEDS:
            cell = all_cells[arm].get(s)
            if cell:
                vals.append(f"{cell['acc']:.6f}")
            else:
                vals.append("N/A     ")
        print(f"  {arm:25s} | " + " | ".join(vals))

    print("\n--- Validator Rule: identical accuracy across >=2 seeds ---")
    seed_identity_violation = False
    for arm in CONFIRMATORY_ARMS:
        accs = [(s, all_cells[arm][s]["acc"]) for s in CONFIRMATORY_SEEDS if s in all_cells[arm]]
        if len(accs) >= 2:
            vals = [a for _, a in accs]
            unique = set(vals)
            if len(unique) < len(vals):
                seed_identity_violation = True
                dup_seeds = [s for s, a in accs if vals.count(a) > 1]
                print(f"  VIOLATION: {arm} has identical accuracies (duplicates at seeds: {dup_seeds})")
                for s, a in accs:
                    cell = all_cells[arm][s]
                    print(f"    seed={s}: acc={a:.10f} eval_loss={cell.get('eval_loss', 'N/A')}")
            else:
                print(f"  {arm}: OK ({len(accs)} seeds, all unique)")

    print(f"\n  matrix_invalidated = {seed_identity_violation}")

    print("\n--- Primary: Δacc(lgs − exact_random) paired t-test ---")
    completed_seeds = [s for s in CONFIRMATORY_SEEDS if s in all_cells["ler_guided_stratified"] and s in all_cells["exact_random"]]
    deltas = []
    for s in completed_seeds:
        lgs = all_cells["ler_guided_stratified"][s]
        rnd = all_cells["exact_random"][s]
        d = lgs["acc"] - rnd["acc"]
        deltas.append(d)
        print(f"  seed={s}: {lgs['acc']:.10f} - {rnd['acc']:.10f} = {d:+.10f}")

    if len(deltas) >= 2:
        n = len(deltas)
        mean_d = sum(deltas) / n
        se = math.sqrt(sum(d**2 for d in deltas) / n - mean_d**2) / math.sqrt(n) if n > 1 else 0
        t_stat = mean_d / se if se > 0 else float("inf")
        sig = "SIGNIFICANT" if abs(t_stat) > t_crit else "NOT SIGNIFICANT"
        print(f"\n  n={n} | Mean Δ: {mean_d:+.10f} | SE: {se:.10f} | t({n-1}): {t_stat:.3f} | crit: {t_crit}")
        print(f"  Result: {sig} at α=0.05 (two-sided)")

    print("\n--- Secondary: Δacc(lgs − full_finetune) TOST non-inferiority ---")
    deltas_ni = []
    for s in completed_seeds:
        if s in all_cells["ler_guided_stratified"] and s in all_cells["full_finetune"]:
            lgs = all_cells["ler_guided_stratified"][s]
            ftf = all_cells["full_finetune"][s]
            d = lgs["acc"] - ftf["acc"]
            deltas_ni.append(d)
            print(f"  seed={s}: {lgs['acc']:.10f} - {ftf['acc']:.10f} = {d:+.10f}")

    if deltas_ni:
        mean_ni = sum(deltas_ni) / len(deltas_ni)
        ci_lower = mean_ni - t_crit * se if se > 0 else mean_ni
        print(f"\n  Mean Δ: {mean_ni:+.10f} | SE: {se:.10f} | 95% CI lower: {ci_lower:+.10f}")
        print(f"  Non-inferiority (margin 0.5pt / 0.005): {'PASS' if ci_lower >= -0.005 else 'FAIL'}")

    print("\n--- Overhead: mean_step_ms_executed ---")
    for arm in CONFIRMATORY_ARMS:
        cells = [all_cells[arm][s] for s in completed_seeds if s in all_cells[arm]]
        ms_vals = [c["mean_step_ms_executed"] for c in cells if c.get("mean_step_ms_executed")]
        if ms_vals:
            mean_ms = sum(ms_vals) / len(ms_vals)
            print(f"  {arm:25s}: {mean_ms:.1f} ms/step (n={len(ms_vals)})")

    return {
        "seed_identity_violation": seed_identity_violation,
        "completed_seeds": completed_seeds,
        "deltas_primary": deltas,
        "deltas_secondary": deltas_ni,
        "mean_delta_primary": mean_d if len(deltas) >= 2 else None,
        "t_statistic": t_stat if len(deltas) >= 2 else None,
        "t_critical": t_crit,
        "primary_significant": abs(t_stat) > t_crit if len(deltas) >= 2 and se > 0 else False,
    }


if __name__ == "__main__":
    results = run_analysis()
    summary_path = Path(BASE_OUTPUT_DIR) / "gate3_analysis.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Analysis saved: {summary_path}")
