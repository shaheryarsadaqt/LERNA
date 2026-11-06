#!/usr/bin/env python3
"""Generate summary tables from results.json + matrix_validation.json only.

Usage:
    python scripts/summarize_matrix.py /path/to/experiment_dir

Outputs:
    - Console: formatted tables
    - experiment_dir/summary_tables.json: all tables as JSON
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_results_json(results_path: Path) -> Optional[Dict[str, Any]]:
    """Load a results.json file, handling potential trailing newlines."""
    try:
        with open(results_path) as f:
            content = f.read().strip()
            if not content:
                return None
            return json.loads(content)
    except Exception as e:
        print(f"Warning: Failed to load {results_path}: {e}", file=sys.stderr)
        return None


def collect_cell_data(exp_dir: Path) -> List[Dict[str, Any]]:
    """Collect all cell data from experiment directory."""
    cells = []
    for arm_dir in sorted(exp_dir.iterdir()):
        if not arm_dir.is_dir() or arm_dir.name.startswith("."):
            continue
        for cell_dir in sorted(arm_dir.iterdir()):
            if not cell_dir.is_dir():
                continue
            attempt_dir = cell_dir / "attempt-001"
            results_path = attempt_dir / "results.json"
            if not results_path.exists():
                continue
            data = load_results_json(results_path)
            if data is None:
                continue
            cells.append({
                "arm": arm_dir.name,
                "cell_id": cell_dir.name,
                "data": data,
            })
    return cells


def extract_cell_summary(data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract summary fields from a single cell's results.json."""
    eval_metrics = data.get("eval_metrics", {})
    instr = data.get("true_skip_instrumentation", {})
    return {
        "arm": data.get("run_config", {}).get("control", "unknown"),
        "target_skip_rate": data.get("run_config", {}).get("target_skip_rate", 0.0),
        "train_runtime_s": data.get("train_runtime_s", 0.0),
        "eval_f1": eval_metrics.get("eval_f1"),
        "eval_accuracy": eval_metrics.get("eval_accuracy"),
        "train_loss": data.get("train_loss"),
        "skipped_backward_steps": data.get("skipped_backward_steps", 0),
        "optimizer_step_attempts": instr.get("optimizer_step_attempts", 0),
        "grad_scaler_step_calls": instr.get("grad_scaler_step_calls", 0),
        "forward_calls": instr.get("forward_calls", 0),
        "backward_calls": instr.get("backward_calls", 0),
        "skipped_batches": instr.get("skipped_batches", 0),
        "run_invalidated": instr.get("run_invalidated", False),
        "last_policy_error_type": instr.get("last_policy_error_type"),
        # M2c fields
        "nonfinite_forward_loss_policy": instr.get("nonfinite_forward_loss_policy"),
        "forward_loss_observations": instr.get("forward_loss_observations", 0),
        "nonfinite_forward_loss_count": instr.get("nonfinite_forward_loss_count", 0),
        "first_nonfinite_forward_loss_step": instr.get("first_nonfinite_forward_loss_step"),
        "first_nonfinite_forward_loss_value": instr.get("first_nonfinite_forward_loss_value"),
        # AMP accounting
        "optimizer_steps_skipped_amp": instr.get("optimizer_steps_skipped_amp", 0),
        "scaler_scale_final": instr.get("scaler_scale_final"),
        "scaler_scale_min": instr.get("scaler_scale_min"),
        # Timing breakdown
        "train_loop_seconds": instr.get("train_loop_seconds", 0.0),
        "eval_seconds": instr.get("eval_seconds", 0.0),
        "setup_seconds": instr.get("setup_seconds", 0.0),
        "telemetry_overhead_seconds": instr.get("telemetry_overhead_seconds", 0.0),
         "mean_step_time_executed": instr.get("mean_step_time_executed"),
        "mean_step_time_skipped": instr.get("mean_step_time_skipped"),
        "dataloader_wait_seconds": instr.get("dataloader_wait_seconds", 0.0),
        # CUDA-event-based timing (new in tracker v2)
        "mean_step_ms_executed": instr.get("mean_step_ms_executed"),
        "mean_step_ms_skipped": instr.get("mean_step_ms_skipped"),
        "avg_dataloader_wait_ms": instr.get("avg_dataloader_wait_ms"),
    }


def build_timing_table(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build timing breakdown table."""
    rows = []
    for c in cells:
        s = c["summary"]
        total = s["train_runtime_s"]
        loop = s["train_loop_seconds"]
        eval_t = s["eval_seconds"]
        setup = s["setup_seconds"]
        telemetry = s["telemetry_overhead_seconds"]
        accounted = loop + eval_t + setup + telemetry
        rows.append({
            "arm": s["arm"],
            "skip_rate": s["target_skip_rate"],
            "total_runtime_s": round(total, 1),
            "train_loop_s": round(loop, 1) if loop else None,
            "eval_s": round(eval_t, 1) if eval_t else None,
            "setup_s": round(setup, 1) if setup else None,
            "telemetry_s": round(telemetry, 1) if telemetry else None,
            "accounted_s": round(accounted, 1),
            "unaccounted_s": round(total - accounted, 1),
            "mean_step_executed_ms": round(s["mean_step_time_executed"] * 1000, 1) if s["mean_step_time_executed"] else None,
            "mean_step_skipped_ms": round(s["mean_step_time_skipped"] * 1000, 1) if s["mean_step_time_skipped"] else None,
            "dataloader_wait_s": round(s["dataloader_wait_seconds"], 1),
            # CUDA-event-based (tracker v2)
            "cuda_mean_step_executed_ms": s["mean_step_ms_executed"],
            "cuda_mean_step_skipped_ms": s["mean_step_ms_skipped"],
            "cuda_avg_dataloader_wait_ms": s["avg_dataloader_wait_ms"],
        })
    return sorted(rows, key=lambda x: x["total_runtime_s"])


def build_metrics_table(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build metrics table sorted by arm, then skip_rate."""
    rows = []
    for c in cells:
        s = c["summary"]
        rows.append({
            "arm": s["arm"],
            "skip_rate": s["target_skip_rate"],
            "eval_f1": round(s["eval_f1"], 4) if s["eval_f1"] is not None else None,
            "eval_accuracy": round(s["eval_accuracy"], 4) if s["eval_accuracy"] is not None else None,
            "train_loss": round(s["train_loss"], 4) if s["train_loss"] is not None else None,
            "train_runtime_s": round(s["train_runtime_s"], 1),
            "skipped_backward": s["skipped_backward_steps"],
            "optimizer_steps": s["optimizer_step_attempts"],
        })
    return sorted(rows, key=lambda x: (x["arm"], x["skip_rate"]))


def build_skip_accounting_table(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build skip accounting table."""
    rows = []
    for c in cells:
        s = c["summary"]
        forward = s["forward_calls"]
        backward = s["backward_calls"]
        skipped = s["skipped_backward_steps"]
        batches_seen = forward
        skip_ratio_batch = skipped / max(batches_seen, 1)
        opt_ops = s["optimizer_step_attempts"] + skipped
        skip_ratio_opt = skipped / max(opt_ops, 1)
        rows.append({
            "arm": s["arm"],
            "skip_rate": s["target_skip_rate"],
            "forward_calls": forward,
            "backward_calls": backward,
            "skipped_backward": skipped,
            "skip_ratio_by_batch": round(skip_ratio_batch, 4),
            "skip_ratio_by_opt_opportunity": round(skip_ratio_opt, 4),
            "optimizer_step_attempts": s["optimizer_step_attempts"],
            "grad_scaler_step_calls": s["grad_scaler_step_calls"],
            "invariant_forward_eq_bwd_plus_skipped": (forward == backward + skipped),
        })
    return sorted(rows, key=lambda x: (x["arm"], x["skip_rate"]))


def build_m2c_table(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build M2c forward-loss numerics table."""
    rows = []
    for c in cells:
        s = c["summary"]
        rows.append({
            "arm": s["arm"],
            "skip_rate": s["target_skip_rate"],
            "policy": s["nonfinite_forward_loss_policy"],
            "forward_loss_observations": s["forward_loss_observations"],
            "nonfinite_count": s["nonfinite_forward_loss_count"],
            "first_nonfinite_step": s["first_nonfinite_forward_loss_step"],
            "first_nonfinite_value": s["first_nonfinite_forward_loss_value"],
            "run_invalidated": s["run_invalidated"],
        })
    return sorted(rows, key=lambda x: (x["arm"], x["skip_rate"]))


def build_amp_table(cells: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build AMP accounting table."""
    rows = []
    for c in cells:
        s = c["summary"]
        rows.append({
            "arm": s["arm"],
            "skip_rate": s["target_skip_rate"],
            "optimizer_step_attempts": s["optimizer_step_attempts"],
            "grad_scaler_step_calls": s["grad_scaler_step_calls"],
            "optimizer_steps_skipped_amp": s["optimizer_steps_skipped_amp"],
            "scaler_scale_final": s["scaler_scale_final"],
            "scaler_scale_min": s["scaler_scale_min"],
        })
    return sorted(rows, key=lambda x: (x["arm"], x["skip_rate"]))


def build_validation_table(exp_dir: Path) -> List[Dict[str, Any]]:
    """Build validation status table from matrix_validation.json."""
    val_path = exp_dir / "matrix_validation.json"
    if not val_path.exists():
        return []
    with open(val_path) as f:
        val = json.load(f)
    rows = []
    for run in val.get("runs", []):
        cell_id = run.get("cell_id", [])
        rows.append({
            "task": cell_id[0] if len(cell_id) > 0 else None,
            "seed": cell_id[1] if len(cell_id) > 1 else None,
            "skip_rate": cell_id[2] if len(cell_id) > 2 else None,
            "arm": cell_id[3] if len(cell_id) > 3 else None,
            "attempt": run.get("attempt"),
            "energy_valid": run.get("energy_valid"),
            "power_evidence_coverage": round(run.get("power_evidence_coverage") or 0.0, 4),
            "manifest_sha256": run.get("manifest_sha256", "")[:16] + "...",
            "results_sha256": run.get("results_sha256", "")[:16] + "...",
        })
    return rows


def print_table(headers: List[str], rows: List[Dict[str, Any]], key_order: List[str]):
    """Print a formatted table."""
    # Calculate column widths
    col_widths = {k: len(k) for k in key_order}
    for row in rows:
        for k in key_order:
            val = str(row.get(k, ""))
            col_widths[k] = max(col_widths[k], len(val))

    # Print header
    header_line = " | ".join(f"{k:<{col_widths[k]}}" for k in key_order)
    print(header_line)
    print("-" * len(header_line))

    # Print rows
    for row in rows:
        line = " | ".join(f"{str(row.get(k, '')):<{col_widths[k]}}" for k in key_order)
        print(line)


def extract_seed_map(cells_raw: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Build {arm: {seed: cell_data}} from collected raw cells.

    Uses identity_inputs.training_seed if present, falls back to 'seed' field.
    """
    seed_map: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for c in cells_raw:
        arm = c["arm"]
        data = c["data"]
        seed = data.get("identity_inputs", {}).get("training_seed", data.get("seed"))
        if seed is None:
            continue
        seed_map.setdefault(arm, {})[int(seed)] = data
    return seed_map


def paired_ttest(
    arm_a: str, arm_b: str, seed_map: Dict[str, Dict[int, Dict[str, Any]]]
) -> Dict[str, Any]:
    """Paired t-test on Δacc(arm_a − arm_b) across shared seeds.

    Returns mean, se, t_stat, df, t_crit, p_value_approx, significant.
    """
    seeds_a = set(seed_map.get(arm_a, {}).keys())
    seeds_b = set(seed_map.get(arm_b, {}).keys())
    shared = sorted(seeds_a & seeds_b)
    if len(shared) < 2:
        return {
            "n": len(shared),
            "deltas": [],
            "mean_delta": None,
            "se": None,
            "t_stat": None,
            "df": len(shared) - 1 if shared else 0,
            "t_crit": None,
            "significant": False,
            "error": "insufficient shared seeds",
        }

    deltas = []
    for s in shared:
        acc_a = (
            seed_map[arm_a][s]
            .get("eval_metrics", {})
            .get("eval_accuracy")
        )
        acc_b = (
            seed_map[arm_b][s]
            .get("eval_metrics", {})
            .get("eval_accuracy")
        )
        if acc_a is None or acc_b is None:
            continue
        deltas.append(float(acc_a) - float(acc_b))

    n = len(deltas)
    if n < 2:
        return {
            "n": n,
            "deltas": deltas,
            "mean_delta": None,
            "se": None,
            "t_stat": None,
            "df": n - 1,
            "t_crit": None,
            "significant": False,
            "error": "insufficient data after filtering",
        }

    mean_d = sum(deltas) / n
    var_d = sum(d**2 for d in deltas) / n - mean_d**2
    if var_d < 0:
        var_d = 0.0
    se = math.sqrt(var_d) / math.sqrt(n) if n > 1 else 0.0
    t_stat = mean_d / se if se > 0 else float("inf")

    # t-critical for two-sided alpha=0.05
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776,
        5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
        9: 2.262, 10: 2.228, 15: 2.131, 20: 2.086,
        30: 1.697,
    }
    t_crit = t_table.get(n - 1, 1.96)
    significant = abs(t_stat) > t_crit if se > 0 else True

    return {
        "n": n,
        "deltas": deltas,
        "mean_delta": mean_d,
        "se": se,
        "t_stat": t_stat,
        "df": n - 1,
        "t_crit": t_crit,
        "significant": significant,
    }


def tost_noninferiority(
    arm_a: str, arm_b: str, seed_map: Dict[str, Dict[int, Dict[str, Any]]],
    margin: float = 0.005, alpha: float = 0.05,
) -> Dict[str, Any]:
    """TOST non-inferiority test: H0: Δ <= -margin vs H1: Δ > -margin.

    Uses the paired t-test results to check if the lower 90% CI
    (one-sided) exceeds -margin.
    """
    pt = paired_ttest(arm_a, arm_b, seed_map)
    if pt["mean_delta"] is None or pt["se"] is None:
        return {"pass": False, "error": pt.get("error", "insufficient data")}

    n = pt["n"]
    mean_d = pt["mean_delta"]
    se = pt["se"]
    df = pt["df"]

    # One-sided 95% CI: mean_d - t_one_sided * se
    # t_one_sided for alpha=0.05 is the same as t_two_sided(0.10, df)
    t_one_sided_table = {
        1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132,
        5: 2.015, 6: 1.943, 7: 1.895, 8: 1.860,
        9: 1.833, 10: 1.812, 15: 1.753, 20: 1.725,
        30: 1.697,
    }
    t_one_sided = t_one_sided_table.get(df, 1.645)

    ci_lower = mean_d - t_one_sided * se if se < float("inf") else mean_d
    passes = ci_lower >= -margin

    return {
        "n": n,
        "mean_delta": mean_d,
        "se": se,
        "ci_lower": ci_lower,
        "margin": margin,
        "pass": passes,
    }


def build_paired_analysis(cells_raw: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Build paired t-test and TOST non-inferiority tables.

    Uses seed-256 field grouping if available; falls back to fingerprint-based
    arm grouping with seed extracted from identity_inputs.
    """
    seed_map = extract_seed_map(cells_raw)

    result: Dict[str, Any] = {"by_arm_seed": {}, "primary": {}, "secondary": {}, "tost": {}}

    for arm, seed_dict in seed_map.items():
        result["by_arm_seed"][arm] = {
            str(s): {
                "eval_accuracy": (
                    seed_dict[s].get("eval_metrics", {}).get("eval_accuracy")
                ),
                "eval_f1": seed_dict[s].get("eval_metrics", {}).get("eval_f1"),
                "train_loss": seed_dict[s].get("train_loss"),
            }
            for s in sorted(seed_dict.keys())
        }

    # Primary: paired t-test (ler_guided_stratified - exact_random)
    if "ler_guided_stratified" in seed_map and "exact_random" in seed_map:
        result["primary"] = paired_ttest(
            "ler_guided_stratified", "exact_random", seed_map
        )

    # Secondary: TOST non-inferiority (ler_guided_stratified - full_finetune)
    if "ler_guided_stratified" in seed_map and "full_finetune" in seed_map:
        result["tost"]["ler_vs_fullfinetune"] = tost_noninferiority(
            "ler_guided_stratified", "full_finetune", seed_map
        )

    # TOST: ler_guided_stratified vs exact_random
    if "ler_guided_stratified" in seed_map and "exact_random" in seed_map:
        result["tost"]["ler_vs_exactrandom"] = tost_noninferiority(
            "ler_guided_stratified", "exact_random", seed_map
        )

    # TOST: phase_strat_guarded vs exact_random
    if "phase_strat_guarded" in seed_map and "exact_random" in seed_map:
        result["tost"]["phase_vs_exactrandom"] = tost_noninferiority(
            "phase_strat_guarded", "exact_random", seed_map
        )

    return result


def main():
    parser = argparse.ArgumentParser(description="Generate summary tables from experiment results")
    parser.add_argument("experiment_dir", help="Path to experiment directory")
    parser.add_argument("--output", "-o", help="Output JSON file for tables")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    if not exp_dir.exists():
        print(f"Error: {exp_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    cells_raw = collect_cell_data(exp_dir)
    if not cells_raw:
        print("No cell data found", file=sys.stderr)
        sys.exit(1)

    # Extract summaries
    for c in cells_raw:
        c["summary"] = extract_cell_summary(c["data"])

    # Build all tables
    metrics_table = build_metrics_table(cells_raw)
    skip_table = build_skip_accounting_table(cells_raw)
    timing_table = build_timing_table(cells_raw)
    m2c_table = build_m2c_table(cells_raw)
    amp_table = build_amp_table(cells_raw)
    validation_table = build_validation_table(exp_dir)

    # Aggregate run counts: attempted / failed / completed / validated / energy-valid
    val_path = exp_dir / "matrix_validation.json"
    run_counts = {
        "attempted": len(cells_raw),
        "failed": 0,
        "completed": 0,
        "validated": 0,
        "energy_valid": 0,
    }
    if val_path.exists():
        with open(val_path) as f:
            val_data = json.load(f)
        val_runs = val_data.get("runs", [])
        run_counts["attempted"] = len(val_runs)
        run_counts["failed"] = sum(1 for r in val_runs if not r.get("completed", r.get("valid", False)))
        run_counts["completed"] = sum(1 for r in val_runs if r.get("completed", False))
        run_counts["validated"] = sum(1 for r in val_runs if r.get("run_validated", False) is False and r.get("valid", False))
        run_counts["energy_valid"] = sum(1 for r in val_runs if r.get("energy_valid", False))

    # Print to console
    print("\n=== METRICS TABLE ===")
    print_table(
        ["arm", "skip_rate", "eval_f1", "eval_accuracy", "train_loss", "train_runtime_s", "skipped_backward", "optimizer_steps"],
        metrics_table,
        ["arm", "skip_rate", "eval_f1", "eval_accuracy", "train_loss", "train_runtime_s", "skipped_backward", "optimizer_steps"]
    )

    print("\n=== SKIP ACCOUNTING TABLE ===")
    print_table(
        ["arm", "skip_rate", "forward_calls", "backward_calls", "skipped_backward", "skip_ratio_by_batch", "skip_ratio_by_opt_opportunity", "optimizer_step_attempts", "grad_scaler_step_calls", "invariant_forward_eq_bwd_plus_skipped"],
        skip_table,
        ["arm", "skip_rate", "forward_calls", "backward_calls", "skipped_backward", "skip_ratio_by_batch", "skip_ratio_by_opt_opportunity", "optimizer_step_attempts", "grad_scaler_step_calls", "invariant_forward_eq_bwd_plus_skipped"]
    )

    print("\n=== TIMING BREAKDOWN TABLE ===")
    print_table(
        ["arm", "skip_rate", "total_runtime_s", "train_loop_s", "eval_s", "setup_s", "telemetry_s", "accounted_s", "unaccounted_s", "mean_step_executed_ms", "mean_step_skipped_ms", "dataloader_wait_s"],
        timing_table,
        ["arm", "skip_rate", "total_runtime_s", "train_loop_s", "eval_s", "setup_s", "telemetry_s", "accounted_s", "unaccounted_s", "mean_step_executed_ms", "mean_step_skipped_ms", "dataloader_wait_s"]
    )

    print("\n=== M2c FORWARD-LOSS NUMERICS TABLE ===")
    print_table(
        ["arm", "skip_rate", "policy", "forward_loss_observations", "nonfinite_count", "first_nonfinite_step", "first_nonfinite_value", "run_invalidated"],
        m2c_table,
        ["arm", "skip_rate", "policy", "forward_loss_observations", "nonfinite_count", "first_nonfinite_step", "first_nonfinite_value", "run_invalidated"]
    )

    print("\n=== AMP ACCOUNTING TABLE ===")
    print_table(
        ["arm", "skip_rate", "optimizer_step_attempts", "grad_scaler_step_calls", "optimizer_steps_skipped_amp", "scaler_scale_final", "scaler_scale_min"],
        amp_table,
        ["arm", "skip_rate", "optimizer_step_attempts", "grad_scaler_step_calls", "optimizer_steps_skipped_amp", "scaler_scale_final", "scaler_scale_min"]
    )

    print("\n=== VALIDATION STATUS TABLE ===")
    print_table(
        ["task", "seed", "skip_rate", "arm", "attempt", "energy_valid", "power_evidence_coverage", "manifest_sha256", "results_sha256"],
        validation_table,
        ["task", "seed", "skip_rate", "arm", "attempt", "energy_valid", "power_evidence_coverage", "manifest_sha256", "results_sha256"]
    )

    # Paired analysis (paired t-test + TOST)
    paired_analysis = build_paired_analysis(cells_raw)

    # Print paired analysis
    print("\n=== PAIRED ANALYSIS ===")
    seed_map = paired_analysis.get("by_arm_seed", {})
    if seed_map:
        arms = sorted(seed_map.keys())
        seeds = sorted(
            set(s for arm in arms for s in seed_map[arm].keys()),
            key=lambda x: int(x),
        )
        print(f"\n  {'Arm':25s} | " + " | ".join(f"s{s}" for s in seeds))
        print("  " + "-" * (25 + 7 * len(seeds)))
        for arm in arms:
            vals = []
            for s in seeds:
                cell = seed_map[arm].get(s, {})
                acc = cell.get("eval_accuracy")
                vals.append(f"{acc:.6f}" if acc is not None else "N/A")
            print(f"  {arm:25s} | " + " | ".join(vals))

    pt = paired_analysis.get("primary", {})
    if pt and pt.get("mean_delta") is not None:
        print(f"\n  Primary: Δacc(lgs − exact_random), n={pt['n']}")
        print(f"    Mean Δ: {pt['mean_delta']:+.10f}")
        print(f"    SE: {pt['se']:.10f}" if pt.get('se') else "    SE: N/A")
        print(f"    t({pt['df']}): {pt['t_stat']:.3f}" if pt.get('t_stat') else "    t: N/A")
        print(f"    t_crit: {pt['t_crit']}")
        sig = "SIGNIFICANT" if pt.get("significant") else "NOT SIGNIFICANT"
        print(f"    Result: {sig} at α=0.05 (two-sided)")

    print("\n--- TOST Non-Inferiority ---")
    for label, tost_key in [
        ("ler vs full_finetune", "ler_vs_fullfinetune"),
        ("ler vs exact_random", "ler_vs_exactrandom"),
        ("phase_strat vs exact_random", "phase_vs_exactrandom"),
    ]:
        tr = paired_analysis.get("tost", {}).get(tost_key, {})
        if tr and tr.get("pass") is not None:
            print(f"  {label:25s}: mean Δ={tr['mean_delta']:+.6f}, CI_lower={tr['ci_lower']:+.6f}, "
                  f"margin={tr['margin']}, {'PASS' if tr['pass'] else 'FAIL'}")

    # Write JSON output
    output = {
        "metrics": metrics_table,
        "skip_accounting": skip_table,
        "timing": timing_table,
        "m2c_numerics": m2c_table,
        "amp_accounting": amp_table,
        "validation": validation_table,
        "run_counts": run_counts,
        "paired_analysis": paired_analysis,
    }

    if args.output:
        out_path = Path(args.output)
    else:
        out_path = exp_dir / "summary_tables.json"

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nWritten JSON tables to: {out_path}")

    # Print aggregate run counts
    print(f"\n=== RUN COUNTS ===")
    print(f"  Attempted:    {run_counts['attempted']}")
    print(f"  Failed:       {run_counts['failed']}")
    print(f"  Completed:    {run_counts['completed']}")
    print(f"  Validated:    {run_counts['validated']}")
    print(f"  Energy-valid: {run_counts['energy_valid']}")


if __name__ == "__main__":
    main()