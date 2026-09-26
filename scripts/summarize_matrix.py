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
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        ["task", "seed", "skip_rate", "arm", "attempt", "energy_valid", "manifest_sha256", "results_sha256"],
        validation_table,
        ["task", "seed", "skip_rate", "arm", "attempt", "energy_valid", "manifest_sha256", "results_sha256"]
    )

    # Write JSON output
    output = {
        "metrics": metrics_table,
        "skip_accounting": skip_table,
        "timing": timing_table,
        "m2c_numerics": m2c_table,
        "amp_accounting": amp_table,
        "validation": validation_table,
        "run_counts": run_counts,
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