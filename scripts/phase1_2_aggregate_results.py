#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2 — Step 1/5: Results Aggregation (Ettin-150m)
=============================================================================

Walks `experiments/phase1_2_ettin150m_2026/` (Phase 1.2 Ettin-150m runs),
parses each `results.json`, and emits a single canonical long-format DataFrame:

    columns: baseline, task, seed, primary_metric, energy_kwh,
             train_runtime_s, train_steps, skip_ratio, steps_skipped,
             power_avg_watts, learning_rate, source_path,
             backward_call_fraction, within_run_backward_call_reduction,
             backward_call_reduction_vs_control

Also emits:
    - aggregated_wide.csv (one row per (task, seed), columns = baselines × metrics)
    - aggregation_manifest.json (file counts, missing-cell map, integrity hash)
    - per_baseline_summary.csv (mean/std/n_seeds per (baseline, task))

Usage:
    python scripts/phase1_2_aggregate_results.py \
        --input-dir experiments/phase1_2_ettin150m_2026 \
        --output-dir experiments/phase1_2_analysis

Author: LERNA Research Team
Phase:  1.2 — Analysis pipeline step 1/5
=============================================================================
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phase1_2_aggregate")

PRIMARY_METRIC_KEYS = (
    "eval_accuracy",
    "eval_matthews_correlation",
    "eval_pearson",
    "eval_pearsonr",
    "eval_spearmanr",
    "eval_f1",
)

EXPECTED_MODEL = "jhu-clsp/ettin-encoder-150m"
EXPECTED_REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"


# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunRecord:
    baseline: str
    task: str
    seed: int
    primary_metric: float
    energy_kwh: float
    energy_valid: bool
    train_runtime_s: float
    train_steps: int
    skip_ratio: float
    steps_skipped: int
    power_avg_watts: float
    learning_rate: float
    primary_metric_name: str
    source_path: str
    forward_calls: int
    backward_calls: int
    skipped_backward_steps: int
    backward_call_fraction: float
    within_run_backward_call_reduction: float
    backward_call_reduction_vs_control: float


def _extract_primary_metric(eval_metrics: dict[str, Any]) -> tuple[float, str]:
    """Return (value, key_used). Handles GLUE's heterogenous metric keys."""
    if not eval_metrics:
        return float("nan"), "missing"
    for k in PRIMARY_METRIC_KEYS:
        v = eval_metrics.get(k)
        if v is not None and isinstance(v, (int, float)) and not np.isnan(v):
            return float(v), k
    for k, v in eval_metrics.items():
        if (k.startswith("eval_")
                and isinstance(v, (int, float))
                and not k.endswith(("loss", "runtime", "_per_second", "_steps_per_second"))):
            return float(v), k
    return float("nan"), "missing"


def _coerce_float(value) -> float | None:
    """results.json uses json.dump(default=str), so numpy scalars / NaN can
    arrive as strings. Return a float, or None if not numeric."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_phase1_2_run(path: Path) -> RunRecord | None:
    """Phase 1.2 (baseline) results.json schema."""
    try:
        data = json.loads(path.read_text())
    except Exception as e:
        log.warning(f"  failed to read {path}: {e}")
        return None

    if data.get("model") != EXPECTED_MODEL:
        log.warning(f"  reject {path}: wrong model {data.get('model')}")
        return None
    if data.get("model_revision") != EXPECTED_REVISION:
        log.warning(f"  reject {path}: wrong revision {data.get('model_revision')}")
        return None
    if data.get("phase") != "1.2":
        log.warning(f"  reject {path}: wrong phase {data.get('phase')}")
        return None

    primary = _coerce_float(data.get("primary_metric"))
    key = data.get("primary_metric_name") or "missing"
    if primary is None or np.isnan(primary):
        eval_metrics = data.get("eval_metrics", {}) or {}
        primary, key = _extract_primary_metric(eval_metrics)
    if primary is None or np.isnan(primary):
        return None

    bstats = data.get("baseline_stats", {}) or {}
    forward_calls = _coerce_int(data.get("forward_calls")) or 0
    backward_calls = _coerce_int(data.get("backward_calls")) or 0
    skipped_backward_steps = _coerce_int(data.get("skipped_backward_steps")) or 0
    backward_call_fraction = backward_calls / max(forward_calls, 1)
    within_run_backward_call_reduction = 1.0 - backward_call_fraction

    energy_raw = data.get("energy_kwh")
    energy_valid = bool(data.get("energy_valid", False))
    energy_kwh = _coerce_float(energy_raw)
    if not energy_valid:
        energy_kwh = float("nan")

    return RunRecord(
        baseline=str(data.get("baseline", "unknown")),
        task=str(data.get("task", "unknown")),
        seed=int(data.get("seed", -1)),
        primary_metric=float(primary),
        energy_kwh=energy_kwh,
        energy_valid=energy_valid,
        train_runtime_s=float(data.get("train_runtime_s", 0.0) or 0.0),
        train_steps=int(data.get("train_steps", 0) or 0),
        skip_ratio=float(bstats.get("skip_ratio", 0.0) or 0.0),
        steps_skipped=skipped_backward_steps,
        power_avg_watts=float(data.get("power_avg_watts", 0.0) or 0.0),
        learning_rate=float(data.get("learning_rate", 0.0) or 0.0),
        primary_metric_name=key,
        source_path=str(path),
        forward_calls=forward_calls,
        backward_calls=backward_calls,
        skipped_backward_steps=skipped_backward_steps,
        backward_call_fraction=backward_call_fraction,
        within_run_backward_call_reduction=within_run_backward_call_reduction,
        backward_call_reduction_vs_control=0.0,
    )


def _walk(root: Path, parser) -> list[RunRecord]:
    out: list[RunRecord] = []
    if not root.exists():
        log.warning(f"  directory not found: {root}")
        return out
    for p in root.rglob("results.json"):
        if p.name == "phase1_2_summary.json":
            continue
        rec = parser(p)
        if rec is not None:
            out.append(rec)
    return out


def _digest_records(records: list[RunRecord]) -> str:
    payload = sorted(
        f"{r.baseline}|{r.task}|{r.seed}|{r.primary_metric:.6f}|{r.energy_kwh:.6e}"
        for r in records
    )
    return hashlib.sha256("\n".join(payload).encode()).hexdigest()[:16]


def _compute_reduction_vs_control(df: pd.DataFrame) -> pd.DataFrame:
    """Compute backward_call_reduction_vs_control against full_finetune control."""
    control = df[df["baseline"] == "full_finetune"][["task", "seed", "backward_calls"]].rename(
        columns={"backward_calls": "control_backward_calls"}
    )
    df = df.merge(control, on=["task", "seed"], how="left")
    df["backward_call_reduction_vs_control"] = 1.0 - df["backward_calls"] / df["control_backward_calls"]
    return df


# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input-dir", default="experiments/phase1_2_ettin150m_2026",
                    help="Directory containing Phase 1.2 Ettin-150m results.json files")
    ap.add_argument("--output-dir", default="experiments/phase1_2_analysis",
                    help="Where to write aggregated CSV / manifest / summary")
    ap.add_argument("--strict", action="store_true",
                    help="Fail if any baseline×task cell has fewer seeds than --min-seeds")
    ap.add_argument("--min-seeds", type=int, default=3)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"  scanning Phase 1.2 (Ettin-150m) :: {args.input_dir}")
    records = _walk(Path(args.input_dir), _parse_phase1_2_run)
    log.info(f"     -> {len(records)} accepted runs")

    if not records:
        log.error("  no records parsed — aborting")
        sys.exit(2)

    df = pd.DataFrame([asdict(r) for r in records])
    df = _compute_reduction_vs_control(df)
    df = df.sort_values(["baseline", "task", "seed"]).reset_index(drop=True)

    long_path = out_dir / "aggregated_long.csv"
    df.to_csv(long_path, index=False)
    log.info(f"  wrote long-format: {long_path}  shape={df.shape}")

    grp = df.groupby(["baseline", "task"])
    summary = grp.agg(
        n_seeds=("seed", "count"),
        mean_metric=("primary_metric", "mean"),
        std_metric=("primary_metric", "std"),
        sem_metric=("primary_metric", lambda x: x.std(ddof=1) / np.sqrt(max(1, x.count()))),
        median_metric=("primary_metric", "median"),
        mean_energy_kwh=("energy_kwh", "mean"),
        std_energy_kwh=("energy_kwh", "std"),
        mean_runtime_s=("train_runtime_s", "mean"),
        mean_skip_ratio=("skip_ratio", "mean"),
        mean_backward_call_reduction_vs_control=("backward_call_reduction_vs_control", "mean"),
        primary_metric_name=("primary_metric_name", "first"),
    ).reset_index()
    summary_path = out_dir / "per_baseline_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info(f"  wrote summary:    {summary_path}")

    wide_metric = df.pivot_table(index=["task", "seed"], columns="baseline",
                                 values="primary_metric", aggfunc="first")
    wide_metric.columns = [f"metric__{c}" for c in wide_metric.columns]
    wide_energy = df.pivot_table(index=["task", "seed"], columns="baseline",
                                 values="energy_kwh", aggfunc="first")
    wide_energy.columns = [f"energy__{c}" for c in wide_energy.columns]
    wide = pd.concat([wide_metric, wide_energy], axis=1).reset_index()
    wide_path = out_dir / "aggregated_wide.csv"
    wide.to_csv(wide_path, index=False)
    log.info(f"  wrote wide-format:{wide_path}  shape={wide.shape}")

    coverage = grp.size().unstack("task", fill_value=0)
    coverage_path = out_dir / "coverage_matrix.csv"
    coverage.to_csv(coverage_path)
    log.info(f"  wrote coverage:   {coverage_path}")

    missing_cells = []
    for baseline in df["baseline"].unique():
        for task in df["task"].unique():
            n = ((df["baseline"] == baseline) & (df["task"] == task)).sum()
            if n < args.min_seeds:
                missing_cells.append({"baseline": baseline, "task": task, "n_seeds": int(n)})

    manifest = {
        "phase": "1.2-ettin-aggregate",
        "input_dir": str(args.input_dir),
        "output_dir": str(args.output_dir),
        "expected_model": EXPECTED_MODEL,
        "expected_revision": EXPECTED_REVISION,
        "n_runs": len(records),
        "baselines": sorted(df["baseline"].unique().tolist()),
        "tasks": sorted(df["task"].unique().tolist()),
        "seeds": sorted(df["seed"].unique().tolist()),
        "min_seeds_threshold": args.min_seeds,
        "under_sampled_cells": missing_cells,
        "integrity_digest_sha256_16": _digest_records(records),
    }
    manifest_path = out_dir / "aggregation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    log.info(f"  wrote manifest:   {manifest_path}")

    print("\n  === COVERAGE MATRIX (n_seeds per baseline × task) ===")
    print(coverage.to_string())

    if missing_cells:
        log.warning(f"  {len(missing_cells)} cells under-sampled (n < {args.min_seeds}):")
        for c in missing_cells[:10]:
            log.warning(f"      {c['baseline']:<18s} {c['task']:<8s} n={c['n_seeds']}")
        if args.strict:
            log.error("  --strict was set, exiting with code 1")
            sys.exit(1)

    log.info("  DONE")


if __name__ == "__main__":
    main()
