#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2 — Master Orchestrator
=============================================================================

Single-command driver for the complete Phase 1.2 analysis pipeline.

Stages (each optional via --skip-*):
    [0] EXPERIMENTS   run_phase1_2_simple_baselines.py        (heavy GPU work)
    [1] AGGREGATE     phase1_2_aggregate_results.py
    [2] SIGNIFICANCE  phase1_2_significance_analysis.py
    [3] EQUIVALENCE   phase1_2_equivalence_test.py
    [4] PARETO        phase1_2_pareto_analysis.py
    [5] REPORT        phase1_2_report_generator.py

Typical workflows:

    # Just analyse pre-computed results.json files (no GPU):
    python scripts/phase1_2_orchestrator.py --skip-experiments

    # Full pipeline including experiment runs (full prod):
    python scripts/phase1_2_orchestrator.py --mode production --wandb

    # Re-run only the analysis stages after editing scripts:
    python scripts/phase1_2_orchestrator.py --skip-experiments --skip-aggregate

Outputs everything under `experiments/phase1_2_analysis/` and
`experiments/phase1_2_analysis/report/`.

Author: LERNA Research Team
Phase:  1.2 — Master orchestrator
=============================================================================
"""
from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phase1_2_orchestrator")

ROOT = Path(__file__).resolve().parent.parent  # repo root
SCRIPTS = ROOT / "scripts"
PY = sys.executable


def _run(cmd: list[str], stage: str):
    log.info(f"  ───────────────────────  STAGE: {stage}  ───────────────────────")
    log.info(f"  $ {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, cwd=ROOT)
    dt = time.time() - t0
    if res.returncode != 0:
        log.error(f"  STAGE FAILED ({stage}) after {dt:.1f}s — exit {res.returncode}")
        sys.exit(res.returncode)
    log.info(f"  stage OK in {dt:.1f}s")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Experiment stage args (forwarded)
    ap.add_argument("--mode", choices=["smoke", "quick", "full", "production"],
                    default="quick")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--unlimited", action="store_true")
    ap.add_argument("--target-skip-rate", type=float, default=0.33)

    # Pipeline structure args
    ap.add_argument("--phase1-1-dir", default="experiments/phase1_1")
    ap.add_argument("--phase1-2-dir", default="experiments/phase1_2_baselines")
    ap.add_argument("--analysis-dir", default="experiments/phase1_2_analysis")
    ap.add_argument("--reference-baseline", default="lerna")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--n-bootstrap", type=int, default=10000)
    ap.add_argument("--margin-metric", type=float, default=0.005)
    ap.add_argument("--margin-energy-pct", type=float, default=5.0)

    # Skip switches
    ap.add_argument("--skip-experiments", action="store_true")
    ap.add_argument("--skip-aggregate", action="store_true")
    ap.add_argument("--skip-significance", action="store_true")
    ap.add_argument("--skip-equivalence", action="store_true")
    ap.add_argument("--skip-pareto", action="store_true")
    ap.add_argument("--skip-report", action="store_true")
    args = ap.parse_args()

    overall_t0 = time.time()

    # ── [0] Experiments ──────────────────────────────────────────────
    if not args.skip_experiments:
        cmd = [PY, str(SCRIPTS / "run_phase1_2_simple_baselines.py"),
               "--mode", args.mode,
               "--target-skip-rate", str(args.target_skip_rate)]
        if args.wandb:
            cmd.append("--wandb")
        if args.unlimited:
            cmd.append("--unlimited")
        _run(cmd, "EXPERIMENTS")
    else:
        log.info("  --skip-experiments set; expecting pre-existing results.json files")

    # ── [1] Aggregate ────────────────────────────────────────────────
    if not args.skip_aggregate:
        _run([PY, str(SCRIPTS / "phase1_2_aggregate_results.py"),
              "--phase1-1-dir", args.phase1_1_dir,
              "--phase1-2-dir", args.phase1_2_dir,
              "--output-dir", args.analysis_dir],
             "AGGREGATE")

    # ── [2] Significance ─────────────────────────────────────────────
    if not args.skip_significance:
        _run([PY, str(SCRIPTS / "phase1_2_significance_analysis.py"),
              "--input", f"{args.analysis_dir}/aggregated_long.csv",
              "--output-dir", args.analysis_dir,
              "--reference-baseline", args.reference_baseline,
              "--alpha", str(args.alpha),
              "--n-bootstrap", str(args.n_bootstrap)],
             "SIGNIFICANCE")

    # ── [3] Equivalence ──────────────────────────────────────────────
    if not args.skip_equivalence:
        _run([PY, str(SCRIPTS / "phase1_2_equivalence_test.py"),
              "--input", f"{args.analysis_dir}/aggregated_long.csv",
              "--output-dir", args.analysis_dir,
              "--reference-baseline", args.reference_baseline,
              "--margin-accuracy", str(args.margin_metric),
              "--margin-energy-pct", str(args.margin_energy_pct),
              "--alpha", str(args.alpha)],
             "EQUIVALENCE")

    # ── [4] Pareto ───────────────────────────────────────────────────
    if not args.skip_pareto:
        _run([PY, str(SCRIPTS / "phase1_2_pareto_analysis.py"),
              "--input", f"{args.analysis_dir}/aggregated_long.csv",
              "--output-dir", args.analysis_dir,
              "--reference-baseline", args.reference_baseline],
             "PARETO")

    # ── [5] Report ───────────────────────────────────────────────────
    if not args.skip_report:
        _run([PY, str(SCRIPTS / "phase1_2_report_generator.py"),
              "--analysis-dir", args.analysis_dir,
              "--output-dir", f"{args.analysis_dir}/report",
              "--reference-baseline", args.reference_baseline,
              "--alpha", str(args.alpha),
              "--n-bootstrap", str(args.n_bootstrap),
              "--margin-metric", str(args.margin_metric),
              "--margin-energy-pct", str(args.margin_energy_pct)],
             "REPORT")

    log.info("=" * 70)
    log.info(f"  PHASE 1.2 PIPELINE COMPLETE in {time.time() - overall_t0:.1f}s")
    log.info(f"  Open: {args.analysis_dir}/report/phase1_2_report.md")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
