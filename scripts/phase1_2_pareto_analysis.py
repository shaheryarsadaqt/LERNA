#!/usr/bin/env python3
"""
=============================================================================
LERNA Phase 1.2 — Step 4/5: Pareto-Frontier Analysis
=============================================================================

The Phase 1.2 verdict is "LERNA wins on the accuracy–energy tradeoff".
This script makes that claim rigorous by computing, per task and pooled
across tasks:

    1. The Pareto front on (mean_metric ↑, mean_energy_kwh ↓)
    2. Per-baseline dominance count: how many baselines does each strictly
       dominate, how many dominate it, how many are mutually non-dominated.
    3. Hypervolume indicator (Zitzler 1999) of each baseline's seed cloud,
       wrt a reference point (worst observed metric, worst observed energy).
    4. Per-seed dominance: across all seeds, how often does LERNA dominate
       each baseline pairwise?

Visualisations:
    pareto_<task>.pdf / .png   (one per task, with confidence ellipses)
    pareto_grand.pdf / .png    (all tasks faceted)
    hypervolume_bar.pdf / .png

Outputs:
    pareto_results.csv  pareto_results.json  pareto_summary.txt

Usage:
    python scripts/phase1_2_pareto_analysis.py \
        --input experiments/phase1_2_analysis/aggregated_long.csv \
        --output-dir experiments/phase1_2_analysis

Author: LERNA Research Team
Phase:  1.2 — Analysis pipeline step 4/5
=============================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("phase1_2_pareto")


def dominates(a_acc: float, a_e: float, b_acc: float, b_e: float) -> bool:
    """a dominates b iff a is no worse in both AND strictly better in at least one.
    accuracy: higher is better. energy: lower is better."""
    return (a_acc >= b_acc and a_e <= b_e) and (a_acc > b_acc or a_e < b_e)


def pareto_front(points: dict[str, tuple[float, float]]) -> list[str]:
    """Return labels whose points are non-dominated."""
    labels = list(points.keys())
    front = []
    for lab in labels:
        ax, ay = points[lab]
        if not any(dominates(points[o][0], points[o][1], ax, ay)
                   for o in labels if o != lab):
            front.append(lab)
    return front


def hypervolume_2d(points: list[tuple[float, float]],
                   ref_acc: float, ref_energy: float) -> float:
    """Hypervolume of a 2-objective set wrt reference (acc_min, energy_max).
    Standard sweep algorithm."""
    if not points:
        return 0.0
    pts = [(a, e) for (a, e) in points if a > ref_acc and e < ref_energy]
    if not pts:
        return 0.0
    pts.sort(key=lambda p: -p[0])
    hv = 0.0
    prev_e = ref_energy
    for a, e in pts:
        if e < prev_e:
            hv += (a - ref_acc) * (prev_e - e)
            prev_e = e
    return hv


# =============================================================================
@dataclass
class PerTaskPareto:
    task: str
    on_front: list[str]
    dominated_by: dict[str, list[str]]
    dominates_count: dict[str, int]
    dominated_count: dict[str, int]
    hypervolume: dict[str, float]


def per_task_analysis(df: pd.DataFrame) -> list[PerTaskPareto]:
    tasks = sorted(df.task.unique())
    baselines = sorted(df.baseline.unique())
    out: list[PerTaskPareto] = []

    for task in tasks:
        sub = df[df.task == task]
        means = {b: (sub[sub.baseline == b].primary_metric.mean(),
                     sub[sub.baseline == b].energy_kwh.mean())
                 for b in baselines if (sub.baseline == b).any()}
        front = pareto_front(means)

        dominates_count: dict[str, int] = {}
        dominated_count: dict[str, int] = {}
        dominated_by: dict[str, list[str]] = {}
        for b in means:
            ax, ay = means[b]
            ds = [o for o in means if o != b and dominates(ax, ay, *means[o])]
            db = [o for o in means if o != b and dominates(*means[o], ax, ay)]
            dominates_count[b] = len(ds)
            dominated_count[b] = len(db)
            dominated_by[b] = db

        ref_acc = sub.primary_metric.min() - 1e-6
        ref_e = sub.energy_kwh.max() + 1e-12
        hv: dict[str, float] = {}
        for b in means:
            pts = list(zip(sub[sub.baseline == b].primary_metric.to_numpy(),
                           sub[sub.baseline == b].energy_kwh.to_numpy()))
            hv[b] = hypervolume_2d(pts, ref_acc, ref_e)

        out.append(PerTaskPareto(
            task=task, on_front=front,
            dominated_by=dominated_by,
            dominates_count=dominates_count,
            dominated_count=dominated_count,
            hypervolume=hv,
        ))
    return out


def pairwise_seed_dominance(df: pd.DataFrame, ref: str) -> pd.DataFrame:
    """For each (other_baseline, task), fraction of (seed_lerna × seed_other)
    pairs where LERNA strictly dominates."""
    rows = []
    tasks = sorted(df.task.unique())
    others = sorted([b for b in df.baseline.unique() if b != ref])
    for b in others:
        for t in tasks:
            le = df[(df.baseline == ref) & (df.task == t)]
            ot = df[(df.baseline == b) & (df.task == t)]
            if le.empty or ot.empty:
                continue
            le_pts = list(zip(le.primary_metric, le.energy_kwh))
            ot_pts = list(zip(ot.primary_metric, ot.energy_kwh))
            total = 0
            wins = 0
            ties = 0
            for (la, lE), (oa, oE) in product(le_pts, ot_pts):
                total += 1
                if dominates(la, lE, oa, oE):
                    wins += 1
                elif not dominates(oa, oE, la, lE):
                    ties += 1
            rows.append({
                "baseline": b, "task": t,
                "n_pairs": total,
                "lerna_dominates_pct": 100 * wins / total if total else 0,
                "mutually_nondominated_pct": 100 * ties / total if total else 0,
            })
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def _plot_one_task(ax, sub: pd.DataFrame, front_labels: list[str], task: str):
    palette = sns.color_palette("Set2", n_colors=sub.baseline.nunique())
    color_map = {b: palette[i] for i, b in enumerate(sorted(sub.baseline.unique()))}

    for b in sorted(sub.baseline.unique()):
        s = sub[sub.baseline == b]
        ax.scatter(s.energy_kwh, s.primary_metric, color=color_map[b],
                   alpha=0.35, s=22, edgecolors="none")

    for b in sorted(sub.baseline.unique()):
        s = sub[sub.baseline == b]
        mx, my = s.energy_kwh.mean(), s.primary_metric.mean()
        ex, ey = s.energy_kwh.std(ddof=1), s.primary_metric.std(ddof=1)
        marker = "*" if b in front_labels else "o"
        size = 240 if b in front_labels else 95
        edge = "black" if b in front_labels else "white"
        ax.errorbar(mx, my, xerr=ex, yerr=ey, fmt="none", color=color_map[b],
                    alpha=0.6, linewidth=1.0, capsize=2.5)
        ax.scatter(mx, my, color=color_map[b], marker=marker, s=size,
                   edgecolor=edge, linewidth=1.4, zorder=10, label=b)

    front_pts = []
    for b in front_labels:
        s = sub[sub.baseline == b]
        front_pts.append((s.energy_kwh.mean(), s.primary_metric.mean()))
    front_pts.sort()
    if len(front_pts) > 1:
        fx, fy = zip(*front_pts)
        ax.plot(fx, fy, "k--", alpha=0.6, linewidth=1.2, zorder=5,
                label="Pareto front")

    ax.set_xlabel("Energy (kWh) — lower is better")
    ax.set_ylabel("Primary metric — higher is better")
    ax.set_title(f"{task}")
    ax.grid(True, alpha=0.25)


def make_pareto_plots(df: pd.DataFrame, per_task: list[PerTaskPareto],
                      out_dir: Path):
    tasks = sorted(df.task.unique())
    fronts = {p.task: p.on_front for p in per_task}

    for task in tasks:
        sub = df[df.task == task]
        fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
        _plot_one_task(ax, sub, fronts.get(task, []), task)
        ax.legend(fontsize=8, loc="best", frameon=True)
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"pareto_{task}.{ext}", dpi=180)
        plt.close(fig)

    ncol = 4
    nrow = int(np.ceil(len(tasks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.5 * ncol, 3.6 * nrow),
                             constrained_layout=True)
    axes = np.atleast_1d(axes).flatten()
    for i, task in enumerate(tasks):
        sub = df[df.task == task]
        _plot_one_task(axes[i], sub, fronts.get(task, []), task)
        if i == 0:
            axes[i].legend(fontsize=6, loc="best", frameon=True)
    for j in range(len(tasks), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Phase 1.2 — Accuracy vs Energy Pareto Fronts", fontsize=13)
    for ext in ("png", "pdf"):
        fig.savefig(out_dir / f"pareto_grand.{ext}", dpi=180)
    plt.close(fig)

    hv_rows = []
    for p in per_task:
        for b, v in p.hypervolume.items():
            hv_rows.append({"task": p.task, "baseline": b, "hypervolume": v})
    hv_df = pd.DataFrame(hv_rows)
    if not hv_df.empty:
        fig, ax = plt.subplots(figsize=(11, 4.6), constrained_layout=True)
        sns.barplot(data=hv_df, x="task", y="hypervolume", hue="baseline", ax=ax)
        ax.set_title("Hypervolume per baseline per task (higher = better)")
        ax.set_ylabel("Hypervolume (metric × kWh)")
        ax.legend(fontsize=7, ncol=3, loc="best")
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"hypervolume_bar.{ext}", dpi=180)
        plt.close(fig)


# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="experiments/phase1_2_analysis/aggregated_long.csv")
    ap.add_argument("--output-dir", default="experiments/phase1_2_analysis")
    ap.add_argument("--reference-baseline", default="lerna")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_task = per_task_analysis(df)

    pt_rows = []
    for p in per_task:
        for b in p.dominates_count:
            pt_rows.append({
                "task": p.task,
                "baseline": b,
                "on_pareto_front": b in p.on_front,
                "dominates_count": p.dominates_count[b],
                "dominated_count": p.dominated_count[b],
                "dominated_by": ";".join(p.dominated_by[b]),
                "hypervolume": p.hypervolume[b],
            })
    pt_df = pd.DataFrame(pt_rows)
    pt_df.to_csv(out_dir / "pareto_results.csv", index=False)

    pw_df = pairwise_seed_dominance(df, args.reference_baseline)
    pw_df.to_csv(out_dir / "pareto_pairwise_seed_dominance.csv", index=False)

    payload = {
        "phase": "1.2-pareto",
        "reference_baseline": args.reference_baseline,
        "per_task": [asdict(p) for p in per_task],
    }
    (out_dir / "pareto_results.json").write_text(json.dumps(payload, indent=2))

    log.info(f"  rendering plots to {out_dir}/pareto_*.pdf / .png")
    make_pareto_plots(df, per_task, out_dir)

    lines = []
    lines.append("=" * 90)
    lines.append("  LERNA Phase 1.2 — Pareto Analysis (accuracy ↑ vs energy ↓)")
    lines.append("=" * 90)
    for p in per_task:
        lines.append(f"\n  TASK: {p.task}")
        lines.append(f"    Pareto front: {p.on_front}")
        for b in sorted(p.dominates_count):
            star = "  ★" if b in p.on_front else "   "
            lines.append(f"     {star} {b:<18s} dominates={p.dominates_count[b]:>2}  "
                         f"dominated_by={p.dominated_count[b]:>2}  HV={p.hypervolume[b]:.6e}")
    lines.append("\n  ★ = on Pareto front for that task.")
    if not pw_df.empty:
        lines.append("\n  PAIRWISE SEED DOMINANCE vs " + args.reference_baseline)
        for t in sorted(pw_df.task.unique()):
            lines.append(f"   {t}:")
            for _, r in pw_df[pw_df.task == t].iterrows():
                lines.append(f"     {r.baseline:<18s}  "
                             f"LERNA dominates {r.lerna_dominates_pct:5.1f}% of seed pairs "
                             f"(n={int(r.n_pairs)})")
    txt = "\n".join(lines)
    print(txt)
    (out_dir / "pareto_summary.txt").write_text(txt)


if __name__ == "__main__":
    main()
