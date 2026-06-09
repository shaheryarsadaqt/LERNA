"""
make_phase1_1_publication_figures.py
=====================================

ICLR / NeurIPS-quality plotting script for LERNA Phase 1.1 diagnostic baseline
(SST-2, seed 42, run name: sst2_s42_lr2e-05).

Consumes a full W&B history CSV and produces high-end publication figures
(PNG @ 600 dpi, PDF + SVG vector). Output folder: phase1_1_publication_figures/

ADVANCED PLOTS INCLUDED (in addition to the core Phase 1.1 figures):
  - mplot_density  -> KDE of GSNR pre-best vs post-best regime  (fig6)
  - mplot_roc      -> Diagnostic ROC: can GSNR / waste predict post-saturation training? (fig7)
  - mplot_cuts     -> Metrics binned by training-step quantile   (fig8)
  - mplot_splits   -> Split distributions (pre/post best step)   (fig8, fig6)
  - mplot_full     -> Multi-metric standardized heatmap          (fig9)
  - mplot_importance -> |Spearman rho| of every signal vs waste  (fig10)

HOW TO USE (VS Code / terminal):
  1. Place this script next to your CSV (or edit CSV_PATH below).
  2. `pip install pandas numpy matplotlib seaborn scipy`
  3. `python make_phase1_1_publication_figures.py`
  4. Figures land in ./phase1_1_publication_figures/

Dependencies: pandas, numpy, matplotlib, seaborn, scipy (for KDE / Spearman).
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats

# =====================================================================
# USER-CONFIGURABLE
# =====================================================================
CSV_PATH: str = "phase_1.1_sst2_s42_lr2e-05_history.csv"
OUTPUT_DIR: str = "phase1_1_publication_figures"
SAVE_SVG: bool = True
DPI_RASTER: int = 600                      # NeurIPS-grade raster
RUN_LABEL: str = "SST-2  ·  seed 42  ·  lr 2e-5"
SHOW_POST_BEST_SHADING: bool = True        # Light tint on post-best-step region


# =====================================================================
# Publication style  (ICLR / NeurIPS aesthetic)
# =====================================================================
# Wong colorblind-safe palette (8 colours)
WONG = {
    "black":  "#000000",
    "orange": "#E69F00",
    "skyblu": "#56B4E9",
    "green":  "#009E73",
    "yellow": "#F0E442",
    "blue":   "#0072B2",
    "red":    "#D55E00",
    "purple": "#CC79A7",
}
C_ACC, C_LOSS = WONG["blue"],   WONG["red"]
C_GSNR        = WONG["green"]
C_WASTE       = WONG["purple"]
C_PWR         = WONG["red"]
C_KWH         = WONG["orange"]
C_REF         = "#555555"
C_BEST        = "#222222"
C_SHADE       = "#9aa6b2"

PHASE_SHADE_KW = dict(facecolor="#d9b4a1", alpha=0.12, edgecolor="none", zorder=0)


def apply_publication_style() -> None:
    """ICLR / NeurIPS publication style."""
    available = {f.name for f in mpl.font_manager.fontManager.ttflist}
    preferred = ["CMU Serif", "Computer Modern Roman", "Nimbus Roman",
                 "Times New Roman", "Times", "Liberation Serif", "DejaVu Serif"]
    chosen = next((f for f in preferred if f in available), "DejaVu Serif")

    sns.set_theme(context="paper", style="ticks")
    mpl.rcParams.update({
        "font.family":          "serif",
        "font.serif":           [chosen],
        "mathtext.fontset":     "cm",
        "font.size":            10.0,
        "axes.titlesize":       10.5,
        "axes.titleweight":     "bold",
        "axes.labelsize":       10.0,
        "axes.labelweight":     "regular",
        "xtick.labelsize":      9.0,
        "ytick.labelsize":      9.0,
        "legend.fontsize":      8.5,
        "legend.title_fontsize":9.0,
        "figure.titlesize":     11.5,
        "figure.dpi":           120,
        "savefig.dpi":          DPI_RASTER,
        "savefig.bbox":         "tight",
        "savefig.pad_inches":   0.04,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.linewidth":       0.8,
        "axes.edgecolor":       "#222222",
        "axes.labelcolor":      "#111111",
        "xtick.color":          "#222222",
        "ytick.color":          "#222222",
        "xtick.direction":      "out",
        "ytick.direction":      "out",
        "xtick.major.width":    0.8,
        "ytick.major.size":     3.2,
        "axes.grid":            True,
        "axes.axisbelow":       True,
        "grid.linestyle":       (0, (1, 3)),
        "grid.linewidth":       0.45,
        "grid.color":           "#b8b8b8",
        "grid.alpha":           0.55,
        "lines.linewidth":      1.6,
        "lines.solid_capstyle": "round",
        "lines.markeredgewidth":0.0,
        "legend.frameon":       False,
        "legend.handlelength":  1.5,
        "legend.handletextpad": 0.5,
        "legend.borderaxespad": 0.4,
        "legend.columnspacing": 1.2,
        "pdf.fonttype":         42,
        "ps.fonttype":          42,
        "svg.fonttype":         "none",
        "pdf.compression":      9,
    })


# =====================================================================
# Helpers
# =====================================================================
def load_data(csv_path: str) -> pd.DataFrame:
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(
            f"CSV not found: {p.resolve()}\nEdit CSV_PATH at the top of the script."
        )
    df = pd.read_csv(p)
    print(f"[load] '{p.name}' shape={df.shape}")
    return df


def find_step_column(df: pd.DataFrame) -> str:
    for c in ("train/global_step", "_step"):
        if c in df.columns:
            return c
    raise KeyError("No step column ('train/global_step' or '_step') found.")


def find_best_validation_step(df: pd.DataFrame, step_col: str
                              ) -> Tuple[Optional[float], Optional[float], str]:
    """Return (best_step, best_value, criterion)."""
    if "eval/accuracy" in df.columns:
        sub = df[[step_col, "eval/accuracy"]].dropna()
        if not sub.empty:
            row = sub.loc[sub["eval/accuracy"].idxmax()]
            print(f"[best] step={row[step_col]:.0f}, acc={row['eval/accuracy']:.4f} "
                  f"(criterion=eval/accuracy max)")
            return float(row[step_col]), float(row["eval/accuracy"]), "max eval/accuracy"
    if "eval/loss" in df.columns:
        sub = df[[step_col, "eval/loss"]].dropna()
        if not sub.empty:
            row = sub.loc[sub["eval/loss"].idxmin()]
            print(f"[best] step={row[step_col]:.0f}, loss={row['eval/loss']:.4f} "
                  f"(criterion=eval/loss min)")
            return float(row[step_col]), float(row["eval/loss"]), "min eval/loss"
    print("[best] no eval column -> no best-step marker.")
    return None, None, ""


def has_cols(df: pd.DataFrame, required: Sequence[str], figure: str) -> bool:
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[skip] {figure}: missing -> {missing}")
        return False
    return True


def clean_xy(df: pd.DataFrame, step_col: str, metric: str) -> pd.DataFrame:
    return df[[step_col, metric]].dropna(subset=[metric]).sort_values(step_col)


def shade_post_best(ax: plt.Axes, df: pd.DataFrame, step_col: str,
                    best_step: Optional[float], label: bool = False) -> None:
    if not SHOW_POST_BEST_SHADING or best_step is None:
        return
    xmax = float(df[step_col].max())
    ax.axvspan(best_step, xmax, **PHASE_SHADE_KW,
               label="post-best regime" if label else None)


def mark_best_step(ax: plt.Axes, best_step: Optional[float],
                   show_label: bool = True, annotate_value: bool = False,
                   value_text: str = "") -> None:
    if best_step is None:
        return
    ax.axvline(best_step, color=C_BEST, linestyle=(0, (4, 3)),
               linewidth=0.9, alpha=0.95, zorder=2,
               label="best val step" if show_label else None)
    if annotate_value and value_text:
        ax.annotate(value_text,
                    xy=(best_step, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -8), textcoords="offset points",
                    fontsize=8.0, color=C_BEST, ha="left", va="top",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white",
                              ec="#bbbbbb", lw=0.5, alpha=0.85))


def panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.10, 1.06, f"({letter})", transform=ax.transAxes,
            fontsize=11, fontweight="bold", ha="left", va="bottom",
            color="#111111")


def thousands_fmt(ax: plt.Axes, axis: str = "x") -> None:
    fmt = mticker.FuncFormatter(lambda v, _: f"{int(v):,}")
    (ax.xaxis if axis == "x" else ax.yaxis).set_major_formatter(fmt)


def save_figure(fig: plt.Figure, outdir: Path, stem: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{stem}.png", dpi=DPI_RASTER)
    fig.savefig(outdir / f"{stem}.pdf")
    if SAVE_SVG:
        fig.savefig(outdir / f"{stem}.svg")
        print(f"[saved] {stem}.{{png,pdf,svg}}")
    else:
        print(f"[saved] {stem}.{{png,pdf}}")


# =====================================================================
# FIGURE 1: Evaluation trajectory
# =====================================================================
def fig1_eval_trajectory(df, step_col, best_step, best_val, criterion, outdir):
    name = "fig1_eval_trajectory"
    have_acc = "eval/accuracy" in df.columns
    have_loss = "eval/loss" in df.columns
    if not (have_acc or have_loss):
        print(f"[skip] {name}"); return None

    fig, axes = plt.subplots(2, 1, figsize=(6.8, 5.4), sharex=True,
                             constrained_layout=True)
    if have_acc:
        s = clean_xy(df, step_col, "eval/accuracy")
        axes[0].plot(s[step_col], s["eval/accuracy"],
                     color=C_ACC, marker="o", markersize=3.6,
                     markerfacecolor=C_ACC, markeredgewidth=0,
                     label="validation accuracy")
        shade_post_best(axes[0], df, step_col, best_step, label=True)
        mark_best_step(axes[0], best_step,
                       annotate_value=True,
                       value_text=(f"best @ step {int(best_step)}\n"
                                   f"acc = {best_val:.4f}"
                                   if best_step and "accuracy" in criterion else
                                   (f"best @ step {int(best_step)}"
                                    if best_step else "")))
        axes[0].set_ylabel("Validation accuracy")
        axes[0].legend(loc="lower right", ncol=2)
    if have_loss:
        s = clean_xy(df, step_col, "eval/loss")
        axes[1].plot(s[step_col], s["eval/loss"],
                     color=C_LOSS, marker="s", markersize=3.2,
                     markerfacecolor=C_LOSS, markeredgewidth=0,
                     label="validation loss")
        shade_post_best(axes[1], df, step_col, best_step)
        mark_best_step(axes[1], best_step)
        axes[1].set_ylabel("Validation loss")
        axes[1].legend(loc="upper right")
    axes[-1].set_xlabel(f"Training step  ({step_col})")
    thousands_fmt(axes[-1], "x")
    fig.suptitle(f"Evaluation trajectory  —  {RUN_LABEL}",
                 x=0.02, y=1.03, ha="left", fontsize=11.5, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 2: GSNR decay
# =====================================================================
def fig2_gsnr_decay(df, step_col, best_step, outdir):
    name = "fig2_gsnr_decay"
    if not has_cols(df, ["gsnr/global"], name): return None
    s = clean_xy(df, step_col, "gsnr/global")
    if s.empty: print(f"[skip] {name}: empty"); return None

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    shade_post_best(ax, df, step_col, best_step, label=True)
    ax.plot(s[step_col], s["gsnr/global"], color=C_GSNR,
            marker="o", markersize=3.0, markerfacecolor=C_GSNR,
            markeredgewidth=0, label=r"$\mathrm{GSNR}_{\mathrm{global}}$")
    if (s["gsnr/global"] > 0).all():
        ax.set_yscale("log")
        ax.set_ylabel(r"Global GSNR  (log scale)")
    else:
        ax.set_ylabel("Global GSNR")
        print(f"[note] {name}: non-positive values -> linear y.")
    ax.axhline(1.0, color=C_REF, linestyle=":", linewidth=1.0,
               label=r"$\mathrm{GSNR}=1$")
    mark_best_step(ax, best_step)
    ax.set_xlabel(f"Training step  ({step_col})")
    thousands_fmt(ax, "x")
    ax.set_title("Gradient-signal-quality decay", loc="left", pad=4)
    ax.legend(loc="upper right", ncol=2)
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 3: Waste trajectory
# =====================================================================
def fig3_waste_ratio(df, step_col, best_step, outdir):
    name = "fig3_waste_ratio"
    if not has_cols(df, ["waste/ratio"], name): return None
    s = clean_xy(df, step_col, "waste/ratio")
    if s.empty: print(f"[skip] {name}: empty"); return None

    fig, ax = plt.subplots(figsize=(6.8, 3.6), constrained_layout=True)
    shade_post_best(ax, df, step_col, best_step, label=True)

    if {"waste/ci_95_low", "waste/ci_95_high"}.issubset(df.columns):
        b = df[[step_col, "waste/ci_95_low", "waste/ci_95_high"]].dropna()
        if not b.empty:
            ax.fill_between(b[step_col], b["waste/ci_95_low"], b["waste/ci_95_high"],
                            color=C_WASTE, alpha=0.18, linewidth=0,
                            label="95% CI band")

    ax.plot(s[step_col], s["waste/ratio"], color=C_WASTE,
            marker="o", markersize=3.0, markerfacecolor=C_WASTE,
            markeredgewidth=0, label="waste ratio")
    vmin, vmax = float(s["waste/ratio"].min()), float(s["waste/ratio"].max())
    if vmin >= 0.0 and vmax <= 1.0:
        ax.set_ylim(0.0, 1.0)
    mark_best_step(ax, best_step)
    ax.set_xlabel(f"Training step  ({step_col})")
    ax.set_ylabel("Waste ratio")
    thousands_fmt(ax, "x")
    ax.set_title("Late-training waste trajectory", loc="left", pad=4)
    ax.legend(loc="upper left")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 4: Power and energy
# =====================================================================
def fig4_power_energy(df, step_col, best_step, outdir):
    name = "fig4_power_energy"
    have_p = "power/current_watts" in df.columns
    have_e = "power/cumulative_kwh" in df.columns
    if not (have_p or have_e):
        print(f"[skip] {name}"); return None
    n = int(have_p) + int(have_e)
    fig, axes = plt.subplots(n, 1, figsize=(6.8, 2.8 * n + 0.4),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    i = 0
    if have_p:
        s = clean_xy(df, step_col, "power/current_watts")
        shade_post_best(axes[i], df, step_col, best_step, label=True)
        axes[i].plot(s[step_col], s["power/current_watts"], color=C_PWR,
                     linewidth=1.0, alpha=0.95, label="instantaneous power")
        if len(s) > 12:
            roll = s["power/current_watts"].rolling(11, min_periods=3, center=True).mean()
            axes[i].plot(s[step_col], roll, color="#222222", linewidth=1.4,
                         alpha=0.85, label="rolling mean (11 pts)")
        mark_best_step(axes[i], best_step)
        axes[i].set_ylabel("Power (W)")
        axes[i].legend(loc="best", ncol=2)
        i += 1
    if have_e:
        s = clean_xy(df, step_col, "power/cumulative_kwh")
        shade_post_best(axes[i], df, step_col, best_step,
                        label=(not have_p))
        axes[i].plot(s[step_col], s["power/cumulative_kwh"], color=C_KWH,
                     linewidth=1.8, label="cumulative energy")
        if best_step is not None:
            post = s[s[step_col] >= best_step]
            if not post.empty:
                e0 = float(post["power/cumulative_kwh"].iloc[0])
                e1 = float(post["power/cumulative_kwh"].iloc[-1])
                axes[i].annotate(
                    f"post-best energy spent\n"
                    f"$\\Delta$ = {e1 - e0:.3f} kWh",
                    xy=(post[step_col].iloc[-1], e1),
                    xytext=(-10, -28), textcoords="offset points",
                    ha="right", fontsize=8.5, color="#222222",
                    arrowprops=dict(arrowstyle="-", color="#666", lw=0.6))
        mark_best_step(axes[i], best_step)
        axes[i].set_ylabel("Cumulative energy (kWh)")
        axes[i].legend(loc="best")
    axes[-1].set_xlabel(f"Training step  ({step_col})")
    thousands_fmt(axes[-1], "x")
    fig.suptitle("Power and energy", x=0.02, y=1.02, ha="left",
                 fontsize=11.5, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 5: Master 4-panel case study (a)(b)(c)(d)
# =====================================================================
def fig5_case_study(df, step_col, best_step, best_val, criterion, outdir):
    name = "fig5_phase1_1_case_study"
    have_acc  = "eval/accuracy" in df.columns
    have_loss = "eval/loss" in df.columns
    have_gsnr = "gsnr/global" in df.columns
    have_wst  = "waste/ratio" in df.columns
    have_kwh  = "power/cumulative_kwh" in df.columns
    if not any([have_acc, have_loss, have_gsnr, have_wst, have_kwh]):
        print(f"[skip] {name}"); return None

    fig, axes = plt.subplots(4, 1, figsize=(7.0, 10.2), sharex=True,
                             constrained_layout=True,
                             gridspec_kw=dict(hspace=0.08))
    (axA, axB, axC, axD) = axes

    # (a) accuracy + loss
    shade_post_best(axA, df, step_col, best_step, label=True)
    h_a = []
    if have_acc:
        s = clean_xy(df, step_col, "eval/accuracy")
        l1, = axA.plot(s[step_col], s["eval/accuracy"], color=C_ACC,
                       marker="o", markersize=3.0, markerfacecolor=C_ACC,
                       markeredgewidth=0, label="eval accuracy")
        h_a.append(l1)
        axA.set_ylabel("Validation accuracy", color=C_ACC)
        axA.tick_params(axis="y", labelcolor=C_ACC)
    if have_loss:
        ax2 = axA.twinx()
        ax2.spines["top"].set_visible(False)
        ax2.grid(False)
        s = clean_xy(df, step_col, "eval/loss")
        l2, = ax2.plot(s[step_col], s["eval/loss"], color=C_LOSS,
                       marker="s", markersize=2.8, markerfacecolor=C_LOSS,
                       markeredgewidth=0, linestyle=(0, (4, 2)),
                       label="eval loss")
        h_a.append(l2)
        ax2.set_ylabel("Validation loss", color=C_LOSS)
        ax2.tick_params(axis="y", labelcolor=C_LOSS)
    mark_best_step(axA, best_step,
                   annotate_value=True,
                   value_text=(f"best @ {int(best_step)}\n"
                               f"acc = {best_val:.4f}"
                               if best_step and "accuracy" in criterion else
                               (f"best @ {int(best_step)}" if best_step else "")))
    if h_a:
        axA.legend(h_a, [h.get_label() for h in h_a],
                   loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=3,
                   frameon=False)
    panel_label(axA, "a")

    # (b) GSNR
    shade_post_best(axB, df, step_col, best_step)
    if have_gsnr:
        s = clean_xy(df, step_col, "gsnr/global")
        axB.plot(s[step_col], s["gsnr/global"], color=C_GSNR,
                 marker="o", markersize=2.6, markerfacecolor=C_GSNR,
                 markeredgewidth=0, label=r"$\mathrm{GSNR}_{\mathrm{global}}$")
        if (s["gsnr/global"] > 0).all():
            axB.set_yscale("log")
        axB.axhline(1.0, color=C_REF, linestyle=":", linewidth=0.9,
                    label=r"$\mathrm{GSNR}=1$")
        axB.set_ylabel("Global GSNR")
        mark_best_step(axB, best_step)
        axB.legend(loc="upper right", ncol=2)
    else:
        axB.text(0.5, 0.5, "gsnr/global not available", transform=axB.transAxes,
                 ha="center", va="center", color="#888")
    panel_label(axB, "b")

    # (c) waste
    shade_post_best(axC, df, step_col, best_step)
    if have_wst:
        s = clean_xy(df, step_col, "waste/ratio")
        axC.plot(s[step_col], s["waste/ratio"], color=C_WASTE,
                 marker="o", markersize=2.6, markerfacecolor=C_WASTE,
                 markeredgewidth=0, label="waste ratio")
        vmin, vmax = float(s["waste/ratio"].min()), float(s["waste/ratio"].max())
        if vmin >= 0 and vmax <= 1: axC.set_ylim(0, 1)
        axC.set_ylabel("Waste ratio")
        mark_best_step(axC, best_step)
        axC.legend(loc="upper left")
    else:
        axC.text(0.5, 0.5, "waste/ratio not available", transform=axC.transAxes,
                 ha="center", va="center", color="#888")
    panel_label(axC, "c")

    # (d) energy
    shade_post_best(axD, df, step_col, best_step)
    if have_kwh:
        s = clean_xy(df, step_col, "power/cumulative_kwh")
        axD.plot(s[step_col], s["power/cumulative_kwh"], color=C_KWH,
                 linewidth=1.8, label="cumulative energy")
        axD.set_ylabel("Cumulative kWh")
        mark_best_step(axD, best_step)
        axD.legend(loc="upper left")
    else:
        axD.text(0.5, 0.5, "power/cumulative_kwh not available",
                 transform=axD.transAxes, ha="center", va="center", color="#888")
    panel_label(axD, "d")
    axD.set_xlabel(f"Training step  ({step_col})")
    thousands_fmt(axD, "x")

    fig.suptitle(f"Phase 1.1 diagnostic baseline  —  {RUN_LABEL}",
                 x=0.02, y=1.005, ha="left", fontsize=12.0, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 6: mplot_density + mplot_splits
#          KDE of log10(GSNR) pre- vs post-best regime
# =====================================================================
def fig6_gsnr_density(df, step_col, best_step, outdir):
    name = "fig6_gsnr_density_splits"
    if "gsnr/global" not in df.columns or best_step is None:
        print(f"[skip] {name}"); return None
    s = clean_xy(df, step_col, "gsnr/global")
    s = s[s["gsnr/global"] > 0].copy()
    if s.empty: print(f"[skip] {name}"); return None
    s["regime"] = np.where(s[step_col] < best_step, "pre-best", "post-best")
    s["log10_gsnr"] = np.log10(s["gsnr/global"])

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4),
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[3, 2]))

    axL = axes[0]
    palette = {"pre-best": WONG["blue"], "post-best": WONG["red"]}
    for reg, sub in s.groupby("regime"):
        vals = sub["log10_gsnr"].to_numpy()
        if len(vals) < 3: continue
        kde = stats.gaussian_kde(vals)
        xs = np.linspace(vals.min() - 0.3, vals.max() + 0.3, 400)
        axL.fill_between(xs, kde(xs), alpha=0.30, color=palette[reg], linewidth=0)
        axL.plot(xs, kde(xs), color=palette[reg], linewidth=1.6, label=reg)
        axL.plot(vals, np.full_like(vals, -0.02), "|", color=palette[reg],
                 markersize=6, alpha=0.7)
        axL.axvline(np.median(vals), color=palette[reg], linestyle=":",
                    linewidth=1.0, alpha=0.9)
    axL.axvline(0.0, color=C_REF, linestyle="--", linewidth=0.9)
    axL.text(0.02, 0.0 - 0.07, r"$\mathrm{GSNR}=1$", color=C_REF,
             transform=axL.get_xaxis_transform(), fontsize=8.5)
    axL.set_xlabel(r"$\log_{10}\,\mathrm{GSNR}_{\mathrm{global}}$")
    axL.set_ylabel("Density")
    axL.set_title("Distribution shift across best-step", loc="left", pad=4)
    axL.legend(loc="upper right", title="regime")
    panel_label(axL, "a")

    axR = axes[1]
    order = ["pre-best", "post-best"]
    sns.boxplot(data=s, x="regime", y="log10_gsnr", order=order,
                ax=axR, palette=palette, width=0.5, fliersize=0,
                linewidth=0.8, boxprops=dict(alpha=0.55))
    sns.stripplot(data=s, x="regime", y="log10_gsnr", order=order,
                  ax=axR, palette=palette, size=2.6, alpha=0.85, jitter=0.18)
    axR.axhline(0.0, color=C_REF, linestyle="--", linewidth=0.9)
    axR.set_xlabel("")
    axR.set_ylabel(r"$\log_{10}\,\mathrm{GSNR}$")
    axR.set_title("Split comparison", loc="left", pad=4)
    panel_label(axR, "b")

    pre = s.loc[s["regime"] == "pre-best", "log10_gsnr"].to_numpy()
    post = s.loc[s["regime"] == "post-best", "log10_gsnr"].to_numpy()
    if len(pre) > 1 and len(post) > 1:
        try:
            mw = stats.mannwhitneyu(pre, post, alternative="greater")
            ymax = s["log10_gsnr"].max()
            axR.text(0.5, ymax + 0.05,
                     f"Mann-Whitney  pre > post:  p = {mw.pvalue:.2g}",
                     ha="center", fontsize=8.5, color="#222")
        except Exception:
            pass

    fig.suptitle("GSNR density & regime split  —  pre- vs post-best step",
                 x=0.02, y=1.05, ha="left", fontsize=11.5, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 7: mplot_roc
#          Diagnostic ROC: can GSNR / waste detect post-best regime?
# =====================================================================
def _roc_curve(scores: np.ndarray, labels: np.ndarray):
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    tps = np.cumsum(y == 1)
    fps = np.cumsum(y == 0)
    P = max(int((labels == 1).sum()), 1)
    N = max(int((labels == 0).sum()), 1)
    tpr = np.concatenate(([0.0], tps / P))
    fpr = np.concatenate(([0.0], fps / N))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    auc = float(trapezoid(tpr, fpr))
    return fpr, tpr, auc


def fig7_diagnostic_roc(df, step_col, best_step, outdir):
    name = "fig7_diagnostic_roc"
    if best_step is None:
        print(f"[skip] {name}: no best-step"); return None

    candidates = {
        r"$1-\mathrm{GSNR}/(1+\mathrm{GSNR})$": "gsnr/global",
        "waste/ratio":                           "waste/ratio",
        "train/grad_norm":                       "train/grad_norm",
    }
    avail = {k: v for k, v in candidates.items() if v in df.columns}
    if not avail:
        print(f"[skip] {name}: no diagnostic signal available"); return None

    fig, ax = plt.subplots(figsize=(5.0, 4.8), constrained_layout=True)
    ax.plot([0, 1], [0, 1], color=C_REF, linestyle="--", linewidth=0.9,
            label="chance (AUC = 0.50)")

    colors = [WONG["green"], WONG["purple"], WONG["orange"], WONG["skyblu"]]
    for (label_disp, col), color in zip(avail.items(), colors):
        sub = df[[step_col, col]].dropna()
        if sub.empty: continue
        y = (sub[step_col].to_numpy() >= best_step).astype(int)
        x = sub[col].to_numpy(dtype=float)
        if col == "gsnr/global":
            score = -x
            disp  = r"$-\,\mathrm{GSNR}_{\mathrm{global}}$"
        else:
            score = x
            disp  = col
        if y.sum() == 0 or y.sum() == len(y): continue
        fpr, tpr, auc = _roc_curve(score, y)
        ax.plot(fpr, tpr, color=color, linewidth=1.8,
                label=f"{disp}   (AUC = {auc:.3f})")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1.001)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Diagnostic ROC  —  predicting post-best regime", loc="left", pad=4)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_aspect("equal", adjustable="box")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 8: mplot_cuts — boxplots by training-step quintile
# =====================================================================
def fig8_quantile_cuts(df, step_col, best_step, outdir):
    name = "fig8_quantile_cuts"
    metrics = [
        ("eval/loss",   "Validation loss"),
        ("gsnr/global", r"$\log_{10}\,\mathrm{GSNR}$"),
        ("waste/ratio", "Waste ratio"),
    ]
    metrics = [(m, lbl) for m, lbl in metrics if m in df.columns]
    if not metrics:
        print(f"[skip] {name}"); return None

    sub = df[[step_col] + [m for m, _ in metrics]].copy()
    sub = sub.dropna(subset=[step_col])
    sub["q"] = pd.qcut(sub[step_col], q=5,
                       labels=["Q1\n(early)", "Q2", "Q3", "Q4", "Q5\n(late)"])

    if "gsnr/global" in sub.columns:
        sub["gsnr/global"] = np.where(sub["gsnr/global"] > 0,
                                      np.log10(sub["gsnr/global"]), np.nan)

    fig, axes = plt.subplots(1, len(metrics),
                             figsize=(2.7 * len(metrics) + 0.8, 3.6),
                             constrained_layout=True)
    if len(metrics) == 1: axes = [axes]

    palette = sns.color_palette("YlOrRd", n_colors=5)
    for ax, (col, lbl), letter in zip(axes, metrics, "abcdef"):
        data = sub[["q", col]].dropna()
        sns.boxplot(data=data, x="q", y=col, ax=ax, palette=palette,
                    width=0.62, fliersize=0, linewidth=0.7,
                    boxprops=dict(alpha=0.85))
        sns.stripplot(data=data, x="q", y=col, ax=ax, color="#222",
                      size=2.0, alpha=0.5, jitter=0.18)
        ax.set_xlabel("Training step quintile")
        ax.set_ylabel(lbl)
        ax.set_title(f"({letter})  {col}", loc="left", pad=3, fontsize=10)
        for tick in ax.get_xticklabels():
            tick.set_fontsize(8.5)

    fig.suptitle("Metric cuts across training quintiles  —  pre→post saturation",
                 x=0.02, y=1.04, ha="left", fontsize=11.5, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 9: mplot_full — multi-metric standardized heatmap
# =====================================================================
def fig9_metric_heatmap(df, step_col, best_step, outdir):
    name = "fig9_multi_metric_heatmap"
    cand = ["eval/accuracy", "eval/loss", "gsnr/global", "fisher/global",
            "waste/ratio", "train/loss", "train/grad_norm",
            "train/learning_rate", "power/current_watts",
            "power/cumulative_kwh", "power/gpu_utilization_pct"]
    cols = [c for c in cand if c in df.columns]
    if len(cols) < 3:
        print(f"[skip] {name}: not enough metrics"); return None

    sub = df[[step_col] + cols].dropna(subset=[step_col]).copy()
    n_bins = min(40, max(15, len(sub) // 4))
    sub["bin"] = pd.cut(sub[step_col], bins=n_bins, labels=False)
    binned = sub.groupby("bin")[cols].mean()
    bin_steps = sub.groupby("bin")[step_col].mean().to_numpy()

    Z = binned.copy()
    for c in cols:
        v = Z[c].to_numpy(dtype=float)
        mu, sd = np.nanmean(v), np.nanstd(v)
        Z[c] = (v - mu) / (sd if sd > 1e-12 else 1.0)

    fig, ax = plt.subplots(figsize=(7.6, 0.42 * len(cols) + 2.0),
                           constrained_layout=True)
    mat = Z.to_numpy().T
    vmax = float(np.nanpercentile(np.abs(mat), 98))
    vmax = max(vmax, 0.5)

    im = ax.imshow(mat, aspect="auto", origin="lower",
                   cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[bin_steps.min(), bin_steps.max(),
                           -0.5, len(cols) - 0.5],
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(cols)))
    ax.set_yticklabels(cols, fontsize=9)
    ax.set_xlabel(f"Training step  ({step_col})")
    ax.grid(False)
    if best_step is not None:
        ax.axvline(best_step, color="#111", linestyle=(0, (3, 2)), linewidth=1.0)
        ax.text(best_step, len(cols) - 0.4, " best step",
                ha="left", va="top", fontsize=8.5, color="#111")
    thousands_fmt(ax, "x")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
    cbar.set_label("z-score (per metric)")
    ax.set_title("Multi-metric trajectory  —  standardized across training",
                 loc="left", pad=4)
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# FIGURE 10: mplot_importance — Spearman rho vs waste/ratio
# =====================================================================
def fig10_signal_importance(df, step_col, outdir, target: str = "waste/ratio",
                            top_k: int = 18):
    name = "fig10_signal_importance"
    if target not in df.columns:
        print(f"[skip] {name}: no '{target}'"); return None
    num = df.select_dtypes(include=[np.number]).copy()
    if target not in num.columns:
        print(f"[skip] {name}: '{target}' not numeric"); return None

    drop_like = (step_col, target, "_step", "_runtime", "_timestamp", "train/epoch")
    feats = [c for c in num.columns if c not in drop_like
             and not c.startswith("run_meta/")
             and not c.startswith("dynamics/")
             and num[c].notna().sum() > 5]

    rows = []
    for c in feats:
        a = num[[c, target]].dropna()
        if len(a) < 6 or a[c].nunique() < 3: continue
        try:
            rho, p = stats.spearmanr(a[c], a[target])
        except Exception:
            continue
        if np.isnan(rho): continue
        rows.append((c, rho, p))
    if not rows:
        print(f"[skip] {name}: no usable features"); return None
    imp = pd.DataFrame(rows, columns=["feature", "rho", "p"])
    imp["abs_rho"] = imp["rho"].abs()
    imp = imp.sort_values("abs_rho", ascending=False).head(top_k)
    imp = imp.iloc[::-1]

    fig, ax = plt.subplots(figsize=(6.6, 0.32 * len(imp) + 1.6),
                           constrained_layout=True)
    colors = [WONG["red"] if r > 0 else WONG["blue"] for r in imp["rho"]]
    bars = ax.barh(imp["feature"], imp["rho"], color=colors,
                   edgecolor="#333", linewidth=0.4, height=0.7)
    ax.axvline(0, color="#333", linewidth=0.8)
    for bar, (_, row) in zip(bars, imp.iterrows()):
        star = ("***" if row["p"] < 1e-3 else
                "**"  if row["p"] < 1e-2 else
                "*"   if row["p"] < 5e-2 else "")
        if star:
            x = row["rho"]
            ax.text(x + (0.015 if x >= 0 else -0.015),
                    bar.get_y() + bar.get_height() / 2,
                    star, va="center",
                    ha="left" if x >= 0 else "right",
                    fontsize=9, color="#111")
    ax.set_xlim(-1.05, 1.05)
    ax.set_xlabel(rf"Spearman $\rho$  with  {target}")
    ax.set_title(rf"Signal importance  —  ranked by $|\rho|$ vs  {target}",
                 loc="left", pad=4)
    legend_handles = [
        mpatches.Patch(color=WONG["red"],  label=r"$\rho>0$  (co-rises with waste)"),
        mpatches.Patch(color=WONG["blue"], label=r"$\rho<0$  (anti-correlated)"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8.5)
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# APPENDIX A1: Training dynamics
# =====================================================================
def appendix_training_dynamics(df, step_col, best_step, outdir):
    name = "appendix_training_dynamics"
    cols = ["train/loss", "train/grad_norm", "train/learning_rate"]
    pres = [c for c in cols if c in df.columns]
    if not pres: print(f"[skip] {name}"); return None
    fig, axes = plt.subplots(len(pres), 1,
                             figsize=(6.8, 2.4 * len(pres) + 0.6),
                             sharex=True, constrained_layout=True)
    axes = np.atleast_1d(axes)
    cmap = {"train/loss": WONG["blue"], "train/grad_norm": WONG["green"],
            "train/learning_rate": WONG["purple"]}
    ylab = {"train/loss": "Training loss", "train/grad_norm": "Gradient norm",
            "train/learning_rate": "Learning rate"}
    for ax, col, letter in zip(axes, pres, "abc"):
        shade_post_best(ax, df, step_col, best_step, label=(letter == "a"))
        s = clean_xy(df, step_col, col)
        ax.plot(s[step_col], s[col], color=cmap[col], linewidth=1.0,
                alpha=0.92, label=col)
        if len(s) > 12:
            roll = s[col].rolling(11, min_periods=3, center=True).mean()
            ax.plot(s[step_col], roll, color="#222", linewidth=1.4,
                    alpha=0.85, label="rolling mean (11 pts)")
        mark_best_step(ax, best_step)
        ax.set_ylabel(ylab[col])
        ax.legend(loc="best", ncol=2)
        panel_label(ax, letter)
    axes[-1].set_xlabel(f"Training step  ({step_col})")
    thousands_fmt(axes[-1], "x")
    fig.suptitle("Appendix A1  —  Training dynamics",
                 x=0.02, y=1.02, ha="left", fontsize=11.5, fontweight="bold")
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# APPENDIX A2: Layer-resolved GSNR heatmap
# =====================================================================
def appendix_layer_gsnr_heatmap(df, step_col, best_step, outdir):
    name = "appendix_layer_gsnr_heatmap"
    layer_cols = [c for c in df.columns
                  if (c.startswith("gsnr/top_") or c.startswith("gsnr/bottom_"))
                  and c not in ("gsnr/global", "gsnr/global_log10")]
    if not layer_cols: print(f"[skip] {name}"); return None

    sub = df[[step_col] + layer_cols].dropna(subset=[step_col]).copy()
    sub = sub.dropna(axis=1, how="all")
    layer_cols = [c for c in sub.columns if c != step_col]
    sub = sub.dropna(subset=layer_cols, how="all")
    if sub.empty: print(f"[skip] {name}"); return None

    top = sorted([c for c in layer_cols if c.startswith("gsnr/top_")])
    bot = sorted([c for c in layer_cols if c.startswith("gsnr/bottom_")])
    ordered = top + bot

    steps = sub[step_col].to_numpy()
    M = sub[ordered].to_numpy().T
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(M > 0, np.log10(M), np.nan)
    if np.isnan(L).all(): print(f"[skip] {name}"); return None

    vmin = np.nanpercentile(L, 2); vmax = np.nanpercentile(L, 98)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(L), np.nanmax(L)

    fig_h = max(4.5, 0.34 * len(ordered) + 1.8)
    fig, ax = plt.subplots(figsize=(8.4, fig_h), constrained_layout=True)
    im = ax.imshow(L, aspect="auto", origin="lower", cmap="viridis",
                   vmin=vmin, vmax=vmax,
                   extent=[steps.min(), steps.max(),
                           -0.5, len(ordered) - 0.5],
                   interpolation="nearest")
    ax.set_yticks(np.arange(len(ordered)))
    ax.set_yticklabels([c.replace("gsnr/", "") for c in ordered], fontsize=7.8)
    if best_step is not None:
        ax.axvline(best_step, color="white", linestyle=(0, (3, 2)), linewidth=1.1)
        ax.text(best_step, len(ordered) - 0.4, " best step",
                ha="left", va="top", fontsize=8.5, color="white")
    thousands_fmt(ax, "x")
    ax.set_xlabel(f"Training step  ({step_col})")
    ax.set_ylabel("Parameter group")
    ax.grid(False)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
    cbar.set_label(r"$\log_{10}\,\mathrm{GSNR}$")
    ax.set_title("Appendix A2  —  Layer-resolved GSNR heatmap",
                 loc="left", pad=4)
    save_figure(fig, outdir, name)
    plt.close(fig); return name


# =====================================================================
# CAPTIONS
# =====================================================================
CAPTIONS = {
"fig1_eval_trajectory":
    "Figure 1 — Evaluation trajectory. Top: validation accuracy. Bottom: validation loss. "
    "The dashed vertical line marks the best validation step; the salmon tint marks the "
    "post-best (saturation) regime. Pattern: accuracy rises then saturates while loss "
    "reaches a minimum and may climb afterwards. Why it matters: anchors the LERNA story — "
    "validation gains stop, yet training continues.",
"fig2_gsnr_decay":
    "Figure 2 — GSNR decay diagnostic. Global GSNR on log y-axis with a GSNR=1 reference; "
    "post-best region tinted. Pattern: monotone decay through training, with values crossing "
    "below 1 in the saturation regime. Why it matters: this is the gradient-quality signature "
    "underpinning LERNA's backward-skipping hypothesis.",
"fig3_waste_ratio":
    "Figure 3 — Waste trajectory. Waste ratio per step, with optional 95% CI band when "
    "available, and the post-best region shaded. Pattern: rising waste ratio after the best "
    "validation step. Why it matters: a direct, model-agnostic measure of compute wastage.",
"fig4_power_energy":
    "Figure 4 — Power and energy. Top: instantaneous GPU power with a rolling-mean overlay. "
    "Bottom: cumulative training energy (kWh), annotated with energy spent after the best "
    "validation step. Why it matters: anchors the diagnostic story in a real, billable cost.",
"fig5_phase1_1_case_study":
    "Figure 5 — Phase 1.1 case study (main paper figure). Four stacked panels with shared "
    "x-axis: (a) eval accuracy + eval loss, (b) global GSNR, (c) waste ratio, (d) cumulative "
    "energy. Best step marked across all panels; post-best regime shaded. Pattern: "
    "saturation (a), GSNR decay (b), waste rise (c), continued energy spend (d). "
    "Why it matters: this single composite figure summarises the Phase 1.1 narrative.",
"fig6_gsnr_density_splits":
    "Figure 6 — GSNR density and pre/post split. (a) KDE of log10(GSNR) before vs after the "
    "best validation step with rug ticks and median lines. (b) Box-and-strip comparison of "
    "the same two regimes, with a one-sided Mann-Whitney p-value. Why it matters: makes the "
    "regime shift in gradient quality quantitative, not just visual.",
"fig7_diagnostic_roc":
    "Figure 7 — Diagnostic ROC. Treating 'step >= best step' as a binary label, we score each "
    "step with -GSNR_global, waste/ratio, and train/grad_norm. Higher AUC means the signal "
    "more cleanly separates the pre-best from the post-best regime. Why it matters: motivates "
    "which signal is best for triggering backward-skipping in later phases of LERNA.",
"fig8_quantile_cuts":
    "Figure 8 — Quantile cuts. Boxplots (with overlaid strip) of validation loss, log10(GSNR), "
    "and waste ratio across step-quintiles Q1..Q5. Why it matters: shows the monotone shift "
    "of every diagnostic signal across the run, robust to outliers.",
"fig9_multi_metric_heatmap":
    "Figure 9 — Multi-metric heatmap. Per-bin mean of each available metric, standardized "
    "(z-score) per metric and plotted against training step. Best step shown as a dashed "
    "guide. Why it matters: a single bird's-eye view that makes the joint dynamics of "
    "evaluation, gradients, waste and energy directly comparable.",
"fig10_signal_importance":
    "Figure 10 — Signal importance. Spearman rho between every available numeric signal and "
    "waste/ratio, sorted by |rho|; */**/*** mark p<5e-2, 1e-2, 1e-3. Red bars co-rise with "
    "waste, blue bars are anti-correlated. Why it matters: identifies which available "
    "telemetry channels best explain waste — the candidate triggers for LERNA's policy.",
"appendix_training_dynamics":
    "Appendix A1 — Training dynamics. Training loss, gradient norm, and learning rate vs "
    "step, with rolling-mean overlays, best-step marker, and post-best shading. Documents the "
    "raw optimisation behaviour underlying the main figures.",
"appendix_layer_gsnr_heatmap":
    "Appendix A2 — Layer-resolved GSNR heatmap. log10(GSNR) for each available gsnr/top_* "
    "and gsnr/bottom_* parameter group across training, with the best step overlaid. "
    "Reveals where in the network gradient quality decays fastest.",
}


def print_captions(gen):
    print("\n" + "=" * 78)
    print("FIGURE CAPTIONS")
    print("=" * 78)
    for n in gen:
        if n in CAPTIONS:
            print(f"\n{CAPTIONS[n]}")
    print("\n" + "=" * 78)


# =====================================================================
# MAIN
# =====================================================================
def _add(lst, name):
    if name: lst.append(name)


def main() -> int:
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    apply_publication_style()

    outdir = Path(OUTPUT_DIR); outdir.mkdir(parents=True, exist_ok=True)

    try:
        df = load_data(CSV_PATH)
    except FileNotFoundError as e:
        print(f"[error] {e}"); return 1

    try:
        step_col = find_step_column(df)
        print(f"[step-col] using '{step_col}'")
    except KeyError as e:
        print(f"[error] {e}"); return 1

    if "phase/current" in df.columns:
        print("[note] 'phase/current' present but intentionally NOT plotted "
              "(noisy phase detector).")

    best_step, best_val, criterion = find_best_validation_step(df, step_col)

    generated: List[str] = []
    # core figures
    _add(generated, fig1_eval_trajectory(df, step_col, best_step, best_val, criterion, outdir))
    _add(generated, fig2_gsnr_decay(df, step_col, best_step, outdir))
    _add(generated, fig3_waste_ratio(df, step_col, best_step, outdir))
    _add(generated, fig4_power_energy(df, step_col, best_step, outdir))
    _add(generated, fig5_case_study(df, step_col, best_step, best_val, criterion, outdir))
    # advanced figures
    _add(generated, fig6_gsnr_density(df, step_col, best_step, outdir))
    _add(generated, fig7_diagnostic_roc(df, step_col, best_step, outdir))
    _add(generated, fig8_quantile_cuts(df, step_col, best_step, outdir))
    _add(generated, fig9_metric_heatmap(df, step_col, best_step, outdir))
    _add(generated, fig10_signal_importance(df, step_col, outdir))
    # appendices
    _add(generated, appendix_training_dynamics(df, step_col, best_step, outdir))
    _add(generated, appendix_layer_gsnr_heatmap(df, step_col, best_step, outdir))

    print(f"\n[done] {len(generated)} figure(s) -> {outdir.resolve()}")
    print_captions(generated)
    return 0


if __name__ == "__main__":
    sys.exit(main())