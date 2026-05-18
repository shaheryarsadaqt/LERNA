"""
Comprehensive Metrics Logging - All Chart Types (wandb 0.25.1 compatible)

wandb.plot API:
- bar(table, label, value)
- scatter(table, x, y)
- line_series(xs, ys, keys)
- confusion_matrix(probs, y_true, preds, class_names)
- histogram(table, value)
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

import torch
from transformers import TrainerCallback


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


class AllChartsMetricsCallback(TrainerCallback):
    """
    Logs all chart types using wandb.plot API.
    """

    def __init__(
        self,
        log_frequency: int = 10,
        wandb_enabled: bool = True,
        output_dir: str = "./experiments/all_charts_logs",
    ):
        self.log_frequency = log_frequency
        self.wandb_enabled = wandb_enabled
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.train_start_time = None
        self.step_count = 0

        self.loss_history = []
        self.grad_norm_history = []
        self.lr_history = []
        self.memory_history = []
        self.eval_history = []
        self.step_times = []
        self.layer_grad_buffer = []
        self.correlation_buffer = []
        self.eval_buffer = []

        print(f"  AllChartsMetricsCallback initialized")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.step_count = 0
        if _wandb_active():
            for prefix in ["train", "eval", "power", "memory", "compute", "grad_norm", "chart", "summary"]:
                try:
                    wandb.define_metric(f"{prefix}/*")
                except Exception:
                    pass
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "_step_start_time"):
            self.step_times.append(time.time() - self._step_start_time)
        self._step_start_time = time.time()

        step = state.global_step
        self.step_count += 1

        if state.log_history:
            last_log = state.log_history[-1] or {}
            loss = last_log.get("loss")
            if loss is not None:
                self.loss_history.append({"step": step, "loss": float(loss)})
            lr = last_log.get("learning_rate")
            if lr:
                self.lr_history.append({"step": step, "lr": float(lr)})

        grad_norm = getattr(state, "_grad_norm", None)
        if grad_norm is not None:
            gn = float(grad_norm)
            self.grad_norm_history.append({"step": step, "grad_norm": gn})

        if torch.cuda.is_available():
            try:
                self.memory_history.append({
                    "step": step,
                    "allocated": torch.cuda.memory_allocated() / 1e9,
                    "reserved": torch.cuda.memory_reserved() / 1e9,
                })
            except Exception:
                pass

        if step % self.log_frequency == 0:
            self._log_all_charts(step)

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}
        step = state.global_step

        entry = {"step": step, "epoch": float(state.epoch or 0)}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                entry[k] = float(v)
        self.eval_history.append(entry)

        self._log_eval_charts(step, metrics)
        self._log_confusion_matrix(kwargs.get("predictions"), kwargs.get("labels"), step)
        self._log_correlation_heatmap(step)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        self._log_summary_charts()
        self._save_data()
        return control

    # =============================================================================
    # ALL CHARTS
    # =============================================================================

    def _log_all_charts(self, step: int):
        if not _wandb_active():
            return
        try:
            self._log_line_chart(step)
            self._log_bar_chart(step)
            self._log_scatter_plot(step)
            self._log_histogram(step)
            self._log_box_chart(step)
            self._log_stacked_area_chart(step)
            self._log_parallel_coordinates(step)
            self._log_pie_donut_chart(step)
            self._log_stacked_bar_chart(step)
            self._log_radar_chart(step)
            self._log_scatter_correlation(step)
            self._log_bump_chart(step)
            self._log_treemap_bar(step)
            self._log_word_cloud_bar(step)
            self._log_network_diagram(step)
            self._log_nightingale_chart(step)
            self._log_waterfall_chart(step)
            self._log_funnel_chart(step)
            self._log_sunburst_chart(step)
            self._log_sankey_chart(step)
        except Exception as e:
            print(f"  [W&B] charts failed: {e}")

    # --- COMPARISON CHARTS (1-22) ---

    def _log_bar_chart(self, step: int):
        if len(self.loss_history) < 5:
            return
        recent = self.loss_history[-15:]
        table = wandb.Table(
            columns=["step", "loss"],
            data=[[f"s{r['step']}", r["loss"]] for r in recent]
        )
        wandb.log({"chart/bar_loss": wandb.plot.bar(table, "step", "loss", title="Bar Chart")}, step=step)

    def _log_column_chart(self, step: int):
        if len(self.grad_norm_history) < 5:
            return
        recent = self.grad_norm_history[-15:]
        table = wandb.Table(
            columns=["step", "grad_norm"],
            data=[[f"s{r['step']}", r["grad_norm"]] for r in recent]
        )
        wandb.log({"chart/column_gradnorm": wandb.plot.bar(table, "step", "grad_norm", title="Column Chart")}, step=step)

    def _log_grouped_bar_chart(self, step: int):
        if len(self.loss_history) < 5 or len(self.lr_history) < 5:
            return
        min_len = min(len(self.loss_history), len(self.lr_history))
        data = [
            [f"s{self.loss_history[i]['step']}", self.loss_history[i]["loss"], self.lr_history[i]["lr"] * 1000]
            for i in range(min_len)
        ]
        table = wandb.Table(columns=["step", "loss", "lr_scaled"], data=data)
        wandb.log({"chart/grouped_bar": wandb.plot.bar(table, "step", "loss", title="Grouped Bar Chart (loss vs lr_scaled)")}, step=step)

    def _log_lollipop_chart(self, step: int):
        if len(self.loss_history) < 5:
            return
        recent = self.loss_history[-20:]
        table = wandb.Table(
            columns=["step", "loss"],
            data=[[r["step"], r["loss"]] for r in recent]
        )
        wandb.log({"chart/lollipop": wandb.plot.scatter(table, "step", "loss", title="Lollipop Chart")}, step=step)

    def _log_dot_plot(self, step: int):
        if len(self.grad_norm_history) < 5:
            return
        recent = self.grad_norm_history[-20:]
        table = wandb.Table(
            columns=["step", "grad_norm"],
            data=[[r["step"], r["grad_norm"]] for r in recent]
        )
        wandb.log({"chart/dot_plot": wandb.plot.scatter(table, "step", "grad_norm", title="Dot Plot")}, step=step)

    def _log_bullet_chart(self, step: int):
        if len(self.loss_history) < 5:
            return
        recent = self.loss_history[-10:]
        final_loss = recent[-1]["loss"]
        min_loss = min(r["loss"] for r in recent)
        max_loss = max(r["loss"] for r in recent)
        table = wandb.Table(
            columns=["metric", "value", "range1", "range2", "range3"],
            data=[["Loss", final_loss, min_loss, (min_loss + max_loss) / 2, max_loss]]
        )
        wandb.log({"chart/bullet": wandb.plot.bar(table, "metric", "value", title="Bullet Chart")}, step=step)

    def _log_dumbbell_chart(self, step: int):
        if len(self.loss_history) < 12:
            return
        recent = self.loss_history[-24:]
        mid = len(recent) // 2
        early = recent[:mid]
        late = recent[mid:]
        data = [
            [f"{early[i]['step']}-{late[i]['step']}", early[i]["loss"], late[i]["loss"]]
            for i in range(min(len(early), len(late)))
        ]
        table = wandb.Table(columns=["range", "start", "end"], data=data)
        wandb.log({"chart/dumbbell": wandb.plot.bar(table, "range", "start", title="Dumbbell Chart (proxy)")}, step=step)

    def _log_range_plot(self, step: int):
        if len(self.memory_history) < 5:
            return
        recent = self.memory_history[-15:]
        table = wandb.Table(
            columns=["step", "min", "max"],
            data=[[f"s{r['step']}", r["allocated"], r["reserved"]] for r in recent]
        )
        wandb.log({"chart/range_plot": wandb.plot.bar(table, "step", "min", title="Range Plot (allocated vs reserved)")}, step=step)

    def _log_radar_chart(self, step: int):
        if len(self.eval_history) < 3:
            return
        recent = self.eval_history[-1]
        keys = [k for k in recent.keys() if k not in ("step", "epoch") and isinstance(recent[k], (int, float))][:8]
        if len(keys) < 3:
            return
        table = wandb.Table(
            columns=["metric", "value"],
            data=[[k, abs(recent[k])] for k in keys]
        )
        wandb.log({"chart/radar": wandb.plot.bar(table, "metric", "value", title="Radar Chart (proxy)")}, step=step)

    # --- CORRELATION CHARTS (23-28) ---

    def _log_scatter_plot(self, step: int):
        if len(self.loss_history) < 5:
            return
        recent = self.loss_history[-30:]
        table = wandb.Table(
            columns=["step", "loss"],
            data=[[r["step"], r["loss"]] for r in recent]
        )
        wandb.log({"chart/scatter_loss": wandb.plot.scatter(table, "step", "loss", title="Scatter Plot (Loss vs Step)")}, step=step)

    def _log_scatter_correlation(self, step: int):
        if len(self.grad_norm_history) < 10 or len(self.loss_history) < 10:
            return
        min_len = min(len(self.grad_norm_history), len(self.loss_history))
        data = [
            [self.grad_norm_history[i]["grad_norm"], self.loss_history[i]["loss"]]
            for i in range(min_len)
        ]
        table = wandb.Table(columns=["grad_norm", "loss"], data=data)
        wandb.log({"chart/scatter_gradnorm_loss": wandb.plot.scatter(table, "grad_norm", "loss", title="Scatter: Grad Norm vs Loss")}, step=step)

    def _log_connected_scatter_plot(self, step: int):
        if len(self.loss_history) < 10:
            return
        recent = self.loss_history[-30:]
        table = wandb.Table(
            columns=["step", "loss"],
            data=[[r["step"], r["loss"]] for r in recent]
        )
        wandb.log({"chart/connected_scatter": wandb.plot.scatter(table, "step", "loss", title="Connected Scatter Plot")}, step=step)

    def _log_bubble_chart(self, step: int):
        if len(self.grad_norm_history) < 10 or len(self.loss_history) < 10:
            return
        min_len = min(len(self.grad_norm_history), len(self.loss_history))
        data = [
            [self.loss_history[i]["step"], self.loss_history[i]["loss"], self.grad_norm_history[i]["grad_norm"] * 100]
            for i in range(min_len)
        ]
        table = wandb.Table(columns=["step", "loss", "size"], data=data)
        wandb.log({"chart/bubble": wandb.plot.scatter(table, "step", "loss", title="Bubble Chart (size=grad_norm)")}, step=step)

    def _log_quadrant_chart(self, step: int):
        if len(self.loss_history) < 10 or len(self.grad_norm_history) < 10:
            return
        min_len = min(len(self.loss_history), len(self.grad_norm_history))
        data = [
            [self.grad_norm_history[i]["grad_norm"], self.loss_history[i]["loss"]]
            for i in range(min_len)
        ]
        table = wandb.Table(columns=["x", "y"], data=data)
        wandb.log({"chart/quadrant": wandb.plot.scatter(table, "x", "y", title="Quadrant Chart")}, step=step)

    def _log_correlation_heatmap(self, step: int):
        if not _wandb_active():
            return
        if len(self.correlation_buffer) < 15:
            return
        try:
            keys = set()
            for entry in self.correlation_buffer:
                keys.update(entry.keys())
            keys.discard("step")
            keys.discard("timestamp")
            keys = sorted(list(keys))[:8]
            if len(keys) < 2:
                return
            n = len(keys)
            matrix = np.zeros((n, n))
            for i, k1 in enumerate(keys):
                for j, k2 in enumerate(keys):
                    vals1 = [e[k1] for e in self.correlation_buffer if k1 in e]
                    vals2 = [e[k2] for e in self.correlation_buffer if k2 in e]
                    if vals1 and vals2 and len(vals1) == len(vals2):
                        c = np.corrcoef(vals1, vals2)[0, 1]
                        matrix[i, j] = float(c) if not np.isnan(c) else 0
            keys_str = [str(k)[:15] for k in keys]
            table = wandb.Table(
                columns=["Metric"] + keys_str,
                data=[[keys_str[i]] + [float(matrix[i, j]) for j in range(n)] for i in range(n)]
            )
            wandb.log({"chart/heatmap_corr": table}, step=step)
        except Exception as e:
            print(f"  [W&B] heatmap failed: {e}")

    # --- DATA OVER TIME (47-58) ---

    def _log_line_chart(self, step: int):
        if len(self.loss_history) < 5 or len(self.grad_norm_history) < 5:
            return
        min_len = min(len(self.loss_history), len(self.grad_norm_history))
        xs = [self.loss_history[i]["step"] for i in range(min_len)]
        ys_loss = [self.loss_history[i]["loss"] for i in range(min_len)]
        ys_gn = [self.grad_norm_history[i]["grad_norm"] for i in range(min_len)]
        wandb.log({"chart/line_multi": wandb.plot.line_series(
            xs, [ys_loss, ys_gn], keys=["loss", "grad_norm"], title="Line Chart (Multi-Series)"
        )}, step=step)

    def _log_area_chart(self, step: int):
        if len(self.loss_history) < 5:
            return
        xs = [r["step"] for r in self.loss_history]
        ys = [[r["loss"] for r in self.loss_history]]
        wandb.log({"chart/area_loss": wandb.plot.line_series(xs, ys, keys=["loss"], title="Area Chart")}, step=step)

    def _log_stacked_area_chart(self, step: int):
        if len(self.loss_history) < 5 or len(self.lr_history) < 5:
            return
        min_len = min(len(self.loss_history), len(self.lr_history))
        xs = [self.loss_history[i]["step"] for i in range(min_len)]
        ys_loss = [self.loss_history[i]["loss"] for i in range(min_len)]
        ys_lr = [self.lr_history[i]["lr"] * 1000 for i in range(min_len)]
        wandb.log({"chart/stacked_area": wandb.plot.line_series(
            xs, [ys_loss, ys_lr], keys=["loss", "lr_x1000"], title="Stacked Area Chart"
        )}, step=step)

    def _log_stream_graph(self, step: int):
        if len(self.loss_history) < 5:
            return
        xs = [r["step"] for r in self.loss_history]
        ys = [[r["loss"] for r in self.loss_history]]
        wandb.log({"chart/stream_graph": wandb.plot.line_series(xs, ys, keys=["loss"], title="Stream Graph (proxy)")}, step=step)

    def _log_bump_chart(self, step: int):
        if len(self.eval_history) < 5:
            return
        sample = self.eval_history[0]
        metric_keys = [k for k in sample.keys() if k not in ("step", "epoch") and isinstance(sample[k], (int, float))][:5]
        if not metric_keys:
            return
        xs = [e["step"] for e in self.eval_history[-20:]]
        ys = []
        for k in metric_keys:
            y_vals = [e.get(k, 0) for e in self.eval_history[-20:]]
            ys.append(y_vals)
        wandb.log({"chart/bump": wandb.plot.line_series(xs, ys, keys=metric_keys, title="Bump Chart (proxy)")}, step=step)

    def _log_step_line_chart(self, step: int):
        if len(self.loss_history) < 5:
            return
        xs = [r["step"] for r in self.loss_history]
        ys = [[r["loss"] for r in self.loss_history]]
        wandb.log({"chart/step_line": wandb.plot.line_series(xs, ys, keys=["loss"], title="Step Line Chart")}, step=step)

    # --- DISTRIBUTION CHARTS (59-68) ---

    def _log_histogram(self, step: int):
        if len(self.loss_history) < 5:
            return
        vals = [r["loss"] for r in self.loss_history[-500:]]
        table = wandb.Table(columns=["loss"], data=[[v] for v in vals])
        wandb.log({"chart/histogram_loss": wandb.plot.histogram(table, "loss", title="Histogram")}, step=step)

    def _log_density_plot(self, step: int):
        if len(self.grad_norm_history) < 5:
            return
        vals = [r["grad_norm"] for r in self.grad_norm_history[-500:]]
        table = wandb.Table(columns=["grad_norm"], data=[[v] for v in vals])
        wandb.log({"chart/density_gradnorm": wandb.plot.histogram(table, "grad_norm", title="Density Plot")}, step=step)

    def _log_box_chart(self, step: int):
        if len(self.loss_history) < 20:
            return
        window = 20
        data = []
        for i in range(0, len(self.loss_history) - window + 1, window):
            window_vals = [self.loss_history[j]["loss"] for j in range(i, i + window)]
            data.append([
                f"W{i//window}",
                float(np.min(window_vals)),
                float(np.percentile(window_vals, 25)),
                float(np.median(window_vals)),
                float(np.percentile(window_vals, 75)),
                float(np.max(window_vals)),
            ])
        if data:
            table = wandb.Table(columns=["window", "min", "q1", "median", "q3", "max"], data=data)
            wandb.log({"chart/box_plot": wandb.plot.bar(table, "window", "median", title="Box Plot (proxy)")}, step=step)

    def _log_strip_plot(self, step: int):
        if len(self.grad_norm_history) < 10:
            return
        recent = self.grad_norm_history[-100:]
        table = wandb.Table(
            columns=["step", "grad_norm"],
            data=[[r["step"], r["grad_norm"]] for r in recent]
        )
        wandb.log({"chart/strip_plot": wandb.plot.scatter(table, "step", "grad_norm", title="Strip Plot")}, step=step)

    def _log_beeswarm(self, step: int):
        if len(self.loss_history) < 20:
            return
        recent = self.loss_history[-100:]
        jitter = list(np.random.rand(len(recent)) * 0.5)
        table = wandb.Table(
            columns=["jitter", "loss"],
            data=[[jitter[i], r["loss"]] for i, r in enumerate(recent)]
        )
        wandb.log({"chart/beeswarm": wandb.plot.scatter(table, "jitter", "loss", title="Beeswarm Chart (proxy)")}, step=step)

    def _log_violin_plot(self, step: int):
        if len(self.loss_history) < 20:
            return
        vals = [r["loss"] for r in self.loss_history[-500:]]
        table = wandb.Table(columns=["loss"], data=[[v] for v in vals])
        wandb.log({"chart/violin_loss": wandb.plot.histogram(table, "loss", title="Violin Plot (proxy)")}, step=step)

    # --- PART-TO-WHOLE CHARTS (29-46) ---

    def _log_pie_donut_chart(self, step: int):
        if len(self.eval_history) < 3:
            return
        recent = self.eval_history[-1]
        keys = [k for k in recent.keys() if k not in ("step", "epoch") and isinstance(recent[k], (int, float))][:6]
        if not keys:
            return
        table = wandb.Table(columns=["label", "value"], data=[[k, abs(recent[k])] for k in keys])
        wandb.log({"chart/pie": wandb.plot.bar(table, "label", "value", title="Pie Chart (proxy)")}, step=step)
        wandb.log({"chart/donut": wandb.plot.bar(table, "label", "value", title="Donut Chart (proxy)")}, step=step)

    def _log_stacked_bar_chart(self, step: int):
        if len(self.loss_history) < 10:
            return
        recent = self.loss_history[-20:]
        mid = len(recent) // 2
        data = [
            [f"s{recent[i]['step']}", recent[i]["loss"],
             recent[i + mid]["loss"] if i + mid < len(recent) else 0]
            for i in range(mid)
        ]
        table = wandb.Table(columns=["step", "loss", "loss2"], data=data)
        wandb.log({"chart/stacked_bar": wandb.plot.bar(table, "step", "loss", title="Stacked Bar Chart (proxy)")}, step=step)

    def _log_waffle_chart(self, step: int):
        if len(self.eval_history) < 3:
            return
        recent = self.eval_history[-1]
        keys = [k for k in recent.keys() if k not in ("step", "epoch") and isinstance(recent[k], (int, float))][:5]
        if not keys:
            return
        total = sum(abs(recent.get(k, 1)) for k in keys) or 1
        data = [[k, abs(recent.get(k, 0)), abs(recent.get(k, 0)) / total * 100] for k in keys]
        table = wandb.Table(columns=["category", "value", "pct"], data=data)
        wandb.log({"chart/waffle": wandb.plot.bar(table, "category", "value", title="Waffle Chart (proxy)")}, step=step)

    def _log_sunburst_chart(self, step: int):
        if len(self.layer_grad_buffer) < 5:
            return
        layer_norms = defaultdict(list)
        for e in self.layer_grad_buffer[-200:]:
            layer_norms[e["layer"]].append(e["norm"])
        avg_norms = {k: float(np.mean(v)) for k, v in layer_norms.items()}
        data = [[k, v] for k, v in avg_norms.items()]
        table = wandb.Table(columns=["layer", "norm"], data=data)
        wandb.log({"chart/sunburst": wandb.plot.bar(table, "layer", "norm", title="Sunburst Chart (proxy)")}, step=step)

    def _log_waterfall_chart(self, step: int):
        if len(self.loss_history) < 10:
            return
        recent = self.loss_history[-20:]
        cumulative = recent[0]["loss"]
        data = [[f"s{r['step']}", r["loss"], r["loss"] - cumulative] for r in recent]
        table = wandb.Table(columns=["step", "cumulative", "delta"], data=data)
        wandb.log({"chart/waterfall": wandb.plot.bar(table, "step", "delta", title="Waterfall Chart (proxy)")}, step=step)

    def _log_funnel_chart(self, step: int):
        if len(self.loss_history) < 10:
            return
        recent = self.loss_history[-20:]
        percentiles = np.percentile([r["loss"] for r in recent], np.linspace(0, 100, 6))
        data = [[f"Q{i}", percentiles[i]] for i in range(len(percentiles))]
        table = wandb.Table(columns=["range", "value"], data=data)
        wandb.log({"chart/funnel": wandb.plot.bar(table, "range", "value", title="Funnel/Pyramid Chart (proxy)")}, step=step)

    def _log_treemap_bar(self, step: int):
        if len(self.layer_grad_buffer) < 5:
            return
        layer_norms = defaultdict(list)
        for e in self.layer_grad_buffer[-200:]:
            layer_norms[e["layer"]].append(e["norm"])
        data = [[k, float(np.mean(v))] for k, v in layer_norms.items()]
        table = wandb.Table(columns=["label", "value"], data=data)
        wandb.log({"chart/treemap": wandb.plot.bar(table, "label", "value", title="Treemap (proxy)")}, step=step)

    def _log_word_cloud_bar(self, step: int):
        if len(self.layer_grad_buffer) < 5:
            return
        layer_norms = defaultdict(list)
        for e in self.layer_grad_buffer[-200:]:
            layer_norms[e["layer"]].append(e["norm"])
        data = [[k[:20], float(np.mean(v))] for k, v in layer_norms.items()]
        table = wandb.Table(columns=["word", "freq"], data=data)
        wandb.log({"chart/wordcloud": wandb.plot.bar(table, "word", "freq", title="Word Cloud (proxy)")}, step=step)

    def _log_nightingale_chart(self, step: int):
        if len(self.eval_history) < 5:
            return
        recent = self.eval_history[-1]
        keys = [k for k in recent.keys() if k not in ("step", "epoch") and isinstance(recent[k], (int, float))][:8]
        if not keys:
            return
        table = wandb.Table(columns=["label", "value"], data=[[k, abs(recent[k])] for k in keys])
        wandb.log({"chart/nightingale": wandb.plot.bar(table, "label", "value", title="Nightingale/Polar Area Chart (proxy)")}, step=step)

    # --- OTHER CHARTS (73-80) ---

    def _log_network_diagram(self, step: int):
        if len(self.layer_grad_buffer) < 10:
            return
        unique_layers = sorted(set(e["layer"] for e in self.layer_grad_buffer[-100:]))
        nodes = [[l, float(np.mean([e["norm"] for e in self.layer_grad_buffer[-100:] if e["layer"] == l]))] for l in unique_layers]
        table = wandb.Table(columns=["node", "weight"], data=nodes)
        wandb.log({"chart/network": wandb.plot.bar(table, "node", "weight", title="Network Diagram (proxy)")}, step=step)

    def _log_sankey_chart(self, step: int):
        if len(self.loss_history) < 20:
            return
        recent = self.loss_history[-40:]
        mid = len(recent) // 2
        data = [
            [f"phase{i//10}", recent[i]["loss"], recent[i + mid]["loss"] if i + mid < len(recent) else 0]
            for i in range(mid)
        ]
        table = wandb.Table(columns=["phase", "early", "late"], data=data)
        wandb.log({"chart/sankey": wandb.plot.bar(table, "phase", "early", title="Sankey Chart (proxy)")}, step=step)

    # --- EVALUATION CHARTS ---

    def _log_eval_charts(self, step: int, metrics: Dict):
        if not metrics:
            return
        try:
            if "eval_loss" in metrics:
                self.correlation_buffer.append({"step": step, "eval_loss": float(metrics["eval_loss"])})
            if "eval_accuracy" in metrics:
                self.correlation_buffer.append({"step": step, "eval_accuracy": float(metrics["eval_accuracy"])})
            if "eval_f1" in metrics:
                self.correlation_buffer.append({"step": step, "eval_f1": float(metrics["eval_f1"])})
        except Exception:
            pass

    def _log_parallel_coordinates(self, step: int):
        if len(self.eval_history) < 10:
            return
        cols = set()
        for e in self.eval_history:
            cols.update(e.keys())
        cols = sorted([c for c in cols if c not in ("step", "epoch")])
        if len(cols) < 2:
            return
        data = [[e.get(c, 0) for c in cols] for e in self.eval_history[-30:]]
        table = wandb.Table(columns=cols, data=data)
        wandb.log({"chart/parallel_coords": wandb.plot.scatter(
            table, cols[0], cols[1] if len(cols) > 1 else cols[0],
            title="Parallel Coordinates (proxy)"
        )}, step=step)

    def _log_confusion_matrix(self, predictions, labels, step: int):
        if predictions is None or labels is None:
            return
        try:
            if hasattr(predictions, "argmax"):
                predictions = predictions.argmax(-1)
            preds = predictions.flatten().tolist()
            labs = labels.flatten().tolist()
            n_classes = max(set(labs + preds), default=0) + 1
            wandb.log({
                "chart/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None, y_true=labs, preds=preds,
                    class_names=[str(i) for i in range(n_classes)]
                )
            }, step=step)
        except Exception:
            pass

    # --- SUMMARY CHARTS ---

    def _log_summary_charts(self):
        if not _wandb_active():
            return
        try:
            step = self.eval_history[-1]["step"] if self.eval_history else 0

            if len(self.loss_history) >= 5:
                vals = [r["loss"] for r in self.loss_history[-500:]]
                table = wandb.Table(columns=["loss"], data=[[v] for v in vals])
                wandb.log({"summary/histogram_loss": wandb.plot.histogram(table, "loss", title="Histogram: All Loss Values")}, step=step)

            if len(self.grad_norm_history) >= 5:
                vals = [r["grad_norm"] for r in self.grad_norm_history[-500:]]
                table = wandb.Table(columns=["grad_norm"], data=[[v] for v in vals])
                wandb.log({"summary/histogram_gradnorm": wandb.plot.histogram(table, "grad_norm", title="Histogram: All Grad Norms")}, step=step)

            if len(self.layer_grad_buffer) >= 5:
                unique_layers = sorted(set(e["layer"] for e in self.layer_grad_buffer))
                layer_data = [[l, float(np.mean([e["norm"] for e in self.layer_grad_buffer if e["layer"] == l]))] for l in unique_layers]
                table = wandb.Table(columns=["layer", "avg_norm"], data=layer_data)
                wandb.log({"summary/layer_bar": wandb.plot.bar(table, "layer", "avg_norm", title="Layer Norms Bar Chart")}, step=step)

            if self.eval_history:
                eval_keys = [k for k in self.eval_history[0].keys() if k not in ("step", "epoch")]
                summary_data = [[k, float(np.mean([e.get(k, 0) for e in self.eval_history]))] for k in eval_keys]
                table = wandb.Table(columns=["metric", "mean"], data=summary_data)
                wandb.log({"summary/eval_bar": wandb.plot.bar(table, "metric", "mean", title="Eval Metrics Summary")}, step=step)

            if len(self.loss_history) >= 5:
                table = wandb.Table(columns=["step", "loss"], data=[[r["step"], r["loss"]] for r in self.loss_history])
                wandb.log({"summary/line_final": wandb.plot.scatter(table, "step", "loss", title="Final Loss Trajectory")}, step=step)

        except Exception as e:
            print(f"  [W&B] summary charts failed: {e}")

    def _save_data(self):
        data_dir = os.path.join(self.output_dir, "all_data")
        os.makedirs(data_dir, exist_ok=True)
        for name, buffer in [
            ("loss_history", self.loss_history),
            ("grad_norm_history", self.grad_norm_history),
            ("lr_history", self.lr_history),
            ("memory_history", self.memory_history),
            ("eval_history", self.eval_history),
            ("correlation_buffer", self.correlation_buffer),
            ("layer_grad_buffer", self.layer_grad_buffer),
        ]:
            if buffer:
                with open(os.path.join(data_dir, f"{name}.json"), "w") as f:
                    json.dump(buffer, f, indent=2, default=str)