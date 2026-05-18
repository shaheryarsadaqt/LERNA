"""
Comprehensive Metrics Logging Callback for W&B

Logs all possible metrics in every W&B-supported format:
- Scalars (line charts)
- Histograms
- Bar charts
- Heatmaps / Correlation matrices
- Scatter plots
- Area charts
- Box plots / Violin plots
- Tables
- Distribution plots
- Scatter matrices
- Parallel coordinates
- Confusion matrices (for eval)
- Media (images, audio)
- Custom visualizations
"""

import os
import json
import time
import numpy as np
from typing import Dict, List, Optional, Any
from collections import defaultdict
from datetime import datetime

try:
    import wandb
except ImportError:
    wandb = None

import torch
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments


def _wandb_active() -> bool:
    return wandb is not None and getattr(wandb, "run", None) is not None


def _safe(val, default=None):
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


class ComprehensiveMetricsCallback(TrainerCallback):
    """
    Comprehensive metrics logging to W&B with all visualization types.

    Logs:
    - Training dynamics (loss, learning rate, steps) - LINE CHART
    - Gradient statistics (norms, distributions) - LINE + HISTOGRAM
    - Memory and compute metrics - LINE + AREA
    - Skip decisions and policy diagnostics - BAR CHART + LINE
    - Energy consumption - LINE + AREA
    - Correlation heatmaps - HEATMAP
    - Distribution histograms - HISTOGRAM + BOX PLOT
    - Bar charts for comparisons - BAR CHART
    - Scatter plots (e.g., loss vs grad_norm) - SCATTER
    - Custom tables - TABLE
    - Parallel coordinates - PARALLEL COORDS
    - Confusion matrices - CONFUSION MATRIX
    - Parallel sets / alluvial - custom TABLE
    """

    def __init__(
        self,
        log_frequency: int = 10,
        histogram_frequency: int = 100,
        table_frequency: int = 500,
        scatter_frequency: int = 50,
        wandb_enabled: bool = True,
        output_dir: str = "./experiments/comprehensive_logs",
    ):
        self.log_frequency = log_frequency
        self.histogram_frequency = histogram_frequency
        self.table_frequency = table_frequency
        self.scatter_frequency = scatter_frequency
        self.wandb_enabled = wandb_enabled
        self.output_dir = output_dir

        os.makedirs(output_dir, exist_ok=True)

        self.step_count = 0
        self.train_start_time = None
        self.step_times = []
        self.loss_history = []
        self.grad_norm_history = []
        self.skip_history = []
        self.lr_history = []
        self.memory_history = []
        self.energy_history = []
        self.eval_history = []
        self.custom_table_data = []
        self.histogram_data = defaultdict(list)
        self.correlation_buffer = []
        self.scatter_buffer = []

        self._last_log_time = 0
        self._last_histogram_time = 0
        self._last_table_time = 0
        self._last_scatter_time = 0

        self._confusion_matrix_buffer = []
        self._parallel_coords_buffer = []

        print(f"  Comprehensive Metrics Callback initialized")
        print(f"  Output directory: {output_dir}")

    def on_train_begin(self, args, state, control, **kwargs):
        self.train_start_time = time.time()
        self.step_count = 0
        print(f"  Training started at {datetime.now().isoformat()}")

        if _wandb_active():
            try:
                wandb.define_metric("train/global_step")
                wandb.define_metric("eval/global_step")
                wandb.define_metric("power/*", step_metric="train/global_step")
                wandb.define_metric("gradients/*", step_metric="train/global_step")
                wandb.define_metric("memory/*", step_metric="train/global_step")
                wandb.define_metric("compute/*", step_metric="train/global_step")
                wandb.define_metric("skip/*", step_metric="train/global_step")
                wandb.define_metric("efficiency/*", step_metric="train/global_step")
                wandb.define_metric("scatter/*", step_metric="train/global_step")
            except Exception as e:
                print(f"  [W&B] define_metric failed: {e}")
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self._step_start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if hasattr(self, "_step_start_time"):
            step_time = time.time() - self._step_start_time
            self.step_times.append(step_time)

        self.step_count += 1
        step = state.global_step

        metrics = {}

        if state.log_history:
            last_log = state.log_history[-1] if state.log_history else {}
            loss = last_log.get("loss")
            if loss is not None:
                metrics["train/loss"] = float(loss)
                self.loss_history.append({"step": step, "value": float(loss)})

            lr = last_log.get("learning_rate") or last_log.get("lr")
            if lr is not None:
                metrics["train/learning_rate"] = float(lr)
                self.lr_history.append({"step": step, "value": float(lr)})

        grad_norm = getattr(state, "_grad_norm", None)
        if grad_norm is not None:
            gn = float(grad_norm)
            metrics["gradients/norm"] = gn
            self.grad_norm_history.append({"step": step, "value": gn})

            if loss is not None:
                self.scatter_buffer.append({
                    "step": step,
                    "loss": float(loss),
                    "grad_norm": gn,
                    "lr": float(last_log.get("learning_rate", 0)) if last_log.get("learning_rate") else None,
                })

        if torch.cuda.is_available():
            try:
                alloc_gb = torch.cuda.memory_allocated() / 1e9
                res_gb = torch.cuda.memory_reserved() / 1e9
                max_gb = torch.cuda.max_memory_allocated() / 1e9
                metrics["memory/allocated_gb"] = alloc_gb
                metrics["memory/reserved_gb"] = res_gb
                metrics["memory/max_allocated_gb"] = max_gb
                self.memory_history.append({
                    "step": step, "allocated": alloc_gb,
                    "reserved": res_gb, "max": max_gb,
                })
            except Exception:
                pass

        if self.step_times:
            recent = self.step_times[-50:]
            avg_step_time = float(np.mean(recent))
            metrics["compute/avg_step_time_ms"] = avg_step_time * 1000
            metrics["compute/steps_per_second"] = 1.0 / avg_step_time if avg_step_time > 0 else 0

        if self.train_start_time:
            elapsed = time.time() - self.train_start_time
            metrics["compute/elapsed_time_s"] = elapsed
            metrics["compute/elapsed_time_min"] = elapsed / 60

        skip_decisions = getattr(state, "_skip_decisions", None)
        if skip_decisions is not None:
            metrics["skip/count"] = skip_decisions
            self.skip_history.append({"step": step, "count": skip_decisions})

        if step % self.log_frequency == 0 and metrics:
            self._log_scalars(metrics, step)

        if step % self.histogram_frequency == 0:
            self._log_histograms(step, kwargs)
            self._log_box_plots(step)

        if step % self.table_frequency == 0:
            self._log_tables(step)

        if step % self.scatter_frequency == 0:
            self._log_scatter_plots(step)

        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}
        step = state.global_step

        eval_metrics = {
            "eval/global_step": float(step),
            "eval/epoch": float(state.epoch) if state.epoch else 0,
        }

        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                eval_metrics[f"eval/{key}"] = float(value)

        if "eval_loss" in metrics:
            self._log_to_correlation_buffer("eval_loss", metrics["eval_loss"], step)
            self.eval_history.append({"step": step, "loss": float(metrics["eval_loss"])})

        if "eval_accuracy" in metrics:
            self.eval_history[-1]["accuracy"] = float(metrics["eval_accuracy"])
            self._log_to_correlation_buffer("eval_accuracy", float(metrics["eval_accuracy"]), step)

        predictions = kwargs.get("predictions")
        labels = kwargs.get("labels")
        if predictions is not None and labels is not None:
            self._log_confusion_matrix(predictions, labels, step)

        if eval_metrics:
            self._log_scalars(eval_metrics, step, prefix="eval")

        self._log_eval_summary(metrics, step)
        self._log_parallel_coordinates(step, metrics)

        return control

    def on_train_end(self, args, state, control, **kwargs):
        total_time = time.time() - self.train_start_time if self.train_start_time else 0

        self._save_all_histories()

        summary = {
            "total_steps": self.step_count,
            "total_training_time_s": total_time,
            "total_training_time_min": total_time / 60,
            "avg_step_time_ms": float(np.mean(self.step_times) * 1000) if self.step_times else 0,
            "total_loss_samples": len(self.loss_history),
            "total_grad_norm_samples": len(self.grad_norm_history),
            "total_skip_samples": len(self.skip_history),
        }

        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        self._log_summary_metrics(summary)

        if _wandb_active():
            self._log_correlation_heatmap()
            self._log_distribution_summary()
            self._log_bar_chart_comparison()
            self._log_scatter_matrix()
            self._log_wide_table()
            self._log_multi_series_line_chart()
            self._log_alluvial_sankey()

        print(f"\n{'=' * 60}")
        print("COMPREHENSIVE METRICS - TRAINING COMPLETE")
        print(f"{'=' * 60}")
        print(f"  Total steps: {self.step_count}")
        print(f"  Training time: {total_time:.1f}s ({total_time/60:.1f}min)")
        print(f"  Avg step time: {summary['avg_step_time_ms']:.1f}ms")
        print(f"  Summary saved: {summary_path}")
        print(f"{'=' * 60}")

        return control

    def _log_scalars(self, metrics: Dict, step: int, prefix: str = "train"):
        if not _wandb_active():
            return
        try:
            wandb.log(metrics, step=step)
        except Exception as e:
            print(f"  [W&B] scalar log failed: {e}")

    def _log_histograms(self, step: int, kwargs):
        if not _wandb_active():
            return
        try:
            histograms = {}

            model = kwargs.get("model")
            if model is not None:
                grad_norms = []
                param_norms = []
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_data = param.grad.detach().float().cpu().numpy().flatten()
                        grad_norms.append(float(np.linalg.norm(grad_data)))
                        self.histogram_data[f"grad/{name}"].append(grad_data)

                    param_data = param.detach().float().cpu().numpy().flatten()
                    param_norms.append(float(np.linalg.norm(param_data)))
                    self.histogram_data[f"param/{name}"].append(param_data)

                if grad_norms:
                    histograms["histogram/grad_norms"] = wandb.Histogram(grad_norms)
                if param_norms:
                    histograms["histogram/param_norms"] = wandb.Histogram(param_norms)

            if self.loss_history:
                recent_losses = [x["value"] for x in self.loss_history[-100:]]
                histograms["histogram/loss_distribution"] = wandb.Histogram(recent_losses)

            if histograms:
                wandb.log(histograms, step=step)

        except Exception as e:
            print(f"  [W&B] histogram log failed: {e}")

    def _log_box_plots(self, step: int):
        if not _wandb_active():
            return
        try:
            if len(self.loss_history) < 10:
                return

            window_size = 20
            box_data = []
            for i in range(0, len(self.loss_history), window_size):
                window = self.loss_history[i:i+window_size]
                if len(window) >= 5:
                    box_data.append({
                        "name": f"step_{window[0]['step']}",
                        "values": [x["value"] for x in window],
                    })

            if len(box_data) >= 2:
                table = wandb.Table(
                    columns=["window", "min", "q1", "median", "q3", "max"],
                    data=[
                        [
                            d["name"],
                            float(np.min(d["values"])),
                            float(np.percentile(d["values"], 25)),
                            float(np.median(d["values"])),
                            float(np.percentile(d["values"], 75)),
                            float(np.max(d["values"])),
                        ]
                        for d in box_data
                    ]
                )
                wandb.log({"box/loss_windows": table}, step=step)

        except Exception as e:
            print(f"  [W&B] box plot log failed: {e}")

    def _log_tables(self, step: int):
        if not _wandb_active():
            return
        try:
            if self.loss_history:
                recent = self.loss_history[-50:]
                table = wandb.Table(
                    columns=["step", "loss"],
                    data=[[r["step"], r["value"]] for r in recent]
                )
                wandb.log({"table/loss_history": table}, step=step)

            if self.grad_norm_history:
                recent = self.grad_norm_history[-50:]
                table = wandb.Table(
                    columns=["step", "grad_norm"],
                    data=[[r["step"], r["value"]] for r in recent]
                )
                wandb.log({"table/grad_norm_history": table}, step=step)

            if self.memory_history:
                recent = self.memory_history[-20:]
                table = wandb.Table(
                    columns=["step", "allocated_gb", "reserved_gb", "max_gb"],
                    data=[[r["step"], r["allocated"], r["reserved"], r["max"]] for r in recent]
                )
                wandb.log({"table/memory_usage": table}, step=step)

            if self.eval_history:
                recent = self.eval_history[-20:]
                cols = ["step"]
                sample = recent[0] if recent else {}
                cols.extend([k for k in sample.keys() if k != "step"])
                data = [[r.get(c) for c in cols] for r in recent]
                wandb.log({"table/eval_history": wandb.Table(columns=cols, data=data)}, step=step)

        except Exception as e:
            print(f"  [W&B] table log failed: {e}")

    def _log_scatter_plots(self, step: int):
        if not _wandb_active():
            return
        try:
            if len(self.scatter_buffer) < 5:
                return

            recent = self.scatter_buffer[-200:]

            loss_vals = [x["loss"] for x in recent]
            grad_norm_vals = [x["grad_norm"] for x in recent]
            step_vals = [x["step"] for x in recent]

            loss_vs_grad = wandb.plot.scatter(
                wandb.Table(
                    columns=["step", "loss", "grad_norm"],
                    data=[[s, l, g] for s, l, g in zip(step_vals, loss_vals, grad_norm_vals)]
                ),
                x="step",
                y="loss",
                title="Loss vs Step (Scatter)"
            )
            wandb.log({"scatter/loss_vs_step": loss_vs_grad}, step=step)

            grad_vs_step = wandb.plot.scatter(
                wandb.Table(
                    columns=["step", "grad_norm"],
                    data=[[s, g] for s, g in zip(step_vals, grad_norm_vals)]
                ),
                x="step",
                y="grad_norm",
                title="Gradient Norm vs Step (Scatter)"
            )
            wandb.log({"scatter/grad_norm_vs_step": grad_vs_step}, step=step)

            if len(recent) > 10:
                loss_vs_gn = wandb.plot.scatter(
                    wandb.Table(
                        columns=["grad_norm", "loss"],
                        data=[[g, l] for g, l in zip(grad_norm_vals, loss_vals)]
                    ),
                    x="grad_norm",
                    y="loss",
                    title="Loss vs Gradient Norm (Scatter)"
                )
                wandb.log({"scatter/loss_vs_grad_norm": loss_vs_gn}, step=step)

        except Exception as e:
            print(f"  [W&B] scatter plot log failed: {e}")

    def _log_eval_summary(self, metrics: Dict, step: int):
        if not _wandb_active():
            return
        try:
            eval_summary = {}

            if "eval_loss" in metrics:
                eval_summary["eval_summary/loss"] = float(metrics["eval_loss"])
            if "eval_accuracy" in metrics:
                eval_summary["eval_summary/accuracy"] = float(metrics["eval_accuracy"])
            if "eval_matthews_correlation" in metrics:
                eval_summary["eval_summary/matthews_correlation"] = float(metrics["eval_matthews_correlation"])
            if "eval_pearson" in metrics:
                eval_summary["eval_summary/pearson"] = float(metrics["eval_pearson"])
            if "eval_f1" in metrics:
                eval_summary["eval_summary/f1"] = float(metrics["eval_f1"])

            if eval_summary:
                wandb.log(eval_summary, step=step)

        except Exception as e:
            print(f"  [W&B] eval summary log failed: {e}")

    def _log_to_correlation_buffer(self, metric_name: str, value: float, step: int):
        self.correlation_buffer.append({
            "step": step,
            metric_name: value,
            "timestamp": time.time(),
        })
        if len(self.correlation_buffer) > 1000:
            self.correlation_buffer = self.correlation_buffer[-500:]

    def _log_confusion_matrix(self, predictions, labels, step: int):
        if not _wandb_active():
            return
        try:
            if predictions is None or labels is None:
                return

            if hasattr(predictions, 'argmax'):
                predictions = predictions.argmax(-1)
            if hasattr(predictions, 'flatten'):
                predictions = predictions.flatten()
            if hasattr(labels, 'flatten'):
                labels = labels.flatten()

            unique_labels = sorted(set(int(x) for x in labels))
            n_classes = max(unique_labels) + 1 if unique_labels else 2
            n_classes = max(n_classes, 2)

            cm = np.zeros((n_classes, n_classes), dtype=np.int32)
            for p, l in zip(predictions.flatten(), labels.flatten()):
                if 0 <= int(p) < n_classes and 0 <= int(l) < n_classes:
                    cm[int(l), int(p)] += 1

            wandb.log({
                "confusion_matrix/eval": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=labels.flatten().tolist(),
                    preds=predictions.flatten().tolist(),
                    class_names=[str(i) for i in range(n_classes)],
                )
            }, step=step)

        except Exception as e:
            print(f"  [W&B] confusion matrix failed: {e}")

    def _log_parallel_coordinates(self, step: int, metrics: Dict):
        if not _wandb_active():
            return
        try:
            if not metrics:
                return

            pc_data = {
                "step": float(step),
            }
            for key, val in metrics.items():
                if isinstance(val, (int, float)):
                    pc_data[key] = float(val)

            if len(pc_data) >= 2:
                self._parallel_coords_buffer.append(pc_data)

            if len(self._parallel_coords_buffer) % 10 == 0 and len(self._parallel_coords_buffer) >= 20:
                recent = self._parallel_coords_buffer[-100:]
                keys = list(recent[0].keys())

                table_data = []
                for entry in recent:
                    row = [entry.get(k) for k in keys]
                    table_data.append(row)

                table = wandb.Table(columns=keys, data=table_data)
                wandb.log({"parallel_coords/training_trajectory": table}, step=step)

        except Exception as e:
            print(f"  [W&B] parallel coordinates failed: {e}")

    def _log_correlation_heatmap(self):
        if not _wandb_active() or len(self.correlation_buffer) < 10:
            return
        try:
            keys = set()
            for entry in self.correlation_buffer:
                keys.update(entry.keys())
            keys.discard("step")
            keys.discard("timestamp")
            keys = sorted(keys)

            if len(keys) < 2:
                return

            matrix = np.zeros((len(keys), len(keys)))
            values_matrix = {k: [] for k in keys}
            for entry in self.correlation_buffer:
                for k in keys:
                    if k in entry:
                        values_matrix[k].append(entry[k])

            for i, k1 in enumerate(keys):
                for j, k2 in enumerate(keys):
                    if values_matrix[k1] and values_matrix[k2]:
                        if len(values_matrix[k1]) == len(values_matrix[k2]):
                            corr = np.corrcoef(values_matrix[k1], values_matrix[k2])[0, 1]
                            matrix[i, j] = float(corr) if not np.isnan(corr) else 0

            keys_str = [str(k) for k in keys]

            wandb.log({
                "heatmap/correlation": wandb.plots.Heatmap(
                    matrix.tolist() if hasattr(matrix, 'tolist') else matrix,
                    keys_str,
                    keys_str,
                    title="Metric Correlation Heatmap"
                )
            })

        except Exception as e:
            print(f"  [W&B] correlation heatmap failed: {e}")

    def _log_distribution_summary(self):
        if not _wandb_active():
            return
        try:
            if self.loss_history:
                values = [x["value"] for x in self.loss_history]
                wandb.log({
                    "distribution/loss": wandb.Histogram(values),
                    "distribution/loss_summary": {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                        "p25": float(np.percentile(values, 25)),
                        "p75": float(np.percentile(values, 75)),
                    }
                })

            if self.grad_norm_history:
                values = [x["value"] for x in self.grad_norm_history]
                wandb.log({
                    "distribution/grad_norm": wandb.Histogram(values),
                    "distribution/grad_norm_summary": {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "median": float(np.median(values)),
                    }
                })

            if self.memory_history:
                alloc_vals = [x["allocated"] for x in self.memory_history]
                wandb.log({
                    "distribution/memory_allocated": wandb.Histogram(alloc_vals),
                })

        except Exception as e:
            print(f"  [W&B] distribution summary failed: {e}")

    def _log_bar_chart_comparison(self):
        if not _wandb_active():
            return
        try:
            bar_data = []

            if self.loss_history:
                bar_data.append({"name": "Final Loss", "value": self.loss_history[-1]["value"]})
                bar_data.append({"name": "Min Loss", "value": min(x["value"] for x in self.loss_history)})
                bar_data.append({"name": "Max Loss", "value": max(x["value"] for x in self.loss_history)})

            if self.grad_norm_history:
                bar_data.append({"name": "Final Grad Norm", "value": self.grad_norm_history[-1]["value"]})
                bar_data.append({"name": "Avg Grad Norm", "value": float(np.mean([x["value"] for x in self.grad_norm_history]))})

            if self.step_times:
                bar_data.append({"name": "Avg Step Time (ms)", "value": float(np.mean(self.step_times) * 1000)})

            if bar_data:
                table = wandb.Table(
                    columns=["Metric", "Value"],
                    data=[[d["name"], d["value"]] for d in bar_data]
                )
                wandb.log({"bar/metrics_comparison": wandb.plot.bar(table, "Metric", "Value", title="Metrics Comparison")})

        except Exception as e:
            print(f"  [W&B] bar chart failed: {e}")

    def _log_scatter_matrix(self):
        if not _wandb_active() or len(self.scatter_buffer) < 10:
            return
        try:
            recent = self.scatter_buffer[-500:]
            if len(recent) < 10:
                return

            all_keys = set()
            for entry in recent:
                all_keys.update(entry.keys())
            all_keys.discard("step")
            all_keys = sorted(all_keys)

            if len(all_keys) < 2:
                return

            cols = ["step"] + list(all_keys)
            data = []
            for entry in recent:
                row = [entry.get(c) for c in cols]
                data.append(row)

            table = wandb.Table(columns=cols, data=data)

            if hasattr(wandb, 'plot') and hasattr(wandb.plot, 'scatter'):
                scatter_matrix = wandb.plot.scatter(
                    table,
                    x="step",
                    y="loss",
                    title="Scatter Matrix: Key Metrics"
                )
                wandb.log({"scatter/matrix": scatter_matrix})

        except Exception as e:
            print(f"  [W&B] scatter matrix failed: {e}")

    def _log_wide_table(self):
        if not _wandb_active():
            return
        try:
            if not self.eval_history:
                return

            cols = set()
            for entry in self.eval_history:
                cols.update(entry.keys())
            cols = sorted(cols)

            table_data = []
            for entry in self.eval_history:
                row = [entry.get(c) for c in cols]
                table_data.append(row)

            table = wandb.Table(columns=cols, data=table_data)
            wandb.log({"table/wide_eval": table})

        except Exception as e:
            print(f"  [W&B] wide table log failed: {e}")

    def _log_multi_series_line_chart(self):
        if not _wandb_active():
            return
        try:
            if len(self.loss_history) < 10:
                return

            steps = [x["step"] for x in self.loss_history]
            loss_vals = [x["value"] for x in self.loss_history]

            table = wandb.Table(
                columns=["step", "loss", "grad_norm"],
                data=[
                    [s, l, g["value"]]
                    for (s, l), g in zip(
                        [(x["step"], x["value"]) for x in self.loss_history],
                        self.grad_norm_history[:len(self.loss_history)]
                    )
                ]
            )

            line_chart = wandb.plot.line_series(
                table,
                x="step",
                keys=["loss", "grad_norm"],
                title="Multi-Metric Time Series",
                xname="step"
            )
            wandb.log({"series/multi_metric": line_chart})

        except Exception as e:
            print(f"  [W&B] multi-series line chart failed: {e}")

    def _log_alluvial_sankey(self):
        if not _wandb_active():
            return
        try:
            if not self.skip_history or len(self.skip_history) < 20:
                return

            bins = [0, 0.25, 0.5, 0.75, 1.0]
            labels = ["0-25%", "25-50%", "50-75%", "75-100%"]

            skip_ratios = [s["count"] / max(1, s["step"]) for s in self.skip_history[-100:]]
            if not skip_ratios:
                return

            binned = np.digitize(skip_ratios, bins)
            unique, counts = np.unique(binned, return_counts=True)

            table_data = []
            for u, c in zip(unique, counts):
                if u < len(labels):
                    table_data.append([labels[u], int(c)])

            if table_data:
                table = wandb.Table(
                    columns=["skip_ratio_range", "count"],
                    data=table_data
                )
                wandb.log({"sankey/skip_distribution": wandb.plot.bar(table, "skip_ratio_range", "count", title="Skip Ratio Distribution (Bar/Sankey proxy)")})

        except Exception as e:
            print(f"  [W&B] alluvial/sankey failed: {e}")

    def _log_summary_metrics(self, summary: Dict):
        if not _wandb_active():
            return
        try:
            wandb.log({
                "summary/total_steps": summary["total_steps"],
                "summary/training_time_min": summary["total_training_time_min"],
                "summary/avg_step_time_ms": summary["avg_step_time_ms"],
            })
        except Exception as e:
            print(f"  [W&B] summary log failed: {e}")

    def _save_all_histories(self):
        histories = {
            "loss_history": self.loss_history,
            "grad_norm_history": self.grad_norm_history,
            "skip_history": self.skip_history,
            "lr_history": self.lr_history,
            "memory_history": self.memory_history,
            "step_times": self.step_times,
            "correlation_buffer": self.correlation_buffer,
            "scatter_buffer": self.scatter_buffer,
            "eval_history": self.eval_history,
            "parallel_coords": self._parallel_coords_buffer,
        }

        for name, data in histories.items():
            if data:
                path = os.path.join(self.output_dir, f"{name}.json")
                with open(path, "w") as f:
                    json.dump(data, f, indent=2, default=str)


class GradNormDetailedCallback(TrainerCallback):
    """
    Detailed gradient norm logging with full W&B visualization support.
    """

    def __init__(self, wandb_enabled: bool = True, log_frequency: int = 10):
        self.wandb_enabled = wandb_enabled
        self.log_frequency = log_frequency
        self.grad_norms = []
        self.param_grad_stats = []
        self.layer_grad_stats = []
        self.layer_comparison_buffer = []

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None:
            return control

        step = state.global_step
        if step % self.log_frequency != 0:
            return control

        grad_norms = []
        layer_stats = {}
        total_sq = 0.0

        for name, param in model.named_parameters():
            if param.grad is not None:
                grad = param.grad.detach().float()
                norm = float(grad.norm().item())
                grad_norms.append(norm)
                total_sq += norm ** 2

                layer_name = ".".join(name.split(".")[:2])
                if layer_name not in layer_stats:
                    layer_stats[layer_name] = {"norms": [], "count": 0}
                layer_stats[layer_name]["norms"].append(norm)
                layer_stats[layer_name]["count"] += 1

        total_norm = total_sq ** 0.5
        self.grad_norms.append({"step": step, "total_norm": total_norm})

        for layer, stats in layer_stats.items():
            if stats["norms"]:
                layer_norm = (sum(x**2 for x in stats["norms"]) / stats["count"]) ** 0.5
                self.layer_grad_stats.append({
                    "step": step,
                    "layer": layer,
                    "norm": layer_norm,
                    "count": stats["count"],
                })
                self.layer_comparison_buffer.append({
                    "step": step,
                    "layer": layer,
                    "norm": layer_norm,
                })

        if _wandb_active():
            try:
                wandb.log({
                    "grad_norm/detailed/total": total_norm,
                    "grad_norm/detailed/mean": float(np.mean(grad_norms)) if grad_norms else 0,
                    "grad_norm/detailed/std": float(np.std(grad_norms)) if grad_norms else 0,
                    "grad_norm/detailed/max": float(np.max(grad_norms)) if grad_norms else 0,
                    "grad_norm/detailed/min": float(np.min(grad_norms)) if grad_norms else 0,
                    "grad_norm/detailed/median": float(np.median(grad_norms)) if grad_norms else 0,
                    "grad_norm/detailed/q25": float(np.percentile(grad_norms, 25)) if grad_norms else 0,
                    "grad_norm/detailed/q75": float(np.percentile(grad_norms, 75)) if grad_norms else 0,
                }, step=step)

                wandb.log({
                    "grad_norm/histogram": wandb.Histogram(grad_norms),
                }, step=step)

            except Exception as e:
                print(f"  [W&B] grad_norm detailed failed: {e}")

        return control

    def on_train_end(self, args, state, control, **kwargs):
        if _wandb_active():
            try:
                if self.grad_norms:
                    table = wandb.Table(
                        columns=["step", "total_norm"],
                        data=[[g["step"], g["total_norm"]] for g in self.grad_norms]
                    )
                    wandb.log({"grad_norm/table": table})

                if self.layer_grad_stats:
                    table = wandb.Table(
                        columns=["step", "layer", "norm", "count"],
                        data=[[s["step"], s["layer"], s["norm"], s["count"]] for s in self.layer_grad_stats]
                    )
                    wandb.log({"grad_norm/layer_table": table})

                if len(self.layer_comparison_buffer) > 10:
                    unique_layers = sorted(set(x["layer"] for x in self.layer_comparison_buffer))
                    if len(unique_layers) > 1:
                        layer_data = {}
                        for entry in self.layer_comparison_buffer[-200:]:
                            layer = entry["layer"]
                            if layer not in layer_data:
                                layer_data[layer] = {"steps": [], "norms": []}
                            layer_data[layer]["steps"].append(entry["step"])
                            layer_data[layer]["norms"].append(entry["norm"])

                        recent = self.layer_comparison_buffer[-50:]
                        scatter_data = [[e["step"], e["layer"], e["norm"]] for e in recent]
                        scatter_table = wandb.Table(
                            columns=["step", "layer", "norm"],
                            data=scatter_data
                        )
                        scatter_plot = wandb.plot.scatter(
                            scatter_table,
                            x="step",
                            y="norm",
                            title="Layer Gradient Norms Over Time"
                        )
                        wandb.log({"grad_norm/layer_scatter": scatter_plot}, step=state.global_step)

            except Exception as e:
                print(f"  [W&B] grad_norm train_end failed: {e}")

        return control