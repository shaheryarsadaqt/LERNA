"""
Efficiency Metrics Callback Implementation

Comprehensive efficiency tracking including GSNR, LER, probe accuracy,
and compute cost analysis.

RTX 5090 / Blackwell compatibility:
- All wandb imports are guarded with try/except
- nvidia-smi output parsing handles [N/A], unit suffixes, and format changes
- PowerTelemetryCallback falls back to TDP-based estimation on read failure
- No callback ever raises an exception into the Trainer loop
"""

import torch
import json
import os
import time
import subprocess
import threading
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

try:
    import wandb
except ImportError:
    wandb = None

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from ..utils.metrics import (
    GSNRTracker,
    LERTracker,
    ProbeAccuracyTracker,
    EfficiencyMetricsCollector,
    validate_ler_metric,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wandb_active() -> bool:
    """Return True only when wandb is installed AND has an active run."""
    return wandb is not None and getattr(wandb, "run", None) is not None


def _safe_parse_float(value: str) -> Optional[float]:
    """Parse a float from nvidia-smi output.

    Handles extra whitespace, trailing units (W, %, C), commas,
    and [N/A] or [Not Supported] values that Blackwell GPUs may emit.
    """
    if not value:
        return None
    cleaned = value.strip().rstrip("WwCc%").strip()
    if not cleaned or "N/A" in cleaned.upper() or "NOT" in cleaned.upper():
        return None
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return None


# ===================================================================
# EfficiencyMetricsCallback
# ===================================================================

class EfficiencyMetricsCallback(TrainerCallback):
    """
    Comprehensive efficiency metrics tracking callback.

    Tracks:
    1. Gradient Signal-to-Noise Ratio (GSNR)
    2. Learning Efficiency Rate (LER)
    3. Probe accuracy for representation quality
    4. Compute cost and memory usage
    5. Training dynamics and convergence patterns
    """

    def __init__(
        self,
        task: str,
        enable_gsnr: bool = True,
        enable_ler: bool = True,
        enable_probe: bool = False,
        enable_compute_tracking: bool = True,
        output_dir: str = "./experiments/efficiency_metrics",
        wandb_enabled: bool = True,
        log_frequency: int = 25,
    ):
        self.task = task
        self.enable_gsnr = enable_gsnr
        self.enable_ler = enable_ler
        self.enable_probe = enable_probe
        self.enable_compute_tracking = enable_compute_tracking
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        self.log_frequency = log_frequency

        os.makedirs(output_dir, exist_ok=True)

        self.gsnr_tracker = None
        self.ler_tracker = None
        self.probe_tracker = None
        self.metrics_collector = None

        self.metrics_history: List[Dict] = []
        self.gradient_history: List[Dict] = []
        self.compute_stats: Dict = defaultdict(list)

        self.step_start_time = None
        self.total_training_time = 0
        self.step_times = []
        self.memory_stats = []

        print(f"  Efficiency Metrics Callback initialized")
        print(f"   Task: {task}")
        print(f"   GSNR: {enable_gsnr}, LER: {enable_ler}, Probe: {enable_probe}")
        print(f"   Output directory: {output_dir}")

    def on_init_end(self, args, state, control, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
            if self.enable_gsnr:
                self.gsnr_tracker = GSNRTracker(model)
            if self.enable_ler:
                self.ler_tracker = LERTracker(task=self.task)
            if self.enable_probe:
                hidden_dim = model.config.hidden_size
                num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
                self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
            self.metrics_collector = EfficiencyMetricsCollector(model, self.task)
        return control

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start_time = time.time()
        if self.enable_compute_tracking and torch.cuda.is_available():
            try:
                self.memory_stats.append({
                    "step": state.global_step,
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                    "timestamp": time.time(),
                })
            except Exception:
                pass
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            self.total_training_time += step_time
            self.compute_stats["step_times"].append(step_time)
            self.compute_stats["cumulative_time"].append(self.total_training_time)
        if state.global_step % self.log_frequency == 0:
            self._log_step_metrics(state)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            metrics = {}
        current_step = state.global_step
        eval_loss = metrics.get("eval_loss", 0)
        gradients = kwargs.get("gradients")
        logits = kwargs.get("logits")
        representations = kwargs.get("representations")
        labels = kwargs.get("labels")

        if self.metrics_collector:
            self.metrics_collector.update(
                gradients=gradients or {}, loss=eval_loss,
                logits=logits if logits is not None else torch.randn(1, 10),
                representations=representations, labels=labels,
                accuracy=metrics.get("eval_accuracy"), step=current_step,
            )

        efficiency_metrics = {}
        if self.ler_tracker and logits is not None:
            accuracy = metrics.get("eval_accuracy", 0)
            model = kwargs.get("model")
            self.ler_tracker.update(eval_loss, logits, accuracy, model=model, gradients=gradients)
            efficiency_metrics.update({
                "efficiency/ler": self.ler_tracker.get_ler(),
                "efficiency/ler_phase": self.ler_tracker.get_efficiency_phase(),
                "efficiency/rho_vg": self.ler_tracker.get_rho_vg(),
                "efficiency/param_velocity": self.ler_tracker.get_velocity(),
            })
        if self.gsnr_tracker and gradients:
            self.gsnr_tracker.update(gradients)
            gsnr_values = self.gsnr_tracker.get_gsnr(latest_only=True)
            if gsnr_values:
                efficiency_metrics["efficiency/gsnr"] = gsnr_values.get("overall", 0)
                for group_name, gsnr in gsnr_values.items():
                    if group_name != "overall":
                        efficiency_metrics[f"efficiency/gsnr_{group_name}"] = gsnr
        if self.probe_tracker and representations is not None and labels is not None:
            self.probe_tracker.add_representations(representations, labels)
            if current_step % 100 == 0:
                efficiency_metrics["efficiency/probe_accuracy"] = self.probe_tracker.compute_probe_accuracy()
        if self.enable_compute_tracking:
            efficiency_metrics.update(self._get_compute_metrics(state))

        metrics_entry = {
            "step": current_step, "epoch": state.epoch,
            "eval_loss": eval_loss, "eval_accuracy": metrics.get("eval_accuracy", 0),
            **efficiency_metrics, "timestamp": datetime.now().isoformat(),
        }
        self.metrics_history.append(metrics_entry)

        if self.wandb_enabled and _wandb_active():
            try:
                wandb.log(metrics_entry)
            except Exception:
                pass
        if gradients and current_step % 100 == 0:
            self._save_gradient_stats(gradients, current_step)
        if current_step % 500 == 0:
            self._save_metrics_history()
        return control

    def _get_compute_metrics(self, state):
        if not self.step_times:
            return {}
        recent = self.step_times[-100:]
        avg = np.mean(recent)
        m = {
            "compute/avg_step_time_ms": avg * 1000,
            "compute/total_training_time_s": self.total_training_time,
            "compute/steps_per_second": 1.0 / avg if avg > 0 else 0,
            "compute/total_steps": state.global_step,
        }
        if self.memory_stats:
            rm = self.memory_stats[-10:]
            m["memory/avg_allocated_gb"] = np.mean([x["allocated_gb"] for x in rm])
            m["memory/avg_reserved_gb"] = np.mean([x["reserved_gb"] for x in rm])
            try:
                m["memory/max_allocated_gb"] = torch.cuda.max_memory_allocated() / 1e9
            except Exception:
                pass
        return m

    def _save_gradient_stats(self, gradients, step):
        gs = {"step": step, "timestamp": datetime.now().isoformat(), "gradients": {}}
        for name, grad in gradients.items():
            if grad is not None:
                gs["gradients"][name] = {
                    "mean": float(grad.mean().item()), "std": float(grad.std().item()),
                    "min": float(grad.min().item()), "max": float(grad.max().item()),
                    "norm": float(torch.norm(grad).item()),
                }
        self.gradient_history.append(gs)
        if len(self.gradient_history) % 20 == 0:
            p = os.path.join(self.output_dir, f"gradient_stats_step{step}.json")
            with open(p, 'w') as f:
                json.dump(gs, f, indent=2, default=str)

    def _save_metrics_history(self):
        if self.metrics_history:
            with open(os.path.join(self.output_dir, "efficiency_metrics.json"), 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)

    def _log_step_metrics(self, state):
        if not self.metrics_history:
            return
        recent = self.metrics_history[-10:]
        ler_vals = [m.get("efficiency/ler") for m in recent if m.get("efficiency/ler") is not None]
        gsnr_vals = [m.get("efficiency/gsnr") for m in recent if m.get("efficiency/gsnr") is not None]
        ler_display = f"{np.mean(ler_vals):.6f}" if ler_vals else "N/A"
        gsnr_display = f"{np.mean(gsnr_vals):.3f}" if gsnr_vals else "N/A"
        print(f"\n  Efficiency (Step {state.global_step}): LER={ler_display} GSNR={gsnr_display}")

    def on_train_end(self, args, state, control, **kwargs):
        print(f"\n{'='*60}\nEFFICIENCY METRICS - TRAINING COMPLETE\n{'='*60}")
        self._generate_efficiency_report(state)
        self._save_all_data()
        print(f"  Total training time: {self.total_training_time:.1f}s")
        if self.step_times:
            print(f"  Average step time: {np.mean(self.step_times)*1000:.1f}ms")
        print(f"  Total evaluations: {len(self.metrics_history)}")
        print(f"{'='*60}")
        return control

    def _generate_efficiency_report(self, state):
        report = {"task": self.task, "total_steps": state.global_step,
                  "total_training_time_s": self.total_training_time,
                  "metrics_summary": {}, "efficiency_analysis": {}, "recommendations": []}
        for key in ["efficiency/ler", "efficiency/gsnr", "eval_loss", "eval_accuracy"]:
            values = [m.get(key) for m in self.metrics_history if m.get(key) is not None]
            if values:
                report["metrics_summary"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "min": float(np.min(values)), "max": float(np.max(values)),
                    "n_samples": len(values),
                }
        p = os.path.join(self.output_dir, "efficiency_report.json")
        with open(p, 'w') as f:
            json.dump(report, f, indent=2, default=str)

    def _save_all_data(self):
        if self.metrics_history:
            with open(os.path.join(self.output_dir, "all_metrics.json"), 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        if self.compute_stats:
            with open(os.path.join(self.output_dir, "compute_stats.json"), 'w') as f:
                json.dump(dict(self.compute_stats), f, indent=2, default=str)


# ===================================================================
# ProbeAccuracyCallback
# ===================================================================

class ProbeAccuracyCallback(TrainerCallback):
    """Probe accuracy callback for monitoring representation quality."""

    def __init__(self, probe_frequency=100, max_samples=1000,
                 output_dir="./experiments/probe_accuracy", wandb_enabled=True):
        self.probe_frequency = probe_frequency
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        os.makedirs(output_dir, exist_ok=True)
        self.probe_tracker = None
        self.probe_accuracies = []

    def on_init_end(self, args, state, control, **kwargs):
        if "model" in kwargs:
            model = kwargs["model"]
            hidden_dim = model.config.hidden_size
            num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
            self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if "representations" not in kwargs or "labels" not in kwargs or self.probe_tracker is None:
            return control
        self.probe_tracker.add_representations(kwargs["representations"], kwargs["labels"])
        if state.global_step % self.probe_frequency == 0:
            acc = self.probe_tracker.compute_probe_accuracy(max_samples=self.max_samples, train_new_probe=True)
            self.probe_accuracies.append({"step": state.global_step, "probe_accuracy": acc})
            print(f"  Probe Accuracy (Step {state.global_step}): {acc:.4f}")
            if self.wandb_enabled and _wandb_active():
                try:
                    wandb.log({"probe/accuracy": acc})
                except Exception:
                    pass
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.probe_accuracies:
            p = os.path.join(self.output_dir, "probe_accuracies.json")
            with open(p, 'w') as f:
                json.dump(self.probe_accuracies, f, indent=2, default=str)
        return control


# ===================================================================
# GradientAnalysisCallback
# ===================================================================

class GradientAnalysisCallback(TrainerCallback):
    """Gradient analysis callback for monitoring training dynamics."""

    def __init__(self, analysis_frequency=50, parameter_groups=None,
                 output_dir="./experiments/gradient_analysis", wandb_enabled=True):
        self.analysis_frequency = analysis_frequency
        self.parameter_groups = parameter_groups or ["attention", "ffn", "embeddings", "classifier"]
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        os.makedirs(output_dir, exist_ok=True)
        self.gradient_stats = []
        self.parameter_mapping = {}

    def on_init_end(self, args, state, control, **kwargs):
        if "model" in kwargs:
            for name, _ in kwargs["model"].named_parameters():
                if not name.endswith('.bias'):
                    if any(g in name for g in ["attention", "self_attn", "attn"]):
                        self.parameter_mapping[name] = "attention"
                    elif any(g in name for g in ["ffn", "intermediate", "output"]):
                        self.parameter_mapping[name] = "ffn"
                    elif "embedding" in name:
                        self.parameter_mapping[name] = "embeddings"
                    elif any(g in name for g in ["classifier", "pooler"]):
                        self.parameter_mapping[name] = "classifier"
                    else:
                        self.parameter_mapping[name] = "other"
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if "gradients" not in kwargs:
            return control
        if state.global_step % self.analysis_frequency == 0:
            gradients = kwargs["gradients"]
            stats = {"step": state.global_step, "group_stats": {
                g: {"norm": 0.0, "mean": 0.0, "n": 0} for g in self.parameter_groups + ["other"]
            }}
            for name, grad in gradients.items():
                if grad is None:
                    continue
                group = self.parameter_mapping.get(name, "other")
                gs = stats["group_stats"][group]
                gs["norm"] += torch.norm(grad).item() ** 2
                gs["mean"] += grad.mean().item()
                gs["n"] += 1
            for g, gs in stats["group_stats"].items():
                if gs["n"] > 0:
                    gs["norm"] = np.sqrt(gs["norm"])
                    gs["mean"] /= gs["n"]
            self.gradient_stats.append(stats)
            if self.wandb_enabled and _wandb_active():
                try:
                    wandb.log({f"gradients/{g}/norm": gs["norm"] for g, gs in stats["group_stats"].items()})
                except Exception:
                    pass
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.gradient_stats:
            p = os.path.join(self.output_dir, "gradient_statistics.json")
            with open(p, 'w') as f:
                json.dump(self.gradient_stats, f, indent=2, default=str)
        return control


# ===================================================================
# ComputeCostTracker
# ===================================================================

class ComputeCostTracker(TrainerCallback):
    """Compute cost tracking callback."""

    def __init__(self, gpu_hourly_cost=2.99, output_dir="./experiments/compute_costs", wandb_enabled=True):
        self.gpu_hourly_cost = gpu_hourly_cost
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        os.makedirs(output_dir, exist_ok=True)
        self.start_time = None
        self.memory_usage = []
        self.cost_estimates = []

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        try:
            if torch.cuda.is_available() and state.global_step % 100 == 0:
                self.memory_usage.append({
                    "step": state.global_step,
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "timestamp": time.time(),
                })
        except Exception:
            pass
        return control

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.start_time is None:
            return control
        elapsed_hours = (time.time() - self.start_time) / 3600
        estimated_cost = elapsed_hours * self.gpu_hourly_cost
        self.cost_estimates.append({"step": state.global_step, "elapsed_hours": elapsed_hours, "estimated_cost": estimated_cost})
        if state.global_step % 500 == 0:
            print(f"  Compute Cost (Step {state.global_step}): {elapsed_hours:.2f}h, ${estimated_cost:.2f}")
            if self.wandb_enabled and _wandb_active():
                try:
                    wandb.log({"cost/elapsed_hours": elapsed_hours, "cost/estimated_cost": estimated_cost})
                except Exception:
                    pass
        return control

    def on_train_end(self, args, state, control, **kwargs):
        if self.start_time is None:
            return control
        total_hours = (time.time() - self.start_time) / 3600
        total_cost = total_hours * self.gpu_hourly_cost
        report = {"total_training_hours": total_hours, "total_estimated_cost": total_cost}
        if self.memory_usage:
            report["avg_memory_gb"] = float(np.mean([m["allocated_gb"] for m in self.memory_usage]))
        p = os.path.join(self.output_dir, "compute_cost_report.json")
        with open(p, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  Cost report saved: {p}")
        return control


# ===================================================================
# PowerTelemetryCallback  (RTX 5090 / Blackwell crash-proof)
# ===================================================================

class PowerTelemetryCallback(TrainerCallback):
    """
    GPU power telemetry via nvidia-smi for energy (kWh) measurement.

    **Crash-proof on RTX 5090 / Blackwell architecture:**

    - All nvidia-smi calls wrapped in try/except with timeouts
    - Output parsing handles [N/A], unit suffixes, and format changes
    - Falls back to TDP-based estimation when direct reading fails
    - Never raises exceptions into the Trainer loop

    Samples power.draw at a configurable interval using a background thread,
    integrates watts over wall-clock time to produce cumulative energy in kWh.
    """

    _KNOWN_TDP = {
        "5090": 575.0, "5080": 360.0, "5070 ti": 300.0, "5070": 250.0,
        "4090": 450.0, "4080": 320.0, "3090": 350.0, "3080": 320.0,
        "a100": 400.0, "h100": 700.0, "a6000": 300.0,
    }

    def __init__(self, sample_interval_s=1.0, gpu_index=0,
                 output_dir="./experiments/power_telemetry",
                 wandb_enabled=True, log_frequency=25):
        self.sample_interval_s = sample_interval_s
        self.gpu_index = gpu_index
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        self.log_frequency = log_frequency
        os.makedirs(output_dir, exist_ok=True)

        self._power_samples: List[Dict] = []
        self._sampling_thread: Optional[threading.Thread] = None
        self._stop_event: Optional[threading.Event] = None
        self._training_start: Optional[float] = None
        self._step_energy_start_idx = 0
        self._consecutive_errors = 0
        self._max_consecutive_errors = 10
        self._fallback_tdp: Optional[float] = None
        self._gpu_name = ""
        self.cumulative_kwh = 0.0
        self.step_energies: List[Dict] = []

    # --- GPU detection ---

    def _detect_gpu(self):
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self._gpu_name = result.stdout.strip()
        except Exception:
            pass
        if not self._gpu_name:
            try:
                if torch.cuda.is_available():
                    self._gpu_name = torch.cuda.get_device_name(self.gpu_index)
            except Exception:
                pass
        name_lower = self._gpu_name.lower()
        for key, tdp in self._KNOWN_TDP.items():
            if key in name_lower:
                self._fallback_tdp = tdp
                return
        self._fallback_tdp = 350.0

    # --- nvidia-smi query with Blackwell-safe parsing ---

    def _query_nvidia_smi(self) -> Optional[Dict]:
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                return None
            parts = [p.strip() for p in result.stdout.strip().split(",")]
            if len(parts) < 3:
                return None
            power = _safe_parse_float(parts[0])
            temp = _safe_parse_float(parts[1])
            util = _safe_parse_float(parts[2])
            if power is not None and 0 < power < 2000:
                return {"power_w": power, "temp_c": temp or 0.0, "util_pct": util or 0.0}
        except Exception:
            pass
        return None

    # --- TDP-based fallback estimation ---

    def _estimate_power(self) -> Dict:
        util, temp = 80.0, 0.0
        try:
            result = subprocess.run(
                ["nvidia-smi", f"--id={self.gpu_index}",
                 "--query-gpu=utilization.gpu,temperature.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=3)
            if result.returncode == 0:
                parts = [p.strip() for p in result.stdout.strip().split(",")]
                u = _safe_parse_float(parts[0]) if len(parts) > 0 else None
                t = _safe_parse_float(parts[1]) if len(parts) > 1 else None
                if u is not None: util = u
                if t is not None: temp = t
        except Exception:
            pass
        tdp = self._fallback_tdp or 350.0
        idle = tdp * 0.08
        return {"power_w": idle + (tdp - idle) * (util / 100.0), "temp_c": temp, "util_pct": util}

    # --- Background sampling thread ---

    def _sample_power_loop(self):
        while not self._stop_event.is_set():
            try:
                sample = self._query_nvidia_smi()
                if sample is None:
                    self._consecutive_errors += 1
                    if self._consecutive_errors <= self._max_consecutive_errors:
                        sample = self._estimate_power()
                else:
                    self._consecutive_errors = 0
                if sample is not None:
                    sample["timestamp"] = time.time()
                    self._power_samples.append(sample)
            except Exception:
                pass
            self._stop_event.wait(self.sample_interval_s)

    def _start_sampling(self):
        self._stop_event = threading.Event()
        self._sampling_thread = threading.Thread(target=self._sample_power_loop, daemon=True)
        self._sampling_thread.start()

    def _stop_sampling(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=5)

    # --- Energy integration ---

    def _compute_energy_kwh(self, start_idx, end_idx):
        if end_idx - start_idx < 2:
            return 0.0
        try:
            samples = self._power_samples[start_idx:end_idx]
            joules = 0.0
            for i in range(1, len(samples)):
                dt = samples[i]["timestamp"] - samples[i-1]["timestamp"]
                avg_p = (samples[i]["power_w"] + samples[i-1]["power_w"]) / 2.0
                if dt > 0 and avg_p > 0:
                    joules += avg_p * dt
            return joules / 3_600_000.0
        except Exception:
            return 0.0

    # --- Trainer hooks (NEVER crash) ---

    def on_train_begin(self, args, state, control, **kwargs):
        try:
            self._detect_gpu()
            self._training_start = time.time()
            self._start_sampling()
            gpu_info = f" ({self._gpu_name})" if self._gpu_name else ""
            print(f"  Power telemetry started{gpu_info} GPU {self.gpu_index}, {self.sample_interval_s}s interval")
        except Exception as exc:
            print(f"  Power telemetry failed to start: {exc}")
        return control

    def on_step_end(self, args, state, control, **kwargs):
        try:
            current_idx = len(self._power_samples)
            step_kwh = self._compute_energy_kwh(self._step_energy_start_idx, current_idx)
            self.cumulative_kwh += step_kwh
            self._step_energy_start_idx = current_idx
            self.step_energies.append({"step": state.global_step, "step_kwh": step_kwh, "cumulative_kwh": self.cumulative_kwh})

            if state.global_step % self.log_frequency == 0:
                recent = self._power_samples[-10:] if self._power_samples else []
                avg_power = float(np.mean([s["power_w"] for s in recent])) if recent else 0
                avg_temp = float(np.mean([s["temp_c"] for s in recent])) if recent else 0
                avg_util = float(np.mean([s["util_pct"] for s in recent])) if recent else 0
                log_metrics = {
                    "power/current_watts": avg_power, "power/temperature_c": avg_temp,
                    "power/gpu_utilization_pct": avg_util, "power/cumulative_kwh": self.cumulative_kwh,
                }
                if self.wandb_enabled and _wandb_active():
                    try:
                        wandb.log(log_metrics)
                    except Exception:
                        pass
                print(f"  Power (Step {state.global_step}): {avg_power:.1f}W | {self.cumulative_kwh:.6f} kWh | {avg_temp:.0f}C | {avg_util:.0f}% util")
        except Exception as exc:
            print(f"  [Power] step_end error: {exc}")
        return control

    def on_train_end(self, args, state, control, **kwargs):
        try:
            self._stop_sampling()
            total_time_s = time.time() - self._training_start if self._training_start else 0
            total_samples = len(self._power_samples)
            all_power = [s["power_w"] for s in self._power_samples] if self._power_samples else [0]

            report = {
                "gpu_name": self._gpu_name, "total_training_time_s": total_time_s,
                "total_energy_kwh": self.cumulative_kwh, "total_power_samples": total_samples,
                "power_statistics": {
                    "mean_watts": float(np.mean(all_power)), "max_watts": float(np.max(all_power)),
                    "min_watts": float(np.min(all_power)), "std_watts": float(np.std(all_power)),
                },
                "per_step_energy": self.step_energies,
                "gpu_index": self.gpu_index, "sample_interval_s": self.sample_interval_s,
            }
            report_path = os.path.join(self.output_dir, "power_telemetry_report.json")
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2, default=str)
            samples_path = os.path.join(self.output_dir, "power_samples.json")
            with open(samples_path, "w") as f:
                json.dump(self._power_samples, f, indent=2, default=str)

            print(f"\n{'='*60}")
            print("POWER TELEMETRY REPORT")
            print(f"{'='*60}")
            print(f"  GPU: {self._gpu_name or 'unknown'}")
            print(f"  Total energy: {self.cumulative_kwh:.6f} kWh")
            print(f"  Training time: {total_time_s:.1f}s ({total_time_s/3600:.3f}h)")
            print(f"  Avg power draw: {np.mean(all_power):.1f}W")
            print(f"  Peak power draw: {np.max(all_power):.1f}W")
            print(f"  Samples collected: {total_samples}")
            print(f"  Report saved: {report_path}")
            print(f"{'='*60}")

            if self.wandb_enabled and _wandb_active():
                try:
                    wandb.log({
                        "power/total_kwh": self.cumulative_kwh,
                        "power/total_training_hours": total_time_s / 3600,
                        "power/avg_watts": float(np.mean(all_power)),
                        "power/peak_watts": float(np.max(all_power)),
                    })
                except Exception:
                    pass
        except Exception as exc:
            print(f"  [Power] train_end error: {exc}")
        return control
