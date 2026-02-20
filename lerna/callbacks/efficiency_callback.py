"""
Efficiency Metrics Callback Implementation

Comprehensive efficiency tracking including GSNR, LER, probe accuracy,
and compute cost analysis.
"""

import torch
import wandb
import json
import os
import time
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments

from ..utils.metrics import (
    GSNRTracker, 
    LERTracker, 
    ProbeAccuracyTracker,
    EfficiencyMetricsCollector,
    validate_ler_metric,
)


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
        """
        Initialize efficiency metrics callback.
        
        Args:
            task: Task name for task-specific calibration
            enable_gsnr: Whether to track GSNR
            enable_ler: Whether to track LER
            enable_probe: Whether to track probe accuracy
            enable_compute_tracking: Whether to track compute costs
            output_dir: Directory for output files
            wandb_enabled: Whether to log to Weights & Biases
            log_frequency: How often to log metrics (in steps)
        """
        self.task = task
        self.enable_gsnr = enable_gsnr
        self.enable_ler = enable_ler
        self.enable_probe = enable_probe
        self.enable_compute_tracking = enable_compute_tracking
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        self.log_frequency = log_frequency
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize trackers (will be set when model is available)
        self.gsnr_tracker = None
        self.ler_tracker = None
        self.probe_tracker = None
        self.metrics_collector = None
        
        # Metrics storage
        self.metrics_history: List[Dict] = []
        self.gradient_history: List[Dict] = []
        self.compute_stats: Dict = defaultdict(list)
        
        # Timing
        self.step_start_time = None
        self.total_training_time = 0
        self.step_times = []
        
        # Memory tracking
        self.memory_stats = []
        
        print(f"✅ Efficiency Metrics Callback initialized")
        print(f"   Task: {task}")
        print(f"   GSNR: {enable_gsnr}, LER: {enable_ler}, Probe: {enable_probe}")
        print(f"   Output directory: {output_dir}")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize trackers when model is available."""
        if "model" in kwargs:
            model = kwargs["model"]
            
            if self.enable_gsnr:
                self.gsnr_tracker = GSNRTracker(model)
                print(f"   ✓ GSNR tracker initialized")
            
            if self.enable_ler:
                self.ler_tracker = LERTracker(task=self.task)
                print(f"   ✓ LER tracker initialized")
            
            if self.enable_probe:
                hidden_dim = model.config.hidden_size
                num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
                self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
                print(f"   ✓ Probe accuracy tracker initialized")
            
            # Initialize metrics collector
            self.metrics_collector = EfficiencyMetricsCollector(model, self.task)
            print(f"   ✓ Efficiency metrics collector initialized")
        
        return control
    
    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Record step start time and memory usage."""
        self.step_start_time = time.time()
        
        if self.enable_compute_tracking and torch.cuda.is_available():
            memory_stats = {
                "step": state.global_step,
                "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
                "timestamp": time.time(),
            }
            self.memory_stats.append(memory_stats)
        
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Record step completion time and compute metrics."""
        if self.step_start_time is not None:
            step_time = time.time() - self.step_start_time
            self.step_times.append(step_time)
            self.total_training_time += step_time
            
            # Store compute stats
            self.compute_stats["step_times"].append(step_time)
            self.compute_stats["cumulative_time"].append(self.total_training_time)
        
        # Log metrics periodically
        if state.global_step % self.log_frequency == 0:
            self._log_step_metrics(state)
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Update efficiency metrics during evaluation."""
        current_step = state.global_step
        eval_loss = metrics.get("eval_loss", 0)
        
        # Collect gradients if available
        gradients = None
        if "gradients" in kwargs:
            gradients = kwargs["gradients"]
        
        # Collect logits if available
        logits = None
        if "logits" in kwargs:
            logits = kwargs["logits"]
        
        # Collect representations if available
        representations = None
        labels = None
        if "representations" in kwargs and "labels" in kwargs:
            representations = kwargs["representations"]
            labels = kwargs["labels"]
        
        # Update metrics collector
        if self.metrics_collector:
            self.metrics_collector.update(
                gradients=gradients or {},
                loss=eval_loss,
                logits=logits if logits is not None else torch.randn(1, 10),
                representations=representations,
                labels=labels,
                accuracy=metrics.get("eval_accuracy"),
                step=current_step,
            )
        
        # Update individual trackers
        efficiency_metrics = {}
        
        if self.ler_tracker and logits is not None:
            accuracy = metrics.get("eval_accuracy", 0)
            model = kwargs.get("model")
            self.ler_tracker.update(
                eval_loss, logits, accuracy,
                model=model, gradients=gradients,
            )
            
            ler_value = self.ler_tracker.get_ler()
            ler_phase = self.ler_tracker.get_efficiency_phase()
            rho_vg = self.ler_tracker.get_rho_vg()
            param_velocity = self.ler_tracker.get_velocity()
            
            efficiency_metrics.update({
                "efficiency/ler": ler_value,
                "efficiency/ler_phase": ler_phase,
                "efficiency/rho_vg": rho_vg,
                "efficiency/param_velocity": param_velocity,
            })
        
        if self.gsnr_tracker and gradients:
            self.gsnr_tracker.update(gradients)
            gsnr_values = self.gsnr_tracker.get_gsnr(latest_only=True)
            if gsnr_values:
                efficiency_metrics["efficiency/gsnr"] = gsnr_values.get("overall", 0)
                
                # Log per-group GSNR
                for group_name, gsnr in gsnr_values.items():
                    if group_name != "overall":
                        efficiency_metrics[f"efficiency/gsnr_{group_name}"] = gsnr
        
        if self.probe_tracker and representations is not None and labels is not None:
            self.probe_tracker.add_representations(representations, labels)
            
            # Compute probe accuracy periodically
            if current_step % 100 == 0:
                probe_accuracy = self.probe_tracker.compute_probe_accuracy()
                efficiency_metrics["efficiency/probe_accuracy"] = probe_accuracy
        
        # Add compute metrics
        if self.enable_compute_tracking:
            compute_metrics = self._get_compute_metrics(state)
            efficiency_metrics.update(compute_metrics)
        
        # Store metrics
        metrics_entry = {
            "step": current_step,
            "epoch": state.epoch,
            "eval_loss": eval_loss,
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            **efficiency_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.metrics_history.append(metrics_entry)
        
        # Log to W&B
        if self.wandb_enabled and wandb.run:
            wandb.log(metrics_entry)
        
        # Save gradient statistics
        if gradients and current_step % 100 == 0:
            self._save_gradient_stats(gradients, current_step)
        
        # Save metrics periodically
        if current_step % 500 == 0:
            self._save_metrics_history()
        
        return control
    
    def _get_compute_metrics(self, state: TrainerState) -> Dict[str, float]:
        """Get compute-related metrics."""
        if not self.step_times:
            return {}
        
        recent_step_times = self.step_times[-100:] if len(self.step_times) >= 100 else self.step_times
        
        metrics = {
            "compute/avg_step_time_ms": np.mean(recent_step_times) * 1000,
            "compute/total_training_time_s": self.total_training_time,
            "compute/steps_per_second": 1.0 / np.mean(recent_step_times) if np.mean(recent_step_times) > 0 else 0,
            "compute/total_steps": state.global_step,
        }
        
        # Memory metrics
        if self.memory_stats:
            recent_memory = self.memory_stats[-10:] if len(self.memory_stats) >= 10 else self.memory_stats
            avg_allocated = np.mean([m["allocated_gb"] for m in recent_memory])
            avg_reserved = np.mean([m["reserved_gb"] for m in recent_memory])
            
            metrics.update({
                "memory/avg_allocated_gb": avg_allocated,
                "memory/avg_reserved_gb": avg_reserved,
                "memory/max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            })
        
        return metrics
    
    def _save_gradient_stats(self, gradients: Dict[str, torch.Tensor], step: int):
        """Save gradient statistics."""
        gradient_stats = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "gradients": {},
        }
        
        for name, grad in gradients.items():
            if grad is not None:
                gradient_stats["gradients"][name] = {
                    "mean": float(grad.mean().item()),
                    "std": float(grad.std().item()),
                    "min": float(grad.min().item()),
                    "max": float(grad.max().item()),
                    "norm": float(torch.norm(grad).item()),
                }
        
        self.gradient_history.append(gradient_stats)
        
        # Save periodically
        if len(self.gradient_history) % 20 == 0:
            gradient_path = os.path.join(self.output_dir, f"gradient_stats_step{step}.json")
            with open(gradient_path, 'w') as f:
                json.dump(gradient_stats, f, indent=2, default=str)
    
    def _save_metrics_history(self):
        """Save metrics history to disk."""
        if not self.metrics_history:
            return
        
        metrics_path = os.path.join(self.output_dir, "efficiency_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def _log_step_metrics(self, state: TrainerState):
        """Log step-level metrics."""
        if not self.metrics_history:
            return
        
        # Get recent metrics
        recent_metrics = self.metrics_history[-10:] if len(self.metrics_history) >= 10 else self.metrics_history
        
        # Compute averages
        avg_ler = np.mean([m.get("efficiency/ler", 0) for m in recent_metrics 
                          if m.get("efficiency/ler") is not None])
        avg_gsnr = np.mean([m.get("efficiency/gsnr", 0) for m in recent_metrics 
                           if m.get("efficiency/gsnr") is not None])
        
        print(f"\n  Efficiency Metrics (Step {state.global_step}):")
        print(f"   LER: {avg_ler:.6f}")
        print(f"   GSNR: {avg_gsnr:.3f}")
        
        if self.ler_tracker:
            ler_phase = self.ler_tracker.get_efficiency_phase()
            rho_vg = self.ler_tracker.get_rho_vg()
            velocity = self.ler_tracker.get_velocity()
            print(f"   Learning Phase: {ler_phase}")
            if rho_vg is not None:
                print(f"   rho_VG: {rho_vg:.4f}")
            if velocity is not None:
                print(f"   Param Velocity: {velocity:.6f}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize efficiency tracking and generate reports."""
        print(f"\n{'='*60}")
        print("EFFICIENCY METRICS - TRAINING COMPLETE")
        print(f"{'='*60}")
        
        # Generate comprehensive reports
        self._generate_efficiency_report(state)
        
        # Save all data
        self._save_all_data()
        
        print(f"\n📊 Summary Statistics:")
        print(f"   Total training time: {self.total_training_time:.1f}s")
        print(f"   Average step time: {np.mean(self.step_times)*1000:.1f}ms")
        print(f"   Total evaluations: {len(self.metrics_history)}")
        
        if self.metrics_collector:
            collector_report = self.metrics_collector.get_comprehensive_report()
            print(f"\n📈 Efficiency Analysis:")
            for key, value in collector_report.get("current_metrics", {}).items():
                if value is not None:
                    print(f"   {key}: {value:.4f}")
        
        print(f"\n📁 Analysis files saved in: {self.output_dir}")
        print(f"{'='*60}")
        
        return control
    
    def _generate_efficiency_report(self, state: TrainerState):
        """Generate comprehensive efficiency report."""
        report = {
            "task": self.task,
            "total_steps": state.global_step,
            "total_training_time_s": self.total_training_time,
            "metrics_summary": {},
            "efficiency_analysis": {},
            "recommendations": [],
        }
        
        # Summarize metrics
        metric_keys = ["efficiency/ler", "efficiency/gsnr", "efficiency/probe_accuracy",
                      "eval_loss", "eval_accuracy"]
        
        for key in metric_keys:
            values = [m.get(key) for m in self.metrics_history if m.get(key) is not None]
            if values:
                report["metrics_summary"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_samples": len(values),
                    "trend": self._compute_trend(values),
                }
        
        # Compute efficiency trends
        ler_values = [m.get("efficiency/ler") for m in self.metrics_history 
                     if m.get("efficiency/ler") is not None]
        gsnr_values = [m.get("efficiency/gsnr") for m in self.metrics_history 
                      if m.get("efficiency/gsnr") is not None]
        
        if ler_values and len(ler_values) > 10:
            # LER trend analysis
            recent_ler = ler_values[-10:]
            ler_trend = np.polyfit(range(len(recent_ler)), recent_ler, 1)[0]
            
            report["efficiency_analysis"]["ler_trend"] = {
                "slope": float(ler_trend),
                "trend": "decreasing" if ler_trend < -0.001 else "stable" if abs(ler_trend) < 0.001 else "increasing",
                "interpretation": self._interpret_ler_trend(ler_trend),
            }
        
        if gsnr_values and len(gsnr_values) > 10:
            # GSNR trend analysis
            recent_gsnr = gsnr_values[-10:]
            gsnr_trend = np.polyfit(range(len(recent_gsnr)), recent_gsnr, 1)[0]
            
            report["efficiency_analysis"]["gsnr_trend"] = {
                "slope": float(gsnr_trend),
                "trend": "decreasing" if gsnr_trend < -0.01 else "stable" if abs(gsnr_trend) < 0.01 else "increasing",
                "warning": "GSNR decreasing significantly" if gsnr_trend < -0.05 else None,
            }
        
        # Generate recommendations
        self._generate_recommendations(report)
        
        # Add metrics collector report if available
        if self.metrics_collector:
            collector_report = self.metrics_collector.get_comprehensive_report()
            report["detailed_analysis"] = collector_report
        
        # Save report
        report_path = os.path.join(self.output_dir, "efficiency_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📁 Efficiency report saved: {report_path}")
        
        return report
    
    def _compute_trend(self, values: List[float]) -> str:
        """Compute trend of values."""
        if len(values) < 10:
            return "insufficient_data"
        
        x = np.arange(len(values))
        slope, _ = np.polyfit(x, values, 1)
        
        if slope < -0.001:
            return "decreasing"
        elif slope > 0.001:
            return "increasing"
        else:
            return "stable"
    
    def _interpret_ler_trend(self, slope: float) -> str:
        """Interpret LER trend slope."""
        if slope < -0.005:
            return "Strong decreasing trend - efficiency dropping rapidly"
        elif slope < -0.001:
            return "Moderate decreasing trend - efficiency declining"
        elif slope < 0.001:
            return "Stable - consistent efficiency"
        elif slope < 0.005:
            return "Moderate increasing trend - efficiency improving"
        else:
            return "Strong increasing trend - significant efficiency gains"
    
    def _generate_recommendations(self, report: Dict):
        """Generate recommendations based on efficiency analysis."""
        recommendations = []
        
        # Check LER trend
        ler_trend = report.get("efficiency_analysis", {}).get("ler_trend", {})
        if ler_trend.get("trend") == "decreasing":
            slope = ler_trend.get("slope", 0)
            if slope < -0.005:
                recommendations.append(
                    "LER decreasing rapidly. Consider reducing learning rate or "
                    "implementing early stopping."
                )
            else:
                recommendations.append(
                    "LER decreasing. Monitor closely for potential plateau."
                )
        
        # Check GSNR trend
        gsnr_trend = report.get("efficiency_analysis", {}).get("gsnr_trend", {})
        if gsnr_trend.get("trend") == "decreasing" and gsnr_trend.get("warning"):
            recommendations.append(
                "GSNR decreasing significantly. Gradient quality may be deteriorating."
            )
        
        # Check compute efficiency
        avg_step_time = report.get("metrics_summary", {}).get("compute/avg_step_time_ms", {})
        if avg_step_time.get("mean", 0) > 100:  # More than 100ms per step
            recommendations.append(
                f"High step time ({avg_step_time['mean']:.1f}ms). "
                "Consider optimizing data loading or model architecture."
            )
        
        # Check memory usage
        if self.memory_stats:
            avg_memory = np.mean([m["allocated_gb"] for m in self.memory_stats])
            if avg_memory > 20:  # More than 20GB
                recommendations.append(
                    f"High memory usage ({avg_memory:.1f}GB). "
                    "Consider gradient checkpointing or model parallelism."
                )
        
        report["recommendations"] = recommendations
    
    def _save_all_data(self):
        """Save all collected data."""
        # Save metrics history
        if self.metrics_history:
            metrics_path = os.path.join(self.output_dir, "all_metrics.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
        
        # Save compute stats
        if self.compute_stats:
            compute_path = os.path.join(self.output_dir, "compute_stats.json")
            with open(compute_path, 'w') as f:
                json.dump(dict(self.compute_stats), f, indent=2, default=str)
        
        # Save memory stats
        if self.memory_stats:
            memory_path = os.path.join(self.output_dir, "memory_stats.json")
            with open(memory_path, 'w') as f:
                json.dump(self.memory_stats, f, indent=2, default=str)
        
        # Save gradient history (sampled)
        if self.gradient_history and len(self.gradient_history) > 100:
            # Save only every 10th entry to reduce file size
            sampled_gradients = self.gradient_history[::10]
            gradient_path = os.path.join(self.output_dir, "gradient_history_sampled.json")
            with open(gradient_path, 'w') as f:
                json.dump(sampled_gradients, f, indent=2, default=str)


class ProbeAccuracyCallback(TrainerCallback):
    """
    Probe accuracy callback for monitoring representation quality.
    
    Trains linear probes on model representations to measure
    how well the model is learning useful features.
    """
    
    def __init__(
        self,
        probe_frequency: int = 100,
        max_samples: int = 1000,
        output_dir: str = "./experiments/probe_accuracy",
        wandb_enabled: bool = True,
    ):
        self.probe_frequency = probe_frequency
        self.max_samples = max_samples
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.probe_tracker = None
        self.probe_accuracies = []
        self.representation_history = []
        
        print(f"✅ Probe Accuracy Callback initialized")
        print(f"   Probe frequency: every {probe_frequency} steps")
        print(f"   Max samples per probe: {max_samples}")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize probe tracker when model is available."""
        if "model" in kwargs:
            model = kwargs["model"]
            hidden_dim = model.config.hidden_size
            num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
            
            self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
            print(f"   ✓ Probe tracker initialized")
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Collect representations for probe training."""
        if "representations" not in kwargs or "labels" not in kwargs:
            return control
        
        if self.probe_tracker is None:
            return control
        
        representations = kwargs["representations"]
        labels = kwargs["labels"]
        current_step = state.global_step
        
        # Store representations
        self.representation_history.append({
            "step": current_step,
            "representations_shape": representations.shape,
            "labels_shape": labels.shape,
            "timestamp": datetime.now().isoformat(),
        })
        
        # Add to probe tracker
        self.probe_tracker.add_representations(representations, labels)
        
        # Train probe periodically
        if current_step % self.probe_frequency == 0:
            probe_accuracy = self.probe_tracker.compute_probe_accuracy(
                max_samples=self.max_samples,
                train_new_probe=True,
            )
            
            self.probe_accuracies.append({
                "step": current_step,
                "probe_accuracy": probe_accuracy,
                "representation_quality": self.probe_tracker.get_representation_quality(),
                "accuracy_trend": self.probe_tracker.get_accuracy_trend(),
                "timestamp": datetime.now().isoformat(),
            })
            
            print(f"\n🔍 Probe Accuracy (Step {current_step}): {probe_accuracy:.4f}")
            print(f"   Representation quality: {self.probe_tracker.get_representation_quality()}")
            
            # Log to W&B
            if self.wandb_enabled and wandb.run:
                wandb.log({
                    "probe/accuracy": probe_accuracy,
                    "probe/representation_quality": self.probe_tracker.get_representation_quality(),
                    "probe/accuracy_trend": self.probe_tracker.get_accuracy_trend() or 0,
                })
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize probe analysis."""
        if not self.probe_accuracies:
            return control
        
        print(f"\n📊 Probe Accuracy Analysis:")
        print(f"   Total probe measurements: {len(self.probe_accuracies)}")
        
        if self.probe_accuracies:
            final_accuracy = self.probe_accuracies[-1]["probe_accuracy"]
            print(f"   Final probe accuracy: {final_accuracy:.4f}")
            
            # Compute trend
            accuracies = [pa["probe_accuracy"] for pa in self.probe_accuracies]
            if len(accuracies) > 5:
                x = np.arange(len(accuracies))
                slope, _ = np.polyfit(x, accuracies, 1)
                print(f"   Accuracy trend: {slope:.6f} per measurement")
        
        # Save probe results
        probe_path = os.path.join(self.output_dir, "probe_accuracies.json")
        with open(probe_path, 'w') as f:
            json.dump(self.probe_accuracies, f, indent=2, default=str)
        
        print(f"📁 Probe results saved: {probe_path}")
        
        return control


class GradientAnalysisCallback(TrainerCallback):
    """
    Gradient analysis callback for monitoring training dynamics.
    
    Analyzes gradient statistics, norms, and distributions.
    """
    
    def __init__(
        self,
        analysis_frequency: int = 50,
        parameter_groups: Optional[List[str]] = None,
        output_dir: str = "./experiments/gradient_analysis",
        wandb_enabled: bool = True,
    ):
        self.analysis_frequency = analysis_frequency
        self.parameter_groups = parameter_groups or ["attention", "ffn", "embeddings", "classifier"]
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.gradient_stats = []
        self.parameter_mapping = {}
        
        print(f"✅ Gradient Analysis Callback initialized")
        print(f"   Analysis frequency: every {analysis_frequency} steps")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize parameter mapping when model is available."""
        if "model" in kwargs:
            model = kwargs["model"]
            self._create_parameter_mapping(model)
            print(f"   ✓ Parameter mapping created: {len(self.parameter_mapping)} parameters")
        
        return control
    
    def _create_parameter_mapping(self, model: torch.nn.Module):
        """Map parameters to groups based on name patterns."""
        for name, _ in model.named_parameters():
            if not name.endswith('.bias'):  # Skip biases
                if any(group in name for group in ["attention", "self_attn", "attn"]):
                    self.parameter_mapping[name] = "attention"
                elif any(group in name for group in ["ffn", "intermediate", "output"]):
                    self.parameter_mapping[name] = "ffn"
                elif "embedding" in name or "embeddings" in name:
                    self.parameter_mapping[name] = "embeddings"
                elif any(group in name for group in ["classifier", "pooler"]):
                    self.parameter_mapping[name] = "classifier"
                else:
                    self.parameter_mapping[name] = "other"
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Analyze gradients."""
        if "gradients" not in kwargs:
            return control
        
        gradients = kwargs["gradients"]
        current_step = state.global_step
        
        # Analyze gradients periodically
        if current_step % self.analysis_frequency == 0:
            gradient_stats = self._analyze_gradients(gradients, current_step)
            self.gradient_stats.append(gradient_stats)
            
            # Print summary
            print(f"\n📈 Gradient Analysis (Step {current_step}):")
            for group, stats in gradient_stats["group_stats"].items():
                print(f"   {group}: norm={stats['norm']:.4f}, mean={stats['mean']:.6f}")
            
            # Log to W&B
            if self.wandb_enabled and wandb.run:
                wandb.log({
                    f"gradients/{group}/norm": stats["norm"]
                    for group, stats in gradient_stats["group_stats"].items()
                })
        
        return control
    
    def _analyze_gradients(self, gradients: Dict[str, torch.Tensor], step: int) -> Dict:
        """Analyze gradient statistics."""
        stats = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "total_norm": 0.0,
                "mean_abs": 0.0,
                "std": 0.0,
                "max_abs": 0.0,
            },
            "group_stats": {group: {
                "norm": 0.0,
                "mean": 0.0,
                "std": 0.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "n_parameters": 0,
            } for group in self.parameter_groups + ["other"]},
        }
        
        # Analyze each gradient
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Get parameter group
            group = self.parameter_mapping.get(name, "other")
            
            # Compute statistics
            grad_norm = torch.norm(grad).item()
            grad_mean = grad.mean().item()
            grad_std = grad.std().item()
            grad_mean_abs = grad.abs().mean().item()
            grad_max_abs = grad.abs().max().item()
            
            # Update overall stats
            stats["overall"]["total_norm"] += grad_norm ** 2
            stats["overall"]["mean_abs"] += grad_mean_abs
            stats["overall"]["std"] += grad_std
            stats["overall"]["max_abs"] = max(stats["overall"]["max_abs"], grad_max_abs)
            
            # Update group stats
            group_stat = stats["group_stats"][group]
            group_stat["norm"] += grad_norm ** 2
            group_stat["mean"] += grad_mean
            group_stat["std"] += grad_std
            group_stat["mean_abs"] += grad_mean_abs
            group_stat["max_abs"] = max(group_stat["max_abs"], grad_max_abs)
            group_stat["n_parameters"] += 1
        
        # Finalize overall stats
        stats["overall"]["total_norm"] = np.sqrt(stats["overall"]["total_norm"])
        
        # Finalize group stats
        for group, group_stat in stats["group_stats"].items():
            if group_stat["n_parameters"] > 0:
                group_stat["norm"] = np.sqrt(group_stat["norm"])
                group_stat["mean"] /= group_stat["n_parameters"]
                group_stat["std"] /= group_stat["n_parameters"]
                group_stat["mean_abs"] /= group_stat["n_parameters"]
        
        return stats
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize gradient analysis."""
        if not self.gradient_stats:
            return control
        
        print(f"\n📊 Gradient Analysis Summary:")
        print(f"   Total analyses: {len(self.gradient_stats)}")
        
        # Save gradient statistics
        stats_path = os.path.join(self.output_dir, "gradient_statistics.json")
        with open(stats_path, 'w') as f:
            json.dump(self.gradient_stats, f, indent=2, default=str)
        
        print(f"📁 Gradient statistics saved: {stats_path}")
        
        # Generate trend analysis
        self._generate_gradient_trend_analysis()
        
        return control
    
    def _generate_gradient_trend_analysis(self):
        """Generate gradient trend analysis."""
        if len(self.gradient_stats) < 10:
            return
        
        trends = {}
        groups = list(self.parameter_groups) + ["other"]
        
        for group in groups:
            norms = [gs["group_stats"][group]["norm"] for gs in self.gradient_stats 
                    if gs["group_stats"][group]["n_parameters"] > 0]
            
            if len(norms) >= 10:
                x = np.arange(len(norms))
                slope, intercept = np.polyfit(x, norms, 1)
                
                trends[group] = {
                    "initial_norm": norms[0],
                    "final_norm": norms[-1],
                    "slope": slope,
                    "trend": "decreasing" if slope < -0.001 else "increasing" if slope > 0.001 else "stable",
                    "percent_change": (norms[-1] - norms[0]) / norms[0] * 100 if norms[0] > 0 else 0,
                }
        
        # Save trend analysis
        trends_path = os.path.join(self.output_dir, "gradient_trends.json")
        with open(trends_path, 'w') as f:
            json.dump(trends, f, indent=2, default=str)
        
        print(f"📁 Gradient trends saved: {trends_path}")


class ComputeCostTracker(TrainerCallback):
    """
    Compute cost tracking callback.
    
    Tracks GPU hours, memory usage, and estimates training costs.
    """
    
    def __init__(
        self,
        gpu_hourly_cost: float = 2.99,  # Default for cloud GPU
        output_dir: str = "./experiments/compute_costs",
        wandb_enabled: bool = True,
    ):
        self.gpu_hourly_cost = gpu_hourly_cost
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.start_time = None
        self.total_gpu_seconds = 0
        self.memory_usage = []
        self.cost_estimates = []
        
        print(f"✅ Compute Cost Tracker initialized")
        print(f"   GPU hourly cost: ${gpu_hourly_cost}/hour")
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Start timing."""
        self.start_time = time.time()
        return control
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Track compute usage."""
        if torch.cuda.is_available():
            # Track memory usage periodically
            if state.global_step % 100 == 0:
                memory_stats = {
                    "step": state.global_step,
                    "allocated_gb": torch.cuda.memory_allocated() / 1e9,
                    "reserved_gb": torch.cuda.memory_reserved() / 1e9,
                    "utilization": torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0,
                    "timestamp": time.time(),
                }
                self.memory_usage.append(memory_stats)
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Update cost estimates."""
        if self.start_time is None:
            return control
        
        current_time = time.time()
        elapsed_hours = (current_time - self.start_time) / 3600
        self.total_gpu_seconds = elapsed_hours * 3600
        
        # Estimate cost
        estimated_cost = elapsed_hours * self.gpu_hourly_cost
        
        cost_estimate = {
            "step": state.global_step,
            "elapsed_hours": elapsed_hours,
            "estimated_cost": estimated_cost,
            "gpu_hourly_cost": self.gpu_hourly_cost,
            "timestamp": datetime.now().isoformat(),
        }
        self.cost_estimates.append(cost_estimate)
        
        # Log periodically
        if state.global_step % 500 == 0:
            print(f"\n💰 Compute Cost (Step {state.global_step}):")
            print(f"   Elapsed time: {elapsed_hours:.2f} hours")
            print(f"   Estimated cost: ${estimated_cost:.2f}")
            
            # Log to W&B
            if self.wandb_enabled and wandb.run:
                wandb.log({
                    "cost/elapsed_hours": elapsed_hours,
                    "cost/estimated_cost": estimated_cost,
                    "cost/cumulative_cost": estimated_cost,
                })
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize cost tracking and generate report."""
        if self.start_time is None:
            return control
        
        total_time = time.time() - self.start_time
        total_hours = total_time / 3600
        total_cost = total_hours * self.gpu_hourly_cost
        
        print(f"\n{'='*60}")
        print("COMPUTE COST ANALYSIS")
        print(f"{'='*60}")
        print(f"Total training time: {total_hours:.2f} hours")
        print(f"Estimated GPU cost: ${total_cost:.2f}")
        print(f"GPU hourly rate: ${self.gpu_hourly_cost}/hour")
        
        # Memory analysis
        if self.memory_usage:
            avg_memory = np.mean([m["allocated_gb"] for m in self.memory_usage])
            max_memory = max([m["allocated_gb"] for m in self.memory_usage])
            print(f"Average memory usage: {avg_memory:.1f} GB")
            print(f"Peak memory usage: {max_memory:.1f} GB")
        
        # Generate cost report
        report = self._generate_cost_report(total_hours, total_cost)
        
        # Save report
        report_path = os.path.join(self.output_dir, "compute_cost_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📁 Cost report saved: {report_path}")
        print(f"{'='*60}")
        
        return control
    
    def _generate_cost_report(self, total_hours: float, total_cost: float) -> Dict:
        """Generate comprehensive cost report."""
        report = {
            "total_training_hours": total_hours,
            "total_estimated_cost": total_cost,
            "gpu_hourly_cost": self.gpu_hourly_cost,
            "cost_breakdown": {
                "compute_cost": total_cost,
                "estimated_waste_cost": 0,  # Will be calculated if plateau analysis available
                "potential_savings": 0,
            },
            "efficiency_metrics": {},
            "recommendations": [],
        }
        
        # Add memory statistics
        if self.memory_usage:
            memory_gb = [m["allocated_gb"] for m in self.memory_usage]
            report["memory_statistics"] = {
                "average_gb": float(np.mean(memory_gb)),
                "maximum_gb": float(np.max(memory_gb)),
                "minimum_gb": float(np.min(memory_gb)),
                "samples": len(memory_gb),
            }
        
        # Calculate waste if plateau information is available
        # This would typically come from IES callback analysis
        if hasattr(self, 'plateau_step') and hasattr(self, 'total_steps'):
            waste_ratio = (self.total_steps - self.plateau_step) / self.total_steps
            waste_cost = total_cost * waste_ratio
            potential_savings = waste_cost
            
            report["cost_breakdown"].update({
                "estimated_waste_cost": waste_cost,
                "potential_savings": potential_savings,
                "waste_ratio": waste_ratio,
                "plateau_step": self.plateau_step,
                "total_steps": self.total_steps,
            })
        
        # Generate recommendations
        if total_cost > 100:  # More than $100
            report["recommendations"].append(
                f"High compute cost (${total_cost:.2f}). Consider implementing "
                "early stopping to reduce waste."
            )
        
        if self.memory_usage:
            avg_memory = np.mean([m["allocated_gb"] for m in self.memory_usage])
            if avg_memory > 20:  # More than 20GB average
                report["recommendations"].append(
                    f"High memory usage ({avg_memory:.1f}GB average). "
                    "Consider using gradient checkpointing or model parallelism."
                )
        
        return report


class PowerTelemetryCallback(TrainerCallback):
    """
    GPU power telemetry via nvidia-smi for energy (kWh) measurement.
    
    Samples power.draw at a configurable interval using a background thread,
    integrates watts over wall-clock time to produce cumulative energy in kWh.
    Logs per-step and cumulative energy to W&B and saves a final report.
    """
    
    def __init__(
        self,
        sample_interval_s: float = 1.0,
        gpu_index: int = 0,
        output_dir: str = "./experiments/power_telemetry",
        wandb_enabled: bool = True,
        log_frequency: int = 25,
    ):
        self.sample_interval_s = sample_interval_s
        self.gpu_index = gpu_index
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        self.log_frequency = log_frequency
        
        os.makedirs(output_dir, exist_ok=True)
        
        self._power_samples: List[Dict] = []
        self._sampling_thread = None
        self._stop_event = None
        self._training_start = None
        self._step_energy_start_idx = 0
        
        self.cumulative_kwh = 0.0
        self.step_energies: List[Dict] = []
    
    def _sample_power_loop(self):
        import subprocess
        
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        f"--id={self.gpu_index}",
                        "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True, text=True, timeout=5,
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 3:
                        self._power_samples.append({
                            "timestamp": time.time(),
                            "power_w": float(parts[0]),
                            "temp_c": float(parts[1]),
                            "util_pct": float(parts[2]),
                        })
            except Exception:
                pass
            
            self._stop_event.wait(self.sample_interval_s)
    
    def _start_sampling(self):
        import threading
        self._stop_event = threading.Event()
        self._sampling_thread = threading.Thread(
            target=self._sample_power_loop, daemon=True
        )
        self._sampling_thread.start()
    
    def _stop_sampling(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._sampling_thread is not None:
            self._sampling_thread.join(timeout=5)
    
    def _compute_energy_kwh(self, start_idx: int, end_idx: int) -> float:
        if end_idx - start_idx < 2:
            return 0.0
        
        samples = self._power_samples[start_idx:end_idx]
        total_joules = 0.0
        for i in range(1, len(samples)):
            dt = samples[i]["timestamp"] - samples[i - 1]["timestamp"]
            avg_power = (samples[i]["power_w"] + samples[i - 1]["power_w"]) / 2.0
            total_joules += avg_power * dt
        
        return total_joules / 3_600_000.0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self._training_start = time.time()
        self._start_sampling()
        print(f"  Power telemetry started (GPU {self.gpu_index}, {self.sample_interval_s}s interval)")
        return control
    
    def on_step_end(self, args, state, control, **kwargs):
        current_idx = len(self._power_samples)
        step_kwh = self._compute_energy_kwh(self._step_energy_start_idx, current_idx)
        self.cumulative_kwh += step_kwh
        self._step_energy_start_idx = current_idx
        
        self.step_energies.append({
            "step": state.global_step,
            "step_kwh": step_kwh,
            "cumulative_kwh": self.cumulative_kwh,
        })
        
        if state.global_step % self.log_frequency == 0:
            recent = self._power_samples[-10:] if self._power_samples else []
            avg_power = np.mean([s["power_w"] for s in recent]) if recent else 0
            avg_temp = np.mean([s["temp_c"] for s in recent]) if recent else 0
            avg_util = np.mean([s["util_pct"] for s in recent]) if recent else 0
            
            metrics = {
                "power/current_watts": avg_power,
                "power/temperature_c": avg_temp,
                "power/gpu_utilization_pct": avg_util,
                "power/cumulative_kwh": self.cumulative_kwh,
                "power/step": state.global_step,
            }
            
            if self.wandb_enabled and wandb.run:
                wandb.log(metrics)
            
            print(
                f"  Power (Step {state.global_step}): "
                f"{avg_power:.1f}W | {self.cumulative_kwh:.6f} kWh | "
                f"{avg_temp:.0f}C | {avg_util:.0f}% util"
            )
        
        return control
    
    def on_train_end(self, args, state, control, **kwargs):
        self._stop_sampling()
        
        total_time_s = time.time() - self._training_start if self._training_start else 0
        total_samples = len(self._power_samples)
        
        all_power = [s["power_w"] for s in self._power_samples] if self._power_samples else [0]
        
        report = {
            "total_training_time_s": total_time_s,
            "total_energy_kwh": self.cumulative_kwh,
            "total_power_samples": total_samples,
            "power_statistics": {
                "mean_watts": float(np.mean(all_power)),
                "max_watts": float(np.max(all_power)),
                "min_watts": float(np.min(all_power)),
                "std_watts": float(np.std(all_power)),
            },
            "per_step_energy": self.step_energies,
            "gpu_index": self.gpu_index,
            "sample_interval_s": self.sample_interval_s,
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
        print(f"  Total energy: {self.cumulative_kwh:.6f} kWh")
        print(f"  Training time: {total_time_s:.1f}s ({total_time_s/3600:.3f}h)")
        print(f"  Avg power draw: {np.mean(all_power):.1f}W")
        print(f"  Peak power draw: {np.max(all_power):.1f}W")
        print(f"  Samples collected: {total_samples}")
        print(f"  Report saved: {report_path}")
        print(f"{'='*60}")
        
        if self.wandb_enabled and wandb.run:
            wandb.log({
                "power/total_kwh": self.cumulative_kwh,
                "power/total_training_hours": total_time_s / 3600,
                "power/avg_watts": float(np.mean(all_power)),
                "power/peak_watts": float(np.max(all_power)),
            })
        
        return control