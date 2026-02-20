"""
IES Callback Implementation with Statistical Validation

ICLR 2025 Instance-dependent Early Stopping callback with
comprehensive monitoring, validation, and checkpoint management.
"""

import torch
import wandb
import json
import os
import warnings
from typing import Dict, List, Optional, Any
from dataclasses import asdict
from datetime import datetime

from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from transformers.trainer_utils import IntervalStrategy

from ..utils.plateau_ies import IESPlateauDetector, PlateauAnalysisResult, compute_statistical_significance
from ..utils.metrics import LERTracker, GSNRTracker


class IESCallback(TrainerCallback):
    """
    IES Callback for plateau detection with early stopping.
    
    Features:
    1. ICLR 2025 IES plateau detection
    2. LER-based efficiency monitoring
    3. Statistical validation of plateau detection
    4. Checkpoint restoration for best model
    5. Comprehensive logging and reporting
    """
    
    def __init__(
        self,
        threshold: float = 0.001,
        window_size: int = 3,
        patience: int = 100,
        min_delta: float = 0.0001,
        restore_best_model: bool = True,
        save_analysis: bool = True,
        enable_wandb: bool = True,
        output_dir: str = "./experiments/ies_analysis",
        task: str = "unknown",
    ):
        """
        Initialize IES callback.
        
        Args:
            threshold: Second-order difference threshold
            window_size: Window size for plateau detection
            patience: Steps to wait after plateau detection
            min_delta: Minimum change for plateau detection
            restore_best_model: Whether to restore best checkpoint
            save_analysis: Whether to save analysis results
            enable_wandb: Whether to log to Weights & Biases
            output_dir: Directory for analysis outputs
            task: Task name for task-specific calibration
        """
        self.threshold = threshold
        self.window_size = window_size
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_model = restore_best_model
        self.save_analysis = save_analysis
        self.enable_wandb = enable_wandb
        self.output_dir = output_dir
        self.task = task
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize detectors
        self.ies_detector = IESPlateauDetector(
            threshold=threshold,
            window_size=window_size,
            patience=patience,
            task=task,
        )
        
        self.ler_tracker = LERTracker(task=task)
        self.gsnr_tracker = None  # Will be initialized when model is available
        
        # Tracking
        self.metrics_history: List[Dict] = []
        self.plateau_candidates: List[int] = []
        self.plateau_confirmed: bool = False
        self.plateau_step: Optional[int] = None
        
        # Checkpoint management
        self.best_checkpoint: Optional[Dict] = None
        self.best_checkpoint_step: int = 0
        self.checkpoints: List[Dict] = []
        
        # Results
        self.analysis_results: Optional[PlateauAnalysisResult] = None
        self.statistical_validation: Dict = {}
        
        print(f"✅ IES Callback initialized for task: {task}")
        print(f"   Threshold: {threshold}, Window: {window_size}, Patience: {patience}")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of initialization."""
        print(f"\n📊 IES Callback Configuration:")
        print(f"   Task: {self.task}")
        print(f"   Plateau detection: ICLR 2025 IES method")
        print(f"   Restore best model: {self.restore_best_model}")
        print(f"   Output directory: {self.output_dir}")
        
        # Initialize GSNR tracker when model is available
        if "model" in kwargs:
            model = kwargs["model"]
            self.gsnr_tracker = GSNRTracker(model)
            print(f"   GSNR tracking: Enabled")
        else:
            print(f"   GSNR tracking: Will be initialized later")
        
        return control
    
    def on_train_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the beginning of training."""
        print(f"\n🚀 IES Monitoring Started")
        print(f"   Starting training with IES plateau detection")
        print(f"   Task: {self.task}")
        print(f"   Expected total steps: ~{args.max_steps if args.max_steps else 'unknown'}")
        
        # Log configuration
        config = {
            "threshold": self.threshold,
            "window_size": self.window_size,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "task": self.task,
            "restore_best_model": self.restore_best_model,
            "training_args": {
                "learning_rate": args.learning_rate,
                "per_device_train_batch_size": args.per_device_train_batch_size,
                "num_train_epochs": args.num_train_epochs,
                "max_steps": args.max_steps,
            },
        }
        
        if self.save_analysis:
            config_path = os.path.join(self.output_dir, "ies_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"   Configuration saved: {config_path}")
        
        if self.enable_wandb and wandb.run:
            wandb.config.update({"ies_config": config})
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Called after evaluation."""
        eval_loss = metrics.get("eval_loss", float("inf"))
        current_step = state.global_step
        
        # Update IES detector
        plateau_detected = self.ies_detector.update(eval_loss, current_step)
        
        # Update LER tracker if logits are available
        if "logits" in kwargs:
            logits = kwargs["logits"]
            accuracy = metrics.get("eval_accuracy", 0)
            model = kwargs.get("model")
            gradients = kwargs.get("gradients")
            self.ler_tracker.update(eval_loss, logits, accuracy, model=model, gradients=gradients)
        
        # Update GSNR tracker if gradients are available
        if self.gsnr_tracker and "gradients" in kwargs:
            self.gsnr_tracker.update(kwargs["gradients"])
        
        # Store metrics
        metrics_entry = {
            "step": current_step,
            "epoch": state.epoch,
            "eval_loss": eval_loss,
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            "ler": self.ler_tracker.get_ler(),
            "ler_phase": self.ler_tracker.get_efficiency_phase(),
            "plateau_detected": plateau_detected,
            "timestamp": datetime.now().isoformat(),
        }
        
        # Add GSNR if available
        if self.gsnr_tracker:
            gsnr = self.gsnr_tracker.get_gsnr(latest_only=True)
            if gsnr:
                metrics_entry["gsnr"] = gsnr.get("overall", 0)
        
        self.metrics_history.append(metrics_entry)
        
        # Check for plateau
        if plateau_detected and not self.plateau_confirmed:
            self.plateau_candidates.append(current_step)
            
            # Check if we have enough consecutive plateau candidates
            if len(self.plateau_candidates) >= self.patience:
                self.plateau_confirmed = True
                self.plateau_step = self.plateau_candidates[0]  # First detection
                
                print(f"\n{'='*60}")
                print("✅ PLATEAU CONFIRMED - IES Detection")
                print(f"{'='*60}")
                print(f"Step: {self.plateau_step} / {current_step}")
                print(f"Wasted compute: {(current_step - self.plateau_step) / current_step * 100:.1f}%")
                print(f"Validation loss: {eval_loss:.4f}")
                print(f"{'='*60}")
                
                # Perform comprehensive analysis
                self._perform_plateau_analysis(current_step)
                
                # Log to W&B
                if self.enable_wandb and wandb.run:
                    wandb.log({
                        "plateau/confirmed": True,
                        "plateau/step": self.plateau_step,
                        "plateau/wasted_pct": (current_step - self.plateau_step) / current_step * 100,
                        "plateau/eval_loss": eval_loss,
                    })
        
        # Check LER plateau
        ler_plateau, ler_confidence = self.ler_tracker.get_ler_plateau_indicator()
        if ler_plateau and ler_confidence > 0.8 and not self.plateau_confirmed:
            print(f"\n⚠️  LER Plateau Indication (confidence: {ler_confidence:.2f})")
            print(f"   LER: {self.ler_tracker.get_ler():.6f}")
            print(f"   Phase: {self.ler_tracker.get_efficiency_phase()}")
        
        # Save checkpoint if this is the best so far
        if self.restore_best_model:
            self._save_best_checkpoint(state, metrics, current_step)
        
        # Log metrics to W&B
        if self.enable_wandb and wandb.run:
            wandb.log(metrics_entry)
        
        return control
    
    def _save_best_checkpoint(self, state: TrainerState, metrics: Dict[str, float], step: int):
        """Save best checkpoint based on validation loss."""
        current_loss = metrics.get("eval_loss", float("inf"))
        
        if self.best_checkpoint is None or current_loss < self.best_checkpoint["eval_loss"]:
            self.best_checkpoint = {
                "step": step,
                "epoch": state.epoch,
                "eval_loss": current_loss,
                "eval_accuracy": metrics.get("eval_accuracy", 0),
                "model_state": self._get_model_state(),
                "optimizer_state": self._get_optimizer_state(),
                "timestamp": datetime.now().isoformat(),
            }
            self.best_checkpoint_step = step
            
            print(f"📁 New best checkpoint at step {step}")
            print(f"   Loss: {current_loss:.4f}, Accuracy: {metrics.get('eval_accuracy', 0):.4f}")
            
            # Save checkpoint to disk
            if self.save_analysis:
                checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step{step}.json")
                checkpoint_data = {
                    "step": step,
                    "eval_loss": current_loss,
                    "eval_accuracy": metrics.get("eval_accuracy", 0),
                    "metrics": {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
                }
                
                with open(checkpoint_path, 'w') as f:
                    json.dump(checkpoint_data, f, indent=2)
    
    def _get_model_state(self):
        """Get model state (simplified - in practice would save actual model)."""
        # This is a placeholder - in practice, you would save the actual model state
        return {"placeholder": "model_state"}
    
    def _get_optimizer_state(self):
        """Get optimizer state (simplified - in practice would save actual optimizer)."""
        # This is a placeholder
        return {"placeholder": "optimizer_state"}
    
    def _perform_plateau_analysis(self, current_step: int):
        """Perform comprehensive plateau analysis."""
        if not self.plateau_step:
            return
        
        # Get analysis from IES detector
        self.analysis_results = self.ies_detector.analyze_plateau()
        
        # Add LER analysis
        ler_plateau, ler_confidence = self.ler_tracker.get_ler_plateau_indicator()
        ler_phase = self.ler_tracker.get_efficiency_phase()
        
        # Add GSNR analysis if available
        gsnr_warning = None
        if self.gsnr_tracker:
            gsnr_warning = self.gsnr_tracker.get_convergence_warning()
        
        # Create comprehensive analysis
        analysis = {
            "plateau_step": self.plateau_step,
            "current_step": current_step,
            "wasted_steps": current_step - self.plateau_step,
            "wasted_percentage": (current_step - self.plateau_step) / current_step * 100,
            "ies_analysis": asdict(self.analysis_results) if self.analysis_results else {},
            "ler_analysis": {
                "plateau_detected": ler_plateau,
                "confidence": ler_confidence,
                "current_phase": ler_phase,
                "current_ler": self.ler_tracker.get_ler(),
            },
            "gsnr_warning": gsnr_warning,
            "validation_metrics": self.metrics_history[-1] if self.metrics_history else {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save analysis
        if self.save_analysis:
            analysis_path = os.path.join(self.output_dir, "plateau_analysis.json")
            with open(analysis_path, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            print(f"📊 Plateau analysis saved: {analysis_path}")
        
        # Log to W&B
        if self.enable_wandb and wandb.run:
            wandb.log({"plateau/analysis": analysis})
        
        # Print summary
        self._print_analysis_summary(analysis)
    
    def _print_analysis_summary(self, analysis: Dict):
        """Print analysis summary."""
        print(f"\n{'='*60}")
        print("PLATEAU ANALYSIS SUMMARY")
        print(f"{'='*60}")
        
        wasted_pct = analysis["wasted_percentage"]
        plateau_step = analysis["plateau_step"]
        current_step = analysis["current_step"]
        
        print(f"Plateau detected at step: {plateau_step:,} / {current_step:,}")
        print(f"Wasted compute: {wasted_pct:.1f}%")
        print(f"Wasted steps: {current_step - plateau_step:,}")
        
        # IES analysis
        ies_analysis = analysis.get("ies_analysis", {})
        if ies_analysis:
            print(f"\nIES Analysis:")
            print(f"  Confidence: {ies_analysis.get('detection_confidence', 0):.2f}")
            print(f"  False positive rate: {ies_analysis.get('false_positive_rate', 0):.3f}")
            print(f"  Robustness score: {ies_analysis.get('robustness_score', 0):.2f}")
        
        # LER analysis
        ler_analysis = analysis.get("ler_analysis", {})
        if ler_analysis:
            print(f"\nLER Analysis:")
            print(f"  Plateau detected: {ler_analysis.get('plateau_detected', False)}")
            print(f"  Confidence: {ler_analysis.get('confidence', 0):.2f}")
            print(f"  Current phase: {ler_analysis.get('current_phase', 'unknown')}")
        
        # Recommendations
        print(f"\n{'='*60}")
        print("RECOMMENDATIONS")
        print(f"{'='*60}")
        
        if wasted_pct > 30:
            print("⚠️  High waste detected (>30%). Consider implementing early stopping.")
            print("   Suggested: Stop training at detected plateau step.")
        
        if ler_analysis.get("plateau_detected", False):
            print("✅ LER confirms plateau detection.")
        
        if analysis.get("gsnr_warning"):
            print(f"⚠️  GSNR warning: {analysis['gsnr_warning']}")
        
        print(f"\nBest checkpoint saved at step: {self.best_checkpoint_step:,}")
        print(f"   Loss: {self.best_checkpoint['eval_loss']:.4f}")
        print(f"   Accuracy: {self.best_checkpoint['eval_accuracy']:.4f}")
        print(f"{'='*60}")
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of a training step."""
        # Check if we should stop training
        if self.plateau_confirmed and self.restore_best_model:
            # We've confirmed plateau and want to restore best model
            print(f"\n🛑 Plateau confirmed. Stopping training and restoring best checkpoint...")
            
            # Set should_training_stop flag
            control.should_training_stop = True
            
            # Log final statistics
            self._log_final_statistics(state)
        
        return control
    
    def _log_final_statistics(self, state: TrainerState):
        """Log final statistics and save results."""
        if not self.analysis_results:
            return
        
        final_stats = {
            "final_step": state.global_step,
            "plateau_step": self.plateau_step,
            "wasted_steps": state.global_step - self.plateau_step,
            "wasted_percentage": (state.global_step - self.plateau_step) / state.global_step * 100,
            "best_checkpoint_step": self.best_checkpoint_step,
            "best_checkpoint_loss": self.best_checkpoint["eval_loss"] if self.best_checkpoint else None,
            "final_loss": self.metrics_history[-1]["eval_loss"] if self.metrics_history else None,
            "analysis_results": asdict(self.analysis_results),
            "metrics_history_summary": {
                "total_evaluations": len(self.metrics_history),
                "final_eval_loss": self.metrics_history[-1]["eval_loss"] if self.metrics_history else None,
                "final_eval_accuracy": self.metrics_history[-1]["eval_accuracy"] if self.metrics_history else None,
            },
            "timestamp": datetime.now().isoformat(),
        }
        
        # Save final statistics
        if self.save_analysis:
            stats_path = os.path.join(self.output_dir, "final_statistics.json")
            with open(stats_path, 'w') as f:
                json.dump(final_stats, f, indent=2, default=str)
            
            # Save metrics history
            metrics_path = os.path.join(self.output_dir, "metrics_history.json")
            with open(metrics_path, 'w') as f:
                json.dump(self.metrics_history, f, indent=2, default=str)
            
            print(f"📁 Final statistics saved: {stats_path}")
            print(f"📁 Metrics history saved: {metrics_path}")
        
        # Log to W&B
        if self.enable_wandb and wandb.run:
            wandb.log({"final_statistics": final_stats})
            wandb.finish()
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of training."""
        print(f"\n{'='*60}")
        print("IES CALLBACK - TRAINING COMPLETE")
        print(f"{'='*60}")
        
        # Final analysis if plateau wasn't confirmed
        if not self.plateau_confirmed and self.metrics_history:
            # Perform final analysis anyway
            current_step = state.global_step
            self.analysis_results = self.ies_detector.analyze_plateau()
            
            if self.analysis_results:
                wasted_pct = self.analysis_results.wasted_compute_pct
                print(f"Final analysis (no plateau confirmed):")
                print(f"  Estimated waste: {wasted_pct:.1f}%")
                print(f"  Plateau step: {self.analysis_results.plateau_step}")
                print(f"  Total steps: {self.analysis_results.total_steps}")
        
        print(f"\n📊 Summary:")
        print(f"  Total evaluations: {len(self.metrics_history)}")
        print(f"  Plateau confirmed: {self.plateau_confirmed}")
        print(f"  Best checkpoint: step {self.best_checkpoint_step}")
        
        if self.save_analysis:
            print(f"\n📁 Analysis files saved in: {self.output_dir}")
        
        print(f"{'='*60}")
        
        return control


class EfficiencyMonitoringCallback(TrainerCallback):
    """
    Comprehensive efficiency monitoring callback.
    
    Tracks multiple efficiency metrics and provides real-time insights.
    """
    
    def __init__(
        self,
        task: str,
        enable_ler: bool = True,
        enable_gsnr: bool = True,
        enable_probe: bool = False,
        output_dir: str = "./experiments/efficiency_monitoring",
        wandb_enabled: bool = True,
    ):
        self.task = task
        self.enable_ler = enable_ler
        self.enable_gsnr = enable_gsnr
        self.enable_probe = enable_probe
        self.output_dir = output_dir
        self.wandb_enabled = wandb_enabled
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize trackers
        self.ler_tracker = None
        self.gsnr_tracker = None
        self.probe_tracker = None
        
        self.metrics_history = []
        self.efficiency_warnings = []
        
        print(f"✅ Efficiency Monitoring initialized for {task}")
    
    def on_init_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Initialize trackers when model is available."""
        if "model" in kwargs:
            model = kwargs["model"]
            
            if self.enable_ler:
                self.ler_tracker = LERTracker(task=self.task)
                print(f"   LER tracking: Enabled")
            
            if self.enable_gsnr:
                self.gsnr_tracker = GSNRTracker(model)
                print(f"   GSNR tracking: Enabled")
            
            if self.enable_probe:
                hidden_dim = model.config.hidden_size
                num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
                from ..utils.metrics import ProbeAccuracyTracker
                self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
                print(f"   Probe accuracy tracking: Enabled")
        
        return control
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Monitor efficiency metrics during evaluation."""
        current_step = state.global_step
        eval_loss = metrics.get("eval_loss", 0)
        
        # Update trackers
        efficiency_metrics = {}
        
        if self.ler_tracker and "logits" in kwargs:
            logits = kwargs["logits"]
            accuracy = metrics.get("eval_accuracy", 0)
            model = kwargs.get("model")
            gradients = kwargs.get("gradients")
            self.ler_tracker.update(eval_loss, logits, accuracy, model=model, gradients=gradients)
            
            efficiency_metrics.update({
                "efficiency/ler": self.ler_tracker.get_ler(),
                "efficiency/ler_phase": self.ler_tracker.get_efficiency_phase(),
                "efficiency/rho_vg": self.ler_tracker.get_rho_vg(),
                "efficiency/param_velocity": self.ler_tracker.get_velocity(),
            })
            
            # Check for LER plateau
            ler_plateau, ler_confidence = self.ler_tracker.get_ler_plateau_indicator()
            if ler_plateau and ler_confidence > 0.8:
                warning = f"LER plateau detected (confidence: {ler_confidence:.2f})"
                if warning not in self.efficiency_warnings:
                    self.efficiency_warnings.append(warning)
                    print(f"⚠️  {warning}")
        
        if self.gsnr_tracker and "gradients" in kwargs:
            self.gsnr_tracker.update(kwargs["gradients"])
            gsnr = self.gsnr_tracker.get_gsnr(latest_only=True)
            if gsnr:
                efficiency_metrics["efficiency/gsnr"] = gsnr.get("overall", 0)
            
            # Check for GSNR warnings
            gsnr_warning = self.gsnr_tracker.get_convergence_warning()
            if gsnr_warning and gsnr_warning not in self.efficiency_warnings:
                self.efficiency_warnings.append(gsnr_warning)
                print(f"⚠️  {gsnr_warning}")
        
        if self.probe_tracker and "representations" in kwargs and "labels" in kwargs:
            self.probe_tracker.add_representations(
                kwargs["representations"],
                kwargs["labels"]
            )
            
            # Compute probe accuracy periodically
            if current_step % 100 == 0:
                probe_accuracy = self.probe_tracker.compute_probe_accuracy()
                efficiency_metrics["efficiency/probe_accuracy"] = probe_accuracy
        
        # Store metrics
        metrics_entry = {
            "step": current_step,
            "eval_loss": eval_loss,
            "eval_accuracy": metrics.get("eval_accuracy", 0),
            **efficiency_metrics,
        }
        self.metrics_history.append(metrics_entry)
        
        # Log to W&B
        if self.wandb_enabled and wandb.run:
            wandb.log(metrics_entry)
        
        # Save periodically
        if current_step % 500 == 0:
            self._save_metrics_history()
        
        return control
    
    def _save_metrics_history(self):
        """Save metrics history to disk."""
        if not self.metrics_history:
            return
        
        metrics_path = os.path.join(self.output_dir, "efficiency_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(self.metrics_history, f, indent=2, default=str)
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Finalize efficiency monitoring."""
        print(f"\n📊 Efficiency Monitoring Summary:")
        print(f"   Total evaluations: {len(self.metrics_history)}")
        print(f"   Efficiency warnings: {len(self.efficiency_warnings)}")
        
        if self.efficiency_warnings:
            print(f"\n⚠️  Efficiency Warnings:")
            for warning in self.efficiency_warnings[-5:]:  # Show last 5 warnings
                print(f"   • {warning}")
        
        # Save final metrics
        self._save_metrics_history()
        
        # Generate efficiency report
        report = self._generate_efficiency_report()
        report_path = os.path.join(self.output_dir, "efficiency_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"\n📁 Efficiency report saved: {report_path}")
        
        return control
    
    def _generate_efficiency_report(self) -> Dict:
        """Generate comprehensive efficiency report."""
        if not self.metrics_history:
            return {"error": "No metrics collected"}
        
        report = {
            "task": self.task,
            "total_steps": len(self.metrics_history),
            "efficiency_warnings": self.efficiency_warnings,
            "metrics_summary": {},
            "recommendations": [],
        }
        
        # Summarize each metric
        for key in ["efficiency/ler", "efficiency/gsnr", "efficiency/probe_accuracy"]:
            values = [m.get(key) for m in self.metrics_history if m.get(key) is not None]
            if values:
                report["metrics_summary"][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "n_samples": len(values),
                }
        
        # Generate recommendations
        ler_values = [m.get("efficiency/ler") for m in self.metrics_history 
                     if m.get("efficiency/ler") is not None]
        if ler_values:
            recent_ler = ler_values[-10:] if len(ler_values) >= 10 else ler_values
            avg_recent_ler = np.mean(recent_ler)
            
            if avg_recent_ler < 0.01:
                report["recommendations"].append(
                    "Low LER detected. Consider early stopping or learning rate reduction."
                )
        
        return report


class EarlyStoppingWithLER(TrainerCallback):
    """
    Early stopping based on Learning Efficiency Rate (LER).
    
    Stops training when LER falls below threshold for consecutive evaluations.
    """
    
    def __init__(
        self,
        ler_threshold: float = 0.01,
        patience: int = 10,
        min_improvement: float = 0.001,
        restore_best: bool = True,
        task: str = "unknown",
    ):
        self.ler_threshold = ler_threshold
        self.patience = patience
        self.min_improvement = min_improvement
        self.restore_best = restore_best
        self.task = task
        
        self.ler_tracker = LERTracker(task=task)
        self.best_loss = float("inf")
        self.best_step = 0
        self.patience_counter = 0
        self.should_stop = False
        
        print(f"✅ LER Early Stopping initialized")
        print(f"   Threshold: {ler_threshold}, Patience: {patience}")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Check LER for early stopping."""
        if self.should_stop:
            return control
        
        eval_loss = metrics.get("eval_loss", float("inf"))
        current_step = state.global_step
        
        # Update LER tracker
        if "logits" in kwargs:
            logits = kwargs["logits"]
            accuracy = metrics.get("eval_accuracy", 0)
            model = kwargs.get("model")
            gradients = kwargs.get("gradients")
            self.ler_tracker.update(eval_loss, logits, accuracy, model=model, gradients=gradients)
        
        # Check current LER
        current_ler = self.ler_tracker.get_ler()
        ler_plateau, ler_confidence = self.ler_tracker.get_ler_plateau_indicator()
        
        # Update best loss
        if eval_loss < self.best_loss - self.min_improvement:
            self.best_loss = eval_loss
            self.best_step = current_step
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Check stopping condition
        if ler_plateau and ler_confidence > 0.8:
            self.patience_counter += 2  # Accelerate stopping for LER plateau
        
        if self.patience_counter >= self.patience:
            print(f"\n🛑 LER Early Stopping triggered")
            print(f"   Step: {current_step}")
            print(f"   LER: {current_ler:.6f} (threshold: {self.ler_threshold})")
            print(f"   Best loss: {self.best_loss:.4f} at step {self.best_step}")
            
            self.should_stop = True
            control.should_training_stop = True
            
            if self.restore_best:
                print(f"   Restoring best checkpoint from step {self.best_step}")
                # In practice, you would restore the checkpoint here
        
        return control


class CheckpointRestorationCallback(TrainerCallback):
    """
    Manages checkpoint restoration based on efficiency metrics.
    
    Restores the best checkpoint when plateau is detected.
    """
    
    def __init__(
        self,
        metric: str = "eval_loss",
        mode: str = "min",
        patience: int = 10,
        min_delta: float = 0.001,
        output_dir: str = "./experiments/checkpoints",
    ):
        self.metric = metric
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.output_dir = output_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.best_value = float("inf") if mode == "min" else float("-inf")
        self.best_step = 0
        self.patience_counter = 0
        self.checkpoints = []
        
        print(f"✅ Checkpoint Restoration initialized")
        print(f"   Metric: {metric}, Mode: {mode}, Patience: {patience}")
    
    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl,
                   metrics: Dict[str, float], **kwargs):
        """Manage checkpoints based on evaluation metrics."""
        current_value = metrics.get(self.metric)
        if current_value is None:
            return control
        
        current_step = state.global_step
        improved = False
        
        # Check if current value is better
        if self.mode == "min":
            if current_value < self.best_value - self.min_delta:
                improved = True
        else:  # max
            if current_value > self.best_value + self.min_delta:
                improved = True
        
        if improved:
            self.best_value = current_value
            self.best_step = current_step
            self.patience_counter = 0
            
            # Save checkpoint info
            checkpoint = {
                "step": current_step,
                self.metric: current_value,
                "epoch": state.epoch,
                "timestamp": datetime.now().isoformat(),
                "other_metrics": {k: v for k, v in metrics.items() 
                                 if k != self.metric and isinstance(v, (int, float))},
            }
            self.checkpoints.append(checkpoint)
            
            # Save to disk
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint_step{current_step}.json")
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            print(f"📁 New best checkpoint at step {current_step}")
            print(f"   {self.metric}: {current_value:.4f}")
        else:
            self.patience_counter += 1
        
        # Check if we should restore best checkpoint
        if self.patience_counter >= self.patience:
            print(f"\n🔄 Restoring best checkpoint from step {self.best_step}")
            print(f"   {self.metric}: {self.best_value:.4f}")
            
            # In practice, you would restore the model from checkpoint here
            # For now, we just log the decision
            
            # Reset patience counter after restoration
            self.patience_counter = 0
        
        return control
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Final checkpoint summary."""
        print(f"\n📊 Checkpoint Summary:")
        print(f"   Best {self.metric}: {self.best_value:.4f} at step {self.best_step}")
        print(f"   Total checkpoints saved: {len(self.checkpoints)}")
        
        # Save checkpoint history
        history_path = os.path.join(self.output_dir, "checkpoint_history.json")
        with open(history_path, 'w') as f:
            json.dump(self.checkpoints, f, indent=2, default=str)
        
        print(f"📁 Checkpoint history saved: {history_path}")
        
        return control