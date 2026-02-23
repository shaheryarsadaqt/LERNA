"""
LERNA Switching Logic - Logs all statistics to W&B for real-time visualization.
"""

import torch
import wandb
from transformers import TrainerCallback
import logging

logger = logging.getLogger(__name__)

class LERNASwitchingCallback(TrainerCallback):
    """
    Implements LERNA's core innovation with full W&B logging.
    All switching statistics appear in your W&B dashboard.
    """
    
    def __init__(self, ler_tracker, threshold=1e-5, min_step=100, wandb_enabled=True):
        self.ler_tracker = ler_tracker
        self.threshold = threshold
        self.min_step = min_step
        self.wandb_enabled = wandb_enabled
        
        self.steps_skipped = 0
        self.total_energy_saved = 0.0
        self.plateau_steps = []
        self.active_skipping = False
        self.step_log = []
        
        # For tracking accuracy before/after switching
        self.last_accuracy = 0
        self.accuracy_before_skip = []
        self.accuracy_after_skip = []
        
    def on_step_begin(self, args, state, control, **kwargs):
        """Check if we should skip this step"""
        if state.global_step < self.min_step:
            return control
            
        # Get current LER
        diag = self.ler_tracker.get_diagnostics()
        current_ler = diag.get('ler')
        current_vel = diag.get('param_velocity', 0)
        current_rho = diag.get('rho_vg', 0)
        current_phase = diag.get('phase', 'unknown')
        
        if current_ler is None:
            return control
            
        # Plateau detection
        should_skip = current_ler < self.threshold
        
        if should_skip:
            if not self.active_skipping:
                logger.info(f"📉 Plateau detected at step {state.global_step}: LER={current_ler:.2e}")
                if self.wandb_enabled:
                    wandb.log({
                        "lerna/plateau_detected": state.global_step,
                        "lerna/plateau_ler": current_ler,
                        "lerna/plateau_step": state.global_step
                    })
                self.active_skipping = True
                self.plateau_steps.append(state.global_step)
            
            # Skip this step
            control.should_skip_backward = True
            self.steps_skipped += 1
            
            # Estimate energy saved (rough estimate: backward pass ~60% of step)
            step_energy = 0.0006  # ~0.0006 kWh per skipped step on 5090
            self.total_energy_saved += step_energy
            
            # Log skipping event
            if self.wandb_enabled and state.global_step % 10 == 0:
                wandb.log({
                    "lerna/steps_skipped": self.steps_skipped,
                    "lerna/energy_saved_kwh": self.total_energy_saved,
                    "lerna/skip_ratio": self.steps_skipped / max(state.global_step, 1),
                    "lerna/current_ler": current_ler,
                    "lerna/current_velocity": current_vel,
                    "lerna/current_rho_vg": current_rho,
                    "lerna/phase": current_phase,
                    "lerna/active": 1,
                    "step": state.global_step
                })
        else:
            if self.active_skipping:
                # Just exited a plateau
                if self.wandb_enabled:
                    wandb.log({
                        "lerna/plateau_ended": state.global_step,
                        "lerna/plateau_duration": state.global_step - self.plateau_steps[-1] if self.plateau_steps else 0,
                        "step": state.global_step
                    })
            self.active_skipping = False
            if self.wandb_enabled and state.global_step % 10 == 0:
                wandb.log({
                    "lerna/current_ler": current_ler,
                    "lerna/current_velocity": current_vel,
                    "lerna/current_rho_vg": current_rho,
                    "lerna/phase": current_phase,
                    "lerna/active": 0,
                    "step": state.global_step
                })
            
        return control
    
    def on_step_end(self, args, state, control, model=None, **kwargs):
        """If we skipped backward, do momentum extrapolation"""
        if hasattr(control, 'should_skip_backward') and control.should_skip_backward:
            if model is not None and hasattr(args, 'optimizer'):
                optimizer = args.optimizer
                lr = args.learning_rate
                
                # Momentum extrapolation
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            if hasattr(optimizer, 'state') and param in optimizer.state:
                                p_state = optimizer.state[param]
                                if 'momentum_buffer' in p_state:
                                    momentum = p_state['momentum_buffer']
                                    param.data -= lr * momentum
                                else:
                                    param.data -= lr * param.grad
                            else:
                                param.data -= lr * param.grad
                
                optimizer.zero_grad(set_to_none=True)
        
        return control
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Track accuracy during evaluation"""
        if metrics and 'eval_accuracy' in metrics:
            self.last_accuracy = metrics['eval_accuracy']
            
            if self.active_skipping:
                self.accuracy_during_skip.append(self.last_accuracy)
            else:
                self.accuracy_during_normal.append(self.last_accuracy)
                
            if self.wandb_enabled:
                wandb.log({
                    "lerna/accuracy_during_skip": self.last_accuracy if self.active_skipping else 0,
                    "lerna/accuracy_during_normal": self.last_accuracy if not self.active_skipping else 0,
                    "lerna/skipping_active": int(self.active_skipping),
                    "step": state.global_step
                })
    
    def on_train_end(self, args, state, control, **kwargs):
        """Report and log final switching statistics"""
        total_steps = state.global_step
        skip_ratio = self.steps_skipped / max(total_steps, 1)
        
        # Calculate accuracy impact
        avg_acc_during_skip = sum(self.accuracy_during_skip) / max(len(self.accuracy_during_skip), 1)
        avg_acc_during_normal = sum(self.accuracy_during_normal) / max(len(self.accuracy_during_normal), 1)
        acc_diff = avg_acc_during_skip - avg_acc_during_normal
        
        # Print summary
        print("\n" + "="*60)
        print("📊 LERNA SWITCHING STATISTICS")
        print("="*60)
        print(f"Total steps: {total_steps}")
        print(f"Steps skipped: {self.steps_skipped} ({skip_ratio*100:.1f}%)")
        print(f"Estimated energy saved: {self.total_energy_saved:.6f} kWh")
        print(f"Plateau steps: {self.plateau_steps[:10]}")
        print(f"Accuracy during skipping: {avg_acc_during_skip:.4f}")
        print(f"Accuracy during normal: {avg_acc_during_normal:.4f}")
        print(f"Accuracy difference: {acc_diff:+.4f}")
        print("="*60)
        
        # Log final metrics to W&B
        if self.wandb_enabled:
            wandb.log({
                "lerna/final/steps_skipped": self.steps_skipped,
                "lerna/final/skip_ratio": skip_ratio,
                "lerna/final/energy_saved_kwh": self.total_energy_saved,
                "lerna/final/num_plateaus": len(self.plateau_steps),
                "lerna/final/avg_acc_during_skip": avg_acc_during_skip,
                "lerna/final/avg_acc_during_normal": avg_acc_during_normal,
                "lerna/final/acc_difference": acc_diff
            })
            
            # Create a table of plateau steps
            if self.plateau_steps:
                plateau_table = wandb.Table(
                    columns=["step", "energy_saved_kwh"],
                    data=[[step, 0.0006] for step in self.plateau_steps[:100]]
                )
                wandb.log({"lerna/plateau_steps": plateau_table})
        
        # Save local stats
        stats = {
            "total_steps": total_steps,
            "steps_skipped": self.steps_skipped,
            "skip_ratio": skip_ratio,
            "energy_saved_kwh": self.total_energy_saved,
            "plateau_steps": self.plateau_steps,
            "avg_acc_during_skip": avg_acc_during_skip,
            "avg_acc_during_normal": avg_acc_during_normal,
            "acc_difference": acc_diff
        }
        
        import json
        import os
        stats_path = os.path.join(args.output_dir, "lerna_switching_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        print(f"Statistics saved to {stats_path}")
        
        return control
