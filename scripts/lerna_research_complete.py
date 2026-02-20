# Create the new comprehensive research script
#!/usr/bin/env python3
"""
LERNA — Complete Research Pipeline
ModernBERT Fine-Tuning with GSNR/LER Tracking + Plateau Detection
Optimized for RTX 3050 4GB VRAM
Fixes ALL 6 critical failures from analysis
"""

# ══════════════════════════════════════════════════════════════════
# MUST BE BEFORE ANY TORCH IMPORTS
# ══════════════════════════════════════════════════════════════════
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import math
import time
import copy
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
torch._dynamo.config.disable = True

import wandb
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset, load_metric
from scipy import stats


# ══════════════════════════════════════════════════════════════════
# CONFIGURATION — RTX 3050 4GB Optimized with FULL GLUE TASKS
# ══════════════════════════════════════════════════════════════════
CONFIG = {
    # ---- Model ----
    "model_name": "answerdotai/ModernBERT-base",  # 149M parameters, fits 4GB

    # ---- FULL GLUE Tasks (no sampling, use official splits) ----
    # FIX 1: Using GLUE tasks instead of IMDB
    "tasks": {
        "sst2": {
            "dataset": "glue", "subset": "sst2",
            "text_keys": ["sentence"], "num_labels": 2,
            "metric": "accuracy",
        },
        "mrpc": {
            "dataset": "glue", "subset": "mrpc",
            "text_keys": ["sentence1", "sentence2"], "num_labels": 2,
            "metric": "f1",
        },
        "cola": {
            "dataset": "glue", "subset": "cola",
            "text_keys": ["sentence"], "num_labels": 2,
            "metric": "matthews_correlation",
        },
        "rte": {
            "dataset": "glue", "subset": "rte",
            "text_keys": ["sentence1", "sentence2"], "num_labels": 2,
            "metric": "accuracy",
        },
        "qnli": {
            "dataset": "glue", "subset": "qnli",
            "text_keys": ["question", "sentence"], "num_labels": 2,
            "metric": "accuracy",
        },
    },

    # ---- Statistical Power: 10 seeds as per analysis ----
    # FIX 4: Multiple seeds for statistical rigor
    "seeds": [42, 123, 456, 789, 1011, 2022, 3033, 4044, 5055, 6066],

    # ---- Learning rates for sensitivity analysis ----
    "learning_rates": [1e-5, 2e-5, 5e-5],

    # ---- RTX 3050 4GB Memory-Safe Settings ----
    "batch_size": 2,             # Micro-batch (VRAM safe)
    "grad_accum": 4,             # Effective batch = 8 (balanced)
    "max_len": 128,              # Standard for GLUE
    "epochs": 10,                # More epochs to observe plateaus
    "weight_decay": 0.01,
    "warmup_ratio": 0.06,
    "max_grad_norm": 1.0,
    "patience": 4,               # FIX 6: Reasonable patience

    # ---- Plateau Detection ----
    # FIX 5: Proper plateau detection
    "plateau_patience_steps": 50,
    "plateau_min_delta": 0.001,
    "ler_window": 30,
    "ler_threshold": 0.005,      # More sensitive for real plateaus
    "gsnr_window": 30,

    # ---- Evaluation ----
    "eval_every_steps": 20,      # Frequent for plateau detection
    "log_every_steps": 10,

    # ---- Output ----
    "output_dir": "./experiments/lerna/runs",
}


# ══════════════════════════════════════════════════════════════════
# GSNR TRACKER — FIX 3: Correct Per-Parameter Implementation
# ══════════════════════════════════════════════════════════════════
class GSNRTracker:
    """
    Gradient Signal-to-Noise Ratio tracker.
    GSNR(θ) = E[∇θL]² / Var[∇θL]
    Computed per-parameter then averaged (matches published definition).
    Memory-efficient for RTX 3050 4GB.
    """
    def __init__(self, model, window_size=30):
        self.window_size = window_size
        self.step_count = 0
        self.param_groups = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.param_groups[name] = {
                    'means': [],
                    'sq_means': []
                }

    def update(self, model):
        """Call after backward(), before optimizer.step()"""
        self.step_count += 1
        for name, param in model.named_parameters():
            if param.grad is not None and name in self.param_groups:
                g = param.grad.detach()
                if g.numel() > 0:
                    self.param_groups[name]['means'].append(g.mean().item())
                    self.param_groups[name]['sq_means'].append((g ** 2).mean().item())
                    
                    if len(self.param_groups[name]['means']) > self.window_size:
                        self.param_groups[name]['means'].pop(0)
                        self.param_groups[name]['sq_means'].pop(0)

    def compute_gsnr(self):
        """Compute aggregate GSNR across all tracked parameters."""
        if self.step_count < 5:
            return float('nan')

        gsnr_values = []
        for name, stats in self.param_groups.items():
            if len(stats['means']) > 1:
                means = np.array(stats['means'])
                sq_means = np.array(stats['sq_means'])
                
                signal = np.mean(means) ** 2
                variance = np.mean(sq_means) - np.mean(means) ** 2
                variance = max(variance, 1e-12)
                
                gsnr = signal / variance
                if not np.isnan(gsnr) and not np.isinf(gsnr):
                    gsnr_values.append(gsnr)

        return float(np.mean(gsnr_values)) if gsnr_values else float('nan')


# ══════════════════════════════════════════════════════════════════
# LER TRACKER — FIX 3: Learning Efficiency Rate
# ══════════════════════════════════════════════════════════════════
class LERTracker:
    """
    Learning Efficiency Rate: measures the rate of useful learning
    per training step. Computed as the smoothed derivative of
    eval_loss reduction normalized by compute spent.
    
    LER = -Δ(eval_loss) / Δ(steps)  (smoothed over window)
    High LER = model is learning efficiently
    Low LER = model has plateaued (wasted compute)
    """
    def __init__(self, window_size=30, threshold=0.005):
        self.window_size = window_size
        self.threshold = threshold
        self.eval_losses = []
        self.eval_steps = []
        self.ler_values = []
        self.plateau_step = None
        self.plateau_detected = False
        self._low_ler_counter = 0

    def update(self, step, eval_loss):
        self.eval_losses.append(eval_loss)
        self.eval_steps.append(step)

        if len(self.eval_losses) < 3:
            self.ler_values.append(float('nan'))
            return float('nan')

        window = min(len(self.eval_losses), self.window_size)
        recent_losses = self.eval_losses[-window:]
        recent_steps = self.eval_steps[-window:]

        if len(recent_losses) < 2:
            self.ler_values.append(0.0)
            return 0.0

        step_diffs = np.diff(recent_steps).astype(float)
        loss_diffs = np.diff(recent_losses)

        step_diffs = np.where(step_diffs == 0, 1, step_diffs)
        rates = -loss_diffs / step_diffs
        
        ler = float(np.mean(rates[-3:])) if len(rates) >= 3 else float(np.mean(rates))
        self.ler_values.append(ler)

        # Plateau detection
        if not self.plateau_detected:
            if ler < self.threshold:
                self._low_ler_counter += 1
                if self._low_ler_counter >= 3:
                    self.plateau_step = step
                    self.plateau_detected = True
            else:
                self._low_ler_counter = 0

        return ler

    def get_waste_percentage(self, total_steps):
        """Calculate wasted compute percentage."""
        if self.plateau_step is None or total_steps <= 0:
            return 0.0
        waste_steps = max(0, total_steps - self.plateau_step)
        return (waste_steps / total_steps) * 100


# ══════════════════════════════════════════════════════════════════
# LOAD GLUE DATA — FIX 1 & 2: Use full GLUE datasets
# ══════════════════════════════════════════════════════════════════
def load_glue_data(task_name, tokenizer, max_length=128):
    """Load FULL GLUE dataset with no sampling."""
    print(f"    Loading FULL {task_name} dataset...")
    
    dataset = load_dataset("glue", task_name)
    
    task_config = CONFIG["tasks"][task_name]
    text_keys = task_config["text_keys"]
    
    def tokenize_function(examples):
        if len(text_keys) == 1:
            return tokenizer(examples[text_keys[0]], 
                           truncation=True, 
                           max_length=max_length,
                           padding=False)
        else:
            return tokenizer(examples[text_keys[0]], 
                           examples[text_keys[1]],
                           truncation=True, 
                           max_length=max_length,
                           padding=False)
    
    tokenized_train = dataset["train"].map(tokenize_function, batched=True)
    tokenized_val = dataset["validation"].map(tokenize_function, batched=True)
    
    columns_to_remove = [col for col in tokenized_train.column_names 
                        if col not in ["input_ids", "attention_mask", "label"]]
    tokenized_train = tokenized_train.remove_columns(columns_to_remove)
    tokenized_val = tokenized_val.remove_columns(columns_to_remove)
    
    print(f"    Train samples: {len(tokenized_train)}, Val samples: {len(tokenized_val)}")
    return tokenized_train, tokenized_val


# ══════════════════════════════════════════════════════════════════
# COMPUTE GLUE METRICS
# ══════════════════════════════════════════════════════════════════
def compute_glue_metric(task_name, predictions, references):
    """Compute appropriate GLUE metric for each task."""
    if task_name == "sst2" or task_name == "rte" or task_name == "qnli":
        correct = sum(p == r for p, r in zip(predictions, references))
        return correct / len(predictions)
    elif task_name == "mrpc":
        from sklearn.metrics import f1_score
        return f1_score(references, predictions, average='binary')
    elif task_name == "cola":
        from sklearn.metrics import matthews_corrcoef
        return matthews_corrcoef(references, predictions)
    return 0.0


# ══════════════════════════════════════════════════════════════════
# SINGLE EXPERIMENT RUN
# ══════════════════════════════════════════════════════════════════
def run_single_experiment(task_name, seed, lr, config):
    """Run a single experiment with FULL GLUE dataset."""
    
    run_id = f"{task_name}_seed{seed}_lr{lr}"
    print(f"\n{'='*60}")
    print(f"  RUN: {run_id}")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    
    # Initialize wandb
    wandb.init(
        project="lerna-research",
        name=run_id,
        config={
            "model": config["model_name"],
            "task": task_name,
            "seed": seed,
            "learning_rate": lr,
            "batch_size": config["batch_size"],
            "grad_accum": config["grad_accum"],
            "effective_batch": config["batch_size"] * config["grad_accum"],
            "max_len": config["max_len"],
            "epochs": config["epochs"],
            "patience": config["patience"],
        },
        reinit=True,
    )
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    task_config = config["tasks"][task_name]
    
    model = AutoModelForSequenceClassification.from_pretrained(
        config["model_name"],
        num_labels=task_config["num_labels"],
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        attn_implementation="sdpa",
    )
    
    try:
        model.gradient_checkpointing_enable()
        print("    ✅ Gradient checkpointing enabled")
    except:
        print("    ⚠️  Gradient checkpointing not available")
    
    model.to(DEVICE)
    
    # Load FULL GLUE dataset (no sampling)
    train_dataset, val_dataset = load_glue_data(task_name, tokenizer, config["max_len"])
    
    # Create data loaders
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"] * 2,
        shuffle=False,
        collate_fn=data_collator,
        num_workers=0,
        pin_memory=True,
    )
    
    # Optimizer
    no_decay = ["bias", "LayerNorm.weight", "layernorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": config["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Scheduler
    total_steps = len(train_loader) // config["grad_accum"] * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Initialize trackers
    gsnr_tracker = GSNRTracker(model, window_size=config["gsnr_window"])
    ler_tracker = LERTracker(window_size=config["ler_window"], 
                            threshold=config["ler_threshold"])
    
    # Training state
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    global_step = 0
    start_time = time.time()
    
    # Metrics history
    history = {
        "train_loss": [], "val_loss": [], "val_accuracy": [],
        "gsnr": [], "ler": [], "learning_rate": [], "step": [],
    }
    
    # Initial evaluation
    model.eval()
    val_loss, val_accuracy = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item() * len(batch["labels"])
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    val_loss /= len(val_dataset)
    val_accuracy = compute_glue_metric(task_name, all_preds, all_labels)
    
    print(f"    Initial: val_loss={val_loss:.4f}, val_acc={val_accuracy:.4f}")
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / config["grad_accum"]
            loss.backward()
            
            epoch_loss += outputs.loss.item()
            
            if (batch_idx + 1) % config["grad_accum"] == 0:
                global_step += 1
                
                # Update GSNR before optimizer step
                gsnr_tracker.update(model)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                # Logging
                if global_step % config["log_every_steps"] == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    current_gsnr = gsnr_tracker.compute_gsnr()
                    
                    wandb.log({
                        "train/step_loss": loss.item() * config["grad_accum"],
                        "train/learning_rate": current_lr,
                        "train/gsnr": current_gsnr if not np.isnan(current_gsnr) else 0,
                        "global_step": global_step,
                    })
                
                # Evaluation
                if global_step % config["eval_every_steps"] == 0:
                    model.eval()
                    val_loss, val_accuracy = 0, 0
                    all_preds, all_labels = [], []
                    
                    with torch.no_grad():
                        for val_batch in val_loader:
                            val_batch = {k: v.to(DEVICE) for k, v in val_batch.items()}
                            outputs = model(**val_batch)
                            val_loss += outputs.loss.item() * len(val_batch["labels"])
                            preds = torch.argmax(outputs.logits, dim=-1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(val_batch["labels"].cpu().numpy())
                    
                    val_loss /= len(val_dataset)
                    val_accuracy = compute_glue_metric(task_name, all_preds, all_labels)
                    current_ler = ler_tracker.update(global_step, val_loss)
                    current_gsnr = gsnr_tracker.compute_gsnr()
                    
                    # Store history
                    history["step"].append(global_step)
                    history["train_loss"].append(epoch_loss / (batch_idx + 1))
                    history["val_loss"].append(val_loss)
                    history["val_accuracy"].append(val_accuracy)
                    history["gsnr"].append(current_gsnr)
                    history["ler"].append(current_ler)
                    history["learning_rate"].append(scheduler.get_last_lr()[0])
                    
                    # Log to wandb
                    wandb.log({
                        "val/loss": val_loss,
                        "val/accuracy": val_accuracy,
                        "efficiency/gsnr": current_gsnr if not np.isnan(current_gsnr) else 0,
                        "efficiency/ler": current_ler if not np.isnan(current_ler) else 0,
                        "efficiency/ler_plateau_detected": ler_tracker.plateau_detected,
                        "global_step": global_step,
                    })
                    
                    # Check for best model
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model_state = copy.deepcopy(model.state_dict())
                        patience_counter = 0
                    else:
                        patience_counter += 1
                    
                    model.train()
        
        # End of epoch
        epoch_time = time.time() - start_time
        avg_epoch_loss = epoch_loss / len(train_loader)
        
        print(f"    Epoch {epoch+1}/{config['epochs']}: "
              f"train_loss={avg_epoch_loss:.4f}, "
              f"val_loss={history['val_loss'][-1] if history['val_loss'] else 0:.4f}, "
              f"val_acc={history['val_accuracy'][-1] if history['val_accuracy'] else 0:.4f}, "
              f"GSRN={history['gsnr'][-1] if history['gsnr'] and not np.isnan(history['gsnr'][-1]) else 0:.4f}, "
              f"LER={history['ler'][-1] if history['ler'] and not np.isnan(history['ler'][-1]) else 0:.6f}")
        
        # Early stopping
        if patience_counter >= config["patience"]:
            print(f"    ⏹️ Early stopping at epoch {epoch+1}")
            break
    
    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final evaluation
    model.eval()
    final_val_loss, final_val_accuracy = 0, 0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(**batch)
            final_val_loss += outputs.loss.item() * len(batch["labels"])
            preds = torch.argmax(outputs.logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch["labels"].cpu().numpy())
    
    final_val_loss /= len(val_dataset)
    final_val_accuracy = compute_glue_metric(task_name, all_preds, all_labels)
    
    # Calculate waste percentage
    ler_waste = ler_tracker.get_waste_percentage(global_step)
    
    # Prepare results
    results = {
        "run_id": run_id,
        "task": task_name,
        "seed": seed,
        "learning_rate": lr,
        "final_metrics": {
            "val_loss": final_val_loss,
            "val_accuracy": final_val_accuracy,
        },
        "plateau_analysis": {
            "ler_plateau_step": ler_tracker.plateau_step,
            "total_steps": global_step,
            "ler_waste_percentage": ler_waste,
        },
        "efficiency": {
            "final_gsnr": gsnr_tracker.compute_gsnr(),
            "final_ler": ler_tracker.ler_values[-1] if ler_tracker.ler_values else None,
        },
        "compute": {
            "total_time": time.time() - start_time,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 1024**3,
        },
        "metrics_history": history,
    }
    
    # Save results
    output_dir = Path(config["output_dir"]) / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "results.json", "w") as f:
        json.dump({k: v for k, v in results.items() if k != "metrics_history"}, 
                 f, indent=2, default=str)
    
    with open(output_dir / "metrics_history.json", "w") as f:
        json.dump(history, f, indent=2, default=str)
    
    # Final wandb logging
    wandb.log({
        "summary/final_val_loss": final_val_loss,
        "summary/final_val_accuracy": final_val_accuracy,
        "summary/ler_waste_percentage": ler_waste,
        "summary/total_steps": global_step,
        "summary/total_time": results["compute"]["total_time"],
    })
    
    wandb.finish()
    
    # Cleanup
    del model, optimizer
    torch.cuda.empty_cache()
    
    return results


# ══════════════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS — FIX 4: Proper statistical tests
# ══════════════════════════════════════════════════════════════════
def analyze_results(all_results):
    """Perform statistical analysis on all results."""
    
    print(f"\n{'='*60}")
    print(f"  STATISTICAL ANALYSIS")
    print(f"{'='*60}")
    
    task_results = defaultdict(list)
    for result in all_results:
        task = result["task"]
        waste = result["plateau_analysis"]["ler_waste_percentage"]
        task_results[task].append(waste)
    
    # Calculate statistics per task
    for task, wastes in task_results.items():
        if len(wastes) >= 2:
            mean_waste = np.mean(wastes)
            std_waste = np.std(wastes, ddof=1)
            sem = stats.sem(wastes)
            ci_95 = stats.t.interval(0.95, len(wastes)-1, loc=mean_waste, scale=sem)
            
            t_stat, p_value = stats.ttest_1samp(wastes, 0)
            
            print(f"\n  {task.upper()}:")
            print(f"    Samples: {len(wastes)}")
            print(f"    Mean waste: {mean_waste:.2f}%")
            print(f"    Std: {std_waste:.2f}%")
            print(f"    95% CI: [{ci_95[0]:.2f}%, {ci_95[1]:.2f}%]")
            print(f"    t-statistic: {t_stat:.4f}")
            print(f"    p-value: {p_value:.6f}")
            print(f"    Significant (p<0.05): {'YES' if p_value < 0.05 else 'NO'}")
    
    # Overall statistics
    all_wastes = [r["plateau_analysis"]["ler_waste_percentage"] for r in all_results]
    if len(all_wastes) >= 2:
        overall_mean = np.mean(all_wastes)
        overall_std = np.std(all_wastes, ddof=1)
        overall_sem = stats.sem(all_wastes)
        overall_ci = stats.t.interval(0.95, len(all_wastes)-1, loc=overall_mean, scale=overall_sem)
        
        print(f"\n  OVERALL ({len(all_wastes)} runs):")
        print(f"    Mean waste: {overall_mean:.2f}% ± {overall_std:.2f}%")
        print(f"    95% CI: [{overall_ci[0]:.2f}%, {overall_ci[1]:.2f}%]")
    
    return task_results


# ══════════════════════════════════════════════════════════════════
# MAIN FUNCTION
# ══════════════════════════════════════════════════════════════════
def main():
    """Main execution function."""
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available! Need GPU for training.")
        return
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"🖥️  GPU: {gpu_name}")
    print(f"💾 VRAM: {vram_gb:.1f} GB")
    print(f"🔥 PyTorch: {torch.__version__}")
    
    if vram_gb < 4:
        print("⚠️  Warning: Less than 4GB VRAM, may encounter OOM errors!")
    
    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["single", "sweep"], default="single",
                       help="single=one run, sweep=full statistical sweep")
    parser.add_argument("--task", default="sst2", 
                       help="GLUE task for single mode")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for single mode")
    parser.add_argument("--lr", type=float, default=2e-5,
                       help="Learning rate for single mode")
    args = parser.parse_args()
    
    # Memory optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    
    if args.mode == "single":
        print(f"\n🧪 Running single experiment: {args.task}, seed={args.seed}, lr={args.lr}")
        result = run_single_experiment(args.task, args.seed, args.lr, CONFIG)
        
        print(f"\n✅ Experiment completed!")
        print(f"   Final accuracy: {result['final_metrics']['val_accuracy']:.4f}")
        print(f"   Waste detected: {result['plateau_analysis']['ler_waste_percentage']:.2f}%")
        print(f"   Total steps: {result['plateau_analysis']['total_steps']}")
        
    else:
        print(f"\n🔬 Starting FULL statistical sweep...")
        print(f"   Tasks: {list(CONFIG['tasks'].keys())}")
        print(f"   Seeds: {CONFIG['seeds']}")
        print(f"   Learning rates: {CONFIG['learning_rates']}")
        
        total_runs = len(CONFIG['tasks']) * len(CONFIG['seeds']) * len(CONFIG['learning_rates'])
        print(f"   Total runs: {total_runs}")
        
        all_results = []
        run_count = 0
        
        for task_name in CONFIG["tasks"]:
            for seed in CONFIG["seeds"]:
                for lr in CONFIG["learning_rates"]:
                    run_count += 1
                    print(f"\n🚀 Run {run_count}/{total_runs}: {task_name}, seed={seed}, lr={lr}")
                    
                    try:
                        result = run_single_experiment(task_name, seed, lr, CONFIG)
                        all_results.append(result)
                    except torch.cuda.OutOfMemoryError:
                        print(f"❌ Out of memory! Skipping this run.")
                        torch.cuda.empty_cache()
                        continue
                    except Exception as e:
                        print(f"❌ Error: {str(e)[:100]}")
                        continue
        
        if all_results:
            print(f"\n📊 Analyzing {len(all_results)} successful runs...")
            task_results = analyze_results(all_results)
            
            agg_path = Path(CONFIG["output_dir"]) / "aggregated_results.json"
            agg_data = {
                "total_runs": len(all_results),
                "per_task_stats": {},
                "all_results": [r for r in all_results]
            }
            
            for task, wastes in task_results.items():
                if len(wastes) >= 2:
                    agg_data["per_task_stats"][task] = {
                        "mean_waste": float(np.mean(wastes)),
                        "std_waste": float(np.std(wastes, ddof=1)),
                        "n_samples": len(wastes),
                        "wastes": wastes
                    }
            
            with open(agg_path, "w") as f:
                json.dump(agg_data, f, indent=2, default=str)
            
            print(f"\n✅ Sweep completed! Results saved to {agg_path}")
            
            all_wastes = [r["plateau_analysis"]["ler_waste_percentage"] for r in all_results]
            if all_wastes:
                mean_waste = np.mean(all_wastes)
                print(f"\n🎯 OVERALL FINDING:")
                print(f"   Average compute waste: {mean_waste:.1f}%")
                print(f"   Range: {min(all_wastes):.1f}% - {max(all_wastes):.1f}%")
                
                if len(all_wastes) >= 5:
                    _, p_value = stats.ttest_1samp(all_wastes, 0)
                    print(f"   Statistically significant (p={p_value:.6f}): {'YES' if p_value < 0.05 else 'NO'}")
        else:
            print("❌ No successful runs completed!")


if __name__ == "__main__":
    main()