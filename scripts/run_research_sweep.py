#!/usr/bin/env python3
"""
Main research execution script for comprehensive experiments.

Runs multiple experiments across models, tasks, and seeds to achieve
statistical significance (n=50+ runs).

Key features:
1. Config-driven experiment management
2. Statistical power planning
3. Parallel execution support
4. Comprehensive result collection
5. Publication-ready output generation
"""

import os
import sys
import json
import yaml
import torch
import wandb
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import logging
import warnings

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    set_seed,
)

from lerna.utils.plateau_ies import IESPlateauDetector, compute_statistical_significance
from lerna.utils.efficiency_metrics import EfficiencyMetricsCollector, validate_ler_metric
from lerna.utils.experiment_tracking import ResearchExperimentLogger, StatisticalAnalysisEngine
from lerna.callbacks.ies_callback import IESCallback, EfficiencyMonitoringCallback
from lerna.callbacks.efficiency_callback import EfficiencyMetricsCallback, ProbeAccuracyCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./experiments/research_sweep.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    model_name: str
    task: str
    seed: int
    learning_rate: float
    batch_size: int = 32
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_length: int = 128
    output_dir: str = "./experiments/runs"
    use_wandb: bool = True
    enable_ies: bool = True
    enable_efficiency_metrics: bool = True
    plateau_threshold: float = 0.001
    plateau_patience: int = 100


class GLUEDataset:
    """Wrapper for GLUE dataset loading."""
    
    @staticmethod
    def load_dataset(task_name: str, tokenizer, max_length: int = 128):
        """Load and tokenize GLUE dataset."""
        from datasets import load_dataset
        
        # Map GLUE task names
        task_to_keys = {
            "sst2": ("sentence", None),
            "qnli": ("question", "sentence"),
            "qqp": ("question1", "question2"),
            "mnli": ("premise", "hypothesis"),
            "rte": ("sentence1", "sentence2"),
            "mrpc": ("sentence1", "sentence2"),
            "cola": ("sentence", None),
            "stsb": ("sentence1", "sentence2"),
        }
        
        # Load dataset
        if task_name == "mnli":
            dataset = load_dataset("glue", "mnli")
        else:
            dataset = load_dataset("glue", task_name)
        
        # Get sentence keys
        sentence1_key, sentence2_key = task_to_keys[task_name]
        
        def preprocess_function(examples):
            """Tokenize examples."""
            if sentence2_key is None:
                return tokenizer(
                    examples[sentence1_key],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
            else:
                return tokenizer(
                    examples[sentence1_key],
                    examples[sentence2_key],
                    truncation=True,
                    padding="max_length",
                    max_length=max_length,
                )
        
        # Tokenize dataset
        tokenized_datasets = dataset.map(preprocess_function, batched=True)
        
        return tokenized_datasets
    
    @staticmethod
    def get_num_labels(task_name: str) -> int:
        """Get number of labels for a GLUE task."""
        if task_name == "stsb":  # Regression
            return 1
        elif task_name == "mnli":  # 3 classes
            return 3
        else:  # Binary classification
            return 2


def run_experiment(config: ExperimentConfig) -> Dict[str, Any]:
    """
    Run a single experiment with given configuration.
    
    Returns:
        Dictionary with experiment results
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting experiment: {config.model_name} on {config.task}")
    logger.info(f"Seed: {config.seed}, LR: {config.learning_rate}")
    logger.info(f"{'='*60}")
    
    # Set random seeds for reproducibility
    set_seed(config.seed)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    run_name = f"{config.model_name.replace('/', '_')}_{config.task}_s{config.seed}_lr{config.learning_rate:.1e}"
    output_dir = Path(config.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize experiment logger
    experiment_logger = ResearchExperimentLogger(
        experiment_name=run_name,
        researcher="LERNA Research Team",
        institution="Harbin Institute of Technology",
        enable_wandb=config.use_wandb,
        output_dir=str(output_dir),
    )
    
    try:
        # Load model and tokenizer
        logger.info(f"Loading model: {config.model_name}")
        num_labels = GLUEDataset.get_num_labels(config.task)
        
        model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name,
            num_labels=num_labels,
            ignore_mismatched_sizes=True,
        )
        
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Load and preprocess dataset
        logger.info(f"Loading dataset: {config.task}")
        tokenized_datasets = GLUEDataset.load_dataset(
            config.task,
            tokenizer,
            max_length=config.max_length
        )
        
        # Set up training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            evaluation_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            save_steps=100,
            learning_rate=config.learning_rate,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size * 2,
            num_train_epochs=config.num_epochs,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            logging_dir=str(output_dir / "logs"),
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            seed=config.seed,
            data_seed=config.seed,
            report_to="wandb" if config.use_wandb else "none",
            run_name=run_name,
            fp16=torch.cuda.is_available(),
            gradient_accumulation_steps=1,
            max_grad_norm=1.0,
            save_total_limit=2,
        )
        
        # Log experiment metadata
        experiment_id = experiment_logger.log_experiment(
            model=model,
            task=config.task,
            dataset=f"GLUE-{config.task}",
            hyperparameters={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "max_length": config.max_length,
                "seed": config.seed,
            },
            random_seeds={
                "torch": config.seed,
                "numpy": config.seed,
                "transformers": config.seed,
            }
        )
        
        # Define compute_metrics function
        def compute_metrics(p: EvalPrediction):
            """Compute metrics for GLUE tasks."""
            from scipy.stats import pearsonr, spearmanr
            from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
            
            preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
            labels = p.label_ids
            
            if config.task == "stsb":  # Regression
                preds = np.squeeze(preds)
                return {
                    "pearson": pearsonr(preds, labels)[0],
                    "spearman": spearmanr(preds, labels)[0],
                    "mse": ((preds - labels) ** 2).mean(),
                }
            else:  # Classification
                preds = np.argmax(preds, axis=1)
                accuracy = accuracy_score(labels, preds)
                
                if config.task in ["mrpc", "qqp"]:
                    f1 = f1_score(labels, preds)
                    return {"accuracy": accuracy, "f1": f1}
                elif config.task == "cola":
                    mcc = matthews_corrcoef(labels, preds)
                    return {"accuracy": accuracy, "mcc": mcc}
                else:
                    return {"accuracy": accuracy}
        
        # Initialize trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metrics,
        )
        
        # Add callbacks
        callbacks = []
        
        if config.enable_ies:
            ies_callback = IESCallback(
                threshold=config.plateau_threshold,
                patience=config.plateau_patience,
                task=config.task,
                restore_best_model=True,
                save_analysis=True,
                enable_wandb=config.use_wandb,
                output_dir=str(output_dir / "ies_analysis"),
            )
            callbacks.append(ies_callback)
        
        if config.enable_efficiency_metrics:
            efficiency_callback = EfficiencyMetricsCallback(
                task=config.task,
                enable_gsnr=True,
                enable_ler=True,
                enable_probe=False,
                enable_compute_tracking=True,
                output_dir=str(output_dir / "efficiency_metrics"),
                wandb_enabled=config.use_wandb,
                log_frequency=25,
            )
            callbacks.append(efficiency_callback)
            
            # Add probe accuracy callback for representation quality
            probe_callback = ProbeAccuracyCallback(
                probe_frequency=100,
                max_samples=1000,
                output_dir=str(output_dir / "probe_accuracy"),
                wandb_enabled=config.use_wandb,
            )
            callbacks.append(probe_callback)
        
        # Add callbacks to trainer
        for callback in callbacks:
            trainer.add_callback(callback)
        
        # Train the model
        logger.info(f"Starting training...")
        train_result = trainer.train()
        
        # Evaluate on validation set
        logger.info(f"Evaluating on validation set...")
        eval_result = trainer.evaluate()
        
        # Save model and tokenizer
        trainer.save_model(str(output_dir / "final_model"))
        tokenizer.save_pretrained(str(output_dir / "final_model"))
        
        # Prepare results
        results = {
            "experiment_id": experiment_id,
            "run_name": run_name,
            "config": asdict(config),
            "training_results": {
                "train_loss": train_result.training_loss,
                "train_epochs": train_result.epoch,
                "train_steps": train_result.global_step,
                "train_runtime": train_result.metrics.get("train_runtime", 0),
                "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            },
            "evaluation_results": eval_result,
            "model_info": {
                "model_name": config.model_name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            },
        }
        
        # Log results
        experiment_logger.log_results(
            experiment_id=experiment_id,
            results=results,
            generate_figures=True,
        )
        
        # Generate experiment report
        report = experiment_logger.generate_experiment_report(experiment_id)
        
        logger.info(f"\n✅ Experiment completed successfully!")
        logger.info(f"   Validation accuracy: {eval_result.get('eval_accuracy', 0):.4f}")
        logger.info(f"   Training time: {train_result.metrics.get('train_runtime', 0):.1f}s")
        
        # Return comprehensive results
        return {
            "success": True,
            "experiment_id": experiment_id,
            "run_name": run_name,
            "config": asdict(config),
            "results": results,
            "report": report,
            "output_dir": str(output_dir),
        }
        
    except Exception as e:
        logger.error(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "experiment_id": experiment_id if 'experiment_id' in locals() else "unknown",
            "run_name": run_name,
            "config": asdict(config),
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_experiment_plan(config: Dict) -> List[ExperimentConfig]:
    """Create experiment plan from configuration."""
    experiments = []
    
    # Parse configuration
    models = config.get("models", {}).get("primary_models", ["bert-base-uncased"])
    tasks = config.get("tasks", {}).get("glue_tasks", {}).get("classification", ["sst2"])
    seeds = config.get("experimental_design", {}).get("seeds", [42])
    learning_rates = config.get("experimental_design", {}).get("learning_rates", [1e-5])
    
    # Get other parameters
    batch_size = config.get("training", {}).get("batch_size", 32)
    num_epochs = config.get("training", {}).get("epochs", 3)
    warmup_ratio = config.get("training", {}).get("warmup_ratio", 0.1)
    weight_decay = config.get("training", {}).get("weight_decay", 0.01)
    max_length = config.get("training", {}).get("max_length", 128)
    output_dir = config.get("output", {}).get("directory", "./experiments/comprehensive_2026")
    use_wandb = config.get("wandb", {}).get("enabled", True)
    
    # IES parameters
    plateau_threshold = config.get("plateau_detection", {}).get("default_threshold", 0.001)
    plateau_patience = config.get("plateau_detection", {}).get("default_patience", 100)
    
    # Create all combinations
    for model in models:
        for task in tasks:
            for seed in seeds:
                for lr in learning_rates:
                    experiments.append(ExperimentConfig(
                        model_name=model,
                        task=task,
                        seed=seed,
                        learning_rate=lr,
                        batch_size=batch_size,
                        num_epochs=num_epochs,
                        warmup_ratio=warmup_ratio,
                        weight_decay=weight_decay,
                        max_length=max_length,
                        output_dir=output_dir,
                        use_wandb=use_wandb,
                        enable_ies=True,
                        enable_efficiency_metrics=True,
                        plateau_threshold=plateau_threshold,
                        plateau_patience=plateau_patience,
                    ))
    
    return experiments


def run_experiment_sweep(config_path: str, max_parallel: int = 1) -> Dict:
    """
    Run a sweep of experiments based on configuration.
    
    Args:
        config_path: Path to configuration YAML file
        max_parallel: Maximum number of parallel experiments
    
    Returns:
        Summary of all experiments
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)
    
    # Create experiment plan
    experiments = create_experiment_plan(config)
    
    logger.info(f"\n📊 Experiment Plan Summary:")
    logger.info(f"   Total experiments: {len(experiments)}")
    logger.info(f"   Models: {len(set(e.model_name for e in experiments))}")
    logger.info(f"   Tasks: {len(set(e.task for e in experiments))}")
    logger.info(f"   Seeds: {len(set(e.seed for e in experiments))}")
    logger.info(f"   Learning rates: {len(set(e.learning_rate for e in experiments))}")
    
    # Estimate compute requirements
    estimated_time_per_experiment = 0.5  # hours (conservative estimate)
    total_estimated_time = len(experiments) * estimated_time_per_experiment
    logger.info(f"\n⏱️  Compute Estimate:")
    logger.info(f"   Estimated time per experiment: {estimated_time_per_experiment} hours")
    logger.info(f"   Total estimated time: {total_estimated_time:.1f} hours")
    
    # Run experiments
    results = []
    successful = 0
    failed = 0
    
    logger.info(f"\n{'='*60}")
    logger.info("STARTING EXPERIMENT SWEEP")
    logger.info(f"{'='*60}")
    
    for i, exp_config in enumerate(experiments, 1):
        logger.info(f"\n▶️  Running experiment {i}/{len(experiments)}")
        logger.info(f"   Config: {exp_config.model_name} on {exp_config.task}")
        logger.info(f"   Seed: {exp_config.seed}, LR: {exp_config.learning_rate:.1e}")
        
        try:
            result = run_experiment(exp_config)
            results.append(result)
            
            if result["success"]:
                successful += 1
                logger.info(f"   ✅ Success")
            else:
                failed += 1
                logger.info(f"   ❌ Failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"   ❌ Unexpected error: {e}")
            failed += 1
            results.append({
                "success": False,
                "config": asdict(exp_config),
                "error": str(e),
            })
    
    # Generate summary
    summary = generate_sweep_summary(results, config)
    
    logger.info(f"\n{'='*60}")
    logger.info("EXPERIMENT SWEEP COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Total experiments: {len(experiments)}")
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info(f"Success rate: {successful/len(experiments)*100:.1f}%")
    
    # Save summary
    summary_path = Path(config.get("output", {}).get("directory", "./experiments")) / "sweep_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    logger.info(f"\n📁 Sweep summary saved: {summary_path}")
    
    # Perform statistical analysis on successful runs
    if successful > 0:
        perform_statistical_analysis(results, config)
    
    return summary


def generate_sweep_summary(results: List[Dict], config: Dict) -> Dict:
    """Generate comprehensive sweep summary."""
    successful_results = [r for r in results if r.get("success", False)]
    failed_results = [r for r in results if not r.get("success", False)]
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(results),
        "successful": len(successful_results),
        "failed": len(failed_results),
        "success_rate": len(successful_results) / len(results) * 100 if len(results) > 0 else 0,
        "config_summary": {
            "models": list(set(r["config"]["model_name"] for r in successful_results)),
            "tasks": list(set(r["config"]["task"] for r in successful_results)),
            "seeds": list(set(r["config"]["seed"] for r in successful_results)),
            "learning_rates": list(set(r["config"]["learning_rate"] for r in successful_results)),
        },
        "performance_summary": {},
        "efficiency_summary": {},
        "failed_experiments": [
            {
                "config": r["config"],
                "error": r.get("error", "Unknown"),
            }
            for r in failed_results
        ],
    }
    
    # Add performance summary if we have successful results
    if successful_results:
        # Accuracy statistics by task
        task_accuracies = {}
        for result in successful_results:
            task = result["config"]["task"]
            accuracy = result.get("results", {}).get("evaluation_results", {}).get("eval_accuracy", 0)
            
            if task not in task_accuracies:
                task_accuracies[task] = []
            task_accuracies[task].append(accuracy)
        
        summary["performance_summary"]["task_accuracies"] = {
            task: {
                "mean": float(np.mean(accuracies)),
                "std": float(np.std(accuracies, ddof=1)),
                "min": float(np.min(accuracies)),
                "max": float(np.max(accuracies)),
                "n": len(accuracies),
            }
            for task, accuracies in task_accuracies.items()
        }
        
        # Training time statistics
        training_times = [
            r.get("results", {}).get("training_results", {}).get("train_runtime", 0)
            for r in successful_results
        ]
        if training_times:
            summary["performance_summary"]["training_times"] = {
                "mean": float(np.mean(training_times)),
                "std": float(np.std(training_times, ddof=1)),
                "total": float(np.sum(training_times)),
            }
    
    return summary


def perform_statistical_analysis(results: List[Dict], config: Dict):
    """Perform statistical analysis on experiment results."""
    successful_results = [r for r in results if r.get("success", False)]
    
    if len(successful_results) < 5:
        logger.warning("Insufficient successful results for statistical analysis")
        return
    
    logger.info(f"\n📊 Performing statistical analysis on {len(successful_results)} successful runs...")
    
    # Initialize statistical analysis engine
    stats_engine = StatisticalAnalysisEngine(alpha=0.05)
    
    # Group results by task
    task_results = {}
    for result in successful_results:
        task = result["config"]["task"]
        accuracy = result.get("results", {}).get("evaluation_results", {}).get("eval_accuracy", 0)
        
        if task not in task_results:
            task_results[task] = []
        task_results[task].append(accuracy)
    
    # Perform analysis for each task
    for task, accuracies in task_results.items():
        if len(accuracies) >= 5:
            logger.info(f"\n📈 Statistical analysis for {task}:")
            logger.info(f"   n={len(accuracies)}, mean={np.mean(accuracies):.4f}, std={np.std(accuracies, ddof=1):.4f}")
            
            # One-sample t-test against random chance
            if task == "sst2":
                chance_level = 0.5
            elif task == "mnli":
                chance_level = 1/3
            else:
                chance_level = 0.5
            
            t_stat, p_value = stats.ttest_1samp(accuracies, chance_level)
            logger.info(f"   t-test vs chance ({chance_level:.3f}): t={t_stat:.2f}, p={p_value:.4f}")
            
            # Confidence interval
            ci = stats.t.interval(
                0.95,
                len(accuracies) - 1,
                loc=np.mean(accuracies),
                scale=stats.sem(accuracies),
            )
            logger.info(f"   95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
    
    # Compare performance across tasks if we have enough data
    if len(task_results) >= 2:
        logger.info(f"\n📊 Cross-task comparison:")
        
        # Prepare data for ANOVA
        anova_data = {task: np.array(accuracies) for task, accuracies in task_results.items()}
        
        # Perform ANOVA
        analysis = stats_engine.perform_comprehensive_analysis(anova_data)
        
        # Log key findings
        anova_result = analysis.get("comparison_tests", {}).get("anova", {})
        if anova_result.get("significant", False):
            logger.info(f"   Significant differences across tasks (ANOVA: F={anova_result['f_statistic']:.2f}, p={anova_result['p_value']:.4f})")
        else:
            logger.info(f"   No significant differences across tasks (ANOVA: F={anova_result.get('f_statistic', 0):.2f}, p={anova_result.get('p_value', 1):.4f})")
        
        # Save statistical analysis
        stats_path = Path(config.get("output", {}).get("directory", "./experiments")) / "statistical_analysis.json"
        with open(stats_path, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"📁 Statistical analysis saved: {stats_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run comprehensive research experiment sweep")
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/lerna_research_2026.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=1,
        help="Maximum number of parallel experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiment plan without running",
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Load configuration
    config = load_config(args.config)
    
    # Create experiment plan
    experiments = create_experiment_plan(config)
    
    if args.dry_run:
        logger.info(f"\n📋 DRY RUN - Experiment Plan:")
        logger.info(f"   Total experiments: {len(experiments)}")
        logger.info(f"   Estimated compute time: {len(experiments) * 0.5:.1f} hours")
        
        # Show first few experiments
        for i, exp in enumerate(experiments[:5], 1):
            logger.info(f"\n   Experiment {i}:")
            logger.info(f"     Model: {exp.model_name}")
            logger.info(f"     Task: {exp.task}")
            logger.info(f"     Seed: {exp.seed}")
            logger.info(f"     LR: {exp.learning_rate:.1e}")
        
        if len(experiments) > 5:
            logger.info(f"\n   ... and {len(experiments) - 5} more experiments")
        
        return
    
    # Run experiment sweep
    summary = run_experiment_sweep(args.config, args.max_parallel)
    
    logger.info(f"\n🎉 Research sweep completed!")
    logger.info(f"   Results saved in: {config.get('output', {}).get('directory', './experiments')}")


if __name__ == "__main__":
    main()