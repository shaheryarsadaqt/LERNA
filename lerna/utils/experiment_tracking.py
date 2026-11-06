"""
Professional Experiment Tracking and Metadata Management

Comprehensive tracking system for academic research experiments
with statistical analysis, publication-ready outputs, and
reproducibility management.
"""

import json
import torch
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
import hashlib
import warnings
from scipy import stats

try:
    import wandb
except ImportError:
    wandb = None

try:
    import yaml
except ImportError:
    yaml = None

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    import seaborn as sns
except ImportError:
    sns = None


@dataclass
class ExperimentMetadata:
    """Comprehensive experiment metadata for reproducibility."""
    
    # Basic information
    experiment_id: str
    timestamp: str
    researcher: str
    institution: str
    
    # Model information
    model_name: str
    model_hash: str
    model_parameters: int
    
    # Task information
    task: str
    dataset: str
    dataset_size: int
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    
    # System information
    system_info: Dict[str, Any]
    
    # Code versioning
    git_commit: Optional[str]
    code_hash: str
    
    # Random seeds
    random_seeds: Dict[str, int]
    
    # Results reference
    results_path: Optional[str]
    wandb_run_id: Optional[str]


class ResearchExperimentLogger:
    """
    Professional logger for academic research experiments.
    
    Features:
    1. Comprehensive metadata tracking
    2. Statistical analysis integration
    3. Publication-ready outputs
    4. W&B integration
    5. Reproducibility management
    """
    
    def __init__(
        self,
        experiment_name: str,
        researcher: str = "LERNA Research Team",
        institution: str = "Harbin Institute of Technology",
        enable_wandb: bool = True,
        wandb_entity: Optional[str] = None,
        wandb_project: Optional[str] = None,
        output_dir: str = "./experiments/research_logs",
    ):
        self.experiment_name = experiment_name
        self.researcher = researcher
        self.institution = institution
        self.enable_wandb = enable_wandb
        self.wandb_entity = wandb_entity
        self.wandb_project = wandb_project
        self.output_dir = Path(output_dir)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Experiment tracking
        self.experiments: List[ExperimentMetadata] = []
        self.results: List[Dict] = []
        self.statistical_analyses: List[Dict] = []
        
        # W&B initialization
        self.wandb_run = None
        if enable_wandb:
            self._initialize_wandb()
    
    def _initialize_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        try:
            self.wandb_run = wandb.init(
                entity=self.wandb_entity,
                project=self.wandb_project or f"lerna-{self.experiment_name}",
                name=self.experiment_name,
                config={
                    "experiment_name": self.experiment_name,
                    "researcher": self.researcher,
                    "institution": self.institution,
                    "timestamp": datetime.now().isoformat(),
                },
                tags=["research", "efficiency", "transformer", "fine-tuning"],
                dir=str(self.output_dir / "wandb"),
            )
        except Exception as e:
            warnings.warn(f"Failed to initialize W&B: {e}")
            self.wandb_run = None
    
    def log_experiment(
        self,
        model: torch.nn.Module,
        task: str,
        dataset: str,
        hyperparameters: Dict[str, Any],
        random_seeds: Optional[Dict[str, int]] = None,
        additional_metadata: Optional[Dict] = None,
    ) -> str:
        """
        Log a new experiment with comprehensive metadata.
        
        Returns:
            Experiment ID
        """
        # Generate experiment ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_id = f"{self.experiment_name}_{task}_{timestamp}"
        
        # Compute model hash
        model_hash = self._compute_model_hash(model)
        
        # Get system information
        system_info = self._get_system_info()
        
        # Get code hash
        code_hash = self._compute_code_hash()
        
        # Prepare random seeds
        if random_seeds is None:
            random_seeds = {
                "torch": torch.initial_seed(),
                "numpy": np.random.get_state()[1][0],
                "python": hash(datetime.now()),
            }
        
        # Create metadata
        metadata = ExperimentMetadata(
            experiment_id=experiment_id,
            timestamp=timestamp,
            researcher=self.researcher,
            institution=self.institution,
            model_name=model.__class__.__name__,
            model_hash=model_hash,
            model_parameters=sum(p.numel() for p in model.parameters()),
            task=task,
            dataset=dataset,
            dataset_size=self._get_dataset_size(dataset),
            hyperparameters=hyperparameters,
            system_info=system_info,
            git_commit=self._get_git_commit(),
            code_hash=code_hash,
            random_seeds=random_seeds,
            results_path=None,
            wandb_run_id=self.wandb_run.id if self.wandb_run else None,
        )
        
        # Add additional metadata
        if additional_metadata:
            metadata_dict = asdict(metadata)
            metadata_dict.update(additional_metadata)
            metadata = ExperimentMetadata(**metadata_dict)
        
        # Store metadata
        self.experiments.append(metadata)
        
        # Save to file
        self._save_metadata(metadata)
        
        # Log to W&B
        if self.wandb_run:
            wandb.config.update(asdict(metadata))
        
        return experiment_id
    
    def log_results(
        self,
        experiment_id: str,
        results: Dict[str, Any],
        statistical_analysis: Optional[Dict] = None,
        generate_figures: bool = True,
    ):
        """
        Log experiment results with statistical analysis.
        
        Args:
            experiment_id: ID of the experiment
            results: Dictionary of results
            statistical_analysis: Optional statistical analysis
            generate_figures: Whether to generate publication figures
        """
        # Find experiment
        experiment = self._find_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Create results entry
        results_entry = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(),
            "results": results,
            "statistical_analysis": statistical_analysis,
        }
        
        self.results.append(results_entry)
        
        # Save results
        results_path = self.output_dir / f"{experiment_id}_results.json"
        with open(results_path, 'w') as f:
            json.dump(results_entry, f, indent=2, default=str)
        
        # Update metadata with results path
        experiment.results_path = str(results_path)
        self._save_metadata(experiment)
        
        # Log to W&B
        if self.wandb_run:
            # Log basic results
            wandb.log(results)
            
            # Log statistical analysis if available
            if statistical_analysis:
                wandb.log({"statistical_analysis": statistical_analysis})
        
        # Generate figures if requested
        if generate_figures and "metrics" in results:
            self._generate_experiment_figures(experiment_id, results)
        
        # Perform statistical analysis if not provided
        if statistical_analysis is None and "metrics" in results:
            statistical_analysis = self._perform_statistical_analysis(results["metrics"])
            self.statistical_analyses.append(statistical_analysis)
            
            # Save statistical analysis
            stats_path = self.output_dir / f"{experiment_id}_statistics.json"
            with open(stats_path, 'w') as f:
                json.dump(statistical_analysis, f, indent=2)
    
    def _compute_model_hash(self, model: torch.nn.Module) -> str:
        """Compute hash of model architecture and parameters."""
        # Create a string representation of model architecture
        arch_str = str(model)
        
        # Add parameter statistics
        param_str = f"params:{sum(p.numel() for p in model.parameters())}"
        
        # Combine and hash
        combined = arch_str + param_str
        return hashlib.md5(combined.encode()).hexdigest()[:16]
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "timestamp": datetime.now().isoformat(),
            "python_version": torch.__version__,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            info.update({
                "cuda_version": torch.version.cuda,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
            })
        
        return info
    
    def _compute_code_hash(self) -> str:
        """Compute hash of relevant code files."""
        # List of directories to include
        code_dirs = ["./lerna", "./scripts"]
        
        # Collect file contents
        code_content = ""
        for code_dir in code_dirs:
            code_path = Path(code_dir)
            if code_path.exists():
                for file in code_path.rglob("*.py"):
                    try:
                        with open(file, 'r') as f:
                            code_content += f.read()
                    except:
                        pass
        
        # Compute hash
        return hashlib.md5(code_content.encode()).hexdigest()[:16]
    
    def _get_dataset_size(self, dataset: str) -> int:
        """Get dataset size for common datasets."""
        sizes = {
            "sst2": 67349,
            "qnli": 104743,
            "qqp": 363846,
            "mnli": 392702,
            "rte": 2490,
            "mrpc": 3668,
            "cola": 8551,
            "stsb": 5749,
        }
        return sizes.get(dataset, 0)
    
    def _get_git_commit(self) -> Optional[str]:
        """Get current git commit hash."""
        try:
            import subprocess
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                cwd=".",
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except:
            pass
        return None
    
    def _save_metadata(self, metadata: ExperimentMetadata):
        """Save experiment metadata to file."""
        metadata_path = self.output_dir / f"{metadata.experiment_id}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def _find_experiment(self, experiment_id: str) -> Optional[ExperimentMetadata]:
        """Find experiment by ID."""
        for exp in self.experiments:
            if exp.experiment_id == experiment_id:
                return exp
        return None
    
    def _perform_statistical_analysis(self, metrics: Dict) -> Dict:
        """Perform statistical analysis on metrics."""
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "n_samples": 0,
            "descriptive_statistics": {},
            "inferential_statistics": {},
            "effect_sizes": {},
            "confidence_intervals": {},
        }
        
        # Analyze each metric
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, list) and len(metric_values) > 1:
                values = np.array(metric_values)
                
                # Descriptive statistics
                analysis["descriptive_statistics"][metric_name] = {
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "q1": float(np.percentile(values, 25)),
                    "q3": float(np.percentile(values, 75)),
                    "n": len(values),
                }
                
                # Inferential statistics (one-sample t-test against 0)
                if len(values) >= 3:
                    t_stat, p_value = stats.ttest_1samp(values, 0)
                    analysis["inferential_statistics"][metric_name] = {
                        "t_statistic": float(t_stat),
                        "p_value": float(p_value),
                        "significant": p_value < 0.05,
                    }
                
                # Confidence interval
                if len(values) >= 5:
                    ci = stats.t.interval(
                        0.95,
                        len(values) - 1,
                        loc=np.mean(values),
                        scale=stats.sem(values),
                    )
                    analysis["confidence_intervals"][metric_name] = {
                        "lower": float(ci[0]),
                        "upper": float(ci[1]),
                        "width": float(ci[1] - ci[0]),
                    }
        
        analysis["n_samples"] = len(metrics.get(list(metrics.keys())[0], [])) if metrics else 0
        
        return analysis
    
    def _generate_experiment_figures(self, experiment_id: str, results: Dict):
        """Generate publication-ready figures for experiment."""
        if "metrics" not in results:
            return
        
        metrics = results["metrics"]
        figures_dir = self.output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)
        
        # Create figure for each metric trend
        for metric_name, metric_values in metrics.items():
            if isinstance(metric_values, list) and len(metric_values) > 5:
                self._create_metric_figure(
                    metric_name, metric_values, experiment_id, figures_dir
                )
        
        # Create combined figure if multiple metrics
        if len(metrics) >= 2:
            self._create_combined_figure(metrics, experiment_id, figures_dir)
    
    def _create_metric_figure(self, metric_name: str, values: List, 
                            experiment_id: str, output_dir: Path):
        """Create figure for a single metric."""
        plt.figure(figsize=(10, 6))
        
        # Plot values
        plt.plot(values, marker='o', markersize=3, linewidth=1.5)
        
        # Add statistics
        mean_val = np.mean(values)
        std_val = np.std(values, ddof=1)
        
        plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7, 
                   label=f'Mean: {mean_val:.4f}')
        plt.fill_between(range(len(values)), 
                        mean_val - std_val, 
                        mean_val + std_val, 
                        alpha=0.2, color='gray',
                        label=f'±1 std: {std_val:.4f}')
        
        # Formatting
        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.title(f'{metric_name.replace("_", " ").title()} - {experiment_id}', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save
        filename = output_dir / f"{experiment_id}_{metric_name}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Also save as PDF for publication
        pdf_filename = output_dir / f"{experiment_id}_{metric_name}.pdf"
        plt.figure(figsize=(10, 6))
        plt.plot(values, linewidth=1.5)
        plt.axhline(y=mean_val, color='r', linestyle='--', alpha=0.7)
        plt.xlabel('Step', fontsize=12)
        plt.ylabel(metric_name.replace('_', ' ').title(), fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.savefig(pdf_filename, bbox_inches='tight')
        plt.close()
    
    def _create_combined_figure(self, metrics: Dict, experiment_id: str, 
                              output_dir: Path):
        """Create combined figure for multiple metrics."""
        n_metrics = len(metrics)
        if n_metrics == 0:
            return
        
        # Determine subplot layout
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        axes = axes.flatten() if n_metrics > 1 else [axes]
        
        for idx, (metric_name, values) in enumerate(metrics.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            if isinstance(values, list) and len(values) > 1:
                ax.plot(values, linewidth=1.5)
                ax.set_xlabel('Step')
                ax.set_ylabel(metric_name.replace('_', ' ').title())
                ax.set_title(metric_name.replace('_', ' ').title())
                ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(metrics), len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Experiment {experiment_id}', fontsize=16)
        plt.tight_layout()
        
        # Save
        filename = output_dir / f"{experiment_id}_combined.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_experiment_report(self, experiment_id: str) -> Dict:
        """Generate comprehensive experiment report."""
        experiment = self._find_experiment(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        # Find results
        results = None
        for res in self.results:
            if res["experiment_id"] == experiment_id:
                results = res
                break
        
        # Find statistical analysis
        stats_analysis = None
        for stats in self.statistical_analyses:
            # Check if this analysis corresponds to the experiment
            if results and stats.get("timestamp", "").startswith(results.get("timestamp", "")[:10]):
                stats_analysis = stats
                break
        
        # Create report
        report = {
            "experiment_summary": {
                "id": experiment_id,
                "timestamp": experiment.timestamp,
                "researcher": experiment.researcher,
                "institution": experiment.institution,
                "model": experiment.model_name,
                "task": experiment.task,
                "dataset": experiment.dataset,
            },
            "methodology": {
                "hyperparameters": experiment.hyperparameters,
                "random_seeds": experiment.random_seeds,
                "system_info": experiment.system_info,
            },
            "results_summary": results["results"] if results else None,
            "statistical_analysis": stats_analysis,
            "reproducibility": {
                "model_hash": experiment.model_hash,
                "code_hash": experiment.code_hash,
                "git_commit": experiment.git_commit,
                "results_path": experiment.results_path,
                "wandb_run_id": experiment.wandb_run_id,
            },
            "quality_assessment": self._assess_experiment_quality(experiment, results, stats_analysis),
        }
        
        # Save report
        report_path = self.output_dir / f"{experiment_id}_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        return report
    
    def _assess_experiment_quality(
        self, 
        experiment: ExperimentMetadata, 
        results: Optional[Dict], 
        stats: Optional[Dict]
    ) -> Dict:
        """Assess the quality of the experiment."""
        quality = {
            "completeness": 0.0,
            "statistical_rigor": 0.0,
            "reproducibility": 0.0,
            "overall_quality": 0.0,
            "issues": [],
            "recommendations": [],
        }
        
        # Check completeness
        completeness_score = 0
        if experiment.hyperparameters:
            completeness_score += 25
        if experiment.random_seeds:
            completeness_score += 25
        if results:
            completeness_score += 25
        if stats:
            completeness_score += 25
        
        quality["completeness"] = completeness_score
        
        # Check statistical rigor
        rigor_score = 0
        if stats:
            if stats.get("n_samples", 0) >= 30:
                rigor_score += 25
            if stats.get("confidence_intervals"):
                rigor_score += 25
            if stats.get("inferential_statistics"):
                rigor_score += 25
            if stats.get("effect_sizes"):
                rigor_score += 25
        
        quality["statistical_rigor"] = rigor_score
        
        # Check reproducibility
        repro_score = 0
        if experiment.model_hash:
            repro_score += 20
        if experiment.code_hash:
            repro_score += 20
        if experiment.git_commit:
            repro_score += 20
        if experiment.results_path:
            repro_score += 20
        if experiment.wandb_run_id:
            repro_score += 20
        
        quality["reproducibility"] = repro_score
        
        # Overall quality
        quality["overall_quality"] = (completeness_score + rigor_score + repro_score) / 3
        
        # Identify issues
        if completeness_score < 75:
            quality["issues"].append("Incomplete experiment documentation")
            quality["recommendations"].append("Ensure all hyperparameters and random seeds are documented")
        
        if rigor_score < 50:
            quality["issues"].append("Insufficient statistical analysis")
            quality["recommendations"].append("Perform statistical tests with confidence intervals")
        
        if repro_score < 60:
            quality["issues"].append("Limited reproducibility")
            quality["recommendations"].append("Include code hashes and detailed environment information")
        
        if quality["overall_quality"] < 70:
            quality["issues"].append("Overall quality needs improvement")
            quality["recommendations"].append("Review and address all quality assessment items")
        
        return quality


class StatisticalAnalysisEngine:
    """
    Advanced statistical analysis engine for research experiments.
    
    Features:
    1. Hypothesis testing
    2. Effect size calculation
    3. Power analysis
    4. Multiple comparisons correction
    5. Non-parametric alternatives
    """
    
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.analyses = []
    
    def perform_comprehensive_analysis(
        self,
        data: Dict[str, np.ndarray],
        groups: Optional[List[str]] = None,
        paired: bool = False,
    ) -> Dict:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            data: Dictionary of data arrays
            groups: Group labels for comparison
            paired: Whether data is paired
        
        Returns:
            Comprehensive analysis results
        """
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "n_groups": len(data),
            "descriptive_statistics": {},
            "normality_tests": {},
            "homogeneity_tests": {},
            "comparison_tests": {},
            "effect_sizes": {},
            "power_analysis": {},
            "interpretation": {},
        }
        
        # Descriptive statistics for each group
        for group_name, group_data in data.items():
            analysis["descriptive_statistics"][group_name] = self._compute_descriptive_stats(group_data)
        
        # Normality tests
        for group_name, group_data in data.items():
            analysis["normality_tests"][group_name] = self._test_normality(group_data)
        
        # Homogeneity of variance
        if len(data) >= 2:
            analysis["homogeneity_tests"] = self._test_homogeneity(data)
        
        # Group comparisons
        if len(data) == 2:
            # Two-group comparison
            group_names = list(data.keys())
            group1_data = data[group_names[0]]
            group2_data = data[group_names[1]]
            
            analysis["comparison_tests"] = self._compare_two_groups(
                group1_data, group2_data, group_names[0], group_names[1], paired
            )
            
            # Effect sizes
            analysis["effect_sizes"] = self._compute_effect_sizes_two_groups(
                group1_data, group2_data, paired
            )
            
            # Power analysis
            analysis["power_analysis"] = self._compute_power_two_groups(
                group1_data, group2_data
            )
            
        elif len(data) > 2:
            # Multiple groups
            analysis["comparison_tests"] = self._compare_multiple_groups(data, groups)
            analysis["effect_sizes"] = self._compute_effect_sizes_multiple_groups(data)
        
        # Interpretation
        analysis["interpretation"] = self._interpret_analysis(analysis)
        
        self.analyses.append(analysis)
        return analysis
    
    def _compute_descriptive_stats(self, data: np.ndarray) -> Dict:
        """Compute descriptive statistics."""
        if len(data) == 0:
            return {}
        
        return {
            "n": len(data),
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "std": float(np.std(data, ddof=1)),
            "sem": float(stats.sem(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "q1": float(np.percentile(data, 25)),
            "q3": float(np.percentile(data, 75)),
            "range": float(np.ptp(data)),
            "iqr": float(np.percentile(data, 75) - np.percentile(data, 25)),
            "cv": float(np.std(data, ddof=1) / np.mean(data) if np.mean(data) != 0 else 0),
        }
    
    def _test_normality(self, data: np.ndarray) -> Dict:
        """Test data for normality."""
        if len(data) < 8:
            return {"test": "insufficient_data", "n": len(data)}
        
        # Shapiro-Wilk test
        stat, p_value = stats.shapiro(data)
        
        # D'Agostino K² test
        stat2, p_value2 = stats.normaltest(data)
        
        return {
            "shapiro_wilk": {
                "statistic": float(stat),
                "p_value": float(p_value),
                "normal": p_value > self.alpha,
            },
            "dagostino_k2": {
                "statistic": float(stat2),
                "p_value": float(p_value2),
                "normal": p_value2 > self.alpha,
            },
            "interpretation": "normal" if p_value > self.alpha else "not_normal",
        }
    
    def _test_homogeneity(self, data: Dict[str, np.ndarray]) -> Dict:
        """Test homogeneity of variance."""
        if len(data) < 2:
            return {}
        
        # Levene's test
        stat, p_value = stats.levene(*data.values())
        
        # Brown-Forsythe test
        stat2, p_value2 = stats.levene(*data.values(), center='median')
        
        return {
            "levene": {
                "statistic": float(stat),
                "p_value": float(p_value),
                "homogeneous": p_value > self.alpha,
            },
            "brown_forsythe": {
                "statistic": float(stat2),
                "p_value": float(p_value2),
                "homogeneous": p_value2 > self.alpha,
            },
            "interpretation": "homogeneous" if p_value > self.alpha else "heterogeneous",
        }
    
    def _compare_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        group1_name: str = "Group 1",
        group2_name: str = "Group 2",
        paired: bool = False,
    ) -> Dict:
        """Compare two groups with appropriate tests."""
        results = {
            "test_type": "paired" if paired else "independent",
            "parametric_tests": {},
            "nonparametric_tests": {},
            "recommended_test": "",
        }
        
        # Check assumptions
        normality1 = self._test_normality(group1)
        normality2 = self._test_normality(group2)
        homogeneity = self._test_homogeneity({"g1": group1, "g2": group2})
        
        assumptions_met = (
            normality1.get("interpretation") == "normal" and
            normality2.get("interpretation") == "normal" and
            homogeneity.get("interpretation") == "homogeneous"
        )
        
        # Parametric tests
        if paired:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(group1, group2)
            results["parametric_tests"]["paired_t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }
        else:
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(group1, group2, equal_var=True)
            results["parametric_tests"]["independent_t_test"] = {
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }
            
            # Welch's t-test (unequal variances)
            t_stat_welch, p_value_welch = stats.ttest_ind(group1, group2, equal_var=False)
            results["parametric_tests"]["welchs_t_test"] = {
                "statistic": float(t_stat_welch),
                "p_value": float(p_value_welch),
                "significant": p_value_welch < self.alpha,
            }
        
        # Non-parametric tests
        if paired:
            # Wilcoxon signed-rank test
            stat, p_value = stats.wilcoxon(group1, group2)
            results["nonparametric_tests"]["wilcoxon"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            results["nonparametric_tests"]["mann_whitney"] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "significant": p_value < self.alpha,
            }
        
        # Recommend appropriate test
        if assumptions_met:
            results["recommended_test"] = "parametric"
            if paired:
                results["recommended_results"] = results["parametric_tests"]["paired_t_test"]
            else:
                if homogeneity.get("levene", {}).get("homogeneous", True):
                    results["recommended_results"] = results["parametric_tests"]["independent_t_test"]
                else:
                    results["recommended_results"] = results["parametric_tests"]["welchs_t_test"]
        else:
            results["recommended_test"] = "nonparametric"
            if paired:
                results["recommended_results"] = results["nonparametric_tests"]["wilcoxon"]
            else:
                results["recommended_results"] = results["nonparametric_tests"]["mann_whitney"]
        
        return results
    
    def _compute_effect_sizes_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        paired: bool = False,
    ) -> Dict:
        """Compute effect sizes for two groups."""
        effect_sizes = {}
        
        # Cohen's d
        pooled_std = np.sqrt(
            (np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2
        )
        cohens_d = (np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
        effect_sizes["cohens_d"] = {
            "value": float(cohens_d),
            "interpretation": self._interpret_cohens_d(cohens_d),
        }
        
        # Hedges' g (small sample correction)
        n1, n2 = len(group1), len(group2)
        df = n1 + n2 - 2
        correction = 1 - (3 / (4 * df - 1))
        hedges_g = cohens_d * correction
        effect_sizes["hedges_g"] = {
            "value": float(hedges_g),
            "interpretation": self._interpret_cohens_d(hedges_g),
        }
        
        # Glass's delta (using control group SD)
        glass_delta = (np.mean(group1) - np.mean(group2)) / (np.std(group2, ddof=1) + 1e-10)
        effect_sizes["glass_delta"] = {
            "value": float(glass_delta),
            "interpretation": self._interpret_cohens_d(glass_delta),
        }
        
        if paired:
            # Cohen's d for paired samples
            differences = group1 - group2
            cohens_dz = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
            effect_sizes["cohens_dz"] = {
                "value": float(cohens_dz),
                "interpretation": self._interpret_cohens_d(cohens_dz),
            }
        
        return effect_sizes
    
    def _interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def _compute_power_two_groups(
        self,
        group1: np.ndarray,
        group2: np.ndarray,
        power: float = 0.8,
    ) -> Dict:
        """Compute statistical power for two-group comparison."""
        # Effect size
        pooled_std = np.sqrt(
            (np.std(group1, ddof=1)**2 + np.std(group2, ddof=1)**2) / 2
        )
        effect_size = abs(np.mean(group1) - np.mean(group2)) / (pooled_std + 1e-10)
        
        # Sample sizes
        n1, n2 = len(group1), len(group2)
        
        # Power calculation (simplified)
        # For more accurate power, would use statsmodels or specialized library
        critical_t = stats.t.ppf(1 - self.alpha/2, n1 + n2 - 2)
        noncentrality = effect_size * np.sqrt((n1 * n2) / (n1 + n2))
        power_actual = 1 - stats.nct.cdf(critical_t, n1 + n2 - 2, noncentrality)
        
        # Required sample size for desired power
        # Using Cohen's power tables approximation
        if effect_size > 0:
            # Z-values
            z_alpha = stats.norm.ppf(1 - self.alpha/2)
            z_beta = stats.norm.ppf(power)
            
            # Required sample size per group (approximation)
            n_required = 2 * ((z_alpha + z_beta) / effect_size) ** 2
        else:
            n_required = float('inf')
        
        return {
            "effect_size": float(effect_size),
            "current_power": float(power_actual),
            "current_sample_sizes": {"group1": n1, "group2": n2},
            "required_sample_per_group": float(n_required),
            "adequately_powered": power_actual >= power,
        }
    
    def _compare_multiple_groups(
        self,
        data: Dict[str, np.ndarray],
        groups: Optional[List[str]] = None,
    ) -> Dict:
        """Compare multiple groups."""
        results = {}
        
        # ANOVA
        f_stat, p_value = stats.f_oneway(*data.values())
        results["anova"] = {
            "f_statistic": float(f_stat),
            "p_value": float(p_value),
            "significant": p_value < self.alpha,
        }
        
        # Kruskal-Wallis (non-parametric)
        h_stat, p_value_kw = stats.kruskal(*data.values())
        results["kruskal_wallis"] = {
            "h_statistic": float(h_stat),
            "p_value": float(p_value_kw),
            "significant": p_value_kw < self.alpha,
        }
        
        # Post-hoc tests if significant
        if p_value < self.alpha:
            results["post_hoc"] = self._perform_post_hoc_tests(data, groups)
        
        return results
    
    def _perform_post_hoc_tests(
        self,
        data: Dict[str, np.ndarray],
        groups: Optional[List[str]] = None,
    ) -> Dict:
        """Perform post-hoc pairwise comparisons."""
        post_hoc = {}
        
        group_names = list(data.keys())
        n_groups = len(group_names)
        
        # Tukey's HSD
        try:
            from statsmodels.stats.multicomp import pairwise_tukeyhsd
            # Flatten data
            values = []
            group_labels = []
            for name, values_arr in data.items():
                values.extend(values_arr)
                group_labels.extend([name] * len(values_arr))
            
            tukey = pairwise_tukeyhsd(values, group_labels, alpha=self.alpha)
            post_hoc["tukey_hsd"] = {
                "results": str(tukey.summary()),
                "significant_pairs": [
                    (group_names[i], group_names[j])
                    for i in range(n_groups)
                    for j in range(i+1, n_groups)
                    if tukey.reject[i, j]
                ],
            }
        except ImportError:
            post_hoc["tukey_hsd"] = {"error": "statsmodels not installed"}
        
        # Bonferroni correction for pairwise t-tests
        n_comparisons = n_groups * (n_groups - 1) // 2
        bonferroni_alpha = self.alpha / n_comparisons
        
        bonferroni_results = []
        for i in range(n_groups):
            for j in range(i+1, n_groups):
                t_stat, p_value = stats.ttest_ind(data[group_names[i]], data[group_names[j]])
                bonferroni_results.append({
                    "group1": group_names[i],
                    "group2": group_names[j],
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "p_adjusted": min(1.0, p_value * n_comparisons),
                    "significant": p_value < bonferroni_alpha,
                })
        
        post_hoc["bonferroni"] = bonferroni_results
        
        return post_hoc
    
    def _compute_effect_sizes_multiple_groups(self, data: Dict[str, np.ndarray]) -> Dict:
        """Compute effect sizes for multiple groups."""
        # Eta squared (η²)
        # Total sum of squares
        all_data = np.concatenate(list(data.values()))
        ss_total = np.sum((all_data - np.mean(all_data)) ** 2)
        
        # Between-groups sum of squares
        ss_between = 0
        for group_data in data.values():
            group_mean = np.mean(group_data)
            ss_between += len(group_data) * (group_mean - np.mean(all_data)) ** 2
        
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
        
        # Omega squared (ω²) - less biased
        k = len(data)
        n_total = len(all_data)
        ms_within = (ss_total - ss_between) / (n_total - k)
        omega_squared = (ss_between - (k-1) * ms_within) / (ss_total + ms_within) if (ss_total + ms_within) > 0 else 0
        
        return {
            "eta_squared": {
                "value": float(eta_squared),
                "interpretation": self._interpret_eta_squared(eta_squared),
            },
            "omega_squared": {
                "value": float(omega_squared),
                "interpretation": self._interpret_eta_squared(omega_squared),
            },
        }
    
    def _interpret_eta_squared(self, eta2: float) -> str:
        """Interpret eta squared effect size."""
        if eta2 < 0.01:
            return "negligible"
        elif eta2 < 0.06:
            return "small"
        elif eta2 < 0.14:
            return "medium"
        else:
            return "large"
    
    def _interpret_analysis(self, analysis: Dict) -> Dict:
        """Provide interpretation of statistical analysis."""
        interpretation = {
            "summary": "",
            "key_findings": [],
            "limitations": [],
            "recommendations": [],
        }
        
        # Build summary
        n_groups = analysis["n_groups"]
        
        if n_groups == 2:
            # Two-group comparison
            comparison = analysis.get("comparison_tests", {})
            recommended = comparison.get("recommended_results", {})
            
            if recommended.get("significant", False):
                interpretation["summary"] = "Statistically significant difference found between groups."
            else:
                interpretation["summary"] = "No statistically significant difference found between groups."
            
            # Effect size
            effect_sizes = analysis.get("effect_sizes", {})
            if effect_sizes:
                for name, effect in effect_sizes.items():
                    if "cohens_d" in name:
                        interpretation["key_findings"].append(
                            f"Effect size ({name}): {effect['value']:.2f} ({effect['interpretation']})"
                        )
            
            # Power
            power = analysis.get("power_analysis", {})
            if power.get("adequately_powered", False):
                interpretation["key_findings"].append("Study adequately powered.")
            else:
                interpretation["limitations"].append(f"Underpowered study (power={power.get('current_power', 0):.2f}).")
                interpretation["recommendations"].append(
                    f"Increase sample size to {power.get('required_sample_per_group', 0):.0f} per group for 80% power."
                )
        
        elif n_groups > 2:
            # Multiple groups
            comparison = analysis.get("comparison_tests", {})
            anova = comparison.get("anova", {})
            
            if anova.get("significant", False):
                interpretation["summary"] = "Statistically significant differences found among groups (ANOVA)."
                interpretation["key_findings"].append("Post-hoc tests needed to identify specific group differences.")
                
                if "post_hoc" in comparison:
                    post_hoc = comparison["post_hoc"]
                    if "tukey_hsd" in post_hoc:
                        sig_pairs = post_hoc["tukey_hsd"].get("significant_pairs", [])
                        if sig_pairs:
                            interpretation["key_findings"].append(
                                f"Significant differences in {len(sig_pairs)} pairwise comparisons."
                            )
            else:
                interpretation["summary"] = "No statistically significant differences found among groups."
            
            # Effect size
            effect_sizes = analysis.get("effect_sizes", {})
            if effect_sizes:
                eta2 = effect_sizes.get("eta_squared", {}).get("value", 0)
                interpretation["key_findings"].append(
                    f"Effect size (η²): {eta2:.3f} ({self._interpret_eta_squared(eta2)})"
                )
        
        # Check assumptions
        normality = analysis.get("normality_tests", {})
        if normality:
            non_normal = [name for name, test in normality.items() 
                         if test.get("interpretation") == "not_normal"]
            if non_normal:
                interpretation["limitations"].append(
                    f"Non-normal distributions detected in groups: {', '.join(non_normal)}"
                )
                interpretation["recommendations"].append(
                    "Consider non-parametric tests or data transformations."
                )
        
        homogeneity = analysis.get("homogeneity_tests", {})
        if homogeneity and homogeneity.get("interpretation") == "heterogeneous":
            interpretation["limitations"].append("Heterogeneous variances detected.")
            interpretation["recommendations"].append("Use Welch's ANOVA or non-parametric tests.")
        
        return interpretation