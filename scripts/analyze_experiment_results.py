#!/usr/bin/env python3
"""
Comprehensive Experiment Results Analysis

Analyzes results from multiple experiments and generates:
1. Statistical significance tests
2. Publication-ready tables and figures
3. Economic impact analysis
4. Method comparison reports
"""

import json
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import argparse
import logging
from scipy import stats
import warnings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)


class ExperimentAnalyzer:
    """Comprehensive analyzer for experiment results."""
    
    def __init__(self, results_dir: str = "./experiments"):
        self.results_dir = Path(results_dir)
        self.results = []
        self.summary_stats = {}
        self.analysis_results = {}
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-paper')
        sns.set_palette("husl")
    
    def load_experiment_results(self, pattern: str = "**/*_results.json"):
        """Load experiment results from files."""
        logger.info(f"Loading experiment results from {self.results_dir}")
        
        result_files = list(self.results_dir.rglob(pattern))
        logger.info(f"Found {len(result_files)} result files")
        
        for file_path in result_files:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                self.results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")
        
        logger.info(f"Successfully loaded {len(self.results)} experiments")
    
    def analyze_statistical_significance(self) -> Dict:
        """Perform statistical significance analysis."""
        logger.info("\n📊 Performing statistical significance analysis...")
        
        if len(self.results) < 5:
            logger.warning("Insufficient results for statistical analysis")
            return {}
        
        # Group results by task and model
        task_groups = {}
        for result in self.results:
            if "config" not in result:
                continue
            
            task = result["config"].get("task", "unknown")
            model = result["config"].get("model_name", "unknown")
            accuracy = result.get("results", {}).get("evaluation_results", {}).get("eval_accuracy", 0)
            
            key = f"{model}_{task}"
            if key not in task_groups:
                task_groups[key] = []
            task_groups[key].append(accuracy)
        
        # Perform statistical tests
        statistical_tests = {}
        
        for group_name, accuracies in task_groups.items():
            if len(accuracies) >= 5:
                # Basic statistics
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies, ddof=1)
                sem_acc = stats.sem(accuracies)
                
                # One-sample t-test against random chance
                if "sst2" in group_name or "mrpc" in group_name or "qqp" in group_name:
                    chance_level = 0.5
                elif "mnli" in group_name:
                    chance_level = 1/3
                else:
                    chance_level = 0.5
                
                t_stat, p_value = stats.ttest_1samp(accuracies, chance_level)
                
                # Confidence interval
                ci = stats.t.interval(
                    0.95,
                    len(accuracies) - 1,
                    loc=mean_acc,
                    scale=sem_acc,
                )
                
                statistical_tests[group_name] = {
                    "n": len(accuracies),
                    "mean": float(mean_acc),
                    "std": float(std_acc),
                    "sem": float(sem_acc),
                    "ci_95": (float(ci[0]), float(ci[1])),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                    "effect_size": float((mean_acc - chance_level) / (std_acc + 1e-8)),
                }
        
        self.analysis_results["statistical_tests"] = statistical_tests
        
        # Print summary
        logger.info(f"\nStatistical Significance Summary:")
        for group_name, test in statistical_tests.items():
            significance = "✓" if test["significant"] else "✗"
            logger.info(f"  {group_name}: {significance} p={test['p_value']:.4f}, "
                       f"mean={test['mean']:.4f} ± {test['std']:.4f}")
        
        return statistical_tests
    
    def analyze_waste_computation(self) -> Dict:
        """Analyze wasted computation across experiments."""
        logger.info("\n💰 Analyzing wasted computation...")
        
        waste_results = []
        
        for result in self.results:
            if "results" not in result:
                continue
            
            # Try to extract plateau analysis
            plateau_info = self._extract_plateau_info(result)
            if plateau_info:
                waste_results.append(plateau_info)
        
        if not waste_results:
            logger.warning("No plateau information found in results")
            return {}
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(waste_results)
        
        # Calculate waste statistics
        waste_stats = {
            "total_experiments": len(waste_results),
            "mean_waste_percentage": float(df["waste_percentage"].mean()),
            "median_waste_percentage": float(df["waste_percentage"].median()),
            "std_waste_percentage": float(df["waste_percentage"].std(ddof=1)),
            "min_waste_percentage": float(df["waste_percentage"].min()),
            "max_waste_percentage": float(df["waste_percentage"].max()),
            "total_wasted_steps": int(df["wasted_steps"].sum()),
            "total_compute_steps": int(df["total_steps"].sum()),
            "overall_waste_percentage": float(df["wasted_steps"].sum() / df["total_steps"].sum() * 100),
        }
        
        # Group by task
        task_stats = {}
        for task in df["task"].unique():
            task_df = df[df["task"] == task]
            task_stats[task] = {
                "n": len(task_df),
                "mean_waste": float(task_df["waste_percentage"].mean()),
                "std_waste": float(task_df["waste_percentage"].std(ddof=1)),
                "min_waste": float(task_df["waste_percentage"].min()),
                "max_waste": float(task_df["waste_percentage"].max()),
            }
        
        waste_analysis = {
            "summary": waste_stats,
            "by_task": task_stats,
            "raw_data": waste_results,
        }
        
        self.analysis_results["waste_analysis"] = waste_analysis
        
        # Print summary
        logger.info(f"\nWaste Computation Analysis:")
        logger.info(f"  Total experiments: {waste_stats['total_experiments']}")
        logger.info(f"  Mean waste: {waste_stats['mean_waste_percentage']:.1f}%")
        logger.info(f"  Overall waste: {waste_stats['overall_waste_percentage']:.1f}%")
        logger.info(f"  Total wasted steps: {waste_stats['total_wasted_steps']:,}")
        
        logger.info(f"\nWaste by task:")
        for task, stats in task_stats.items():
            logger.info(f"  {task}: {stats['mean_waste']:.1f}% ± {stats['std_waste']:.1f}% (n={stats['n']})")
        
        return waste_analysis
    
    def _extract_plateau_info(self, result: Dict) -> Optional[Dict]:
        """Extract plateau information from experiment result."""
        try:
            # Check for plateau analysis in various locations
            plateau_data = None
            
            # Look in results
            if "results" in result:
                # Check for IES analysis
                if "plateau_analysis" in result["results"]:
                    plateau_data = result["results"]["plateau_analysis"]
                # Check for efficiency metrics
                elif "efficiency_analysis" in result["results"]:
                    plateau_data = result["results"]["efficiency_analysis"]
            
            # Look in report
            if not plateau_data and "report" in result:
                if "efficiency_analysis" in result["report"]:
                    plateau_data = result["report"]["efficiency_analysis"]
            
            if plateau_data:
                task = result["config"].get("task", "unknown")
                model = result["config"].get("model_name", "unknown")
                
                # Extract waste information
                waste_percentage = plateau_data.get("wasted_percentage", 0)
                plateau_step = plateau_data.get("plateau_step", 0)
                total_steps = plateau_data.get("current_step", 0)
                wasted_steps = total_steps - plateau_step if total_steps > plateau_step else 0
                
                return {
                    "task": task,
                    "model": model,
                    "waste_percentage": waste_percentage,
                    "plateau_step": plateau_step,
                    "total_steps": total_steps,
                    "wasted_steps": wasted_steps,
                }
            
        except Exception as e:
            logger.debug(f"Failed to extract plateau info: {e}")
        
        return None
    
    def analyze_efficiency_metrics(self) -> Dict:
        """Analyze efficiency metrics (LER, GSNR, etc.)."""
        logger.info("\n📈 Analyzing efficiency metrics...")
        
        efficiency_data = []
        
        for result in self.results:
            if "results" not in result:
                continue
            
            # Extract efficiency metrics
            metrics = self._extract_efficiency_metrics(result)
            if metrics:
                metrics.update({
                    "task": result["config"].get("task", "unknown"),
                    "model": result["config"].get("model_name", "unknown"),
                })
                efficiency_data.append(metrics)
        
        if not efficiency_data:
            logger.warning("No efficiency metrics found")
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(efficiency_data)
        
        # Analyze each metric
        efficiency_analysis = {}
        
        for metric in ["ler", "gsnr", "probe_accuracy"]:
            if metric in df.columns:
                values = df[metric].dropna()
                if len(values) > 0:
                    efficiency_analysis[metric] = {
                        "n": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=1)),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "correlation_with_accuracy": self._calculate_correlation(df, metric, "accuracy"),
                    }
        
        self.analysis_results["efficiency_metrics"] = efficiency_analysis
        
        # Print summary
        logger.info(f"\nEfficiency Metrics Summary:")
        for metric, stats in efficiency_analysis.items():
            logger.info(f"  {metric.upper()}: mean={stats['mean']:.4f} ± {stats['std']:.4f} (n={stats['n']})")
            if "correlation_with_accuracy" in stats:
                corr = stats["correlation_with_accuracy"]
                logger.info(f"    Correlation with accuracy: r={corr.get('correlation', 0):.3f}, "
                           f"p={corr.get('p_value', 1):.4f}")
        
        return efficiency_analysis
    
    def _extract_efficiency_metrics(self, result: Dict) -> Optional[Dict]:
        """Extract efficiency metrics from result."""
        try:
            metrics = {}
            
            # Look in various locations
            sources = [
                result.get("results", {}).get("efficiency_metrics", {}),
                result.get("report", {}).get("efficiency_analysis", {}),
                result.get("results", {}).get("metrics_history", [])[-1] if result.get("results", {}).get("metrics_history") else {},
            ]
            
            for source in sources:
                if not source:
                    continue
                
                # Extract LER
                if "ler" in source:
                    metrics["ler"] = source["ler"]
                elif "efficiency/ler" in source:
                    metrics["ler"] = source["efficiency/ler"]
                
                # Extract GSNR
                if "gsnr" in source:
                    metrics["gsnr"] = source["gsnr"]
                elif "efficiency/gsnr" in source:
                    metrics["gsnr"] = source["efficiency/gsnr"]
                
                # Extract probe accuracy
                if "probe_accuracy" in source:
                    metrics["probe_accuracy"] = source["probe_accuracy"]
                
                # Extract accuracy
                if "accuracy" in source:
                    metrics["accuracy"] = source["accuracy"]
                elif "eval_accuracy" in source:
                    metrics["accuracy"] = source["eval_accuracy"]
            
            return metrics if metrics else None
            
        except Exception as e:
            logger.debug(f"Failed to extract efficiency metrics: {e}")
            return None
    
    def _calculate_correlation(self, df: pd.DataFrame, x_col: str, y_col: str) -> Dict:
        """Calculate correlation between two columns."""
        if x_col not in df.columns or y_col not in df.columns:
            return {}
        
        # Drop missing values
        valid_data = df[[x_col, y_col]].dropna()
        if len(valid_data) < 5:
            return {}
        
        x = valid_data[x_col]
        y = valid_data[y_col]
        
        # Calculate Pearson correlation
        correlation, p_value = stats.pearsonr(x, y)
        
        # Calculate R²
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            "correlation": float(correlation),
            "p_value": float(p_value),
            "r_squared": float(r_squared),
            "n": len(valid_data),
            "significant": p_value < 0.05,
        }
    
    def analyze_economic_impact(self, gpu_cost_per_hour: float = 2.99) -> Dict:
        """Analyze economic impact of wasted compute."""
        logger.info("\n💸 Analyzing economic impact...")
        
        # Get waste analysis
        if "waste_analysis" not in self.analysis_results:
            self.analyze_waste_computation()
        
        waste_analysis = self.analysis_results.get("waste_analysis", {})
        if not waste_analysis:
            logger.warning("No waste analysis available")
            return {}
        
        # Calculate costs
        summary = waste_analysis.get("summary", {})
        total_wasted_steps = summary.get("total_wasted_steps", 0)
        total_compute_steps = summary.get("total_compute_steps", 0)
        
        # Estimate time per step (conservative: 0.1 seconds per step)
        time_per_step_seconds = 0.1
        total_wasted_seconds = total_wasted_steps * time_per_step_seconds
        total_wasted_hours = total_wasted_seconds / 3600
        
        # Calculate costs
        wasted_cost = total_wasted_hours * gpu_cost_per_hour
        total_cost = (total_compute_steps * time_per_step_seconds / 3600) * gpu_cost_per_hour
        
        # Market segmentation analysis
        market_segments = {
            "academic_research": {
                "annual_runs": 500000,
                "avg_cost_per_run": 30,
                "waste_rate": summary.get("overall_waste_percentage", 0) / 100,
                "adoption_rate": 0.05,
            },
            "industry_production": {
                "annual_runs": 2000000,
                "avg_cost_per_run": 200,
                "waste_rate": summary.get("overall_waste_percentage", 0) / 100 * 0.8,  # 20% less waste
                "adoption_rate": 0.20,
            },
            "industry_experimentation": {
                "annual_runs": 5000000,
                "avg_cost_per_run": 50,
                "waste_rate": summary.get("overall_waste_percentage", 0) / 100 * 1.2,  # 20% more waste
                "adoption_rate": 0.50,
            },
        }
        
        # Calculate potential savings
        potential_savings = {}
        total_annual_savings = 0
        
        for segment, params in market_segments.items():
            segment_waste_cost = params["annual_runs"] * params["avg_cost_per_run"] * params["waste_rate"]
            segment_savings = segment_waste_cost * params["adoption_rate"]
            
            potential_savings[segment] = {
                "annual_waste_cost": segment_waste_cost,
                "potential_savings": segment_savings,
                "adoption_rate": params["adoption_rate"],
            }
            
            total_annual_savings += segment_savings
        
        # Scenario analysis
        scenarios = {
            "conservative": {
                "market_adoption": 0.05,
                "waste_reduction": 0.25,
                "implementation_cost": 10000000,
            },
            "moderate": {
                "market_adoption": 0.20,
                "waste_reduction": 0.50,
                "implementation_cost": 5000000,
            },
            "optimistic": {
                "market_adoption": 0.50,
                "waste_reduction": 0.75,
                "implementation_cost": 2000000,
            },
        }
        
        scenario_results = {}
        for scenario_name, params in scenarios.items():
            total_waste_cost = sum(seg["annual_waste_cost"] for seg in potential_savings.values())
            potential_savings_scenario = total_waste_cost * params["market_adoption"] * params["waste_reduction"]
            net_benefit = potential_savings_scenario - params["implementation_cost"]
            
            scenario_results[scenario_name] = {
                "potential_savings": potential_savings_scenario,
                "implementation_cost": params["implementation_cost"],
                "net_benefit": net_benefit,
                "roi": net_benefit / params["implementation_cost"] if params["implementation_cost"] > 0 else float('inf'),
            }
        
        economic_analysis = {
            "experiment_level": {
                "total_wasted_steps": total_wasted_steps,
                "total_wasted_hours": total_wasted_hours,
                "wasted_cost": wasted_cost,
                "total_cost": total_cost,
                "waste_percentage": summary.get("overall_waste_percentage", 0),
                "gpu_cost_per_hour": gpu_cost_per_hour,
            },
            "market_analysis": potential_savings,
            "scenario_analysis": scenario_results,
            "total_annual_savings": total_annual_savings,
            "recommendations": self._generate_economic_recommendations(total_annual_savings),
        }
        
        self.analysis_results["economic_analysis"] = economic_analysis
        
        # Print summary
        logger.info(f"\nEconomic Impact Analysis:")
        logger.info(f"  Experiment waste: {wasted_cost:.2f} (${total_wasted_hours:.1f} GPU-hours)")
        logger.info(f"  Overall waste: {summary.get('overall_waste_percentage', 0):.1f}%")
        
        logger.info(f"\nMarket Analysis:")
        for segment, data in potential_savings.items():
            logger.info(f"  {segment}: ${data['annual_waste_cost']/1e6:.1f}M waste, "
                       f"${data['potential_savings']/1e6:.1f}M potential savings")
        
        logger.info(f"\nScenario Analysis:")
        for scenario, data in scenario_results.items():
            logger.info(f"  {scenario}: ${data['net_benefit']/1e6:.1f}M net benefit, "
                       f"ROI: {data['roi']:.1f}x")
        
        logger.info(f"\nTotal annual savings potential: ${total_annual_savings/1e6:.1f}M")
        
        return economic_analysis
    
    def _generate_economic_recommendations(self, total_savings: float) -> List[str]:
        """Generate economic recommendations."""
        recommendations = []
        
        if total_savings > 100000000:  # > $100M
            recommendations.append(
                f"High impact potential (${total_savings/1e6:.0f}M annually). "
                "Strong business case for implementation."
            )
        elif total_savings > 10000000:  # > $10M
            recommendations.append(
                f"Significant impact potential (${total_savings/1e6:.0f}M annually). "
                "Worth pursuing with dedicated resources."
            )
        else:
            recommendations.append(
                f"Moderate impact potential (${total_savings/1e6:.1f}M annually). "
                "Consider implementation as part of broader efficiency initiatives."
            )
        
        recommendations.extend([
            "Focus initial implementation on industry experimentation segment "
            "(highest adoption rate, significant waste).",
            "Develop open-source implementation to drive academic adoption.",
            "Create integration with popular ML frameworks (PyTorch, HuggingFace).",
        ])
        
        return recommendations
    
    def generate_publication_figures(self, output_dir: str = "./experiments/analysis/figures"):
        """Generate publication-ready figures."""
        logger.info("\n🎨 Generating publication figures...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        # 1. Waste distribution histogram
        if "waste_analysis" in self.analysis_results:
            fig1 = self._create_waste_distribution_plot(output_path)
            figures["waste_distribution"] = str(fig1)
        
        # 2. Task comparison box plot
        if "waste_analysis" in self.analysis_results:
            fig2 = self._create_task_comparison_plot(output_path)
            figures["task_comparison"] = str(fig2)
        
        # 3. Efficiency metrics correlation
        if "efficiency_metrics" in self.analysis_results:
            fig3 = self._create_efficiency_correlation_plot(output_path)
            figures["efficiency_correlation"] = str(fig3)
        
        # 4. Economic impact visualization
        if "economic_analysis" in self.analysis_results:
            fig4 = self._create_economic_impact_plot(output_path)
            figures["economic_impact"] = str(fig4)
        
        self.analysis_results["figures"] = figures
        
        logger.info(f"Generated {len(figures)} figures in {output_path}")
        
        return figures
    
    def _create_waste_distribution_plot(self, output_path: Path) -> Path:
        """Create waste distribution histogram."""
        waste_data = self.analysis_results["waste_analysis"]["raw_data"]
        df = pd.DataFrame(waste_data)
        
        plt.figure(figsize=(10, 6))
        
        # Histogram
        plt.hist(df["waste_percentage"], bins=20, edgecolor='black', alpha=0.7)
        
        # Add statistics
        mean_waste = df["waste_percentage"].mean()
        median_waste = df["waste_percentage"].median()
        
        plt.axvline(mean_waste, color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {mean_waste:.1f}%')
        plt.axvline(median_waste, color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {median_waste:.1f}%')
        
        plt.xlabel('Wasted Compute (%)', fontsize=12)
        plt.ylabel('Number of Runs', fontsize=12)
        plt.title('Distribution of Compute Waste in Fine-Tuning', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Save
        filepath = output_path / "waste_distribution.pdf"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_task_comparison_plot(self, output_path: Path) -> Path:
        """Create box plot comparing waste by task."""
        waste_data = self.analysis_results["waste_analysis"]["raw_data"]
        df = pd.DataFrame(waste_data)
        
        plt.figure(figsize=(12, 6))
        
        # Box plot
        sns.boxplot(data=df, x='task', y='waste_percentage')
        
        # Add mean line
        mean_waste = df["waste_percentage"].mean()
        plt.axhline(mean_waste, color='red', linestyle='--', alpha=0.5,
                   label=f'Overall mean: {mean_waste:.1f}%')
        
        plt.xlabel('GLUE Task', fontsize=12)
        plt.ylabel('Wasted Compute (%)', fontsize=12)
        plt.title('Compute Waste by Task Type', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save
        filepath = output_path / "task_comparison.pdf"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def _create_efficiency_correlation_plot(self, output_path: Path) -> Path:
        """Create correlation plot for efficiency metrics."""
        # This would need efficiency metrics data
        # For now, create a placeholder
        filepath = output_path / "efficiency_correlation.pdf"
        
        # Create a simple correlation matrix if we have the data
        try:
            # Extract efficiency metrics
            efficiency_data = []
            for result in self.results:
                metrics = self._extract_efficiency_metrics(result)
                if metrics:
                    efficiency_data.append(metrics)
            
            if efficiency_data:
                df = pd.DataFrame(efficiency_data)
                
                # Select numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) >= 2:
                    # Create correlation matrix
                    corr_matrix = df[numeric_cols].corr()
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', 
                               center=0, fmt='.2f', square=True)
                    plt.title('Correlation Matrix of Efficiency Metrics', fontsize=14)
                    
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Could not create efficiency correlation plot: {e}")
        
        return filepath
    
    def _create_economic_impact_plot(self, output_path: Path) -> Path:
        """Create economic impact visualization."""
        economic_data = self.analysis_results.get("economic_analysis", {})
        if not economic_data:
            return output_path / "economic_impact.pdf"
        
        market_data = economic_data.get("market_analysis", {})
        scenario_data = economic_data.get("scenario_analysis", {})
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Market segment waste
        if market_data:
            segments = list(market_data.keys())
            waste_costs = [data["annual_waste_cost"] / 1e6 for data in market_data.values()]
            potential_savings = [data["potential_savings"] / 1e6 for data in market_data.values()]
            
            x = np.arange(len(segments))
            width = 0.35
            
            axes[0].bar(x - width/2, waste_costs, width, label='Annual Waste Cost', color='red', alpha=0.7)
            axes[0].bar(x + width/2, potential_savings, width, label='Potential Savings', color='green', alpha=0.7)
            
            axes[0].set_xlabel('Market Segment', fontsize=12)
            axes[0].set_ylabel('Cost (Millions USD)', fontsize=12)
            axes[0].set_title('Annual Waste Cost and Potential Savings by Market Segment', fontsize=14)
            axes[0].set_xticks(x)
            axes[0].set_xticklabels([s.replace('_', ' ').title() for s in segments])
            axes[0].legend(fontsize=10)
            axes[0].grid(True, alpha=0.3, axis='y')
        
        # Plot 2: Scenario analysis
        if scenario_data:
            scenarios = list(scenario_data.keys())
            net_benefits = [data["net_benefit"] / 1e6 for data in scenario_data.values()]
            rois = [data.get("roi", 0) for data in scenario_data.values()]
            
            x = np.arange(len(scenarios))
            width = 0.35
            
            ax2 = axes[1]
            bars = ax2.bar(x, net_benefits, width, color='blue', alpha=0.7)
            
            # Add ROI on secondary axis
            ax2_secondary = ax2.twinx()
            ax2_secondary.plot(x, rois, color='orange', marker='o', linewidth=2, markersize=8)
            
            ax2.set_xlabel('Scenario', fontsize=12)
            ax2.set_ylabel('Net Benefit (Millions USD)', fontsize=12, color='blue')
            ax2_secondary.set_ylabel('ROI (x)', fontsize=12, color='orange')
            ax2.set_title('Scenario Analysis: Net Benefit and ROI', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([s.title() for s in scenarios])
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, value in zip(bars, net_benefits):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        f'${value:.1f}M', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        filepath = output_path / "economic_impact.pdf"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filepath
    
    def generate_latex_tables(self, output_dir: str = "./experiments/analysis/tables"):
        """Generate LaTeX tables for publication."""
        logger.info("\n📋 Generating LaTeX tables...")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        tables = {}
        
        # 1. Statistical significance table
        if "statistical_tests" in self.analysis_results:
            tables["statistical_significance"] = self._create_statistical_table(output_path)
        
        # 2. Waste analysis table
        if "waste_analysis" in self.analysis_results:
            tables["waste_analysis"] = self._create_waste_table(output_path)
        
        # 3. Economic impact table
        if "economic_analysis" in self.analysis_results:
            tables["economic_impact"] = self._create_economic_table(output_path)
        
        self.analysis_results["latex_tables"] = tables
        
        logger.info(f"Generated {len(tables)} LaTeX tables in {output_path}")
        
        return tables
    
    def _create_statistical_table(self, output_path: Path) -> Path:
        """Create LaTeX table for statistical significance."""
        stats = self.analysis_results["statistical_tests"]
        
        # Convert to DataFrame
        rows = []
        for group_name, test in stats.items():
            model, task = group_name.split('_', 1) if '_' in group_name else (group_name, 'unknown')
            
            rows.append({
                'Model': model,
                'Task': task,
                'n': test['n'],
                'Mean': f"{test['mean']:.4f}",
                'Std': f"{test['std']:.4f}",
                '95\\% CI': f"[{test['ci_95'][0]:.4f}, {test['ci_95'][1]:.4f}]",
                'p-value': f"{test['p_value']:.4f}",
                'Significant': 'Yes' if test['significant'] else 'No',
            })
        
        df = pd.DataFrame(rows)
        
        # Generate LaTeX
        latex = df.to_latex(
            index=False,
            caption='Statistical Significance Analysis of Model Performance',
            label='tab:statistical_significance',
            position='htbp',
            escape=False,
        )
        
        filepath = output_path / "statistical_significance.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return filepath
    
    def _create_waste_table(self, output_path: Path) -> Path:
        """Create LaTeX table for waste analysis."""
        waste_data = self.analysis_results["waste_analysis"]
        
        # Create summary table
        summary = waste_data.get("summary", {})
        task_stats = waste_data.get("by_task", {})
        
        # Summary row
        summary_row = {
            'Task': 'Overall',
            'n': summary.get('total_experiments', 0),
            'Mean Waste (\\%)': f"{summary.get('mean_waste_percentage', 0):.1f}",
            'Std (\\%)': f"{summary.get('std_waste_percentage', 0):.1f}",
            'Min (\\%)': f"{summary.get('min_waste_percentage', 0):.1f}",
            'Max (\\%)': f"{summary.get('max_waste_percentage', 0):.1f}",
        }
        
        # Task rows
        task_rows = []
        for task, stats in task_stats.items():
            task_rows.append({
                'Task': task.upper(),
                'n': stats['n'],
                'Mean Waste (\\%)': f"{stats['mean_waste']:.1f}",
                'Std (\\%)': f"{stats['std_waste']:.1f}",
                'Min (\\%)': f"{stats['min_waste']:.1f}",
                'Max (\\%)': f"{stats['max_waste']:.1f}",
            })
        
        # Combine
        df = pd.DataFrame([summary_row] + task_rows)
        
        # Generate LaTeX
        latex = df.to_latex(
            index=False,
            caption='Compute Waste Analysis by Task',
            label='tab:waste_analysis',
            position='htbp',
            escape=False,
        )
        
        filepath = output_path / "waste_analysis.tex"
        with open(filepath, 'w') as f:
            f.write(latex)
        
        return filepath
    
    def _create_economic_table(self, output_path: Path) -> Path:
        """Create LaTeX table for economic impact."""
        economic_data = self.analysis_results["economic_analysis"]
        market_data = economic_data.get("market_analysis", {})
        scenario_data = economic_data.get("scenario_analysis", {})
        
        # Market analysis table
        market_rows = []
        for segment, data in market_data.items():
            market_rows.append({
                'Segment': segment.replace('_', ' ').title(),
                'Annual Runs': f"{data.get('annual_waste_cost', 0)/data.get('avg_cost_per_run', 1):,.0f}",
                'Avg Cost/Run (\\$)': f"${data.get('avg_cost_per_run', 0):,.0f}",
                'Waste Rate (\\%)': f"{data.get('waste_rate', 0)*100:.1f}\\%",
                'Annual Waste (M\\$)': f"${data.get('annual_waste_cost', 0)/1e6:.1f}",
                'Potential Savings (M\\$)': f"${data.get('potential_savings', 0)/1e6:.1f}",
            })
        
        df_market = pd.DataFrame(market_rows)
        
        # Scenario analysis table
        scenario_rows = []
        for scenario, data in scenario_data.items():
            scenario_rows.append({
                'Scenario': scenario.title(),
                'Market Adoption (\\%)': f"{data.get('market_adoption', 0)*100:.0f}\\%",
                'Waste Reduction (\\%)': f"{data.get('waste_reduction', 0)*100:.0f}\\%",
                'Potential Savings (M\\$)': f"${data.get('potential_savings', 0)/1e6:.1f}",
                'Implementation Cost (M\\$)': f"${data.get('implementation_cost', 0)/1e6:.1f}",
                'Net Benefit (M\\$)': f"${data.get('net_benefit', 0)/1e6:.1f}",
                'ROI (x)': f"{data.get('roi', 0):.1f}",
            })
        
        df_scenario = pd.DataFrame(scenario_rows)
        
        # Generate LaTeX
        latex_market = df_market.to_latex(
            index=False,
            caption='Market Analysis of Compute Waste',
            label='tab:market_analysis',
            position='htbp',
            escape=False,
        )
        
        latex_scenario = df_scenario.to_latex(
            index=False,
            caption='Scenario Analysis of Economic Impact',
            label='tab:scenario_analysis',
            position='htbp',
            escape=False,
        )
        
        # Combine tables
        combined_latex = latex_market + "\n\n" + latex_scenario
        
        filepath = output_path / "economic_analysis.tex"
        with open(filepath, 'w') as f:
            f.write(combined_latex)
        
        return filepath
    
    def save_analysis_report(self, output_path: str = "./experiments/analysis/final_report.json"):
        """Save comprehensive analysis report."""
        logger.info(f"\n💾 Saving analysis report to {output_path}")
        
        # Ensure all analyses are complete
        if "statistical_tests" not in self.analysis_results:
            self.analyze_statistical_significance()
        
        if "waste_analysis" not in self.analysis_results:
            self.analyze_waste_computation()
        
        if "efficiency_metrics" not in self.analysis_results:
            self.analyze_efficiency_metrics()
        
        if "economic_analysis" not in self.analysis_results:
            self.analyze_economic_impact()
        
        # Generate figures and tables
        self.generate_publication_figures()
        self.generate_latex_tables()
        
        # Create final report
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_experiments": len(self.results),
                "key_findings": self._extract_key_findings(),
            },
            "analyses": self.analysis_results,
            "recommendations": self._generate_overall_recommendations(),
            "metadata": {
                "analysis_timestamp": datetime.now().isoformat(),
                "analyzer_version": "1.0.0",
                "results_directory": str(self.results_dir),
            },
        }
        
        # Save report
        output_path_obj = Path(output_path)
        output_path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path_obj, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"✅ Analysis report saved: {output_path}")
        
        # Print executive summary
        self._print_executive_summary(report)
        
        return report
    
    def _extract_key_findings(self) -> List[str]:
        """Extract key findings from analyses."""
        findings = []
        
        # Statistical findings
        if "statistical_tests" in self.analysis_results:
            sig_tests = [t for t in self.analysis_results["statistical_tests"].values() 
                        if t.get("significant", False)]
            if sig_tests:
                findings.append(f"Found statistically significant performance in {len(sig_tests)} experiments")
        
        # Waste findings
        if "waste_analysis" in self.analysis_results:
            waste_stats = self.analysis_results["waste_analysis"]["summary"]
            findings.append(
                f"Average compute waste: {waste_stats.get('mean_waste_percentage', 0):.1f}% "
                f"(overall: {waste_stats.get('overall_waste_percentage', 0):.1f}%)"
            )
        
        # Economic findings
        if "economic_analysis" in self.analysis_results:
            econ_data = self.analysis_results["economic_analysis"]
            total_savings = econ_data.get("total_annual_savings", 0)
            findings.append(f"Potential annual savings: ${total_savings/1e6:.1f}M")
        
        return findings
    
    def _generate_overall_recommendations(self) -> List[str]:
        """Generate overall recommendations."""
        recommendations = [
            "Implement early stopping based on efficiency metrics to reduce compute waste",
            "Use mixed precision training (BF16/FP16) for faster training and lower memory usage",
            "Regularly monitor efficiency metrics (LER, GSNR) during training",
            "Validate model performance with statistical significance testing",
            "Consider economic impact when designing training schedules",
        ]
        
        # Add waste-specific recommendations
        if "waste_analysis" in self.analysis_results:
            waste_stats = self.analysis_results["waste_analysis"]["summary"]
            waste_pct = waste_stats.get("overall_waste_percentage", 0)
            
            if waste_pct > 30:
                recommendations.insert(0, 
                    f"High compute waste detected ({waste_pct:.1f}%). "
                    "Prioritize efficiency improvements in training pipelines."
                )
        
        return recommendations
    
    def _print_executive_summary(self, report: Dict):
        """Print executive summary of findings."""
        logger.info(f"\n{'='*60}")
        logger.info("EXECUTIVE SUMMARY")
        logger.info(f"{'='*60}")
        
        summary = report.get("summary", {})
        findings = summary.get("key_findings", [])
        
        logger.info(f"\n📊 Key Findings:")
        for finding in findings:
            logger.info(f"  • {finding}")
        
        recommendations = report.get("recommendations", [])
        logger.info(f"\n💡 Recommendations:")
        for rec in recommendations[:5]:  # Show top 5
            logger.info(f"  • {rec}")
        
        # Economic impact
        if "economic_analysis" in report.get("analyses", {}):
            econ_data = report["analyses"]["economic_analysis"]
            total_savings = econ_data.get("total_annual_savings", 0)
            
            logger.info(f"\n💰 Economic Impact:")
            logger.info(f"  Annual savings potential: ${total_savings/1e6:.1f}M")
            
            # Best scenario
            scenarios = econ_data.get("scenario_analysis", {})
            if scenarios:
                best_scenario = max(scenarios.items(), key=lambda x: x[1].get("net_benefit", 0))
                logger.info(f"  Best case ({best_scenario[0]}): ${best_scenario[1].get('net_benefit', 0)/1e6:.1f}M net benefit")
        
        logger.info(f"\n📁 Report saved: {report.get('metadata', {}).get('results_directory', '')}")
        logger.info(f"{'='*60}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./experiments",
        help="Directory containing experiment results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./experiments/analysis",
        help="Output directory for analysis results",
    )
    parser.add_argument(
        "--gpu-cost",
        type=float,
        default=2.99,
        help="GPU cost per hour for economic analysis",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick analysis (skip some visualizations)",
    )
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = ExperimentAnalyzer(results_dir=args.results_dir)
    
    # Load results
    analyzer.load_experiment_results()
    
    if not analyzer.results:
        logger.error("No experiment results found!")
        return
    
    # Run analyses
    analyzer.analyze_statistical_significance()
    analyzer.analyze_waste_computation()
    analyzer.analyze_efficiency_metrics()
    analyzer.analyze_economic_impact(gpu_cost_per_hour=args.gpu_cost)
    
    # Generate outputs
    if not args.quick:
        analyzer.generate_publication_figures(output_dir=f"{args.output_dir}/figures")
        analyzer.generate_latex_tables(output_dir=f"{args.output_dir}/tables")
    
    # Save comprehensive report
    report_path = f"{args.output_dir}/final_report.json"
    analyzer.save_analysis_report(report_path)
    
    logger.info("\n🎉 Analysis complete!")


if __name__ == "__main__":
    main()