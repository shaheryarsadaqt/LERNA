"""
VeLoRA Pillar 0: Wasted Compute Quantification in Transformer Fine-tuning

Academic research package for quantifying compute waste and developing
efficiency metrics for transformer model fine-tuning.

Version: research-2.0
Authors: VeLoRA Research Team, Harbin Institute of Technology
License: MIT
"""

__version__ = "2.0.0"
__author__ = "VeLoRA Research Team"
__email__ = "research@velora.ai"
__license__ = "MIT"

# Core exports
from .utils.plateau_ies import (
    IESPlateauDetector,
    SecondOrderDifferenceDetector,
    PlateauAnalysisResult,
    compute_statistical_significance,
)

from .utils.efficiency_metrics import (
    GSNRTracker,
    LERTracker,
    ProbeAccuracyTracker,
    EfficiencyMetricsCollector,
    validate_ler_metric,
    compute_effect_sizes,
)

from .utils.experiment_tracking import (
    ResearchExperimentLogger,
    StatisticalAnalysisEngine,
    PublicationFigureGenerator,
    EconomicImpactAnalyzer,
    ReproducibilityManager,
)

from .callbacks.ies_callback import (
    IESCallback,
    EfficiencyMonitoringCallback,
    EarlyStoppingWithLER,
    CheckpointRestorationCallback,
)

from .callbacks.efficiency_callback import (
    EfficiencyMetricsCallback,
    ProbeAccuracyCallback,
    GradientAnalysisCallback,
    ComputeCostTracker,
)

# Constants
SUPPORTED_MODELS = {
    "bert-base-uncased": {"parameters": 110_000_000, "type": "encoder"},
    "roberta-base": {"parameters": 125_000_000, "type": "encoder"},
    "deberta-v3-base": {"parameters": 184_000_000, "type": "encoder"},
    "modernbert-base": {"parameters": 149_000_000, "type": "encoder"},
}

GLUE_TASKS = {
    "classification": ["sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola"],
    "regression": ["stsb"],
}

STATISTICAL_REQUIREMENTS = {
    "minimum_runs": 50,
    "confidence_level": 0.95,
    "power_threshold": 0.80,
}

__all__ = [
    # Core components
    "IESPlateauDetector",
    "GSNRTracker",
    "LERTracker",
    "ResearchExperimentLogger",
    
    # Callbacks
    "IESCallback",
    "EfficiencyMetricsCallback",
    
    # Statistical analysis
    "StatisticalAnalysisEngine",
    "compute_statistical_significance",
    "compute_effect_sizes",
    
    # Constants
    "SUPPORTED_MODELS",
    "GLUE_TASKS",
    "STATISTICAL_REQUIREMENTS",
]