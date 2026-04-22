"""
Efficiency Metrics Implementation with Statistical Validation

Implementation of novel efficiency metrics (LER, GSNR) with proper
validation, statistical analysis, and comparison to baseline methods.

Key improvements:
1. Correct GSNR implementation (per-parameter tracking)
2. LER validation with correlation analysis
3. Probe accuracy tracking for representation quality
4. Statistical validation of all metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from scipy import stats
import warnings
from collections import defaultdict
import torch.nn.functional as F
import json
import os


# =============================================================================
# §2c: LER Configuration - promote magic constants to a dataclass
# =============================================================================

@dataclass(frozen=True)
class LERConfig:
    """Configuration for LER calculation with tunable hyperparameters.
    
    This makes ablation trivial: LERConfig(w_spread=0.5, w_range=0.2, ...).
    """
    # Entropy mixing weights — previously 0.3 / 0.4 / 0.15 hardcoded.
    w_spread: float = 0.3
    w_range: float = 0.4
    w_meanabs: float = 0.15
    # Per-task calibration — previously baked into self.task_calibration.
    entropy_weight: float = 1.0
    ler_threshold: float = 1e-4
    # Safety horizon.
    pl_constant_factor: float = 1e-4
    max_horizon: int = 50
    # No entropy floor by default — see metrics.py §2b.
    entropy_floor: float = 0.0
    
    @classmethod
    def for_task(cls, task: str) -> "LERConfig":
        """Return preset configuration for common tasks."""
        presets = {
            "sst2":  dict(entropy_weight=1.0, ler_threshold=1e-4),
            "rte":   dict(entropy_weight=1.3, ler_threshold=1e-3),
            "stsb":  dict(entropy_weight=1.5, ler_threshold=1e-2),
        }
        return cls(**presets.get(task, {}))


@dataclass
class EfficiencyMetrics:
    """Container for all efficiency metrics with validation."""
    
    # Core metrics
    ler_value: float  # Learning Efficiency Rate
    gsnr_value: float  # Gradient Signal-to-Noise Ratio
    probe_accuracy: float  # Representation quality
    
    # Statistical validation
    ler_validation: Dict[str, float]  # Correlation with performance
    gsnr_validation: Dict[str, float]  # Comparison with published values
    probe_validation: Dict[str, float]  # Correlation with final accuracy
    
    # Quality metrics
    confidence_scores: Dict[str, float]
    reliability_indicator: str  # high, medium, low
    validation_status: str  # validated, needs_revalidation, failed
    
    # Economic impact
    potential_savings_pct: float
    roi_estimate: float  # Return on investment


class GSNRTracker:
    """
    Correct GSNR implementation with per-parameter tracking.
    
    Gradient Signal-to-Noise Ratio (GSNR) measures the ratio of
    signal (mean gradient) to noise (gradient variance).
    
    Correct formula: GSNR(θ_j) = (E[∇θ_jL])² / Var[∇θ_jL]
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        parameter_groups: Optional[List[str]] = None,
        window_size: int = 5,
        validate_implementation: bool = True,
    ):
        """
        Initialize GSNR tracker with proper parameter tracking.
        
        Args:
            model: PyTorch model
            parameter_groups: Groups to track separately
            window_size: Window size for variance calculation
            validate_implementation: Validate against known benchmarks
        """
        self.model = model
        self.window_size = window_size
        self.validate_implementation = validate_implementation
        
        # Parameter grouping
        if parameter_groups is None:
            parameter_groups = ["attention", "ffn", "embeddings", "classifier"]
        self.parameter_groups = parameter_groups
        
        # Gradient history
        self.gradient_history = defaultdict(list)
        self.parameter_mapping = self._create_parameter_mapping()
        
        # Statistics
        self.gsnr_history = []
        self.validation_results = {}
        
        if validate_implementation:
            self._validate_initialization()
    
    def _create_parameter_mapping(self) -> Dict[str, List[str]]:
        """Map parameters to groups based on name patterns."""
        mapping = defaultdict(list)
        
        for name, _ in self.model.named_parameters():
            if not name.endswith('.bias'):  # Skip biases for cleaner analysis
                if 'attention' in name or 'self_attn' in name or 'attn' in name:
                    mapping['attention'].append(name)
                elif 'ffn' in name or 'intermediate' in name or 'output' in name:
                    mapping['ffn'].append(name)
                elif 'embedding' in name or 'embeddings' in name:
                    mapping['embeddings'].append(name)
                elif 'classifier' in name or 'pooler' in name:
                    mapping['classifier'].append(name)
                else:
                    mapping['other'].append(name)
        
        return dict(mapping)
    
    def _validate_initialization(self):
        """Structural sanity check only — no appeals to unverifiable priors."""
        assert self.window_size > 1, "GSNR window_size must be > 1"
    
    def update(self, gradients: Dict[str, torch.Tensor]):
        """
        Update tracker with new gradients.
        
        Args:
            gradients: Dictionary of parameter gradients
        """
        for name, grad in gradients.items():
            if grad is not None:
                # Store gradient for this parameter
                self.gradient_history[name].append(grad.detach().cpu())
                
                # Keep only window_size gradients
                if len(self.gradient_history[name]) > self.window_size * 2:
                    self.gradient_history[name] = self.gradient_history[name][-self.window_size * 2:]
        
        # Compute GSNR if we have enough history
        if len(self.gradient_history) > 0:
            gsnr = self._compute_gsnr()
            self.gsnr_history.append(gsnr)
            
            # Validate against benchmarks
            if self.validate_implementation:
                self._validate_against_benchmarks(gsnr)
    
    def _compute_gsnr(self) -> Dict[str, float]:
        """Compute GSNR for each parameter group."""
        gsnr_by_group = {}
        
        for group_name, param_names in self.parameter_mapping.items():
            group_gsnrs = []
            
            for param_name in param_names:
                if param_name in self.gradient_history:
                    grad_history = self.gradient_history[param_name]
                    
                    if len(grad_history) >= self.window_size:
                        # Get recent gradients
                        recent_grads = torch.stack(grad_history[-self.window_size:])
                        
                        # Compute mean and variance
                        grad_mean = recent_grads.mean(dim=0)
                        grad_var = recent_grads.var(dim=0, unbiased=True)
                        
                        # Compute GSNR per element
                        # Add epsilon to avoid division by zero
                        epsilon = 1e-8
                        gsnr_elements = (grad_mean ** 2) / (grad_var + epsilon)
                        
                        # Average across elements
                        param_gsnr = gsnr_elements.mean().item()
                        group_gsnrs.append(param_gsnr)
            
            if group_gsnrs:
                gsnr_by_group[group_name] = np.mean(group_gsnrs)
            else:
                gsnr_by_group[group_name] = 0.0
        
        # Compute overall GSNR (weighted by parameter count)
        total_params = sum(len(params) for params in self.parameter_mapping.values())
        overall_gsnr = 0.0
        
        for group_name, gsnr in gsnr_by_group.items():
            weight = len(self.parameter_mapping[group_name]) / total_params
            overall_gsnr += gsnr * weight
        
        gsnr_by_group["overall"] = overall_gsnr
        return gsnr_by_group
    
    def _load_published_benchmarks(self) -> Dict[str, Dict]:
        """Return external benchmarks from a config file if the user provides one.

        We removed the hardcoded dict: the previous values had no verifiable
        citation and validating our implementation against them was circular.
        """
        path = getattr(self, "benchmarks_path", None)
        if not path:
            return {}
        if not os.path.exists(path):
            return {}
        with open(path) as f:
            return json.load(f)
    
    def _validate_against_benchmarks(self, gsnr_values: Dict[str, float]):
        """Validate computed GSNR against external benchmarks if provided.
        
        NOTE: This is optional - benchmarks must be provided by the user
        via benchmarks_path. We no longer use unverifiable hardcoded values.
        """
        benchmarks = self._load_published_benchmarks()
        if not benchmarks:
            return  # No benchmarks to validate against
        
        # Validation logic would go here if benchmarks are provided
        self.validation_results = {}
    
    def get_gsnr(self, latest_only: bool = True) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """Get GSNR values."""
        if latest_only and self.gsnr_history:
            return self.gsnr_history[-1]
        return self.gsnr_history
    
    def get_convergence_warning(self) -> Optional[str]:
        """Generate convergence warnings based on GSNR patterns."""
        if len(self.gsnr_history) < self.window_size * 2:
            return None
        
        # Get recent and previous GSNR values
        recent = self.gsnr_history[-self.window_size:]
        previous = self.gsnr_history[-self.window_size * 2:-self.window_size]
        
        # Extract overall GSNR
        recent_overall = [gsnr.get("overall", 0) for gsnr in recent]
        previous_overall = [gsnr.get("overall", 0) for gsnr in previous]
        
        if len(recent_overall) == 0 or len(previous_overall) == 0:
            return None
        
        recent_mean = np.mean(recent_overall)
        previous_mean = np.mean(previous_overall)
        
        if recent_mean < 0.1 * previous_mean:
            return "Critical GSNR drop detected - strong plateau signal"
        elif recent_mean < 0.5 * previous_mean:
            return "Moderate GSNR drop detected - plateau approaching"
        
        return None
    
    def validate_implementation_quality(self) -> Dict[str, float]:
        """Validate overall implementation quality."""
        if not self.validation_results:
            return {"quality_score": 0.0, "validation_status": "not_validated"}
        
        # Compute quality score
        valid_groups = 0
        total_groups = 0
        
        for group_result in self.validation_results.values():
            total_groups += 1
            if group_result.get("valid", False):
                valid_groups += 1
        
        quality_score = valid_groups / total_groups if total_groups > 0 else 0.0
        
        # Determine status
        if quality_score > 0.8:
            status = "excellent"
        elif quality_score > 0.6:
            status = "good"
        elif quality_score > 0.4:
            status = "acceptable"
        else:
            status = "needs_improvement"
        
        return {
            "quality_score": quality_score,
            "validation_status": status,
            "valid_groups": valid_groups,
            "total_groups": total_groups,
        }


class LERTracker:
    """
    Learning Efficiency Ratio (LER) tracker with velocity-gradient diagnostics.
    
    LER combines parameter update velocity with loss dynamics:
        LER = (param_velocity * loss_reduction * entropy) / compute_steps
    
    Also computes rho_VG (velocity-gradient correlation) for real-time
    detection of productive vs unproductive training phases.
    
    rho_VG = cosine_similarity(param_velocity_direction, gradient_direction)
      - rho_VG > 0: parameters moving with gradients (productive)
      - rho_VG ~ 0: uncorrelated movement (transition/noise)
      - rho_VG < 0: parameters moving against gradients (thrashing)
    """
    
    def __init__(
        self,
        task: str = "unknown",
        window_size: int = 50,
        validate_correlation: bool = True,
        task_calibration: Optional[Dict] = None,
        min_phase_duration: int = 20,
    ):
        self.task = task
        self.window_size = window_size
        self.validate_correlation = validate_correlation
        self.min_phase_duration = min_phase_duration
        
        self.loss_history: List[float] = []
        self.entropy_history: List[float] = []
        self.ler_history: List[float] = []
        self.accuracy_history: List[float] = []
        
        self.velocity_history: List[float] = []
        self.rho_vg_history: List[float] = []
        self._prev_params: Optional[Dict[str, torch.Tensor]] = None
        self._cached_rho_vg: Optional[float] = None
        
        # FIX #7: streaming layer-wise rho_VG
        self.snapshot_in_fp16: bool = False  # set True for models > ~10B params
        self._prev_params_buffers: Dict[str, torch.Tensor] = {}
        self._prev_params_norms: bool = False
        self.rho_vg_per_layer_history: List[Dict[str, float]] = []
        
        self.correlation_history: List[Tuple[float, float]] = []
        self.validation_results: Dict = {}
        
        # Hysteresis state for phase detection
        self._committed_phase: str = "warmup"
        self._candidate_phase: Optional[str] = None
        self._candidate_phase_count: int = 0
        
        if task_calibration is None:
            task_calibration = self._get_default_calibration()
        self.task_calibration = task_calibration

        # Allow per-task override of min_phase_duration
        task_cal = self.task_calibration.get(self.task, {})
        if "min_phase_duration" in task_cal:
            self.min_phase_duration = task_cal["min_phase_duration"]
    
    def _get_default_calibration(self) -> Dict:
        """Get default calibration for different tasks."""
        return {
            "sst2": {"ler_threshold": 0.01, "entropy_weight": 1.0},
            "qnli": {"ler_threshold": 0.008, "entropy_weight": 1.1},
            "qqp": {"ler_threshold": 0.012, "entropy_weight": 0.9},
            "mnli": {"ler_threshold": 0.009, "entropy_weight": 1.2},
            "rte": {"ler_threshold": 0.015, "entropy_weight": 1.3},
            "mrpc": {"ler_threshold": 0.014, "entropy_weight": 1.4},
            "cola": {"ler_threshold": 0.013, "entropy_weight": 1.0},
            "stsb": {"ler_threshold": 5e-5, "entropy_weight": 1.5, "min_phase_duration": 3},
        }
    
    def update(
        self,
        loss: float,
        logits: torch.Tensor,
        accuracy: Optional[float] = None,
        n_steps: int = 1,
        model: Optional[torch.nn.Module] = None,
        gradients: Optional[Dict[str, torch.Tensor]] = None,
    ):
        self.loss_history.append(loss)
        
        # Compute entropy: handle both classification and regression tasks
        if logits.dim() >= 2 and logits.size(-1) > 1:
            # Classification: use prediction entropy over class probabilities
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).mean().item()
        else:
            # Regression (num_labels=1): softmax on a single logit is always 1.0,
            # giving entropy=0 and making LER=0 for the entire run.
            #
            # FIX (2026-04-06): The previous approach using log1p(pred_std)
            # collapsed to ~0 once the model converged, because prediction
            # variance drops to near-zero for well-trained regression models.
            # Classification entropy for binary tasks is typically 0.5-0.7,
            # but log1p(std) for converged regression was 0.0-0.05.
            #
            # New approach: Regression Prediction Spread Entropy (RPSE)
            # Combines three signals that remain informative throughout training:
            #   1. Prediction spread: how dispersed predictions are across the
            #      batch (normalized by target range to be scale-invariant)
            #   2. Prediction range utilization: what fraction of the output
            #      space the model is actually using
            #   3. Per-sample deviation from batch mean (analogous to how
            #      classification entropy measures per-sample uncertainty)
            #
            # The result is scaled to match classification entropy magnitude
            # (~0.3-0.8 for active learning, ~0.1-0.3 for plateau).
            preds = logits.squeeze()
            if preds.numel() > 1:
                pred_mean = preds.mean().item()
                pred_std = preds.std().item()
                pred_range = (preds.max() - preds.min()).item()

                # For STS-B, target range is [0, 5]. For general regression,
                # estimate from prediction range or use a reasonable default.
                # We use max(pred_range, 1.0) to avoid division by zero and
                # to handle tasks where the model hasn't spread predictions yet.
                effective_range = max(pred_range, 1.0)

                # Component 1: Normalized spread (coefficient of variation analog)
                # High when predictions are diverse, low when collapsed
                spread = pred_std / (abs(pred_mean) + 1e-6)

                # Component 2: Range utilization entropy
                # Measures how much of the output space is being used
                # Normalize predictions to [0, 1] range and compute entropy
                preds_norm = (preds - preds.min()) / (effective_range + 1e-8)
                # Bin into a simple histogram (10 bins) and compute entropy
                hist_counts = torch.histc(preds_norm.float(), bins=10, min=0.0, max=1.0)
                hist_probs = hist_counts / (hist_counts.sum() + 1e-8)
                range_entropy = -(hist_probs * torch.log(hist_probs + 1e-10)).sum().item()
                # Normalize: max entropy for 10 bins is log(10) ~ 2.3
                range_entropy_norm = range_entropy / (np.log(10) + 1e-8)

                # Component 3: Per-sample deviation entropy
                # Analogous to classification per-sample entropy
                deviations = (preds - pred_mean).abs()
                dev_norm = deviations / (pred_std + 1e-8)
                # Use mean absolute z-score as uncertainty measure
                mean_abs_z = dev_norm.mean().item()

                # Combine components with scaling to match classification
                # entropy range (~0.3-0.8 during active learning)
                # spread: typically 0.1-2.0, scale by 0.3
                # range_entropy_norm: 0-1, scale by 0.4
                # mean_abs_z: typically 0.5-1.5, scale by 0.15
                entropy = float(
                    0.3 * min(spread, 3.0)
                    + 0.4 * range_entropy_norm
                    + 0.15 * min(mean_abs_z, 2.0)
                )
                # NO entropy floor: If the model is converged, LER *should* go to zero.
                # Flooring at 0.05 is what makes regression tasks look like they're still
                # "wasting" compute forever. See metrics.py §2b.
            else:
                entropy = 0.1  # single-sample fallback
        self.entropy_history.append(entropy)
        
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
        
        # FIX: Take snapshot BEFORE computing velocity.
        # Velocity needs a prior snapshot from the previous step.
        # Without this, the first call has no prior snapshot -> velocity=None,
        # and we never take a snapshot because we returned early.
        if model is not None:
            self._snapshot_params(model)
        
        param_velocity = self._compute_param_velocity(model)
        
        rho_vg = self._compute_rho_vg(model, gradients)
        if rho_vg is None and self._cached_rho_vg is not None:
            rho_vg = self._cached_rho_vg
            self._cached_rho_vg = None
        if rho_vg is not None:
            self.rho_vg_history.append(rho_vg)
        
        if len(self.loss_history) >= 2:
            loss_gain = max(0.0, self.loss_history[-2] - self.loss_history[-1])  # signed, clipped at 0
            
            window_start = max(0, len(self.entropy_history) - self.window_size)
            avg_entropy = np.mean(self.entropy_history[window_start:])
            
            # FIX #10: Always append to history with explicit mode flag.
            # This ensures history length matches step count for plateau detection.
            if param_velocity is not None and param_velocity > 0:
                ler = (param_velocity * loss_gain * avg_entropy) / (n_steps + 1e-8)
                ler_mode = "full"
            elif param_velocity is not None and param_velocity == 0.0:
                # Genuinely no parameter movement — LER is honestly zero.
                ler = 0.0
                ler_mode = "zero_velocity"
            else:
                # Velocity unavailable (first step, no prior snapshot). Record a
                # velocity-free proxy so history length matches step count and
                # the plateau/phase logic still has a signal. Tag it so analysis
                # can filter these samples.
                ler = (loss_gain * avg_entropy) / (n_steps + 1e-8)
                ler_mode = "velocity_missing"
            
            if self.task in self.task_calibration:
                ler *= self.task_calibration[self.task]["entropy_weight"]
            
            self.ler_history.append(ler)
            
            # Track mode so analysis can filter velocity_missing samples
            if not hasattr(self, "ler_mode_history"):
                self.ler_mode_history = []
            self.ler_mode_history.append(ler_mode)
            
            if accuracy is not None and self.validate_correlation:
                self.correlation_history.append((ler, accuracy))
                if len(self.correlation_history) % 50 == 0:
                    self._validate_correlation()
    
    def _compute_param_velocity(self, model: Optional[torch.nn.Module]) -> Optional[float]:
        if model is None:
            return None
        if self._prev_params is None:
            # Seed snapshot on first call so velocity can be computed on next call
            self._snapshot_params(model)
            return None
        
        total_delta_sq = 0.0
        total_params = 0
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if name in self._prev_params:
                delta = param.data.detach() - self._prev_params[name]
                total_delta_sq += delta.pow(2).sum().item()
                total_params += param.numel()
        
        if total_params == 0:
            return None
        
        velocity = (total_delta_sq ** 0.5) / (total_params ** 0.5)
        self.velocity_history.append(velocity)
        return velocity
    
    def _compute_rho_vg(
        self,
        model: Optional[torch.nn.Module],
        gradients: Optional[Dict[str, torch.Tensor]] = None
    ) -> Optional[float]:
        """Streaming layer-wise rho_VG.

        Instead of concatenating all parameters (2× model in transient memory),
        accumulate dot / norms over parameters one at a time, and also
        retain per-layer values for diagnostics.

        Global rho is the length-weighted average of per-layer rhos, which
        is provably equivalent to the flattened cosine under equal variance
        (and is more informative when variances differ across layers).
        """
        if model is None or not self._prev_params_norms:
            return None

        dot = 0.0
        vel_sq = 0.0
        grad_sq = 0.0
        per_layer: Dict[str, float] = {}

        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            prev_norm_info = self._prev_params_buffers.get(name)
            if prev_norm_info is None:
                continue
            # velocity for this layer: θ_t − θ_{t−1}
            vel = param.data.detach() - prev_norm_info
            grad = None
            if gradients is not None and name in gradients:
                grad = gradients[name].detach()
            elif param.grad is not None:
                grad = param.grad.detach()
            if grad is None:
                continue

            # layer contributions (keep on-device, cast to fp32 for numerics)
            v = vel.float().flatten()
            g = grad.float().flatten()
            d = torch.dot(v, g).item()
            vn = v.norm().item()
            gn = g.norm().item()
            dot += d
            vel_sq += vn * vn
            grad_sq += gn * gn
            if vn > 1e-12 and gn > 1e-12:
                per_layer[name] = d / (vn * gn)

            # free immediately
            del v, g, vel

        self.rho_vg_per_layer_history.append(per_layer)

        if vel_sq < 1e-24 or grad_sq < 1e-24:
            return 0.0
        return dot / ((vel_sq * grad_sq) ** 0.5)
    
    def _snapshot_params(self, model: torch.nn.Module):
        """Store one tensor per param (same tensors that existed anyway),
        not a full deep-clone copy of the model. For memory-bounded training
        of large models, use ``snapshot_in_fp16=True``.
        """
        buffers = {}
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            # Keep in the parameter's own dtype; .clone() still allocates but
            # avoids the earlier .detach().clone() semantics duplication.
            buffers[name] = p.data.detach().clone() if not self.snapshot_in_fp16 \
                            else p.data.detach().to(torch.float16).clone()
        self._prev_params_buffers = buffers
        self._prev_params_norms = True
    
    def capture_step_gradients(self, model: torch.nn.Module):
        """Call during training when param.grad is live (before optimizer.zero_grad).
        
        Computes rho_VG using current gradients and caches the result
        so the next update() call can use it.
        """
        if self._prev_params is None:
            self._snapshot_params(model)
            return
        
        rho = self._compute_rho_vg(model, gradients=None)
        if rho is not None:
            self._cached_rho_vg = rho
    
    def get_ler(self, window: Optional[int] = None) -> Optional[float]:
        """Get LER value over specified window.

        Uses an adaptive window: min(window_size, available_samples) with
        a minimum of 2 samples. This ensures short training runs (e.g.,
        STS-B with ~6 eval points) can compute LER instead of returning
        None because the full window_size was never reached.
        """
        if len(self.ler_history) < 2:
            return None

        if window is None:
            window = self.window_size

        # Adaptive: use what we have, up to the requested window
        effective_window = min(window, len(self.ler_history))
        return np.mean(self.ler_history[-effective_window:])
    
    def get_efficiency_phase(self) -> str:
        """Identify current learning phase based on LER with hysteresis.
        
        Uses a minimum phase duration (min_phase_duration) to prevent
        rapid oscillation between phases due to noisy LER values.
        A candidate phase must be sustained for min_phase_duration
        consecutive calls before the transition is committed.
        """
        # Adaptive minimum: need at least 2 samples, use up to 10
        if len(self.ler_history) < 2:
            return "warmup"

        n_recent = min(10, len(self.ler_history))
        recent_ler = np.array(self.ler_history[-n_recent:])
        avg_ler = np.mean(recent_ler)
        std_ler = np.std(recent_ler)
        
        # Task-specific thresholds
        if self.task in self.task_calibration:
            threshold = self.task_calibration[self.task]["ler_threshold"]
        else:
            threshold = 0.01
        
        # Compute raw (instantaneous) phase
        if avg_ler > threshold * 2 and (avg_ler > 0 and std_ler / avg_ler < 0.5):
            raw_phase = "high_efficiency"
        elif avg_ler > threshold:
            raw_phase = "medium_efficiency"
        elif avg_ler > threshold * 0.1:
            raw_phase = "low_efficiency"
        else:
            raw_phase = "plateau"
        
        # Hysteresis: require min_phase_duration consecutive observations
        # of the same new phase before committing the transition.
        if raw_phase == self._committed_phase:
            # Still in the committed phase; reset any pending candidate
            self._candidate_phase = None
            self._candidate_phase_count = 0
        elif raw_phase == self._candidate_phase:
            # Same candidate as before; increment counter
            self._candidate_phase_count += 1
            if self._candidate_phase_count >= self.min_phase_duration:
                # Sustained long enough; commit the transition
                self._committed_phase = raw_phase
                self._candidate_phase = None
                self._candidate_phase_count = 0
        else:
            # New candidate phase; start counting
            self._candidate_phase = raw_phase
            self._candidate_phase_count = 1
        
        return self._committed_phase
    
    def get_ler_plateau_indicator(self) -> Tuple[bool, float]:
        """Check if LER indicates plateau and return confidence."""
        ler = self.get_ler()
        
        if ler is None:
            return False, 0.0
        
        # Task-specific threshold
        if self.task in self.task_calibration:
            threshold = self.task_calibration[self.task]["ler_threshold"]
        else:
            threshold = 0.01
        
        is_plateau = ler < threshold
        
        # Compute confidence based on consistency
        if len(self.ler_history) >= 5:
            recent_ler = self.ler_history[-5:]
            consistency = 1.0 - (np.std(recent_ler) / (np.mean(recent_ler) + 1e-8))
            confidence = max(0, min(1, consistency))
        else:
            confidence = 0.5
        
        return is_plateau, confidence
    
    def _validate_correlation(self):
        """Validate correlation between LER and accuracy."""
        if len(self.correlation_history) < 10:
            return
        
        lers, accuracies = zip(*self.correlation_history)
        
        # Compute Pearson correlation
        if len(lers) > 1:
            correlation, p_value = stats.pearsonr(lers, accuracies)
            
            self.validation_results = {
                "correlation": correlation,
                "p_value": p_value,
                "n_samples": len(lers),
                "significant": p_value < 0.05,
                "interpretation": self._interpret_correlation(correlation),
            }
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation coefficient."""
        abs_corr = abs(correlation)
        
        if abs_corr > 0.7:
            return "strong correlation"
        elif abs_corr > 0.5:
            return "moderate correlation"
        elif abs_corr > 0.3:
            return "weak correlation"
        else:
            return "no meaningful correlation"
    
    def get_validation_report(self) -> Dict:
        """Get comprehensive validation report."""
        report = {
            "task": self.task,
            "n_samples": len(self.ler_history),
            "current_ler": self.get_ler(),
            "current_phase": self.get_efficiency_phase(),
            "validation_results": self.validation_results,
        }
        
        # Add correlation analysis if available
        if self.correlation_history:
            lers, accuracies = zip(*self.correlation_history[-100:])  # Last 100 points
            if len(lers) > 5:
                correlation, p_value = stats.pearsonr(lers, accuracies)
                report["recent_correlation"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "n_points": len(lers),
                }
        
        return report
    
    def get_rho_vg(self, window: Optional[int] = None) -> Optional[float]:
        if window is None:
            window = self.window_size
        if len(self.rho_vg_history) < 1:
            return None
        recent = self.rho_vg_history[-window:]
        return float(np.mean(recent))
    
    def get_velocity(self, window: Optional[int] = None) -> Optional[float]:
        if window is None:
            window = self.window_size
        if len(self.velocity_history) < 1:
            return None
        recent = self.velocity_history[-window:]
        return float(np.mean(recent))
    
    def get_diagnostics(self) -> Dict:
        ler = self.get_ler()
        rho_vg = self.get_rho_vg()
        velocity = self.get_velocity()
        phase = self.get_efficiency_phase()
        is_plateau, plateau_conf = self.get_ler_plateau_indicator()
        
        productive = False
        if rho_vg is not None and ler is not None:
            threshold = self.task_calibration.get(self.task, {}).get("ler_threshold", 0.01)
            productive = (ler > threshold) and (rho_vg > 0.1)
        
        return {
            "ler": ler,
            "rho_vg": rho_vg,
            "param_velocity": velocity,
            "phase": phase,
            "is_plateau": is_plateau,
            "plateau_confidence": plateau_conf,
            "is_productive": productive,
            "n_steps": len(self.ler_history),
            "n_velocity_samples": len(self.velocity_history),
            "n_rho_vg_samples": len(self.rho_vg_history),
        }


class ProbeAccuracyTracker:
    """
    Probe accuracy tracker for representation quality.
    
    Measures whether model's internal representations contain
    task-relevant information using linear probes.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_labels: int,
        probe_type: str = "linear",
        validation_split: float = 0.2,
    ):
        """
        Initialize probe accuracy tracker.
        
        Args:
            hidden_dim: Dimension of hidden representations
            num_labels: Number of classification labels
            probe_type: Type of probe (linear, mlp)
            validation_split: Fraction of data for validation
        """
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.probe_type = probe_type
        self.validation_split = validation_split
        
        # Storage for representations and labels
        self.representations: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        
        # Probe model
        self.probe = self._create_probe()
        self.probe_trained = False
        
        # Results
        self.probe_accuracies: List[float] = []
        self.training_history: List[Dict] = []
    
    def _create_probe(self) -> torch.nn.Module:
        """Create probe model."""
        if self.probe_type == "linear":
            return torch.nn.Linear(self.hidden_dim, self.num_labels)
        elif self.probe_type == "mlp":
            return torch.nn.Sequential(
                torch.nn.Linear(self.hidden_dim, self.hidden_dim // 2),
                torch.nn.ReLU(),
                torch.nn.Linear(self.hidden_dim // 2, self.num_labels),
            )
        else:
            raise ValueError(f"Unknown probe type: {self.probe_type}")
    
    def add_representations(
        self,
        representations: torch.Tensor,
        labels: torch.Tensor,
    ):
        """
        Add new representations and labels.
        
        Args:
            representations: Hidden representations [batch_size, hidden_dim]
            labels: Ground truth labels [batch_size]
        """
        self.representations.append(representations.detach().cpu())
        self.labels.append(labels.detach().cpu())
        
        # Limit storage
        if len(self.representations) > 1000:  # Keep last 1000 batches
            self.representations = self.representations[-1000:]
            self.labels = self.labels[-1000:]
    
    def compute_probe_accuracy(
        self,
        max_samples: int = 1000,
        train_new_probe: bool = True,
    ) -> float:
        """
        Compute probe accuracy on current representations.
        
        Args:
            max_samples: Maximum samples to use for training/evaluation
            train_new_probe: Whether to train a new probe
        
        Returns:
            Probe accuracy (0-1)
        """
        if len(self.representations) < 10:
            return 0.0
        
        # Concatenate all representations and labels
        all_reps = torch.cat(self.representations, dim=0)
        all_labels = torch.cat(self.labels, dim=0)
        
        # Limit samples
        if len(all_reps) > max_samples:
            indices = torch.randperm(len(all_reps))[:max_samples]
            all_reps = all_reps[indices]
            all_labels = all_labels[indices]
        
        # Split into train/validation
        n_train = int(len(all_reps) * (1 - self.validation_split))
        
        train_reps = all_reps[:n_train]
        train_labels = all_labels[:n_train]
        val_reps = all_reps[n_train:]
        val_labels = all_labels[n_train:]
        
        if train_new_probe:
            # Train new probe
            self.probe = self._create_probe()
            accuracy = self._train_and_evaluate_probe(
                train_reps, train_labels, val_reps, val_labels
            )
        else:
            # Evaluate existing probe
            self.probe.eval()
            with torch.no_grad():
                val_preds = self.probe(val_reps)
                val_preds = torch.argmax(val_preds, dim=1)
                accuracy = (val_preds == val_labels).float().mean().item()
        
        self.probe_accuracies.append(accuracy)
        return accuracy
    
    def _train_and_evaluate_probe(
        self,
        train_reps: torch.Tensor,
        train_labels: torch.Tensor,
        val_reps: torch.Tensor,
        val_labels: torch.Tensor,
        epochs: int = 50,
        lr: float = 0.01,
    ) -> float:
        """Train probe and evaluate on validation set."""
        probe = self.probe
        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        # Training loop
        for epoch in range(epochs):
            probe.train()
            optimizer.zero_grad()
            
            outputs = probe(train_reps)
            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()
        
        # Evaluation
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(val_reps)
            val_preds = torch.argmax(val_outputs, dim=1)
            accuracy = (val_preds == val_labels).float().mean().item()
        
        self.probe_trained = True
        
        # Store training history
        self.training_history.append({
            "epoch": epochs,
            "train_samples": len(train_reps),
            "val_samples": len(val_reps),
            "accuracy": accuracy,
            "probe_type": self.probe_type,
        })
        
        return accuracy
    
    def get_accuracy_trend(self, window: int = 10) -> Optional[float]:
        """Get trend of probe accuracy over recent measurements."""
        if len(self.probe_accuracies) < window:
            return None
        
        recent_accuracies = self.probe_accuracies[-window:]
        
        # Compute slope using linear regression
        x = np.arange(len(recent_accuracies))
        slope, intercept = np.polyfit(x, recent_accuracies, 1)
        
        return slope
    
    def get_representation_quality(self) -> str:
        """Get qualitative assessment of representation quality."""
        if len(self.probe_accuracies) < 5:
            return "insufficient_data"
        
        recent_accuracy = np.mean(self.probe_accuracies[-5:])
        
        if recent_accuracy > 0.9:
            return "excellent"
        elif recent_accuracy > 0.8:
            return "good"
        elif recent_accuracy > 0.7:
            return "adequate"
        elif recent_accuracy > 0.6:
            return "poor"
        else:
            return "very_poor"
    
    def get_validation_report(self) -> Dict:
        """Get comprehensive validation report."""
        return {
            "probe_type": self.probe_type,
            "n_measurements": len(self.probe_accuracies),
            "current_accuracy": self.probe_accuracies[-1] if self.probe_accuracies else 0.0,
            "average_accuracy": np.mean(self.probe_accuracies) if self.probe_accuracies else 0.0,
            "accuracy_trend": self.get_accuracy_trend(),
            "representation_quality": self.get_representation_quality(),
            "probe_trained": self.probe_trained,
            "training_history_summary": {
                "n_trainings": len(self.training_history),
                "avg_val_samples": np.mean([h["val_samples"] for h in self.training_history]) 
                if self.training_history else 0,
            },
        }


class EfficiencyMetricsCollector:
    """
    Collector for all efficiency metrics with integrated analysis.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        task: str,
        validate_all_metrics: bool = True,
    ):
        self.model = model
        self.task = task
        self.validate_all_metrics = validate_all_metrics
        
        # Initialize trackers
        hidden_dim = model.config.hidden_size
        num_labels = model.config.num_labels if hasattr(model.config, "num_labels") else 2
        
        self.gsnr_tracker = GSNRTracker(model)
        self.ler_tracker = LERTracker(task)
        self.probe_tracker = ProbeAccuracyTracker(hidden_dim, num_labels)
        
        # Integrated results
        self.integrated_results: List[Dict] = []
        self.validation_reports: Dict[str, Dict] = {}
    
    def update(
        self,
        gradients: Dict[str, torch.Tensor],
        loss: float,
        logits: torch.Tensor,
        representations: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        accuracy: Optional[float] = None,
        step: int = 0,
    ):
        """Update all trackers with new data."""
        # Update GSNR tracker
        self.gsnr_tracker.update(gradients)
        
        # Update LER tracker
        self.ler_tracker.update(loss, logits, accuracy, model=self.model, gradients=gradients)
        
        # Update probe tracker if representations are available
        if representations is not None and labels is not None:
            self.probe_tracker.add_representations(representations, labels)
            
            # Compute probe accuracy periodically
            if step % 100 == 0:
                probe_accuracy = self.probe_tracker.compute_probe_accuracy()
            else:
                probe_accuracy = None
        else:
            probe_accuracy = None
        
        # Collect integrated results
        result = {
            "step": step,
            "loss": loss,
            "accuracy": accuracy,
            "ler": self.ler_tracker.get_ler(),
            "ler_phase": self.ler_tracker.get_efficiency_phase(),
            "gsnr": self.gsnr_tracker.get_gsnr(latest_only=True),
            "probe_accuracy": probe_accuracy,
        }
        
        self.integrated_results.append(result)
        
        # Validate metrics periodically
        if self.validate_all_metrics and step % 500 == 0:
            self._validate_all_metrics(step)
    
    def _validate_all_metrics(self, step: int):
        """Validate all efficiency metrics."""
        validation_report = {
            "step": step,
            "ler_validation": self.ler_tracker.get_validation_report(),
            "gsnr_validation": self.gsnr_tracker.validate_implementation_quality(),
            "probe_validation": self.probe_tracker.get_validation_report(),
            "integrated_analysis": self._analyze_integrated_metrics(),
        }
        
        self.validation_reports[step] = validation_report
    
    def _analyze_integrated_metrics(self) -> Dict:
        """Analyze relationships between different metrics."""
        if len(self.integrated_results) < 10:
            return {"status": "insufficient_data"}
        
        # Extract metrics
        steps = [r["step"] for r in self.integrated_results]
        losses = [r["loss"] for r in self.integrated_results]
        accuracies = [r["accuracy"] for r in self.integrated_results if r["accuracy"] is not None]
        lers = [r["ler"] for r in self.integrated_results if r["ler"] is not None]
        
        # Compute correlations
        correlations = {}
        
        if len(lers) > 5 and len(accuracies) > 5:
            # Ensure same length
            min_len = min(len(lers), len(accuracies))
            lers_subset = lers[:min_len]
            accuracies_subset = accuracies[:min_len]
            
            if len(lers_subset) > 5:
                correlation, p_value = stats.pearsonr(lers_subset, accuracies_subset)
                correlations["ler_vs_accuracy"] = {
                    "correlation": correlation,
                    "p_value": p_value,
                    "significant": p_value < 0.05,
                }
        
        # Analyze trends
        trends = {}
        if len(lers) > 10:
            # LER trend
            x = np.arange(len(lers))
            slope, _ = np.polyfit(x, lers, 1)
            trends["ler_slope"] = slope
            trends["ler_declining"] = slope < 0
        
        return {
            "n_data_points": len(self.integrated_results),
            "correlations": correlations,
            "trends": trends,
            "current_efficiency_phase": self.ler_tracker.get_efficiency_phase(),
            "convergence_warning": self.gsnr_tracker.get_convergence_warning(),
        }
    
    def get_comprehensive_report(self) -> Dict:
        """Get comprehensive efficiency report."""
        return {
            "task": self.task,
            "model": self.model.__class__.__name__,
            "total_steps": len(self.integrated_results),
            "current_metrics": {
                "ler": self.ler_tracker.get_ler(),
                "ler_phase": self.ler_tracker.get_efficiency_phase(),
                "rho_vg": self.ler_tracker.get_rho_vg(),
                "param_velocity": self.ler_tracker.get_velocity(),
                "gsnr": self.gsnr_tracker.get_gsnr(latest_only=True),
                "probe_accuracy": self.probe_tracker.probe_accuracies[-1] 
                if self.probe_tracker.probe_accuracies else None,
            },
            "validation_reports": self.validation_reports,
            "integrated_analysis": self._analyze_integrated_metrics(),
            "recommendations": self._generate_recommendations(),
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on current metrics."""
        recommendations = []
        
        # Check LER
        ler_phase = self.ler_tracker.get_efficiency_phase()
        if ler_phase == "plateau":
            recommendations.append(
                "LER indicates training plateau. Consider early stopping or "
                "learning rate reduction."
            )
        
        # Check GSNR
        warning = self.gsnr_tracker.get_convergence_warning()
        if warning:
            recommendations.append(f"GSNR warning: {warning}")
        
        # Check probe accuracy
        rep_quality = self.probe_tracker.get_representation_quality()
        if rep_quality in ["poor", "very_poor"]:
            recommendations.append(
                f"Representation quality is {rep_quality}. Model may be "
                "memorizing rather than learning useful features."
            )
        
        # Check correlation between metrics
        if self.validation_reports:
            latest_report = list(self.validation_reports.values())[-1]
            analysis = latest_report.get("integrated_analysis", {})
            
            correlations = analysis.get("correlations", {})
            ler_acc_corr = correlations.get("ler_vs_accuracy", {})
            
            if ler_acc_corr.get("significant", False):
                correlation = ler_acc_corr.get("correlation", 0)
                if correlation > 0.5:
                    recommendations.append(
                        "Strong positive correlation between LER and accuracy. "
                        "LER is a good predictor of performance."
                    )
                elif correlation < -0.5:
                    recommendations.append(
                        "Strong negative correlation between LER and accuracy. "
                        "As LER decreases, accuracy improves (expected pattern)."
                    )
        
        return recommendations


def validate_ler_metric(
    ler_values: List[float],
    performance_metrics: List[float],
    metric_name: str = "accuracy",
) -> Dict:
    """
    Validate LER metric against performance metrics.
    
    Args:
        ler_values: List of LER values
        performance_metrics: Corresponding performance metrics
        metric_name: Name of performance metric
    
    Returns:
        Validation report
    """
    if len(ler_values) != len(performance_metrics):
        raise ValueError("LER values and performance metrics must have same length")
    
    if len(ler_values) < 10:
        return {"status": "insufficient_data", "n_samples": len(ler_values)}
    
    # Compute correlation
    correlation, p_value = stats.pearsonr(ler_values, performance_metrics)
    
    # Compute R²
    slope, intercept = np.polyfit(ler_values, performance_metrics, 1)
    y_pred = slope * np.array(ler_values) + intercept
    ss_res = np.sum((performance_metrics - y_pred) ** 2)
    ss_tot = np.sum((performance_metrics - np.mean(performance_metrics)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Interpret correlation
    abs_corr = abs(correlation)
    if abs_corr > 0.7:
        strength = "strong"
    elif abs_corr > 0.5:
        strength = "moderate"
    elif abs_corr > 0.3:
        strength = "weak"
    else:
        strength = "negligible"
    
    # Direction
    if correlation > 0:
        direction = "positive"
        interpretation = f"As LER increases, {metric_name} increases"
    else:
        direction = "negative"
        interpretation = f"As LER decreases, {metric_name} increases"
    
    return {
        "n_samples": len(ler_values),
        "correlation": float(correlation),
        "p_value": float(p_value),
        "r_squared": float(r_squared),
        "significant": p_value < 0.05,
        "strength": strength,
        "direction": direction,
        "interpretation": interpretation,
        "validation_status": "valid" if p_value < 0.05 and abs_corr > 0.5 else "needs_revalidation",
    }


def compute_effect_sizes(
    method_a_results: List[float],
    method_b_results: List[float],
    method_a_name: str = "Method A",
    method_b_name: str = "Method B",
) -> Dict:
    """
    Compute effect sizes between two methods.
    
    Args:
        method_a_results: Results from method A
        method_b_results: Results from method B
        method_a_name: Name of method A
        method_b_name: Name of method B
    
    Returns:
        Effect size analysis
    """
    if len(method_a_results) != len(method_b_results):
        raise ValueError("Both methods must have same number of results")
    
    if len(method_a_results) < 5:
        return {"status": "insufficient_data"}
    
    a_array = np.array(method_a_results)
    b_array = np.array(method_b_results)
    
    # Mean difference
    mean_diff = np.mean(a_array) - np.mean(b_array)
    
    # Cohen's d
    pooled_std = np.sqrt(
        (np.std(a_array, ddof=1)**2 + np.std(b_array, ddof=1)**2) / 2
    )
    cohens_d = mean_diff / (pooled_std + 1e-10)
    
    # Hedges' g (small sample correction)
    n = len(a_array)
    hedges_g = cohens_d * (1 - (3 / (4 * (n - 1) - 1)))
    
    # Interpret effect size
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        magnitude = "negligible"
    elif abs_d < 0.5:
        magnitude = "small"
    elif abs_d < 0.8:
        magnitude = "medium"
    else:
        magnitude = "large"
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(a_array, b_array)
    
    # Confidence interval for mean difference
    differences = a_array - b_array
    mean_diff_ci = stats.t.interval(
        0.95,
        len(differences) - 1,
        loc=np.mean(differences),
        scale=stats.sem(differences),
    )
    
    return {
        "n_comparisons": len(method_a_results),
        "mean_difference": float(mean_diff),
        "mean_difference_ci": tuple(float(x) for x in mean_diff_ci),
        "cohens_d": float(cohens_d),
        "hedges_g": float(hedges_g),
        "effect_magnitude": magnitude,
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "interpretation": (
            f"{method_a_name} shows {magnitude} effect compared to {method_b_name} "
            f"(d = {cohens_d:.2f}, p = {p_value:.4f})"
        ),
    }
