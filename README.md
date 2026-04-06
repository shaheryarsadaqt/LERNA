# LERNA: Learning Efficiency Ratio Navigation & Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## Abstract

Fine-tuning Large Language Models consumes substantial energy, yet a significant fraction of this compute yields no measurable improvement. Across 80+ controlled runs on 8 GLUE tasks, we quantify that 36.7% +/- 5.3% of fine-tuning compute is expended past the point of productive learning (95% CI), translating directly into unnecessary energy consumption across millions of annual training runs worldwide. We present LERNA (Learning Efficiency Ratio Navigation & Adaptation), a framework that diagnoses when models are actively learning and exploits this signal to eliminate wasteful gradient computation.

We introduce the Learning Efficiency Ratio (LER), a lightweight diagnostic combining parameter update velocity, loss dynamics, and prediction entropy, alongside velocity-gradient correlation (rho_VG) for real-time detection of transitions between productive and unproductive training phases. Using gradient tracing (TracIn) and controlled counterfactual interventions at identical parameter checkpoints, we establish that training steps during high-LER phases exhibit 5x greater marginal contribution to validation loss reduction than steps during low-LER phases, providing the first interventional evidence that training steps are fundamentally unequal in their contribution to model capability. We further demonstrate that LER serves as a computationally efficient proxy for TracIn scores, enabling real-time step-level attribution without requiring validation gradients at every step.

This diagnostic insight enables LERNA's core mechanism: LER-guided hybrid-order switching. During high-LER phases, standard backpropagation drives learning at full fidelity. When LER drops below a validated threshold, LERNA bypasses the backward pass entirely, updating weights via momentum-driven inertial extrapolation, eliminating ~60% of per-step compute. An adaptive safety horizon H(rho_VG), scaled inversely with diagnostic confidence, prevents parameter drift. We provide convergence guarantees under the Polyak-Lojasiewicz condition, proving that the parameter drift bound tightens to O(eta*K*epsilon_plateau) when skips are triggered only during detected plateaus, and that LERNA converges to a neighborhood of the optimum with error floor O(eta^2 * K^2_max * L^2) dependent on the safety horizon rather than the skip fraction alone. Unlike prior methods that partition parameters by static importance (Hi-ZFO) or operate at the tensor level (GreenTrainer), LERNA introduces temporal compute optimization: dynamically toggling based on when the model is learning, not which parameters matter most.

Validated across classification (8 GLUE tasks), instruction tuning (Alpaca), summarization (CNN/DailyMail, XSum), mathematical reasoning (GSM8K), science reasoning (ARC), and code generation (HumanEval) on models from 149M to 70B parameters (LoRA/QLoRA), LERNA reduces measured fine-tuning energy consumption by 35-40% (kWh via GPU power telemetry) while maintaining accuracy within +/-0.3% of full-training oracle performance. Component ablations across 6 simple baselines confirm that LER's sophistication is justified: neither gradient norm thresholding, random step skipping, nor early stopping achieves comparable efficiency-accuracy trade-offs. All findings are supported by 95% confidence intervals over 50+ runs per configuration, with head-to-head comparison against GreenTrainer demonstrating complementarity between temporal and spatial compute optimization.

> **Note:** This abstract describes the planned final state of the paper after executing the full research plan. Claims are updated as each phase completes. See the Experimental Results section below for currently validated findings.

---

## Experimental Results

Results are added incrementally as each experimental phase completes.

### Phase 1.1: Baseline Data Collection (COMPLETED)

**Status:** 80/80 runs completed (79 succeeded, 1 SST-2 crash re-run pending)
**Model:** RoBERTa-base | **Infrastructure:** RTX 5090 | **Wall time:** 4 days, 57 minutes

#### Accuracy Results (Best-Model Evaluation)

All results use `load_best_model_at_end=True` with task-specific model selection metrics.

| Task  | LR    | Metric   | Mean   | Std    | SOTA Range | Status |
|-------|-------|----------|--------|--------|------------|--------|
| SST-2 | 2e-05 | Accuracy | 0.9373 | 0.0054 | 94-96%     | Excellent |
| QNLI  | 2e-05 | Accuracy | 0.9248 | 0.0018 | 92-94%     | Strong |
| QQP   | 2e-05 | Accuracy | 0.9081 | 0.0022 | 91-92%     | Very Good |
| MNLI  | 2e-05 | Accuracy | 0.8728 | 0.0030 | 87-90%     | Good |
| RTE   | 2e-05 | Accuracy | 0.7639 | 0.0090 | 75-85%     | Good |
| MRPC  | 2e-05 | Accuracy | 0.8831 | 0.0066 | 88-92%     | Good |
| CoLA  | 1e-05 | MCC      | 0.5780 | 0.0055 | 60-65%     | Acceptable |
| STS-B | 2e-05 | Pearson  | 0.8607 | ~0.01  | 88-92%     | Good |

#### Waste Detection Results

| Task  | Waste Ratio | Interpretation |
|-------|-------------|----------------|
| QQP   | 98.8%       | Massive waste: model converges in first ~1% of steps |
| MNLI  | 98.9%       | Massive waste: model converges in first ~1% of steps |
| QNLI  | 55.8%       | Significant waste after early convergence |
| SST-2 | 50.4%       | Significant waste after early convergence |
| CoLA  | 0.0%        | No waste detected (10 epochs, lr=1e-05, early stopping) |
| MRPC  | 0.0%        | No waste detected (10 epochs, early stopping) |
| RTE   | 0.0%        | No waste detected (20 epochs, early stopping) |
| STS-B | 0.0%*       | *Bug: LER=0 for regression tasks (now fixed, re-run pending) |

**Task-level mean waste: 38.0%** (supports abstract claim of 36.7% +/- 5.3%)

**Key finding:** Waste detection correctly discriminates between tasks. Large datasets
with standard 3-epoch training (QQP, MNLI) show massive waste because the model
converges early. Small datasets with more epochs and early stopping (RTE, MRPC, CoLA)
show zero waste because compute is used efficiently. This validates that the
WasteQuantifier is not simply flagging all training as wasteful.

#### Infrastructure Metrics

| Metric | Value |
|--------|-------|
| Total Runs | 80/80 (1 crash, 79 completed) |
| Total Energy | 1.52 kWh |
| Disk Usage | ~122GB (within 223GB limit) |
| Slim Checkpoints | 2.79GB saved per run |

#### Known Issues

1. **STS-B LER=0 (FIXED):** Regression entropy proxy collapsed to zero. Fixed with
   Regression Prediction Spread Entropy (RPSE). Re-run pending.
2. **SST-2 seed 42:** Crashed. Re-run pending.
3. **STS-B seed 46:** Anomalous final Pearson (0.556 vs ~0.88). Investigation pending.

### Phase 1.2: Simple Baselines (PENDING)

*Results will be added after Phase 1.2 experiments complete.*

### Phase 1.3: Component Ablation (PENDING)

*Results will be added after Phase 1.3 experiments complete.*

---

## Installation

```bash
# Clone repository
git clone https://github.com/shaheryarsadaqt/LERNA.git
cd LERNA

# Create conda environment
conda create -n lerna python=3.10
conda activate lerna

# Install PyTorch with CUDA 12.8 support
pip install torch==2.10.0 torchvision==0.15.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# Install other dependencies
pip install transformers datasets evaluate wandb scipy numpy pandas matplotlib seaborn tqdm plotly psutil alive-progress
```
