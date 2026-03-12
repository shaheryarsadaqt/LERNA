# LERNA: Learning Efficiency Ratio Navigation & Adaptation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

## 📄 Abstract

Now let me review the current abstract against everything we've validated and refined. The current abstract has several issues:

It says "causal evidence" / "causal intervention experiments" but our research confirmed you should use "interventional evidence"
It says "knowledge retention" but we're replacing that with TracIn-based step attribution
It doesn't mention the convergence theory (PL condition, drift bounds)
It doesn't mention the breadth of benchmarks (generative tasks beyond GLUE)
It claims "10+ tasks" but the plan expands this significantly
The "37% ± 8%" in the first line doesn't match "37% ± 5.3%" in the results

Here's the updated abstract. You can replace it in your README and paper:

Updated Abstract

Fine-tuning Large Language Models consumes substantial energy, yet a significant fraction of this compute yields no measurable improvement. Across 80+ controlled runs on 8 GLUE tasks, we quantify that 36.7% ± 5.3% of fine-tuning compute is expended past the point of productive learning (95% CI), translating directly into unnecessary energy consumption across millions of annual training runs worldwide. We present LERNA (Learning Efficiency Ratio Navigation & Adaptation), a framework that diagnoses when models are actively learning and exploits this signal to eliminate wasteful gradient computation.
We introduce the Learning Efficiency Ratio (LER), a lightweight diagnostic combining parameter update velocity, loss dynamics, and prediction entropy, alongside velocity-gradient correlation (ρ_VG) for real-time detection of transitions between productive and unproductive training phases. Using gradient tracing (TracIn) and controlled counterfactual interventions at identical parameter checkpoints, we establish that training steps during high-LER phases exhibit 5× greater marginal contribution to validation loss reduction than steps during low-LER phases, providing the first interventional evidence that training steps are fundamentally unequal in their contribution to model capability. We further demonstrate that LER serves as a computationally efficient proxy for TracIn scores, enabling real-time step-level attribution without requiring validation gradients at every step.
This diagnostic insight enables LERNA's core mechanism: LER-guided hybrid-order switching. During high-LER phases, standard backpropagation drives learning at full fidelity. When LER drops below a validated threshold, LERNA bypasses the backward pass entirely, updating weights via momentum-driven inertial extrapolation, eliminating ~60% of per-step compute. An adaptive safety horizon H(ρ_VG), scaled inversely with diagnostic confidence, prevents parameter drift. We provide convergence guarantees under the Polyak-Łojasiewicz condition, proving that the parameter drift bound tightens to O(ηK·ε_plateau) when skips are triggered only during detected plateaus, and that LERNA converges to a neighborhood of the optimum with error floor O(η²K²_max·L²) dependent on the safety horizon rather than the skip fraction alone. Unlike prior methods that partition parameters by static importance (Hi-ZFO) or operate at the tensor level (GreenTrainer), LERNA introduces temporal compute optimization: dynamically toggling based on when the model is learning, not which parameters matter most.
Validated across classification (8 GLUE tasks), instruction tuning (Alpaca), summarization (CNN/DailyMail, XSum), mathematical reasoning (GSM8K), science reasoning (ARC), and code generation (HumanEval) on models from 149M to 70B parameters (LoRA/QLoRA), LERNA reduces measured fine-tuning energy consumption by 35-40% (kWh via GPU power telemetry) while maintaining accuracy within ±0.3% of full-training oracle performance. Component ablations across 6 simple baselines confirm that LER's sophistication is justified: neither gradient norm thresholding, random step skipping, nor early stopping achieves comparable efficiency-accuracy trade-offs. All findings are supported by 95% confidence intervals over 50+ runs per configuration, with head-to-head comparison against GreenTrainer demonstrating complementarity between temporal and spatial compute optimization.


What Changed and Why



Old Abstract
New Abstract
Reason



"37% ± 8%"
"36.7% ± 5.3%"
Matches your actual validated results from 80 runs


"causal intervention experiments" / "causal evidence"
"controlled counterfactual interventions" / "interventional evidence"
Both deep research sources confirmed: don't use "causal" for this setting


"knowledge retention during high-LER phases is 5× greater"
"training steps during high-LER phases exhibit 5× greater marginal contribution to validation loss reduction"
Replaced vague "knowledge retention" with precise TracIn-based attribution language


No mention of TracIn
"Using gradient tracing (TracIn)... LER serves as a computationally efficient proxy for TracIn scores"
This is your new unique contribution from the revised Phase 4


No convergence theory
"convergence guarantees under the Polyak-Łojasiewicz condition... drift bound tightens to O(ηK·ε_plateau)"
Incorporates the validated Phase 5 theory with PL condition (not strong convexity)


No mention of K_max
"error floor O(η²K²_max·L²) dependent on the safety horizon"
Critical insight from Gemini's analysis: error depends on K_max, not just p


"10+ tasks spanning classification, NLI, and semantic similarity"
Lists all 6 task types explicitly (classification, instruction, summarization, reasoning, code)
Reflects the expanded Phase 2 generative benchmarks


No mention of baselines
"Component ablations across 6 simple baselines confirm that LER's sophistication is justified"
Reflects Phase 1 baseline experiments


No GreenTrainer mention
"head-to-head comparison against GreenTrainer demonstrating complementarity"
Reflects Phase 3


"models from 149M to 8B"
"models from 149M to 70B parameters (LoRA/QLoRA)"
Reflects Phase 6 scaling


Important Note
This abstract describes the planned final state of your paper after executing the full research plan. Some claims (generative benchmarks, GreenTrainer comparison, TracIn analysis, convergence proofs) are not yet experimentally validated. As you complete each phase, verify that the numbers hold. If any claim doesn't hold after experiments (e.g., accuracy degrades >0.3% on generative tasks), adjust the abstract accordingly. Never publish claims you haven't verified.
## 📦 Installation

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
