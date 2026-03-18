#!/bin/bash
#SBATCH --job-name=lerna_gen
#SBATCH --output=lerna_gen_%j.out
#SBATCH --error=lerna_gen_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu

# ============================================================
# LERNA Phase 2: Generative Benchmark SLURM Launcher
# ============================================================
#
# Usage:
#   # Single task:
#   sbatch scripts/run_slurm_generative.sh alpaca llama2_7b_lora
#
#   # Submit all tasks:
#   for task in alpaca cnndm xsum gsm8k arc code; do
#     sbatch scripts/run_slurm_generative.sh $task llama2_7b_lora
#   done
#
# Arguments:
#   $1 = task name (alpaca, cnndm, xsum, gsm8k, arc, code)
#   $2 = model config key from configs/lora_configs.yaml
# ============================================================

TASK=${1:-alpaca}
MODEL_CONFIG=${2:-llama2_7b_lora}
NUM_SEEDS=${3:-5}
OUTPUT_DIR="/ssd_xs/home/scvi383/scvi383/experiments/phase2_generative"

echo "=== LERNA Generative Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "Task: $TASK"
echo "Model: $MODEL_CONFIG"
echo "Seeds: $NUM_SEEDS"
echo "Output: $OUTPUT_DIR"
echo "Started: $(date)"
echo "================================"

# Load modules
module load cuda/12.1
module load python/3.10

# Activate conda
source ~/.bashrc
conda activate torch_cuda

# Ensure Phase 2 deps are installed
pip install --quiet peft bitsandbytes rouge-score accelerate

# Set HuggingFace cache (adjust to your cluster)
export HF_HOME=/ssd_xs/home/scvi383/.cache/huggingface
export WANDB_PROJECT=lerna-2026
export WANDB_RUN_GROUP="phase2-${TASK}-${MODEL_CONFIG}"

# Run experiment
# (This script will be created in Phase 1/2 when we build the generative runner)
# For now, this is the SLURM template.
echo "[TODO] python scripts/run_generative_experiment.py \\"
echo "  --task $TASK \\"
echo "  --model-config $MODEL_CONFIG \\"
echo "  --num-seeds $NUM_SEEDS \\"
echo "  --output-dir $OUTPUT_DIR \\"
echo "  --wandb"

echo "================================"
echo "Finished: $(date)"
echo "================================"
