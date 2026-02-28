#!/bin/bash
#SBATCH --job-name=lerna_full
#SBATCH --output=lerna_%j.out
#SBATCH --error=lerna_%j.err
#SBATCH --time=24:00:00          # Request 24 hours
#SBATCH --gres=gpu:1              # 1 GPU
#SBATCH --cpus-per-task=8         # 8 CPU cores
#SBATCH --partition=gpu            # GPU partition
##SBATCH --mem=64G                 # COMMENTED OUT - use default per-GPU memory

echo "=== LERNA Full Experiment ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Started: $(date)"
echo "================================"

# Load modules (adjust to your cluster)
module load cuda/12.1
module load python/3.10

# Activate conda
source ~/.bashrc
conda activate torch_cuda

# Install missing packages (just in case)
pip install --quiet plotly scipy psutil

# Run experiment with checkpointing
python scripts/run_full_experiment.py \
  --output-dir /ssd_xs/home/scvi383/scvi383/experiments/full_baseline \
  --num-seeds 10 \
  --cooldown 2 \
  --wandb \
  --max-retries 3

echo "================================"
echo "Finished: $(date)"
echo "================================"
