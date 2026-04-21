#!/usr/bin/env bash
#SBATCH --job-name=lerna
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

: "${LERNA_ROOT:?set LERNA_ROOT to the repo path}"
: "${LERNA_OUT:?set LERNA_OUT to the results dir}"
: "${LERNA_CONDA_ENV:=lerna}"

cd "$LERNA_ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$LERNA_CONDA_ENV"

python -m lerna.experiments.glue.cli \
    --task "${LERNA_TASK:?}" --seed "${LERNA_SEED:?}" \
    --out "$LERNA_OUT/${LERNA_TASK}_${LERNA_SEED}"
