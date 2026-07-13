#!/usr/bin/env bash
set -euo pipefail

# Offline bundle preparation script for Ettin-150m Phase 1.2 experiments
# Run this on an internet-connected Linux machine

echo "=== Step 1: Create isolated venv ==="
python3 -m venv ettin_download_env
source ettin_download_env/bin/activate

echo "=== Step 2: Install exact package versions ==="
pip install \
  torch \
  transformers==4.48.3 \
  datasets \
  evaluate \
  huggingface-hub \
  safetensors \
  scipy \
  scikit-learn

echo "=== Step 3: Set isolated cache ==="
export HF_HOME="$PWD/ettin_hf_bundle"
mkdir -p "$HF_HOME"

echo "=== Step 4: Download Ettin model ==="
python3 - <<'PY'
from huggingface_hub import snapshot_download

MODEL = "jhu-clsp/ettin-encoder-150m"
REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"

print("Downloading pinned Ettin checkpoint...")
path = snapshot_download(
    repo_id=MODEL,
    revision=REVISION,
)
print("Model snapshot:", path)
PY

echo "=== Step 5: Download GLUE datasets and metrics ==="
python3 - <<'PY'
from datasets import load_dataset
import evaluate

TASKS = ["sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb"]

for task in TASKS:
    print("Downloading GLUE dataset:", task)
    load_dataset("glue", task)

    print("Downloading GLUE metric:", task)
    evaluate.load("glue", task)

print("Offline bundle preparation complete")
PY

echo "=== Step 6: Verify weights exist ==="
find "$HF_HOME" -type f \
  \( -name 'model.safetensors' -o -name 'pytorch_model.bin' \) \
  -ls

echo "=== Step 7: Package bundle ==="
tar -czf ettin_hf_bundle.tar.gz ettin_hf_bundle
ls -lh ettin_hf_bundle.tar.gz

echo ""
echo "=== Bundle ready ==="
echo "Transfer ettin_hf_bundle.tar.gz to the DGX and run:"
echo "  tar -xzf ettin_hf_bundle.tar.gz"
echo "  export HF_HOME=\$HOME/ettin_hf_bundle"
echo "  export HF_HUB_OFFLINE=1"
echo "  export TRANSFORMERS_OFFLINE=1"
echo "  export HF_DATASETS_OFFLINE=1"
