#!/usr/bin/env bash
set -euo pipefail

# Verify offline bundle on the DGX
# Run after extracting ettin_hf_bundle.tar.gz

echo "=== Step 1: Set offline environment ==="
export HF_HOME="$HOME/ettin_hf_bundle"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_EVALUATE_OFFLINE=1
export WANDB_DISABLED=true
export WANDB_MODE=disabled

echo "HF_HOME=$HF_HOME"
echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
echo "HF_DATASETS_OFFLINE=$HF_DATASETS_OFFLINE"

echo ""
echo "=== Step 2: Verify Ettin model files ==="
find "$HF_HOME" -type f \
  \( -name 'model.safetensors' -o -name 'pytorch_model.bin' \) \
  -ls

echo ""
echo "=== Step 3: Verify Ettin offline load ==="
python3 - <<'PY'
import os
from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import scan_cache_dir

model = "jhu-clsp/ettin-encoder-150m"
revision = "45d08642849e5c5701b162671ac811b7654bfd9f"

config = AutoConfig.from_pretrained(
    model,
    revision=revision,
    local_files_only=True,
)
tokenizer = AutoTokenizer.from_pretrained(
    model,
    revision=revision,
    local_files_only=True,
)

print("Model type:", config.model_type)
print("Tokenizer:", tokenizer.__class__.__name__)
print("Vocabulary:", len(tokenizer))
print(scan_cache_dir())
PY

echo ""
echo "=== Step 4: Verify full weight load ==="
python3 - <<'PY'
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(
    "jhu-clsp/ettin-encoder-150m",
    revision="45d08642849e5c5701b162671ac811b7654bfd9f",
    num_labels=2,
    local_files_only=True,
    attn_implementation="sdpa",
    reference_compile=False,
)

print("Loaded:", model.__class__.__name__)
print("Parameters:", sum(p.numel() for p in model.parameters()))
PY

echo ""
echo "=== Step 5: Verify GLUE assets offline ==="
python3 - <<'PY'
from datasets import load_dataset
import evaluate

tasks = ["sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb"]

for task in tasks:
    dataset = load_dataset("glue", task)
    metric = evaluate.load("glue", task)
    print(
        task,
        "train=", len(dataset["train"]),
        "metric=", metric.__class__.__name__,
    )
PY

echo ""
echo "=== All offline verifications passed ==="
