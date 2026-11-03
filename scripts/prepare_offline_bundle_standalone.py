#!/usr/bin/env python3
import gc
import os
import subprocess
import sys
import tarfile
from pathlib import Path

MODEL = "jhu-clsp/ettin-encoder-150m"
REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
TASKS = ("sst2", "qnli", "qqp", "mnli", "rte", "mrpc", "cola", "stsb")

bundle = Path("ettin_hf_bundle").resolve()
archive = Path("ettin_hf_bundle.tar.gz").resolve()

if bundle.exists() and any(bundle.iterdir()):
    raise SystemExit(f"Refusing to overwrite non-empty directory: {bundle}")
if archive.exists():
    raise SystemExit(f"Refusing to overwrite existing archive: {archive}")

bundle.mkdir(parents=True, exist_ok=True)

# Set before importing Hugging Face packages.
os.environ["HF_HOME"] = str(bundle)
os.environ["HF_HUB_DISABLE_XET"] = "1"

import torch
import evaluate
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoModelForSequenceClassification, AutoTokenizer

print(f"Downloading {MODEL}@{REVISION}")
snapshot = Path(snapshot_download(MODEL, revision=REVISION))

weights = [
    path
    for pattern in ("*.safetensors", "pytorch_model*.bin")
    for path in snapshot.glob(pattern)
    if path.is_file()
]
weight_bytes = sum(path.stat().st_size for path in weights)

if not weights or weight_bytes < 100_000_000:
    raise RuntimeError(
        f"Model weights missing: files={weights}, bytes={weight_bytes}"
    )

print(f"Weights verified: {weight_bytes / 1024**2:.1f} MiB")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    revision=REVISION,
    local_files_only=True,
)

kwargs = {
    "revision": REVISION,
    "local_files_only": True,
    "num_labels": 2,
    "attn_implementation": "sdpa",
    "reference_compile": False,
}

try:
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, **kwargs)
except TypeError:
    kwargs.pop("reference_compile", None)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, **kwargs)

parameters = sum(parameter.numel() for parameter in model.parameters())
if parameters < 100_000_000:
    raise RuntimeError(f"Unexpected parameter count: {parameters}")

print(f"Model loaded: {parameters:,} parameters")
print(f"Tokenizer vocabulary: {len(tokenizer):,}")

del model, tokenizer
gc.collect()

for task in TASKS:
    print(f"Caching GLUE dataset and metric: {task}")
    dataset = load_dataset("glue", task)
    if "train" not in dataset or len(dataset["train"]) == 0:
        raise RuntimeError(f"Missing GLUE train split: {task}")
    evaluate.load("glue", task)

offline_test = f"""
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset
import evaluate

model = AutoModelForSequenceClassification.from_pretrained(
    {MODEL!r},
    revision={REVISION!r},
    num_labels=2,
    local_files_only=True,
    attn_implementation="sdpa",
    reference_compile=False,
)
assert sum(p.numel() for p in model.parameters()) >= 100_000_000

for task in {TASKS!r}:
    dataset = load_dataset("glue", task)
    assert len(dataset["train"]) > 0
    evaluate.load("glue", task)

print("OFFLINE_VERIFICATION_OK")
"""

offline_env = os.environ.copy()
offline_env.update({
    "HF_HOME": str(bundle),
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
})

subprocess.run(
    [sys.executable, "-c", offline_test],
    env=offline_env,
    check=True,
)

with tarfile.open(archive, "w:gz", dereference=False) as tar:
    tar.add(bundle, arcname=bundle.name)

if archive.stat().st_size < 100_000_000:
    raise RuntimeError(f"Archive is unexpectedly small: {archive.stat().st_size}")

print(f"BUNDLE_READY={archive}")
print(f"ARCHIVE_BYTES={archive.stat().st_size}")
