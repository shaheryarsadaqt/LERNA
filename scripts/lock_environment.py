#!/usr/bin/env python3
"""
Environment fingerprinting for reproducible experiment execution.

Records GPU, PyTorch, Transformers, and CUDA versions.
On resume, stops execution if the fingerprint has changed.
"""

import json
import os
import sys
from pathlib import Path

import torch
from transformers import __version__ as transformers_version


def capture_fingerprint() -> dict:
    fingerprint = {
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
        "cuda_version": torch.version.cuda,
    }
    return fingerprint


def save_fingerprint(path: Path, fingerprint: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(fingerprint, indent=2))


def load_fingerprint(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def check_environment(output_dir: str) -> dict:
    fingerprint_path = Path(output_dir) / "environment_fingerprint.json"
    current = capture_fingerprint()
    saved = load_fingerprint(fingerprint_path)

    if saved is None:
        save_fingerprint(fingerprint_path, current)
        print(f"  [ENV] Fingerprint saved: {fingerprint_path}")
        return current

    if saved != current:
        changed = [k for k in current if saved.get(k) != current[k]]
        raise RuntimeError(
            f"Environment fingerprint mismatch. Changed fields: {changed}. "
            f"Expected: {saved}. Got: {current}. Aborting to preserve reproducibility."
        )

    print(f"  [ENV] Fingerprint matches saved environment.")
    return current


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()
    check_environment(args.output_dir)
