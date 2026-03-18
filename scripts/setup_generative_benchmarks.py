#!/usr/bin/env python3
"""
LERNA Phase 0.4: Download and verify all generative benchmark datasets.

This script downloads all datasets needed for Phase 2 (generative benchmarks)
and verifies they load correctly. Run this ONCE before starting experiments.

Usage:
    python scripts/setup_generative_benchmarks.py
    python scripts/setup_generative_benchmarks.py --cache-dir /data/hf_cache
    python scripts/setup_generative_benchmarks.py --verify-only
"""

import argparse
import os
import sys
import time
from pathlib import Path


def check_imports():
    """Check that all required packages are installed."""
    missing = []
    packages = {
        "datasets": "datasets",
        "transformers": "transformers",
        "peft": "peft",
        "bitsandbytes": "bitsandbytes",
        "rouge_score": "rouge-score",
        "torch": "torch",
        "evaluate": "evaluate",
    }
    for module, pip_name in packages.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(pip_name)

    if missing:
        print("\n" + "=" * 60)
        print("MISSING PACKAGES - Install before running:")
        print("=" * 60)
        print(f"  pip install {' '.join(missing)}")
        print("  # or:")
        print("  pip install -r requirements/requirements_phase2.txt")
        print("=" * 60)
        return False
    return True


def download_dataset(name, config, split=None, cache_dir=None):
    """Download a single dataset and return info."""
    from datasets import load_dataset

    print(f"\n  Downloading: {name}" + (f" ({config})" if config else ""))
    start = time.time()
    try:
        kwargs = {}
        if cache_dir:
            kwargs["cache_dir"] = cache_dir
        if config:
            ds = load_dataset(name, config, **kwargs)
        else:
            ds = load_dataset(name, **kwargs)

        elapsed = time.time() - start

        # Print split info
        for split_name, split_ds in ds.items():
            print(f"    {split_name}: {len(split_ds):,} examples")
        print(f"    Downloaded in {elapsed:.1f}s")

        return {"status": "ok", "name": name, "config": config, "splits": {k: len(v) for k, v in ds.items()}}
    except Exception as e:
        print(f"    FAILED: {e}")
        return {"status": "error", "name": name, "config": config, "error": str(e)}


def verify_tokenizer(model_name):
    """Verify a model tokenizer can be loaded (does NOT download full model weights)."""
    from transformers import AutoTokenizer

    print(f"\n  Verifying tokenizer: {model_name}")
    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        print(f"    Vocab size: {tok.vocab_size:,}")
        print(f"    Model max length: {tok.model_max_length:,}")
        test = tok("Hello, LERNA!", return_tensors="pt")
        print(f"    Test encode OK: {test['input_ids'].shape}")
        return True
    except Exception as e:
        print(f"    FAILED: {e}")
        print(f"    Make sure you have access: huggingface-cli login")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download & verify LERNA generative benchmarks")
    parser.add_argument("--cache-dir", default=None, help="HuggingFace cache directory")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, don't download")
    parser.add_argument("--skip-tokenizers", action="store_true", help="Skip tokenizer verification")
    args = parser.parse_args()

    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["HF_DATASETS_CACHE"] = os.path.join(args.cache_dir, "datasets")

    print("=" * 60)
    print("  LERNA Phase 0.4: Generative Benchmark Setup")
    print("=" * 60)

    # Step 1: Check imports
    print("\n[1/4] Checking package imports...")
    if not check_imports():
        sys.exit(1)
    print("  All packages OK")

    # Step 2: Download datasets
    print("\n[2/4] Downloading datasets...")
    datasets_to_download = [
        # Instruction tuning
        ("tatsu-lab/alpaca", None),                    # Alpaca-52k
        ("OpenAssistant/oasst1", None),                # OpenAssistant
        # Summarization
        ("cnn_dailymail", "3.0.0"),                    # CNN/DailyMail
        ("EdinburghNLP/xsum", None),                   # XSum
        # Reasoning
        ("openai/gsm8k", "main"),                      # GSM8K (math)
        ("allenai/ai2_arc", "ARC-Challenge"),           # ARC-Challenge (science)
        # Code generation
        ("sahil2801/CodeAlpaca-20k", None),             # CodeAlpaca
    ]

    results = []
    for name, config in datasets_to_download:
        if args.verify_only:
            print(f"  [verify-only] Skipping download: {name}")
            continue
        result = download_dataset(name, config, cache_dir=args.cache_dir)
        results.append(result)

    # Step 3: Download evaluation metrics
    print("\n[3/4] Downloading evaluation metrics...")
    try:
        import evaluate
        for metric_name in ["rouge", "accuracy", "exact_match"]:
            print(f"  Loading metric: {metric_name}")
            evaluate.load(metric_name)
            print(f"    OK")
    except Exception as e:
        print(f"  Warning: Could not load some metrics: {e}")

    # Step 4: Verify tokenizers (requires HF login for Llama)
    if not args.skip_tokenizers:
        print("\n[4/4] Verifying model tokenizers...")
        print("  (Requires: huggingface-cli login with Llama access)")
        tokenizer_models = [
            "meta-llama/Llama-2-7b-hf",
            "meta-llama/Meta-Llama-3-8B",
            "codellama/CodeLlama-7b-hf",
        ]
        tokenizer_results = []
        for model_name in tokenizer_models:
            ok = verify_tokenizer(model_name)
            tokenizer_results.append((model_name, ok))
    else:
        print("\n[4/4] Skipping tokenizer verification (--skip-tokenizers)")
        tokenizer_results = []

    # Summary
    print("\n" + "=" * 60)
    print("  SETUP SUMMARY")
    print("=" * 60)

    if results:
        ok_count = sum(1 for r in results if r["status"] == "ok")
        fail_count = sum(1 for r in results if r["status"] == "error")
        print(f"  Datasets: {ok_count} OK, {fail_count} failed")
        for r in results:
            status = "OK" if r["status"] == "ok" else "FAILED"
            print(f"    [{status}] {r['name']}")
            if r["status"] == "error":
                print(f"           Error: {r['error']}")

    if tokenizer_results:
        tok_ok = sum(1 for _, ok in tokenizer_results if ok)
        tok_fail = sum(1 for _, ok in tokenizer_results if not ok)
        print(f"  Tokenizers: {tok_ok} OK, {tok_fail} failed")
        for model, ok in tokenizer_results:
            status = "OK" if ok else "FAILED"
            short = model.split("/")[-1]
            print(f"    [{status}] {short}")

    if any(r["status"] == "error" for r in results):
        print("\n  Some datasets failed. Fix errors and re-run.")
    elif any(not ok for _, ok in tokenizer_results):
        print("\n  Some tokenizers failed. Run: huggingface-cli login")
        print("  Then request access at: https://huggingface.co/meta-llama")
    else:
        print("\n  All checks passed! Ready for Phase 1 & 2 experiments.")

    print("=" * 60)


if __name__ == "__main__":
    main()
