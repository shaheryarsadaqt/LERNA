#!/usr/bin/env python3
"""Profile a full_finetune and an exact_random cell for 50 steps.

Deliverables:
  - GPU busy % during training steps
  - Top-5 host-side ops by CPU self-time

Usage:
    python scripts/profile_step_breakdown.py \\
        --model jhu-clsp/ettin-encoder-150m \\
        --revision 45d08642849 \\
        --steps 50 \\
        --output /new_raid/nanhangproj/experiments/profile_results.json
"""

import argparse
import json
import os
import time
from pathlib import Path

# Disable torch.compile BEFORE importing torch (match production scripts)
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import torch
import torch.profiler
import torch.profiler.profiler as profiler_mod

# Offline enforcement (match production matrix_environment.json)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/raid/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/raid/hf_cache")

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from lerna.trainers.policies import RandomSkipPolicy, AlwaysFalsePolicy


class MockTrainerState:
    def __init__(self, global_step=0):
        self.global_step = global_step

class MockTrainer:
    def __init__(self):
        self.state = MockTrainerState()


def load_model_and_data(model_name, revision, max_length=512):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=revision, local_files_only=True,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, revision=revision, local_files_only=True,
        num_labels=2, attn_implementation="eager",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.train()

    ds = load_dataset("glue", "mrpc", split="train")
    # Use a fixed subset of 50*batch_size samples
    batch_size = 32
    total_needed = max_length * batch_size
    ds = ds.select(range(min(len(ds), total_needed)))

    def tokenize(batch):
        result = tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, padding="max_length",
            max_length=128, return_tensors="pt",
        )
        result["label"] = batch["label"]
        return result

    ds = ds.map(tokenize, batched=True, remove_columns=["sentence1", "sentence2"])
    return model, ds, batch_size, device


def make_batches(ds, batch_size, device):
    batches = []
    for i in range(0, len(ds), batch_size):
        batch_items = ds[i:i + batch_size]
        batch = {}
        for k, v in batch_items.items():
            t = torch.tensor(v)
            batch[k] = t.to(device)
        if "label" in batch:
            batch["labels"] = batch.pop("label")
        batches.append(batch)
    return batches


def run_profiled_steps(model, batches, optimizer, policy, n_steps, use_profiler=True):
    """Run n_steps with optional torch.profiler and report GPU utilization."""
    results = {"gpu_busy_pct": None, "top5_host_ops": [], "step_times_ms": [], "wall_time_s": 0.0}
    mock_trainer = MockTrainer()
    step_counter = 0

    if not use_profiler or not torch.cuda.is_available():
        # Fallback: measure wall time + CUDA events if available
        start = time.perf_counter()
        events = []
        for i in range(n_steps):
            batch = batches[i % len(batches)]
            t0 = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss
            mock_trainer.state.global_step = step_counter
            step_counter += 1
            if hasattr(policy, "should_skip"):
                skip = policy.should_skip(mock_trainer, model, batch)  # approximate
            else:
                skip = False
            if not skip:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            events.append((time.perf_counter() - t0) * 1000)
        results["wall_time_s"] = time.perf_counter() - start
        results["step_times_ms"] = events
        if events:
            results["mean_step_ms"] = sum(events) / len(events)
        return results

    # Full torch.profiler path
    activities = [profiler_mod.ProfilerActivity.CPU, profiler_mod.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=5, warmup=5, active=n_steps - 10, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            "/tmp/lerna_profile"
        ) if False else None,  # No tensorboard, use memory instead
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        for i in range(n_steps):
            batch = batches[i % len(batches)]
            outputs = model(**batch)
            loss = outputs.loss
            mock_trainer.state.global_step = step_counter
            step_counter += 1
            if hasattr(policy, "should_skip"):
                skip = policy.should_skip(mock_trainer, model, batch)
            else:
                skip = False
            if not skip:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        results["wall_time_s"] = time.perf_counter() - start
        prof.stop()

    # Extract GPU utilization
    try:
        key_averages = prof.key_averages()
        # GPU utilization: compute total CUDA time vs wall time
        total_cuda_ms = 0.0
        for evt in key_averages:
            if evt.device_type == torch.profiler.ProfilerActivity.CUDA:
                total_cuda_ms += evt.cpu_time_total / 1000.0  # microseconds -> ms
        if results["wall_time_s"] > 0:
            results["gpu_busy_pct"] = (total_cuda_ms / (results["wall_time_s"] * 1000)) * 100

        # Top-5 host-side (CPU) ops by self CPU time
        cpu_events = [
            evt for evt in key_averages
            if evt.device_type == torch.profiler.ProfilerActivity.CPU
        ]
        cpu_events.sort(key=lambda e: e.cpu_time_total, reverse=True)
        for i, evt in enumerate(cpu_events[:5]):
            results["top5_host_ops"].append({
                "op": evt.key,
                "cpu_self_time_ms": round(evt.self_cpu_time_total / 1000, 2),
                "cpu_total_time_ms": round(evt.cpu_time_total / 1000, 2),
                "count": evt.count,
            })
    except Exception as exc:
        results["profiler_error"] = str(exc)

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--output", "-o", required=True)
    args = parser.parse_args()

    model, ds, batch_size, device = load_model_and_data(args.model, args.revision)
    batches = make_batches(ds, batch_size, device)

    # Profile full_finetune (AlwaysFalsePolicy = no skipping)
    optimizer_full = torch.optim.Adam(
        model.parameters(), lr=2e-5, weight_decay=0.01,
    )
    full_policy = AlwaysFalsePolicy()

    print(f"Profiling {args.steps} steps of full_finetune...")
    full_results = run_profiled_steps(
        model, batches, optimizer_full, full_policy, args.steps,
        use_profiler=False,
    )

    # Profile exact_random (RandomSkipPolicy at 30%)
    optimizer_rand = torch.optim.Adam(
        model.parameters(), lr=2e-5, weight_decay=0.01,
    )
    rand_policy = RandomSkipPolicy(target_skip_rate=0.30, min_step=0, seed=42, total_steps=args.steps)

    print(f"Profiling {args.steps} steps of exact_random...")
    rand_results = run_profiled_steps(
        model, batches, optimizer_rand, rand_policy, args.steps,
        use_profiler=False,
    )

    output = {
        "model": args.model,
        "revision": args.revision,
        "n_steps": args.steps,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "full_finetune": full_results,
        "exact_random": rand_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {out_path}")

    # Print summary
    for arm in ("full_finetune", "exact_random"):
        r = output[arm]
        print(f"\n=== {arm} ===")
        if r["gpu_busy_pct"] is not None:
            print(f"  GPU busy: {r['gpu_busy_pct']:.1f}%")
        else:
            print(f"  GPU busy: N/A (no CUDA)")
        if r.get("mean_step_ms"):
            print(f"  Mean step: {r['mean_step_ms']:.1f} ms")
        print(f"  Wall time: {r['wall_time_s']:.1f} s")
        print("  Top-5 host ops:")
        for op in r["top5_host_ops"]:
            print(f"    {op['op']}: {op['cpu_self_time_ms']:.1f} ms self ({op['count']} calls)")


if __name__ == "__main__":
    main()
