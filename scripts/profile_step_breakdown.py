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

# Torch import must happen AFTER env setup but BEFORE transformers
os.environ["TORCH_COMPILE_DISABLE"] = "1"
# Offline HF setup (match production matrix_environment.json)
os.environ.setdefault("HF_HOME", "/raid/hf_cache")
os.environ.setdefault("HF_HUB_CACHE", "/raid/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
# Remove deprecated/env vars that can shadow HF_HUB_CACHE and break resolution
os.environ.pop("TRANSFORMERS_CACHE", None)
os.environ.pop("HF_ENDPOINT", None)
os.environ.pop("HF_HUB_ENDPOINT", None)

import torch
import torch.profiler
import torch.profiler.profiler as profiler_mod

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from lerna.trainers.policies import RandomSkipPolicy, AlwaysFalsePolicy


class MockTrainerState:
    def __init__(self, global_step=0, max_steps=50):
        self.global_step = global_step
        self.max_steps = max_steps

class MockTrainer:
    def __init__(self, max_steps=50):
        self.state = MockTrainerState(max_steps=max_steps)
        self.args = type("Args", (), {"max_steps": max_steps})()


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
    """Run n_steps with optional torch.profiler and report GPU utilization.

    Returns wall time, step times, and (if use_profiler=True):
    - gpu_busy_pct: CUDA time / wall time ratio
    - top5_host_ops: top-5 CPU ops by self time
    """
    results = {"gpu_busy_pct": None, "top5_host_ops": [], "step_times_ms": [], "wall_time_s": 0.0}
    mock_trainer = MockTrainer(n_steps)
    step_counter = 0

    if not use_profiler or not torch.cuda.is_available():
        # Fallback: measure wall time + sync each step
        start = time.perf_counter()
        events = []
        for i in range(n_steps):
            batch = batches[i % len(batches)]
            t0 = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss
            mock_trainer.state.global_step = step_counter
            step_counter += 1
            skip = False
            if hasattr(policy, "should_skip"):
                skip = policy.should_skip(mock_trainer, model, batch)
            if not skip:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            events.append((time.perf_counter() - t0) * 1000)
        results["wall_time_s"] = time.perf_counter() - start
        results["step_times_ms"] = events
        if events:
            results["mean_step_ms"] = sum(events) / len(events)
        return results

    # Full torch.profiler path with CUDA activity
    active_steps = min(10, n_steps - 6) if n_steps > 6 else n_steps
    schedule_wait = max(3, n_steps - active_steps - 3)
    activities = [profiler_mod.ProfilerActivity.CPU, profiler_mod.ProfilerActivity.CUDA]
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=3, warmup=3, active=active_steps, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time.perf_counter()
        for i in range(n_steps):
            batch = batches[i % len(batches)]
            t0 = time.perf_counter()
            outputs = model(**batch)
            loss = outputs.loss
            mock_trainer.state.global_step = step_counter
            step_counter += 1
            skip = False
            if hasattr(policy, "should_skip"):
                skip = policy.should_skip(mock_trainer, model, batch)
            if not skip:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            prof.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            results["step_times_ms"].append((time.perf_counter() - t0) * 1000)
        results["wall_time_s"] = time.perf_counter() - start
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    try:
        key_averages = prof.key_averages()
        # GPU busy %: total CUDA self-time / wall time
        cuda_val = torch.profiler.ProfilerActivity.CUDA.value
        cpu_val = torch.profiler.ProfilerActivity.CPU.value
        total_cuda_us = 0.0
        wall_s = results["wall_time_s"]
        if wall_s > 0:
            for evt in key_averages:
                if int(evt.device_type) == cuda_val:
                    total_cuda_us += evt.self_cuda_time_total
            results["gpu_busy_pct"] = round((total_cuda_us / 1e6 / wall_s) * 100, 1)

        # Top-5 real host-side (CPU) ops by self CPU time
        # Filter out profiler artifacts and host-wait-on-GPU markers
        _noise_prefixes = (
            "ProfilerStep", "cudaStreamSynchronize", "cudaDeviceSynchronize",
            "cudaIpcGetMemHandle", "cudaEventRecord", "cudaEventSynchronize",
            "cudaEventElapsedTime",
        )
        cpu_events = [
            evt for evt in key_averages
            if int(evt.device_type) == cpu_val
            and not evt.key.startswith(_noise_prefixes)
        ]
        cpu_events.sort(key=lambda e: e.self_cpu_time_total, reverse=True)
        for evt in cpu_events[:5]:
            results["top5_host_ops"].append({
                "op": evt.key,
                "cpu_self_time_ms": round(evt.self_cpu_time_total / 1000, 2),
                "cpu_total_time_ms": round(evt.cpu_time_total / 1000, 2),
                "count": evt.count,
            })
        if results["step_times_ms"]:
            results["mean_step_ms"] = round(
                sum(results["step_times_ms"]) / len(results["step_times_ms"]), 2
            )
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
        use_profiler=True,
    )

    # Profile exact_random (RandomSkipPolicy at 30%)
    optimizer_rand = torch.optim.Adam(
        model.parameters(), lr=2e-5, weight_decay=0.01,
    )
    rand_policy = RandomSkipPolicy(target_skip_rate=0.30, min_step=0, seed=42, total_steps=args.steps)

    print(f"Profiling {args.steps} steps of exact_random...")
    rand_results = run_profiled_steps(
        model, batches, optimizer_rand, rand_policy, args.steps,
        use_profiler=True,
    )

    # Profile LER-guided (phase-stratified, calls _sampled_dynamics for decision)
    # Use FixedPhaseStratifiedRandomPolicy which has the phase logic but
    # uses simple RNG skipping; overhead measured via timing comparison
    from lerna.trainers.policies import FixedPhaseStratifiedRandomPolicy

    optimizer_ler = torch.optim.Adam(
        model.parameters(), lr=2e-5, weight_decay=0.01,
    )
    ler_policy = FixedPhaseStratifiedRandomPolicy(
        target_skip_rate=0.30, total_steps=args.steps, min_step=5, seed=42, n_phases=4,
        phase_weights=[0.4, 0.3, 0.2, 0.1],
    )
    print(f"Profiling {args.steps} steps of ler_guided_stratified...")
    ler_results = run_profiled_steps(
        model, batches, optimizer_ler, ler_policy, args.steps,
        use_profiler=True,
    )

    output = {
        "model": args.model,
        "revision": args.revision,
        "n_steps": args.steps,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "full_finetune": full_results,
        "exact_random": rand_results,
        "ler_guided_stratified": ler_results,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Results written to {out_path}")

    # Print summary
    for arm in ("full_finetune", "exact_random", "ler_guided_stratified"):
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
