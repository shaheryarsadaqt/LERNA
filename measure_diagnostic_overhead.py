from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

import amp_harness


REPEATS = 3
TRAIN_SEED = 20260916
STEPS = 8
SKIP_AFTER_REAL_STEPS = 4
OUT = Path(os.environ.get("LERNA_OVERHEAD_OUT", "diagnostic_overhead_evidence"))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_once(mode: str, repetition: int, order_index: int) -> dict:
    set_seed(TRAIN_SEED)
    path = OUT / f"repeat_{repetition:02d}" / mode
    path.mkdir(parents=True, exist_ok=True)
    model = amp_harness.TinyModel().cuda()
    policy = amp_harness.ScriptedPolicy(skip_after_real_steps=SKIP_AFTER_REAL_STEPS)
    tracker = None
    if mode == "sampled_lagged":
        tracker = amp_harness.SampledLaggedLERTracker(
            parameter_sample_size=2,
            sample_seed=17,
        )
    trainer, recorder = amp_harness.build(
        path,
        model,
        policy,
        STEPS,
        tracker=tracker,
    )
    torch.cuda.synchronize()
    started = time.perf_counter()
    trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    instrumentation = trainer.get_instrumentation()
    tracker_diagnostics = tracker.get_diagnostics() if tracker is not None else None
    return {
        "mode": mode,
        "repetition": repetition,
        "order_index": order_index,
        "seed": TRAIN_SEED,
        "steps": STEPS,
        "skip_after_real_steps": SKIP_AFTER_REAL_STEPS,
        "elapsed_seconds": elapsed,
        "forward_calls": instrumentation["forward_calls"],
        "backward_calls": instrumentation["backward_calls"],
        "skipped_backward_steps": instrumentation["skipped_backward_steps"],
        "online_ler_update_attempts": instrumentation["online_ler_update_attempts"],
        "online_ler_update_successes": instrumentation["online_ler_update_successes"],
        "tracker_diagnostics": tracker_diagnostics,
        "events": recorder.events,
    }


def main() -> None:
    identity = amp_harness.source_identity()
    if not torch.cuda.is_available():
        raise RuntimeError("diagnostic overhead requires CUDA")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            "run with exactly one visible GPU, for example CUDA_VISIBLE_DEVICES=0"
        )
    OUT.mkdir(parents=True, exist_ok=True)
    print("BEGIN warmup", flush=True)
    run_once("off", -1, 0)
    run_once("sampled_lagged", -1, 1)
    print("PASS warmup", flush=True)
    runs = []
    for repetition in range(REPEATS):
        modes = ("off", "sampled_lagged") if repetition % 2 == 0 else (
            "sampled_lagged",
            "off",
        )
        for order_index, mode in enumerate(modes):
            print(f"BEGIN repeat={repetition} mode={mode}", flush=True)
            result = run_once(mode, repetition, order_index)
            runs.append(result)
            print(
                f"PASS repeat={repetition} mode={mode} "
                f"elapsed={result['elapsed_seconds']:.6f}s",
                flush=True,
            )
    by_repeat = {}
    for run in runs:
        by_repeat.setdefault(run["repetition"], {})[run["mode"]] = run
    paired = []
    for repetition, modes in sorted(by_repeat.items()):
        off = modes["off"]["elapsed_seconds"]
        sampled = modes["sampled_lagged"]["elapsed_seconds"]
        paired.append({
            "repetition": repetition,
            "off_seconds": off,
            "sampled_lagged_seconds": sampled,
            "sampled_minus_off_seconds": sampled - off,
            "sampled_over_off_ratio": sampled / off,
        })
    summary = {
        "source_identity": identity,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "repeats": REPEATS,
        "matched_configuration": {
            "seed": TRAIN_SEED,
            "steps": STEPS,
            "skip_after_real_steps": SKIP_AFTER_REAL_STEPS,
            "modes": ["off", "sampled_lagged"],
        },
        "runs": runs,
        "paired": paired,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"output": str(OUT), "paired": paired}, indent=2), flush=True)


if __name__ == "__main__":
    main()