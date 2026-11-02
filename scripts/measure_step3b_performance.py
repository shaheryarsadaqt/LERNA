#!/usr/bin/env python3
"""
Measure initialization, peak traced allocation, and decision/query cost for the
ACTUAL submitted helper and integrated pure gate on task-sized configurations.

Per STEP3B_CLI_CORRECTION_PROMPT.md section 6:
  * Add a declared pre-allocation guard for helper memory/work.
  * Correct the unconditional O(1) query claim: actual phase search is O(P),
    and Python big-integer operations are width-dependent.
  * Measure initialization, peak traced allocation and decision/query cost for
    the ACTUAL submitted helper and integrated pure gate on task-sized
    configurations, with repeat count/environment/instrumentation noted.
  * Include script and raw output. Do not copy prior reference timings.

CPU-only; no torch/Trainer/model; no training.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lerna.utils.phase_suffix_feasibility import (
    PhaseSuffixFeasibility, phase_bounds,
)
from lerna.utils.forward_gate import ScoreQuantileGate, _eligible_phase_quota


def _source_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except Exception:
        return "unknown"


_MEASUREMENT_SOURCE_BINDINGS = {
    "forward_gate": _source_hash(
        Path(__file__).resolve().parents[1] / "lerna" / "utils" / "forward_gate.py"
    ),
    "phase_suffix_feasibility": _source_hash(
        Path(__file__).resolve().parents[1] / "lerna" / "utils" / "phase_suffix_feasibility.py"
    ),
    "runner": _source_hash(Path(__file__).resolve()),
    "max_allowed_cells": PhaseSuffixFeasibility.MAX_ALLOWED_CELLS,
}


def const_score(val):
    def src(di, model, inputs):
        return val
    return src


def measure_helper(task_name: str, T: int, P: int, W: int, Q: int, K: int, repeats: int = 5):
    """Measure PhaseSuffixFeasibility init + witness.

    Uses production quotas from _eligible_phase_quota (proportional allocation
    over eligible steps, largest-remainder method) instead of even allocation.
    Queries follow actual remaining-quota/streak states from a legal trajectory
    (running the gate sweep and sampling its state at each decision point).
    No silent fallback: allocator failures propagate as errors.
    """
    # Use production quotas from the allocator, not even allocation.
    phase_quota, requested_total, total_eligible, eligible_per_window = (
        _eligible_phase_quota(Q / T, T, W, P)
    )

    times = []
    peak_bytes = 0
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        h = PhaseSuffixFeasibility(T, P, phase_quota, W, K)
        init_ms = (time.perf_counter() - t0) * 1000
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(init_ms)
        peak_bytes = max(peak_bytes, peak)

    # Witness
    t0 = time.perf_counter()
    witness = h.witness()
    witness_ms = (time.perf_counter() - t0) * 1000

    # Per-query cost: sample states from a legal gate trajectory.
    # Build the gate, run the full sweep, and at each sampled decision point
    # query feasibility with the actual remaining quota and streak state.
    gate = ScoreQuantileGate(
        rate=Q / T, total_steps=T, min_step=W,
        max_consecutive_skips=K, n_phases=P,
        score_history_window=64, anneal_fraction=0.88,
        seed=12, score_source=lambda *a: 0.5,
        controller_version="phase_exact_v1",
    )
    # Pre-run to establish legal state at each step
    legal_states = []
    for t in range(T):
        c = gate._consecutive_skips
        q_rem = gate._remaining_in_phase(t)[0]
        legal_states.append((t, c, q_rem))
        gate.decide(t, None, None)

    # Sample 1000 query points from legal trajectory states
    query_times = []
    step = max(1, T // 1000)
    for i in range(0, T, step):
        t, c, q_rem = legal_states[i]
        t0 = time.perf_counter()
        h.is_feasible_at(t, c, q_rem)
        query_times.append((time.perf_counter() - t0) * 1_000_000)  # microseconds

    return {
        "task": task_name,
        "T": T, "P": P, "W": W, "Q": Q, "K": K,
        "quotas": phase_quota,
        "requested_total": requested_total,
        "total_eligible": total_eligible,
        "eligible_per_window": eligible_per_window,
        "repeats": repeats,
        "source_bindings": _MEASUREMENT_SOURCE_BINDINGS,
        "init_ms": {
            "min": min(times), "max": max(times),
            "mean": sum(times) / len(times),
        },
        "peak_traced_bytes": peak_bytes,
        "peak_traced_mb": peak_bytes / (1024 * 1024),
        "witness_ms": witness_ms,
        "witness_found": witness is not None,
        "query_us": {
            "min": min(query_times), "max": max(query_times),
            "mean": sum(query_times) / len(query_times),
        },
    }


def measure_gate(task_name: str, T: int, P: int, W: int, Q: int, K: int, repeats: int = 3):
    """Measure ScoreQuantileGate init + full-horizon decide sweep."""
    rate = Q / T
    times = []
    peak_bytes = 0
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        g = ScoreQuantileGate(
            rate=rate, total_steps=T, min_step=W,
            max_consecutive_skips=K, n_phases=P,
            score_history_window=64, anneal_fraction=0.88,
            seed=12, score_source=const_score(0.5),
            controller_version="phase_exact_v1",
        )
        init_ms = (time.perf_counter() - t0) * 1000
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(init_ms)
        peak_bytes = max(peak_bytes, peak)

    # Full sweep
    t0 = time.perf_counter()
    for t in range(T):
        g.decide(t, None, None)
    sweep_ms = (time.perf_counter() - t0) * 1000

    diag = g.diagnostics()
    return {
        "task": task_name,
        "T": T, "P": P, "W": W, "Q": Q, "K": K,
        "rate": rate,
        "repeats": repeats,
        "source_bindings": _MEASUREMENT_SOURCE_BINDINGS,
        "init_ms": {
            "min": min(times), "max": max(times),
            "mean": sum(times) / len(times),
        },
        "peak_traced_bytes": peak_bytes,
        "peak_traced_mb": peak_bytes / (1024 * 1024),
        "sweep_ms": sweep_ms,
        "sweep_us_per_decision": (sweep_ms * 1000) / T,
        "realized": diag["realized_total"],
        "requested": diag["requested_total"],
        "longest_skip_run": diag["longest_skip_run"],
        "suffix_force_skip": diag["suffix_force_skip"],
        "suffix_force_retain": diag["suffix_force_retain"],
    }


def main():
    results = {
        "scope": "Actual helper/gate measurement, CPU-only, no training.",
        "note": "Phase lookup is O(P), not strict O(1). Big-integer bit operations "
                f"scale with quota in machine words. Pre-allocation guard: "
                f"MAX_ALLOWED_CELLS={PhaseSuffixFeasibility.MAX_ALLOWED_CELLS}.",
        "source_bindings": _MEASUREMENT_SOURCE_BINDINGS,
        "config": {
            "MAX_ALLOWED_CELLS": PhaseSuffixFeasibility.MAX_ALLOWED_CELLS,
        },
        "helper_measurements": [],
        "gate_measurements": [],
    }

    # Task-sized configurations (from V2 review)
    tasks = [
        ("qnli", 6548, 4, 392, 3929, 4),
        ("mrpc", 377, 4, 22, 226, 4),
        ("rte", 257, 4, 15, 154, 4),
        # P=1 variants
        ("qnli_p1", 6548, 1, 392, 3929, 4),
        ("mrpc_p1", 377, 1, 22, 226, 4),
        ("rte_p1", 257, 1, 15, 154, 4),
        # Small config for guard test
        ("tiny", 10, 2, 0, 5, 1),
    ]

    for task_name, T, P, W, Q, K in tasks:
        try:
            m = measure_helper(task_name, T, P, W, Q, K)
            m["source_bindings"] = _MEASUREMENT_SOURCE_BINDINGS
            results["helper_measurements"].append(m)
        except Exception as e:
            results["helper_measurements"].append({
                "task": task_name, "error": str(e),
                "source_bindings": _MEASUREMENT_SOURCE_BINDINGS,
            })

    for task_name, T, P, W, Q, K in tasks:
        try:
            m = measure_gate(task_name, T, P, W, Q, K)
            m["source_bindings"] = _MEASUREMENT_SOURCE_BINDINGS
            results["gate_measurements"].append(m)
        except Exception as e:
            results["gate_measurements"].append({
                "task": task_name, "error": str(e),
                "source_bindings": _MEASUREMENT_SOURCE_BINDINGS,
            })

    # Environment
    results["environment"] = {
        "python": sys.version,
        "platform": sys.platform,
        "instrumentation": "tracemalloc + time.perf_counter",
    }

    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    main()
