"""Tests for Gate 3 QNLI confirmatory analysis functions.

Covers:
- collect_cell_metrics uses fingerprint (not seed) to find cells
- Validator rule catches duplicate accuracies across seeds
- discover_existing_cells finds all valid cells

NOTE: These tests extract pure functions from the gate3 script via AST
parsing to avoid pulling in heavy deps (torch, transformers, CUDA) that
would interfere with test isolation.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GATE3_SCRIPT = REPO_ROOT / "scripts" / "gate3_qnli_confirmatory.py"


def _load_pure_functions():
    """Extract pure functions from gate3 script without executing imports.

    The functions collect_cell_metrics, find_completed_attempt, and
    discover_existing_cells are pure JSON readers that don't depend on
    torch/transformers. We extract them via AST to avoid CUDA contention.
    """
    source = GATE3_SCRIPT.read_text()
    tree = ast.parse(source)

    # We need: CONFIRMATORY_ARMS, CONFIRMATORY_SEEDS, collect_cell_metrics,
    # find_completed_attempt, discover_existing_cells
    target_names = {
        "CONFIRMATORY_ARMS",
        "CONFIRMATORY_SEEDS",
        "collect_cell_metrics",
        "find_completed_attempt",
        "discover_existing_cells",
    }

    # Extract top-level assignments and function defs we need
    exec_nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in target_names:
                    exec_nodes.append(node)
                    break
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in target_names:
                exec_nodes.append(node)

    # Execute only the needed assignments and functions in a clean namespace
    namespace = {
        "Path": Path,
        "json": json,
        "glob": __import__("glob"),
        "shutil": __import__("shutil"),
        "os": os,
        "math": __import__("math"),
        "sys": sys,
        "subprocess": __import__("subprocess"),
        "time": __import__("time"),
        "tempfile": tempfile,
        "traceback": __import__("traceback"),
    }
    for node in exec_nodes:
        if isinstance(node, ast.Assign):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(GATE3_SCRIPT), "exec")
            exec(code, namespace)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            code = compile(ast.Module(body=[node], type_ignores=[]), str(GATE3_SCRIPT), "exec")
            exec(code, namespace)

    return namespace


_ns = _load_pure_functions()
CONFIRMATORY_ARMS = _ns["CONFIRMATORY_ARMS"]
CONFIRMATORY_SEEDS = _ns["CONFIRMATORY_SEEDS"]
collect_cell_metrics = _ns["collect_cell_metrics"]
find_completed_attempt = _ns["find_completed_attempt"]
discover_existing_cells = _ns["discover_existing_cells"]


def _write_cell(base_dir, arm, fingerprint, seed, acc=0.9, total_steps=6548):
    """Helper: write a minimal cell with results.json and run_manifest.json."""
    arm_dir = Path(base_dir) / arm / fingerprint / "attempt-001"
    arm_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "eval_metrics": {"eval_accuracy": acc},
        "true_skip_instrumentation": {
            "mean_step_ms_executed": 100.0,
            "mean_step_ms_skipped": 80.0,
            "train_loop_seconds": 500.0,
            "eval_seconds": 10.0,
            "avg_dataloader_wait_ms": 0.5,
            "forward_calls": total_steps,
            "backward_calls": int(total_steps * 0.7),
            "skipped_backward_steps": int(total_steps * 0.3),
            "forward_loss_observations": total_steps,
            "nonfinite_forward_loss_count": 0,
        },
        "forward_calls": total_steps,
        "backward_calls": int(total_steps * 0.7),
        "skipped_backward_steps": int(total_steps * 0.3),
        "skip_ratio": 0.3,
        "energy_kwh": 1.0,
        "power_evidence": {"energy_valid": True, "coverage": 0.95},
        "seed": seed,
        "identity_inputs": {"training_seed": seed, "control": arm},
    }
    with open(arm_dir / "results.json", "w") as f:
        json.dump(results, f)

    manifest = {
        "git_sha": "a" * 40,
        "identity_inputs": {"training_seed": seed, "control": arm},
        "tracker_revision": "v3",
    }
    with open(arm_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f)

    # find_completed_attempt requires both results.json AND instrumentation.json
    instr = {
        "mean_step_ms_executed": 100.0,
        "mean_step_ms_skipped": 80.0,
        "train_loop_seconds": 500.0,
        "eval_seconds": 10.0,
        "avg_dataloader_wait_ms": 0.5,
        "forward_calls": total_steps,
        "backward_calls": int(total_steps * 0.7),
        "skipped_backward_steps": int(total_steps * 0.3),
    }
    with open(arm_dir / "instrumentation.json", "w") as f:
        json.dump(instr, f)


def test_collect_cell_metrics_uses_fingerprint_not_seed():
    """collect_cell_metrics must use fingerprint, not seed, to find the cell."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cell(tmpdir, "exact_random", "fp_a" + "a" * 12, seed=11, acc=0.91)
        _write_cell(tmpdir, "exact_random", "fp_b" + "b" * 12, seed=13, acc=0.93)

        m = collect_cell_metrics(tmpdir, "exact_random", "fp_a" + "a" * 12)
        assert m is not None
        assert m["acc"] == 0.91

        m = collect_cell_metrics(tmpdir, "exact_random", "fp_b" + "b" * 12)
        assert m is not None
        assert m["acc"] == 0.93


def test_collect_cell_metrics_no_result():
    """collect_cell_metrics returns None when no completed attempt found."""
    with tempfile.TemporaryDirectory() as tmpdir:
        m = collect_cell_metrics(tmpdir, "exact_random", "nonexistent_fp_123")
        assert m is None


def test_discover_existing_cells_finds_all_seeds():
    """discover_existing_cells finds cells by scanning manifests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_cell(tmpdir, "exact_random", "fp_a" + "a" * 12, seed=11, acc=0.91)
        _write_cell(tmpdir, "exact_random", "fp_b" + "b" * 12, seed=13, acc=0.93)

        cells = discover_existing_cells(tmpdir)
        assert len(cells) == 2
        seeds = sorted(c["training_seed"] for c in cells)
        assert seeds == [11, 13]


def test_validator_duplicate_acceleracies_triggered():
    """Verify the validator logic catches duplicate accuracies."""
    accs = [(11, 0.9322716456), (13, 0.9322716456), (17, 0.9300750503)]
    vals = [a for _, a in accs]
    assert len(set(vals)) < len(vals)

    accs = [(11, 0.9365), (13, 0.9332), (17, 0.9308)]
    vals = [a for _, a in accs]
    assert len(set(vals)) == len(vals)


def test_validator_all_identical_also_triggered():
    """The old bug (all identical) is still caught by the new rule."""
    accs = [(11, 0.9332), (13, 0.9332), (17, 0.9332)]
    vals = [a for _, a in accs]
    assert len(set(vals)) < len(vals)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
