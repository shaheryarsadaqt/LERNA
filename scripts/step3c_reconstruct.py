#!/usr/bin/env python3
"""
LERNA Step 3C — Reconstruction Script

Reproduces the corrected source files from the base commit into an isolated
destination. Path mapping and helper dependencies are listed explicitly. No
pretrained model, real dataset, downloads, or GPU are required.

Usage:
    python scripts/step3c_reconstruct.py --check        # verify current state
    python scripts/step3c_reconstruct.py --apply --dest /tmp/step3c_recon  # copy to isolated dest
    python scripts/step3c_reconstruct.py --diff         # show base-relative diff

Base: commit 54678fb (fix/e3-reduced-steps-budget)
Corrected files:
  scripts/run_controller_ablation.py   — R4-A/B/C/R5 execution-contract fixes
  tests/test_step3c_execution_contract.py — NEW, R4-D targeted CPU tests
  tests/test_phase_exact_v1_acceptance.py — QNLI prerequisite fix
  REQUIREMENTS_TO_EVIDENCE_STEP3C.md   — NEW, R5 evidence table
  COPY_BACK_SUMMARY.md                 — NEW, summary

Preserved (pre-existing, not re-implemented):
  lerna/utils/forward_gate.py          — phase_exact_v1, count equalities, S1
  tests/test_forward_gate.py           — existing S1/gate tests
"""

from __future__ import annotations

import difflib
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BASE_COMMIT = "54678fb"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Path mapping: repo-relative path -> {description, expected_sha256}
# expected_sha256 is the full hash captured at bundle time.
# ---------------------------------------------------------------------------

FILES = {
    "scripts/run_controller_ablation.py": {
        "description": "Execution-contract corrections (R4-A, R4-B, R4-C, R5)",
        "required": True,
        "expected_sha256": "d0e05cfc0fded697a67542d1533d5efec041bd46c9d8677ba63535e0b276ba00",
    },
    "tests/test_step3c_execution_contract.py": {
        "description": "Targeted CPU contract tests (R4-D)",
        "expected_sha256": "b8665bce6dd3d2305e23af7ed7f51a18ad31d7c2d46944ace9e56c145e13b967",
    },
    "tests/test_phase_exact_v1_acceptance.py": {
        "description": "QNLI prerequisite fix in existing test",
        "required": True,
        "expected_sha256": "92ed55337d92218cfc3c86bb149c170802efd981aed4540a0c8c8944910127ce",
    },
    # NOTE: scripts/step3c_reconstruct.py is intentionally NOT in FILES.
    # A script cannot verify its own hash — editing the script changes its
    # hash, so self-check would always fail. The reconstruct script's own
    # hash is tracked in source_hashes.txt instead.
    "lerna/utils/model_loader.py": {
        "description": "Model/tokenizer loading with revision/local_files_only propagation",
        "required": True,
        "expected_sha256": "d8c8cbc89efeab09d933f08ded791ce10663593f1ddf1e898f4e7056c4660752",
    },
    "REQUIREMENTS_TO_EVIDENCE_STEP3C.md": {
        "description": "Requirements-to-evidence table (R5)",
        "required": True,
        "expected_sha256": "9d7560517a98333414b38828159ea1b1b5a16b12b7d9c502eac17df8051466c3",
    },
    "COPY_BACK_SUMMARY.md": {
        "description": "Copy-back summary",
        "required": True,
        "expected_sha256": "6a6c2d4a2cba46ea0c598cdf3d4c40594afe8290f25fbb4ba1a753fb7b181ccf",
    },
    "scripts/measure_step3b_performance.py": {
        "description": "PhaseSuffixFeasibility / ScoreQuantileGate performance measurement script (R5)",
        "required": True,
        "expected_sha256": "ee5997db82b62fb2c0efbf59eec9b836343ee103cd164b8a9bd41671b27cfc26",
    },
}

# Helper dependencies (external modules the corrected code relies on)
HELPERS = {
    "lerna/utils/model_loader.py": "Approved model/tokenizer loading primitives",
    "lerna/utils/ettin_constants.py": "Ettin model ID, revision, and validation",
    "lerna/utils/forward_gate.py": "ScoreQuantileGate, ScoreQuantileGatePolicy, forward_gain_score",
    "lerna/utils/phase_suffix_feasibility.py": "PhaseSuffixFeasibility for phase_exact_v1",
    "lerna/trainers/true_skip_trainer.py": "TrueBackwardSkippingTrainer, SchedulerStepPolicy",
    "lerna/trainers/policies.py": "Skip policies including AlwaysFalsePolicy",
}


# ---------------------------------------------------------------------------
# --check: verify current files match expected hashes
# ---------------------------------------------------------------------------

def cmd_check() -> int:
    """Verify all corrected files exist and match expected hashes."""
    print(f"Base commit: {BASE_COMMIT}")
    print()
    all_ok = True
    for relpath, meta in FILES.items():
        path = REPO_ROOT / relpath
        if not path.exists():
            print(f"  [MISS] {relpath}  — {meta['description']}")
            all_ok = False
            continue
        actual = sha256_file(path)
        expected = meta.get("expected_sha256")
        if expected is not None and actual != expected:
            print(f"  [HASH] {relpath}  (actual={actual[:16]}..., "
                  f"expected={expected[:16]}...)  — {meta['description']}")
            all_ok = False
        else:
            print(f"  [OK]   {relpath}  ({actual[:16]}...)  — {meta['description']}")
    print()
    print("Helper dependencies (must be present):")
    for relpath, desc in HELPERS.items():
        path = REPO_ROOT / relpath
        if path.exists():
            h = sha256_file(path)
            print(f"  [OK]   {relpath}  ({h[:16]}...)  — {desc}")
        else:
            print(f"  [MISS] {relpath}  — {desc}")
            all_ok = False
    print()
    if all_ok:
        print("All files present and hashes match. Reconstruction check PASSED.")
        return 0
    print("Some files missing or hashes differ. Reconstruction check FAILED.")
    return 1


# ---------------------------------------------------------------------------
# --apply: copy corrected files to an isolated destination
# ---------------------------------------------------------------------------

def cmd_apply(dest: str) -> int:
    """Idempotent apply: copy corrected files to an isolated destination.

    For each corrected file:
      - If the destination file does not exist: copy it (reconstruct).
      - If it exists and content matches: skip (idempotent).
      - If it exists and content differs: refuse (conflicting newer target).

    Returns 0 if all files reconstructed or already match, 1 otherwise.
    """
    dest_root = Path(dest)
    dest_root.mkdir(parents=True, exist_ok=True)

    print(f"Applying corrections to isolated destination: {dest_root}")
    print(f"Base commit: {BASE_COMMIT}")
    print()
    all_ok = True

    for relpath, meta in FILES.items():
        src = REPO_ROOT / relpath
        dst = dest_root / relpath
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not src.exists():
            print(f"  [MISS] {relpath}  — source not found: {meta['description']}")
            all_ok = False
            continue

        actual_src_hash = sha256_file(src)
        expected = meta.get("expected_sha256")
        if expected is not None and actual_src_hash != expected:
            print(f"  [HASH] {relpath}  source hash mismatch "
                  f"(actual={actual_src_hash[:16]}..., expected={expected[:16]}...)")
            all_ok = False
            continue

        if dst.exists():
            dst_hash = sha256_file(dst)
            if dst_hash == actual_src_hash:
                print(f"  [SKIP] {relpath}  — already matches (idempotent)")
                continue
            else:
                print(f"  [REFUSE] {relpath}  — destination exists with "
                      f"different content; refusing to overwrite conflicting target")
                all_ok = False
                continue

        # Copy the corrected file to the destination
        shutil.copy2(src, dst)
        print(f"  [OK]   {relpath}  — reconstructed ({actual_src_hash[:16]}...)")

    print()
    print("Helper dependencies (must be present in source repo):")
    for relpath, desc in HELPERS.items():
        path = REPO_ROOT / relpath
        if path.exists():
            h = sha256_file(path)
            print(f"  [OK]   {relpath}  ({h[:16]}...)  — {desc}")
        else:
            print(f"  [MISS] {relpath}  — {desc}")
            all_ok = False

    print()
    if all_ok:
        print("All files reconstructed or already present. Apply PASSED.")
        print(f"Isolated destination ready at: {dest_root}")
        return 0
    print("Some files could not be reconstructed. Apply FAILED.")
    return 1


# ---------------------------------------------------------------------------
# --diff: show base-relative diff for corrected files
# ---------------------------------------------------------------------------

def cmd_diff() -> int:
    """Show base-relative diff for corrected files."""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=REPO_ROOT,
        )
        head = r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        head = "unknown"
    print(f"HEAD: {head}")
    print(f"Base: {BASE_COMMIT}")
    print()

    for relpath in FILES:
        path = REPO_ROOT / relpath
        # Determine file status relative to base commit and working directory.
        in_base = False
        try:
            r = subprocess.run(
                ["git", "cat-file", "-e", f"{BASE_COMMIT}:{relpath}"],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            in_base = (r.returncode == 0)
        except Exception:
            pass
        exists_locally = path.exists()

        if not exists_locally and not in_base:
            print(f"=== {relpath} (never in base; not present locally) ===")
            print()
            continue
        if not exists_locally and in_base:
            print(f"=== {relpath} (LOCALLY DELETED; present in base {BASE_COMMIT}) ===")
            print()
            continue
        if exists_locally and not in_base:
            print(f"=== {relpath} (NEW FILE; not in base {BASE_COMMIT}) ===")
            print(path.read_text())
            print()
            continue
        # exists_locally and in_base: show diff
        try:
            r = subprocess.run(
                ["git", "diff", BASE_COMMIT, "--", relpath],
                capture_output=True, text=True, cwd=REPO_ROOT,
            )
            if r.stdout:
                print(f"=== {relpath} ===")
                print(r.stdout)
            else:
                print(f"=== {relpath} (no diff vs {BASE_COMMIT}) ===")
        except Exception as e:
            print(f"=== {relpath} (diff failed: {e}) ===")
        print()
    return 0


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    import argparse
    parser = argparse.ArgumentParser(
        description="LERNA Step 3C reconstruction script"
    )
    parser.add_argument("--check", action="store_true", help="verify file presence")
    parser.add_argument("--apply", action="store_true", help="reconstruct files to isolated destination")
    parser.add_argument("--diff", action="store_true", help="show base-relative diff")
    parser.add_argument("--dest", default="/tmp/step3c_recon",
                        help="isolated destination for --apply (default: /tmp/step3c_recon)")
    args = parser.parse_args()

    if args.check:
        return cmd_check()
    if args.diff:
        return cmd_diff()
    if args.apply:
        return cmd_apply(args.dest)
    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
