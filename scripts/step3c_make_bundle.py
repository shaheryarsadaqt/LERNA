#!/usr/bin/env python3
"""Step 3C Part C: build and verify the complete bundle.

Run after scripts/step3c_cpu_evidence.py, from the repository root:
    python scripts/step3c_make_bundle.py --evidence step3c_evidence --out step3c_final_bundle

What it does, in order. Every command, exit status and raw output is recorded.
  1. Runs pytest with -rs: the contract tests, the acceptance tests, the full suite (tests/), and
     the five Step 3C test files together ("focused"). The focused run loads
     scripts/step3c_files_plugin.py, which records every module imported and every file read.
     It also records the collected test IDs of the focused files.
  2. Runs scripts/measure_step3b_performance.py and keeps its raw output.
  3. Records git state: HEAD, status, recent log, and for each bundled file whether it is tracked
     and unchanged from HEAD.
  4. Bundles every repository file that the focused tests or the evidence runs imported or read,
     plus the five test files, the Part C scripts and any pytest configuration at the repository
     root. FILES.json lists each one with its SHA-256, size, role and git state.
  5. Writes step3c_reconstruct.py into the bundle and verifies it: --check against this checkout,
     --apply into a new temporary directory, then the focused tests inside that directory. This
     passes only if the tests pass there and nothing was imported or read from this checkout.
  6. Writes README.md and bundle_report.json, then source_hashes.txt last.

Exit status 0 only if every step and check passed. It writes only inside --out and a temporary
directory. Refuses to write into an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
FOCUSED = ["tests/test_step3c_execution_contract.py", "tests/test_phase_exact_v1_acceptance.py",
           "tests/test_forward_gate.py", "tests/test_model_loader.py", "tests/test_true_skip_trainer.py"]
KIT = ["scripts/step3c_cpu_evidence.py", "scripts/step3c_make_bundle.py", "scripts/step3c_files_plugin.py",
       "scripts/measure_step3b_performance.py"]
REQUIRED = ["scripts/run_controller_ablation.py", "lerna/utils/forward_gate.py",
            "lerna/utils/phase_suffix_feasibility.py", "lerna/utils/model_loader.py",
            "lerna/utils/ettin_constants.py", "lerna/trainers/true_skip_trainer.py",
            "lerna/trainers/policies.py"]
PYTEST_CONFIG = ["pytest.ini", "pyproject.toml", "setup.cfg", "tox.ini", "conftest.py"]
PLUGIN = "scripts.step3c_files_plugin"
SKIP_PARTS = {".git", ".pytest_cache", "__pycache__", "site-packages", "dist-packages"}
MAX_FILE_BYTES = 20 * 1024 * 1024  # larger files are listed in the report, not copied, and fail a check

RECONSTRUCT = '''#!/usr/bin/env python3
"""Check or rebuild the Step 3C file set listed in FILES.json, next to this script.

  --check [--root DIR]   compare every listed file under DIR (default: the current directory)
                         with its recorded SHA-256. The bundle's own copies are checked too.
  --apply --dest DIR     copy the bundle's files into DIR, which must be new or empty, then
                         check them there.

Exit status 0 when every file matches. This script and FILES.json are not listed in FILES.json
(no self-hash). source_hashes.txt, written last, covers every file in the bundle.
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TABLE = json.loads((HERE / "FILES.json").read_text())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def check(root):
    problems = []
    for rel, info in sorted(TABLE["files"].items()):
        for label, base in (("bundle copy", HERE / "files"), ("target", root)):
            p = base / rel
            if not p.is_file():
                problems.append(label + " missing: " + rel)
            elif sha(p) != info["sha256"]:
                problems.append(label + " differs: " + rel)
    for line in problems:
        print(line)
    print(str(len(TABLE["files"])) + " files listed, " + str(len(problems)) + " problems, root " + str(root))
    return 0 if not problems else 1


def apply(dest):
    if dest.exists() and any(dest.iterdir()):
        print("refusing: " + str(dest) + " exists and is not empty")
        return 2
    for rel in sorted(TABLE["files"]):
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(HERE / "files" / rel, target)
    return check(dest)


def main():
    ap = argparse.ArgumentParser(description="Check or rebuild the Step 3C file set.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--apply", action="store_true")
    ap.add_argument("--root", default=".")
    ap.add_argument("--dest")
    a = ap.parse_args()
    if a.apply:
        if not a.dest:
            ap.error("--apply needs --dest")
        return apply(Path(a.dest).resolve())
    return check(Path(a.root).resolve())


if __name__ == "__main__":
    sys.exit(main())
'''


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(cmd, log_path: Path, cwd: Path, env=None, stdout_path: Path = None):
    """Run cmd; write a header plus raw stdout and stderr to log_path. Return a record."""
    t0 = time.time()
    p = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True,
                       env=dict(os.environ, **(env or {})))
    seconds = round(time.time() - t0, 1)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if stdout_path is not None:
        stdout_path.write_text(p.stdout)
        body = f"(stdout written to {stdout_path.name})\n--- stderr ---\n{p.stderr}"
    else:
        body = p.stdout + ("" if not p.stderr else "\n--- stderr ---\n" + p.stderr)
    log_path.write_text(f"# command: {' '.join(cmd)}\n# cwd: {cwd}\n# exit status: {p.returncode}\n"
                        f"# seconds: {seconds}\n{body}")
    return {"command": cmd, "cwd": str(cwd), "exit_status": p.returncode, "seconds": seconds,
            "log": log_path.name, "summary_line": _summary_line(p.stdout)}


def _summary_line(text):
    lines = [l.strip("= ").strip() for l in text.splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    return lines[-1] if lines else None


def under(path: str, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def repo_rel(path: str, root: Path, exclude):
    """Repo-relative path of a file under root, or None if it is outside or excluded."""
    p = Path(path).resolve()
    try:
        rel = p.relative_to(root.resolve())
    except ValueError:
        return None
    if any(part in SKIP_PARTS for part in rel.parts) or p.suffix == ".pyc":
        return None
    if any(under(str(p), e) for e in exclude):
        return None
    if under(str(p), Path(sys.prefix)) or under(str(p), Path(sys.base_prefix)):
        return None
    return rel.as_posix() if p.is_file() else None


def git(args):
    try:
        p = subprocess.run(["git", *args], cwd=str(REPO_ROOT), capture_output=True, text=True)
    except OSError as e:
        return 127, f"git unavailable: {e}\n"
    return p.returncode, p.stdout


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--focused", nargs="+", default=FOCUSED)
    ap.add_argument("--contract", default="tests/test_step3c_execution_contract.py")
    ap.add_argument("--acceptance", default="tests/test_phase_exact_v1_acceptance.py")
    ap.add_argument("--full-suite", default="tests")
    ap.add_argument("--required", nargs="*", default=REQUIRED)
    ap.add_argument("--measure", default="scripts/measure_step3b_performance.py")
    a = ap.parse_args()
    evidence = Path(a.evidence).resolve()
    out = Path(a.out).resolve()
    if out.exists():
        print(f"refusing to write into existing {out}", file=sys.stderr)
        return 2
    summary_path = evidence / "summary.json"
    if not summary_path.is_file():
        print(f"no evidence summary at {summary_path}; run scripts/step3c_cpu_evidence.py first", file=sys.stderr)
        return 2
    out.mkdir(parents=True)
    logs = out / "logs"
    logs.mkdir()
    tmp = Path(tempfile.mkdtemp(prefix="step3c_bundle_"))
    py = sys.executable
    report = {"repo_root": str(REPO_ROOT), "python": sys.version, "commands": {}, "checks": {}}
    cmds = report["commands"]

    # 1. tests
    cmds["pytest_contract"] = run([py, "-m", "pytest", "-rs", a.contract], logs / "pytest_contract.txt", REPO_ROOT)
    cmds["pytest_acceptance"] = run([py, "-m", "pytest", "-rs", a.acceptance], logs / "pytest_acceptance.txt", REPO_ROOT)
    cmds["pytest_focused"] = run([py, "-m", "pytest", "-rs", "-p", PLUGIN, *a.focused],
                                 logs / "pytest_focused.txt", REPO_ROOT,
                                 env={"STEP3C_FILES_OUT": str(tmp / "files_used_repo.json")})
    cmds["pytest_collected_focused"] = run([py, "-m", "pytest", "--collect-only", "-q", *a.focused],
                                           logs / "pytest_collected_focused.txt", REPO_ROOT)
    cmds["pytest_full"] = run([py, "-m", "pytest", "-rs", a.full_suite], logs / "pytest_full.txt", REPO_ROOT)

    # 2. measurement script
    cmds["measure_step3b_performance"] = run([py, a.measure], logs / "measure_step3b_performance.log",
                                             REPO_ROOT, stdout_path=logs / "measure_step3b_performance.json")
    try:
        meas = json.loads((logs / "measure_step3b_performance.json").read_text())
        meas_errors = [r.get("task") for k in ("helper_measurements", "gate_measurements")
                       for r in meas.get(k, []) if "error" in r]
        meas_ok = isinstance(meas.get("helper_measurements"), list) and not meas_errors
    except ValueError:
        meas_ok, meas_errors = False, ["output is not JSON"]

    # 3. git state
    for name, args in (("git_head", ["rev-parse", "HEAD"]), ("git_status", ["status", "--porcelain"]),
                       ("git_log", ["log", "--oneline", "-15"]), ("git_diff_stat_vs_HEAD", ["diff", "--stat", "HEAD"])):
        rc, text = git(args)
        (logs / f"{name}.txt").write_text(f"# command: git {' '.join(args)}\n# exit status: {rc}\n{text}")
    head = git(["rev-parse", "HEAD"])[1].strip()

    # 4. file set
    exclude = [out, evidence, tmp]
    roles = {}
    try:
        used = json.loads((tmp / "files_used_repo.json").read_text())
    except (OSError, ValueError):
        used = {"modules": {}, "opened_for_reading": []}
    for f in list(used["modules"].values()) + list(used["opened_for_reading"]):
        rel = repo_rel(f, REPO_ROOT, exclude)
        if rel:
            roles.setdefault(rel, "used by the focused tests (imported or read)")
    ev = json.loads(summary_path.read_text())
    for rel in ev.get("repo_modules_loaded_sha256", {}):
        if (REPO_ROOT / rel).is_file():
            roles.setdefault(rel, "imported by the evidence runs")
    for rel in a.focused:
        roles[rel] = "Step 3C test file"
    for rel in KIT:
        if (REPO_ROOT / rel).is_file():
            roles[rel] = "Part C script"
    for rel in PYTEST_CONFIG:
        if (REPO_ROOT / rel).is_file():
            roles.setdefault(rel, "pytest configuration")
    files, too_large = {}, []
    for rel in sorted(roles):
        src = REPO_ROOT / rel
        if src.stat().st_size > MAX_FILE_BYTES:
            too_large.append({"path": rel, "bytes": src.stat().st_size, "role": roles[rel]})
            continue
        dst = out / "files" / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        tracked = git(["ls-files", "--error-unmatch", "--", rel])[0] == 0
        unchanged = tracked and git(["diff", "--quiet", "HEAD", "--", rel])[0] == 0
        files[rel] = {"sha256": sha256(src), "bytes": src.stat().st_size, "role": roles[rel],
                      "git_tracked": tracked, "unchanged_from_HEAD": unchanged}
    (out / "FILES.json").write_text(json.dumps({"repo_root": str(REPO_ROOT), "git_head": head,
                                                "files": files}, indent=1, sort_keys=True))
    shutil.copytree(evidence, out / "evidence")

    # 5. reconstruction script and its verification
    (out / "step3c_reconstruct.py").write_text(RECONSTRUCT)
    cmds["reconstruct_check_repo"] = run([py, str(out / "step3c_reconstruct.py"), "--check", "--root", str(REPO_ROOT)],
                                         logs / "reconstruct_check_repo.txt", REPO_ROOT)
    dest = tmp / "reconstructed"
    cmds["reconstruct_apply_tmp"] = run([py, str(out / "step3c_reconstruct.py"), "--apply", "--dest", str(dest)],
                                        logs / "reconstruct_apply_tmp.txt", REPO_ROOT)
    cmds["pytest_focused_in_reconstruction"] = run(
        [py, "-m", "pytest", "-rs", "-p", PLUGIN, *a.focused], logs / "pytest_focused_in_reconstruction.txt",
        dest, env={"STEP3C_FILES_OUT": str(tmp / "files_used_reconstruction.json")})
    try:
        used_dest = json.loads((tmp / "files_used_reconstruction.json").read_text())
    except (OSError, ValueError):
        used_dest = {"modules": {}, "opened_for_reading": []}
    # Any file under this checkout counts here, including __pycache__/*.pyc, which Python reads instead
    # of the source when the cache is valid.
    from_checkout = sorted({f for f in list(used_dest["modules"].values()) + used_dest["opened_for_reading"]
                            if under(f, REPO_ROOT) and not any(under(f, e) for e in exclude)})
    own_pkgs = sorted({n.split(".")[0] for n, f in used["modules"].items() if repo_rel(f, REPO_ROOT, exclude)})
    outside = sorted(f"{n} -> {f}" for n, f in used_dest["modules"].items()
                     if n.split(".")[0] in own_pkgs and not under(f, dest))
    (logs / "reconstruction_isolation.json").write_text(json.dumps(
        {"files_from_this_checkout": from_checkout, "own_modules_loaded_from_outside_reconstruction": outside,
         "own_top_level_packages": own_pkgs}, indent=1))

    # checks
    c = report["checks"]
    c["evidence_all_checks_passed"] = ev.get("all_checks_passed") is True
    for name in ("pytest_contract", "pytest_acceptance", "pytest_focused", "pytest_collected_focused", "pytest_full",
                 "reconstruct_check_repo", "reconstruct_apply_tmp", "pytest_focused_in_reconstruction"):
        c[f"{name}_exit_0"] = cmds[name]["exit_status"] == 0
    c["measure_exit_0_and_no_record_errors"] = cmds["measure_step3b_performance"]["exit_status"] == 0 and meas_ok
    c["required_files_present"] = all(r in files for r in a.required)
    c["focused_test_files_present"] = all(r in files for r in a.focused)
    c["reconstruction_used_nothing_from_this_checkout"] = not from_checkout and not outside
    c["no_file_too_large_to_bundle"] = not too_large
    report["files_too_large_to_bundle"] = too_large
    m = re.search(r"(\d+) tests? collected", (logs / "pytest_collected_focused.txt").read_text())
    report["focused_tests_collected"] = int(m.group(1)) if m else None
    report["measure_record_errors"] = meas_errors
    report["missing_required_files"] = [r for r in a.required if r not in files]
    report["files_not_tracked_or_changed_from_HEAD"] = sorted(r for r, i in files.items() if not i["unchanged_from_HEAD"])
    report["all_checks_passed"] = all(c.values())
    (out / "bundle_report.json").write_text(json.dumps(report, indent=1, sort_keys=True))

    # 6. README, then source_hashes.txt last
    lines = ["# Step 3C final bundle", "",
             f"Built from `{REPO_ROOT}` at HEAD `{head}` by `scripts/step3c_make_bundle.py`.",
             f"All checks passed: **{report['all_checks_passed']}** (details in `bundle_report.json`).", "",
             "## Checks", "", "| Check | Result |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in c.items()]
    lines += ["", "## Commands", "", "| Name | Exit status | Result line | Log |", "|---|---|---|---|"]
    lines += [f"| {k} | {v['exit_status']} | {v['summary_line'] or ''} | `logs/{v['log']}` |" for k, v in cmds.items()]
    lines += ["", "## Contents", "",
              "- `files/`: every repository file the focused tests or the evidence runs imported or read, the five "
              "Step 3C test files, the Part C scripts and any pytest configuration, at their repository paths. "
              "`FILES.json` lists each with SHA-256, size, role and git state.",
              "- `evidence/`: the output of `scripts/step3c_cpu_evidence.py`: one `execution_manifest.json` per run, "
              "`cli/` with raw `main()` dry-run output, and `summary.json`.",
              "- `logs/`: raw pytest, measurement, reconstruction and git output, each with its command and exit status.",
              "- `step3c_reconstruct.py`: `--check --root <checkout>` or `--apply --dest <new dir>`.",
              "- `source_hashes.txt`: SHA-256 of every other file in this bundle, written last.", "",
              "## Labels and limits", "",
              "- The execution manifests come from non-dry runs in which the four production loader functions were "
              "replaced by CPU fixtures (TinyModel and small tokenized examples). They are not real training. The "
              "40-hex revisions in them are synthetic fixture values, not verified assets.",
              "- `main()` was run only with `--dry-run`.",
              "- Earlier `step3c_manifests/`, `logs/cli_*.json`, `REQUIREMENTS_TO_EVIDENCE_STEP3C.md`, "
              "`COPY_BACK_SUMMARY.md` and the repository's old `scripts/step3c_reconstruct.py` are superseded and "
              "not part of this bundle.", ""]
    (out / "README.md").write_text("\n".join(lines))
    hashes = sorted((p.relative_to(out).as_posix(), sha256(p)) for p in out.rglob("*")
                    if p.is_file() and p.name != "source_hashes.txt")
    (out / "source_hashes.txt").write_text("".join(f"{h}  {rel}\n" for rel, h in hashes))
    shutil.rmtree(tmp, ignore_errors=True)
    print(json.dumps({"all_checks_passed": report["all_checks_passed"], "checks": c,
                      "files_bundled": len(files)}, indent=2))
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
