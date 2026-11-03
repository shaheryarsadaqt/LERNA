#!/usr/bin/env python3
"""Step 3C Part C: non-dry CPU evidence through explicit loader replacements.

What it runs
  1. run_execution(cfg) with cfg.dry_run=False for three configurations:
       gate_phase_exact_v1_cosine_W5  gate_s1_p1, phase_exact_v1, S1, T=20, rate 0.5 (Q=10),
                                      min_step 2, LR warmup 5, cosine
       full_finetune_linear_W3        full_finetune, T=20, LR warmup 3, linear
       reduced_steps_cosine_W2        reduced_steps, T=20, rate 0.5 (10 steps), LR warmup 2, cosine
     For these runs only, the four production loader functions are replaced by CPU fixtures:
       _load_model_production          -> TinyModel(hidden=8, classes=2)
       _load_tokenizer_production      -> a pad-only tokenizer (pad_token_id=0, padding_side="right")
       _load_train_dataset_production  -> 40 tokenized fixture examples, lengths 2 to 6
       _load_eval_dataset_production   -> 16 tokenized fixture examples, lengths 2 to 5
     TrueBackwardSkippingTrainer is replaced by a subclass that only adds a TrainerCallback. The
     callback records the learning rate and the trainer's counters after every scheduled batch.
     The real Trainer loop, gate, policy, collator, LR check and evaluator run unchanged.
  2. main() with --dry-run, in subprocesses: one accepted plan and two rejected configs. The raw
     stdout, stderr and exit status are recorded.

What it does not do
  No pretrained weights, no real data, no downloads, no GPU. The 40-hex revisions are synthetic
  fixture values, not verified assets. main() is never run without --dry-run.

Usage:  python scripts/step3c_cpu_evidence.py --out step3c_evidence
Writes <out>/<run>/execution_manifest.json, <out>/cli/<case>.json and <out>/summary.json.
Exit status 0 only if every check passes. Refuses to write into an existing --out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch  # noqa: E402
import transformers  # noqa: E402
from transformers import TrainerCallback  # noqa: E402

import scripts.run_controller_ablation as R  # noqa: E402

LR = 2e-5
LR_TOL = 1e-12
SYNTHETIC_PINS = {"model_revision": "a" * 40, "tokenizer_revision": "b" * 40, "dataset_revision": "c" * 40}
COMMON = dict(task="sst-2", n_phases=1, max_consecutive_skips=4, policy_seed=0, learning_rate=LR,
              train_batch_size=8, eval_batch_size=8, dry_run=False, **SYNTHETIC_PINS)
RUNS = [
    {"name": "gate_phase_exact_v1_cosine_W5",
     "cfg": dict(arm="gate_s1_p1", controller_version="phase_exact_v1", score_source="s1",
                 total_steps=20, min_step=2, target_skip_rate=0.5,
                 lr_warmup_steps=5, lr_schedule_shape="cosine"),
     "expect": {"T_eff": 20, "Q": 10}},
    {"name": "full_finetune_linear_W3",
     "cfg": dict(arm="full_finetune", controller_version="legacy_v4", score_source="uniform",
                 total_steps=20, min_step=0, target_skip_rate=0.0,
                 lr_warmup_steps=3, lr_schedule_shape="linear"),
     "expect": {"T_eff": 20, "Q": 0}},
    {"name": "reduced_steps_cosine_W2",
     "cfg": dict(arm="reduced_steps", controller_version="legacy_v4", score_source="uniform",
                 total_steps=20, min_step=0, target_skip_rate=0.5,
                 lr_warmup_steps=2, lr_schedule_shape="cosine"),
     "expect": {"T_eff": 10, "Q": 0}},
]
CLI_CASES = [
    {"name": "dry_run_plan_gate_phase_exact_v1",
     "args": ["--dry-run", "--task", "sst-2", "--arm", "gate_s1_p1", "--controller-version", "phase_exact_v1",
              "--score-source", "s1", "--total-steps", "20", "--min-step", "2", "--n-phases", "1",
              "--skip-rate", "0.5"],
     "expect": {"exit": 0, "status": "OK", "execution_mode": "plan_only"}},
    {"name": "dry_run_rejected_skip_rate_1.5",
     "args": ["--dry-run", "--task", "sst-2", "--skip-rate", "1.5"],
     "expect": {"exit": 1, "status": "REJECTED", "execution_mode": "plan_only"}},
    {"name": "dry_run_rejected_blank_model_id",
     "args": ["--dry-run", "--task", "sst-2", "--model-id", ""],
     "expect": {"exit": 1, "status": "REJECTED", "execution_mode": "plan_only"}},
]
REPLACED = {
    "_load_model_production": "TinyModel(hidden=8, classes=2) CPU fixture; no pretrained weights",
    "_load_tokenizer_production": "pad-only tokenizer fixture: pad_token_id=0, padding_side='right'",
    "_load_train_dataset_production": "40 tokenized fixture examples (input_ids lengths 2-6, labels i % 2)",
    "_load_eval_dataset_production": "16 tokenized fixture examples (input_ids lengths 2-5, labels i % 2)",
}
EVIDENCE_LABEL = {
    "kind": "non_dry_cpu_run_with_replaced_loaders",
    "replaced_boundaries": REPLACED,
    "observation_wrapper": ("TrueBackwardSkippingTrainer subclass that only adds a TrainerCallback recording "
                            "the learning rate and counters after every scheduled batch"),
    "unchanged": "Trainer loop, gate and policy, collator, LR endpoint check, evaluator",
    "pins": "synthetic 40-hex fixture revisions, not verified assets",
    "not_real_training": True,
    "note": ("provenance.execution_mode says 'production_loaders' because run_execution selected the "
             "production loader functions; this script replaced those functions with the fixtures above"),
}


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _PadTokenizer:
    pad_token_id = 0
    padding_side = "right"


def _example(i, n):
    return {"input_ids": torch.tensor([1 + (i * 7 + j) % 50 for j in range(n)]),
            "attention_mask": torch.tensor([1] * n),
            "labels": torch.tensor(i % 2)}


def fixture_model(cfg):
    return R.TinyModel(hidden=8, classes=2)


def fixture_tokenizer(cfg):
    return _PadTokenizer()


def fixture_train(cfg, num_examples):
    return [_example(i, 2 + i % 5) for i in range(40)]


def fixture_eval(cfg):
    return [_example(i, 2 + i % 4) for i in range(16)]


class _Recorder(TrainerCallback):
    """Observation only: copies the LR and counters after every scheduled batch."""

    def __init__(self, trainer, rows):
        self.trainer, self.rows = trainer, rows

    def on_step_end(self, args, state, control, **kwargs):
        ins = self.trainer.instr
        self.rows.append({
            "global_step": int(state.global_step),
            "lr": float(self.trainer.optimizer.param_groups[0]["lr"]),
            "forward_calls": int(ins.forward_calls),
            "backward_calls": int(ins.backward_calls),
            "optimizer_step_attempts": int(ins.optimizer_step_attempts),
            "scheduler_step_calls": int(ins.scheduler_step_calls),
            "skipped_backward_steps": int(ins.skipped_backward_steps),
        })


def recording_trainer(base, rows):
    class RecordingTrueBackwardSkippingTrainer(base):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback(_Recorder(self, rows))

    return RecordingTrueBackwardSkippingTrainer


def hf_lr_trajectory(shape, lr, warmup, steps):
    """LR after each of `steps` scheduler steps, from the installed transformers scheduler."""
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=lr)
    sched = transformers.get_scheduler(shape, opt, num_warmup_steps=warmup, num_training_steps=steps)
    out = []
    for _ in range(steps):
        opt.step()
        sched.step()
        out.append(float(opt.param_groups[0]["lr"]))
    return out


def run_one(spec, out_dir: Path):
    run_dir = out_dir / spec["name"]
    run_dir.mkdir(parents=True)
    kw = dict(COMMON, **spec["cfg"], output_dir=str(run_dir / "trainer_output"), attempt_id=spec["name"])
    cfg = R.ExperimentConfig(**kw)
    rows = []
    base = R.TrueBackwardSkippingTrainer
    with mock.patch.object(R, "_load_model_production", side_effect=fixture_model), \
            mock.patch.object(R, "_load_tokenizer_production", side_effect=fixture_tokenizer), \
            mock.patch.object(R, "_load_train_dataset_production", side_effect=fixture_train), \
            mock.patch.object(R, "_load_eval_dataset_production", side_effect=fixture_eval), \
            mock.patch.object(R, "TrueBackwardSkippingTrainer", recording_trainer(base, rows)):
        result = R.run_execution(cfg)
    T_eff, Q = spec["expect"]["T_eff"], spec["expect"]["Q"]
    res = result.get("result") or {}
    expected_lr = hf_lr_trajectory(spec["cfg"]["lr_schedule_shape"], LR, spec["cfg"]["lr_warmup_steps"], T_eff)
    measured_lr = [r["lr"] for r in rows]
    per_batch, prev = [], {"backward_calls": 0, "optimizer_step_attempts": 0, "scheduler_step_calls": 0,
                           "forward_calls": 0, "skipped_backward_steps": 0}
    for k, r in enumerate(rows):
        d = {key: r[key] - prev[key] for key in prev}
        per_batch.append({"batch": k + 1, "lr": r["lr"],
                          "lr_expected_installed_hf": expected_lr[k] if k < len(expected_lr) else None,
                          "decision": "skip" if d["backward_calls"] == 0 else "retain",
                          "delta_F": d["forward_calls"], "delta_B": d["backward_calls"],
                          "delta_O": d["optimizer_step_attempts"], "delta_S": d["scheduler_step_calls"]})
        prev = {key: r[key] for key in prev}
    lr_diffs = [abs(a - b) for a, b in zip(measured_lr, expected_lr)]
    prov, conf = result.get("provenance") or {}, result.get("config") or {}
    nums = [res.get("final_eval_loss"), res.get("final_eval_accuracy")]
    checks = {
        "status_OK": result.get("status") == "OK",
        "config_and_provenance_dry_run_false": conf.get("dry_run") is False and prov.get("dry_run") is False,
        "provenance_execution_mode_production_loaders": prov.get("execution_mode") == "production_loaders",
        "one_record_per_scheduled_batch": len(rows) == T_eff,
        "F_equals_T_eff": res.get("forwards") == T_eff,
        "S_equals_T_eff": res.get("scheduler_advances") == T_eff,
        "B_equals_T_eff_minus_Q": res.get("backwards") == T_eff - Q,
        "O_equals_T_eff_minus_Q": res.get("optimizer_updates") == T_eff - Q,
        "skipped_equals_Q": res.get("skipped") == Q,
        "requested_and_realized_equal_Q": res.get("requested_total") == Q and res.get("realized_total") == Q,
        "every_batch_one_forward_one_scheduler_step": all(p["delta_F"] == 1 and p["delta_S"] == 1 for p in per_batch),
        "optimizer_step_only_with_backward": all(p["delta_O"] == p["delta_B"] in (0, 1) for p in per_batch),
        "lr_trajectory_equals_installed_hf_scheduler": (len(measured_lr) == len(expected_lr)
                                                        and all(x <= LR_TOL for x in lr_diffs)),
        "final_lr_equals_last_trajectory_point": bool(rows) and res.get("final_learning_rate") == rows[-1]["lr"],
        "eval_loss_and_accuracy_finite": all(isinstance(x, (int, float)) and math.isfinite(x) for x in nums)
        and 0.0 <= float(res.get("final_eval_accuracy", -1)) <= 1.0,
    }
    manifest = {
        "evidence_label": EVIDENCE_LABEL,
        "run": spec["name"],
        "expected": {"T_eff": T_eff, "Q": Q, "F": T_eff, "B": T_eff - Q, "O": T_eff - Q, "S": T_eff},
        "run_execution_output": result,
        "lr_trajectory": {"measured_after_each_batch": measured_lr, "installed_hf_scheduler": expected_lr,
                          "max_abs_difference": max(lr_diffs) if lr_diffs else None, "tolerance": LR_TOL},
        "per_batch": per_batch,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
    }
    (run_dir / "execution_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


PYTHON = [sys.executable]


def run_cli(case, out_dir: Path):
    cli_dir = out_dir / "cli"
    cli_dir.mkdir(exist_ok=True)
    cmd = [*PYTHON, str(REPO_ROOT / "scripts" / "run_controller_ablation.py"), *case["args"],
           "--output-dir", str(cli_dir / f"{case['name']}_output")]
    p = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    try:
        parsed = json.loads(p.stdout)
    except ValueError:
        parsed = None
    exp = case["expect"]
    got = {"exit": p.returncode,
           "status": (parsed or {}).get("status"),
           "execution_mode": ((parsed or {}).get("provenance") or {}).get("execution_mode")}
    rec = {"command": cmd, "cwd": str(REPO_ROOT), "exit_status": p.returncode,
           "stdout": p.stdout, "stderr": p.stderr, "expected": exp, "observed": got,
           "matches_expected": got == exp}
    (cli_dir / f"{case['name']}.json").write_text(json.dumps(rec, indent=2))
    return rec


def repo_modules_loaded():
    files = set()
    root = str(REPO_ROOT.resolve()) + os.sep
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and os.path.realpath(f).startswith(root):
            files.add(os.path.relpath(os.path.realpath(f), str(REPO_ROOT.resolve())))
    return {f: sha256(REPO_ROOT / f) for f in sorted(files) if (REPO_ROOT / f).exists()}


def environment():
    env = {"python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
           "transformers": transformers.__version__}
    for name in ("numpy", "datasets", "accelerate"):
        try:
            env[name] = __import__(name).__version__
        except Exception as e:  # recorded, not fatal
            env[name] = f"unavailable: {type(e).__name__}"
    return env


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_dir = Path(a.out).resolve()
    if out_dir.exists():
        print(f"refusing to write into existing {out_dir}", file=sys.stderr)
        return 2
    out_dir.mkdir(parents=True)
    runs = {}
    for spec in RUNS:
        try:
            m = run_one(spec, out_dir)
            runs[spec["name"]] = {"all_checks_passed": m["all_checks_passed"], "checks": m["checks"],
                                  "status": m["run_execution_output"].get("status"),
                                  "reason": m["run_execution_output"].get("reason")}
        except Exception as e:  # recorded; the summary fails
            runs[spec["name"]] = {"all_checks_passed": False, "error": f"{type(e).__name__}: {e}"}
    cli = {}
    for case in CLI_CASES:
        try:
            r = run_cli(case, out_dir)
            cli[case["name"]] = {"matches_expected": r["matches_expected"], "observed": r["observed"]}
        except Exception as e:
            cli[case["name"]] = {"matches_expected": False, "error": f"{type(e).__name__}: {e}"}
    test_files = ["tests/test_step3c_execution_contract.py", "tests/test_phase_exact_v1_acceptance.py"]
    summary = {
        "command": [sys.executable, *sys.argv],
        "cwd": os.getcwd(),
        "evidence_label": EVIDENCE_LABEL,
        "runs": runs,
        "cli_cases": cli,
        "all_checks_passed": all(r["all_checks_passed"] for r in runs.values())
        and all(c["matches_expected"] for c in cli.values()),
        "environment": environment(),
        "repo_modules_loaded_sha256": repo_modules_loaded(),
        "test_files_sha256": {f: sha256(REPO_ROOT / f) for f in test_files if (REPO_ROOT / f).exists()},
        "this_script_sha256": sha256(__file__),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({"all_checks_passed": summary["all_checks_passed"],
                      "runs": {k: v["all_checks_passed"] for k, v in runs.items()},
                      "cli_cases": {k: v["matches_expected"] for k, v in cli.items()}}, indent=2))
    return 0 if summary["all_checks_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
