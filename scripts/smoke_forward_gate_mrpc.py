"""LER v4 smoke test on MRPC (200 steps), both arms.

GPU only. Output to /raid (NOT /new_raid, which is full).
Goal: verify the gate and its uniform control share identical per-window skip
machinery and identical LR curves under an always_step schedule, and report the
gate's ms/step overhead. MECHANICS check, not a result cell.

    python scripts/smoke_forward_gate_mrpc.py
"""

import os
os.environ.setdefault("HF_HOME", "/raid/hf_cache")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCH_LOGS", "+0")

import sys, json, time
from pathlib import Path
import torch
if hasattr(torch, "_dynamo"):
    torch._dynamo.config.disable = True
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, TrainerCallback,
)
from datasets import load_dataset

sys.path.insert(0, "/home/nanhangproj/LERNA")
from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer
from lerna.utils.forward_gate import ScoreQuantileGatePolicy

MODEL_ID = "jhu-clsp/ettin-encoder-150m"
MODEL_REV = "45d08642849e5c5701b162671ac811b7654bfd9f"
V4_PREREG_SHA = "2d20e935139dc97a0d962e576e0f659e6861ecbc"
TOTAL_STEPS = 200
RATE = 0.45
SEED = 11
OUT = Path("/raid/smoke_forward_gate_mrpc")
OUT.mkdir(parents=True, exist_ok=True)


class _Diag:
    def __init__(self, name):
        self.name = name
        self.lr = []
        self.policy_diag = None
        self.skipped_batches = 0
        self.forward_calls = 0


class _LRCallback(TrainerCallback):
    def __init__(self, outer):
        self.o = outer
    def on_step_end(self, args, state, control, **kwargs):
        try:
            lr = float(self.o_trainer.optimizer.param_groups[0]["lr"]) if self.o_trainer else None
        except Exception:
            lr = None
        if lr is not None:
            self.o.lr.append(lr)
    def on_train_end(self, args, state, control, **kwargs):
        try:
            instr = getattr(self.o_trainer, "instr", None)
            if instr is not None:
                self.o.scheduler_step_calls = getattr(instr, "scheduler_step_calls", 0)
                self.o.batches_seen = getattr(instr, "batches_seen", 0)
                self.o.optimizer_step_attempts = getattr(instr, "optimizer_step_attempts", 0)
                self.o.skipped_backward = getattr(instr, "skipped_backward_steps", 0)
            self.o.policy_diag = self.o_trainer.skip_policy.get_diagnostics()
            self.o.skipped_batches = getattr(self.o_trainer.instr, "skipped_batches", 0)
            self.o.forward_calls = getattr(self.o_trainer.instr, "forward_calls", 0)
        except Exception as e:
            import traceback; traceback.print_exc()
            self.o.policy_diag = None


def _collate(tokenizer):
    import torch as _T
    def coll(batch):
        return {k: _T.tensor([d[k] for d in batch]) for k in
                ("input_ids", "attention_mask")} | {"labels": _T.tensor([d["label"] for d in batch])}
    return coll


def make_trainer(arm):
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REV)
    ds_tr = load_dataset("glue", "mrpc", split="train")
    ds_ev = load_dataset("glue", "mrpc", split="validation")

    def tok_fn(x):
        out = tok(x["sentence1"], x["sentence2"], truncation=True,
                  padding="max_length", max_length=128)
        out["label"] = x["label"]
        return out

    ds_tr = ds_tr.map(tok_fn, remove_columns=list(ds_tr.column_names))
    ds_ev = ds_ev.map(tok_fn, remove_columns=list(ds_ev.column_names))

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, revision=MODEL_REV, num_labels=2)
    args = TrainingArguments(
        output_dir=str(OUT / arm),
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_steps=12,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False, bf16=False,
        eval_strategy="no", save_strategy="no",
        report_to="none", logging_steps=625, seed=SEED,
        dataloader_num_workers=0, remove_unused_columns=False,
        disable_tqdm=True,
    )
    pol = ScoreQuantileGatePolicy(
        target_skip_rate=RATE, total_steps=TOTAL_STEPS, min_step=50,
        n_phases=4, max_consecutive_skips=4, policy_seed=SEED,
        score_source="gain" if arm == "gate" else "uniform",
    )
    diag = _Diag(arm)
    cb = _LRCallback(diag); cb.o_trainer = None  # set after trainer built

    def _acc(p):
        import numpy as np
        preds = np.asarray(p.predictions).argmax(-1)
        return {"accuracy": float((preds == np.asarray(p.label_ids)).mean())}

    t = TrueBackwardSkippingTrainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_ev,
        data_collator=_collate(tok),
        compute_metrics=_acc,
        skip_policy=pol,
        scheduler_step_policy="always_step",
        capture_logits=True, online_ler_mode="legacy_dense",
        instrumentation_path=str(OUT / arm / "instrumentation.json"),
        callbacks=[cb],
    )
    cb.o_trainer = t
    diag._trainer = t
    return t, diag


def run(arm):
    t, diag = make_trainer(arm)
    torch.cuda.empty_cache()
    t0 = time.perf_counter()
    t.train()
    wall = time.perf_counter() - t0
    metrics = t.evaluate()
    rep = {
        "arm": arm, "rate": RATE, "total_steps": TOTAL_STEPS, "seed": SEED,
        "warmup_steps": 12,
        "lr_curve": diag.lr,
        "policy_diagnostics": diag.policy_diag,
        "scheduler_step_calls": getattr(diag, "scheduler_step_calls", 0),
        "batches_seen": getattr(diag, "batches_seen", 0),
        "skipped_batches": diag.skipped_batches,
        "forward_calls": diag.forward_calls,
        "wall_seconds": wall,
        "ms_per_step": wall / TOTAL_STEPS * 1000,
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_loss": metrics.get("eval_loss"),
    }
    (OUT / arm).mkdir(parents=True, exist_ok=True)
    json.dump(rep, open(OUT / arm / "smoke_report.json", "w"), indent=1)
    pd = diag.policy_diag
    acc = rep['eval_accuracy']
    acc_s = f"{acc:.4f}" if acc is not None else "N/A"
    print(f"[{arm}] acc={acc_s} ms/step={rep['ms_per_step']:.1f} "
          f"skipped={diag.skipped_batches} phases={pd['phase_used'] if pd else None} "
          f"quota={pd['phase_quota'] if pd else None} "
          f"realized={pd.get('realized_total') if pd else None} "
          f"requested={pd.get('requested_total') if pd else None}")
    return rep


def _git_sha() -> str:
    import subprocess
    try:
        h = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
        return h.stdout.strip()
    except Exception:
        return "unknown"


def _tree_clean() -> bool:
    import subprocess
    try:
        r = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]))
        return r.stdout.strip() == ""
    except Exception:
        return False


def main():
    r_gate = run("gate")
    r_ctrl = run("control")

    pg = r_gate["policy_diagnostics"]; pc = r_ctrl["policy_diagnostics"]
    # Budget invariant: realized total must equal the RandomSkipPolicy-style quota
    # round(rate * total_steps), realized over eligible steps only.
    for name, pd in (("gate", pg), ("control", pc)):
        assert pd["realized_total"] == pd["requested_total"], \
            f"{name}: realized_total={pd['realized_total']} != requested_total={pd['requested_total']} " \
            f"(per-window used={pd['phase_used']} quota={pd['phase_quota']})"
        assert sum(pd["phase_used"]) == pd["requested_total"], \
            f"{name}: phase_used sum {sum(pd['phase_used'])} != requested {pd['requested_total']}"

    q = pc["phase_quota"]; elig = pc["eligible_per_window"]
    ug = pg["phase_used"]; uc = pc["phase_used"]
    assert list(ug) == list(uc), \
        f"gate vs control per-window skip counts must match: gate={ug} ctrl={uc}"
    # Eligible windows hit quota exactly; 0-eligible windows are 0 (quota transferred).
    for i, (u, qt, e) in enumerate(zip(ug, q, elig)):
        if e > 0:
            assert u == qt, f"phase {i} used {u} != quota {qt} (eligible {e})"
        else:
            assert u == 0, f"phase {i} used {u} != 0 (0-eligible window must transfer quota)"

    assert r_gate["scheduler_step_calls"] == r_gate["batches_seen"], \
        f"always_step contract violated (gate): sched={r_gate['scheduler_step_calls']} batches={r_gate['batches_seen']}"
    assert r_ctrl["scheduler_step_calls"] == r_ctrl["batches_seen"], \
        f"always_step contract violated (control): sched={r_ctrl['scheduler_step_calls']} batches={r_ctrl['batches_seen']}"

    lrg = r_gate["lr_curve"]; lrc = r_ctrl["lr_curve"]
    max_diff = max(abs(a - b) for a, b in zip(lrg, lrc)) if lrg and lrc else None
    assert max_diff is not None and max_diff < 1e-9, \
        f"LR curves must be identical under always_step; max|diff|={max_diff}"

    overhead = r_gate["ms_per_step"] - r_ctrl["ms_per_step"]
    print("\nSMOKE PASS")
    print(f"  per-window skip counts (both arms identical): {list(ug)}")
    print(f"  quota (phase 0 is min_step-gated to 0): {list(q)}")
    print(f"  LR curves identical: max|diff|={max_diff:.2e}")
    print(f"  gate ms/step={r_gate['ms_per_step']:.1f} control={r_ctrl['ms_per_step']:.1f} "
          f"overhead={overhead:+.1f} ms/step")

    # Provenance: the smoke is a new, additive Phase 3 artifact. It does NOT
    # modify the frozen V3 QNLI block (SHA 93ec6e4) — the forward gate lives in
    # the V4 code commit. The manifest records both the V3 retrospective SHA
    # (for the frozen block) and the V4 code SHA (for the forward gate artifacts),
    # matching the dual-SHA registry in experiments/PREREG_V4.md. The validator
    # checks provenance.v3_retrospective_sha or v4_prereg_sha against --expected-sha.
    sha = _git_sha()
    summary = {
        "gate": r_gate, "control": r_ctrl,
        "max_lr_diff": max_diff, "overhead_ms_per_step": overhead,
             "provenance": {
            "v3_retrospective_sha": "93ec6e488a6d74f63d021d0f517cf81fdd892801",
            "v4_prereg_sha": V4_PREREG_SHA,
            "resolved_code_sha": sha,
            "tree_dirty": not _tree_clean(),
            "scheduler_step_policy": "always_step",
            "rate": RATE, "total_steps": TOTAL_STEPS, "warmup_steps": 12,
            "seed": SEED, "min_step": 50, "max_consecutive_skips": 4, "n_phases": 4,
            "model": MODEL_ID, "model_rev": MODEL_REV,
            "arms": {"gate": "gain", "control": "uniform"},
            "note": "additive Phase 3 artifact; V3 block SHA preserved; fp32 (no AMP) per contract-test path",
        },
    }
    json.dump(summary, open(OUT / "smoke_summary.json", "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
