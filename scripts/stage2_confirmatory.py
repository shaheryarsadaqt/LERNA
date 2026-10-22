"""Stage 2: QNLI confirmatory run at rate 0.60.

Runs the chosen score (from Stage 0) against its uniform control, plus
full_finetune and reduced_steps, over 24 seeds on QNLI.

Arms:
- gate: ScoreQuantileGatePolicy with chosen score (s1 or s2)
- control: ScoreQuantileGatePolicy with score_source="uniform"
- full_finetune: AlwaysFalsePolicy (0% skip, 6548 steps)
- reduced_steps: train 40% of steps (2619), matched backward budget

**Amendment r3 -> r4** (applied after dry-run but before any real cell):
- B17: ModernBert models produce NaN loss with sdpa attention on CUDA V100.
  Fixed by forcing `attn_implementation="eager"` in `_load_model`.
  Affects only GPU+ModernBert; bert-tiny (CPU) and Stage 0 are unaffected.

Pre-registered: experiments/PREREG_V4.md §7.

Usage:
    python scripts/stage2_confirmatory.py \
        --score s1 --seeds 11 13 17 19 23 29 31 37 41 43 47 53 59 61 67 71 73 79 83 89 97 107 109 113 \
        --rate 0.60 \
        --total-steps 6548 \
        --model jhu-clsp/ettin-encoder-150m@45d0864 \
        --output-dir /raid/stage2_confirmatory/

Seeds 157 and 163 are reserves for failed cells only.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


MIN_STEP_RATIO = 0.06
WARMUP_STEPS_QNLI = 392
ANNEAL_STEP = 5762
PROBE_REFRESH_INTERVAL = 64


def _max_skips_possible(total_steps, K=4):
    """B12: maximum number of skips possible with cap K.

    Formula: K * floor(r / (K+1)) + min(r % (K+1), K)
    where r = total_steps - min_step (skips only possible after min_step).
    """
    r = max(0, total_steps - int(total_steps * MIN_STEP_RATIO))
    return K * (r // (K + 1)) + min(r % (K + 1), K)


def _check_lerna_in_checkout():
    """B9: Verify lerna package is loaded from the repo checkout, not site-packages."""
    try:
        import lerna
        lerna_path = Path(lerna.__file__).resolve()
        repo_root = Path(__file__).resolve().parents[1]
        return str(lerna_path.parent) == str(repo_root / "lerna")
    except Exception:
        return False


def arm_needs_probe(args) -> bool:
    """B4: Does this arm need a P_s probe refresher?
    Only the gate arm with score_source='s2' needs it.
    """
    return args.score == "s2" and args.arm == "gate"


def _load_tokenizer(model_name: str, revision: str):
    """Load tokenizer, falling back to BertTokenizer for models without
    tokenizer_config.json (e.g. prajjwal1/bert-tiny)."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, revision=revision)
    except (ValueError, OSError):
        from transformers import BertTokenizer
        from huggingface_hub import hf_hub_download
        vocab_path = hf_hub_download(model_name, "vocab.txt", revision=revision)
        return BertTokenizer(vocab_path, model_max_length=512)


def _load_model(model_name: str, revision: str):
    """Load model, falling back to BertForSequenceClassification for models
    whose config.json lacks model_type (e.g. prajjwal1/bert-tiny)."""
    from transformers import AutoModelForSequenceClassification
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, revision=revision, num_labels=2, torch_dtype=torch.float32,
            attn_implementation="eager",
        )
    except (ValueError, OSError):
        import json
        from transformers import BertConfig, BertForSequenceClassification
        from huggingface_hub import hf_hub_download
        config_path = hf_hub_download(model_name, "config.json", revision=revision)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict["model_type"] = "bert"
        config = BertConfig.from_dict(cfg_dict)
        config.num_labels = 2
        return BertForSequenceClassification(config)


def _build_ps_refresher(model, trainer, probe_s_batch, probe_s_ids, device):
    """B4: Build a P_s refresher callable that computes G_P every 64 steps.

    The returned zero-arg callable returns G_P = gradient of probe loss w.r.t.
    classifier.weight, computed via _classifier_grad_from_logits on P_s.

    Refresh protocol (every 64 calls):
    - model.eval() / no_grad / fp32
    - P_s forward (triggers AlignmentProbe hook)
    - Save/restore the hook's training last_hidden around the probe forward
    - Use live Adam exp_avg_sq (consumed by alignment_score from the trainer)

    Time is charged to the gate: the refresh happens inside score computation.
    """
    from lerna.utils.forward_gate import _classifier_grad_from_logits

    state = {"G_P": None, "call_count": 0}

    def probe_fn():
        state["call_count"] += 1
        if state["G_P"] is not None and state["call_count"] % PROBE_REFRESH_INTERVAL != 0:
            return state["G_P"]

        saved_training = model.training
        model.eval()
        probe = getattr(trainer, "_alignment_probe", None)
        saved_hidden = probe.last_hidden if probe is not None else None
        if probe is not None:
            probe._last_hidden = None

        try:
            with torch.no_grad():
                out = model(**{k: v.to(device) for k, v in probe_s_batch.items()})
                logits = out.logits.detach().float()
                labels = probe_s_batch["labels"].to(device)

                if probe is None or probe.last_hidden is None:
                    return None
                hidden = probe.last_hidden.detach().float()
                G_P = _classifier_grad_from_logits(logits, labels, hidden)
        finally:
            if probe is not None:
                probe._last_hidden = saved_hidden
            model.train(saved_training)

        state["G_P"] = G_P
        return G_P

    return probe_fn
REDUCED_TOTAL_STEPS = 2619
REDUCED_WARMUP_STEPS = int(REDUCED_TOTAL_STEPS * MIN_STEP_RATIO)  # 157
FULL_STEPS = 6548
FULL_WARMUP = int(FULL_STEPS * MIN_STEP_RATIO)  # 392


def _prereg_tag_sha() -> str | None:
    """Resolve the v4-prereg-r3 tag to a commit SHA (None if unresolvable)."""
    repo_root = Path(__file__).resolve().parents[1]
    try:
        return subprocess.run(
            ["git", "rev-parse", "v4-prereg-r3^{commit}"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip() or None
    except Exception:
        return None


def _tree_is_dirty() -> bool:
    """True iff the git checkout has uncommitted changes.

    Used for the provenance block written into every results.json. The
    previous implementation returned `not _check_lerna_in_checkout()`, which
    reported whether the lerna package was importable from the checkout and
    NOT whether the tree was clean. Every results.json produced before this
    fix therefore carries a meaningless `tree_dirty` value.
    """
    repo_root = Path(__file__).resolve().parents[1]
    try:
        r = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, cwd=str(repo_root)
        )
        return bool(r.stdout.strip())
    except Exception:
        return True


def _check_provenance(probe_ids_sha: str | None = None, post_hoc: bool = False):
    """F11/B5: refuse to run unless tree is clean and HEAD == v4-prereg-r3.

    Runs BEFORE training in every mode (including single-cell). The probe
    ids_sha256 is verified here against the loaded probe data.

    ``post_hoc=True`` is the escape hatch for POST-HOC EXTENSION experiments
    (E1 Pareto, E3 MRPC/RTE) that are not covered by the V4 pre-registration.
    In that mode the tree must still be clean, but HEAD need not equal the
    pre-registration tag; the actual commit SHA is recorded instead, and the
    caller must stamp the run as post-hoc in the manifest.
    """
    repo_root = Path(__file__).resolve().parents[1]

    # Check dirty tree
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(repo_root)
    )
    if r.stdout.strip():
        print(f"FATAL: git tree is dirty. Commit or stash first.\n{r.stdout}")
        sys.exit(1)

    # Check HEAD == v4-prereg-r3
    try:
        tag_sha = subprocess.run(
            ["git", "rev-parse", "v4-prereg-r3^{commit}"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        if tag_sha != head_sha and not post_hoc:
            print(f"FATAL: HEAD ({head_sha}) != v4-prereg-r3 ({tag_sha}).")
            sys.exit(1)
    except Exception:
        print("FATAL: could not resolve v4-prereg-r3 tag.")
        sys.exit(1)

    # B9: Verify we are running the installed (or path-inserted) lerna
    # from inside the tagged checkout.
    try:
        import lerna
        lerna_file = lerna.__file__
        repo_root_str = str(repo_root)
        if not lerna_file.startswith(repo_root_str):
            print(f"FATAL: lerna imported from {lerna_file}, not inside the checkout at {repo_root_str}")
            sys.exit(1)
    except ImportError:
        print("FATAL: cannot import lerna. Run 'pip install -e .' or ensure repo root is on sys.path.")
        sys.exit(1)

    return head_sha


def _load_probe_data(probe_file: str) -> dict:
    with open(probe_file) as f:
        data = json.load(f)
    # B1: Verify ids_sha256 (content hash of probe ids), not file bytes.
    # A file cannot contain its own hash.
    ids_json = json.dumps({
        "probe_s_ids": data["probe_s_ids"],
        "probe_e_ids": data["probe_e_ids"],
    }, sort_keys=True, separators=(",", ":"))
    ids_sha = hashlib.sha256(ids_json.encode()).hexdigest()
    expected_ids_sha = data.get("ids_sha256")
    if expected_ids_sha and ids_sha != expected_ids_sha:
        print(f"FATAL: probe ids hash mismatch. Expected {expected_ids_sha}, got {ids_sha}")
        sys.exit(1)

    # B1: Assert probe set sizes and disjointness
    assert len(data["probe_s_ids"]) == 256, \
        f"FATAL: probe_s_ids must have exactly 256 entries, got {len(data['probe_s_ids'])}"
    assert len(data["probe_e_ids"]) == 256, \
        f"FATAL: probe_e_ids must have exactly 256 entries, got {len(data['probe_e_ids'])}"
    assert set(data["probe_s_ids"]).isdisjoint(data["probe_e_ids"]), \
        "FATAL: P_s and P_e probe index sets overlap"

    data["_ids_sha256"] = ids_sha
    return data


def _build_policy(arm, seed, rate, total_steps, score_kind, min_step, probe_fn=None):
    """Build the skip policy for the given arm."""
    from lerna.utils.forward_gate import (
        ScoreQuantileGatePolicy,
        AlignmentProbe,
    )
    from lerna.trainers.policies import AlwaysFalsePolicy

    if arm == "full_finetune":
        return AlwaysFalsePolicy()

    if arm in ("gate", "control"):
        score_source = score_kind if arm == "gate" else "uniform"
        pol = ScoreQuantileGatePolicy(
            target_skip_rate=rate,
            total_steps=total_steps,
            min_step=min_step,
            n_phases=4,
            max_consecutive_skips=4,
            policy_seed=seed,
            score_source=score_source,
        )

        # F8: fail hard if S2 probe_fn is missing
        if score_source == "s2":
            if probe_fn is None:
                raise ValueError(
                    f"FATAL: arm={arm} score_source='s2' but probe_fn is None. "
                    f"S2 alignment score will be NaN and the gate will be non-functional."
                )
            pol.alignment_probe_fn = probe_fn

        return pol

    # reduced_steps handled as a TrainingArguments difference, not a policy
    return AlwaysFalsePolicy()


def _resolve_reduced_steps(total_steps, reduced_fraction=None, reduced_steps=None):
    """Resolve the absolute step count for the reduced_steps arm.

    Precedence: explicit ``reduced_steps`` (absolute) > ``reduced_fraction``
    (relative) > the QNLI-specific ``REDUCED_TOTAL_STEPS`` constant.

    The absolute form exists because the relative form cannot express every
    integer budget exactly: round(0.40 * 377) = 151 but int(377 * 0.40) = 150.
    E3 (MRPC/RTE) needs the backward budget of the gate/control arms to match
    exactly, so the step count is passed explicitly rather than derived.
    """
    if reduced_steps is not None:
        return int(reduced_steps)
    if reduced_fraction is not None:
        return int(total_steps * reduced_fraction)
    return REDUCED_TOTAL_STEPS


def _build_training_args(
    arm, seed, total_steps, warmup_steps, output_dir, no_early_stopping=True,
    reduced_fraction=None, reduced_steps=None,
):
    """Build TrainingArguments for the given arm."""
    from transformers import TrainingArguments

    if arm == "reduced_steps":
        steps = _resolve_reduced_steps(total_steps, reduced_fraction, reduced_steps)
        warmup = int(steps * MIN_STEP_RATIO)
    else:
        steps = total_steps
        warmup = warmup_steps

    return TrainingArguments(
        output_dir=str(output_dir / arm / str(seed)),
        max_steps=steps,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_steps=warmup,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False, bf16=False,
        eval_strategy="no",  # B8: no mid-run eval; evaluate once at end (final checkpoint only)
        save_strategy="no",
        load_best_model_at_end=False,  # B8: final checkpoint only, no dev-set selection
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", logging_steps=100, seed=seed,
        dataloader_num_workers=0, remove_unused_columns=True,
        disable_tqdm=True,
        use_cpu=False,
    )


def run_cell(
    arm, seed, rate, total_steps, score_kind, model_id, model_rev,
    output_dir, probe_data=None, run_sha=None, probe_fn=None,
    reduced_fraction=None, reduced_steps=None, task="qnli", post_hoc=False,
):
    """Run one cell and return results dict.

    F7: reduced_steps uses max_steps=2619, warmup=157, linear decay to 0.
    F8: S2 gate fails hard if probe_fn missing or score non-finite at step 392.
    F11: Each cell writes a manifest and runs validate_skip_policy_results.
    B5: provenance checked in main() BEFORE training; run_cell receives run_sha.
    """
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
    )
    from datasets import load_dataset
    from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = _load_tokenizer(model_id, revision=model_rev)
    ds_tr = load_dataset("glue", task, split="train")
    ds_ev = load_dataset("glue", task, split="validation")

    # Task-aware tokenization: QNLI uses question/sentence, MRPC uses sentence1/sentence2.
    if task == "qnli":
        _a, _b = "question", "sentence"
    else:
        _a, _b = "sentence1", "sentence2"

    def tok_fn(x):
        out = tok(x[_a], x[_b], truncation=True,
                   max_length=512)
        out["labels"] = x["label"]
        return out

    # B11: tokenize fully, then select probe subset from unshuffled order.
    ds_tr_raw = ds_tr.map(tok_fn, remove_columns=list(ds_tr.column_names))
    ds_ev = ds_ev.map(tok_fn, remove_columns=list(ds_tr.column_names))

    # Probe exclusion only applies to qnli (the probe indices are QNLI-specific
    # and only the S2 gate arm needs P_s/P_e held out).
    if task == "qnli":
        probe_s_ids = list(probe_data["probe_s_ids"])
        probe_e_ids = list(probe_data["probe_e_ids"])
        probe_s_set = set(probe_s_ids)
        probe_e_set = set(probe_e_ids)

        # F6: exclude probe indices from training
        all_train_ids = list(range(len(ds_tr_raw)))
        filtered_ids = [i for i in all_train_ids if i not in probe_s_set and i not in probe_e_set]

        assert probe_s_set.isdisjoint(probe_e_set), "P_s and P_e overlap"
        assert not probe_s_set.intersection(filtered_ids), "P_s in train"
        assert not probe_e_set.intersection(filtered_ids), "P_e in train"

        # B4: prepare P_s batch from raw (unfiltered) dataset for refresher probe_fn
        ps_raw = ds_tr_raw.select(sorted(probe_s_ids)[:32])
        probe_s_batch = tok.pad(
            {"input_ids": ps_raw["input_ids"], "attention_mask": ps_raw["attention_mask"]},
            padding="longest", max_length=512, return_tensors="pt")
    else:
        filtered_ids = list(range(len(ds_tr_raw)))
        probe_s_batch = None

    ds_tr = ds_tr_raw.select(filtered_ids)

    model = _load_model(model_id, revision=model_rev)

    min_step = int(total_steps * MIN_STEP_RATIO) if arm not in ("reduced_steps",) else int(
        _resolve_reduced_steps(total_steps, reduced_fraction, reduced_steps)
        * MIN_STEP_RATIO
    )

    args = _build_training_args(
        arm, seed, total_steps, FULL_WARMUP, output_dir,
        reduced_fraction=reduced_fraction,
        reduced_steps=reduced_steps,
    )

    # B4: For S2 gate, build live P_s refresher probe_fn if not already provided.
    # S2 is QNLI-only (probe indices and P_s/P_e holdout are QNLI-specific).
    if score_kind == "s2" and arm == "gate":
        if probe_fn is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            probe_fn = _build_ps_refresher(model, None, probe_s_batch, sorted(probe_s_ids), device)

    policy = _build_policy(arm, seed, rate, total_steps, score_kind, min_step, probe_fn)

    def _acc(p):
        preds = np.asarray(p.predictions).argmax(-1)
        return {"accuracy": float((preds == np.asarray(p.label_ids)).mean())}

    def _collate(tokenizer):
        def coll(batch):
            to_pad = {k: [d[k] for d in batch] for k in ("input_ids", "attention_mask")}
            labels = torch.tensor([d["labels"] for d in batch])
            padded = tokenizer.pad(to_pad, padding="longest", max_length=512, return_tensors="pt")
            padded["labels"] = labels
            return padded
        return coll

    from transformers import TrainerCallback
    lr_history: list = []

    class _LRCallback(TrainerCallback):
        def __init__(self):
            self._trainer = None

        def on_step_end(self, args, state, control, **kwargs):
            # transformers 5.x calls on_step_end(args, state, control) without
            # a trainer kwarg; use the bound trainer reference instead.
            trainer = self._trainer if self._trainer is not None else kwargs.get("trainer")
            opt = getattr(trainer, "optimizer", None) if trainer else None
            if opt is not None:
                lr_history.append(float(opt.param_groups[0]["lr"]))

    _lr_cb = _LRCallback()

    t = TrueBackwardSkippingTrainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_ev,
        data_collator=_collate(tok),
        compute_metrics=_acc,
        skip_policy=policy,
        scheduler_step_policy="always_step",
        capture_logits=True,
        online_ler_mode="legacy_dense",  # B3: logit capture + legacy dense mode
        instrumentation_path=str(output_dir / arm / str(seed) / "instrumentation.json"),
        callbacks=[_lr_cb],
    )
    _lr_cb._trainer = t

    # B3: Attach AlignmentProbe to model before train() for S2 score
    if arm == "gate" and score_kind == "s2":
        policy.setup_probe(t, model)

    # F8: for S2, verify the probe is attached and score is non-finite at step 392
    if arm == "gate" and score_kind == "s2":
        original_should_skip = policy.should_skip

        def checking_should_skip(trainer, model, inputs):
            instr = getattr(trainer, "instr", None)
            batches_seen = getattr(instr, "batches_seen", 0) if instr else 0
            if batches_seen - 1 == min_step:
                probe = getattr(trainer, "_alignment_probe", None)
                assert probe is not None, \
                    "FATAL: S2 arm but AlignmentProbe not attached to trainer"
                assert probe.last_hidden is not None, \
                    "FATAL: S2 arm but AlignmentProbe.last_hidden is None at step 392"
                from lerna.utils.forward_gate import alignment_score
                s = alignment_score(trainer, model, inputs, probe_fn)
                assert math.isfinite(s), \
                    f"FATAL: S2 score is non-finite at step 392 ({s})"
            return original_should_skip(trainer, model, inputs)

        policy.should_skip = checking_should_skip

    t0 = time.perf_counter()
    t.train()
    wall = time.perf_counter() - t0

    # F7: assert final LR <= 1e-8 for the primary test (always_step contract)
    final_lr = float(t.optimizer.param_groups[0]["lr"])
    assert final_lr <= 1e-8, \
        f"FATAL: final LR={final_lr} > 1e-8 — scheduler step contract violated"

    scheduler_calls = getattr(getattr(t, "instr", None), "scheduler_step_calls", 0)
    batches_seen = getattr(getattr(t, "instr", None), "batches_seen", 0)

    # Assert scheduler steps == batches seen (always_step contract)
    expected_steps = (_resolve_reduced_steps(total_steps, reduced_fraction, reduced_steps)
                      if arm == "reduced_steps" else FULL_STEPS)
    assert scheduler_calls == batches_seen == expected_steps, \
        f"FATAL: scheduler_step_calls={scheduler_calls} != batches_seen={batches_seen} != expected={expected_steps}"

    metrics = t.evaluate()

    instr = getattr(t, "instr", None)

    # B2: capture policy diagnostics from the gate/control policy
    pol_diag = {}
    if hasattr(t.skip_policy, "get_diagnostics"):
        pol_diag = t.skip_policy.get_diagnostics()

    # B2: restructured results with required sections
    result = {
        "run_config": {
            "arm": arm,
            "seed": seed,
            "rate": rate,
            "total_steps": total_steps,
            "actual_steps_run": expected_steps,
            "model_id": model_id,
            "model_rev": model_rev,
            "score_kind": score_kind,
            "min_step": min_step,
            "skip_update_mode": "freeze",
            "scheduler_step_policy": "always_step",
        },
        "policy_diagnostics": pol_diag,
        "true_skip_instrumentation": {
            "batches_seen": batches_seen,
            "scheduler_step_calls": scheduler_calls,
            "optimizer_step_attempts": getattr(instr, "optimizer_step_attempts", 0) if instr else 0,
            "skipped_backward_steps": getattr(instr, "skipped_backward_steps", 0) if instr else 0,
            "skipped_batches": getattr(instr, "skipped_batches", 0) if instr else 0,
            "forward_calls": getattr(instr, "forward_calls", 0) if instr else 0,
            "backward_calls": getattr(instr, "backward_calls", 0) if instr else 0,
            "skip_ratio_by_batch": getattr(instr, "skip_ratio_by_batch", None) if instr else None,
            "scheduler_step_policy": getattr(instr, "scheduler_step_policy", "always_step") if instr else "always_step",
            "skip_update_mode": getattr(instr, "skip_update_mode", "freeze") if instr else "freeze",
        },
        "eval_metrics": {
            "eval_accuracy": metrics.get("eval_accuracy"),
            "eval_loss": metrics.get("eval_loss"),
        },
        "provenance": {
            "v4_prereg_sha": run_sha,
            "probe_ids_sha256": probe_data.get("ids_sha256") if probe_data else None,
            "probe_file_sha256": probe_data.get("sha256") if probe_data else None,
            "codebase_checkout": _check_lerna_in_checkout(),
            "tree_dirty": _tree_is_dirty(),
            "post_hoc": bool(post_hoc),
            "prereg_tag_sha": _prereg_tag_sha(),
        },
        "lr_history": list(lr_history),
        "final_lr": final_lr,
        "wall_seconds": wall,
        "ms_per_step": wall / expected_steps * 1000,
    }

    # F11: write manifest + run validate_skip_policy_results.py --check-v4-gate
    cell_output = Path(str(output_dir)) / arm / str(seed)
    os.makedirs(str(cell_output), exist_ok=True)

    manifest = {
        "arm": arm,
        "seed": seed,
        "v4_prereg_sha": run_sha,
        "probe_ids_sha256": probe_data.get("ids_sha256") if probe_data else None,
        "run_config": result["run_config"],
        "policy_diagnostics": result["policy_diagnostics"],
        "provenance": result["provenance"],
    }
    with open(cell_output / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    results_path = cell_output / "results.json"
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    if run_sha:
        repo_root = Path(__file__).resolve().parents[1]
        validator = str(repo_root / "scripts" / "validate_skip_policy_results.py")
        val = subprocess.run(
            [sys.executable, validator, str(results_path),
             "--expected-sha", run_sha, "--check-v4-gate"],
            capture_output=True, text=True, cwd=str(repo_root),
        )
        result["validation_passed"] = (val.returncode == 0)
        result["validation_stderr"] = val.stderr[:500] if val.returncode != 0 else None
        if val.returncode != 0:
            print(f"WARNING: validation failed for arm={arm} seed={seed}:\n{val.stderr[:500]}")
            print(f"  Cell will be REFUSED in final analysis (validation_passed=False)")

    return result


def main():
    parser = argparse.ArgumentParser(description="Stage 2: QNLI confirmatory at rate 0.60")
    parser.add_argument("--score", choices=["s1", "s2"], default="s1",
                        help="Score chosen by Stage 0")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="24 Stage 2 seeds")
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--model", type=str, default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--model-rev", type=str, default="45d08642849e5c5701b162671ac811b7654bfd9f")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--reserves", type=int, nargs="+", default=[157, 163])
    parser.add_argument("--probe-file", type=str,
                        default="experiments/probe_indices_v4.json")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index for single-cell runs (launcher sets CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--single-cell", action="store_true",
                        help="Run a single cell (for launcher dispatch)")
    parser.add_argument("--arm", type=str, choices=["gate", "control", "full_finetune", "reduced_steps"])
    parser.add_argument("--single-seed", type=int, help="Seed for single-cell mode")
    parser.add_argument("--reduced-fraction", type=float, default=None,
                        help="E1 override: fraction of total_steps for the reduced_steps "
                             "arm (e.g. 0.20, 0.80). Defaults to REDUCED_TOTAL_STEPS/total_steps.")
    parser.add_argument("--reduced-steps", type=int, default=None,
                        help="Absolute step count for the reduced_steps arm. Takes "
                             "precedence over --reduced-fraction. Used to match the "
                             "backward budget of the gate/control arms exactly "
                             "(E3: MRPC 151, RTE 103).")
    parser.add_argument("--task", type=str, default="qnli",
                        help="GLUE task name (qnli for Stage 2 confirmatory; mrpc for E3).")
    parser.add_argument("--post-hoc", action="store_true",
                        help="Mark the run as a POST-HOC EXTENSION (E1/E3) that is not "
                             "covered by the v4-prereg-r3 pre-registration. Requires a "
                             "clean tree; records the actual HEAD sha as provenance.")
    args = parser.parse_args()

    # B9: Ensure the repo root is on sys.path so `import lerna` works when
    # this script is run as `python scripts/stage2_confirmatory.py`.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # B5: provenance check runs BEFORE training in every mode (single-cell and full).
    probe_data = _load_probe_data(str(repo_root / "experiments" / "probe_indices_v4.json"))
    run_sha = _check_provenance(probe_ids_sha=probe_data.get("ids_sha256"),
                                post_hoc=args.post_hoc)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    # Single-cell mode (for launcher)
    if args.single_cell:
        assert args.arm and args.single_seed, "Need --arm and --single-seed in single-cell mode"
        r = run_cell(
            args.arm, args.single_seed, args.rate, args.total_steps,
            args.score, args.model, args.model_rev,
            Path(args.output_dir),
            probe_data=probe_data, run_sha=run_sha,
            reduced_fraction=args.reduced_fraction,
            reduced_steps=args.reduced_steps,
            task=args.task,
            post_hoc=args.post_hoc,
        )
        out_path = Path(args.output_dir) / args.arm / str(args.single_seed)
        out_path.mkdir(parents=True, exist_ok=True)
        print(json.dumps(r, indent=2))
        return

    # Full run mode
    assert len(args.seeds) == 24, f"Expected 24 seeds, got {len(args.seeds)}"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    probe_data = _load_probe_data(args.probe_file)
    probe_sha = hashlib.sha256(open(args.probe_file, "rb").read()).hexdigest()

    arms = ["gate", "control", "full_finetune", "reduced_steps"]

    # F9: reserve seed replacement replaces in ALL arms, not just the failing one
    # B7: re-run finished arms with the reserve seed when a swap occurs
    reserve_seeds = list(args.reserves)
    crash_log = []

    # Build master task list: one entry per (arm, seed)
    task_list: list[tuple[str, int]] = []
    for arm in arms:
        for seed in args.seeds:
            task_list.append((arm, seed))

    # Track seed lineage: original_seed -> replacement_seed (for B7 re-runs)
    seed_map: dict[int, int] = {}
    # Results indexed by (arm, effective_seed)
    results_by_key: dict[tuple[str, int], dict] = {}

    while task_list:
        arm, seed = task_list.pop(0)
        # Resolve through seed_map (in case this seed was itself a replacement)
        effective_seed = seed
        while effective_seed in seed_map:
            effective_seed = seed_map[effective_seed]

        print(f"\n=== arm={arm} seed={effective_seed} (rate={args.rate}) ===")

        attempt = 0
        success = False
        while attempt < 2 and not success:
            try:
                r = run_cell(
                    arm, effective_seed, args.rate, args.total_steps, args.score,
                    args.model, args.model_rev, output,
                    probe_data=probe_data, run_sha=run_sha,
                    reduced_fraction=args.reduced_fraction,
                    reduced_steps=args.reduced_steps,
                    task=args.task,
                    post_hoc=args.post_hoc,
                )
                r["attempts"] = attempt + 1
                r["crashed"] = False
                results_by_key[(arm, effective_seed)] = r
                success = True
            except Exception as e:
                print(f"  FAILED (attempt {attempt + 1}): {e}")
                if attempt == 0:
                    print(f"  Retrying with same seed={effective_seed}")
                    attempt += 1
                elif reserve_seeds:
                    new_seed = reserve_seeds.pop(0)
                    crash_log.append({
                        "arm": arm, "orig_seed": seed,
                        "replacement_seed": new_seed,
                        "effective_seed": effective_seed,
                        "error": str(e),
                        "attempts": 2,
                    })
                    seed_map[effective_seed] = new_seed
                    print(f"  Replacing seed {effective_seed} with reserve {new_seed} in ALL arms")
                    # B7: remove ALL results for the old effective_seed and re-queue all arms
                    for a in arms:
                        old_key = (a, effective_seed)
                        if old_key in results_by_key:
                            del results_by_key[old_key]
                        # Re-add to task list if not already there
                        if (a, effective_seed) not in task_list and (a, effective_seed) not in [(t[0], t[1]) for t in task_list]:
                            task_list.insert(0, (a, effective_seed))
                    # Re-queue this arm with the new seed
                    task_list.insert(0, (arm, new_seed))
                    # Reset attempt counter so the new task starts fresh
                    success = True  # exit the while loop; will re-enter from task_list
                    # But we haven't successfully completed this arm yet
                    # Remove any partial result
                    results_by_key.pop((arm, effective_seed), None)
                else:
                    crash_log.append({
                        "arm": arm, "orig_seed": seed,
                        "replacement_seed": None,
                        "effective_seed": effective_seed,
                        "error": str(e),
                        "attempts": 2,
                    })
                    success = True  # give up on this cell

    # Flatten results
    results = list(results_by_key.values())

    # Write summary
    cells = {}
    for r in results:
        cfg = r.get("run_config", {})
        key = (cfg.get("arm", r.get("arm")), cfg.get("seed", r.get("seed")))
        cells[key] = r

    # F11: analysis refuses cells that failed validation
    validated_results = [r for r in results if r.get("validation_passed") is True]
    failed_validation = [r for r in results if r.get("validation_passed") is not True]
    if failed_validation:
        print(f"\nWARNING: {len(failed_validation)} cells failed validation and are REFUSED:")
        for r in failed_validation:
            cfg = r.get("run_config", {})
            print(f"  arm={cfg.get('arm')} seed={cfg.get('seed')}: {r.get('validation_stderr', 'unknown')[:200]}")

    # Paired analysis: gate vs control (primary) — only validated cells
    gate_by_seed = {}
    ctrl_by_seed = {}
    for r in validated_results:
        cfg = r.get("run_config", {})
        em = r.get("eval_metrics", {})
        arm = cfg.get("arm")
        seed = cfg.get("seed")
        acc = em.get("eval_accuracy")
        if arm == "gate":
            gate_by_seed[seed] = acc
        elif arm == "control":
            ctrl_by_seed[seed] = acc

    paired = []
    for s in sorted(gate_by_seed.keys() & ctrl_by_seed.keys()):
        paired.append(gate_by_seed[s] - ctrl_by_seed[s])

    if paired:
        diff = np.array(paired)
        from scipy.stats import t as t_dist
        n = len(diff)
        df = n - 1
        mean_diff = float(diff.mean())
        sd = float(diff.std(ddof=1))
        se = sd / math.sqrt(n)
        t_stat = mean_diff / se if se > 0 else float("inf")
        p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df)) if se > 0 else 0.0

        # Bootstrap CI
        rng = np.random.RandomState(42)
        n_boot = 10000
        boot_means = [float(rng.choice(diff, size=n, replace=True).mean()) for _ in range(n_boot)]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        primary = {
            "mean_delta": mean_diff,
            "sd": sd,
            "t_stat": t_stat,
            "p_val": p_val,
            "n": n,
            "df": df,
            "ci_95": [ci_lo, ci_hi],
            "passed": p_val < 0.05,
        }
    else:
        primary = {"error": "no paired results"}

    summary = {
        "stage": "stage2_confirmatory",
        "score_source": args.score,
        "rate": args.rate,
        "total_steps": args.total_steps,
        "seeds": args.seeds,
        "reserves_used": [c["replacement_seed"] for c in crash_log],
        "arms": arms,
        "probe_sha256": probe_sha,
        "run_sha": run_sha,
        "cell_results": results,
        "primary_test": primary,
        "crash_log": crash_log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Stage 2 Summary ===")
    print(f"  Primary: Δacc(gate - control) = {primary.get('mean_delta', 'N/A')}")
    print(f"  t({primary.get('df', '?')}) = {primary.get('t_stat', 'N/A')}")
    print(f"  p = {primary.get('p_val', 'N/A')}")
    print(f"  95% CI: {primary.get('ci_95', [None, None])}")
    print(f"  Written to: {output / 'stage2_summary.json'}")


if __name__ == "__main__":
    main()
