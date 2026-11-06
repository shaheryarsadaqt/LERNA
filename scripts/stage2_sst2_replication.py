"""SST-2 replication for LERNA v4.

Runs gate vs control at rate 0.60 on SST-2, 10 seeds.

Pre-registered: experiments/PREREG_V4.md §10.

Usage:
    python scripts/stage2_sst2_replication.py \
        --score s1 \
        --seeds 101 103 107 109 113 127 131 137 139 149 \
        --rate 0.60 \
        --total-steps 4210 \
        --model jhu-clsp/ettin-encoder-150m \
        --model-rev 45d0864 \
        --output-dir /raid/sst2_replication/
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
ANNEAL_RATIO = 0.88
TOTAL_STEPS = 4210
REDUCED_TOTAL_STEPS = 1684
REDUCED_WARMUP_STEPS = int(REDUCED_TOTAL_STEPS * MIN_STEP_RATIO)  # 101
PROBE_REFRESH_INTERVAL = 64


def _max_skips_possible(total_steps, K=4):
    r = max(0, total_steps - int(total_steps * MIN_STEP_RATIO))
    return K * (r // (K + 1)) + min(r % (K + 1), K)


def _check_lerna_in_checkout():
    try:
        import lerna
        lerna_path = Path(lerna.__file__).resolve()
        repo_root = Path(__file__).resolve().parents[1]
        return str(lerna_path.parent) == str(repo_root / "lerna")
    except Exception:
        return False


def _check_provenance(probe_ids_sha: str | None = None):
    repo_root = Path(__file__).resolve().parents[1]

    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(repo_root)
    )
    if r.stdout.strip():
        print(f"FATAL: git tree is dirty.\n{r.stdout}")
        sys.exit(1)

    try:
        tag_sha = subprocess.run(
            ["git", "rev-parse", "v4-prereg-r3^{commit}"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        if tag_sha != head_sha:
            print(f"FATAL: HEAD ({head_sha}) != v4-prereg-r3 ({tag_sha}).")
            sys.exit(1)
    except Exception:
        print("FATAL: could not resolve v4-prereg-r3 tag.")
        sys.exit(1)

    try:
        import lerna
        lerna_file = lerna.__file__
        repo_root_str = str(repo_root)
        if not lerna_file.startswith(repo_root_str):
            print(f"FATAL: lerna imported from {lerna_file}, not inside checkout.")
            sys.exit(1)
    except ImportError:
        print("FATAL: cannot import lerna.")
        sys.exit(1)

    return head_sha


def _load_probe_data(probe_file: str) -> dict:
    with open(probe_file) as f:
        data = json.load(f)
    ids_json = json.dumps({
        "probe_s_ids": data["probe_s_ids"],
        "probe_e_ids": data["probe_e_ids"],
    }, sort_keys=True, separators=(",", ":"))
    ids_sha = hashlib.sha256(ids_json.encode()).hexdigest()
    expected_ids_sha = data.get("ids_sha256")
    if expected_ids_sha and ids_sha != expected_ids_sha:
        print(f"FATAL: probe ids hash mismatch.")
        sys.exit(1)
    assert len(data["probe_s_ids"]) == 256
    assert len(data["probe_e_ids"]) == 256
    assert set(data["probe_s_ids"]).isdisjoint(data["probe_e_ids"])
    data["_ids_sha256"] = ids_sha
    return data


def _load_tokenizer(model_name: str, revision: str):
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(model_name, revision=revision)
    except (ValueError, OSError):
        from transformers import BertTokenizer
        from huggingface_hub import hf_hub_download
        vocab_path = hf_hub_download(model_name, "vocab.txt", revision=revision)
        return BertTokenizer(vocab_path, model_max_length=512)


def _load_model(model_name: str, revision: str):
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


def _build_policy(arm, seed, rate, total_steps, score_kind, min_step):
    from lerna.utils.forward_gate import ScoreQuantileGatePolicy
    from lerna.trainers.policies import AlwaysFalsePolicy

    if arm in ("full_finetune",):
        return AlwaysFalsePolicy()

    if arm in ("gate", "control"):
        score_source = score_kind if arm == "gate" else "uniform"
        return ScoreQuantileGatePolicy(
            target_skip_rate=rate,
            total_steps=total_steps,
            min_step=min_step,
            n_phases=4,
            max_consecutive_skips=4,
            policy_seed=seed,
            score_source=score_source,
        )

    return AlwaysFalsePolicy()


def _build_training_args(arm, seed, total_steps, warmup_steps, output_dir):
    from transformers import TrainingArguments

    if arm == "reduced_steps":
        steps = REDUCED_TOTAL_STEPS
        warmup = REDUCED_WARMUP_STEPS
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
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        report_to="none", logging_steps=100, seed=seed,
        dataloader_num_workers=0, remove_unused_columns=True,
        disable_tqdm=True,
        use_cpu=False,
    )


def run_cell(arm, seed, rate, total_steps, score_kind, model_id, model_rev,
             output_dir, probe_data=None, run_sha=None):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from datasets import load_dataset
    from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer
    from lerna.utils.forward_gate import ScoreQuantileGatePolicy
    from lerna.trainers.policies import AlwaysFalsePolicy

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = _load_tokenizer(model_id, revision=model_rev)
    ds_tr = load_dataset("glue", "sst2", split="train")
    ds_ev = load_dataset("glue", "sst2", split="validation")

    def tok_fn(x):
        out = tok(x["sentence"], truncation=True, max_length=512)
        out["labels"] = x["label"]
        return out

    ds_tr_raw = ds_tr.map(tok_fn, remove_columns=list(ds_tr.column_names))
    ds_ev = ds_ev.map(tok_fn, remove_columns=list(ds_ev.column_names))

    probe_s_ids = list(probe_data["probe_s_ids"])
    probe_e_ids = list(probe_data["probe_e_ids"])
    probe_s_set = set(probe_s_ids)
    probe_e_set = set(probe_e_ids)

    all_train_ids = list(range(len(ds_tr_raw)))
    filtered_ids = [i for i in all_train_ids if i not in probe_s_set and i not in probe_e_set]

    assert probe_s_set.isdisjoint(probe_e_set)
    assert not probe_s_set.intersection(filtered_ids)
    assert not probe_e_set.intersection(filtered_ids)

    ps_raw = ds_tr_raw.select(sorted(probe_s_ids)[:32])
    probe_s_batch = tok.pad(
        {"input_ids": ps_raw["input_ids"], "attention_mask": ps_raw["attention_mask"]},
        padding="longest", max_length=512, return_tensors="pt")

    ds_tr = ds_tr_raw.select(filtered_ids)
    model = _load_model(model_id, revision=model_rev)

    min_step = int(total_steps * MIN_STEP_RATIO) if arm != "reduced_steps" else REDUCED_WARMUP_STEPS

    args = _build_training_args(arm, seed, total_steps, min_step, output_dir)

    policy = _build_policy(arm, seed, rate, total_steps, score_kind, min_step)

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
    lr_history = []

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
        online_ler_mode="legacy_dense",
        instrumentation_path=str(output_dir / arm / str(seed) / "instrumentation.json"),
        callbacks=[_lr_cb],
    )
    _lr_cb._trainer = t

    t0 = time.perf_counter()
    t.train()
    wall = time.perf_counter() - t0

    final_lr = float(t.optimizer.param_groups[0]["lr"])
    assert final_lr <= 1e-8, f"FATAL: final LR={final_lr} > 1e-8"

    instr = getattr(t, "instr", None)
    scheduler_calls = getattr(instr, "scheduler_step_calls", 0) if instr else 0
    batches_seen = getattr(instr, "batches_seen", 0) if instr else 0

    expected_steps = REDUCED_TOTAL_STEPS if arm == "reduced_steps" else TOTAL_STEPS
    assert scheduler_calls == batches_seen == expected_steps, \
        f"FATAL: scheduler_step_calls={scheduler_calls} != batches_seen={batches_seen} != expected={expected_steps}"

    metrics = t.evaluate()
    pol_diag = {}
    if hasattr(t.skip_policy, "get_diagnostics"):
        pol_diag = t.skip_policy.get_diagnostics()

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
            "tree_dirty": not _check_lerna_in_checkout(),
        },
        "final_lr": final_lr,
        "wall_seconds": wall,
        "ms_per_step": wall / expected_steps * 1000,
    }

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
             "--expected-sha", run_sha],
            capture_output=True, text=True, cwd=str(repo_root),
        )
        result["validation_passed"] = (val.returncode == 0)
        result["validation_stderr"] = val.stderr[:500] if val.returncode != 0 else None
        if val.returncode != 0:
            print(f"WARNING: validation failed for arm={arm} seed={seed}: {val.stderr[:200]}")

    return result


def main():
    parser = argparse.ArgumentParser(description="SST-2 replication for LERNA v4")
    parser.add_argument("--score", choices=["s1", "s2"], default="s1")
    parser.add_argument("--seeds", type=int, nargs="+", required=True,
                        help="10 SST-2 seeds")
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=4210)
    parser.add_argument("--model", type=str, default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--model-rev", type=str, default="45d08642849e5c5701b162671ac811b7654bfd9f")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--probe-file", type=str, default="experiments/probe_indices_v4.json")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--single-cell", action="store_true")
    parser.add_argument("--arm", type=str, choices=["gate", "control", "full_finetune", "reduced_steps"])
    parser.add_argument("--single-seed", type=int)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    probe_data = _load_probe_data(str(repo_root / args.probe_file))
    run_sha = _check_provenance(probe_ids_sha=probe_data.get("ids_sha256"))

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    if args.single_cell:
        assert args.arm and args.single_seed, "Need --arm and --single-seed"
        r = run_cell(
            args.arm, args.single_seed, args.rate, args.total_steps,
            args.score, args.model, args.model_rev,
            Path(args.output_dir), probe_data=probe_data, run_sha=run_sha,
        )
        out_path = Path(args.output_dir) / args.arm / str(args.single_seed)
        out_path.mkdir(parents=True, exist_ok=True)
        print(json.dumps(r, indent=2))
        return

    assert len(args.seeds) == 10, f"Expected 10 seeds, got {len(args.seeds)}"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    arms = ["gate", "control", "full_finetune", "reduced_steps"]
    results_by_key = {}
    crash_log = []

    for arm in arms:
        for seed in args.seeds:
            print(f"\n=== arm={arm} seed={seed} ===")
            try:
                r = run_cell(
                    arm, seed, args.rate, args.total_steps, args.score,
                    args.model, args.model_rev, output,
                    probe_data=probe_data, run_sha=run_sha,
                )
                r["crashed"] = False
                results_by_key[(arm, seed)] = r
            except Exception as e:
                print(f"  FAILED: {e}")
                crash_log.append({"arm": arm, "seed": seed, "error": str(e)})

    results = list(results_by_key.values())

    gate_by_seed = {}
    ctrl_by_seed = {}
    for r in results:
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

        rng = np.random.RandomState(42)
        boot_means = [float(rng.choice(diff, size=n, replace=True).mean()) for _ in range(10000)]
        ci_lo = float(np.percentile(boot_means, 2.5))
        ci_hi = float(np.percentile(boot_means, 97.5))

        primary = {
            "mean_delta": mean_diff, "sd": sd, "t_stat": t_stat, "p_val": p_val,
            "n": n, "df": df, "ci_95": [ci_lo, ci_hi], "passed": p_val < 0.05,
        }
    else:
        primary = {"error": "no paired results"}

    summary = {
        "stage": "sst2_replication",
        "score_source": args.score,
        "rate": args.rate,
        "total_steps": args.total_steps,
        "seeds": args.seeds,
        "arms": arms,
        "run_sha": run_sha,
        "cell_results": results,
        "primary_test": primary,
        "crash_log": crash_log,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output / "sst2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== SST-2 Replication Summary ===")
    print(f"  Primary: Δacc(gate - control) = {primary.get('mean_delta', 'N/A')}")
    print(f"  t({primary.get('df', '?')}) = {primary.get('t_stat', 'N/A')}")
    print(f"  p = {primary.get('p_val', 'N/A')}")
    print(f"  95% CI: {primary.get('ci_95', [None, None])}")
    print(f"  Written to: {output / 'sst2_summary.json'}")


if __name__ == "__main__":
    main()
