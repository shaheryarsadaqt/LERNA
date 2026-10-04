"""Stage 2: QNLI confirmatory run at rate 0.60.

Runs the chosen score (from Stage 0) against its uniform control, plus
full_finetune and reduced_steps, over 24 seeds on QNLI.

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
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _build_cell(arm, seed, rate, total_steps, score_kind, min_step=392, **kw):
    """Build a training cell for the given arm and seed."""
    from lerna.utils.forward_gate import (
        ScoreQuantileGatePolicy,
        AlignmentProbe,
    )

    if arm == "full_finetune":
        from lerna.trainers.policies import AlwaysFalsePolicy
        return AlwaysFalsePolicy(seed=seed)

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
        if score_source == "s2":
            pol.alignment_probe_fn = kw.get("probe_fn")
        return pol

    # reduced_steps handled as a callback, not a policy
    return None


def run_cell(arm, seed, rate, total_steps, score_kind, model_id, model_rev, output_dir):
    """Run one cell and return results dict."""
    import torch.distributed as dist
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, TrainerCallback,
    )
    from datasets import load_dataset
    from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    torch.manual_seed(seed)
    np.random.seed(seed)

    tok = AutoTokenizer.from_pretrained(model_id, revision=model_rev)
    ds_tr = load_dataset("glue", "qnli", split="train")
    ds_ev = load_dataset("glue", "qnli", split="validation")

    def tok_fn(x):
        out = tok(x["question"], x["sentence"], truncation="max_length",
                  padding="max_length", max_length=512)
        out["label"] = x["label"]
        return out

    ds_tr = ds_tr.map(tok_fn, remove_columns=list(ds_tr.column_names))
    ds_ev = ds_ev.map(tok_fn, remove_columns=list(ds_ev.column_names))

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, revision=model_rev, num_labels=2, torch_dtype=torch.float32
    )

    args = TrainingArguments(
        output_dir=str(output_dir / arm / str(seed)),
        max_steps=total_steps,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        gradient_accumulation_steps=1,
        learning_rate=2e-5,
        warmup_steps=392,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=False, bf16=False,
        eval_strategy="no", save_strategy="no",
        report_to="none", logging_steps=100, seed=seed,
        dataloader_num_workers=0, remove_unused_columns=False,
        disable_tqdm=True,
    )

    policy = _build_cell(arm, seed, rate, total_steps, score_kind)

    def _collate(tokenizer):
        import torch as _T
        def coll(batch):
            return {
                "input_ids": _T.tensor([d["input_ids"] for d in batch]),
                "attention_mask": _T.tensor([d["attention_mask"] for d in batch]),
                "labels": _T.tensor([d["label"] for d in batch]),
            }
        return coll

    def _acc(p):
        preds = np.asarray(p.predictions).argmax(-1)
        return {"accuracy": float((preds == np.asarray(p.label_ids)).mean())}

    t = TrueBackwardSkippingTrainer(
        model=model, args=args,
        train_dataset=ds_tr, eval_dataset=ds_ev,
        data_collator=_collate(tok),
        compute_metrics=_acc,
        skip_policy=policy,
        scheduler_step_policy="always_step",
        capture_logits=True,
        instrumentation_path=str(output_dir / arm / str(seed) / "instrumentation.json"),
    )

    t0 = time.perf_counter()
    t.train()
    wall = time.perf_counter() - t0
    metrics = t.evaluate()

    return {
        "arm": arm,
        "seed": seed,
        "rate": rate,
        "total_steps": total_steps,
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_loss": metrics.get("eval_loss"),
        "wall_seconds": wall,
        "ms_per_step": wall / total_steps * 1000,
        "scheduler_step_calls": getattr(getattr(t, "instr", None), "scheduler_step_calls", 0),
        "batches_seen": getattr(getattr(t, "instr", None), "batches_seen", 0),
    }


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
    args = parser.parse_args()

    assert len(args.seeds) == 24, f"Expected 24 seeds, got {len(args.seeds)}"

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # Arms: gate (chosen score), control (uniform), full_finetune, reduced_steps
    arms = ["gate", "control", "full_finetune", "reduced_steps"]

    results = []
    for arm in arms:
        for seed in args.seeds:
            print(f"\n=== arm={arm} seed={seed} (rate={args.rate}) ===")
            try:
                r = run_cell(
                    arm, seed, args.rate, args.total_steps, args.score,
                    args.model, args.model_rev, output,
                )
                results.append(r)
            except Exception as e:
                print(f"  FAILED: {e}")
                # Use reserve seed
                if args.reserves:
                    rs = args.reserves.pop(0)
                    print(f"  Retrying with reserve seed={rs}")
                    r = run_cell(
                        arm, rs, args.rate, args.total_steps, args.score,
                        args.model, args.model_rev, output,
                    )
                    results.append(r)

    # Write summary
    cells = {}
    for r in results:
        key = (r["arm"], r["seed"])
        cells[key] = r

    # Paired analysis: gate vs control (primary)
    gate_accs = sorted([r["eval_accuracy"] for r in results if r["arm"] == "gate"],
                       key=lambda x: x if x is not None else -1)
    ctrl_accs = sorted([r["eval_accuracy"] for r in results if r["arm"] == "control"],
                       key=lambda x: x if x is not None else -1)

    # Pair by seed
    gate_by_seed = {r["seed"]: r["eval_accuracy"] for r in results if r["arm"] == "gate"}
    ctrl_by_seed = {r["seed"]: r["eval_accuracy"] for r in results if r["arm"] == "control"}

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
        "reserves_used": args.reserves,
        "arms": arms,
        "cell_results": results,
        "primary_test": primary,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    with open(output / "stage2_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n=== Stage 2 Summary ===")
    print(f"  Primary: Δacc(gate - control) = {primary.get('mean_delta', 'N/A'):.4f} pt")
    print(f"  t({primary.get('df', '?')}) = {primary.get('t_stat', 'N/A'):.4f}")
    print(f"  p = {primary.get('p_val', 'N/A'):.4f}")
    print(f"  95% CI: [{primary.get('ci_95', [None, None])[0]:.4f}, {primary.get('ci_95', [None, None])[1]:.4f}]")
    print(f"  Written to: {output / 'stage2_summary.json'}")


if __name__ == "__main__":
    main()
