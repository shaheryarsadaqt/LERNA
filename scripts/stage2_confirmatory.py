"""Stage 2: QNLI confirmatory run at rate 0.60.

Runs the chosen score (from Stage 0) against its uniform control, plus
full_finetune and reduced_steps, over 24 seeds on QNLI.

Arms:
- gate: ScoreQuantileGatePolicy with chosen score (s1 or s2)
- control: ScoreQuantileGatePolicy with score_source="uniform"
- full_finetune: AlwaysFalsePolicy (0% skip, 6548 steps)
- reduced_steps: train 40% of steps (2619), matched backward budget

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
REDUCED_TOTAL_STEPS = 2619
REDUCED_WARMUP_STEPS = int(REDUCED_TOTAL_STEPS * MIN_STEP_RATIO)  # 157
FULL_STEPS = 6548
FULL_WARMUP = int(FULL_STEPS * MIN_STEP_RATIO)  # 392


def _check_provenance():
    """F11: refuse to run unless tree is clean and HEAD == v4-prereg-r2."""
    import subprocess
    repo_root = Path(__file__).resolve().parents[1]

    # Check dirty tree
    r = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True, text=True, cwd=str(repo_root)
    )
    if r.stdout.strip():
        print(f"FATAL: git tree is dirty. Commit or stash first.\n{r.stdout}")
        sys.exit(1)

    # Check HEAD == v4-prereg-r2
    try:
        tag_sha = subprocess.run(
            ["git", "rev-parse", "v4-prereg-r2"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        if tag_sha != head_sha:
            print(f"FATAL: HEAD ({head_sha}) != v4-prereg-r2 ({tag_sha}).")
            sys.exit(1)
    except Exception:
        print("FATAL: could not resolve v4-prereg-r2 tag.")
        sys.exit(1)

    return head_sha


def _load_probe_data(probe_file: str) -> dict:
    with open(probe_file) as f:
        data = json.load(f)
    sha = hashlib.sha256(open(probe_file, "rb").read()).hexdigest()
    if data.get("sha256") and sha != data["sha256"]:
        print(f"FATAL: probe file hash mismatch. Expected {data['sha256']}, got {sha}")
        sys.exit(1)
    return data


def _build_policy(arm, seed, rate, total_steps, score_kind, min_step, probe_fn=None):
    """Build the skip policy for the given arm."""
    from lerna.utils.forward_gate import (
        ScoreQuantileGatePolicy,
        AlignmentProbe,
    )
    from lerna.trainers.policies import AlwaysFalsePolicy

    if arm == "full_finetune":
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
    return AlwaysFalsePolicy(seed=seed)


def _build_training_args(
    arm, seed, total_steps, warmup_steps, output_dir, no_early_stopping=True,
):
    """Build TrainingArguments for the given arm."""
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
        eval_strategy="steps",
        eval_steps=max(steps // 20, 10),
        save_strategy="no",
        load_best_model_at_end=not no_early_stopping,  # False — final checkpoint only
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", logging_steps=100, seed=seed,
        dataloader_num_workers=0, remove_unused_columns=True,
        disable_tqdm=True,
        use_cpu=False,
    )


def run_cell(
    arm, seed, rate, total_steps, score_kind, model_id, model_rev,
    output_dir, probe_fn=None,
):
    """Run one cell and return results dict.

    F7: reduced_steps uses max_steps=2619, warmup=157, linear decay to 0.
    F8: S2 gate fails hard if probe_fn missing or score non-finite at step 392.
    F11: Each cell writes a manifest and runs validate_skip_policy_results.
    """
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
    )
    from datasets import load_dataset
    from lerna.trainers.true_skip_trainer import TrueBackwardSkippingTrainer

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

    args = _build_training_args(
        arm, seed, total_steps, FULL_WARMUP, output_dir,
    )

    # F6: exclude probe indices from training
    probe_data = _load_probe_data(
        os.path.join(os.path.dirname(__file__), "..", "experiments", "probe_indices_v4.json")
    )
    probe_s_ids = set(probe_data["probe_s_ids"])
    probe_e_ids = set(probe_data["probe_e_ids"])

    all_train_ids = list(range(len(ds_tr)))
    filtered_ids = [i for i in all_train_ids if i not in probe_s_ids and i not in probe_e_ids]

    assert probe_s_ids.isdisjoint(probe_e_ids), "P_s and P_e overlap"
    assert not probe_s_ids.intersection(filtered_ids), "P_s in train"
    assert not probe_e_ids.intersection(filtered_ids), "P_e in train"

    ds_tr = ds_tr.select(filtered_ids)

    min_step = int(total_steps * MIN_STEP_RATIO) if arm not in ("reduced_steps",) else int(REDUCED_TOTAL_STEPS * MIN_STEP_RATIO)

    policy = _build_policy(arm, seed, rate, total_steps, score_kind, min_step, probe_fn)

    def _acc(p):
        preds = np.asarray(p.predictions).argmax(-1)
        return {"accuracy": float((preds == np.asarray(p.label_ids)).mean())}

    def _collate(tokenizer):
        import torch as _T
        def coll(batch):
            return {
                "input_ids": _T.tensor([d["input_ids"] for d in batch]),
                "attention_mask": _T.tensor([d["attention_mask"] for d in batch]),
                "labels": _T.tensor([d["label"] for d in batch]),
            }
        return coll

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

    # F8: for S2, verify the probe is attached and score is non-finite at step 392
    if arm == "gate" and score_kind == "s2":
        original_should_skip = policy.should_skip

        def checking_should_skip(trainer, model, inputs):
            if trainer.instr.decisions_seen == FULL_WARMUP:
                probe = getattr(trainer, "_alignment_probe", None)
                assert probe is not None, \
                    "FATAL: S2 arm but AlignmentProbe not attached to trainer"
                assert probe.last_hidden is not None, \
                    "FATAL: S2 arm but AlignmentProbe.last_hidden is None at step 392"
                # Verify score is finite
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
    expected_final_lr = 0.0
    # For always_step with linear decay: lr at step max_steps should be ~0
    assert final_lr <= 1e-8, \
        f"FATAL: final LR={final_lr} > 1e-8 — scheduler step contract violated"

    scheduler_calls = getattr(getattr(t, "instr", None), "scheduler_step_calls", 0)
    batches_seen = getattr(getattr(t, "instr", None), "batches_seen", 0)

    # Assert scheduler steps == batches seen (always_step contract)
    expected_steps = REDUCED_TOTAL_STEPS if arm == "reduced_steps" else FULL_STEPS
    assert scheduler_calls == batches_seen == expected_steps, \
        f"FATAL: scheduler_step_calls={scheduler_calls} != batches_seen={batches_seen} != expected={expected_steps}"

    metrics = t.evaluate()

    result = {
        "arm": arm,
        "seed": seed,
        "rate": rate,
        "total_steps": total_steps,
        "actual_steps_run": expected_steps,
        "eval_accuracy": metrics.get("eval_accuracy"),
        "eval_loss": metrics.get("eval_loss"),
        "final_lr": final_lr,
        "wall_seconds": wall,
        "ms_per_step": wall / expected_steps * 1000,
        "scheduler_step_calls": scheduler_calls,
        "batches_seen": batches_seen,
        "min_step": min_step,
    }

    # F11: write results.json and run validate_skip_policy_results.py --expected-sha
    # The analysis script refuses any cell that fails validation.
    cell_output = Path(str(output_dir)) / arm / str(seed)
    results_path = cell_output / "results.json"
    os.makedirs(str(cell_output), exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(result, f, indent=2)

    expected_sha = _check_provenance()
    repo_root = Path(__file__).resolve().parents[1]
    validator = str(repo_root / "scripts" / "validate_skip_policy_results.py")
    val = subprocess.run(
        [sys.executable, validator, str(results_path), "--expected-sha", expected_sha],
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
    args = parser.parse_args()

    # F11: provenance checks (skip in single-cell mode — launcher handles it)
    if not args.single_cell:
        run_sha = _check_provenance()
    else:
        run_sha = "single-cell"

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    # Single-cell mode (for launcher)
    if args.single_cell:
        assert args.arm and args.single_seed, "Need --arm and --single-seed in single-cell mode"
        probe_fn = None  # The S2 probe_fn is set up per-model in the trainer
        r = run_cell(
            args.arm, args.single_seed, args.rate, args.total_steps,
            args.score, args.model, args.model_rev,
            Path(args.output_dir), probe_fn=probe_fn,
        )
        out_path = Path(args.output_dir) / args.arm / str(args.single_seed)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / "manifest.json", "w") as f:
            json.dump({
                "arm": r["arm"], "seed": r["seed"],
                "run_sha": run_sha,
                "probe_sha256": hashlib.sha256(
                    open(args.probe_file, "rb").read()
                ).hexdigest(),
                "config": r,
            }, f, indent=2)
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
    active_seeds = list(args.seeds)
    reserve_seeds = list(args.reserves)
    crash_log = []

    results = []
    for arm in arms:
        for i, seed in enumerate(active_seeds[:]):
            print(f"\n=== arm={arm} seed={seed} (rate={args.rate}) ===")
            attempt = 0
            success = False
            while attempt < 2 and not success:
                try:
                    r = run_cell(
                        arm, seed, args.rate, args.total_steps, args.score,
                        args.model, args.model_rev, output,
                    )
                    r["attempts"] = attempt + 1
                    r["crashed"] = False
                    results.append(r)
                    success = True
                except Exception as e:
                    print(f"  FAILED (attempt {attempt + 1}): {e}")
                    if attempt == 0:
                        # F9: retry same seed once
                        print(f"  Retrying with same seed={seed}")
                        attempt += 1
                    elif reserve_seeds:
                        # F9: replace seed in ALL arms with next reserve
                        new_seed = reserve_seeds.pop(0)
                        crash_log.append({
                            "arm": arm, "orig_seed": seed,
                            "replacement_seed": new_seed,
                            "error": str(e),
                            "attempts": 2,
                        })
                        # Replace in all arms
                        # F9: replace seed in ALL arms with next reserve
                        if seed in active_seeds:
                            idx = active_seeds.index(seed)
                            active_seeds[idx] = new_seed
                        seed = new_seed
                        print(f"  Replacing seed {seed} with reserve {new_seed} in ALL arms")
                        attempt = 0
                    else:
                        crash_log.append({
                            "arm": arm, "orig_seed": seed,
                            "replacement_seed": None,
                            "error": str(e),
                            "attempts": 2,
                        })
                        break

    # Write summary
    cells = {}
    for r in results:
        key = (r["arm"], r["seed"])
        cells[key] = r

    # F11: analysis refuses cells that failed validation
    validated_results = [r for r in results if r.get("validation_passed") is True]
    failed_validation = [r for r in results if r.get("validation_passed") is not True]
    if failed_validation:
        print(f"\nWARNING: {len(failed_validation)} cells failed validation and are REFUSED:")
        for r in failed_validation:
            print(f"  arm={r['arm']} seed={r['seed']}: {r.get('validation_stderr', 'unknown')[:200]}")

    # Paired analysis: gate vs control (primary) — only validated cells
    gate_by_seed = {r["seed"]: r["eval_accuracy"] for r in validated_results if r["arm"] == "gate"}
    ctrl_by_seed = {r["seed"]: r["eval_accuracy"] for r in validated_results if r["arm"] == "control"}

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
