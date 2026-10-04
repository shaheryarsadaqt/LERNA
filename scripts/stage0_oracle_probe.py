"""Stage 0: oracle probe to select S1 or S2 for Stage 2.

Runs full_finetune in shadow mode on two disjoint seeds (101, 103).
At every 8th step, logs S1 (forward gain), S2 (alignment), old LER score,
and realized gain on P_e (fp32, eval mode). Computes local Spearman ρ in
trailing 64-step windows with moving-block bootstrap (block 64, 10k resamples)
and selects the score with the higher per-seed ρ.

Usage:
    python scripts/stage0_oracle_probe.py \
        --seeds 101 103 \
        --rate 0.60 \
        --total-steps 6548 \
        --model jhu-clsp/ettin-encoder-150m@45d0864 \
        --probe-size 256 \
        --log-interval 8 \
        --probe-refresh 64 \
        --output-dir /raid/stage0_oracle_probe/

Pre-registered: experiments/PREREG_V4.md §4.
"""
import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


def _max_skips_possible(r: int, k: int) -> int:
    if r <= 0:
        return 0
    return k * (r // (k + 1)) + min(r % (k + 1), k)


def _eligible_phase_quota(
    rate: float, total_steps: int, min_step: int, n_phases: int
) -> tuple[list[int], int, int, list[int]]:
    requested_total = int(round(rate * total_steps))
    windows = _phase_bounds(total_steps, n_phases)
    eligible = [_eligible_steps_in_window(s, e, min_step) for (s, e) in windows]
    total_eligible = sum(eligible)
    if total_eligible <= 0:
        return [0] * n_phases, 0, 0, eligible
    if requested_total > total_eligible:
        raise ValueError(
            f"Infeasible: requested {requested_total} > eligible {total_eligible}"
        )
    raw = [requested_total * elig / total_eligible for elig in eligible]
    quota = [int(math.floor(r)) for r in raw]
    remainder = requested_total - sum(quota)
    fracs = [raw[i] - quota[i] for i in range(n_phases)]
    for _ in range(remainder):
        j = max(range(n_phases), key=lambda i: (fracs[i] if eligible[i] > 0 else -1.0))
        quota[j] += 1
        fracs[j] = raw[j] - quota[j]
    return quota, requested_total, total_eligible, eligible


def _phase_bounds(total_steps: int, n_phases: int) -> list[tuple[int, int]]:
    bounds = []
    step = 0
    for i in range(n_phases):
        end = total_steps if i == n_phases - 1 else step + total_steps // n_phases
        bounds.append((step, end))
        step = end
    return bounds


def moving_block_bootstrap_ci(
    data_x: list[float],
    data_y: list[float],
    block_size: int = 64,
    n_resamples: int = 10000,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Local Spearman with moving-block bootstrap 95% CI.

    Returns (point_estimate, ci_lower, ci_upper).
    """
    n = len(data_x)
    if n < 4:
        return float("nan"), float("nan"), float("nan")

    # Point estimate
    rho, _ = spearmanr(data_x, data_y)
    if math.isnan(rho):
        rho = 0.0
    rho = float(rho)

    if n <= block_size:
        # Single block — no resampling possible
        return rho, float("nan"), float("nan")

    # Moving block bootstrap
    rng = np.random.RandomState(42)
    n_blocks = math.ceil(n / block_size)
    boot_rhos = []
    for _ in range(n_resamples):
        idx = []
        for _b in range(n_blocks):
            start = rng.randint(0, n - block_size) if n > block_size else 0
            idx.extend(range(start, start + block_size))
        idx = idx[:n]
        if len(set(idx)) < n:
            # Handle potential degenerate cases
            pass
        boot_x = [data_x[i] for i in idx]
        boot_y = [data_y[i] for i in idx]
        try:
            b_rho, _ = spearmanr(boot_x, boot_y)
            if b_rho is not None and not math.isnan(b_rho):
                boot_rhos.append(b_rho)
        except Exception:
            pass

    if not boot_rhos:
        return rho, float("nan"), float("nan")

    ci_lower = float(np.percentile(boot_rhos, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_rhos, 100 * (1 - alpha / 2)))
    return rho, ci_lower, ci_upper


@dataclass
class Stage0Result:
    seed: int
    s1_spearman: list[float] = field(default_factory=list)
    s1_ci_lower: list[float] = field(default_factory=list)
    s1_ci_upper: list[float] = field(default_factory=list)
    s2_spearman: list[float] = field(default_factory=list)
    s2_ci_lower: list[float] = field(default_factory=list)
    s2_ci_upper: list[float] = field(default_factory=list)
    s1_max_rho: float = float("nan")
    s2_max_rho: float = float("nan")
    selected_score: str = "s1"


def run_stage0(
    seed: int,
    model_name: str,
    total_steps: int,
    rate: float,
    probe_size: int,
    log_interval: int,
    probe_refresh: int,
    output_dir: str,
) -> Stage0Result:
    """Run Stage 0 oracle probe for a single seed."""
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        torch_dtype=torch.float32,
    )

    # Get training data (QNLI)
    from datasets import load_dataset
    raw = load_dataset("glue", "qnli", split="train")
    raw = raw.shuffle(seed=seed)

    # Split into P_s, P_e, and train
    all_ids = list(range(len(raw)))
    probe_s_ids = set(all_ids[:probe_size])
    probe_e_ids = set(all_ids[probe_size : 2 * probe_size])
    train_ids = [i for i in all_ids if i not in probe_s_ids and i not in probe_e_ids]

    assert len(probe_s_ids & probe_e_ids) == 0, "P_s and P_e overlap"
    assert len(probe_s_ids & set(train_ids)) == 0, "P_s in train"
    assert len(probe_e_ids & set(train_ids)) == 0, "P_e in training"

    probe_s = raw.select(sorted(probe_s_ids))
    probe_e = raw.select(sorted(probe_e_ids))
    train_ds = raw.select(train_ids)

    # Tokenize
    def tokenize(batch):
        return tokenizer(batch["sentence"], batch["sentence2"], truncation="max_length", padding="max_length", max_length=512)

    probe_s = probe_s.map(tokenize, batched=True)
    probe_e = probe_e.map(tokenize, batched=True)
    train_ds = train_ds.map(tokenize, batched=True)

    # Attach alignment probe hook to classifier
    from lerna.utils.forward_gate import AlignmentProbe
    alignment_probe = AlignmentProbe(model, hook_module_name="classifier")

    # Optimizer (need state for S2)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # LR schedule (always_step contract)
    warmup_steps = 392
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / warmup_steps) * max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)) if step < total_steps else 0.0,
    )

    # Pre-tokenize probe batches
    probe_s_batch = {
        "input_ids": torch.tensor(probe_s["input_ids"]),
        "attention_mask": torch.tensor(probe_s["attention_mask"]),
        "labels": torch.tensor(probe_s["label"]),
    }
    probe_e_batch = {
        "input_ids": torch.tensor(probe_e["input_ids"]),
        "attention_mask": torch.tensor(probe_e["attention_mask"]),
        "labels": torch.tensor(probe_e["label"]),
    }

    def compute_probe_loss(model, batch):
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=False):
                out = model(**{k: v.to(model.device) for k, v in batch.items()})
                return out.loss.item()

    def compute_gp(model, probe_s_batch, optimizer):
        """Compute G_P = gradient of probe loss w.r.t. classifier.weight."""
        saved = {
            "training": model.training,
        }
        model.eval()
        try:
            # Zero gradients, compute forward, get gradient
            model.zero_grad()
            with torch.enable_grad():
                out = model(**{k: v.to(model.device) for k, v in probe_s_batch.items()})
                loss = out.loss
            G_P = None
            for name, p in model.named_parameters():
                if name == "classifier.weight":
                    G_P = p.grad.detach().clone() / labels_shape_0(probe_s_batch)
                    break
            model.zero_grad()
            return G_P
        finally:
            model.train(saved["training"])

    def labels_shape_0(batch):
        return batch["labels"].shape[0]

    # Training loop (shadow mode: full_finetune, no skipping)
    dataloader = torch.utils.data.DataLoader(
        train_ds.remove_columns(["sentence", "sentence2", "idx"]),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    dataloader_iter = iter(dataloader)

    # Logging
    step = 0
    s1_values = []
    s2_values = {}
    s2_values["gp"] = []    # G_P history for alignment
    old_ler_values = []
    pe_loss_history = []    # realized gain on P_e

    # Track P_e loss at each log step for gain computation
    pe_loss_at_step = {}

    print(f"[Stage 0] seed={seed}, total_steps={total_steps}, rate={rate}")
    print(f"  P_s size={probe_size}, P_e size={probe_size}, log_interval={log_interval}")

    # Initial P_e loss
    model.eval()
    with torch.no_grad():
        pe_out = model(**{k: v.to(model.device) for k, v in probe_e_batch.items()})
        pe_loss_initial = pe_out.loss.item()
    pe_loss_at_step[0] = pe_loss_initial
    model.train()

    # Refresh G_P at step 0
    gp = compute_gp(model, probe_s_batch, optimizer)
    v = optimizer.state[dict(model.named_parameters())["classifier.weight"]]
    v = v.get("exp_avg_sq") if v else None

    last_gp = gp
    last_v = v
    last_v_step = 0

    while step < total_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(model.device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}

        # Forward
        outputs = model(**batch)
        loss = outputs.loss

        # Backward (shadow mode: always backward)
        optimizer.zero_grad()
        loss.backward()

        # Log at interval
        if step > 0 and step % log_interval == 0:
            # S1: forward gain score
            from lerna.utils.forward_gate import forward_gain_score
            logits = outputs.logits.detach()
            s1 = forward_gain_score(logits, batch["labels"])
            s1_values.append(s1)

            # S2: alignment score
            if last_gp is not None and last_v is not None:
                from lerna.utils.forward_gate import alignment_score
                # alignment_score uses trainer._last_real_logits and trainer._alignment_probe
                # but here we compute directly
                h = alignment_probe.last_hidden
                if h is not None:
                    from lerna.utils.forward_gate import _classifier_grad_from_logits
                    G_B = _classifier_grad_from_logits(logits, batch["labels"], h)
                    normalized = last_gp / (last_v.sqrt() + 1e-8)
                    s2 = float(torch.dot(G_B.flatten(), normalized.flatten()).item())
                    s2_values["gp"].append(s2)
                else:
                    s2_values["gp"].append(float("nan"))
            else:
                s2_values["gp"].append(float("nan"))

            # Old LER score (lagged sampled-LER) — logged but not used for selection
            old_ler_values.append(0.0)  # placeholder; not implemented in shadow mode

            # Realized gain on P_e
            model.eval()
            with torch.no_grad():
                pe_out = model(**{k: v.to(model.device) for k, v in probe_e_batch.items()})
                pe_loss = pe_out.loss.item()
            pe_loss_at_step[step] = pe_loss
            model.train()

            # Refresh G_P every probe_refresh steps
            if step % probe_refresh == 0 and step > 0:
                last_gp = compute_gp(model, probe_s_batch, optimizer)
                v_dict = optimizer.state[dict(model.named_parameters())["classifier.weight"]]
                last_v = v_dict.get("exp_avg_sq") if v_dict else None
                last_v_step = step

        # Optimizer step (shadow mode: always step)
        optimizer.step()
        scheduler.step()

        step += 1

    # Remove hook
    alignment_probe.remove()

    # Compute realized gain
    pe_gains = []
    log_steps = list(range(log_interval, total_steps, log_interval))
    pe_loss_final = pe_loss_at_step.get(step, pe_loss_initial)
    for i, s in enumerate(log_steps):
        if s in pe_loss_at_step:
            gain = pe_loss_at_step[s] - pe_loss_initial  # change in loss
            pe_gains.append(gain)

    # Compute Spearman in windows of 64 log steps
    result = Stage0Result(seed=seed)

    # Use log_steps as x (step index), pe_gains as y (realized gain)
    # Compute local Spearman in trailing 64-step windows
    window_size = 64  # log entries per window = 64 * 8 = 512 actual steps

    for score_name, score_values, spearmans, ci_lowers, ci_uppers, attr in [
        ("s1", s1_values, result.s1_spearman, result.s1_ci_lower, result.s1_ci_upper, "s1_max_rho"),
        ("s2", s2_values["gp"], result.s2_spearman, result.s2_ci_lower, result.s2_ci_upper, "s2_max_rho"),
    ]:
        n = len(score_values)
        if n < 4:
            continue

        # Trailing windows of 64 entries
        for start_idx in range(0, n, window_size):
            end_idx = min(start_idx + window_size, n)
            if end_idx - start_idx < 4:
                continue

            x = score_values[start_idx:end_idx]
            y = pe_gains[start_idx:end_idx] if len(pe_gains) >= end_idx else pe_gains[start_idx:start_idx+len(pe_gains)]

            # Align lengths
            min_len = min(len(x), len(y))
            if min_len < 4:
                continue
            x = x[:min_len]
            y = y[:min_len]

            rho, ci_lo, ci_hi = moving_block_bootstrap_ci(
                x, y, block_size=min(64, min_len), n_resamples=10000
            )
            spearmans.append(rho)
            ci_lowers.append(ci_lo)
            ci_uppers.append(ci_hi)

        # Max per-seed rho
        valid_rhos = [r for r in spearmans if not math.isnan(r)]
        if valid_rhos:
            setattr(result, attr, max(valid_rhos))

    # Selection: higher max_rho wins; ties go to S1
    if not math.isnan(result.s2_max_rho) and result.s2_max_rho > result.s1_max_rho:
        result.selected_score = "s2"
    else:
        result.selected_score = "s1"

    return result


def labels_shape_0(batch):
    return batch["labels"].shape[0]


def main():
    parser = argparse.ArgumentParser(description="Stage 0: oracle probe")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--model", type=str, default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--probe-size", type=int, default=256)
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--probe-refresh", type=int, default=64)
    parser.add_argument("--output-dir", type=str, default="/raid/stage0_oracle_probe/")
    args = parser.parse_args()

    results = {}
    for seed in args.seeds:
        print(f"\n=== Stage 0: seed={seed} ===")
        result = run_stage0(
            seed=seed,
            model_name=args.model,
            total_steps=args.total_steps,
            rate=args.rate,
            probe_size=args.probe_size,
            log_interval=args.log_interval,
            probe_refresh=args.probe_refresh,
            output_dir=args.output_dir,
        )
        results[seed] = {
            "s1_spearman": result.s1_spearman,
            "s1_ci_lower": result.s1_ci_lower,
            "s1_ci_upper": result.s1_ci_upper,
            "s1_max_rho": result.s1_max_rho,
            "s2_spearman": result.s2_spearman,
            "s2_ci_lower": result.s2_ci_lower,
            "s2_ci_upper": result.s2_ci_upper,
            "s2_max_rho": result.s2_max_rho,
            "selected_score": result.selected_score,
        }
        print(f"  S1 max ρ: {result.s1_max_rho}")
        print(f"  S2 max ρ: {result.s2_max_rho}")
        print(f"  Selected: {result.selected_score}")

    # Overall selection: take the higher of the two per-seed ρ values for each
    # score, pick the score that is higher. Ties → S1.
    s1_best = max(r["s1_max_rho"] for r in results.values() if not math.isnan(r["s1_max_rho"]))
    s2_best = max(r["s2_max_rho"] for r in results.values() if not math.isnan(r["s2_max_rho"]))
    selected = "s2" if s2_best > s1_best else "s1"

    output = {
        "stage": "stage0_oracle_probe",
        "seeds": args.seeds,
        "rate": args.rate,
        "total_steps": args.total_steps,
        "selected_score": selected,
        "s1_best_rho": s1_best,
        "s2_best_rho": s2_best,
        "results_per_seed": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "gate_a_decision.json"), "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n=== Stage 0 complete ===")
    print(f"  S1 best ρ: {s1_best}")
    print(f"  S2 best ρ: {s2_best}")
    print(f"  Selected score: {selected}")
    print(f"  Written to: {os.path.join(args.output_dir, 'gate_a_decision.json')}")


if __name__ == "__main__":
    main()
