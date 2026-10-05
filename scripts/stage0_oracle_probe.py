"""Stage 0: oracle probe to select S1 or S2 for Stage 2.

Runs full_finetune in shadow mode on two disjoint seeds (101, 103).
At every 8th step, logs S1 (forward gain), S2 (alignment), and realized
per-step gain on P_e. Computes local Spearman rho in non-overlapping
windows of 64 logged points (512 actual steps), with cluster bootstrap
(10k resamples) for descriptive CIs only. Selection uses the mean of
per-window rho values, pooled over seeds 101 and 103, only in steps
392-5761. Ties go to S1.

F11 provenance: refuses to run on dirty git tree or unless HEAD == v4-prereg-r2.
Writes a manifest with run_sha, probe_sha256, and selected_score.

Pre-registered: experiments/PREREG_V4.md §5.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr


def _phase_bounds(total_steps: int, n_phases: int) -> list[tuple[int, int]]:
    bounds = []
    step = 0
    for i in range(n_phases):
        end = total_steps if i == n_phases - 1 else step + total_steps // n_phases
        bounds.append((step, end))
        step = end
    return bounds


def _eligible_steps_in_window(start: int, end: int, min_step: int) -> int:
    elig_start = max(start, min_step)
    return max(0, end - elig_start)


@dataclass
class Stage0Result:
    seed: int
    s1_spearman: list[float] = field(default_factory=list)
    s1_ci_lower: list[float] = field(default_factory=list)
    s1_ci_upper: list[float] = field(default_factory=list)
    s2_spearman: list[float] = field(default_factory=list)
    s2_ci_lower: list[float] = field(default_factory=list)
    s2_ci_upper: list[float] = field(default_factory=list)
    s1_mean_rho: float = float("nan")
    s2_mean_rho: float = float("nan")
    s1_ci_per_seed: tuple[float, float] = (float("nan"), float("nan"))
    s2_ci_per_seed: tuple[float, float] = (float("nan"), float("nan"))
    log_steps: list[int] = field(default_factory=list)
    # B12: reference predictors (descriptive only)
    step_index_rho: float = float("nan")
    batch_loss_rho: float = float("nan")


def load_probe_indices(probe_file: str) -> dict:
    with open(probe_file) as f:
        data = json.load(f)
    return data


def compute_probe_loss(model, batch, device):
    saved = model.training
    model.eval()
    try:
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in batch.items()})
            return out.loss.item()
    finally:
        model.train(saved)


def compute_gp_from_logits(model, probe_s_batch, device, alignment_probe_fn):
    """Compute G_P = gradient of probe loss w.r.t. classifier.weight.

    Uses _classifier_grad_from_logits on P_s (eval-mode hook), the same
    function as G_B. Does NOT do a backward pass or zero_grad, so it
    cannot wipe the training gradient.
    """
    from lerna.utils.forward_gate import _classifier_grad_from_logits

    saved = {"training": model.training}
    model.eval()
    try:
        probe_s_batch = {k: v.to(device) for k, v in probe_s_batch.items()}
        with torch.no_grad():
            out = model(**probe_s_batch)
            logits = out.logits.detach()
            labels = probe_s_batch["labels"]

        hidden = alignment_probe_fn() if alignment_probe_fn else None
        if hidden is None:
            return None

        G_P = _classifier_grad_from_logits(logits, labels, hidden)
        return G_P
    finally:
        model.train(saved["training"])


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
            model_name, revision=revision, num_labels=2, torch_dtype=torch.float32)
    except (ValueError, OSError):
        from transformers import BertConfig, BertForSequenceClassification
        from huggingface_hub import hf_hub_download
        import json
        config_path = hf_hub_download(model_name, "config.json", revision=revision)
        with open(config_path) as f:
            cfg_dict = json.load(f)
        cfg_dict["model_type"] = "bert"
        config = BertConfig.from_dict(cfg_dict)
        config.num_labels = 2
        model = BertForSequenceClassification(config)
        return model


def run_stage0(
    seed: int,
    model_name: str,
    model_rev: str,
    total_steps: int,
    rate: float,
    probe_indices_file: str,
    log_interval: int = 8,
    probe_refresh: int = 512,
    output_dir: str = "/raid/stage0_oracle_probe/",
) -> Stage0Result:
    """Run Stage 0 oracle probe for a single seed.

    Shadow mode: full_finetune (no skipping, backward every step).
    """
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    torch.manual_seed(seed)
    np.random.seed(seed)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer = _load_tokenizer(model_name, revision=model_rev)
    model = _load_model(model_name, revision=model_rev)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    from datasets import load_dataset
    raw = load_dataset("glue", "qnli", split="train")

    # B11: select probes from unshuffled order (shuffle happens only for training)
    probe_data = load_probe_indices(probe_indices_file)
    probe_s_ids = set(probe_data["probe_s_ids"])
    probe_e_ids = set(probe_data["probe_e_ids"])
    all_ids = set(range(len(raw)))
    train_ids = sorted(all_ids - probe_s_ids - probe_e_ids)

    assert probe_s_ids.isdisjoint(probe_e_ids), "P_s and P_e overlap"
    assert probe_s_ids.isdisjoint(train_ids), "P_s in train"
    assert probe_e_ids.isdisjoint(train_ids), "P_e in train"

    probe_s = raw.select(sorted(probe_s_ids))
    probe_e = raw.select(sorted(probe_e_ids))
    train_ds = raw.select(train_ids)

    # B11: shuffle only the training dataset, not the probe selections
    train_ds = train_ds.shuffle(seed=seed)

    def tokenize(batch):
        # B12: dynamic padding — no padding here, handled by DataLoader collate
        out = tokenizer(
            batch["question"], batch["sentence"],
            truncation=True, max_length=512,
        )
        out["labels"] = batch["label"]
        return out

    probe_s = probe_s.map(tokenize, batched=True)
    probe_e = probe_e.map(tokenize, batched=True)
    train_ds = train_ds.map(tokenize, batched=True)

    # Attach alignment probe
    from lerna.utils.forward_gate import AlignmentProbe
    alignment_probe = AlignmentProbe(model, hook_module_name="classifier")

    def get_last_hidden():
        return alignment_probe.last_hidden

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
    )

    # LR schedule (always_step contract)
    warmup_steps = int(total_steps * 0.06)  # 392 for QNLI
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        if step >= total_steps:
            return 0.0
        return max(0.0, 1.0 - (step - warmup_steps) / (total_steps - warmup_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    probe_s_batch = tokenizer.pad(
        {"input_ids": probe_s["input_ids"], "attention_mask": probe_s["attention_mask"]},
        padding="longest", max_length=512, return_tensors="pt")
    probe_s_batch["labels"] = torch.tensor(probe_s["labels"])
    probe_s_batch = {k: v.to(device) for k, v in probe_s_batch.items()}
    probe_e_batch = tokenizer.pad(
        {"input_ids": probe_e["input_ids"], "attention_mask": probe_e["attention_mask"]},
        padding="longest", max_length=512, return_tensors="pt")
    probe_e_batch["labels"] = torch.tensor(probe_e["labels"])
    probe_e_batch = {k: v.to(device) for k, v in probe_e_batch.items()}

    def _collate_fn(batch):
        batch_dict = {k: [d[k] for d in batch] for k in ("input_ids", "attention_mask")}
        batch_dict["labels"] = torch.tensor([d["labels"] for d in batch])
        padded = tokenizer.pad(batch_dict, padding="longest", max_length=512, return_tensors="pt")
        return {k: v.to(device) for k, v in padded.items()}

    dataloader = torch.utils.data.DataLoader(
        train_ds.remove_columns(["sentence", "sentence2", "idx", "label", "question"]),
        batch_size=32,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=_collate_fn,
    )
    dataloader_iter = iter(dataloader)

    # Logging
    step = 0
    s1_values = []
    s2_values = []
    pe_gain_values = []  # F1 fix: per-step, correct sign
    step_index_values = []  # B12: reference predictor
    batch_loss_values = []  # B12: reference predictor
    log_steps = []

    # Track P_e loss at each log step
    pe_loss_at_step = {}

    print(f"[Stage 0] seed={seed}, total_steps={total_steps}, rate={rate}")
    print(f"  P_s size={len(probe_s_ids)}, P_e size={len(probe_e_ids)}, log_interval={log_interval}")

    # Initial P_e loss
    initial_pe_loss = compute_probe_loss(model, probe_e_batch, device)
    pe_loss_at_step[0] = initial_pe_loss

    # Refresh G_P at step 0
    last_gp = compute_gp_from_logits(model, probe_s_batch, device, get_last_hidden)
    v_dict = optimizer.state[dict(model.named_parameters())["classifier.weight"]]
    last_v = v_dict.get("exp_avg_sq") if v_dict else None
    last_v_step = 0

    # Anneal range: steps 392-5761 only
    min_step = warmup_steps
    max_step = int(total_steps * 0.88)  # 5762 for QNLI, so range is 392-5761

    while step < total_steps:
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)

        batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask", "labels")}

        # Forward
        outputs = model(**batch)
        logits = outputs.logits.detach()
        loss = outputs.loss

        # Backward (shadow mode: always backward)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Log at interval
        if step > 0 and step % log_interval == 0:
            # S1: forward gain score
            from lerna.utils.forward_gate import forward_gain_score
            s1 = forward_gain_score(logits, batch["labels"])
            s1_values.append(s1)

            # S2: alignment score — B12: uses shared _s2_alignment_core
            from lerna.utils.forward_gate import _classifier_grad_from_logits, _s2_alignment_core
            h = alignment_probe.last_hidden
            if h is not None and last_gp is not None and last_v is not None:
                G_B = _classifier_grad_from_logits(logits.float(), batch["labels"], h)
                s2 = _s2_alignment_core(G_B, last_gp.float(), last_v.float())
            else:
                s2 = float("nan")
            s2_values.append(s2)

            # B12: reference predictors (descriptive only)
            step_index_values.append(float(step))
            batch_loss_values.append(float(loss.item()))

            # F1 fix: per-step gain, correct sign (before - after)
            # We compute P_e loss BEFORE optimizer.step(), step, then compute AFTER.
            pe_loss_before = compute_probe_loss(model, probe_e_batch, device)

            # Optimizer step
            optimizer.step()
            scheduler.step()

            pe_loss_after = compute_probe_loss(model, probe_e_batch, device)
            gain = pe_loss_before - pe_loss_after  # positive = loss decreased = gain
            pe_gain_values.append(gain)
            pe_loss_at_step[step] = pe_loss_after
            log_steps.append(step)

            # Refresh G_P at probe_refresh interval
            if step % probe_refresh == 0 and step > 0:
                last_gp = compute_gp_from_logits(model, probe_s_batch, device, get_last_hidden)
                v_dict = optimizer.state[dict(model.named_parameters())["classifier.weight"]]
                last_v = v_dict.get("exp_avg_sq") if v_dict else None
                last_v_step = step
        else:
            # Optimizer step (shadow mode: always step)
            optimizer.step()
            scheduler.step()

        step += 1

    # Remove hook
    alignment_probe.remove()

    # Final P_e loss
    final_pe_loss = compute_probe_loss(model, probe_e_batch, device)
    pe_loss_at_step[step] = final_pe_loss

    # F2/F3: compute rho in non-overlapping windows of 64 logged points
    # (512 actual steps each), only in steps 392-5761
    result = Stage0Result(seed=seed)
    result.log_steps = list(log_steps)

    # B12: eligible range is 392-5761 (exclusive of 5762)
    eligible_indices = [
        i for i, s in enumerate(log_steps) if min_step <= s < max_step
    ]

    window_size = 64  # 64 logged points = 512 actual steps

    for score_name, score_values, spearmans, ci_lowers, ci_uppers, attr_mean in [
        ("s1", s1_values, result.s1_spearman, result.s1_ci_lower, result.s1_ci_upper, "s1_mean_rho"),
        ("s2", s2_values, result.s2_spearman, result.s2_ci_lower, result.s2_ci_upper, "s2_mean_rho"),
    ]:
        # Filter score values to eligible range
        eligible_scores = [
            (score_values[i] if i < len(score_values) else float("nan"))
            for i in eligible_indices
        ]
        eligible_gains = [
            (pe_gain_values[i] if i < len(pe_gain_values) else float("nan"))
            for i in eligible_indices
        ]

        n = len(eligible_scores)
        if n < 4:
            continue

        # Non-overlapping windows of 64 entries
        window_rhos = []
        window_ci_los = []
        window_ci_his = []
        for start_idx in range(0, n, window_size):
            end_idx = min(start_idx + window_size, n)
            if end_idx - start_idx < 4:
                continue

            x = eligible_scores[start_idx:end_idx]
            y = eligible_gains[start_idx:end_idx]

            # Filter NaN
            valid = [(xi, yi) for xi, yi in zip(x, y)
                     if not (math.isnan(xi) if isinstance(xi, float) else False)
                     and not (math.isnan(yi) if isinstance(yi, float) else False)]
            if len(valid) < 4:
                continue

            vx = [v[0] for v in valid]
            vy = [v[1] for v in valid]
            min_len = min(len(vx), len(vy))
            rho, _ = spearmanr(vx[:min_len], vy[:min_len])
            if rho is not None and not math.isnan(rho):
                window_rhos.append(float(rho))
                spearmans.append(float(rho))

        if not window_rhos:
            continue

        # F2: mean of per-window rho values
        mean_rho = float(np.mean(window_rhos))
        setattr(result, attr_mean, mean_rho)

        # F3: cluster bootstrap CI (descriptive only)
        if len(window_rhos) >= 2:
            rng = np.random.RandomState(42)
            boot_means = []
            for _ in range(10000):
                idx = rng.randint(0, len(window_rhos), size=len(window_rhos))
                boot_means.append(float(np.mean([window_rhos[i] for i in idx])))
            ci_lo = float(np.percentile(boot_means, 2.5))
            ci_hi = float(np.percentile(boot_means, 97.5))
        else:
            ci_lo, ci_hi = float("nan"), float("nan")

        if score_name == "s1":
            result.s1_ci_per_seed = (ci_lo, ci_hi)
        else:
            result.s2_ci_per_seed = (ci_lo, ci_hi)

    # B12: reference predictor rho (step index and batch loss vs realized gain)
    for ref_name, ref_values, attr in [
        ("step_index", step_index_values, "step_index_rho"),
        ("batch_loss", batch_loss_values, "batch_loss_rho"),
    ]:
        eligible_ref = [ref_values[i] for i in eligible_indices]
        eligible_gains_ref = [pe_gain_values[i] for i in eligible_indices]
        ref_rhos = []
        for start_idx in range(0, len(eligible_ref), window_size):
            end_idx = min(start_idx + window_size, len(eligible_ref))
            if end_idx - start_idx < 4:
                continue
            rx = eligible_ref[start_idx:end_idx]
            ry = eligible_gains_ref[start_idx:end_idx]
            min_len = min(len(rx), len(ry))
            rho, _ = spearmanr(rx[:min_len], ry[:min_len])
            if rho is not None and not math.isnan(rho):
                ref_rhos.append(float(rho))
        if ref_rhos:
            setattr(result, attr, float(np.mean(ref_rhos)))

    return result


def _check_provenance():
    """F11: refuse to run unless tree is clean and HEAD == v4-prereg-r3."""
    repo_root = Path(__file__).resolve().parents[1]
    status = subprocess.run(
        ["git", "status", "--porcelain"], capture_output=True, text=True,
        cwd=str(repo_root),
    )
    if status.stdout.strip():
        print(f"FATAL: git tree is dirty. Commit or stash first.\n{status.stdout}")
        sys.exit(1)
    try:
        tag_sha = subprocess.run(
            ["git", "rev-parse", "v4-prereg-r3"],
            capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
        head_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=str(repo_root)
        ).stdout.strip()
    except Exception:
        print("FATAL: could not resolve v4-prereg-r3 tag.")
        sys.exit(1)
    if head_sha != tag_sha:
        print(f"FATAL: HEAD ({head_sha}) != v4-prereg-r3 ({tag_sha}).")
        sys.exit(1)
    return head_sha


def main():
    parser = argparse.ArgumentParser(description="Stage 0: oracle probe")
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 103])
    parser.add_argument("--rate", type=float, default=0.60)
    parser.add_argument("--total-steps", type=int, default=6548)
    parser.add_argument("--model", type=str, default="jhu-clsp/ettin-encoder-150m")
    parser.add_argument("--model-rev", type=str, default="45d08642849e5c5701b162671ac811b7654bfd9f")
    parser.add_argument("--probe-file", type=str,
                        default="experiments/probe_indices_v4.json")
    parser.add_argument("--log-interval", type=int, default=8)
    parser.add_argument("--probe-refresh", type=int, default=512)
    parser.add_argument("--output-dir", type=str, default="/raid/stage0_oracle_probe/")
    args = parser.parse_args()

    # F11: provenance check
    run_sha = _check_provenance()

    # B9: ensure repo root is on sys.path so `import lerna` works
    repo_root = str(Path(__file__).resolve().parents[1])
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # F6: verify probe file hash matches pre-reg
    with open(args.probe_file) as f:
        probe_data = json.load(f)
    # B1: Verify ids_sha256 (content hash of the probe ids), not the file bytes
    # (a file cannot contain its own hash).
    ids_json = json.dumps({
        "probe_s_ids": probe_data["probe_s_ids"],
        "probe_e_ids": probe_data["probe_e_ids"],
    }, sort_keys=True, separators=(",", ":"))
    ids_sha = hashlib.sha256(ids_json.encode()).hexdigest()
    expected_ids_sha = probe_data.get("ids_sha256")
    if expected_ids_sha and ids_sha != expected_ids_sha:
        print(f"FATAL: probe ids hash mismatch. Expected {expected_ids_sha}, got {ids_sha}")
        sys.exit(1)

    # B1: Assert probe set sizes and disjointness
    assert len(probe_data["probe_s_ids"]) == 256, \
        f"FATAL: probe_s_ids must have exactly 256 entries, got {len(probe_data['probe_s_ids'])}"
    assert len(probe_data["probe_e_ids"]) == 256, \
        f"FATAL: probe_e_ids must have exactly 256 entries, got {len(probe_data['probe_e_ids'])}"
    assert set(probe_data["probe_s_ids"]).isdisjoint(probe_data["probe_e_ids"]), \
        "FATAL: P_s and P_e probe index sets overlap"

    results = {}
    for seed in args.seeds:
        print(f"\n=== Stage 0: seed={seed} ===")
        result = run_stage0(
            seed=seed,
            model_name=args.model,
            model_rev=args.model_rev,
            total_steps=args.total_steps,
            rate=args.rate,
            probe_indices_file=args.probe_file,
            log_interval=args.log_interval,
            probe_refresh=args.probe_refresh,
            output_dir=args.output_dir,
        )
        results[seed] = {
            "s1_spearman": result.s1_spearman,
            "s1_ci_per_seed": list(result.s1_ci_per_seed),
            "s1_mean_rho": result.s1_mean_rho,
            "s2_spearman": result.s2_spearman,
            "s2_ci_per_seed": list(result.s2_ci_per_seed),
            "s2_mean_rho": result.s2_mean_rho,
            "log_steps": result.log_steps,
            "step_index_rho": result.step_index_rho,
            "batch_loss_rho": result.batch_loss_rho,
        }
        print(f"  S1 mean ρ: {result.s1_mean_rho}")
        print(f"  S2 mean ρ: {result.s2_mean_rho}")

    # F2: selection = mean of per-window rho, pooled over seeds, only in steps 392-5761
    # Ties go to S1.
    s1_mean = np.nanmean([r["s1_mean_rho"] for r in results.values()])
    s2_mean = np.nanmean([r["s2_mean_rho"] for r in results.values()])

    selected = "s1"  # ties go to S1
    if not math.isnan(s2_mean) and (math.isnan(s1_mean) or s2_mean > s1_mean):
        selected = "s2"

    output = {
        "stage": "stage0_oracle_probe",
        "seeds": args.seeds,
        "rate": args.rate,
        "total_steps": args.total_steps,
        "model": args.model,
        "model_rev": args.model_rev,
        "probe_file": args.probe_file,
        "probe_ids_sha256": ids_sha,
        "run_sha": run_sha,
        "selected_score": selected,
        "s1_pooled_mean_rho": float(s1_mean) if not math.isnan(s1_mean) else None,
        "s2_pooled_mean_rho": float(s2_mean) if not math.isnan(s2_mean) else None,
        "results_per_seed": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "gate_a_decision.json"), "w") as f:
        json.dump(output, f, indent=2)
    with open(os.path.join(args.output_dir, "manifest.json"), "w") as f:
        json.dump({
            "run_sha": run_sha,
            "probe_ids_sha256": ids_sha,
            "selected_score": selected,
            "seeds": args.seeds,
            "rate": args.rate,
            "total_steps": args.total_steps,
        }, f, indent=2)

    print(f"\n=== Stage 0 complete ===")
    print(f"  S1 pooled mean ρ: {s1_mean}")
    print(f"  S2 pooled mean ρ: {s2_mean}")
    print(f"  Selected score: {selected}")
    print(f"  Probe SHA256: {ids_sha}")


if __name__ == "__main__":
    main()
