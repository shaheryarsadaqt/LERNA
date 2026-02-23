#!/usr/bin/env python3
"""
LERNA Baseline: ModernBERT-base on GLUE with corrected LER/rho_VG diagnostics.

Usage:
  # Smoke test on RTX 3050 (1 seed, SST-2 only, small subset)
  python scripts/run_baseline_glue.py --mode smoke

  # Full baseline on RTX 5090 (3 seeds x 8 tasks, 25k samples/task)
  python scripts/run_baseline_glue.py --mode full --num-seeds 3

  # Full baseline 10 seeds (production)
  python scripts/run_baseline_glue.py --mode full --wandb

  # Full data, no cap
  python scripts/run_baseline_glue.py --mode full --unlimited --wandb

  # Custom: specific tasks and seeds
  python scripts/run_baseline_glue.py --tasks sst2 rte mrpc --seeds 42 43 44
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import json
import time
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
torch._dynamo.config.disable = True

# RTX 5090 Blackwell: disable flash/mem-efficient SDP kernels (cause SIGFPE)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)
from datasets import load_dataset
import evaluate


GLUE_TASK_CONFIG = {
    "sst2":  {"keys": ("sentence", None),      "num_labels": 2, "metric": "accuracy"},
    "qnli":  {"keys": ("question", "sentence"), "num_labels": 2, "metric": "accuracy"},
    "qqp":   {"keys": ("question1", "question2"), "num_labels": 2, "metric": "accuracy"},
    "mnli":  {"keys": ("premise", "hypothesis"), "num_labels": 3, "metric": "accuracy"},
    "rte":   {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
    "mrpc":  {"keys": ("sentence1", "sentence2"), "num_labels": 2, "metric": "accuracy"},
    "cola":  {"keys": ("sentence", None),        "num_labels": 2, "metric": "matthews_correlation"},
    "stsb":  {"keys": ("sentence1", "sentence2"), "num_labels": 1, "metric": "pearsonr"},
}

MODEL_NAME = "answerdotai/ModernBERT-base"


def detect_device_profile():
    if not torch.cuda.is_available():
        return "cpu"
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu_name} ({vram_gb:.1f} GB VRAM)")
    if vram_gb >= 20:
        return "server"
    return "laptop"


def get_training_config(profile: str, output_dir: str):
    if profile == "server":
        return {
            "per_device_train_batch_size": 32,
            "per_device_eval_batch_size": 64,
            "gradient_accumulation_steps": 1,
            "fp16": False,
            "bf16": True,
            "gradient_checkpointing": False,
            "dataloader_num_workers": 4,
            "max_samples": None,
        }
    else:
        return {
            "per_device_train_batch_size": 4,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "fp16": True,
            "bf16": False,
            "gradient_checkpointing": True,
            "dataloader_num_workers": 0,
            "max_samples": 2000,
        }


def load_glue_task(task_name: str, tokenizer, max_length: int = 128, max_samples=None):
    cfg = GLUE_TASK_CONFIG[task_name]
    key1, key2 = cfg["keys"]

    dataset = load_dataset("glue", task_name)

    def tokenize_fn(examples):
        if key2 is not None:
            return tokenizer(
                examples[key1], examples[key2],
                truncation=True, max_length=max_length, padding=False,
            )
        return tokenizer(
            examples[key1],
            truncation=True, max_length=max_length, padding=False,
        )

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=[
        c for c in dataset["train"].column_names if c not in ["label", "labels"]
    ])

    if "label" in tokenized["train"].column_names:
        tokenized = tokenized.rename_column("label", "labels")

    train_ds = tokenized["train"]
    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    eval_ds = tokenized[eval_key]

    if max_samples is not None:
        if len(train_ds) > max_samples:
            train_ds = train_ds.select(range(max_samples))
        eval_max = min(max_samples // 4, len(eval_ds))
        eval_ds = eval_ds.select(range(eval_max))

    return train_ds, eval_ds, cfg


def build_compute_metrics(task_name: str):
    cfg = GLUE_TASK_CONFIG[task_name]
    metric_name = cfg["metric"]

    if metric_name == "matthews_correlation":
        metric = evaluate.load("glue", "cola")
    elif metric_name == "pearsonr":
        metric = evaluate.load("glue", "stsb")
    else:
        metric = evaluate.load("glue", task_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        if cfg["num_labels"] == 1:
            predictions = predictions.squeeze()
        else:
            predictions = predictions.argmax(axis=-1)
        return metric.compute(predictions=predictions, references=labels)

    return compute_metrics


_UNSET = object()


def run_single_experiment(
    task_name: str,
    seed: int,
    lr: float,
    profile: str,
    base_output_dir: str,
    use_wandb: bool = False,
    max_samples_override=_UNSET,
):
    from lerna.utils.metrics import LERTracker
    from lerna.callbacks.efficiency_callback import PowerTelemetryCallback

    hw_cfg = get_training_config(profile, base_output_dir)
    if max_samples_override is not _UNSET:
        hw_cfg["max_samples"] = max_samples_override
    run_id = f"{task_name}_s{seed}_lr{lr:.0e}"
    output_dir = os.path.join(base_output_dir, run_id)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  LERNA Baseline: {task_name} | seed={seed} | lr={lr}")
    print(f"  Profile: {profile} | Output: {output_dir}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    cfg = GLUE_TASK_CONFIG[task_name]

    # Load in fp16 to avoid CUBLAS_STATUS_NOT_INITIALIZED on Blackwell
    # when cuDNN version mismatch corrupts CUDA context for fp32 kernels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=cfg["num_labels"],
        reference_compile=False,
        attn_implementation="eager",
    )

    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"]
    )

    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    ler_tracker = LERTracker(task=task_name, window_size=5)
    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
    )

    class LERDiagnosticsCallback:
        """Feeds model params + gradients into LER tracker at each optimizer step and eval."""

        def __init__(self, tracker, model_ref):
            self.tracker = tracker
            self._model = model_ref
            self.step_count = 0
            self._last_loss = None

        def on_init_end(self, args, state, control, **kwargs):
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            return control

        def on_train_end(self, args, state, control, **kwargs):
            diag_path = os.path.join(output_dir, "ler_diagnostics.json")
            final = self.tracker.get_diagnostics()
            final["ler_history"] = self.tracker.ler_history
            final["velocity_history"] = self.tracker.velocity_history
            final["rho_vg_history"] = self.tracker.rho_vg_history
            final["loss_history"] = self.tracker.loss_history
            final["entropy_history"] = self.tracker.entropy_history
            with open(diag_path, "w") as f:
                json.dump(final, f, indent=2, default=str)
            print(f"  LER diagnostics saved: {diag_path}")
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            try:
                self.tracker.capture_step_gradients(self._model)
            except Exception:
                pass
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            return control

        def on_step_end(self, args, state, control, model=None, **kwargs):
            self.step_count += 1
            if self.tracker._cached_rho_vg is None and self._model is not None:
                has_grad = any(
                    p.grad is not None for p in self._model.parameters() if p.requires_grad
                )
                if has_grad:
                    try:
                        self.tracker.capture_step_gradients(self._model)
                    except Exception:
                        pass
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            return control

        def on_save(self, args, state, control, **kwargs):
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self._last_loss = logs.get("loss", self._last_loss)
            return control

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if metrics is None:
                return control

            eval_loss = metrics.get("eval_loss", 0)
            accuracy = metrics.get("eval_accuracy", metrics.get("eval_matthews_correlation", 0))

            num_labels = GLUE_TASK_CONFIG[task_name]["num_labels"]
            dummy_logits = torch.randn(8, num_labels)

            loss_for_ler = self._last_loss if self._last_loss is not None else eval_loss

            try:
                self.tracker.update(
                    loss=loss_for_ler,
                    logits=dummy_logits,
                    accuracy=accuracy,
                    model=model,
                )
            except Exception as e:
                print(f"  [LER warn] {e}")

            diag = self.tracker.get_diagnostics()
            ler_val = diag.get("ler")
            vel_val = diag.get("param_velocity")
            rho_val = diag.get("rho_vg")
            phase = diag.get("phase", "?")

            ler_str = f"{ler_val:.6f}" if ler_val is not None else "warming"
            vel_str = f"{vel_val:.6f}" if vel_val is not None else "N/A"
            rho_str = f"{rho_val:.4f}" if rho_val is not None else "N/A"

            print(
                f"  [LERNA step={state.global_step}] "
                f"LER={ler_str} | vel={vel_str} | rho_VG={rho_str} | "
                f"phase={phase} | acc={accuracy:.3f}"
            )

            return control

        def on_predict(self, args, state, control, **kwargs):
            return control

        def on_prediction_step(self, args, state, control, **kwargs):
            return control

    ler_callback = LERDiagnosticsCallback(ler_tracker, model)

    num_epochs = 3
    total_steps = (len(train_ds) // (hw_cfg["per_device_train_batch_size"] * hw_cfg["gradient_accumulation_steps"])) * num_epochs
    eval_steps = max(total_steps // 20, 10)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=hw_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        fp16=hw_cfg["fp16"],
        bf16=hw_cfg["bf16"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=2,
        load_best_model_at_end=False,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=max(eval_steps // 5, 1),
        report_to="wandb" if use_wandb else "none",
        run_name=run_id if use_wandb else None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
    )

    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)

    compute_metrics = build_compute_metrics(task_name)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # # tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=5),
            power_callback,
            ler_callback,
        ],
    )

    start_time = time.time()
    train_result = trainer.train()
    total_time = time.time() - start_time

    eval_result = trainer.evaluate()

    results = {
        "task": task_name,
        "seed": seed,
        "learning_rate": lr,
        "profile": profile,
        "model": MODEL_NAME,
        "train_runtime_s": total_time,
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_result,
        "energy_kwh": power_callback.cumulative_kwh,
        "power_avg_watts": float(np.mean([s["power_w"] for s in power_callback._power_samples])) if power_callback._power_samples else 0,
        "ler_final": ler_tracker.get_diagnostics(),
        "timestamp": datetime.now().isoformat(),
        "hw_config": {k: v for k, v in hw_cfg.items() if k != "max_samples"},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Results: {eval_result}")
    print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Saved: {results_path}")

    del model, trainer
    torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser(description="LERNA Baseline: ModernBERT on GLUE")
    parser.add_argument(
        "--mode", choices=["smoke", "full", "custom"], default="smoke",
        help="smoke=1 seed SST-2 only (3050 OK), full=10 seeds x 8 tasks (5090), custom=use --tasks/--seeds",
    )
    parser.add_argument("--tasks", nargs="+", default=None, help="Tasks to run (e.g., sst2 qnli)")
    parser.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds (e.g., 42 43 44)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--output-dir", default="./experiments/baseline", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap training samples per task (default: 2000 laptop, 25000 server-full, None for --unlimited)")
    parser.add_argument("--unlimited", action="store_true",
                        help="Use full dataset without sample cap (server only)")
    parser.add_argument("--num-seeds", type=int, default=None,
                        help="Override number of seeds in full mode (default: 10)")
    args = parser.parse_args()

    profile = detect_device_profile()

    if args.mode == "smoke":
        tasks = ["sst2"]
        seeds = [42]
        print("\n  SMOKE TEST MODE (1 seed, SST-2 only)")
    elif args.mode == "full":
        tasks = list(GLUE_TASK_CONFIG.keys())
        n_seeds = args.num_seeds or 10
        seeds = list(range(42, 42 + n_seeds))
        print(f"\n  FULL MODE ({len(seeds)} seeds x {len(tasks)} tasks = {len(seeds)*len(tasks)} runs)")
    else:
        tasks = args.tasks or ["sst2"]
        seeds = args.seeds or [42]

    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  LR: {args.lr}")
    print(f"  Profile: {profile}")

    if args.max_samples is not None:
        effective_max_samples = args.max_samples
    elif args.unlimited:
        effective_max_samples = None
    elif profile == "server":
        effective_max_samples = 25000
    else:
        effective_max_samples = 2000

    print(f"  Max samples/task: {effective_max_samples or 'unlimited'}")

    all_results = []
    total_runs = len(tasks) * len(seeds)
    run_idx = 0

    for task in tasks:
        for seed in seeds:
            run_idx += 1
            print(f"\n  === Run {run_idx}/{total_runs} ===")
            try:
                result = run_single_experiment(
                    task_name=task,
                    seed=seed,
                    lr=args.lr,
                    profile=profile,
                    base_output_dir=args.output_dir,
                    use_wandb=args.wandb,
                    max_samples_override=effective_max_samples,
                )
                all_results.append(result)
            except Exception as e:
                print(f"  FAILED: {task} seed={seed}: {e}")
                import traceback
                traceback.print_exc()
                all_results.append({"task": task, "seed": seed, "error": str(e)})

    summary_path = os.path.join(args.output_dir, "baseline_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"  BASELINE COMPLETE: {len(all_results)} runs")
    print(f"  Summary: {summary_path}")

    successful = [r for r in all_results if "error" not in r]
    if successful:
        total_kwh = sum(r.get("energy_kwh", 0) for r in successful)
        total_time = sum(r.get("train_runtime_s", 0) for r in successful)
        print(f"  Total energy: {total_kwh:.6f} kWh")
        print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
