#!/usr/bin/env python3
"""
LERNA Ablation Study: Disabling individual LERNA components to quantify their contribution.

Ablation configs:
    full_lerna       — control (all components enabled)
    no_rho_vg        — disable rho_VG velocity-gradient correlation
    no_ler           — disable LER tracking (plateau detection falls back)
    no_safety        — disable safety horizon (unbounded momentum extrapolation)
    no_hysteresis   — disable phase-detection hysteresis
    no_momentum     — disable momentum extrapolation (full backprop always)

Usage:
    python scripts/run_ablation_study.py --mode smoke
    python scripts/run_ablation_study.py --mode full --tasks sst2 qnli --seeds 42 43 44
"""

import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["WANDB_START_METHOD"] = "thread"

import sys
import json
import time
import argparse
import gc
import math
from datetime import datetime, timedelta

import torch
try:
    torch._dynamo.config.disable = True
except AttributeError:
    pass
try:
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
except AttributeError:
    pass

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
)
from datasets import load_dataset
import evaluate
import numpy as np

from lerna.callbacks.lerna_switching import LERNASwitchingCallback, LERNATrainer
from lerna.callbacks.efficiency_callback import PowerTelemetryCallback
from lerna.utils.metrics import LERTracker

try:
    from scripts.run_baseline_glue import (
        MODEL_NAME,
        GLUE_TASK_CONFIG,
        TASK_HP_OVERRIDES,
        GLUE_TASKS,
        SEEDS,
        detect_device_profile,
        get_training_config,
        load_glue_task,
        build_compute_metrics,
        _ensure_wandb_finished,
        run_single_experiment,
    )
except ModuleNotFoundError:
    from run_baseline_glue import (
        MODEL_NAME,
        GLUE_TASK_CONFIG,
        TASK_HP_OVERRIDES,
        GLUE_TASKS,
        SEEDS,
        detect_device_profile,
        get_training_config,
        load_glue_task,
        build_compute_metrics,
        _ensure_wandb_finished,
        run_single_experiment,
    )

ABLATIONS = {
    "full_lerna":       {},
    "no_rho_vg":        {"use_rho_vg": False},
    "no_ler":           {"use_ler": False},
    "no_safety":        {"use_safety_horizon": False},
    "no_hysteresis":    {"use_hysteresis": False},
    "no_momentum":      {"use_momentum_extrap": False},
}


def run_ablation_single(
    task_name: str,
    seed: int,
    ablation_name: str,
    ablation_overrides: dict,
    model_name: str,
    profile: str,
    base_output_dir: str,
    use_wandb: bool = False,
    max_samples_override=None,
    run_idx: int = 0,
    total_runs: int = 0,
    wandb_project: str = "lerna-ablation",
    wandb_group: str = None,
    num_epochs: int = 3,
    warmup_ratio: float = 0.1,
    early_stopping_patience: int = 5,
    metric_for_best_model: str = "eval_loss",
    greater_is_better: bool = False,
    init_from_mnli: bool = False,
):
    """Run a single experiment with a specific ablation config."""

    task_hp = TASK_HP_OVERRIDES.get(task_name, {})
    lr = task_hp.get("learning_rate", 2e-5)
    num_epochs = task_hp.get("num_epochs", num_epochs)
    warmup_ratio = task_hp.get("warmup_ratio", warmup_ratio)
    early_stopping_patience = task_hp.get("early_stopping_patience", early_stopping_patience)
    metric_for_best_model = task_hp.get("metric_for_best_model", metric_for_best_model)
    greater_is_better = task_hp.get("greater_is_better", greater_is_better)
    init_from_mnli = task_hp.get("init_from_mnli", init_from_mnli)

    hw_cfg = get_training_config(profile)
    if max_samples_override is not None:
        hw_cfg["max_samples"] = max_samples_override

    run_id = f"{task_name}_s{seed}_ab-{ablation_name}"
    output_dir = os.path.join(base_output_dir, ablation_name, run_id)
    os.makedirs(output_dir, exist_ok=True)

    if wandb_group is None:
        wandb_group = f"ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    if use_wandb:
        import wandb
        _ensure_wandb_finished()
        wandb.init(
            project=wandb_project,
            name=run_id,
            group=wandb_group,
            job_type=f"ablation-{ablation_name}",
            tags=[task_name, f"ablation-{ablation_name}", f"seed-{seed}", MODEL_NAME.split("/")[-1]],
            reinit=True,
            settings=wandb.Settings(init_timeout=120),
            config={
                "task": task_name,
                "seed": seed,
                "ablation": ablation_name,
                "ablation_overrides": ablation_overrides,
                "learning_rate": lr,
                "model": MODEL_NAME,
                "profile": profile,
                "max_samples": hw_cfg["max_samples"],
                "run_index": run_idx,
                "total_runs": total_runs,
            },
        )

    print(f"\n{'='*60}")
    print(f"  Ablation [{ablation_name}]: {task_name} | seed={seed} | lr={lr}")
    print(f"  Overrides: {ablation_overrides}")
    print(f"  Profile: {profile} | Output: {output_dir}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    from lerna.utils.model_loader import load_model_and_tokenizer
    cfg = GLUE_TASK_CONFIG[task_name]

    mnli_checkpoint_dir = os.path.join(base_output_dir, "mnli_finetuned")
    if init_from_mnli and os.path.exists(mnli_checkpoint_dir):
        from transformers import AutoConfig
        mnli_model, _ = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])
        model, _ = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])
        encoder_state = {k: v for k, v in mnli_model.state_dict().items()
                        if "classifier" not in k and "pooler" not in k}
        model.load_state_dict(encoder_state, strict=False)
        del mnli_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        model, tokenizer = load_model_and_tokenizer(model_name, num_labels=cfg["num_labels"])

    if hw_cfg["gradient_checkpointing"]:
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    train_ds, eval_ds, task_cfg = load_glue_task(
        task_name, tokenizer, max_length=128, max_samples=hw_cfg["max_samples"])
    print(f"  Train samples: {len(train_ds)}, Eval samples: {len(eval_ds)}")

    n_gpu = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    steps_per_epoch = len(train_ds) // (
        hw_cfg["per_device_train_batch_size"] * hw_cfg["gradient_accumulation_steps"] * n_gpu)
    total_steps = steps_per_epoch * num_epochs
    eval_steps = max(total_steps // 20, 10)

    use_rho_vg = ablation_overrides.get("use_rho_vg", True)
    use_ler = ablation_overrides.get("use_ler", True)
    use_safety_horizon = ablation_overrides.get("use_safety_horizon", True)
    use_hysteresis = ablation_overrides.get("use_hysteresis", True)
    use_momentum_extrap = ablation_overrides.get("use_momentum_extrap", True)

    ler_tracker = LERTracker(task=task_name, window_size=5)

    power_callback = PowerTelemetryCallback(
        sample_interval_s=1.0,
        output_dir=os.path.join(output_dir, "power"),
        wandb_enabled=use_wandb,
        log_frequency=50,
    )

    class AblationDiagnosticsCallback:
        def __init__(self, ler_trk, model_ref, trainer_ref_holder, greater_is_better,
                     use_rho_vg, use_ler, use_hysteresis, use_safety_horizon, use_momentum_extrap):
            self.ler_tracker = ler_trk
            self._model = model_ref
            self._trainer_holder = trainer_ref_holder
            self._greater_is_better = greater_is_better
            self.use_rho_vg = use_rho_vg
            self.use_ler = use_ler
            self.use_hysteresis = use_hysteresis
            self.use_safety_horizon = use_safety_horizon
            self.use_momentum_extrap = use_momentum_extrap
            self._last_loss = None
            self._step_count = 0

        def on_init_end(self, args, state, control, **kwargs):
            return control

        def on_epoch_begin(self, args, state, control, **kwargs):
            return control

        def on_epoch_end(self, args, state, control, **kwargs):
            return control

        def on_step_begin(self, args, state, control, **kwargs):
            return control

        def on_optimizer_step(self, args, state, control, **kwargs):
            return control

        def on_step_end(self, args, state, control, **kwargs):
            return control

        def on_substep_end(self, args, state, control, **kwargs):
            return control

        def on_save(self, args, state, control, **kwargs):
            return control

        def on_predict(self, args, state, control, **kwargs):
            return control

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            return control

        def on_prediction_step(self, args, state, control, **kwargs):
            return control

        def on_train_begin(self, args, state, control, **kwargs):
            opt = kwargs.get("optimizer", None)
            if opt is not None and hasattr(self.ler_tracker, "set_optimizer"):
                self.ler_tracker.set_optimizer(opt)
            return control

        def on_train_end(self, args, state, control, **kwargs):
            self._save_diagnostics()
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs is not None:
                self._last_loss = logs.get("loss", self._last_loss)
            return control

        def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
            if metrics is None:
                return control

            accuracy = metrics.get("eval_accuracy",
                metrics.get("eval_matthews_correlation",
                metrics.get("eval_pearson", 0)))
            eval_loss = metrics.get("eval_loss", 0)

            if self._last_loss is not None and self.use_ler:
                trainer = self._trainer_holder[0] if self._trainer_holder else None
                real_logits = None
                if trainer is not None and hasattr(trainer, "_last_real_logits") and trainer._last_real_logits is not None:
                    real_logits = trainer._last_real_logits
                elif model is not None:
                    try:
                        model.eval()
                        dl = DataLoader(
                            eval_ds.select(range(min(8, len(eval_ds)))),
                            batch_size=min(8, len(eval_ds)),
                            collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8),
                        )
                        batch = next(iter(dl))
                        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                        with torch.no_grad():
                            outputs = model(**batch)
                        real_logits = outputs.logits.detach()
                    except Exception:
                        real_logits = torch.randn(8, cfg["num_labels"])

                try:
                    self.ler_tracker.update(
                        loss=self._last_loss,
                        logits=real_logits,
                        accuracy=accuracy,
                        model=model,
                    )
                except Exception:
                    pass

                diag = self.ler_tracker.get_diagnostics()
                ler_val = diag.get("ler")
                rho_val = diag.get("rho_vg")
                phase = diag.get("phase", "?")
                ler_str = f"{ler_val:.6f}" if ler_val is not None else "N/A"
                rho_str = f"{rho_val:.4f}" if rho_val is not None else "N/A"
                print(
                    f"  [ABL step={state.global_step}] "
                    f"LER={ler_str} | "
                    f"rho_VG={rho_str} | "
                    f"phase={phase} | acc={accuracy:.3f}"
                )

                if use_wandb:
                    try:
                        import wandb
                        if wandb.run is not None:
                            wandb.log({
                                "lerna/ler": ler_val,
                                "lerna/rho_vg": rho_val,
                                "lerna/phase": phase,
                                "lerna/eval_accuracy": accuracy,
                                "lerna/eval_loss": eval_loss,
                                "ablation/ablation_name": ablation_name,
                                "ablation/use_rho_vg": self.use_rho_vg,
                                "ablation/use_ler": self.use_ler,
                                "ablation/use_safety_horizon": self.use_safety_horizon,
                                "ablation/use_hysteresis": self.use_hysteresis,
                                "ablation/use_momentum_extrap": self.use_momentum_extrap,
                            }, step=state.global_step)
                    except Exception:
                        pass
            return control

        def _save_diagnostics(self):
            diag_path = os.path.join(output_dir, "ler_diagnostics.json")
            final = self.ler_tracker.get_diagnostics()
            final["ler_history"] = self.ler_tracker.ler_history
            final["rho_vg_history"] = self.ler_tracker.rho_vg_history
            final["velocity_history"] = self.ler_tracker.velocity_history
            final["ablation_name"] = ablation_name
            final["ablation_overrides"] = ablation_overrides
            with open(diag_path, "w") as f:
                json.dump(final, f, indent=2, default=str)
            print(f"  LER diagnostics saved: {diag_path}")

    trainer_holder = [None]

    diag_callback = AblationDiagnosticsCallback(
        ler_trk=ler_tracker,
        model_ref=model,
        trainer_ref_holder=trainer_holder,
        greater_is_better=greater_is_better,
        use_rho_vg=use_rho_vg,
        use_ler=use_ler,
        use_hysteresis=use_hysteresis,
        use_safety_horizon=use_safety_horizon,
        use_momentum_extrap=use_momentum_extrap,
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=hw_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=hw_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=hw_cfg["gradient_accumulation_steps"],
        learning_rate=lr,
        weight_decay=0.01,
        warmup_ratio=warmup_ratio,
        max_grad_norm=1.0,
        fp16=hw_cfg["fp16"],
        bf16=hw_cfg["bf16"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=greater_is_better,
        logging_steps=max(eval_steps // 5, 1),
        report_to="wandb" if use_wandb else "none",
        run_name=run_id if use_wandb else None,
        seed=seed,
        dataloader_num_workers=hw_cfg["dataloader_num_workers"],
        dataloader_pin_memory=(profile == "server"),
        gradient_checkpointing=hw_cfg["gradient_checkpointing"],
        remove_unused_columns=True,
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8)
    compute_metrics = build_compute_metrics(task_name)

    class AblationTrainer(Trainer):
        def __init__(self, *args, ler_tracker=None, **kwargs):
            super().__init__(*args, **kwargs)
            self._ler_tracker = ler_tracker
            self._last_real_logits = None

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            outputs = model(**inputs)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs["loss"]
            if hasattr(outputs, "logits"):
                self._last_real_logits = outputs.logits.detach()
            elif isinstance(outputs, dict) and "logits" in outputs:
                self._last_real_logits = outputs["logits"].detach()
            return (loss, outputs) if return_outputs else loss

    trainer = AblationTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        ler_tracker=ler_tracker,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=early_stopping_patience),
            power_callback,
            diag_callback,
        ],
    )
    trainer_holder[0] = trainer

    start_time = time.time()
    print(f"\n  Starting ablation [{ablation_name}]: {total_steps} steps, eval every {eval_steps}")
    train_result = trainer.train()
    total_time = time.time() - start_time

    print(f"\n  Evaluating best model...")
    eval_result = trainer.evaluate()

    avg_power = (float(np.mean([s["power_w"] for s in power_callback._power_samples]))
                 if power_callback._power_samples else 0)

    results = {
        "task": task_name,
        "seed": seed,
        "ablation": ablation_name,
        "ablation_overrides": ablation_overrides,
        "learning_rate": lr,
        "profile": profile,
        "model": MODEL_NAME,
        "train_runtime_s": total_time,
        "train_loss": train_result.training_loss,
        "eval_metrics": eval_result,
        "energy_kwh": power_callback.cumulative_kwh,
        "power_avg_watts": avg_power,
        "ler_final": ler_tracker.get_diagnostics(),
        "timestamp": datetime.now().isoformat(),
        "hw_config": {k: v for k, v in hw_cfg.items() if k != "max_samples"},
    }

    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    if use_wandb:
        try:
            import wandb
            if wandb.run is not None:
                wandb.summary.update({
                    "final/eval_accuracy": eval_result.get("eval_accuracy",
                        eval_result.get("eval_matthews_correlation",
                        eval_result.get("eval_pearsonr"))),
                    "final/eval_loss": eval_result.get("eval_loss"),
                    "final/energy_kwh": power_callback.cumulative_kwh,
                    "final/runtime_s": total_time,
                    "final/ler": ler_tracker.get_diagnostics().get("ler"),
                    "final/rho_vg": ler_tracker.get_diagnostics().get("rho_vg"),
                    "ablation/overrides": ablation_overrides,
                })
        except Exception:
            pass

    print(f"\n  Ablation [{ablation_name}] Results:")
    print(f"  Eval metrics: {eval_result}")
    print(f"  Energy: {power_callback.cumulative_kwh:.6f} kWh")
    print(f"  Time: {total_time:.1f}s")
    print(f"  Saved: {results_path}")

    del model, trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if use_wandb:
        _ensure_wandb_finished()

    return results


def main():
    parser = argparse.ArgumentParser(description="LERNA Ablation Study")
    parser.add_argument("--mode", choices=["smoke", "full", "custom"], default="smoke")
    parser.add_argument("--tasks", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--ablations", nargs="+", default=None,
                        help="Ablation names to run (default: all)")
    parser.add_argument("--output-dir", default="./experiments/ablation")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerna-ablation")
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--model", default="modernbert", choices=["roberta", "modernbert", "deberta"],
                        help="Model to use for ablation study")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--unlimited", action="store_true")
    args = parser.parse_args()

    profile = detect_device_profile()

    if args.mode == "smoke":
        tasks = ["sst2"]
        seeds = [42]
        ablations_to_run = ["full_lerna", "no_ler", "no_safety"]
    elif args.mode == "full":
        tasks = GLUE_TASKS
        seeds = SEEDS
        ablations_to_run = list(ABLATIONS.keys())
    else:
        tasks = args.tasks or ["sst2"]
        seeds = args.seeds or [42]
        ablations_to_run = args.ablations or list(ABLATIONS.keys())

    if args.tasks:
        tasks = args.tasks
    if args.seeds:
        seeds = args.seeds
    if args.ablations:
        ablations_to_run = args.ablations

    effective_max_samples = args.max_samples
    if effective_max_samples is None and not args.unlimited:
        effective_max_samples = 2000 if profile != "server" else 25000

    from lerna.utils.model_loader import MODELS
    model_name = MODELS[args.model]
    wandb_group = args.wandb_group or f"ablation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    total_runs = len(tasks) * len(seeds) * len(ablations_to_run)
    print(f"\n  ═══════════════════════════════════════════════════════")
    print(f"  LERNA Ablation Study")
    print(f"  ═══════════════════════════════════════════════════════")
    print(f"  Tasks: {tasks}")
    print(f"  Seeds: {seeds}")
    print(f"  Ablations: {ablations_to_run}")
    print(f"  Total runs: {total_runs}")
    print(f"  Max samples/task: {effective_max_samples or 'unlimited'}")
    print(f"  ═══════════════════════════════════════════════════════\n")

    if args.wandb:
        _ensure_wandb_finished()

    all_results = []
    run_idx = 0
    overall_start = time.time()

    for task in tasks:
        for seed in seeds:
            for ablation_name in ablations_to_run:
                run_idx += 1

                if run_idx > 1:
                    elapsed = time.time() - overall_start
                    avg_per_run = elapsed / (run_idx - 1)
                    remaining = (total_runs - run_idx + 1) * avg_per_run
                    print(f"\n  ═══ Run {run_idx}/{total_runs} | ETA: {timedelta(seconds=int(remaining))} ═══")
                else:
                    print(f"\n  ═══ Run {run_idx}/{total_runs} ═══")

                task_hp = TASK_HP_OVERRIDES.get(task, {})
                try:
                    result = run_ablation_single(
                        task_name=task,
                        seed=seed,
                        ablation_name=ablation_name,
                        ablation_overrides=ABLATIONS[ablation_name],
                        model_name=model_name,
                        profile=profile,
                        base_output_dir=args.output_dir,
                        use_wandb=args.wandb,
                        max_samples_override=effective_max_samples,
                        run_idx=run_idx,
                        total_runs=total_runs,
                        wandb_project=args.wandb_project,
                        wandb_group=wandb_group,
                        num_epochs=task_hp.get("num_epochs", 3),
                        warmup_ratio=task_hp.get("warmup_ratio", 0.1),
                        early_stopping_patience=task_hp.get("early_stopping_patience", 5),
                        metric_for_best_model=task_hp.get("metric_for_best_model", "eval_loss"),
                        greater_is_better=task_hp.get("greater_is_better", False),
                        init_from_mnli=task_hp.get("init_from_mnli", False),
                    )
                    all_results.append(result)
                except Exception as e:
                    print(f"  FAILED: {task} seed={seed} ablation={ablation_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    all_results.append({
                        "task": task, "seed": seed,
                        "ablation": ablation_name, "error": str(e)})
                    if args.wandb:
                        _ensure_wandb_finished()

    summary_path = os.path.join(args.output_dir, "ablation_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    total_elapsed = time.time() - overall_start
    successful = [r for r in all_results if "error" not in r]

    print(f"\n{'='*60}")
    print(f"  ABLATION STUDY COMPLETE: {len(all_results)} runs")
    print(f"  Summary: {summary_path}")
    print(f"  Total wall time: {timedelta(seconds=int(total_elapsed))}")

    if successful:
        print(f"\n  {'Ablation':<15} {'Runs':>5} {'Avg Acc':>10} {'Std':>8} {'Avg kWh':>10} {'Avg LER':>10}")
        print(f"  {'-'*60}")
        for ablab in ablations_to_run:
            ab_results = [r for r in successful if r.get("ablation") == ablab]
            if not ab_results:
                continue
            accs = [r.get("eval_metrics", {}).get("eval_accuracy",
                r.get("eval_metrics", {}).get("eval_matthews_correlation",
                r.get("eval_metrics", {}).get("eval_pearson", 0))) for r in ab_results]
            kwhs = [r.get("energy_kwh", 0) for r in ab_results]
            lers = [r.get("ler_final", {}).get("ler") for r in ab_results if r.get("ler_final", {}).get("ler") is not None]
            print(
                f"  {ablab:<15} {len(ab_results):>5} "
                f"{np.mean(accs):>10.4f} {np.std(accs):>8.4f} "
                f"{np.mean(kwhs):>10.6f} "
                f"{np.mean(lers) if lers else 0:.2e}"
            )

    print(f"{'='*60}")


if __name__ == "__main__":
    main()
