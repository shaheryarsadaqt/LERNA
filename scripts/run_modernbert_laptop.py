#!/usr/bin/env python3
"""ModernBERT Fine-Tuning — RTX 3050 4GB (WSL2 Safe, No Triton Required)"""

# ══════════════════════════════════════════════════════════════════════
# MUST BE BEFORE ANY TORCH IMPORTS
# ══════════════════════════════════════════════════════════════════════
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

import sys
import math
import time
import copy

import torch
torch._dynamo.config.disable = True  # Belt + suspenders

from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset


def main():
    print("=" * 60)
    print("  ModernBERT Fine-Tuning — RTX 3050 4GB (WSL2 Safe)")
    print("=" * 60)

    assert torch.cuda.is_available(), "CUDA not available!"
    DEVICE = torch.device("cuda")

    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"VRAM:     {vram_gb:.1f} GB")
    print(f"PyTorch:  {torch.__version__}")
    print(f"Dynamo:   DISABLED ✅")

    # ─────────────────────────────────────────────
    # CUDA Performance Flags
    # ─────────────────────────────────────────────
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.cuda.empty_cache()

    # ─────────────────────────────────────────────
    # Hyperparameters
    # ─────────────────────────────────────────────
    MODEL_NAME    = "answerdotai/ModernBERT-base"
    BATCH_SIZE    = 2
    GRAD_ACCUM    = 8
    MAX_LEN       = 128
    EPOCHS        = 3
    LR            = 2e-5
    WEIGHT_DECAY  = 0.01
    WARMUP_RATIO  = 0.06
    MAX_GRAD_NORM = 1.0
    TRAIN_SAMPLES = 2000
    EVAL_SAMPLES  = 500
    PATIENCE      = 2

    # ─────────────────────────────────────────────
    # Model & Tokenizer (with compile disabled)
    # ─────────────────────────────────────────────
    print(f"\n🔧 Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "Negative", 1: "Positive"},
        label2id={"Negative": 0, "Positive": 1},
        reference_compile=False,       # ← Disable ModernBERT's internal compile
        attn_implementation="sdpa",    # ← Use PyTorch SDPA, not Flash/Triton
    )
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"✅ Loaded ({param_count / 1e6:.0f}M params)")

    # Gradient checkpointing
    try:
        model.gradient_checkpointing_enable()
        print("✅ Gradient checkpointing: enabled")
    except Exception:
        print("⚠️  Gradient checkpointing: not available")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    model.to(DEVICE)

    # ─────────────────────────────────────────────
    # Dataset
    # ─────────────────────────────────────────────
    print(f"\n📦 Loading IMDB ({TRAIN_SAMPLES} train / {EVAL_SAMPLES} eval)...")
    train_ds = load_dataset("imdb", split=f"train[:{TRAIN_SAMPLES}]")
    eval_ds  = load_dataset("imdb", split=f"test[:{EVAL_SAMPLES}]")

    def tokenize_fn(batch):
        out = tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)
        out["labels"] = batch["label"]
        return out

    collator = DataCollatorWithPadding(
        tokenizer=tokenizer, pad_to_multiple_of=8, return_tensors="pt"
    )

    cols = train_ds.column_names
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=cols)
    eval_ds  = eval_ds.map(tokenize_fn, batched=True, remove_columns=cols)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=0, pin_memory=False, collate_fn=collator
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
        num_workers=0, pin_memory=False, collate_fn=collator
    )

    # ─────────────────────────────────────────────
    # Optimizer & Scheduler
    # ─────────────────────────────────────────────
    no_decay = {"bias", "LayerNorm.weight", "layernorm.weight"}
    param_groups = [
        {"params": [p for n, p in model.named_parameters() 
                    if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": WEIGHT_DECAY},
        {"params": [p for n, p in model.named_parameters() 
                    if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]

    try:
        optimizer = torch.optim.AdamW(param_groups, lr=LR, fused=True)
        print("✅ Optimizer: AdamW (fused)")
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=LR)
        print("✅ Optimizer: AdamW")

    steps_per_epoch = math.ceil(len(train_loader) / GRAD_ACCUM)
    total_steps  = steps_per_epoch * EPOCHS
    warmup_steps = max(1, int(total_steps * WARMUP_RATIO))

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler("cuda")

    # ─────────────────────────────────────────────
    # Evaluation
    # ─────────────────────────────────────────────
    @torch.no_grad()
    def evaluate():
        model.eval()
        total_loss = correct = total = 0
        for batch in eval_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.amp.autocast("cuda"):
                out = model(**batch)
            bs = batch["labels"].size(0)
            total_loss += out.loss.item() * bs
            correct += (out.logits.argmax(-1) == batch["labels"]).sum().item()
            total += bs
        return total_loss / max(total, 1), correct / max(total, 1)

    # ─────────────────────────────────────────────
    # Training Loop
    # ─────────────────────────────────────────────
    print(f"\n🚀 Training:")
    print(f"  Batch: {BATCH_SIZE} × {GRAD_ACCUM} = {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps: {total_steps} ({warmup_steps} warmup)")
    print()

    best_loss = float("inf")
    best_state = None
    patience_ctr = 0
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = torch.zeros((), device=DEVICE)
        micro = 0

        for batch in train_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            with torch.amp.autocast("cuda"):
                out = model(**batch)
                scaled = out.loss / GRAD_ACCUM

            scaler.scale(scaled).backward()
            running_loss += out.loss.detach()
            micro += 1

            if micro % GRAD_ACCUM == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Flush tail
        if micro % GRAD_ACCUM != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss = (running_loss / micro).item()
        eval_loss, eval_acc = evaluate()
        elapsed = time.time() - start
        mem = torch.cuda.max_memory_allocated() / 1024**3

        print(f"  Epoch {epoch}/{EPOCHS} │ Train: {train_loss:.4f} │ "
              f"Eval: {eval_loss:.4f} │ Acc: {eval_acc:.2%} │ "
              f"Mem: {mem:.2f}GB │ {elapsed:.0f}s")

        if eval_loss < best_loss:
            best_loss = eval_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            print(f"         ↳ 🏆 New best!")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n⏹️  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    print(f"\n✅ Done! Best eval loss: {best_loss:.4f}")
    print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
