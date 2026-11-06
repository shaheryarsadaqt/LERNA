#!/usr/bin/env python3
"""
Batch-9 forward replay: isolate fp16 forward numerics on seed-7 MRPC step 9.
"""
import os, sys, json, math
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

# --- exact same config as the trainer ---
MODEL = "jhu-clsp/ettin-encoder-150m"
REVISION = "45d08642849e5c5701b162671ac811b7654bfd9f"
TASK = "mrpc"
SEED = 7
MAX_LENGTH = 128
BATCH_SIZE = 32

def build_dataloader():
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, local_files_only=True)
    ds = load_dataset("glue", TASK, split="train")
    ds = ds.shuffle(seed=SEED)
    
    def tokenize(ex):
        return tokenizer(ex["sentence1"], ex["sentence2"], truncation=True, max_length=MAX_LENGTH, padding="max_length")
    
    ds = ds.map(tokenize, batched=True, remove_columns=["sentence1", "sentence2", "idx"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

def get_batch_9(loader):
    for i, batch in enumerate(loader):
        if i == 9:  # 0-indexed, step 9 = batch index 9
            return {k: v for k, v in batch.items()}
    return None

def run_forward(model, batch, device, dtype, attn_impl):
    model.to(device)
    model.train()
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    labels = batch["label"].to(device)
    
    with torch.autocast(device_type="cuda", dtype=dtype, enabled=(dtype != torch.float32)):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
    return loss, outputs.logits if hasattr(outputs, "logits") else None

def hook_finder(model, batch, device, dtype):
    """Find first module with non-finite output."""
    first_bad = {"module": None, "input_finite": None, "output_finite": None}
    handles = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if first_bad["module"] is not None:
                return
            if isinstance(out, tuple):
                tensors = [t for t in out if isinstance(t, torch.Tensor)]
            elif isinstance(out, torch.Tensor):
                tensors = [out]
            else:
                return
            for t in tensors:
                f = t.detach().float()
                fin = torch.isfinite(f)
                if not fin.all():
                    first_bad["module"] = name
                    # check inputs
                    if isinstance(inp, tuple):
                        in_tensors = [t for t in inp if isinstance(t, torch.Tensor)]
                    elif isinstance(inp, torch.Tensor):
                        in_tensors = [inp]
                    else:
                        in_tensors = []
                    in_finite = all(torch.isfinite(t.detach().float()).all() for t in in_tensors)
                    first_bad["input_finite"] = in_finite
                    first_bad["output_finite"] = False
                    return
        return hook
    
    for name, mod in model.named_modules():
        handles.append(mod.register_forward_hook(make_hook(name)))
    
    _ = run_forward(model, batch, device, dtype, None)
    
    for h in handles:
        h.remove()
    return first_bad

def main():
    device = torch.device("cuda:0")
    loader = build_dataloader()
    batch = get_batch_9(loader)
    
    if batch is None:
        print("ERROR: batch 9 not found")
        return 1
    
    lengths = batch["attention_mask"].sum(dim=1)
    print(f"Batch 9: size={batch['input_ids'].shape[0]}, max_len={lengths.max().item()}, min_len={lengths.min().item()}, spread={lengths.max().item() - lengths.min().item()}")
    
    # Also check batch 8
    loader2 = build_dataloader()
    batch8 = None
    for i, b in enumerate(loader2):
        if i == 8:
            batch8 = {k: v for k, v in b.items()}
            break
    if batch8:
        lengths8 = batch8["attention_mask"].sum(dim=1)
        print(f"Batch 8: size={batch8['input_ids'].shape[0]}, max_len={lengths8.max().item()}, min_len={lengths8.min().item()}, spread={lengths8.max().item() - lengths8.min().item()}")
    
    print("\n=== 2x2 Grid: {fp16, fp32} x {eager, sdpa} ===")
    
    for attn_impl in ["eager", "sdpa"]:
        for dtype_name, dtype in [("fp16", torch.float16), ("fp32", torch.float32)]:
            print(f"\n--- {dtype_name} + {attn_impl} ---")
            model = AutoModelForSequenceClassification.from_pretrained(
                MODEL, revision=REVISION, local_files_only=True,
                num_labels=2, ignore_mismatched_sizes=True,
                attn_implementation=attn_impl if attn_impl != "eager" else None
            )
            model.config._attn_implementation = attn_impl
            
            try:
                loss, logits = run_forward(model, batch, device, dtype, attn_impl)
                print(f"  loss: {loss.item():.6f} (finite={math.isfinite(loss.item())})")
                if logits is not None:
                    f = logits.detach().float()
                    fin = torch.isfinite(f)
                    print(f"  logits: shape={tuple(logits.shape)}, n_nan={int(torch.isnan(f).sum())}, n_inf={int(torch.isinf(f).sum())}, max_abs={f[fin].abs().max().item() if fin.any() else 'N/A'}")
            except Exception as e:
                print(f"  ERROR: {type(e).__name__}: {e}")
            
            # Hook finder
            bad = hook_finder(model, batch, device, dtype)
            if bad["module"]:
                print(f"  FIRST NON-FINITE: module={bad['module']}, inputs_finite={bad['input_finite']}")
            else:
                print(f"  All modules finite")
            
            del model
            torch.cuda.empty_cache()

if __name__ == "__main__":
    sys.exit(main())