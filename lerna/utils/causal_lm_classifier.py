"""Round 18: decoder-only (causal) language models as sentence-pair classifiers.

Prof. Chen asked for results on causal language models (Llama, Qwen) and on a mixture-of-experts
LLM. Round 18 fine-tunes three of them on MRPC and RTE:

  alpindale/Llama-3.2-1B                   Meta's Llama-3.2-1B (base), 1.24B parameters. Meta's own
                                           repository is gated; this public copy holds the same files
                                           (the download step compares them with Meta's listing).
  Qwen/Qwen3-0.6B-Base                     Alibaba's Qwen3-0.6B base model, 0.6B parameters.
  ibm-granite/granite-3.1-1b-a400m-base    IBM's Granite-3.1 MoE base model: 1.3B parameters,
                                           32 experts per layer, 8 used per token (400M active).

transformers 4.56 has LlamaForSequenceClassification and Qwen3ForSequenceClassification but no
GraniteMoe one, so all three models get the same head here, built like transformers' own
*ForSequenceClassification heads:
  - the pretrained causal LM's body (its `.model`, loaded through AutoModelForCausalLM so that the
    checkpoint must match the whole language model, with nothing missing or unused);
  - the hidden state of each sequence's last real token. transformers finds it from a pad-token
    id; here it comes from the attention mask (padding is on the right);
  - a linear layer without bias, initialized normal(0, initializer_range) as transformers does.
FP32 weights, eager attention, no KV cache, and gradient checkpointing (non-reentrant), which
recomputes each layer's activations in the backward pass instead of storing them: the gradients
are the same, the memory is small enough for 32 GB V100s, and the arithmetic costs about one more
forward pass.

The input. A decoder has no [CLS]/[SEP] pair template, so each example is written as text,
  "Sentence 1: <sentence1>\nSentence 2: <sentence2>"   (single sentences: "Sentence: <sentence>")
encoded with the model's own tokenizer and its own special tokens (Llama adds its BOS token; Qwen3
and Granite add none), and ended with the model's EOS token, so every example ends on the same
token and the head reads that token's hidden state. Padding uses the tokenizer's pad token, or the
EOS token when it has none (the attention mask, not the id, marks padding).
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import torch
from torch import nn

# model id -> (name in the tables, the configuration's model_type)
CAUSAL_LM_MODELS = {
    "alpindale/Llama-3.2-1B": ("Llama-3.2-1B", "llama"),
    "Qwen/Qwen3-0.6B-Base": ("Qwen3-0.6B-Base", "qwen3"),
    "ibm-granite/granite-3.1-1b-a400m-base": ("Granite-3.1-1B-A400M-Base", "granitemoe"),
}
PAIR_TEMPLATE = "Sentence 1: {a}\nSentence 2: {b}"
SINGLE_TEMPLATE = "Sentence: {a}"
HEAD_KEYS = ("score.weight",)


def is_causal_lm_model(model_name) -> bool:
    return isinstance(model_name, str) and model_name in CAUSAL_LM_MODELS


def require_pinned(model_name: str, revision) -> str:
    if (not isinstance(revision, str) or len(revision) != 40
            or any(ch not in "0123456789abcdef" for ch in revision)):
        raise ValueError(f"{model_name} requires an explicit 40-character lowercase hex revision; got {revision!r}")
    return revision


def resolve_snapshot(model_name: str, revision: str) -> Path:
    """The local snapshot folder of model_name at revision, from the Hub cache (no download)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_name, revision=revision, local_files_only=True))


def format_example(text_a: str, text_b: Optional[str] = None) -> str:
    if text_b is None:
        return SINGLE_TEMPLATE.format(a=text_a)
    return PAIR_TEMPLATE.format(a=text_a, b=text_b)


class PromptTokenizer:
    """The model's own tokenizer behind the round-18 input format (see the module docstring).

    Called like a transformers tokenizer by the runner, tokenizer(texts, pairs, truncation=True,
    max_length=512), it returns {"input_ids": [...], "attention_mask": [...]}. With truncation, a
    sequence longer than max_length keeps its first max_length - 1 tokens and the EOS token. It
    pads on the right (the runner's collator does the padding)."""

    padding_side = "right"
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self, tokenizer):
        eos = getattr(tokenizer, "eos_token_id", None)
        if eos is None:
            raise ValueError("the tokenizer has no EOS token")
        self.tokenizer = tokenizer
        self.eos_token_id = int(eos)
        pad = getattr(tokenizer, "pad_token_id", None)
        self.pad_token_id = int(pad) if pad is not None else int(eos)
        self.pad_source = "tokenizer" if pad is not None else "eos"
        self.bos_token_id = getattr(tokenizer, "bos_token_id", None)
        self.name_or_path = getattr(tokenizer, "name_or_path", None)

    def __len__(self):
        return len(self.tokenizer)

    def encode_example(self, text_a: str, text_b: Optional[str] = None, *, truncation: bool = False,
                       max_length: Optional[int] = None) -> list:
        ids = list(self.tokenizer(format_example(text_a, text_b), add_special_tokens=True)["input_ids"])
        if ids and ids[-1] == self.eos_token_id:
            ids = ids[:-1]  # a tokenizer that ends sequences with EOS itself: keep exactly one
        if truncation and max_length is not None and len(ids) + 1 > int(max_length):
            ids = ids[: max(int(max_length) - 1, 0)]
        return ids + [self.eos_token_id]

    def __call__(self, text, text_pair=None, *, truncation: bool = False, max_length: Optional[int] = None,
                 padding=False, return_tensors=None, add_special_tokens: bool = True, **unused):
        if not add_special_tokens:
            raise ValueError("the round-18 input format always uses the model's special tokens")
        batched = isinstance(text, (list, tuple))
        texts = list(text) if batched else [text]
        if text_pair is None:
            pairs = [None] * len(texts)
        else:
            pairs = list(text_pair) if isinstance(text_pair, (list, tuple)) else [text_pair]
            if len(pairs) != len(texts):
                raise ValueError(f"{len(texts)} first sentences but {len(pairs)} second sentences")
        ids = [self.encode_example(a, b, truncation=truncation, max_length=max_length) for a, b in zip(texts, pairs)]
        masks = [[1] * len(x) for x in ids]
        if padding:
            width = max(len(x) for x in ids)
            ids = [x + [self.pad_token_id] * (width - len(x)) for x in ids]
            masks = [m + [0] * (width - len(m)) for m in masks]
        if return_tensors == "pt":
            if not padding and len({len(x) for x in ids}) > 1:
                raise ValueError("return_tensors='pt' needs padding=True for sequences of different lengths")
            return {"input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(masks, dtype=torch.long)}
        if return_tensors is not None:
            raise ValueError(f"return_tensors={return_tensors!r} is not supported")
        if batched:
            return {"input_ids": ids, "attention_mask": masks}
        return {"input_ids": ids[0], "attention_mask": masks[0]}

    def decode(self, ids, **kwargs):
        return self.tokenizer.decode(ids, **kwargs)


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """The position of each row's last real token (the last 1 of its attention mask)."""
    mask = attention_mask.to(torch.long)
    positions = torch.arange(mask.shape[1], device=mask.device, dtype=torch.long)
    return (mask * positions).argmax(dim=1)


class CausalLMForSequenceClassification(nn.Module):
    """A causal LM's body with a last-token linear head (see the module docstring)."""

    main_input_name = "input_ids"

    def __init__(self, body: nn.Module, num_labels: int = 2):
        super().__init__()
        self.model = body
        self.config = body.config
        self.num_labels = int(num_labels)
        self.config.num_labels = self.num_labels
        self.score = nn.Linear(int(self.config.hidden_size), self.num_labels, bias=False)
        std = float(getattr(self.config, "initializer_range", 0.02))
        with torch.no_grad():
            self.score.weight.normal_(mean=0.0, std=std)

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.model, "is_gradient_checkpointing", False))

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {"use_reentrant": False})

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        if input_ids is None:
            raise ValueError("input_ids are required")
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        hidden = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        pooled = hidden[rows, last_token_index(attention_mask).to(hidden.device)]
        logits = self.score(pooled)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels.view(-1).long().to(logits.device))
        from transformers.modeling_outputs import SequenceClassifierOutput

        return SequenceClassifierOutput(loss=loss, logits=logits)


def _loading_report(info) -> dict:
    info = info or {}
    get = info.get if isinstance(info, dict) else (lambda k, d=None: getattr(info, k, d))
    return {k: sorted(str(x) for x in (get(k) or [])) for k in ("missing_keys", "unexpected_keys", "mismatched_keys")}


def load_causal_lm(model_name: str, *, revision: str, snapshot: Optional[Path] = None,
                   dtype: torch.dtype = torch.float32):
    """The pretrained causal LM (with its LM head), FP32 and eager attention, from the local Hub
    cache at a pinned revision. Fails unless every tensor loads and none is left unused."""
    from transformers import AutoConfig, AutoModelForCausalLM

    if not is_causal_lm_model(model_name):
        raise ValueError(f"{model_name} is not a round-18 causal LM")
    require_pinned(model_name, revision)
    snapshot = Path(snapshot) if snapshot is not None else resolve_snapshot(model_name, revision)
    want_type = CAUSAL_LM_MODELS[model_name][1]
    config = AutoConfig.from_pretrained(str(snapshot), local_files_only=True)
    if config.model_type != want_type:
        raise ValueError(f"{model_name}: configuration model_type {config.model_type!r}, expected {want_type!r}")
    model, info = AutoModelForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=dtype, attn_implementation="eager", output_loading_info=True)
    report = _loading_report(info)
    if any(report.values()):
        raise ValueError(f"{model_name}: the checkpoint does not match the model exactly: {report}")
    model.config.use_cache = False
    return model, report


def load_causal_lm_tokenizer(model_name: str, *, revision: str, snapshot: Optional[Path] = None) -> PromptTokenizer:
    from transformers import AutoTokenizer

    if not is_causal_lm_model(model_name):
        raise ValueError(f"{model_name} is not a round-18 causal LM")
    require_pinned(model_name, revision)
    snapshot = Path(snapshot) if snapshot is not None else resolve_snapshot(model_name, revision)
    tok = AutoTokenizer.from_pretrained(str(snapshot), local_files_only=True)
    return PromptTokenizer(tok)


def load_causal_lm_for_sequence_classification(model_name: str, *, revision: str, num_labels: int = 2,
                                               problem_type: Optional[str] = None,
                                               snapshot: Optional[Path] = None,
                                               gradient_checkpointing: bool = True,
                                               dtype: torch.dtype = torch.float32):
    if problem_type not in (None, "single_label_classification"):
        raise ValueError(f"problem_type {problem_type!r} is not supported for the causal LMs")
    lm, report = load_causal_lm(model_name, revision=revision, snapshot=snapshot, dtype=dtype)
    body = lm.model
    model = CausalLMForSequenceClassification(body, num_labels=num_labels)
    model.to(dtype)
    if gradient_checkpointing:
        model.gradient_checkpointing_enable({"use_reentrant": False})
    model.config.name_or_path = model_name
    model.lerna_checkpoint_report = dict(report, new=list(HEAD_KEYS))
    return model


def parameter_counts(model: nn.Module, config=None) -> dict:
    """Total parameters, the head's, and for a mixture-of-experts body the parameters one token uses."""
    total = int(sum(p.numel() for p in model.parameters()))
    head = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("score.")))
    out = {"total": total, "head": head, "body": total - head}
    config = config if config is not None else getattr(model, "config", None)
    n_exp = getattr(config, "num_local_experts", None) if config is not None else None
    k = getattr(config, "num_experts_per_tok", None) if config is not None else None
    if n_exp and k:
        experts = int(sum(p.numel() for n, p in model.named_parameters()
                          if ".input_linear." in n or ".output_linear." in n or ".experts." in n))
        out["experts"] = experts
        out["active_per_token"] = total - experts + experts * int(k) // int(n_exp)
    return out


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = [
    "CAUSAL_LM_MODELS",
    "PAIR_TEMPLATE",
    "SINGLE_TEMPLATE",
    "CausalLMForSequenceClassification",
    "PromptTokenizer",
    "format_example",
    "is_causal_lm_model",
    "last_token_index",
    "load_causal_lm",
    "load_causal_lm_tokenizer",
    "load_causal_lm_for_sequence_classification",
    "parameter_counts",
    "require_pinned",
    "resolve_snapshot",
]
