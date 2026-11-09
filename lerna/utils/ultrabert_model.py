"""UltraBERT for transformers 4.56 (LERNA round 17, 8 November 2026).

UltraBERT (Park, Lee and Kang, "UltraBERT: Beyond the Limit of BERT Pre-Training",
arXiv 2610.15028, October 2026; checkpoints dmis-lab/UltraBERT-150M and -380M, Apache-2.0)
is natively supported only from transformers 5.20. The qwen-v100 environment that ran every
LERNA round has transformers 4.56.0 and no package index, and transformers 5.22 needs
Python 3.11. This module re-implements transformers 5.22.2's native UltraBERT
(src/transformers/models/ultrabert/modeling_ultrabert.py and configuration_ultrabert.py,
Apache-2.0, copyright 2026 the HuggingFace Team) on transformers 4.56's base classes:

  - the configuration, with the same defaults and the same handling of the original
    checkpoints' fields (`attention_windows`, `rope_theta` given per layer);
  - the encoder: token embeddings; per layer RMSNorm -> attention -> residual, RMSNorm ->
    SwiGLU MLP -> residual; a final RMSNorm. Attention uses rotary embeddings with one base
    per layer type; "sliding_attention" layers see the tokens within `sliding_window` positions
    on either side (inclusive), "full_attention" layers see every token; padding keys are
    masked in both. Eager attention only (the additive mask is torch.finfo(dtype).min);
  - the sequence-classification model (first-token pooling by default, dropout, linear
    classifier) and the masked-LM model (used only to verify the checkpoint);
  - loading a checkpoint from its files, with the same key renaming as transformers 5.22
    (`ultrabert.` -> `model.`, `cls.` -> `head.decoder.`), and the exact sets of keys that may
    be missing or unused.

Round 17's model check compares this implementation, on the real checkpoint, with the float32
outputs that transformers 5.22.2's integration test (tests/models/ultrabert/
test_modeling_ultrabert.py) expects from the original checkpoint code. The kit's local tests
compare it with transformers 5.22.2's own implementation on random weights.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, MaskedLMOutput, SequenceClassifierOutput

ULTRABERT_MODEL_ID = "dmis-lab/UltraBERT-150M"
ULTRABERT_PREFIXES = ("dmis-lab/UltraBERT-",)
LAYER_TYPES = ("sliding_attention", "full_attention")
# transformers 5.22.2, conversion_mapping.py: the original checkpoints' key prefixes
KEY_RENAMES = ((re.compile(r"^ultrabert\."), "model."), (re.compile(r"^cls\."), "head.decoder."))
CLASSIFIER_KEYS = frozenset({"classifier.weight", "classifier.bias"})
MLM_HEAD_KEYS = frozenset({"head.linear1.weight", "head.linear2.weight", "head.decoder.weight"})


def is_ultrabert_model(model_name: str) -> bool:
    return isinstance(model_name, str) and any(model_name.startswith(p) for p in ULTRABERT_PREFIXES)


class UltraBertConfig(PretrainedConfig):
    """transformers 5.22.2's UltrabertConfig: same fields, defaults and legacy handling."""

    model_type = "ultrabert"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 4096,
        intermediate_size: int = 11008,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: Optional[int] = None,
        hidden_act: str = "silu",
        max_position_embeddings: int = 32768,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        use_cache: bool = False,
        pad_token_id: Optional[int] = 0,
        bos_token_id: Optional[int] = 0,
        eos_token_id=0,
        tie_word_embeddings: bool = False,
        rope_parameters: Optional[dict] = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        mlp_bias: bool = False,
        head_dim: Optional[int] = None,
        classifier_pooling: str = "cls",
        sliding_window: int = 128,
        layer_types: Optional[list] = None,
        prediction_head_size: int = 768,
        classifier_dropout: float = 0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_parameters = rope_parameters
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.mlp_bias = mlp_bias
        self.head_dim = head_dim
        if classifier_pooling not in ("cls", "mean"):
            raise ValueError(f"classifier_pooling must be 'cls' or 'mean', got {classifier_pooling!r}")
        self.classifier_pooling = classifier_pooling
        self.sliding_window = sliding_window
        self.layer_types = list(layer_types) if layer_types is not None else None
        self.prediction_head_size = prediction_head_size
        self.classifier_dropout = classifier_dropout

        # BC -> existing checkpoints store attention windows per layer.
        attention_windows = kwargs.pop("attention_windows", None)
        if attention_windows is not None:
            self.sliding_window = next((w for w in attention_windows if w), self.sliding_window)
            if self.layer_types is None:
                self.layer_types = ["sliding_attention" if w else "full_attention" for w in attention_windows]
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention" if (i + 1) % 3 else "full_attention" for i in range(self.num_hidden_layers)
            ]
        # BC -> existing checkpoints store RoPE bases in rope_theta.
        rope_theta = kwargs.pop("rope_theta", None)
        if rope_theta is not None and self.rope_parameters is None:
            self.rope_parameters = {
                kind: {"rope_type": "default", "rope_theta": theta} for kind, theta in zip(self.layer_types, rope_theta)
            }
        if self.rope_parameters is None:
            self.rope_parameters = {
                "sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                "full_attention": {"rope_type": "default", "rope_theta": 2_560_000.0},
            }
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({self.hidden_size}) is not a multiple of the number of attention "
                f"heads ({self.num_attention_heads})."
            )
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError(f"{len(self.layer_types)} layer types for {self.num_hidden_layers} layers")
        for kind in self.layer_types:
            if kind not in LAYER_TYPES:
                raise ValueError(f"unknown layer type {kind!r}")
        for kind in set(self.layer_types):
            params = self.rope_parameters.get(kind)
            if params is None:
                raise ValueError(f"no rotary parameters for layer type {kind!r}")
            if params.get("rope_type", "default") != "default":
                raise NotImplementedError(f"rope_type {params.get('rope_type')!r} is not implemented here")
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def convert_rope_params_to_dict(self, **kwargs):
        """transformers 5.x calls this from the base constructor (4.56 never does). The rotary
        parameters are already set per layer type above, so nothing is converted."""
        kwargs.pop("rope_scaling", None)
        return kwargs


class UltraBertRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class UltraBertRotaryEmbedding(nn.Module):
    """One set of inverse frequencies per layer type (default RoPE only)."""

    def __init__(self, config: UltraBertConfig):
        super().__init__()
        self.layer_types = sorted(set(config.layer_types))
        dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        for kind in self.layer_types:
            base = config.rope_parameters[kind]["rope_theta"]
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
            self.register_buffer(f"{kind}_inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor, layer_type: str):
        inv_freq = getattr(self, f"{layer_type}_inv_freq")
        freqs = position_ids[..., None].float() * inv_freq.to(device=x.device, dtype=torch.float)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(dtype=x.dtype), emb.sin().to(dtype=x.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim: int = 1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def additive_masks(attention_mask: Optional[torch.Tensor], length: int, batch: int, window: int,
                   dtype: torch.dtype, device) -> Dict[str, torch.Tensor]:
    """The eager masks of transformers 5.22 (masking_utils.eager_mask): 0 where a query may attend
    to a key, torch.finfo(dtype).min elsewhere, shape (batch, 1, length, length).

    full_attention: the key is not padding. sliding_attention: also |query - key| <= window.
    """
    if attention_mask is None:
        keys = torch.ones(batch, length, dtype=torch.bool, device=device)
    else:
        if attention_mask.dim() != 2 or tuple(attention_mask.shape) != (batch, length):
            raise ValueError(f"attention_mask must be (batch, length) = ({batch}, {length}), "
                             f"got {tuple(attention_mask.shape)}")
        keys = attention_mask.to(device=device).bool()
    allowed = keys[:, None, None, :].expand(batch, 1, length, length)
    idx = torch.arange(length, device=device)
    near = (idx[:, None] - idx[None, :]).abs() <= int(window)
    zero = torch.zeros((), device=device, dtype=dtype)
    low = torch.tensor(torch.finfo(dtype).min, device=device, dtype=dtype)
    return {"full_attention": torch.where(allowed, zero, low),
            "sliding_attention": torch.where(allowed & near[None, None], zero, low)}


class UltraBertAttention(nn.Module):
    def __init__(self, config: UltraBertConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim,
                                bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dim,
                                bias=config.attention_bias)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size,
                                bias=config.attention_bias)
        self.layer_type = config.layer_types[layer_idx]

    def forward(self, hidden_states: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout if self.training else 0.0,
                                             training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class UltraBertMLP(nn.Module):
    def __init__(self, config: UltraBertConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class UltraBertDecoderLayer(nn.Module):
    """Named as in transformers ("DecoderLayer"); the attention is bidirectional."""

    def __init__(self, config: UltraBertConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = UltraBertAttention(config=config, layer_idx=layer_idx)
        self.mlp = UltraBertMLP(config)
        self.input_layernorm = UltraBertRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = UltraBertRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor],
                position_embeddings: Tuple[torch.Tensor, torch.Tensor]):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states,
                                                     position_embeddings=position_embeddings,
                                                     attention_mask=attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, attn_weights


class UltraBertPreTrainedModel(PreTrainedModel):
    config_class = UltraBertConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["UltraBertDecoderLayer"]
    _supports_sdpa = False
    _supports_flash_attn = False
    _supports_flash_attn_2 = False
    _supports_flex_attn = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, UltraBertRMSNorm):
            module.weight.data.fill_(1.0)


class UltraBertModel(UltraBertPreTrainedModel):
    def __init__(self, config: UltraBertConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [UltraBertDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = UltraBertRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = UltraBertRotaryEmbedding(config=config)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(self, input_ids: Optional[torch.Tensor] = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: bool = False, output_hidden_states: bool = False, **kwargs) -> BaseModelOutput:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if position_ids is None:
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device).unsqueeze(0)
        batch, length = inputs_embeds.shape[0], inputs_embeds.shape[1]
        masks = additive_masks(attention_mask, length, batch, self.config.sliding_window, inputs_embeds.dtype,
                               inputs_embeds.device)
        position_embeddings = {kind: self.rotary_emb(inputs_embeds, position_ids, kind)
                               for kind in set(self.config.layer_types)}
        hidden_states = inputs_embeds
        all_hidden = [] if output_hidden_states else None
        all_attn = [] if output_attentions else None
        for i, layer in enumerate(self.layers):
            kind = self.config.layer_types[i]
            if all_hidden is not None:
                all_hidden.append(hidden_states)
            hidden_states, attn_weights = layer(hidden_states, attention_mask=masks[kind],
                                                position_embeddings=position_embeddings[kind])
            if all_attn is not None:
                all_attn.append(attn_weights)
        hidden_states = self.norm(hidden_states)
        if all_hidden is not None:
            all_hidden.append(hidden_states)
        return BaseModelOutput(last_hidden_state=hidden_states,
                               hidden_states=tuple(all_hidden) if all_hidden is not None else None,
                               attentions=tuple(all_attn) if all_attn is not None else None)


class UltraBertPredictionHead(nn.Module):
    def __init__(self, config: UltraBertConfig):
        super().__init__()
        self.linear1 = nn.Linear(config.hidden_size, config.prediction_head_size, bias=False)
        self.linear2 = nn.Linear(config.prediction_head_size, config.hidden_size, bias=False)
        self.act_fn = ACT2FN["gelu_pytorch_tanh"]
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.act_fn(self.linear1(hidden_states))
        hidden_states = self.act_fn(self.linear2(hidden_states))
        return self.decoder(hidden_states)


class UltraBertForMaskedLM(UltraBertPreTrainedModel):
    """Used only to verify a checkpoint (its prediction head is the one trained with 5% masking)."""

    def __init__(self, config: UltraBertConfig):
        super().__init__(config)
        self.model = UltraBertModel(config)
        self.head = UltraBertPredictionHead(config)
        self.post_init()

    def get_output_embeddings(self):
        return self.head.decoder

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, labels=None,
                **kwargs) -> MaskedLMOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             inputs_embeds=inputs_embeds)
        logits = self.head(outputs.last_hidden_state)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits.view(-1, self.config.vocab_size).float(), labels.view(-1),
                                               ignore_index=-100)
        return MaskedLMOutput(loss=loss, logits=logits)


class UltraBertForSequenceClassification(UltraBertPreTrainedModel):
    """Pooling (first token by default) -> dropout -> linear, as in transformers 5.22.

    The forward pass takes the same arguments as transformers 4.56's ModernBERT classifier that
    ran Ettin, including **kwargs (so the Trainer treats both models alike); extra inputs such as
    token_type_ids are ignored, as the native model ignores them.
    """

    def __init__(self, config: UltraBertConfig):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = UltraBertModel(config)
        self.head = nn.Identity()
        self.drop = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids: Optional[torch.LongTensor] = None, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None, inputs_embeds: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None,
                **kwargs) -> SequenceClassifierOutput:
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                             inputs_embeds=inputs_embeds, output_attentions=bool(output_attentions),
                             output_hidden_states=bool(output_hidden_states))
        last_hidden_state = outputs.last_hidden_state
        if self.config.classifier_pooling == "cls":
            pooled = last_hidden_state[:, 0]
        else:
            if attention_mask is None:
                attention_mask = torch.ones(last_hidden_state.shape[:2], device=last_hidden_state.device,
                                            dtype=torch.bool)
            pooled = (last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True)
        logits = self.classifier(self.drop(self.head(pooled)))
        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"
            if self.config.problem_type == "regression":
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze()) if self.num_labels == 1 else loss_fct(logits,
                                                                                                         labels)
            elif self.config.problem_type == "single_label_classification":
                loss = nn.CrossEntropyLoss()(logits.view(-1, self.num_labels), labels.view(-1))
            else:
                loss = nn.BCEWithLogitsLoss()(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states,
                                        attentions=outputs.attentions)


# --------------------------------------------------------------------------- loading
def rename_key(key: str) -> str:
    for pattern, new in KEY_RENAMES:
        key = pattern.sub(new, key)
    return key


def checkpoint_files(snapshot: Path) -> list:
    """The checkpoint's safetensors files: the index's shards, or model.safetensors."""
    snapshot = Path(snapshot)
    index = snapshot / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        return [snapshot / s for s in shards]
    single = snapshot / "model.safetensors"
    if single.is_file():
        return [single]
    raise FileNotFoundError(f"no model.safetensors (or index) in {snapshot}")


def read_state_dict(snapshot: Path) -> Dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    state = {}
    for f in checkpoint_files(snapshot):
        for k, v in load_file(str(f)).items():
            nk = rename_key(k)
            if nk in state:
                raise ValueError(f"two checkpoint tensors map to {nk}")
            state[nk] = v
    return state


def _is_head_key(key: str) -> bool:
    """A tensor outside the encoder (a prediction head): anything not under `model.`."""
    return not key.startswith("model.")


def load_checkpoint_into(model: UltraBertPreTrainedModel, snapshot: Path, *, allowed_missing: Iterable[str],
                         allowed_unused: Iterable[str] = (), unused_heads_ok: bool = False) -> dict:
    """Load the checkpoint's tensors; fail unless the missing and unused keys are the allowed ones
    and every loaded tensor has its parameter's shape. With unused_heads_ok, any unused tensor
    outside the encoder (a prediction head) is allowed too; an unused encoder tensor never is.
    Returns the key accounting."""
    state = read_state_dict(Path(snapshot))
    own = model.state_dict()
    bad_shapes = sorted(k for k in state if k in own and tuple(state[k].shape) != tuple(own[k].shape))
    if bad_shapes:
        raise ValueError(f"checkpoint shapes differ for {bad_shapes[:5]}")
    missing = sorted(k for k in own if k not in state)
    unused = sorted(k for k in state if k not in own)
    if set(missing) - set(allowed_missing):
        raise ValueError(f"checkpoint lacks {sorted(set(missing) - set(allowed_missing))[:8]}")
    bad_unused = sorted(k for k in unused if k not in set(allowed_unused) and not (unused_heads_ok and _is_head_key(k)))
    if bad_unused:
        raise ValueError(f"checkpoint has unexpected tensors {bad_unused[:8]}")
    dtypes = sorted({str(v.dtype) for v in state.values()})
    with torch.no_grad():
        for k, v in state.items():
            if k in own:
                own[k].copy_(v.to(own[k].dtype))
    return {"loaded": len(state) - len(unused), "missing": missing, "unused": unused, "checkpoint_dtypes": dtypes}


def resolve_snapshot(model_name: str, revision: str) -> Path:
    """The local snapshot folder of model_name at revision, from the Hub cache (no download)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo_id=model_name, revision=revision, local_files_only=True))


def _read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text()) if Path(path).is_file() else {}


def _token_text(v):
    return v.get("content") if isinstance(v, dict) else v


def load_ultrabert_tokenizer(model_name: str, *, revision: str, snapshot: Optional[Path] = None):
    """The checkpoint's fast tokenizer, built from its tokenizer.json (vocabulary, added tokens with
    their flags, pre-tokenizer and single/pair template), with the named special tokens of
    special_tokens_map.json or tokenizer_config.json. Other tokenizer_config.json fields, which
    transformers 5 may have written, are not read. Inputs are input_ids and attention_mask
    (unless the configuration names others); padding is on the right, so the first token stays
    first."""
    from transformers import PreTrainedTokenizerFast

    snapshot = Path(snapshot) if snapshot is not None else resolve_snapshot(model_name, revision)
    tok_file = snapshot / "tokenizer.json"
    if not tok_file.is_file():
        raise FileNotFoundError(f"no tokenizer.json in {snapshot}")
    cfg = _read_json(snapshot / "tokenizer_config.json")
    smap = _read_json(snapshot / "special_tokens_map.json")
    specials = {}
    for key in ("bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token", "mask_token"):
        v = _token_text(smap.get(key, cfg.get(key)))
        if isinstance(v, str) and v:
            specials[key] = v
    names = cfg.get("model_input_names") or ["input_ids", "attention_mask"]
    tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_file), model_input_names=list(names),
                                  padding_side="right",
                                  clean_up_tokenization_spaces=bool(cfg.get("clean_up_tokenization_spaces", False)),
                                  **specials)
    vocab = tok.get_vocab()
    unknown = sorted(v for v in specials.values() if v not in vocab)
    if unknown:
        raise ValueError(f"special tokens {unknown} are not in tokenizer.json")
    if tok.pad_token_id is None:
        raise ValueError("the UltraBERT tokenizer has no pad token")
    return tok


def load_ultrabert_for_sequence_classification(model_name: str, *, revision: str, num_labels: int = 2,
                                               problem_type: Optional[str] = None,
                                               snapshot: Optional[Path] = None,
                                               dtype: torch.dtype = torch.float32):
    snapshot = Path(snapshot) if snapshot is not None else resolve_snapshot(model_name, revision)
    kwargs = {"num_labels": num_labels}
    if problem_type is not None:
        kwargs["problem_type"] = problem_type
    config = UltraBertConfig.from_pretrained(str(snapshot), **kwargs)
    config._attn_implementation = "eager"
    model = UltraBertForSequenceClassification(config)
    report = load_checkpoint_into(model, snapshot, allowed_missing=CLASSIFIER_KEYS, unused_heads_ok=True)
    if set(report["missing"]) != set(CLASSIFIER_KEYS):
        raise ValueError(f"expected only the classifier to be new, missing {report['missing']}")
    model.to(dtype)
    model.lerna_checkpoint_report = report
    model.config.name_or_path = model_name
    return model


def load_ultrabert_for_masked_lm(model_name: str, *, revision: str, snapshot: Optional[Path] = None,
                                 dtype: torch.dtype = torch.float32):
    snapshot = Path(snapshot) if snapshot is not None else resolve_snapshot(model_name, revision)
    config = UltraBertConfig.from_pretrained(str(snapshot))
    config._attn_implementation = "eager"
    model = UltraBertForMaskedLM(config)
    # every encoder and 5%-masking head tensor must be present; another head may be unused
    report = load_checkpoint_into(model, snapshot, allowed_missing=(), unused_heads_ok=True)
    model.to(dtype)
    model.lerna_checkpoint_report = report
    return model


def dropbp_macs_from_config(config) -> dict:
    """The multiply-adds per token of one layer's linear maps, from the configuration alone
    (round 17's DropBP check): attention q, k, v, o; the SwiGLU MLP's three maps."""
    h = int(config.hidden_size)
    nh = int(config.num_attention_heads)
    nkv = int(config.num_key_value_heads or nh)
    hd = int(getattr(config, "head_dim", None) or h // nh)
    return {"layers": int(config.num_hidden_layers),
            "attn_linear": h * nh * hd + 2 * h * nkv * hd + nh * hd * h,
            "attn_scores_per_position": 2 * nh * hd,
            "mlp": 3 * h * int(config.intermediate_size)}


__all__ = [
    "ULTRABERT_MODEL_ID",
    "UltraBertConfig",
    "UltraBertModel",
    "UltraBertForMaskedLM",
    "UltraBertForSequenceClassification",
    "is_ultrabert_model",
    "load_ultrabert_tokenizer",
    "load_ultrabert_for_sequence_classification",
    "load_ultrabert_for_masked_lm",
    "dropbp_macs_from_config",
    "resolve_snapshot",
]
