"""Round 18: decoder-only LLMs (Llama, Qwen3, a Granite mixture-of-experts model) as sentence-pair
classifiers: the input format, the last-token head, gradient checkpointing, loading from the Hub
cache, the plain run at 35%, and the runner end to end with each round-18 method on tiny models."""
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lerna.utils import causal_lm_classifier as cl  # noqa: E402
from lerna.utils import model_loader  # noqa: E402
import run_controller_ablation as runner  # noqa: E402
from run_controller_ablation import ExperimentConfig, run_execution  # noqa: E402

from transformers import (GraniteMoeConfig, GraniteMoeForCausalLM, LlamaConfig, LlamaForCausalLM,  # noqa: E402
                          LlamaForSequenceClassification, Qwen3Config, Qwen3ForCausalLM,
                          Qwen3ForSequenceClassification)

SHA = "0123456789abcdef0123456789abcdef01234567"
LLAMA, QWEN, GRANITE = "alpindale/Llama-3.2-1B", "Qwen/Qwen3-0.6B-Base", "ibm-granite/granite-3.1-1b-a400m-base"
CORPUS = ["Sentence 1: the cat sat on the mat", "Sentence 2: a dog ran in the park", "Sentence: hello world"] * 20


# --------------------------------------------------------------------------- fixtures
def _write_tokenizer(folder: Path, *, bos: bool, pad: bool, eos_in_template: bool = False) -> int:
    """A small byte-level BPE tokenizer: with or without a BOS template (Llama adds BOS; Qwen3 and
    Granite add none), with or without a pad token, optionally ending every sequence with EOS."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers

    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    specials = ["<|bos|>", "<|eos|>"] + (["<|pad|>"] if pad else [])
    trainer = trainers.BpeTrainer(vocab_size=120, special_tokens=specials,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(CORPUS, trainer)
    if bos or eos_in_template:
        single = ("<|bos|> " if bos else "") + "$A" + (" <|eos|>" if eos_in_template else "")
        tok.post_processor = processors.TemplateProcessing(
            single=single, pair=single + " $B",
            special_tokens=[("<|bos|>", tok.token_to_id("<|bos|>")), ("<|eos|>", tok.token_to_id("<|eos|>"))])
    folder.mkdir(parents=True, exist_ok=True)
    tok.save(str(folder / "tokenizer.json"))
    cfg = {"tokenizer_class": "PreTrainedTokenizerFast", "bos_token": "<|bos|>", "eos_token": "<|eos|>"}
    if pad:
        cfg["pad_token"] = "<|pad|>"
    (folder / "tokenizer_config.json").write_text(json.dumps(cfg))
    return tok.get_vocab_size()


def _hf_tokenizer(tmp_path, **kw):
    from transformers import AutoTokenizer

    d = tmp_path / ("tok_" + "_".join(f"{k}{int(v)}" for k, v in sorted(kw.items())))
    _write_tokenizer(d, **kw)
    return AutoTokenizer.from_pretrained(str(d))


FAMILIES = {
    LLAMA: (LlamaConfig, LlamaForCausalLM, {}),
    QWEN: (Qwen3Config, Qwen3ForCausalLM, {"head_dim": 8}),
    GRANITE: (GraniteMoeConfig, GraniteMoeForCausalLM, {"num_local_experts": 4, "num_experts_per_tok": 2}),
}


def _lm_config(name, vocab=120, **kw):
    cfg_cls, _, extra = FAMILIES[name]
    d = dict(vocab_size=vocab, hidden_size=32, intermediate_size=48, num_hidden_layers=3, num_attention_heads=4,
             num_key_value_heads=2, max_position_embeddings=128, tie_word_embeddings=True, bos_token_id=0,
             eos_token_id=1, attention_dropout=0.0)
    d.update(extra)
    d.update(kw)
    return cfg_cls(**d)


def _tiny_lm(name, seed=0, **kw):
    torch.manual_seed(seed)
    cfg = _lm_config(name, **kw)
    return FAMILIES[name][1](cfg)


def _fake_hub(tmp_path, monkeypatch, name, lm=None, *, bos=None, pad=None, seed=0):
    """A local Hub cache holding `name` at SHA: a tiny causal LM and a tokenizer."""
    import huggingface_hub.constants as hc

    cache = tmp_path / "hub"
    repo = cache / ("models--" + name.replace("/", "--"))
    snap = repo / "snapshots" / SHA
    vocab = _write_tokenizer(snap, bos=(name == LLAMA) if bos is None else bos,
                             pad=(name == QWEN) if pad is None else pad)
    lm = lm if lm is not None else _tiny_lm(name, seed=seed, vocab=vocab)
    lm.save_pretrained(str(snap))
    (repo / "refs").mkdir(parents=True, exist_ok=True)
    (repo / "refs" / "main").write_text(SHA)
    monkeypatch.setattr(hc, "HF_HUB_CACHE", str(cache))
    return snap, lm


def _classifier(name, seed=0, **kw):
    lm = _tiny_lm(name, seed=seed, **kw)
    lm.config._attn_implementation = "eager"
    torch.manual_seed(seed + 100)
    return cl.CausalLMForSequenceClassification(lm.model, num_labels=2)


# --------------------------------------------------------------------------- the input format
def test_registry_and_template():
    assert set(cl.CAUSAL_LM_MODELS) == {LLAMA, QWEN, GRANITE}
    assert {v[1] for v in cl.CAUSAL_LM_MODELS.values()} == {"llama", "qwen3", "granitemoe"}
    assert cl.format_example("A b.", "C d.") == "Sentence 1: A b.\nSentence 2: C d."
    assert cl.format_example("A b.") == "Sentence: A b."
    assert all(model_loader.is_causal_lm_model(n) for n in cl.CAUSAL_LM_MODELS)
    assert not model_loader.is_causal_lm_model("jhu-clsp/ettin-encoder-150m")
    assert not model_loader.is_causal_lm_model("meta-llama/Llama-3.2-1B")


@pytest.mark.parametrize("bos", [True, False])
def test_prompt_tokenizer_formats_and_ends_with_eos(tmp_path, bos):
    hf = _hf_tokenizer(tmp_path, bos=bos, pad=False)
    tok = cl.PromptTokenizer(hf)
    eos = hf.convert_tokens_to_ids("<|eos|>")
    assert tok.eos_token_id == eos and tok.pad_token_id == eos and tok.pad_source == "eos"
    assert tok.padding_side == "right"
    enc = tok(["the cat sat", "a dog"], ["on the mat", "ran in the park"], truncation=True, max_length=512)
    assert set(enc) == {"input_ids", "attention_mask"}
    want0 = hf("Sentence 1: the cat sat\nSentence 2: on the mat")["input_ids"] + [eos]
    assert enc["input_ids"][0] == want0
    assert enc["attention_mask"][0] == [1] * len(want0)
    assert (enc["input_ids"][0][0] == hf.convert_tokens_to_ids("<|bos|>")) == bos
    assert all(ids[-1] == eos and ids.count(eos) == 1 for ids in enc["input_ids"])
    single = tok("hello world", truncation=True, max_length=512)
    assert single["input_ids"] == hf("Sentence: hello world")["input_ids"] + [eos]
    text = tok.decode(enc["input_ids"][0])
    assert text.endswith("Sentence 2: on the mat<|eos|>")


def test_prompt_tokenizer_truncation_padding_and_errors(tmp_path):
    hf = _hf_tokenizer(tmp_path, bos=True, pad=True)
    tok = cl.PromptTokenizer(hf)
    eos, pad = hf.convert_tokens_to_ids("<|eos|>"), hf.convert_tokens_to_ids("<|pad|>")
    assert tok.pad_token_id == pad and tok.pad_source == "tokenizer"
    full = tok("the cat sat on the mat", "a dog ran in the park")["input_ids"]
    cut = tok("the cat sat on the mat", "a dog ran in the park", truncation=True, max_length=7)["input_ids"]
    assert len(cut) == 7 and cut[:6] == full[:6] and cut[-1] == eos
    assert tok("the cat sat on the mat", "a dog", truncation=False, max_length=3)["input_ids"][-1] == eos
    batch = tok(["the cat", "a dog ran in the park"], padding=True, return_tensors="pt")
    assert batch["input_ids"].shape == batch["attention_mask"].shape
    short = int(batch["attention_mask"][0].sum())
    assert batch["input_ids"][0, short - 1] == eos and (batch["input_ids"][0, short:] == pad).all()
    assert (batch["attention_mask"][0, short:] == 0).all()
    with pytest.raises(ValueError):
        tok(["a", "b"], ["c"])
    with pytest.raises(ValueError):
        tok("a", add_special_tokens=False)
    with pytest.raises(ValueError):
        tok(["a", "b c d e"], return_tensors="pt")


def test_prompt_tokenizer_keeps_a_single_eos_when_the_tokenizer_adds_one(tmp_path):
    hf = _hf_tokenizer(tmp_path, bos=False, pad=False, eos_in_template=True)
    tok = cl.PromptTokenizer(hf)
    ids = tok("the cat", "a dog")["input_ids"]
    assert ids == hf("Sentence 1: the cat\nSentence 2: a dog")["input_ids"]
    assert ids.count(tok.eos_token_id) == 1 and ids[-1] == tok.eos_token_id


def test_last_token_index():
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]])
    assert cl.last_token_index(mask).tolist() == [2, 4, 0]


# --------------------------------------------------------------------------- the head
@pytest.mark.parametrize("name,hf_cls", [(LLAMA, LlamaForSequenceClassification),
                                         (QWEN, Qwen3ForSequenceClassification)])
def test_head_equals_transformers_own_head(name, hf_cls):
    """With a pad id that is not the EOS id, transformers' head reads the same last token."""
    model = _classifier(name, seed=3)
    model.eval()
    cfg = _lm_config(name, pad_token_id=7, num_labels=2)
    cfg._attn_implementation = "eager"
    torch.manual_seed(0)
    ref = hf_cls(cfg)
    ref.eval()
    ref.model.load_state_dict(model.model.state_dict())
    ref.score.weight.data.copy_(model.score.weight.data)
    ids = torch.randint(10, 120, (3, 9))
    ids[:, -1] = 1
    mask = torch.ones_like(ids)
    mask[0, 5:] = 0
    ids[0, 4] = 1
    ids[0, 5:] = 7
    with torch.no_grad():
        a = model(input_ids=ids, attention_mask=mask).logits
        b = ref(input_ids=ids, attention_mask=mask).logits
    assert torch.allclose(a, b, atol=1e-6, rtol=0)


@pytest.mark.parametrize("name", [LLAMA, QWEN, GRANITE])
def test_padding_does_not_change_an_example(name):
    model = _classifier(name, seed=4)
    model.eval()
    ids = torch.randint(10, 120, (2, 11))
    mask = torch.ones_like(ids)
    mask[1, 6:] = 0
    with torch.no_grad():
        both = model(input_ids=ids, attention_mask=mask).logits
        alone = model(input_ids=ids[1:, :6], attention_mask=mask[1:, :6]).logits
    assert torch.allclose(both[1:], alone, atol=1e-5, rtol=0)


@pytest.mark.parametrize("name,dropout", [(LLAMA, 0.0), (QWEN, 0.0), (GRANITE, 0.0), (GRANITE, 0.1)])
def test_gradient_checkpointing_gives_the_same_gradients(name, dropout):
    model = _classifier(name, seed=5, attention_dropout=dropout)
    ids = torch.randint(10, 120, (4, 8))
    mask = torch.ones_like(ids)
    mask[2, 5:] = 0
    labels = torch.tensor([0, 1, 1, 0])
    grads = []
    for gc in (True, False):
        if gc:
            model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable()
        assert model.is_gradient_checkpointing == gc
        model.train()
        model.zero_grad(set_to_none=True)
        torch.manual_seed(11)
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        grads.append((float(out.loss), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}))
    (l1, g1), (l2, g2) = grads
    assert l1 == l2 and set(g1) == set(g2) and len(g1) == len(list(model.parameters()))
    assert all(torch.allclose(g1[n], g2[n], atol=1e-7, rtol=1e-6) for n in g1)


def test_output_loss_and_extra_inputs():
    model = _classifier(QWEN, seed=6)
    ids = torch.randint(10, 120, (3, 6))
    out = model(input_ids=ids, labels=torch.tensor([0, 1, 0]), token_type_ids=torch.zeros_like(ids),
                num_items_in_batch=3)
    assert out.logits.shape == (3, 2)
    assert torch.isclose(out.loss, torch.nn.functional.cross_entropy(out.logits, torch.tensor([0, 1, 0])))
    assert model(input_ids=ids).loss is None
    with pytest.raises(ValueError):
        model(attention_mask=torch.ones(1, 2))


def test_score_head_initialization():
    model = _classifier(LLAMA, seed=7, initializer_range=0.5, hidden_size=64, num_attention_heads=4)
    w = model.score.weight
    assert w.shape == (2, 64) and model.score.bias is None
    assert 0.3 < float(w.std()) < 0.7


def test_parameter_counts_for_the_mixture_of_experts():
    model = _classifier(GRANITE, seed=8)
    c = cl.parameter_counts(model)
    cfg = model.config
    experts = cfg.num_hidden_layers * cfg.num_local_experts * (cfg.hidden_size * 2 * cfg.intermediate_size
                                                               + cfg.intermediate_size * cfg.hidden_size)
    assert c["experts"] == experts and c["head"] == 2 * cfg.hidden_size
    assert c["active_per_token"] == c["total"] - experts + experts // 2
    assert "experts" not in cl.parameter_counts(_classifier(LLAMA, seed=8))


# --------------------------------------------------------------------------- loading
@pytest.mark.parametrize("name", [LLAMA, QWEN, GRANITE])
def test_model_loader_loads_each_llm_from_the_hub_cache(tmp_path, monkeypatch, name):
    snap, lm = _fake_hub(tmp_path, monkeypatch, name, seed=9)
    model, tok = model_loader.load_model_and_tokenizer(name, num_labels=2, revision=SHA, local_files_only=True)
    assert isinstance(model, cl.CausalLMForSequenceClassification)
    assert isinstance(tok, cl.PromptTokenizer) and tok.padding_side == "right"
    assert model.is_gradient_checkpointing and model.config._attn_implementation == "eager"
    assert model.config.model_type == cl.CAUSAL_LM_MODELS[name][1] and model.config.use_cache is False
    assert all(p.dtype == torch.float32 for p in model.parameters())
    rep = model.lerna_checkpoint_report
    assert rep == {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": [], "new": ["score.weight"]}
    own = lm.model.state_dict()
    assert all(torch.equal(v, own[k]) for k, v in model.model.state_dict().items())
    tok2 = model_loader.load_tokenizer(name, revision=SHA, local_files_only=True)
    assert tok2("a b", "c")["input_ids"] == tok("a b", "c")["input_ids"]
    batch = tok(["the cat", "a dog ran"], ["sat", "in the park"], padding=True, return_tensors="pt")
    with torch.no_grad():
        assert model(**batch).logits.shape == (2, 2)
    for bad in (None, "main", SHA[:12]):
        with pytest.raises(ValueError):
            model_loader.load_model_and_tokenizer(name, revision=bad)
        with pytest.raises(ValueError):
            model_loader.load_tokenizer(name, revision=bad)
    with pytest.raises(Exception):
        model_loader.load_model_and_tokenizer(name, revision="f" * 40)  # not in the cache
    with pytest.raises(ValueError):
        model_loader.load_model_and_tokenizer(name, revision=SHA, device_map="auto")


def test_loading_stops_on_the_wrong_architecture(tmp_path, monkeypatch):
    lm = _tiny_lm(LLAMA, seed=1)
    _fake_hub(tmp_path, monkeypatch, QWEN, lm=lm)  # Llama weights under the Qwen3 id
    with pytest.raises(ValueError, match="model_type"):
        model_loader.load_model_and_tokenizer(QWEN, revision=SHA)


def test_loading_stops_on_a_missing_tensor(tmp_path, monkeypatch):
    from safetensors.torch import load_file, save_file

    snap, _ = _fake_hub(tmp_path, monkeypatch, LLAMA, seed=2)
    f = snap / "model.safetensors"
    state = load_file(str(f))
    state.pop("model.layers.1.mlp.up_proj.weight")
    save_file(state, str(f), metadata={"format": "pt"})
    with pytest.raises(ValueError, match="does not match"):
        model_loader.load_model_and_tokenizer(LLAMA, revision=SHA)


# --------------------------------------------------------------------------- the runner
def test_plain_035_runs_35_percent_of_the_steps():
    assert "plain_035" in runner.ARMS and runner.PLAIN_FRACTIONS["plain_035"] == 0.35
    assert "plain_035" in runner.PLAIN_ARMS and "plain_035" not in runner.STORED_SCORE_ARMS
    for T, steps in ((345, 121), (234, 82), (1150, 402)):
        cfg = _cfg("plain_035", total_steps=T)
        assert cfg.plain_fraction_steps == steps and cfg.arm_horizon() == steps
        assert cfg.warmup_steps_for(steps) == int(0.06 * steps)


class _CausalAdapter(torch.nn.Module):
    """Feeds the CPU fixture batches (float vectors) to a tiny round-18 classifier as token ids
    ending in EOS, with right padding, so the runner trains the real modules end to end."""

    def __init__(self, name, seed=0):
        super().__init__()
        torch.manual_seed(seed)
        self.inner = _classifier(name, seed=seed)
        self.inner.gradient_checkpointing_enable()
        self.model = self.inner.model
        self.config = self.inner.config

    @property
    def is_gradient_checkpointing(self):
        return self.inner.is_gradient_checkpointing

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        ids = (input_ids.float().clamp(0, 0.999) * 100).long() + 10
        mask = torch.ones_like(ids)
        cut = 4 + (ids[:, 0] % 4)
        for i in range(ids.shape[0]):
            ids[i, cut[i] - 1] = 1
            mask[i, cut[i]:] = 0
            ids[i, cut[i]:] = 1
        return self.inner(input_ids=ids, attention_mask=mask, labels=labels)


def _cfg(arm, total_steps=48, **kw):
    d = dict(task="mrpc", arm=arm, controller_version="legacy_v4", score_source="uniform",
             total_steps=total_steps, min_step=3, n_phases=1, max_consecutive_skips=4, policy_seed=0,
             target_skip_rate=0.6, output_dir="/tmp/test_round18", attempt_id="t18", capture_logits=True,
             lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
             train_batch_size=8, lr_schedule_shape="linear", lr_warmup_ratio=0.06, learning_rate=1e-5)
    d.update(kw)
    return ExperimentConfig(**d)


R18_ARMS = ["full_finetune", "plain_035", "equal_compute", "lerna2_nolr_b035", "lerna2_nolr_b060", "randkeep_b035",
            "infobatch"]


@pytest.mark.parametrize("name", [LLAMA, QWEN, GRANITE])
@pytest.mark.parametrize("arm", R18_ARMS)
def test_the_runner_trains_each_llm_with_each_method(arm, name, monkeypatch):
    monkeypatch.setattr(runner, "_load_model", lambda cfg, model_loader=None: _CausalAdapter(name, cfg.policy_seed))
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    res = result["result"]
    assert res["final_learning_rate"] == 0.0 and result["config"]["learning_rate"] == 1e-5
    rt = result["runtime"]
    assert rt["gradient_checkpointing"] is True and rt["model_type"] == cl.CAUSAL_LM_MODELS[name][1]
    assert rt["model_class"] == "_CausalAdapter" and rt["model_dtype"] == "torch.float32"
    horizon = _cfg(arm).arm_horizon()
    assert res["forwards"] == horizon and res["backwards"] == horizon and res["skipped"] == 0
    if arm.startswith(("lerna2", "randkeep")):
        sel = res["selection"]
        assert sel["policy"] == "lerna_budget" and abs(sel["realized_fraction"] - sel["budget"]) < 0.05


@pytest.mark.parametrize("name", [LLAMA, GRANITE])
def test_the_runner_runs_lerna_v1_on_the_llms(name, monkeypatch):
    monkeypatch.setattr(runner, "_load_model", lambda cfg, model_loader=None: _CausalAdapter(name, cfg.policy_seed))
    result = run_execution(_cfg("gate_s1_p4", total_steps=60, controller_version="phase_exact_v1", score_source="s1",
                                n_phases=4, min_step=3))
    assert result["status"] == "OK", result.get("reason")
    assert result["result"]["skipped"] == 36 and result["result"]["backwards"] == 24


def _fake_glue(monkeypatch, rows):
    """datasets.load_dataset stand-in: a small MRPC-like split."""
    import datasets

    def load_dataset(path, name=None, split=None, revision=None, **kw):
        n = rows if split == "train" else 6
        data = {"sentence1": [f"the cat sat {i}" for i in range(n)],
                "sentence2": [f"a dog ran in the park {'x ' * (i % 5)}".strip() for i in range(n)],
                "label": [i % 2 for i in range(n)], "idx": list(range(n))}
        return datasets.Dataset.from_dict(data)

    monkeypatch.setattr(datasets, "load_dataset", load_dataset)


@pytest.mark.parametrize("name", [LLAMA, QWEN, GRANITE])
@pytest.mark.parametrize("arm", ["full_finetune", "lerna2_nolr_b035"])
def test_the_production_path_end_to_end_on_cpu(tmp_path, monkeypatch, name, arm):
    """The real loaders (Hub cache, the round-18 tokenizer, the runner's tokenization and collator)."""
    _fake_hub(tmp_path, monkeypatch, name, seed=12)
    _fake_glue(monkeypatch, rows=24)
    cfg = _cfg(arm, total_steps=12, dry_run=False, device="cpu", model_id=name, tokenizer_id=name,
               model_revision=SHA, tokenizer_revision=SHA, dataset_revision=SHA, output_dir=str(tmp_path / "out"))
    ds = runner._load_train_dataset_production(cfg, 12)
    tok = runner._load_tokenizer_production(cfg)
    first = ds[0]
    assert int(first["input_ids"][-1]) == tok.eos_token_id and int(first["attention_mask"].sum()) == len(
        first["input_ids"])
    assert first["input_ids"].tolist() == tok("the cat sat 0", "a dog ran in the park")["input_ids"]
    result = run_execution(cfg)
    assert result["status"] == "OK", result.get("reason")
    rt = result["runtime"]
    assert rt["model_class"] == "CausalLMForSequenceClassification" and rt["gradient_checkpointing"] is True
    assert rt["attn_implementation"] == "eager" and rt["model_type"] == cl.CAUSAL_LM_MODELS[name][1]
    assert 0.0 <= result["result"]["final_eval_accuracy"] <= 1.0
