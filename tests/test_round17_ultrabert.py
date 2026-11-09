"""Round 17: UltraBERT (the second model) in LERNA's own implementation, its loading, DropBP on
its layers, the --learning-rate option, and the runner end to end with it."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lerna.utils import ultrabert_model as ub  # noqa: E402
from lerna.utils import model_loader  # noqa: E402
from lerna.utils.compared_methods import DropBPPolicy  # noqa: E402
import run_controller_ablation as runner  # noqa: E402
from run_controller_ablation import ExperimentConfig, run_execution  # noqa: E402

SHA = "0123456789abcdef0123456789abcdef01234567"


def _config(**kw):
    d = dict(vocab_size=64, hidden_size=32, intermediate_size=40, num_hidden_layers=6, num_attention_heads=2,
             max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2, sliding_window=3,
             prediction_head_size=16, num_labels=2)
    d.update(kw)
    return ub.UltraBertConfig(**d)


def _tiny(seed=0, **kw):
    torch.manual_seed(seed)
    cfg = _config(**kw)
    cfg._attn_implementation = "eager"
    return ub.UltraBertForSequenceClassification(cfg)


# --------------------------------------------------------------------------- configuration
def test_config_defaults_follow_transformers_5_22():
    c = ub.UltraBertConfig(num_hidden_layers=24, hidden_size=768, num_attention_heads=6, intermediate_size=64)
    assert c.layer_types == ["full_attention" if (i + 1) % 3 == 0 else "sliding_attention" for i in range(24)]
    assert c.sliding_window == 128 and c.head_dim == 128 and c.num_key_value_heads == 6
    assert c.rope_parameters == {"sliding_attention": {"rope_type": "default", "rope_theta": 10_000.0},
                                 "full_attention": {"rope_type": "default", "rope_theta": 2_560_000.0}}
    assert c.rms_norm_eps == 1e-6 and c.hidden_act == "silu" and c.classifier_pooling == "cls"
    assert c.attention_bias is False and c.mlp_bias is False and c.tie_word_embeddings is False


def test_config_reads_the_original_checkpoints_fields():
    c = ub.UltraBertConfig(num_hidden_layers=3, hidden_size=32, num_attention_heads=2, intermediate_size=8,
                           attention_windows=[2, 2, 0], rope_theta=[10000.0, 10000.0, 2560000.0])
    assert c.sliding_window == 2
    assert c.layer_types == ["sliding_attention", "sliding_attention", "full_attention"]
    assert c.rope_parameters == {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                 "full_attention": {"rope_type": "default", "rope_theta": 2560000.0}}
    assert "rope_theta" not in c.to_dict() and "attention_windows" not in c.to_dict()


def test_config_rejects_what_is_not_implemented():
    with pytest.raises(NotImplementedError):
        ub.UltraBertConfig(num_hidden_layers=3, hidden_size=32, num_attention_heads=2, rope_parameters={
            "sliding_attention": {"rope_type": "linear", "rope_theta": 1.0, "factor": 2.0},
            "full_attention": {"rope_type": "default", "rope_theta": 1.0}})
    with pytest.raises(ValueError):
        ub.UltraBertConfig(num_hidden_layers=2, hidden_size=32, num_attention_heads=2, layer_types=["x", "y"])
    with pytest.raises(ValueError):
        ub.UltraBertConfig(hidden_size=30, num_attention_heads=4)


# --------------------------------------------------------------------------- the encoder
def test_sliding_layers_see_exactly_the_window():
    model = _tiny().eval()
    ids = torch.randint(3, 64, (2, 12))
    with torch.no_grad():
        att = model.model(input_ids=ids, output_attentions=True).attentions
    w = model.config.sliding_window
    for a, kind in zip(att, model.config.layer_types):
        if kind == "sliding_attention":
            assert torch.all(a[:, :, 0, w] > 0) and torch.all(a[:, :, 0, w + 1] == 0)
            assert torch.all(a[:, :, w, 0] > 0) and torch.all(a[:, :, w + 1, 0] == 0)
        else:
            assert torch.all(a[:, :, 0, -1] > 0)


def test_padding_does_not_change_a_sequence():
    model = _tiny(classifier_pooling="cls").eval()
    ids = torch.randint(3, 64, (1, 9))
    padded = torch.cat([ids, torch.zeros(1, 5, dtype=torch.long)], dim=1)
    mask = torch.cat([torch.ones(1, 9, dtype=torch.long), torch.zeros(1, 5, dtype=torch.long)], dim=1)
    with torch.no_grad():
        a = model(input_ids=ids).logits
        b = model(input_ids=padded, attention_mask=mask).logits
    assert torch.allclose(a, b, atol=1e-6)
    mean = _tiny(classifier_pooling="mean").eval()
    with torch.no_grad():
        assert torch.allclose(mean(input_ids=ids).logits, mean(input_ids=padded, attention_mask=mask).logits,
                              atol=1e-6)


def test_classifier_output_loss_and_extra_inputs():
    model = _tiny()
    ids = torch.randint(3, 64, (3, 7))
    labels = torch.tensor([0, 1, 1])
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels,
                token_type_ids=torch.zeros_like(ids), num_items_in_batch=3)
    assert out.logits.shape == (3, 2)
    assert torch.allclose(out.loss, torch.nn.functional.cross_entropy(out.logits, labels))
    assert model.config.problem_type == "single_label_classification"
    with pytest.raises(ValueError):
        model(input_ids=ids, attention_mask=torch.ones(3, 6, dtype=torch.long))


def test_grouped_query_attention_and_explicit_head_size():
    model = _tiny(num_attention_heads=4, num_key_value_heads=2, head_dim=16, attention_bias=True, mlp_bias=True)
    attn = model.model.layers[0].self_attn
    assert attn.q_proj.weight.shape == (64, 32) and attn.k_proj.weight.shape == (32, 32)
    out = model(input_ids=torch.randint(3, 64, (2, 6)), labels=torch.tensor([0, 1]))
    out.loss.backward()
    assert attn.k_proj.bias.grad is not None


# --------------------------------------------------------------------------- checkpoints
def _write_snapshot(folder: Path, model, hub_keys=True, extra=None, config_extra=None):
    from safetensors.torch import save_file

    folder.mkdir(parents=True, exist_ok=True)
    mlm = ub.UltraBertForMaskedLM(model.config)
    mlm.model.load_state_dict(model.model.state_dict())
    sd = {}
    for k, v in mlm.state_dict().items():
        if hub_keys and k.startswith("model."):
            k = "ultrabert." + k[len("model."):]
        elif hub_keys and k == "head.decoder.weight":
            k = "cls.weight"
        sd[k] = v.contiguous().clone()
    sd.update(extra or {})
    save_file(sd, str(folder / "model.safetensors"))
    cfg = model.config.to_dict()
    for k in ("num_labels", "id2label", "label2id", "problem_type"):
        cfg.pop(k, None)
    cfg.update(config_extra or {})
    (folder / "config.json").write_text(json.dumps(cfg))
    return mlm


@pytest.mark.parametrize("hub_keys", [True, False])
def test_checkpoint_loads_with_only_the_classifier_new(tmp_path, hub_keys):
    src = _tiny(seed=3)
    _write_snapshot(tmp_path, src, hub_keys=hub_keys,
                    extra={"head_50.linear1.weight": torch.zeros(16, 32)})
    model = ub.load_ultrabert_for_sequence_classification("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path)
    rep = model.lerna_checkpoint_report
    assert set(rep["missing"]) == {"classifier.weight", "classifier.bias"}
    assert set(rep["unused"]) == {"head.linear1.weight", "head.linear2.weight", "head.decoder.weight",
                                  "head_50.linear1.weight"}
    for k, v in src.model.state_dict().items():
        assert torch.equal(model.model.state_dict()[k], v)
    assert next(model.parameters()).dtype == torch.float32
    assert model.config._attn_implementation == "eager"
    mlm = ub.load_ultrabert_for_masked_lm("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path)
    assert mlm.lerna_checkpoint_report["missing"] == []


def test_checkpoint_problems_stop_the_load(tmp_path):
    src = _tiny(seed=3)
    _write_snapshot(tmp_path / "a", src, extra={"ultrabert.layers.9.mlp.up_proj.weight": torch.zeros(40, 32)})
    with pytest.raises(ValueError, match="unexpected"):
        ub.load_ultrabert_for_sequence_classification("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path / "a")
    _write_snapshot(tmp_path / "b", src)
    from safetensors.torch import load_file, save_file
    sd = load_file(str(tmp_path / "b" / "model.safetensors"))
    sd.pop("ultrabert.layers.0.mlp.up_proj.weight")
    save_file(sd, str(tmp_path / "b" / "model.safetensors"))
    with pytest.raises(ValueError, match="lacks"):
        ub.load_ultrabert_for_sequence_classification("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path / "b")
    _write_snapshot(tmp_path / "c", src, config_extra={"intermediate_size": 44})
    with pytest.raises(ValueError, match="shapes"):
        ub.load_ultrabert_for_sequence_classification("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path / "c")


def _write_tokenizer(folder: Path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors

    words = ["<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>", "<|mask|>"] + [f"w{i}" for i in range(59)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<|unk|>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.post_processor = processors.TemplateProcessing(
        single="<|bos|> $A <|eos|>", pair="<|bos|> $A <|eos|> $B <|eos|>",
        special_tokens=[("<|bos|>", 1), ("<|eos|>", 2)])
    from tokenizers import AddedToken
    tok.add_special_tokens([AddedToken(w, special=True) for w in words[:5]])
    tok.save(str(folder / "tokenizer.json"))
    (folder / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "TokenizersBackend", "bos_token": "<|bos|>", "eos_token": "<|eos|>",
        "pad_token": "<|pad|>", "unk_token": "<|unk|>", "mask_token": "<|mask|>",
        "model_input_names": ["input_ids", "attention_mask"], "padding_side": "left"}))


def _fake_hub(tmp_path, monkeypatch, model):
    import huggingface_hub.constants as hc

    cache = tmp_path / "hub"
    repo = cache / "models--dmis-lab--UltraBERT-150M"
    snap = repo / "snapshots" / SHA
    _write_snapshot(snap, model)
    _write_tokenizer(snap)
    (repo / "refs").mkdir(parents=True)
    (repo / "refs" / "main").write_text(SHA)
    monkeypatch.setattr(hc, "HF_HUB_CACHE", str(cache))
    return snap


def test_model_loader_loads_ultrabert_from_the_hub_cache(tmp_path, monkeypatch):
    src = _tiny(seed=5)
    _fake_hub(tmp_path, monkeypatch, src)
    model, tok = model_loader.load_model_and_tokenizer("dmis-lab/UltraBERT-150M", num_labels=2, revision=SHA,
                                                        local_files_only=True)
    assert isinstance(model, ub.UltraBertForSequenceClassification)
    assert tok.padding_side == "right" and tok.pad_token_id == 0 and tok.mask_token_id == 4
    enc = tok(["w1 w2 w3", "w4"], ["w5", "w6 w7"], truncation=True, max_length=512)
    assert enc["input_ids"][0] == [1, 6, 7, 8, 2, 10, 2] and "token_type_ids" not in enc
    tok2 = model_loader.load_tokenizer("dmis-lab/UltraBERT-150M", revision=SHA, local_files_only=True)
    assert tok2("w1")["input_ids"] == [1, 6, 2]
    for bad in (None, "main", SHA[:12]):
        with pytest.raises(ValueError):
            model_loader.load_model_and_tokenizer("dmis-lab/UltraBERT-150M", revision=bad)
    with pytest.raises(Exception):
        model_loader.load_model_and_tokenizer("dmis-lab/UltraBERT-150M", revision="f" * 40)  # not in the cache
    batch = tok(["w1 w2", "w3"], padding=True, return_tensors="pt")
    with torch.no_grad():
        assert model(**batch).logits.shape == (2, 2)


def test_dropbp_macs_from_the_configuration():
    c = _config(num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    assert ub.dropbp_macs_from_config(c) == {"layers": 6, "attn_linear": 32 * 64 + 2 * 32 * 32 + 64 * 32,
                                             "attn_scores_per_position": 128, "mlp": 3 * 32 * 40}


# --------------------------------------------------------------------------- DropBP on UltraBERT
def _grads(model, ids, mask, labels):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, attention_mask=mask, labels=labels)
    out.loss.backward()
    return float(out.loss), {n: (None if p.grad is None else p.grad.clone()) for n, p in model.named_parameters()}


def test_dropbp_on_ultrabert_layers():
    model = _tiny(seed=1, num_hidden_layers=3)
    model.train()
    ids = torch.randint(3, 64, (4, 10))
    mask = torch.ones_like(ids)
    mask[0, 7:] = 0
    labels = torch.tensor([0, 1, 1, 0])
    l0, g0 = _grads(model, ids, mask, labels)
    pol = DropBPPolicy(model=model, drop_rate=0.5, total_steps=10)
    assert len(pol._blocks) == 6
    assert pol.flops_per_token == ub.dropbp_macs_from_config(model.config)
    l1, g1 = _grads(model, ids, mask, labels)
    assert l0 == l1 and all(torch.equal(g0[n], g1[n]) for n in g0)
    for li, kind, mod in pol._blocks:
        mod._lerna_dropbp = (li == 1 and kind == "attn") or (li == 2 and kind == "mlp")
    l2, g2 = _grads(model, ids, mask, labels)
    assert l2 == l0
    assert sorted(n for n in g2 if g2[n] is None) == sorted([
        "model.layers.1.self_attn.q_proj.weight", "model.layers.1.self_attn.k_proj.weight",
        "model.layers.1.self_attn.v_proj.weight", "model.layers.1.self_attn.o_proj.weight",
        "model.layers.1.input_layernorm.weight",
        "model.layers.2.mlp.gate_proj.weight", "model.layers.2.mlp.up_proj.weight",
        "model.layers.2.mlp.down_proj.weight", "model.layers.2.post_attention_layernorm.weight"])
    assert g2["model.embed_tokens.weight"] is not None and g2["classifier.weight"] is not None


# --------------------------------------------------------------------------- the runner
def test_learning_rate_option(tmp_path):
    parser_args = SimpleNamespace(lr_schedule_shape=None, lr_warmup_steps=None, lr_warmup_ratio=None, device=None,
                                  infobatch_prune_ratio=None, infobatch_anneal_fraction=None, learning_rate=1e-5)
    assert runner._recipe_overrides(parser_args) == {"learning_rate": 1e-5}
    parser_args.learning_rate = None
    assert runner._recipe_overrides(parser_args) == {}


class _UltraAdapter(torch.nn.Module):
    """Feeds the CPU fixture batches (float vectors) to a tiny UltraBERT as token ids, with a
    padding mask, so the runner trains the real UltraBERT modules end to end."""

    def __init__(self, seed=0):
        super().__init__()
        self.inner = _tiny(seed=seed, num_hidden_layers=3, sliding_window=2)
        self.model = self.inner.model
        self.config = self.inner.config

    def forward(self, input_ids, labels=None, attention_mask=None, **kwargs):
        ids = (input_ids.float().clamp(0, 0.999) * 59).long() + 5
        mask = torch.ones_like(ids)
        cut = 4 + (ids[:, 0] % 4)
        for i in range(ids.shape[0]):
            mask[i, cut[i]:] = 0
            ids[i, cut[i]:] = 0
        return self.inner(input_ids=ids, attention_mask=mask, labels=labels)


def _cfg(arm, total_steps=48, **kw):
    d = dict(task="mrpc", arm=arm, controller_version="legacy_v4", score_source="uniform",
             total_steps=total_steps, min_step=3, n_phases=1, max_consecutive_skips=4, policy_seed=0,
             target_skip_rate=0.6, output_dir="/tmp/test_round17", attempt_id="t17", capture_logits=True,
             lr_warmup_steps=0, probe_exclusions_file=None, probe_exclusions_hash=None, dry_run=True,
             train_batch_size=8, lr_schedule_shape="linear", lr_warmup_ratio=0.06, learning_rate=1e-5)
    d.update(kw)
    return ExperimentConfig(**d)


ARMS = ["full_finetune", "plain_050", "lerna2_nolr_b035", "lerna2_b060", "randkeep_b040", "el2n_topk_b035",
        "egreedy_b035", "ucb_b035", "orderdp_b035", "isfresh_b060", "sb_b083", "kakurenbo", "ies", "infobatch",
        "dropbp_p050", "dropbp_p0875"]


@pytest.mark.parametrize("arm", ARMS)
def test_the_runner_trains_ultrabert_with_each_method(arm, monkeypatch):
    monkeypatch.setattr(runner, "_load_model", lambda cfg, model_loader=None: _UltraAdapter(cfg.policy_seed))
    result = run_execution(_cfg(arm))
    assert result["status"] == "OK", result.get("reason")
    res = result["result"]
    assert res["final_learning_rate"] == 0.0
    assert result["config"]["learning_rate"] == 1e-5 if "config" in result else True
    if arm.startswith("dropbp"):
        sel = res["selection"]
        assert sel["flops_per_token"]["layers"] == 3 and len(sel["drop_counts"]) == 48
        assert res["compute_accounting"]["backward_token_equivalents"] < res["compute_accounting"][
            "forward_padded_tokens"]


@pytest.mark.parametrize("arm", ["gate_s1_p4", "control_uniform_p4"])
def test_the_runner_runs_the_batch_gates_on_ultrabert(arm, monkeypatch):
    monkeypatch.setattr(runner, "_load_model", lambda cfg, model_loader=None: _UltraAdapter(cfg.policy_seed))
    score = "s1" if arm == "gate_s1_p4" else "uniform"
    result = run_execution(_cfg(arm, total_steps=60, controller_version="phase_exact_v1", score_source=score,
                                n_phases=4, min_step=3))
    assert result["status"] == "OK", result.get("reason")
    assert result["result"]["skipped"] == 36 and result["result"]["backwards"] == 24


def test_tokenizer_matches_transformers_5_22_on_a_5_22_saved_tokenizer(tmp_path):
    """A tokenizer saved by transformers 5.22.2 (tokenizer_class TokenizersBackend, which 4.56's
    AutoTokenizer does not know) loads, and encodes sentences and pairs as 5.22.2's AutoTokenizer did."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import round17_tokenizer_fixture as fx

    (tmp_path / "tokenizer.json").write_text(fx.TOKENIZER_JSON)
    (tmp_path / "tokenizer_config.json").write_text(fx.TOKENIZER_CONFIG_JSON)
    tok = ub.load_ultrabert_tokenizer("dmis-lab/UltraBERT-150M", revision=SHA, snapshot=tmp_path)
    s = ["The capital of France is <|mask|>.", "The capital of Germany is <|mask|>.", "Revenue rose."]
    p = (["He said hello to her.", "Revenue rose"], ["She said hi back.", "the firm said."])
    ref = fx.REFERENCE
    assert ref["transformers"] == "5.22.2" and ref["class"] == "TokenizersBackend"
    assert tok(s)["input_ids"] == ref["single"]
    assert tok(s, padding=True)["input_ids"] == ref["single_padded"]
    assert tok(*p, truncation=True, max_length=512)["input_ids"] == ref["pair"]
    assert sorted(tok(*p).keys()) == ref["pair_keys"]
    assert tok.pad_token_id == ref["pad"] and tok.mask_token_id == ref["mask"]
    assert tok.batch_decode([[tok.convert_tokens_to_ids("ĠParis")]]) == ref["decode"]
    assert tok.convert_ids_to_tokens(tok(s[0])["input_ids"])[5:7] == ["Ġis", "<|mask|>"]
