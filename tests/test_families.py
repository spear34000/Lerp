"""Model-family coverage: tensor-name rules (dense, multimodal, mixture-of-experts) and user-defined rules."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

from lerp.spec import SpecError, parse_spec, spec_to_dict
from lerp.weighting import tensor_group

CASES = {
    # dense decoder-only
    "model.layers.0.self_attn.q_proj.weight": "attention",
    "model.layers.0.mlp.up_proj.weight": "mlp",
    "model.layers.0.input_layernorm.weight": "norm",
    "model.embed_tokens.weight": "embedding",
    "lm_head.weight": "embedding",
    "transformer.h.3.attn.c_attn.weight": "attention",
    # Gemma 4 mixture-of-experts: fused expert tensors, router, dense shared MLP
    "model.language_model.layers.3.experts.gate_up_proj": "mlp",
    "model.language_model.layers.3.experts.down_proj": "mlp",
    "model.language_model.layers.3.router.proj.weight": "router",
    "model.language_model.layers.3.router.per_expert_scale": "router",
    "model.language_model.layers.3.mlp.gate_proj.weight": "mlp",
    # Qwen3-MoE / OLMoE: per-expert tensors, router named mlp.gate, shared expert
    "model.layers.2.mlp.gate.weight": "router",
    "model.layers.2.mlp.experts.17.up_proj.weight": "mlp",
    "model.layers.2.mlp.shared_expert.down_proj.weight": "mlp",
    "model.layers.2.mlp.shared_expert_gate.weight": "router",
    # Mixtral
    "model.layers.1.block_sparse_moe.gate.weight": "router",
    "model.layers.1.block_sparse_moe.experts.3.w1.weight": "mlp",
}


@pytest.mark.parametrize("name,group", CASES.items())
def test_default_rules_cover_dense_and_moe_families(name, group):
    assert tensor_group(name) == group


def test_dense_gate_proj_is_not_mistaken_for_a_router():
    assert tensor_group("model.layers.0.mlp.gate_proj.weight") == "mlp"


def test_user_rules_support_an_unfamiliar_family_without_code_changes():
    name = "transformer.blocks.0.ffn.w1.weight"
    assert tensor_group(name) == "other"
    assert tensor_group(name, ((r"\.ffn\.", "mlp"),)) == "mlp"


def _raw(**extra):
    return {"name": "fam", "base_model": "./base", "genes": 2, "gene_groups": ["attention", "mlp", "router", "other"],
            "parents": [{"name": "a", "model": "./a"}, {"name": "b", "model": "./b"}],
            "evaluation": {"tasks": {"x": {"metric": "acc,none"}, "y": {"metric": "acc,none"}}}, **extra}


def test_spec_accepts_router_group_and_round_trips_rules():
    spec = parse_spec(_raw(tensor_rules=[{"match": r"\.ffn\.", "group": "mlp"}]))
    assert spec.tensor_rules == ((r"\.ffn\.", "mlp"),)
    again = parse_spec(yaml.safe_load(yaml.safe_dump(spec_to_dict(spec))))
    assert again.tensor_rules == spec.tensor_rules


def test_spec_rejects_bad_rules():
    with pytest.raises(SpecError, match="invalid regex"):
        parse_spec(_raw(tensor_rules=[{"match": "(", "group": "mlp"}]))
    with pytest.raises(SpecError, match="unknown group"):
        parse_spec(_raw(tensor_rules=[{"match": "x", "group": "banana"}]))


def _moe_checkpoint(root: Path, value: float) -> None:
    root.mkdir(parents=True)
    (root / "config.json").write_text(json.dumps({"model_type": "toy_moe", "num_hidden_layers": 1, "hidden_size": 2}))
    save_file({
        "model.layers.0.self_attn.q_proj.weight": torch.full((2, 2), value),
        "model.layers.0.mlp.experts.0.up_proj.weight": torch.full((2, 2), value + 1),
        "model.layers.0.experts.gate_up_proj": torch.full((4, 2, 2), value + 2),  # fused experts, 3-D
        "model.layers.0.mlp.gate.weight": torch.full((2, 2), value + 3),           # router
        "model.embed_tokens.weight": torch.full((2, 2), value + 4),
    }, str(root / "model.safetensors"))


@pytest.mark.parametrize("force_chunking", [False, True])
def test_lite_merge_applies_a_separate_weight_to_each_moe_group(tmp_path, monkeypatch, force_chunking):
    import lerp.lite as lite
    if force_chunking:
        monkeypatch.setattr(lite, "LARGE_TENSOR_ELEMS", 1)
        monkeypatch.setattr(lite, "CHUNK_ELEMS", 3)  # splits the (4, 2, 2) expert tensor into 1-expert chunks
    _moe_checkpoint(tmp_path / "a", 0.0)
    _moe_checkpoint(tmp_path / "b", 10.0)
    raw = _raw()
    raw["base_model"], raw["out_dtype"] = str(tmp_path / "a"), "float32"
    raw["parents"] = [{"name": "a", "model": str(tmp_path / "a")}, {"name": "b", "model": str(tmp_path / "b")}]
    spec = parse_spec(raw)
    # weight of parent a per group, two control points each: attention 0.0, mlp 1.0, router 0.5, other 0.25
    genes = [0.0, 0.0, 1.0, 1.0, 0.5, 0.5, 0.25, 0.25]
    out = tmp_path / "child"
    lite.build_lite(spec, genes, out)
    index = json.loads((out / "model.safetensors.index.json").read_text())["weight_map"]

    def read(name):
        with safe_open(str(out / index[name]), framework="pt") as f:
            return f.get_tensor(name)

    assert torch.allclose(read("model.layers.0.self_attn.q_proj.weight"), torch.full((2, 2), 10.0))        # attention: all parent b
    assert torch.allclose(read("model.layers.0.mlp.experts.0.up_proj.weight"), torch.full((2, 2), 1.0))    # mlp: all parent a
    assert torch.allclose(read("model.layers.0.experts.gate_up_proj"), torch.full((4, 2, 2), 2.0))         # fused experts follow mlp
    assert torch.allclose(read("model.layers.0.mlp.gate.weight"), torch.full((2, 2), 0.5 * 3 + 0.5 * 13))  # router: 50/50
    assert torch.allclose(read("model.embed_tokens.weight"), torch.full((2, 2), 0.25 * 4 + 0.75 * 14))    # embedding -> other


def test_lora_targets_skip_experts_routers_towers_and_the_head():
    from lerp.targets import lora_target_modules
    names = [
        "model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj",
        "model.layers.0.mlp.gate_proj",                       # dense MLP (or Gemma 4 shared MLP): trained
        "model.layers.0.mlp.experts.5.up_proj",               # one of hundreds of experts: skipped
        "model.layers.0.mlp.gate",                            # router: skipped
        "model.layers.0.input_layernorm", "lm_head",
        "model.vision_tower.encoder.layers.0.self_attn.q_proj",  # non-text tower: skipped
    ]
    assert lora_target_modules(names) == ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj",
                                          "model.layers.0.mlp.gate_proj"]
    assert lora_target_modules(names, groups=("attention",)) == ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.o_proj"]
