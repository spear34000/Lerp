"""Multimodal checkpoints (Gemma 4 style): nested text_config and audio/vision towers."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

import lerp.experiment as exp
from lerp.weighting import is_non_text_tensor, language_config, language_layer_count


def _nested_checkpoint(root: Path, text_value: float, audio_value: float) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps({
        "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
        "text_config": {"model_type": "gemma4_text", "num_hidden_layers": 2, "hidden_size": 2, "vocab_size": 10},
        "audio_config": {"num_hidden_layers": 12},
    }))
    (root / "tokenizer.json").write_text("{}")
    save_file({
        "model.language_model.layers.0.mlp.up_proj.weight": torch.full((2, 2), text_value),
        "model.language_model.layers.1.mlp.up_proj.weight": torch.full((2, 2), text_value + 1),
        # Index 7 is a position inside the 12-layer audio tower, not valid for a 2-layer text model.
        "model.audio_tower.layers.7.self_attn.q_proj.weight": torch.full((2, 2), audio_value),
        "model.embed_tokens.weight": torch.full((2, 2), 0.5),
    }, str(root / "model.safetensors"))


def _read(model: Path, name: str):
    index = json.loads((model / "model.safetensors.index.json").read_text())
    with safe_open(str(model / index["weight_map"][name]), framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def test_helpers_read_nested_text_config_and_detect_towers():
    nested = {"text_config": {"num_hidden_layers": 42, "hidden_size": 8}, "audio_config": {"num_hidden_layers": 12}}
    assert language_layer_count(nested) == 42
    assert language_config({"num_hidden_layers": 3}) == {"num_hidden_layers": 3}
    assert is_non_text_tensor("model.audio_tower.layers.3.mlp.weight")
    assert is_non_text_tensor("model.vision_tower.encoder.layers.0.self_attn.q_proj.weight")
    assert not is_non_text_tensor("model.language_model.layers.3.mlp.up_proj.weight")
    assert not is_non_text_tensor("model.layers.3.self_attn.q_proj.weight")


def test_lite_merges_multimodal_checkpoint_without_misreading_tower_depth(tmp_path):
    _nested_checkpoint(tmp_path / "base", 0.0, 0.0)
    _nested_checkpoint(tmp_path / "a", 1.0, 10.0)
    _nested_checkpoint(tmp_path / "b", 5.0, 30.0)
    cfg = {"name": "mm", "base_model": str(tmp_path / "base"),
           "parents": [{"name": "a", "model": str(tmp_path / "a")}, {"name": "b", "model": str(tmp_path / "b")}],
           "method": "linear", "genes": 3, "out_dtype": "float32"}
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(cfg))
    run = tmp_path / "run"
    exp.init_run(path, run)
    model = exp.build_candidate(run, 0, 0, engine="lite")  # candidate 0: 0.25 parent a, 0.75 parent b
    assert torch.allclose(_read(model, "model.language_model.layers.0.mlp.up_proj.weight"), torch.full((2, 2), 0.25 * 1 + 0.75 * 5))
    assert torch.allclose(_read(model, "model.language_model.layers.1.mlp.up_proj.weight"), torch.full((2, 2), 0.25 * 2 + 0.75 * 6))
    assert torch.allclose(_read(model, "model.audio_tower.layers.7.self_attn.q_proj.weight"), torch.full((2, 2), 0.25 * 10 + 0.75 * 30))


@pytest.mark.parametrize("method", ["linear", "task_arithmetic"])
def test_chunked_large_tensor_path_matches_in_memory_path(tmp_path, monkeypatch, method):
    import lerp.lite as lite

    def run(subdir: str):
        root = tmp_path / subdir
        for name, text, audio in (("base", 0.0, 0.0), ("a", 1.0, 10.0), ("b", 5.0, 30.0)):
            _nested_checkpoint(root / name, text, audio)
        # A tensor whose rows differ, so chunking mistakes would show up.
        for name, scale in (("base", 0.0), ("a", 1.0), ("b", 2.0)):
            tensors = {"model.language_model.embed_tokens.weight": torch.arange(14, dtype=torch.float32).reshape(7, 2) * scale + scale}
            with safe_open(str(root / name / "model.safetensors"), framework="pt") as f:
                for key in f.keys():
                    tensors[key] = f.get_tensor(key)
            save_file(tensors, str(root / name / "model.safetensors"))
        cfg = {"name": "mm", "base_model": str(root / "base"),
               "parents": [{"name": "a", "model": str(root / "a")}, {"name": "b", "model": str(root / "b")}],
               "method": method, "genes": 3, "out_dtype": "bfloat16", "task_scale": 0.8}
        path = root / "experiment.yaml"
        path.write_text(yaml.safe_dump(cfg))
        exp.init_run(path, root / "run")
        return exp.build_candidate(root / "run", 0, 0, engine="lite")

    reference = run("plain")
    monkeypatch.setattr(lite, "LARGE_TENSOR_ELEMS", 1)
    monkeypatch.setattr(lite, "CHUNK_ELEMS", 3)  # forces several 1-2 row chunks
    chunked = run("chunked")
    index = json.loads((reference / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == set(json.loads((chunked / "model.safetensors.index.json").read_text())["weight_map"])
    for name in index:
        assert torch.equal(_read(reference, name), _read(chunked, name)), name
