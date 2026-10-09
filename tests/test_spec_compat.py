import json
from pathlib import Path

import pytest

from lerp.compat import check_compatibility, resolve_spec_paths
from lerp.merge import mergekit_config
from lerp.spec import SpecError, load_spec, parse_spec


def sample_spec(method="linear"):
    return parse_spec({
        "name": "test", "base_model": "./base",
        "parents": [{"name": "a", "model": "./a"}, {"name": "b", "model": "./b"}],
        "method": method, "genes": 3,
        "evaluation": {"tasks": {"arc_easy": {"metric": "acc_norm,none", "weight": 2},
                                 "hellaswag": {"metric": "acc_norm,none", "weight": 1}}},
    })


def create_configs(tmp_path, *, bad=False):
    cfg = {"model_type": "llama", "hidden_size": 64, "num_hidden_layers": 2,
           "num_attention_heads": 4, "vocab_size": 32}
    for folder in ("base", "a", "b"):
        root = tmp_path / folder
        root.mkdir()
        current = dict(cfg)
        if bad and folder == "b":
            current["hidden_size"] = 128
        (root / "config.json").write_text(json.dumps(current))
        (root / "tokenizer.json").write_text('{"model": "toy"}')


def test_preflight_identical(tmp_path):
    create_configs(tmp_path)
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert report.ok, report.errors
    assert len(report.checked) == 3


def test_preflight_rejects_mismatch(tmp_path):
    create_configs(tmp_path, bad=True)
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert not report.ok
    assert "hidden_size" in " ".join(report.errors)


def test_missing_path_rejected(tmp_path):
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert not report.ok


def test_unequal_tokenizer_rejected(tmp_path):
    create_configs(tmp_path)
    (tmp_path / "b" / "tokenizer.json").write_text('{"model": "different"}')
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert not report.ok
    assert "Tokenizer" in " ".join(report.errors)


@pytest.mark.parametrize("method", ["linear", "task_arithmetic", "ties", "dare_ties"])
def test_genome_conversion(method):
    spec = sample_spec(method)
    cfg = mergekit_config(spec, [0.2, 0.5, 0.8])
    assert cfg["merge_method"] == method
    assert cfg["models"][0]["parameters"]["weight"] == [0.2, 0.5, 0.8]
    assert cfg["models"][1]["parameters"]["weight"] == [0.8, 0.5, 0.2]
    if method == "linear":
        assert cfg["parameters"]["normalize"]
        assert "base_model" not in cfg
    else:
        assert cfg["base_model"] == spec.base_model
    if method in ("ties", "dare_ties"):
        assert cfg["models"][0]["parameters"]["density"] == 0.5


def test_invalid_spec():
    with pytest.raises(SpecError):
        parse_spec({"name": "t", "base_model": "base", "parents": [
            {"name": "a", "model": "x"}, {"name": "a", "model": "y"}]})
    with pytest.raises(SpecError):
        parse_spec({"name": "t", "base_model": "base", "parents": [
            {"name": "a", "model": "x"}, {"name": "b", "model": "y"}], "genes": 1})


def _tokenizer(root: Path, *, vocab=("a", "b"), post=None):
    (root / "tokenizer.json").write_text(json.dumps({
        "model": {"type": "BPE", "vocab": list(vocab)}, "added_tokens": [], "normalizer": None,
        "pre_tokenizer": None, "decoder": None, "post_processor": post}))


def test_tokenizer_post_processor_difference_is_a_warning_not_an_error(tmp_path):
    create_configs(tmp_path)
    bos = {"type": "TemplateProcessing", "single": [{"SpecialToken": {"id": "<bos>", "type_id": 0}}]}
    _tokenizer(tmp_path / "base", post=bos)
    _tokenizer(tmp_path / "a", post=bos)
    _tokenizer(tmp_path / "b", post=None)  # e.g. an -it checkpoint that adds <bos> via its chat template
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert report.ok, report.errors
    assert any("post_processor" in w for w in report.warnings)


def test_tokenizer_vocab_difference_is_still_an_error(tmp_path):
    create_configs(tmp_path)
    for folder in ("base", "a"):
        _tokenizer(tmp_path / folder)
    _tokenizer(tmp_path / "b", vocab=("a", "c"))
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert any("Tokenizer JSON files differ" in e for e in report.errors)


def test_tokenizer_serialization_and_extra_added_tokens_are_warnings(tmp_path):
    """Qwen3 base vs post-trained: merges as strings vs pairs, ignore_merges null vs false, extra <think> tokens."""
    create_configs(tmp_path)

    def write(root, merges, ignore, added):
        (root / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}, "merges": merges, "ignore_merges": ignore},
            "added_tokens": added, "normalizer": None, "pre_tokenizer": None, "decoder": None, "post_processor": None}))

    base_tokens = [{"id": 5, "content": "<s>", "special": True}]
    write(tmp_path / "base", ["a b"], None, base_tokens)
    write(tmp_path / "a", ["a b"], None, base_tokens)
    write(tmp_path / "b", [["a", "b"]], False, base_tokens + [{"id": 6, "content": "<think>", "special": False}])
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert report.ok, report.errors
    assert any("registered in only one checkpoint" in w for w in report.warnings)


def test_conflicting_added_token_ids_are_an_error(tmp_path):
    create_configs(tmp_path)
    for folder, content in (("base", "<s>"), ("a", "<s>"), ("b", "<other>")):
        (tmp_path / folder / "tokenizer.json").write_text(json.dumps({
            "model": {"type": "BPE", "vocab": {"a": 0}, "merges": []}, "added_tokens": [{"id": 5, "content": content}]}))
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert any("Tokenizer JSON files differ" in e for e in report.errors)


def test_context_length_difference_is_only_a_warning(tmp_path):
    create_configs(tmp_path)
    cfg = json.loads((tmp_path / "b" / "config.json").read_text())
    cfg["max_position_embeddings"] = 40960
    (tmp_path / "b" / "config.json").write_text(json.dumps(cfg))
    base_cfg = json.loads((tmp_path / "base" / "config.json").read_text())
    base_cfg["max_position_embeddings"] = 32768
    (tmp_path / "base" / "config.json").write_text(json.dumps(base_cfg))
    report = check_compatibility(resolve_spec_paths(sample_spec(), tmp_path))
    assert report.ok, report.errors
    assert any("max_position_embeddings" in w for w in report.warnings)
