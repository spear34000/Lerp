"""Resident evaluator: blended-in-place weights must equal the checkpoints the merge engines write; scoring must be exact."""
from __future__ import annotations

import json
import random
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from transformers import AutoModelForCausalLM, LlamaConfig, OlmoeConfig, PreTrainedTokenizerFast  # noqa: E402

import lerp.experiment as exp  # noqa: E402
from lerp.lite import build_lite  # noqa: E402
from lerp.resident import ResidentError, ResidentSession  # noqa: E402
from lerp.search import search_run  # noqa: E402
from lerp.spec import parse_spec, spec_to_dict  # noqa: E402

TASK = {"dataset": "local/toy", "split": "test", "prompt": "{q}", "choices": {"field": "options"}, "label": {"field": "gold"}}
IDS = torch.tensor([[3, 7, 11, 5, 9, 2, 40, 41]])


def _rows(n: int = 12, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)

    def word() -> str:
        return f"w{rng.randrange(1, 50)}"

    return [{"q": " ".join(word() for _ in range(rng.randrange(3, 7))),
             "options": [" ".join(word() for _ in range(rng.randrange(1, 4))) for _ in range(3)],
             "gold": rng.randrange(3)} for _ in range(n)]


def _save_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers
    words = ["[UNK]"] + [f"w{i}" for i in range(1, 63)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]").save_pretrained(path)


def _llama(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, max_position_embeddings=128, tie_word_embeddings=True)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    _save_tokenizer(path)


def _olmoe(path: Path, seed: int) -> None:
    torch.manual_seed(seed)
    cfg = OlmoeConfig(vocab_size=64, hidden_size=32, intermediate_size=16, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=4, num_experts=4, num_experts_per_tok=2, max_position_embeddings=128,
                      tie_word_embeddings=False)
    AutoModelForCausalLM.from_config(cfg).save_pretrained(path)
    _save_tokenizer(path)


def _spec(base: Path, parents: list[Path], **extra):
    raw = {"name": "tiny", "base_model": str(base), "parents": [{"name": f"p{i}", "model": str(p)} for i, p in enumerate(parents)],
           "genes": 2, "out_dtype": "float32", "gene_groups": ["attention", "mlp", "other"], "population": 3,
           "evaluation": {"limit": 8, "tasks": {"toy": {"metric": "acc_norm,none", "task": TASK},
                                                "toy2": {"metric": "acc,none", "task": TASK}}}, **extra}
    return parse_spec(raw)


def _three(tmp_path: Path, maker=_llama) -> None:
    for name, seed in (("base", 1), ("p0", 2), ("p1", 3)):
        maker(tmp_path / name, seed)


def _session(spec, **kwargs) -> ResidentSession:
    return ResidentSession(spec, "cpu", "float32", rows={"toy": _rows(), "toy2": _rows()}, **kwargs)


def _logits(model, ids):
    with torch.no_grad():
        return model(input_ids=ids).logits


def _load(path: Path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()


def test_full_checkpoint_blend_equals_the_lite_engine_output(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"])
    genes = [0.8, 0.8, 0.2, 0.2, 0.5, 0.5]
    build_lite(spec, genes, tmp_path / "child")
    session = _session(spec)
    session.blender.apply(genes)
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child"), IDS), atol=1e-6)
    second = [0.1, 0.1, 0.9, 0.9, 0.5, 0.5]
    session.blender.apply(second)  # a later candidate starts from the parents, not from the previous blend
    build_lite(spec, second, tmp_path / "child2")
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child2"), IDS), atol=1e-6)
    session.close()


def test_task_arithmetic_blend_equals_the_lite_engine_output(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], method="task_arithmetic", task_scale=0.7)
    genes = [0.3, 0.3, 0.6, 0.6, 0.5, 0.5]
    build_lite(spec, genes, tmp_path / "child")
    session = _session(spec)
    session.blender.apply(genes)
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child"), IDS), atol=1e-6)
    session.close()


def test_lora_blend_equals_the_lora_engine_output(tmp_path):
    peft = pytest.importorskip("peft")
    from lerp.lora import build_lora
    _llama(tmp_path / "base", 1)
    adapters = []
    for n in range(2):
        torch.manual_seed(10 + n)
        wrapped = peft.get_peft_model(_load(tmp_path / "base"), peft.LoraConfig(
            r=2 + n, lora_alpha=4, target_modules=["q_proj", "v_proj", "up_proj"], lora_dropout=0.0))
        with torch.no_grad():
            for name, p in wrapped.named_parameters():
                if "lora_B" in name:
                    p.normal_(0, 0.2)
        wrapped.save_pretrained(tmp_path / f"a{n}")
        adapters.append(tmp_path / f"a{n}")
    spec = _spec(tmp_path / "base", adapters, mode="lora")
    genes = [0.7, 0.7, 0.3, 0.3, 0.4, 0.4]
    build_lora(spec, genes, tmp_path / "child")
    child = peft.PeftModel.from_pretrained(_load(tmp_path / "base"), tmp_path / "child").eval()
    session = _session(spec)
    session.blender.apply(genes)
    assert torch.allclose(_logits(session.model, IDS), _logits(child, IDS), atol=1e-4)
    session.blender.apply_reference("base")
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "base"), IDS), atol=1e-6)


def test_moe_blend_maps_per_expert_checkpoint_tensors_onto_fused_parameters(tmp_path):
    from safetensors import safe_open
    _three(tmp_path, _olmoe)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], gene_groups=["attention", "mlp", "router", "other"])
    genes = [0.9, 0.9, 0.1, 0.1, 0.6, 0.6, 0.3, 0.3]
    build_lite(spec, genes, tmp_path / "child")
    session = _session(spec)
    session.blender.apply(genes)
    assert torch.allclose(_logits(session.model, IDS), _logits(_load(tmp_path / "child"), IDS), atol=1e-6)
    keys = set()
    for f in (tmp_path / "p0").glob("*.safetensors"):
        with safe_open(str(f), "pt") as handle:
            keys |= set(handle.keys())
    if any(".experts.0.gate_proj.weight" in k for k in keys):  # per-expert checkpoint layout: the conversion had to run
        assert {"gate_up", "down"} <= {kind for _, kind, _ in session.blender.plan}


def _reference_accuracy(session: ResidentSession, rows, kind: str) -> float:
    hits = 0
    for row in rows:
        values = []
        for choice in row["options"]:
            ctx = session.tokenizer(row["q"])["input_ids"]
            whole = session.tokenizer(row["q"] + " " + choice)["input_ids"]
            with torch.no_grad():
                logits = session.model(input_ids=torch.tensor([whole])).logits[0]
            logp = torch.log_softmax(logits[len(ctx) - 1:len(whole) - 1], -1)
            total = float(sum(logp[i, t] for i, t in enumerate(whole[len(ctx):])))
            values.append(total / len(choice) if kind == "acc_norm" else total)
        hits += int(max(range(3), key=values.__getitem__) == row["gold"])
    return hits / len(rows)


def test_scorer_matches_a_brute_force_loglikelihood_computation(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"])
    rows = _rows(12)
    session = ResidentSession(spec, "cpu", "float32", rows={"toy": rows, "toy2": rows})
    session.blender.apply([0.5] * 6)
    metrics = session._score((0, 12), max_tokens=200)  # small batches exercise the padding
    assert metrics["toy"] == pytest.approx(_reference_accuracy(session, rows, "acc_norm"))
    assert metrics["toy2"] == pytest.approx(_reference_accuracy(session, rows, "acc"))
    assert session._score((0, 12), max_tokens=100000) == metrics  # batch size does not change the answer


def test_self_check_refuses_when_logits_are_post_processed(tmp_path):
    _three(tmp_path)
    session = _session(_spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"]))
    session.softcap = 0.02  # pretend the model soft-caps its logits but the resident path was not told
    with pytest.raises(ResidentError, match="disagrees"):
        session._self_check()


def test_search_run_scores_generations_in_the_normal_run_folder(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"], search="gp")
    config = tmp_path / "experiment.yaml"
    config.write_text(yaml.safe_dump(spec_to_dict(spec)))
    run = tmp_path / "run"
    exp.init_run(config, run)
    loaded, _ = exp.load_run(run)
    session = _session(loaded)
    logs: list[str] = []
    assert search_run(run, 2, session=session, baselines=True, log=logs.append) == [0, 1]
    for gen in (0, 1):
        for idx in range(3):
            score = json.loads((run / "generations" / f"gen-{gen:03d}" / f"cand-{idx:03d}" / "score.json").read_text())
            assert score["source"] == "lerp_eval" and score["evaluation_settings"]["backend"] == "lerp-resident/float32"
    assert (run / "baselines" / "base" / "score.json").is_file() and (run / "baselines" / "p1" / "score.json").is_file()
    assert (run / "resident_protocol.json").is_file()
    assert len(logs) == 6 + 3  # six candidates and three baselines
    assert len(exp.leaderboard(run, include_simulated=False)) == 6
    assert exp.comparisons(run)  # the resident evaluator counts as a verified source, comparable with its own baselines
    path = run / "resident_protocol.json"
    doc = json.loads(path.read_text())
    doc["items"] = [0, 99]
    path.write_text(json.dumps(doc))
    with pytest.raises(exp.BreederError, match="protocol changed"):
        search_run(run, 1, session=session)


def test_self_check_does_not_false_alarm_in_bfloat16(tmp_path):
    _three(tmp_path)
    spec = _spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"])
    session = ResidentSession(spec, "cpu", "bfloat16", rows={"toy": _rows(), "toy2": _rows()})  # the constructor runs the check
    session.blender.apply([0.5] * 6)
    assert set(session._score((0, 8), max_tokens=500)) == {"toy", "toy2"}


def test_scoring_retries_with_smaller_batches_after_an_out_of_memory_error(tmp_path, monkeypatch):
    _three(tmp_path)
    session = _session(_spec(tmp_path / "base", [tmp_path / "p0", tmp_path / "p1"]))
    session.blender.apply([0.5] * 6)
    expected = session._score((0, 8), max_tokens=100000)
    original, calls = session._logprobs, {"n": 0}

    def flaky(scorer, batch):
        calls["n"] += 1
        if len(batch) > 6:  # pretend big batches do not fit
            raise torch.OutOfMemoryError("simulated")
        return original(scorer, batch)

    monkeypatch.setattr(session, "_logprobs", flaky)
    assert session._score((0, 8), max_tokens=100000) == expected
    assert session._token_budget is not None and session._token_budget < 100000
