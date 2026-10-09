"""Shared genome interpretation for full tensors and LoRA factors.

Genomes are grouped by tensor families; each family has independent layer-wise
per-parent simplex coefficients. Naming heuristics are explicit and testable.
"""
from __future__ import annotations

import math
import re

from .genetics import weights_by_parent
from .spec import Spec

_LAYER = re.compile(r"(?:^|\.)(?:layers|h|blocks)\.(\d+)\.")
# Multimodal checkpoints (Gemma 4, Qwen-VL, ...) hold audio/vision towers whose "layers.N" are NOT
# positions in the language model's depth; they must not be interpolated along text depth.
_NON_TEXT = ("audio", "vision", "visual", "image", "video", "multi_modal", "mm_projector")


def is_non_text_tensor(name: str) -> bool:
    lower = name.lower()
    return any(marker in lower for marker in _NON_TEXT)


def language_config(config: dict) -> dict:
    """Language-model settings: multimodal configs nest them under ``text_config``."""
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


def language_layer_count(config: dict):
    cfg = language_config(config)
    return cfg.get("num_hidden_layers") or cfg.get("n_layer") or cfg.get("num_layers")


# Ordered (group, substrings) rules. First match wins. Mixture-of-experts routers are checked first: they are
# tiny, decide which expert runs and are usually the most fragile tensors to interpolate, so they get their own group
# (when "router" is not one of the experiment's gene_groups they fall back to "other" like any unknown tensor).
DEFAULT_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("router", ("router", ".mlp.gate.weight", "block_sparse_moe.gate.", "moe.gate.weight", "shared_expert_gate")),
    ("attention", ("self_attn", "attention", ".attn.", "q_proj", "k_proj", "v_proj", "o_proj")),
    ("mlp", (".mlp.", "feed_forward", "gate_proj", "up_proj", "down_proj", "gate_up_proj", ".fc1.", ".fc2.",
             ".experts.", "block_sparse_moe", "shared_expert")),
    ("norm", ("norm", "layernorm", "layer_norm")),
    ("embedding", ("embed_tokens", "embed_in", "wte.", "lm_head", "word_embeddings")),
)


def tensor_group(name: str, rules: tuple[tuple[str, str], ...] = ()) -> str:
    """Module family of a tensor. ``rules`` are user (regex, group) pairs from the experiment; they win over the defaults,
    so a new model family can be supported from configuration without touching this file."""
    for pattern, group in rules:
        if re.search(pattern, name):
            return group
    lower = name.lower()
    for group, markers in DEFAULT_RULES:
        if any(x in lower for x in markers):
            return group
    return "other"


def tensor_coefficients(spec: Spec, genes: list[float], name: str, layer_count: int) -> list[float]:
    if len(genes) != spec.genome_size:
        raise ValueError(f"Expected {spec.genome_size} genes, got {len(genes)}")
    group = tensor_group(name, spec.tensor_rules)
    group_name = group if group in spec.gene_groups else ("other" if "other" in spec.gene_groups else "all")
    group_idx = spec.gene_groups.index(group_name)
    start = group_idx * spec.block_size
    matrices = weights_by_parent(genes[start:start + spec.block_size], len(spec.parents), spec.genes)
    match = None if is_non_text_tensor(name) else _LAYER.search(name)
    if match:
        layer = int(match.group(1))
        if layer < 0 or layer >= layer_count:
            raise ValueError(f"Unexpected layer index {layer} in {name}")
        x = layer / max(1, layer_count - 1) * (spec.genes - 1)
        left, right = math.floor(x), math.ceil(x)
        t = x - left
        return [float(a[left] * (1 - t) + a[right] * t) for a in matrices]
    return [sum(row) / len(row) for row in matrices]
