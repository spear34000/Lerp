"""Generate compatible MergeKit configs for two-to-six parent convex merges."""
from __future__ import annotations

from .genetics import weights_by_parent
from .spec import Spec


def mergekit_config(spec: Spec, genes: list[float], *, method: str | None = None) -> dict:
    chosen_method = method or spec.method
    if chosen_method == "slerp":  # implemented by lerp's own engines; no MergeKit recipe is emitted
        if len(spec.parents) != 2:
            raise ValueError("slerp merges exactly two parents")
        return {"merge_method": "slerp", "backend": "lerp_only", "note": "SLERP runs in the lite engine and the resident evaluator",
                "genome": list(genes), "gene_groups": list(spec.gene_groups)}
    if chosen_method not in {"linear", "task_arithmetic", "ties", "dare_ties", "dare_linear"}:
        raise ValueError(f"Invalid concrete merge method: {chosen_method}")
    if len(genes) != spec.genome_size:
        raise ValueError(f"Expected {spec.genome_size} genes, got {len(genes)}")
    if not all(0 <= x <= 1 for x in genes):
        raise ValueError("Gene values must be within [0,1]")
    # MergeKit gradient parameters vary over layer depth; a module-family-specific
    # genome cannot be faithfully represented in a single generic YAML recipe.
    # Such recipes use the tested Lite backend; fail closed rather than ignore genes.
    if len(spec.gene_groups) > 1:
        return {"merge_method": chosen_method, "backend": "modelbreeder_lite_only",
                "note": "Grouped attention/MLP genes require the Lite backend (or LoRA engine)",
                "genome": list(genes), "gene_groups": list(spec.gene_groups)}
    weights = weights_by_parent(genes, len(spec.parents), spec.genes)
    models = []
    for parent, values in zip(spec.parents, weights):
        params: dict = {"weight": values}
        if chosen_method in ("ties", "dare_ties", "dare_linear"):
            params["density"] = spec.density
        models.append({"model": parent.model, "parameters": params})
    cfg = {"merge_method": chosen_method, "models": models, "out_dtype": spec.out_dtype}
    if chosen_method == "linear":
        cfg["parameters"] = {"normalize": True}
        cfg["tokenizer"] = {"source": spec.parents[0].model}
    else:
        cfg["base_model"] = spec.base_model
        cfg["parameters"] = {"lambda": spec.task_scale}
        cfg["tokenizer"] = {"source": "base"}
    return cfg
