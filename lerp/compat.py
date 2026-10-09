"""Conservative metadata compatibility checks (never load remote Python code)."""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from .spec import Parent, Spec

# Checking config.json is necessary but not sufficient. The actual weights and
# their ancestry must also be compatible, and tokenizer equivalence matters.
CONFIG_KEYS = (
    "model_type", "architectures", "hidden_size", "num_hidden_layers",
    "num_attention_heads", "num_key_value_heads", "intermediate_size",
    "vocab_size", "head_dim", "tie_word_embeddings", "rope_theta",
    "hidden_act", "rms_norm_eps", "max_position_embeddings",
)

SOFT_CONFIG_KEYS = frozenset({"max_position_embeddings"})


@dataclass
class Report:
    errors: list[str]
    warnings: list[str]
    checked: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


_MSYS_DRIVE = re.compile(r"^/([A-Za-z])(?:/(.*))?$")


def normalize_platform_path(ref: str) -> str:
    """Turn Git-Bash/MSYS style '/c/Users/x' into 'C:/Users/x' on Windows.

    Otherwise Windows resolves it to 'C:\\c\\Users\\x', which does not exist and
    produces a misleading "Missing config.json" error.
    """
    if os.name == "nt":
        match = _MSYS_DRIVE.match(ref)
        if match:
            return f"{match.group(1).upper()}:/{match.group(2) or ''}"
    return ref


def local_reference(ref: str) -> bool:
    ref = normalize_platform_path(ref)
    return (
        Path(ref).exists() or ref.startswith(("./", "../", ".\\", "..\\", "/", "\\\\"))
        or bool(re.match(r"^[A-Za-z]:[/\\]", ref))
    )


def absolute_local_ref(ref: str, config_dir: Path) -> str:
    ref = normalize_platform_path(ref)
    if local_reference(ref) or ref.startswith(("./", "../", ".\\", "..\\")):
        return str((config_dir / ref).resolve())
    return ref


def resolve_spec_paths(spec: Spec, config_dir: Path) -> Spec:
    from dataclasses import replace

    return replace(
        spec,
        base_model=absolute_local_ref(spec.base_model, config_dir),
        parents=tuple(
            Parent(p.name, absolute_local_ref(p.model, config_dir)) for p in spec.parents
        ),
    )


def _get_metadata(ref: str, remote: bool) -> tuple[dict | None, Path | None]:
    if local_reference(ref):
        root = Path(ref)
        path = root / "config.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing checkpoint config.json: {path}")
    elif remote:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise RuntimeError("Install huggingface_hub for remote metadata checks") from exc
        path = Path(hf_hub_download(repo_id=ref, filename="config.json"))
        root = None
    else:
        return None, None
    doc = json.loads(path.read_text(encoding="utf-8"))
    # Multimodal configs (Gemma 4, ...) nest language-model settings under text_config; surface them
    # so hidden_size / num_hidden_layers / ... are still compared between checkpoints.
    text = doc.get("text_config")
    if isinstance(text, dict):
        for key in CONFIG_KEYS:
            if key in text and key not in doc:
                doc[key] = text[key]
    return doc, root


def _safetensors_index(root: Path) -> dict[str, tuple[int, ...]] | None:
    """Read tensor shape headers without materializing tensor data in RAM."""
    files = sorted(root.glob("*.safetensors"))
    if not files:
        return None
    try:
        from safetensors import safe_open
    except ImportError:
        return None
    index: dict[str, tuple[int, ...]] = {}
    for file in files:
        with safe_open(file, framework="np", device="cpu") as handle:
            for name in handle.keys():
                if name in index:
                    raise ValueError(f"Duplicated tensor {name} in {root}")
                index[name] = tuple(handle.get_slice(name).get_shape())
    return index


def _check_lora_compatibility(spec: Spec, remote: bool = False) -> Report:
    """Adapter metadata is not a full model config, never treat it as one."""
    from .lora import LoRAMergeError, verify_adapters, _adapter_source
    errors, warnings, checked = [], [], []
    try:
        base_config, base_root = _get_metadata(spec.base_model, remote)
        if base_config is None:
            warnings.append("Base metadata unchecked: remote model reference without --remote")
        else:
            checked.append(f"base: {spec.base_model}")
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        errors.append(f"base: {exc}")
        base_root = None
    try:
        meta = verify_adapters(spec)
        for parent in spec.parents:
            checked.append(f"adapter {parent.name}: {parent.model}")
        declared = meta['declared_base']
        if declared != spec.base_model:
            if not (local_reference(declared) and local_reference(spec.base_model) and
                    Path(declared).resolve() == Path(spec.base_model).resolve()):
                warnings.append(f"Declared adapter base {declared!r} differs from experiment base {spec.base_model!r}; manually verify revision/ancestry")
        if meta['output_rank'] > 256:
            warnings.append(f"Combined LoRA rank {meta['output_rank']} is large; adapter may be slow or memory-heavy")
        for parent in spec.parents:
            cfg, weights_file = _adapter_source(Path(parent.model))
            index = _safetensors_index(weights_file.parent)
            if index is None or not index:
                errors.append(f"{parent.name}: adapter safetensors shape headers unavailable")
            elif not any(k.endswith('.lora_A.weight') for k in index):
                errors.append(f"{parent.name}: no standard LoRA matrix pairs in safetensors")
            if cfg.get('lora_dropout', 0):
                warnings.append(f"{parent.name}: input dropout is nonzero; inference mode disables it, but train-time equivalence is not guaranteed")
    except (LoRAMergeError, ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
        errors.append(f"Adapter compatibility: {exc}")
    warnings.append("Adapter config strings cannot prove identical base *revisions*; pin a commit and verify tokenizers")
    return Report(errors=errors, warnings=warnings, checked=checked)


# Parts of tokenizer.json that decide which ids a text maps to. post_processor (BOS/EOS insertion), truncation and
# padding only change how special tokens are added, not what the weights' embeddings mean. Added tokens are compared
# separately: a token registered in only one checkpoint (e.g. <think>) leaves ordinary text tokenization unchanged.
_TOKENIZER_CORE = ("model", "normalizer", "pre_tokenizer", "decoder")


def _normalized_core(doc: dict) -> str:
    core = {k: doc.get(k) for k in _TOKENIZER_CORE}
    model = core.get("model")
    if isinstance(model, dict) and isinstance(model.get("merges"), list):
        # "a b" strings and ["a", "b"] pairs are two serializations of the same merge rule
        model = dict(model)
        model["merges"] = [m if isinstance(m, str) else " ".join(m) for m in model["merges"]]
        model["ignore_merges"] = bool(model.get("ignore_merges"))  # null and false are the same default
        core["model"] = model
    return json.dumps(core, sort_keys=True, ensure_ascii=False)


def _tokenizer_fingerprints(raw: bytes) -> tuple[str, str, dict]:
    """(hash of id-defining parts, hash of whole file, added tokens by id). Unparseable files fall back to the raw hash."""
    full = hashlib.sha256(raw).hexdigest()
    try:
        doc = json.loads(raw.decode("utf-8"))
        added = {int(t["id"]): json.dumps(t, sort_keys=True, ensure_ascii=False) for t in doc.get("added_tokens") or []}
        return hashlib.sha256(_normalized_core(doc).encode("utf-8")).hexdigest(), full, added
    except (UnicodeDecodeError, ValueError, AttributeError, KeyError, TypeError):
        return full, full, {}


def check_compatibility(spec: Spec, remote: bool = False) -> Report:
    if spec.mode == "lora":
        return _check_lora_compatibility(spec, remote)
    errors: list[str] = []
    warnings: list[str] = []
    checked: list[str] = []
    sources = [("base", spec.base_model)] + [(p.name, p.model) for p in spec.parents]
    metadata: dict[str, dict] = {}
    roots: dict[str, Path] = {}
    for name, ref in sources:
        try:
            config, root = _get_metadata(ref, remote=remote)
            if config is None:
                warnings.append(f"{name}: remote ref {ref!r}; metadata unchecked (use --remote)")
            else:
                metadata[name] = config
                if root is not None:
                    roots[name] = root
                checked.append(f"{name}: {ref}")
        except (ValueError, FileNotFoundError, RuntimeError, OSError) as exc:
            errors.append(f"{name}: {exc}")
        except Exception as exc:
            errors.append(f"{name}: metadata lookup failed: {exc}")

    base = metadata.get("base")
    if base is not None:
        for parent in spec.parents:
            meta = metadata.get(parent.name)
            if meta is None:
                continue
            for key in CONFIG_KEYS:
                if key in base and key in meta and base[key] != meta[key]:
                    message = f"{parent.name}: {key} differs from base: {meta[key]!r} != {base[key]!r}"
                    # Context-length metadata changes no tensor shape or weight meaning.
                    (warnings if key in SOFT_CONFIG_KEYS else errors).append(message)
            if not all(k in meta for k in ("model_type", "hidden_size", "num_hidden_layers")):
                warnings.append(f"{parent.name}: incomplete config.json; tensor compatibility not fully checked")

    core_hashes: dict[str, str] = {}
    full_hashes: dict[str, str] = {}
    added_tokens: dict[str, dict] = {}
    for name, root in roots.items():
        tok = root / "tokenizer.json"
        if tok.is_file():
            core_hashes[name], full_hashes[name], added_tokens[name] = _tokenizer_fingerprints(tok.read_bytes())
        else:
            warnings.append(f"{name}: tokenizer.json not found; verify tokenizer manually")
    conflict = False
    names = list(added_tokens)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            if any(added_tokens[first][k] != added_tokens[second][k] for k in added_tokens[first].keys() & added_tokens[second].keys()):
                conflict = True
    if len(set(core_hashes.values())) > 1 or conflict:
        errors.append("Tokenizer JSON files differ. Align vocab and tokens before attempting a merge")
    elif len({tuple(sorted(a.items())) for a in added_tokens.values()}) > 1:
        warnings.append("Some added/special tokens are registered in only one checkpoint (for example <think>); "
                        "ordinary text tokenizes identically. The child inherits one parent's tokenizer files")
    elif len(set(full_hashes.values())) > 1:
        warnings.append(
            "Tokenizer vocabulary/merges/added tokens match, but other tokenizer.json settings differ "
            "(e.g. post_processor: automatic <bos> insertion). The merged child inherits one parent's "
            "tokenizer files; evaluate every model with the same tokenizer behaviour")
    # Only safetensors header metadata is inspected; weights are never loaded.
    tensor_indexes: dict[str, dict[str, tuple[int, ...]]] = {}
    for name, root in roots.items():
        try:
            index = _safetensors_index(root)
            if index is not None:
                tensor_indexes[name] = index
            else:
                warnings.append(f"{name}: safetensors headers unavailable; tensor shapes not preflight-checked")
        except (ValueError, OSError, RuntimeError) as exc:
            errors.append(f"{name}: malformed safetensors checkpoint: {exc}")
    if "base" in tensor_indexes:
        base_index = tensor_indexes["base"]
        for parent in spec.parents:
            index = tensor_indexes.get(parent.name)
            if index is None:
                continue
            missing = set(base_index) - set(index)
            extra = set(index) - set(base_index)
            if missing or extra:
                errors.append(f"{parent.name}: tensor key mismatch versus base; missing={len(missing)}, extra={len(extra)}; examples: {sorted(missing | extra)[:4]}")
            bad_shapes = [(key, base_index[key], index[key]) for key in base_index.keys() & index.keys()
                          if base_index[key] != index[key]]
            if bad_shapes:
                errors.append(f"{parent.name}: {len(bad_shapes)} tensor shape mismatch(es): {bad_shapes[:3]}")
    warnings.append("Metadata cannot prove common fine-tuning ancestry; confirm all parents truly descend from the declared base")
    warnings.append("Metadata check cannot verify every tensor. MergeKit validates checkpoint tensors during the actual build")
    return Report(errors=errors, warnings=warnings, checked=checked)
