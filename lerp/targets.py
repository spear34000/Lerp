"""Which Linear layers of an arbitrary Hugging Face model should receive LoRA?

Training and merging must agree on module families, so this reuses the merge-side tensor rules (`tensor_group`).
Selected: attention and dense-MLP projections of the *language model*. Skipped: mixture-of-experts experts
(hundreds of tiny Linear layers per block would make the adapter huge), routers, norms, embeddings / LM head and
non-text towers (vision, audio).
"""
from __future__ import annotations

from typing import Iterable

from .weighting import is_non_text_tensor, tensor_group


def lora_target_modules(named_linears: Iterable[str], rules: tuple[tuple[str, str], ...] = (),
                        groups: tuple[str, ...] = ("attention", "mlp")) -> list[str]:
    """Filter module names (as given by ``model.named_modules()`` for ``nn.Linear`` layers)."""
    selected = []
    for name in named_linears:
        if name.endswith("lm_head") or is_non_text_tensor(name) or ".experts." in name:
            continue
        if tensor_group(name + ".weight", rules) in groups:
            selected.append(name)
    return selected


def discover(model, rules: tuple[tuple[str, str], ...] = (), groups: tuple[str, ...] = ("attention", "mlp")) -> list[str]:
    import torch
    return lora_target_modules((n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)), rules, groups)
