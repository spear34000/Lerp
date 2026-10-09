"""Resident evaluation: one model stays on the accelerator, candidates are written into it in place and scored in seconds.

A normal cycle pays a fixed price per candidate (write the merged weights, start the harness, load the model and datasets again).
Merging is linear in the weights, so a candidate can instead be produced by overwriting the target tensors of a model that is already
loaded, and scored by the same process:

* ``LoraBlender``       W = W0 + sum_i c_i * scale_i * B_i @ A_i           (the coefficients ``build_lora`` uses)
* ``CheckpointBlender`` W = sum_i c_i * P_i   (linear)  or  base + sum_i s * c_i * (P_i - base)   (task arithmetic),
                        read tensor by tensor from the parents' safetensors files, fp32 accumulation, cast to the model dtype
                        (the arithmetic of the ``lite`` engine). Checkpoint names are mapped onto the loaded model, including
                        per-expert tensors that transformers fuses into 3-D parameters.

``Scorer`` reproduces lm-eval's log-likelihood scoring (prompt, continuation tokenization, ``acc`` / ``acc_norm``). Before any
candidate is scored the session cross-checks its fast path against the model's own forward pass and refuses to run if they
disagree (for example a model that post-processes its logits in a way this module does not know).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import re
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Any, Sequence

from .spec import Spec
from .tasks import metric_kind, resolve_task
from .weighting import language_config, language_layer_count, tensor_coefficients

CHUNK_ELEMS = 32 * 1024 * 1024  # rows of very large tensors are blended in pieces of about this many elements
PROTOCOL_VERSION = 1


class ResidentError(RuntimeError):
    pass


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise ResidentError('The resident evaluator needs torch: pip install -e ".[lora]"') from exc
    return torch


def _is_out_of_memory(exc: BaseException) -> bool:
    """torch.OutOfMemoryError (torch >= 2.5) or the RuntimeError older builds raise for CUDA / XPU allocation failures."""
    return type(exc).__name__ == "OutOfMemoryError" or "out of memory" in str(exc).lower()


def _sync(device) -> None:
    torch = _torch()
    if str(device).startswith("xpu") and hasattr(torch, "xpu"):
        torch.xpu.synchronize()
    elif str(device).startswith("cuda"):
        torch.cuda.synchronize()


# --------------------------------------------------------------------------------------------------- scoring
class Scorer:
    """All requests of one window of items; ``score`` returns {task: accuracy}."""

    def __init__(self, spec: Spec, tokenizer, lo: int, hi: int, *, add_special_tokens: bool = False,
                 max_length: int = 4096, rows: dict[str, Sequence[Any]] | None = None):
        if not spec.evaluation.tasks:
            raise ResidentError("No evaluation.tasks configured")
        self.window = (lo, hi)
        choice_tasks = [t for t in spec.evaluation.tasks if not resolve_task(t.name, t.definition).generative]
        self.kinds = {t.name: metric_kind(t.metric) for t in choice_tasks}
        self.requests: list[tuple[str, int, int, list[int], list[int]]] = []  # (task, doc, choice, context ids, continuation ids)
        self.gold: dict[tuple[str, int], int] = {}
        self.char_len: dict[tuple[str, int, int], float] = {}
        for task in choice_tasks:
            definition = resolve_task(task.name, task.definition)
            docs = definition.documents(lo, hi, rows=None if rows is None else rows[task.name])
            if not docs:
                raise ResidentError(f"task {task.name}: no documents in items [{lo}, {hi})")
            for d, (context, choices, label) in enumerate(docs):
                self.gold[(task.name, d)] = label
                for c, choice in enumerate(choices):
                    ctx_ids, cont_ids = self._encode_pair(tokenizer, context, " " + choice, add_special_tokens, max_length)
                    self.requests.append((task.name, d, c, ctx_ids, cont_ids))
                    self.char_len[(task.name, d, c)] = float(len(choice))
        self.order = sorted(range(len(self.requests)), key=lambda i: len(self.requests[i][3]) + len(self.requests[i][4]))
        self.by_doc: dict[tuple[str, int], list[int]] = {}
        for i, request in enumerate(self.requests):
            self.by_doc.setdefault((request[0], request[1]), []).append(i)

    @staticmethod
    def _encode_pair(tokenizer, context: str, continuation: str, special: bool, max_length: int) -> tuple[list[int], list[int]]:
        """lm-eval's ``_encode_pair``: trailing context whitespace moves to the continuation; the continuation is the
        token suffix of the joint encoding."""
        spaces = len(context) - len(context.rstrip())
        if spaces:
            continuation = context[-spaces:] + continuation
            context = context[:-spaces]
        whole = tokenizer(context + continuation, add_special_tokens=special)["input_ids"]
        ctx = tokenizer(context, add_special_tokens=special)["input_ids"]
        ctx_ids, cont_ids = whole[:len(ctx)], whole[len(ctx):]
        if not cont_ids:
            raise ResidentError(f"empty continuation after tokenization: {continuation!r}")
        overflow = len(ctx_ids) + len(cont_ids) - max_length
        if overflow > 0:  # keep the end of the context, like lm-eval's left truncation
            ctx_ids = ctx_ids[overflow:]
        return ctx_ids, cont_ids

    @property
    def tokens(self) -> int:
        return sum(len(r[3]) + len(r[4]) for r in self.requests)

    def batches(self, max_tokens: int):
        pos = 0
        while pos < len(self.order):
            width = 0
            batch: list[int] = []
            while pos < len(self.order):
                i = self.order[pos]
                w = len(self.requests[i][3]) + len(self.requests[i][4])
                if batch and (len(batch) + 1) * max(w, width) > max_tokens:
                    break
                batch.append(i)
                width = max(width, w)
                pos += 1
            yield batch

    def accuracy(self, loglik: dict[int, float]) -> dict[str, float]:
        metrics = {}
        for name, kind in self.kinds.items():
            docs = sorted(d for (task, d) in self.gold if task == name)
            hits = 0
            for d in docs:
                indexes = self.by_doc[(name, d)]
                values = [loglik[i] / (self.char_len[(name, d, self.requests[i][2])] if kind == "acc_norm" else 1.0) for i in indexes]
                hits += int(max(range(len(values)), key=values.__getitem__) == self.gold[(name, d)])
            metrics[name] = hits / len(docs)
        return metrics


# --------------------------------------------------------------------------------------------------- blenders
class LoraBlender:
    """Overwrites the adapted Linear weights with ``W0 + sum_i c_i * scale_i * B_i @ A_i``."""

    def __init__(self, spec: Spec, model, device):
        from safetensors import safe_open
        from .lora import verify_adapters
        torch = _torch()
        self.spec, self.device = spec, device
        verify_adapters(spec)
        self.n_layers = language_layer_count(json.loads((Path(spec.base_model) / "config.json").read_text(encoding="utf-8")))
        if not isinstance(self.n_layers, int):
            raise ResidentError("base_model/config.json has no layer count")
        modules: dict[str, dict[str, Any]] = {}
        for parent in spec.parents:
            adapter = Path(parent.model)
            cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
            scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
            weights = adapter / "adapter_model.safetensors"
            with safe_open(str(weights), framework="pt") as f:
                for key in f.keys():
                    if ".lora_A." not in key:
                        continue
                    prefix = key.split(".lora_A.")[0]
                    entry = modules.setdefault(prefix, {"A": [], "B": [], "scale": []})
                    entry["A"].append(f.get_tensor(key).to(device, torch.float32))
                    entry["B"].append(f.get_tensor(key.replace(".lora_A.", ".lora_B.")).to(device, torch.float32))
                    entry["scale"].append(scale)
        if not modules:
            raise ResidentError("no LoRA modules found in the adapters")
        for prefix, entry in modules.items():
            name = prefix.replace("base_model.model.", "", 1)
            try:
                layer = model.get_submodule(name)
            except AttributeError as exc:
                raise ResidentError(f"adapter module {prefix!r} does not exist in the model ({name!r})") from exc
            weight = getattr(layer, "weight", None)
            B, A = torch.cat(entry["B"], 1), torch.cat(entry["A"], 0)
            if weight is None or weight.ndim != 2 or weight.shape != (B.shape[0], A.shape[1]):
                raise ResidentError(f"{name}: weight {None if weight is None else tuple(weight.shape)} does not match the adapter "
                                    f"delta {(B.shape[0], A.shape[1])}")
            entry.update(layer=layer, W0=weight.detach().cpu().clone(), A_cat=A, B_cat=B, ranks=[a.shape[0] for a in entry["A"]])
        self.modules = modules

    def _write(self, coefficients: dict[str, Sequence[float]], task_scale: float) -> None:
        torch = _torch()
        with torch.no_grad():
            for prefix, e in self.modules.items():
                columns = torch.cat([torch.full((r,), c * s * task_scale, device=self.device)
                                     for r, c, s in zip(e["ranks"], coefficients[prefix], e["scale"])])
                delta = (e["B_cat"] * columns) @ e["A_cat"]
                e["layer"].weight.copy_((e["W0"].to(self.device).float() + delta).to(e["layer"].weight.dtype))

    def apply(self, genes: Sequence[float], method: str | None = None) -> None:
        method = method or self.spec.method
        if method not in ("linear", "task_arithmetic"):
            raise ResidentError(f"LoRA blending supports linear and task_arithmetic, not {method}")
        self._write({p: tensor_coefficients(self.spec, list(genes), p, self.n_layers) for p in self.modules},
                    self.spec.task_scale if method == "task_arithmetic" else 1.0)

    def apply_reference(self, name: str) -> None:
        torch = _torch()
        if name == "base":
            with torch.no_grad():
                for e in self.modules.values():
                    e["layer"].weight.copy_(e["W0"].to(self.device))
            return
        index = [p.name for p in self.spec.parents].index(name)
        self._write({p: [1.0 if i == index else 0.0 for i in range(len(self.spec.parents))] for p in self.modules}, 1.0)


_FUSED = re.compile(r"^(?P<prefix>.*)\.experts\.(?P<kind>gate_up_proj|down_proj)$")


class CheckpointBlender:
    """Writes ``sum_i c_i * P_i`` (or the task-arithmetic form) of full checkpoints into the loaded model, tensor by tensor."""

    def __init__(self, spec: Spec, model, device):
        from safetensors import safe_open
        from .lite import _tensor_index
        self.spec, self.device, self.model = spec, device, model
        self.n_layers = language_layer_count(json.loads((Path(spec.parents[0].model) / "config.json").read_text(encoding="utf-8")))
        if not isinstance(self.n_layers, int):
            raise ResidentError("config.json has no layer count (num_hidden_layers / text_config.num_hidden_layers)")
        self._stack = ExitStack()
        roots = [Path(spec.base_model)] + [Path(p.model) for p in spec.parents]
        self.indexes = [_tensor_index(root, self._stack, safe_open) for root in roots]  # [0] is the base
        reference = set(self.indexes[1])
        for root, index in zip(roots[2:], self.indexes[2:]):
            if set(index) != reference:
                raise ResidentError(f"{root}: tensor names differ from the first parent; run `lerp check`")
        self.state = model.state_dict()
        self.plan = self._plan(reference)

    def close(self) -> None:
        self._stack.close()

    def _plan(self, checkpoint_keys: set[str]) -> list[tuple[str, str, Any]]:
        """(model entry, kind, source) with kind 'direct' (source = checkpoint key) or 'gate_up' / 'down' (source = (prefix, E))."""
        plan: list[tuple[str, str, Any]] = []
        used: set[str] = set()
        unresolved = []
        for name, tensor in self.state.items():
            if name in checkpoint_keys:
                plan.append((name, "direct", name))
                used.add(name)
                continue
            fused = _FUSED.match(name)
            if fused:
                prefix, kind = fused["prefix"], fused["kind"]
                experts = 0
                while f"{prefix}.experts.{experts}.down_proj.weight" in checkpoint_keys:
                    experts += 1
                if experts:
                    parts = ("gate_proj", "up_proj") if kind == "gate_up_proj" else ("down_proj",)
                    keys = {f"{prefix}.experts.{e}.{part}.weight" for e in range(experts) for part in parts}
                    if keys <= checkpoint_keys:
                        plan.append((name, "gate_up" if kind == "gate_up_proj" else "down", (prefix, experts)))
                        used |= keys
                        continue
            unresolved.append((name, tensor))
        covered = {self.state[n].data_ptr() for n, _, _ in plan}
        for name, tensor in unresolved:
            if tensor.data_ptr() in covered:  # tied weight (lm_head sharing the embedding): its twin is written
                continue
            if tensor.numel() == 0:
                continue
            raise ResidentError(f"model entry {name!r} {tuple(tensor.shape)} has no counterpart in the checkpoints; "
                                "this architecture needs a name mapping (see ECOSYSTEM.md) - use the lm-eval path instead")
        self.unused = sorted(checkpoint_keys - used)  # e.g. Gemma 4 K/V tensors of layers that share another layer's cache
        return plan

    # -- tensor access --------------------------------------------------------------------------------
    def _source_tensor(self, source: int, key: str):
        return self.indexes[source][key].get_tensor(key)

    def _blend_direct(self, dst, key: str, sources: list[int], weights: list[float], base_weight: float | None) -> None:
        torch = _torch()
        rows = dst.shape[0] if dst.ndim else 1
        per_row = max(1, dst.numel() // max(rows, 1))
        step = max(1, CHUNK_ELEMS // per_row)
        for start in range(0, rows, step):
            sl = slice(start, min(rows, start + step))
            view = dst[sl] if dst.ndim else dst

            def read(source):
                handle = self.indexes[source][key]
                part = handle.get_slice(key)[sl] if dst.ndim else handle.get_tensor(key)
                return part.to(self.device).float()

            if base_weight is None:  # linear
                acc = torch.zeros_like(view, dtype=torch.float32)
                for source, w in zip(sources, weights):
                    acc.add_(read(source), alpha=w)
            else:  # task arithmetic: base + task_scale * sum_i c_i (P_i - base)
                base = read(0)
                acc = base.clone()
                for source, w in zip(sources, weights):
                    acc.add_(read(source) - base, alpha=base_weight * w)
            view.copy_(acc.to(dst.dtype))

    def _blend_experts(self, dst, kind: str, prefix: str, experts: int, sources: list[int], weights: list[float], base_weight: float | None) -> None:
        """Fused expert parameter <- per-expert checkpoint tensors. A group of experts is stacked on the host and moved in one transfer
        (thousands of tiny per-expert transfers would dominate the time); groups keep the fp32 temporaries within CHUNK_ELEMS."""
        torch = _torch()
        names = ("gate_proj", "up_proj") if kind == "gate_up" else ("down_proj",)
        step = max(1, CHUNK_ELEMS // max(1, dst[0].numel()))

        for lo in range(0, experts, step):
            hi = min(experts, lo + step)

            def stacked(source: int, part: str):
                keys = [f"{prefix}.experts.{e}.{part}.weight" for e in range(lo, hi)]
                return torch.stack([self._source_tensor(source, key) for key in keys]).to(self.device).float()

            pieces = []
            for part in names:
                acc = None
                if base_weight is None:
                    for source, w in zip(sources, weights):
                        value = stacked(source, part)
                        acc = value * w if acc is None else acc.add_(value, alpha=w)
                else:
                    base = stacked(0, part)
                    acc = base.clone()
                    for source, w in zip(sources, weights):
                        acc.add_(stacked(source, part) - base, alpha=base_weight * w)
                pieces.append(acc)
            dst[lo:hi].copy_((torch.cat(pieces, 1) if len(pieces) > 1 else pieces[0]).to(dst.dtype))

    def _run(self, sources: list[int], coefficient_for, base_weight: float | None) -> None:
        torch = _torch()
        with torch.no_grad():
            for name, kind, source in self.plan:
                dst = self.state[name]
                if not dst.is_floating_point():
                    continue
                if kind == "direct":
                    self._blend_direct(dst, source, sources, coefficient_for(name), base_weight)
                else:
                    prefix, experts = source
                    self._blend_experts(dst, kind, prefix, experts, sources, coefficient_for(name), base_weight)

    def apply(self, genes: Sequence[float], method: str | None = None) -> None:
        method = method or self.spec.method
        if method not in ("linear", "task_arithmetic"):
            raise ResidentError(f"checkpoint blending supports linear and task_arithmetic, not {method}")
        parents = list(range(1, len(self.indexes)))
        self._run(parents, lambda n: tensor_coefficients(self.spec, list(genes), n, self.n_layers),
                  self.spec.task_scale if method == "task_arithmetic" else None)

    def apply_reference(self, name: str) -> None:
        source = 0 if name == "base" else 1 + [p.name for p in self.spec.parents].index(name)
        self._run([source], lambda n: [1.0], None)


# --------------------------------------------------------------------------------------------------- session
class ResidentSession:
    def __init__(self, spec: Spec, device: str = "cpu", dtype: str = "bfloat16", *,
                 rows: dict[str, Sequence[Any]] | None = None, trust_self_check: bool = False):
        torch = _torch()
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ResidentError('The resident evaluator needs transformers: pip install -e ".[lora]"') from exc
        if dtype not in ("bfloat16", "float16", "float32"):
            raise ResidentError("dtype must be bfloat16, float16 or float32")
        self.spec, self.device, self.dtype, self.rows = spec, torch.device(device), dtype, rows
        source = spec.base_model if spec.mode == "lora" else spec.parents[0].model
        self.tokenizer = AutoTokenizer.from_pretrained(spec.base_model)
        self.model = AutoModelForCausalLM.from_pretrained(source, dtype=getattr(torch, dtype)).to(self.device).eval()
        self.config = self.model.config
        self.model_type = getattr(self.config, "model_type", "")
        self.softcap = language_config(self.config.to_dict()).get("final_logit_softcapping")
        max_pos = language_config(self.config.to_dict()).get("max_position_embeddings") or 4096
        self.max_length = min(int(max_pos), 8192)
        self.blender = (LoraBlender if spec.mode == "lora" else CheckpointBlender)(spec, self.model, self.device)
        self._scorers: dict[tuple[int, int], Scorer] = {}
        self._gen_scorers: dict[tuple[int, int], Any] = {}
        self._token_budget: int | None = None
        self._gen_batch = 16
        if not trust_self_check:
            self._self_check()

    def close(self) -> None:
        if hasattr(self.blender, "close"):
            self.blender.close()

    # -- forward ---------------------------------------------------------------------------------------
    def _scorer(self, window: tuple[int, int]) -> Scorer:
        if window not in self._scorers:
            self._scorers[window] = Scorer(self.spec, self.tokenizer, *window, add_special_tokens="gemma" in self.model_type,
                                           max_length=self.max_length, rows=self.rows)
        return self._scorers[window]

    def _generative(self, window: tuple[int, int]):
        if window not in self._gen_scorers:
            from .generative import GenerativeScorer
            self._gen_scorers[window] = GenerativeScorer(
                self.spec, self.tokenizer, *window, add_special_tokens="gemma" in self.model_type,
                max_length=self.max_length, rows=self.rows)
        return self._gen_scorers[window]

    def _generate_batch(self, gen, batch: list[int]) -> list[str]:
        torch = _torch()
        width = max(len(gen.items[i][2]) for i in batch)
        ids = torch.full((len(batch), width), gen.pad_id(), dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, i in enumerate(batch):  # left padding: generation continues from the last real token
            seq = gen.items[i][2]
            ids[row, width - len(seq):] = torch.tensor(seq)
            mask[row, width - len(seq):] = 1
        limits = gen.tasks[gen.items[batch[0]][0]].generate
        out = self.model.generate(
            input_ids=ids.to(self.device), attention_mask=mask.to(self.device), do_sample=False, temperature=None, top_p=None,
            top_k=None, max_new_tokens=limits.get("max_new_tokens", 256), pad_token_id=gen.pad_id(),
            stop_strings=limits.get("stop") or None, tokenizer=self.tokenizer if limits.get("stop") else None)
        return self.tokenizer.batch_decode(out[:, width:], skip_special_tokens=True)

    def _score_generative(self, window: tuple[int, int]) -> dict[str, float]:
        """Greedy completions of the window's prompts. The batch size halves on out-of-memory and is kept for later candidates."""
        torch = _torch()
        gen = self._generative(window)
        if not gen:
            return {}
        size = min(self._gen_batch, len(gen.items))
        while True:
            texts: dict[int, str] = {}
            try:
                with torch.no_grad():
                    for batch in gen.batches(size):
                        for i, text in zip(batch, self._generate_batch(gen, batch)):
                            texts[i] = text
                return gen.accuracy(texts)
            except RuntimeError as exc:
                if not _is_out_of_memory(exc):
                    raise
                self._release_cache()
                if size == 1:
                    raise ResidentError("out of accelerator memory even generating one sequence at a time; "
                                        "use a smaller model or fewer new tokens") from None
                size = max(1, size // 2)
                self._gen_batch = size

    def _pad(self, scorer: Scorer, batch: list[int]):
        torch = _torch()
        width = max(len(scorer.requests[i][3]) + len(scorer.requests[i][4]) for i in batch)
        ids = torch.zeros(len(batch), width, dtype=torch.long)
        mask = torch.zeros(len(batch), width, dtype=torch.long)
        for row, i in enumerate(batch):
            seq = scorer.requests[i][3] + scorer.requests[i][4]
            ids[row, :len(seq)] = torch.tensor(seq)
            mask[row, :len(seq)] = 1
        return ids.to(self.device), mask.to(self.device)

    def _logprobs(self, scorer: Scorer, batch: list[int]) -> list[float]:
        """Continuation log-likelihoods: only the hidden states at continuation positions go through the LM head."""
        torch = _torch()
        ids, mask = self._pad(scorer, batch)
        hidden = self.model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
        picked, targets, counts = [], [], []
        for row, i in enumerate(batch):
            c0, n = len(scorer.requests[i][3]), len(scorer.requests[i][4])
            picked.append(hidden[row, c0 - 1:c0 - 1 + n])
            targets.extend(scorer.requests[i][4])
            counts.append(n)
        picked = torch.cat(picked)
        target = torch.tensor(targets, device=self.device)
        pieces = []
        for s in range(0, len(picked), 1024):
            logits = self.model.lm_head(picked[s:s + 1024]).float()
            if self.softcap:
                logits = torch.tanh(logits / self.softcap) * self.softcap
            pieces.append(torch.log_softmax(logits, -1).gather(1, target[s:s + 1024, None])[:, 0])
        flat = torch.cat(pieces).cpu().tolist()
        out, at = [], 0
        for n in counts:
            out.append(sum(flat[at:at + n]))
            at += n
        return out

    def _self_check(self) -> None:
        """Does ``lm_head(base_model(x))`` (plus the soft-cap this module knows) equal ``model(x).logits``?

        Both sides run the identical computation on the identical padded batch, so without output post-processing they agree to
        rounding. A model that scales or caps its logits in a way this module does not reproduce differs by far more, and the
        resident path then refuses to score instead of returning plausible but wrong accuracies."""
        torch = _torch()
        scorer = self._scorer((0, min(2, self.spec.evaluation.limit or 2)))
        if not scorer.requests:  # only generative tasks: generate() runs the model's own forward pass
            return
        batch = scorer.order[:3]
        ids, mask = self._pad(scorer, batch)
        with torch.no_grad():
            hidden = self.model.base_model(input_ids=ids, attention_mask=mask).last_hidden_state
            mine = self.model.lm_head(hidden).float()
            if self.softcap:
                mine = torch.tanh(mine / self.softcap) * self.softcap
            reference = self.model(input_ids=ids, attention_mask=mask).logits.float()
        worst = float((mine - reference).abs()[mask.bool()].max())
        if not worst <= 0.05:
            raise ResidentError(f"resident scoring disagrees with the model's own forward pass (largest logit difference {worst:.3f}); "
                                "this architecture post-processes its logits in a way the resident path does not reproduce. "
                                "Use the lm-eval path (`lerp cycle`) for this model.")

    # -- public ----------------------------------------------------------------------------------------
    def _score(self, window: tuple[int, int], max_tokens: int) -> dict[str, float]:
        """Accuracy per task. Batches shrink automatically when the accelerator runs out of memory (large models leave little room
        for activations), and the smaller size is remembered for the rest of the session."""
        torch = _torch()
        scorer = self._scorer(window)
        metrics = self._score_generative(window)
        if not scorer.requests:
            return metrics
        max_tokens = min(max_tokens, self._token_budget or max_tokens)
        while True:
            loglik: dict[int, float] = {}
            try:
                with torch.no_grad():
                    for batch in scorer.batches(max_tokens):
                        for i, value in zip(batch, self._logprobs(scorer, batch)):
                            loglik[i] = value
                return {**scorer.accuracy(loglik), **metrics}
            except RuntimeError as exc:
                if not _is_out_of_memory(exc):
                    raise
                self._release_cache()
                longest = max(len(r[3]) + len(r[4]) for r in scorer.requests)
                if max_tokens <= longest:
                    raise ResidentError("out of accelerator memory even with one sequence per batch; "
                                        "use a smaller model, a shorter context or a bigger device") from None
                max_tokens = max(longest, max_tokens // 2)
                self._token_budget = max_tokens

    def _release_cache(self) -> None:
        torch = _torch()
        for name in ("xpu", "cuda"):
            module = getattr(torch, name, None)
            if module is not None and str(self.device).startswith(name) and hasattr(module, "empty_cache"):
                module.empty_cache()

    def evaluate(self, genes: Sequence[float], window: tuple[int, int], *, method: str | None = None,
                 max_tokens: int = 9000) -> dict[str, float]:
        self.blender.apply(genes, method)
        _sync(self.device)
        self._release_cache()
        return self._score(window, max_tokens)

    def evaluate_reference(self, name: str, window: tuple[int, int], *, max_tokens: int = 9000) -> dict[str, float]:
        self.blender.apply_reference(name)
        _sync(self.device)
        self._release_cache()
        return self._score(window, max_tokens)

    def protocol(self, window: tuple[int, int]) -> dict[str, Any]:
        import platform
        torch = _torch()
        try:
            import transformers
            tf_version = transformers.__version__
        except Exception:  # pragma: no cover
            tf_version = None
        scorer = self._scorer(window)
        return {
            "engine": "lerp-resident", "protocol_version": PROTOCOL_VERSION, "mode": self.spec.mode,
            "items": list(window), "dtype": self.dtype, "device": str(self.device), "model_type": self.model_type,
            "tasks": {t.name: {"metric": t.metric, "definition_sha256": hashlib.sha256(
                (t.definition or json.dumps(dataclasses.asdict(resolve_task(t.name)), sort_keys=True)).encode("utf-8")).hexdigest()}
                for t in self.spec.evaluation.tasks},
            "requests": len(scorer.requests), "tokens": scorer.tokens, "softcap": self.softcap,
            "generation": {"decoding": "greedy", "prompts": len(self._generative(window).items)},
            "versions": {"torch": torch.__version__, "transformers": tf_version, "python": platform.python_version()},
        }
