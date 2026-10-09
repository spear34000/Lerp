"""Offline PEFT integration gate: random tiny Llama, two adapters, delta & logits.

This test does NOT download pretrained weights, demonstrate a trained model, or
measure real-world intelligence. It validates the PEFT load path and the
merged adapter's runtime equivalence to explicitly applied dense updates.

  pip install -e '.[lora]'
  python examples/offline_peft_equivalence.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path


def run(workdir: Path) -> dict:
    import torch
    from safetensors import safe_open
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import LlamaConfig, LlamaForCausalLM
    from lerp.lora import build_lora
    from lerp.spec import parse_spec

    torch.manual_seed(1123)
    base = workdir / 'base'
    base.mkdir(parents=True, exist_ok=True)
    cfg = LlamaConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=128)
    LlamaForCausalLM(cfg).save_pretrained(base, safe_serialization=True)

    adapters = []
    for index, (rank, alpha) in enumerate(((2, 4), (3, 9))):
        parent = workdir / f'parent-{index}'
        model = LlamaForCausalLM.from_pretrained(base)
        lora = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=0.0, bias='none',
                          target_modules=['q_proj', 'up_proj'], task_type='CAUSAL_LM')
        peft_model = get_peft_model(model, lora)
        with torch.no_grad():
            for name, parameter in peft_model.named_parameters():
                if '.lora_A.' in name or '.lora_B.' in name:
                    parameter.normal_(mean=0, std=0.08)
        peft_model.save_pretrained(parent, safe_serialization=True)
        # An actual PEFT export is required here; never invent the tensor names.
        config_path = parent / 'adapter_config.json'
        config = json.loads(config_path.read_text(encoding='utf-8'))
        assert config['peft_type'] == 'LORA', config
        config['base_model_name_or_path'] = str(base)
        config_path.write_text(json.dumps(config), encoding='utf-8')
        adapters.append(parent)
        del peft_model, model

    spec = parse_spec({
        'name': 'offline-peft-proof', 'mode': 'lora', 'base_model': str(base),
        'parents': [{'name': f'parent-{i}', 'model': str(p)} for i, p in enumerate(adapters)],
        'method': 'linear', 'genes': 2, 'out_dtype': 'float32',
        'gene_groups': ['attention', 'mlp', 'other'],
        'population': 2, 'evaluation': {'tasks': {'dummy': {'metric': 'acc,none'}}},
    })
    # Two parents: independent weights for attention, MLP and other.
    genes = [.8, .25,  .15, .65,  .5, .5]
    child_dir = workdir / 'child'
    details = build_lora(spec, genes, child_dir)

    dense = LlamaForCausalLM.from_pretrained(base).eval()
    child = PeftModel.from_pretrained(LlamaForCausalLM.from_pretrained(base),
                                      child_dir, is_trainable=False).eval()
    factors = child_dir / 'adapter_model.safetensors'
    with safe_open(str(factors), framework='pt', device='cpu') as f, torch.no_grad():
        for name in sorted(f.keys()):
            if not name.endswith('.lora_A.weight'):
                continue
            paired = name[:-len('.lora_A.weight')] + '.lora_B.weight'
            # Standard PEFT checkpoint names add 'base_model.model.' to the
            # underlying HF module names. Remove only that explicit prefix.
            assert name.startswith('base_model.model.'), name
            module_name = name[len('base_model.model.'):-len('.lora_A.weight')]
            layer = dense.get_submodule(module_name)
            delta = f.get_tensor(paired) @ f.get_tensor(name)
            assert layer.weight.shape == delta.shape, (module_name, layer.weight.shape, delta.shape)
            layer.weight.add_(delta)

    sample = torch.tensor([[1, 17, 25, 7, 4]], dtype=torch.long)
    with torch.inference_mode():
        dense_logits = dense(sample).logits
        child_logits = child(input_ids=sample).logits
        tokens = child.generate(input_ids=sample, max_new_tokens=3,
                                do_sample=False, pad_token_id=0)
    max_delta = (dense_logits-child_logits).abs().max().item()
    torch.testing.assert_close(dense_logits, child_logits, atol=1e-5, rtol=1e-4)
    if not torch.isfinite(child_logits).all().item():
        raise AssertionError('Non-finite PEFT logits')
    return {'status': 'REAL_PEFT_TINY_OFFLINE_EQUIVALENCE_PASS',
            'note': 'Random tiny network, not a trained LLM or benchmark',
            'max_abs_logits_error': max_delta,
            'generated_token_count': int(tokens.shape[1] - sample.shape[1]),
            'parents': 2, 'output_rank': details['output_rank'],
            'modules': details['modules']}


def main() -> None:
    parser = argparse.ArgumentParser(description='Run offline PEFT exact-merge integration test')
    parser.add_argument('--workdir', type=Path, default=None)
    args = parser.parse_args()
    if args.workdir is None:
        with tempfile.TemporaryDirectory(prefix='lerp-peft-') as tmp:
            print(json.dumps(run(Path(tmp)), indent=2))
    else:
        args.workdir.mkdir(parents=True, exist_ok=True)
        print(json.dumps(run(args.workdir), indent=2))


if __name__ == '__main__':
    main()
