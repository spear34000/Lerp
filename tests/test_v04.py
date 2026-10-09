"""v0.4 regression, adversarial, and numerical tests for new safety gates."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors.torch import save_file

import lerp.experiment as exp
from lerp.artifacts import ArtifactError, verify_artifact
from lerp.integrity import IntegrityError, freeze_inputs, verify_frozen_inputs
from lerp.leakage import LeakageError, audit_split_overlap
from lerp.lora import LoRAMergeError, build_lora
from lerp.spec import SpecError, parse_spec
from lerp.statistics import StatisticsError, compare_samples


def _toy(tmp_path: Path):
    root = tmp_path / 'base'
    root.mkdir()
    (root / 'config.json').write_text(json.dumps({'model_type': 'llama', 'num_hidden_layers': 1}))
    parents = []
    for n in range(2):
        p = tmp_path / f'parent{n}'
        p.mkdir()
        cfg = dict(peft_type='LORA', r=2, lora_alpha=4,
                   base_model_name_or_path=str(root), target_modules=['q_proj'],
                   task_type='CAUSAL_LM', bias='none', lora_dropout=0.)
        (p / 'adapter_config.json').write_text(json.dumps(cfg))
        name = 'base_model.model.model.layers.0.self_attn.q_proj'
        save_file({name+'.lora_A.weight': torch.full((2, 4), .1 + n),
                   name+'.lora_B.weight': torch.full((4, 2), .2 + n)}, str(p/'adapter_model.safetensors'))
        parents.append(p)
    raw = {'name': 'toy-lora', 'base_model': str(root), 'mode': 'lora', 'method': 'linear',
           'parents': [{'name': 'one', 'model': str(parents[0])},
                       {'name': 'two', 'model': str(parents[1])}],
           'genes': 2, 'population': 2, 'out_dtype': 'float32',
           'evaluation': {'tasks': {'a': {'metric': 'acc,none'}}}}
    config_path = tmp_path / 'experiment.yaml'
    config_path.write_text(yaml.safe_dump(raw))
    run = tmp_path / 'run'
    exp.init_run(config_path, run)
    return run, parse_spec(raw), root, parents


def _jsonlines(path: Path, rows: list[dict]):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')


def test_freeze_checks_content_sha_and_blocks_subsequent_build(tmp_path):
    run, spec, base, parents = _toy(tmp_path)
    record = freeze_inputs(run, spec)
    assert record['sources'][0]['verified'] is False  # config alone is not model weights
    assert verify_frozen_inputs(run, spec)['passed']
    with pytest.raises(IntegrityError, match='already frozen'):
        freeze_inputs(run, spec)
    # Same-length config rewrite evades size-based caches, but not SHA-256.
    cfg = parents[0] / 'adapter_config.json'
    content = cfg.read_text()
    cfg.write_text(content.replace('CAUSAL_LM', 'CAUSAL_LN'))
    with pytest.raises(IntegrityError, match='changed since freeze'):
        verify_frozen_inputs(run, spec)
    with pytest.raises(IntegrityError):
        exp.build_candidate(run, 0, 0, engine='lora')


def test_strict_freeze_requires_real_base_weights(tmp_path):
    run, spec, _, _ = _toy(tmp_path)
    with pytest.raises(IntegrityError, match='requires local weights'):
        freeze_inputs(run, spec, strict=True)
    assert not (run/'input_fingerprints.json').exists()


def test_strict_freeze_succeeds_after_base_weight_added(tmp_path):
    run, spec, base, _ = _toy(tmp_path)
    save_file({'dummy': torch.zeros((2, 2))}, str(base/'model.safetensors'))
    assert all(entry['verified'] for entry in freeze_inputs(run, spec, strict=True)['sources'])
    assert verify_frozen_inputs(run, spec)['fully_verified']


def test_freeze_refuses_after_score(tmp_path):
    run, spec, _, _ = _toy(tmp_path)
    exp.simulate_generation(run, 0)
    with pytest.raises(IntegrityError, match='BEFORE recording scores'):
        freeze_inputs(run, spec)


def test_symlink_weight_rejected_by_freeze(tmp_path):
    run, spec, _, parents = _toy(tmp_path)
    data = parents[0]/'adapter_model.safetensors'
    real = tmp_path/'real.safetensors'
    data.replace(real)
    try:
        data.symlink_to(real)
    except OSError as exc:  # Windows without Developer Mode / admin rights
        pytest.skip(f"cannot create symlinks here: {exc}")
    with pytest.raises(IntegrityError, match='symlink'):
        freeze_inputs(run, spec)


def test_artifact_pinned_and_tampering_blocks_evaluation(tmp_path):
    run, _, _, _ = _toy(tmp_path)
    output = exp.build_candidate(run, 0, 0, engine='lora')
    assert verify_artifact(output)['pinned']
    # Modifying even non-weight metadata after build is detectable.
    with (output/'adapter_config.json').open('a') as f:
        f.write(' ')
    with pytest.raises(ArtifactError, match='modified since build'):
        exp.evaluate_candidate(run, 0, 0)


def test_rank_budget_enforced_before_creating_output(tmp_path):
    _, spec, _, _ = _toy(tmp_path)
    raw = {'name':spec.name, 'base_model':spec.base_model, 'mode':'lora',
           'parents':[{'name':p.name,'model':p.model} for p in spec.parents],
           'max_output_rank':3, 'genes':2}
    altered = parse_spec(raw)
    path = tmp_path/'out'
    with pytest.raises(LoRAMergeError, match='exceeds max_output_rank'):
        build_lora(altered, [.5, .5], path)
    assert not path.exists()
    with pytest.raises(SpecError, match='max_output_rank'):
        parse_spec({**raw, 'max_output_rank':0})


def test_lora_output_size_budget_rejects_large_shapes_from_headers(tmp_path):
    _, spec, _, parents = _toy(tmp_path)
    for parent in parents:
        filename = parent/'adapter_model.safetensors'
        prefix = 'base_model.model.model.layers.0.self_attn.q_proj'
        save_file({prefix+'.lora_A.weight':torch.ones((2, 50_000)),
                   prefix+'.lora_B.weight':torch.ones((50_000,2))}, str(filename))
    raw = {'name':'budget','base_model':spec.base_model,'mode':'lora',
           'parents':[{'name':p.name,'model':p.model} for p in spec.parents],
           'max_lora_output_mib':1, 'genes':2, 'out_dtype':'float32'}
    with pytest.raises(LoRAMergeError,match='max_lora_output_mib'):
        build_lora(parse_spec(raw),[.5,.5],tmp_path/'large')
    with pytest.raises(SpecError,match='max_lora_output_mib'):
        parse_spec({**raw, 'max_lora_output_mib':0})


def test_bad_parent_header_shape_rejected_before_factor_allocation(tmp_path):
    _,spec,_,parents = _toy(tmp_path)
    filename=parents[1]/'adapter_model.safetensors'
    prefix='base_model.model.model.layers.0.self_attn.q_proj'
    save_file({prefix+'.lora_A.weight':torch.ones((2, 100000)),
               prefix+'.lora_B.weight':torch.ones((4,2))},str(filename))
    with pytest.raises(LoRAMergeError,match='Incompatible LoRA factor header'):
        build_lora(spec,[.5,.5],tmp_path/'badshape')


def test_unknown_adapter_behavior_flag_rejected(tmp_path):
    _, spec, _, parents = _toy(tmp_path)
    cfg = parents[0]/'adapter_config.json'
    d = json.loads(cfg.read_text()); d['trainable_token_indices'] = [1, 2]
    cfg.write_text(json.dumps(d))
    with pytest.raises(LoRAMergeError, match='trainable_token_indices'):
        build_lora(spec, [.5,.5], tmp_path/'out')


def test_pairstats_perfect_improvement_ci_and_reproducibility(tmp_path):
    a, b = tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    _jsonlines(a, [{'id':i,'score':1} for i in range(50)])
    _jsonlines(b, [{'id':i,'score':0} for i in reversed(range(50))])
    result = compare_samples(a, b, replicates=300, seed=10)
    assert result['samples'] == 50
    assert result['paired_mean_difference'] == pytest.approx(1)
    assert result['paired_bootstrap_ci']['lower'] == 1
    assert result['paired_bootstrap_ci']['upper'] == 1
    assert result['wins'] == 50 and result['losses'] == 0
    assert compare_samples(a,b,replicates=300,seed=10) == result


def test_pairstats_no_difference_signflip_neutral(tmp_path):
    a,b=tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    data=[{'id':i,'score':i%2} for i in range(40)]
    _jsonlines(a,data); _jsonlines(b,list(reversed(data)))
    r=compare_samples(a,b,replicates=300)
    assert r['paired_mean_difference'] == 0
    assert r['paired_bootstrap_ci']['lower'] == r['paired_bootstrap_ci']['upper'] == 0
    assert r['signflip_exploratory_p_two_sided'] == 1


def test_pairstats_rejects_misalignment_and_duplicates(tmp_path):
    a,b=tmp_path/'a.jsonl', tmp_path/'b.jsonl'
    _jsonlines(a,[{'id':'1','score':1}, {'id':'2','score':0}])
    _jsonlines(b,[{'id':'1','score':1}, {'id':'3','score':0}])
    with pytest.raises(StatisticsError, match='Unpaired sample IDs'):
        compare_samples(a,b,replicates=200)
    _jsonlines(b,[{'id':'1','score':1}, {'id':'1','score':0}])
    with pytest.raises(StatisticsError,match='duplicate sample id'):
        compare_samples(a,b,replicates=200)
    _jsonlines(b,[{'id':'1','score':float('nan')},{'id':'2','score':0}])
    with pytest.raises(StatisticsError,match='finite'):
        compare_samples(a,b,replicates=200)


def test_leakage_normalized_unicode_and_ids(tmp_path):
    dev,held=tmp_path/'dev.jsonl',tmp_path/'held.jsonl'
    _jsonlines(dev,[{'id':1,'text':'Hello, WORLD!'}, {'id':2,'text':'ABC'}])
    _jsonlines(held,[{'id':3,'text':'hello world'}, {'id':4,'text':'A B C'}])
    report=audit_split_overlap(dev,held)
    assert report['status']=='FAIL_OVERLAP'
    assert report['normalized_duplicate_texts']==1
    assert report['repeated_ids']==0
    _jsonlines(held,[{'id':3,'text':'new content'}, {'id':4,'text':'unrelated'}])
    assert audit_split_overlap(dev,held)['status']=='PASS_ONLY_EXACT_NORMALIZED_CHECK'


def test_leakage_refuses_repeated_ids_and_malformed(tmp_path):
    dev,held=tmp_path/'dev.jsonl',tmp_path/'held.jsonl'
    _jsonlines(dev,[{'id':'a','text':'test text'}])
    _jsonlines(held,[{'id':'a','text':'completely different'}])
    report=audit_split_overlap(dev,held)
    assert report['repeated_ids']==1
    assert report['status']=='FAIL_OVERLAP'
    _jsonlines(dev,[{'id':'a','text':'one'},{'id':'a','text':'two'}])
    with pytest.raises(LeakageError,match='duplicate ID'):
        audit_split_overlap(dev,held)


def test_v04_commands_parse_and_config_preserved():
    from lerp.cli import create_parser
    from lerp.spec import spec_to_dict
    p=create_parser()
    assert p.parse_args(['freeze','-r','somewhere','--strict']).strict
    assert p.parse_args(['verify-inputs','-r','somewhere']).command=='verify-inputs'
    assert p.parse_args(['compare-samples','--candidate','a','--baseline','b','--out','c']).seed==42
    assert p.parse_args(['audit-splits','--development','a','--holdout','b']).text_field=='text'
    spec=parse_spec({'name':'test','base_model':'q','parents':[{'name':'a','model':'a'}, {'name':'b','model':'b'}]})
    assert parse_spec(spec_to_dict(spec)).max_output_rank == 256


def test_cli_run_lock_blocks_simultaneous_writers_and_releases(tmp_path):
    from lerp.locking import RunBusyError, locked_run
    run = tmp_path/'run'; run.mkdir()
    with locked_run(run):
        with pytest.raises(RunBusyError, match='Another Lerp process'):
            with locked_run(run):
                pass
    with locked_run(run):
        pass


def test_device_and_eval_command_recorded_and_mismatch_not_compared(tmp_path,monkeypatch):
    run, _, _, _ = _toy(tmp_path)
    exp.build_candidate(run, 0, 0, engine='lora')
    monkeypatch.setattr(exp,'_require_command',lambda *args: None)
    def fake(command,log):
        folder=Path(command[command.index('--output_path')+1]); folder.mkdir(parents=True,exist_ok=True)
        (folder/'result.json').write_text(json.dumps({'results': {'a': {'acc,none': .75}}}))
    monkeypatch.setattr(exp,'_execute_logged',fake)
    score=exp.evaluate_candidate(run,0,0,device='cpu')
    assert score['evaluation_settings']['device']=='cpu'
    protocol=run/'generations'/'gen-000'/'cand-000'/'evaluation'/'modelbreeder_protocol.json'
    doc=json.loads(protocol.read_text())
    assert doc['device']=='cpu' and doc['command'][:2]==['lm-eval','run']
    # An external evaluator might accept an invalid device string in this mock;
    # protocol comparison must still reject incomparable measured conditions.
    baseline=exp.evaluate_baseline(run,'base',device='cuda:0')
    assert baseline['evaluation_settings']['device']=='cuda:0'
    assert exp.comparisons(run)==[]
    exp.evaluate_baseline(run,'base',device='cpu',overwrite=True)
    assert exp.comparisons(run)[0]['id']=='g000-c000'


def test_recipe_written_byte_exact_so_hash_survives_windows(tmp_path):
    """Text-mode writes turn \n into \r\n on Windows and break recipe_sha256."""
    import hashlib
    run, _, _, _ = _toy(tmp_path)
    cand = run / 'generations' / 'gen-000' / 'cand-000'
    raw = (cand / 'merge.yaml').read_bytes()
    assert b'\r' not in raw
    recorded = json.loads((cand / 'genome.json').read_text(encoding='utf-8'))['recipe_sha256']
    assert hashlib.sha256(raw).hexdigest() == recorded


def test_msys_style_paths_are_mapped_to_drive_letters_on_windows(monkeypatch):
    import types
    import lerp.compat as compat
    monkeypatch.setattr(compat, 'os', types.SimpleNamespace(name='nt'))
    assert compat.normalize_platform_path('/c/Users/me/model') == 'C:/Users/me/model'
    assert compat.normalize_platform_path('/d') == 'D:/'
    assert compat.normalize_platform_path('C:/already/fine') == 'C:/already/fine'
    assert compat.normalize_platform_path('./relative') == './relative'
    monkeypatch.setattr(compat, 'os', types.SimpleNamespace(name='posix'))
    assert compat.normalize_platform_path('/c/Users/me/model') == '/c/Users/me/model'


def test_target_module_order_does_not_matter_but_set_difference_does(tmp_path):
    """PEFT serializes target_modules from a set; adapters from different processes list them differently."""
    from lerp.lora import verify_adapters
    _, spec, _, parents = _toy(tmp_path)
    for path, modules in ((parents[0], ['q_proj', 'v_proj', 'o_proj']), (parents[1], ['o_proj', 'q_proj', 'v_proj'])):
        cfg = json.loads((path / 'adapter_config.json').read_text())
        cfg['target_modules'] = modules
        (path / 'adapter_config.json').write_text(json.dumps(cfg))
    assert verify_adapters(spec)['parents'] == 2
    cfg = json.loads((parents[1] / 'adapter_config.json').read_text())
    cfg['target_modules'] = ['q_proj', 'v_proj']
    (parents[1] / 'adapter_config.json').write_text(json.dumps(cfg))
    with pytest.raises(LoRAMergeError, match='target_modules differs'):
        verify_adapters(spec)
