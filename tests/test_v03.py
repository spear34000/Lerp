"""v0.3 adversarial and numerical integration tests (not claims of LLM quality)."""
import json
from pathlib import Path

import pytest
import torch
import yaml
from safetensors import safe_open
from safetensors.torch import save_file

import lerp.experiment as exp
from lerp.compat import check_compatibility, resolve_spec_paths
from lerp.genetics import crowding_distance, rank_entries
from lerp.lora import LoRAMergeError, build_lora
from lerp.spec import SpecError, parse_spec
from lerp.weighting import tensor_coefficients, tensor_group


def _create_adapters(tmp_path, *, population=4, stage=False, method='auto'):
    base = tmp_path / 'base'
    base.mkdir(exist_ok=True)
    (base / 'config.json').write_text(json.dumps({
        'model_type': 'llama', 'hidden_size': 4, 'num_hidden_layers': 2,
        'num_attention_heads': 1, 'vocab_size': 100,
    }))
    (base / 'tokenizer.json').write_text('{}')
    roots = []
    for i, r in enumerate((1, 2, 2)):
        root = tmp_path / f'adapter{i}'
        root.mkdir(exist_ok=True)
        config = {
            'peft_type': 'LORA', 'base_model_name_or_path': str(base),
            'r': r, 'lora_alpha': [2, 4, 8][i],
            'target_modules': ['q_proj', 'up_proj'],
            'bias': 'none', 'task_type': 'CAUSAL_LM', 'inference_mode': True,
            'use_rslora': i == 2, 'fan_in_fan_out': False,
            'lora_dropout': 0.0,
        }
        (root / 'adapter_config.json').write_text(json.dumps(config))
        state = {}
        for layer in range(2):
            for group, module in (('self_attn', 'q_proj'), ('mlp', 'up_proj')):
                prefix = f'base_model.model.model.layers.{layer}.{group}.{module}'
                # Deterministic non-collinear factors; different parents/ranks.
                a = torch.arange(r * 4, dtype=torch.float32).reshape(r, 4) / 10.0 + (i + 1) / 20.0 + layer / 10
                b = torch.arange(3 * r, dtype=torch.float32).reshape(3, r) / 8.0 + (i + 1) / 11.0
                state[prefix + '.lora_A.weight'] = a.contiguous()
                state[prefix + '.lora_B.weight'] = b.contiguous()
        save_file(state, str(root / 'adapter_model.safetensors'))
        roots.append(root)
    raw = {
        'name': 'v03-real-lora', 'base_model': str(base),
        'parents': [{'name': f'p{i}', 'model': str(root)} for i, root in enumerate(roots)],
        'population': population, 'mode': 'lora', 'method': method,
        'search_methods': ['linear', 'task_arithmetic'] if method == 'auto' else None,
        'genes': 2, 'seed': 2317, 'task_scale': .7, 'selection': 'pareto',
        'out_dtype': 'float32', 'gene_groups': ['attention', 'mlp', 'other'],
        'evaluation': {
            'tasks': {'coding': {'metric': 'acc,none'}, 'reasoning': {'metric': 'acc,none'}},
            'num_fewshot': 0, 'limit': 20 if stage else None,
            'screening_limit': 3 if stage else None,
            'promote_top': 2 if stage else None,
        },
    }
    if method != 'auto':
        raw.pop('search_methods')
    path = tmp_path / 'experiment.yaml'
    path.write_text(yaml.safe_dump(raw))
    return path, raw, roots


def _tensor(path, key):
    with safe_open(str(path), framework='pt', device='cpu') as reader:
        return reader.get_tensor(key)


def test_lora_3_parent_numerical_equivalence_per_module(tmp_path):
    path, raw, roots = _create_adapters(tmp_path)
    spec = resolve_spec_paths(parse_spec(raw), tmp_path)
    report = check_compatibility(spec)
    assert report.ok, report.errors
    # Attention and MLP have distinct simplex coefficients at every layer.
    genome = [.7,.7, .2,.2, .1,.1,  .1,.1, .3,.3, .6,.6,  .2,.2, .4,.4, .4,.4]
    output = tmp_path / 'result'
    details = build_lora(spec, genome, output, method='linear')
    assert details['output_rank'] == 5
    config = json.loads((output / 'adapter_config.json').read_text())
    assert config['r'] == config['lora_alpha'] == 5
    assert config['inference_mode'] and not config.get('use_rslora')
    for layer in range(2):
        for group, module in (('self_attn', 'q_proj'), ('mlp', 'up_proj')):
            prefix = f'base_model.model.model.layers.{layer}.{group}.{module}'
            akey, bkey = prefix+'.lora_A.weight', prefix+'.lora_B.weight'
            actual = _tensor(output/'adapter_model.safetensors', bkey) @ _tensor(output/'adapter_model.safetensors', akey)
            weights = tensor_coefficients(spec, genome, akey, 2)
            expected = torch.zeros((3, 4))
            for i, root in enumerate(roots):
                cfg = json.loads((root/'adapter_config.json').read_text())
                scale = cfg['lora_alpha'] / (cfg['r'] ** .5 if cfg['use_rslora'] else cfg['r'])
                expected += weights[i] * scale * (_tensor(root/'adapter_model.safetensors', bkey) @ _tensor(root/'adapter_model.safetensors', akey))
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    assert tensor_group('model.layers.0.self_attn.q_proj.lora_A.weight') == 'attention'
    assert tensor_group('model.layers.0.mlp.up_proj.lora_B.weight') == 'mlp'
    assert (output/'modelbreeder_provenance.json').exists()


def test_lora_task_arithmetic_scales_deltas(tmp_path):
    _, raw, _ = _create_adapters(tmp_path)
    spec = parse_spec(raw)
    genome = [.5]*spec.genome_size
    linear, arithmetic = tmp_path/'linear', tmp_path/'arithmetic'
    build_lora(spec, genome, linear, method='linear')
    build_lora(spec, genome, arithmetic, method='task_arithmetic')
    key='base_model.model.model.layers.0.mlp.up_proj'
    delta_linear = _tensor(linear/'adapter_model.safetensors', key+'.lora_B.weight') @ _tensor(linear/'adapter_model.safetensors', key+'.lora_A.weight')
    delta_arith = _tensor(arithmetic/'adapter_model.safetensors', key+'.lora_B.weight') @ _tensor(arithmetic/'adapter_model.safetensors', key+'.lora_A.weight')
    torch.testing.assert_close(delta_arith, delta_linear*spec.task_scale, atol=1e-6, rtol=1e-6)


def test_lora_real_build_and_lora_model_args(tmp_path, monkeypatch):
    path, raw, roots = _create_adapters(tmp_path, method='linear')
    run = tmp_path/'run'
    exp.init_run(path, run)
    output = exp.build_candidate(run, 0, 0, engine='lora')
    assert exp.statuses(run)[0]['built'] is True
    assert 'adapter_model.safetensors' in [p.name for p in output.iterdir()]
    monkeypatch.setattr(exp, '_require_command', lambda *args: None)
    seen=[]
    def fake(cmd, log):
        seen.append(cmd)
        target=Path(cmd[cmd.index('--output_path') + 1]); target.mkdir(parents=True, exist_ok=True)
        (target/'result.json').write_text(json.dumps({'results': {'coding': {'acc,none': .7}, 'reasoning': {'acc,none': .8}}}))
    monkeypatch.setattr(exp, '_execute_logged', fake)
    exp.evaluate_candidate(run, 0, 0)
    args=seen[-1][seen[-1].index('--model_args')+1]
    assert args == f"pretrained={raw['base_model']},peft={output.resolve()}"
    exp.evaluate_baseline(run, 'p1')
    assert seen[-1][seen[-1].index('--model_args')+1] == f"pretrained={raw['base_model']},peft={roots[1]}"
    exp.evaluate_baseline(run, 'base')
    assert seen[-1][seen[-1].index('--model_args')+1] == f"pretrained={raw['base_model']}"


def test_lora_rejects_nonfinite_factors(tmp_path):
    _, raw, roots = _create_adapters(tmp_path)
    spec = parse_spec(raw)
    filename = roots[0]/'adapter_model.safetensors'
    with safe_open(str(filename), framework='pt', device='cpu') as reader:
        state={k:reader.get_tensor(k) for k in reader.keys()}
    key=next(k for k in state if k.endswith('.lora_A.weight'))
    state[key][0,0] = float('nan')
    save_file(state, str(filename))
    with pytest.raises(LoRAMergeError, match='Non-finite'):
        build_lora(spec, [.3]*spec.genome_size, tmp_path/'invalid')


def test_lora_refuses_wrong_base(tmp_path):
    _, raw, roots = _create_adapters(tmp_path)
    cfgpath=roots[1]/'adapter_config.json'
    cfg=json.loads(cfgpath.read_text()); cfg['base_model_name_or_path']='other/different-base'
    cfgpath.write_text(json.dumps(cfg))
    assert not check_compatibility(parse_spec(raw)).ok


def test_lora_refuses_unsafe_extensions(tmp_path):
    _, raw, roots = _create_adapters(tmp_path)
    cfgpath=roots[2]/'adapter_config.json'
    cfg=json.loads(cfgpath.read_text()); cfg['alora_invocation_tokens']=[1,2,3]
    cfgpath.write_text(json.dumps(cfg))
    report=check_compatibility(parse_spec(raw))
    assert not report.ok
    assert 'alora' in str(report.errors)


def test_lora_rejects_differing_tensor_keys(tmp_path):
    _, raw, roots = _create_adapters(tmp_path)
    filename=roots[1]/'adapter_model.safetensors'
    with safe_open(str(filename), framework='pt', device='cpu') as reader:
        state={k:reader.get_tensor(k) for k in reader.keys()}
    state.pop(next(k for k in state if k.endswith('.lora_A.weight')))
    save_file(state, str(filename))
    with pytest.raises(LoRAMergeError, match='keys differ'):
        build_lora(parse_spec(raw), [.5]*18, tmp_path/'bad')


def test_staged_lora_cycle_only_promotes_top(tmp_path, monkeypatch):
    path, raw, _ = _create_adapters(tmp_path, stage=True)
    run = tmp_path/'run'; exp.init_run(path, run)
    with pytest.raises(exp.BreederError, match='screened'):
        exp.promotion_list(run, 0)
    monkeypatch.setattr(exp, '_require_command', lambda *args: None)
    seen=[]
    def fake(cmd, log):
        seen.append(cmd)
        target=Path(cmd[cmd.index('--output_path') + 1]); target.mkdir(parents=True, exist_ok=True)
        idx=int(next(x.split('-')[-1] for x in target.parts if x.startswith('cand-')))
        # Same ranking in both metrics, top two are #3 and #2.
        val=.5+idx*.1
        (target/'result.json').write_text(json.dumps({'results': {
            'coding': {'acc,none': val}, 'reasoning': {'acc,none': val-.04}}}))
    monkeypatch.setattr(exp, '_execute_logged', fake)
    exp.cycle(run, 2, engine='lora')
    stages=[Path(c[c.index('--output_path')+1]).name for c in seen]
    assert stages.count('.screening.partial') == 8
    assert stages.count('.evaluation.partial') == 4
    assert len([c for c in exp.leaderboard(run) if c['score']['source']=='lm_eval']) == 4
    for gen in (0,1):
        assert {c['id'] for c in exp.promotion_list(run, gen)} == {f'g{gen:03d}-c002', f'g{gen:03d}-c003'}
        assert all(exp.load_candidate(run, gen, i).get('screen_score') is not None for i in range(4))
        assert all(exp.load_candidate(run, gen, i).get('score') is None for i in (0,1))
        assert all(exp.load_candidate(run, gen, i).get('score') is not None for i in (2,3))
        for i in range(4):
            screen=exp.load_candidate(run, gen, i)['screen_score']
            assert screen['status']=='screen_only_not_full_evaluation'
            assert screen['evaluation_settings']['limit'] == 3
    assert exp.statuses(run)[0]['screened'] is True
    assert exp.load_run(run)[1]['generation'] == 1
    assert exp.advance(run) == 2


def test_staged_not_enough_full_scores(tmp_path, monkeypatch):
    path, _, _ = _create_adapters(tmp_path, stage=True)
    run=tmp_path/'run'; exp.init_run(path, run)
    for i in range(4):
        folder=exp.candidate_dir(run,0,i)
        (folder/'screen_score.json').write_text(json.dumps({'source':'lm_eval', 'fitness':float(i)/5,
                         'metrics': {'coding':i/5,'reasoning':i/5}}))
    exp.record_score(run, 0, 3, {'coding':.7,'reasoning':.7},source='manual')
    with pytest.raises(exp.BreederError, match='At least two'):
        exp.advance(run)


def test_gene_tampering_rejected(tmp_path):
    path, raw, _ = _create_adapters(tmp_path, method='linear')
    run=tmp_path/'run'; exp.init_run(path, run)
    genome=exp.candidate_dir(run,0,0)/'genome.json'
    info=json.loads(genome.read_text()); info['genes']=[0.9]*len(info['genes'])
    genome.write_text(json.dumps(info))
    with pytest.raises(exp.BreederError, match='modified'):
        exp.build_candidate(run,0,0,engine='lora')


def test_pareto_ties_do_not_spuriously_mark_extremes():
    scores=[{'id': f'{i}', 'score': {'fitness': .5, 'metrics': {'constant': .5, 'variable':.5}}} for i in range(5)]
    distances=crowding_distance(scores, ['constant'])
    assert all(value==0 for value in distances.values())


def test_auto_export_refuses_unverified_manual(tmp_path):
    path,_,_=_create_adapters(tmp_path)
    run=tmp_path/'run'; exp.init_run(path,run)
    exp.record_score(run,0,0,{'coding':.9,'reasoning':.9},source='manual')
    with pytest.raises(exp.BreederError,match='unverified manual'):
        exp.export_recipe(run,tmp_path/'winner')


def test_invalid_gene_group_type_rejected():
    raw={'name':'g', 'base_model':'x', 'parents':[
        {'name':'a','model':'p/a'}, {'name':'b','model':'p/b'}],
         'gene_groups':'attention'}
    with pytest.raises(SpecError,match='gene_groups'):
        parse_spec(raw)


def test_heldout_separate_protocol_and_parent_comparison(tmp_path, monkeypatch):
    path, raw, roots = _create_adapters(tmp_path, method='linear')
    run=tmp_path/'run'; exp.init_run(path,run)
    exp.build_candidate(run,0,0,engine='lora')
    protocol=tmp_path/'holdout.yaml'
    protocol.write_text(yaml.safe_dump({'evaluation': {'tasks': {'piqa': {'metric': 'acc_norm,none'}},
                                           'limit': 15, 'device':'cpu'}}))
    monkeypatch.setattr(exp,'_require_command',lambda *args: None)
    called=[]
    def fake(cmd, log):
        called.append(cmd)
        target=Path(cmd[cmd.index('--output_path')+1]); target.mkdir(parents=True,exist_ok=True)
        tasks=cmd[cmd.index('--tasks')+1]
        scores = {'coding': {'acc,none': .65}, 'reasoning': {'acc,none': .64}} if 'coding' in tasks else {'piqa': {'acc_norm,none': .72}}
        (target/'result.json').write_text(json.dumps({'results': scores}))
    monkeypatch.setattr(exp, '_execute_logged', fake)
    exp.evaluate_candidate(run,0,0)
    before=exp.leaderboard(run)[0]['score']
    summary=exp.validate_holdout(run, protocol, baseline='all')
    assert summary['delta_vs_best_baseline']==0
    assert len(summary['results']) == 5
    assert summary['candidate_holdout_fitness'] == .72
    assert exp.leaderboard(run)[0]['score'] == before
    for result in summary['results'].values():
        assert result['stage'] == 'holdout_validation'
        assert result['status'] == 'smoke_test' # holdout limit=15, NOT official claim
    previous_count=len(called)
    same=exp.validate_holdout(run, protocol, baseline='all')
    assert len(called) == previous_count # idempotent, cached under protocol hash
    assert same['protocol_sha256'] == summary['protocol_sha256']


def test_heldout_rejects_benchmark_leakage(tmp_path, monkeypatch):
    path, raw, _ = _create_adapters(tmp_path, method='linear')
    run=tmp_path/'run'; exp.init_run(path,run)
    protocol=tmp_path/'holdout.yaml'
    protocol.write_text(yaml.safe_dump({'evaluation': {'tasks': {'coding': {'metric':'acc,none'}}}}))
    with pytest.raises(exp.BreederError,match='overlap'):
        exp.validate_holdout(run,protocol)


def test_atomic_generation_state_recovery(tmp_path,monkeypatch):
    from lerp.genetics import rank_entries
    from lerp.experiment import generation_dir
    path,_,_ = _create_adapters(tmp_path,method='linear')
    run=tmp_path/'run'; exp.init_run(path,run)
    exp.simulate_generation(run,0)
    original=exp._write_json
    def crash(path,doc):
        if path.name == 'state.json' and doc.get('generation') == 1:
            raise OSError('simulated outage after atomic generation directory rename')
        original(path,doc)
    monkeypatch.setattr(exp,'_write_json',crash)
    with pytest.raises(OSError,match='simulated outage'):
        exp.advance(run,allow_simulated=True)
    assert generation_dir(run,1).is_dir()
    assert exp.load_run(run)[1]['generation']==0
    monkeypatch.setattr(exp,'_write_json',original)
    assert exp.advance(run,allow_simulated=True)==1
    assert exp.load_run(run)[1]['generation']==1


def test_generation_partial_requires_explicit_clean(tmp_path):
    path,_,_=_create_adapters(tmp_path,method='linear')
    run=tmp_path/'run'; exp.init_run(path,run)
    exp.simulate_generation(run,0)
    pending=run/'generations'/'.gen-001.partial'
    pending.mkdir()
    (pending/'bad.txt').write_text('stale')
    with pytest.raises(exp.BreederError,match='Interrupted generation'):
        exp.advance(run,allow_simulated=True)
    assert exp.advance(run,allow_simulated=True,retry_partial=True)==1
    assert not pending.exists()


def test_lora_cli_engine_available():
    from lerp.cli import create_parser
    parser=create_parser()
    args=parser.parse_args(['build','-r','run','-g','0','-i','0','--engine','lora'])
    assert args.engine=='lora'
    args=parser.parse_args(['cycle','-r','run','--engine','lora'])
    assert args.engine=='lora'
    assert parser.parse_args(['advance','-r','run','--retry-partial']).retry_partial


def test_lora_cast_overflow_rejected(tmp_path):
    _, raw, roots = _create_adapters(tmp_path)
    raw['out_dtype']='float16'
    root=roots[0]
    file=root/'adapter_model.safetensors'
    with safe_open(str(file),framework='pt',device='cpu') as reader:
        state={k:reader.get_tensor(k) for k in reader.keys()}
    key=next(k for k in state if '.lora_B.' in k)
    state[key].fill_(1e10)
    save_file(state,str(file))
    with pytest.raises(LoRAMergeError,match='overflowed target dtype'):
        build_lora(parse_spec(raw), [.3]*18, tmp_path/'bad-dtype')


def test_comparisons_excludes_unverified_manual_and_mismatched_protocol(tmp_path):
    path, _, _ = _create_adapters(tmp_path, method='linear')
    run = tmp_path / 'run'
    exp.init_run(path, run)
    # A forged 100% manual score is NOT evidence of beating the measured baseline.
    exp.record_score(run, 0, 0, {'coding': 1.0, 'reasoning': 1.0}, source='manual')
    with pytest.raises(exp.BreederError, match='baseline'):
        exp.comparisons(run)
    spec, _ = exp.load_run(run)
    baseline = exp._score_document(spec, {'coding': .65, 'reasoning': .65}, 'lm_eval')
    folder = run / 'baselines' / 'base'; folder.mkdir(parents=True)
    exp._write_json(folder / 'score.json', baseline)
    assert exp.comparisons(run) == []
    exp.record_score(run, 0, 1, {'coding': .72, 'reasoning': .72}, source='lm_eval')
    results = exp.comparisons(run)
    assert [item['id'] for item in results] == ['g000-c001']
    assert results[0]['delta'] == pytest.approx(.07)
    # Same task labels but different fewshot/limit settings are not comparable.
    s = exp.candidate_dir(run, 0, 1) / 'score.json'
    payload = json.loads(s.read_text())
    payload['evaluation_settings']['fewshot'] = 5
    s.write_text(json.dumps(payload))
    assert exp.comparisons(run) == []


def test_lora_export_bfloat16_rank_sum_is_finite(tmp_path):
    _, raw, roots = _create_adapters(tmp_path, method='linear')
    raw['out_dtype'] = 'bfloat16'
    spec = parse_spec(raw)
    output = tmp_path / 'bf16'
    build_lora(spec, [.5] * spec.genome_size, output, method='linear')
    config = json.loads((output / 'adapter_config.json').read_text())
    assert config['r'] == 5
    with safe_open(str(output / 'adapter_model.safetensors'), framework='pt', device='cpu') as handle:
        for k in handle.keys():
            tensor = handle.get_tensor(k)
            assert tensor.dtype == torch.bfloat16
            assert torch.isfinite(tensor).all()


def test_lite_grouped_attention_vs_mlp_actual_tensors(tmp_path):
    from lerp.lite import build_lite
    base = tmp_path / 'base'; base.mkdir()
    parents = []
    keys = ('model.layers.0.self_attn.q_proj.weight', 'model.layers.0.mlp.up_proj.weight')
    for parent_idx, offset in enumerate((0.0, 10.0)):
        parent = tmp_path / f'full_{parent_idx}'; parent.mkdir()
        (parent / 'config.json').write_text(json.dumps({'model_type': 'llama', 'num_hidden_layers': 1}))
        save_file({key: torch.full((2,2), offset + (1 if 'mlp' in key else 0), dtype=torch.float32) for key in keys}, str(parent / 'model.safetensors'))
        parents.append(parent)
    (base / 'config.json').write_text(json.dumps({'model_type': 'llama', 'num_hidden_layers': 1}))
    raw = {'name':'groups', 'base_model': str(base), 'method':'linear',
           'parents': [{'name': f'p{i}', 'model': str(p)} for i, p in enumerate(parents)],
           'genes':2, 'out_dtype':'float32', 'gene_groups':['attention','mlp','other']}
    spec = parse_spec(raw)
    # Parent-0 alpha = 0.9 for attention and 0.1 for MLP.
    genes = [.9,.9, .1,.1, .5,.5]
    dest = tmp_path/'merged'
    build_lite(spec, genes, dest)
    with safe_open(str(dest/'model-00001.safetensors'), framework='pt', device='cpu') as reader:
        assert reader.get_tensor(keys[0]) == pytest.approx(torch.full((2,2), 1.0))
        assert reader.get_tensor(keys[1]) == pytest.approx(torch.full((2,2), 10.0))


def test_manual_scores_must_be_explicitly_opted_into_evolution(tmp_path):
    path, _, _ = _create_adapters(tmp_path, method='linear')
    run = tmp_path / 'run'
    exp.init_run(path, run)
    for i in range(4):
        exp.record_score(run, 0, i, {'coding': .7, 'reasoning': .6}, source='manual')
    with pytest.raises(exp.BreederError, match='Unverified manual'):
        exp.advance(run)
    assert exp.advance(run, allow_manual=True) == 1


def test_version_metadata_and_optional_smoke_cli_help():
    import lerp
    import subprocess
    import sys
    assert lerp.__version__ == '0.5.0'
    cmd = [sys.executable, str(Path(__file__).resolve().parents[1]/'examples/smoke_peft_load.py'), '--help']
    check = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert '--adapter' in check.stdout and '--device' in check.stdout
