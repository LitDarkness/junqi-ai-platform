"""Small frozen inference grouping diagnosis, without new games or updates."""
from contextlib import ExitStack
from pathlib import Path
import hashlib


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, read_json, atomic_json, utc_now
    from junqi_az.config import ModelConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.arena import select_families
    from junqi_open_v4.layouts import backend_for
    from junqi_search_v3.search import BatchedNeuralPUCT
    root=Path(__file__).resolve().parents[1]
    plan=read_json(root/'config/experiments/open_v4_tactical_training_20261004_065605.json')
    run=root/plan['directory'];arena=read_json(run/'arena/block_000006_screen.json')
    suite=select_families(read_json(root/plan['suite']['path'])['states'],16,plan['evaluation_seed'])
    groups={}
    for g in arena['games']:
        if g['candidate'] in ('parent','repeat') and g['opponent'].startswith('teacher'):
            groups.setdefault((g['position'],g['candidate_side'],g['opponent']),{})[g['candidate']]=g
    different=[]
    for key,pair in groups.items():
        a,b=pair['parent']['trace'],pair['repeat']['trace']
        for ply,(left,right) in enumerate(zip(a,b)):
            if left!=right:
                different.append((ply,key,pair,left,right));break
    different.sort(key=lambda r:r[0])
    ply,key,pair,left,right=different[0];d=suite[key[0]]
    s=backend_for(d['red'],d['black']).new_state(d['setup_seed'])
    for a in d['trace']+pair['parent']['trace'][:ply]:s.apply(a)
    assert s.current_player()==key[1]
    others=[]
    for desc in suite:
        other=backend_for(desc['red'],desc['black']).new_state(desc['setup_seed'])
        for a in desc['trace']:other.apply(a)
        if not other.is_terminal() and other.native_state.key()!=s.native_state.key():others.append(other)
    with ExitStack() as locks:
        for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json')):
            for name in ('SUPERVISOR','TRAINING'):locks.enter_context(ProcessLock(p.parent/f'supervision/{name}.lock'))
        src=plan['source_checkpoint'];path=root/src['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==src['sha256']
        p=torch.load(path,map_location='cpu',weights_only=False)
        torch.set_num_threads(4);device=torch.device('cuda')
        model=PolicyValueNet(ModelConfig(**p['config']['model'])).to(device).eval();model.load_state_dict(p['model']);model.search_v3_leaf_value_scale=p['leaf_value_scale']
        adjustment={k:float(v.abs().max()) for k,v in p['model'].items() if k.startswith('repetition_value_adjustment')}
        results={}
        for kind in ('dynamic','fixed32'):
            results[kind]=[]
            for size in (1,2,8,32):
                states=[s]+[others[i%len(others)].clone() for i in range(size-1)]
                search=BatchedNeuralPUCT(model,device=device,cuda_graphs=False,fixed_inference_batch_size=32 if kind=='fixed32' else 0)
                rows,metrics=search.search(states,256);r=rows[0]
                record=dict(batch=size,action=r.action,visits=r.visits.tolist(),q=r.q_values.tolist(),prior=r.priors.tolist(),metrics=metrics)
                results[kind].append(record)
                print(kind,size,r.action,flush=True)
        fixed=results['fixed32'];dynamic=results['dynamic']
        evidence=dict(at=utc_now(),phase='frozen_evaluation_determinism_diagnosis',new_games=0,updates=0,source=src,
            case=dict(position=key[0],side=key[1],teacher=key[2],relative_ply=ply,absolute_ply=ply+d['start_ply'],state_key=int(s.native_state.key()),
                      historical_parent_action=left,historical_repeat_action=right),
            differing_paired_traces=len(different),repetition_value_adjustment_parameter_maxima=adjustment,
            results=results,dynamic_exact_equal=all(r['visits']==dynamic[0]['visits'] and r['q']==dynamic[0]['q'] and r['prior']==dynamic[0]['prior'] for r in dynamic),
            fixed32_exact_equal=all(r['visits']==fixed[0]['visits'] and r['q']==fixed[0]['q'] and r['prior']==fixed[0]['prior'] for r in fixed),
            caveat='One divergence state, fixed BF16 batchshape32. No claim all numerical sources explained, and no retroactive model promotion.')
        evidence['production_search_code_sha256']=hashlib.sha256((root/'python/junqi_search_v3/search.py').read_bytes()).hexdigest()
        if not evidence['fixed32_exact_equal']:raise AssertionError('fixed-shape production search not identical across grouping')
        atomic_json(root/'reports/current_training_audit_20261004/BATCH_STABILITY_AFTER.json',evidence)


if __name__=='__main__':main()
