"""Finish the paused 62-round gate in a new directory; never trains here."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from contextlib import ExitStack
from pathlib import Path

def main():
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now, supervision_status
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',required=True)
    parser.add_argument('--status',action='store_true')
    parser.add_argument('--check-only',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    def inside(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root) or p==root:raise ValueError('outside alphazero')
        return p
    path=inside(args.plan);plan=read_json(path);directory=inside(plan['directory'])
    if args.status:
        state=supervision_status(directory);state['stop_requested']=any(inside(p).exists() for p in plan['stop_paths'])
        print(json.dumps(state,ensure_ascii=False));return
    if plan['schema']!='junqi-frozen-completion-plan-1':raise ValueError('schema mismatch')
    import torch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import model_hash
    from junqi_open_v4.arena import run_arena,select_families
    from junqi_open_v4.pipeline import promotion_decision
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.search import BatchedNeuralPUCT
    from run_open_v4_long_behavior import screening_signal,confirmed_route
    from run_open_v4_search_training import verify_learning_state
    from open_v4_experiment_evidence import teacher_deltas
    from open_v4_cached_arena import matchups,suite_digest,compare_repeat,assemble
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    original=read_json(inside(plan['original_plan']['path']))
    cached=read_json(inside(plan['cached_arena']['path']))
    base=OpenConfig.load(inside(original['parent_config']))
    suite=select_families(read_json(inside(original['suite']['path']))['states'],64,original['evaluation_seed'])
    expected={'physical_batch':32,'precision':'BF16','padding':'repeat final input, discard padded outputs',
              'root_noise':False,'root_guard':False,'budget':256,'historical_dynamic_batch_scores_not_same_compute_context':True}
    if cached['inference_contract']!=expected or plan['inference_contract']!=expected:
        raise ValueError('cached inference context changed')
    if cached['seed']!=original['evaluation_seed'] or cached['suite_digest']!=suite_digest(suite):
        raise ValueError('cached suite/seed mismatch')
    if len(cached['layout_families'])!=64:raise ValueError('full64 cache required')
    engine={'cpuct':1.5,'cache_size':0,'leaves_per_tree':1,'reuse_roots':False,'cuda_graphs':False,
            'physical_batch':32,'precision':'BF16','root_noise':False,'root_guard':False,'budget':256}
    if plan['engine_contract']!=engine or base.device!='cuda' or base.cache_size!=0 or base.leaves_per_tree!=1:
        raise ValueError('frozen search engine context changed')
    if original['evaluation_budget']!=256 or original['evaluation_inference_batch_size']!=32 or original['full_families']!=64:
        raise ValueError('original gate compute context changed')
    for name in ('champion','parent'):
        for k in ('sha256','model_hash'):
            if cached['checkpoints'][name][k]!=plan['participants'][name][k]:raise ValueError('baseline identity changed')
    if plan['participants']['parent']['sha256']!=plan['participants']['repeat']['sha256']:
        raise ValueError('fresh repeat source differs')
    def verify():
        for entry in [plan['original_plan'],plan['cached_arena'],*plan['frozen_files']]:
            if sha(inside(entry['path']))!=entry['sha256']:raise ValueError('frozen input changed: '+entry['path'])
        for name,digest in plan['source'].items():
            if sha(inside(name))!=digest:raise ValueError('executed source changed: '+name)
        for item in plan['participants'].values():
            if sha(inside(item['path']))!=item['sha256']:raise ValueError('checkpoint changed')
    stops=[inside(p) for p in plan['stop_paths']]
    if any(p.exists() for p in stops):raise RuntimeError('STOP requested; do not clear automatically')
    with ExitStack() as locks:
        for run in plan['lock_directories']:
            for name in ('SUPERVISOR','TRAINING'):
                locks.enter_context(ProcessLock(inside(run)/f'supervision/{name}.lock'))
        verify()
        proofs={}
        for name,item in plan['participants'].items():
            saved=torch.load(inside(item['path']),map_location='cpu',weights_only=False)
            proofs[name]=verify_learning_state(saved,saved)
            if saved['model_hash']!=item['model_hash'] or saved['counters']['iteration']!=item['iteration']:
                raise ValueError('checkpoint metadata mismatch')
            if saved['config']['model']!=base.as_dict()['model'] or saved['leaf_value_scale']!=item['leaf_value_scale']:
                raise ValueError('checkpoint architecture or Value scale changed')
            if hashlib.sha256(json.dumps(saved['config'],sort_keys=True).encode()).hexdigest()!=saved['config_hash']:
                raise ValueError('full learning configuration digest mismatch')
            if set(saved['rng'])!={'python','numpy','torch','cuda','batch_generator'}:
                raise ValueError('incomplete random number generator state')
            for field in ('model','optimizer','scheduler'):
                def finite(value):
                    if isinstance(value,torch.Tensor):return bool(torch.isfinite(value).all())
                    if isinstance(value,dict):return all(finite(v) for v in value.values())
                    if isinstance(value,(list,tuple)):return all(finite(v) for v in value)
                    return True
                if not finite(saved[field]):raise ValueError('non-finite learning state')
            proofs[name].update(config_verified=True,all_tensor_states_finite=True)
        # Full cached arena split/merge must reproduce original intervals exactly.
        keep=[g for g in cached['games'] if g['candidate'] not in ('champion','parent') or not g['opponent'].startswith('teacher')]
        rebuilt=assemble(cached,[{'games':keep}],suite,original['evaluation_seed'],reuse=True)
        if rebuilt['summaries']!=cached['summaries'] or rebuilt['paired_deltas']!=cached['paired_deltas']:
            raise ValueError('canonical reaggregation does not reproduce original arena')
        if args.check_only:
            print(json.dumps({'valid':True,'original_statistics_exactly_reproduced':True,'full_learning_states_verified':True,
                              'planned_fresh_games':2916,'planned_reused_games':1296,'fallback_fresh_games':4212}));return
        if directory.exists()!=args.resume:raise ValueError('new run or reviewed explicit resume required')
        directory.mkdir(parents=True,exist_ok=True)
        for name in ('SUPERVISOR','TRAINING'):
            locks.enter_context(ProcessLock(directory/f'supervision/{name}.lock'))
        contract={'plan_sha256':sha(path),'source':plan['source']}
        if args.resume:
            if read_json(directory/'CONTRACT.json')!=contract:raise ValueError('resume contract changed')
            progress=read_json(directory/'PROGRESS.json')
        else:
            atomic_json(directory/'CONTRACT.json',contract);progress={'parts':{},'new_training_games':0,'completed_updates':0}
            atomic_json(directory/'verification/PREFLIGHT.json',{'states':proofs,'canonical_statistics_reproduced':True})
        state={'schema':plan['schema'],'state':'checking','pid':os.getpid(),'started_at':utc_now(),
               'plan':str(path.relative_to(root)),'phase':'frozen_development_evaluation','completed_iteration':62,
               'new_training_games':0,'completed_updates':0,'registered_training_games_before_job':12288,
               'planned_fresh_games':2916,'planned_reused_games':1296,'route_confirmed':False}
        def update(**fields):
            state.update(fields,updated_at=utc_now());atomic_json(directory/'supervision/STATUS.json',state);atomic_json(directory/'STATUS.json',state)
        def check():
            if any(p.exists() for p in stops):
                update(state='stopped_by_user',review_required=False);raise RuntimeError('STOP requested; frozen checkpoints preserved')
        def commit():atomic_json(directory/'PROGRESS.json',progress)
        atomic_json(directory/'CURRENT.json',{'iteration':62,'learner':plan['participants']['candidate'],
            'champion':plan['participants']['champion'],'actor':plan['participants']['champion'],
            'experimental_collectors':{k:plan['participants'][k] for k in ('control','candidate')}})
        device=torch.device(base.device);torch.set_num_threads(base.workers)
        job_started=time.monotonic()
        update()
        telemetry={}
        class MeasuredSearch:
            def __init__(self,name,impl):self.name=name;self.impl=impl;self.last=0.
            def __getattr__(self,name):return getattr(self.impl,name)
            def search(self,states,simulations,**kwargs):
                check();results,metrics=self.impl.search(states,simulations,**kwargs)
                totals=telemetry.setdefault(self.name,{'physical_inference_rows':0,'logical_inference_rows':0,'neural_moves':0,
                    'physical_inference_calls':0,'padding_rows':0,'completed_simulations':0})
                totals['physical_inference_rows']+=metrics['physical_inference_rows'];totals['logical_inference_rows']+=metrics['inference_rows'];totals['neural_moves']+=len(states)
                physical=metrics['physical_inference_rows'];logical=metrics['inference_rows']
                if physical%32 or physical<logical:raise ValueError('physical batch accounting mismatch')
                totals['physical_inference_calls']+=physical//32;totals['padding_rows']+=physical-logical
                totals['completed_simulations']+=sum(r.simulations for r in results)
                if time.monotonic()-self.last>20:
                    self.last=time.monotonic();update(active_agent=self.name,inference_accounting=telemetry)
                return results,metrics
        models={};agents={}
        for name,item in plan['participants'].items():
            saved=torch.load(inside(item['path']),map_location='cpu',weights_only=False)
            model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(saved['model']);model.search_v3_leaf_value_scale=saved['leaf_value_scale']
            if model_hash(model)!=item['model_hash']:raise ValueError('tensor identity mismatch')
            models[name]=model
            agents[name]=MeasuredSearch(name,BatchedNeuralPUCT(model,device=device,inference_batch_size=base.inference_batch_size,
                cpuct=1.5,cache_size=0,leaves_per_tree=1,reuse_roots=False,root_tactical_guard=False,
                fixed_inference_batch_size=32,cuda_graphs=False))
        def part(name,pairs,teachers):
            check();verify()
            if name in progress['parts']:
                record=progress['parts'][name]
                if sha(inside(record['path']))!=record['sha256']:raise ValueError('committed part changed')
                return read_json(inside(record['path']))
            update(state='arena',arena=name,planned_games=len(suite)*len(pairs)*2)
            used={a for pair in pairs for a in pair if a in agents}
            before={k:dict(v) for k,v in telemetry.items()}
            result=run_arena(suite,pairs,{k:agents[k] for k in agents if k in used},teachers,{k:256 for k in used},seed=original['evaluation_seed'])
            result['physical_inference_accounting']={k:{field:v[field]-before.get(k,{}).get(field,0) for field in v} for k,v in telemetry.items() if k in used}
            result['inference_contract']=expected
            result['peak_gpu_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            output=directory/'arena'/f'{name}.json';atomic_json(output,result)
            progress['parts'][name]={'path':str(output.relative_to(root)),'sha256':sha(output),'games':len(result['games'])};commit()
            update(committed_fresh_games=sum(r['games'] for r in progress['parts'].values()))
            return result
        try:
            with TeacherLeague(root,base.teacher_candidates,base.workers) as teachers:
                if teachers.provenance!=original['teachers']:raise ValueError('teacher identity changed')
                submit=teachers.submit_actions
                def checked_submit(*a,**kw):check();return submit(*a,**kw)
                teachers.submit_actions=checked_submit
                repeat=part('repeat_source38',[('repeat',f'teacher{i}') for i in range(2)],teachers)
                reliability=compare_repeat(cached,repeat,suite)
                atomic_json(directory/'verification/CACHE_REPEAT.json',reliability)
                cached_pairs=[(r,f'teacher{i}') for i in range(2) for r in ('champion','parent')]
                new_pairs=[p for p in matchups() if p not in cached_pairs and p[0]!='repeat']
                parts=[repeat,part('new62_matches',new_pairs,teachers)]
                reuse=reliability['all_exact']
                if not reuse:
                    update(cache_disabled=True,reason='fresh source traces differ; rerun all cached baseline games',planned_fresh_games=4212,planned_reused_games=0)
                    parts.append(part('fresh_baselines',cached_pairs,teachers))
                full=assemble(cached,parts,suite,original['evaluation_seed'],reuse=reuse)
                full.update(checkpoints=plan['participants'],inference_contract=expected,cache_repeat=reliability,
                    elapsed_seconds=sum(p['elapsed_seconds'] for p in parts),teacher_workers=teachers.workers,
                    current_job_wall_seconds=time.monotonic()-job_started,engine_contract=engine,
                    teacher_execution='spawned CPU processes overlapping neural search',physical_inference_accounting={k:v['physical_inference_accounting'] for k,v in zip(['repeat','new62','fallback'],parts)},
                    cache_provenance={'source':plan['cached_arena'],'only_roles':['champion','parent'],'reused':reuse},
                    budget_comparison='fixed added simulations; fresh compute separately accounted')
                gains={k:teacher_deltas(full,before,'candidate') for k,before in [('champion','champion'),('parent','parent'),('control','control')]}
                duplicate=teacher_deltas(full,'parent','repeat');gains['duplicate']=duplicate
                gate=promotion_decision(full,base);duels=[r['family_score'] for r in full['summaries'][:3]]
                full['review']={'gains':gains,'gate':gate,'screen_promising':screening_signal([gains[k] for k in ('champion','parent','control')],duels),
                    'route_confirmed':confirmed_route([gains[k] for k in ('champion','parent','control')],duplicate,duels,gate,64,.02),'screen_only':False}
                output=directory/'arena/block_000024_full.json';atomic_json(output,full)
                progress['complete_arena']={'path':str(output.relative_to(root)),'sha256':sha(output),'review':full['review']};commit()
                atomic_json(directory/'RESULTS.json',{'state':'review_due','new_training_games':0,'completed_updates':0,
                    'arena':progress['complete_arena'],'fresh_games':full['fresh_games'],'reused_games':full['reused_games'],
                    'interpretation':'Frozen completion of original62 full64 gate; no automatic promotion or horizon extension; original STOP, states and all negative results preserved.'})
                update(state='review_due',review_required=True,route_confirmed=full['review']['route_confirmed'],
                    committed_fresh_games=full['fresh_games'],reused_games=full['reused_games'],completed_games=len(full['games']))
        except BaseException as error:
            if state['state']!='stopped_by_user':update(state='needs_attention',reason=repr(error),review_required=True)
            raise

if __name__=='__main__':main()
