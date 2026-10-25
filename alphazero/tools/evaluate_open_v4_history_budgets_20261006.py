"""User-authorized finite frozen history assessment; never learns or promotes."""
from __future__ import annotations
import argparse, hashlib, json, os, time
from pathlib import Path
from datetime import datetime, timezone
from contextlib import ExitStack
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inside(p):
    q=(ROOT/p).resolve()
    if q==ROOT or not q.is_relative_to(ROOT): raise ValueError('Outside alphazero')
    return q

def prepare():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from evaluate_open_v4_longitudinal import archive_snapshot
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    master=read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    assert master['paused_by_user'] and not master['auto_next_stage']['automatic_continuation_enabled']
    assert not any(lock_held(ROOT/p) for p in ['runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',master['optimization']['run_directory']+'/supervision/SUPERVISOR.lock',master['optimization']['run_directory']+'/supervision/TRAINING.lock'])
    prior=read_json(ROOT/'config/experiments/open_v4_longitudinal_20261005_035049.json')
    current=ROOT/master['optimization']['run_directory']
    paths={
       'import0':prior['participants']['old150']['path'],
       'champ26':prior['participants']['champ26']['path'],
       'candidate62':prior['participants']['candidate62']['path'],
       'control86':prior['participants']['control86']['path'],
       'control110':prior['participants']['control110']['path'],
       'control134':prior['participants']['control134']['path'],
       'single158':'runs/open_junqi_v4_single_20261005_044709/learner/checkpoints/learner_stage_000024.pt',
       'fourfold182':'runs/open_junqi_v4_single_ratio_20261005_070831/learner/checkpoints/learner_stage_000024.pt',
       'restored170':master['optimization']['config'],
       'continuous206':str((current/'learner/checkpoints/learner_stage_000036.pt').relative_to(ROOT)),
       'qualified230':str((current/'learner/checkpoints/learner_stage_000060.pt').relative_to(ROOT)),
       'continuous266':str((current/'learner/checkpoints/learner_stage_000096.pt').relative_to(ROOT)),
       'latest302':str((current/'learner/checkpoints/learner_stage_000132.pt').relative_to(ROOT))}
    paths['restored170']=read_json(inside(paths['restored170']))['source_checkpoint']['path']
    basepath='config/experiments/open_v4_root_explore_base_20261004_214715.json'
    base=RootExploreConfig.load(ROOT/basepath)
    torch.set_num_threads(2)
    participants={}
    for name,path in paths.items():
        payload=torch.load(inside(path),map_location='cpu',weights_only=False)
        model=PolicyValueNet(base.model).eval();model.load_state_dict(payload['model'])
        digest=model_hash(model)
        assert all(torch.isfinite(t).all() for t in payload['model'].values())
        assert not payload.get('model_hash') or digest==payload['model_hash']
        assert payload['leaf_value_scale']==0.9076346023345448
        iteration=payload.get('counters',{}).get('iteration',payload.get('iteration'))
        participants[name]=dict(path=path,sha256=sha(inside(path)),model_hash=digest,
            iteration=iteration,plot_iteration=0 if name=='import0' else iteration,
            lineage='independent_failed_fourfold_branch' if name=='fourfold182' else ('continuous_restored_branch' if name in ['restored170','continuous206','qualified230','continuous266','latest302'] else 'historical_source'),
            leaf_value_scale=payload['leaf_value_scale'])
        del payload,model
    participants['repeat302']=participants['latest302'].copy()
    models=list(paths)
    matchups=[[name,'teacher0'] for name in models]+[['repeat302','teacher0'],['latest302','restored170'],['latest302','qualified230']]
    suite_path=ROOT/'runs/open_junqi_v4_fresh_continuation/development/suite.json'
    suite=read_json(suite_path)
    assert suite['contract']['role']=='dev'
    assert hashlib.sha256(json.dumps(suite['states'],sort_keys=True).encode()).hexdigest()==suite['states_hash']
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    directory=f'runs/open_junqi_v4_history_budgets_{stamp}'
    from junqi_open_v4.arena import select_families
    selected=select_families(suite['states'],16,2026100409)
    plan=dict(schema='junqi-frozen-history-budgets-20261006',registered_at=utc_now(),directory=directory,
        authorization='Latest user explicitly requests charts and testing missing data, plus bottleneck analysis. Frozen evaluation only; all training stays paused.',
        config=basepath,config_sha256=sha(ROOT/basepath),participants=participants,history=models,matchups=matchups,budgets=[0,64,256],
        suite=dict(path=str(suite_path.relative_to(ROOT)),sha256=sha(suite_path),states_hash=suite['states_hash'],families=16,starts=len(selected)),
        evaluation_seed=2026100409,workers=12,physical_batch=32,precision='BF16',cpuct=1.5,root_noise=False,guard=False,forced=False,cache=False,tree_reuse=False,
        planned_games=2*len(selected)*len(matchups)*3,
        interpretation='Fresh same-context16-family descriptive budget/history diagnostics; no production gate, no selection, no hypothesis-causal claim. Historical branches explicitly distinct; oldV3 150 imported asV4 seed0.',
        repeat_rule='Report actual same-model teacher repeat point drift and complete trajectories. Drift>2pp flags timing/load sensitivity; do not discard or selectively rerun.',
        stop_paths=[directory+'/STOP',directory+'/supervision/STOP'],
        historical_stop_and_role_preservation=archive_snapshot(),
        preserved_current_run=dict(directory=str(current.relative_to(ROOT)),progress_sha256=sha(current/'PROGRESS.json'),status_sha256=sha(current/'STATUS.json'),current_sha256=sha(current/'CURRENT.json')),
        frozen_sources={p:sha(ROOT/p) for p in ['build/JunqiAZ.cp310-win_amd64.pyd','python/junqi_open_v4/arena.py','python/junqi_open_v4/teachers.py','python/junqi_open_v4/layouts.py','python/junqi_search_v3/search.py','python/junqi_az/model.py','tools/evaluate_open_v4_longitudinal.py','tools/evaluate_open_v4_history_budgets_20261006.py']})
    target=ROOT/f'config/experiments/open_v4_history_budgets_{stamp}.json'
    assert not target.exists();atomic_json(target,plan)
    print(json.dumps(dict(config=str(target.relative_to(ROOT)),directory=directory,planned_games=plan['planned_games'],models=len(models),all_checkpoint_checks_passed=True)))

def evaluate(planpath):
    import torch,numpy as np
    from open_v4_runtime import read_json,atomic_json,utc_now,ProcessLock
    from evaluate_open_v4_longitudinal import FixedBare,verify_native
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_open_v4.arena import run_arena,select_families,paired_interval
    from junqi_open_v4.teachers import TeacherLeague
    plan=read_json(planpath);directory=inside(plan['directory'])
    if directory.exists():raise ValueError('Preserve existing assessment; no implicit restart')
    masterpath=ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json'
    with ExitStack() as locks:
        for name in ['NEXT_STAGE','SUPERVISOR','TRAINING']:
            locks.enter_context(ProcessLock(ROOT/f'runs/open_junqi_v4_repair_pilot/supervision/{name}.lock'))
        directory.mkdir(parents=True)
        for name in ['SUPERVISOR','TRAINING']:locks.enter_context(ProcessLock(directory/f'supervision/{name}.lock'))
        status=dict(schema=plan['schema'],state='checking',pid=os.getpid(),started_at=utc_now(),planned_games=plan['planned_games'],new_evaluation_games=0,new_training_games=0,effective_updates=0,training_paused=True)
        def update(**kw):status.update(kw,updated_at=utc_now());atomic_json(directory/'STATUS.json',status)
        def check():
            if any(inside(p).exists() for p in plan['stop_paths']):
                update(state='stopped_by_user');raise RuntimeError('Assessment STOP requested')
        def verify():
            check();master=read_json(masterpath)
            assert master['paused_by_user'] and not master['auto_next_stage']['automatic_continuation_enabled']
            for group in ['historical_stop_and_role_preservation','frozen_sources']:
                for p,digest in plan[group].items():assert sha(inside(p))==digest,(group,p)
            for p in plan['participants'].values():assert sha(inside(p['path']))==p['sha256']
            assert sha(inside(plan['suite']['path']))==plan['suite']['sha256']
            cur=inside(plan['preserved_current_run']['directory'])
            for name,key in [('PROGRESS.json','progress_sha256'),('STATUS.json','status_sha256'),('CURRENT.json','current_sha256')]:assert sha(cur/name)==plan['preserved_current_run'][key]
        class MeasuredSearch(FixedBare):
            def __init__(self,search):
                super().__init__(search,check);self.cost=defaultdict(float)
            def search(self,states,simulations,**kw):
                t=time.perf_counter();output,report=super().search(states,simulations,**kw)
                self.cost['search_wall_seconds']+=time.perf_counter()-t
                self.cost['calls']+=1;self.cost['neural_moves']+=len(states)
                for k in ['physical_inference_rows','inference_rows','inference_calls','simulations']:self.cost[k]+=report.get(k,simulations*len(states) if k=='simulations' else 0)
                return output,report
            def bare(self,states):
                t=time.perf_counter();output=super().bare(states)
                self.cost['bare_wall_seconds']+=time.perf_counter()-t;self.cost['neural_moves']+=len(states)
                return output
        update()
        try:
            verify();atomic_json(directory/'CONTRACT.json',plan|dict(plan_sha256=sha(planpath)))
            base=RootExploreConfig.load(inside(plan['config']));device=torch.device(base.device)
            torch.set_num_threads(base.workers);torch.manual_seed(plan['evaluation_seed'])
            suite=select_families(read_json(inside(plan['suite']['path']))['states'],16,plan['evaluation_seed'])
            assert len(suite)==plan['suite']['starts'] and len({s['family'] for s in suite})==16
            models={}
            for name,item in plan['participants'].items():
                if name=='repeat302':models[name]=models['latest302'];continue
                payload=torch.load(inside(item['path']),map_location='cpu',weights_only=False)
                model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(payload['model'])
                model.search_v3_leaf_value_scale=payload['leaf_value_scale'];assert model_hash(model)==item['model_hash']
                models[name]=model;del payload
            outputs={}
            with TeacherLeague(ROOT,base.teacher_candidates,workers=plan['workers']) as teachers:
                atomic_json(directory/'TEACHER_PROVENANCE.json',teachers.provenance)
                for budget in plan['budgets']:
                    verify();agents={name:MeasuredSearch(BatchedNeuralPUCT(model,device=device,cpuct=1.5,inference_batch_size=256,fixed_inference_batch_size=32,cache_size=0,reuse_roots=False,root_tactical_guard=False,leaves_per_tree=1,cuda_graphs=False)) for name,model in models.items()}
                    update(state='arena',budget=budget)
                    print(f'[history_budget] budget={budget} starts={len(suite)} matchups={len(plan["matchups"])} planned={2*len(suite)*len(plan["matchups"])}',flush=True)
                    with torch.inference_mode():arena=run_arena(suite,[tuple(x) for x in plan['matchups']],agents,teachers,{name:budget for name in agents},seed=plan['evaluation_seed'])
                    update(state='verifying',budget=budget)
                    native=verify_native(arena,suite,check)
                    assert native['games']==2*len(suite)*len(plan['matchups'])
                    indexed=lambda role:{(g['position'],g['candidate_side']):g for g in arena['games'] if g['candidate']==role and g['opponent']=='teacher0'}
                    original,duplicate=indexed('latest302'),indexed('repeat302')
                    same=sum(all(original[k][f]==duplicate[k][f] for f in ['trace','reason','value']) for k in original)
                    family_values=lambda rows:{f:float(np.mean([g['score'] for g in rows.values() if g['family']==f])) for f in {g['family'] for g in rows.values()}}
                    ov,dv=family_values(original),family_values(duplicate)
                    drift=paired_interval([dv[f]-ov[f] for f in sorted(ov)],plan['evaluation_seed']^0xD00D)
                    costs={name:dict(a.cost)|dict(bare_calls=a.bare_calls,bare_physical_rows=a.bare_physical_rows,bare_logical_rows=a.bare_logical_rows) for name,a in agents.items()}
                    arena.update(checkpoints=plan['participants'],native_verification=native,inference_contract={k:plan[k] for k in ['physical_batch','precision','cpuct','root_noise','guard','forced','cache','tree_reuse']}|dict(budget=budget),physical_compute_cost=costs,repeat_diagnostic=dict(games=len(duplicate),complete_trajectory_same=same,point_drift=drift,drift_within2pp=abs(drift['mean'])<=.02),purpose='Frozen descriptive history/budget assessment. No learning, promotion or weight selection.')
                    path=directory/f'arena_{budget}.json';atomic_json(path,arena)
                    outputs[str(budget)]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),native=native,summaries=arena['summaries'],repeat_diagnostic=arena['repeat_diagnostic'],elapsed_seconds=arena['elapsed_seconds'],physical_compute_cost=costs)
                    verify();assert all(model_hash(model)==plan['participants'][name]['model_hash'] for name,model in models.items())
                    atomic_json(directory/'PROGRESS.json',dict(outputs=outputs))
                    update(state='committed',budget=budget,new_evaluation_games=sum(o['native']['games'] for o in outputs.values()))
                    print(f'[history_budget] committed budget={budget} games={native["games"]} native_plies={native["native_plies"]} seconds={arena["elapsed_seconds"]:.1f} duplicate_same={same}/{len(duplicate)} drift={drift["mean"]:.6f}',flush=True)
                    del agents,arena;torch.cuda.empty_cache()
            verify()
            result=dict(schema=plan['schema'],outputs=outputs,new_evaluation_games=status['new_evaluation_games'],new_training_games=0,effective_updates=0,training_still_paused=True,promotion='none',weights_roles_archives_unchanged=True,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,completed_at=utc_now())
            assert result['new_evaluation_games']==plan['planned_games']
            atomic_json(directory/'RESULTS.json',result);update(state='complete',completed_at=utc_now())
            print(json.dumps(dict(complete=True,games=result['new_evaluation_games'],peak_gpu_allocated_mib=result['peak_gpu_allocated_mib'])),flush=True)
        except BaseException as error:
            if status['state']!='stopped_by_user':update(state='needs_attention',error=repr(error))
            raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--plan')
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.plan:evaluate(inside(args.plan))
    else:parser.error('--prepare or --plan required')
