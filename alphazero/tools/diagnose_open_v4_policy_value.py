"""Bounded frozen complete-output diagnosis on saved training roots, not new games."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
from contextlib import ExitStack


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock, supervision_status
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--status', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def inside(p):
        p = (root / p).resolve()
        if not p.is_relative_to(root) or p == root: raise ValueError('outside project')
        return p
    plan_path = inside(args.plan); plan = read_json(plan_path); run = inside(plan['directory'])
    if args.status:
        print(json.dumps(supervision_status(run), ensure_ascii=False)); return
    master_path = root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json'
    if plan['schema'] != 'junqi-complete-output-diagnostic-1' or plan['roots'] != 16 or plan['budget'] != 64:
        raise ValueError('bounded diagnostic contract required')
    stops = [inside(p) for p in plan['stop_paths']]
    if run.exists(): raise ValueError('never repeat a completed or partial diagnostic without review')
    def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    def check():
        master = read_json(master_path)
        if master.get('paused_by_user') or master['optimization']['config'] != str(plan_path.relative_to(root)).replace('\\','/') or any(p.exists() for p in stops):
            raise RuntimeError('latest authorization/STOP wins')
    def verify():
        for p,h in plan['source'].items():
            if sha(inside(p)) != h: raise ValueError('frozen execution/source changed: '+p)
        for item in plan['historical_stops'] + list(plan['models'].values()):
            if sha(inside(item['path'])) != item['sha256']: raise ValueError('protected artifact changed')
        for records in plan['corpora'].values():
            for item in records:
                if sha(inside(item['path']))!=item['sha256'] or sha(inside(item['metadata']))!=item['metadata_sha256']:
                    raise ValueError('saved training source changed')
    check()
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
        run.mkdir(parents=True)
        locks.enter_context(ProcessLock(run/'supervision/SUPERVISOR.lock'))
        locks.enter_context(ProcessLock(run/'supervision/TRAINING.lock'))
        status = dict(state='diagnosing', pid=os.getpid(), phase='frozen_saved_training_search', started_at=utc_now(),new_training_games=0,effective_updates=0)
        def update(**kw):
            status.update(kw, updated_at=utc_now()); atomic_json(run/'STATUS.json',status); atomic_json(run/'supervision/STATUS.json',status)
        update()
        try:
            verify()
            import numpy as np, torch
            from junqi_az.model import PolicyValueNet
            from junqi_open_v4.root_explore_config import RootExploreConfig
            from junqi_open_v4.training_replay import complete_games
            from junqi_open_v4.layouts import load_role, MultiLayoutBackend
            from junqi_open_v4.learner import model_hash
            from junqi_search_v3.loop import read_selfplay_shard
            from junqi_search_v3.search import BatchedNeuralPUCT
            from run_open_v4_fresh_data import is_holdout
            from evaluate_open_v4_search_budgets import StopAwareSearch
            base = RootExploreConfig.load(inside(plan['parent_config']))
            torch.set_num_threads(2); device = torch.device('cuda')
            if not torch.cuda.is_available(): raise ValueError('registered BF16 diagnostic requires CUDA')
            torch.manual_seed(base.seed)
            class CompleteOutputs(torch.nn.Module):
                def __init__(self,p,v):
                    super().__init__(); self.p=p; self.v=v; self.config=p.config
                    self.search_v3_leaf_value_scale=v.search_v3_leaf_value_scale
                def policy_value(self,*inputs):
                    pp,pv=self.p.policy_value(*inputs)
                    if self.p is self.v: return pp,pv
                    _,vv=self.v.policy_value(*inputs)
                    return pp,vv
            models={}
            for name,item in plan['models'].items():
                payload=torch.load(inside(item['path']),map_location='cpu',weights_only=False)
                model=PolicyValueNet(base.model).to(device).eval(); model.load_state_dict(payload['model'])
                model.search_v3_leaf_value_scale=payload['leaf_value_scale']
                if model_hash(model)!=item['model_hash']:raise ValueError('tensor identity changed')
                models[name]=model
            if models['158'].search_v3_leaf_value_scale != models['182'].search_v3_leaf_value_scale:
                raise ValueError('diagnostic scale mismatch')
            backend=MultiLayoutBackend(load_role(inside(base.layout_directory),'train'),base.seed)
            selected=[]; seen=set()
            quotas=[('selfplay','early',2),('selfplay','middle',1),('teacher_vs_actor','early',1),('teacher_vs_actor','middle',2),('teacher_selfplay','early',1),('teacher_selfplay','middle',1)]
            for corpus,records in plan['corpora'].items():
                candidates=[]
                for entry in records:
                    meta=read_json(inside(entry['metadata']))
                    if meta['layout_role']!='train':raise ValueError('only saved training roots')
                    samples=read_selfplay_shard(inside(entry['path']))
                    for game,rows in complete_games(samples,meta):
                        if not is_holdout(int(game['family']),base.seed):continue
                        if len(rows)<28:continue
                        candidates.append((entry,game,rows))
                for mode,phase,count in quotas:
                    items=[x for x in candidates if x[1]['mode']==mode]
                    items.sort(key=lambda x:hashlib.sha256(f"{base.seed}:output_diag:{corpus}:{phase}:{x[1]['family']}:{x[1]['index']}".encode()).hexdigest())
                    got=0
                    for entry,game,rows in items:
                        if game['family'] in seen:continue
                        sample=rows[min(6,len(rows)-1) if phase=='early' else max(24,len(rows)//2)]
                        state=backend.new_state(game['provenance']['setup_seed'])
                        for action in game['trace'][:sample.ply]:
                            if action not in state.legal_actions():raise ValueError('illegal full-history prefix')
                            state.apply(action)
                        if state.is_terminal() or int(state.native_state.key())!=sample.state_hash or not np.array_equal(state.legal_actions(),sample.legal_actions) or not np.array_equal(state.repetition_contexts(),sample.repetition_contexts) or not np.array_equal(state.feature_planes().astype(np.float16),sample.observation):
                            raise ValueError('saved complete root mismatch')
                        seen.add(game['family']);selected.append((state,dict(corpus=corpus,mode=mode,phase=phase,family=game['family'],game_index=game['index'],ply=sample.ply,state_key=int(state.native_state.key()),source=entry,prefix=game['trace'][:sample.ply],provenance=game['provenance'])))
                        got+=1
                        if got==count:break
                    if got!=count:raise ValueError('insufficient stratified roots')
            if len(selected)!=16:raise ValueError('root budget mismatch')
            states=[x[0] for x in selected]; identities=[x[1] for x in selected]
            atomic_json(run/'SELECTION.json',{'roots':identities,'selection_is_training_holdout_not_development':True,'no_counterfactual_terminal_labels':True})
            combos={'P158_V158':('158','158'),'P182_V158':('182','158'),'P158_V182':('158','182'),'P182_V182':('182','182')}
            outputs={};metrics={}
            for label,(p,v) in combos.items():
                check();update(combination=label)
                engine=StopAwareSearch(BatchedNeuralPUCT(CompleteOutputs(models[p],models[v]),device=device,cpuct=1.5,fixed_inference_batch_size=32,cuda_graphs=False,cache_size=0,reuse_roots=False,root_tactical_guard=False),check)
                results,cost=engine.search(states,64,add_root_noise=False)
                metrics[label]=cost
                rows=[]
                for state,result,identity in zip(states,results,identities):
                    if result.visits.sum()!=63 or result.action not in state.legal_actions() or int(state.native_state.key())!=identity['state_key']:
                        raise ValueError('budget/legality/root preservation failed')
                    rows.append(dict(action=int(result.action),actions=result.actions.tolist(),visits=result.visits.tolist(),priors=result.priors.tolist(),q=result.q_values.tolist(),entropy=result.root_entropy))
                outputs[label]=rows
                atomic_json(run/f'{label}.json',{'complete_output_sources':dict(policy=p,value=v),'outputs':rows,'cost':cost})
            changes=[i for i in range(16) if outputs['P158_V158'][i]['action']!=outputs['P182_V182'][i]['action']]
            recovering={label:sum(outputs[label][i]['action']==outputs['P158_V158'][i]['action'] for i in changes) for label in ('P182_V158','P158_V182')}
            result={'schema':plan['schema'],'plan_sha256':sha(plan_path),'models':plan['models'],'checks_passed':True,'roots':16,'simulations':4096,'new_training_games':0,'effective_updates':0,'teacher_queries':0,'changed_complete_recommendation_roots':len(changes),'source_recommendation_recovery_on_changed_roots':recovering,'identities':identities,'outputs':outputs,'metrics':metrics,'interpretation':'Mechanism diagnosis only: complete-output mixing changes search. Recovery of an old recommendation is not proof it wins; no unplayed successor receives old terminal truth; not production adoption.'}
            verify();check();atomic_json(run/'RESULTS.json',result)
            update(state='segment_complete',review_required=True,results='RESULTS.json',completed_simulations=4096)
            print(json.dumps({'completed':True,'directory':plan['directory'],'changed_roots':len(changes),'recovery':recovering,'new_games':0,'updates':0}))
        except BaseException as e:
            update(state='needs_attention',reason=repr(e));raise


if __name__=='__main__':main()
