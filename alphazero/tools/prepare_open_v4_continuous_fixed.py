"""Register one new continuous owner from the latest complete restored learner."""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import unittest


def main():
    import torch
    import numpy as np
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_continuous_fixed_contract import verify_stable_seed,validate_source_config,verify_single_prerequisite
    from open_v4_mechanism_reversion_contract import small_screen_review as historical_screen
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    from junqi_az.model import PolicyValueNet
    from junqi_search_v3.loop import read_selfplay_shard
    from open_v4_root_explore_contract import verify_collection
    from junqi_open_v4.arena import select_families
    from evaluate_open_v4_longitudinal import verify_native
    parser=argparse.ArgumentParser();parser.add_argument('--previous-plan',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def inside(name):
        p=(root/name).resolve()
        if not p.is_relative_to(root) or p==root:raise ValueError('outside alphazero')
        return p
    def entry(p):
        p=inside(p);return dict(path=str(p.relative_to(root)),sha256=sha(p))
    previous_path=inside(args.previous_plan);old=read_json(previous_path);old_run=inside(old['directory'])
    master_path=root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json';master=read_json(master_path)
    if master.get('paused_by_user'):raise ValueError('user pause overrides')
    if any(inside(p).exists() for p in old['stop_paths']):raise ValueError('never bypass an existing current STOP')
    for p in [root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',old_run/'supervision/TRAINING.lock',old_run/'supervision/SUPERVISOR.lock']:
        if lock_held(p):raise ValueError('previous owner must exit before registration')
    status=read_json(old_run/'STATUS.json');progress=read_json(old_run/'PROGRESS.json')
    if status['state'] not in ('review_due','segment_complete'):raise ValueError('complete natural boundary required')
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise ValueError('boundary check failed: '+name)
    for p,h in old['source'].items():check('source:'+p,sha(inside(p))==h)
    for e in old['historical_stops']:check('STOP:'+e['path'],sha(inside(e['path']))==e['sha256'])
    for e in old['frozen_runs']:check('CURRENT:'+e['directory'],sha(inside(e['directory'])/'CURRENT.json')==e['current_sha256'])
    item=progress['variants']['learner'];check('latest_file_sha',sha(inside(item['checkpoint']))==item['sha256'])
    source=torch.load(inside(item['checkpoint']),map_location='cpu',weights_only=False)
    cfg=RootExploreConfig.load(old_run/'learner/config.json')
    check('all_complete_state_fields',all(k in source for k in ('model','optimizer','scheduler','counters','rng','config','leaf_value_scale')))
    check('current_source_counters',source['counters']==item['counters'])
    check('current_source_config',source['config']==cfg.as_dict() and source['config_hash']==cfg.digest())
    validate_source_config(source,cfg)
    model=PolicyValueNet(cfg.model);model.load_state_dict(source['model'])
    check('actual_tensor_hash',model_hash(model)==item['model_hash']==source['model_hash'])
    check('finite_model',all(torch.isfinite(t).all().item() for t in source['model'].values()))
    before=torch.load(inside(old['source_checkpoint']['path']),map_location='cpu',weights_only=False)
    seed=torch.load(old_run/'learner/checkpoints/learner_stage_000000.pt',map_location='cpu',weights_only=False)
    checks.update({'previous_seed:'+k:bool(v) for k,v in verify_stable_seed(before,seed).items()})
    stage_rows=[];counter_before=before['counters'];games=accepted=0
    for record in progress['collections']['learner']:
        stage=record['stage'];meta=read_json(inside(record['metadata']))
        check(f'{stage}:shard_sha',sha(inside(record['path']))==record['sha256'])
        check(f'{stage}:metadata_sha',sha(inside(record['metadata']))==record['metadata_sha256'])
        checkpoint=old_run/f'learner/checkpoints/learner_stage_{stage:06d}.pt'
        after=torch.load(checkpoint,map_location='cpu',weights_only=False)
        report_path=old_run/f'learner/reports/stage_{stage:06d}.json';report=read_json(report_path)
        rows=report['updates'];accepted_stage=sum(int(v['accepted']) for v in rows)
        check(f'{stage}:64_attempts',len(rows)==64 and after['counters']['attempted_updates']-counter_before['attempted_updates']==64)
        check(f'{stage}:effective_count',after['counters']['effective_updates']-counter_before['effective_updates']==accepted_stage)
        check(f'{stage}:iteration',after['counters']['iteration']==counter_before['iteration']+1)
        check(f'{stage}:unchanged_config',after['config']==cfg.as_dict() and after['config_hash']==cfg.digest())
        for i,row in enumerate(rows):
            phase='value' if (counter_before['attempted_updates']+i)%3==2 else 'policy'
            check(f'{stage}:update{i}:PPV',row['phase']==phase)
            check(f'{stage}:update{i}:finite',np.isfinite(row['loss']) and np.isfinite(row['gradient_norm']))
            if row['accepted']:
                factors=dict(encoder=.1,policy=0.,value=1.) if phase=='value' else dict(encoder=.25,policy=1.,value=0.)
                check(f'{stage}:update{i}:LR',all(abs(row['optimizer_group_learning_rates'][k]-cfg.learning_rate*row['accepted_lr_scale']*v)<1e-15 for k,v in factors.items()))
        check(f'{stage}:own_previous_collector',meta['actor_model_hash']==seed['model_hash'])
        samples=read_selfplay_shard(inside(record['path']))
        proof=verify_collection(samples,meta,cfg)
        check(f'{stage}:collection_contract',proof['all_passed'])
        check(f'{stage}:256_fresh',len(meta['games'])==256)
        if stage>=3:check(f'{stage}:initial_replay_retired',all(not k.startswith('initial:') for k in report['actual_policy_replay_origin_sources']))
        games+=len(meta['games']);accepted+=accepted_stage
        stage_rows.append(dict(stage=stage,iteration=after['counters']['iteration'],model_hash=after['model_hash'],checkpoint=entry(checkpoint.relative_to(root)),report=entry(report_path.relative_to(root)),new_games=len(meta['games']),effective_updates=accepted_stage,collection_contract=proof))
        seed=after;counter_before=after['counters']
    arena_record=progress['evaluations'][f"block_{item['stage']:06d}_small"]
    check('arena_sha',sha(inside(arena_record['path']))==arena_record['sha256'])
    arena=read_json(inside(arena_record['path']))
    review=historical_screen(arena,old['evaluation_seed'],old['early_stop']['confidence'],old['initial_source_teacher_score']);review['route_confirmed']=False
    check('raw_recomputed_review',review==arena['review'])
    selected=select_families(read_json(inside(old['suite']['path']))['states'],16,old['evaluation_seed'])
    native=verify_native(arena,selected,lambda:None)
    check('raw_native_arena',native==arena['native_verification'])
    check('cpu_no_gpu_initialized',not torch.cuda.is_initialized())
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    evidence_dir=root/'reports/continuous_fixed_20261005';evidence_dir.mkdir(exist_ok=True)
    audit_path=evidence_dir/f'BOUNDARY_{stamp}.json'
    atomic_json(audit_path,dict(at=utc_now(),all_checks_passed=all(checks.values()),checks=checks,stages=stage_rows,new_games=games,effective_updates=accepted,latest=entry(item['checkpoint']),model_hash=item['model_hash'],native_arena=native,screen_review=review,previous_run_stopped_by_old_rule=True,new_authorization_overrides_performance_stop=True,no_new_training_or_evaluation_games_in_audit=True))
    stream=io.StringIO();tests=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_open_v4_continuous_fixed.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    cpu_path=evidence_dir/f'CPU_{stamp}.json'
    atomic_json(cpu_path,dict(at=utc_now(),all_passed=result.wasSuccessful() and result.testsRun==4 and not result.skipped,tests=result.testsRun,output=stream.getvalue(),cpu_no_gpu_initialized=not torch.cuda.is_initialized()))
    if not read_json(cpu_path)['all_passed'] or torch.cuda.is_initialized():raise ValueError('CPU validation failed')
    authorization_path=evidence_dir/f'AUTHORIZATION_{stamp}.json'
    atomic_json(authorization_path,dict(at=utc_now(),request='不用自动停止？我的意思是有波动是正常的，一直继续训练就好了，就算出现了退步，已经回退到之前确认有效的机制了，一直继续就行了。',continue_despite_strength_decline=True,allow_learning_hyperparameter_change=False,keep_64_updates_per256fresh=True,no_weight_reset=True,stop_for_user_or_operational_failure=True,promotion_gates_unchanged=True))
    plan=copy.deepcopy(old);directory='runs/open_junqi_v4_continuous_fixed_'+stamp
    plan_path=root/f'config/experiments/open_v4_continuous_fixed_{stamp}.json'
    base_path=root/f'config/experiments/open_v4_continuous_fixed_base_{stamp}.json'
    atomic_json(base_path,source['config'])
    current=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=source['counters']['iteration'],leaf_value_scale=source['leaf_value_scale'])
    plan.update(schema='junqi-continuous-fixed-mechanism-plan-1',route='continuous_restored_mechanism',registered_at=utc_now(),directory=directory,parent_config=str(base_path.relative_to(root)),parent_config_sha256=sha(base_path),source_checkpoint=current,source_counters=source['counters'],source_config_hash=source['config_hash'],maximum_stages=None,periodic_screen_stages=12,performance_stop_enabled=False,continuation_requires_promotion=False,collection_start=old['collection_start']+item['stage'],initial_source_teacher_score=arena['summaries'][2]['family_score']['mean'],registered_maximum_new_games=None,registered_maximum_updates=None)
    plan['participants']['parent']=current
    plan['initial_replay']=progress['collections']['learner'][-3:]
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP']
    plan['frozen_runs'].append(dict(directory=old['directory'],current_sha256=sha(old_run/'CURRENT.json')))
    plan['previous_experiment']=dict(plan=entry(previous_path.relative_to(root)),boundary_audit=entry(audit_path.relative_to(root)))
    plan['user_contract']=entry(authorization_path.relative_to(root));plan['boundary_audit']=entry(audit_path.relative_to(root));plan['cpu_verification']=entry(cpu_path.relative_to(root))
    plan.pop('trigger',None)
    plan['authorization']='User explicitly revokes strength-based stopping and requests continuous unchanged64update mechanism even during regression. Major learning changes still require separate explicit confirmation.'
    plan['interpretation']='One latest complete170 learner continues without weight/optimizer/scheduler/counter/RNG/Value-scale reset. Every256 fresh games uses preceding learner then64 PPV updates; all learning/search/data parameters match restored mechanism. Screens record fluctuations and never stop learning or prove promotion.'
    plan['source_role']='Latest complete restored-mechanism170 research learner; no historical weight replacement.'
    plan['frozen_reference_definition']='Parent is frozen actual source170. candidate is the sole evolving learner; control134/86/Champion26 are frozen references, not concurrent online controls.'
    plan['horizon']='Unbounded fresh online stages until user STOP or operational/integrity failure. Screens at6/12/24 then every12; no performance-based stop or promotion prerequisite for continuation.'
    plan['hypothesis']='None added; preserve restored mechanism and comply with user continuous-training instruction.'
    plan['early_stop']['explanation']='Historical thresholds are descriptive diagnostics only. No negative score or unconfirmed promotion stops learning. Source drift or integrity failure still needs attention.'
    plan['early_stop']['maximum_observations']=None
    for p in ['tools/run_open_v4_continuous_fixed.py','tools/open_v4_continuous_fixed_contract.py','tools/prepare_open_v4_continuous_fixed.py','tests/test_open_v4_continuous_fixed.py','start_open_v4_continuous_fixed.ps1','stop_open_v4_continuous_fixed.ps1']:
        plan['source'][p]=sha(inside(p))
    atomic_json(plan_path,plan);verify_single_prerequisite(root,plan)
    print(json.dumps(dict(plan=str(plan_path.relative_to(root)),directory=directory,source=current,boundary_audit=entry(audit_path.relative_to(root)),cpu=entry(cpu_path.relative_to(root)),no_weight_reset=True,performance_stop_enabled=False),ensure_ascii=False))


if __name__=='__main__':main()
