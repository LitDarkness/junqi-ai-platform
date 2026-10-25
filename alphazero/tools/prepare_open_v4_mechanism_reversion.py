"""Boundary preparation: retain current weights; return to the original mechanism."""
from pathlib import Path
import argparse
import copy
import hashlib
import io
import json
import os
import subprocess
import unittest


def main():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_mechanism_reversion_contract import verify_single_prerequisite
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser();parser.add_argument('--contract',required=True);args=parser.parse_args()
    contract_path=Path(args.contract);contract=read_json(root/contract_path);queue=root/contract['queue_directory'];old_run=root/contract['current_run']
    master_path=root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json';master=read_json(master_path)
    if master.get('paused_by_user') or (queue/'STOP').exists():raise ValueError('user pause overrides')
    if (queue/'PREPARED.json').exists():raise ValueError('one-time preparation already recorded')
    if lock_held(old_run/'supervision/TRAINING.lock') or lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):raise ValueError('old GPU owner must exit first')
    status=read_json(old_run/'STATUS.json')
    if status['state'] not in ('stopped_by_user','review_due','segment_complete'):raise ValueError('safe natural exit required')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    def entry(path):
        p=Path(path);return dict(path=str(p),sha256=sha(root/p))
    old=read_json(root/contract['current_config']);progress=read_json(old_run/'PROGRESS.json');item=progress['variants']['learner']
    if sha(root/item['checkpoint'])!=item['sha256']:raise ValueError('latest complete source changed')
    source=torch.load(root/item['checkpoint'],map_location='cpu',weights_only=False)
    trigger=read_json(queue/'TRIGGER.json')
    if trigger['paired_delta']>=0:raise ValueError('negative condition not met')
    if item['stage']!=trigger['boundary_stage']:raise ValueError('unexpected post-boundary learning; preserve and inspect, no automatic weight replacement')
    stream=io.StringIO();tests=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_open_v4_mechanism_reversion.py')
    result=unittest.TextTestRunner(stream=stream,verbosity=2).run(tests)
    cpu_path=Path(contract['queue_directory'])/'CPU_VALIDATION.json'
    atomic_json(root/cpu_path,dict(at=utc_now(),all_passed=result.wasSuccessful() and result.testsRun==3 and not result.skipped,
        tests=result.testsRun,output=stream.getvalue(),cpu_no_gpu_initialized=not torch.cuda.is_initialized()))
    if not result.wasSuccessful() or torch.cuda.is_initialized():raise ValueError('CPU reversion validation failed')
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    directory='runs/open_junqi_v4_mechanism_reversion_'+stamp
    plan_path=Path('config/experiments/open_v4_mechanism_reversion_'+stamp+'.json');base_path=Path('config/experiments/open_v4_mechanism_reversion_base_'+stamp+'.json')
    base=copy.deepcopy(source['config']);base.pop('policy_step_lr_multiplier');base['updates_per_iteration']=64;atomic_json(root/base_path,base)
    plan=copy.deepcopy(old);plan.update(schema='junqi-mechanism-reversion-plan-1',route='restore_pre_fourfold_mechanism',registered_at=utc_now(),directory=directory,
        parent_config=str(base_path),parent_config_sha256=sha(root/base_path),policy_step_lr_multiplier=1.,updates_per_stage=64,
        registered_first_segment_updates=768,registered_maximum_updates=1536,source_counters=source['counters'],source_config_hash=source['config_hash'],
        collection_start=old['collection_start']+item['stage'],no_weight_reset=True,initial_source_teacher_score=trigger['candidate_teacher_score']['mean'])
    plan['source_checkpoint']=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=source['counters']['iteration'],leaf_value_scale=source['leaf_value_scale'])
    plan['participants']['parent']=plan['source_checkpoint']
    plan['initial_replay']=[r for r in progress['collections']['learner'] if r['stage']<=item['stage']][-3:]
    if len(plan['initial_replay'])!=3:raise ValueError('last three current complete training shards required')
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP']
    plan['historical_stops'].append(entry((old_run/'STOP').relative_to(root)))
    plan['historical_stop_markers']=[e['path'] for e in plan['historical_stops']]
    plan['frozen_runs'].append(dict(directory=contract['current_run'],current_sha256=sha(old_run/'CURRENT.json')))
    plan['cpu_verification']=entry(cpu_path);plan['user_contract']=entry(contract_path.relative_to(root) if contract_path.is_absolute() else contract_path)
    plan['trigger']=entry((queue/'TRIGGER.json').relative_to(root))
    plan['previous_experiment']=dict(plan=entry(contract['current_config']),trigger=plan['trigger'])
    plan['authorization']='User explicitly requests conditional mechanism rollback, not historical weight replacement. Preserve current complete learner; no additional experimental factors.'
    plan['source_role']='Current complete learner at negative boundary; no historical checkpoint substituted.'
    plan['frozen_reference_definition']='Parent is the actual current learner at reversion, not restored158weights. Archived134/86/Champion26 are frozen protection only, no concurrent control.'
    plan['interpretation']='Restores original64 PPV updates per256fresh games and original phase LRs. Actual current model/Adam moments/step/scheduler/counters/all RNG/Value scale preserved. Telemetry retained. This is user-directed mechanism reversion, not a new hypothesis or a weight reset.'
    plan['hypothesis']='None added: revert the fourfold-update mechanism and subsequent Policy-rate experiment at the user-defined negative boundary.'
    plan['horizon']='Initial12 new stages,3072fresh games/768planned updates,6/12screen.Only original full64 confirmation permits next12(max24). Any negative external paired point score stops this restored run for reporting; no new automatic experimental factor.'
    plan['efficiency']='Return to64updates; retain actual sampling-exposure and elapsed-time observability. No added search/data/LR method, no claimed strength guarantee.'
    plan['early_stop']['explanation']='User constraint: any negative external paired point score stops; CI still reports uncertainty, no significance-delay rule.'
    for p,h in old['source'].items():
        if sha(root/p)!=h:raise ValueError('frozen old execution dependency changed: '+p)
    ready=read_json(queue/'READY.json')
    for p,h in ready['sources'].items():
        if sha(root/p)!=h:raise ValueError('reversion dependency changed')
        plan['source'][p]=h
    atomic_json(root/plan_path,plan);verify_single_prerequisite(root,plan)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONUTF8='1')
    preflight=subprocess.run([ready['python'],str(root/'tools/run_open_v4_mechanism_reversion.py'),'--plan',str(plan_path),'--check-only'],cwd=root,env=env,text=True,capture_output=True)
    atomic_json(queue/'PRELAUNCH.json',dict(at=utc_now(),returncode=preflight.returncode,stdout=preflight.stdout,stderr=preflight.stderr,plan=entry(plan_path)))
    if preflight.returncode:raise RuntimeError('complete-state restoration preflight failed: '+preflight.stderr)
    master=read_json(master_path)
    if master.get('paused_by_user') or (queue/'STOP').exists():raise ValueError('later user stop overrides preparation')
    master['before_mechanism_reversion']=master['optimization']
    master['optimization']=dict(type=plan['route'],state='registered',config=str(plan_path).replace('\\','/'),config_sha256=sha(root/plan_path),
        run_directory=directory,status_tool='tools/run_open_v4_mechanism_reversion.py',launcher='start_open_v4_mechanism_reversion.ps1',training_active=False,
        training_variants=['learner'],source_iteration=source['counters']['iteration'],source_model_hash=source['model_hash'],no_weight_reset=True,
        new_training_games=0,effective_updates=0,updates_per256fresh=64,original_policy_rate_restored=True)
    master['automatic_continuation_enabled']=True
    master['user_mechanism_reversion'].update(state='prepared_no_weight_reset',new_plan=str(plan_path).replace('\\','/'),trigger=plan['trigger'])
    master['next_action']='CPU boundary controller has prepared exact mechanism reversion and will spawn once. Current source weights/moments/counters/RNG preserved. One restored64update learner; no speculative factors. Hold all old STOP/CURRENT/sources. Any restored-route external decline stops for report; later user stop overrides. At12 only original full64 gain allows next12.'
    for p in [master_path,root/'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json']:atomic_json(p,master)
    atomic_json(queue/'PREPARED.json',dict(at=utc_now(),plan=str(plan_path),directory=directory,source=plan['source_checkpoint'],model_not_replaced=True,updates_per256fresh=64,
        original_phase_rates={'policy':{'encoder':1.25e-5,'head':5e-5},'value':{'encoder':5e-6,'head':5e-5}},source_stage=item['stage']))
    print(json.dumps(dict(plan=str(plan_path),model_hash=source['model_hash'],iteration=source['counters']['iteration'],weight_reset=False,updates_per256fresh=64)))


if __name__=='__main__':main()
