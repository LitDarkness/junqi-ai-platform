"""Register the authorized same-recipe424to600 continuation on CPU."""
from pathlib import Path
from collections import Counter
import copy,hashlib,json,unittest,shutil
ROOT=Path(__file__).resolve().parents[1]

def main():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_recipe50_to600_contract import validate_source_config,fork_stable_payload,verify_stable_seed,verify_single_prerequisite
    from open_v4_monitor_evidence import reconstruct_monitor
    from junqi_open_v4.arena import select_families
    torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def entry(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    master=read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old=read_json(ROOT/master['optimization']['config']);oldrun=ROOT/old['directory']
    progress=read_json(oldrun/'PROGRESS.json');item=progress['variants']['learner']
    checks={}
    def check(k,ok):
        checks[k]=bool(ok)
        if not ok:raise ValueError('600 continuation registration: '+k)
    check('latest424_completed',not master['paused_by_user'] and read_json(oldrun/'STATUS.json')['state']=='horizon_complete' and read_json(oldrun/'COMPLETION.json')['target_iteration']==424 and item['counters']['iteration']==424 and sha(ROOT/item['checkpoint'])==item['sha256'])
    prior_report=master['optimization']['completion_report']
    for name in ('report','data','review','monitor_review'):
        check('completed424_report:'+name,sha(ROOT/prior_report[name]['path'])==prior_report[name]['sha256'])
    check('completed424_review',read_json(ROOT/prior_report['review']['path'])['all_checks_passed'])
    for p in [ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock']+[ROOT/e['directory']/f'supervision/{n}.lock' for e in old['frozen_runs']+[{'directory':old['directory']}] for n in ('SUPERVISOR','TRAINING')]:check('unowned:'+str(p.relative_to(ROOT)),not lock_held(p))
    for p,h in old['source'].items():check('preserved_frozen_source:'+p,sha(ROOT/p)==h)
    for e in old['frozen_runs']:check('preserved_CURRENT:'+e['directory'],sha(ROOT/e['directory']/'CURRENT.json')==e['current_sha256'])
    source=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
    check('source_complete_model_Adam_schedule_RNG',all(k in source for k in ('model','optimizer','scheduler','counters','rng','config','leaf_value_scale')))
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    out=ROOT/f'reports/recipe50_to600_{stamp}';out.mkdir()
    directory=f'runs/open_junqi_v4_recipe50_to600_{stamp}'
    planpath=ROOT/f'config/experiments/open_v4_recipe50_to600_{stamp}.json'
    basepath=ROOT/f'config/experiments/open_v4_recipe50_to600_base_{stamp}.json'
    atomic_json(out/'PRIOR_MASTER.json',master);atomic_json(basepath,source['config'])
    cfg=RootExploreConfig.load(basepath);cfg.run_directory=directory+'/learner';cfg.initial_checkpoint=item['checkpoint'];cfg.validate()
    changed=validate_source_config(source,cfg)
    check('only_paths_changed',set(changed)=={'run_directory','initial_checkpoint'})
    migrated=fork_stable_payload(source,cfg,dict(source=item,target_iteration=600,same_recipe=True,no_weight_reset=True))
    statechecks=verify_stable_seed(source,migrated)
    check('exact_model_Adam_steps_schedule_counters_allRNG_Value_LR',all(statechecks.values()))
    check('source_optimizer_model_finite',all(torch.isfinite(v).all().item() for v in source['model'].values() if v.is_floating_point()) and all(torch.isfinite(v).all().item() for row in source['optimizer']['state'].values() for v in row.values() if isinstance(v,torch.Tensor) and v.is_floating_point()))
    suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'),pattern='test_open_v4_recipe50_to600.py')
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    check('continuation_state_route_horizon_protection_tests',result.wasSuccessful() and result.testsRun==6)
    tests=dict(tests_run=result.testsRun,failures=len(result.failures),errors=len(result.errors),all_passed=result.wasSuccessful())
    for e in progress['collections']['learner'][-3:]:check('initial3_same_recipe_TRAIN:'+e['path'],sha(ROOT/e['path'])==e['sha256'] and sha(ROOT/e['metadata'])==e['metadata_sha256'] and read_json(ROOT/e['metadata'])['layout_role']=='train' and read_json(ROOT/e['metadata'])['training_recipe']=={'selfplay':.5,'teacher_selfplay':0.,'vs_actual_082':.5})
    baseline_ref=progress['evaluations']['startup_baseline'];baseline=read_json(ROOT/baseline_ref['path'])
    check('original400_baseline_sha',sha(ROOT/baseline_ref['path'])==baseline_ref['sha256'])
    selected=select_families(read_json(ROOT/old['suite']['path'])['states'],16,old['evaluation_seed'])
    baseline_review=reconstruct_monitor(baseline,selected,old,ROOT)
    check('original400_baseline_native_stats',baseline_review['all_checks_passed'] and all(x['model_hash']==old['participants']['parent']['model_hash'] for x in baseline['checkpoints'].values()))
    atomic_json(out/'BASELINE400_REVIEW.json',baseline_review)
    stops={e['path']:e['sha256'] for e in old['historical_stops']}
    for p in (ROOT/'runs').rglob('STOP'):
        if p.is_file():stops[str(p.relative_to(ROOT))]=sha(p)
    per_stage=(ROOT/item['checkpoint']).stat().st_size+sum((ROOT/progress['collections']['learner'][-1][k]).stat().st_size for k in ('path','metadata'))+1024**2
    budget=dict(free_bytes=shutil.disk_usage(ROOT).free,estimated176_stage_bytes=per_stage*176,minimum_free_gib=old['minimum_free_gib'],formal_arena_growth_not_assumed_zero=True)
    check('estimated_training_disk_reserve',budget['free_bytes']-budget['estimated176_stage_bytes']>old['minimum_free_gib']*1024**3)
    atomic_json(out/'DISK_BUDGET.json',budget)
    boundary=out/'BOUNDARY_AUDIT.json'
    atomic_json(boundary,dict(all_checks_passed=all(checks.values()),checks=checks,source=item,source_state_preservation=statechecks,changed_config_fields=changed,training_teacher_id=0,old_run=old['directory'],old_current_sha256=sha(oldrun/'CURRENT.json'),initial_replay=progress['collections']['learner'][-3:],qualification_reference400=old['participants']['parent'],original400_baseline=baseline_ref,negative424_results_preserved=True))
    cpu=out/'CPU_VERIFICATION.json'
    atomic_json(cpu,dict(all_passed=all(checks.values()),state_checks=statechecks,tests=tests,baseline_native=baseline_review['native_verification'],zero_training_games=True,zero_updates=True,no_cuda_initialized=not torch.cuda.is_initialized()))
    auth=out/'AUTHORIZATION.json'
    atomic_json(auth,dict(at=utc_now(),request='这说明出现了一些遗忘，这是正常的？训练到 600 轮试试？',recipe={'selfplay':.5,'teacher_selfplay':0.,'vs_actual_082':.5},allow_other_learning_changes=False,no_weight_reset=True,continue_despite_strength_decline=True,source_iteration=424,target_iteration=600,scope='Extend same authorized50/50 recipe to600 from latest424 exact full learning state. Original64PPV/LR/search/labels/replay unchanged. Frozen400 qualification and negative evidence retained; source424 is not a replacement qualification baseline. Assess possible forgetting with actual084 monitoring; no causal or guaranteed recovery claim.'))
    current=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=424,leaf_value_scale=source['leaf_value_scale'])
    plan=copy.deepcopy(old)
    plan.update(schema='junqi-recipe50-424to600-plan-1',route='recipe50_selfplay_vs082_to600',registered_at=utc_now(),directory=directory,parent_config=str(basepath.relative_to(ROOT)),parent_config_sha256=sha(basepath),source_checkpoint=current,source_counters=source['counters'],source_config_hash=source['config_hash'],maximum_stages=176,target_iteration=600,screen_boundaries=[6]+list(range(12,177,12))+[176],monitor084_boundaries=list(range(24,177,24))+[176],collection_start=read_json(ROOT/progress['collections']['learner'][-1]['metadata'])['iteration']+1,initial_replay=progress['collections']['learner'][-3:],registered_maximum_new_games=45056,registered_maximum_updates=11264,qualification_reference400=copy.deepcopy(old['participants']['parent']),frozen_baseline=entry(ROOT/baseline_ref['path']),user_contract=entry(auth),cpu_verification=entry(cpu),boundary_audit=entry(boundary),training_teacher_ids=[0])
    plan['participants']['champion']=copy.deepcopy(progress['champion'])
    plan['initial_source_teacher_score']=next(r['family_score']['mean'] for r in baseline['summaries'] if r['candidate']=='parent' and r['opponent']=='teacher0')
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP']
    plan['historical_stops']=[dict(path=p,sha256=h) for p,h in sorted(stops.items())];plan['historical_stop_markers']=list(stops)
    plan['frozen_runs'].append(dict(directory=old['directory'],current_sha256=sha(oldrun/'CURRENT.json')))
    plan['previous_experiment']=dict(plan=entry(ROOT/master['optimization']['config']),completed424_report=prior_report)
    plan.update(authorization='Latest human explicitly authorizes extending the same50/50 recipe from complete424 to600. No new learning hyperparameters.',source_role='Latest complete424 learning state. Frozen400/Champion230/candidate134/control86 continue protecting adoption.',horizon='176fresh stages424to600, 45056fresh/11264updates. Registered6/12/every12/end research082 screens, every24/end actual084 monitoring. No negative-score stop; endpoint review, no automatic extension.',frozen_reference_definition='parent/repeat remain400; latest424 is learning source, not a new qualification baseline. Champion230/control134/guard86 remain fixed references.',hypothesis='Longer same-recipe learning to600 may recover or improve performance; measure rather than assume. Increased selfplay and teacher selection were jointly changed at400; continuation is not a causal forgetting experiment.',baseline_registration='Import unchanged original400/082/084344-game frozen baseline and independent native/statistical review. No new baseline games or learning; frozen source400 score stays fixed.',interpretation='Same50/50 recipe continues from424; possible084 forgetting remains an observation, not proven cause or guaranteed recovery. Actual084 periodic monitor and frozen400 qualification remain; final layouts unused.',efficiency='Learning/search/labels/LR/update ratio unchanged. No new performance algorithm. All timing and sources recorded separately.')
    plan['reporting_contract'].update(cumulative='Old170to30233792/8448; true302to36215360/3840; engineered362to4009728/2432; priorrecipe400to4246144/1536. New424to600 independently planned45056/11264. Imported baseline, frozen arenas, CPU audit0training games/updates.',initial_replay='Latest424 same-recipe TRAIN22/23/24 preserved; no084 fresh labels or old400 recipe remain. Inherited raw shards naturally exit after3new stages; never relabel.')
    names=['tools/build_open_v4_recipe50_to600.py','tools/open_v4_recipe50_to600_contract.py','tools/run_open_v4_recipe50_to600.py','tools/review_open_v4_recipe50_to600.py','tools/review_open_v4_recipe50_to600_full.py','tools/open_v4_monitor_evidence.py','tools/prepare_open_v4_recipe50_to600.py','tools/activate_open_v4_recipe50_to600.py','tests/test_open_v4_recipe50_to600.py','start_open_v4_recipe50_to600.ps1','stop_open_v4_recipe50_to600.ps1']
    for name in names:plan['source'][name]=sha(ROOT/name)
    atomic_json(planpath,plan);verify_single_prerequisite(ROOT,plan)
    registration=dict(plan=str(planpath.relative_to(ROOT)).replace('\\','/'),directory=directory,report=str(out.relative_to(ROOT)),source=current,qualification_reference400=plan['qualification_reference400'],planned_new_games=45056,planned_updates=11264,checks=len(checks),target_iteration=600,tests=tests)
    atomic_json(out/'REGISTRATION.json',registration);atomic_json(ROOT/'reports/recipe50_to600_latest_registration.json',registration)
    print(json.dumps(registration,ensure_ascii=False))

if __name__=='__main__':main()
