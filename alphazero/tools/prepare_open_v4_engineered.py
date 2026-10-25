"""Register performance-only continuation from complete362; preserve all history."""
from pathlib import Path
import copy,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_engineered_contract import fork_stable_payload,verify_stable_seed,validate_source_config,verify_single_prerequisite
    torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def entry(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    master=read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old=read_json(ROOT/master['optimization']['config']);oldrun=ROOT/old['directory']
    progress=read_json(oldrun/'PROGRESS.json');item=progress['variants']['learner']
    checks={}
    def check(name,ok):
        checks[name]=bool(ok)
        if not ok:raise ValueError('engineering registration check: '+name)
    check('old_paused_and_STOP_preserved',master['paused_by_user'] and read_json(oldrun/'STATUS.json')['state']=='stopped_by_user' and all((ROOT/p).exists() for p in old['stop_paths']))
    check('complete_source362',item['counters']['iteration']==362 and sha(ROOT/item['checkpoint'])==item['sha256'])
    for p in [ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',oldrun/'supervision/SUPERVISOR.lock',oldrun/'supervision/TRAINING.lock']:
        check('unowned:'+str(p),not lock_held(p))
    for p,h in old['source'].items():check('frozen_source:'+p,sha(ROOT/p)==h)
    for e in old['frozen_runs']:check('frozen_CURRENT:'+e['directory'],sha(ROOT/e['directory']/'CURRENT.json')==e['current_sha256'])
    source=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15]
    out=ROOT/f'reports/engineered_{stamp}';out.mkdir()
    directory=f'runs/open_junqi_v4_engineered_{stamp}'
    planpath=ROOT/f'config/experiments/open_v4_engineered_{stamp}.json'
    basepath=ROOT/f'config/experiments/open_v4_engineered_base_{stamp}.json'
    atomic_json(out/'PRIOR_MASTER.json',master);atomic_json(basepath,source['config'])
    cfg=RootExploreConfig.load(basepath);cfg.run_directory=directory+'/learner';cfg.initial_checkpoint=item['checkpoint']
    validate_source_config(source,cfg)
    migrated=fork_stable_payload(source,cfg,dict(engineering_only=True,source=item,target_iteration=400,no_weight_reset=True))
    statechecks=verify_stable_seed(source,migrated)
    check('full_model_Adam_scheduler_counters_allRNG_Value_preserved',all(statechecks.values()))
    bench=ROOT/'reports/search_engineering_20261006/BENCHMARK_RESULTS.json'
    check('benchmark_all_exact',read_json(bench)['all_exact'])
    check('graph_model_revision_invalidation',read_json(bench)['revision_invalidates_graphs'])
    for e in progress['collections']['learner'][-3:]:
        check('last3TRAIN:'+e['path'],sha(ROOT/e['path'])==e['sha256'] and sha(ROOT/e['metadata'])==e['metadata_sha256'] and read_json(ROOT/e['metadata'])['layout_role']=='train')
    stops={e['path']:e['sha256'] for e in old['historical_stops']}
    for p in ROOT.joinpath('runs').rglob('STOP'):
        if p.is_file():stops[str(p.relative_to(ROOT))]=sha(p)
    boundary=out/'BOUNDARY_AUDIT.json';atomic_json(boundary,dict(all_checks_passed=all(checks.values()),checks=checks,source=item,source_state_preservation=statechecks,old_run=old['directory'],old_current_sha256=sha(oldrun/'CURRENT.json'),initial_replay=progress['collections']['learner'][-3:]))
    cpu=out/'CPU_VERIFICATION.json';tests=read_json(ROOT/'reports/search_engineering_20261006/TEST_RESULTS.json')
    check('rules_and_resumable_arena_tests_passed',tests['all_passed'])
    atomic_json(cpu,dict(all_passed=all(statechecks.values()) and tests['all_passed'],state_checks=statechecks,tests=tests,zero_training_games=True,zero_updates=True,no_cuda_initialized=not torch.cuda.is_initialized()))
    auth=out/'AUTHORIZATION.json';atomic_json(auth,dict(at=utc_now(),request='行，放手去做吧',performance_engineering_authorized=True,continue_to_iteration400=True,continue_despite_strength_decline=True,allow_learning_hyperparameter_change=False,keep64_updates_per256fresh=True,no_weight_reset=True,review_at400=True,explicit_scope='Search/inference and resumable evaluation engineering after exact-output/throughput validation, then resume complete362 to400. No change to learning rates, update ratio, budgets, mixture or labels. First save363 online commit, then complete pending frozen362 full gate; current learner never reset.'))
    current=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=362,leaf_value_scale=source['leaf_value_scale'])
    plan=copy.deepcopy(old);plan.update(schema='junqi-engineered-362to400-plan-1',route='engineered_fixed_mechanism_to400',registered_at=utc_now(),directory=directory,parent_config=str(basepath.relative_to(ROOT)),parent_config_sha256=sha(basepath),source_checkpoint=current,source_counters=source['counters'],source_config_hash=source['config_hash'],maximum_stages=38,screen_boundaries=[12,24,36,38],collection_start=old['collection_start']+60,initial_replay=progress['collections']['learner'][-3:],registered_maximum_new_games=9728,registered_maximum_updates=2432,startup_baseline_games=0,frozen_baseline=copy.deepcopy(progress['evaluations']['startup_baseline']),pending_screen=copy.deepcopy(progress['evaluations']['block_000060_small']),benchmark=entry(bench))
    plan['participants']['champion']=copy.deepcopy(progress['champion'])
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP']
    plan['historical_stops']=[dict(path=p,sha256=h) for p,h in sorted(stops.items())];plan['historical_stop_markers']=list(stops)
    plan['frozen_runs'].append(dict(directory=old['directory'],current_sha256=sha(oldrun/'CURRENT.json')))
    plan.update(user_contract=entry(auth),cpu_verification=entry(cpu),boundary_audit=entry(boundary))
    plan['authorization']='User approves concrete performance-only engineering and unchanged362to400 continuation. No new hyperparameters or mixture experiment.'
    plan['source_role']='Latest complete362; original302 remains the frozen qualification reference, not weight source.'
    plan['horizon']='38fresh stages362to400; first363commit then pending frozen362full gate. Original absolute374/386/398/400 screens. No strength stop, method switch or extension.'
    plan['interpretation']='Single current learner; exact-shape BF16 graph inference validated against unchanged eager traversal, journaled arenas preserve original game/group/statistical order. Original64PPV and original recipe retained. Frozen302/230/134/86 are protections, not simultaneous online controls.'
    plan['reporting_contract']['cumulative']='Old170to30233792games/8448updates; true-teacher302to36215360/3840; new362to4009728/2432planned. Frozen362full0fresh/0learning.'
    plan['reporting_contract']['initial_replay']='Latest362 genuine-teacher TRAIN shards58/59/60, retained raw/labels; exit after three fresh stages.'
    plan['engineering_contract']=dict(evaluation_physical_batch=32,precision='BF16',shape_bucketing=False,search_cache=False,tree_reuse=False,graph_slots=8,training_search='unchanged_original_dynamic_RootExploreSearch',training_graph_candidate_adopted=False,collector_graph_benchmark=entry(ROOT/'reports/search_engineering_20261006/COLLECTOR_SHAPES_RESULTS.json'),arena_journal_rounds=20,bounded_replay='At most20uncommitted rounds after explicit reviewed resume; completed chunks immutable.',full_gates_unchanged=True)
    for p in ('tools/run_open_v4_engineered.py','tools/review_open_v4_engineered.py','tools/review_open_v4_engineered_full.py','tools/open_v4_engineered_contract.py','tools/prepare_open_v4_engineered.py','tools/build_engineered_runner.py','tools/build_resumable_arena_module.py','python/junqi_search_v3/graph_inference.py','python/junqi_open_v4/engineered_games.py','python/junqi_open_v4/resumable_arena.py','python/junqi_open_v4/arena_journal.py','tests/test_open_v4_engineering.py','start_open_v4_engineered.ps1','stop_open_v4_engineered.ps1'):
        plan['source'][p]=sha(ROOT/p)
    atomic_json(planpath,plan);verify_single_prerequisite(ROOT,plan)
    result=dict(plan=str(planpath.relative_to(ROOT)).replace('\\','/'),directory=directory,report=str(out.relative_to(ROOT)),source=current,planned_new_games=9728,planned_updates=2432,checks=len(checks))
    atomic_json(out/'REGISTRATION.json',result);atomic_json(ROOT/'reports/search_engineering_20261006/REGISTRATION.json',result)
    print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
