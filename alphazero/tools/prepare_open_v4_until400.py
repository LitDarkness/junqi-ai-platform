"""Register approved302to400 continuation; preserve stopped historical run."""
from pathlib import Path
import copy, hashlib, json

ROOT=Path(__file__).resolve().parents[1]
def main():
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.isolated_teachers import TeacherLeague
    from junqi_fixed_rl.layout import FixedLayout,FixedLayoutPairBackend
    from open_v4_mechanism_reversion_contract import verify_stable_seed,validate_source_config
    from fork_open_v4_qualified import fork_payload
    from open_v4_until400_contract import verify_single_prerequisite
    torch.set_num_threads(2)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def entry(p):
        p=Path(p);p=p if p.is_absolute() else ROOT/p
        return dict(path=str(p.relative_to(ROOT)),sha256=sha(p))
    oldpath=ROOT/'config/experiments/open_v4_continuous_fixed_20261005_121923.json';old=read_json(oldpath);oldrun=ROOT/old['directory'];progress=read_json(oldrun/'PROGRESS.json');status=read_json(oldrun/'STATUS.json')
    checks={}
    def check(name,value):
        checks[name]=bool(value)
        if not value:raise ValueError('registration check failed: '+name)
    check('old_user_stop_preserved',status['state']=='stopped_by_user' and (oldrun/'STOP').exists())
    for p in [ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock',ROOT/'runs/open_junqi_v4_repair_pilot/supervision/SUPERVISOR.lock',ROOT/'runs/open_junqi_v4_repair_pilot/supervision/TRAINING.lock',oldrun/'supervision/SUPERVISOR.lock',oldrun/'supervision/TRAINING.lock']:
        check('unowned:'+str(p.relative_to(ROOT)),not lock_held(p))
    for p,h in old['source'].items():check('historical_source:'+p,sha(ROOT/p)==h)
    item=progress['variants']['learner'];source=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
    check('complete_source302',source['counters']['iteration']==302 and item['model_hash']==source['model_hash'] and sha(ROOT/item['checkpoint'])==item['sha256'])
    for field in ('model','optimizer','scheduler','counters','rng','config','leaf_value_scale'):check('full_state:'+field,field in source)
    for e in progress['collections']['learner'][-3:]:
        check('source_shard:'+e['path'],sha(ROOT/e['path'])==e['sha256']);check('source_meta:'+e['metadata'],sha(ROOT/e['metadata'])==e['metadata_sha256'] and read_json(ROOT/e['metadata'])['layout_role']=='train')
    stamp=utc_now().replace('-','').replace(':','').replace('T','_')[:15];report=ROOT/f'reports/until400_{stamp}';report.mkdir()
    directory=f'runs/open_junqi_v4_until400_{stamp}';planpath=ROOT/f'config/experiments/open_v4_until400_{stamp}.json';basepath=ROOT/f'config/experiments/open_v4_until400_base_{stamp}.json'
    atomic_json(report/'PRIOR_MASTER.json',read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json'))
    atomic_json(basepath,source['config']);cfg=RootExploreConfig.load(basepath);cfg.run_directory=directory+'/learner';cfg.initial_checkpoint=item['checkpoint'];validate_source_config(source,cfg)
    migrated=fork_payload(source,cfg,dict(approved_teacher_identity_repair=True,no_weight_reset=True,target_iteration=400))
    statechecks=verify_stable_seed(source,migrated);check('all_full_seed_checks',all(statechecks.values()))
    model=PolicyValueNet(cfg.model).cpu().eval();model.load_state_dict(source['model']);check('source_model_tensor_digest',model_hash(model)==item['model_hash']);del model
    diagnostic=ROOT/'reports/teacher_saturation_20261006_112322';rootprotocol=read_json(diagnostic/'TIMING_PROTOCOL.json')['roots'];entries=read_json(diagnostic/'FIT_PROTOCOL.json')['corpora']['latest302_training'];meta=read_json(ROOT/entries[-1]['metadata']);states=[]
    for r in rootprotocol:
        g=meta['games'][r['game_index']];p=g['provenance'];red=FixedLayout(p['red_layout_id'],p['red_layout_id'],tuple(p['red_layout']),g['family']);black=FixedLayout(p['black_layout_id'],p['black_layout_id'],tuple(p['black_layout']),g['family']);s=FixedLayoutPairBackend(red,black).new_state(p['setup_seed'])
        for a in g['trace'][:r['ply']]:check('saved_root_legal:'+str(len(states))+':'+str(a),a in s.legal_actions());s.apply(int(a))
        states.append(s)
    expected0=[x['original']['082'] for x in read_json(diagnostic/'LOADER_082084.json')['rows']];expected1=[x['original']['084'] for x in read_json(diagnostic/'LOADER_084082.json')['rows']];actions=[]
    with TeacherLeague(ROOT,cfg.teacher_candidates,12) as teachers:
        receipts=teachers.identity_report();check('all12actual_workers_verified',receipts['all_workers_verified'] and len(receipts['verified_workers'])==12 and receipts['worker_counts']==[6,6])
        for i,s in enumerate(states):
            result=teachers.actions([s,s],[0,1]);actions.append(result);check('082_original_recommendation:'+str(i),result[0]==expected0[i]);check('084_original_recommendation:'+str(i),result[1]==expected1[i])
        teacher_provenance=teachers.provenance
    check('two_teacher_difference',sum(a!=b for a,b in actions)==6)
    check('cpu_no_cuda_initialized',not torch.cuda.is_initialized())
    cpu=report/'CPU_VERIFICATION.json';atomic_json(cpu,dict(all_passed=True,checks=statechecks,teacher_identity=receipts,saved_roots=24,queries=48,recommendations_match_individual_originals=True,workers_closed=True,no_new_games_or_updates=True))
    stops={e['path']:e['sha256'] for e in old['historical_stops']}
    for p in ROOT.joinpath('runs').rglob('STOP'):
        if p.is_file():stops[str(p.relative_to(ROOT))]=sha(p)
    for p,h in stops.items():check('STOP:'+p,sha(ROOT/p)==h)
    boundary=report/'BOUNDARY_AUDIT.json';atomic_json(boundary,dict(all_checks_passed=True,checks=checks,source=entry(item['checkpoint']),source_counters=source['counters'],source_model_hash=item['model_hash'],previous_run=old['directory'],previous_run_frozen=True,old_current_sha256=sha(oldrun/'CURRENT.json'),source_state_preservation=statechecks,initial_replay=progress['collections']['learner'][-3:],prior_dual_teacher_evidence_needs_reverification=True))
    auth=report/'AUTHORIZATION.json';atomic_json(auth,dict(at=utc_now(),request='行，继续训练吧，如果直到400轮依旧平台期，我们再考虑换一些方法？',continue_to_iteration400=True,continue_despite_strength_decline=True,teacher_identity_repair_authorized=True,allow_learning_hyperparameter_change=False,keep64_updates_per256fresh=True,no_weight_reset=True,review_at400=True,no_automatic_method_change=True,interpretation='Approval follows concrete report proposing teacher isolation and same-mechanism continuation. Real teacher084 replaces alias082 on its registered requests; label provider changes explicitly disclosed.'))
    current=dict(path=item['checkpoint'],sha256=item['sha256'],model_hash=item['model_hash'],iteration=302,leaf_value_scale=source['leaf_value_scale'])
    plan=copy.deepcopy(old);plan.update(schema='junqi-until400-isolated-teachers-plan-1',route='restored_mechanism_isolated_teachers_to400',registered_at=utc_now(),directory=directory,parent_config=str(basepath.relative_to(ROOT)),parent_config_sha256=sha(basepath),source_checkpoint=current,source_counters=source['counters'],source_config_hash=source['config_hash'],target_iteration=400,maximum_stages=98,screen_boundaries=[6,12,24,98],collection_start=old['collection_start']+item['stage'],initial_replay=progress['collections']['learner'][-3:],registered_maximum_new_games=25088,registered_maximum_updates=6272,teacher_identity_contract='one_original_native_teacher_per_worker_actual_sha_verified',startup_baseline_games=344,teacher_repair_effect='Actual supervision becomes original082+084 instead of duplicate082, original labels remain hard teacher targets; old shards unmodified.',baseline_registration='Frozen302 parent/repeat versus each real teacher before fresh training. Newly measured source reference fixed for all subsequent screens.',final_true084_monitor=True)
    plan['participants']['parent']=current;plan['participants']['champion']=copy.deepcopy(progress['champion']);plan['participants']['champion']['inherited_dual_teacher_gate_needs_reverification']=True
    plan['initial_source_teacher_score']=.5885416666666666
    plan['stop_paths']=[directory+'/STOP',directory+'/supervision/STOP'];plan['historical_stops']=[dict(path=p,sha256=h) for p,h in sorted(stops.items())];plan['historical_stop_markers']=list(stops)
    if not any(e['directory']==old['directory'] for e in plan['frozen_runs']):plan['frozen_runs'].append(dict(directory=old['directory'],current_sha256=sha(oldrun/'CURRENT.json')))
    plan['teachers']=teacher_provenance;plan['user_contract']=entry(auth);plan['boundary_audit']=entry(boundary);plan['cpu_verification']=entry(cpu)
    plan['authorization']='Latest human approves continuing single learner from current302 to review400 after disclosed teacher isolation repair; keep all learning/search/data settings, no autonomous new methods.'
    plan['horizon']='98fresh stages302to400, no strength-based early stop. Review at400, no automatic method switch or unregistered extension.'
    plan['interpretation']='One own-preceding-model online learner,256new terminal games then64PPV. Full model/Adam/scheduler/counters/config/RNG/Value scale inherited from302; paths and effective teacher identity corrected. Frozen230/302/134/86 references are not concurrent learners or strict causal controls. Full adoption gates require distinct actual082/084 receipts.'
    plan['source_role']='Current complete research302, not old230/170 weights.'
    plan['reporting_contract']=dict(online='Own preceding complete learner generates fresh terminal games and learns; inherited replay is old data; frozen arenas add0training games/updates.',teacher_evidence='Prior two-teacher claims were alias082; do not reuse them as true084 proof. Corrected genuine teachers are related, not independent final evidence.',cumulative='Old170to30233792games/8448updates separate from new302to40025088games/6272plannedupdates.',initial_replay='Original last3rawshards retain historical provider metadata and labels; never relabel as true084.')
    for p in ('tools/run_open_v4_until400.py','tools/open_v4_until400_contract.py','tools/prepare_open_v4_until400.py','python/junqi_open_v4/isolated_teachers.py','start_open_v4_until400.ps1','stop_open_v4_until400.ps1'):
        plan['source'][p]=sha(ROOT/p)
    atomic_json(planpath,plan);verify_single_prerequisite(ROOT,plan)
    result=dict(plan=str(planpath.relative_to(ROOT)),directory=directory,report=str(report.relative_to(ROOT)),source=current,target_iteration=400,planned_new_games=25088,planned_updates=6272,cpu_verification=entry(cpu),boundary_audit=entry(boundary),historical_stops=len(stops),checks=len(checks))
    atomic_json(report/'REGISTRATION.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
