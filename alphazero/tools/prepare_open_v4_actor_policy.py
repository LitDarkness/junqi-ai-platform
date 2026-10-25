"""Register an independent86-to110 online labeling experiment."""
from pathlib import Path


def main():
    import copy,json,torch
    from datetime import datetime,timezone
    from open_v4_runtime import read_json,atomic_json,utc_now
    from open_v4_queue_contract import sha
    from junqi_open_v4.queued_config import QueuedConfig
    from open_v4_actor_policy_contract import check_factor_parameters,verify_actor_policy_prerequisite
    root=Path(__file__).resolve().parents[1]
    supervision=read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old_path=supervision['optimization']['config'];old=read_json(root/old_path)
    if old['route']!='late_noise_factor' or read_json(root/old['directory']/'STATUS.json')['state']!='review_due':
        raise ValueError('complete and review preceding factor first')
    progress=read_json(root/old['directory']/'PROGRESS.json');item=progress['variants']['control']
    payload=torch.load(root/item['checkpoint'],map_location='cpu',weights_only=False)
    source={'path':item['checkpoint'],'sha256':item['sha256'],'model_hash':item['model_hash'],
            'iteration':payload['counters']['iteration'],'leaf_value_scale':payload['leaf_value_scale']}
    if source['iteration']!=86 or sha(root/source['path'])!=source['sha256']:raise ValueError('latest full source86 required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');directory=f'runs/open_junqi_v4_actor_policy_{stamp}'
    base_path=f'config/experiments/open_v4_actor_policy_base_{stamp}.json'
    atomic_json(root/base_path,payload['config']);QueuedConfig.load(root/base_path)
    def entry(path):return {'path':str(Path(path)),'sha256':sha(root/path)}
    review_path=supervision['optimization']['final_review']
    audit_path='reports/current_training_audit_20261004/ACTOR_POLICY_BOUNDARY_20261004_183200.json'
    names=set(old['source'])|{'tools/prepare_open_v4_actor_policy.py','tools/run_open_v4_actor_policy.py',
        'tools/open_v4_actor_policy_contract.py','tools/audit_open_v4_actor_policy_boundary.py',
        'python/junqi_open_v4/actor_policy_games.py','tests/test_open_v4_actor_policy.py','start_open_v4_actor_policy.ps1'}
    frozen=[{'directory':str(p.parent.relative_to(root)),'current_sha256':sha(p)} for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    markers=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))]
    control=dict(old['control'],actor_policy_target='teacher',root_noise_until_ply=None)
    candidate=dict(control,actor_policy_target='search')
    plan={'schema':'junqi-actor-owned-policy-plan-1','registered_at':utc_now(),'directory':directory,
        'route':'actor_owned_policy_factor','training_variants':['control','candidate'],
        'parent_config':base_path,'parent_config_sha256':sha(root/base_path),'source_checkpoint':source,
        'source_counters':payload['counters'],'source_config_hash':payload['config_hash'],
        'participants':{'parent':source,'champion':old['participants']['champion']},
        'suite':old['suite'],'teachers':old['teachers'],'layout_manifest_hash':old['layout_manifest_hash'],
        'initial_replay':progress['collections']['control'][-3:],'frozen_runs':frozen,
        'historical_stops':markers,'historical_stop_markers':[m['path'] for m in markers],
        'source':{name:sha(root/name) for name in sorted(names)},
        'stop_paths':[directory+'/STOP',directory+'/supervision/STOP']+old['stop_paths'],
        'previous_experiment':{'plan':entry(old_path),'final_review':entry(review_path)},'boundary_audit':entry(audit_path),
        'evaluation_seed':old['evaluation_seed'],'screen_families':16,'full_families':64,'evaluation_budget':256,
        'evaluation_inference_batch_size':32,'collection_start':800,'stages_per_block':12,'maximum_stages':24,
        'games_per_stage':256,'updates_per_stage':64,'replay_stages':3,'control':control,'candidate':candidate,
        'authorization':'Latest user authorizes continual fresh training, necessary diagnosis/optimization and automatic direct continuation; routine independent experiments need no new approval.',
        'source_role':'Latest unchanged-noise researchcontrol86 full learning state. Chosen to continue reference mechanism, not by its exploratory45.05% point score. No adoption; production26 remains reference.',
        'hypothesis':'An actor versus a teacher currently learns teacher hard Policy advice on its own MCTS behavior turns. Replacing only these fresh actor-owned Policy targets with its actual search visits may close a larger autonomous policy-improvement loop while keeping genuine teacher-turn anchors. Not a proven bug or guaranteed gain.',
        'interpretation':'Single online label factor, not pure selfplay or a weight/replay change: candidate actor-owned teacher_vs_actor rows use real native neural visits; selfplay uses visits in both; actual teacher-owned and teacher_selfplay rows retain teacher hard labels. All actual terminal WDL and independent draws retained. Old initial teacher shards keep their original metadata and raw labels; no relabeling or excluded old Policy rows. Both query shadow teachers on all non-selfplay turns to match time-capped teacher workload; no savings claim. Trajectories can diverge after learning.',
        'collector':'Own preceding complete learner every stage; paired starts/modes/colors namespace800+,256newgames then64updates,3stage replay.24stages perbranch,86to110,12288newgames/3072updates total.',
        'promotion_rule':old['promotion_rule'],'horizon':old['horizon'],
        'registered_new_games':12288,'registered_updates':3072}
    check_factor_parameters(plan);verify_actor_policy_prerequisite(root,plan)
    path=f'config/experiments/open_v4_actor_policy_{stamp}.json';atomic_json(root/path,plan)
    print(json.dumps({'plan':path,'directory':directory,'source':source,'registered_new_games':12288,'registered_updates':3072},ensure_ascii=False))


if __name__=='__main__':main()
