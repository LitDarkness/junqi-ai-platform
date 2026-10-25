"""Preregister134-to158 with64/256 learning steps and unchanged fresh games."""
from pathlib import Path

def main():
    import argparse,copy,json,torch
    from datetime import datetime,timezone
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_queue_contract import sha
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_update_ratio_contract import check_factor_parameters,verify_update_ratio_prerequisite
    parser=argparse.ArgumentParser();parser.add_argument('--audit',required=True);parser.add_argument('--cpu-proof',required=True);a=parser.parse_args()
    root=Path(__file__).resolve().parents[1];supervision=read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old_path=supervision['optimization']['config'];old=read_json(root/old_path)
    if old['route']!='root_exploration_target_factor' or read_json(root/old['directory']/'STATUS.json')['state']!='review_due':raise ValueError('preceding factor must have ended')
    if lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock') or supervision.get('paused_by_user'):raise ValueError('another owner or user pause')
    progress=read_json(root/old['directory']/'PROGRESS.json');item=progress['variants']['control'];payload=torch.load(root/item['checkpoint'],map_location='cpu',weights_only=False)
    source={'path':item['checkpoint'],'sha256':item['sha256'],'model_hash':item['model_hash'],'iteration':payload['counters']['iteration'],'leaf_value_scale':payload['leaf_value_scale']}
    if source['iteration']!=134 or sha(root/source['path'])!=source['sha256']:raise ValueError('latest unchanged complete134 required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');directory=f'runs/open_junqi_v4_update_ratio_{stamp}'
    base_path=f'config/experiments/open_v4_update_ratio_base_{stamp}.json';atomic_json(root/base_path,payload['config']);RootExploreConfig.load(root/base_path)
    def entry(name):return {'path':str(Path(name)),'sha256':sha(root/name)}
    names=set(old['source'])|{'tools/prepare_open_v4_update_ratio.py','tools/run_open_v4_update_ratio.py',
        'tools/open_v4_update_ratio_contract.py','tools/audit_open_v4_value_rehearsal_boundary.py','tools/audit_open_v4_update_ratio_boundary.py',
        'tests/test_open_v4_update_ratio.py','tools/verify_open_v4_update_ratio_cpu.py','start_open_v4_update_ratio.ps1'}
    frozen=[{'directory':str(p.parent.relative_to(root)),'current_sha256':sha(p)} for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    markers=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))]
    control=dict(old['control'],exploration_target_pruning=False,updates_per_stage=64);candidate=dict(control,updates_per_stage=256)
    plan={'schema':'junqi-update-ratio-plan-1','registered_at':utc_now(),'directory':directory,'route':'learning_update_ratio_factor',
        'training_variants':['control','candidate'],'parent_config':base_path,'parent_config_sha256':sha(root/base_path),
        'source_checkpoint':source,'source_counters':payload['counters'],'source_config_hash':payload['config_hash'],'source_route_qualified':False,
        'participants':{'parent':source,'champion':old['participants']['champion'],'guard86':old['participants']['guard86']},
        'suite':old['suite'],'teachers':old['teachers'],'layout_manifest_hash':old['layout_manifest_hash'],'initial_replay':progress['collections']['control'][-3:],
        'frozen_runs':frozen,'historical_stops':markers,'historical_stop_markers':[m['path'] for m in markers],
        'source':{n:sha(root/n) for n in sorted(names)},'stop_paths':[directory+'/STOP',directory+'/supervision/STOP']+old['stop_paths'],
        'previous_experiment':{'plan':entry(old_path),'final_review':entry(supervision['optimization']['final_review'])},
        'boundary_audit':entry(a.audit),'cpu_verification':entry(a.cpu_proof),'native_build':old['native_build'],
        'evaluation_seed':old['evaluation_seed'],'screen_families':16,'full_families':64,'evaluation_budget':256,'evaluation_inference_batch_size':32,
        'collection_start':1000,'stages_per_block':12,'maximum_stages':24,'games_per_stage':256,
        'updates_per_stage':{'control':64,'candidate':256},'replay_stages':3,'control':control,'candidate':candidate,
        'authorization':'User authorizes continued new online training, accurate diagnosis and necessary independent validated experiments, with historical STOP and source preservation.',
        'source_role':'Latest unchanged-label unpruned reference134 full state, selected by continuity/role rather than exploratory strength. Unqualified;26production and86guard retained.',
        'hypothesis':'At64 learning updates per256freshgames, one measured shard received0.49Value passes/37.29%distinct-example coverage before eviction.256PPV updates may improve data use; underfitting or strength benefit is not established.',
        'interpretation':'Only all-PPV update count differs. Both retain original2:1Policy/Value phase, numerical learning rates, AdamW/KL/backtracking, all actual terminal WDL/independent draws,3stage replay,25/25/50teacher labels and64/128sampling. Both use shared forcedk2 raw targets; old pruning factor disabled. Initial3shards remain unchanged. Additional optimizer clocks/weight decay/RNG advancement are intrinsic effects; not a Value-only or isolated-coverage causal claim. More training compute is charged and not claimed as speedup. No frozen86 replay/holdout additions.',
        'collector':'Own preceding complete learner256freshgames perstage, then control64/candidate256updates. Pair initial layouts/modes/colors namespace1000+.24stages134to158;12288freshgames total.',
        'promotion_rule':old['promotion_rule'],'horizon':old['horizon'],'registered_new_games':12288,'registered_updates':7680,
        'registered_attempted_updates':7680,'effective_update_contract':'Actual accepted counts reported separately; control1536/candidate6144 if all accepted. No claim of effective7680 in advance.',
        'evaluation_matchups':16,'screen_games':1376,'full_games':5184,
        'cost_contract':'Record collection walltime/search/moves/inference rows/padding and learning walltime/exposure/update acceptance; compare strength versus total walltime and training compute.',
        'coverage_contract':'Distinct saved full-context examples keyed by shard/family/ply/player/native state hash. Every batch source counted; family coverage is not board or layout generalization.'}
    check_factor_parameters(plan);verify_update_ratio_prerequisite(root,plan)
    path=f'config/experiments/open_v4_update_ratio_{stamp}.json';atomic_json(root/path,plan)
    print(json.dumps({'plan':path,'directory':directory,'source':source,'registered_new_games':12288,'registered_updates':7680},ensure_ascii=False))
if __name__=='__main__':main()
