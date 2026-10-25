"""Preregister110-to134 target pruning factor with independent native build and stronger86 guard."""
from pathlib import Path

def main():
    import argparse,copy,json,torch
    from datetime import datetime,timezone
    from open_v4_runtime import read_json,atomic_json,utc_now,lock_held
    from open_v4_queue_contract import sha
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_root_explore_contract import check_factor_parameters,verify_root_explore_prerequisite
    parser=argparse.ArgumentParser();parser.add_argument('--native-manifest',required=True);parser.add_argument('--audit',required=True);parser.add_argument('--cpu-proof',required=True)
    a=parser.parse_args();root=Path(__file__).resolve().parents[1]
    supervision=read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json');old_path=supervision['optimization']['config'];old=read_json(root/old_path)
    if old['route']!='actor_owned_policy_factor' or read_json(root/old['directory']/'STATUS.json')['state']!='review_due':raise ValueError('completed preceding factor required')
    if lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):raise ValueError('another task owns handoff')
    progress=read_json(root/old['directory']/'PROGRESS.json');item=progress['variants']['control']
    payload=torch.load(root/item['checkpoint'],map_location='cpu',weights_only=False)
    source={'path':item['checkpoint'],'sha256':item['sha256'],'model_hash':item['model_hash'],
            'iteration':payload['counters']['iteration'],'leaf_value_scale':payload['leaf_value_scale']}
    if source['iteration']!=110 or sha(root/source['path'])!=source['sha256']:raise ValueError('latest full control110 required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S');directory=f'runs/open_junqi_v4_root_explore_{stamp}'
    base_path=f'config/experiments/open_v4_root_explore_base_{stamp}.json'
    atomic_json(root/base_path,dict(payload['config'],forced_root_coeff=2.0,exploration_target_pruning=False));RootExploreConfig.load(root/base_path)
    def entry(name):return {'path':str(Path(name)),'sha256':sha(root/name)}
    review_path=supervision['optimization']['final_review'];native=read_json(root/a.native_manifest)
    names=set(old['source'])|{'tools/prepare_open_v4_root_explore.py','tools/run_open_v4_root_explore.py',
        'tools/open_v4_root_explore_contract.py','tools/audit_open_v4_root_explore_boundary.py',
        'tools/root_explore_training_diagnostic.py','tools/build_open_v4_root_explore.py','tools/verify_open_v4_root_explore_cpu.py',
        'python/junqi_open_v4/root_explore_config.py','python/junqi_open_v4/root_explore_games.py',
        'python/junqi_search_v3/root_explore_search.py','tests/test_open_v4_root_explore.py','start_open_v4_root_explore.ps1',
        'cpp/root_explore/neural_puct.h','cpp/root_explore/neural_puct.cpp','cpp/root_explore/bindings.cpp',
        native['binary']['path'],a.native_manifest}
    names.update(native['identity']['inputs'])
    frozen=[{'directory':str(p.parent.relative_to(root)),'current_sha256':sha(p)} for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    markers=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))]
    control=dict(old['control'],actor_policy_target='teacher',forced_root_coeff=2.0,exploration_target_pruning=False)
    candidate=dict(control,exploration_target_pruning=True)
    plan={'schema':'junqi-root-explore-target-plan-1','registered_at':utc_now(),'directory':directory,
        'route':'root_exploration_target_factor','training_variants':['control','candidate'],
        'parent_config':base_path,'parent_config_sha256':sha(root/base_path),'source_checkpoint':source,
        'source_counters':payload['counters'],'source_config_hash':payload['config_hash'],'source_route_qualified':False,
        'participants':{'parent':source,'champion':old['participants']['champion'],'guard86':old['source_checkpoint']},
        'suite':old['suite'],'teachers':old['teachers'],'layout_manifest_hash':old['layout_manifest_hash'],
        'initial_replay':progress['collections']['control'][-3:],'frozen_runs':frozen,'historical_stops':markers,
        'historical_stop_markers':[m['path'] for m in markers],'source':{name:sha(root/name) for name in sorted(names)},
        'stop_paths':[directory+'/STOP',directory+'/supervision/STOP']+old['stop_paths'],
        'previous_experiment':{'plan':entry(old_path),'final_review':entry(review_path)},'boundary_audit':entry(a.audit),
        'cpu_verification':entry(a.cpu_proof),'native_build':{'manifest':entry(a.native_manifest),'directory':native['directory'],'binary':native['binary']},
        'evaluation_seed':old['evaluation_seed'],'screen_families':16,'full_families':64,'evaluation_budget':256,
        'evaluation_inference_batch_size':32,'collection_start':900,'stages_per_block':12,'maximum_stages':24,
        'games_per_stage':256,'updates_per_stage':64,'replay_stages':3,'control':control,'candidate':candidate,
        'target_math':{'forced_coeff':2.0,'forced_only_already_visited':True,'quota':'sqrt(2*noised_prior*sum(raw_edge_visits))',
            'force_tie':'lowest action id','ordinary_puct_scale':'unchanged sqrt(root_node_visits)',
            'best_edge':'raw visits, then exported Q, then lowest action id','quota_rounding':'ceil',
            'prune':'at most ceil(quota) decrements; fixed final Q,P,root_count; trial ordinary score strictly less than best final ordinary score',
            'singleton_cutoff':False,'behavior_uses_raw_visits':True,'additional_simulations':0,'leaves_per_tree':1},
        'authorization':'User authorized continual new online training and necessary validated independent mechanisms, preserve all archival STOP and failed branches.',
        'source_role':'Latest unchanged-label referencecontrol110 complete learning state, selected by continuity and role, not exploratory strength. Unqualified; stronger frozencontrol86 and productionChampion26 protect adoption.',
        'hypothesis':'Low-budget exploratory visits may burden neural Policy labels. Under common k2 forced-root exploration, pruning only non-best counts allowed by fixed-final PUCT may improve learning.',
        'interpretation':'Single target factor within a common forced-root search baseline. Not full KataGo replication: both behave with raw visits, no singleton cleanup. Shared forced exploration itself is not causally isolated. Only actual selfplay neural Policy labels differ; genuine teacher/actor hard labels, all actual terminal WDL/independent draws, old sparse replay and25/25/50 remain. Initial three shards retain original labels. No extra simulations or asserted speed/strength gain. All development uses original standard256search with noise/forced/guard/cache/reuse off.',
        'collector':'Own preceding complete learner256freshgames then64updates each stage,3stage replay; pairednamespace900+,24stages110to134,12288new/3072updates total.',
        'promotion_rule':old['promotion_rule']+' PLUS both teacher paired95lower>0 vs frozencontrol86 and direct candidate-control86 lower>50%; original gates never relaxed.',
        'horizon':old['horizon'],'registered_new_games':12288,'registered_updates':3072,
        'evaluation_matchups':16,'screen_games':1376,'full_games':5184,
        'research_sources':['https://arxiv.org/html/1902.10565#S3.SS2','https://raw.githubusercontent.com/lightvector/KataGo/v1.0/cpp/search/search.cpp']}
    check_factor_parameters(plan);verify_root_explore_prerequisite(root,plan)
    path=f'config/experiments/open_v4_root_explore_{stamp}.json';atomic_json(root/path,plan)
    print(json.dumps({'plan':path,'directory':directory,'source':source,'registered_new_games':12288,'registered_updates':3072},ensure_ascii=False))
if __name__=='__main__':main()
