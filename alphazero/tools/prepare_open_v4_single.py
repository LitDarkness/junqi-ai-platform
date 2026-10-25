"""Register a new single learner from control134 without touching archival STOPs."""
from pathlib import Path


def main():
    import json, torch
    from datetime import datetime, timezone
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from open_v4_queue_contract import sha
    from open_v4_single_contract import verify_single_prerequisite
    from junqi_open_v4.root_explore_config import RootExploreConfig
    root = Path(__file__).resolve().parents[1]
    if lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):
        raise ValueError('another GPU/supervisor task owns global lock')
    old_path='config/experiments/open_v4_root_explore_20261004_214715.json'
    old=read_json(root/old_path); progress=read_json(root/old['directory']/'PROGRESS.json')
    saved=progress['variants']['control']
    payload=torch.load(root/saved['checkpoint'],map_location='cpu',weights_only=False)
    source=dict(path=saved['checkpoint'],sha256=saved['sha256'],model_hash=saved['model_hash'],
                iteration=payload['counters']['iteration'],leaf_value_scale=payload['leaf_value_scale'])
    if source['iteration']!=134: raise ValueError('completed134 required')
    stamp=datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    directory=f'runs/open_junqi_v4_single_{stamp}'
    base_path=f'config/experiments/open_v4_single_base_{stamp}.json'
    atomic_json(root/base_path,payload['config']); RootExploreConfig.load(root/base_path)
    def entry(path): return dict(path=str(Path(path)),sha256=sha(root/path))
    other=progress['variants']['candidate']
    other_item=dict(path=other['checkpoint'],sha256=other['sha256'],model_hash=other['model_hash'],iteration=134)
    evidence_dir='runs/open_junqi_v4_longitudinal_20261005_035049'
    names=set(old['source'])|{'tools/prepare_open_v4_single.py','tools/run_open_v4_single.py',
        'tools/open_v4_single_contract.py','tools/evaluate_open_v4_longitudinal.py',
        'tools/run_open_v4_fresh_data.py','tests/test_open_v4_single.py','start_open_v4_single.ps1','stop_open_v4_single.ps1'}
    frozen=[dict(directory=str(p.parent.relative_to(root)),current_sha256=sha(p))
            for p in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json'))]
    markers=[entry(str(p.relative_to(root))) for p in sorted((root/'runs').glob('*/**/STOP'))]
    plan=dict(schema='junqi-single-online-plan-1',route='single_online_continuation',registered_at=utc_now(),
        directory=directory,training_variants=['learner'],parent_config=base_path,parent_config_sha256=sha(root/base_path),
        source_checkpoint=source,source_counters=payload['counters'],source_config_hash=payload['config_hash'],
        source_route_qualified=False,participants=dict(parent=source,control=other_item,
            champion=old['participants']['champion'],guard86=old['participants']['guard86']),
        suite=old['suite'],teachers=old['teachers'],layout_manifest_hash=old['layout_manifest_hash'],
        initial_replay=progress['collections']['control'][-3:],frozen_runs=frozen,historical_stops=markers,
        historical_stop_markers=[m['path'] for m in markers],source={n:sha(root/n) for n in sorted(names)},
        stop_paths=[directory+'/STOP',directory+'/supervision/STOP'],previous_experiment=entry(old_path),
        longitudinal_evidence=dict(review=entry(evidence_dir+'/REVIEW.json'),results=entry(evidence_dir+'/RESULTS.json')),
        native_build=old['native_build'],learner=dict(old['control']),collection_start=1000,
        evaluation_seed=old['evaluation_seed'],screen_families=16,full_families=64,evaluation_budget=256,
        evaluation_inference_batch_size=32,stages_per_block=12,segment_stages=24,maximum_stages=96,
        games_per_stage=256,updates_per_stage=64,replay_stages=3,minimum_free_gib=10,
        authorization='User explicitly revoked pause for new single learner continuation; old runs and STOPs remain archived.',
        source_role='control134 complete research learning state; longitudinal eligible, not production promoted.',
        frozen_reference_definition='arena control is archived candidate134, frozen throughout, never a concurrently trained control.',
        interpretation='One evolving learner. Other134 and86 are frozen protection, no new factorial/causal claim or equivalence claim. Full gates unchanged; losses do not establish strength.',
        promotion_rule=old['promotion_rule'],
        horizon='First24 stages134to158; intermediate12 uncertainty continues24. After each24, only native-verified full64 route confirmation allows immediate next24 without a heartbeat. Otherwise review_due. Up to four segments before renewed audit.',
        registered_first_segment_new_games=6144,registered_first_segment_updates=1536,
        registered_maximum_new_games=24576,registered_maximum_updates=6144,
        efficiency='One vs two evolving branches halves planned collection and learning work; evaluation remains. No measured end-to-end2x claim.',
        evaluation_matchups=16,screen_games=1376,full_games=5184)
    verify_single_prerequisite(root,plan)
    path=f'config/experiments/open_v4_single_{stamp}.json'; atomic_json(root/path,plan)
    print(json.dumps(dict(plan=path,directory=directory,source=source),ensure_ascii=False))


if __name__=='__main__': main()
