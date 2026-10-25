"""Read-only CPU audit of first single online learning commit; no GPU initialization."""
from pathlib import Path


def main():
    import argparse, json, hashlib
    from collections import Counter
    import numpy as np
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from open_v4_queue_contract import inside, sha
    from open_v4_single_ratio_contract import verify_single_prerequisite
    from run_open_v4_search_training import verify_learning_state
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.training_replay import complete_games
    from junqi_open_v4.layouts import backend_for, load_role
    from junqi_search_v3.loop import read_selfplay_shard
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    plan=read_json(inside(root,args.plan));run=inside(root,plan['directory'])
    checks={}
    def require(name,value):
        checks[name]=bool(value)
        if not checks[name]:raise ValueError(name)
    verify_single_prerequisite(root,plan)
    progress=read_json(run/'PROGRESS.json');status=read_json(run/'STATUS.json')
    require('single_learner',set(progress['variants'])=={'learner'})
    require('first_learning_commit',progress['variants']['learner']['stage']>=1)
    source=torch.load(inside(root,plan['source_checkpoint']['path']),map_location='cpu',weights_only=False)
    seed=torch.load(run/'learner/checkpoints/learner_stage_000000.pt',map_location='cpu',weights_only=False)
    checks.update({'seed_'+k:v for k,v in verify_learning_state(source,seed).items()})
    cfg=RootExploreConfig.load(run/'learner/config.json')
    changed={k for k,v in cfg.as_dict().items() if v!=source['config'][k]}
    require('only_new_path_and_source',changed == {'run_directory','initial_checkpoint','updates_per_iteration'} and cfg.updates_per_iteration==256)
    path=run/'learner/checkpoints/learner_stage_000001.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    require('checkpoint_digest_contract',payload['config_hash']==cfg.digest())
    require('first_iteration159',payload['counters']['iteration']==159)
    report=read_json(run/'learner/reports/stage_000001.json')
    effective=sum(int(row['accepted']) for row in report['updates'])
    require('effective_updates_match_accepted',payload['counters']['effective_updates']-source['counters']['effective_updates']==effective and 0<effective<=256)
    require('256_update_attempt_records',len(report['updates'])==256)
    require('256_attempts',payload['counters']['attempted_updates']-source['counters']['attempted_updates']==256)
    require('new_model_learned',payload['model_hash']!=source['model_hash'])
    require('inherited_value_scale',payload['leaf_value_scale']==source['leaf_value_scale'])
    require('all_model_tensors_finite',all(torch.isfinite(v).all().item() for v in payload['model'].values() if v.is_floating_point()))
    require('all_optimizer_tensors_finite',all(torch.isfinite(v).all().item() for row in payload['optimizer']['state'].values() for v in row.values() if isinstance(v,torch.Tensor) and v.is_floating_point()))
    for key in ('model','optimizer','scheduler','counters','rng','config','leaf_value_scale'):
        require('complete_'+key,key in payload)
    item=progress['collections']['learner'][0]
    require('first_shard_hash',sha(inside(root,item['path']))==item['sha256'])
    require('first_metadata_hash',sha(inside(root,item['metadata']))==item['metadata_sha256'])
    meta=read_json(inside(root,item['metadata']));samples=read_selfplay_shard(inside(root,item['path']))
    require('own_source158_collection',meta['actor_model_hash']==source['model_hash'])
    require('256_actual_new_games',len(meta['games'])==256 and meta['game_count']==256)
    require('source_namespace1100',meta['iteration']==1100)
    require('raw_target_unchanged',meta['root_explore_verification']['all_passed'] and not meta['root_explore_verification']['changed_neural_targets'])
    modes=Counter(g['mode'] for g in meta['games'])
    # Fractions are probabilities per paired family, not fixed per-shard quotas.
    expected_modes=Counter()
    for game in meta['games']:
        family=meta['iteration']*1_000_000+game['index']//2
        require('paired_family_namespace',game['family']==family)
        key=np.random.default_rng(cfg.seed ^ family).random()
        expected='selfplay' if key<.25 else ('teacher_selfplay' if key<.5 else 'teacher_vs_actor')
        require('seeded_mode_assignment',game['mode']==expected)
        expected_modes[expected]+=1
    require('25_25_50_mode_probabilities',modes==expected_modes)
    training_layouts={row['id']:row for row in load_role(inside(root,cfg.layout_directory),'train')}
    native_plies=0;games=0
    for game,rows in complete_games(samples,meta):
        provenance=game['provenance']
        red=training_layouts[provenance['red_layout_id']];black=training_layouts[provenance['black_layout_id']]
        state=backend_for(red,black,mirrored=list(red['setup'])!=provenance['red_layout']).new_state(provenance['setup_seed'])
        require('initial_training_board',state.provenance()['initial_board']==provenance['initial_board'])
        for action,row in zip(game['trace'],rows):
            require('native_player',state.current_player()==row.player)
            require('native_key',int(state.native_state.key())==int(row.state_hash))
            require('native_sparse_legal',np.array_equal(np.asarray(state.legal_actions()),row.legal_actions))
            require('played_move_legal',action in state.legal_actions())
            state.apply(int(action));native_plies+=1
        require('native_terminal',state.is_terminal())
        require('native_reason',state.terminal_reason()==game['reason'])
        require('native_result',state.terminal_value(1)==game['result_red'])
        games+=1
    require('256_native_terminal_games',games==256)
    report=read_json(run/'learner/reports/stage_000001.json')
    require('fresh_policy_actually_sampled',sum(v for k,v in report['actual_policy_replay_origin_sources'].items() if k.startswith('fresh:'))>0)
    for name,digest in plan['source'].items(): require('immutable_source:'+name,sha(inside(root,name))==digest)
    for marker in plan['historical_stops']:require('preserved_STOP:'+marker['path'],sha(inside(root,marker['path']))==marker['sha256'])
    for frozen in plan['frozen_runs']:require('preserved_CURRENT:'+frozen['directory'],sha(inside(root,frozen['directory'])/'CURRENT.json')==frozen['current_sha256'])
    require('sole_global_owner',lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
    require('no_GPU_initialized_by_audit',not torch.cuda.is_initialized())
    result=dict(at=utc_now(),all_passed=all(checks.values()),checks=checks,new_training_games=256,effective_updates=effective,
        source_iteration=158,committed_iteration=159,source_model_hash=source['model_hash'],current_model_hash=payload['model_hash'],
        checkpoint=dict(path=str(path.relative_to(root)),sha256=sha(path)),native_games=games,native_plies=native_plies,
        collection_metrics=report['collection_metrics'],actual_modes=dict(modes),actual_policy_replay_origin_sources=report['actual_policy_replay_origin_sources'],
        example_exposure=report['example_exposure'],learning_elapsed_seconds=report['learning_elapsed_seconds'],observed_status=status,full_learning_state_preserved=True,production_promotion=False)
    atomic_json(run/'verification/FIRST_SINGLE_RATIO_TRAINING.json',result)
    atomic_json(run/'supervision/REVIEWS/STARTUP_REVIEW.json',result)
    print(json.dumps(dict(all_passed=result['all_passed'],checks=len(checks),new_training_games=256,effective_updates=effective,iteration=159,native_games=games,native_plies=native_plies),ensure_ascii=False))


if __name__=='__main__':main()
