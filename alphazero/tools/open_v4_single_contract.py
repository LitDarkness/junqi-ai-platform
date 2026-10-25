"""Single learner continuation eligibility, frozen references and finite segments."""
from pathlib import Path
from open_v4_runtime import read_json
from open_v4_queue_contract import inside, sha


def check_single_parameters(plan):
    if plan['schema'] != 'junqi-single-online-plan-1' or plan['route'] != 'single_online_continuation':
        raise ValueError('single online continuation contract required')
    if plan['training_variants'] != ['learner']:
        raise ValueError('exactly one evolving learner required')
    if (plan['maximum_stages'], plan['segment_stages'], plan['stages_per_block']) != (96, 24, 12):
        raise ValueError('four bounded24 segments with12/24 checks required')
    if (plan['games_per_stage'], plan['updates_per_stage'], plan['replay_stages']) != (256, 64, 3):
        raise ValueError('inherited data/update/replay schedule required')
    expected = dict(selfplay_fraction=.25, teacher_selfplay_fraction=None,
                    actor_policy_target='teacher', root_tactical_guard=False,
                    training_greedy_after_ply=24, root_noise_until_ply=None,
                    forced_root_coeff=2.0, exploration_target_pruning=False,
                    deduplicate=False, neural_policy_only=False)
    if plan['learner'] != expected:
        raise ValueError('unchanged control134 training mechanism required')
    if plan['stop_paths'] != [plan['directory']+'/STOP', plan['directory']+'/supervision/STOP']:
        raise ValueError('new branch STOPs only; archival STOPs must remain immutable')
    if plan['source_checkpoint']['iteration'] != 134 or plan['participants']['parent'] != plan['source_checkpoint']:
        raise ValueError('complete unchanged-control134 source required')
    if plan['participants']['control']['model_hash'] == plan['source_checkpoint']['model_hash']:
        raise ValueError('other134 must be a distinct frozen reference')


def continuation_action(stage, confirmed, severe, maximum=96):
    if severe:
        return 'review_due'
    if stage % 24:
        return 'continue_registered_segment'
    if not confirmed:
        return 'review_due'
    return 'auto_next_segment' if stage < maximum else 'registered_horizon_complete'


def verify_single_prerequisite(root, plan):
    root = Path(root)
    check_single_parameters(plan)
    for entry in [*plan['longitudinal_evidence'].values(), plan['previous_experiment'],
                  plan['native_build']['manifest'], plan['native_build']['binary'], plan['source_checkpoint']]:
        if sha(inside(root, entry['path'])) != entry['sha256']:
            raise ValueError('eligibility/source/native evidence changed')
    review = read_json(inside(root, plan['longitudinal_evidence']['review']['path']))
    results = read_json(inside(root, plan['longitudinal_evidence']['results']['path']))
    if review['new_evaluation_games'] != 9072 or review['native_replayed_plies'] != 279644:
        raise ValueError('completed native longitudinal assessment required')
    primary = next(r for r in results['primary'] if r['candidate'] == 'control134')
    if not primary['research_continuation_signal'] or primary['ci975'][0] <= .5 or primary['families'] != 64:
        raise ValueError('preregistered research continuation signal missing')
    if review['source_hash'] != plan['source_checkpoint']['model_hash'] or review['production_promotion']:
        raise ValueError('research eligibility must not masquerade as production promotion')
    old = read_json(inside(root, plan['previous_experiment']['path']))
    progress = read_json(inside(root, old['directory'])/'PROGRESS.json')
    source = progress['variants']['control']
    if any(plan['source_checkpoint'][a] != source[b] for a,b in
           [('path','checkpoint'),('sha256','sha256'),('model_hash','model_hash')]):
        raise ValueError('exact completed control134 required')
    if plan['initial_replay'] != progress['collections']['control'][-3:]:
        raise ValueError('source134 actual last three shards required')
    for item in plan['initial_replay']:
        for path_key, sha_key in [('path','sha256'),('metadata','metadata_sha256')]:
            if sha(inside(root,item[path_key])) != item[sha_key]:
                raise ValueError('source replay changed')
        meta=read_json(inside(root,item['metadata']))
        if meta['layout_role'] != 'train' or meta['shard_sha256'] != item['sha256']:
            raise ValueError('source replay must be training-only')
    return {'research_eligible':True, 'production_qualified':False,
            'single_learner':True, 'exact_replay_preserved':True}
