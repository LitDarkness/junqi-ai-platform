"""Single online learning/data-ratio experiment from complete158; strict frozen gates."""
from pathlib import Path
from open_v4_runtime import read_json
from open_v4_queue_contract import inside, sha
from open_v4_single_contract import continuation_action
from open_v4_update_ratio_contract import exposure_summary


def check_single_parameters(plan):
    if plan['schema'] != 'junqi-single-ratio-plan-1' or plan['route'] != 'single_learning_update_ratio':
        raise ValueError('new single learning-ratio contract required')
    if plan['training_variants'] != ['learner']:
        raise ValueError('exactly one evolving learner required')
    if (plan['maximum_stages'],plan['segment_stages'],plan['stages_per_block']) != (96,24,12):
        raise ValueError('bounded24-stage segments and12/24 evaluation required')
    if (plan['games_per_stage'],plan['updates_per_stage'],plan['replay_stages']) != (256,256,3):
        raise ValueError('only all-PPV update count64to256 may change')
    expected=dict(selfplay_fraction=.25,teacher_selfplay_fraction=None,actor_policy_target='teacher',root_tactical_guard=False,
                  training_greedy_after_ply=24,root_noise_until_ply=None,forced_root_coeff=2.0,exploration_target_pruning=False,deduplicate=False,neural_policy_only=False)
    if plan['learner'] != expected: raise ValueError('sampling/search/labels must remain unchanged')
    if plan['source_checkpoint']['iteration'] != 158 or plan['participants']['parent'] != plan['source_checkpoint']:
        raise ValueError('complete latest158 source required')
    if plan['stop_paths'] != [plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:
        raise ValueError('new branch STOP scope; preserve archived STOPs')
    if plan['participants']['control']['iteration'] != 134 or plan['participants']['guard86']['iteration'] != 86:
        raise ValueError('frozen stronger134/86 references required')
    if plan['registered_first_segment_updates'] != 6144 or plan['registered_first_segment_new_games'] != 6144:
        raise ValueError('distinct actual games and updates required')


def verify_single_prerequisite(root,plan):
    root=Path(root);check_single_parameters(plan)
    for entry in [*plan['previous_experiment'].values(),plan['boundary_audit'],plan['cpu_verification'],plan['native_build']['manifest'],plan['native_build']['binary'],plan['source_checkpoint']]:
        if sha(inside(root,entry['path'])) != entry['sha256']:raise ValueError('frozen evidence changed')
    old=read_json(inside(root,plan['previous_experiment']['plan']['path']));run=inside(root,old['directory']);progress=read_json(run/'PROGRESS.json')
    review=read_json(inside(root,plan['previous_experiment']['review']['path']))
    if read_json(run/'STATUS.json')['state'] != 'review_due' or not review['all_checks_passed'] or review['route_confirmed'] or review['native_games_checked'] != 1376:
        raise ValueError('preceding unconfirmed route must be naturally complete and reviewed')
    latest=progress['variants']['learner']
    for a,b in [('path','checkpoint'),('sha256','sha256'),('model_hash','model_hash')]:
        if plan['source_checkpoint'][a] != latest[b]:raise ValueError('latest complete158 learning state required')
    if plan['initial_replay'] != progress['collections']['learner'][-3:]:raise ValueError('actual latest3 replay shards required')
    if plan['learner'] != old['learner']:raise ValueError('single numerical factor only')
    audit=read_json(inside(root,plan['boundary_audit']['path']));proof=read_json(inside(root,plan['cpu_verification']['path']))
    if not audit['complete'] or not audit['cpu_only'] or not audit['training_only'] or audit['new_training_games'] or audit['effective_updates'] or not all(audit['checks'].values()):raise ValueError('small completed saved-training audit required')
    if audit['source_checkpoint'] != plan['source_checkpoint'] or not proof['all_passed']:raise ValueError('source/CPU validation mismatch')
    for item in plan['initial_replay']:
        if sha(inside(root,item['path']))!=item['sha256'] or sha(inside(root,item['metadata']))!=item['metadata_sha256']:raise ValueError('source replay changed')
        meta=read_json(inside(root,item['metadata']))
        if meta['layout_role']!='train' or meta['shard_sha256']!=item['sha256']:raise ValueError('training-only initial replay')
    return dict(research_eligible=True,production_qualified=False,single_learner=True,complete158_preserved=True)
