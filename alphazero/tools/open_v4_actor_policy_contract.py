"""Bound prerequisite for a new actor-owned Policy experiment."""
from pathlib import Path
from open_v4_runtime import read_json
from open_v4_queue_contract import sha, inside


def verify_actor_policy_prerequisite(root, plan):
    root = Path(root)
    previous = plan['previous_experiment']
    for entry in (previous['plan'], previous['final_review'], plan['boundary_audit']):
        if sha(inside(root, entry['path'])) != entry['sha256']:
            raise ValueError('previous result or boundary evidence changed')
    old = read_json(inside(root, previous['plan']['path']))
    review = read_json(inside(root, previous['final_review']['path']))
    status = read_json(inside(root, old['directory'])/'STATUS.json')
    progress = read_json(inside(root, old['directory'])/'PROGRESS.json')
    item = progress['variants']['control']
    source = plan['source_checkpoint']
    if status['state'] != 'review_due' or review['native_games_checked'] != 1118 or not all(review['checks'].values()):
        raise ValueError('completed and verified preceding experiment required')
    if (source['path'], source['sha256'], source['model_hash'], source['iteration']) != (
            item['checkpoint'], item['sha256'], item['model_hash'], item['counters']['iteration']):
        raise ValueError('must use latest unchanged control full state, not choose by score')
    if source['iteration'] != 86 or plan['initial_replay'] != progress['collections']['control'][-3:]:
        raise ValueError('latest86 and its actual last three training shards required')
    audit = read_json(inside(root, plan['boundary_audit']['path']))
    if not audit['complete'] or audit['previous_plan'] != previous['plan'] or audit['source_checkpoint'] != source:
        raise ValueError('training audit must bind the exact completed branch')
    if audit['new_training_games'] or audit['effective_updates'] or not audit['training_only']:
        raise ValueError('boundary diagnostic must not add replay or training')
    return audit


def check_factor_parameters(plan):
    control, candidate = plan['control'], plan['candidate']
    differences = {k for k in control.keys() | candidate.keys() if control.get(k) != candidate.get(k)}
    if differences != {'actor_policy_target'} or control['actor_policy_target'] != 'teacher' or candidate['actor_policy_target'] != 'search':
        raise ValueError('only actor-owned teacher_vs_actor Policy targets may change')
    for branch in (control, candidate):
        teacher_fraction = ((1 - branch['selfplay_fraction']) / 3 if branch['teacher_selfplay_fraction'] is None
                            else branch['teacher_selfplay_fraction'])
        if teacher_fraction != .25:
            raise ValueError('effective teacher mutual-game fraction must remain25%')
        expected = {'selfplay_fraction': .25,
                    'root_tactical_guard': False, 'training_greedy_after_ply': 24,
                    'root_noise_until_ply': None, 'deduplicate': False, 'neural_policy_only': False}
        if any(branch[k] != value for k, value in expected.items()):
            raise ValueError('registered behavior, mixture and replay qualifications cannot change')
