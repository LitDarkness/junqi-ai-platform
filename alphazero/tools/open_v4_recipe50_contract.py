"""Authorized 50% selfplay / 50% actual082 pilot, with exact full-state inheritance."""
from pathlib import Path
import hashlib
from open_v4_runtime import read_json
from open_v4_engineered_contract import small_screen_review, continuation_action, is_screen_stage


def validate_source_config(source, cfg):
    actual = cfg.as_dict()
    before = source['config']
    changed = {k for k in set(actual) | set(before) if actual.get(k) != before.get(k)}
    allowed = {'run_directory', 'initial_checkpoint', 'selfplay_fraction', 'teacher_selfplay_fraction'}
    if changed - allowed or actual['selfplay_fraction'] != .5 or actual['teacher_selfplay_fraction'] != 0.:
        raise ValueError('only approved50/0/50 recipe and new paths may change: ' + repr(changed))
    if actual['updates_per_iteration'] != 64 or actual['games_per_iteration'] != 256:
        raise ValueError('original64PPV per256fresh required')
    if actual['actor_policy_target'] != 'teacher' or actual['simulations'] != 64 or actual['repeated_simulations'] != 128:
        raise ValueError('preserve original hard teacher labels and64/128 searches')
    return sorted(changed)


def verify_stable_seed(before, after):
    from run_open_v4_search_training import verify_learning_state
    return verify_learning_state(before, after)


def fork_stable_payload(source, cfg, evidence):
    from fork_open_v4_qualified import fork_payload
    validate_source_config(source, cfg)
    result = fork_payload(source, cfg, evidence)
    verify_stable_seed(source, result)
    return result


def check_single_parameters(plan):
    if (plan['schema'], plan['route']) != ('junqi-recipe50-400to424-plan-1', 'recipe50_selfplay_vs082'):
        raise ValueError('explicitly approved recipe required')
    if (plan['training_variants'], plan['games_per_stage'], plan['updates_per_stage'], plan['replay_stages']) != (['learner'], 256, 64, 3):
        raise ValueError('single own-learner256fresh/64PPV/3replay required')
    if (plan['source_checkpoint']['iteration'], plan['target_iteration'], plan['maximum_stages']) != (400, 424, 24):
        raise ValueError('bounded complete400to424 recipe pilot required')
    if plan['participants']['parent'] != plan['source_checkpoint']:
        raise ValueError('source400 must remain frozen qualification protection')
    if plan['screen_boundaries'] != [6, 12, 24] or plan['periodic_screen_stages'] != 12:
        raise ValueError('registered6/12/24 monitoring required')
    if plan['performance_stop_enabled'] or plan['continuation_requires_promotion'] or not plan['no_weight_reset']:
        raise ValueError('negative or uncertain strength cannot stop the authorized pilot')
    if plan['training_teacher_ids'] != [0] or plan['learner']['selfplay_fraction'] != .5 or plan['learner']['teacher_selfplay_fraction'] != 0.:
        raise ValueError('50%selfplay/50%actual082 only; zero teacher selfplay')
    if plan['small_screen_matchups'] != [['candidate', 'parent'], ['parent', 'teacher0'], ['candidate', 'teacher0'], ['repeat', 'teacher0']]:
        raise ValueError('research screen must retain frozen source repeat and direct duel')
    if (plan['screen_games'], plan['full_games'], plan['evaluation_budget'], plan['evaluation_inference_batch_size']) != (344, 5184, 256, 32):
        raise ValueError('original formal gates and physical evaluation shape required')
    if len(plan['teachers']) != 2 or plan['teacher_identity_contract'] != 'one_original_native_teacher_per_worker_actual_sha_verified':
        raise ValueError('genuine082/084 evaluation protections must remain')
    if plan['stop_paths'] != [plan['directory'] + '/STOP', plan['directory'] + '/supervision/STOP']:
        raise ValueError('fresh current STOP scope required')


def verify_single_prerequisite(root, plan):
    check_single_parameters(plan)
    root = Path(root)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for entry in (plan['source_checkpoint'], plan['user_contract'], plan['cpu_verification'], plan['boundary_audit'], plan['benchmark']):
        if sha(root / entry['path']) != entry['sha256']:
            raise ValueError('source or authorization evidence changed')
    auth = read_json(root / plan['user_contract']['path'])
    if auth['recipe'] != {'selfplay': .5, 'teacher_selfplay': 0., 'vs_actual_082': .5} or auth['allow_other_learning_changes']:
        raise ValueError('latest human authorization scope mismatch')
    if not read_json(root / plan['cpu_verification']['path'])['all_passed'] or not read_json(root / plan['boundary_audit']['path'])['all_checks_passed']:
        raise ValueError('state and recipe boundary verification required')
    bench = read_json(root / plan['benchmark']['path'])
    if not bench['all_exact'] or not bench['revision_invalidates_graphs']:
        raise ValueError('retained evaluation graph proof required')
    if len(plan['initial_replay']) != 3:
        raise ValueError('retain latest3 original TRAIN shards')
    for entry in plan['initial_replay']:
        if sha(root / entry['path']) != entry['sha256'] or sha(root / entry['metadata']) != entry['metadata_sha256'] or read_json(root / entry['metadata'])['layout_role'] != 'train':
            raise ValueError('inherited original TRAIN shards must not be relabeled')
    return dict(source400_complete=True, recipe_authorized=True, no_weight_reset=True,
                all_learning_state_preserved=True, evaluation_both_teachers_retained=True)
