"""User-authorized continuous training; strength screens never stop learning.

Learning is exactly the restored pre-fourfold mechanism. Formal adoption gates
remain independent of the decision to keep producing fresh training games.
"""
from pathlib import Path
import hashlib
from open_v4_runtime import read_json
from open_v4_mechanism_reversion_contract import (
    fork_stable_payload, verify_stable_seed, validate_source_config,
    small_screen_review as previous_screen,
)


def is_screen_stage(stage, plan):
    return stage in plan['screen_boundaries'] or stage % plan['periodic_screen_stages'] == 0


def continuation_action(stage, confirmed, negative, maximum=None):
    # User explicitly permits fluctuations and regression under this mechanism.
    # Neither screening uncertainty nor failed adoption suspends learning.
    return 'continue_monitor_only'


def small_screen_review(arena, seed, confidence, expected_source_score):
    result = previous_screen(arena, seed, confidence, expected_source_score)
    result['observed_negative_external_point'] = result['paired']['mean'] < 0
    result['historical_stop_would_have_fired'] = result['early_stop']
    result['early_stop'] = False
    result['early_stop_reason'] = None
    result['continuation_policy'] = 'continue_with_unchanged_mechanism_regardless_strength'
    result['interpretation'] += ' Strength monitoring is not a training stop or a promotion.'
    return result


def check_single_parameters(plan):
    if plan['schema'] != 'junqi-continuous-fixed-mechanism-plan-1' or plan['route'] != 'continuous_restored_mechanism':
        raise ValueError('fixed-mechanism continuous authorization required')
    if (plan['training_variants'], plan['games_per_stage'], plan['updates_per_stage'], plan['replay_stages']) != (['learner'], 256, 64, 3):
        raise ValueError('one learner,256fresh/64updates/3replay unchanged')
    if plan['maximum_stages'] is not None or plan['screen_boundaries'] != [6,12,24] or plan['periodic_screen_stages'] != 12:
        raise ValueError('no performance-gated horizon; fixed periodic monitoring')
    if plan['performance_stop_enabled'] or plan['continuation_requires_promotion']:
        raise ValueError('strength-based stopping explicitly revoked by user')
    if plan['participants']['parent'] != plan['source_checkpoint'] or not plan['no_weight_reset']:
        raise ValueError('latest complete learning state must be preserved')
    if plan['small_screen_matchups'] != [['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']]:
        raise ValueError('development measurement cannot change')
    if (plan['screen_games'], plan['full_games'], plan['evaluation_budget'], plan['evaluation_inference_batch_size']) != (344,5184,256,32):
        raise ValueError('formal gates and measurement unchanged')
    if plan['stop_paths'] != [plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:
        raise ValueError('new branch respects all current STOPs')


def verify_single_prerequisite(root, plan):
    check_single_parameters(plan)
    root = Path(root)
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    for e in [plan['source_checkpoint'],plan['user_contract'],plan['cpu_verification'],plan['boundary_audit']]:
        if sha(root/e['path']) != e['sha256']:
            raise ValueError('declared source/evidence changed')
    authorization = read_json(root/plan['user_contract']['path'])
    if not authorization['continue_despite_strength_decline'] or authorization['allow_learning_hyperparameter_change']:
        raise ValueError('continuous fixed mechanism authorization only')
    if not read_json(root/plan['cpu_verification']['path'])['all_passed']:
        raise ValueError('CPU state/schedule validation required')
    if not read_json(root/plan['boundary_audit']['path'])['all_checks_passed']:
        raise ValueError('previous complete boundary must be verified')
    if len(plan['initial_replay']) != 3:
        raise ValueError('latest3 real training shards required')
    for e in plan['initial_replay']:
        if sha(root/e['path']) != e['sha256'] or sha(root/e['metadata']) != e['metadata_sha256']:
            raise ValueError('training replay changed')
        if read_json(root/e['metadata'])['layout_role'] != 'train':
            raise ValueError('development/final trajectories cannot enter replay')
    return dict(one_online_learner=True,no_weight_reset=True,
                unchanged64update_mechanism=True,performance_stop_enabled=False,
                production_qualified=False)
