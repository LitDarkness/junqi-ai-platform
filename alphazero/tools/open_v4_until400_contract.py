"""Approved corrected-teacher continuation from complete302 to review400."""
from pathlib import Path
import hashlib
from open_v4_runtime import read_json
from open_v4_continuous_fixed_contract import (
    fork_stable_payload, verify_stable_seed, validate_source_config, small_screen_review,
)

def is_screen_stage(stage,plan):
    return stage in plan['screen_boundaries'] or stage%plan['periodic_screen_stages']==0

def continuation_action(stage,confirmed,negative,maximum=None):
    return 'continue_monitor_only'

def check_single_parameters(plan):
    if (plan['schema'],plan['route'])!=('junqi-until400-isolated-teachers-plan-1','restored_mechanism_isolated_teachers_to400'):
        raise ValueError('approved isolated teacher302to400 contract required')
    if (plan['training_variants'],plan['games_per_stage'],plan['updates_per_stage'],plan['replay_stages'])!=(['learner'],256,64,3):
        raise ValueError('one learner,256fresh/64updates/3replay required')
    if (plan['source_checkpoint']['iteration'],plan['target_iteration'],plan['maximum_stages'])!=(302,400,98):
        raise ValueError('absolute302to400 horizon required; no counter reset')
    if plan['screen_boundaries']!=[6,12,24,98] or plan['periodic_screen_stages']!=12:
        raise ValueError('registered monitoring schedule required')
    if plan['performance_stop_enabled'] or plan['continuation_requires_promotion']:
        raise ValueError('strength fluctuations must not stop training before400')
    if plan['participants']['parent']!=plan['source_checkpoint'] or not plan['no_weight_reset']:
        raise ValueError('complete current source302 required')
    if plan['small_screen_matchups']!=[['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']]:
        raise ValueError('original teacher0 small measurement unchanged')
    if (plan['screen_games'],plan['full_games'],plan['evaluation_budget'],plan['evaluation_inference_batch_size'])!=(344,5184,256,32):
        raise ValueError('measurement and formal gates unchanged')
    if plan['teacher_identity_contract']!='one_original_native_teacher_per_worker_actual_sha_verified':
        raise ValueError('original actual teachers must be isolated')
    if plan['stop_paths']!=[plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:
        raise ValueError('new branch STOP scope required; old STOPs immutable')

def verify_single_prerequisite(root,plan):
    check_single_parameters(plan);root=Path(root)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for e in [plan['source_checkpoint'],plan['user_contract'],plan['cpu_verification'],plan['boundary_audit']]:
        if sha(root/e['path'])!=e['sha256']: raise ValueError('source/evidence changed')
    auth=read_json(root/plan['user_contract']['path'])
    if not auth['continue_to_iteration400'] or auth['allow_learning_hyperparameter_change'] or not auth['teacher_identity_repair_authorized']:
        raise ValueError('latest user authorization mismatch')
    if not read_json(root/plan['cpu_verification']['path'])['all_passed'] or not read_json(root/plan['boundary_audit']['path'])['all_checks_passed']:
        raise ValueError('CPU identity and full boundary validation required')
    if len(plan['initial_replay'])!=3: raise ValueError('latest3 real training shards required')
    for e in plan['initial_replay']:
        if sha(root/e['path'])!=e['sha256'] or sha(root/e['metadata'])!=e['metadata_sha256'] or read_json(root/e['metadata'])['layout_role']!='train':
            raise ValueError('unaltered source TRAIN replay required')
    return dict(one_online_learner=True,no_weight_reset=True,unchanged64update_mechanism=True,
                actual_teacher_identity_corrected=True,performance_stop_enabled=False,target_iteration=400)
