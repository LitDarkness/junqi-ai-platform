"""Approved performance engineering; fixed learning mechanism through400."""
from pathlib import Path
import hashlib
from open_v4_runtime import read_json
from open_v4_continuous_fixed_contract import fork_stable_payload,verify_stable_seed,validate_source_config,small_screen_review as previous_screen

def small_screen_review(*args):
    review=previous_screen(*args)
    if not review['source_drift_ok']:
        review['positive_signal']=False
        review['measurement_review_required']=True
    return review

def is_screen_stage(stage,plan): return stage in plan['screen_boundaries']
def continuation_action(*args): return 'continue_monitor_only'

def check_single_parameters(plan):
    if (plan['schema'],plan['route'])!=('junqi-engineered-362to400-plan-1','engineered_fixed_mechanism_to400'):
        raise ValueError('approved engineering route required')
    if (plan['training_variants'],plan['games_per_stage'],plan['updates_per_stage'],plan['replay_stages'])!=(['learner'],256,64,3):
        raise ValueError('single learner256fresh/64PPV/3replay must remain')
    if (plan['source_checkpoint']['iteration'],plan['target_iteration'],plan['maximum_stages'])!=(362,400,38):
        raise ValueError('complete362 to400 without reset')
    if plan['screen_boundaries']!=[12,24,36,38] or plan['periodic_screen_stages']!=12:
        raise ValueError('retain absolute374/386/398/400 monitoring')
    if plan['performance_stop_enabled'] or plan['continuation_requires_promotion'] or not plan['no_weight_reset']:
        raise ValueError('unchanged continuous authorization required')
    if plan['participants']['parent']['iteration']!=302:
        raise ValueError('original frozen302 qualification reference retained')
    if plan['small_screen_matchups']!=[['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']]:
        raise ValueError('measurement unchanged')
    if (plan['screen_games'],plan['full_games'],plan['evaluation_budget'],plan['evaluation_inference_batch_size'])!=(344,5184,256,32):
        raise ValueError('formal gates and physical evaluation shape unchanged')
    if plan['teacher_identity_contract']!='one_original_native_teacher_per_worker_actual_sha_verified':
        raise ValueError('genuine isolated teachers required')
    if plan['stop_paths']!=[plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:
        raise ValueError('current STOP required')

def verify_single_prerequisite(root,plan):
    check_single_parameters(plan);root=Path(root)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for e in (plan['source_checkpoint'],plan['user_contract'],plan['cpu_verification'],plan['boundary_audit'],plan['benchmark'],plan['frozen_baseline'],plan['pending_screen']):
        if sha(root/e['path'])!=e['sha256']:raise ValueError('engineering/source evidence changed')
    auth=read_json(root/plan['user_contract']['path'])
    if not auth['performance_engineering_authorized'] or not auth['continue_to_iteration400'] or auth['allow_learning_hyperparameter_change']:
        raise ValueError('authorization scope mismatch')
    bench=read_json(root/plan['benchmark']['path'])
    if not bench['all_exact'] or not bench['revision_invalidates_graphs']:raise ValueError('strict graph equivalence required')
    if not read_json(root/plan['cpu_verification']['path'])['all_passed'] or not read_json(root/plan['boundary_audit']['path'])['all_checks_passed']:
        raise ValueError('state/rules/journal validation required')
    if len(plan['initial_replay'])!=3:raise ValueError('latest3 original TRAIN shards required')
    for e in plan['initial_replay']:
        if sha(root/e['path'])!=e['sha256'] or sha(root/e['metadata'])!=e['metadata_sha256'] or read_json(root/e['metadata'])['layout_role']!='train':
            raise ValueError('unaltered TRAIN shards required')
    if not read_json(root/plan['pending_screen']['path'])['review']['positive_signal']:
        raise ValueError('pending full362 gate needs original positive screen')
    return dict(source362_complete=True,unchanged_mechanism=True,no_weight_reset=True,
        performance_only_engineering=True,original_references_retained=True,target400=True)
