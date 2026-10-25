"""Restore the actual pre-fourfold mechanism, preserving current learned state."""
from copy import deepcopy
from pathlib import Path
import hashlib
from open_v4_runtime import read_json
from open_v4_stable_policy_contract import small_screen_review as original_screen,continuation_action


REFERENCE='runs/open_junqi_v4_single_20261005_044709/learner/checkpoints/learner_stage_000024.pt'


def validate_source_config(source,cfg):
    import torch
    root=Path(__file__).resolve().parents[1]
    original=torch.load(root/REFERENCE,map_location='cpu',weights_only=False)['config']
    actual=cfg.as_dict()
    ignored={'run_directory','initial_checkpoint'}
    differences={k for k in set(actual)|set(original) if k not in ignored and actual.get(k)!=original.get(k)}
    if differences:raise ValueError('mechanism must exactly match pre-fourfold158 config: '+repr(differences))
    if actual['updates_per_iteration']!=64 or 'policy_step_lr_multiplier' in actual:raise ValueError('64update and original Policy rate mechanism required')
    before=source['config']
    changed={k for k in set(before)|set(actual) if before.get(k)!=actual.get(k)}
    if changed-{'run_directory','initial_checkpoint','updates_per_iteration','policy_step_lr_multiplier'}:raise ValueError('undeclared mechanism change')
    return sorted(changed)


def verify_stable_seed(before,after):
    from run_open_v4_search_training import verify_learning_state
    checked=deepcopy(after)
    policy=(before['counters']['attempted_updates']-1)%3!=2
    factor=1/before['config'].get('policy_step_lr_multiplier',1.)
    for old,new in zip(before['optimizer']['param_groups'],checked['optimizer']['param_groups']):
        expected=old['lr']*(factor if policy and old['name'] in ('encoder','policy') else 1.)
        if new['lr']!=expected:raise ValueError('only original current-phase LR restoration allowed')
        new['lr']=old['lr']
    result=verify_learning_state(before,checked)
    result['model_weights_not_replaced']=True;result['Adam_moments_steps_scheduler_counters_all_RNG_preserved']=True
    return result


def fork_stable_payload(source,cfg,evidence):
    from fork_open_v4_qualified import fork_payload
    validate_source_config(source,cfg)
    result=fork_payload(source,cfg,evidence)
    if (source['counters']['attempted_updates']-1)%3!=2:
        for group in result['optimizer']['param_groups']:
            if group['name'] in ('encoder','policy'):group['lr']/=source['config'].get('policy_step_lr_multiplier',1.)
    verify_stable_seed(source,result)
    return result


def small_screen_review(arena,seed,confidence,expected_source_score):
    result=original_screen(arena,seed,confidence,expected_source_score)
    # User's instruction takes priority over a stricter statistical-delay rule.
    # The CI describes uncertainty; it is not a reason to extend declining work.
    result['early_stop']=result['paired']['mean']<0
    result['early_stop_reason']='negative_external_point_score_user_constraint' if result['early_stop'] else None
    return result


def check_single_parameters(plan):
    if plan['schema']!='junqi-mechanism-reversion-plan-1' or plan['route']!='restore_pre_fourfold_mechanism':raise ValueError('user mechanism reversion only')
    if plan['training_variants']!=['learner'] or plan['games_per_stage']!=256 or plan['updates_per_stage']!=64 or plan['replay_stages']!=3:raise ValueError('original one-learner update/data ratio required')
    if (plan['maximum_stages'],plan['segment_stages'],plan['stages_per_block'],plan['screen_boundaries'])!=(24,12,6,[6,12,24]):raise ValueError('bounded unchanged evaluation schedule')
    if plan['participants']['parent']!=plan['source_checkpoint'] or not plan['no_weight_reset']:raise ValueError('current complete state required')
    if plan['small_screen_matchups']!=[['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']] or plan['screen_games']!=344 or plan['full_games']!=5184:raise ValueError('frozen development context required')
    if plan['stop_paths']!=[plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:raise ValueError('fresh branch stop scope required')


def verify_single_prerequisite(root,plan):
    check_single_parameters(plan);root=Path(root)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for e in [plan['source_checkpoint'],plan['user_contract'],plan['trigger'],plan['cpu_verification']]:
        if sha(root/e['path'])!=e['sha256']:raise ValueError('declared source/evidence changed')
    if read_json(root/plan['trigger']['path'])['paired_delta']>=0:raise ValueError('negative user condition not met')
    if not read_json(root/plan['cpu_verification']['path'])['all_passed']:raise ValueError('CPU full-current-state preservation required')
    for e in plan['initial_replay']:
        if sha(root/e['path'])!=e['sha256'] or sha(root/e['metadata'])!=e['metadata_sha256'] or read_json(root/e['metadata'])['layout_role']!='train':raise ValueError('latest real training replay required')
    return dict(user_requested_mechanism_restored=True,no_weight_reset=True,production_qualified=False,one_online_learner=True)
