"""Prospective bounded one-learner contract, with no old-rule reinterpretation."""
from copy import deepcopy
from pathlib import Path
import numpy as np
from open_v4_runtime import read_json
from open_v4_queue_contract import inside, sha


def validate_source_config(source, cfg):
    before = source['config']; after = cfg.as_dict()
    changed = {k for k in set(before)|set(after) if before.get(k) != after.get(k)}
    if changed - {'run_directory','initial_checkpoint','updates_per_iteration','policy_step_lr_multiplier'}:
        raise ValueError('unexpected factor relative to complete source: '+repr(changed))
    if before['updates_per_iteration'] != 64 or after['updates_per_iteration'] != 256 or after['policy_step_lr_multiplier'] != .25 or after['learning_rate'] != 5e-5:
        raise ValueError('256-update route, quarter Policy phase only, unchanged Value base LR')
    return sorted(changed)


def fork_stable_payload(source, cfg, evidence):
    from fork_open_v4_qualified import fork_payload
    validate_source_config(source,cfg)
    result=fork_payload(source,cfg,evidence)
    # Preserve optimizer moments/step, but make the saved current-phase LR
    # describe the new contract too. The actual next phase always resets LR.
    last_phase='value' if (source['counters']['attempted_updates']-1)%3==2 else 'policy'
    factor=cfg.policy_step_lr_multiplier/source['config'].get('policy_step_lr_multiplier',1.)
    if last_phase=='policy':
        for group in result['optimizer']['param_groups']:
            if group['name'] in ('encoder','policy'):group['lr']*=factor
    verify_stable_seed(source,result)
    return result


def verify_stable_seed(before, after):
    from run_open_v4_search_training import verify_learning_state
    checked=deepcopy(after)
    last_phase='value' if (before['counters']['attempted_updates']-1)%3==2 else 'policy'
    factor=after['config']['policy_step_lr_multiplier']/before['config'].get('policy_step_lr_multiplier',1.)
    for old,new in zip(before['optimizer']['param_groups'],checked['optimizer']['param_groups']):
        expected=old['lr']*(factor if last_phase=='policy' and old['name'] in ('encoder','policy') else 1.)
        if new['lr']!=expected:raise ValueError('undeclared optimizer LR change')
        new['lr']=old['lr']
    checks=verify_learning_state(before,checked)
    checks['only_declared_current_phase_group_lr_change']=True
    return checks


def check_single_parameters(plan):
    if plan['schema'] != 'junqi-stable-policy-plan-1' or plan['route'] != 'single_policy_step_rate':
        raise ValueError('independent Policy rate experiment required')
    if plan['training_variants'] != ['learner']:
        raise ValueError('one evolving learner only')
    if (plan['maximum_stages'],plan['segment_stages'],plan['stages_per_block']) != (24,12,6):
        raise ValueError('6/12 boundaries and conditionally24 maximum')
    if plan['screen_boundaries'] != [6,12,24]:raise ValueError('only three prospective negative looks')
    if (plan['games_per_stage'],plan['updates_per_stage'],plan['replay_stages']) != (256,256,3):
        raise ValueError('fresh data and PPV exposure unchanged from failed ratio route')
    if plan['policy_step_lr_multiplier'] != .25 or plan['source_checkpoint']['iteration'] != 158:
        raise ValueError('declared multiplier and complete158 source')
    if plan['participants']['parent'] != plan['source_checkpoint'] or plan['participants']['control']['iteration'] != 134 or plan['participants']['guard86']['iteration'] != 86 or plan['participants']['champion']['iteration'] != 26:
        raise ValueError('source and four protected reference roles required')
    if plan['stop_paths'] != [plan['directory']+'/STOP',plan['directory']+'/supervision/STOP']:
        raise ValueError('new branch STOP scope; historical STOPs immutable')
    expected=dict(selfplay_fraction=.25,teacher_selfplay_fraction=None,actor_policy_target='teacher',root_tactical_guard=False,
                  training_greedy_after_ply=24,root_noise_until_ply=None,forced_root_coeff=2.,exploration_target_pruning=False,deduplicate=False,neural_policy_only=False)
    if plan['learner'] != expected:raise ValueError('original sampling/behavior/targets must remain unchanged')
    if plan['small_screen_matchups'] != [['candidate','parent'],['parent','teacher0'],['candidate','teacher0'],['repeat','teacher0']]:
        raise ValueError('preregistered four matchups required')
    if plan['screen_games'] != 344 or plan['full_games'] != 5184:
        raise ValueError('distinct small research/full original gate counts')
    if plan['early_stop']['confidence'] != 1-.05/6 or plan['early_stop']['paired_delta_limit'] != -.05:
        raise ValueError('three negative looks times two endpoints and declared effect floor')


def family_interval(values, seed, confidence, resamples=8000):
    values=np.asarray(values,dtype=np.float64)
    if len(values)<2 or not np.isfinite(values).all():raise ValueError('finite family observations required')
    rng=np.random.default_rng(seed)
    estimates=np.concatenate([values[rng.integers(0,len(values),(min(256,resamples-start),len(values)))].mean(1) for start in range(0,resamples,256)])
    alpha=(1-confidence)/2
    return dict(mean=float(values.mean()),interval=list(map(float,np.quantile(estimates,[alpha,1-alpha]))),confidence=confidence,families=len(values),unit='paired layout family, all starts and colors clustered',resamples=resamples)


def small_screen_review(arena, seed, confidence, expected_source_score=.4609375):
    if arena['role']!='dev' or len(arena['summaries'])!=4:raise ValueError('four-matchup dev only')
    family_maps=[]
    for matchup in range(4):
        groups={}
        for game in arena['games']:
            if game['matchup']==matchup:groups.setdefault(int(game['family']),[]).append(game['score'])
        family_maps.append({k:float(np.mean(v)) for k,v in groups.items()})
    if not family_maps[0] or any(set(m)!=set(family_maps[0]) for m in family_maps):raise ValueError('matched family sets required')
    ids=sorted(family_maps[0])
    paired=family_interval([family_maps[2][i]-family_maps[1][i] for i in ids],seed,confidence)
    duel=family_interval([family_maps[0][i] for i in ids],seed^0x4a81,confidence)
    repeat=family_interval([family_maps[3][i]-family_maps[1][i] for i in ids],seed^0x328a,confidence)
    source=family_interval([family_maps[1][i] for i in ids],seed^0x9a1,confidence)
    early = paired['mean']<=-.05 and paired['interval'][1]<0
    early_direct = duel['mean']<=.45 and duel['interval'][1]<.5
    # Positive discovery uses ordinary95% bounds. It cannot grant adoption.
    paired95=family_interval([family_maps[2][i]-family_maps[1][i] for i in ids],seed, .95)
    duel95=family_interval([family_maps[0][i] for i in ids],seed^0x4a81,.95)
    positive=paired95['interval'][0]>0 and duel95['interval'][0]>.5 and abs(repeat['mean'])<=.02
    return dict(paired=paired,direct_source=duel,duplicate=repeat,source=source,paired95=paired95,direct95=duel95,
                early_stop=bool(early or early_direct),early_stop_reason='external_negative' if early else ('direct_source_negative' if early_direct else None),
                source_drift_ok=abs(source['mean']-expected_source_score)<=.02,positive_signal=bool(positive),screen_only=True,
                interpretation='Correlated082/084 not independent; repeated development; bootstrap not finite-sample guaranteed. New rule prospective only.')


def continuation_action(stage, confirmed, severe, maximum):
    if stage not in (6,12,18,24) or maximum!=24:raise ValueError('fixed diagnostic horizons')
    if severe:return 'review_due'
    if stage==6:return 'continue'
    if stage==12:return 'auto_next_segment' if confirmed else 'review_due'
    if stage==18:return 'continue'
    return 'segment_complete' if confirmed else 'review_due'


def verify_single_prerequisite(root,plan):
    check_single_parameters(plan);root=Path(root)
    for entry in [plan['source_checkpoint'],plan['diagnostic'],plan['prior_negative_review'],plan['cpu_verification']]:
        if sha(inside(root,entry['path']))!=entry['sha256']:raise ValueError('evidence/source changed')
    diagnostic=read_json(inside(root,plan['diagnostic']['path']))
    if not diagnostic['checks_passed'] or diagnostic['simulations']!=4096 or diagnostic['new_training_games'] or diagnostic['effective_updates']:
        raise ValueError('completed bounded output diagnosis required')
    if not read_json(inside(root,plan['cpu_verification']['path']))['all_passed']:raise ValueError('CPU migration/phase/control validation required')
    for item in plan['initial_replay']:
        if sha(inside(root,item['path']))!=item['sha256'] or sha(inside(root,item['metadata']))!=item['metadata_sha256']:raise ValueError('initial real replay changed')
        if read_json(inside(root,item['metadata']))['layout_role']!='train':raise ValueError('training-only replay')
    return dict(research_eligible=True,production_qualified=False,source158_protected=True,one_online_learner=True)
