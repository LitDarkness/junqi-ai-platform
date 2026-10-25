"""Immutable evidence, single-factor, sparse raw/target and terminal contracts."""
from pathlib import Path
from open_v4_runtime import read_json
from open_v4_queue_contract import inside,sha

def check_factor_parameters(plan):
    control,candidate=plan['control'],plan['candidate']
    differences={k for k in control.keys()|candidate.keys() if control.get(k)!=candidate.get(k)}
    if differences!={'exploration_target_pruning'} or control['exploration_target_pruning'] or not candidate['exploration_target_pruning']:
        raise ValueError('only target pruning may differ between two common forced-root branches')
    for b in (control,candidate):
        expected={'selfplay_fraction':.25,'teacher_selfplay_fraction':None,'actor_policy_target':'teacher',
                  'root_tactical_guard':False,'training_greedy_after_ply':24,'root_noise_until_ply':None,
                  'forced_root_coeff':2.0,'deduplicate':False,'neural_policy_only':False}
        if any(b.get(k)!=v for k,v in expected.items()):raise ValueError('other factors cannot change')
    if plan['target_math']['singleton_cutoff'] or plan['target_math']['quota_rounding']!='ceil':
        raise ValueError('registered integer core only; no singleton cutoff')
    if plan['source_checkpoint']['iteration']!=110 or not plan['participants'].get('guard86'):
        raise ValueError('latest research110 plus stronger frozen86 protection required')

def verify_root_explore_prerequisite(root,plan):
    root=Path(root);old=read_json(inside(root,plan['previous_experiment']['plan']['path']))
    review=read_json(inside(root,plan['previous_experiment']['final_review']['path']))
    for e in [*plan['previous_experiment'].values(),plan['boundary_audit'],plan['cpu_verification'],plan['native_build']['manifest']]:
        if sha(inside(root,e['path']))!=e['sha256']:raise ValueError('bound prior result or verification changed')
    if not all(review['checks'].values()) or review['native_games_checked']!=1118:
        raise ValueError('previous final native/statistical review incomplete')
    progress=read_json(inside(root,old['directory'])/'PROGRESS.json');item=progress['variants']['control']
    if read_json(inside(root,old['directory'])/'STATUS.json')['state']!='review_due':raise ValueError('preceding run still active')
    if any(plan['source_checkpoint'][key]!=item[other] for key,other in [('path','checkpoint'),('sha256','sha256'),('model_hash','model_hash')]):
        raise ValueError('must use latest unchanged control; no score-based reset')
    if plan['initial_replay']!=progress['collections']['control'][-3:]:raise ValueError('actual latest3 training shards required')
    audit=read_json(inside(root,plan['boundary_audit']['path']));tests=read_json(inside(root,plan['cpu_verification']['path']))
    if not audit['complete'] or not audit['training_only'] or audit['new_training_games'] or audit['effective_updates']:
        raise ValueError('saved-training diagnostic only')
    if audit['source_checkpoint']!=plan['source_checkpoint'] or not audit['root_diagnostic']['checks_passed']:
        raise ValueError('exact source and native target diagnostic required')
    if not tests['all_passed']:raise ValueError('required CPU conformance tests failed')
    native=plan['native_build']['binary']
    if sha(inside(root,native['path']))!=native['sha256']:raise ValueError('independent native binary changed')
    return audit

def verify_collection(samples,metadata,config):
    import numpy as np
    from junqi_open_v4.training_replay import complete_games
    counts={'rows':0,'neural_target_rows':0,'teacher_target_rows':0,'changed_neural_targets':0,'deleted_counts':0}
    if metadata['layout_role']!='train' or metadata['policy_target_contract']!='teacher':raise ValueError('source/actor label drift')
    for game,rows in complete_games(samples,metadata):
        for row in rows:
            counts['rows']+=1
            if not np.array_equal(row.raw_visits,row.visits) and game['mode']!='selfplay':raise ValueError('teacher labels changed')
            if game['mode']=='selfplay':
                raw=np.asarray(row.raw_visits);target=np.asarray(row.visits)
                budget=config.repeated_simulations if np.any(row.repetition_contexts>0) else config.simulations
                if raw.sum()!=budget-1 or np.any(target<0) or np.any(target>raw) or not target.sum():raise ValueError('true raw budget or sparse target invalid')
                if not config.exploration_target_pruning and not np.array_equal(raw,target):raise ValueError('control target changed')
                if not np.any((raw==raw.max())&(target==raw)):raise ValueError('no most-visited edge protected')
                counts['neural_target_rows']+=1
                counts['changed_neural_targets']+=int(not np.array_equal(raw,target))
                counts['deleted_counts']+=int(raw.sum()-target.sum())
            else:
                if row.visits.sum()!=1 or np.count_nonzero(row.visits)!=1:raise ValueError('teacher hard label changed')
                counts['teacher_target_rows']+=1
    return {'all_passed':True,**counts,'raw_and_target_separate':True,'true_terminal_wdl_checked':True}
