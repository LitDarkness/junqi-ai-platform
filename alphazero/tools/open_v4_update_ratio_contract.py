"""One online learning/new-experience ratio factor; original gates stay intact."""
from pathlib import Path
from open_v4_runtime import read_json
from open_v4_queue_contract import inside,sha

def check_factor_parameters(plan):
    control,candidate=plan['control'],plan['candidate']
    differences={k for k in control.keys()|candidate.keys() if control.get(k)!=candidate.get(k)}
    if differences!={'updates_per_stage'} or (control['updates_per_stage'],candidate['updates_per_stage'])!=(64,256):
        raise ValueError('only all-PPV update count64/256 may differ')
    expected={'selfplay_fraction':.25,'teacher_selfplay_fraction':None,'actor_policy_target':'teacher',
        'root_tactical_guard':False,'training_greedy_after_ply':24,'root_noise_until_ply':None,
        'forced_root_coeff':2.0,'exploration_target_pruning':False,'deduplicate':False,'neural_policy_only':False}
    for branch in (control,candidate):
        if any(branch.get(k)!=v for k,v in expected.items()):raise ValueError('sampling/search/target contract cannot change')
    if plan['source_checkpoint']['iteration']!=134 or not plan['participants'].get('guard86'):
        raise ValueError('latest unchanged reference134 plus frozen86 guard required')
    if plan['updates_per_stage']!={'control':64,'candidate':256} or plan['registered_updates']!=7680:
        raise ValueError('unequal optimizer clocks must be explicit')
    if (plan['maximum_stages'],plan['stages_per_block'],plan['games_per_stage'],plan['replay_stages'],plan['collection_start'])!=(24,12,256,3,1000):
        raise ValueError('registered data/horizon/source window cannot change')

def verify_update_ratio_prerequisite(root,plan):
    root=Path(root)
    for e in [*plan['previous_experiment'].values(),plan['boundary_audit'],plan['cpu_verification'],plan['native_build']['manifest']]:
        if sha(inside(root,e['path']))!=e['sha256']:raise ValueError('bound evidence changed')
    old=read_json(inside(root,plan['previous_experiment']['plan']['path']))
    review=read_json(inside(root,plan['previous_experiment']['final_review']['path']))
    if not review['all_checks_passed'] or not all(review['checks'].values()) or review['native_games_checked']!=1376 or review['route_confirmed']:
        raise ValueError('preceding negative factor must be fully reviewed')
    arena=read_json(inside(root,review['final_arena_review']['path']))
    if sha(inside(root,review['final_arena_review']['path']))!=review['final_arena_review']['sha256'] or not arena['all_checks_passed']:
        raise ValueError('all16 matchups/native/gates required')
    progress=read_json(inside(root,old['directory'])/'PROGRESS.json');item=progress['variants']['control']
    if read_json(inside(root,old['directory'])/'STATUS.json')['state']!='review_due':raise ValueError('preceding task not complete')
    if any(plan['source_checkpoint'][k]!=item[j] for k,j in [('path','checkpoint'),('sha256','sha256'),('model_hash','model_hash')]):
        raise ValueError('latest unchanged complete reference required')
    if plan['initial_replay']!=progress['collections']['control'][-3:]:raise ValueError('actual current last3 shards required')
    audit=read_json(inside(root,plan['boundary_audit']['path']));proof=read_json(inside(root,plan['cpu_verification']['path']))
    if not audit['complete'] or not audit['training_only'] or audit['new_training_games'] or audit['effective_updates'] or not all(audit['checks'].values()):
        raise ValueError('small saved-training audit required')
    if audit['source_checkpoint']!=plan['source_checkpoint'] or not proof['all_passed']:raise ValueError('source/CPU validation mismatch')
    if sha(inside(root,plan['native_build']['binary']['path']))!=plan['native_build']['binary']['sha256']:raise ValueError('native binary changed')
    return audit

def exposure_summary(counts,pools):
    """Distinct examples, rather than distinct boards or layout generalization."""
    from collections import Counter
    result={}
    for phase,rows in counts.items():
        family=Counter();origins=Counter()
        for identity,n in rows.items():
            # shard, game family, ply, player, full native state hash
            family[identity[1]]+=n;origins[identity[0]]+=n
        result[phase]={'sampled_rows':sum(rows.values()),'unique_examples':len(rows),
            'pool_rows':pools[phase],'unique_pool_fraction':len(rows)/pools[phase],
            'families_seen':len(family),'max_family_samples':max(family.values(),default=0),'by_shard':dict(origins)}
    return result
