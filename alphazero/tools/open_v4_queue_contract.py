"""Strict completion checks shared by the queue and its online runner."""
from pathlib import Path
import hashlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(root,name):
    path=(Path(root)/name).resolve()
    if not path.is_relative_to(Path(root).resolve()) or path==Path(root).resolve():
        raise ValueError('path outside alphazero')
    return path


def validate_finished_arena(root,entry,*,native=False):
    from open_v4_runtime import read_json
    from open_v4_cached_arena import assemble,compare_repeat
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.arena import select_families
    from junqi_open_v4.pipeline import promotion_decision
    from run_open_v4_long_behavior import confirmed_route,screening_signal
    root=Path(root);path=inside(root,entry['path'])
    if sha(path)!=entry['sha256']:raise ValueError('completion plan changed')
    plan=read_json(path);directory=inside(root,plan['directory'])
    status=read_json(directory/'STATUS.json');results=read_json(directory/'RESULTS.json');progress=read_json(directory/'PROGRESS.json')
    if status['state']!='review_due' or results['state']!='review_due':raise ValueError('review not complete')
    for item in plan['frozen_files']+[plan['original_plan'],plan['cached_arena']]+list(plan['participants'].values()):
        if sha(inside(root,item['path']))!=item['sha256']:raise ValueError('frozen input changed')
    for name,digest in plan['source'].items():
        if sha(inside(root,name))!=digest:raise ValueError('executed evaluation source changed')
    record=progress['complete_arena'];arena_path=inside(root,record['path'])
    if results['arena']!=record or sha(arena_path)!=record['sha256']:raise ValueError('final arena commit mismatch')
    arena=read_json(arena_path);parts=[]
    for name in ('repeat_source38','new62_matches','fresh_baselines'):
        if name not in progress['parts']:continue
        part=progress['parts'][name];p=inside(root,part['path'])
        if sha(p)!=part['sha256']:raise ValueError('partial arena commit changed')
        data=read_json(p)
        if len(data['games'])!=part['games']:raise ValueError('part game count mismatch')
        parts.append(data)
    original=read_json(inside(root,plan['original_plan']['path']));cached=read_json(inside(root,plan['cached_arena']['path']))
    suite=select_families(read_json(inside(root,original['suite']['path']))['states'],64,original['evaluation_seed'])
    repeat=compare_repeat(cached,parts[0],suite)
    if repeat!=arena['cache_repeat'] or repeat!=read_json(directory/'verification/CACHE_REPEAT.json'):
        raise ValueError('repeat proof changed')
    use=repeat['all_exact']
    if not use and 'fresh_baselines' not in progress['parts']:raise ValueError('invalid cache needs complete fresh baselines')
    rebuilt=assemble(cached,parts,suite,original['evaluation_seed'],reuse=use)
    for key in ('summaries','paired_deltas','games','fresh_games','reused_games','suite_digest'):
        if rebuilt[key]!=arena[key]:raise ValueError('final arena recomputation mismatch: '+key)
    if len(arena['games'])!=4212 or len(arena['layout_families'])!=64 or arena['checkpoints']!=plan['participants']:
        raise ValueError('full64 participants/count mismatch')
    if arena['engine_contract']!=plan['engine_contract'] or arena['inference_contract']!=plan['inference_contract']:
        raise ValueError('evaluation context changed')
    base=OpenConfig.load(inside(root,original['parent_config']))
    gains={name:teacher_deltas(arena,name,'candidate') for name in ('champion','parent','control')}
    gains['duplicate']=teacher_deltas(arena,'parent','repeat');gate=promotion_decision(arena,base)
    duels=[r['family_score'] for r in arena['summaries'][:3]]
    review={'gains':gains,'gate':gate,'screen_promising':screening_signal([gains[k] for k in ('champion','parent','control')],duels),
        'route_confirmed':confirmed_route([gains[k] for k in ('champion','parent','control')],gains['duplicate'],duels,gate,64,.02),'screen_only':False}
    if review!=arena['review'] or review!=record['review']:raise ValueError('original gate recomputation mismatch')
    checked=0
    if native:
        from junqi_open_v4.layouts import backend_for
        for g in arena['games']:
            desc=suite[g['position']];state=backend_for(desc['red'],desc['black']).new_state(desc['setup_seed'])
            for action in desc['trace']:state.apply(int(action))
            for action in g['trace']:
                if state.is_terminal() or action not in state.legal_actions():raise ValueError('invalid saved arena trace')
                state.apply(int(action))
            if not state.is_terminal() or state.terminal_value(g['candidate_side'])!=g['value'] or state.terminal_reason()!=g['reason']:
                raise ValueError('actual native terminal mismatch')
            checked+=1
    return {'arena':{'path':record['path'],'sha256':record['sha256']},'review':review,'native_games_checked':checked,
            'fresh_games':arena['fresh_games'],'reused_games':arena['reused_games'],'new_training_games':0,'effective_updates':0}


def queue_decision(status,*,stopped,locked,complete):
    if stopped:return 'stop'
    if status in ('stopped_by_user','needs_attention'):return 'stop' if status=='stopped_by_user' else 'fail'
    if status=='review_due':return 'wait' if locked else ('launch' if complete else 'fail')
    if not locked:return 'fail'
    return 'wait'


def verify_boundary_audit(root,queue):
    import math
    from open_v4_runtime import read_json
    audit=read_json(inside(root,queue['directory'])/'BOUNDARY_AUDIT.json')
    identity={key:queue[key] for key in ('directory','source','completion_plan')}
    if audit.get('queue_identity')!=identity or audit.get('complete') is not True:
        raise ValueError('boundary diagnostic has not committed for this queue')
    completion=read_json(inside(root,queue['completion_plan']['path']))
    if audit.get('checkpoints')!=completion['participants'] or audit.get('new_training_games')!=0 or audit.get('effective_updates')!=0:
        raise ValueError('boundary diagnostic identity/work mismatch')
    diagnostic=audit['root_diagnostic'];count=diagnostic['roots']
    if not 0<count<=24 or diagnostic['total_simulations']!=128*count or len(diagnostic['comparisons'])!=count:
        raise ValueError('saved-root diagnostic incomplete')
    if set(audit['value'])!={'parent','control','candidate'}:raise ValueError('incomplete Value model comparison')
    for results in audit['value'].values():
        if set(results)!={'old38','control62','candidate62'}:raise ValueError('incomplete old/new training audit')
        for result in results.values():
            if not 0<result['rows']<=108 or not result['groups']>0 or not all(math.isfinite(v) for v in result['group_mean'].values()):
                raise ValueError('empty or nonfinite Value audit')
    return audit


def verify_training_prerequisite(root,plan):
    from open_v4_runtime import read_json
    proof=validate_finished_arena(root,plan['completion_plan'],native=False)
    saved=read_json(inside(root,plan['queue_directory'])/'COMPLETION_REVIEW.json')
    prerequisite=plan['prerequisite']
    if saved.get('native_games_checked')!=prerequisite['native_all_games_replayed'] or saved.get('native_games_checked')!=4212:
        raise ValueError('complete native review proof required before online training')
    if any(saved.get(key)!=value for key,value in proof.items() if key!='native_games_checked'):
        raise ValueError('saved completion proof does not match current full review')
    if proof['review']['route_confirmed'] is not prerequisite['required_route_confirmed']:
        raise ValueError('training route does not match the unchanged full gate')
    if not proof['review']['route_confirmed']:
        # Bind the diagnostic to the same immutable source contract and completion.
        verify_boundary_audit(root,{'directory':plan['queue_directory'],'source':plan['source'],'completion_plan':plan['completion_plan']})
    return proof
