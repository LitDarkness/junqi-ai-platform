"""Independent CPU verification at a naturally completed single online boundary."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    import numpy as np
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from open_v4_queue_contract import inside, sha
    from junqi_open_v4.arena import select_families, paired_interval
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.pipeline import promotion_decision
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_longitudinal import verify_native
    from run_open_v4_single import confirmed_route, screening_signal, severe_regression
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stage',type=int,default=24)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan = read_json(inside(root, args.plan)); run = inside(root, plan['directory'])
    out = inside(root, args.output)
    if out.exists(): raise ValueError('preserve prior review')
    checks = {}
    def require(name, value):
        checks[name] = bool(value)
        if not checks[name]: raise ValueError(name)
    def check_stop():
        if not read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user'):
            raise InterruptedError('Pause must remain in effect during read-only report verification')
    check_stop()
    status = read_json(run/'STATUS.json'); progress = read_json(run/'PROGRESS.json')
    require('naturally_completed_review_due', status['state']=='review_due' and status['stage']==args.stage)
    for path in [root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock', run/'supervision/TRAINING.lock', run/'supervision/SUPERVISOR.lock']:
        require('released:'+str(path.relative_to(root)), not lock_held(path))
    entry = progress['evaluations'][f'block_{args.stage:06d}_screen']; arena = read_json(inside(root, entry['path']))
    require('arena_sha', sha(inside(root, entry['path'])) == entry['sha256'])
    selected = select_families(read_json(inside(root, plan['suite']['path']))['states'], 16, plan['evaluation_seed'])
    require('suite_digest', arena['suite_digest'] == hashlib.sha256(json.dumps(selected,sort_keys=True).encode()).hexdigest())
    require('development_only', arena['role'] == 'dev')
    require('1376_games', len(arena['games']) == 1376)
    require('16_families', len(arena['layout_families']) == 16)
    native = verify_native(arena, selected, check_stop)
    require('all_native_games', native == arena['native_verification'] and native['games'] == 1376)
    for name, item in arena['checkpoints'].items():
        require('checkpoint:'+name, sha(inside(root, item['path'])) == item['sha256'])
    latest = dict(checkpoint=str((run/'learner/checkpoints'/f'learner_stage_{args.stage:06d}.pt').relative_to(root)))
    saved_boundary=torch.load(root/latest['checkpoint'],map_location='cpu',weights_only=False)
    latest.update(sha256=sha(root/latest['checkpoint']),model_hash=saved_boundary['model_hash'],counters=saved_boundary['counters'],stage=args.stage)
    require('latest_candidate', arena['checkpoints']['candidate']['model_hash'] == latest['model_hash'])
    require('parent_source', arena['checkpoints']['parent'] == plan['source_checkpoint'])
    require('other134_frozen', arena['checkpoints']['control'] == plan['participants']['control'])
    scores = {}
    for i, summary in enumerate(arena['summaries']):
        games = [g for g in arena['games'] if g['matchup'] == i]; groups = defaultdict(list)
        for game in games: groups[game['family']].append(game['score'])
        scores[i] = {family: float(np.mean(rows)) for family, rows in groups.items()}
        require('summary_CI:'+str(i), paired_interval(list(scores[i].values()), arena['seed'] ^ i) == summary['family_score'])
        require('summary_counts:'+str(i), summary['games'] == 86 and summary['wins'] == sum(g['value'] > 0 for g in games) and summary['draws'] == sum(g['value'] == 0 for g in games) and summary['losses'] == sum(g['value'] < 0 for g in games))
    for pair in arena['paired_deltas']:
        first, second = pair['from_matchup'], pair['to_matchup']; common = sorted(scores[first].keys() & scores[second].keys())
        require(f'paired_CI:{first}:{second}', pair['delta'] == paired_interval([scores[second][f]-scores[first][f] for f in common], arena['seed'] ^ (first << 8) ^ second))
    gains = {key: teacher_deltas(arena,before,after) for key,before,after in [('champion','champion','candidate'),('parent','parent','candidate'),('control','control','candidate'),('guard86','guard86','candidate'),('duplicate','parent','repeat')]}
    require('paired_gains', gains == arena['review']['gains'])
    gate = promotion_decision(arena, RootExploreConfig.load(inside(root,plan['parent_config'])))
    require('original_gate', gate == arena['review']['gate'])
    duels = [s['family_score'] for s in arena['summaries'][:4]]; gain_rows = [gains[k] for k in ('champion','parent','control','guard86')]
    require('screening_signal', screening_signal(gain_rows,duels) == arena['review']['screen_promising'])
    require('full_route_unconfirmed', not confirmed_route(gain_rows,gains['duplicate'],duels,gate,16,.02) and not arena['review']['route_confirmed'])
    def traces(name, teacher):
        return {(g['position'],g['candidate_side']):(g['trace'],g['reason'],g['value'],g['score']) for g in arena['games'] if g['candidate']==name and g['opponent']==teacher}
    repeats = 0; correlated = 0; related_pairs = 0; related_by_model = {}
    for teacher in ('teacher0','teacher1'):
        require('repeat_exact:'+teacher, traces('parent',teacher) == traces('repeat',teacher)); repeats += len(traces('parent',teacher))
    for name in ('champion','parent','guard86','control','candidate','repeat'):
        first=traces(name,'teacher0');second=traces(name,'teacher1')
        require('related_pair_alignment:'+name,set(first)==set(second))
        equal=sum(first[key]==second[key] for key in first);correlated+=equal;related_pairs+=len(first)
        related_by_model[name]=dict(pairs=len(first),identical_full_traces=equal)
    for path,digest in plan['source'].items(): require('source:'+path,sha(inside(root,path)) == digest)
    for item in plan['historical_stops']: require('archived_STOP:'+item['path'],sha(inside(root,item['path'])) == item['sha256'])
    for item in plan['frozen_runs']: require('archived_CURRENT:'+item['directory'],sha(inside(root,item['directory'])/'CURRENT.json') == item['current_sha256'])
    payload = torch.load(inside(root,latest['checkpoint']),map_location='cpu',weights_only=False)
    require('full_latest_state', all(k in payload for k in ('model','optimizer','scheduler','counters','rng','config','leaf_value_scale')))
    require('boundary_actual_counters', payload['counters']['iteration'] == plan['source_checkpoint']['iteration']+args.stage and payload['counters']['effective_updates']-plan['source_counters']['effective_updates'] == args.stage*plan['updates_per_stage'])
    require('boundary_new_games',len(progress['collections']['learner']) >= args.stage and status['new_training_games'] >= args.stage*plan['games_per_stage'])
    require('CPU_only',not torch.cuda.is_initialized())
    result = dict(at=utc_now(),all_checks_passed=all(checks.values()),checks=checks,native_games_checked=native['games'],native_plies=native['native_plies'],
                  completed_iteration=plan['source_checkpoint']['iteration']+args.stage,new_training_games=args.stage*plan['games_per_stage'],effective_updates=args.stage*plan['updates_per_stage'],source_checkpoint=plan['source_checkpoint'],current_checkpoint=latest,
                  final_arena_review=entry,route_confirmed=False,screening_signal=arena['review']['screen_promising'],severe_regression=severe_regression(arena),
                  teacher082=arena['summaries'][8]['family_score'],direct_vs_source=duels[1],direct_vs_other134=duels[2],gains=gains,
                  exact_repeat_games=repeats,related_teacher_trace_pairs=related_pairs,related_teacher_identical_traces=correlated,related_teacher_by_model=related_by_model,
                  interpretation='Final182 fixed16 screen shows negative external development gain versus158; no full64 screening signal or production promotion. User paused all training/evaluation and auto-continuation after natural completion. Verification here uses only saved records and native CPU rules: zero new games/updates. Related time-bounded teachers and reused development are not independent final evidence; single learner cannot prove strict update-count causality.')
    atomic_json(out,result)
    print(json.dumps({k:result[k] for k in ('all_checks_passed','native_games_checked','native_plies','teacher082','direct_vs_source','direct_vs_other134','route_confirmed')},ensure_ascii=False))

if __name__ == '__main__': main()
