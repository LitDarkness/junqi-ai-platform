"""CPU-only reconstruction of committed full arena statistics and formal gates."""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import sys


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now
    parser = argparse.ArgumentParser()
    parser.add_argument('--arena', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    masters = [root / f'runs/{name}/supervision/PLAN.json' for name in (
        'open_junqi_v4_repair_pilot', 'open_junqi_v4_fresh_continuation')]
    master = read_json(masters[0])
    plan_path = root / master['optimization']['config']
    plan = read_json(plan_path)
    run = root / plan['directory']
    progress = read_json(run / 'PROGRESS.json')
    entry = progress['evaluations'][args.arena]
    path = root / entry['path']
    arena = read_json(path)
    checks = {}
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

    def check(name, ok):
        checks[name] = bool(ok)
        if not ok:
            raise ValueError('full review check failed: ' + name)

    check('registered_plan', sha(plan_path) == master['optimization']['config_sha256'])
    check('complete_arena_receipt', args.arena.endswith('_full') and sha(path) == entry['sha256'])
    check('64_development_only', arena['role'] == 'dev' and len(arena['layout_families']) == 64)
    check('seed', arena['seed'] == plan['evaluation_seed'])
    for name, digest in plan['source'].items():
        check('frozen_source:' + name, sha(root / name) == digest)
    sys.path.insert(0, str(root / plan['native_build']['directory']))
    import numpy as np
    import torch
    torch.set_num_threads(2)
    from junqi_open_v4.arena import paired_interval, select_families
    from junqi_open_v4.pipeline import promotion_decision
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from evaluate_open_v4_longitudinal import verify_native
    from open_v4_experiment_evidence import teacher_deltas
    from run_open_v4_recipe50 import confirmed_route, screening_signal
    cfg = RootExploreConfig.load(run / 'learner/config.json')
    selected = select_families(read_json(root / plan['suite']['path'])['states'], 64, arena['seed'])
    check('registered_suite', arena['suite_digest'] == hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest())
    matchups = [('candidate', x) for x in ('champion', 'parent', 'control', 'guard86')]
    order = ('champion', 'parent', 'guard86', 'control', 'candidate', 'repeat')
    matchups += [(x, f'teacher{i}') for i in range(2) for x in order]
    check('16_matchups_5184_games', len(arena['games']) == plan['full_games'] == 5184 and len(arena['summaries']) == len(matchups) == 16)
    check('unique_game_ids', len({g['id'] for g in arena['games']}) == 5184)
    summaries, scores = [], {}
    repeat_diagnostics = {}
    for matchup, names in enumerate(matchups):
        games = [g for g in arena['games'] if g['matchup'] == matchup]
        check(f'm{matchup}:names', all((g['candidate'], g['opponent']) == names for g in games))
        check(f'm{matchup}:complete_positions_colors', Counter((g['position'], g['candidate_side']) for g in games) == Counter((i, side) for i in range(len(selected)) for side in (-1, 1)))
        check(f'm{matchup}:layout_identity', all(g['family'] == selected[g['position']]['family'] and g['start_ply'] == selected[g['position']]['start_ply'] for g in games))
        families, starts = defaultdict(list), defaultdict(list)
        for g in games:
            families[g['family']].append(g['score'])
            starts[g['start_ply']].append(g['score'])
        scores[matchup] = {family: float(np.mean(values)) for family, values in families.items()}
        summary = dict(candidate=names[0], opponent=names[1], games=len(games),
            wins=sum(g['value'] > 0 for g in games), draws=sum(g['value'] == 0 for g in games),
            losses=sum(g['value'] < 0 for g in games), score=float(np.mean([g['score'] for g in games])),
            family_score=paired_interval(list(scores[matchup].values()), arena['seed'] ^ matchup),
            score_by_start_ply={str(k): float(np.mean(v)) for k, v in starts.items()},
            repetition_fraction=float(np.mean([g['reason'] == 'REPETITION' for g in games])))
        check(f'm{matchup}:reconstructed_summary', summary == arena['summaries'][matchup])
        summaries.append(summary)
    pairs = []
    for first in range(len(matchups)):
        for second in range(first + 1, len(matchups)):
            if matchups[first][1] != matchups[second][1]:
                continue
            common = sorted(scores[first].keys() & scores[second].keys())
            pairs.append(dict(from_matchup=first, to_matchup=second,
                delta=paired_interval([scores[second][f] - scores[first][f] for f in common], arena['seed'] ^ (first << 8) ^ second)))
    check('all_paired_intervals_reconstructed', pairs == arena['paired_deltas'])
    reconstructed = dict(arena, summaries=summaries, paired_deltas=pairs)
    gains = {key: teacher_deltas(reconstructed, before, after) for key, before, after in (
        ('champion', 'champion', 'candidate'), ('parent', 'parent', 'candidate'),
        ('control', 'control', 'candidate'), ('guard86', 'guard86', 'candidate'), ('duplicate', 'parent', 'repeat'))}
    gate = promotion_decision(reconstructed, cfg)
    duels = [s['family_score'] for s in summaries[:4]]
    review = dict(gains=gains, gate=gate,
        screen_promising=screening_signal([gains[k] for k in ('champion', 'parent', 'control', 'guard86')], duels),
        route_confirmed=confirmed_route([gains[k] for k in ('champion', 'parent', 'control', 'guard86')], gains['duplicate'], duels, gate, 64, .02), screen_only=False)
    check('all_original_formal_gates', review == arena['review'] == entry['review'])
    native = verify_native(arena, selected, lambda: None)
    check('native_legal_terminal_score', native == arena['native_verification'])
    for name, receipt in arena['checkpoints'].items():
        saved = torch.load(root / receipt['path'], map_location='cpu', weights_only=False)
        check('checkpoint:' + name, sha(root / receipt['path']) == receipt['sha256'] and saved['model_hash'] == receipt['model_hash'] and all(k in saved for k in ('model', 'optimizer', 'scheduler', 'counters', 'rng', 'config', 'leaf_value_scale')))
        digest = hashlib.sha256()
        for key, tensor in sorted(saved['model'].items()):
            digest.update(key.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        check('model_tensor_hash:' + name, digest.hexdigest() == receipt['model_hash'])
    identities = arena['teacher_runtime_identity']
    check('genuine082084_6plus6', identities['all_workers_verified'] and identities['worker_counts'] == [6, 6] and Counter(x['binary_sha256'] for x in identities['verified_workers']) == Counter({t['binary_sha256']: 6 for t in plan['teachers']}))
    check('256_BF16_physical32_no_noise_guard', arena['inference_contract'] == dict(physical_batch=32, precision='BF16', padding='repeat final input, discard padded outputs', root_noise=False, root_guard=False, budget=256, historical_dynamic_batch_scores_not_same_compute_context=True))
    for name, metrics in arena['search_metrics'].items():
        check('fresh256_no_cache_reuse:' + name, metrics['simulations'] == 256 * metrics['moves'] and all(metrics[k] == 0 for k in ('cache_hits', 'deduplicated_rows', 'reused_roots', 'inherited_root_visits')))
    for teacher in ('teacher0', 'teacher1'):
        def keyed(candidate):
            return {(g['position'], g['candidate_side']): (g['trace'], g['value'], g['reason']) for g in arena['games'] if g['candidate'] == candidate and g['opponent'] == teacher}
        original, repeat = keyed('parent'), keyed('repeat')
        repeat_diagnostics[teacher] = dict(games=len(original), same_complete_trajectory=sum(original[k] == repeat[k] for k in original), score_difference=gains['duplicate'][int(teacher[-1])]['mean'])
    failed = [f'{name}_teacher{i}_paired95_lower_not_above0' for name in ('champion', 'parent', 'control', 'guard86') for i, row in enumerate(gains[name]) if row['ci95'][0] <= 0]
    failed += [f'direct_{matchups[i][1]}_95_lower_not_above50pct' for i, row in enumerate(duels) if row['ci95'][0] <= .5]
    failed += [f'source_repeat_teacher{i}_drift_above2pp' for i, row in enumerate(gains['duplicate']) if abs(row['mean']) > .02]
    if not gate['champion_accepted']: failed.append('original_Champion_gate')
    check('adoption_conjunction', review['route_confirmed'] == (not failed))
    current = read_json(run / 'CURRENT.json')
    if not review['route_confirmed']:
        check('no_unqualified_promotion', current['champion']['model_hash'] == arena['checkpoints']['champion']['model_hash'])
    check('CPU_only_no_training', not torch.cuda.is_initialized())
    result = dict(at=utc_now(), arena=entry['path'], arena_sha256=sha(path), all_checks_passed=True, checks_count=len(checks), checks=checks,
        native_verification=native, qualification_independently_recomputed=True, review=review, failed_protection=failed,
        candidate_teacher_scores={s['opponent']: s['family_score'] for s in summaries if s['candidate'] == 'candidate' and s['opponent'].startswith('teacher')},
        direct_protected_duels={matchups[i][1]: row for i, row in enumerate(duels)}, source_repeat=repeat_diagnostics,
        production_promotion=review['route_confirmed'], current_registered_champion=current['champion'],
        new_training_games=0, effective_updates=0, mechanism_changed=False,
        interpretation='Independent reconstruction from saved games, original native terminal replay and registered gates. Repeated development and related teachers are not independent final evidence. Full adoption requires every registered protection; continuation through400 is unchanged.')
    output = run / 'supervision/REVIEWS' / f'FULL_{args.arena.upper()}_REVIEW.json'
    atomic_json(output, result)
    ref = dict(path=str(output.relative_to(root)), sha256=sha(output), arena=args.arena, all_checks_passed=True, checks_count=len(checks), route_confirmed=review['route_confirmed'])
    for filename in masters:
        p = read_json(filename)
        p['optimization']['last_formal_review'] = ref
        p['optimization']['last_strength_screen']['full_gate_pending'] = False
        p['optimization']['last_strength_screen']['full_gate_route_confirmed'] = review['route_confirmed']
        atomic_json(filename, p)
    print(json.dumps(dict(review=ref, candidate_teacher_scores=result['candidate_teacher_scores'], direct_protected_duels=result['direct_protected_duels'], parent_deltas=gains['parent'], failed_protection=failed, native_verification=native, production_promotion=result['production_promotion']), ensure_ascii=False))


if __name__ == '__main__':
    main()
