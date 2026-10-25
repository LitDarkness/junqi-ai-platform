"""Read committed first stage only; no teacher queries, games or learning."""
import argparse
from collections import Counter
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    from open_v4_runtime import read_json, atomic_json, utc_now
    root = Path(__file__).resolve().parents[1]
    plan = read_json(root / args.plan)
    sys.path.insert(0, str(root / plan['native_build']['directory']))
    from audit_open_v4_reversion_startup import main as native_first_stage
    native_first_stage()
    import numpy as np
    import torch
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_open_v4.training_replay import complete_games
    run = root / plan['directory']
    proof = read_json(root / args.output)
    checks = proof['checks']

    def check(name, value):
        checks[name] = bool(value)
        if not value:
            raise ValueError('first corrected-teacher audit failed: ' + name)

    progress = read_json(run / 'PROGRESS.json')
    item = next(x for x in progress['collections']['learner'] if x['stage'] == 1)
    meta = read_json(root / item['metadata'])
    samples = read_selfplay_shard(root / item['path'])
    check('absolute_first_iteration303', proof['source']['iteration'] == 302)
    check('new_namespace1244', meta['iteration'] == plan['collection_start'] == 1244)
    identity = meta['teacher_runtime_identity']
    check('all12actual_workers', identity['all_workers_verified'] and len(identity['verified_workers']) == 12)
    check('two_dedicated_teacher_pools', identity['worker_counts'] == [6, 6])
    actual = Counter(x['binary_sha256'] for x in identity['verified_workers'])
    expected = Counter({x['binary_sha256']: 6 for x in plan['teachers']})
    check('actual082_and084_original_binary_digests', actual == expected)
    check('teacher_wrapper_provenance', meta['teacher_provenance'] == plan['teachers'])
    check('search_target_verification', meta['root_explore_verification']['all_passed'])
    check('no_policy_target_pruning', not meta['root_explore_verification']['changed_neural_targets'])
    check('no_extra_search_simulations', meta['root_explore_contract']['additional_simulations'] == 0)
    check('retained_k2_forced_exploration', meta['root_explore_contract']['forced_coeff'] == 2.0)
    check('greedy24_noise_all_plies', meta['behavior_contract']['greedy_after_ply'] == 24 and meta['behavior_contract']['root_noise_until_ply'] is None)
    targets = Counter()
    modes = Counter()
    colors = Counter()
    actual_teacher_games = Counter()
    digests = Counter()
    import hashlib
    cfg = read_json(run / 'learner/config.json')
    for game, rows in complete_games(samples, meta):
        modes[game['mode']] += 1
        colors[game['actor_side']] += 1
        actual_teacher_games[game['teacher_id']] += int(game['mode'] != 'selfplay')
        family = meta['iteration'] * 1_000_000 + game['index'] // 2
        key = np.random.default_rng(cfg['seed'] ^ family).random()
        expected_mode = 'selfplay' if key < .25 else ('teacher_selfplay' if key < .5 else 'teacher_vs_actor')
        check(f"game{game['index']}:paired_namespace_mode", game['family'] == family and game['mode'] == expected_mode)
        digests[hashlib.sha256(str((game['provenance']['initial_board'], game['trace'])).encode()).hexdigest()] += 1
        for row in rows:
            label = 'search' if game['mode'] == 'selfplay' else 'teacher'
            targets[label] += 1
            check(f"game{game['index']}:sparse{row.ply}", row.visits.shape == row.legal_actions.shape and np.array_equal(row.raw_visits, row.visits) and np.all(row.visits >= 0) and row.visits.sum() > 0)
            if label == 'teacher':
                check(f"game{game['index']}:hard_label{row.ply}", row.visits.sum() == 1 and np.count_nonzero(row.visits) == 1)
    check('both_actual_teachers_generate_fresh_data', all(actual_teacher_games[i] > 0 for i in (0, 1)))
    check('balanced_actor_colors', colors == {1: 128, -1: 128})
    report = read_json(run / 'learner/reports/stage_000001.json')
    check('actual64PPV_original_policy_rate', report['planned_updates_per_stage'] == 64 and report['policy_step_lr_multiplier'] == 1.0)
    check('all_model_parameters_finite', all(torch.isfinite(x).all().item() for x in torch.load(run / 'learner/checkpoints/learner_stage_000001.pt', map_location='cpu', weights_only=False)['model'].values() if x.is_floating_point()))
    baseline = read_json(run / 'arena/startup_baseline.json')
    check('verified344_frozen_baseline', baseline['native_verification']['games'] == 344 and baseline['review']['new_training_games'] == 0)
    keyed_repeats = {}
    for teacher in ('teacher0', 'teacher1'):
        aligned = {}
        for role in ('parent', 'repeat'):
            aligned[role] = {(g['position'], g['candidate_side']): g for g in baseline['games'] if g['candidate'] == role and g['opponent'] == teacher}
        check('baseline_paired_keys:' + teacher, aligned['parent'].keys() == aligned['repeat'].keys() and len(aligned['parent']) == 86)
        same = sum((g['trace'], g['value']) == (aligned['repeat'][key]['trace'], aligned['repeat'][key]['value']) for key, g in aligned['parent'].items())
        keyed_repeats[teacher] = dict(games=86, identical_trace_and_result=same,
                                    comparison_key='saved position and candidate color, independent of finish order')
    proof.update(at=utc_now(), all_checks_passed=all(checks.values()), checks_count=len(checks), teacher_runtime_identity=identity,
                 source_iteration=302, first_committed_iteration=303, actual_modes=dict(modes), actual_actor_colors=dict(colors),
                 teacher_game_counts=dict(actual_teacher_games), actual_target_rows=dict(targets),
                 full_trajectory_duplicate_games=sum(n - 1 for n in digests.values()),
                 collection_metrics=report['collection_metrics'], actual_policy_replay_origin_sources=report['actual_policy_replay_origin_sources'],
                 actual_policy_target_types=report['actual_policy_target_types'], example_exposure=report['example_exposure'],
                 frozen_baseline_native_verification=baseline['native_verification'], frozen_baseline_keyed_repeats=keyed_repeats,
                 new_games_in_this_audit=0, updates_in_this_audit=0)
    atomic_json(root / args.output, proof)
    atomic_json(run / 'supervision/REVIEWS/FIRST_CORRECTED_TEACHER_TRAINING.json', proof)
    print({k: proof[k] for k in ('all_checks_passed', 'checks_count', 'first_committed_iteration', 'native_games', 'native_plies', 'effective_updates')})


if __name__ == '__main__':
    main()
