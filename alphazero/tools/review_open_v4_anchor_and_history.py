"""Read-only result proof and training-only history eligibility audit. No GPU."""
from collections import Counter, defaultdict
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from open_v4_runtime import read_json, atomic_json, utc_now, supervision_status
from run_open_v4_search_training import verify_learning_state, route_signal
from run_open_v4_fresh_data import is_holdout
from run_open_v4_policy_anchor_factor import screening_signal, legacy_guard
from open_v4_experiment_evidence import teacher_deltas
from junqi_open_v4.arena import paired_interval, select_families
from junqi_open_v4.layouts import backend_for
from junqi_open_v4.pipeline import promotion_decision
from junqi_open_v4.config import OpenConfig
from junqi_open_v4.training_replay import replay_pools
from junqi_search_v3.loop import read_selfplay_shard


def main():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan = read_json(root / 'config/experiments/open_v4_policy_anchor_factor_20261004_042349.json')
    run = root / plan['directory']
    status = supervision_status(run)
    assert status['state'] == 'review_due' and not status['training_active'] and not status['supervisor_active']
    assert not any((root / p).exists() for p in plan['stop_paths'])
    for p, digest in plan['source'].items(): assert sha(root / p) == digest, p
    for item in plan['frozen_runs']:
        assert sha(root / item['directory'] / 'CURRENT.json') == item['current_sha256']
    base = OpenConfig.load(root / plan['parent_config'])
    assert sha(root / plan['parent_config']) == plan['parent_config_sha256']
    assert sha(root / base.layout_directory / 'manifest.json') == plan['layout_manifest_hash']
    assert sha(root / plan['suite']['path']) == plan['suite']['sha256']
    for item in plan['initial_replay'] + plan['frozen_collections']:
        assert sha(root / item['path']) == item['sha256']
        assert sha(root / item['metadata']) == item['metadata_sha256']
    for item in plan['participants'].values(): assert sha(root / item['path']) == item['sha256']
    progress = read_json(run / 'PROGRESS.json')
    references = []
    for item in plan['reference_control']:
        stage = item['stage']
        assert sha(root / item['checkpoint']) == item['sha256']
        actual = torch.load(run / f'control/checkpoints/learner_stage_{stage:06d}.pt', map_location='cpu', weights_only=False)
        expected = torch.load(root / item['checkpoint'], map_location='cpu', weights_only=False)
        checks = verify_learning_state(expected, actual)
        assert all(read_json(run / f'verification/control_stage_{stage:06d}.json')['all_learning_state_equal'].values())
        references.append(dict(stage=stage, checks=checks))
        del actual, expected
    final_states, samples = {}, {}
    for variant, record in progress['variants'].items():
        assert record['stage'] == 6 and sha(root / record['checkpoint']) == record['sha256']
        assert sha(root / record['report']) == record['report_sha256']
        saved = torch.load(root / record['checkpoint'], map_location='cpu', weights_only=False)
        required = ('model', 'optimizer', 'scheduler', 'counters', 'config', 'rng', 'leaf_value_scale')
        assert all(k in saved for k in required)
        assert saved['counters']['effective_updates'] == 2048
        assert saved['counters']['backtracking_rejections'] == 0
        assert all(torch.isfinite(t).all() for t in saved['model'].values())
        total = Counter()
        for stage in range(1, 7):
            report = read_json(run / variant / f'reports/stage_{stage:06d}.json')
            total.update(report['actual_policy_batch_sources'])
        samples[variant] = dict(total)
        final_states[variant] = dict(record, full_state_fields=list(required), all_model_tensors_finite=True,
                                    rng_fields=list(saved['rng']))
        del saved
    current = read_json(run / 'CURRENT.json')
    assert current['champion']['model_hash'] == plan['source_checkpoint']['model_hash']
    assert sha(run / current['champion']['path']) == current['champion']['sha256']
    arena_record = progress['evaluations']['block_000006_screen']
    arena_path = root / arena_record['path']
    assert sha(arena_path) == arena_record['sha256']
    arena = read_json(arena_path)
    assert arena['role'] == 'dev' and len(arena['games']) == 1118 and len(arena['layout_families']) == 16
    scores = {}
    for index, summary in enumerate(arena['summaries']):
        games = [g for g in arena['games'] if g['matchup'] == index]
        families = defaultdict(list)
        for game in games:
            assert game['score'] == .5 + .5 * game['value']
            families[game['family']].append(game['score'])
        scores[index] = {f: float(np.mean(rows)) for f, rows in families.items()}
        assert paired_interval(list(scores[index].values()), arena['seed'] ^ index) == summary['family_score']
        assert summary['games'] == len(games)
        assert summary['wins'] == sum(g['value'] > 0 for g in games)
        assert summary['draws'] == sum(g['value'] == 0 for g in games)
        assert summary['losses'] == sum(g['value'] < 0 for g in games)
    for pair in arena['paired_deltas']:
        first, second = pair['from_matchup'], pair['to_matchup']
        common = sorted(scores[first].keys() & scores[second].keys())
        assert paired_interval([scores[second][f] - scores[first][f] for f in common],
                               arena['seed'] ^ (first << 8) ^ second) == pair['delta']
    gains = {k: teacher_deltas(arena, before, after) for k, before, after in (
        ('parent', 'parent', 'candidate'), ('control', 'control', 'candidate'),
        ('duplicate', 'parent', 'repeat'), ('legacy', 'legacy', 'candidate'))}
    gate = promotion_decision(arena, base)
    duels = [r['family_score'] for r in arena['summaries'][:3]]
    signal = screening_signal([gains['parent'], gains['control']], duels[:2]) and legacy_guard(gains['legacy'], duels[2], full=False)
    assert gains == arena['review']['gains'] and gate == arena['review']['gate']
    assert signal == arena['review']['screen_promising'] == False
    result = read_json(run / 'BLOCK_000006_RESULTS.json')
    assert result['screen'] == arena['review'] and result['full'] is None and not result['route_confirmed']
    suite = select_families(read_json(root / plan['suite']['path'])['states'], 16, arena['seed'])
    replay_proof = []
    selected = list(np.linspace(0, len(arena['games']) - 1, 8, dtype=int))
    for index in selected:
        game = arena['games'][int(index)]; desc = suite[game['position']]
        state = backend_for(desc['red'], desc['black']).new_state(desc['setup_seed'])
        for action in desc['trace']: state.apply(action)
        for action in game['trace']: state.apply(action)
        assert state.is_terminal() and state.terminal_reason() == game['reason']
        assert state.terminal_value(game['candidate_side']) == game['value']
        replay_proof.append(dict(game_id=game['id'], reason=game['reason'], value=game['value'], plies=len(game['trace'])))
    stop_hashes = {p: sha(root / p) for p in plan['historical_stop_markers']}
    previous = read_json(root / 'runs/open_junqi_v4_dedup_factor_20261004_035152/supervision/REVIEWS/RESULT_REVIEW_20261004_041357.json')
    assert stop_hashes == previous['original_stop_hashes']
    proof = dict(at=utc_now(), heartbeat_received_at='2026-10-04T05:13:58.504Z', state=status['state'],
        locks_free=True, source_contract_unchanged=True, original_stop_hashes=stop_hashes,
        all_six_control_learning_states_reverified=references, complete_final_states=final_states,
        actual_policy_batch_samples=samples, arena_sha256=sha(arena_path), games_verified=1118,
        all_family_and_paired_intervals_recomputed=True, native_terminal_replays=replay_proof,
        summaries=arena['summaries'], gains=gains, gate=gate, screen_promising=signal,
        route_confirmed=False, full64_not_run=True, champion_unchanged=True,
        conclusion='Teacher-owned Policy anchor improves over weak neural-only control on reused fixed16, but no confirmed advantage over qualified parent or stronger legacy32. No adoption or continuation.')
    target = run / 'supervision/REVIEWS/RESULT_REVIEW_20261004_051358.json'
    atomic_json(target, proof)
    print('RESULT_PROOF ' + str(target.relative_to(root)), flush=True)
    legacy_plan = read_json(root / 'config/experiments/open_v4_dedup_factor_20261004_035152.json')
    audited = []
    for item in legacy_plan['initial_replay'] + legacy_plan['frozen_collections']:
        assert sha(root / item['path']) == item['sha256']
        assert sha(root / item['metadata']) == item['metadata_sha256']
        rows = read_selfplay_shard(root / item['path'])
        meta = read_json(root / item['metadata'])
        values, policy, audit = replay_pools([(rows, meta)], deduplicate=False,
            neural_policy_only=False, holdout=lambda f: is_holdout(f, base.seed))
        counts = Counter()
        for row in policy:
            counts[row.training_policy_source] += 1
            unsafe = bool(row.cycle_entry or row.repetition_count > 1 or np.any(row.repetition_contexts))
            if unsafe: counts[row.training_policy_source + '_history_sensitive'] += 1
            if row.training_policy_source == 'teacher':
                assert np.count_nonzero(row.visits) == 1
                context = int(row.repetition_contexts[int(np.argmax(row.visits))])
                if context: counts['teacher_label_repeated_successor'] += 1
                if context == 2: counts['teacher_label_immediate_repetition_draw'] += 1
        audited.append(dict(source=item, audit=audit, counts=dict(counts)))
        print('HISTORY_AUDIT ' + str(item['path']) + ' ' + str(dict(counts)), flush=True)
        del rows, values, policy
        gc.collect()
    atomic_json(root / 'reports/self_training_preflight/TEACHER_HISTORY_ELIGIBILITY_AUDIT.json',
        dict(at=utc_now(), training_only=True, sources=audited, new_queries=0, new_games=0,
             interpretation='History-sensitive teacher labels are not proved wrong. Teacher sees board/legal but no actual game repetition history; audit supports a preregistered Policy-only eligibility test, with real Value and all history-aware neural targets retained.'))


if __name__ == '__main__': main()
