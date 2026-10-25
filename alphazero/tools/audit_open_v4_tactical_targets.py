"""Replay existing train-only games and measure exact, shallow tactical errors."""
from __future__ import annotations
import hashlib
from collections import Counter
from pathlib import Path
import numpy as np


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now
    from junqi_open_v4.layouts import backend_for, load_role
    from junqi_open_v4.training_replay import complete_games
    from junqi_search_v3.loop import read_selfplay_shard
    root = Path(__file__).resolve().parents[1]
    progress = read_json(root / 'runs/open_junqi_v4_self_training_20261004_023128/PROGRESS.json')
    layouts = {r['id']: r for r in load_role(root / 'config/layouts/open_v4', 'train')}
    report = {'at': utc_now(), 'role': 'train', 'new_games': 0, 'variants': {},
        'definition': 'immediate flag win uses native legal moves; avoidable loss means selected move allows opponent flag capture next turn and at least one legal alternative avoids it. Full history is replayed. This is a tactical diagnostic, not strength or a proof of all causes.'}
    for variant in ('control', 'candidate'):
        item = progress['collections'][variant][-1]
        path = root / item['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item['sha256']
        assert hashlib.sha256((root / item['metadata']).read_bytes()).hexdigest() == item['metadata_sha256']
        metadata = read_json(root / item['metadata'])
        samples = read_selfplay_shard(path)
        counts, cases = Counter(), []
        for game, rows in complete_games(samples, metadata):
            p = game['provenance']
            state = backend_for(layouts[p['red_layout_id']], layouts[p['black_layout_id']],
                mirrored=list(layouts[p['red_layout_id']]['setup']) != p['red_layout']).new_state(p['setup_seed'])
            assert list(state.native_state.board()) == p['initial_board']
            for ply, (row, action) in enumerate(zip(rows, game['trace'])):
                legal = state.legal_actions()
                assert np.array_equal(legal, row.legal_actions)
                assert int(state.native_state.key()) == row.state_hash
                neural = game['mode'] == 'selfplay' or (game['mode'] == 'teacher_vs_actor' and row.player == game['actor_side'])
                if neural:
                    counts['neural_decisions'] += 1
                    board = state.native_state.board()
                    wins = [int(a) for a in legal if board[int(a) % 60] == -3]
                    if wins:
                        counts['immediate_flag_win_available'] += 1
                        mass = sum(int(n) for a, n in zip(legal, row.visits) if int(a) in wins)
                        counts['immediate_win_selected'] += int(action in wins)
                        counts['immediate_win_zero_target'] += int(mass == 0)
                        counts['immediate_win_not_selected'] += int(action not in wins)
                    child = state.clone(); child.apply(action)
                    exposed = False
                    if not child.is_terminal():
                        b = child.native_state.board()
                        exposed = any(b[int(a) % 60] == -3 for a in child.legal_actions())
                    if exposed:
                        counts['selected_allows_next_flag_capture'] += 1
                        # Inspect every exposed root in these two frozen train shards.
                        if True:
                            counts['checked_exposed_roots'] += 1
                            safe = []
                            for a in legal:
                                c = state.clone(); c.apply(int(a))
                                if c.is_terminal():
                                    ok = c.terminal_value(row.player) >= 0
                                else:
                                    b = c.native_state.board()
                                    ok = not any(b[int(reply) % 60] == -3 for reply in c.legal_actions())
                                if ok: safe.append(int(a))
                            counts['avoidable_next_flag_capture'] += int(bool(safe))
                            if safe and len(cases) < 12:
                                cases.append({'family': game['family'], 'ply': ply, 'mode': game['mode'],
                                    'player': row.player, 'provenance': p, 'prefix': game['trace'][:ply],
                                    'action': action, 'safe_actions': safe, 'legal': list(map(int, legal)),
                                    'visits': list(map(int, row.visits)), 'policy_contract': metadata['policy_target_contract']})
                state.apply(action)
            assert state.is_terminal() and state.terminal_reason() == game['reason']
            counts['complete_native_games'] += 1
        report['variants'][variant] = {'source': item, 'counts': dict(counts), 'cases': cases}
        print(variant, dict(counts), flush=True)
    target = root / 'reports/current_training_audit_20261004/TACTICAL_TARGET_AUDIT.json'
    atomic_json(target, report)
    print(target, flush=True)


if __name__ == '__main__': main()
