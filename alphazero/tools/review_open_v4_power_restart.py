"""CPU-only integrity review before the one explicit power-interruption resume."""
from pathlib import Path
import hashlib
import math
import sys


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    root = Path(__file__).resolve().parents[1]
    master = read_json(root / 'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    plan_path = root / master['optimization']['config']
    plan = read_json(plan_path)
    run = root / plan['directory']
    evidence = run / 'supervision/REVIEWS/POWER_INTERRUPTION_20261006'
    evidence.mkdir(parents=True, exist_ok=True)
    checks = {}
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def check(name, value):
        checks[name] = bool(value)
        if not value:
            atomic_json(evidence / 'PRE_RESUME_FAILURE.json', dict(at=utc_now(), checks=checks))
            raise ValueError(name)
    check('user_not_paused', not master['paused_by_user'])
    check('registered_plan_unchanged', sha(plan_path) == master['optimization']['config_sha256'])
    check('current_STOP_absent', all(not (root / p).exists() for p in plan['stop_paths']))
    check('not_already_resumed', not (evidence / 'RESUME_ATTEMPT.json').exists())
    check('not_normal_completed_endpoint', read_json(run / 'STATUS.json')['state'] not in ('horizon_complete', 'review_due', 'segment_complete'))
    check('global_owner_released', not lock_held(root / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
    for directory in [run] + [root / p['directory'] for p in plan['frozen_runs']]:
        for name in ('SUPERVISOR', 'TRAINING'):
            check('released:' + str(directory.relative_to(root)) + ':' + name,
                  not lock_held(directory / f'supervision/{name}.lock'))
    check('source_contract', read_json(run / 'CONTRACT.json') == dict(plan_sha256=sha(plan_path), source=plan['source']))
    for path, value in plan['source'].items(): check('frozen_source:' + path, sha(root / path) == value)
    for item in plan['historical_stops']: check('historical_STOP:' + item['path'], sha(root / item['path']) == item['sha256'])
    for item in plan['frozen_runs']: check('historical_CURRENT:' + item['directory'], sha(root / item['directory'] / 'CURRENT.json') == item['current_sha256'])
    for item in plan['participants'].values(): check('reference:' + item['path'], sha(root / item['path']) == item['sha256'])
    progress = read_json(run / 'PROGRESS.json')
    latest = progress['variants']['learner']
    check('last_complete363', latest['stage'] == 1 and latest['counters']['iteration'] == 363)
    check('latest_checkpoint_SHA', sha(root / latest['checkpoint']) == latest['sha256'])
    check('latest_report_SHA', sha(root / latest['report']) == latest['report_sha256'])
    for key in ('first_complete_training_review', 'last_review'):
        item = master['optimization'][key]
        proof = read_json(root / item['path'])
        check('prior_review:' + key, sha(root / item['path']) == item['sha256'] and proof['all_checks_passed'])
    import numpy as np
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0, str(root / plan['native_build']['directory']))
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role, backend_for
    from junqi_open_v4.training_replay import complete_games
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_open_v4.arena import select_families
    from junqi_open_v4.arena_journal import ArenaJournal
    from run_open_v4_engineered import ordered_participants
    saved = torch.load(root / latest['checkpoint'], map_location='cpu', weights_only=False)
    for field in ('model', 'optimizer', 'scheduler', 'counters', 'config', 'rng', 'leaf_value_scale'):
        check('complete_state:' + field, field in saved)
    check('counter_commit', saved['counters'] == latest['counters'])
    check('config_preserved', saved['config'] == read_json(run / 'learner/config.json'))
    check('Value_scale_finite', math.isfinite(saved['leaf_value_scale']) and saved['leaf_value_scale'] > 0)
    model_digest = hashlib.sha256()
    for name, value in sorted(saved['model'].items()):
        model_digest.update(name.encode()); model_digest.update(value.contiguous().numpy().tobytes())
        check('model_finite:' + name, not value.is_floating_point() or torch.isfinite(value).all().item())
    check('model_digest_preserved', model_digest.hexdigest() == latest['model_hash'] == saved['model_hash'])
    check('optimizer_finite', all(torch.isfinite(v).all().item() for row in saved['optimizer']['state'].values() for v in row.values() if isinstance(v, torch.Tensor) and v.is_floating_point()))
    check('all_RNG_saved', all(k in saved['rng'] for k in ('python', 'numpy', 'torch', 'cuda')))
    receipt = progress['collections']['learner'][0]
    check('raw_shard_SHA', sha(root / receipt['path']) == receipt['sha256'])
    check('metadata_SHA', sha(root / receipt['metadata']) == receipt['metadata_sha256'])
    meta = read_json(root / receipt['metadata'])
    stage0 = torch.load(run / 'learner/checkpoints/learner_stage_000000.pt', map_location='cpu', weights_only=False)
    check('own_preceding_collector', meta['actor_model_hash'] == stage0['model_hash'] and meta['actor_checkpoint']['sha256'] == sha(run / 'learner/checkpoints/learner_stage_000000.pt'))
    check('256_actual_new_terminal_games', meta['game_count'] == len(meta['games']) == 256 and meta['layout_role'] == 'train')
    report = read_json(root / latest['report'])
    check('64_actual_accepted_updates', len(report['updates']) == sum(bool(u['accepted']) for u in report['updates']) == 64 and saved['counters']['effective_updates'] - stage0['counters']['effective_updates'] == 64)
    check('actual_original_LRs', all(all(abs(u['optimizer_group_learning_rates'][k] - saved['config']['learning_rate'] * u['accepted_lr_scale'] * v) < 1e-15 for k, v in (dict(encoder=.1, policy=0., value=1.) if u['phase'] == 'value' else dict(encoder=.25, policy=1., value=0.)).items()) for u in report['updates']))
    backend = MultiLayoutBackend(load_role(root / saved['config']['layout_directory'], 'train'), saved['config']['seed'])
    plies = 0
    for game, samples in complete_games(read_selfplay_shard(root / receipt['path']), meta):
        state = backend.new_state(game['provenance']['setup_seed']); ok = True
        for action, row in zip(game['trace'], samples):
            ok &= (not state.is_terminal() and state.current_player() == row.player and int(state.native_state.key()) == int(row.state_hash) and action in state.legal_actions() and np.array_equal(row.raw_visits, row.visits))
            state.apply(int(action)); plies += 1
        check('TRAIN_terminal:' + str(game['index']), ok and state.is_terminal() and state.terminal_value(1) == game['result_red'] and state.terminal_reason() == game['reason'])
    participants = ordered_participants(dict(candidate=plan['source_checkpoint'], control=plan['participants']['control'], parent=plan['participants']['parent'], repeat=plan['participants']['parent'], guard86=plan['participants']['guard86'], champion=progress['champion']))
    suite = select_families(read_json(root / plan['suite']['path'])['states'], 64, plan['evaluation_seed'])
    matchups = [('candidate', 'champion'), ('candidate', 'parent'), ('candidate', 'control'), ('candidate', 'guard86')] + [(key, f'teacher{i}') for i in range(len(plan['teachers'])) for key in participants]
    initial = []
    for position, descriptor in enumerate(suite):
        for matchup, (candidate, opponent) in enumerate(matchups):
            for side in (1, -1):
                state = backend_for(descriptor['red'], descriptor['black']).new_state(descriptor['setup_seed'])
                for action in descriptor['trace']: state.apply(int(action))
                check('DEV_start:' + str(len(initial)), not state.is_terminal() and int(state.native_state.key()) == descriptor['state_key'])
                initial.append(dict(id=len(initial), position=position, family=descriptor['family'], start_ply=descriptor['start_ply'], matchup=matchup, candidate=candidate, opponent=opponent, candidate_side=side, state=state, trace=[], candidate_seconds=0., opponent_seconds=0.))
    journal_dir = run / 'arena_chunks/pending_iteration362_full'
    journal = ArenaJournal(journal_dir, suite, matchups, {key: 256 for key in participants}, plan['evaluation_seed'], dict(plan_sha256=sha(plan_path), participants=participants, teachers=plan['teachers'], physical_batch=32, precision='BF16', source=plan['source']))
    active, completed, snapshot = journal.restore(initial)
    check('all5184_arena_games_bound_and_reconstructed', len(active) + len(completed) == 5184)
    check('full_gate_not_committed', 'pending_iteration362_full' not in progress['evaluations'])
    check('CPU_only_no_GPU_task', not torch.cuda.is_initialized())
    original = {}
    for path in [run / 'STATUS.json', run / 'LAUNCH.json', run / 'PROGRESS.json', journal_dir / 'MANIFEST.json', root / plan['user_contract']['path'].replace('AUTHORIZATION.json', 'SPAWN_ONCE.json')]:
        label = path.name if path.name != 'MANIFEST.json' else 'ARENA_MANIFEST.json'
        target = evidence / ('ORIGINAL_' + label)
        if not target.exists(): target.write_bytes(path.read_bytes())
        original[str(path.relative_to(root))] = dict(sha256=sha(path), archive=str(target.relative_to(root)))
    proof = dict(at=utc_now(), cause='Human confirmed shutdown/reboot: 关机重启了; no Python exception or numerical failure recorded.', all_checks_passed=True, checks_count=len(checks), checks=checks, original_artifacts=original, current_checkpoint=latest, online_games=256, effective_updates=64, TRAIN_native_plies_verified=plies, evaluation_candidate_iteration=362, learning_iteration_preserved=363, evaluation_rounds=snapshot['rounds'], evaluation_completed_games=len(completed), evaluation_active_games=len(active), mechanism_changed=False, new_training_games_by_review=0, updates_by_review=0, explicit_resume_limit=1)
    atomic_json(evidence / 'PRE_RESUME_VERIFIED.json', proof)
    print(dict(all_passed=True, checks=len(checks), iteration=363, training_games=256, updates=64, native_plies=plies, arena_rounds=snapshot['rounds'], completed=len(completed), active=len(active), proof=str((evidence / 'PRE_RESUME_VERIFIED.json').relative_to(root))))


if __name__ == '__main__': main()
