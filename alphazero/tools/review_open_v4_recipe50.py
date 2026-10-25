"""Read-only CPU review of committed training while its sole owner runs."""
from pathlib import Path
from collections import Counter
import hashlib, math, sys


def main():
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    root = Path(__file__).resolve().parents[1]
    master_path = root / 'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json'
    master = read_json(master_path)
    plan_path = root / master['optimization']['config']
    plan = read_json(plan_path)
    run = root / plan['directory']
    progress = read_json(run / 'PROGRESS.json')
    latest = progress['variants']['learner']
    n = latest['stage']
    checks = {}

    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    def check(name, ok):
        checks[name] = bool(ok)
        if not ok: raise ValueError('supervision check failed: ' + name)

    check('active_authorization', not master['paused_by_user'])
    check('frozen_plan', sha(plan_path) == master['optimization']['config_sha256'])
    check('contract', read_json(run / 'CONTRACT.json') == dict(plan_sha256=sha(plan_path), source=plan['source']))
    check('authorized50_50_recipe_to424', plan['updates_per_stage'] == 64 and plan['target_iteration'] == 424 and plan['training_teacher_ids'] == [0] and plan['learner']['selfplay_fraction'] == .5 and plan['learner']['teacher_selfplay_fraction'] == 0. and not plan['performance_stop_enabled'])
    for path, h in plan['source'].items(): check('source:' + path, sha(root / path) == h)
    for e in plan['historical_stops']: check('old_STOP:' + e['path'], sha(root / e['path']) == e['sha256'])
    for e in plan['frozen_runs']: check('old_CURRENT:' + e['directory'], sha(root / e['directory'] / 'CURRENT.json') == e['current_sha256'])
    runtime=read_json(run/'STATUS.json'); expected_owned=runtime['state']!='horizon_complete'
    for name in ('NEXT_STAGE', 'SUPERVISOR', 'TRAINING'):
        check('global_lock:' + name, lock_held(root / f'runs/open_junqi_v4_repair_pilot/supervision/{name}.lock')==expected_owned)
    for name in ('SUPERVISOR', 'TRAINING'): check('run_lock:' + name, lock_held(run / f'supervision/{name}.lock')==expected_owned)
    check('current_STOP_absent', all(not (root / p).exists() for p in plan['stop_paths']))
    status = read_json(run / 'STATUS.json')
    check('original_single_owner', read_json(run / 'LAUNCH.json')['pid'] == status['pid'] == master['optimization']['pid'])
    check('owner_no_failure', status['state'] not in ('needs_attention', 'stopped_by_user'))
    sys.path.insert(0, str(root / plan['native_build']['directory']))
    import numpy as np, torch
    torch.set_num_threads(2)
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.training_replay import complete_games, trajectory_digest
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_open_v4.arena import select_families
    from junqi_search_v3.loop import read_selfplay_shard
    from evaluate_open_v4_longitudinal import verify_native
    from open_v4_recipe50_contract import small_screen_review
    cfg = RootExploreConfig.load(run / 'learner/config.json')
    previous = torch.load(run / 'learner/checkpoints/learner_stage_000000.pt', map_location='cpu', weights_only=False)
    if n == 0: raise ValueError('No complete fresh training stage yet; use status for baseline supervision')
    records = progress['collections']['learner'][:n]
    check('complete_prefix', len(records) == n and [x['stage'] for x in records] == list(range(1, n + 1)))
    expected_teachers = Counter({x['binary_sha256']: 6 for x in plan['teachers']})
    totals, stages = Counter(), []
    attempted = effective = 0
    for stage, item in enumerate(records, 1):
        check(f's{stage}:shard', sha(root / item['path']) == item['sha256'])
        check(f's{stage}:metadata', sha(root / item['metadata']) == item['metadata_sha256'])
        meta = read_json(root / item['metadata'])
        report = read_json(run / f'learner/reports/stage_{stage:06d}.json')
        checkpoint = run / f'learner/checkpoints/learner_stage_{stage:06d}.pt'
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        check(f's{stage}:complete_state', all(k in saved for k in ('model', 'optimizer', 'scheduler', 'counters', 'rng', 'config', 'leaf_value_scale')))
        check(f's{stage}:exact_config', saved['config'] == cfg.as_dict() and saved['config_hash'] == cfg.digest())
        check(f's{stage}:own_preceding_collector', meta['actor_model_hash'] == previous['model_hash'] and meta['actor_checkpoint']['sha256'] == sha(run / f'learner/checkpoints/learner_stage_{stage-1:06d}.pt'))
        check(f's{stage}:256newTRAIN', len(meta['games']) == meta['game_count'] == 256 and meta['layout_role'] == 'train' and meta['shard_sha256'] == item['sha256'])
        check(f's{stage}:continuity', saved['counters']['iteration'] == plan['source_checkpoint']['iteration'] + stage and meta['iteration'] == plan['collection_start'] + stage - 1)
        check(f's{stage}:64attempts', len(report['updates']) == 64 and saved['counters']['attempted_updates'] - previous['counters']['attempted_updates'] == 64 and report['planned_updates_per_stage'] == 64 and report['policy_step_lr_multiplier'] == 1.)
        accepted = sum(int(u['accepted']) for u in report['updates'])
        check(f's{stage}:effective_count', saved['counters']['effective_updates'] - previous['counters']['effective_updates'] == accepted and saved['counters'] == report['counters'])
        for i, u in enumerate(report['updates']):
            phase = 'value' if (previous['counters']['attempted_updates'] + i) % 3 == 2 else 'policy'
            factors = dict(encoder=.1, policy=0., value=1.) if phase == 'value' else dict(encoder=.25, policy=1., value=0.)
            check(f's{stage}:u{i}:phase_finite', u['phase'] == phase and math.isfinite(u['loss']) and math.isfinite(u['gradient_norm']))
            if u['accepted']: check(f's{stage}:u{i}:actual_original_LR', all(abs(u['optimizer_group_learning_rates'][k] - cfg.learning_rate * u['accepted_lr_scale'] * v) < 1e-15 for k, v in factors.items()))
        identity = meta['teacher_runtime_identity']
        check(f's{stage}:actual12teachers', identity['all_workers_verified'] and identity['worker_counts'] == [6, 6] and Counter(x['binary_sha256'] for x in identity['verified_workers']) == expected_teachers and meta['teacher_provenance'] == plan['teachers'])
        check(f's{stage}:sparse_targets_terminal_verification', meta['root_explore_verification']['all_passed'] and meta['root_explore_verification']['true_terminal_wdl_checked'] and not meta['root_explore_verification']['changed_neural_targets'] and meta['root_explore_audit'].get('deleted_visits', 0) == 0)
        check(f's{stage}:greedy24_full_noise', meta['behavior_contract']['greedy_after_ply'] == 24 and meta['behavior_contract']['root_noise_until_ply'] is None and meta['behavior_contract']['noise_decisions'].get('with_noise', 0) == meta['root_explore_audit']['all_neural_roots'])
        check(f's{stage}:original_actor_hard_labels', meta['policy_target_contract'] == 'teacher' and meta['policy_label_audit']['actor_owned_rows'] == meta['policy_label_audit']['actor_owned_teacher_targets'])
        expected_paths = {x['path'] for x in (plan['initial_replay'] + records[:stage])[-3:]}
        for phase in ('policy', 'value'): check(f's{stage}:{phase}:actual_three_shard_samples', set(report['example_exposure'][phase]['by_shard']) == expected_paths)
        if stage >= 3: check(f's{stage}:old_shards_exited', not any(k.startswith('initial:') and v for k, v in report['actual_policy_replay_origin_sources'].items()))
        mode_ok = True
        for g in meta['games']:
            family = meta['iteration'] * 1000000 + g['index'] // 2
            key = np.random.default_rng(cfg.seed ^ family).random()
            mode = 'selfplay' if key < .5 else 'teacher_vs_actor'
            mode_ok &= g['teacher_id'] == 0 and g['family'] == family and g['mode'] == mode and g['actor_side'] == (1 if g['index'] % 2 == 0 else -1)
            for label, value in (('mode', g['mode']), ('terminal', g['reason']), ('color', g['actor_side'])): totals[f'{label}:{value}'] += 1
        check(f's{stage}:seeded_modes_colors', mode_ok)
        check(f's{stage}:actual_fresh_training082_only', meta['training_recipe'] == {'selfplay': .5, 'teacher_selfplay': 0., 'vs_actual_082': .5} and set(meta['training_teacher_requests']) == {'0'} and meta['training_teacher_provenance'] == plan['teachers'][0] and meta['training_teacher_requests']['0'] == meta['teacher_query_audit']['shadow_actor_owned_requests'] + meta['teacher_query_audit']['actual_teacher_turn_requests'])
        for k in ('physical_inference_rows', 'padding_rows', 'simulation_count', 'total_plies'): totals[k] += meta[k]
        totals['collection_seconds'] += meta['elapsed_seconds']; totals['learning_seconds'] += report['learning_elapsed_seconds']
        stages.append(dict(stage=stage, iteration=plan['source_checkpoint']['iteration']+stage, games=256, attempted_updates=64, effective_updates=accepted, collector_model_hash=meta['actor_model_hash'], model_hash=saved['model_hash'], checkpoint_sha256=sha(checkpoint), policy_replay_origin_sources=report['actual_policy_replay_origin_sources'], actual_policy_target_types=report['actual_policy_target_types'], exposure=report['example_exposure'], root_explore_audit=meta['root_explore_audit'], behavior_metrics=meta['behavior_metrics'], replay_audit=report['replay_audit'], collection_metrics=report['collection_metrics'], learning_seconds=report['learning_elapsed_seconds']))
        attempted += 64; effective += accepted; previous = saved
    check('latest_receipt', sha(root / latest['checkpoint']) == latest['sha256'] and previous['counters'] == latest['counters'] and previous['model_hash'] == latest['model_hash'] and (n == 0 or sha(root / latest['report']) == latest['report_sha256']))
    digest = hashlib.sha256()
    for name, v in sorted(previous['model'].items()): digest.update(name.encode()); digest.update(v.detach().cpu().contiguous().numpy().tobytes())
    check('latest_tensor_digest', digest.hexdigest() == latest['model_hash'])
    check('latest_model_optimizer_finite', all(torch.isfinite(v).all().item() for v in previous['model'].values() if v.is_floating_point()) and all(torch.isfinite(v).all().item() for row in previous['optimizer']['state'].values() for v in row.values() if isinstance(v, torch.Tensor) and v.is_floating_point()))
    check('Value_scale_finite', math.isfinite(previous['leaf_value_scale']) and previous['leaf_value_scale'] > 0)
    backend = MultiLayoutBackend(load_role(root / cfg.layout_directory, 'train'), cfg.seed)
    native_steps, digests = 0, Counter()
    for game, samples in complete_games(read_selfplay_shard(root / records[-1]['path']), meta):
        state = backend.new_state(game['provenance']['setup_seed']); okay = True
        for action, row in zip(game['trace'], samples):
            okay &= not state.is_terminal() and int(state.native_state.key()) == int(row.state_hash) and state.current_player() == row.player and action in state.legal_actions() and np.array_equal(state.legal_actions(), row.legal_actions) and np.array_equal(state.repetition_contexts(), row.repetition_contexts) and np.array_equal(state.feature_planes().astype(np.float16), row.observation) and np.array_equal(row.raw_visits, row.visits) and row.visits.sum() > 0 and np.all(row.visits >= 0)
            if game['mode'] != 'selfplay': okay &= row.visits.sum() == 1 and np.count_nonzero(row.visits) == 1
            state.apply(int(action)); native_steps += 1
        check(f'latest:g{game["index"]}:native_states_labels_terminal_WDL', okay and state.is_terminal() and state.terminal_value(1) == game['result_red'] and state.terminal_reason() == game['reason'])
        digests[trajectory_digest(game, samples, meta['policy_target_contract'])] += 1
    arena_reviews = {}
    selected = select_families(read_json(root / plan['suite']['path'])['states'], 16, plan['evaluation_seed'])
    for name, entry in progress['evaluations'].items():
        check('arena_sha:' + name, sha(root / entry['path']) == entry['sha256'])
        arena = read_json(root / entry['path'])
        check('arena_actual_teachers:' + name, arena['teacher_runtime_identity']['all_workers_verified'] and Counter(x['binary_sha256'] for x in arena['teacher_runtime_identity']['verified_workers']) == expected_teachers)
        if name == 'startup_baseline':
            native = verify_native(arena, selected, lambda: None)
            check('baseline_native344', native == arena['native_verification'] and native['games'] == 344)
            check('baseline_frozen400', all(x['model_hash'] == plan['source_checkpoint']['model_hash'] for x in arena['checkpoints'].values()))
            continue
        if name.endswith('_full') or name == 'iteration424_true084':
            families = 64 if name.endswith('_full') else 16
            native = verify_native(arena, select_families(read_json(root / plan['suite']['path'])['states'], families, plan['evaluation_seed']), lambda: None)
            check('arena_native:' + name, native == arena['native_verification'])
            formal_ref = master['optimization'].get('last_formal_review', {})
            proof = None
            if formal_ref.get('arena') == name:
                proof_path = root / formal_ref['path']
                proof = read_json(proof_path)
                check('arena_independent_review:' + name, sha(proof_path) == formal_ref['sha256'] and proof['all_checks_passed']
                      and proof['qualification_independently_recomputed'] and proof['arena_sha256'] == entry['sha256']
                      and proof['review'] == arena['review'])
            arena_reviews[name] = dict(native_verification=native, requires_separate_statistical_and_formal_gate_review=proof is None,
                                      stored_review=arena['review'], qualification_independently_recomputed=proof is not None)
            if proof is not None:
                arena_reviews[name].update(formal_review=formal_ref, failed_protection=proof['failed_protection'],
                    candidate_teacher_scores=proof['candidate_teacher_scores'], direct_protected_duels=proof['direct_protected_duels'],
                    production_promotion=proof['production_promotion'])
            continue
        native = verify_native(arena, selected, lambda: None)
        check('arena_native:' + name, native == arena['native_verification'] and native['games'] == 344)
        review = small_screen_review(arena, plan['evaluation_seed'], plan['early_stop']['confidence'], progress['segment_parent_screen_score'])
        check('arena_statistics_and_gates:' + name, all(review[k] == arena['review'][k] for k in review))
        arena_reviews[name] = dict(native_verification=native, teacher082=next(s['family_score'] for s in arena['summaries'] if s['candidate'] == 'candidate' and s['opponent'] == 'teacher0'), paired95=review['paired95'], direct95=review['direct95'], positive_signal=review['positive_signal'], promotion=False)
    check('audit_CPU_only', not torch.cuda.is_initialized())
    previous_review = master['optimization'].get('last_review', {})
    previously_reviewed = set()
    if previous_review.get('path') and (root / previous_review['path']).exists():
        previously_reviewed = set(read_json(root / previous_review['path']).get('arena_reviews', {}))
    new_arena_names = set(arena_reviews) - previously_reviewed
    material = any(name.endswith('_full') or name == 'iteration424_true084' or arena_reviews[name]['paired95']['interval'][0] > 0 or arena_reviews[name]['paired95']['interval'][1] < 0 for name in new_arena_names)
    now = utc_now(); current = read_json(run / 'STATUS.json')
    result = dict(at=now, all_checks_passed=all(checks.values()), checks_count=len(checks), checks=checks, pid=current['pid'], source_iteration=plan['source_checkpoint']['iteration'], latest_verified_iteration=plan['source_checkpoint']['iteration']+n, latest_verified_stage=n, current_phase=current['phase'], snapshot_status=current, new_branch_online_games=n*256, new_branch_attempted_updates=attempted, new_branch_effective_updates=effective, prior170to302_online_games=33792, prior170to302_effective_updates=8448, latest_checkpoint=latest, latest_leaf_value_scale=previous['leaf_value_scale'], latest_native_games_verified=256, latest_native_plies_verified=native_steps, latest_full_trajectory_duplicates=sum(v-1 for v in digests.values()), stage_reviews=stages, totals=dict(totals), arena_reviews=arena_reviews, current_frozen_evaluation_new_games=0, current_frozen_evaluation_updates=0, mechanism_changed=True, authorized_recipe_only=True, other_learning_mechanism_changed=False, new_games_generated_by_audit=0, updates_by_audit=0, notify=False, strength_change_material=False, interpretation='Read-only CPU review. Fixed16 results and intervals are recorded per arena; repeated development is not independent final evidence, a training stop or promotion. Completed full gates require separate statistical review. Frozen evaluation adds0training games/updates; genuine084 is distinct from old alias evidence.')
    stamp = now.replace(':', '').replace('-', '').split('.')[0]
    result.update(notify=material, strength_change_material=material, new_arena_results=sorted(new_arena_names))
    out = run / 'supervision/REVIEWS' / f'HEARTBEAT_{stamp}.json'
    atomic_json(out, result)
    for filename in (master_path, root / 'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json'):
        p = read_json(filename)
        p['optimization'].update(training_active=expected_owned, automatic_continuation_enabled=expected_owned,state=current['state'], phase=current['phase'], latest_iteration=plan['source_checkpoint']['iteration']+n, latest_model_hash=latest['model_hash'], new_training_games=n*256, effective_updates=effective, verified_iteration=plan['source_checkpoint']['iteration']+n, verified_new_training_games=n*256, verified_effective_updates=effective, last_review=dict(path=str(out.relative_to(root)), sha256=sha(out), all_checks_passed=True, checks_count=len(checks)), observed_at=now, current_evaluation=current.get('arena') if current['phase'] == 'frozen_development_evaluation' else None)
        p['auto_next_stage'].update(state=current['state'], phase=current['phase'], current_stage=current.get('stage', n), training_active=expected_owned, waiting_controller_active=False, automatic_continuation_enabled=expected_owned)
        p['updated_at'] = now
        p['next_action'] = 'Existing sole owner performs registered monitoring then immediately returns to own-preceding-learner256fresh/64PPV through424. Maintain read-only supervision and REVIEWS; no second spawn or frozen mechanism edits. Fluctuations do not stop; no unapproved method change.'
        atomic_json(filename, p)
    print(dict(all_passed=True, checks=len(checks), review=str(out.relative_to(root)), iteration=plan['source_checkpoint']['iteration']+n, new_games=n*256, attempted_updates=attempted, effective_updates=effective, latest_native_games=256, latest_native_plies=native_steps, phase=current['phase'], arenas=arena_reviews, notify=material))


if __name__ == '__main__': main()
