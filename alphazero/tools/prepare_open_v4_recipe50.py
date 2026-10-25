"""Register only the authorized50/50 recipe from latest complete400."""
from pathlib import Path
import copy, hashlib, json, unittest
ROOT = Path(__file__).resolve().parents[1]


def main():
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_recipe50_contract import validate_source_config, fork_stable_payload, verify_stable_seed, verify_single_prerequisite
    torch.set_num_threads(2)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def entry(p):
        p = Path(p); p = p if p.is_absolute() else ROOT / p
        return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
    master = read_json(ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old = read_json(ROOT / master['optimization']['config'])
    oldrun = ROOT / old['directory']
    progress = read_json(oldrun / 'PROGRESS.json')
    item = progress['variants']['learner']
    checks = {}
    def check(name, ok):
        checks[name] = bool(ok)
        if not ok: raise ValueError('recipe registration check: ' + name)
    check('old400_horizon_completed_not_paused', not master['paused_by_user'] and read_json(oldrun / 'STATUS.json')['state'] == 'horizon_complete' and read_json(oldrun / 'COMPLETION.json')['target_iteration'] == 400)
    check('latest_complete400', item['counters']['iteration'] == 400 and sha(ROOT / item['checkpoint']) == item['sha256'])
    for path in [ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'] + [ROOT / e['directory'] / f'supervision/{n}.lock' for e in old['frozen_runs'] + [{'directory': old['directory']}] for n in ('SUPERVISOR', 'TRAINING')]:
        check('unowned:' + str(path.relative_to(ROOT)), not lock_held(path))
    for path, digest in old['source'].items(): check('frozen_source:' + path, sha(ROOT / path) == digest)
    for e in old['frozen_runs']: check('frozen_CURRENT:' + e['directory'], sha(ROOT / e['directory'] / 'CURRENT.json') == e['current_sha256'])
    source = torch.load(ROOT / item['checkpoint'], map_location='cpu', weights_only=False)
    stamp = utc_now().replace('-', '').replace(':', '').replace('T', '_')[:15]
    out = ROOT / f'reports/recipe50_{stamp}'; out.mkdir()
    directory = f'runs/open_junqi_v4_recipe50_{stamp}'
    planpath = ROOT / f'config/experiments/open_v4_recipe50_{stamp}.json'
    basepath = ROOT / f'config/experiments/open_v4_recipe50_base_{stamp}.json'
    atomic_json(out / 'PRIOR_MASTER.json', master)
    atomic_json(basepath, source['config'])
    cfg = RootExploreConfig.load(basepath)
    cfg.run_directory = directory + '/learner'; cfg.initial_checkpoint = item['checkpoint']
    cfg.selfplay_fraction = .5; cfg.teacher_selfplay_fraction = 0.
    cfg.validate()
    changed = validate_source_config(source, cfg)
    migrated = fork_stable_payload(source, cfg, dict(source=item, approved_recipe=True, target_iteration=424, no_weight_reset=True))
    statechecks = verify_stable_seed(source, migrated)
    check('full_model_Adam_scheduler_counters_allRNG_Value_originalLR_preserved', all(statechecks.values()))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_open_v4_recipe50.py')
    test_result = unittest.TextTestRunner(verbosity=2).run(suite)
    check('recipe_routing_state_and_formal_protection_tests', test_result.wasSuccessful() and test_result.testsRun == 4)
    tests = dict(tests_run=test_result.testsRun, failures=len(test_result.failures), errors=len(test_result.errors), all_passed=test_result.wasSuccessful())
    for e in progress['collections']['learner'][-3:]:
        check('last3TRAIN:' + e['path'], sha(ROOT / e['path']) == e['sha256'] and sha(ROOT / e['metadata']) == e['metadata_sha256'] and read_json(ROOT / e['metadata'])['layout_role'] == 'train')
    stops = {e['path']:e['sha256'] for e in old['historical_stops']}
    for path in (ROOT / 'runs').rglob('STOP'):
        if path.is_file(): stops[str(path.relative_to(ROOT))] = sha(path)
    bench = ROOT / 'reports/search_engineering_20261006/BENCHMARK_RESULTS.json'
    check('unchanged_evaluation_graph_exact_proof', read_json(bench)['all_exact'] and read_json(bench)['revision_invalidates_graphs'])
    original_tests = read_json(ROOT / 'reports/search_engineering_20261006/TEST_RESULTS.json')
    check('unchanged_native_rules_and_resumable_arena_tests', original_tests['all_passed'])
    boundary = out / 'BOUNDARY_AUDIT.json'
    atomic_json(boundary, dict(all_checks_passed=all(checks.values()), checks=checks, source=item, source_state_preservation=statechecks, changed_config_fields=changed, training_teacher_id=0, old_run=old['directory'], old_current_sha256=sha(oldrun / 'CURRENT.json'), initial_replay=progress['collections']['learner'][-3:]))
    cpu = out / 'CPU_VERIFICATION.json'
    atomic_json(cpu, dict(all_passed=all(checks.values()), state_checks=statechecks, recipe_tests=tests, retained_native_arena_tests=original_tests, zero_training_games=True, zero_updates=True, no_cuda_initialized=not torch.cuda.is_initialized()))
    auth = out / 'AUTHORIZATION.json'
    atomic_json(auth, dict(at=utc_now(), request='我觉得可以更改 recipe 试一试，例如我觉得可以尝试把教师互相打改成 0 ，全部换成自我博弈：1. 50% 自我博弈 2. 50% vs Heu082 请你进行更改和训练', recipe={'selfplay':.5, 'teacher_selfplay':0., 'vs_actual_082':.5}, allow_other_learning_changes=False, no_weight_reset=True, continue_despite_strength_decline=True, pilot_horizon_chosen=424, scope='Recipe experiment: latest complete400, exact model/Adam/scheduler/counters/allRNG/Value and originalLR/64PPV/search/labels. Single new run24 stages then review. Initial3raw TRAIN shards exit after3fresh stages. Actual084 excluded from fresh training and retained for evaluation; strict formal gates unchanged.'))
    current = dict(path=item['checkpoint'], sha256=item['sha256'], model_hash=item['model_hash'], iteration=400, leaf_value_scale=source['leaf_value_scale'])
    plan = copy.deepcopy(old)
    for name in ('frozen_baseline', 'pending_screen'): plan.pop(name, None)
    plan.update(schema='junqi-recipe50-400to424-plan-1', route='recipe50_selfplay_vs082', registered_at=utc_now(), directory=directory, parent_config=str(basepath.relative_to(ROOT)), parent_config_sha256=sha(basepath), source_checkpoint=current, source_counters=source['counters'], source_config_hash=source['config_hash'], maximum_stages=24, target_iteration=424, screen_boundaries=[6,12,24], collection_start=read_json(ROOT / progress['collections']['learner'][-1]['metadata'])['iteration']+1, initial_replay=progress['collections']['learner'][-3:], registered_maximum_new_games=6144, registered_maximum_updates=1536, startup_baseline_games=344, benchmark=entry(bench), user_contract=entry(auth), cpu_verification=entry(cpu), boundary_audit=entry(boundary), training_teacher_ids=[0])
    plan['learner'].update(selfplay_fraction=.5, teacher_selfplay_fraction=0.)
    plan['participants']['parent'] = copy.deepcopy(current)
    plan['participants']['champion'] = copy.deepcopy(progress['champion'])
    plan['initial_source_teacher_score'] = .703125
    plan['stop_paths'] = [directory + '/STOP', directory + '/supervision/STOP']
    plan['historical_stops'] = [dict(path=p, sha256=h) for p,h in sorted(stops.items())]
    plan['historical_stop_markers'] = list(stops)
    plan['frozen_runs'].append(dict(directory=old['directory'], current_sha256=sha(oldrun / 'CURRENT.json')))
    plan['previous_experiment'] = dict(plan=entry(ROOT / master['optimization']['config']), completed400_report=master['optimization']['final_report'])
    plan['authorization'] = 'Latest human explicitly approves changing recipe to50%selfplay/50%vsHeu082 and starting training. No other learning hyperparameter changes.'
    plan['source_role'] = 'Latest complete400 full-state weight source; frozen400/Champion230/candidate134/control86 protect adoption.'
    plan['horizon'] = '24fresh stages400to424; fixed6/12/24 screens, positive signal only for full5184. Negative or uncertain scores never stop the pilot; endpoint requires review, no automatic extension.'
    plan['frozen_reference_definition'] = 'parent/repeat: frozen complete400. candidate: single evolving recipe50 learner. champion230/control134/guard86 are frozen historical protections, not simultaneous online controls.'
    plan['interpretation'] = 'Fresh games use50%selfplay/50%vsactual082 probabilities, teacher selfplay0. This changes recipe and selected training teacher together; no single-factor causal claim. Game probabilities are not guaranteed50/50 realized games, rows or Policy/Value exposure. Original hard teacher labels and true sparse selfplay visits are retained. Actual084 still evaluated; no final-layout selection.'
    plan.update(hypothesis='Increase fresh selfplay supervision while removing teacher selfplay and selecting actual082 as the sole training opponent; evaluate improvement versus frozen400 and retain actual084 protection.',
                efficiency='Only authorized recipe/routing changes. Original learning rate,64PPV ratio,64/128 training search and validated fixed32/BF16 evaluation retained; actual timing and exposure measured.',
                teacher_repair_effect='Actual082 only in fresh training; both original isolated082/084 binaries retained in evaluation. Historical alias evidence and inherited raw labels unchanged.',
                baseline_registration='Frozen400 parent/repeat versus actual082 and actual084 before fresh training. This measured source baseline stays fixed during the pilot.',
                registered_first_segment_new_games=1536, registered_first_segment_updates=384)
    plan['reporting_contract'].update(cumulative='Prior170to30233792/8448; genuine302to36215360/3840; engineer362to4009728/2432. New400to424 separately planned6144fresh/1536updates. Startup baseline and arenas0training games/updates.', initial_replay='Latest400 raw TRAIN shards36/37/38 remain untouched, include old25/25/50 and actual084 labels; exit after3fresh stages.')
    for name in ('tools/open_v4_recipe50_contract.py', 'tools/run_open_v4_recipe50.py', 'python/junqi_open_v4/recipe50_games.py', 'tools/review_open_v4_recipe50.py', 'tools/review_open_v4_recipe50_full.py', 'tools/prepare_open_v4_recipe50.py', 'tools/activate_open_v4_recipe50.py', 'tests/test_open_v4_recipe50.py', 'start_open_v4_recipe50.ps1', 'stop_open_v4_recipe50.ps1'):
        plan['source'][name] = sha(ROOT / name)
    atomic_json(planpath, plan)
    verify_single_prerequisite(ROOT, plan)
    result = dict(plan=str(planpath.relative_to(ROOT)).replace('\\','/'), directory=directory, report=str(out.relative_to(ROOT)), source=current, planned_new_games=6144, planned_updates=1536, checks=len(checks), target_iteration=424, tests=tests)
    atomic_json(out / 'REGISTRATION.json', result)
    atomic_json(ROOT / 'reports/recipe50_latest_registration.json', result)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__': main()
