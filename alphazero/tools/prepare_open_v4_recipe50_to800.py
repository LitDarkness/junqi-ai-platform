"""Register unchanged600to800 continuation with exact full learning state on CPU."""
from pathlib import Path
import copy, hashlib, json, unittest, shutil
ROOT = Path(__file__).resolve().parents[1]

def main():
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from open_v4_recipe50_to800_contract import validate_source_config, fork_stable_payload, verify_stable_seed, verify_single_prerequisite
    from open_v4_monitor_evidence import reconstruct_monitor
    from junqi_open_v4.arena import select_families
    torch.set_num_threads(2)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    def entry(p):
        p = Path(p); p = p if p.is_absolute() else ROOT / p
        return dict(path=str(p.relative_to(ROOT)), sha256=sha(p))
    master = read_json(ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    old = read_json(ROOT / master['optimization']['config']); oldrun = ROOT / old['directory']
    progress = read_json(oldrun / 'PROGRESS.json'); item = progress['variants']['learner']
    checks = {}
    def check(k, ok):
        checks[k] = bool(ok)
        if not ok: raise ValueError('800 continuation registration: ' + k)
    check('latest600_completed', not master['paused_by_user'] and read_json(oldrun / 'STATUS.json')['state'] == 'horizon_complete' and read_json(oldrun / 'COMPLETION.json')['target_iteration'] == 600 and item['counters']['iteration'] == 600 and sha(ROOT / item['checkpoint']) == item['sha256'])
    prior_report = master['optimization']['completion_report']
    for label, path_key, hash_key in [('report', 'path', 'sha256'), ('data', 'data', 'data_sha256'), ('qa', 'qa', 'qa_sha256')]:
        check('completed600_report:' + label, sha(ROOT / prior_report[path_key]) == prior_report[hash_key])
    for label in ('final_completion_review', 'last_review'):
        e = master['optimization'][label]
        check('completed600_review:' + label, sha(ROOT / e['path']) == e['sha256'] and read_json(ROOT / e['path'])['all_checks_passed'])
    check('completed600_report_review_closed', prior_report['complete'] and prior_report['latest_iteration'] == 600 and master['optimization']['phase'] == 'review_complete')
    for p in [ROOT / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'] + [ROOT / e['directory'] / f'supervision/{n}.lock' for e in old['frozen_runs'] + [{'directory': old['directory']}] for n in ('SUPERVISOR', 'TRAINING')]:
        check('unowned:' + str(p.relative_to(ROOT)), not lock_held(p))
    for p, h in old['source'].items(): check('preserved_frozen_source:' + p, sha(ROOT / p) == h)
    for e in old['frozen_runs']: check('preserved_CURRENT:' + e['directory'], sha(ROOT / e['directory'] / 'CURRENT.json') == e['current_sha256'])
    source = torch.load(ROOT / item['checkpoint'], map_location='cpu', weights_only=False)
    check('source_complete_model_Adam_schedule_RNG', all(k in source for k in ('model', 'optimizer', 'scheduler', 'counters', 'rng', 'config', 'leaf_value_scale')))
    stamp = utc_now().replace('-', '').replace(':', '').replace('T', '_')[:15]
    out = ROOT / f'reports/recipe50_to800_{stamp}'; out.mkdir()
    directory = f'runs/open_junqi_v4_recipe50_to800_{stamp}'
    planpath = ROOT / f'config/experiments/open_v4_recipe50_to800_{stamp}.json'
    basepath = ROOT / f'config/experiments/open_v4_recipe50_to800_base_{stamp}.json'
    atomic_json(out / 'PRIOR_MASTER.json', master); atomic_json(basepath, source['config'])
    cfg = RootExploreConfig.load(basepath); cfg.run_directory = directory + '/learner'; cfg.initial_checkpoint = item['checkpoint']; cfg.validate()
    changed = validate_source_config(source, cfg)
    check('only_paths_changed', set(changed) == {'run_directory', 'initial_checkpoint'})
    migrated = fork_stable_payload(source, cfg, dict(source=item, target_iteration=800, same_recipe=True, no_weight_reset=True))
    statechecks = verify_stable_seed(source, migrated)
    check('exact_model_Adam_steps_schedule_counters_allRNG_Value_LR', all(statechecks.values()))
    check('source_optimizer_model_finite', all(torch.isfinite(v).all().item() for v in source['model'].values() if v.is_floating_point()) and all(torch.isfinite(v).all().item() for row in source['optimizer']['state'].values() for v in row.values() if isinstance(v, torch.Tensor) and v.is_floating_point()))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_open_v4_recipe50_to800.py')
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    check('continuation_state_route_horizon_protection_tests', result.wasSuccessful() and result.testsRun == 6)
    tests = dict(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors), all_passed=result.wasSuccessful())
    for e in progress['collections']['learner'][-3:]:
        check('initial3_same_recipe_TRAIN:' + e['path'], sha(ROOT / e['path']) == e['sha256'] and sha(ROOT / e['metadata']) == e['metadata_sha256'] and read_json(ROOT / e['metadata'])['layout_role'] == 'train' and read_json(ROOT / e['metadata'])['training_recipe'] == {'selfplay': .5, 'teacher_selfplay': 0., 'vs_actual_082': .5})
    baseline_ref = progress['evaluations']['startup_baseline']; baseline = read_json(ROOT / baseline_ref['path'])
    check('original400_baseline_sha', sha(ROOT / baseline_ref['path']) == baseline_ref['sha256'])
    selected = select_families(read_json(ROOT / old['suite']['path'])['states'], 16, old['evaluation_seed'])
    baseline_review = reconstruct_monitor(baseline, selected, old, ROOT)
    check('original400_baseline_native_stats', baseline_review['all_checks_passed'] and all(x['model_hash'] == old['participants']['parent']['model_hash'] for x in baseline['checkpoints'].values()))
    atomic_json(out / 'BASELINE400_REVIEW.json', baseline_review)
    stops = {e['path']: e['sha256'] for e in old['historical_stops']}
    for p in (ROOT / 'runs').rglob('STOP'):
        if p.is_file(): stops[str(p.relative_to(ROOT))] = sha(p)
    per_stage = (ROOT / item['checkpoint']).stat().st_size + sum((ROOT / progress['collections']['learner'][-1][k]).stat().st_size for k in ('path', 'metadata')) + 1024**2
    budget = dict(free_bytes=shutil.disk_usage(ROOT).free, estimated200_stage_bytes=per_stage * 200, minimum_free_gib=old['minimum_free_gib'], formal_arena_growth_not_assumed_zero=True)
    check('estimated_training_disk_reserve', budget['free_bytes'] - budget['estimated200_stage_bytes'] > old['minimum_free_gib'] * 1024**3)
    atomic_json(out / 'DISK_BUDGET.json', budget)
    changes = read_json(ROOT / 'reports/recipe50_to800_entry_changes.json')
    for e in changes['changes']:
        check('copied_entry_original:' + e['original'], sha(ROOT / e['original']) == e['original_sha256'])
        check('copied_entry_new:' + e['created'], sha(ROOT / e['created']) == e['new_sha256'])
    boundary = out / 'BOUNDARY_AUDIT.json'
    atomic_json(boundary, dict(all_checks_passed=all(checks.values()), checks=checks, source=item, source_state_preservation=statechecks, changed_config_fields=changed, training_teacher_id=0, old_run=old['directory'], old_current_sha256=sha(oldrun / 'CURRENT.json'), initial_replay=progress['collections']['learner'][-3:], qualification_reference400=old['participants']['parent'], original400_baseline=baseline_ref, all_completed600_and_negative_historical_results_preserved=True, new_entry_changes=entry(ROOT / 'reports/recipe50_to800_entry_changes.json')))
    cpu = out / 'CPU_VERIFICATION.json'
    atomic_json(cpu, dict(all_passed=all(checks.values()), state_checks=statechecks, tests=tests, baseline_native=baseline_review['native_verification'], zero_training_games=True, zero_updates=True, no_cuda_initialized=not torch.cuda.is_initialized()))
    auth = out / 'AUTHORIZATION.json'
    atomic_json(auth, dict(at=utc_now(), request='继续训练更多轮次', recipe={'selfplay': .5, 'teacher_selfplay': 0., 'vs_actual_082': .5}, allow_other_learning_changes=False, no_weight_reset=True, continue_despite_strength_decline=True, source_iteration=600, target_iteration=800, target_selection='Agent selected a bounded200-stage extension to800 for review under explicit human authorization to continue more rounds; no new recipe or learning mechanism.', scope='Extend unchanged50/50 recipe from complete600 to800. Full600 learning state inherited; original64PPV/LR/search/labels/replay/RNG/Value preserved. Frozen400 qualification and all previous results retained. Negative or uncertain scores do not stop training. Target800 is a review endpoint, not an automatic further extension.'))
    current = dict(path=item['checkpoint'], sha256=item['sha256'], model_hash=item['model_hash'], iteration=600, leaf_value_scale=source['leaf_value_scale'])
    plan = copy.deepcopy(old)
    plan.update(schema='junqi-recipe50-600to800-plan-1', route='recipe50_selfplay_vs082_to800', registered_at=utc_now(), directory=directory, parent_config=str(basepath.relative_to(ROOT)), parent_config_sha256=sha(basepath), source_checkpoint=current, source_counters=source['counters'], source_config_hash=source['config_hash'], maximum_stages=200, target_iteration=800, screen_boundaries=[6] + list(range(12, 201, 12)) + [200], monitor084_boundaries=list(range(24, 201, 24)) + [200], collection_start=read_json(ROOT / progress['collections']['learner'][-1]['metadata'])['iteration'] + 1, initial_replay=progress['collections']['learner'][-3:], registered_maximum_new_games=51200, registered_maximum_updates=12800, qualification_reference400=copy.deepcopy(old['participants']['parent']), frozen_baseline=entry(ROOT / baseline_ref['path']), user_contract=entry(auth), cpu_verification=entry(cpu), boundary_audit=entry(boundary), training_teacher_ids=[0])
    plan['participants']['champion'] = copy.deepcopy(progress['champion'])
    plan['initial_source_teacher_score'] = next(r['family_score']['mean'] for r in baseline['summaries'] if r['candidate'] == 'parent' and r['opponent'] == 'teacher0')
    plan['stop_paths'] = [directory + '/STOP', directory + '/supervision/STOP']
    plan['historical_stops'] = [dict(path=p, sha256=h) for p, h in sorted(stops.items())]; plan['historical_stop_markers'] = list(stops)
    plan['frozen_runs'].append(dict(directory=old['directory'], current_sha256=sha(oldrun / 'CURRENT.json')))
    plan['previous_experiment'] = dict(plan=entry(ROOT / master['optimization']['config']), completed600_report=prior_report, completed600_review=master['optimization']['final_completion_review'])
    plan.update(authorization='Latest human explicitly requests more training. Agent registers unchanged600to800 continuation and endpoint review. No new learning hyperparameters.', source_role='Latest complete600 learning state. Frozen400/Champion230/candidate134/control86 protect adoption.', horizon='200fresh stages600to800, 51200fresh/12800updates. Registered6/12/every12/end research082 screens, every24/end actual084 monitoring. No negative-score stop; endpoint review, no automatic further extension.', frozen_reference_definition='parent/repeat remain400; latest600 is learning source, not a replacement qualification baseline. Champion230/control134/guard86 remain fixed references.', hypothesis='Measure further unchanged-recipe progress through800 without assuming recovery, monotonic strength or convergence.', baseline_registration='Import unchanged original400/082/084344-game historical frozen baseline and independent native/statistical review. Zero new baseline matches or learning; frozen400 source score stays fixed.', interpretation='Same50/50 recipe continues from600; strength and possible forgetting assessed from actual scores and paired intervals, not losses. Original formal protections and final-layout isolation remain.', efficiency='Learning/search/labels/LR/update ratio unchanged. Retain evaluation graph proof; independent detached WMI owner performs same-process continuation.')
    plan['reporting_contract'].update(cumulative='Old170to30233792/8448; true302to36215360/3840; engineered362to4009728/2432; recipe400to4246144/1536; recipe424to60045056/11264 preserved. New600to800 independently planned51200/12800. Imported baseline, frozen arenas, CPU audits add0training games/updates.', initial_replay='Latest600 same-recipe actual082 TRAIN174/175/176 preserved. Inherited shards naturally exit after3new stages; never relabel.')
    names = ['tools/build_open_v4_recipe50_to800.py', 'tools/open_v4_recipe50_to800_contract.py', 'tools/run_open_v4_recipe50_to800.py', 'tools/review_open_v4_recipe50_to800.py', 'tools/review_open_v4_recipe50_to800_full.py', 'tools/prepare_open_v4_recipe50_to800.py', 'tools/activate_open_v4_recipe50_to800.py', 'tools/detached_open_v4_recipe50_to800.py', 'tests/test_open_v4_recipe50_to800.py', 'start_open_v4_recipe50_to800.ps1', 'stop_open_v4_recipe50_to800.ps1']
    for name in names: plan['source'][name] = sha(ROOT / name)
    atomic_json(planpath, plan); verify_single_prerequisite(ROOT, plan)
    registration = dict(plan=str(planpath.relative_to(ROOT)).replace('\\', '/'), directory=directory, report=str(out.relative_to(ROOT)), source=current, qualification_reference400=plan['qualification_reference400'], planned_new_games=51200, planned_updates=12800, checks=len(checks), target_iteration=800, tests=tests)
    atomic_json(out / 'REGISTRATION.json', registration); atomic_json(ROOT / 'reports/recipe50_to800_latest_registration.json', registration)
    print(json.dumps(registration, ensure_ascii=False))

if __name__ == '__main__': main()
