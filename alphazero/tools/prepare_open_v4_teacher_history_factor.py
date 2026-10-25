"""Register one hypothesis and immutable matched sources before execution."""
from collections import Counter
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from open_v4_runtime import atomic_json, read_json, utc_now


def main():
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    path = root / f'config/experiments/open_v4_teacher_history_factor_{stamp}.json'
    plan = copy.deepcopy(read_json(root / 'config/experiments/open_v4_dedup_factor_20261004_035152.json'))
    anchor = read_json(root / 'config/experiments/open_v4_policy_anchor_factor_20261004_042349.json')
    anchor_run = root / anchor['directory']
    assert read_json(anchor_run / 'STATUS.json')['state'] == 'review_due'
    plan['registered_at'] = utc_now()
    plan['directory'] = f'runs/open_junqi_v4_teacher_history_factor_{stamp}'
    assert not (root / plan['directory']).exists() and not path.exists()
    plan['experiment_type'] = 'teacher_history_policy_factor'
    plan['control'].update(exclude_history_sensitive_teacher=False)
    plan['candidate'] = dict(plan['control'], exclude_history_sensitive_teacher=True)
    plan['hypothesis'] = ('A board-only teacher does not see actual repetition history. Excluding only its Policy '
        'labels at history-sensitive positions may reduce history-blind supervision, relative to the stronger '
        'legacy route. Teacher labels are not assumed wrong. Every real terminal Value and all neural search '
        'Policy targets remain. Both branches keep old25/25/50mix,teacher actor imitation labels,dedupFalse, '
        '64/128search,three-stage replay,full26state and384updates. This is a diagnostic, not a pure selfplay route.')
    plan['collector'] = ('No new collection. Reuse exactly original stronger legacy-control trajectories '
                         'and actual saved Actor provenance,2048 existing games. Neither learner collects games.')
    plan['interpretation'] = ('Single primary eligibility factor, frozen off-policy trajectories, one seed '
        'and repeatedly reused development families. No proof of universal causality or final generalization.')
    plan['failure_policy'] = ('First unconfirmed screen/full block stops for review. No blind extension, '
                             'fallback adoption, auto-restart or promotion based on loss.')
    plan['repair_ancestry_record'] = plan.pop('repair', None)
    review = anchor_run / 'supervision/REVIEWS/RESULT_REVIEW_20261004_051358.json'
    audit = root / 'reports/self_training_preflight/TEACHER_HISTORY_ELIGIBILITY_AUDIT.json'
    plan['result_review'] = {'path': str(review.relative_to(root)), 'sha256': sha(review)}
    plan['history_eligibility_audit'] = {'path': str(audit.relative_to(root)), 'sha256': sha(audit)}
    plan['frozen_runs'] = copy.deepcopy(anchor['frozen_runs']) + [
        {'directory': anchor['directory'], 'current_sha256': sha(anchor_run / 'CURRENT.json')}]
    plan['stop_paths'] = list(anchor['stop_paths']) + [
        plan['directory'] + '/STOP', plan['directory'] + '/supervision/STOP']
    plan['source'] = copy.deepcopy(anchor['source'])
    for name in ('tools/run_open_v4_teacher_history_factor.py',
                 'tools/prepare_open_v4_teacher_history_factor.py',
                 'python/junqi_open_v4/history_policy_replay.py',
                 'start_open_v4_teacher_history_factor.ps1',
                 'tests/test_open_v4_teacher_history_factor.py'):
        plan['source'][name] = sha(root / name)
    plan['authorization'] = ('User reauthorized necessary ongoing evaluation, optimization and training. '
        'Independent registered branch after unconfirmed anchor; archived STOPs and negative evidence preserved.')
    for item in plan['initial_replay'] + plan['frozen_collections']:
        assert sha(root / item['path']) == item['sha256']
        assert sha(root / item['metadata']) == item['metadata_sha256']
    assert not any((root / p).exists() for p in plan['stop_paths'])
    atomic_json(path, plan)
    totals = Counter()
    for item in read_json(audit)['sources']: totals.update(item['counts'])
    print(str(path.relative_to(root)), flush=True)
    print(dict(totals), flush=True)
    for relative in ('runs/open_junqi_v4_repair_pilot/supervision/PLAN.json',
                     'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json'):
        main_path = root / relative; main_plan = read_json(main_path)
        old = main_plan['optimization']
        old.update(state='review_due', source_result_review=plan['result_review'],
            final_screen=dict(route_confirmed=False, full64_not_run=True,
                candidate_teacher_score=.2526041666666667,
                candidate_teacher_ci95=[.16145833333333334, .3567708333333333],
                vs_neural_control_delta=.078125, vs_parent_delta=.018229166666666664,
                vs_legacy_delta=-.04947916666666667))
        main_plan.setdefault('optimization_history', []).append(copy.deepcopy(old))
        main_plan['optimization'] = dict(type=plan['experiment_type'], state='registered',
            config=str(path.relative_to(root)), run_directory=plan['directory'],
            launcher='start_open_v4_teacher_history_factor.ps1',
            status_tool='tools/run_open_v4_teacher_history_factor.py',
            source_review=plan['result_review'], history_eligibility_audit=plan['history_eligibility_audit'],
            stop_paths=plan['stop_paths'], historical_stops_are_archived=True, tests_passed=16,
            control_is_stronger_legacy_route=True, full_seed_state_verified=False)
        main_plan['next_action'] = ('Follow latest optimization config/run_directory/status_tool only. '
            'Package,dedup,and teacher-owned anchor stopped unconfirmed; never resume or relax gates. '
            'Current teacher_history_policy_factor changes only candidate Policy eligibility: exclude teacher '
            'labels at cycle_entry,repetition_count>1,or any repeated successor context. Keep all neural Policy '
            'targets and true terminal Value,old25/25/50mix,old actor teacher labels,dedupFalse,64/128search '
            'and three-stage replay. Both full26state +384updates on same2048 existing games,new0. '
            'All six unfiltered controls must exactly reproduce original stronger legacycontrol32 complete '
            'state before evaluation. Fixed16 cannot promote; signal only opens originalfull64 strict '
            'teacher paired/direct/repeat/Champion gates against parent and stronger matched control. '
            'Healthy locked GPU process permits read/record only. First unconfirmed block stops review, '
            'preserve negatives and investigate distinct hypothesis,not blind extension. All current plan '
            'STOP paths effective; three old userSTOPs archived never cleared. No dev/final replay or final tuning.')
        main_plan['updated_at'] = utc_now()
        atomic_json(main_path, main_plan)


if __name__ == '__main__': main()
