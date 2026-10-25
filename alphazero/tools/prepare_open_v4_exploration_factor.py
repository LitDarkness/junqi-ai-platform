"""Preregister one frozen-search coefficient, before any diagnostic/evaluation."""
import copy
from datetime import datetime, timezone
import hashlib
from pathlib import Path

from open_v4_runtime import read_json, atomic_json, utc_now


def main():
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    old_path = root / 'config/experiments/open_v4_teacher_history_factor_20261004_052542.json'
    old = read_json(old_path); old_run = root / old['directory']
    assert read_json(old_run / 'STATUS.json')['state'] == 'review_due'
    assert not read_json(old_run / 'BLOCK_000006_RESULTS.json')['route_confirmed']
    review = old_run / 'supervision/REVIEWS/RESULT_REVIEW_20261004_054358.json'
    assert read_json(review)['games_verified'] == 860
    stamp = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    directory = f'runs/open_junqi_v4_exploration_factor_{stamp}'
    path = root / f'config/experiments/open_v4_exploration_factor_{stamp}.json'
    assert not path.exists() and not (root / directory).exists()
    parent = old['source_checkpoint']; legacy = old['participants']['control']
    plan = {k: copy.deepcopy(old[k]) for k in ('parent_config', 'parent_config_sha256',
        'teachers', 'suite', 'layout_manifest_hash', 'source', 'historical_stop_markers', 'frozen_runs', 'stop_paths')}
    plan.update(schema='junqi-frozen-exploration-plan-1', experiment_type='frozen_puct_exploration_factor',
        registered_at=utc_now(), directory=directory, source_checkpoint=parent,
        participants={'parent': parent, 'control': legacy, 'candidate': parent, 'repeat': parent},
        stronger_legacy_reference=legacy,
        search_variants={n: {'cpuct': c, 'simulations': 256} for n, c in
            [('parent', 1.5), ('control', 1.5), ('candidate', .75), ('repeat', 1.5)]},
        result_review={'path': str(review.relative_to(root)), 'sha256': sha(review)},
        diagnostic_source=old['frozen_collections'][0], diagnostic_positions=48,
        evaluation_seed=2026100409, screen_families=16, full_families=64,
        training_updates=0, new_training_games=0, automatic_adoption=False,
        hypothesis='Reduced PUCT exploration may improve action selection from fixed qualified26 Policy/Value, '
            'given noisy learned estimates and weak low-budget neural targets. Only cpuct1.5->0.75 changes for candidate. '
            'This is a distinct frozen inference mechanism test after negative replay eligibility experiments, not more training.',
        primary_comparison='Candidate qualified26 cpuct0.75 versus identical qualified26 cpuct1.5; fixed256 added simulations. '
            'Frozen stronger legacy32 cpuct1.5 is an extra strength guard, not the causal matched-weight control.',
        diagnostic_rule='At most48 full-history training prefixes, excluding internal holdout. Record64/256 action/entropy '
            'changes for both coefficients, no root noise. Descriptive only; no gate or secondary rescue.',
        screen_rule='Unchanged fixed16 futility screen: both teacher positive means relative to identical parent and '
            'stronger legacy reference, both direct mean scores>0.5; cannot qualify.',
        promotion_rule='Only full64 original teacher pairedCI95 lower>0 versus parent and strongerlegacy reference, '
            'both directCI95 lower>0.5, duplicate-parent point abs<=0.02 and original Champion gate support a search route. '
            'No automatic model/Actor/search-parameter adoption; review and separate registered training experiment required.',
        failure_policy='First unconfirmed screen/full stops review. No blind continuation, grid search, automatic restart or fallback adoption.',
        held_constant='Qualified26 full weights,leaf Value scale,terminal exact values,legal/repetition rules,NN batching, '
            'cache/reuse defaults,suite/colors,teacher binaries and budget. Existing complete learning states mirrored byte-identically.',
        development_reuse_caveat='Repeatedly used development families, related082/084, not independent final testing. '
            'No development/final replay or final selection. Lower entropy alone does not mean stronger targets.',
        authorization='User authorized necessary ongoing training, evaluation and optimization, including independent '
            'registered hypothesis tests after a stopped unconfirmed branch; all old STOPs and negatives preserved.')
    plan['frozen_runs'].append({'directory': old['directory'], 'current_sha256': sha(old_run / 'CURRENT.json')})
    plan['stop_paths'] += [directory + '/STOP', directory + '/supervision/STOP']
    for name in ('tools/run_open_v4_exploration_factor.py', 'tools/prepare_open_v4_exploration_factor.py',
                 'start_open_v4_exploration_factor.ps1', 'tests/test_open_v4_exploration_factor.py',
                 'python/junqi_az/model.py', 'python/junqi_az/config.py', 'python/junqi_search_v3/recovery.py'):
        plan['source'][name] = sha(root / name)
    assert not any((root / p).exists() for p in plan['stop_paths'])
    atomic_json(path, plan)
    for rel in ('runs/open_junqi_v4_repair_pilot/supervision/PLAN.json', 'runs/open_junqi_v4_fresh_continuation/supervision/PLAN.json'):
        target = root / rel; main_plan = read_json(target); previous = main_plan['optimization']
        previous.update(state='review_due', final_review=plan['result_review'], final_screen=dict(route_confirmed=False,
            full64_not_run=True, candidate_teacher_score=.20052083333333331,
            candidate_teacher_ci95=[.13802083333333331, .26822916666666663],
            candidate_minus_control=-.1015625, delta_ci95=[-.19010416666666669, -.015625000000000007]))
        main_plan.setdefault('optimization_history', []).append(copy.deepcopy(previous))
        main_plan['optimization'] = dict(type=plan['experiment_type'], state='registered',
            config=str(path.relative_to(root)), run_directory=directory, launcher='start_open_v4_exploration_factor.ps1',
            status_tool='tools/run_open_v4_exploration_factor.py', source_review=plan['result_review'],
            historical_stops_are_archived=True, stop_paths=plan['stop_paths'], tests_passed=10,
            training_updates=0, new_training_games=0, automatic_adoption=False)
        main_plan['next_action'] = ('Follow latest optimization config/run_directory/status_tool. All four new training '
            'factors stopped unconfirmed; preserve negative results and never resume them. Current frozen_puct_exploration_factor '
            'uses exactly qualified26 full weights with cpuct1.5 parent/repeat and0.75 candidate, equal256sim; stronger legacy32 '
            'cpuct1.5 is additional guard, not causal same-weight control. First max48 train-only full-history prefixes compare '
            '64/256 descriptively,no strength inference from entropy; then original fixed16 screen and only if promising full64 '
            'strict paired teacher/direct/repeat/Champion gates versus parent and stronger reference. Training updates/new games0, '
            'complete state mirrored byte-identically,production roles/search settings untouched,no automatic adoption. '
            'Healthy locked GPU process permits read/record only,no second GPU or executed-source edit. On review_due verify '
            'complete arena,source hashes,unchanged weights and diagnostic provenance; only a fully supported route informs '
            'a separately preregistered training experiment,otherwise preserve negative hypothesis and investigate distinct mechanism. '
            'All plan STOP paths effective; original three userSTOPs archived never cleared. No dev/final replay or final tuning.')
        main_plan['updated_at'] = utc_now(); atomic_json(target, main_plan)
    print(str(path.relative_to(root)), flush=True)


if __name__ == '__main__': main()
