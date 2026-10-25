"""Frozen-weight PUCT exploration diagnosis. No learning or automatic adoption."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path


def validate_plan(plan):
    if plan['experiment_type'] != 'frozen_puct_exploration_factor':
        raise ValueError('explicit frozen exploration factor required')
    expected = {name: {'cpuct': value, 'simulations': 256} for name, value in
                [('parent', 1.5), ('control', 1.5), ('candidate', .75), ('repeat', 1.5)]}
    if plan['search_variants'] != expected:
        raise ValueError('only candidate cpuct1.5->0.75 at equal256 may change')
    if (plan['screen_families'], plan['full_families'], plan['diagnostic_positions']) != (16, 64, 48):
        raise ValueError('registered screening/full/diagnostic sizes required')
    participants = plan['participants']
    if list(participants) != ['parent', 'control', 'candidate', 'repeat']:
        raise ValueError('baseline must precede candidate for paired sentinels')
    if participants['parent'] != participants['candidate'] or participants['parent'] != participants['repeat']:
        raise ValueError('parent/candidate/repeat complete weights and Value scale must be identical')
    if participants['control'] != plan['stronger_legacy_reference']:
        raise ValueError('stronger legacy reference guard required')
    if plan['training_updates'] != 0 or plan['new_training_games'] != 0 or plan['automatic_adoption']:
        raise ValueError('frozen diagnosis cannot train, collect or adopt')
    return True


def main():
    from open_v4_runtime import atomic_json, read_json, utc_now, ProcessLock, supervision_status
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    parser.add_argument('--status', action='store_true')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def inside(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root: raise ValueError('path outside alphazero')
        return path
    plan_path = inside(args.plan); plan = read_json(plan_path); directory = inside(plan['directory'])
    stops = [inside(p) for p in plan['stop_paths']]
    if args.status:
        state = supervision_status(directory)
        state['stop_requested'] = any(p.exists() for p in stops)
        print(json.dumps(state)); return
    validate_plan(plan)
    import numpy as np
    import torch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import model_hash
    from junqi_open_v4.layouts import load_role, MultiLayoutBackend
    from junqi_open_v4.arena import select_families, run_arena
    from junqi_open_v4.pipeline import promotion_decision
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.search import BatchedNeuralPUCT
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from open_v4_experiment_evidence import teacher_deltas
    from run_open_v4_fresh_data import is_holdout
    from run_open_v4_policy_anchor_factor import screening_signal
    from run_open_v4_search_training import route_signal
    base = OpenConfig.load(inside(plan['parent_config']))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    def verify():
        for name, digest in plan['source'].items():
            if sha(inside(name)) != digest: raise ValueError(f'source changed: {name}')
        if sha(inside(plan['parent_config'])) != plan['parent_config_sha256']: raise ValueError('base config changed')
        if sha(inside(base.layout_directory) / 'manifest.json') != plan['layout_manifest_hash']: raise ValueError('layouts changed')
        if sha(inside(plan['suite']['path'])) != plan['suite']['sha256']: raise ValueError('dev suite changed')
        for item in plan['frozen_runs']:
            if sha(inside(item['directory']) / 'CURRENT.json') != item['current_sha256']: raise ValueError('archived roles changed')
        for item in plan['participants'].values():
            if sha(inside(item['path'])) != item['sha256']: raise ValueError('complete weights changed')
        item = plan['result_review']
        if sha(inside(item['path'])) != item['sha256']: raise ValueError('registered source review changed')
        item = plan['diagnostic_source']
        if sha(inside(item['path'])) != item['sha256'] or sha(inside(item['metadata'])) != item['metadata_sha256']:
            raise ValueError('training-only diagnostic source changed')
    if any(p.exists() for p in stops): raise RuntimeError('STOP requested; never clear')
    with ExitStack() as locks:
        for item in plan['frozen_runs']:
            for name in ('SUPERVISOR', 'TRAINING'):
                locks.enter_context(ProcessLock(inside(item['directory']) / f'supervision/{name}.lock'))
        verify()
        for item in (plan['participants']['parent'], plan['participants']['control']):
            saved = torch.load(inside(item['path']), map_location='cpu', weights_only=False)
            if not all(k in saved for k in ('model', 'optimizer', 'scheduler', 'counters', 'config', 'rng', 'leaf_value_scale')):
                raise ValueError('incomplete source learning state')
            if saved['config']['model'] != base.as_dict()['model']: raise ValueError('architecture changed')
            if saved['model_hash'] != item['model_hash']: raise ValueError('source tensor digest changed')
            del saved
        if args.check_only:
            print(json.dumps({'valid': True, 'frozen_complete_states': True, 'training_updates': 0})); return
        if directory.exists() != args.resume: raise ValueError('fresh directory or explicit reviewed resume required')
        directory.mkdir(parents=True, exist_ok=True)
        for name in ('SUPERVISOR', 'TRAINING'):
            locks.enter_context(ProcessLock(directory / f'supervision/{name}.lock'))
        contract = {'plan_sha256': sha(plan_path), 'source': plan['source']}
        status = dict(schema=plan['schema'], state='checking', pid=os.getpid(), started_at=utc_now(),
            plan=str(plan_path.relative_to(root)), completed_updates=0, new_training_games=0, stage=0,
            variant='candidate', frozen_weights=True)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / 'STATUS.json', status); atomic_json(directory / 'supervision/STATUS.json', status)
        def check():
            if any(p.exists() for p in stops):
                update(state='stopped_by_user', review_required=False)
                raise RuntimeError('STOP requested; frozen sources preserved')
        if args.resume:
            if read_json(directory / 'CONTRACT.json') != contract: raise ValueError('resume contract changed')
            progress = read_json(directory / 'PROGRESS.json')
        else:
            atomic_json(directory / 'CONTRACT.json', contract)
            progress = {'evaluations': {}, 'sources': plan['participants'], 'variants': {}, 'diagnostic': None}
            records = {}
            for name in ('parent', 'control'):
                item = plan['participants'][name]; path = directory / 'checkpoints' / f'{name}_frozen.pt'
                path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(inside(item['path']).read_bytes())
                if sha(path) != item['sha256']: raise ValueError('frozen mirror differs')
                records[name] = dict(item, path=str(path.relative_to(directory)))
            atomic_json(directory / 'CURRENT.json', {'iteration': 26, 'learner': records['parent'],
                'actor': records['parent'], 'champion': records['parent'], 'frozen_reference': records['control'],
                'experimental_search': plan['search_variants'], 'training_updates': 0, 'automatic_adoption': False})
        def commit(): atomic_json(directory / 'PROGRESS.json', progress)
        commit(); update()
        try:
            torch.set_num_threads(base.workers); device = torch.device(base.device)
            models = {}
            for name, item in plan['participants'].items():
                if item['model_hash'] not in models:
                    saved = torch.load(inside(item['path']), map_location='cpu', weights_only=False)
                    model = PolicyValueNet(base.model).to(device).eval(); model.load_state_dict(saved['model'])
                    model.search_v3_leaf_value_scale = saved['leaf_value_scale']
                    if model_hash(model) != item['model_hash']: raise ValueError('loaded weights differ')
                    models[item['model_hash']] = model
            def agent(name):
                return StopAwareSearch(BatchedNeuralPUCT(models[plan['participants'][name]['model_hash']], device=device,
                    cpuct=plan['search_variants'][name]['cpuct'], inference_batch_size=base.inference_batch_size,
                    cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree), check)
            if not progress['diagnostic']:
                update(state='search_diagnostic', planned_positions=48)
                source = plan['diagnostic_source']; meta = read_json(inside(source['metadata']))
                if meta['layout_role'] != 'train' or meta['shard_sha256'] != source['sha256']: raise ValueError('diagnostic must be training-only')
                backend = MultiLayoutBackend(load_role(inside(base.layout_directory), 'train'), base.seed)
                states, descriptors, seen = [], [], set()
                for game in meta['games']:
                    if game['mode'] != 'selfplay' or game['family'] in seen or is_holdout(game['family'], base.seed): continue
                    seen.add(game['family'])
                    for ply in (0, 24, 48):
                        if ply >= len(game['trace']): continue
                        state = backend.new_state(game['provenance']['setup_seed'])
                        if state.provenance() != game['provenance']: raise ValueError('training history source differs')
                        for action in game['trace'][:ply]: state.apply(action)
                        if state.is_terminal(): raise ValueError('terminal diagnostic prefix')
                        states.append(state); descriptors.append({'family': game['family'], 'ply': ply,
                            'setup_seed': game['provenance']['setup_seed'], 'trace': game['trace'][:ply],
                            'state_key': int(state.native_state.key())})
                        if len(states) == 48: break
                    if len(states) == 48: break
                if len(states) != 48: raise ValueError('48 distinct training prefixes required')
                results, metrics = {}, {}
                for budget in (64, 256):
                    for name in ('parent', 'candidate'):
                        check(); results[(name, budget)], metrics[f'{name}_{budget}'] = agent(name).search(states, budget)
                diagnostic = {'role': 'train', 'source': source, 'positions': descriptors, 'metrics': metrics,
                    'comparisons': {str(b): {'action_agreement': float(np.mean([a.action == c.action for a, c in zip(results[('parent', b)], results[('candidate', b)])])),
                        'mean_root_entropy': {n: float(np.mean([r.root_entropy for r in results[(n, b)]])) for n in ('parent', 'candidate')}} for b in (64, 256)},
                    'peak_gpu_allocated_mib': torch.cuda.max_memory_allocated() / 2**20,
                    'interpretation': 'Fixed full-history training prefixes; no root noise. Descriptive only, not a strength gate or training-target adoption.'}
                path = directory / 'SEARCH_DIAGNOSTIC.json'; atomic_json(path, diagnostic)
                progress['diagnostic'] = {'path': str(path.relative_to(root)), 'sha256': sha(path)}; commit()
                del states, results
            suite = read_json(inside(plan['suite']['path']))['states']
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != plan['teachers']: raise ValueError('teacher provenance changed')
                submit = teachers.submit_actions
                def checked_submit(states, ids): check(); return submit(states, ids)
                teachers.submit_actions = checked_submit
                def evaluate(name, families):
                    check(); verify()
                    if name in progress['evaluations']:
                        item = progress['evaluations'][name]
                        if sha(inside(item['path'])) != item['sha256']: raise ValueError('committed arena changed')
                        return read_json(inside(item['path']))
                    selected = select_families(suite, families, plan['evaluation_seed'])
                    agents = {n: agent(n) for n in plan['participants']}
                    matchups = [('candidate', 'parent'), ('candidate', 'control')] + [(n, f'teacher{i}') for i in range(len(teachers.modules)) for n in agents]
                    update(state='arena', arena=name, evaluation_families=families, planned_games=2 * len(selected) * len(matchups))
                    arena = run_arena(selected, matchups, agents, teachers, {n: 256 for n in agents}, seed=plan['evaluation_seed'])
                    gains = {key: teacher_deltas(arena, before, after) for key, before, after in
                        [('parent', 'parent', 'candidate'), ('control', 'control', 'candidate'), ('duplicate', 'parent', 'repeat')]}
                    duels = [r['family_score'] for r in arena['summaries'][:2]]; gate = promotion_decision(arena, base)
                    arena['review'] = {'gains': gains, 'gate': gate, 'screen_promising': screening_signal([gains['parent'], gains['control']], duels),
                        'route_confirmed': route_signal([gains['parent'], gains['control']], gains['duplicate'], duels, gate, families, .02), 'screen_only': families < 64}
                    arena.update(checkpoints=plan['participants'], search_variants=plan['search_variants'],
                        purpose='Only qualified26 exploration cpuct changes; legacy32 is an extra frozen strength guard',
                        peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
                    path = directory / 'arena' / f'{name}.json'; atomic_json(path, arena)
                    progress['evaluations'][name] = {'path': str(path.relative_to(root)), 'sha256': sha(path), 'review': arena['review']}; commit()
                    return arena
                screen = evaluate('exploration_screen', 16)
                full = evaluate('exploration_full', 64) if screen['review']['screen_promising'] else None
            check(); verify()
            for key, model in models.items():
                if model_hash(model) != key: raise ValueError('frozen inference mutated weights')
            confirmed = full is not None and full['review']['route_confirmed']
            atomic_json(directory / 'RESULTS.json', {'screen': screen['review'], 'full': full['review'] if full else None,
                'route_confirmed': confirmed, 'training_updates': 0, 'weights_unchanged': True, 'production_roles_unchanged': True,
                'automatic_adoption': False, 'interpretation': 'Any positive search route needs review and a new registered training experiment; no automatic setting or model adoption.'})
            update(state='review_due', review_required=True, route_confirmed=confirmed, reason='completed_frozen_exploration_diagnosis; review_before_any_training')
        except BaseException as error:
            if status['state'] != 'stopped_by_user': update(state='needs_attention', reason=repr(error), review_required=True)
            raise


if __name__ == '__main__': main()
