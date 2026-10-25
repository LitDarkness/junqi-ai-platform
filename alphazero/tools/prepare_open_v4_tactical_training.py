"""Register the currently authorized fresh-data repair without restarting archives."""
import hashlib
from pathlib import Path


def main():
    import torch
    from open_v4_runtime import atomic_json, read_json, utc_now
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    stamp = utc_now().replace('-', '').replace(':', '').replace('T', '_')[:15]
    old = read_json(root / 'config/experiments/open_v4_self_training_20261004_023128.json')
    old_progress = read_json(root / old['directory'] / 'PROGRESS.json')
    current = old_progress['variants']['control']
    source = {'path': current['checkpoint'], 'sha256': current['sha256'],
              'model_hash': current['model_hash'], 'iteration': current['counters']['iteration']}
    if sha(root / source['path']) != source['sha256'] or source['iteration'] != 32:
        raise ValueError('current experimental source32 identity changed')
    payload = torch.load(root / source['path'], map_location='cpu', weights_only=False)
    base_path = f'config/experiments/open_v4_tactical_base_{stamp}.json'
    atomic_json(root / base_path, payload['config'])
    directory = f'runs/open_junqi_v4_tactical_training_{stamp}'
    if (root / directory).exists(): raise ValueError('never replace existing run')
    initial = old_progress['collections']['control'][-3:]
    if len(initial) != 3: raise ValueError('three original recent training shards required')
    for item in initial:
        if sha(root / item['path']) != item['sha256'] or sha(root / item['metadata']) != item['metadata_sha256']:
            raise ValueError('original replay changed')
    sources = list(old['source'])
    sources.remove('tools/run_open_v4_self_training.py')
    sources.remove('tools/prepare_open_v4_self_training.py')
    sources += ['tools/run_open_v4_tactical_training.py', 'tools/prepare_open_v4_tactical_training.py',
                'cpp/neural_puct.cpp', 'cpp/neural_puct.h', 'cpp/junqi_az_bindings.cpp',
                'cpp/az_state.cpp', 'cpp/az_state.h', 'python/junqi_az/model.py']
    frozen = []
    for p in sorted((root / 'runs').glob('open_junqi_v4_*/CURRENT.json')):
        frozen.append({'directory': str(p.parent.relative_to(root)), 'current_sha256': sha(p)})
    markers = [{'path': str(p.relative_to(root)), 'sha256': sha(p)}
               for p in sorted((root / 'runs').glob('*/**/STOP'))]
    control = dict(selfplay_fraction=.25, teacher_selfplay_fraction=None, actor_policy_target='teacher',
                   deduplicate=False, neural_policy_only=False, root_tactical_guard=False)
    candidate = dict(control, root_tactical_guard=True)
    plan = dict(schema='junqi-tactical-training-plan-1', registered_at=utc_now(), directory=directory,
        parent_config=base_path, parent_config_sha256=sha(root / base_path), source_checkpoint=source,
        source_counters=payload['counters'], source_config_hash=payload['config_hash'],
        participants={'parent': source, 'champion': old['source_checkpoint']},
        suite=old['suite'], teachers=old['teachers'], layout_manifest_hash=old['layout_manifest_hash'],
        initial_replay=initial, frozen_runs=frozen, source={name: sha(root / name) for name in sources},
        historical_stops=markers, historical_stop_markers=[m['path'] for m in markers],
        stop_paths=[f'{directory}/STOP', f'{directory}/supervision/STOP'],
        evaluation_seed=old['evaluation_seed'], screen_families=16, full_families=64, evaluation_budget=256,
        collection_start=100, stages_per_block=6, maximum_stages=12,
        games_per_stage=256, updates_per_stage=64, replay_stages=3, control=control, candidate=candidate,
        authorization='Latest user explicitly requested accurate current problem localization, repair and continual training. New branch only; all archived STOPs and failed states remain preserved.',
        hypothesis='Exact native root tactical constraints may improve fresh neural selfplay outcomes and search targets. Only candidate neural collector uses guard. The proof-based extra CPU transitions are recorded separately from64/128 MCTS simulations; this is not equal total compute.',
        source_role='Experimental current32 f9eb complete learner, not approved production Actor/Champion. Production26 remains an additional promotion baseline.',
        collector='Both branches collect256 NEW games before64updates using their own preceding complete learner; paired setup namespace100+, modes and colors; no frozen26 collector.',
        policy_contract='Original hybrid25%network selfplay/25%teacher mutual/50%network-teacher retained. Selfplay Policy actual neural visits; actor and teacher hard labels remain explicitly teacher labels. All Value labels actual native terminal WDL. No dedup/history filtering.',
        evaluation_contract='All models use standard PUCT256, guard OFF. Fixed16 is exploratory screening. Full64 requires paired gains over production Champion, source32 and matched control, with repetition baseline check. Development/final games never enter replay.',
        screen_rule='All three teacher mean-delta groups positive for both related teachers, all three direct mean scores>.5; no promotion. Failed screen means no confirmed gain.',
        promotion_rule='Full64 only: all six paired teacher CI95 lower>0 against champion/source32/control, all three direct CI95 lower>.5, repeat source32 teacher point drift<=.02, unchanged production Champion gate. Teachers082/084 correlated, not independent evidence.',
        failure_policy='First unconfirmed block ends at review_due with no automatic continuation. Confirmed block may continue to pre-registered12stages. Afterwards review the saved results and next mechanism rather than blindly extending.',
        validation_policy='Internal held-out trajectory groups, not unseen layout families; not strength or generalization evidence. Final layouts unused.')
    path = root / f'config/experiments/open_v4_tactical_training_{stamp}.json'
    if path.exists(): raise ValueError('preserve prior registration')
    atomic_json(path, plan)
    atomic_json(root / 'reports/current_training_audit_20261004/REGISTERED.json',
                {'plan': str(path.relative_to(root)), 'directory': directory, 'source_checkpoint': source})
    print(str(path.relative_to(root)))


if __name__ == '__main__': main()
