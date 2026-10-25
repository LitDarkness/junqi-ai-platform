"""Small saved-training WDL retention audit, performed only after natural exit."""
from pathlib import Path
from collections import defaultdict
import copy


def main():
    import argparse, json, numpy as np, torch
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock, lock_held
    from open_v4_queue_contract import sha, inside
    from run_open_v4_fresh_data import is_holdout
    from junqi_open_v4.training_replay import complete_games
    from junqi_search_v3.loop import read_selfplay_shard, search_distill_batch
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    p = argparse.ArgumentParser()
    p.add_argument('--previous-plan', required=True)
    p.add_argument('--retention-plan', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    old = read_json(inside(root, args.previous_plan))
    retention = read_json(inside(root, args.retention_plan))
    directory = inside(root, old['directory'])
    progress = read_json(directory / 'PROGRESS.json')
    retention_progress = read_json(inside(root, retention['directory']) / 'PROGRESS.json')
    if read_json(directory / 'STATUS.json')['state'] != 'review_due':
        raise ValueError('natural completed review boundary required')
    out = inside(root, args.output)
    if out.exists():
        raise ValueError('preserve saved audit')
    def check():
        if any(inside(root, n).exists() for n in old['stop_paths']) or read_json(
                root / 'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user'):
            raise InterruptedError('STOP or pause requested')
    check()
    if lock_held(root / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock') or lock_held(directory / 'supervision/TRAINING.lock'):
        raise ValueError('another GPU task owns the boundary')
    with ProcessLock(root / 'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'), ProcessLock(directory / 'supervision/TRAINING.lock'):
        check()
        base = RootExploreConfig.load(inside(root, old['parent_config']))
        corpora = {}
        for name, records in (
            ('retention86', retention_progress['collections']['control'][-3:]),
            ('current_control134', progress['collections']['control'][-3:]),
            ('current_candidate134', progress['collections']['candidate'][-3:])):
            rows, identities = [], []
            groups = defaultdict(set)
            for record in records:
                if sha(inside(root, record['path'])) != record['sha256'] or sha(inside(root, record['metadata'])) != record['metadata_sha256']:
                    raise ValueError('training corpus changed')
                meta = read_json(inside(root, record['metadata']))
                if meta['layout_role'] != 'train':
                    raise ValueError('training-only audit')
                samples = read_selfplay_shard(inside(root, record['path']))
                for game, game_rows in complete_games(samples, meta):
                    family, mode = int(game['family']), game['mode']
                    if not is_holdout(family, base.seed) or (family not in groups[mode] and len(groups[mode]) >= 6):
                        continue
                    groups[mode].add(family)
                    for ply in sorted({0, len(game_rows) // 2, len(game_rows) - 1}):
                        row = copy.deepcopy(game_rows[ply])
                        rows.append(row)
                        identities.append({'family': family, 'mode': mode, 'ply': int(row.ply),
                            'reason': game['reason'], 'repetition_context': bool(np.any(row.repetition_contexts > 0)),
                            'state_hash': int(row.state_hash), 'source': record['path'], 'wdl': row.wdl.tolist()})
            if not rows:
                raise ValueError('missing saved training holdout')
            corpora[name] = {'rows': rows, 'identities': identities, 'sources': records}
        models = {'retention86': retention['source_checkpoint']}
        guard = old['participants']['guard86']
        models['retention86'] = guard
        models['source110'] = old['source_checkpoint']
        for variant in ('control', 'candidate'):
            item = progress['variants'][variant]
            models[variant + '134'] = {'path': item['checkpoint'], 'sha256': item['sha256'], 'model_hash': item['model_hash'], 'iteration': item['counters']['iteration']}
        device = torch.device('cuda')
        torch.set_num_threads(4)
        results = {}
        for name, item in models.items():
            check()
            if sha(inside(root, item['path'])) != item['sha256']:
                raise ValueError('audit weights changed')
            saved = torch.load(inside(root, item['path']), map_location='cpu', weights_only=False)
            model = PolicyValueNet(base.model).to(device).eval()
            model.load_state_dict(saved['model'])
            if model_hash(model) != item['model_hash']:
                raise ValueError('audit tensor digest changed')
            results[name] = {}
            for corpus_name, corpus in corpora.items():
                check()
                probabilities = []
                with torch.inference_mode():
                    for start in range(0, len(corpus['rows']), 32):
                        part = corpus['rows'][start:start + 32]
                        padded = part + [part[-1]] * (32 - len(part))
                        batch = search_distill_batch(padded, device)
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                            logits = model.terminal_wdl_logits(batch['observations'])
                        probabilities.extend(torch.softmax(logits.float(), -1)[:len(part)].cpu().numpy().tolist())
                prediction = np.asarray(probabilities)
                truth = np.stack([row.wdl for row in corpus['rows']])
                ce = -(truth * np.log(np.maximum(prediction, 1e-9))).sum(-1)
                brier = ((prediction - truth) ** 2).sum(-1)
                bias = (prediction[:, 2] - prediction[:, 0]) - (truth[:, 2] - truth[:, 0])
                grouped = defaultdict(list)
                for i, row in enumerate(corpus['identities']):
                    grouped[(row['mode'], row['family'])].append(i)
                per_group = [{'mode': mode, 'family': family, 'wdl_ce': float(ce[ids].mean()),
                    'brier': float(brier[ids].mean()), 'value_bias': float(bias[ids].mean())}
                    for (mode, family), ids in grouped.items()]
                results[name][corpus_name] = {'rows': len(truth), 'groups': len(grouped),
                    'group_mean': {k: float(np.mean([g[k] for g in per_group])) for k in ('wdl_ce', 'brier', 'value_bias')},
                    'by_mode': {mode: {k: float(np.mean([g[k] for g in per_group if g['mode'] == mode]))
                        for k in ('wdl_ce', 'brier', 'value_bias')} for mode in sorted({g['mode'] for g in per_group})},
                    'by_reason': {reason: {'rows': len(ids), 'wdl_ce': float(ce[ids].mean()), 'brier': float(brier[ids].mean())}
                        for reason in sorted({i['reason'] for i in corpus['identities']})
                        for ids in [[j for j, i in enumerate(corpus['identities']) if i['reason'] == reason]]},
                    'probabilities': probabilities}
            del model, saved
        check()
        atomic_json(out, {'at': utc_now(), 'complete': True, 'training_only': True, 'new_training_games': 0, 'effective_updates': 0,
            'previous_plan': {'path': args.previous_plan, 'sha256': sha(inside(root, args.previous_plan))},
            'retention_plan': {'path': args.retention_plan, 'sha256': sha(inside(root, args.retention_plan))},
            'source_checkpoint': models['control134'], 'checkpoints': models,
            'corpora': {name: {'identities': item['identities'], 'sources': item['sources']} for name, item in corpora.items()},
            'value': results, 'precision': 'BF16 fixed physical32, FP32 probabilities',
            'interpretation': 'Same saved actual-terminal targets across models; descriptive trajectory-group holdout retention/calibration, not optimal value, layout generalization, chess strength or proof of forgetting as causal bottleneck. No counterfactual successors and no replay additions.'})
    print(json.dumps({'output': str(out.relative_to(root)), 'results': {name: {c: item['group_mean'] for c, item in values.items()} for name, values in results.items()}}))


if __name__ == '__main__':
    main()
