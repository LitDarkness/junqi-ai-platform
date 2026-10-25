"""Frozen output crossover and train-only Value calibration; no weight updates."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path


def make_cross_model(policy, value):
    import torch
    from torch import nn
    if repr(policy.config) != repr(value.config):
        raise ValueError("crossover requires identical observation/model contracts")
    if policy.search_v3_leaf_value_scale != value.search_v3_leaf_value_scale:
        raise ValueError("crossover must preserve the common Value scale")

    class OutputCrossover(nn.Module):
        def __init__(self):
            super().__init__()
            self.policy_model, self.value_model = policy, value
            self.config = policy.config
            self.search_v3_leaf_value_scale = value.search_v3_leaf_value_scale

        def forward(self, observations, action_mask=None, repetition_context=None):
            if self.policy_model is self.value_model:
                return self.policy_model(observations, action_mask, repetition_context)
            logits = self.policy_model.policy_logits(observations, action_mask, repetition_context)
            _, values = self.value_model(observations, action_mask, repetition_context)
            return logits, values

        @torch.inference_mode()
        def policy_value(self, observations, action_mask, repetition_context=None):
            logits, values = self(observations, action_mask, repetition_context)
            return torch.softmax(logits.float(), dim=-1), values.float()

    return OutputCrossover().eval()


def calibration_summary(probabilities, targets, search_values, families):
    import numpy as np
    p, y, v = np.asarray(probabilities), np.asarray(targets), np.asarray(search_values)
    if p.shape != y.shape or p.ndim != 2 or p.shape[1] != 3 or len(v) != len(p) or len(families) != len(p):
        raise ValueError("invalid WDL audit shapes")
    if not np.all(np.isfinite(p)) or not np.allclose(p.sum(-1), 1., atol=1e-5) or not np.allclose(y.sum(-1), 1.):
        raise ValueError("invalid WDL probabilities or targets")
    if not np.all((y == 0) | (y == 1)) or np.any(p < 0):
        raise ValueError("native terminal audit needs hard WDL labels")
    true_value = y[:, 2] - y[:, 0]
    ce = -(y * np.log(np.maximum(p, 1e-9))).sum(-1)
    brier = ((p-y)**2).sum(-1)
    rows = np.stack((ce, brier, (v-true_value)**2, v-true_value), axis=1)
    grouped = {}
    for family, row in zip(families, rows):
        grouped.setdefault(int(family), []).append(row)
    means = {str(k): np.mean(a, axis=0).tolist() for k, a in grouped.items()}
    mean = np.mean(list(means.values()), axis=0)
    confidence, correct = p.max(-1), p.argmax(-1) == y.argmax(-1)
    bins, ece = [], 0.
    for low in np.arange(0., 1., .1):
        selected = (confidence >= low) & (confidence < low+.1 if low < .9 else confidence <= 1.)
        if selected.any():
            gap = abs(float(confidence[selected].mean()) - float(correct[selected].mean()))
            ece += float(selected.mean()) * gap
            bins.append(dict(low=float(low), rows=int(selected.sum()), confidence=float(confidence[selected].mean()), accuracy=float(correct[selected].mean())))
    return dict(samples=len(p), training_game_families=len(means), family_wdl_ce=float(mean[0]),
        family_brier=float(mean[1]), family_scaled_search_value_mse=float(mean[2]), family_value_bias=float(mean[3]),
        row_weighted_confidence_ece=ece, confidence_bins=bins, true_wdl_fraction=y.mean(0).tolist(),
        predicted_wdl_fraction=p.mean(0).tolist(), family_metrics=means,
        caveat="Frozen training-layout internal holdout; calibration is not external strength or a promotion gate.")


def head_primary_signal(gains, duplicate, duels, families, tolerance):
    enough = lambda rows: len(rows) == 2 and all(r['families'] >= families and r.get('uncertainty_estimable', False) for r in rows)
    return bool(len(gains) == 2 and all(enough(rows) and all(r['ci95'][0] > 0 for r in rows) for rows in gains)
        and enough(duplicate) and all(abs(r['mean']) <= tolerance for r in duplicate)
        and len(duels) == 2 and all(r['families'] >= families and r['ci95'][0] > .5 for r in duels))


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from run_open_v4_fresh_data import is_holdout
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.arena import run_arena, select_families, paired_interval
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.learner import model_hash
    from junqi_az.model import PolicyValueNet
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_search_v3.loop import read_selfplay_shard, search_distill_batch
    parser = argparse.ArgumentParser(); parser.add_argument('--experiment', required=True); args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def inside(name):
        path = (root/name).resolve()
        if not path.is_relative_to(root) or path == root: raise ValueError('paths must stay inside alphazero')
        return path
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path = inside(args.experiment); plan = read_json(plan_path); base = OpenConfig.load(inside(plan['parent_config']))
    if base.digest() != plan['parent_config_hash'] or plan['evaluation_families'] != 32 or plan['evaluation_budget'] != 256 or sha(Path(__file__)) != plan['tool_sha256']:
        raise ValueError('registered frozen screen required')
    directory = inside(plan['directory']); stops = [inside(p) for p in plan['stop_paths']]+[directory/'STOP']
    if any(p.exists() for p in stops): raise RuntimeError('STOP requested; never clear automatically')
    with ExitStack() as locks:
        for row in plan['frozen_runs']:
            for name in ('SUPERVISOR', 'TRAINING'): locks.enter_context(ProcessLock(inside(row['directory'])/f'supervision/{name}.lock'))
        if directory.exists(): raise ValueError('preserve existing experiment; no automatic retry')
        directory.mkdir(parents=True)
        status = dict(schema='junqi-frozen-head-crossover-1', state='checking', pid=os.getpid(), started_at=utc_now(), updates=0, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now()); atomic_json(directory/'STATUS.json', status)
        def check():
            if any(p.exists() for p in stops):
                update(state='stopped_by_user'); raise RuntimeError('STOP requested; frozen checkpoints preserved')
            if shutil.disk_usage(directory).free < 10*2**30: raise RuntimeError('minimum disk reserve reached')
        def verify():
            for row in plan['frozen_runs']:
                r = inside(row['directory']); current = row['current']
                if read_json(r/'CURRENT.json') != current or sha(r/'CONTRACT.json') != current['contract_hash']: raise ValueError('frozen source run changed')
                for role in ('learner','actor','champion'):
                    if sha(r/current[role]['path']) != current[role]['sha256']: raise ValueError('frozen role changed')
            for path, digest in {**plan['source'], **plan['evidence_files']}.items():
                if sha(inside(path)) != digest: raise ValueError('bound source/evidence changed')
        update()
        try:
            check(); verify()
            if sha(inside(base.layout_directory)/'manifest.json') != plan['layout_manifest_hash']: raise ValueError('layout boundary changed')
            atomic_json(directory/'CONTRACT.json', plan | dict(tool_sha256=sha(Path(__file__)), plan_sha256=sha(plan_path)))
            torch.set_num_threads(base.workers); device = torch.device(base.device); models = {}; checkpoint_proof = {}
            for name, item in plan['checkpoints'].items():
                path = inside(item['path'])
                if sha(path) != item['sha256']: raise ValueError('complete checkpoint changed')
                payload = torch.load(path, map_location='cpu', weights_only=False)
                if not all(k in payload for k in ('model','optimizer','scheduler','counters','rng','config','config_hash','leaf_value_scale','model_hash')): raise ValueError('incomplete learning state')
                model = PolicyValueNet(base.model).to(device).eval(); model.load_state_dict(payload['model']); model.search_v3_leaf_value_scale = payload['leaf_value_scale']
                if model_hash(model) != item['model_hash'] or payload['leaf_value_scale'] != plan['leaf_value_scale']: raise ValueError('tensor or Value scale mismatch')
                models[name] = model; checkpoint_proof[name] = item | dict(counters=payload['counters'], full_state_verified=True)
            validation = []
            for item in plan['internal_holdout_sources']:
                path = inside(item['path']); metadata = inside(item['metadata'])
                if sha(path) != item['sha256'] or sha(metadata) != item['metadata_sha256'] or read_json(metadata)['layout_role'] != 'train': raise ValueError('fixed train-only audit source changed')
                validation.extend(s for s in read_selfplay_shard(path) if is_holdout(s.game_id, base.seed))
            validation = validation[:4096]
            if not validation: raise ValueError('internal training holdout missing')
            update(state='calibrating', internal_samples=len(validation))
            calibration = {}; started = time.perf_counter()
            for name, model in models.items():
                predictions, targets, values, families = [], [], [], []
                with torch.inference_mode():
                    for start in range(0,len(validation),64):
                        check(); part = validation[start:start+64]; batch = search_distill_batch(part,device)
                        with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
                            _, value, _, wdl, _ = model.forward_heads(batch['observations'],batch['legal_mask'],batch['repetition'])
                        predictions.extend(torch.softmax(wdl.float(),-1).cpu().numpy()); targets.extend(batch['wdl'].cpu().numpy())
                        values.extend((value.float()*model.search_v3_leaf_value_scale).cpu().numpy()); families.extend(s.game_id for s in part)
                calibration[name] = calibration_summary(predictions,targets,values,families)
                print(f"[calibration] model={name} samples={len(validation)} family_wdl_ce={calibration[name]['family_wdl_ce']:.6f} family_brier={calibration[name]['family_brier']:.6f}",flush=True)
            deltas = {}
            for name in models:
                if name == 'parent': continue
                first, second = calibration['parent']['family_metrics'], calibration[name]['family_metrics']; keys = sorted(first)
                deltas[name] = {metric: paired_interval([second[k][i]-first[k][i] for k in keys],plan['evaluation_seed']^i) | dict(unit='training game families; internal diagnostics only') for i,metric in enumerate(('wdl_ce','brier','scaled_value_mse','value_bias'))}
            atomic_json(directory/'CALIBRATION.json',dict(models=calibration,paired_change_vs_parent=deltas,sources=plan['internal_holdout_sources'],elapsed_seconds=time.perf_counter()-started,role='train-internal-only'))
            components = {'parent':('parent','parent'),'trained':('trained','trained'),'hybrid':('trained','parent'),'value_changed':('parent','trained'),'repeat':('parent','parent')}
            agents = {name:StopAwareSearch(BatchedNeuralPUCT(make_cross_model(models[p],models[v]),device=device,inference_batch_size=base.inference_batch_size,cache_size=base.cache_size,leaves_per_tree=base.leaves_per_tree),check) for name,(p,v) in components.items()}
            suite_path = inside(plan['suite']['path']); suite_payload = read_json(suite_path)
            if sha(suite_path) != plan['suite']['sha256'] or suite_payload['contract']['role'] != 'dev': raise ValueError('development suite changed')
            suite = select_families(suite_payload['states'],32,plan['evaluation_seed'])
            matchups = [('hybrid','parent'),('hybrid','trained')]+[(name,f'teacher{i}') for i in range(len(base.teacher_candidates)) for name in agents]
            update(state='arena',evaluation_families=32,evaluation_budget=256,starts=len(suite),planned_games=2*len(suite)*len(matchups),components=components)
            with TeacherLeague(root,base.teacher_candidates,base.workers) as teachers:
                if teachers.provenance != plan['teachers']: raise ValueError('frozen teacher changed')
                arena = run_arena(suite,matchups,agents,teachers,{name:256 for name in agents},seed=plan['evaluation_seed'])
            check(); verify()
            if any(model_hash(model) != plan['checkpoints'][name]['model_hash'] for name,model in models.items()): raise ValueError('frozen weights mutated')
            arena.update(components=components,checkpoints=checkpoint_proof,purpose=plan['hypothesis'],compute_caveat=plan['compute_caveat'])
            atomic_json(directory/'arena.json',arena)
            gains = {key:teacher_deltas(arena,before,after) for key,before,after in (('hybrid_vs_parent','parent','hybrid'),('hybrid_vs_trained','trained','hybrid'),('value_changed_vs_parent','parent','value_changed'),('trained_vs_parent','parent','trained'),('duplicate','parent','repeat'))}
            selected = head_primary_signal([gains['hybrid_vs_parent'],gains['hybrid_vs_trained']],gains['duplicate'],[arena['summaries'][i]['family_score'] for i in (0,1)],32,.02)
            result = dict(schema='junqi-head-crossover-result-1',summaries=arena['summaries'],teacher_deltas=gains,primary_frozen_value_signal=selected,training_updates=0,promotion='none;32-family screen requires full64 confirmation and a justified learning experiment',components=components,checkpoints=checkpoint_proof,arena_seconds=arena['elapsed_seconds'],search_metrics=arena['search_metrics'],peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20,compute_caveat=plan['compute_caveat'])
            atomic_json(directory/'RESULTS.json',result);update(state='review_due',review_required=True,primary_frozen_value_signal=selected)
            print(json.dumps(dict(primary_frozen_value_signal=selected)),flush=True)
        except BaseException as error:
            if status['state'] != 'stopped_by_user': update(state='needs_attention',reason=repr(error),review_required=True)
            raise


if __name__ == '__main__': main()
