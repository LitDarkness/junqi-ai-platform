"""Bounded train-only Policy/Value/search diagnosis at a completed boundary.

Native two-ply proofs label only the actually proven losing successors.
Unknown alternative successors never inherit the recorded trajectory outcome.
This does no training, collection, model selection, or final-layout evaluation.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from contextlib import ExitStack
import hashlib
from pathlib import Path
import time


def main():
    import numpy as np
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock
    from run_open_v4_fresh_data import is_holdout
    from run_open_v4_search_training import verify_learning_state, paired_schedule
    from junqi_az.model import PolicyValueNet
    from junqi_az.config import ModelConfig
    from junqi_open_v4.layouts import backend_for, load_role
    from junqi_open_v4.training_replay import complete_games
    from junqi_open_v4.arena import paired_interval, select_families
    from junqi_search_v3.loop import read_selfplay_shard, search_distill_batch
    from junqi_search_v3.search import BatchedNeuralPUCT
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    run = root / 'runs/open_junqi_v4_tactical_training_20261004_065605'
    plan = read_json(root / 'config/experiments/open_v4_tactical_training_20261004_065605.json')
    progress = read_json(run / 'PROGRESS.json')
    status = read_json(run / 'STATUS.json')
    if status['state'] != 'review_due' or any((root/p).exists() for p in plan['stop_paths']):
        raise ValueError('only an unstopped, naturally completed boundary may be audited')
    out = root / 'reports/current_training_audit_20261004'
    with ExitStack() as locks:
        for path in sorted((root/'runs').glob('open_junqi_v4_*/CURRENT.json')):
            for name in ('SUPERVISOR', 'TRAINING'):
                locks.enter_context(ProcessLock(path.parent / f'supervision/{name}.lock'))
        for name, digest in plan['source'].items():
            assert sha(root/name) == digest, name
        for item in plan['historical_stops']:
            assert sha(root/item['path']) == item['sha256']
        for item in plan['frozen_runs']:
            assert sha(root/item['directory']/'CURRENT.json') == item['current_sha256']
        for variant in ('control', 'candidate'):
            for stage in range(1, 7):
                assert all(read_json(run / f'verification/{variant}_{stage:06d}.json')['checks'].values())
                meta = read_json(root/progress['collections'][variant][stage-1]['metadata'])
                previous = torch.load(run / f'{variant}/checkpoints/learner_stage_{stage-1:06d}.pt', map_location='cpu', weights_only=False)
                assert meta['actor_model_hash'] == previous['model_hash']
                del previous
            assert progress['variants'][variant]['counters']['effective_updates'] - plan['source_counters']['effective_updates'] == 384
        for left, right in zip(progress['collections']['control'], progress['collections']['candidate']):
            assert paired_schedule(read_json(root/left['metadata'])) == paired_schedule(read_json(root/right['metadata']))
        arena_path = root/progress['evaluations']['block_000006_screen']['path']
        arena = read_json(arena_path)
        assert sha(arena_path) == progress['evaluations']['block_000006_screen']['sha256']
        summaries = arena['summaries']; family_scores = []
        for index, item in enumerate(summaries):
            games = [g for g in arena['games'] if g['matchup'] == index]
            grouped = defaultdict(list)
            for g in games:
                assert g['score'] == .5 + .5*g['value']
                grouped[g['family']].append(g['score'])
            scores = {f:float(np.mean(v)) for f,v in grouped.items()}; family_scores.append(scores)
            check = paired_interval(list(scores.values()), arena['seed'] ^ index)
            assert check == item['family_score']
        for item in arena['paired_deltas']:
            a,b = item['from_matchup'], item['to_matchup']
            common = sorted(family_scores[a].keys() & family_scores[b].keys())
            assert paired_interval([family_scores[b][f]-family_scores[a][f] for f in common], arena['seed']^(a<<8)^b) == item['delta']
        dev = {r['id']:r for r in load_role(root/'config/layouts/open_v4','dev')}
        suite = select_families(read_json(root/plan['suite']['path'])['states'],plan['screen_families'],plan['evaluation_seed'])
        for g in sorted(arena['games'], key=lambda g:g['id'])[::140]:
            d=suite[g['position']]
            assert d['red']==dev[d['red']['id']] and d['black']==dev[d['black']['id']]
            s=backend_for(d['red'], d['black']).new_state(d['setup_seed'])
            for a in d['trace']+g['trace']: s.apply(a)
            assert s.is_terminal() and s.terminal_reason()==g['reason'] and s.terminal_value(g['candidate_side'])==g['value']
        review={'at':utc_now(),'phase':'completed_online_training_and_development_screen',
                'new_games':3072,'effective_updates':768,'arena_games':len(arena['games']),
                'arena_sha256':sha(arena_path),'summary_and_paired_intervals_recomputed':True,
                'native_terminal_replays':len(sorted(arena['games'],key=lambda g:g['id'])[::140]),
                'all_previous_learner_collectors_and_paired_schedules_verified':True,
                'screen_passed':False,'route_confirmed':False,'full64_run':False,
                'repeat_baseline_drift_pp':-2.8645833333333335,
                'interpretation':'Unconfirmed guard learning benefit. Repeated baseline also exceeds prespecified2pp tolerance. Near-zero bootstrap lower limits are numerical zero, not positive evidence.',
                'summaries':summaries,'paired_deltas':arena['paired_deltas']}
        atomic_json(run/'supervision/REVIEWS/RESULT_REVIEW_20261004_COMPONENT_BOUNDARY.json',review)
        train={r['id']:r for r in load_role(root/'config/layouts/open_v4','train')}
        audit_path=out/'TACTICAL_TARGET_AUDIT.json'; audit=read_json(audit_path)
        roots=[]; children=[]; cases=[];noise_seeds=[]
        for variant,item in audit['variants'].items():
            historic_meta=read_json(root/item['source']['metadata'])
            for c in item['cases']:
                p=c['provenance']; red=train[p['red_layout_id']]; black=train[p['black_layout_id']]
                s=backend_for(red,black,mirrored=list(red['setup'])!=p['red_layout']).new_state(p['setup_seed'])
                assert list(s.native_state.board())==p['initial_board']
                for a in c['prefix']: s.apply(a)
                assert s.legal_actions().tolist()==c['legal']
                child=s.clone(); child.apply(c['action']); assert not child.is_terminal()
                replies=[]
                for a in child.legal_actions():
                    if child.native_state.board()[int(a)%60]==-3:
                        terminal=child.clone(); terminal.apply(int(a))
                        assert terminal.is_terminal() and terminal.terminal_value(c['player'])==-1
                        replies.append(int(a))
                assert replies
                matches=[g for g in historic_meta['games'] if g['family']==c['family'] and g['trace'][:c['ply']]==c['prefix'] and g['trace'][c['ply']]==c['action']]
                assert len(matches)==1
                noise_seed=2026100307 ^ (historic_meta['iteration']<<20) ^ matches[0]['index'] ^ (c['ply']*65537)
                noise_seeds.append(noise_seed)
                cases.append(dict(variant=variant, family=c['family'],ply=c['ply'],action=c['action'],
                                  safe_actions=c['safe_actions'],native_winning_replies=replies,
                                  root_key=int(s.native_state.key()),historical_target=c['visits'],
                                  policy_contract=c['policy_contract'],original_root_noise_seed=noise_seed))
                roots.append(s);children.append(child)
        assert len(roots)==9
        old_progress=read_json(root/'runs/open_junqi_v4_self_training_20261004_023128/PROGRESS.json')
        sources={'champion26':plan['participants']['champion'],'parent32':plan['source_checkpoint']}
        sources.update({f'{v}38':dict(path=i['checkpoint'],sha256=i['sha256'],model_hash=i['model_hash']) for v,i in progress['variants'].items()})
        for v in ('control','candidate'):
            meta=read_json(root/old_progress['collections'][v][-1]['metadata'])
            actor=meta['actor_checkpoint'];sources[f'historical_{v}31']=dict(path=actor['checkpoint'],sha256=actor['sha256'],model_hash=actor['model_hash'])
        # Bounded held-out trajectories, stratified by collection mode. No dev rows.
        corpora={}; row_ids={}; corpus_sources={}
        for name,items in [('parent32',plan['initial_replay']),('control38',progress['collections']['control'][-3:]),('candidate38',progress['collections']['candidate'][-3:])]:
            selected=[]; ids=[]; seen=Counter(); source_records=[]
            for item in items:
                assert sha(root/item['path'])==item['sha256'] and sha(root/item['metadata'])==item['metadata_sha256']
                meta=read_json(root/item['metadata']);samples=read_selfplay_shard(root/item['path'])
                for g,rows in complete_games(samples,meta):
                    if not is_holdout(int(g['family']),2026100307) or seen[g['mode']]>=12: continue
                    seen[g['mode']]+=1
                    for ply in sorted(set((0,len(rows)//2,len(rows)-1))):
                        import copy
                        r=copy.copy(rows[ply]);r.observation=r.observation.copy()
                        selected.append(r);ids.append(dict(group=g['family'],mode=g['mode'],ply=ply,source=item['path'],result_red=g['result_red']))
                source_records.append(item);del samples
            corpora[name]=selected;row_ids[name]=ids;corpus_sources[name]=source_records
        def request(s):
            return dict(features=s.feature_planes(), legal_actions=s.legal_actions(),repetition_contexts=s.repetition_contexts(),state_key=int(s.native_state.key()))
        torch.set_num_threads(4); device=torch.device('cuda')
        torch.cuda.reset_peak_memory_stats(); started=time.perf_counter()
        results={}
        for name,src in sources.items():
            assert sha(root/src['path'])==src['sha256']
            payload=torch.load(root/src['path'],map_location='cpu',weights_only=False)
            model=PolicyValueNet(ModelConfig(**payload['config']['model'])).to(device).eval()
            model.load_state_dict(payload['model']); model.search_v3_leaf_value_scale=payload['leaf_value_scale']
            from junqi_open_v4.learner import model_hash
            assert model_hash(model)==src['model_hash']
            search=BatchedNeuralPUCT(model,device=device,cuda_graphs=False)
            inferred=search._infer([request(s) for s in roots+children])
            tactical=[]
            for i,c in enumerate(cases):
                probs,val=inferred[i]; _,child_val=inferred[i+9]
                legal=roots[i].legal_actions(); safe=np.isin(legal,c['safe_actions'])
                best=int(legal[int(np.argmax(probs))])
                tactical.append(dict(**c,raw_policy_top=best,raw_top_safe=best in c['safe_actions'],
                    raw_selected_probability=float(probs[np.where(legal==c['action'])[0][0]]),
                    raw_safe_mass=float(probs[safe].sum()),raw_value=val,
                    proven_loss_child_opponent_value=child_val,
                    proven_loss_root_value=-child_val,correct_child_minimax_value=1.,
                    caveat='Safe only means no immediate opposing flag capture, not a winning or minimax-labelled alternative.'))
            for budget in (64,256):
                moves,metrics=search.search(roots,budget)
                for c,move in zip(tactical,moves):
                    idx=int(np.where(move.actions==c['action'])[0][0])
                    safe=np.isin(move.actions,c['safe_actions']);weights=move.visits.astype(float)**(1 if c['ply']<24 else 2)
                    c[f'search{budget}']=dict(greedy_action=move.action,greedy_safe=move.action in c['safe_actions'],
                         selected_visits=int(move.visits[idx]),selected_q=float(move.q_values[idx]),
                         unsafe_sampling_mass=float(weights[~safe].sum()/max(weights.sum(),1)),
                         actions=move.actions.tolist(),visits=move.visits.tolist(),q_values=move.q_values.tolist(),metrics=metrics)
                noisy,metrics=search.search(roots,budget,add_root_noise=True,noise_seeds=noise_seeds)
                for c,move in zip(tactical,noisy):
                    safe=np.isin(move.actions,c['safe_actions']); weights=move.visits.astype(float)**(1 if c['ply']<24 else 2)
                    c[f'noisy_search{budget}']=dict(greedy_action=move.action,greedy_safe=move.action in c['safe_actions'],
                        unsafe_sampling_mass=float(weights[~safe].sum()/max(weights.sum(),1)),metrics=metrics)
            calibration={}
            with torch.inference_mode():
                for corpus,rows in corpora.items():
                    probs=[]
                    for start in range(0,len(rows),64):
                        b=search_distill_batch(rows[start:start+64],device)
                        with torch.autocast('cuda',dtype=torch.bfloat16): logits=model.terminal_wdl_logits(b['observations'])
                        probs.extend(torch.softmax(logits.float(),-1).cpu().numpy())
                    p=np.asarray(probs); y=np.stack([r.wdl for r in rows]); v=p[:,2]-p[:,0];truth=y[:,2]-y[:,0]
                    mode={}
                    for m in ('selfplay','teacher_selfplay','teacher_vs_actor'):
                        mask=np.array([r['mode']==m for r in row_ids[corpus]])
                        mode[m]=dict(rows=int(mask.sum()),wdl_ce=float(-(y[mask]*np.log(np.maximum(p[mask],1e-9))).sum(-1).mean()),
                                     value_mae=float(np.abs(v[mask]-truth[mask]).mean()),mean_value=float(v[mask].mean()),
                                     true_mean=float(truth[mask].mean()))
                    calibration[corpus]=dict(rows=len(rows),trajectory_groups=len(set(r.game_id for r in rows)),
                        wdl_ce=float(-(y*np.log(np.maximum(p,1e-9))).sum(-1).mean()),uniform_ce=float(np.log(3)),
                        brier=float(((p-y)**2).sum(-1).mean()),value_mae=float(np.abs(v-truth).mean()),
                        accuracy=float((p.argmax(-1)==y.argmax(-1)).mean()),modes=mode,
                        target=y.tolist(),prediction=p.tolist())
            results[name]=dict(source=src,counters=payload['counters'],leaf_value_scale=payload['leaf_value_scale'],
                               tactical=tactical,heldout_trajectory_calibration=calibration)
            print(name, 'safe greedy64/256',sum(c['search64']['greedy_safe'] for c in tactical),sum(c['search256']['greedy_safe'] for c in tactical),
                  'known losing child mean',float(np.mean([c['proven_loss_child_opponent_value'] for c in tactical])),
                  'heldout CE', {k:round(v['wdl_ce'],4) for k,v in calibration.items()},flush=True)
            del model,search,payload;torch.cuda.empty_cache()
        report=dict(at=utc_now(),phase='frozen_train_only_component_diagnosis',new_games=0,effective_updates=0,
            tactical_case_source=dict(path=str(audit_path.relative_to(root)),sha256=sha(audit_path)),
            diagnostic_code_sha256=sha(Path(__file__)),models=results,
            calibration_sources=corpus_sources,calibration_row_ids=row_ids,
            inference='Existing BF16, no CUDA graph, same9 full-history roots batched; reruns do not claim historical batch-exact reproduction. Search unguarded and no root noise.',
            elapsed_seconds=time.perf_counter()-started,gpu_peak_mb=torch.cuda.max_memory_allocated()/2**20,
            replay_training_rows=0,
            caveats='9selected failure examples not prevalence/strength sample; heldout trajectory groups not unseen layouts. All target outcomes from actual recorded native trajectories. No counterfactual unknown successors receive those outcomes.')
        atomic_json(out/'RL_COMPONENT_DIAGNOSTIC.json',report)
        print(out/'RL_COMPONENT_DIAGNOSTIC.json',flush=True)


if __name__=='__main__': main()
