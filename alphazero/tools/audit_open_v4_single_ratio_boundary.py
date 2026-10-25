"""Small CPU saved-terminal fitting and exact sampler exposure audit; no training."""
import argparse
import copy
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace


def main():
    import numpy as np
    import torch
    from open_v4_runtime import read_json, atomic_json, utc_now, lock_held
    from open_v4_queue_contract import inside, sha
    from run_open_v4_fresh_data import is_holdout
    from junqi_search_v3.recovery import _family_balanced_select
    from junqi_search_v3.loop import read_selfplay_shard, search_distill_batch
    from junqi_open_v4.training_replay import complete_games
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--review',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    plan=read_json(inside(root,args.plan));run=inside(root,plan['directory']);progress=read_json(run/'PROGRESS.json');out=inside(root,args.output)
    if out.exists():raise ValueError('preserve audit')
    if lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock') or read_json(run/'STATUS.json')['state']!='review_due':raise ValueError('free natural boundary required')
    def check():
        if any(inside(root,p).exists() for p in plan['stop_paths']) or read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user'):raise InterruptedError('STOP/pause')
    check();review=read_json(inside(root,args.review))
    if not review['all_checks_passed'] or review['route_confirmed']:raise ValueError('verified unconfirmed boundary required')
    cfg=RootExploreConfig.load(inside(root,plan['parent_config']));torch.set_num_threads(4)
    latest=progress['variants']['learner'];saved=torch.load(inside(root,latest['checkpoint']),map_location='cpu',weights_only=False)
    source=dict(path=latest['checkpoint'],sha256=latest['sha256'],model_hash=latest['model_hash'],iteration=158,leaf_value_scale=saved['leaf_value_scale'])
    checks={};pools={};sources=progress['collections']['learner'][-5:]
    for entry in sources:
        if sha(inside(root,entry['path']))!=entry['sha256'] or sha(inside(root,entry['metadata']))!=entry['metadata_sha256']:raise ValueError('replay changed')
        meta=read_json(inside(root,entry['metadata']));pool=[]
        for index,game in enumerate(meta['games']):
            if not is_holdout(int(game['family']),cfg.seed):
                pool.extend(SimpleNamespace(game_id=int(game['family']),identity=(entry['path'],index,ply),source_stage=entry['stage']) for ply in range(len(game['trace'])))
        pools[entry['stage']]=pool
    coverage={count:{'policy':Counter(),'value':Counter()} for count in (64,256)};stages=[]
    hypothetical_generator=None;hypothetical_attempt=None
    for stage in (22,23,24):
        before=torch.load(run/'learner/checkpoints'/f'learner_stage_{stage-1:06d}.pt',map_location='cpu',weights_only=False)
        after=torch.load(run/'learner/checkpoints'/f'learner_stage_{stage:06d}.pt',map_location='cpu',weights_only=False)
        pool=[row for n in range(stage,stage-3,-1) for row in pools[n]];prefix=[]
        for count in (64,256):
            if count==64:
                gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(before['rng']['batch_generator']);attempt=before['counters']['attempted_updates']
            else:
                if hypothetical_generator is None:
                    hypothetical_generator=np.random.default_rng();hypothetical_generator.bit_generator.state=copy.deepcopy(before['rng']['batch_generator']);hypothetical_attempt=before['counters']['attempted_updates']
                gen=hypothetical_generator;attempt=hypothetical_attempt
            draws=[];phase_counts=Counter()
            for update in range(count):
                phase='value' if (attempt+update)%3==2 else 'policy';keys=gen.integers(0,2**63-1,size=cfg.logical_batch_size)
                selected=_family_balanced_select(pool,cfg.logical_batch_size,keys,family_cap_fraction=.02)
                draws.append([row.identity for row in selected]);phase_counts[phase]+=1
                coverage[count][phase].update(row.identity for row in selected if row.source_stage==22)
            if count==64:
                checks[f'stage{stage}:exact_actual_rng']=gen.bit_generator.state==after['rng']['batch_generator'];prefix=draws
                checks[f'stage{stage}:64_actual_updates']=after['counters']['effective_updates']-before['counters']['effective_updates']==64
            else:
                if stage==22:checks['hypothetical_first64_matches_actual']=prefix==draws[:64]
                hypothetical_attempt+=256
        stages.append(dict(stage=stage,pool_rows=len(pool),eligible_families=len({r.game_id for r in pool})))
    exposure={str(count):{phase:dict(sampled_rows=sum(counter.values()),unique_rows=len(counter),pool_rows=len(pools[22]),unique_fraction=len(counter)/len(pools[22]),mean_passes=sum(counter.values())/len(pools[22])) for phase,counter in phases.items()} for count,phases in coverage.items()}
    corpora={}
    for name,records in [('old134',plan['initial_replay']),('fresh158',progress['collections']['learner'][-3:])]:
        rows=[];ids=[];groups=defaultdict(set)
        for entry in records:
            check()
            if sha(inside(root,entry['path']))!=entry['sha256'] or sha(inside(root,entry['metadata']))!=entry['metadata_sha256']:raise ValueError('corpus changed')
            meta=read_json(inside(root,entry['metadata']));samples=read_selfplay_shard(inside(root,entry['path']))
            if meta['layout_role']!='train':raise ValueError('training only')
            for game,game_rows in complete_games(samples,meta):
                if is_holdout(int(game['family']),cfg.seed) and (game['family'] in groups[game['mode']] or len(groups[game['mode']])<6):
                    groups[game['mode']].add(game['family'])
                    for ply in sorted(set((0,len(game_rows)//2,len(game_rows)-1))):
                        rows.append(copy.deepcopy(game_rows[ply]));ids.append(dict(family=game['family'],mode=game['mode'],ply=ply,reason=game['reason'],source=entry['path']))
            del samples
        corpora[name]=dict(rows=rows,identities=ids,sources=records)
    models={'source134':plan['source_checkpoint'],'mid146':dict(path=str((run/'learner/checkpoints/learner_stage_000012.pt').relative_to(root))), 'latest158':source}
    value={}
    for name,entry in models.items():
        check();payload=torch.load(inside(root,entry['path']),map_location='cpu',weights_only=False)
        entry.update(sha256=sha(inside(root,entry['path'])),model_hash=payload['model_hash'])
        model=PolicyValueNet(cfg.model).cpu().eval();model.load_state_dict(payload['model']);assert model_hash(model)==payload['model_hash']
        value[name]={}
        for corpus_name,corpus in corpora.items():
            predictions=[]
            with torch.inference_mode():
                for start in range(0,len(corpus['rows']),8):
                    batch=search_distill_batch(corpus['rows'][start:start+8],torch.device('cpu'))
                    predictions.extend(torch.softmax(model.terminal_wdl_logits(batch['observations']).float(),-1).numpy().tolist())
            pred=np.asarray(predictions);truth=np.stack([row.wdl for row in corpus['rows']]);ce=-(truth*np.log(np.maximum(pred,1e-9))).sum(-1);brier=((pred-truth)**2).sum(-1);bias=(pred[:,2]-pred[:,0])-(truth[:,2]-truth[:,0]);grouped=defaultdict(list)
            for i,identity in enumerate(corpus['identities']):grouped[(identity['mode'],identity['family'])].append(i)
            per=[dict(mode=mode,family=family,wdl_ce=float(ce[idx].mean()),brier=float(brier[idx].mean()),value_bias=float(bias[idx].mean())) for (mode,family),idx in grouped.items()]
            value[name][corpus_name]=dict(rows=len(truth),groups=len(per),group_mean={key:float(np.mean([g[key] for g in per])) for key in ('wdl_ce','brier','value_bias')},by_mode={mode:{key:float(np.mean([g[key] for g in per if g['mode']==mode])) for key in ('wdl_ce','brier','value_bias')} for mode in {g['mode'] for g in per}})
        del model,payload
    checks['CPU_no_GPU_initialized']=not torch.cuda.is_initialized();check()
    if not all(checks.values()):raise ValueError(checks)
    atomic_json(out,dict(at=utc_now(),complete=True,training_only=True,cpu_only=True,new_training_games=0,effective_updates=0,
        source_checkpoint=source,previous_review=dict(path=args.review,sha256=sha(inside(root,args.review))),checks=checks,sampler_stages=stages,sources=sources,
        collection22_lifetime_exposure=exposure,hypothetical_sampling='Own continuous generator and attempted-update clock from stage21, advancing256 each stage; frozen original pools, no per-stage RNG reset.',value=value,checkpoints=models,corpora={k:{n:v[n] for n in ('identities','sources')} for k,v in corpora.items()},
        precision='CPU FP32 at identical saved actual-terminal samples; differs from BF16 arena',
        hypothesis='Increase all PPV learning updates64to256 per own fresh256games; both Policy and Value plus optimizer/RNG clocks increase. Frozen-shard hypothetical256 exposure is not actual future experience or a causal proof.',
        interpretation='Limited row coverage is measured, not evidence that every unobserved row must be fitted or that underfitting is the root cause. CE/Brier fit behavior terminal outcomes, not optimal Value, layout generalization or external strength. More reuse may overfit/forget and costs more learning; preserve protected arena gates.'))
    print(str(out.relative_to(root)));print(exposure);print({k:{c:r['group_mean'] for c,r in v.items()} for k,v in value.items()})

if __name__=='__main__':main()
