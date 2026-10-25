"""Exact CPU sampler exposure replay plus saved-terminal calibration evidence."""
from pathlib import Path
from types import SimpleNamespace
from collections import Counter
import copy

def main():
    import argparse,json,numpy as np,torch
    from open_v4_runtime import atomic_json,read_json,utc_now,lock_held
    from open_v4_queue_contract import sha,inside
    from run_open_v4_fresh_data import is_holdout
    from junqi_search_v3.recovery import _family_balanced_select
    p=argparse.ArgumentParser();p.add_argument('--previous-plan',required=True);p.add_argument('--value-audit',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];old=read_json(inside(root,a.previous_plan));directory=inside(root,old['directory']);out=inside(root,a.output)
    if out.exists():raise ValueError('preserve immutable audit')
    if read_json(directory/'STATUS.json')['state']!='review_due' or lock_held(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'):raise ValueError('completed free natural boundary required')
    if any(inside(root,n).exists() for n in old['stop_paths']):raise ValueError('STOP requested')
    progress=read_json(directory/'PROGRESS.json');records=progress['collections']['control'];seed=read_json(inside(root,old['parent_config']))['seed']
    pools={};sources=[]
    for c in records[-5:]:
        if sha(inside(root,c['path']))!=c['sha256'] or sha(inside(root,c['metadata']))!=c['metadata_sha256']:raise ValueError('corpus changed')
        meta=read_json(inside(root,c['metadata']));rows=[]
        if meta['layout_role']!='train':raise ValueError('training only')
        # Use family+ply for distinct identities: one paired family may have two
        # colors, so include the actual metadata game index rather than loop0.
        rows=[]
        for game_index,game in enumerate(meta['games']):
            if not is_holdout(int(game['family']),seed):
                rows.extend(SimpleNamespace(game_id=int(game['family']),identity=(c['path'],game_index,ply),source_stage=c['stage']) for ply in range(len(game['trace'])))
        pools[c['stage']]=rows;sources.append(c)
    checks={};coverage={'policy':Counter(),'value':Counter()};stages=[]
    for stage in (22,23,24):
        before=directory/'control/checkpoints'/f'learner_stage_{stage-1:06d}.pt';after=directory/'control/checkpoints'/f'learner_stage_{stage:06d}.pt'
        previous=torch.load(before,map_location='cpu',weights_only=False);saved=torch.load(after,map_location='cpu',weights_only=False)
        gen=np.random.default_rng();gen.bit_generator.state=copy.deepcopy(previous['rng']['batch_generator']);attempt=previous['counters']['attempted_updates']
        pool=[row for n in range(stage,stage-3,-1) for row in pools[n]];phase_counts=Counter()
        for update in range(64):
            phase='value' if (attempt+update)%3==2 else 'policy';keys=gen.integers(0,2**63-1,size=256)
            selected=_family_balanced_select(pool,256,keys,family_cap_fraction=.02);phase_counts[phase]+=1
            coverage[phase].update(row.identity for row in selected if row.source_stage==22)
        checks[f'stage{stage}:exact_sampler_rng']=gen.bit_generator.state==saved['rng']['batch_generator']
        checks[f'stage{stage}:attempts']=attempt+64==saved['counters']['attempted_updates']
        stages.append({'stage':stage,'pool_rows':len(pool),'families':len({s.game_id for s in pool}),'phase_updates':dict(phase_counts),'before_sha256':sha(before),'after_sha256':sha(after)})
    sample=pools[22];family_by_id={s.identity:s.game_id for s in sample};family_count=len(set(family_by_id.values()))
    totals={}
    for phase,c in coverage.items():
        family=Counter()
        for identity,n in c.items():family[family_by_id[identity]]+=n
        totals[phase]={'pool_rows':len(sample),'eligible_families':family_count,'sampled_rows':sum(c.values()),'unique_rows':len(c),'unique_fraction':len(c)/len(sample),
            'never_sampled_rows':len(sample)-len(c),'mean_passes':sum(c.values())/len(sample),'families_seen':len(family),'min_per_family':min(family.values()),'max_per_family':max(family.values())}
    v=read_json(inside(root,a.value_audit));item=progress['variants']['control'];payload=torch.load(inside(root,item['checkpoint']),map_location='cpu',weights_only=False)
    source={'path':item['checkpoint'],'sha256':item['sha256'],'model_hash':item['model_hash'],'iteration':payload['counters']['iteration'],'leaf_value_scale':payload['leaf_value_scale']}
    checks['value_audit_complete']=v['complete'] and v['training_only'] and not v['new_training_games'] and not v['effective_updates']
    checks['same_complete_source']=v['source_checkpoint']['sha256']==source['sha256']
    if not all(checks.values()):raise ValueError([k for k,x in checks.items() if not x])
    atomic_json(out,{'at':utc_now(),'complete':True,'training_only':True,'cpu_exposure_only':True,'new_training_games':0,'effective_updates':0,'source_checkpoint':source,
        'checks':checks,'sources':sources,'sampler_stages':stages,'collection22_lifetime_exposure':totals,
        'value_audit':{'path':a.value_audit,'sha256':sha(inside(root,a.value_audit))},
        'hypothesis':'More learning per same fresh256games could increase useful optimization before replay eviction. Both Policy and Value updates increase together; not a Value-only intervention.',
        'interpretation':'Exact generator reproduction confirms batch exposure accounting. Limited row exposure is measured, not proof of underfitting or a causal bottleneck. Earlier Value eligibility/encoder factors retained64updates and differ from all-PPV update ratio. Current source improves old86 terminal fitting; general forgetting is unsupported. More updates also advance AdamW/weight decay/random draws and cost time; strength must pass unchanged protected development gates.'})
    print(json.dumps({'output':str(out.relative_to(root)),'checks':checks,'coverage':totals}))
if __name__=='__main__':main()
