"""Small saved-training audit at the naturally completed110 boundary."""
import copy
from collections import defaultdict, Counter
from pathlib import Path


def main():
    import argparse, numpy as np, torch
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock, lock_held
    from open_v4_queue_contract import sha, inside
    from run_open_v4_fresh_data import is_holdout
    from junqi_open_v4.training_replay import complete_games
    from junqi_search_v3.loop import read_selfplay_shard, search_distill_batch
    from junqi_open_v4.queued_config import QueuedConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    parser=argparse.ArgumentParser(); parser.add_argument('--previous-plan',required=True); parser.add_argument('--output',required=True);parser.add_argument('--native-manifest',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    import sys
    native=read_json(inside(root,args.native_manifest));sys.path.insert(0,str(inside(root,native['directory'])))
    old=read_json(inside(root,args.previous_plan));directory=inside(root,old['directory'])
    progress=read_json(directory/'PROGRESS.json');source=progress['variants']['control']
    source_entry={'path':source['checkpoint'],'sha256':source['sha256'],'model_hash':source['model_hash'],
                  'iteration':source['counters']['iteration']}
    source_payload=torch.load(inside(root,source_entry['path']),map_location='cpu',weights_only=False)
    source_entry['leaf_value_scale']=source_payload['leaf_value_scale'];del source_payload
    if read_json(directory/'STATUS.json')['state']!='review_due' or source_entry['iteration']!=110:
        raise ValueError('completed latest110 required')
    output=inside(root,args.output)
    if output.exists():raise ValueError('do not replace an audit')
    stops=[inside(root,n) for n in old['stop_paths']]
    def check():
        if any(p.exists() for p in stops) or read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user',False):
            raise InterruptedError('STOP or pause requested')
    check()
    if lock_held(directory/'supervision/TRAINING.lock'):raise ValueError('GPU still owned')
    with ProcessLock(root/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'), ProcessLock(directory/'supervision/TRAINING.lock'):
        check();base=QueuedConfig.load(inside(root,old['parent_config']));corpora={};labels={}
        for variant in ('control','candidate'):
            rows=[];identities=[];groups=defaultdict(set);counts=Counter();sources=progress['collections'][variant][-3:]
            for entry in sources:
                if sha(inside(root,entry['path']))!=entry['sha256'] or sha(inside(root,entry['metadata']))!=entry['metadata_sha256']:
                    raise ValueError('saved training source changed')
                meta=read_json(inside(root,entry['metadata']));samples=read_selfplay_shard(inside(root,entry['path']))
                for game,game_rows in complete_games(samples,meta):
                    if is_holdout(int(game['family']),base.seed) and (game['family'] in groups[game['mode']] or len(groups[game['mode']])<6):
                        groups[game['mode']].add(game['family'])
                        for ply in sorted(set((0,len(game_rows)//2,len(game_rows)-1))):
                            rows.append(copy.deepcopy(game_rows[ply]));identities.append({'group':game['family'],'mode':game['mode'],'ply':ply,'source':entry['path']})
                    for row,played in zip(game_rows,game['trace']):
                        own=game['mode']=='teacher_vs_actor' and row.player==game['actor_side']
                        if not own:continue
                        counts['actor_owned_rows']+=1
                        hard=bool(np.count_nonzero(row.visits)==1 and row.visits.sum()==1)
                        counts['teacher_hard_rows']+=int(hard)
                        counts['teacher_target_differs_from_played']+=int(hard and int(row.legal_actions[np.argmax(row.visits)])!=played)
                        if row.ply>=24:
                            counts['late_actor_owned_rows']+=1
                            counts['late_teacher_target_differs_from_greedy_behavior']+=int(hard and int(row.legal_actions[np.argmax(row.visits)])!=played)
                del samples
            if not rows:raise ValueError('no saved training holdout')
            corpora[variant]={'rows':rows,'identities':identities,'sources':sources};labels[variant]=dict(counts)
        device=torch.device('cuda');torch.set_num_threads(4)
        models={'parent86':old['source_checkpoint'], 'control110':source_entry,
                'candidate110':dict(path=progress['variants']['candidate']['checkpoint'],sha256=progress['variants']['candidate']['sha256'],model_hash=progress['variants']['candidate']['model_hash'])}
        results={}
        for name,entry in models.items():
            check()
            if sha(inside(root,entry['path']))!=entry['sha256']:raise ValueError('model changed')
            saved=torch.load(inside(root,entry['path']),map_location='cpu',weights_only=False)
            model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(saved['model']);model.search_v3_leaf_value_scale=saved['leaf_value_scale']
            if model_hash(model)!=entry['model_hash']:raise ValueError('model tensor hash changed')
            results[name]={}
            for variant,corpus in corpora.items():
                check();predictions=[]
                with torch.inference_mode():
                    for start in range(0,len(corpus['rows']),32):
                        batch=search_distill_batch(corpus['rows'][start:start+32],device)
                        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):logits=model.terminal_wdl_logits(batch['observations'])
                        predictions.extend(torch.softmax(logits.float(),-1).cpu().numpy().tolist())
                prediction=np.asarray(predictions);truth=np.stack([r.wdl for r in corpus['rows']]);ce=-(truth*np.log(np.maximum(prediction,1e-9))).sum(-1)
                brier=((prediction-truth)**2).sum(-1);bias=(prediction[:,2]-prediction[:,0])-(truth[:,2]-truth[:,0]);grouped=defaultdict(list)
                for i,identity in enumerate(corpus['identities']):grouped[(identity['mode'],identity['group'])].append(i)
                def summary(indices):return {'wdl_ce':float(ce[indices].mean()),'brier':float(brier[indices].mean()),'value_bias':float(bias[indices].mean())}
                per_group=[dict(mode=mode,group=group,**summary(indices)) for (mode,group),indices in grouped.items()]
                results[name][variant]={'rows':len(truth),'groups':len(grouped),'group_mean':{k:float(np.mean([g[k] for g in per_group])) for k in ('wdl_ce','brier','value_bias')},
                    'by_mode':{mode:{k:float(np.mean([g[k] for g in per_group if g['mode']==mode])) for k in ('wdl_ce','brier','value_bias')} for mode in grouped_mode(per_group)}}
            del model,saved
        from root_explore_training_diagnostic import root_diagnostic
        diagnostic=root_diagnostic(root,old,progress,base,source_entry,check)
        check();atomic_json(output,{'at':utc_now(),'complete':True,'new_training_games':0,'effective_updates':0,'training_only':True,
            'previous_plan':{'path':str(Path(args.previous_plan)),'sha256':sha(inside(root,args.previous_plan))},'source_checkpoint':source_entry,'checkpoints':models,
            'corpora':{k:{'identities':v['identities'],'sources':v['sources']} for k,v in corpora.items()},'labels':labels,'value':results,'root_diagnostic':diagnostic,'native_build':native,
            'precision':'BF16 inference, FP32 probabilities; saved actual-terminal WDL','interpretation':'Descriptive grouped training holdout and real actor-owned label/behavior mismatch. Mismatch is expected in distillation and is not proof of an error or causal stall. Test one new exploration-target factor; no imagined successor truth, replay additions, adoption or strength claim.'})
    print(str(output.relative_to(root)))


def grouped_mode(rows):return sorted({r['mode'] for r in rows})


if __name__=='__main__':main()
