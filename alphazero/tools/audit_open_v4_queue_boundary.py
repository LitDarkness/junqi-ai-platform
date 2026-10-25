"""Small saved-training-data audit, called only after the arena releases GPU locks."""
import copy
from collections import defaultdict


def run_audit(root,queue,check_stop):
    import numpy as np
    import torch
    from open_v4_runtime import read_json,atomic_json,utc_now
    from open_v4_queue_contract import sha,inside
    from run_open_v4_fresh_data import is_holdout
    from junqi_open_v4.training_replay import complete_games
    from junqi_search_v3.loop import read_selfplay_shard,search_distill_batch
    from junqi_open_v4.layouts import load_role,backend_for
    from junqi_open_v4.config import OpenConfig
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    from junqi_search_v3.search import BatchedNeuralPUCT
    completion=read_json(inside(root,queue['completion_plan']['path']))
    original=read_json(inside(root,completion['original_plan']['path']))
    progress=read_json(inside(root,original['directory'])/'PROGRESS.json')
    base=OpenConfig.load(inside(root,original['parent_config']))
    corpora={};root_records=[];root_groups=set();root_modes=defaultdict(int)
    for name,items in [('old38',original['initial_replay'])]+[(v+'62',progress['collections'][v][-3:]) for v in ('control','candidate')]:
        selected=[];identities=[];groups=defaultdict(set)
        for item in items:
            check_stop()
            if sha(inside(root,item['path']))!=item['sha256'] or sha(inside(root,item['metadata']))!=item['metadata_sha256']:
                raise ValueError('audit source replay changed')
            metadata=read_json(inside(root,item['metadata']));samples=read_selfplay_shard(inside(root,item['path']))
            for game,rows in complete_games(samples,metadata):
                holdout=is_holdout(int(game['family']),base.seed)
                if holdout and (game['family'] in groups[game['mode']] or len(groups[game['mode']])<6):
                    groups[game['mode']].add(game['family'])
                    for ply in sorted(set((0,len(rows)//2,len(rows)-1))):
                        row=copy.deepcopy(rows[ply]);selected.append(row)
                        identities.append({'group':game['family'],'mode':game['mode'],'index':game['index'],'ply':ply,'source':item['path'],'result_red':game['result_red']})
                if name!='candidate62' or holdout or game['family'] in root_groups or root_modes[game['mode']]>=12:continue
                neural=game['mode']=='selfplay' or game['mode']=='teacher_vs_actor'
                if not neural:continue
                row=next((r for r in rows if r.ply>=24 and (game['mode']=='selfplay' or r.player==game['actor_side'])),None)
                if row is not None and len(root_records)<24:
                    root_groups.add(game['family']);root_modes[game['mode']]+=1
                    root_records.append((copy.deepcopy(game),copy.deepcopy(row),metadata['iteration'],item))
            del samples
        if not selected:raise ValueError('empty held-out training audit')
        corpora[name]={'rows':selected,'identities':identities,'sources':items}
    device=torch.device('cuda');torch.set_num_threads(4)
    def load(source):
        check_stop()
        if sha(inside(root,source['path']))!=source['sha256']:raise ValueError('audit checkpoint changed')
        saved=torch.load(inside(root,source['path']),map_location='cpu',weights_only=False)
        model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(saved['model'])
        model.search_v3_leaf_value_scale=saved['leaf_value_scale']
        if model_hash(model)!=source['model_hash']:raise ValueError('audit model tensor mismatch')
        return model
    results={}
    for name in ('parent','control','candidate'):
        model=load(completion['participants'][name]);results[name]={}
        for label,corpus in corpora.items():
            check_stop();prob=[]
            with torch.inference_mode():
                for start in range(0,len(corpus['rows']),32):
                    rows=corpus['rows'][start:start+32];batch=search_distill_batch(rows,device)
                    with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                        logits=model.terminal_wdl_logits(batch['observations'])
                    prob.extend(torch.softmax(logits.float(),-1).cpu().numpy().tolist())
            prediction=np.array(prob);truth=np.stack([r.wdl for r in corpus['rows']]);ce=-(truth*np.log(np.maximum(prediction,1e-9))).sum(-1)
            brier=((prediction-truth)**2).sum(-1);bias=(prediction[:,2]-prediction[:,0])-(truth[:,2]-truth[:,0])
            grouped=defaultdict(list)
            for i,identity in enumerate(corpus['identities']):grouped[(identity['mode'],identity['group'])].append(i)
            def summary(indices):
                return {'wdl_ce':float(ce[indices].mean()),'brier':float(brier[indices].mean()),'value_bias':float(bias[indices].mean())}
            group_metrics=[{'mode':mode,'group':group,**summary(indices)} for (mode,group),indices in grouped.items()]
            results[name][label]={'rows':len(truth),'groups':len(grouped),'group_metrics':group_metrics,
                'group_mean':{field:float(np.mean([r[field] for r in group_metrics])) for field in ('wdl_ce','brier','value_bias')}}
        del model
    train={r['id']:r for r in load_role(inside(root,base.layout_directory),'train')};states=[];seeds=[];identities=[]
    for game,row,iteration,item in root_records:
        p=game['provenance'];red,black=train[p['red_layout_id']],train[p['black_layout_id']]
        state=backend_for(red,black,mirrored=list(red['setup'])!=p['red_layout']).new_state(p['setup_seed'])
        if list(state.native_state.board())!=p['initial_board']:raise ValueError('saved training start mismatch')
        for action in game['trace'][:row.ply]:state.apply(int(action))
        if int(state.native_state.key())!=row.state_hash or state.current_player()!=row.player or not np.array_equal(state.feature_planes().astype(np.float16),row.observation):
            raise ValueError('saved training history/observation mismatch')
        if not np.array_equal(state.legal_actions(),row.legal_actions) or not np.array_equal(state.repetition_contexts(),row.repetition_contexts):
            raise ValueError('saved training action/history context mismatch')
        states.append(state);seeds.append(base.seed^(iteration<<20)^game['index']^(row.ply*65537))
        identities.append({'group':game['family'],'index':game['index'],'ply':row.ply,'source':item,'state_key':row.state_hash,'noise_seed':seeds[-1]})
    if not states:raise ValueError('no saved late neural training roots')
    model=load(completion['participants']['candidate'])
    search=BatchedNeuralPUCT(model,device=device,cache_size=0,leaves_per_tree=1,reuse_roots=False,root_tactical_guard=False,fixed_inference_batch_size=32,cuda_graphs=False)
    check_stop();clean,clean_metrics=search.search(states,64,add_root_noise=False)
    check_stop();noisy,noisy_metrics=search.search(states,64,add_root_noise=True,noise_seeds=seeds)
    comparison=[]
    for identity,a,b in zip(identities,clean,noisy):
        def record(r):return {'action':r.action,'actions':r.actions.tolist(),'visits':r.visits.tolist(),'q':r.q_values.tolist(),'entropy':r.root_entropy}
        comparison.append(dict(identity,without_noise=record(a),with_noise=record(b),action_changed=a.action!=b.action))
    output={'at':utc_now(),'complete':True,'new_training_games':0,'effective_updates':0,'training_holdout_only':True,
        'queue_identity':{key:queue[key] for key in ('directory','source','completion_plan')},'checkpoints':completion['participants'],
        'heldout_corpora':{k:{'identities':v['identities'],'sources':v['sources']} for k,v in corpora.items()},'value':results,
        'root_diagnostic':{'roots':len(states),'total_simulations':128*len(states),'comparisons':comparison,
            'with_noise_metrics':noisy_metrics,'without_noise_metrics':clean_metrics,'action_changes':sum(r['action_changed'] for r in comparison)},
        'interpretation':'Old/new actual behavior-terminal WDL audits are descriptive, not minimax or causal Value truth. Equal64 noisy/clean saved training roots establish schedule intervention only; no counterfactual terminal labels, no replay additions, no promotion or claim of guaranteed noise benefit.'}
    atomic_json(inside(root,queue['directory'])/'BOUNDARY_AUDIT.json',output)
    return output


def main():
    import argparse
    from pathlib import Path
    from open_v4_runtime import read_json
    from open_v4_queue_contract import inside,sha,verify_boundary_audit
    parser=argparse.ArgumentParser();parser.add_argument('--queue',required=True)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    queue=read_json(inside(root,args.queue))
    if queue['schema']!='junqi-immediate-queue-plan-1':raise ValueError('diagnostic queue contract changed')
    def check_stop():
        if any(inside(root,p).exists() for p in queue['stop_paths']) or read_json(root/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json').get('paused_by_user',False):
            raise InterruptedError('STOP/pause requested during saved-training diagnostic')
    check_stop()
    for name,digest in queue['source'].items():
        if sha(inside(root,name))!=digest:raise ValueError('diagnostic source changed: '+name)
    if sha(inside(root,queue['completion_plan']['path']))!=queue['completion_plan']['sha256']:
        raise ValueError('diagnostic completion contract changed')
    run_audit(root,queue,check_stop);check_stop();verify_boundary_audit(root,queue)
    print('Saved-training diagnostic committed and verified;0 new games,0 updates.',flush=True)


if __name__=='__main__':main()
