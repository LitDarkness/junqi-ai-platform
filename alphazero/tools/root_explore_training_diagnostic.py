"""Small fixed saved-training-root probe; no imagined successor outcome labels."""
from collections import Counter
import time
from open_v4_runtime import read_json
from open_v4_queue_contract import sha,inside

def root_diagnostic(root,old,progress,base,source_entry,check):
    import numpy as np,torch
    from junqi_open_v4.training_replay import complete_games
    from junqi_open_v4.layouts import load_role,MultiLayoutBackend
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_search_v3.root_explore_search import RootExploreSearch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.learner import model_hash
    candidates={'early':[],'late':[]};tail=Counter()
    for entry in progress['collections']['control'][-3:]:
        meta=read_json(inside(root,entry['metadata']));rows=read_selfplay_shard(inside(root,entry['path']))
        for game,samples in complete_games(rows,meta):
            if game['mode']!='selfplay':continue
            for sample in samples:
                visits=np.asarray(sample.visits);tail['rows']+=1;tail['raw_visits']+=int(visits.sum())
                tail['singleton_mass_counts']+=int(visits[visits==1].sum());tail['two_or_less_counts']+=int(visits[visits<=2].sum())
            candidates['early'].append((entry,game,samples[min(6,len(samples)-1)]))
            if len(samples)>26:candidates['late'].append((entry,game,samples[max(24,len(samples)//2)]))
    rng=np.random.default_rng(base.seed^2026100502);chosen=[];seen=set()
    for phase in ('late','early'):
        items=sorted(candidates[phase],key=lambda x:(x[0]['stage'],x[1]['family'],x[1]['index']))
        count=0
        for index in rng.permutation(len(items)):
            e,g,s=items[index]
            if g['family'] in seen:continue
            seen.add(g['family']);chosen.append((phase,e,g,s));count+=1
            if count==12:break
        if count!=12:raise ValueError('24 distinct saved train-family roots required')
    backend=MultiLayoutBackend(load_role(inside(root,base.layout_directory),'train'),base.seed)
    states=[];identities=[];budgets=[]
    for phase,entry,game,sample in chosen:
        state=backend.new_state(game['provenance']['setup_seed'])
        for action in game['trace'][:sample.ply]:
            if action not in state.legal_actions():raise ValueError('saved root prefix illegal')
            state.apply(action)
        if int(state.native_state.key())!=sample.state_hash or not np.array_equal(state.legal_actions(),sample.legal_actions):
            raise ValueError('complete saved-root history mismatch')
        if not np.array_equal(state.feature_planes().astype(np.float16),sample.observation):raise ValueError('saved observation mismatch')
        states.append(state);budgets.append(128 if np.any(sample.repetition_contexts>0) else 64)
        identities.append({'phase':phase,'family':game['family'],'index':game['index'],'ply':sample.ply,
            'state_key':int(state.native_state.key()),'source':entry,'prefix':game['trace'][:sample.ply],
            'native_provenance':game['provenance'],'budget':budgets[-1]})
    saved=torch.load(inside(root,source_entry['path']),map_location='cpu',weights_only=False)
    model=PolicyValueNet(base.model).to('cuda').eval();model.load_state_dict(saved['model']);model.search_v3_leaf_value_scale=saved['leaf_value_scale']
    if model_hash(model)!=source_entry['model_hash']:raise ValueError('diagnostic weight changed')
    device=torch.device('cuda');engines={'standard':BatchedNeuralPUCT(model,device=device,fixed_inference_batch_size=32,cuda_graphs=False),
        'forced':RootExploreSearch(model,device=device,forced_coeff=2.,target_pruning=True,fixed_inference_batch_size=32,cuda_graphs=False)}
    outputs={k:[None]*24 for k in engines};metrics=[]
    for budget in (64,128):
        ix=[i for i,b in enumerate(budgets) if b==budget]
        if not ix:continue
        for name,engine in engines.items():
            check();results,cost=engine.search([states[i] for i in ix],budget,add_root_noise=True,
                noise_seeds=[base.seed^(900<<20)^identities[i]['index']^(identities[i]['ply']*65537) for i in ix])
            for i,result in zip(ix,results):outputs[name][i]=result
            metrics.append({'method':name,'roots':len(ix),'budget':budget,'simulations':len(ix)*budget,**cost})
    rows=[];changed=forced=0
    for i,(a,b) in enumerate(zip(outputs['standard'],outputs['forced'])):
        raw=b.visits;target=b.target_visits
        if raw.sum()!=budgets[i]-1 or np.any(target>raw) or not target.sum():raise ValueError('budget/target probe failed')
        winner=np.lexsort((b.actions,-b.q_values,-raw))[0]
        if target[winner]!=raw[winner]:raise ValueError('best target lost')
        targetp=target/target.sum();rawp=raw/raw.sum();delta=float(np.abs(targetp-rawp).sum())
        changed+=int(delta>0);forced+=b.forced_decisions
        rows.append({**identities[i],'actions':b.actions.tolist(),'raw_visits':raw.tolist(),'target_visits':target.tolist(),
            'priors':b.priors.tolist(),'q':b.q_values.tolist(),'target_type':'integer_core_no_singleton_cutoff',
            'forced_decisions':b.forced_decisions,'deleted_counts':int(raw.sum()-target.sum()),'target_l1':delta,
            'raw_entropy':b.root_entropy,'target_entropy':float(-(targetp*np.log(np.maximum(targetp,1e-12))).sum()),
            'standard_greedy':a.action,'forced_raw_greedy':b.action,'behavior_action_unchanged_by_target_export':True,
            'standard_vs_forced_action_diff':bool(a.action!=b.action)})
    if not changed or not forced:raise ValueError('intervention not active in saved-training probe')
    return {'checks_passed':True,'roots':24,'source_checkpoint':source_entry,'same_root_budget_per_method':True,
            'new_training_games':0,'effective_updates':0,'teacher_queries':0,'changed_target_roots':changed,
            'forced_decisions':forced,'metrics':metrics,'examples':rows,'saved_tail_counts':dict(tail),
            'interpretation':'Implementation signal only. Q is an estimate. No counterfactual WDL or external strength claim.'}
