"""Finite saved-TRAIN-root benchmark: warm alternating reference/new searches."""
from pathlib import Path
from contextlib import ExitStack
import sys, json, hashlib, time
ROOT = Path(__file__).resolve().parents[1]
def main():
    import numpy as np
    import torch
    from open_v4_runtime import read_json, atomic_json, ProcessLock, utc_now
    master = read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json')
    assert master['paused_by_user']
    old = read_json(ROOT/master['optimization']['config'])
    sys.path.insert(0, str(ROOT/old['native_build']['directory']))
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_search_v3.root_explore_search import RootExploreSearch
    from junqi_search_v3.graph_inference import GraphNeuralPUCT, GraphRootExplore
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    from junqi_fixed_rl.layout import FixedLayout, FixedLayoutPairBackend
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out = ROOT/'reports/search_engineering_20261006'
    out.mkdir(exist_ok=True)
    progress = read_json(ROOT/old['directory']/'PROGRESS.json')
    item = progress['variants']['learner']
    assert item['counters']['iteration']==362 and sha(ROOT/item['checkpoint'])==item['sha256']
    meta_entry = progress['collections']['learner'][-1]
    meta = read_json(ROOT/meta_entry['metadata']); assert meta['layout_role']=='train'
    states=[]; roots=[]
    for index,g in enumerate(meta['games']):
        p=g['provenance']
        for ply in sorted(set((0,len(g['trace'])//2,len(g['trace'])-1))):
            red=FixedLayout(p['red_layout_id'],p['red_layout_id'],tuple(p['red_layout']),g['family'])
            black=FixedLayout(p['black_layout_id'],p['black_layout_id'],tuple(p['black_layout']),g['family'])
            s=FixedLayoutPairBackend(red,black).new_state(p['setup_seed'])
            for a in g['trace'][:ply]: assert a in s.legal_actions(); s.apply(int(a))
            assert not s.is_terminal()
            states.append(s); roots.append(dict(game=index,ply=ply,key=int(s.native_state.key())))
        if len(states)>=96: break
    protocol=dict(at=utc_now(),source=item,shard=meta_entry,roots=roots,training_games=0,updates=0,
        repeats=3,prewarm=True,alternating_order=True,budgets=[64,128,256],
        comparison='Exact physical shapes, BF16, original traversal and targets; no bucketing/cache/tree reuse.',
        source_sha=sha(__file__),optimized_sha=sha(ROOT/'python/junqi_search_v3/graph_inference.py'))
    atomic_json(out/'BENCHMARK_PROTOCOL.json',protocol)
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
        locks.enter_context(ProcessLock(ROOT/old['directory']/'supervision/TRAINING.lock'))
        torch.set_num_threads(2); device=torch.device('cuda')
        saved=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
        cfg=RootExploreConfig.from_dict(saved['config']) if hasattr(RootExploreConfig,'from_dict') else RootExploreConfig.load(ROOT/old['parent_config'])
        model=PolicyValueNet(cfg.model).to(device).eval(); model.load_state_dict(saved['model'])
        model.search_v3_leaf_value_scale=saved['leaf_value_scale']; assert model_hash(model)==item['model_hash']
        del saved
        rows=[]
        cases=[('evaluation',256,24),('evaluation',256,96),('training',64,24),('training',128,96),('training',64,256)]
        for mode,budget,count in cases:
            reference_cls, new_cls = (BatchedNeuralPUCT,GraphNeuralPUCT) if mode=='evaluation' else (RootExploreSearch,GraphRootExplore)
            kw=dict(device=device,inference_batch_size=256,cache_size=0,reuse_roots=False,leaves_per_tree=1,
                    root_tactical_guard=False,fixed_inference_batch_size=32 if mode=='evaluation' else 0)
            if mode=='training': kw.update(forced_coeff=2,target_pruning=False)
            searches=[reference_cls(model,cuda_graphs=mode=='training',**kw),new_cls(model,cuda_graphs=True,**kw)]
            batch=[states[i%len(states)] for i in range(count)]
            search_kw=dict(add_root_noise=mode=='training',noise_seeds=list(range(count)))
            reference,_=searches[0].search(batch,budget,**search_kw)
            candidate,_=searches[1].search(batch,budget,**search_kw)
            def equivalent(a,b):
                fields=['actions','visits','priors','q_values','policy']
                if mode=='training': fields+=['target_visits','pruned_visits','target_policy']
                return all(x.action==y.action and all(np.array_equal(getattr(x,k),getattr(y,k)) for k in fields)
                    and (mode!='training' or x.forced_decisions==y.forced_decisions) for x,y in zip(a,b))
            checks=[equivalent(reference,candidate)]; times=[[],[]]; reports=[]
            for repeat in range(3):
                for index in ([0,1] if repeat%2==0 else [1,0]):
                    torch.cuda.synchronize(); start=time.perf_counter()
                    result,report=searches[index].search(batch,budget,**search_kw)
                    torch.cuda.synchronize(); times[index].append(time.perf_counter()-start)
                    checks.append(equivalent(reference,result)); reports.append(dict(implementation=index,repeat=repeat,report=report))
            row=dict(mode=mode,budget=budget,roots=count,all_exact=all(checks),times=times,
                median_speedup=float(np.median(times[0])/np.median(times[1])),reports=reports,
                capture_seconds=searches[1].graph_capture_seconds,
                peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
            rows.append(row); atomic_json(out/'BENCHMARK_RESULTS.json',dict(protocol=protocol,cases=rows,all_exact=all(r['all_exact'] for r in rows)))
            print(json.dumps({k:v for k,v in row.items() if k!='reports'}),flush=True)
            assert row['all_exact'],'Reject output-changing optimization'
            del searches
        # Vary the exact batch size as finished games shrink the collector.
        kw=dict(device=device,inference_batch_size=256,cache_size=0,reuse_roots=False,
            leaves_per_tree=1,root_tactical_guard=False,forced_coeff=2,target_pruning=False)
        reference=RootExploreSearch(model,**kw);optimized=GraphRootExplore(model,**kw)
        sizes=[192,188,184,180,176,172,168,160,128,96,64,32,17,8,3,1]
        times=[0.,0.];checks=[]
        for size in sizes:
            batch=[states[i%len(states)] for i in range(size)]
            results=[]
            for i,search in enumerate((reference,optimized)):
                torch.cuda.synchronize();t=time.perf_counter()
                result,_=search.search(batch,64,add_root_noise=True,noise_seeds=list(range(size)))
                torch.cuda.synchronize();times[i]+=time.perf_counter()-t;results.append(result)
            checks.append(equivalent(*results))
        rows.append(dict(mode='training_changing_shapes_capture_included',budget=64,shapes=sizes,
            all_exact=all(checks),times=times,median_speedup=times[0]/times[1],
            capture_seconds=optimized.graph_capture_seconds,peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20))
        assert all(checks)
        # Captured graphs must be discarded after an in-place model revision.
        before=optimized._revision
        with torch.no_grad(): next(model.parameters()).add_(0.)
        optimized._refresh_revision()
        assert optimized._revision!=before and not optimized._graphs
        atomic_json(out/'BENCHMARK_RESULTS.json',dict(protocol=protocol,cases=rows,all_exact=all(r['all_exact'] for r in rows),revision_invalidates_graphs=True))
        print(json.dumps(rows[-1]),flush=True)
        assert model_hash(model)==item['model_hash'] and sha(ROOT/item['checkpoint'])==item['sha256']
        print('COMPLETE: weights unchanged; no new games or updates',flush=True)
if __name__=='__main__': main()
