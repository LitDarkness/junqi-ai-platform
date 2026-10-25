"""Capture-inclusive replay of actual collector grouping on saved TRAIN games."""
from pathlib import Path
from contextlib import ExitStack
import sys,hashlib,json,time
ROOT=Path(__file__).resolve().parents[1]
def main():
    import numpy as np,torch
    from open_v4_runtime import read_json,atomic_json,ProcessLock,utc_now
    master=read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json');assert master['paused_by_user']
    plan=read_json(ROOT/master['optimization']['config']);sys.path.insert(0,str(ROOT/plan['native_build']['directory']))
    from junqi_search_v3.root_explore_search import RootExploreSearch
    from junqi_search_v3.graph_inference import GraphRootExplore
    from junqi_fixed_rl.layout import FixedLayout,FixedLayoutPairBackend
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    from junqi_az.model import PolicyValueNet
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    progress=read_json(ROOT/plan['directory']/'PROGRESS.json');item=progress['variants']['learner'];shard=progress['collections']['learner'][-1]
    assert sha(ROOT/item['checkpoint'])==item['sha256'] and sha(ROOT/shard['path'])==shard['sha256'] and sha(ROOT/shard['metadata'])==shard['metadata_sha256']
    meta=read_json(ROOT/shard['metadata']);assert meta['layout_role']=='train'
    raw=np.load(ROOT/shard['path'],allow_pickle=False);hashes=raw['state_hash']
    cfg=RootExploreConfig.load(ROOT/plan['parent_config']);cohorts=[]
    for ply in (0,4,8,12,24,32,48,64,80):
        selected={64:[],128:[]};noise={64:[],128:[]};offset=0
        for g in meta['games']:
            length=len(g['trace'])
            if length<=ply:offset+=length;continue
            player=1 if ply%2==0 else -1
            if g['mode']=='selfplay' or (g['mode']=='teacher_vs_actor' and player==g['actor_side']):
                p=g['provenance'];red=FixedLayout(p['red_layout_id'],p['red_layout_id'],tuple(p['red_layout']),g['family']);black=FixedLayout(p['black_layout_id'],p['black_layout_id'],tuple(p['black_layout']),g['family'])
                s=FixedLayoutPairBackend(red,black).new_state(p['setup_seed'])
                for a in g['trace'][:ply]:assert a in s.legal_actions();s.apply(int(a))
                assert int(s.native_state.key())==int(hashes[offset+ply])
                budget=128 if np.any(s.repetition_contexts()>0) else 64
                selected[budget].append(s);noise[budget].append(cfg.seed^(meta['iteration']<<20)^g['index']^(ply*65537))
            offset+=length
        for budget in (64,128):
            if selected[budget]:cohorts.append((ply,budget,selected[budget],noise[budget]))
    out=ROOT/'reports/search_engineering_20261006'
    protocol=dict(at=utc_now(),source=item,shard=shard,cohorts=[dict(ply=p,budget=b,roots=len(s)) for p,b,s,n in cohorts],
        repetitions=3,capture_included=True,fresh_search_instances_per_pass=True,alternating_order=True,
        new_games=0,updates=0,interpretation='Saved TRAIN trajectory groups with current362 inference; no teacher advice, learning, or new complete games. Includes shape capture overhead.',source_sha=sha(__file__))
    atomic_json(out/'COLLECTOR_SHAPES_PROTOCOL.json',protocol)
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
        locks.enter_context(ProcessLock(ROOT/plan['directory']/'supervision/TRAINING.lock'))
        torch.set_num_threads(2);device=torch.device('cuda');saved=torch.load(ROOT/item['checkpoint'],map_location='cpu',weights_only=False)
        model=PolicyValueNet(cfg.model).to(device).eval();model.load_state_dict(saved['model']);model.search_v3_leaf_value_scale=saved['leaf_value_scale'];assert model_hash(model)==item['model_hash'];del saved
        kw=dict(device=device,inference_batch_size=256,cache_size=0,reuse_roots=False,leaves_per_tree=1,root_tactical_guard=False,forced_coeff=2,target_pruning=False)
        warm=RootExploreSearch(model,**kw);warm.search(cohorts[0][2],64);del warm
        times=[[],[]];reference=None;checks=[];capture=[]
        for repeat in range(3):
            for index in ([0,1] if repeat%2==0 else [1,0]):
                search=(RootExploreSearch if index==0 else GraphRootExplore)(model,**kw)
                results=[];torch.cuda.synchronize();start=time.perf_counter()
                for ply,budget,states,noise in cohorts:
                    rows,_=search.search(states,budget,add_root_noise=True,noise_seeds=noise)
                    results.extend(rows)
                torch.cuda.synchronize();times[index].append(time.perf_counter()-start)
                if reference is None:reference=results
                fields=('actions','visits','priors','q_values','policy','target_visits','pruned_visits','target_policy')
                checks.append(all(a.action==b.action and a.forced_decisions==b.forced_decisions and all(np.array_equal(getattr(a,k),getattr(b,k)) for k in fields) for a,b in zip(reference,results)))
                if index:capture.append(search.graph_capture_seconds)
                print(json.dumps(dict(repeat=repeat,implementation=index,seconds=times[index][-1],all_exact=checks[-1])),flush=True)
                del search
        result=dict(protocol=protocol,all_exact=all(checks),times=times,capture_seconds=capture,
            median_speedup=float(np.median(times[0])/np.median(times[1])),peak_gpu_mib=torch.cuda.max_memory_allocated()/2**20)
        assert result['all_exact'] and model_hash(model)==item['model_hash']
        atomic_json(out/'COLLECTOR_SHAPES_RESULTS.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
