from pathlib import Path
from contextlib import ExitStack
import sys,hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    import torch,numpy as np
    from open_v4_runtime import read_json,atomic_json,ProcessLock,utc_now
    master=read_json(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/PLAN.json');assert master['paused_by_user']
    plan=read_json(ROOT/master['optimization']['config']);sys.path.insert(0,str(ROOT/plan['native_build']['directory']))
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_search_v3.graph_inference import GraphNeuralPUCT
    from junqi_open_v4.layouts import backend_for
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.root_explore_config import RootExploreConfig
    from junqi_open_v4.learner import model_hash
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    cfg=RootExploreConfig.load(ROOT/plan['parent_config']);suite=read_json(ROOT/plan['suite']['path'])['states'][:24]
    states=[]
    for row in suite:
        s=backend_for(row['red'],row['black']).new_state(row['setup_seed'])
        for a in row['trace']:s.apply(int(a))
        assert not s.is_terminal() and int(s.native_state.key())==row['state_key'];states.append(s)
    participants=dict(plan['participants']);new=read_json(ROOT/plan['directory']/'PROGRESS.json')['variants']['learner']
    participants['latest362']=dict(path=new['checkpoint'],sha256=new['sha256'],model_hash=new['model_hash'])
    results={}
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(ROOT/'runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock'))
        torch.set_num_threads(2);device=torch.device('cuda')
        for name,e in participants.items():
            assert sha(ROOT/e['path'])==e['sha256']
            saved=torch.load(ROOT/e['path'],map_location='cpu',weights_only=False)
            model=PolicyValueNet(cfg.model).to(device).eval();model.load_state_dict(saved['model']);model.search_v3_leaf_value_scale=saved['leaf_value_scale']
            kw=dict(device=device,inference_batch_size=256,fixed_inference_batch_size=32,cache_size=0,reuse_roots=False,leaves_per_tree=1,root_tactical_guard=False)
            reference=BatchedNeuralPUCT(model,cuda_graphs=False,**kw);optimized=GraphNeuralPUCT(model,cuda_graphs=True,**kw)
            a,_=reference.search(states,256);b,_=optimized.search(states,256)
            ok=all(x.action==y.action and all(np.array_equal(getattr(x,k),getattr(y,k)) for k in ('actions','visits','priors','q_values','policy')) for x,y in zip(a,b))
            assert ok and model_hash(model)==e['model_hash'];results[name]=dict(all_exact=ok,checkpoint=e,roots=len(states),budget=256)
            print(name,ok,flush=True);del model,reference,optimized,saved
    atomic_json(ROOT/'reports/search_engineering_20261006/REFERENCE_VERIFICATION.json',dict(at=utc_now(),all_passed=True,results=results,new_games=0,updates=0))
if __name__=='__main__':main()
