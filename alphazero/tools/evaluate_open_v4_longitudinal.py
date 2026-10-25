"""Frozen historical-self gauntlet; no training, promotion, or auto-resume."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(name):
    path = (ROOT / name).resolve()
    if path == ROOT or not path.is_relative_to(ROOT):
        raise ValueError("path must stay inside alphazero")
    return path


class NeuralOnlyLeague:
    workers = 0


class FixedBare:
    """Use fixed32 physical inputs for bare too; never edit archived search."""
    def __init__(self, search, check):
        self.search_impl, self.check = search, check
        self.bare_calls = self.bare_physical_rows = self.bare_logical_rows = 0

    def __getattr__(self, name):
        return getattr(self.search_impl, name)

    def search(self, states, simulations, **kwargs):
        self.check()
        return self.search_impl.search(states, simulations, **kwargs)

    def bare(self, states):
        output = []
        for begin in range(0, len(states), 32):
            self.check()
            part = states[begin:begin+32]
            count = len(part)
            padded = part + [part[-1]] * (32-count)
            output.extend(self.search_impl.bare(padded)[:count])
            self.bare_calls += 1
            self.bare_physical_rows += 32
            self.bare_logical_rows += count
        return output


def archive_snapshot():
    paths = list((ROOT/"runs").rglob("STOP"))
    paths += list((ROOT/"runs").rglob("CURRENT.json"))
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}


def prepare():
    import torch
    from datetime import datetime, timezone
    from open_v4_runtime import read_json, atomic_json, utc_now
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.root_explore_config import RootExploreConfig as OpenConfig
    from junqi_open_v4.learner import model_hash

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    config_path = ROOT/f"config/experiments/open_v4_longitudinal_{stamp}.json"
    directory = f"runs/open_junqi_v4_longitudinal_{stamp}"
    base_path = "config/experiments/open_v4_root_explore_base_20261004_214715.json"
    base = OpenConfig.load(inside(base_path))
    paths = {
        "old150": "runs/open_junqi_search_v3_strength_continuation/checkpoints/loop_iteration_000150.pt",
        "champ26": "runs/open_junqi_v4_fresh_continuation/checkpoints/champion_000026.pt",
        "source38": "runs/open_junqi_v4_tactical_training_20261004_065605/control/checkpoints/learner_stage_000006.pt",
        "candidate62": "runs/open_junqi_v4_long_behavior_20261004_081501/candidate/checkpoints/learner_stage_000024.pt",
        "control86": "runs/open_junqi_v4_late_noise_factor_20261004_143905/control/checkpoints/learner_stage_000024.pt",
        "control110": "runs/open_junqi_v4_actor_policy_20261004_184157/control/checkpoints/learner_stage_000024.pt",
        "control134": "runs/open_junqi_v4_root_explore_20261004_214715/control/checkpoints/learner_stage_000024.pt",
        "candidate134": "runs/open_junqi_v4_root_explore_20261004_214715/candidate/checkpoints/learner_stage_000024.pt",
    }
    participants = {}
    torch.set_num_threads(4)
    for name, path in paths.items():
        saved = torch.load(inside(path), map_location="cpu", weights_only=False)
        model = PolicyValueNet(base.model).eval()
        model.load_state_dict(saved["model"])
        digest = model_hash(model)
        if saved.get("model_hash") and digest != saved["model_hash"]:
            raise ValueError("checkpoint tensor digest mismatch")
        if saved["leaf_value_scale"] != 0.9076346023345448:
            raise ValueError("historical Value scale mismatch")
        if any(not torch.isfinite(v).all() for v in saved["model"].values()):
            raise ValueError("nonfinite checkpoint")
        participants[name] = dict(path=path, sha256=sha(inside(path)), model_hash=digest,
            iteration=saved.get("counters", {}).get("iteration", saved.get("iteration")),
            namespace="v3" if name=="old150" else "v4", leaf_value_scale=saved["leaf_value_scale"],
            complete_state_fields=[k for k in ["model","optimizer","scheduler","rng","counters","config","leaf_value_scale"] if k in saved])
        del saved, model
    history = list(paths)[:6]
    matchups = [(current, old) for current in ["control134", "candidate134"] for old in history]
    matchups += [("candidate134", "control134"), ("repeat134", "control86")]
    participants["repeat134"] = participants["candidate134"].copy()
    suite_path = "runs/open_junqi_v4_fresh_continuation/development/suite.json"
    suite = read_json(inside(suite_path))
    if suite["contract"]["role"] != "dev":
        raise ValueError("dev corpus required; final never opened")
    if hashlib.sha256(json.dumps(suite["states"],sort_keys=True).encode()).hexdigest()!=suite["states_hash"]:
        raise ValueError("suite digest mismatch")
    plan = dict(schema="junqi-longitudinal-evaluation-1",registered_at=utc_now(),directory=directory,
        authorization="User requested historical-self head-to-head longitudinal assessment and judgment about further training. Evaluation only; training stays paused.",
        config=base_path,config_sha256=sha(inside(base_path)),
        participants=participants,history=history,matchups=matchups,budgets=[0,64],
        suite=dict(path=suite_path,sha256=sha(inside(suite_path)),states_hash=suite["states_hash"],families=64),
        evaluation_seed=2026100501,physical_batch=32,precision="BF16",cpuct=1.5,
        root_noise=False,guard=False,forced_exploration=False,cache=False,tree_reuse=False,
        planned_games=2*len(suite["states"])*len(matchups)*2,
        comparison="same development starts, both colors, family-clustered; bare and64 evaluated separately",
        primary_endpoints=[["candidate134","control86",64],["control134","control86",64]],
        primary_ci_level=.975,primary_bootstrap_resamples=20000,
        continuation_judgment="Research continuation signal only if a primary97.5%CI lower>50% and that branch's champ26 head-to-head95%CI lower>50%. Otherwise present inconclusive/regression. No production promotion, no automatic training startup. Search teacher and original promotion gates unchanged.",
        secondary_endpoints="Other historical matches, same-stage duel and bare; descriptive95% intervals, no independent multiplicity claim.",
        existing256=dict(path="runs/open_junqi_v4_root_explore_20261004_214715/arena/block_000024_screen.json",
                         sha256=sha(inside("runs/open_junqi_v4_root_explore_20261004_214715/arena/block_000024_screen.json")),families=16),
        stop_paths=[f"{directory}/STOP",f"{directory}/supervision/STOP"],
        historical_stop_preservation=archive_snapshot(),
        frozen_sources={p:sha(inside(p)) for p in ["build/JunqiAZ.cp310-win_amd64.pyd","python/junqi_open_v4/arena.py","python/junqi_open_v4/layouts.py","python/junqi_search_v3/search.py","python/junqi_az/model.py","tools/evaluate_open_v4_longitudinal.py"]})
    atomic_json(config_path,plan)
    print(json.dumps(dict(config=str(config_path.relative_to(ROOT)),directory=directory,
                          planned_games=plan["planned_games"],cpu_checkpoint_checks="all passed"),ensure_ascii=False))


def verify_native(arena, suite, check):
    from junqi_open_v4.layouts import backend_for
    counts = defaultdict(int)
    for index, game in enumerate(arena["games"]):
        if index%128==0: check()
        descriptor = suite[game["position"]]
        state = backend_for(descriptor["red"],descriptor["black"]).new_state(descriptor["setup_seed"])
        for action in descriptor["trace"]:
            state.apply(int(action))
        for action in game["trace"]:
            if action not in state.legal_actions(): raise ValueError("illegal saved move")
            state.apply(int(action));counts["native_plies"]+=1
        if not state.is_terminal() or state.terminal_reason()!=game["reason"]:
            raise ValueError("native terminal mismatch")
        if state.terminal_value(game["candidate_side"])!=game["value"] or game["score"]!=.5+.5*game["value"]:
            raise ValueError("native score mismatch")
        counts["games"]+=1
    return dict(counts)


def evaluate(plan_path):
    import torch
    import numpy as np
    from open_v4_runtime import read_json, atomic_json, utc_now, ProcessLock
    from junqi_open_v4.root_explore_config import RootExploreConfig as OpenConfig
    from junqi_open_v4.learner import model_hash
    from junqi_az.model import PolicyValueNet
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_open_v4.arena import run_arena
    plan=read_json(plan_path);directory=inside(plan["directory"])
    if directory.exists(): raise ValueError("preserve existing run; no implicit restart")
    with ExitStack() as locks:
        for name in ["NEXT_STAGE","SUPERVISOR","TRAINING"]:
            locks.enter_context(ProcessLock(ROOT/f"runs/open_junqi_v4_repair_pilot/supervision/{name}.lock"))
        if any(inside(p).exists() for p in plan["stop_paths"]):raise RuntimeError("new assessment STOP present")
        directory.mkdir(parents=True)
        locks.enter_context(ProcessLock(directory/"supervision/SUPERVISOR.lock"))
        locks.enter_context(ProcessLock(directory/"supervision/TRAINING.lock"))
        status=dict(schema=plan["schema"],state="checking",pid=os.getpid(),started_at=utc_now(),
                    new_training_games=0,effective_updates=0,new_evaluation_games=0,planned_games=plan["planned_games"],training_paused=True)
        def update(**kwargs):
            status.update(kwargs,updated_at=utc_now());atomic_json(directory/"STATUS.json",status)
        def check():
            if any(inside(p).exists() for p in plan["stop_paths"]):
                update(state="stopped_by_user");raise RuntimeError("assessment STOP requested")
        def verify():
            check()
            for path,digest in plan["historical_stop_preservation"].items():
                if not inside(path).exists() or sha(inside(path))!=digest:raise ValueError("historicalSTOP/role changed")
            for path,digest in plan["frozen_sources"].items():
                if sha(inside(path))!=digest:raise ValueError("frozen evaluation source changed")
            for name,item in plan["participants"].items():
                if sha(inside(item["path"]))!=item["sha256"]:raise ValueError("checkpoint changed")
            if sha(inside(plan["suite"]["path"]))!=plan["suite"]["sha256"]:raise ValueError("suite changed")
        update()
        try:
            verify()
            atomic_json(directory/"CONTRACT.json",plan|dict(plan_sha256=sha(plan_path),tool_sha256=sha(Path(__file__))))
            base=OpenConfig.load(inside(plan["config"]));device=torch.device(base.device)
            torch.set_num_threads(base.workers);torch.manual_seed(plan["evaluation_seed"])
            suite=read_json(inside(plan["suite"]["path"]))["states"]
            assert len({s["family"] for s in suite})==64
            models={}
            for name,item in plan["participants"].items():
                if name=="repeat134":models[name]=models["candidate134"];continue
                payload=torch.load(inside(item["path"]),map_location="cpu",weights_only=False)
                model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(payload["model"])
                model.search_v3_leaf_value_scale=payload["leaf_value_scale"]
                assert model_hash(model)==item["model_hash"]
                models[name]=model;del payload
            outputs={}
            for budget in plan["budgets"]:
                check();verify()
                agents={name:FixedBare(BatchedNeuralPUCT(model,device=device,cpuct=1.5,
                    inference_batch_size=256,fixed_inference_batch_size=32,cache_size=0,
                    reuse_roots=False,root_tactical_guard=False,leaves_per_tree=1,cuda_graphs=False),check)
                    for name,model in models.items()}
                update(state="arena",budget=budget)
                print(f"[longitudinal] budget={budget} starts={len(suite)} matchups={len(plan['matchups'])}",flush=True)
                t=time.perf_counter()
                with torch.inference_mode():
                    arena=run_arena(suite,[tuple(m) for m in plan["matchups"]],agents,NeuralOnlyLeague(),
                                    {name:budget for name in agents},seed=plan["evaluation_seed"])
                arena.update(checkpoints=plan["participants"],inference_contract=dict(physical_batch=32,precision="BF16",budget=budget,
                    root_noise=False,root_guard=False,forced=False,cache=False,tree_reuse=False),purpose="frozen longitudinal assessment; no promotion/no training")
                arena["bare_metrics"]={name:dict(calls=a.bare_calls,physical_rows=a.bare_physical_rows,logical_rows=a.bare_logical_rows) for name,a in agents.items()}
                update(state="verifying",budget=budget)
                native=verify_native(arena,suite,check)
                target=[g for g in arena["games"] if g["candidate"]=="candidate134" and g["opponent"]=="control86"]
                repeat=[g for g in arena["games"] if g["candidate"]=="repeat134" and g["opponent"]=="control86"]
                keyed=lambda rows:{(g["position"],g["candidate_side"]):(g["trace"],g["value"],g["reason"]) for g in rows}
                if keyed(target)!=keyed(repeat):raise ValueError("fixed32 duplicate trajectories differ")
                native["duplicate_exact_games"]=len(repeat)
                path=directory/f"arena_{budget}.json";atomic_json(path,arena)
                atomic_json(directory/f"verification_{budget}.json",native)
                outputs[budget]=dict(path=str(path.relative_to(ROOT)),sha256=sha(path),native=native,summaries=arena["summaries"],
                    arena_seconds=arena["elapsed_seconds"],total_seconds=time.perf_counter()-t)
                verify()
                for name,model in models.items():assert model_hash(model)==plan["participants"][name]["model_hash"]
                update(state="committed",budget=budget,new_evaluation_games=sum(x["native"]["games"] for x in outputs.values()))
                print(f"[longitudinal] budget={budget} verified_games={native['games']} seconds={outputs[budget]['total_seconds']:.1f}",flush=True)
                del agents,arena;torch.cuda.empty_cache()
            primary=[]
            arena=read_json(inside(outputs[64]["path"]))
            for index,(candidate,opponent,budget) in enumerate(plan["primary_endpoints"]):
                groups=defaultdict(list)
                for g in arena["games"]:
                    if (g["candidate"],g["opponent"])==(candidate,opponent):groups[g["family"]].append(g["score"])
                values=np.array([np.mean(groups[f]) for f in sorted(groups)])
                rng=np.random.default_rng(plan["evaluation_seed"]+1000+index)
                boot=np.empty(20000)
                for i in range(0,len(boot),256):
                    n=min(256,len(boot)-i);boot[i:i+n]=values[rng.integers(0,len(values),(n,len(values)))].mean(axis=1)
                interval=list(map(float,np.quantile(boot,[.0125,.9875])))
                champ=next(r for r in outputs[64]["summaries"] if r["candidate"]==candidate and r["opponent"]=="champ26")
                primary.append(dict(candidate=candidate,opponent=opponent,budget=64,mean=float(values.mean()),ci975=interval,
                    families=len(values),research_continuation_signal=interval[0]>.5 and champ["family_score"]["ci95"][0]>.5))
            result=dict(schema="junqi-longitudinal-result-1",outputs=outputs,primary=primary,
                new_evaluation_games=status["new_evaluation_games"],new_training_games=0,effective_updates=0,
                training_still_paused=True,old_roles_checkpoints_and_STOP_unchanged=True,promotion="none",
                weights_unchanged=True,peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            atomic_json(directory/"RESULTS.json",result)
            update(state="review_due",primary=primary,completed_at=utc_now())
            print(json.dumps(primary,ensure_ascii=False),flush=True)
        except BaseException as error:
            if status["state"]!="stopped_by_user":update(state="needs_attention",error=repr(error))
            raise


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--prepare",action="store_true");parser.add_argument("--plan")
    args=parser.parse_args()
    if args.prepare:prepare()
    elif args.plan:evaluate(inside(args.plan))
    else:parser.error("--prepare or --plan required")
