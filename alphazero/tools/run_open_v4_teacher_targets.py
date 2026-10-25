"""Audit exact training traces and compare one selfplay-policy label factor.

Heavy imports stay inside main/functions so CPU teacher children avoid Torch.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
import shutil
import time
from contextlib import ExitStack
from pathlib import Path

from fork_open_v4_qualified import fork_payload
from run_open_v4_fresh_data import is_holdout
from run_open_v4_search_training import replay_window, verify_learning_state, route_signal


def history_safe(sample):
    return not sample.cycle_entry and sample.repetition_count <= 1 and not any(sample.repetition_contexts)


def hard_policy_target(sample, action):
    import numpy as np
    ids = np.flatnonzero(sample.legal_actions == action)
    if len(ids) != 1:
        raise ValueError("teacher action is not uniquely legal")
    old = sample.visits.copy()
    total = int(old.sum())
    if total <= 0:
        raise ValueError("empty source policy target")
    if sample.raw_visits is None:
        sample.raw_visits = old.copy()
    sample.visits = np.zeros(len(old), dtype=np.int32)
    sample.visits[int(ids[0])] = 1
    return float(old[int(ids[0])]) / total, bool(int(np.argmax(old)) == int(ids[0]))


def relabel_shard(samples, metadata, backend, teachers, check, notify):
    import numpy as np
    if metadata["layout_role"] != "train":
        raise ValueError("only training trajectories may be relabeled")
    offset, changed, skipped, mass, agreement = 0, 0, 0, 0., 0
    cache, pending, labels = {}, {}, []
    def apply(row, index, action, teacher):
        nonlocal changed, mass, agreement
        probability, same = hard_policy_target(row, action)
        changed += 1; mass += probability; agreement += int(same)
        labels.append({"row": index, "action": int(action), "teacher_id": teacher})
    def flush():
        if not pending:
            return
        check()
        keys = list(pending)
        actions = teachers.actions([pending[k][0] for k in keys], [k[-1] for k in keys])
        check()
        for key, action in zip(keys, actions):
            cache[key] = int(action)
            for row, index in pending[key][1]:
                apply(row, index, action, key[-1])
        pending.clear()
    for number, game in enumerate(metadata["games"], 1):
        check()
        state = backend.new_state(game["provenance"]["setup_seed"])
        if state.provenance() != game["provenance"]:
            raise ValueError("training layout reconstruction mismatch")
        rows = samples[offset:offset + len(game["trace"])]
        if len(rows) != len(game["trace"]):
            raise ValueError("missing complete trajectory samples")
        for ply, (row, action) in enumerate(zip(rows, game["trace"])):
            if (row.game_id, row.ply, row.player, row.state_hash) != (
                    game["family"], ply, state.current_player(), int(state.native_state.key())):
                raise ValueError("training sample/history state mismatch")
            if not np.array_equal(row.legal_actions, state.legal_actions()) or not np.array_equal(row.repetition_contexts, state.repetition_contexts()):
                raise ValueError("native legal/repetition target mismatch")
            if not np.array_equal(row.observation.astype(np.float16), state.feature_planes().astype(np.float16)):
                raise ValueError("native training observation mismatch")
            if game["mode"] == "selfplay":
                if history_safe(row):
                    key = (row.game_id, row.state_hash, row.player, row.ply, int(game["teacher_id"]))
                    if key in cache:
                        apply(row, offset + ply, cache[key], key[-1])
                    elif key in pending:
                        pending[key][1].append((row, offset + ply))
                    else:
                        pending[key] = [state.clone(), [(row, offset + ply)]]
                    if len(pending) >= 128:
                        flush()
                else:
                    skipped += 1
            if int(action) not in row.legal_actions:
                raise ValueError("illegal saved trajectory action")
            state.apply(int(action))
        if not state.is_terminal() or state.terminal_reason() != game["reason"] or state.terminal_value(1) != game["result_red"]:
            raise ValueError("reconstructed native terminal differs")
        for row in rows:
            if not np.array_equal(row.wdl, np.eye(3, dtype=np.float32)[int(state.terminal_value(row.player)) + 1]):
                raise ValueError("terminal WDL sign or class mismatch")
        offset += len(rows)
        if number % 64 == 0:
            flush(); notify(number, changed)
    flush()
    if offset != len(samples):
        raise ValueError("unassigned training rows")
    audit = {"games_reconstructed": len(metadata["games"]), "native_rows_verified": offset,
        "changed_selfplay_policy_rows": changed, "history_sensitive_selfplay_rows_preserved": skipped,
        "unique_teacher_queries": len(cache), "teacher_target_mass_in_original_mcts": mass / max(1, changed),
        "original_mcts_argmax_teacher_agreement": agreement / max(1, changed),
        "wdl_trajectory_observation_legality_history_unchanged": True,
        "source_teacher_policy_rows_unchanged": True, "teacher_labels": sorted(labels, key=lambda r:r["row"])}
    return audit


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
    from junqi_open_v4.pipeline import validation_metrics, promotion_decision
    from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.arena import run_arena, select_families

    parser=argparse.ArgumentParser();parser.add_argument("--experiment",required=True);parser.add_argument("--resume",action="store_true")
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    def inside(name):
        path=(root/name).resolve()
        if not path.is_relative_to(root) or path==root:raise ValueError("paths must stay inside alphazero")
        return path
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path=inside(args.experiment);plan=read_json(plan_path);base=OpenConfig.load(inside(plan["parent_config"]))
    if base.digest()!=plan["parent_config_hash"] or plan["stages"]!=6 or plan["evaluation_families"]!=64 or plan["evaluation_budget"]!=256:
        raise ValueError("registered complete-state/fixed-budget experiment required")
    directory=inside(plan["directory"]);stops=[inside(p) for p in plan["stop_paths"]]+[directory/"STOP"]
    if any(p.exists() for p in stops):raise RuntimeError("STOP requested; no automatic clearing")
    with ExitStack() as locks:
        for frozen in plan["frozen_runs"]:
            run=inside(frozen["directory"])
            for name in ("SUPERVISOR","TRAINING"):locks.enter_context(ProcessLock(run/f"supervision/{name}.lock"))
        if directory.exists()!=args.resume:raise ValueError("fresh directory or explicit reviewed --resume required")
        directory.mkdir(parents=True,exist_ok=True)
        status=dict(schema="junqi-teacher-target-experiment-1",state="checking",pid=os.getpid(),started_at=utc_now(),stages=6,updates=384,completed_updates=0)
        def update(**values):
            status.update(values,updated_at=utc_now());atomic_json(directory/"STATUS.json",status)
        def check():
            if any(p.exists() for p in stops):
                update(state="stopped_by_user");raise RuntimeError("STOP requested; complete checkpoints preserved")
            if shutil.disk_usage(directory).free<10*2**30:raise RuntimeError("minimum disk reserve reached")
        def verify_sources():
            for frozen in plan["frozen_runs"]:
                path=inside(frozen["directory"]);current=frozen["current"]
                if read_json(path/"CURRENT.json")!=current or sha(path/"CONTRACT.json")!=current["contract_hash"]:raise ValueError("frozen run changed")
                for role in ("learner","actor","champion"):
                    if sha(path/current[role]["path"])!=current[role]["sha256"]:raise ValueError("frozen role changed")
            for name,digest in {**plan["source"],**plan["evidence_files"]}.items():
                if sha(inside(name))!=digest:raise ValueError("source or qualification/negative evidence changed")
        update()
        try:
            check();verify_sources()
            layouts=inside(base.layout_directory)
            if sha(layouts/"manifest.json")!=plan["layout_manifest_hash"]:raise ValueError("layout boundary changed")
            source=inside(plan["source_checkpoint"]["path"])
            if sha(source)!=plan["source_checkpoint"]["sha256"]:raise ValueError("complete parent changed")
            payload=torch.load(source,map_location="cpu",weights_only=False)
            if payload["counters"]!=plan["source_counters"] or payload["config_hash"]!=base.digest():raise ValueError("complete parent counters/config differ")
            proof=plan|{"tool_sha256":sha(Path(__file__)),"plan_sha256":sha(plan_path)}
            configs={}
            for variant in ("control","candidate"):
                cfg=copy.deepcopy(base);cfg.run_directory=str((directory/variant).relative_to(root));cfg.initial_checkpoint=str(source.relative_to(root))
                cfg.simulations=cfg.repeated_simulations=256;cfg.validate();configs[variant]=cfg
            if args.resume:
                if read_json(directory/"CONTRACT.json")!=proof:raise ValueError("plan/tool changed; preserve prior experiment")
                progress=read_json(directory/"PROGRESS.json")
            else:
                atomic_json(directory/"CONTRACT.json",proof);progress={"labels":{},"variants":{}}
                for variant,cfg in configs.items():
                    migrated=fork_payload(payload,cfg,{"source":plan["source_checkpoint"],"variant":variant,"changed_factor":"noncycle selfplay policy target","optimizer_reset":False,"rng_reset":False})
                    checks=verify_learning_state(payload,migrated);path=directory/variant/"checkpoints/learner_stage_000000.pt"
                    atomic_checkpoint(path,migrated);atomic_json(directory/variant/"config.json",cfg.as_dict());atomic_json(directory/variant/"SEED_VERIFICATION.json",checks)
                    progress["variants"][variant]={"stage":0,"checkpoint":str(path.relative_to(directory)),"sha256":sha(path),"model_hash":migrated["model_hash"]}
                atomic_json(directory/"PROGRESS.json",progress)
            def commit():atomic_json(directory/"PROGRESS.json",progress)
            for item in progress["labels"].values():
                if sha(directory/item["path"])!=item["sha256"] or sha(directory/item["audit"])!=item["audit_sha256"]:raise ValueError("derived label shard changed")
            for item in progress["variants"].values():
                if sha(directory/item["checkpoint"])!=item["sha256"]:raise ValueError("committed learner changed")
            torch.set_num_threads(base.workers);device=torch.device(base.device)
            backend=MultiLayoutBackend(load_role(layouts,"train"),base.seed)
            def read(path):
                rows=read_selfplay_shard(path)
                for row in rows:row.observation=row.observation.astype(np.float16)
                return rows
            validation=[]
            for item in plan["validation_replay"]:
                if sha(inside(item["path"]))!=item["sha256"]:raise ValueError("fixed internal holdout source changed")
                validation.extend(s for s in read(inside(item["path"])) if is_holdout(s.game_id,base.seed))
            validation=validation[:4096]
            if not validation:raise ValueError("fixed internal holdout absent")
            with TeacherLeague(root,base.teacher_candidates,base.workers) as teachers:
                if teachers.provenance!=plan["teachers"]:raise ValueError("frozen teacher differs")
                for identifier,item in plan["replay_sources"].items():
                    if identifier in progress["labels"]:continue
                    check();path=inside(item["path"]);meta_path=inside(item["metadata"])
                    if sha(path)!=item["sha256"] or sha(meta_path)!=item["metadata_sha256"]:raise ValueError("source trajectory changed")
                    metadata=read_json(meta_path)
                    if metadata["layout_role"]!="train" or metadata["shard_sha256"]!=sha(path):raise ValueError("nontraining or unverified replay")
                    update(state="labeling",variant="candidate",shard=identifier,label_shards_done=len(progress["labels"]),label_shards_total=len(plan["replay_sources"]))
                    rows=read(path);started=time.perf_counter()
                    def notify(games,changed):print(f"[labels] shard={identifier} reconstructed_games={games}/256 changed_policy_rows={changed}",flush=True)
                    audit=relabel_shard(rows,metadata,backend,teachers,check,notify)
                    audit.update(source=item,teachers=teachers.provenance,elapsed_seconds=time.perf_counter()-started)
                    target=directory/f"teacher_replay/{identifier}.npz"
                    if target.exists():raise ValueError("uncommitted derived shard; inspect before recovery")
                    write_selfplay_shard(target,rows);audit_path=target.with_suffix(".json");atomic_json(audit_path,audit)
                    progress["labels"][identifier]={"path":str(target.relative_to(directory)),"sha256":sha(target),"audit":str(audit_path.relative_to(directory)),"audit_sha256":sha(audit_path)}
                    commit();del rows
                atomic_json(directory/"LABEL_AUDIT.json",{key:{k:v for k,v in read_json(directory/row["audit"]).items() if k not in ("teacher_labels","source","teachers")} for key,row in progress["labels"].items()})
                for stage in range(1,7):
                    for variant in ("control","candidate"):
                        old=progress["variants"][variant]
                        if old["stage"]>=stage:continue
                        check();cfg=configs[variant];learner=Learner(cfg,root);learner.resume(directory/old["checkpoint"])
                        groups=replay_window(stage,3,[24,25,26],45);training=[];replay=[]
                        for origin,number in groups:
                            identifier=f"{origin}_{number:06d}";source_item=plan["replay_sources"][identifier]
                            path=inside(source_item["path"]) if variant=="control" else directory/progress["labels"][identifier]["path"]
                            part=read(path);training.extend(s for s in part if not is_holdout(s.game_id,base.seed));replay.append({"path":str(path.relative_to(root)),"sha256":sha(path),"role":"train"})
                        if not training or len(training)>base.maximum_replay_samples:raise ValueError("fixed replay cap exceeded")
                        batch_digest=hashlib.sha256();original_batch=learner._batch
                        def tracked(pool):
                            selected=original_batch(pool)
                            for s in selected:batch_digest.update(str((s.game_id,s.state_hash,s.player,s.ply)).encode())
                            return selected
                        learner._batch=tracked
                        before=validation_metrics(learner.model,validation,device)
                        update(state="training",variant=variant,stage=stage,completed_updates=old["stage"]*64,training_samples=len(training))
                        started=time.perf_counter();logs=learner.train(training,updates=64);learner.counters["iteration"]=26+stage
                        after=validation_metrics(learner.model,validation,device);checkpoint=directory/variant/f"checkpoints/learner_stage_{stage:06d}.pt"
                        if checkpoint.exists():raise ValueError("uncommitted checkpoint; inspect before recovery")
                        atomic_checkpoint(checkpoint,learner.payload(role="learner",lineage={"experiment":str(plan_path),"source":plan["source_checkpoint"],"stage":stage,"replay":replay,"batch_digest":batch_digest.hexdigest(),"promotion":"none"}))
                        report={"stage":stage,"variant":variant,"training":logs,"training_samples":len(training),"counters":dict(learner.counters),"batch_digest":batch_digest.hexdigest(),"common_holdout_before":before,"common_holdout_after":after,"replay":replay,"training_seconds":time.perf_counter()-started,"checkpoint_sha256":sha(checkpoint),"peak_gpu_allocated_mib":torch.cuda.max_memory_allocated()/2**20}
                        report_path=directory/variant/f"reports/stage_{stage:06d}.json";atomic_json(report_path,report)
                        progress["variants"][variant]={"stage":stage,"checkpoint":str(checkpoint.relative_to(directory)),"sha256":sha(checkpoint),"model_hash":model_hash(learner.model),"report":str(report_path.relative_to(directory)),"report_sha256":sha(report_path),"effective_updates":learner.counters["effective_updates"],"batch_digest":batch_digest.hexdigest()}
                        commit();del learner,training,part;torch.cuda.empty_cache()
                    if progress["variants"]["control"]["batch_digest"]!=progress["variants"]["candidate"]["batch_digest"]:raise ValueError("paired training update batches diverged")
                check();models={};checkpoint_provenance={}
                for name,path in {"parent":source,**{v:directory/row["checkpoint"] for v,row in progress["variants"].items()}}.items():
                    saved=torch.load(path,map_location="cpu",weights_only=False);model=PolicyValueNet(base.model).to(device).eval();model.load_state_dict(saved["model"]);model.search_v3_leaf_value_scale=saved["leaf_value_scale"]
                    if model_hash(model)!=saved["model_hash"]:raise ValueError("arena tensor mismatch")
                    models[name]=model;checkpoint_provenance[name]={"path":str(path.relative_to(root)),"sha256":sha(path),"model_hash":saved["model_hash"]}
                models["repeat"]=models["parent"]
                suite_path=inside(plan["suite"]["path"])
                if sha(suite_path)!=plan["suite"]["sha256"]:raise ValueError("development suite changed")
                suite=select_families(read_json(suite_path)["states"],64,plan["evaluation_seed"])
                if len({r["family"] for r in suite})!=64:raise ValueError("incomplete development coverage")
                agents={name:StopAwareSearch(BatchedNeuralPUCT(model,device=device,inference_batch_size=base.inference_batch_size,cache_size=base.cache_size,leaves_per_tree=base.leaves_per_tree),check) for name,model in models.items()}
                matchups=[("candidate","parent"),("candidate","control")]+[(name,f"teacher{i}") for i in range(len(teachers.modules)) for name in models]
                update(state="arena",planned_games=2*len(suite)*len(matchups),evaluation_budget=256,evaluation_families=64,completed_updates=384)
                arena=run_arena(suite,matchups,agents,teachers,{name:256 for name in agents},seed=plan["evaluation_seed"])
                arena.update(checkpoints=checkpoint_provenance,teacher_provenance=teachers.provenance,purpose="selfplay policy-label factor; no promotion")
                atomic_json(directory/"arena.json",arena)
            check();verify_sources();gains={key:teacher_deltas(arena,before,after) for key,before,after in (("candidate_vs_parent","parent","candidate"),("candidate_vs_control","control","candidate"),("duplicate","parent","repeat"))}
            gate=promotion_decision(arena,base);selected=route_signal([gains["candidate_vs_parent"],gains["candidate_vs_control"]],gains["duplicate"],[arena["summaries"][i]["family_score"] for i in (0,1)],gate,64,.02)
            result={"schema":"junqi-teacher-target-result-1","experiment":plan,"summaries":arena["summaries"],"teacher_deltas":gains,"existing_gate":gate,"teacher_target_route_confirmed":selected,"progress":progress,"frozen_roles_unchanged":True,"matched_update_batches":True,"promotion":"none; review complete state and full primary gates","arena_seconds":arena["elapsed_seconds"]}
            atomic_json(directory/"RESULTS.json",result);update(state="review_due",review_required=True,teacher_target_route_confirmed=selected)
            print(json.dumps({"teacher_target_route_confirmed":selected,"existing_gate":gate}),flush=True)
        except BaseException as error:
            if status["state"]!="stopped_by_user":update(state="needs_attention",reason=repr(error),review_required=True)
            raise


if __name__=="__main__":main()
