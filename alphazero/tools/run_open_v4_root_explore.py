"""Isolated paired fresh-data root exploration and Policy-target separation experiment.

Heavy imports stay inside main so spawned CPU teachers do not initialize Torch.
No old STOP is removed. Resume requires an explicit reviewed invocation.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import os
from contextlib import ExitStack
from collections import Counter
from pathlib import Path


def screening_signal(gains, duels):
    return (len(gains) == 4 and all(len(rows) == 2 and all(r["mean"] > 0 for r in rows) for rows in gains)
            and len(duels) == 4 and all(row["mean"] > .5 for row in duels))


def confirmed_route(gains, duplicate, duels, gate, families, tolerance):
    return bool(families >= 64 and len(gains) == 4 and all(
        len(rows) == 2 and all(r['families'] >= families and r['ci95'][0] > 0 for r in rows) for rows in gains)
        and len(duels) == 4 and all(r['families'] >= families and r['ci95'][0] > .5 for r in duels)
        and len(duplicate) == 2 and all(r['families'] >= families and abs(r['mean']) <= tolerance for r in duplicate)
        and gate['champion_accepted'])


def ordered_participants(participants):
    # promotion_decision consumes parent-to-candidate paired deltas. The arena
    # emits pairs in matchup order, so the baseline must precede the candidate.
    order = ("champion", "parent", "guard86", "control", "candidate", "repeat")
    if set(participants) != set(order):
        raise ValueError("champion/parent/guard86/control/candidate/repeat participants required")
    return {name: participants[name] for name in order}


def severe_regression(arena):
    """Predeclared safety stop; uncertainty alone is not early failure.

    Both correlated teachers must have upper CI below minus10pp vs source,
    and the candidate-source direct game's upper CI must be below40%.
    No independence is assumed and no promotion threshold is changed.
    """
    gains = arena['review']['gains']['parent']
    duel = arena['summaries'][1]['family_score']
    return len(gains)==2 and all(r['ci95'][1]<-.10 for r in gains) and duel['ci95'][1]<.40


def main():
    from open_v4_runtime import atomic_json, read_json, utc_now, ProcessLock, supervision_status
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def inside(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root: raise ValueError("path outside alphazero")
        return path
    plan_path = inside(args.plan); plan = read_json(plan_path)
    directory = inside(plan["directory"])
    if args.status:
        snapshot = supervision_status(directory)
        snapshot["stop_requested"] = any(inside(p).exists() for p in plan["stop_paths"])
        print(json.dumps(snapshot, ensure_ascii=False)); return
    import sys
    sys.path.insert(0,str(inside(plan["native_build"]["directory"])))
    import numpy as np
    import torch
    from fork_open_v4_qualified import fork_payload
    from run_open_v4_fresh_data import is_holdout
    from run_open_v4_search_training import verify_learning_state, paired_schedule
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.root_explore_config import RootExploreConfig as OpenConfig
    from junqi_open_v4.learner import atomic_checkpoint, model_hash
    from junqi_open_v4.root_explore_games import collect_training_games
    from junqi_open_v4.training_replay import replay_pools, make_self_training_learner, complete_games, trajectory_digest
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_open_v4.pipeline import promotion_decision, validation_metrics
    from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if plan["schema"] != "junqi-root-explore-target-plan-1": raise ValueError("plan contract mismatch")
    if plan['maximum_stages']!=24 or plan['stages_per_block']!=12 or plan['evaluation_inference_batch_size']!=32:
        raise ValueError('registered horizon/evaluation shape cannot change')
    from open_v4_root_explore_contract import check_factor_parameters
    if plan.get('route')!='root_exploration_target_factor' or plan['training_variants']!=['control','candidate']:
        raise ValueError('registered root-target branches required')
    check_factor_parameters(plan)
    if (plan['games_per_stage'],plan['updates_per_stage'],plan['replay_stages'],plan['evaluation_budget'],plan['screen_families'],plan['full_families'])!=(256,64,3,256,16,64):
        raise ValueError('registered data/update/evaluation budget cannot change')
    base = OpenConfig.load(inside(plan["parent_config"]))
    stops = [inside(p) for p in plan["stop_paths"]]
    def verify():
        for name, digest in plan["source"].items():
            if sha(inside(name)) != digest: raise ValueError(f"executed source changed: {name}")
        if sha(inside(plan["parent_config"])) != plan["parent_config_sha256"]: raise ValueError("base config changed")
        if sha(inside(base.layout_directory) / "manifest.json") != plan["layout_manifest_hash"]: raise ValueError("layouts changed")
        if sha(inside(plan["suite"]["path"])) != plan["suite"]["sha256"]: raise ValueError("dev suite changed")
        for marker in plan["historical_stops"]:
            if sha(inside(marker["path"])) != marker["sha256"]: raise ValueError("historical STOP changed")
        for frozen in plan["frozen_runs"]:
            if sha(inside(frozen["directory"]) / "CURRENT.json") != frozen["current_sha256"]: raise ValueError("archived roles changed")
        for item in plan["participants"].values():
            if sha(inside(item["path"])) != item["sha256"]: raise ValueError("frozen complete checkpoint changed")
    if any(p.exists() for p in stops): raise RuntimeError("STOP requested; never clear automatically")
    with ExitStack() as locks:
        locks.enter_context(ProcessLock(root/"runs/open_junqi_v4_repair_pilot/supervision/NEXT_STAGE.lock"))
        for frozen in plan["frozen_runs"]:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(inside(frozen["directory"]) / f"supervision/{name}.lock"))
        verify()
        from open_v4_root_explore_contract import verify_root_explore_prerequisite
        verify_root_explore_prerequisite(root,plan)
        source = torch.load(inside(plan["source_checkpoint"]["path"]), map_location="cpu", weights_only=False)
        for field in ("model", "optimizer", "scheduler", "counters", "rng", "config", "leaf_value_scale"):
            if field not in source: raise ValueError("incomplete learning state")
        if source["counters"] != plan["source_counters"] or source["config"]["model"] != base.as_dict()["model"]:
            raise ValueError("experimental source counters or architecture changed")
        configs = {}
        for variant in plan["training_variants"]:
            cfg = copy.deepcopy(base); params = plan[variant]
            cfg.run_directory = str((directory / variant).relative_to(root))
            cfg.initial_checkpoint = plan["source_checkpoint"]["path"]
            for key in ("selfplay_fraction", "teacher_selfplay_fraction", "actor_policy_target", "root_tactical_guard", "training_greedy_after_ply", "root_noise_until_ply", "forced_root_coeff", "exploration_target_pruning"):
                setattr(cfg, key, params[key])
            cfg.games_per_iteration = plan["games_per_stage"]; cfg.updates_per_iteration = plan["updates_per_stage"]
            cfg.validate(); configs[variant] = cfg
            changed = {k for k, v in cfg.as_dict().items() if v != base.as_dict()[k]}
            if changed - {"run_directory", "initial_checkpoint", "exploration_target_pruning"}:
                raise ValueError("single-factor experiment changed additional numerical settings")
            if params["deduplicate"] or params["neural_policy_only"]:
                raise ValueError("registered replay qualification cannot change")
            expected = 24
            if cfg.root_tactical_guard or cfg.training_greedy_after_ply!=expected or cfg.simulations != 64 or cfg.repeated_simulations != 128:
                raise ValueError("registered behavior/budgets changed")
            target = "teacher"
            if cfg.root_noise_until_ply is not None or cfg.actor_policy_target!=target or cfg.exploration_target_pruning!=(variant=="candidate"):
                raise ValueError("registered behavior/budgets changed")
            migrated = fork_payload(source, cfg, {"plan": str(plan_path.relative_to(root)), "variant": variant,
                                                "source": plan["source_checkpoint"], "optimizer_reset": False})
            verify_learning_state(source, migrated)
        if args.check_only:
            print(json.dumps({"valid": True, "full_seed_state_preserved": True,
                              "historical_stops_preserved": plan["historical_stop_markers"]})); return
        if directory.exists() != args.resume: raise ValueError("fresh branch or explicit --resume required")
        directory.mkdir(parents=True, exist_ok=True)
        for name in ("SUPERVISOR", "TRAINING"):
            locks.enter_context(ProcessLock(directory / f"supervision/{name}.lock"))
        status = dict(schema=plan["schema"], state="checking", pid=os.getpid(), started_at=utc_now(),
                      plan=str(plan_path.relative_to(root)), new_training_games=0, completed_updates=0,phase="online_training",
                      source_iteration=source["counters"]["iteration"],route=plan["route"])
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "supervision/STATUS.json", status)
            atomic_json(directory / "STATUS.json", status)
        def check():
            if any(p.exists() for p in stops) or read_json(root/"runs/open_junqi_v4_repair_pilot/supervision/PLAN.json").get("paused_by_user",False):
                update(state="stopped_by_user", review_required=False)
                raise RuntimeError("STOP requested; complete checkpoints preserved")
        proof = {"plan_sha256": sha(plan_path), "source": plan["source"]}
        if args.resume:
            if read_json(directory / "CONTRACT.json") != proof: raise ValueError("resume contract changed")
            progress = read_json(directory / "PROGRESS.json")
        else:
            atomic_json(directory / "CONTRACT.json", proof)
            progress = {"variants": {}, "collections": {"control": [], "candidate": []}, "evaluations": {},
                        "source": plan["source_checkpoint"], "champion": plan["source_checkpoint"]}
            for variant, cfg in configs.items():
                migrated = fork_payload(source, cfg, {"plan": str(plan_path.relative_to(root)),
                    "variant": variant, "source": plan["source_checkpoint"], "optimizer_reset": False})
                path = directory / variant / "checkpoints/learner_stage_000000.pt"
                atomic_checkpoint(path, migrated); atomic_json(path.parents[1] / "config.json", cfg.as_dict())
                atomic_json(directory / "verification" / f"{variant}_seed.json", {"source": plan["source_checkpoint"], "checks": verify_learning_state(source, torch.load(path, map_location="cpu", weights_only=False)), "intentional_changes": ["config", "role", "lineage"]})
                progress["variants"][variant] = {"stage": 0, "checkpoint": str(path.relative_to(root)),
                    "sha256": sha(path), "model_hash": migrated["model_hash"], "counters": migrated["counters"]}
            seed_champion = directory / "champions/champion_seed.pt"
            production = torch.load(inside(plan["participants"]["champion"]["path"]), map_location="cpu", weights_only=False)
            atomic_checkpoint(seed_champion, production)
            progress["champion"] = {"path": str(seed_champion.relative_to(root)),
                "sha256": sha(seed_champion), "model_hash": production["model_hash"], "iteration": production["counters"]["iteration"]}
            atomic_json(directory / "PROGRESS.json", progress)
        def commit(): atomic_json(directory / "PROGRESS.json", progress)
        def publish_current():
            candidate = progress["variants"]["candidate"]
            learner_record = dict(candidate, path=str(inside(candidate["checkpoint"]).relative_to(directory)),
                                  iteration=candidate["counters"]["iteration"])
            champion = progress["champion"]
            champion_record = dict(champion, path=str(inside(champion["path"]).relative_to(directory)))
            atomic_json(directory / "CURRENT.json", {"iteration": candidate["counters"]["iteration"],
                "learner": learner_record, "champion": champion_record, "actor": champion_record,
                "experimental_collectors": progress["variants"]})
        publish_current()
        device = torch.device(base.device); torch.set_num_threads(base.workers)
        def search(model):
            return StopAwareSearch(BatchedNeuralPUCT(model, device=device, inference_batch_size=base.inference_batch_size,
                cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree,
                fixed_inference_batch_size=plan['evaluation_inference_batch_size'],cuda_graphs=False), check)
        def load_model(item):
            if sha(inside(item["path"])) != item["sha256"]: raise ValueError("arena weights changed")
            saved = torch.load(inside(item["path"]), map_location="cpu", weights_only=False)
            model = PolicyValueNet(base.model).to(device).eval(); model.load_state_dict(saved["model"])
            model.search_v3_leaf_value_scale = saved["leaf_value_scale"]
            if model_hash(model) != item["model_hash"]: raise ValueError("arena tensor digest changed")
            return model
        suite = read_json(inside(plan["suite"]["path"]))["states"]
        def evaluate(name, participants, families, teachers):
            check(); verify()
            participants = ordered_participants(participants)
            if name in progress["evaluations"]:
                record = progress["evaluations"][name]
                if sha(inside(record["path"])) != record["sha256"]: raise ValueError("committed arena changed")
                return read_json(inside(record["path"]))
            selected = select_families(suite, families, plan["evaluation_seed"])
            models = {key: load_model(item) for key, item in participants.items()}
            agents = {key: search(model) for key, model in models.items()}
            matchups = [("candidate", "champion"), ("candidate", "parent"), ("candidate", "control"), ("candidate", "guard86")] + [
                (key, f"teacher{i}") for i in range(len(teachers.modules)) for key in models]
            update(state="arena",phase="frozen_development_evaluation",arena=name, planned_games=2 * len(selected) * len(matchups), evaluation_families=families)
            arena = run_arena(selected, matchups, agents, teachers, {key: plan["evaluation_budget"] for key in agents},
                              seed=plan["evaluation_seed"])
            arena["checkpoints"] = participants
            arena['inference_contract']={'physical_batch':32,'precision':'BF16','padding':'repeat final input, discard padded outputs',
                                         'root_noise':False,'root_guard':False,'budget':256,
                                         'historical_dynamic_batch_scores_not_same_compute_context':True}
            gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
                ("champion", "champion", "candidate"), ("parent", "parent", "candidate"), ("control", "control", "candidate"), ("guard86", "guard86", "candidate"), ("duplicate", "parent", "repeat"))}
            gate = promotion_decision(arena, base)
            duels = [row["family_score"] for row in arena["summaries"][:4]]
            arena["review"] = {"gains": gains, "gate": gate,
                "screen_promising": screening_signal([gains["champion"], gains["parent"], gains["control"], gains["guard86"]], duels),
                "route_confirmed": confirmed_route([gains["champion"], gains["parent"], gains["control"], gains["guard86"]], gains["duplicate"], duels, gate, families, .02),
                "screen_only": families < 64}
            path = directory / "arena" / f"{name}.json"; atomic_json(path, arena)
            progress["evaluations"][name] = {"path": str(path.relative_to(root)), "sha256": sha(path), "review": arena["review"]}
            commit(); return arena
        def read_shard(item):
            path = inside(item["path"]); meta_path = inside(item["metadata"])
            if sha(path) != item["sha256"] or sha(meta_path) != item["metadata_sha256"]: raise ValueError("replay changed")
            meta = read_json(meta_path)
            if meta["layout_role"] != "train" or meta["shard_sha256"] != item["sha256"]: raise ValueError("nontraining replay")
            rows = read_selfplay_shard(path)
            for row in rows: row.observation = row.observation.astype(np.float16)
            origin = 'initial' if any(item['path'] == old['path'] for old in plan['initial_replay']) else 'fresh'
            for row in rows:
                row.training_replay_origin = origin
                row.training_target_type = meta.get('selfplay_target_contract','legacy_raw')
            return rows, meta
        update()
        try:
            train_rows = load_role(inside(base.layout_directory), "train")
            backend = MultiLayoutBackend(train_rows, base.seed)
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != plan["teachers"]: raise ValueError("teacher identity changed")
                original_submit = teachers.submit_actions
                def checked_submit(states, ids): check(); return original_submit(states, ids)
                teachers.submit_actions = checked_submit
                validation = []
                for item in plan["initial_replay"]:
                    samples, _ = read_shard(item)
                    validation.extend(s for s in samples if is_holdout(s.game_id, base.seed))
                validation = validation[:4096]
                for stage in range(1, plan["maximum_stages"] + 1):
                    for variant in plan["training_variants"]:
                        old = progress["variants"][variant]
                        if old["stage"] >= stage: continue
                        check(); verify(); cfg = configs[variant]; params = plan[variant]
                        if sha(inside(old["checkpoint"])) != old["sha256"]: raise ValueError("committed learner changed")
                        learner = make_self_training_learner(cfg, root); learner.resume(inside(old["checkpoint"]))
                        collection = next((r for r in progress["collections"][variant] if r["stage"] == stage), None)
                        if collection is None:
                            update(state="collecting",phase="online_training", variant=variant, stage=stage, collector_model_hash=old["model_hash"])
                            learner.model.eval()
                            samples, meta = collect_training_games(cfg, learner.model, device, backend, teachers,
                                plan["collection_start"] + stage - 1, check_stop=check)
                            from open_v4_root_explore_contract import verify_collection
                            collection_proof=verify_collection(samples,meta,cfg)
                            meta.update(root_explore_verification=collection_proof)
                            meta.update(actor_model_hash=old["model_hash"], actor_checkpoint=old,
                                        policy_eligibility="neural_search_only" if params["neural_policy_only"] else "all")
                            path = directory / variant / "replay" / f"collection_{stage:06d}.npz"
                            write_selfplay_shard(path, samples); meta["shard_sha256"] = sha(path)
                            atomic_json(path.with_suffix(".json"), meta)
                            collection = {"stage": stage, "path": str(path.relative_to(root)), "sha256": sha(path),
                                "metadata": str(path.with_suffix(".json").relative_to(root)), "metadata_sha256": sha(path.with_suffix(".json"))}
                            progress["collections"][variant].append(collection); commit(); del samples
                            update(new_training_games=sum(len(v) for v in progress["collections"].values()) * plan["games_per_stage"])
                        meta = read_json(inside(collection["metadata"]))
                        if variant == "candidate" and "control" in plan["training_variants"]:
                            paired = next(r for r in progress["collections"]["control"] if r["stage"] == stage)
                            if paired_schedule(meta) != paired_schedule(read_json(inside(paired["metadata"]))):
                                raise ValueError("paired initial layouts, modes or colors changed")
                        if meta["actor_model_hash"] != old["model_hash"]:
                            raise ValueError("collector did not use preceding complete learner")
                        records = (plan["initial_replay"] + progress["collections"][variant])[-plan["replay_stages"]:]
                        shards = [read_shard(item) for item in reversed(records)]
                        values, policy, audit = replay_pools(shards, deduplicate=params["deduplicate"],
                            neural_policy_only=params["neural_policy_only"], holdout=lambda family: is_holdout(family, base.seed))
                        digests=[]; by_mode={}; by_side={}
                        for samples,metadata in shards:
                            for game,rows in complete_games(samples,metadata):
                                if is_holdout(int(game['family']),base.seed): continue
                                digests.append(trajectory_digest(game,rows,metadata['policy_target_contract']))
                                by_mode.setdefault(game['mode'],[]).append(game['result_red']*game['actor_side'])
                                by_side.setdefault(str(game['actor_side']),[]).append(game['result_red']*game['actor_side'])
                        audit['exact_unique_complete_trajectories']=len(set(digests))
                        audit['exact_repeated_complete_trajectories']=len(digests)-len(set(digests))
                        audit['actor_pov_score_by_mode']={k:float(np.mean(v))*.5+.5 for k,v in by_mode.items()}
                        audit['actor_pov_score_by_side']={k:float(np.mean(v))*.5+.5 for k,v in by_side.items()}
                        update(state="training",phase="online_training", variant=variant, stage=stage, replay_audit=audit)
                        original_step = learner.step
                        def checked_step(*a, **kw): check(); return original_step(*a, **kw)
                        learner.step = checked_step
                        learner.policy_replay_origin_counts = Counter()
                        learner.policy_target_type_counts = Counter()
                        original_batch = learner._batch
                        def tracked_batch(samples):
                            phase = 'value' if learner.counters['attempted_updates'] % 3 == 2 else 'policy'
                            selected = original_batch(samples)
                            if phase == 'policy':
                                learner.policy_replay_origin_counts.update(
                                    row.training_replay_origin + ':' + row.training_policy_source for row in selected)
                                learner.policy_target_type_counts.update(
                                    row.training_target_type if row.training_policy_source=='search' else 'teacher_hard' for row in selected)
                            return selected
                        learner._batch = tracked_batch
                        updates = learner.train_pools(values, policy, updates=plan["updates_per_stage"])
                        learner.counters["iteration"] += 1
                        checkpoint = directory / variant / "checkpoints" / f"learner_stage_{stage:06d}.pt"
                        payload = learner.payload(role="learner", lineage={"plan": str(plan_path.relative_to(root)),
                            "previous": old, "replay": records, "data_package": params})
                        atomic_checkpoint(checkpoint, payload)
                        atomic_json(directory / "verification" / f"{variant}_{stage:06d}.json", {"checks": verify_learning_state(payload, torch.load(checkpoint, map_location="cpu", weights_only=False)), "collector": old, "stage": stage})
                        report = {"stage": stage, "variant": variant, "counters": learner.counters,
                            "replay_audit": audit, "actual_policy_batch_sources": dict(learner.policy_batch_sources),
                            "actual_policy_replay_origin_sources": dict(learner.policy_replay_origin_counts),
                            "actual_policy_target_types":dict(learner.policy_target_type_counts),
                            "root_explore_audit":meta["root_explore_audit"],
                            "policy_label_audit": meta['policy_label_audit'], "teacher_query_audit": meta['teacher_query_audit'],
                            "updates": updates, "internal_holdout": validation_metrics(learner.model, validation, device),
                            "collection_metrics": {k: meta[k] for k in ("moves_per_second", "simulations_per_second", "mean_inference_batch", "elapsed_seconds", "reasons", "physical_inference_rows", "padding_rows")} if collection["stage"] == stage else {},
                            "tactical_metrics": meta.get("tactical_metrics", {}),
                            "behavior_metrics":meta['behavior_metrics'], "behavior_contract":meta['behavior_contract'],
                            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                        report_path = directory / variant / "reports" / f"stage_{stage:06d}.json"; atomic_json(report_path, report)
                        progress["variants"][variant] = {"stage": stage, "checkpoint": str(checkpoint.relative_to(root)),
                            "sha256": sha(checkpoint), "model_hash": payload["model_hash"], "counters": payload["counters"],
                            "report": str(report_path.relative_to(root)), "report_sha256": sha(report_path)}
                        commit()
                        publish_current()
                        update(completed_updates=sum(v["counters"]["effective_updates"] - source["counters"]["effective_updates"] for v in progress["variants"].values()),
                            new_training_games=sum(len(v) for v in progress["collections"].values()) * plan["games_per_stage"])
                        del learner, values, policy, shards, payload; check()
                    if stage % plan["stages_per_block"]: continue
                    participants = {name: {"path": progress["variants"][name]["checkpoint"],
                        "sha256": progress["variants"][name]["sha256"], "model_hash": progress["variants"][name]["model_hash"]}
                        for name in plan["training_variants"]}
                    if "control" not in participants:participants["control"]=plan["participants"]["control"]
                    participants["parent"] = plan["participants"]["parent"]; participants["repeat"] = participants["parent"]
                    participants["guard86"] = plan["participants"]["guard86"]
                    participants["champion"] = progress["champion"]
                    screen = evaluate(f"block_{stage:06d}_screen", participants, plan["screen_families"], teachers)
                    full = (evaluate(f"block_{stage:06d}_full", participants, plan["full_families"], teachers)
                            if screen["review"]["screen_promising"] else None)
                    confirmed = full is not None and full["review"]["route_confirmed"]
                    atomic_json(directory / f"BLOCK_{stage:06d}_RESULTS.json", {"screen": screen["review"],
                        "full": full["review"] if full else None, "route_confirmed": confirmed,
                        "interpretation": plan["interpretation"]})
                    if not confirmed:
                        if severe_regression(screen) or stage==plan['maximum_stages']:
                            update(state="review_due", review_required=True, route_confirmed=False,
                                   reason="prespecified_severe_regression" if severe_regression(screen) else 'registered_horizon_complete_external_gain_unconfirmed'); return
                        update(state='interim_unconfirmed',review_required=False,route_confirmed=False,
                               reason='continue_only_to_preregistered24_stages; no_promotion_or_factor_change')
                        continue
                    item = participants["candidate"]
                    promoted = torch.load(inside(item["path"]), map_location="cpu", weights_only=False)
                    promoted["role"] = "champion"
                    promoted["lineage"]["qualification"] = progress["evaluations"][f"block_{stage:06d}_full"]
                    path = directory / "champions" / f"champion_{source['counters']['iteration'] + stage:06d}.pt"
                    atomic_checkpoint(path, promoted)
                    progress["champion"] = {"path": str(path.relative_to(root)), "sha256": sha(path),
                        "model_hash": item["model_hash"], "iteration": source["counters"]["iteration"] + stage}; commit()
                    publish_current()
                    update(state="qualified", route_confirmed=True, champion_iteration=source["counters"]["iteration"] + stage)
            update(state="segment_complete", review_required=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user": update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__": main()
