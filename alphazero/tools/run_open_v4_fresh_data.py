"""Paired streaming-data continuation; one GPU, full-state checkpoints.

Collect four identical-size chunks with a frozen Actor. The control sees the
first chunk; the candidate sees all four. Rules and training kernels are reused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path


def chunk_iterations(stage, first_iteration, chunks):
    if stage < 1 or first_iteration < 1 or chunks < 1:
        raise ValueError("positive stage and collection budgets required")
    start = first_iteration + (stage - 1) * chunks
    return list(range(start, start + chunks))


def replay_groups(stage, window, initial_iterations, first_iteration, chunks, variant):
    if variant not in ("control", "fresh") or window < 1:
        raise ValueError("invalid variant or replay window")
    groups = [[("original", i)] for i in initial_iterations]
    for index in range(1, stage + 1):
        ids = chunk_iterations(index, first_iteration, chunks)
        groups.append([("fresh", i) for i in (ids[:1] if variant == "control" else ids)])
    return groups[-window:]


def is_holdout(game_id, seed):
    # Both colors have the same game_id, so they cannot cross this boundary.
    return (game_id * 0x9E3779B1 ^ seed) % 10 == 0


def main():
    import copy
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
    from junqi_open_v4.pipeline import source_contract, validation_metrics
    from junqi_open_v4.games import collect_training_games
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = (root / args.experiment).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = OpenConfig.load(root / plan["parent_config"])
    original = (root / base.run_directory).resolve()
    directory = (root / plan["directory"]).resolve()
    if not plan_path.is_relative_to(root) or not directory.is_relative_to(root) or directory == root:
        raise ValueError("experiment must remain within alphazero")
    if directory.exists() != args.resume:
        raise ValueError("use a fresh directory, or explicit --resume after reviewing a failure")
    if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
        raise RuntimeError("STOP is present; no automatic clearing")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        directory.mkdir(parents=True, exist_ok=True)
        status = dict(schema="junqi-fresh-data-experiment-1", state="checking", pid=os.getpid(),
            started_at=utc_now(), experiment=str(plan_path), updates=plan["stages"] * plan["updates_per_stage"],
            stages=plan["stages"], completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; committed experiment checkpoints preserved")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        update()
        try:
            current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
            contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
            if current != plan["parent_current"] or sha(original / "CONTRACT.json") != current["contract_hash"]:
                raise ValueError("parent state changed after preregistration")
            if source_contract(root) != contract["source"] or contract["source"] != plan["source"]:
                raise ValueError("source contract mismatch")
            for role in ("learner", "actor", "champion"):
                if sha(original / current[role]["path"]) != current[role]["sha256"]:
                    raise ValueError("parent checkpoint digest mismatch")
            if plan["chunks"] != 4 or plan["chunk_games"] != base.games_per_iteration or base.games_per_iteration != 64:
                raise ValueError("preregistered 64/256 paired data experiment required")
            if plan["updates_per_stage"] != base.updates_per_iteration or plan["replay_window"] != base.replay_iterations:
                raise ValueError("update budget and replay window must remain unchanged")
            layouts = root / base.layout_directory
            if sha(layouts / "manifest.json") != plan["layout_manifest_sha256"]:
                raise ValueError("layout boundary changed")
            rows = load_role(layouts, "train")
            backend = MultiLayoutBackend(rows, base.seed)
            provenance = plan | {"tool_sha256": sha(Path(__file__)), "experiment_sha256": sha(plan_path)}
            if args.resume:
                if json.loads((directory / "CONTRACT.json").read_text(encoding="utf-8")) != provenance:
                    raise ValueError("experiment code or plan changed; preserve prior branch")
                progress = json.loads((directory / "PROGRESS.json").read_text(encoding="utf-8"))
            else:
                atomic_json(directory / "CONTRACT.json", provenance)
                progress = {"collections": {}, "variants": {v: {"stage": 0} for v in ("control", "fresh")}}
                atomic_json(directory / "PROGRESS.json", progress)
            def commit():
                atomic_json(directory / "PROGRESS.json", progress)
            for record in progress["collections"].values():
                if sha(directory / record["path"]) != record["sha256"] or sha(directory / record["metadata"]) != record["metadata_sha256"]:
                    raise ValueError("committed collection changed")
            for record in progress["variants"].values():
                if record["stage"] and sha(directory / record["checkpoint"]) != record["sha256"]:
                    raise ValueError("committed experiment checkpoint changed")
            torch.set_num_threads(base.workers)
            device = torch.device(base.device)
            common_validation = []
            originals = {}
            for record in plan["initial_replay"]:
                path = root / record["path"]
                if sha(path) != record["sha256"] or sha(path.with_suffix(".json")) != record["metadata_sha256"]:
                    raise ValueError("original replay changed")
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if metadata["layout_role"] != "train" or metadata["config_hash"] != base.digest():
                    raise ValueError("original replay provenance mismatch")
                originals[record["iteration"]] = path
                common_validation.extend(s for s in read_selfplay_shard(path) if is_holdout(s.game_id, base.seed))
            common_validation = common_validation[:4096]
            if not common_validation:
                raise ValueError("common internal holdout missing")
            actor_payload = torch.load(original / current["actor"]["path"], map_location="cpu", weights_only=False)
            actor = PolicyValueNet(base.model).to(device).eval()
            actor.load_state_dict(actor_payload["model"])
            actor.search_v3_leaf_value_scale = actor_payload["leaf_value_scale"]
            if model_hash(actor) != current["actor"]["model_hash"]:
                raise ValueError("frozen Actor tensor mismatch")
            configs = {}
            for variant, count in (("control", 64), ("fresh", 256)):
                cfg = copy.deepcopy(base)
                cfg.run_directory = str((directory / variant).relative_to(root))
                cfg.initial_checkpoint = str((original / current["learner"]["path"]).relative_to(root))
                cfg.games_per_iteration = count
                configs[variant] = cfg
                atomic_json(directory / variant / "config.json", cfg.as_dict())
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != contract["teachers"]:
                    raise ValueError("frozen teacher changed")
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                for stage in range(1, plan["stages"] + 1):
                    if all(r["stage"] >= stage for r in progress["variants"].values()):
                        continue
                    for chunk, iteration in enumerate(chunk_iterations(stage, plan["first_collection_iteration"], plan["chunks"]), 1):
                        if str(iteration) in progress["collections"]:
                            continue
                        respect_stop()
                        if shutil.disk_usage(directory).free < plan["minimum_free_gib"] * 2**30:
                            raise RuntimeError("insufficient disk space")
                        update(state="collecting", stage=stage, chunk=chunk, chunk_total=plan["chunks"], collection_iteration=iteration)
                        samples, metadata = collect_training_games(base, actor, device, backend, teachers, iteration, count=plan["chunk_games"])
                        path = directory / f"replay/collection_{iteration:06d}.npz"
                        write_selfplay_shard(path, samples)
                        metadata.update(actor_model_hash=current["actor"]["model_hash"], config_hash=base.digest(),
                            shard_sha256=sha(path), experiment=str(plan_path), stage=stage, chunk=chunk,
                            generation_contract="unchanged base config; 64-game batches; training layouts only")
                        atomic_json(path.with_suffix(".json"), metadata)
                        progress["collections"][str(iteration)] = {"path": str(path.relative_to(directory)), "sha256": sha(path),
                            "metadata": str(path.with_suffix(".json").relative_to(directory)), "metadata_sha256": sha(path.with_suffix(".json"))}
                        commit()
                        del samples
                    for variant in ("control", "fresh"):
                        if progress["variants"][variant]["stage"] >= stage:
                            continue
                        respect_stop()
                        cfg = configs[variant]
                        old = progress["variants"][variant]
                        learner = Learner(cfg if old["stage"] else base, root)
                        learner.resume(directory / old["checkpoint"] if old["stage"] else original / current["learner"]["path"])
                        learner.config = cfg
                        learner.provenance = {"full_parent_checkpoint": current["learner"], "optimizer_reset": False,
                            "scheduler_reset": False, "rng_reset": False, "experiment": str(plan_path),
                            "changed_factor": "fresh training games per 64 updates", "frozen_actor": current["actor"],
                            "phase_schedule": "PPV unchanged", "variant": variant}
                        groups = replay_groups(stage, plan["replay_window"], plan["initial_replay_iterations"],
                            plan["first_collection_iteration"], plan["chunks"], variant)
                        training, replay = [], []
                        for source, iteration in [item for group in groups for item in group]:
                            path = originals[iteration] if source == "original" else directory / progress["collections"][str(iteration)]["path"]
                            part = read_selfplay_shard(path)
                            for sample in part:
                                sample.observation = sample.observation.astype(np.float16)
                            training.extend(s for s in part if not is_holdout(s.game_id, base.seed))
                            replay.append({"path": str(path.relative_to(root)), "sha256": sha(path)})
                            del part
                        if not training or len(training) > base.maximum_replay_samples:
                            raise ValueError("training sample cap or empty data; do not change budget silently")
                        before = validation_metrics(learner.model, common_validation, device)
                        update(state="training", stage=stage, variant=variant, completed_updates=old["stage"] * plan["updates_per_stage"],
                            training_samples=len(training))
                        started = time.perf_counter()
                        logs = learner.train(training, updates=plan["updates_per_stage"])
                        after = validation_metrics(learner.model, common_validation, device)
                        learner.counters["iteration"] = current["iteration"] + stage
                        checkpoint = directory / variant / f"checkpoints/learner_stage_{stage:06d}.pt"
                        atomic_checkpoint(checkpoint, learner.payload(role="learner", lineage={"parent": current,
                            "experiment": str(plan_path), "stage": stage, "replay": replay, "promotion": "none; diagnostic branch"}))
                        report = {"stage": stage, "variant": variant, "fresh_games": cfg.games_per_iteration,
                            "training_samples": len(training), "common_holdout_before": before, "common_holdout_after": after,
                            "training": logs, "counters": dict(learner.counters), "replay": replay,
                            "training_seconds": time.perf_counter() - started, "checkpoint_sha256": sha(checkpoint),
                            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                        atomic_json(directory / variant / f"reports/stage_{stage:06d}.json", report)
                        progress["variants"][variant] = {"stage": stage, "checkpoint": str(checkpoint.relative_to(directory)),
                            "sha256": sha(checkpoint), "model_hash": model_hash(learner.model), "holdout": after,
                            "effective_updates": learner.counters["effective_updates"]}
                        commit()
                        update(completed_updates=stage * plan["updates_per_stage"], checkpoint=str(checkpoint),
                            effective_updates=learner.counters["effective_updates"])
                        del learner, training
                        torch.cuda.empty_cache()
                del actor
                torch.cuda.empty_cache()
                respect_stop()
                agents, checkpoint_provenance = {}, {}
                frozen = {"parent": original / current["learner"]["path"], "parent_repeat": original / current["learner"]["path"],
                    **{v: directory / progress["variants"][v]["checkpoint"] for v in ("control", "fresh")}}
                for name, checkpoint in frozen.items():
                    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model = PolicyValueNet(base.model).to(device).eval()
                    model.load_state_dict(payload["model"])
                    model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
                    agents[name] = BatchedNeuralPUCT(model, device=device, inference_batch_size=base.inference_batch_size,
                        cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree)
                    checkpoint_provenance[name] = {"path": str(checkpoint), "sha256": sha(checkpoint), "model_hash": model_hash(model)}
                suite = json.loads((original / "development/suite.json").read_text(encoding="utf-8"))["states"]
                suite = select_families(suite, plan["evaluation_families"], plan["evaluation_seed"])
                matchups = [("fresh", "control"), ("fresh", "parent"), ("control", "parent")]
                matchups += [(name, f"teacher{i}") for i in range(len(base.teacher_candidates)) for name in frozen]
                update(state="arena", matchups=matchups)
                arena = run_arena(suite, matchups, agents, teachers, {name: base.simulations for name in agents}, seed=plan["evaluation_seed"])
                arena.update(checkpoints=checkpoint_provenance, teacher_provenance=teachers.provenance,
                    purpose="fresh-data-per-update paired comparison; no production promotion")
                atomic_json(directory / "arena.json", arena)
            paired = {key: teacher_deltas(arena, before, after) for key, before, after in (
                ("fresh_vs_control", "control", "fresh"), ("fresh_vs_parent", "parent", "fresh"),
                ("control_vs_parent", "parent", "control"), ("duplicate", "parent", "parent_repeat"))}
            def duel(a, b):
                return next(s["family_score"] for s in arena["summaries"] if s["candidate"] == a and s["opponent"] == b)
            reliable = len(paired["duplicate"]) == 2 and all(abs(r["mean"]) <= plan["duplicate_control_tolerance"] for r in paired["duplicate"])
            def positive(rows):
                return len(rows) == 2 and all(r["ci95"][0] > 0 for r in rows)
            fresh_preferred = reliable and positive(paired["fresh_vs_control"]) and positive(paired["fresh_vs_parent"]) and duel("fresh", "control")["ci95"][0] > .5 and duel("fresh", "parent")["ci95"][0] > .5
            control_promising = reliable and positive(paired["control_vs_parent"]) and duel("control", "parent")["ci95"][0] > .5
            result = {"schema": "junqi-fresh-data-result-1", "experiment": plan, "arena_summaries": arena["summaries"],
                "teacher_deltas": paired, "fresh_route_preferred": bool(fresh_preferred), "control_route_promising": bool(control_promising),
                "duplicate_control_within_tolerance": reliable, "progress": progress,
                "promotion": "none; promising route requires existing full development gates", "parent_roles_unchanged":
                json.loads((original / "CURRENT.json").read_text(encoding="utf-8")) == current}
            if not result["parent_roles_unchanged"]:
                raise RuntimeError("original roles changed during experiment")
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, fresh_route_preferred=bool(fresh_preferred),
                control_route_promising=bool(control_promising), result=str(directory / "RESULTS.json"))
            print(json.dumps({"fresh_route_preferred": fresh_preferred, "control_route_promising": control_promising,
                "duplicate_control_within_tolerance": reliable}, ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
