"""Paired search-data training from a qualified complete learning state.

Keep heavyweight imports inside main: spawned CPU teachers must not import Torch.
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


def variant_config(base, directory, source, variant):
    if variant not in ("control", "candidate"):
        raise ValueError("unknown search-data variant")
    cfg = copy.deepcopy(base)
    cfg.run_directory = str(Path(directory) / variant)
    cfg.initial_checkpoint = None if source is None else str(source)
    if variant == "candidate":
        cfg.simulations, cfg.repeated_simulations = 256, 256
    cfg.validate()
    allowed = {"run_directory", "initial_checkpoint", "simulations", "repeated_simulations"}
    changed = {key for key, value in cfg.as_dict().items() if value != base.as_dict()[key]}
    if changed - allowed or base.simulations != 64 or base.repeated_simulations != 128:
        raise ValueError("only the registered search budget may change")
    return cfg


def replay_window(stage, window, initial_iterations, first_collection):
    if stage < 1 or window != len(initial_iterations):
        raise ValueError("invalid rolling replay window")
    groups = [("initial", i) for i in initial_iterations]
    groups += [("new", first_collection + i - 1) for i in range(1, stage + 1)]
    return groups[-window:]


def paired_schedule(metadata):
    fields = ("index", "family", "mode", "actor_side", "teacher_id", "provenance")
    return [{key: game[key] for key in fields} for game in sorted(metadata["games"], key=lambda g: g["index"])]


def verify_learning_state(before, after):
    import numpy as np
    import torch
    def same(a, b):
        if isinstance(a, torch.Tensor):
            return isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a.cpu(), b.cpu())
        if isinstance(a, np.ndarray):
            return isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
        if isinstance(a, dict):
            return isinstance(b, dict) and a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
        if isinstance(a, (list, tuple)):
            return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        return a == b
    fields = ("model", "optimizer", "scheduler", "counters", "rng", "leaf_value_scale", "model_hash")
    checks = {field: same(before[field], after[field]) for field in fields}
    if not all(checks.values()):
        raise ValueError(f"full learning state changed during migration: {checks}")
    return checks


def route_signal(gains, duplicate, duels, gate, families, tolerance):
    reliable = (families >= 64 and len(duplicate) == 2 and all(
        r["families"] >= families and abs(r["mean"]) <= tolerance for r in duplicate))
    positive = len(gains) == 2 and all(len(rows) == 2 and all(r["families"] >= families and r["ci95"][0] > 0 for r in rows)
                   for rows in gains)
    wins = len(duels) == 2 and all(r["families"] >= families and r["ci95"][0] > .5 for r in duels)
    return bool(reliable and positive and wins and gate["champion_accepted"])


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from evaluate_open_v4_search_confirmation import full_budget_confirmed
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
    from junqi_open_v4.pipeline import source_contract, validation_metrics, promotion_decision
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
    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("all experiment paths must stay in alphazero")
        return path
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path = inside(args.experiment)
    plan = read_json(plan_path)
    base = OpenConfig.load(inside(plan["parent_config"]))
    if base.digest() != plan["parent_config_hash"]:
        raise ValueError("parent configuration changed")
    if (base.games_per_iteration, base.updates_per_iteration, base.replay_iterations) != (256, 64, 3):
        raise ValueError("fixed fresh-data/update/window budgets required")
    if plan["stages"] != 6 or plan["evaluation_budgets"] != {"primary": 256, "secondary": 64}:
        raise ValueError("registered six-stage paired experiment required")
    if plan["evaluation_families"] != {"primary": 64, "secondary": 32}:
        raise ValueError("complete primary and diagnostic secondary coverage required")
    if plan["first_collection_iteration"] <= plan["previous_max_collection_iteration"]:
        raise ValueError("new collections must use a fresh explicit seed namespace")
    directory = inside(plan["directory"])
    frozen_runs = [(inside(r["directory"]), r) for r in plan["frozen_runs"]]
    if len({path for path, _ in frozen_runs}) != len(frozen_runs):
        raise ValueError("duplicate frozen lock directories")
    stops = [inside(p) for p in plan["stop_paths"]] + [directory / "STOP"]
    def stop_requested():
        return any(path.exists() for path in stops)
    if stop_requested():
        raise RuntimeError("STOP requested; never automatically clear it")
    with ExitStack() as locks:
        for run, _ in frozen_runs:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
        if directory.exists() != args.resume:
            raise ValueError("fresh directory or explicit reviewed --resume required")
        directory.mkdir(parents=True, exist_ok=True)
        status = dict(schema="junqi-search-data-training-1", state="checking", pid=os.getpid(),
            started_at=utc_now(), experiment=str(plan_path), stages=plan["stages"],
            updates=plan["stages"] * base.updates_per_iteration, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if stop_requested():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; committed checkpoints preserved")
        def frozen_intact():
            for run, record in frozen_runs:
                if read_json(run / "CURRENT.json") != record["current"]:
                    raise ValueError("frozen roles changed")
                contract = read_json(run / "CONTRACT.json")
                if sha(run / "CONTRACT.json") != record["current"]["contract_hash"]:
                    raise ValueError("frozen contract digest mismatch")
                if contract["source"] != plan["source"] or source_contract(root) != plan["source"]:
                    raise ValueError("native/training source changed")
                for role in ("learner", "actor", "champion"):
                    item = record["current"][role]
                    if sha(run / item["path"]) != item["sha256"]:
                        raise ValueError("frozen checkpoint digest mismatch")
            for path, digest in plan["evidence_files"].items():
                if sha(inside(path)) != digest:
                    raise ValueError("registered qualification evidence changed")
        def disk_check():
            if shutil.disk_usage(directory).free < plan["minimum_free_gib"] * 2**30:
                raise RuntimeError("minimum free disk budget reached")
        update()
        try:
            respect_stop()
            disk_check()
            frozen_intact()
            confirmation = read_json(inside(plan["qualification_result"]))
            if not confirmation["search_256_confirmed"] or not full_budget_confirmed(
                    confirmation["teacher_gain_256_vs_64"], confirmation["duplicate_control"],
                    confirmation["summaries"][0]["family_score"], confirmation["existing_gate"], 64, .02):
                raise ValueError("search256 has not passed full development confirmation")
            source = inside(plan["source_checkpoint"]["path"])
            if sha(source) != plan["source_checkpoint"]["sha256"]:
                raise ValueError("qualified complete source changed")
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if payload["config_hash"] != base.digest() or payload["counters"] != plan["source_counters"]:
                raise ValueError("complete source config/counters mismatch")
            if payload["model_hash"] != plan["source_checkpoint"]["model_hash"]:
                raise ValueError("qualified source tensor digest mismatch")
            layouts = inside(base.layout_directory)
            if sha(layouts / "manifest.json") != plan["layout_manifest_hash"]:
                raise ValueError("training/development layout boundary changed")
            suite_path = inside(plan["suite"]["path"])
            if sha(suite_path) != plan["suite"]["sha256"]:
                raise ValueError("development suite changed")
            suite_payload = read_json(suite_path)
            if suite_payload["contract"]["role"] != "dev":
                raise ValueError("only development evaluations may select routes")
            proof = plan | {"tool_sha256": sha(Path(__file__)), "plan_sha256": sha(plan_path)}
            configs = {v: variant_config(base, plan["directory"], plan["source_checkpoint"]["path"], v)
                       for v in ("control", "candidate")}
            if args.resume:
                if read_json(directory / "CONTRACT.json") != proof:
                    raise ValueError("plan/tool changed; preserve prior experiment")
                progress = read_json(directory / "PROGRESS.json")
            else:
                atomic_json(directory / "CONTRACT.json", proof)
                progress = {"collections": {}, "variants": {}, "evaluations": {}}
                for variant, cfg in configs.items():
                    evidence = {"source": plan["source_checkpoint"], "search_qualification": plan["qualification_result"],
                        "experiment": str(plan_path), "variant": variant, "optimizer_reset": False,
                        "scheduler_reset": False, "rng_reset": False, "changed_factor": "collection search budget"}
                    migrated = fork_payload(payload, cfg, evidence)
                    checks = verify_learning_state(payload, migrated)
                    path = directory / variant / "checkpoints/learner_stage_000000.pt"
                    atomic_checkpoint(path, migrated)
                    verify_learning_state(payload, torch.load(path, map_location="cpu", weights_only=False))
                    atomic_json(directory / variant / "config.json", cfg.as_dict())
                    progress["variants"][variant] = {"stage": 0, "checkpoint": str(path.relative_to(directory)),
                        "sha256": sha(path), "model_hash": migrated["model_hash"],
                        "effective_updates": migrated["counters"]["effective_updates"]}
                    atomic_json(directory / variant / "SEED_VERIFICATION.json", checks | {"source": plan["source_checkpoint"]})
                atomic_json(directory / "PROGRESS.json", progress)
            def commit():
                atomic_json(directory / "PROGRESS.json", progress)
            for record in progress["collections"].values():
                for path_key, hash_key in (("path", "sha256"), ("metadata", "metadata_sha256")):
                    if sha(directory / record[path_key]) != record[hash_key]:
                        raise ValueError("committed collection changed")
            for record in progress["variants"].values():
                if sha(directory / record["checkpoint"]) != record["sha256"]:
                    raise ValueError("committed checkpoint changed")
                if record.get("report") and sha(directory / record["report"]) != record["report_sha256"]:
                    raise ValueError("committed training report changed")
            torch.set_num_threads(base.workers)
            device = torch.device(base.device)
            backend = MultiLayoutBackend(load_role(layouts, "train"), base.seed)
            originals, validation = {}, []
            for item in plan["initial_replay"]:
                path = inside(item["path"])
                if sha(path) != item["sha256"] or sha(path.with_suffix(".json")) != item["metadata_sha256"]:
                    raise ValueError("initial rolling training replay changed")
                metadata = read_json(path.with_suffix(".json"))
                if metadata["layout_role"] != "train" or metadata["config_hash"] != base.digest():
                    raise ValueError("initial training replay contract mismatch")
                originals[item["iteration"]] = path
                validation.extend(s for s in read_selfplay_shard(path) if is_holdout(s.game_id, base.seed))
            validation = validation[:4096]
            if not validation:
                raise ValueError("internal training holdout missing")
            actor = PolicyValueNet(base.model).to(device).eval()
            actor.load_state_dict(payload["model"])
            actor.search_v3_leaf_value_scale = payload["leaf_value_scale"]
            if model_hash(actor) != payload["model_hash"]:
                raise ValueError("frozen Actor tensor mismatch")
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != plan["teachers"]:
                    raise ValueError("teacher provenance changed")
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                for stage in range(1, plan["stages"] + 1):
                    iteration = plan["first_collection_iteration"] + stage - 1
                    for chunk, variant in enumerate(("control", "candidate"), 1):
                        key = f"{variant}:{iteration}"
                        if key in progress["collections"]:
                            continue
                        respect_stop()
                        disk_check()
                        update(state="collecting", stage=stage, variant=variant, chunk=chunk, chunk_total=2,
                            collection_iteration=iteration, completed_updates=progress["variants"][variant]["stage"] * 64)
                        cfg = configs[variant]
                        samples, metadata = collect_training_games(cfg, actor, device, backend, teachers, iteration)
                        if model_hash(actor) != payload["model_hash"]:
                            raise ValueError("collection mutated the frozen actor")
                        path = directory / variant / f"replay/collection_{iteration:06d}.npz"
                        if path.exists():
                            raise ValueError("uncommitted shard present; inspect before recovery")
                        write_selfplay_shard(path, samples)
                        metadata.update(actor_model_hash=payload["model_hash"], config_hash=cfg.digest(),
                            shard_sha256=sha(path), stage=stage, variant=variant, experiment=str(plan_path),
                            collection_budget=[cfg.simulations, cfg.repeated_simulations])
                        atomic_json(path.with_suffix(".json"), metadata)
                        progress["collections"][key] = {"path": str(path.relative_to(directory)), "sha256": sha(path),
                            "metadata": str(path.with_suffix(".json").relative_to(directory)),
                            "metadata_sha256": sha(path.with_suffix(".json"))}
                        commit()
                        del samples
                    schedules = [paired_schedule(read_json(directory / progress["collections"][f"{v}:{iteration}"]["metadata"]))
                                 for v in ("control", "candidate")]
                    if schedules[0] != schedules[1]:
                        raise ValueError("paired training layouts/colors/modes/teacher schedule differ")
                    for variant in ("control", "candidate"):
                        old = progress["variants"][variant]
                        if old["stage"] >= stage:
                            continue
                        respect_stop()
                        disk_check()
                        cfg = configs[variant]
                        learner = Learner(cfg, root)
                        learner.resume(directory / old["checkpoint"])
                        training, replay = [], []
                        for origin, number in replay_window(stage, 3, plan["initial_replay_iterations"], plan["first_collection_iteration"]):
                            path = originals[number] if origin == "initial" else directory / progress["collections"][f"{variant}:{number}"]["path"]
                            part = read_selfplay_shard(path)
                            for sample in part:
                                sample.observation = sample.observation.astype(np.float16)
                            training.extend(s for s in part if not is_holdout(s.game_id, base.seed))
                            replay.append({"path": str(path.relative_to(root)), "sha256": sha(path), "role": "train"})
                        if not training or len(training) > base.maximum_replay_samples:
                            raise ValueError("empty/over-cap training replay; no silent budget change")
                        before = validation_metrics(learner.model, validation, device)
                        update(state="training", stage=stage, variant=variant, training_samples=len(training),
                            completed_updates=old["stage"] * 64)
                        started = time.perf_counter()
                        logs = learner.train(training, updates=64)
                        learner.counters["iteration"] = plan["source_counters"]["iteration"] + stage
                        after = validation_metrics(learner.model, validation, device)
                        checkpoint = directory / variant / f"checkpoints/learner_stage_{stage:06d}.pt"
                        if checkpoint.exists():
                            raise ValueError("uncommitted checkpoint present; inspect before recovery")
                        atomic_checkpoint(checkpoint, learner.payload(role="learner", lineage={"experiment": str(plan_path),
                            "source": plan["source_checkpoint"], "stage": stage, "replay": replay, "promotion": "none"}))
                        collection = read_json(directory / progress["collections"][f"{variant}:{iteration}"]["metadata"])
                        report = {"stage": stage, "variant": variant, "collection_budget": [cfg.simulations, cfg.repeated_simulations],
                            "new_games": 256, "training_samples": len(training), "common_holdout_before": before,
                            "common_holdout_after": after, "training": logs, "counters": dict(learner.counters),
                            "replay": replay, "training_seconds": time.perf_counter() - started,
                            "collection_metrics": {k: collection[k] for k in ("elapsed_seconds", "moves_per_second",
                                "simulations_per_second", "mean_inference_batch", "label_sources")},
                            "checkpoint_sha256": sha(checkpoint), "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                        report_path = directory / variant / f"reports/stage_{stage:06d}.json"
                        atomic_json(report_path, report)
                        progress["variants"][variant] = {"stage": stage, "checkpoint": str(checkpoint.relative_to(directory)),
                            "sha256": sha(checkpoint), "model_hash": model_hash(learner.model), "holdout": after,
                            "effective_updates": learner.counters["effective_updates"], "report": str(report_path.relative_to(directory)),
                            "report_sha256": sha(report_path)}
                        commit()
                        update(completed_updates=stage * 64, checkpoint=str(checkpoint),
                            effective_updates=learner.counters["effective_updates"])
                        del learner, training, part
                        torch.cuda.empty_cache()
                del actor, payload
                torch.cuda.empty_cache()
                respect_stop()
                models, checkpoint_provenance = {}, {}
                for name, checkpoint in {"parent": source, **{v: directory / progress["variants"][v]["checkpoint"] for v in configs}}.items():
                    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
                    model = PolicyValueNet(base.model).to(device).eval()
                    model.load_state_dict(saved["model"])
                    model.search_v3_leaf_value_scale = saved["leaf_value_scale"]
                    if model_hash(model) != saved["model_hash"]:
                        raise ValueError("arena checkpoint tensor mismatch")
                    models[name] = model
                    checkpoint_provenance[name] = {"path": str(checkpoint.relative_to(root)), "sha256": sha(checkpoint),
                        "model_hash": saved["model_hash"], "counters": saved["counters"]}
                models["repeat"] = models["parent"]
                arenas = {}
                for label in ("primary", "secondary"):
                    respect_stop()
                    count = plan["evaluation_families"][label]
                    budget = plan["evaluation_budgets"][label]
                    if label in progress["evaluations"]:
                        record = progress["evaluations"][label]
                        if sha(directory / record["path"]) != record["sha256"]:
                            raise ValueError("committed arena changed")
                        arenas[label] = read_json(directory / record["path"])
                        continue
                    suite = select_families(suite_payload["states"], count, plan["evaluation_seed"])
                    if len({row["family"] for row in suite}) != count:
                        raise ValueError("incomplete development family coverage")
                    agents = {name: StopAwareSearch(BatchedNeuralPUCT(model, device=device,
                        inference_batch_size=base.inference_batch_size, cache_size=base.cache_size,
                        leaves_per_tree=base.leaves_per_tree), respect_stop) for name, model in models.items()}
                    matchups = [("candidate", "parent"), ("candidate", "control"), ("control", "parent")]
                    matchups += [(name, f"teacher{i}") for i in range(len(teachers.modules)) for name in models]
                    update(state="arena", evaluation=label, evaluation_budget=budget, evaluation_families=count,
                        planned_games=len(suite) * 2 * len(matchups), matchups=matchups)
                    arena = run_arena(suite, matchups, agents, teachers, {name: budget for name in agents}, seed=plan["evaluation_seed"])
                    arena.update(checkpoints=checkpoint_provenance, teacher_provenance=teachers.provenance,
                        evaluation=label, matched_search_budget=budget, purpose="separate weight learning from runtime search gain; no automatic promotion")
                    path = directory / f"arena_{label}.json"
                    atomic_json(path, arena)
                    progress["evaluations"][label] = {"path": str(path.relative_to(directory)), "sha256": sha(path)}
                    commit()
                    arenas[label] = arena
                    del agents
                frozen_intact()
            respect_stop()
            primary = arenas["primary"]
            paired = {key: teacher_deltas(primary, before, after) for key, before, after in (
                ("candidate_vs_parent", "parent", "candidate"), ("candidate_vs_control", "control", "candidate"),
                ("control_vs_parent", "parent", "control"), ("duplicate", "parent", "repeat"))}
            gate = promotion_decision(primary, base)
            selected = route_signal([paired["candidate_vs_parent"], paired["candidate_vs_control"]], paired["duplicate"],
                [primary["summaries"][i]["family_score"] for i in (0, 1)], gate, 64, plan["duplicate_control_tolerance"])
            result = {"schema": "junqi-search-data-training-result-1", "experiment": plan,
                "primary_summaries": primary["summaries"], "secondary_summaries": arenas["secondary"]["summaries"],
                "primary_teacher_deltas": paired, "existing_gate": gate, "search_data_route_confirmed": selected,
                "progress": progress, "frozen_roles_unchanged": True, "training_games_per_variant": 1536,
                "attempted_updates_per_variant": 384, "promotion": "none; review complete learning/gate evidence before a qualified fork",
                "primary_arena_seconds": primary["elapsed_seconds"], "secondary_arena_seconds": arenas["secondary"]["elapsed_seconds"],
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, search_data_route_confirmed=selected,
                result=str(directory / "RESULTS.json"))
            print(json.dumps({"search_data_route_confirmed": selected, "existing_gate": gate}), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
