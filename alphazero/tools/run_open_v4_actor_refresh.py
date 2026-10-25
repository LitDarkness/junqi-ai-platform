"""Paired actor refresh using full learners only for experimental collection."""
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
from open_v4_value_source_factor import make_learner, audit_shard
from open_v4_actor_refresh_factor import collection_source, verify_collection_pair


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.learner import atomic_checkpoint, model_hash
    from junqi_open_v4.pipeline import validation_metrics, promotion_decision
    from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard
    from junqi_open_v4.games import collect_training_games
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.arena import run_arena, select_families

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    def inside(name):
        path = (root / name).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("paths must stay inside alphazero")
        return path
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path = inside(args.experiment)
    plan = read_json(plan_path)
    base = OpenConfig.load(inside(plan["parent_config"]))
    if (base.digest() != plan["parent_config_hash"] or plan["stages"] != 6
            or plan["updates_per_stage"] != 64 or plan["evaluation_families"] != 64
            or plan["evaluation_budget"] != 256
            or (base.games_per_iteration, base.simulations, base.repeated_simulations, base.replay_iterations) != (256, 64, 128, 3)
            or plan["value_allowed_modes"] != {"control": "all", "candidate": "all"}
            or plan["collector_refresh"] != {"control": "qualified26", "candidate": "previous_complete_learner"}):
        raise ValueError("registered single-factor complete-state experiment required")
    directory = inside(plan["directory"])
    stops = [inside(p) for p in plan["stop_paths"]] + [directory / "STOP", directory / "supervision/STOP"]
    if any(p.exists() for p in stops):
        raise RuntimeError("STOP requested; no automatic clearing")
    with ExitStack() as locks:
        for frozen in plan["frozen_runs"]:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(inside(frozen["directory"]) / f"supervision/{name}.lock"))
        if directory.exists() != args.resume:
            raise ValueError("fresh directory or explicit reviewed --resume required")
        if (directory / "RESULTS.json").exists():
            raise ValueError("completed result exists; preserve it and review")
        directory.mkdir(parents=True, exist_ok=True)
        status = dict(schema="junqi-actor-refresh-experiment-1", state="checking", pid=os.getpid(),
                      started_at=utc_now(), stages=6, updates=384, completed_updates=0, new_training_games=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def check():
            if any(p.exists() for p in stops):
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; complete checkpoints preserved")
            if shutil.disk_usage(directory).free < 10 * 2**30:
                raise RuntimeError("minimum disk reserve reached")
        def verify_sources():
            for frozen in plan["frozen_runs"]:
                run = inside(frozen["directory"]); current = frozen["current"]
                if read_json(run / "CURRENT.json") != current or sha(run / "CONTRACT.json") != current["contract_hash"]:
                    raise ValueError("frozen run changed")
                for role in ("learner", "actor", "champion"):
                    if sha(run / current[role]["path"]) != current[role]["sha256"]:
                        raise ValueError("frozen role changed")
            for name, digest in {**plan["source"], **plan["evidence_files"]}.items():
                if sha(inside(name)) != digest:
                    raise ValueError("source or qualification/negative evidence changed")
            if sha(inside(plan["baseline_control"]["path"])) != plan["baseline_control"]["sha256"]:
                raise ValueError("complete baseline control changed")
        update()
        try:
            check(); verify_sources()
            if sha(inside(base.layout_directory) / "manifest.json") != plan["layout_manifest_hash"]:
                raise ValueError("layout boundary changed")
            source = inside(plan["source_checkpoint"]["path"])
            if sha(source) != plan["source_checkpoint"]["sha256"]:
                raise ValueError("complete parent changed")
            payload = torch.load(source, map_location="cpu", weights_only=False)
            if payload["counters"] != plan["source_counters"] or payload["config_hash"] != base.digest():
                raise ValueError("complete parent counters/config differ")
            proof = plan | {"tool_sha256": sha(Path(__file__)), "plan_sha256": sha(plan_path)}
            configs = {}
            for variant in ("control", "candidate"):
                cfg = copy.deepcopy(base)
                cfg.run_directory = str((directory / variant).relative_to(root))
                cfg.initial_checkpoint = str(source.relative_to(root))
                cfg.validate(); configs[variant] = cfg
            if args.resume:
                if read_json(directory / "CONTRACT.json") != proof:
                    raise ValueError("plan/tool changed; preserve prior experiment")
                progress = read_json(directory / "PROGRESS.json")
            else:
                atomic_json(directory / "CONTRACT.json", proof)
                progress = {"variants": {}, "collections": {}}
                for variant, cfg in configs.items():
                    allowed_modes = plan["value_allowed_modes"][variant]
                    migrated = fork_payload(payload, cfg, {"source": plan["source_checkpoint"],
                        "variant": variant, "changed_factor": plan["changed_factor"],
                        "value_allowed_modes": allowed_modes, "optimizer_reset": False, "rng_reset": False})
                    migrated["experiment_parameters"] = {"value_allowed_modes": allowed_modes}
                    checks = verify_learning_state(payload, migrated)
                    path = directory / variant / "checkpoints/learner_stage_000000.pt"
                    atomic_checkpoint(path, migrated)
                    atomic_json(directory / variant / "config.json", cfg.as_dict())
                    atomic_json(directory / variant / "SEED_VERIFICATION.json", checks)
                    progress["variants"][variant] = {"stage": 0, "checkpoint": str(path.relative_to(directory)),
                        "sha256": sha(path), "model_hash": migrated["model_hash"]}
                atomic_json(directory / "PROGRESS.json", progress)
            def commit(): atomic_json(directory / "PROGRESS.json", progress)
            for item in progress["variants"].values():
                if sha(directory / item["checkpoint"]) != item["sha256"]:
                    raise ValueError("committed learner changed")
                if item.get("report") and sha(directory / item["report"]) != item["report_sha256"]:
                    raise ValueError("committed training report changed")
            for item in progress["collections"].values():
                if sha(inside(item["path"])) != item["sha256"] or sha(inside(item["metadata"])) != item["metadata_sha256"]:
                    raise ValueError("committed refreshed collection changed")
            torch.set_num_threads(base.workers); device = torch.device(base.device)
            def read(item):
                path = inside(item["path"])
                meta = inside(item.get("metadata", str(path.with_suffix(".json").relative_to(root))))
                if sha(path) != item["sha256"] or sha(meta) != item["metadata_sha256"]:
                    raise ValueError("fixed training/holdout source changed")
                metadata = read_json(meta)
                if metadata["layout_role"] != "train" or metadata["shard_sha256"] != sha(path):
                    raise ValueError("nontraining or unverified replay")
                rows = read_selfplay_shard(path)
                for row in rows:
                    row.observation = row.observation.astype(np.float16)
                return rows
            validation = []
            for item in plan["validation_replay"]:
                validation.extend(s for s in read(item) if is_holdout(s.game_id, base.seed))
            validation = validation[:4096]
            if not validation: raise ValueError("fixed internal holdout absent")
            backend = MultiLayoutBackend(load_role(inside(base.layout_directory), "train"), base.seed)
            with TeacherLeague(root, base.teacher_candidates, base.workers) as collectors:
                if collectors.provenance != plan["teachers"]: raise ValueError("collector teacher provenance changed")
                original_submit = collectors.submit_actions
                def checked_submit(states, ids):
                    check(); return original_submit(states, ids)
                collectors.submit_actions = checked_submit
                for stage in range(1, 7):
                    for variant in ("control", "candidate"):
                        old = progress["variants"][variant]
                        if old["stage"] >= stage: continue
                        check(); cfg = configs[variant]
                        learner = make_learner(cfg, root, plan["family_modes"], plan["value_allowed_modes"][variant])
                        learner.resume(directory / old["checkpoint"])
                        collection_iteration = 44 + stage
                        if variant == "candidate" and stage >= 2 and str(collection_iteration) not in progress["collections"]:
                            actor_source = collection_source(stage, variant, old, plan["source_checkpoint"])
                            actor_path = directory / actor_source["path"]
                            if sha(actor_path) != actor_source["sha256"]:
                                raise ValueError("experimental actor full checkpoint changed")
                            saved_actor = torch.load(actor_path, map_location="cpu", weights_only=False)
                            if saved_actor["model_hash"] != actor_source["model_hash"] or saved_actor["counters"]["iteration"] != 26 + stage - 1:
                                raise ValueError("experimental collector lineage mismatch")
                            actor = PolicyValueNet(base.model).to(device).eval()
                            actor.load_state_dict(saved_actor["model"])
                            actor.search_v3_leaf_value_scale = saved_actor["leaf_value_scale"]
                            if model_hash(actor) != actor_source["model_hash"]:
                                raise ValueError("experimental actor tensor mismatch")
                            update(state="collecting", variant=variant, stage=stage, collection_iteration=collection_iteration,
                                   actor_model_hash=actor_source["model_hash"], actor_effective_updates=saved_actor["counters"]["effective_updates"])
                            rows, metadata = collect_training_games(cfg, actor, device, backend, collectors, collection_iteration)
                            if model_hash(actor) != actor_source["model_hash"]:
                                raise ValueError("collection mutated experimental actor")
                            actor_record = dict(actor_source, path=str(actor_path.relative_to(root)),
                                                counters=saved_actor["counters"], experimental_only=True, production_promotion=False)
                            metadata.update(actor_model_hash=actor_source["model_hash"], actor_checkpoint=actor_record,
                                            config_hash=cfg.digest(), collection_budget=[cfg.simulations, cfg.repeated_simulations],
                                            stage=stage, variant=variant, experiment=str(plan_path))
                            frozen_metadata = read_json(inside(plan["replay_sources"][f"new_{collection_iteration:06d}"]["metadata"]))
                            verify_collection_pair(frozen_metadata, metadata, actor_source["model_hash"])
                            actual_modes, _ = audit_shard(rows, metadata)
                            if any(plan["family_modes"].get(str(k), plan["family_modes"].get(k)) != v for k, v in actual_modes.items()):
                                raise ValueError("paired collection mode mapping changed")
                            shard = directory / variant / f"replay/collection_{collection_iteration:06d}.npz"
                            if shard.exists() or shard.with_suffix(".json").exists():
                                raise ValueError("uncommitted collection; inspect before explicit recovery")
                            write_selfplay_shard(shard, rows); metadata["shard_sha256"] = sha(shard)
                            atomic_json(shard.with_suffix(".json"), metadata)
                            progress["collections"][str(collection_iteration)] = {
                                "path": str(shard.relative_to(root)), "sha256": sha(shard),
                                "metadata": str(shard.with_suffix(".json").relative_to(root)),
                                "metadata_sha256": sha(shard.with_suffix(".json")),
                                "actor_checkpoint": actor_record, "new_training_games": 256}
                            commit(); update(new_training_games=256 * len(progress["collections"]))
                            del actor, saved_actor, rows; torch.cuda.empty_cache()
                            # Collection constructs a separate network; restore training RNG too.
                            learner.resume(directory / old["checkpoint"])
                        groups = replay_window(stage, 3, [24, 25, 26], 45)
                        training, replay = [], []
                        for origin, number in groups:
                            item = (progress["collections"][str(number)] if variant == "candidate" and origin == "new" and number >= 46
                                    else plan["replay_sources"][f"{origin}_{number:06d}"])
                            part = read(item)
                            training.extend(s for s in part if not is_holdout(s.game_id, base.seed))
                            replay.append(item | {"role": "train"})
                        if not training or len(training) > base.maximum_replay_samples:
                            raise ValueError("fixed replay cap exceeded")
                        batch_digest = hashlib.sha256(); original_batch = learner._batch
                        def tracked(pool):
                            selected = original_batch(pool)
                            for s in selected:
                                batch_digest.update(str((s.game_id, s.state_hash, s.player, s.ply)).encode())
                            return selected
                        learner._batch = tracked
                        before = validation_metrics(learner.model, validation, device)
                        update(state="training", variant=variant, stage=stage,
                               completed_updates=old["stage"] * 64, training_samples=len(training))
                        started = time.perf_counter(); logs = learner.train(training, updates=64)
                        learner.counters["iteration"] = 26 + stage
                        after = validation_metrics(learner.model, validation, device)
                        checkpoint = directory / variant / f"checkpoints/learner_stage_{stage:06d}.pt"
                        if checkpoint.exists(): raise ValueError("uncommitted checkpoint; inspect before recovery")
                        atomic_checkpoint(checkpoint, learner.payload(role="learner", lineage={"experiment": str(plan_path),
                            "source": plan["source_checkpoint"], "stage": stage, "replay": replay,
                            "changed_factor": plan["changed_factor"], "batch_digest": batch_digest.hexdigest(), "promotion": "none"}))
                        report = {"stage": stage, "variant": variant, "training": logs, "training_samples": len(training),
                            "counters": dict(learner.counters), "batch_digest": batch_digest.hexdigest(),
                            "value_allowed_modes": plan["value_allowed_modes"][variant],
                            "policy_pool_and_anchor_use_all_sources": True,
                            "sampling_rule_unchanged": True,
                            "paired_batches_expected": stage == 1,
                            "value_pool_size": learner.value_pool_size,
                            "phase_batch_digests": {k:v.hexdigest() for k,v in learner.phase_batch_digests.items()},
                            "sampling_key_digest": learner.sampling_key_digest.hexdigest(),
                            "common_holdout_before": before, "common_holdout_after": after, "replay": replay,
                            "training_seconds": time.perf_counter() - started, "checkpoint_sha256": sha(checkpoint),
                            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "new_training_games": 256 if variant == "candidate" and stage >= 2 else 0,
                            "selfplay_moves_per_second": (read_json(inside(progress["collections"][str(collection_iteration)]["metadata"]))["moves_per_second"]
                                                         if variant == "candidate" and stage >= 2 else None),
                            "collector_mode": plan["collector_refresh"][variant],
                            "collector_checkpoint": (progress["collections"][str(collection_iteration)]["actor_checkpoint"]
                                                     if variant == "candidate" and stage >= 2 else plan["source_checkpoint"])}
                        report_path = directory / variant / f"reports/stage_{stage:06d}.json"
                        atomic_json(report_path, report)
                        progress["variants"][variant] = {"stage": stage, "checkpoint": str(checkpoint.relative_to(directory)),
                            "sha256": sha(checkpoint), "model_hash": model_hash(learner.model),
                            "report": str(report_path.relative_to(directory)), "report_sha256": sha(report_path),
                            "effective_updates": learner.counters["effective_updates"], "batch_digest": batch_digest.hexdigest(),
                            "phase_batch_digests": {k:v.hexdigest() for k,v in learner.phase_batch_digests.items()},
                            "sampling_key_digest": learner.sampling_key_digest.hexdigest()}
                        commit(); del learner, training, part; torch.cuda.empty_cache()
                    control = progress["variants"]["control"]; candidate = progress["variants"]["candidate"]
                    if stage == 1 and control["batch_digest"] != candidate["batch_digest"]:
                        raise ValueError("identical first-stage replay must give identical batches")
                    if stage == 1:
                        first_checks = verify_learning_state(
                            torch.load(directory / control["checkpoint"], map_location="cpu", weights_only=False),
                            torch.load(directory / candidate["checkpoint"], map_location="cpu", weights_only=False))
                        atomic_json(directory / "FIRST_STAGE_REPRODUCTION.json", first_checks)
                    if control["sampling_key_digest"] != candidate["sampling_key_digest"]:
                        raise ValueError("paired sampler random keys diverged")
            if progress["variants"]["control"]["model_hash"] != plan["expected_control_model_hash"]:
                raise ValueError("original64-data control not reproduced; do not evaluate/select")
            baseline = torch.load(inside(plan["baseline_control"]["path"]), map_location="cpu", weights_only=False)
            reproduced = torch.load(directory / progress["variants"]["control"]["checkpoint"], map_location="cpu", weights_only=False)
            full_control_checks = verify_learning_state(baseline, reproduced)
            atomic_json(directory / "CONTROL_REPRODUCTION.json", full_control_checks)
            del baseline, reproduced
            check(); models, checkpoint_provenance = {}, {}
            for name, path in {"parent": source, **{v: directory / row["checkpoint"] for v, row in progress["variants"].items()}}.items():
                saved = torch.load(path, map_location="cpu", weights_only=False)
                model = PolicyValueNet(base.model).to(device).eval(); model.load_state_dict(saved["model"])
                model.search_v3_leaf_value_scale = saved["leaf_value_scale"]
                if model_hash(model) != saved["model_hash"]: raise ValueError("arena tensor mismatch")
                models[name] = model
                checkpoint_provenance[name] = {"path": str(path.relative_to(root)), "sha256": sha(path), "model_hash": saved["model_hash"]}
            models["repeat"] = models["parent"]
            suite_path = inside(plan["suite"]["path"])
            if sha(suite_path) != plan["suite"]["sha256"]: raise ValueError("development suite changed")
            suite = select_families(read_json(suite_path)["states"], 64, plan["evaluation_seed"])
            if len({r["family"] for r in suite}) != 64: raise ValueError("incomplete development coverage")
            agents = {name: StopAwareSearch(BatchedNeuralPUCT(model, device=device,
                inference_batch_size=base.inference_batch_size, cache_size=base.cache_size,
                leaves_per_tree=base.leaves_per_tree), check) for name, model in models.items()}
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != plan["teachers"]: raise ValueError("frozen teacher differs")
                matchups = [("candidate", "parent"), ("candidate", "control")] + [
                    (name, f"teacher{i}") for i in range(len(teachers.modules)) for name in models]
                update(state="arena", planned_games=2 * len(suite) * len(matchups),
                       evaluation_budget=256, evaluation_families=64, completed_updates=384,
                       original_control_reproduced=True)
                if "arena" in progress:
                    if sha(directory / "arena.json") != progress["arena"]["sha256"]:
                        raise ValueError("committed arena changed")
                    arena = read_json(directory / "arena.json")
                else:
                    if (directory / "arena.json").exists(): raise ValueError("uncommitted arena; review before recovery")
                    arena = run_arena(suite, matchups, agents, teachers, {name: 256 for name in agents}, seed=plan["evaluation_seed"])
                    arena.update(checkpoints=checkpoint_provenance, teacher_provenance=teachers.provenance,
                                 purpose=plan["hypothesis"])
                    atomic_json(directory / "arena.json", arena)
                    progress["arena"] = {"path": "arena.json", "sha256": sha(directory / "arena.json")}; commit()
            check(); verify_sources()
            gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
                ("candidate_vs_parent", "parent", "candidate"), ("candidate_vs_control", "control", "candidate"), ("duplicate", "parent", "repeat"))}
            gate = promotion_decision(arena, base)
            selected = route_signal([gains["candidate_vs_parent"], gains["candidate_vs_control"]], gains["duplicate"],
                                    [arena["summaries"][i]["family_score"] for i in (0, 1)], gate, 64, .02)
            result = {"schema": "junqi-actor-refresh-result-1", "experiment": plan, "summaries": arena["summaries"],
                "teacher_deltas": gains, "existing_gate": gate, "actor_refresh_route_confirmed": selected,
                "progress": progress, "original_control_reproduced": True, "frozen_roles_unchanged": True,
                "matched_sampling_keys": True, "first_stage_batches_matched": True,
                "later_batches_intentionally_different": True, "new_training_games": 256 * len(progress["collections"]),
                "new_training_games_per_variant": {"control": 0, "candidate": 256 * len(progress["collections"])},
                "control_full_learning_state_reproduced": full_control_checks,
                "promotion": "none; review complete state and full primary gates", "arena_seconds": arena["elapsed_seconds"],
                "search_metrics": arena["search_metrics"], "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, actor_refresh_route_confirmed=selected)
            print(json.dumps({"actor_refresh_route_confirmed": selected, "existing_gate": gate}), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user": update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__": main()
