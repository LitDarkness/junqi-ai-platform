"""One-factor teacher-owned Policy anchor on frozen neural-search trajectories.

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
from pathlib import Path


def screening_signal(gains, duels):
    return (len(gains) == 2 and all(len(rows) == 2 and all(r["mean"] > 0 for r in rows) for rows in gains)
            and len(duels) == 2 and all(row["mean"] > .5 for row in duels))


def ordered_participants(participants):
    # promotion_decision consumes parent-to-candidate paired deltas. The arena
    # emits pairs in matchup order, so the baseline must precede the candidate.
    order = ("parent", "control", "legacy", "candidate", "repeat")
    if set(participants) != set(order):
        raise ValueError("parent/control/legacy/candidate/repeat participants required")
    return {name: participants[name] for name in order}


def validate_factor_plan(plan):
    if plan.get("experiment_type") != "teacher_owned_policy_anchor_factor":
        raise ValueError("explicit teacher-owned Policy anchor factor required")
    expected = {"selfplay_fraction": .4, "teacher_selfplay_fraction": .1,
                "actor_policy_target": "search", "deduplicate": True, "teacher_policy_anchor": False}
    if plan["control"] != expected or plan["candidate"] != dict(expected, teacher_policy_anchor=True):
        raise ValueError("only teacher-owned Policy eligibility may differ")
    if (plan["maximum_stages"], plan["stages_per_block"], plan["games_per_stage"],
        plan["updates_per_stage"], plan["replay_stages"]) != (6, 6, 256, 64, 3):
        raise ValueError("registered six-stage frozen replay schedule required")
    if (plan["screen_families"], plan["full_families"], plan["evaluation_budget"]) != (16, 64, 256):
        raise ValueError("original screening and full qualification budgets required")
    for name in ("frozen_collections", "reference_control"):
        if [row["stage"] for row in plan[name]] != list(range(1, 7)):
            raise ValueError("six complete matched frozen stages required")
    if len({r["path"] for r in plan["frozen_collections"]}) != 6:
        raise ValueError("distinct raw collection sources required")
    if not plan.get("legacy_reference") or plan.get("legacy_guard_required") is not True:
        raise ValueError("stronger legacy reference and extra superiority gate required")
    return True


def legacy_guard(gains, duel, *, full):
    metric = (lambda row: row["ci95"][0]) if full else (lambda row: row["mean"])
    return (len(gains) == 2 and all(metric(row) > 0 for row in gains)
            and metric(duel) > .5 and (not full or all(row["families"] >= 64 for row in gains + [duel])))


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
    validate_factor_plan(plan)
    import numpy as np
    import torch
    from fork_open_v4_qualified import fork_payload
    from run_open_v4_fresh_data import is_holdout
    from run_open_v4_search_training import verify_learning_state, route_signal
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_budgets import StopAwareSearch
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import atomic_checkpoint, model_hash
    from junqi_open_v4.training_replay import make_self_training_learner
    from junqi_open_v4.policy_anchor_replay import policy_anchor_pools
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_open_v4.pipeline import promotion_decision, validation_metrics
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if plan["schema"] != "junqi-self-training-plan-1": raise ValueError("plan contract mismatch")
    base = OpenConfig.load(inside(plan["parent_config"]))
    stops = [inside(p) for p in plan["stop_paths"]]
    def verify():
        for name, digest in plan["source"].items():
            if sha(inside(name)) != digest: raise ValueError(f"executed source changed: {name}")
        if sha(inside(plan["parent_config"])) != plan["parent_config_sha256"]: raise ValueError("base config changed")
        if sha(inside(base.layout_directory) / "manifest.json") != plan["layout_manifest_hash"]: raise ValueError("layouts changed")
        if sha(inside(plan["suite"]["path"])) != plan["suite"]["sha256"]: raise ValueError("dev suite changed")
        for frozen in plan["frozen_runs"]:
            if sha(inside(frozen["directory"]) / "CURRENT.json") != frozen["current_sha256"]: raise ValueError("archived roles changed")
        for item in plan["participants"].values():
            if sha(inside(item["path"])) != item["sha256"]: raise ValueError("frozen complete checkpoint changed")
        for item in plan["reference_control"]:
            if sha(inside(item["checkpoint"])) != item["sha256"]:
                raise ValueError("reference complete control changed")
        for item in plan["initial_replay"] + plan["frozen_collections"]:
            if sha(inside(item["path"])) != item["sha256"] or sha(inside(item["metadata"])) != item["metadata_sha256"]:
                raise ValueError("frozen raw trajectory or metadata changed")
    if any(p.exists() for p in stops): raise RuntimeError("STOP requested; never clear automatically")
    with ExitStack() as locks:
        for frozen in plan["frozen_runs"]:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(inside(frozen["directory"]) / f"supervision/{name}.lock"))
        verify()
        source = torch.load(inside(plan["source_checkpoint"]["path"]), map_location="cpu", weights_only=False)
        for field in ("model", "optimizer", "scheduler", "counters", "rng", "config", "leaf_value_scale"):
            if field not in source: raise ValueError("incomplete learning state")
        if source["counters"] != plan["source_counters"] or source["config"]["model"] != base.as_dict()["model"]:
            raise ValueError("qualified parent counters or architecture changed")
        configs = {}
        for variant in ("control", "candidate"):
            cfg = copy.deepcopy(base); params = plan[variant]
            cfg.run_directory = str((directory / variant).relative_to(root))
            cfg.initial_checkpoint = plan["source_checkpoint"]["path"]
            for key in ("selfplay_fraction", "teacher_selfplay_fraction", "actor_policy_target"):
                setattr(cfg, key, params[key])
            cfg.games_per_iteration = plan["games_per_stage"]; cfg.updates_per_iteration = plan["updates_per_stage"]
            cfg.validate(); configs[variant] = cfg
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
                      plan=str(plan_path.relative_to(root)), new_training_games=0, reused_training_games=0, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "supervision/STATUS.json", status)
            atomic_json(directory / "STATUS.json", status)
        def check():
            if any(p.exists() for p in stops):
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
                progress["variants"][variant] = {"stage": 0, "checkpoint": str(path.relative_to(root)),
                    "sha256": sha(path), "model_hash": migrated["model_hash"], "counters": migrated["counters"]}
            seed_champion = directory / "champions/champion_000026.pt"
            atomic_checkpoint(seed_champion, source)
            progress["champion"] = {"path": str(seed_champion.relative_to(root)),
                "sha256": sha(seed_champion), "model_hash": source["model_hash"], "iteration": 26}
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
                "experimental_learners": progress["variants"],
                "collector_contract": "Frozen original neural-only candidate trajectories; neither new learner collects games."})
        publish_current()
        device = torch.device(base.device); torch.set_num_threads(base.workers)
        def search(model):
            return StopAwareSearch(BatchedNeuralPUCT(model, device=device, inference_batch_size=base.inference_batch_size,
                cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree), check)
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
            matchups = [("candidate", "parent"), ("candidate", "control"), ("candidate", "legacy")] + [
                (key, f"teacher{i}") for i in range(len(teachers.modules)) for key in models]
            update(state="arena", arena=name, planned_games=2 * len(selected) * len(matchups), evaluation_families=families)
            arena = run_arena(selected, matchups, agents, teachers, {key: plan["evaluation_budget"] for key in agents},
                              seed=plan["evaluation_seed"])
            arena["checkpoints"] = participants
            gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
                ("parent", "parent", "candidate"), ("control", "control", "candidate"), ("duplicate", "parent", "repeat"), ("legacy", "legacy", "candidate"))}
            gate = promotion_decision(arena, base)
            duels = [row["family_score"] for row in arena["summaries"][:3]]
            arena["review"] = {"gains": gains, "gate": gate,
                "screen_promising": screening_signal([gains["parent"], gains["control"]], duels[:2]) and legacy_guard(gains["legacy"], duels[2], full=False),
                "route_confirmed": route_signal([gains["parent"], gains["control"]], gains["duplicate"], duels[:2], gate, families, .02) and legacy_guard(gains["legacy"], duels[2], full=True),
                "legacy_gate": {"required": True, "passed": legacy_guard(gains["legacy"], duels[2], full=True)},
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
            return rows, meta
        update()
        try:
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                if teachers.provenance != plan["teachers"]: raise ValueError("teacher identity changed")
                original_submit = teachers.submit_actions
                def checked_submit(states, ids): check(); return original_submit(states, ids)
                teachers.submit_actions = checked_submit
                # Prior screening and diagnostics remain immutable in their original run.
                # This new experiment trains only on identical frozen neural-only candidate trajectories.
                validation = []
                for item in plan["initial_replay"]:
                    samples, _ = read_shard(item)
                    validation.extend(s for s in samples if is_holdout(s.game_id, base.seed))
                validation = validation[:4096]
                for stage in range(1, plan["maximum_stages"] + 1):
                    for variant in ("control", "candidate"):
                        old = progress["variants"][variant]
                        if old["stage"] >= stage: continue
                        check(); verify(); cfg = configs[variant]; params = plan[variant]
                        if sha(inside(old["checkpoint"])) != old["sha256"]: raise ValueError("committed learner changed")
                        learner = make_self_training_learner(cfg, root); learner.resume(inside(old["checkpoint"]))
                        collection = next((r for r in progress["collections"][variant] if r["stage"] == stage), None)
                        if collection is None:
                            collection = dict(plan["frozen_collections"][stage - 1])
                            update(state="loading_frozen_replay", variant=variant, stage=stage,
                                   frozen_collection=collection["path"])
                            progress["collections"][variant].append(collection); commit()
                        meta = read_json(inside(collection["metadata"]))
                        records = (plan["initial_replay"] + progress["collections"][variant])[-plan["replay_stages"]:]
                        shards = [read_shard(item) for item in reversed(records)]
                        values, policy, audit = policy_anchor_pools(shards, deduplicate=params["deduplicate"],
                            teacher_policy_anchor=params["teacher_policy_anchor"], holdout=lambda family: is_holdout(family, base.seed))
                        update(state="training", variant=variant, stage=stage, replay_audit=audit)
                        original_step = learner.step
                        def checked_step(*a, **kw): check(); return original_step(*a, **kw)
                        learner.step = checked_step
                        updates = learner.train_pools(values, policy, updates=plan["updates_per_stage"])
                        learner.counters["iteration"] += 1
                        checkpoint = directory / variant / "checkpoints" / f"learner_stage_{stage:06d}.pt"
                        payload = learner.payload(role="learner", lineage={"plan": str(plan_path.relative_to(root)),
                            "previous": old, "replay": records, "single_factor": params})
                        if variant == "control":
                            reference = plan["reference_control"][stage - 1]
                            expected = torch.load(inside(reference["checkpoint"]), map_location="cpu", weights_only=False)
                            checks = verify_learning_state(expected, payload)
                            atomic_json(directory / "verification" / f"control_stage_{stage:06d}.json",
                                        {"stage": stage, "reference": reference, "all_learning_state_equal": checks})
                            del expected
                        atomic_checkpoint(checkpoint, payload)
                        report = {"stage": stage, "variant": variant, "counters": learner.counters,
                            "replay_audit": audit, "actual_policy_batch_sources": dict(learner.policy_batch_sources),
                            "updates": updates, "internal_holdout": validation_metrics(learner.model, validation, device),
                            "new_training_games": 0, "reused_collection": collection,
                            "historical_collection_metrics": {k: meta[k] for k in ("moves_per_second", "simulations_per_second", "mean_inference_batch", "elapsed_seconds", "reasons")},
                            "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
                        report_path = directory / variant / "reports" / f"stage_{stage:06d}.json"; atomic_json(report_path, report)
                        progress["variants"][variant] = {"stage": stage, "checkpoint": str(checkpoint.relative_to(root)),
                            "sha256": sha(checkpoint), "model_hash": payload["model_hash"], "counters": payload["counters"],
                            "report": str(report_path.relative_to(root)), "report_sha256": sha(report_path)}
                        commit()
                        publish_current()
                        update(completed_updates=sum(v["counters"]["effective_updates"] - source["counters"]["effective_updates"] for v in progress["variants"].values()),
                            new_training_games=0,
                            reused_training_games=len({c["path"] for rows in progress["collections"].values() for c in rows}) * plan["games_per_stage"])
                        del learner, values, policy, shards, payload; check()
                    if stage % plan["stages_per_block"]: continue
                    participants = {name: {"path": progress["variants"][name]["checkpoint"],
                        "sha256": progress["variants"][name]["sha256"], "model_hash": progress["variants"][name]["model_hash"]}
                        for name in ("candidate", "control")}
                    participants["parent"] = progress["champion"]; participants["repeat"] = participants["parent"]
                    participants["legacy"] = plan["legacy_reference"]
                    screen = evaluate(f"block_{stage:06d}_screen", participants, plan["screen_families"], teachers)
                    full = (evaluate(f"block_{stage:06d}_full", participants, plan["full_families"], teachers)
                            if screen["review"]["screen_promising"] else None)
                    confirmed = full is not None and full["review"]["route_confirmed"]
                    atomic_json(directory / f"BLOCK_{stage:06d}_RESULTS.json", {"screen": screen["review"],
                        "full": full["review"] if full else None, "route_confirmed": confirmed,
                        "interpretation": "Single factor: add teacher-owned Policy rows to identical frozen neural-search trajectories. Legacy actor imitation rows stay excluded. Original qualification plus stronger-legacy-reference guard required; one seed and reused development suite."})
                    if not confirmed:
                        update(state="review_due", review_required=True, route_confirmed=False,
                               reason="external_gain_unconfirmed; stop_without_blind_extension"); return
                    item = participants["candidate"]
                    promoted = torch.load(inside(item["path"]), map_location="cpu", weights_only=False)
                    promoted["role"] = "champion"
                    promoted["lineage"]["qualification"] = progress["evaluations"][f"block_{stage:06d}_full"]
                    path = directory / "champions" / f"champion_{26 + stage:06d}.pt"
                    atomic_checkpoint(path, promoted)
                    progress["champion"] = {"path": str(path.relative_to(root)), "sha256": sha(path),
                        "model_hash": item["model_hash"], "iteration": 26 + stage}; commit()
                    publish_current()
                    update(state="qualified", route_confirmed=True, champion_iteration=26 + stage)
            update(state="segment_complete", review_required=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user": update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__": main()
