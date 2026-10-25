"""Paired fixed-replay continuation: PPV control versus policy-only updates.

No production model promotion and no development data enters training.
Windows CPU workers import this entry point without importing Torch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def main():
    import copy
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, lock_held, utc_now
    from open_v4_experiment_evidence import teacher_deltas as select_teacher_deltas
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
    from junqi_open_v4.pipeline import source_contract, validation_metrics
    from junqi_open_v4.arena import run_arena, select_families, paired_interval
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_search_v3.recovery import make_policy_anchor
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    experiment_path = (root / args.experiment).resolve()
    experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    base = OpenConfig.load(root / experiment["parent_config"])
    original = root / base.run_directory
    directory = (root / experiment["directory"]).resolve()
    if not directory.is_relative_to(root) or directory.exists():
        raise ValueError("experiment requires a fresh directory inside alphazero")
    if (original / "supervision/STOP").exists():
        raise RuntimeError("STOP is present; do not start an optimization experiment")
    # The original writer locks also guard its GPU while a controlled experiment
    # is running. Another ordinary training launch will see ownership and exit.
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        directory.mkdir(parents=True)
        status_path = directory / "STATUS.json"
        status = {"schema": "junqi-phase-ablation-1", "state": "checking", "started_at": utc_now(),
            "experiment": str(experiment_path), "pid": __import__("os").getpid(), "updates": experiment["updates"]}
        def update(**values):
            status.update(values)
            status["updated_at"] = utc_now()
            atomic_json(status_path, status)
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        def respect_stop():
            if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; experiment checkpoint is preserved")
        try:
            current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
            contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
            if current["iteration"] != experiment["parent_iteration"] or current["contract_hash"] != sha(original / "CONTRACT.json"):
                raise ValueError("parent run changed since preregistration")
            if source_contract(root) != contract["source"] or source_contract(root) != experiment["source"]:
                raise ValueError("parent/experiment source mismatch")
            for role in ("learner", "actor", "champion"):
                if sha(original / current[role]["path"]) != current[role]["sha256"]:
                    raise ValueError("parent checkpoint digest mismatch")
            if current["learner"]["sha256"] != experiment["parent_checkpoint"]["sha256"]:
                raise ValueError("different parent learner")
            torch.set_num_threads(base.workers)
            device = torch.device(base.device)
            samples = []
            for row in experiment["replay"]:
                path = root / row["path"]
                if sha(path) != row["sha256"]:
                    raise ValueError("immutable training replay changed")
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if metadata["layout_role"] != "train" or metadata["config_hash"] != base.digest() or metadata["shard_sha256"] != row["sha256"]:
                    raise ValueError("replay was not from the production training split")
                part = read_selfplay_shard(path)
                for sample in part:
                    sample.observation = sample.observation.astype(np.float16)
                samples.extend(part)
            training = [s for s in samples if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 != 0]
            validation = [s for s in samples if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 == 0][:4096]
            if not training or not validation or len(training) > base.maximum_replay_samples:
                raise ValueError("unexpected fixed replay size or missing held-out training families")
            atomic_json(directory / "CONTRACT.json", experiment | {"training_samples": len(training), "validation_samples": len(validation),
                "tool_sha256": sha(Path(__file__)), "parent_current": current, "parent_contract": contract})
            rows, checkpoints, batch_digests = [], {}, []
            for variant in ("control", "policy"):
                respect_stop()
                learner = Learner(base, root)
                learner.resume(original / current["learner"]["path"])
                config = copy.deepcopy(base)
                config.run_directory = str((directory / variant).relative_to(root))
                config.initial_checkpoint = str((original / current["learner"]["path"]).relative_to(root))
                learner.config = config
                learner.provenance = {"full_parent_checkpoint": experiment["parent_checkpoint"],
                    "optimizer_reset": False, "scheduler_reset": False, "rng_reset": False,
                    "experiment_config_fork": "path and lineage only; optimizer, model, counters and RNG inherited",
                    "phase_schedule": "PPV" if variant == "control" else "policy only", "replay": experiment["replay"]}
                atomic_json(directory / variant / "config.json", config.as_dict())
                before = validation_metrics(learner.model, validation, device)
                logs = []
                digest = hashlib.sha256()
                started = time.perf_counter()
                update(state="training", variant=variant, completed_updates=0)
                for index in range(experiment["updates"]):
                    respect_stop()
                    phase = "value" if variant == "control" and learner.counters["attempted_updates"] % 3 == 2 else "policy"
                    anchor = make_policy_anchor(learner.model, training, device,
                        seed=base.seed ^ learner.counters["attempted_updates"], count=128) if phase == "value" else None
                    batch = learner._batch(training)
                    for sample in batch:
                        digest.update(str((sample.game_id, sample.state_hash, sample.player, sample.ply)).encode())
                    row = learner.step(batch, phase, anchor=anchor)
                    logs.append(row)
                    if (index + 1) % 64 == 0 or index + 1 == experiment["updates"]:
                        checkpoint = directory / variant / "learner.pt"
                        atomic_checkpoint(checkpoint, learner.payload(role="learner", lineage={
                            "parent": current, "experiment": str(experiment_path), "variant": variant,
                            "completed_experiment_updates": index + 1, "training_batch_digest": digest.hexdigest(),
                            "replay": experiment["replay"], "promotion": "none; diagnostic experiment only"}))
                        update(state="training", variant=variant, completed_updates=index + 1,
                            effective_updates=learner.counters["effective_updates"], checkpoint=str(checkpoint))
                        print(f"[ablation] variant={variant} updates={index+1}/{experiment['updates']} phase={phase} loss={row['loss']:.4f}", flush=True)
                learner.model.eval()
                after = validation_metrics(learner.model, validation, device)
                batch_digests.append(digest.hexdigest())
                checkpoints[variant] = {"path": str(checkpoint), "sha256": sha(checkpoint), "model_hash": model_hash(learner.model)}
                rows.append({"variant": variant, "before": before, "after": after, "training": logs,
                    "counters": dict(learner.counters), "training_seconds": time.perf_counter() - started,
                    "batch_digest": digest.hexdigest(), "checkpoint": checkpoints[variant]})
                atomic_json(directory / variant / "training.json", rows[-1])
                del learner
                torch.cuda.empty_cache()
            if len(set(batch_digests)) != 1:
                raise RuntimeError("paired branches did not see identical ordered training batches")
            respect_stop()
            frozen = {"parent": original / current["learner"]["path"], "champion": original / current["champion"]["path"],
                **{variant: Path(row["path"]) for variant, row in checkpoints.items()}}
            agents, provenance = {}, {}
            for name, path in frozen.items():
                payload = torch.load(path, map_location="cpu", weights_only=False)
                model = PolicyValueNet(base.model).to(device).eval()
                model.load_state_dict(payload["model"])
                model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
                agents[name] = BatchedNeuralPUCT(model, device=device, inference_batch_size=base.inference_batch_size,
                    cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree)
                provenance[name] = {"path": str(path), "sha256": sha(path), "model_hash": model_hash(model)}
            full_suite = json.loads((original / "development/suite.json").read_text(encoding="utf-8"))["states"]
            suite = select_families(full_suite, experiment["evaluation_families"], experiment["evaluation_seed"])
            matchups = [("parent", "champion"), ("control", "parent"), ("policy", "parent"), ("policy", "control")]
            matchups += [(name, f"teacher{teacher_id}") for teacher_id in range(len(base.teacher_candidates)) for name in ("parent", "control", "policy")]
            update(state="arena", variant=None, completed_updates=experiment["updates"], matchups=matchups)
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                submit = teachers.submit_actions
                def checked_submit(states, teacher_ids):
                    respect_stop()
                    return submit(states, teacher_ids)
                teachers.submit_actions = checked_submit
                arena = run_arena(suite, matchups, agents, teachers, {name: base.simulations for name in agents}, seed=experiment["evaluation_seed"])
                arena.update(checkpoints=provenance, teacher_provenance=teachers.provenance,
                    purpose="preregistered fixed-replay phase ablation; no production model promotion")
            atomic_json(directory / "arena.json", arena)
            # Select the experimental route only on prespecified paired evidence.
            # All teacher matchups have identical family units and both colors.
            duel = arena["summaries"][3]["family_score"]
            teacher_deltas = select_teacher_deltas(arena, "control", "policy")
            parent_deltas = {variant: select_teacher_deltas(arena, "parent", variant) for variant in ("control", "policy")}
            preferred = (duel["ci95"][0] > .5 and len(teacher_deltas) == 2
                and all(row["ci95"][0] > 0 for row in teacher_deltas)
                and all(row["ci95"][0] >= -base.teacher_sentinel_tolerance for row in parent_deltas["policy"]))
            result = {"schema": "junqi-phase-ablation-result-1", "experiment": experiment, "training": rows,
                "identical_ordered_training_batches": True, "arena_summaries": arena["summaries"],
                "policy_vs_control": duel, "policy_teacher_delta_vs_control": teacher_deltas,
                "teacher_delta_vs_parent": parent_deltas, "policy_route_preferred": bool(preferred),
                "promotion": "none; a preferred route still requires production-compatible lineage and existing full development gates",
                "interpretation": "Equal update budget; policy-only replaces Value updates with extra Policy updates. Shared encoder can still change Value predictions.",
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, policy_route_preferred=bool(preferred),
                result=str(directory / "RESULTS.json"), finished_at=utc_now())
            print(json.dumps({"policy_route_preferred": preferred, "duel": duel, "teacher_deltas": teacher_deltas}, ensure_ascii=False), flush=True)
        except BaseException as error:
            if status.get("state") != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
