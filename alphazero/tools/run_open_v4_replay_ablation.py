"""One-factor replay-window experiment reusing the frozen PPV control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path


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
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.loop import read_selfplay_shard
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = (root / args.experiment).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    base = OpenConfig.load(root / plan["parent_config"])
    original = root / base.run_directory
    directory = (root / plan["directory"]).resolve()
    if not directory.is_relative_to(root) or directory.exists():
        raise ValueError("replay experiment needs a fresh directory within alphazero")
    if (original / "supervision/STOP").exists():
        raise RuntimeError("user STOP is present")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        directory.mkdir(parents=True)
        status = dict(schema="junqi-replay-ablation-1", state="checking", pid=os.getpid(), started_at=utc_now(),
            experiment=str(plan_path), updates=plan["updates"], variant="wide", completed_updates=0)
        def update(**values):
            status.update(values)
            status["updated_at"] = utc_now()
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; saved experimental checkpoints are preserved")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        update()
        try:
            current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
            contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
            if current != plan["parent_current"] or sha(original / "CONTRACT.json") != current["contract_hash"]:
                raise ValueError("parent state changed after preregistration")
            if source_contract(root) != contract["source"] or contract["source"] != plan["source"]:
                raise ValueError("training source mismatch")
            for role in ("learner", "actor", "champion"):
                if sha(original / current[role]["path"]) != current[role]["sha256"]:
                    raise ValueError("parent checkpoint mismatch")
            control_path = root / plan["control_checkpoint"]["path"]
            if sha(control_path) != plan["control_checkpoint"]["sha256"]:
                raise ValueError("frozen small-window control changed")
            prior = json.loads((root / plan["control_contract"]).read_text(encoding="utf-8"))
            if (prior["parent_current"] != current or prior["updates"] != plan["updates"]
                    or prior["source"] != plan["source"] or prior["evaluation_seed"] != plan["evaluation_seed"]):
                raise ValueError("reused control has a different parent, budget, source or evaluation")
            torch.set_num_threads(base.workers)
            device = torch.device(base.device)
            samples, common_validation = [], []
            for row in plan["replay"]:
                path = root / row["path"]
                if sha(path) != row["sha256"]:
                    raise ValueError("fixed replay changed")
                metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
                if metadata["layout_role"] != "train" or metadata["config_hash"] != base.digest() or metadata["shard_sha256"] != row["sha256"]:
                    raise ValueError("replay split or provenance mismatch")
                part = read_selfplay_shard(path)
                for sample in part:
                    sample.observation = sample.observation.astype(np.float16)
                samples.extend(part)
                if row["iteration"] in plan["common_holdout_iterations"]:
                    common_validation.extend(s for s in part if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 == 0)
            training = [s for s in samples if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 != 0]
            common_validation = common_validation[:4096]
            if len(training) > base.maximum_replay_samples or not common_validation:
                raise ValueError("replay size/holdout mismatch")
            respect_stop()
            learner = Learner(base, root)
            learner.resume(original / current["learner"]["path"])
            cfg = copy.deepcopy(base)
            cfg.replay_iterations = plan["replay_iterations"]
            cfg.run_directory = str(directory.relative_to(root))
            cfg.initial_checkpoint = str((original / current["learner"]["path"]).relative_to(root))
            learner.config = cfg
            learner.provenance = {"full_parent_checkpoint": current["learner"], "optimizer_reset": False,
                "scheduler_reset": False, "rng_reset": False, "phase_schedule": "PPV unchanged",
                "changed_factor": "fixed replay window from 3 to 12 iterations", "replay": plan["replay"]}
            atomic_json(directory / "config.json", cfg.as_dict())
            atomic_json(directory / "CONTRACT.json", plan | {"tool_sha256": sha(Path(__file__)),
                "training_samples": len(training), "common_holdout_samples": len(common_validation),
                "effective_config_hash": cfg.digest()})
            before = validation_metrics(learner.model, common_validation, learner.device)
            logs = []
            started = time.perf_counter()
            for completed in range(0, plan["updates"], 64):
                respect_stop()
                logs.extend(learner.train(training, updates=min(64, plan["updates"] - completed)))
                saved = completed + min(64, plan["updates"] - completed)
                path = directory / "learner.pt"
                atomic_checkpoint(path, learner.payload(role="learner", lineage={"parent": current,
                    "experiment": str(plan_path), "completed_experiment_updates": saved,
                    "replay": plan["replay"], "promotion": "none; diagnostic branch"}))
                update(state="training", completed_updates=saved, effective_updates=learner.counters["effective_updates"], checkpoint=str(path))
            after = validation_metrics(learner.model, common_validation, learner.device)
            atomic_json(directory / "training.json", {"before": before, "after": after, "training": logs,
                "counters": dict(learner.counters), "training_seconds": time.perf_counter() - started,
                "parent": current["learner"], "checkpoint_sha256": sha(path), "training_samples": len(training)})
            del learner
            torch.cuda.empty_cache()
            respect_stop()
            agents, provenance = {}, {}
            frozen = {"parent": original / current["learner"]["path"], "control": control_path, "wide": path}
            for name, checkpoint in frozen.items():
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = PolicyValueNet(base.model).to(device).eval()
                model.load_state_dict(payload["model"])
                model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
                agents[name] = BatchedNeuralPUCT(model, device=device, inference_batch_size=base.inference_batch_size,
                    cache_size=base.cache_size, leaves_per_tree=base.leaves_per_tree)
                provenance[name] = {"path": str(checkpoint), "sha256": sha(checkpoint), "model_hash": model_hash(model)}
            suite = json.loads((original / "development/suite.json").read_text(encoding="utf-8"))["states"]
            suite = select_families(suite, plan["evaluation_families"], plan["evaluation_seed"])
            matchups = [("wide", "parent"), ("wide", "control")]
            matchups += [(name, f"teacher{i}") for i in range(len(base.teacher_candidates)) for name in ("parent", "control", "wide")]
            update(state="arena", matchups=matchups)
            with TeacherLeague(root, base.teacher_candidates, base.workers) as teachers:
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                arena = run_arena(suite, matchups, agents, teachers, {name: base.simulations for name in agents}, seed=plan["evaluation_seed"])
                arena.update(checkpoints=provenance, teacher_provenance=teachers.provenance,
                    purpose="single-factor fixed replay-window comparison; no production promotion")
            atomic_json(directory / "arena.json", arena)
            vs_control = teacher_deltas(arena, "control", "wide")
            vs_parent = teacher_deltas(arena, "parent", "wide")
            duel = arena["summaries"][1]["family_score"]
            preferred = (duel["ci95"][0] > .5 and len(vs_control) == len(vs_parent) == 2
                and all(row["ci95"][0] > 0 for row in vs_control + vs_parent))
            result = {"schema": "junqi-replay-ablation-result-1", "experiment": plan, "common_holdout_before": before,
                "common_holdout_after": after, "arena_summaries": arena["summaries"], "wide_vs_control": duel,
                "teacher_delta_vs_control": vs_control, "teacher_delta_vs_parent": vs_parent,
                "wide_route_preferred": bool(preferred), "promotion": "none; original full development gates still apply",
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, wide_route_preferred=bool(preferred), result=str(directory / "RESULTS.json"))
            print(json.dumps({"wide_route_preferred": preferred, "teacher_delta_vs_parent": vs_parent}, ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
