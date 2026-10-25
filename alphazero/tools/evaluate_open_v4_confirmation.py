"""Full development recheck of a frozen candidate; no training or promotion."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def route_confirmed(parent_deltas, control_deltas, duplicate_deltas, gate, minimum_families, tolerance, duels):
    def enough(rows):
        return len(rows) == 2 and all(row["families"] >= minimum_families for row in rows)
    return (enough(parent_deltas) and enough(control_deltas) and enough(duplicate_deltas)
        and all(row["ci95"][0] > 0 for row in parent_deltas + control_deltas)
        and all(abs(row["mean"]) <= tolerance for row in duplicate_deltas)
        and enough(duels) and all(row["ci95"][0] > .5 for row in duels)
        and gate["champion_accepted"])


def main():
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import model_hash
    from junqi_open_v4.pipeline import source_contract, promotion_decision
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_path = (root / args.experiment).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    config = OpenConfig.load(root / plan["parent_config"])
    original = root / config.run_directory
    directory = (root / plan["directory"]).resolve()
    if not plan_path.is_relative_to(root) or not directory.is_relative_to(root) or directory == root or directory.exists():
        raise ValueError("confirmation needs a fresh directory within alphazero")
    if (original / "supervision/STOP").exists():
        raise RuntimeError("user STOP present")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        directory.mkdir(parents=True)
        status = dict(schema="junqi-frozen-confirmation-1", state="checking", pid=os.getpid(),
            started_at=utc_now(), experiment=str(plan_path), updates=0, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; frozen checkpoints preserved")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        update()
        try:
            current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
            contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
            if current != plan["parent_current"] or sha(original / "CONTRACT.json") != current["contract_hash"]:
                raise ValueError("parent changed since preregistration")
            if source_contract(root) != contract["source"] or contract["source"] != plan["source"]:
                raise ValueError("evaluation source changed")
            for role in ("learner", "actor", "champion"):
                if sha(original / current[role]["path"]) != current[role]["sha256"]:
                    raise ValueError("original role checkpoint mismatch")
            if plan["evaluation_families"] != config.champion_minimum_family_count:
                raise ValueError("full existing development family count required")
            suite_path = root / plan["suite"]["path"]
            if sha(suite_path) != plan["suite"]["sha256"]:
                raise ValueError("development suite changed")
            suite_payload = json.loads(suite_path.read_text(encoding="utf-8"))
            if suite_payload["contract"]["role"] != "dev" or suite_payload["contract"]["layout_manifest_hash"] != contract["layouts"]:
                raise ValueError("wrong development layout contract")
            suite = select_families(suite_payload["states"], plan["evaluation_families"], plan["evaluation_seed"])
            if len({s["family"] for s in suite}) != plan["evaluation_families"]:
                raise ValueError("incomplete development suite")
            atomic_json(directory / "CONTRACT.json", plan | {"tool_sha256": sha(Path(__file__)), "plan_sha256": sha(plan_path)})
            torch.set_num_threads(config.workers)
            device = torch.device(config.device)
            agents, models = {}, {}
            for name, participant in plan["participants"].items():
                checkpoint = (root / participant["path"]).resolve()
                if not checkpoint.is_relative_to(root) or sha(checkpoint) != participant["sha256"]:
                    raise ValueError("frozen participant digest mismatch")
                payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
                model = PolicyValueNet(config.model).to(device).eval()
                model.load_state_dict(payload["model"])
                model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
                if model_hash(model) != participant["model_hash"]:
                    raise ValueError("frozen tensor mismatch")
                models[name] = model
                agents[name] = BatchedNeuralPUCT(model, device=device, inference_batch_size=config.inference_batch_size,
                    cache_size=config.cache_size, leaves_per_tree=config.leaves_per_tree)
            matchups = [("candidate", "champion"), ("candidate", "parent"), ("candidate", "control")]
            matchups += [(name, f"teacher{i}") for i in range(len(config.teacher_candidates)) for name in plan["participants"]]
            update(state="arena", evaluation_families=plan["evaluation_families"], starts=len(suite), matchups=matchups)
            with TeacherLeague(root, config.teacher_candidates, config.workers) as teachers:
                if teachers.provenance != contract["teachers"]:
                    raise ValueError("frozen teacher provenance mismatch")
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                respect_stop()
                arena = run_arena(suite, matchups, agents, teachers, {name: config.simulations for name in agents}, seed=plan["evaluation_seed"])
                arena.update(checkpoints=plan["participants"], teacher_provenance=teachers.provenance,
                    purpose=plan["purpose"], caution=plan["development_reuse_caveat"])
            if any(model_hash(model) != plan["participants"][name]["model_hash"] for name, model in models.items()):
                raise RuntimeError("evaluation changed frozen weights")
            atomic_json(directory / "arena.json", arena)
            gate = promotion_decision(arena, config)
            deltas = {key: teacher_deltas(arena, before, after) for key, before, after in (
                ("candidate_vs_parent", "parent", "candidate"), ("candidate_vs_control", "control", "candidate"),
                ("candidate_vs_champion", "champion", "candidate"), ("duplicate", "parent", "parent_repeat"))}
            confirmed = route_confirmed(deltas["candidate_vs_parent"], deltas["candidate_vs_control"],
                deltas["duplicate"], gate, plan["evaluation_families"], plan["duplicate_control_tolerance"],
                [arena["summaries"][i]["family_score"] for i in (1, 2)])
            unchanged = json.loads((original / "CURRENT.json").read_text(encoding="utf-8")) == current
            if not unchanged:
                raise RuntimeError("original roles changed during confirmation")
            result = dict(schema="junqi-frozen-confirmation-result-1", experiment=plan, summaries=arena["summaries"],
                teacher_deltas=deltas, existing_gate=gate, fresh_route_confirmed=bool(confirmed),
                original_roles_unchanged=unchanged, weights_unchanged=True, training_updates=0,
                promotion="none; evidence review required before a documented continuation fork",
                peak_gpu_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, fresh_route_confirmed=bool(confirmed),
                existing_gate=gate, result=str(directory / "RESULTS.json"))
            print(json.dumps(dict(fresh_route_confirmed=bool(confirmed), existing_gate=gate), ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
