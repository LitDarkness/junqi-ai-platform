"""Frozen-weight Value search ablation, bare Policy, and a duplicate control."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


class ScaledSearch:
    def __init__(self, search, scale):
        self.search_impl = search
        self.scale = scale

    def __getattr__(self, name):
        return getattr(self.search_impl, name)

    def search(self, states, simulations, **kwargs):
        return self.search_impl.search(states, simulations, leaf_value_scale=self.scale, **kwargs)


def main():
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.learner import model_hash
    from junqi_open_v4.pipeline import source_contract
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
    if not directory.is_relative_to(root) or directory.exists():
        raise ValueError("Value ablation needs a fresh directory inside alphazero")
    if (original / "supervision/STOP").exists():
        raise RuntimeError("STOP requested; do not start an experiment")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        directory.mkdir(parents=True)
        status = dict(schema="junqi-value-search-ablation-1", state="checking", started_at=utc_now(),
            pid=os.getpid(), experiment=str(plan_path), updates=0, completed_updates=0)
        def update(**values):
            status.update(values)
            status["updated_at"] = utc_now()
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if (original / "supervision/STOP").exists() or (directory / "STOP").exists():
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; original model is preserved")
        sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
        update()
        try:
            current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
            contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
            if current != plan["parent_current"] or sha(original / "CONTRACT.json") != current["contract_hash"]:
                raise ValueError("parent changed after preregistration")
            if source_contract(root) != contract["source"] or contract["source"] != plan["source"]:
                raise ValueError("search source mismatch")
            for role in ("learner", "actor", "champion"):
                if sha(original / current[role]["path"]) != current[role]["sha256"]:
                    raise ValueError("parent checkpoint digest mismatch")
            torch.set_num_threads(config.workers)
            device = torch.device(config.device)
            payload = torch.load(original / current["learner"]["path"], map_location="cpu", weights_only=False)
            model = PolicyValueNet(config.model).to(device).eval()
            model.load_state_dict(payload["model"])
            model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
            if payload["leaf_value_scale"] != plan["baseline_scale"]:
                raise ValueError("baseline leaf scale changed after preregistration")
            if model_hash(model) != current["learner"]["model_hash"]:
                raise ValueError("model tensor digest mismatch")
            agents, budgets, variants = {}, {}, []
            for variant in plan["variants"]:
                name, scale, budget = variant["name"], variant["scale"], variant["simulations"]
                if name in agents or not 0 <= scale <= 1 or budget not in (0, config.simulations):
                    raise ValueError("unexpected search variant")
                search = BatchedNeuralPUCT(model, device=device, inference_batch_size=config.inference_batch_size,
                    cache_size=config.cache_size, leaves_per_tree=config.leaves_per_tree)
                agents[name] = ScaledSearch(search, scale)
                budgets[name] = budget
                variants.append(variant)
            if agents["baseline"].scale != payload["leaf_value_scale"] or agents["repeat"].scale != payload["leaf_value_scale"]:
                raise ValueError("duplicate controls must use the actual production Value scale")
            atomic_json(directory / "CONTRACT.json", plan | {"tool_sha256": sha(Path(__file__)),
                "checkpoint_leaf_value_scale": payload["leaf_value_scale"], "tensor_model_hash": model_hash(model),
                "weights_shared_and_immutable": True, "training_updates": 0,
                "terminal_values": "native true terminal values remain unscaled; only nonterminal network leaves are scaled"})
            respect_stop()
            suite = json.loads((original / "development/suite.json").read_text(encoding="utf-8"))["states"]
            suite = select_families(suite, plan["evaluation_families"], plan["evaluation_seed"])
            names = [variant["name"] for variant in variants]
            matchups = [(name, f"teacher{i}") for i in range(len(config.teacher_candidates)) for name in names]
            matchups += [(name, "baseline") for name in ("quarter", "zero", "bare")]
            update(state="arena", variants=variants, matchups=matchups)
            with TeacherLeague(root, config.teacher_candidates, config.workers) as teachers:
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                arena = run_arena(suite, matchups, agents, teachers, budgets, seed=plan["evaluation_seed"])
                arena.update(variants=variants, checkpoint=current["learner"], teacher_provenance=teachers.provenance,
                    purpose="prespecified frozen Value search diagnostics; no model promotion or source/configuration switch")
            atomic_json(directory / "arena.json", arena)
            if model_hash(model) != current["learner"]["model_hash"]:
                raise RuntimeError("evaluation mutated frozen model weights")
            deltas = {name: teacher_deltas(arena, "baseline", name) for name in names if name != "baseline"}
            aa = deltas["repeat"]
            primary = deltas["zero"]
            zero_duel = next(row["family_score"] for row in arena["summaries"] if row["candidate"] == "zero" and row["opponent"] == "baseline")
            reliable = len(aa) == 2 and all(abs(row["mean"]) <= plan["duplicate_control_tolerance"] for row in aa)
            preferred = reliable and len(primary) == 2 and all(row["ci95"][0] > 0 for row in primary) and zero_duel["ci95"][0] > .5
            result = {"schema": "junqi-value-search-ablation-result-1", "experiment": plan,
                "summaries": arena["summaries"], "teacher_deltas_vs_baseline": deltas,
                "duplicate_control_within_tolerance": reliable, "primary_zero_value_preferred": bool(preferred),
                "primary_zero_value_duel": zero_duel, "weights_unchanged": True, "training_updates": 0,
                "promotion": "none; secondary variants exploratory, primary route requires full development confirmation",
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20}
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, primary_zero_value_preferred=bool(preferred),
                duplicate_control_within_tolerance=reliable, result=str(directory / "RESULTS.json"))
            print(json.dumps({"primary_zero_value_preferred": preferred, "duplicate_control_within_tolerance": reliable,
                "zero_value_teacher_deltas": primary}, ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
