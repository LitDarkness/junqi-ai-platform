"""Full development recheck of exploratory 256 versus production 64 search."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

from evaluate_open_v4_search_budgets import StopAwareSearch, primary_signal


def full_budget_confirmed(deltas, duplicate, duel, gate, families, tolerance):
    if families < 64:
        return False
    reliable, signal = primary_signal(deltas, duplicate, duel, families, tolerance)
    return reliable and signal and gate["champion_accepted"]


def main():
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
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
    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("all paths must stay in alphazero")
        return path
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    plan_path = inside(args.experiment)
    plan = read_json(plan_path)
    config = OpenConfig.load(inside(plan["config"]))
    budgets = {"baseline": 64, "repeat": 64, "candidate": 256}
    if config.digest() != plan["config_hash"] or config.simulations != 64 or plan["budgets"] != budgets:
        raise ValueError("fixed production config and 64/64/256 budgets required")
    if plan["evaluation_families"] != config.champion_minimum_family_count or plan["evaluation_families"] < 64:
        raise ValueError("full existing development family count required")
    directory = inside(plan["directory"])
    frozen_runs = [(inside(row["directory"]), row) for row in plan["frozen_runs"]]
    if len({path for path, _ in frozen_runs}) != len(frozen_runs):
        raise ValueError("duplicate frozen runs")
    stops = [inside(p) for p in plan["stop_paths"]] + [directory / "STOP"]
    if any(path.exists() for path in stops):
        raise RuntimeError("STOP requested; no automatic clearing")
    with ExitStack() as locks:
        for run, _ in frozen_runs:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
        if directory.exists():
            raise ValueError("preserve existing recheck; inspect before explicit recovery")
        directory.mkdir(parents=True)
        status = dict(schema="junqi-search-budget-confirmation-1", state="checking", pid=os.getpid(),
                      started_at=utc_now(), experiment=str(plan_path), updates=0, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if any(path.exists() for path in stops):
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; frozen states preserved")
        update()
        try:
            respect_stop()
            for path, digest in plan["evidence_files"].items():
                if sha(inside(path)) != digest:
                    raise ValueError("screening evidence or dependency digest changed")
            for run, frozen in frozen_runs:
                if read_json(run / "CURRENT.json") != frozen["current"]:
                    raise ValueError("frozen run changed after registration")
                contract = read_json(run / "CONTRACT.json")
                if sha(run / "CONTRACT.json") != frozen["current"]["contract_hash"]:
                    raise ValueError("run contract digest mismatch")
                if contract["source"] != source_contract(root) or contract["source"] != plan["source"]:
                    raise ValueError("evaluation source mismatch")
                for role in ("learner", "actor", "champion"):
                    record = frozen["current"][role]
                    if sha(run / record["path"]) != record["sha256"]:
                        raise ValueError("frozen checkpoint digest mismatch")
            suite_path = inside(plan["suite"]["path"])
            if sha(suite_path) != plan["suite"]["sha256"]:
                raise ValueError("development suite changed")
            suite_payload = read_json(suite_path)
            if suite_payload["contract"]["role"] != "dev" or suite_payload["contract"]["layout_manifest_hash"] != plan["layout_manifest_hash"]:
                raise ValueError("development layout provenance mismatch")
            if hashlib.sha256(json.dumps(suite_payload["states"], sort_keys=True).encode()).hexdigest() != suite_payload["states_hash"]:
                raise ValueError("development states digest mismatch")
            suite = select_families(suite_payload["states"], plan["evaluation_families"], plan["evaluation_seed"])
            if len({row["family"] for row in suite}) != plan["evaluation_families"]:
                raise ValueError("incomplete development family coverage")
            checkpoint = inside(plan["checkpoint"]["path"])
            if sha(checkpoint) != plan["checkpoint"]["sha256"]:
                raise ValueError("champion digest mismatch")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["leaf_value_scale"] != plan["leaf_value_scale"]:
                raise ValueError("production Value scale changed")
            atomic_json(directory / "CONTRACT.json", plan | {"tool_sha256": sha(Path(__file__)), "plan_sha256": sha(plan_path)})
            torch.set_num_threads(config.workers)
            device = torch.device(config.device)
            model = PolicyValueNet(config.model).to(device).eval()
            model.load_state_dict(payload["model"])
            model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
            if model_hash(model) != plan["checkpoint"]["model_hash"]:
                raise ValueError("frozen tensor digest mismatch")
            agents = {name: StopAwareSearch(BatchedNeuralPUCT(model, device=device,
                inference_batch_size=config.inference_batch_size, cache_size=config.cache_size,
                leaves_per_tree=config.leaves_per_tree), respect_stop) for name in budgets}
            # Keep the direct duel first for the unchanged production gate.
            matchups = [("candidate", "baseline")]
            matchups += [(name, f"teacher{i}") for i in range(len(config.teacher_candidates)) for name in budgets]
            update(state="arena", evaluation_families=plan["evaluation_families"], starts=len(suite),
                   budgets=budgets, matchups=matchups, planned_games=2 * len(suite) * len(matchups))
            with TeacherLeague(root, config.teacher_candidates, config.workers) as teachers:
                if teachers.provenance != plan["teachers"]:
                    raise ValueError("frozen teacher provenance mismatch")
                submit = teachers.submit_actions
                def checked_submit(states, ids):
                    respect_stop()
                    return submit(states, ids)
                teachers.submit_actions = checked_submit
                arena = run_arena(suite, matchups, agents, teachers, budgets, seed=plan["evaluation_seed"])
            respect_stop()
            if model_hash(model) != plan["checkpoint"]["model_hash"]:
                raise RuntimeError("evaluation mutated frozen model")
            if any(read_json(run / "CURRENT.json") != row["current"] for run, row in frozen_runs):
                raise RuntimeError("frozen model roles changed during recheck")
            if any(sha(inside(path)) != digest for path, digest in plan["evidence_files"].items()):
                raise RuntimeError("original screening evidence changed")
            arena.update(checkpoint=plan["checkpoint"], budgets=budgets, teacher_provenance=plan["teachers"],
                         purpose=plan["purpose"], caution=plan["development_reuse_caveat"])
            atomic_json(directory / "arena.json", arena)
            gain = teacher_deltas(arena, "baseline", "candidate")
            duplicate = teacher_deltas(arena, "baseline", "repeat")
            gate = promotion_decision(arena, config)
            confirmed = full_budget_confirmed(gain, duplicate, arena["summaries"][0]["family_score"],
                                              gate, plan["evaluation_families"], plan["duplicate_control_tolerance"])
            result = dict(schema="junqi-search-budget-confirmation-result-1", experiment=plan,
                summaries=arena["summaries"], teacher_gain_256_vs_64=gain, duplicate_control=duplicate,
                existing_gate=gate, search_256_confirmed=bool(confirmed), frozen_weights_unchanged=True,
                frozen_roles_unchanged=True, original_primary_128_outcome_preserved=True,
                training_updates=0, promotion="none; review before a recorded independent runtime or training branch",
                search_metrics=arena["search_metrics"], arena_seconds=arena["elapsed_seconds"],
                evaluation_replay_samples=0, peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, search_256_confirmed=bool(confirmed),
                   existing_gate=gate, result=str(directory / "RESULTS.json"))
            print(json.dumps(dict(search_256_confirmed=bool(confirmed), existing_gate=gate), ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
