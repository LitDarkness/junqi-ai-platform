"""Paired frozen-champion search budget diagnosis; never promote or train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path


def validated_budgets(variants, production_budget):
    budgets = {}
    for row in variants:
        if set(row) != {"name", "simulations"}:
            raise ValueError("only simulation budget may vary")
        name, value = row["name"], row["simulations"]
        if name in budgets or type(value) is not int or value not in (32, 64, 128, 256):
            raise ValueError("invalid or duplicate budget variant")
        budgets[name] = value
    expected = {"baseline": production_budget, "repeat": production_budget,
                "sim32": 32, "sim128": 128, "sim256": 256}
    if budgets != expected:
        raise ValueError("production duplicate and declared diagnostic budgets required")
    return budgets


def primary_signal(deltas, duplicate, duel, families, tolerance):
    def enough(rows):
        return len(rows) == 2 and all(row["families"] >= families for row in rows)
    reliable = enough(duplicate) and all(abs(row["mean"]) <= tolerance for row in duplicate)
    signal = (reliable and enough(deltas) and all(row["ci95"][0] > 0 for row in deltas)
              and duel["families"] >= families and duel["ci95"][0] > .5)
    return bool(reliable), bool(signal)


class StopAwareSearch:
    def __init__(self, search, check):
        self.search_impl, self.check = search, check

    def __getattr__(self, name):
        return getattr(self.search_impl, name)

    def search(self, states, simulations, **kwargs):
        self.check()
        return self.search_impl.search(states, simulations, **kwargs)

    def bare(self, states):
        self.check()
        return self.search_impl.bare(states)


def main():
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
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
    def inside(relative):
        path = (root / relative).resolve()
        if not path.is_relative_to(root) or path == root:
            raise ValueError("all experiment paths must stay in alphazero")
        return path
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    plan_path = inside(args.experiment)
    plan = read_json(plan_path)
    config = OpenConfig.load(inside(plan["config"]))
    budgets = validated_budgets(plan["variants"], config.simulations)
    directory = inside(plan["directory"])
    frozen_runs = [(inside(row["directory"]), row) for row in plan["frozen_runs"]]
    if len({path for path, _ in frozen_runs}) != len(frozen_runs):
        raise ValueError("duplicate frozen run")
    stop_paths = [inside(path) for path in plan["stop_paths"]] + [directory / "STOP"]
    if any(path.exists() for path in stop_paths):
        raise RuntimeError("STOP requested; no automatic clearing")
    with ExitStack() as locks:
        for run, _ in frozen_runs:
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
        if directory.exists():
            raise ValueError("preserve existing experiment; inspect before explicit recovery")
        directory.mkdir(parents=True)
        status = dict(schema="junqi-frozen-search-budget-1", state="checking", pid=os.getpid(),
                      started_at=utc_now(), experiment=str(plan_path), updates=0, completed_updates=0)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(directory / "STATUS.json", status)
        def respect_stop():
            if any(path.exists() for path in stop_paths):
                update(state="stopped_by_user")
                raise RuntimeError("STOP requested; frozen checkpoints preserved")
        update()
        try:
            respect_stop()
            for run, frozen in frozen_runs:
                if read_json(run / "CURRENT.json") != frozen["current"]:
                    raise ValueError("frozen run changed after registration")
                contract = read_json(run / "CONTRACT.json")
                if sha(run / "CONTRACT.json") != frozen["current"]["contract_hash"]:
                    raise ValueError("run contract digest mismatch")
                if contract["source"] != source_contract(root) or contract["source"] != plan["source"]:
                    raise ValueError("search source mismatch")
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
                raise ValueError("insufficient development families")
            checkpoint = inside(plan["checkpoint"]["path"])
            if sha(checkpoint) != plan["checkpoint"]["sha256"]:
                raise ValueError("qualified champion changed")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload["leaf_value_scale"] != plan["leaf_value_scale"]:
                raise ValueError("production Value scale changed")
            atomic_json(directory / "CONTRACT.json", plan | {"tool_sha256": sha(Path(__file__)),
                "plan_sha256": sha(plan_path), "weights_shared_and_immutable": True})
            torch.set_num_threads(config.workers)
            device = torch.device(config.device)
            model = PolicyValueNet(config.model).to(device).eval()
            model.load_state_dict(payload["model"])
            model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
            if model_hash(model) != plan["checkpoint"]["model_hash"]:
                raise ValueError("qualified champion tensor digest mismatch")
            agents = {name: StopAwareSearch(BatchedNeuralPUCT(model, device=device,
                inference_batch_size=config.inference_batch_size, cache_size=config.cache_size,
                leaves_per_tree=config.leaves_per_tree), respect_stop) for name in budgets}
            matchups = [(name, f"teacher{i}") for i in range(len(config.teacher_candidates)) for name in budgets]
            matchups += [(name, "baseline") for name in ("sim32", "sim128", "sim256")]
            update(state="arena", evaluation_families=plan["evaluation_families"], starts=len(suite),
                   variants=plan["variants"], matchups=matchups, planned_games=2 * len(suite) * len(matchups))
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
                raise RuntimeError("evaluation mutated frozen weights")
            if any(read_json(run / "CURRENT.json") != row["current"] for run, row in frozen_runs):
                raise RuntimeError("frozen roles changed during evaluation")
            arena.update(variants=plan["variants"], checkpoint=plan["checkpoint"], teacher_provenance=plan["teachers"],
                         purpose=plan["hypothesis"], caveat=plan["development_reuse_caveat"])
            atomic_json(directory / "arena.json", arena)
            deltas = {name: teacher_deltas(arena, "baseline", name) for name in budgets if name != "baseline"}
            duel = next(row["family_score"] for row in arena["summaries"] if row["candidate"] == "sim128" and row["opponent"] == "baseline")
            reliable, signal = primary_signal(deltas["sim128"], deltas["repeat"], duel,
                                             plan["evaluation_families"], plan["duplicate_control_tolerance"])
            result = dict(schema="junqi-frozen-search-budget-result-1", experiment=plan,
                summaries=arena["summaries"], teacher_deltas_vs_baseline=deltas,
                teacher_deltas_256_vs_128=teacher_deltas(arena, "sim128", "sim256"),
                primary_128_vs_64_duel=duel, duplicate_control_within_tolerance=reliable,
                primary_128_signal=signal, weights_unchanged=True, frozen_roles_unchanged=True,
                training_updates=0, promotion="none; any favorable search route needs full development confirmation",
                search_metrics=arena["search_metrics"], arena_seconds=arena["elapsed_seconds"],
                evaluation_replay_samples=0, peak_gpu_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
            atomic_json(directory / "RESULTS.json", result)
            update(state="review_due", review_required=True, primary_128_signal=signal,
                   duplicate_control_within_tolerance=reliable, result=str(directory / "RESULTS.json"))
            print(json.dumps(dict(primary_128_signal=signal, duplicate_control_within_tolerance=reliable), ensure_ascii=False), flush=True)
        except BaseException as error:
            if status["state"] != "stopped_by_user":
                update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
