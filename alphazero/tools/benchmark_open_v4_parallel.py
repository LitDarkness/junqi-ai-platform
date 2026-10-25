"""Frozen training-layout throughput and development-arena concurrency checks."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from datetime import datetime
from pathlib import Path


def worker_info():
    import os
    return {"pid": os.getpid(), "torch_loaded": "torch" in sys.modules}


def baseline_module(root, name):
    spec = importlib.util.spec_from_file_location("junqi_open_v4._parallel_baseline_" + name,
        root / ("reports/parallel_20261003/before_" + name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    import numpy as np
    import torch
    import psutil
    from junqi_az.model import PolicyValueNet
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.games import collect_training_games
    from junqi_open_v4.teachers import TeacherLeague
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_open_v4.arena import run_arena, select_families
    from junqi_search_v3.dataset import atomic_json
    from junqi_search_v3.search import BatchedNeuralPUCT

    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["cpu", "collect", "arena"], required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--output-directory", help="Optional output directory; default creates a fresh rerun directory")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = (root / args.output_directory).resolve() if args.output_directory else root / "reports/parallel_20261003" / (
        "rerun_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + args.part)
    output.mkdir(parents=True, exist_ok=True)
    config = OpenConfig.load(root / "config/open_junqi_v4_repair_pilot.json")
    old_games = baseline_module(root, "games")
    old_arena = baseline_module(root, "arena")
    backend = MultiLayoutBackend(load_role(root / config.layout_directory, "train"), config.seed)
    rows = []
    if args.part == "cpu":
        states = [backend.new_state(config.seed + i // 2) for i in range(128)]
        rng = np.random.default_rng(719)
        for i, state in enumerate(states):
            for ply in range(i % 48):
                if state.is_terminal():
                    break
                state.apply(int(rng.choice(state.legal_actions())))
        states = [state for state in states if not state.is_terminal()]
        ids = [i % 2 for i in range(len(states))]
        references = []
        for workers in (0, 4, 8, 12):
            teachers = TeacherLeague(root, config.teacher_candidates, workers) if workers else old_games.TeacherLeague(root, config.teacher_candidates, config.workers)
            # Warm persistent workers; startup is separately visible in full games.
            teachers.actions(states[:24], ids[:24])
            repetitions, actions = [], []
            for unused in range(3):
                started = time.perf_counter()
                actions.append(teachers.actions(states, ids))
                repetitions.append(time.perf_counter() - started)
            process = psutil.Process()
            info = []
            if workers:
                info = [f.result() for f in [teachers.executor.submit(worker_info) for _ in range(workers * 2)]]
            row = {"execution": "processes" if workers else "legacy_threads", "workers": workers or config.workers,
                "seconds": repetitions, "mean_seconds": float(np.mean(repetitions)), "actions": actions,
                "worker_info": info, "process_tree_rss_mib": sum(p.memory_info().rss for p in [process] + process.children(recursive=True)) / 2**20,
                "within_mode_action_differences": [sum(a != b for a,b in zip(actions[0], v)) for v in actions[1:]]}
            if not workers:
                references = actions[0]
            row["differences_from_legacy"] = [sum(a != b for a,b in zip(references, v)) for v in actions]
            rows.append(row)
            teachers.executor.shutdown(wait=True)
            print("[cpu-benchmark]", row["execution"], row["workers"], row["mean_seconds"], row["differences_from_legacy"], flush=True)
        atomic_json(output / "cpu.json", {"scope": "training layouts only; unchanged teacher sources and budgets", "states": len(states), "rows": rows})
        return

    torch.set_num_threads(config.workers)
    device = torch.device("cuda")
    current = json.loads((root / config.run_directory / "CURRENT.json").read_text())
    path = root / config.run_directory / current["actor"]["path"]
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyValueNet(config.model).to(device).eval()
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = payload.get("leaf_value_scale", 1.)
    common = {"checkpoint": str(path), "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "model_hash": current["actor"]["model_hash"], "config_hash": config.digest(), "iteration": 7}
    for enabled in (False, True):
        teachers = TeacherLeague(root, config.teacher_candidates, args.workers) if enabled else old_games.TeacherLeague(root, config.teacher_candidates, config.workers)
        torch.cuda.reset_peak_memory_stats()
        if args.part == "collect":
            collect = collect_training_games if enabled else old_games.collect_training_games
            samples, report = collect(config, model, device, backend, teachers, 7, count=args.games)
            digest = hashlib.sha256()
            for sample in samples:
                digest.update(str((sample.game_id, sample.state_hash, sample.player, sample.ply)).encode())
                for array in (sample.observation, sample.legal_actions, sample.repetition_contexts, sample.visits, sample.raw_visits, sample.wdl):
                    digest.update(array.tobytes())
            report["samples_digest"] = digest.hexdigest()
            del samples
        else:
            suite = json.loads((root / config.run_directory / "development/suite.json").read_text())["states"]
            suite = select_families(suite, 8, config.seed ^ 0xA2E)
            agents = {"actor": BatchedNeuralPUCT(model, device=device)}
            arena = run_arena if enabled else old_arena.run_arena
            report = arena(suite, [("actor", "teacher0"), ("actor", "teacher1")], agents, teachers,
                {"actor": config.simulations}, seed=config.seed ^ 0xEAA)
        report["parallel"] = enabled
        report["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        process = psutil.Process()
        report["process_tree_rss_mib"] = sum(p.memory_info().rss for p in [process] + process.children(recursive=True)) / 2**20
        teachers.executor.shutdown(wait=True)
        rows.append(report)
        atomic_json(output / (args.part + "_" + ("parallel" if enabled else "baseline") + ".json"), common | {"report": report})
        print("[parallel-benchmark]", args.part, enabled, report["elapsed_seconds"], flush=True)
    def trajectories(report):
        key = (lambda g: g["index"]) if args.part == "collect" else (lambda g: (g["position"], g["matchup"], g["candidate_side"]))
        return {key(game): game["trace"] for game in report["games"]}
    first, second = map(trajectories, rows)
    comparison = common | {"speedup": rows[0]["elapsed_seconds"] / rows[1]["elapsed_seconds"],
        "trajectory_differences": sum(first[k] != second[k] for k in first), "games": len(first),
        "sample_equivalence": rows[0].get("samples_digest") == rows[1].get("samples_digest") if args.part == "collect" else None}
    atomic_json(output / (args.part + "_comparison.json"), comparison)
    print(json.dumps(comparison), flush=True)


if __name__ == "__main__":
    main()
