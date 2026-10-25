from __future__ import annotations

import functools
import json
from pathlib import Path

def main():
    import torch
    import junqi_open_v4.games as games_module
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.games import TeacherLeague, collect_training_games
    from junqi_open_v4.layouts import MultiLayoutBackend, load_role
    from junqi_search_v3.dataset import atomic_json
    from junqi_search_v3.search import BatchedNeuralPUCT
    from junqi_az.model import PolicyValueNet

    root = Path(__file__).resolve().parents[1]
    config = OpenConfig.load(root / "config/open_junqi_v4_repaired.json")
    config.selfplay_fraction = 1.
    config.games_per_iteration = 16
    torch.set_num_threads(config.workers)
    device = torch.device("cuda")
    model = PolicyValueNet(config.model).to(device).eval()
    payload = torch.load(root / config.initial_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = payload.get("leaf_value_scale", 1.)
    backend = MultiLayoutBackend(load_role(root / config.layout_directory, "train"), config.seed)
    teachers = TeacherLeague(root, config.teacher_candidates, config.workers)
    rows, traces = [], []
    for enabled in (False, True):
        games_module.BatchedNeuralPUCT = functools.partial(BatchedNeuralPUCT, cuda_graphs=enabled)
        torch.cuda.reset_peak_memory_stats()
        samples, report = collect_training_games(config, model, device, backend, teachers, 18)
        report["cuda_graphs"] = enabled
        report["peak_gpu_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        report["peak_gpu_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
        rows.append(report)
        traces.append({game["index"]: game["trace"] for game in report["games"]})
        print("[selfplay-benchmark]", enabled, report["elapsed_seconds"], report["moves_per_second"], flush=True)
        del samples
    if traces[0] != traces[1]:
        raise RuntimeError("graph optimization changed deterministic neural selfplay trajectories")
    atomic_json(root / "reports/repair_20261003/selfplay_benchmark.json", {
        "scope": "16 complete neural selfplay games; identical layout pairs, weights, seeds and visits; no teacher moves",
        "trajectory_equivalence": True, "speedup": rows[0]["elapsed_seconds"] / rows[1]["elapsed_seconds"], "rows": rows})


if __name__ == "__main__":
    main()
