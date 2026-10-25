from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout
from junqi_search_v3.config import SearchV3Config
from junqi_search_v3.search import BatchedNeuralPUCT

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "repaired", "bulk4", "cache", "graphs"], required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    torch.set_num_threads(12)
    config = SearchV3Config.load(ROOT / "config/open_junqi_search_v3_strength_late_stage.json")
    model = PolicyValueNet(config.model).cuda().eval()
    checkpoint = ROOT / "runs/open_junqi_search_v3_strength_continuation/checkpoints/loop_iteration_000150.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = payload.get("leaf_value_scale", 1.)
    search_class = BatchedNeuralPUCT
    kwargs = {"cache_size": 8192 if args.mode == "cache" else 0,
              "leaves_per_tree": 4 if args.mode == "bulk4" else 1,
              "cuda_graphs": args.mode == "graphs"}
    if args.mode == "baseline":
        import JunqiAZ
        backup = ROOT / "reports/repair_20261003"
        expected = next(row["sha256"] for row in json.loads((backup / "baseline_manifest.json").read_text(encoding="utf-8"))["files"]
                        if Path(row["path"]).name.startswith("JunqiAZ.") and row["path"].endswith(".pyd"))
        if hashlib.sha256(Path(JunqiAZ.__file__).read_bytes()).hexdigest() != expected:
            raise ValueError("baseline requires original native module: set PYTHONPATH to reports/repair_20261003/baseline_build;python")
        spec = importlib.util.spec_from_file_location("junqi_search_v3._repair_baseline_search", backup / "baseline_source/python/junqi_search_v3/search.py")
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        search_class, kwargs = original.BatchedNeuralPUCT, {}
    search = search_class(model, device=torch.device("cuda"), **kwargs)
    backend = FixedSameLayoutBackend(load_fixed_layout(ROOT / config.boundary.layout_set,
                                                      config.boundary.layout_id, project_root=ROOT))
    rng = np.random.default_rng(2026100308)
    diverse = []
    for index in range(128):
        state = backend.new_state(index)
        for _ in range(2 + index % 24):
            if state.is_terminal():
                break
            state.apply(int(rng.choice(state.legal_actions())))
        diverse.append(state)
    rows = []
    search.search(diverse[:8], 8)
    for root_type in ("opening_duplicates", "diverse"):
        for count in (1, 8, 64, 128):
            states = [backend.new_state(1) for _ in range(count)] if root_type == "opening_duplicates" else diverse[:count]
            timings, reports, action_rows = [], [], []
            for repeat in range(3):
                if args.mode == "cache":
                    search._cache.clear()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                started = time.perf_counter()
                results, report = search.search(states, 64)
                torch.cuda.synchronize()
                timings.append(time.perf_counter() - started)
                reports.append(report)
                action_rows.append([dict(action=r.action, visits=r.visits.tolist(), actions=r.actions.tolist()) for r in results])
            rows.append({"root_type": root_type, "roots": count, "simulations": 64,
                "elapsed_median": float(np.median(timings)), "elapsed_all": timings,
                "simulations_per_second": count * 64 / float(np.median(timings)),
                "moves_per_second": count / float(np.median(timings)),
                "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "reports": reports, "actions": action_rows})
            print(args.mode, root_type, count, f"{rows[-1]['elapsed_median']:.3f}s", flush=True)
    path = ROOT / args.output if args.output else ROOT / f"reports/repair_20261003/benchmark_{args.mode}.json"
    path.write_text(json.dumps({"mode": args.mode, "checkpoint": str(checkpoint), "seed": 2026100308,
                                "torch": torch.__version__, "rows": rows}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
