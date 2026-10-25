from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout

from .config import SearchV3Config
from .search import BatchedNeuralPUCT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--roots", type=int, default=256)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--checkpoint", default="")
    arguments = parser.parse_args()
    config_path = Path(arguments.config).resolve()
    root = config_path.parent.parent
    config = SearchV3Config.load(config_path)
    run = root / config.run_directory
    if arguments.checkpoint:
        checkpoint = Path(arguments.checkpoint).resolve()
    else:
        gate_path = run / "gates" / "03_supervised.json"
        supervised_passed = (
            gate_path.exists()
            and json.loads(gate_path.read_text(encoding="utf-8")).get(
                "passed"
            )
            is True
        )
        candidates = []
        if supervised_passed:
            candidates.append(
                run / "checkpoints" / "supervised_latest.pt"
            )
        candidates.extend(
            [
                run / "checkpoints" / "initial_random.pt",
                run / "checkpoints" / "initial_v2b_trunk_ablation.pt",
            ]
        )
        checkpoint = next(path for path in candidates if path.exists())
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    model = PolicyValueNet(config.model).to(device).eval()
    model.load_state_dict(payload["model"])
    layout = load_fixed_layout(
        config.boundary.layout_set,
        config.boundary.layout_id,
        project_root=root,
    )
    backend = FixedSameLayoutBackend(layout)
    states = []
    for index in range(arguments.roots):
        state = backend.new_state(config.seed + index)
        rng = np.random.default_rng(config.seed + index)
        for _ in range(index % 8):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            state.apply(int(legal[rng.integers(len(legal))]))
        states.append(state)
    search = BatchedNeuralPUCT(
        model,
        device=device,
        cpuct=config.puct.cpuct,
        inference_batch_size=config.puct.inference_batch_size,
    )
    _, metric = search.search(states, arguments.simulations)
    metric.update(
        {
            "checkpoint": str(checkpoint),
            "device": str(device),
            "roots": arguments.roots,
            "simulations": arguments.simulations,
            "gpu_memory_allocated_mib": (
                torch.cuda.max_memory_allocated() / 2**20
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    print(json.dumps(metric, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
