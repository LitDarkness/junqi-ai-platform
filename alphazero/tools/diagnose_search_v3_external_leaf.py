from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "python"))

from junqi_az.model import PolicyValueNet  # noqa: E402
from junqi_search_v3.config import SearchV3Config  # noqa: E402
from junqi_search_v3.dataset import atomic_json  # noqa: E402
from junqi_search_v3.evaluation import SearchEvaluator  # noqa: E402
from junqi_search_v3.stages import SearchV3Stages  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "open_junqi_search_v3.json"),
    )
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--simulations", type=int, default=64)
    args = parser.parse_args()
    config = SearchV3Config.load(args.config)
    stages = SearchV3Stages(args.config)
    run = ROOT / config.run_directory
    checkpoint = torch.load(
        run / "checkpoints" / "supervised_latest.pt",
        map_location="cpu",
        weights_only=False,
    )
    model = PolicyValueNet(config.model)
    model.load_state_dict(checkpoint["model"])
    scale = json.loads(
        (run / "oracle_scale.json").read_text(encoding="utf-8")
    )["score_scale"]
    evaluator = SearchEvaluator(
        root=ROOT,
        backend=stages.backend,
        config=config,
        model=model,
        score_scale=float(scale),
    )
    rows = {}
    for opponent in ("supplement_wide_quiescence", "exact082"):
        for mode in ("network", "zero"):
            candidate = (
                f"candidate_{args.simulations}"
                if mode == "network"
                else f"candidate_zero_{args.simulations}"
            )
            key = f"{mode}_{args.simulations}_vs_{opponent}"
            print(f"[external-leaf] {key}", flush=True)
            rows[key] = evaluator.paired_candidate_vs(
                candidate,
                opponent,
                args.games,
                config.seed ^ sum(ord(char) for char in key),
            )
    report = {
        "schema": "open-junqi-search-v3-external-leaf-diagnostic",
        "checkpoint": str(
            run / "checkpoints" / "supervised_latest.pt"
        ),
        "games_per_comparison": args.games,
        "simulations": args.simulations,
        "rows": rows,
        "read_only": True,
        "opens_gate": False,
    }
    destination = run / "diagnostics" / "04_external_leaf_ablation.json"
    atomic_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
