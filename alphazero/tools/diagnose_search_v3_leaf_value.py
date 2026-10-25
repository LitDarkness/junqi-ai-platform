from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "python"))

from junqi_az.model import PolicyValueNet  # noqa: E402
from junqi_search_v3.config import SearchV3Config  # noqa: E402
from junqi_search_v3.dataset import atomic_json  # noqa: E402
from junqi_search_v3.search import BatchedNeuralPUCT  # noqa: E402
from junqi_search_v3.stages import SearchV3Stages  # noqa: E402


def classify(probe: dict) -> str:
    cycles = [row for row in probe["actions"] if row["enters_repetition"]]
    exits = [row for row in probe["actions"] if not row["enters_repetition"]]
    if not cycles or not exits:
        return "not_cycle_entry"
    cycle_q = max(float(row["q"]) for row in cycles)
    exit_q = max(float(row["q"]) for row in exits)
    if exit_q >= cycle_q + 0.05:
        return "avoidable_repetition"
    if cycle_q >= exit_q + 0.05:
        return "rational_draw"
    return "unresolved"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "open_junqi_search_v3.json"),
    )
    parser.add_argument(
        "--budgets", type=int, nargs="+", default=[32, 64, 128]
    )
    parser.add_argument(
        "--checkpoint",
        default="",
        help="Checkpoint to diagnose (default: supervised_latest.pt).",
    )
    parser.add_argument(
        "--value-scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="Network leaf-value scales to test; zero mode is always included.",
    )
    args = parser.parse_args()
    config = SearchV3Config.load(args.config)
    stages = SearchV3Stages(args.config)
    run = ROOT / config.run_directory
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run / "checkpoints" / "supervised_latest.pt"
    )
    if not checkpoint_path.is_absolute():
        checkpoint_path = (ROOT / checkpoint_path).resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    model = PolicyValueNet(config.model)
    model.load_state_dict(checkpoint["model"])
    device = torch.device(
        config.device if torch.cuda.is_available() else "cpu"
    )
    search = BatchedNeuralPUCT(
        model.to(device).eval(),
        device=device,
        cpuct=config.puct.cpuct,
        inference_batch_size=config.puct.inference_batch_size,
    )
    probes = json.loads(
        (run / "probe_manifest.json").read_text(encoding="utf-8")
    )["probes"]
    states = [stages._replay_trace(row["action_trace"])[0] for row in probes]
    classes = [classify(probe) for probe in probes]
    rows = {}
    experiments = [
        ("network", float(scale)) for scale in args.value_scales
    ] + [("zero", 0.0)]
    for mode, scale in experiments:
        for budget in args.budgets:
            results, performance = search.search(
                states,
                budget,
                leaf_value_mode=mode,
                leaf_value_scale=scale,
            )
            regrets = []
            exits = Counter()
            totals = Counter()
            repeat_actions = 0
            for probe, result, category in zip(probes, results, classes):
                oracle = {
                    int(row["action"]): row for row in probe["actions"]
                }
                best_q = max(float(row["q"]) for row in oracle.values())
                selected = oracle[result.action]
                regrets.append(best_q - float(selected["q"]))
                repeat_actions += int(selected["enters_repetition"])
                if category != "not_cycle_entry":
                    totals[category] += 1
                    exits[category] += int(
                        not selected["enters_repetition"]
                    )
            key = (
                f"network_scale_{scale:g}_{budget}"
                if mode == "network"
                else f"zero_{budget}"
            )
            rows[key] = {
                "leaf_value_mode": mode,
                "leaf_value_scale": scale,
                "simulations": budget,
                "oracle_regret_mean": float(np.mean(regrets)),
                "oracle_regret_median": float(np.median(regrets)),
                "repetition_action_fraction": (
                    repeat_actions / max(1, len(probes))
                ),
                "exit_fraction_by_class": {
                    name: exits[name] / max(1, count)
                    for name, count in totals.items()
                },
                "performance": performance,
            }
            print(
                f"[leaf-value] mode={mode} scale={scale:g} sims={budget} "
                f"regret={np.mean(regrets):.6f} "
                f"avoidable-exit="
                f"{rows[key]['exit_fraction_by_class'].get('avoidable_repetition', 0):.1%}",
                flush=True,
            )
    report = {
        "schema": "open-junqi-search-v3-leaf-value-diagnostic",
        "checkpoint": str(checkpoint_path),
        "probe_manifest": str(run / "probe_manifest.json"),
        "probe_count": len(probes),
        "classification_counts": dict(Counter(classes)),
        "rows": rows,
        "interpretation": (
            "If zero leaf consistently beats network leaf, WDL calibration "
            "is harming PUCT. This report is diagnostic and cannot open gate 04."
        ),
    }
    destination = (
        run
        / "diagnostics"
        / f"leaf_value_{checkpoint_path.stem}.json"
    )
    atomic_json(destination, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
