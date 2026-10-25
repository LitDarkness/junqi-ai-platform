from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from junqi_az.model import PolicyValueNet
from junqi_search_v3.dataset import atomic_json
from junqi_search_v3.sealed_continuation import (
    MultiCheckpointContinuationEvaluator,
)
from junqi_search_v3.stages import SearchV3Stages
from junqi_search_v3.strength import SUITE_SIZE


def _checkpoint(run: Path, iteration: int) -> Path:
    return run / "checkpoints" / f"loop_iteration_{iteration:06d}.pt"


def _load_model(stages: SearchV3Stages, path: Path, iteration: int):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        payload.get("completed") is not True
        or int(payload.get("iteration", -1)) != iteration
    ):
        raise RuntimeError(f"incomplete checkpoint: {path}")
    model = PolicyValueNet(stages.config.model)
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = float(
        payload.get("leaf_value_scale", 1.0)
    )
    return model.eval()


def _select_candidate(
    run: Path, baseline: int, accepted_updates: int
) -> tuple[int, list[int]]:
    accepted = []
    for report_path in sorted((run / "reports").glob("iteration_*.json")):
        iteration = int(report_path.stem.rsplit("_", 1)[-1])
        if iteration <= baseline:
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        chosen = report.get("chosen_candidate", {})
        changed = bool(
            int(chosen.get("policy_steps", 0))
            or int(chosen.get("value_steps", 0))
        )
        if report.get("passed") is True and changed:
            accepted.append(iteration)
            if len(accepted) == accepted_updates:
                return iteration, accepted
    raise RuntimeError(
        f"only {len(accepted)} accepted updates exist after iteration "
        f"{baseline}; wait for {accepted_updates}"
    )


def run(arguments: argparse.Namespace) -> dict:
    stages = SearchV3Stages(arguments.config)
    run = stages.run
    candidate, accepted = _select_candidate(
        run, arguments.baseline, arguments.accepted_updates
    )
    iterations = (arguments.baseline, candidate)
    checkpoints = {
        iteration: _checkpoint(run, iteration)
        for iteration in iterations
    }
    for path in checkpoints.values():
        if not path.exists():
            raise RuntimeError(f"retained checkpoint is missing: {path}")
    models = {
        iteration: _load_model(stages, checkpoints[iteration], iteration)
        for iteration in iterations
    }
    strength = (
        stages.root / "runs" / "open_junqi_search_v3_strength"
    )
    suite = strength / "single_layout_strength_suite" / "states.jsonl"
    seal_path = (
        strength / "single_layout_strength_suite" / "manifest.json"
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    b = arguments.baseline
    n = candidate
    matchups = (
        (f"bare_{n}", f"bare_{n}"),
        (f"bare_{b}", f"bare_{b}"),
        (f"search64_{n}", f"bare_{n}"),
        (f"search64_{b}", f"bare_{b}"),
        (f"bare_{n}", "exact082"),
        (f"bare_{b}", "exact082"),
        (f"search64_{n}", "exact082"),
        (f"search64_{b}", "exact082"),
        (f"bare_{n}", f"bare_{b}"),
        (f"search64_{n}", f"search64_{b}"),
    )
    output = (
        run
        / "diagnostics"
        / f"cycle_validation_{b:06d}_vs_{n:06d}"
    )
    evaluator = MultiCheckpointContinuationEvaluator(
        stages=stages,
        models=models,
        checkpoints=checkpoints,
        suite_path=suite,
        seal=seal,
        output=output,
        workers=arguments.workers,
        block_states=arguments.block_states,
        matchups=matchups,
    )
    report = evaluator.run(list(matchups), states=SUITE_SIZE)
    repetition = {
        "bare_vs_bare_new_minus_baseline": (
            evaluator._paired_repetition_delta(
                f"bare_{n}_vs_bare_{n}",
                f"bare_{b}_vs_bare_{b}",
                states=SUITE_SIZE,
                seed=stages.config.seed ^ 0xBA11,
            )
        ),
        "search_vs_bare_new_minus_baseline": (
            evaluator._paired_repetition_delta(
                f"search64_{n}_vs_bare_{n}",
                f"search64_{b}_vs_bare_{b}",
                states=SUITE_SIZE,
                seed=stages.config.seed ^ 0x5EAB,
            )
        ),
    }
    score = {
        "bare_exact082_new_minus_baseline": evaluator._paired_delta(
            f"bare_{n}_vs_exact082",
            f"bare_{b}_vs_exact082",
            states=SUITE_SIZE,
            seed=stages.config.seed ^ 0xBAE0,
        ),
        "search_exact082_new_minus_baseline": evaluator._paired_delta(
            f"search64_{n}_vs_exact082",
            f"search64_{b}_vs_exact082",
            states=SUITE_SIZE,
            seed=stages.config.seed ^ 0x5EA2,
        ),
    }
    analysis = {
        "schema": "search-v3-cycle-aware-validation-v1",
        "training_mutated": False,
        "replay_written": False,
        "baseline_iteration": b,
        "candidate_iteration": n,
        "accepted_update_iterations": accepted,
        "states_per_matchup": SUITE_SIZE,
        "games_per_matchup": SUITE_SIZE * 2,
        "repetition_deltas": repetition,
        "external_score_deltas": score,
        "direct_bare": report["matchups"][f"bare_{n}_vs_bare_{b}"],
        "direct_search64": report["matchups"][
            f"search64_{n}_vs_search64_{b}"
        ],
        "engineering_targets": {
            "bare_selfplay_repetition_delta": -0.05,
            "search_vs_bare_repetition_delta": -0.03,
            "exact082_score_minimum_delta": -0.02,
            "direct_new_vs_old_minimum_score": 0.50,
        },
        "arena_report": report["report_path"],
        "summary_csv": report["summary_csv"],
    }
    atomic_json(output / "cycle_validation.json", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_strength_continuation.json",
    )
    parser.add_argument("--baseline", type=int, default=99)
    parser.add_argument(
        "--accepted-updates", type=int, choices=(3, 4, 5), default=3
    )
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--block-states", type=int, default=128)
    arguments = parser.parse_args()
    result = run(arguments)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
