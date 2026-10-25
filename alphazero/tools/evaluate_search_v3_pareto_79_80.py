from __future__ import annotations

import argparse
import csv
import copy
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build"))

from junqi_az.model import PolicyValueNet  # noqa: E402
from junqi_search_v3.alternating_lineage import (  # noqa: E402
    _rng_snapshot,
    _sha256,
)
from junqi_search_v3.dataset import atomic_json  # noqa: E402
from junqi_search_v3.sealed_continuation import (  # noqa: E402
    MultiCheckpointContinuationEvaluator,
)
from junqi_search_v3.strength_continuation import (  # noqa: E402
    StrengthContinuationLineage,
)


TARGETS = {
    79: 1,  # epoch 0.276...
    80: 7,  # epoch 1.724...
}


def _recover(
    lineage: StrengthContinuationLineage,
    iteration: int,
    candidate_index: int,
    output: Path,
) -> Path:
    source_iteration = iteration - 1
    actor, optimizer, scheduler, global_step = (
        lineage._load_training_state(source_iteration)
    )
    fresh = lineage.runner._iteration_samples(iteration)
    historical_path = (
        lineage.output
        / "reports"
        / f"iteration_{iteration:06d}.json"
    )
    historical = json.loads(
        historical_path.read_text(encoding="utf-8")
    )
    checkpoint = output / (
        f"iteration_{iteration:06d}_candidate_{candidate_index:02d}.pt"
    )
    if checkpoint.exists():
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if (
            payload.get("schema")
            == "search-v3-recovered-pareto-candidate-v1"
            and int(payload.get("iteration", -1)) == iteration
            and int(payload.get("candidate_index", -1))
            == candidate_index
        ):
            return checkpoint
    captured: dict[str, Any] = {}

    def capture(
        index: int,
        model: PolicyValueNet,
        candidate_optimizer: torch.optim.AdamW,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        if index != candidate_index:
            return {}
        old = historical["candidates"][index]
        keys = (
            (
                old["validation"]["policy_excess"],
                candidate["validation"]["policy_excess"],
                "held-out Policy excess",
            ),
            (
                old["validation"]["wdl_ce"],
                candidate["validation"]["wdl_ce"],
                "held-out WDL CE",
            ),
            (
                old["probe"]["bare_oracle_regret"],
                candidate["probe"]["bare_oracle_regret"],
                "bare regret",
            ),
            (
                old["probe"]["search64_oracle_regret"],
                candidate["probe"]["search64_oracle_regret"],
                "Search64 regret",
            ),
        )
        reproduction = {}
        for expected, actual, label in keys:
            delta = float(actual) - float(expected)
            reproduction[label] = {
                "expected": float(expected),
                "actual": float(actual),
                "delta": delta,
            }
            if abs(delta) > 0.005:
                raise RuntimeError(
                    f"iteration {iteration} candidate {index} "
                    f"did not reproduce {label}: "
                    f"expected={float(expected):.12f} "
                    f"actual={float(actual):.12f} "
                    f"delta={float(actual) - float(expected):+.12f}"
                )
        payload = {
            "schema": "search-v3-recovered-pareto-candidate-v1",
            "completed": True,
            "resumable": False,
            "phase": "read_only_pareto_recovery",
            "iteration": iteration,
            "source_iteration": source_iteration,
            "candidate_index": index,
            "fractional_epoch": candidate["fractional_epoch"],
            "global_step": (
                global_step
                + int(candidate["policy_steps"])
                + int(candidate["value_steps"])
            ),
            "config": lineage.config.to_dict(),
            "config_hash": lineage.config.hash(),
            "model": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "optimizer": copy.deepcopy(
                candidate_optimizer.state_dict()
            ),
            "scheduler": copy.deepcopy(scheduler),
            "leaf_value_scale": candidate["value_calibration"][
                "leaf_value_scale"
            ],
            "leaf_value_calibration": copy.deepcopy(
                candidate["value_calibration"]
            ),
            "replay": {
                "data_iteration": iteration,
                "new_data_generated": False,
            },
            "rng": _rng_snapshot(
                lineage.generator, lineage.runner
            ),
            "candidate_metrics": copy.deepcopy(candidate),
            "source_checkpoint": str(
                lineage._checkpoint(source_iteration)
            ),
            "source_checkpoint_sha256": _sha256(
                lineage._checkpoint(source_iteration)
            ),
            "historical_report": str(historical_path),
            "historical_report_sha256": _sha256(historical_path),
            "numerical_reproduction": {
                "tolerance": 0.005,
                "reason": (
                    "CUDA/BF16 optimizer replay is not guaranteed "
                    "bitwise deterministic"
                ),
                "metrics": reproduction,
            },
        }
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        captured["candidate"] = copy.deepcopy(candidate)
        captured["numerical_reproduction"] = reproduction
        return {"captured": True}

    lineage._train_iteration(
        iteration,
        actor,
        optimizer,
        fresh,
        candidate_evaluator=capture,
        run_continuation_dev=False,
    )
    if "candidate" not in captured or not checkpoint.exists():
        raise RuntimeError(
            f"failed to recover iteration {iteration} candidate "
            f"{candidate_index}"
        )
    return checkpoint


def _load_model(
    lineage: StrengthContinuationLineage, checkpoint: Path
) -> PolicyValueNet:
    payload = torch.load(
        checkpoint, map_location="cpu", weights_only=False
    )
    model = PolicyValueNet(lineage.config.model)
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = float(
        payload["leaf_value_scale"]
    )
    model.eval()
    return model


def _existing_exact082_reference(
    lineage: StrengthContinuationLineage,
    champion_iteration: int,
    states: int,
) -> dict[str, Any]:
    root = (
        lineage.output
        / "championships"
        / "challenger_000080_vs_champion_000050"
        / "arena"
        / "states_1024"
    )
    result = {}
    for mode in ("bare", "search64"):
        label = f"{mode}_{champion_iteration}_vs_exact082"
        path = root / label / "games.csv"
        rows = [
            row
            for row in csv.DictReader(
                path.open("r", encoding="utf-8-sig", newline="")
            )
            if int(row["suite_id"]) < states
        ]
        by_family: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_family[int(row["trajectory_family_id"])].append(
                float(row["score"])
            )
        family_scores = [
            sum(values) / len(values)
            for values in by_family.values()
        ]
        result[mode] = {
            "source": str(path),
            "games": len(rows),
            "trajectory_families": len(family_scores),
            "state_weighted_score": sum(
                float(row["score"]) for row in rows
            )
            / len(rows),
            "family_weighted_mean_score": sum(family_scores)
            / len(family_scores),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-states", type=int, default=64)
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "config"
            / "open_junqi_search_v3_strength_continuation.json"
        ),
    )
    arguments = parser.parse_args()
    if arguments.states not in {256, 512}:
        raise ValueError("--states must be 256 or 512")
    torch.set_num_threads(max(1, arguments.workers))
    try:
        torch.set_num_interop_threads(
            max(1, min(4, arguments.workers))
        )
    except RuntimeError:
        pass
    lineage = StrengthContinuationLineage(
        arguments.config,
        training_workers=arguments.workers,
        arena_workers=arguments.workers,
        arena_block_states=arguments.block_states,
    )
    output = (
        lineage.output
        / "diagnostics"
        / "pareto_79_80_small_continuation"
    )
    output.mkdir(parents=True, exist_ok=True)
    recovered = {
        iteration: _recover(
            lineage, iteration, candidate_index, output
        )
        for iteration, candidate_index in TARGETS.items()
    }
    champion_payload = json.loads(
        (lineage.output / "championship.json").read_text(
            encoding="utf-8"
        )
    )
    champion_iteration = int(
        champion_payload["current"]["iteration"]
    )
    champion_checkpoint = Path(
        champion_payload["current"]["checkpoint"]
    )
    identifiers = {
        champion_iteration: champion_checkpoint,
        7901: recovered[79],
        8007: recovered[80],
    }
    models = {
        identifier: _load_model(lineage, checkpoint)
        for identifier, checkpoint in identifiers.items()
    }
    suite_root = (
        lineage.output
        / "championships"
        / "challenger_000080_vs_champion_000050"
        / "sealed_suite"
    )
    suite_path = suite_root / "states.jsonl"
    seal_path = suite_root / "SEALED.json"
    if not suite_path.exists() or not seal_path.exists():
        raise RuntimeError("existing sealed continuation suite is missing")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    matchups = (
        ("bare_7901", f"bare_{champion_iteration}"),
        ("search64_7901", f"search64_{champion_iteration}"),
        ("bare_7901", "exact082"),
        ("search64_7901", "exact082"),
        ("bare_8007", f"bare_{champion_iteration}"),
        ("search64_8007", f"search64_{champion_iteration}"),
        ("bare_8007", "exact082"),
        ("search64_8007", "exact082"),
    )
    evaluator = MultiCheckpointContinuationEvaluator(
        stages=lineage.stages,
        models=models,
        checkpoints=identifiers,
        suite_path=suite_path,
        seal=seal,
        output=output / f"arena_{arguments.states}_states",
        workers=arguments.workers,
        block_states=arguments.block_states,
        matchups=matchups,
    )
    report = evaluator.run(
        list(matchups), states=arguments.states
    )
    assessments = {}
    incumbent_exact = _existing_exact082_reference(
        lineage, champion_iteration, arguments.states
    )
    for identifier, source_iteration in ((7901, 79), (8007, 80)):
        bare = report["matchups"][
            f"bare_{identifier}_vs_bare_{champion_iteration}"
        ]
        search = report["matchups"][
            f"search64_{identifier}_vs_search64_{champion_iteration}"
        ]
        bare_low, bare_high = map(float, bare["family_ci95"])
        search_low, search_high = map(
            float, search["family_ci95"]
        )
        candidate_bare_exact = report["matchups"][
            f"bare_{identifier}_vs_exact082"
        ]
        candidate_search_exact = report["matchups"][
            f"search64_{identifier}_vs_exact082"
        ]
        bare_exact_delta = (
            float(
                candidate_bare_exact[
                    "family_weighted_mean_score"
                ]
            )
            - float(
                incumbent_exact["bare"][
                    "family_weighted_mean_score"
                ]
            )
        )
        search_exact_delta = (
            float(
                candidate_search_exact[
                    "family_weighted_mean_score"
                ]
            )
            - float(
                incumbent_exact["search64"][
                    "family_weighted_mean_score"
                ]
            )
        )
        direct_promising = (
            bare_high >= 0.5
            and search_high >= 0.5
            and (
                float(bare["family_weighted_mean_score"]) >= 0.5
                or float(
                    search["family_weighted_mean_score"]
                )
                >= 0.5
                or bare_low > 0.5
                or search_low > 0.5
            )
        )
        promising = (
            direct_promising
            and bare_exact_delta >= -0.02
            and search_exact_delta >= -0.02
        )
        assessments[str(source_iteration)] = {
            "identifier": identifier,
            "candidate_index": TARGETS[source_iteration],
            "promising_for_formal_championship": promising,
            "bare_vs_incumbent": bare,
            "search64_vs_incumbent": search,
            "direct_continuation_promising": direct_promising,
            "bare_vs_exact082": candidate_bare_exact,
            "search64_vs_exact082": candidate_search_exact,
            "bare_exact082_family_delta_vs_incumbent": (
                bare_exact_delta
            ),
            "search64_exact082_family_delta_vs_incumbent": (
                search_exact_delta
            ),
        }
    summary = {
        "schema": "search-v3-pareto-79-80-small-continuation-v1",
        "states": arguments.states,
        "games_per_matchup": arguments.states * 2,
        "new_selfplay_generated": False,
        "source_suite": str(suite_path),
        "source_suite_sha256": _sha256(suite_path),
        "suite_was_sealed_before_candidate_recovery": True,
        "champion_iteration": champion_iteration,
        "incumbent_exact082_same_state_reference": (
            incumbent_exact
        ),
        "recovered_checkpoints": {
            str(iteration): {
                "candidate_index": TARGETS[iteration],
                "path": str(path),
                "sha256": _sha256(path),
                "numerical_reproduction": torch.load(
                    path, map_location="cpu", weights_only=False
                )["numerical_reproduction"],
            }
            for iteration, path in recovered.items()
        },
        "assessments": assessments,
        "arena_report": report["report_path"],
        "formal_championship_started": False,
    }
    atomic_json(output / "summary.json", summary)
    registry_path = lineage.output / "competitive.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for source_iteration, assessment in assessments.items():
        if not assessment["promising_for_formal_championship"]:
            continue
        recovered_checkpoint = recovered[int(source_iteration)]
        registry.setdefault("iterations", {})[source_iteration] = {
            "reason": (
                "recovered Pareto candidate passed the 256-state "
                "small continuation dev screen; formal championship "
                "not yet run"
            ),
            "checkpoint": str(recovered_checkpoint),
            "candidate_index": TARGETS[int(source_iteration)],
            "official_incumbent": False,
            "small_continuation_summary": str(output / "summary.json"),
            "bare_family_score": assessment[
                "bare_vs_incumbent"
            ]["family_weighted_mean_score"],
            "search64_family_score": assessment[
                "search64_vs_incumbent"
            ]["family_weighted_mean_score"],
            "permanent": True,
        }
    atomic_json(registry_path, registry)
    print(
        json.dumps(summary, ensure_ascii=False),
        flush=True,
    )


if __name__ == "__main__":
    main()
