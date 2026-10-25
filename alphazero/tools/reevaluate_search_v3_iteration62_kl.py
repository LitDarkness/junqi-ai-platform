from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build"))

from junqi_search_v3.alternating_lineage import (  # noqa: E402
    _rng_snapshot,
    _sha256,
)
from junqi_search_v3.dataset import atomic_json  # noqa: E402
from junqi_search_v3.strength_continuation import (  # noqa: E402
    StrengthContinuationLineage,
)


def _close(left: float, right: float, tolerance: float = 2e-6) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically re-evaluate iteration 62 candidates after "
            "the candidate-local KL gate correction. No self-play is run."
        )
    )
    parser.add_argument(
        "--config",
        default=str(
            ROOT
            / "config"
            / "open_junqi_search_v3_strength_continuation.json"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=12)
    arguments = parser.parse_args()
    torch.set_num_threads(max(1, int(arguments.cpu_threads)))
    try:
        torch.set_num_interop_threads(
            max(1, min(4, int(arguments.cpu_threads)))
        )
    except RuntimeError:
        pass

    iteration = 62
    source_iteration = iteration - 1
    lineage = StrengthContinuationLineage(
        arguments.config,
        training_workers=arguments.cpu_threads,
        arena_workers=arguments.cpu_threads,
        arena_block_states=128,
    )
    run = lineage.output
    original_report_path = (
        run / "reports" / f"iteration_{iteration:06d}.json"
    )
    source_checkpoint_path = lineage._checkpoint(source_iteration)
    replay_path = (
        run / "search_replay" / f"iteration_{iteration:06d}"
    )
    if not original_report_path.exists():
        raise RuntimeError("iteration 62 report is missing")
    if not source_checkpoint_path.exists():
        raise RuntimeError("iteration 61 atomic checkpoint is missing")
    if not (replay_path / "summary.json").exists():
        raise RuntimeError("iteration 62 committed replay is missing")

    original = json.loads(
        original_report_path.read_text(encoding="utf-8")
    )
    actor, optimizer, scheduler, global_step = (
        lineage._load_training_state(source_iteration)
    )
    fresh = lineage.runner._iteration_samples(iteration)
    diagnostic_root = (
        run / "diagnostics" / "iteration_000062_kl_requalification"
    )
    diagnostic_root.mkdir(parents=True, exist_ok=True)
    requalified_path = (
        run
        / "candidates"
        / "requalified_iteration_000062.pt"
    )
    best: dict[str, Any] | None = None

    def evaluate_candidate(
        index: int,
        model: Any,
        candidate_optimizer: torch.optim.AdamW,
        candidate: dict[str, Any],
    ) -> dict[str, Any]:
        nonlocal best
        arena = lineage.runner._arena_metrics(
            model, iteration, "iteration62_kl_recheck"
        )
        diagnostics = {
            "continuation_dev_score": arena[
                "external_search64_mean"
            ],
            "continuation_dev_bare_score": arena[
                "external_bare_mean"
            ],
            "search64_vs_bare_score": arena[
                "direct_search64_vs_bare"
            ]["score"],
            "continuation": arena,
        }
        if index > 0 and candidate["eligible"]:
            rank = (
                -float(candidate["probe"]["search64_oracle_regret"]),
                float(diagnostics["continuation_dev_score"]),
                -float(candidate["validation"]["policy_excess"]),
            )
            if best is None or rank > best["rank"]:
                payload = {
                    "schema": (
                        "open-junqi-search-v3-kl-requalified-candidate-v1"
                    ),
                    "completed": False,
                    "resumable": False,
                    "phase": "diagnostic_requalified_candidate",
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
                    "scheduler": {
                        **copy.deepcopy(scheduler),
                        "diagnostic_candidate_iteration": iteration,
                        "diagnostic_candidate_index": index,
                    },
                    "leaf_value_scale": candidate[
                        "value_calibration"
                    ]["leaf_value_scale"],
                    "leaf_value_calibration": copy.deepcopy(
                        candidate["value_calibration"]
                    ),
                    "replay": {
                        "data_iteration": iteration,
                        "read_only_requalification": True,
                    },
                    "rng": _rng_snapshot(
                        lineage.generator, lineage.runner
                    ),
                    "qualification": {
                        "candidate_final_kl": candidate[
                            "candidate_final_kl"
                        ],
                        "candidate_final_family_kl": candidate[
                            "candidate_final_family_kl"
                        ],
                        "value_step_peak_kl": candidate[
                            "value_step_peak_kl"
                        ],
                        **diagnostics,
                    },
                }
                temporary = requalified_path.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, requalified_path)
                best = {
                    "index": index,
                    "rank": rank,
                    "checkpoint": str(requalified_path),
                }
        print(
            "[iteration62 recheck] "
            f"candidate={index} epoch={candidate['fractional_epoch']:.4f} "
            f"final-kl={candidate['candidate_final_kl']:.7f} "
            f"family-kl={candidate['candidate_final_family_kl']:.7f} "
            f"peak-kl={candidate['value_step_peak_kl']:.7f} "
            f"dev={diagnostics['continuation_dev_score']:.4f} "
            f"eligible={candidate['eligible']}",
            flush=True,
        )
        return diagnostics

    _, _, _, details = lineage._train_iteration(
        iteration,
        actor,
        optimizer,
        fresh,
        candidate_evaluator=evaluate_candidate,
        run_continuation_dev=False,
    )
    rows = []
    historical = original["candidates"]
    if len(historical) != len(details["candidates"]):
        raise RuntimeError("candidate count changed during re-evaluation")
    for index, (old, candidate) in enumerate(
        zip(historical, details["candidates"])
    ):
        if not (
            _close(
                old["paired_policy"]["mean_delta"],
                candidate["paired_policy"]["mean_delta"],
            )
            and _close(
                old["paired_wdl"]["mean_delta"],
                candidate["paired_wdl"]["mean_delta"],
            )
            and _close(
                old["probe"]["bare_oracle_regret"],
                candidate["probe"]["bare_oracle_regret"],
            )
            and _close(
                old["probe"]["search64_oracle_regret"],
                candidate["probe"]["search64_oracle_regret"],
            )
        ):
            raise RuntimeError(
                f"candidate {index} did not reproduce historical metrics"
            )
        diagnostics = candidate["candidate_diagnostics"]
        rows.append(
            {
                "candidate_index": index,
                "fractional_epoch": candidate["fractional_epoch"],
                "policy_steps": candidate["policy_steps"],
                "value_steps": candidate["value_steps"],
                "value_step_peak_kl": candidate[
                    "value_step_peak_kl"
                ],
                "candidate_final_kl": candidate[
                    "candidate_final_kl"
                ],
                "candidate_final_family_kl": candidate[
                    "candidate_final_family_kl"
                ],
                "candidate_final_family_kl_max": candidate[
                    "candidate_final_family_kl_max"
                ],
                "policy_family": candidate["paired_policy"],
                "wdl_family": candidate["paired_wdl"],
                "bare_oracle_regret": candidate["probe"][
                    "bare_oracle_regret"
                ],
                "search64_oracle_regret": candidate["probe"][
                    "search64_oracle_regret"
                ],
                "continuation_dev_score": diagnostics[
                    "continuation_dev_score"
                ],
                "continuation_dev_bare_score": diagnostics[
                    "continuation_dev_bare_score"
                ],
                "search64_vs_bare_score": diagnostics[
                    "search64_vs_bare_score"
                ],
                "eligible_after_kl_fix": candidate["eligible"],
            }
        )

    result = {
        "schema": "open-junqi-search-v3-iteration62-kl-recheck-v1",
        "iteration": iteration,
        "source_checkpoint": str(source_checkpoint_path),
        "source_checkpoint_sha256": _sha256(source_checkpoint_path),
        "original_report": str(original_report_path),
        "original_report_sha256": _sha256(original_report_path),
        "committed_replay": str(replay_path),
        "no_new_data": True,
        "no_training_lineage_mutation": True,
        "gate": {
            "candidate_final_kl_limit": 1e-3,
            "value_step_peak_kl_hard_limit": 1.5e-3,
            "family_kl_role": "diagnostic; family-weighted mean",
        },
        "historical_reproduction_passed": True,
        "candidates": rows,
        "requalified_candidate": best,
        "official_iteration_62_unchanged": True,
    }
    report_path = diagnostic_root / "report.json"
    atomic_json(report_path, result)
    atomic_json(
        run / "reports" / "iteration_000062_kl_requalification.json",
        result,
    )
    if best is not None:
        registry_path = run / "competitive.json"
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        selected = rows[int(best["index"])]
        registry.setdefault("iterations", {})["62"] = {
            "reason": (
                "iteration 62 candidate restored to competitive-only "
                "status after candidate-local KL gate correction"
            ),
            "checkpoint": str(requalified_path),
            "candidate_index": int(best["index"]),
            "official_incumbent": False,
            "policy_classification": selected["policy_family"][
                "classification"
            ],
            "wdl_classification": selected["wdl_family"][
                "classification"
            ],
            "search64_oracle_regret": selected[
                "search64_oracle_regret"
            ],
            "continuation_dev_score": selected[
                "continuation_dev_score"
            ],
            "permanent": True,
        }
        atomic_json(registry_path, registry)
    print(
        json.dumps(
            {
                "report": str(report_path),
                "requalified_candidate": best,
                "candidate_count": len(rows),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
