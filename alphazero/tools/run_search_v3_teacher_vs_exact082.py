from __future__ import annotations

import argparse
import json
from pathlib import Path

from junqi_search_v3.continuation import SuiteContinuationEvaluator
from junqi_search_v3.strength import SearchV3Strength, _sha256


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Formal paired continuation benchmark of the Search-v3 warm-start "
            "Oracle against exact082."
        )
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_strength.json",
    )
    parser.add_argument("--oracle-budget", type=int, default=2000)
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-states", type=int, default=128)
    parser.add_argument(
        "--output",
        default=(
            "runs/open_junqi_search_v3_teacher_vs_exact082/"
            "formal_continuation"
        ),
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    output = Path(args.output).resolve()
    strength = SearchV3Strength(config_path, workers=args.workers)
    scale_path = strength.run_dir / "oracle_scale.json"
    evaluator = SuiteContinuationEvaluator(
        strength,
        workers=args.workers,
        block_states=args.block_states,
        output_directory=output,
        contract_extra={
            "benchmark_purpose": (
                "initial Search-v3 warm-start Oracle vs exact082"
            ),
            "primary_agent": f"oracle_{args.oracle_budget}",
            "secondary_agent": "exact082",
            "oracle_budget_total_root_nodes": args.oracle_budget,
            "oracle_scale_path": str(scale_path),
            "oracle_scale_sha256": (
                _sha256(scale_path) if scale_path.exists() else None
            ),
            "training_data_excluded": True,
        },
    )
    report = evaluator.run(
        [(f"oracle_{args.oracle_budget}", "exact082")],
        states=args.states,
    )
    print(
        json.dumps(
            {
                "report": str(
                    output / f"report_states_{args.states:04d}.json"
                ),
                "result": report["matchups"][
                    f"oracle_{args.oracle_budget}_vs_exact082"
                ],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
