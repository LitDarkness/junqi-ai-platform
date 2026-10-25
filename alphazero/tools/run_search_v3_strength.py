from __future__ import annotations

import argparse
import json

from junqi_search_v3.strength import run_strength


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and evaluate the fixed-layout Search-v3 strength suite."
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_strength.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    arguments = parser.parse_args()
    if arguments.build_only and arguments.evaluate_only:
        parser.error("--build-only and --evaluate-only are exclusive")
    result = run_strength(
        arguments.config,
        workers=arguments.workers,
        build_only=arguments.build_only,
        evaluate_only=arguments.evaluate_only,
    )
    evaluation = result.get("evaluation")
    print(
        json.dumps(
            {
                "suite": result["suite"].get("validation"),
                "evaluation_report": (
                    "runs/open_junqi_search_v3_strength/evaluation/report.json"
                    if evaluation is not None
                    else None
                ),
                "oracle": (
                    {
                        budget: row["overall"]
                        for budget, row in evaluation["oracle"].items()
                    }
                    if evaluation is not None
                    else None
                ),
                "exact082": (
                    evaluation["exact082"]
                    if evaluation is not None
                    else None
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
