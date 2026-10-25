from __future__ import annotations

import argparse
import json
import sys

from junqi_search_v3.scaled_training import run_scaled_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the long-lived fixed-layout Search-v3 successor."
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_scaled_training.json",
    )
    parser.add_argument("--through-iteration", type=int, default=30)
    arguments = parser.parse_args()
    result = run_scaled_training(
        arguments.config, arguments.through_iteration
    )
    summary = {
        "completed_iteration": result.get("completed_iteration"),
        "paused_iteration": result.get("paused_iteration"),
        "training_healthy": (
            result.get("paused_iteration") is None
            and all(
                row.get("passed") is True
                for row in result.get("reports", [])
            )
        ),
        "iterations": [
            {
                "iteration": row.get("iteration"),
                "passed": row.get("passed"),
                "reasons": row.get("reasons", []),
                "selected_epoch": row.get(
                    "chosen_candidate", {}
                ).get("fractional_epoch"),
            }
            for row in result.get("reports", [])
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if result.get("paused_iteration") is not None:
        sys.exit(2)


if __name__ == "__main__":
    main()
