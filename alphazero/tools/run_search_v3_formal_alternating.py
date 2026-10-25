from __future__ import annotations

import argparse
import json

from junqi_search_v3.alternating_lineage import run_formal_alternating


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the formal Search-v3 alternating lineage."
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_alternating.json",
    )
    parser.add_argument("--through-iteration", type=int, default=5)
    arguments = parser.parse_args()
    result = run_formal_alternating(
        arguments.config, arguments.through_iteration
    )
    print(
        json.dumps(
            {
                "completed_iteration": result.get(
                    "completed_iteration"
                ),
                "paused_iteration": result.get("paused_iteration"),
                "recovery_confirmed": result.get(
                    "recovery_confirmed", False
                ),
                "iterations": [
                    {
                        "iteration": row.get("iteration"),
                        "passed": row.get("passed"),
                        "reasons": row.get("reasons", []),
                    }
                    for row in result.get("reports", [])
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
