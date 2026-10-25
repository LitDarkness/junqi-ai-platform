from __future__ import annotations

import argparse
import json

from junqi_search_v3.value_step_ablation import run_value_step_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Attach 0/1/2/3 fixed Value steps to the two selected "
            "iteration-107 Policy candidates."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "config/open_junqi_search_v3_strength_continuation.json"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=12)
    arguments = parser.parse_args()
    report = run_value_step_ablation(
        arguments.config, cpu_threads=arguments.cpu_threads
    )
    print(
        json.dumps(
            {
                "source_iteration": report["source_iteration"],
                "value_steps": report["value_steps"],
                "arena_run": report["arena_run"],
                "report": (
                    "runs/open_junqi_search_v3_strength_continuation/"
                    "diagnostics/iteration_000107_value_step_ablation/"
                    "report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
