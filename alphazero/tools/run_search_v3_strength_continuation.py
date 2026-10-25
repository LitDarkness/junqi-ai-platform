from __future__ import annotations

import argparse
import json

import torch

from junqi_search_v3.strength_continuation import (
    run_strength_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Continue the fixed-layout Search-v3 lineage from frozen "
            "champion iteration 50."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "config/"
            "open_junqi_search_v3_strength_continuation.json"
        ),
    )
    parser.add_argument("--through-iteration", type=int, default=100)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--arena-workers", type=int, default=12)
    parser.add_argument(
        "--arena-block-states", type=int, default=128
    )
    arguments = parser.parse_args()
    if arguments.cpu_threads < 1:
        parser.error("--cpu-threads must be positive")
    torch.set_num_threads(arguments.cpu_threads)
    try:
        torch.set_num_interop_threads(
            max(1, min(4, arguments.cpu_threads))
        )
    except RuntimeError:
        pass
    report = run_strength_continuation(
        arguments.config,
        arguments.through_iteration,
        training_workers=arguments.cpu_threads,
        arena_workers=arguments.arena_workers,
        arena_block_states=arguments.arena_block_states,
    )
    print(
        json.dumps(
            {
                "completed_iteration": report.get(
                    "completed_iteration"
                ),
                "paused_iteration": report.get("paused_iteration"),
                "iterations_in_this_invocation": len(
                    report.get("reports", [])
                ),
                "run": (
                    "runs/"
                    "open_junqi_search_v3_strength_continuation"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
