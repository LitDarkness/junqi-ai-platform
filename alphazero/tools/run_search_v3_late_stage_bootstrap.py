from __future__ import annotations

import argparse
import json

from junqi_search_v3.late_stage_bootstrap import (
    run_late_stage_bootstrap,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild the iteration-107 0.5x/3+2 full atomic candidate, "
            "run its internal gate, then run the formal sealed championship."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "config/"
            "open_junqi_search_v3_strength_late_stage.json"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--arena-workers", type=int, default=12)
    parser.add_argument(
        "--arena-block-states", type=int, default=128
    )
    parser.add_argument(
        "--internal-gate-only", action="store_true"
    )
    arguments = parser.parse_args()
    result = run_late_stage_bootstrap(
        arguments.config,
        cpu_threads=arguments.cpu_threads,
        arena_workers=arguments.arena_workers,
        arena_block_states=arguments.arena_block_states,
        run_championship=not arguments.internal_gate_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
