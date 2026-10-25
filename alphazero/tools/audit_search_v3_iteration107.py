from __future__ import annotations

import argparse
import json

from junqi_search_v3.iteration107_audit import run_iteration107_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze iteration 107, replay its 0.12/0.24/0.48 candidates, "
            "and compare legacy Search64 with cycle-aware-v1 held-out labels."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "config/open_junqi_search_v3_strength_continuation.json"
        ),
    )
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--search-batch-states", type=int, default=256)
    arguments = parser.parse_args()
    report = run_iteration107_audit(
        arguments.config,
        cpu_threads=arguments.cpu_threads,
        search_batch_states=arguments.search_batch_states,
    )
    print(
        json.dumps(
            {
                "iteration": report["iteration"],
                "code_path_semantics_match": report[
                    "code_path_semantics_match"
                ],
                "target_semantic_difference": report[
                    "target_semantic_difference"
                ],
                "report": (
                    "runs/open_junqi_search_v3_strength_continuation/"
                    "diagnostics/iteration_000107_target_semantics_audit/"
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
