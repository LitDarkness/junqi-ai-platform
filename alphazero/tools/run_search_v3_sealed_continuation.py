from __future__ import annotations

import argparse
import json

from junqi_search_v3.sealed_continuation import (
    run_sealed_continuation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build and seal a fresh 1024-state suite, then run all formal "
            "iteration 10/49/50 continuation matchups."
        )
    )
    parser.add_argument(
        "--config",
        default=(
            "config/"
            "open_junqi_search_v3_sealed_continuation_v1.json"
        ),
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-states", type=int, default=128)
    arguments = parser.parse_args()
    report = run_sealed_continuation(
        arguments.config,
        workers=arguments.workers,
        block_states=arguments.block_states,
    )
    print(
        json.dumps(
            {
                "states_per_matchup": report["states_per_matchup"],
                "games_per_matchup": report["games_per_matchup"],
                "total_matchups": report["total_matchups"],
                "total_games": report["total_games"],
                "matchups": {
                    label: {
                        "wins": summary["wins"],
                        "draws": summary["draws"],
                        "losses": summary["losses"],
                        "score": summary["score"],
                        "family_ci95": summary["family_ci95"],
                        "repetition_fraction": summary[
                            "repetition_fraction"
                        ],
                        "capture_flag_fraction": summary[
                            "capture_flag_fraction"
                        ],
                        "average_continuation_ply": summary[
                            "average_continuation_ply"
                        ],
                    }
                    for label, summary in report["matchups"].items()
                },
                "paired_comparisons": report["paired_comparisons"],
                "report": report["report_path"],
                "summary_csv": report["summary_csv"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
