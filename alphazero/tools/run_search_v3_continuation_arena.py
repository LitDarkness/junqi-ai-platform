from __future__ import annotations

import argparse
import json

from junqi_search_v3.continuation import run_suite_continuations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continue paired games from immutable strength-suite states."
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_strength.json",
    )
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-states", type=int, default=128)
    parser.add_argument("--states", type=int, default=1024)
    parser.add_argument(
        "--matchups",
        default="bare_vs_exact082,search64_vs_exact082,search64_vs_bare",
    )
    arguments = parser.parse_args()
    matchups = []
    for raw in arguments.matchups.split(","):
        primary, separator, secondary = raw.strip().partition("_vs_")
        if not separator:
            parser.error(f"invalid matchup: {raw}")
        matchups.append((primary, secondary))
    report = run_suite_continuations(
        arguments.config,
        workers=arguments.workers,
        block_states=arguments.block_states,
        states=arguments.states,
        matchups=matchups,
    )
    print(
        json.dumps(
            {
                "states_per_matchup": report["states_per_matchup"],
                "games_per_matchup": report["games_per_matchup"],
                "matchups": {
                    key: {
                        "wins": value["wins"],
                        "draws": value["draws"],
                        "losses": value["losses"],
                        "score": value["score"],
                        "family_ci95": value["family_ci95"],
                        "average_continuation_ply": value[
                            "average_continuation_ply"
                        ],
                        "search64_sps": value["performance"][
                            "search64_simulations_per_second"
                        ],
                    }
                    for key, value in report["matchups"].items()
                },
                "search64_minus_bare_vs_exact082": report[
                    "search64_minus_bare_vs_exact082"
                ],
                "report": (
                    "runs/open_junqi_search_v3_strength/"
                    "continuation_arena/report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
