from __future__ import annotations

import argparse
import json

from junqi_search_v3.policy_lr_ablation import run_policy_lr_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the fixed iteration-107 Policy-only LR multiplier ablation."
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
    report = run_policy_lr_ablation(
        arguments.config, cpu_threads=arguments.cpu_threads
    )
    print(
        json.dumps(
            {
                "source_iteration": report["source_iteration"],
                "lr_multipliers": report["lr_multipliers"],
                "save_steps": report["save_steps"],
                "arena_run": report["arena_run"],
                "report": (
                    "runs/open_junqi_search_v3_strength_continuation/"
                    "diagnostics/iteration_000107_policy_lr_ablation/"
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
