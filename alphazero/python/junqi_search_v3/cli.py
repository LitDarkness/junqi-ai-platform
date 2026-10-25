from __future__ import annotations

import argparse
import json
import sys

from .stages import SearchV3Stages


def main() -> None:
    parser = argparse.ArgumentParser(description="Open Junqi Search v3 stage gates")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "stage",
        choices=[
            "boundary",
            "oracle",
            "diversity",
            "supervised",
            "search-gain",
            "closed-loop",
        ],
        help="Run exactly one stage; later stages remain blocked on failure.",
    )
    arguments = parser.parse_args()
    stages = SearchV3Stages(arguments.config)
    if arguments.stage == "boundary":
        result = stages.boundary()
    elif arguments.stage == "oracle":
        result = stages.oracle()
    elif arguments.stage == "diversity":
        result = stages.diversity()
    elif arguments.stage == "supervised":
        result = stages.supervised()
    elif arguments.stage == "search-gain":
        result = stages.search_gain()
    else:
        result = stages.closed_loop()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not result["passed"]:
        sys.exit(2)
