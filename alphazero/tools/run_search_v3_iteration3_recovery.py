from __future__ import annotations

import argparse
import json

from junqi_search_v3.recovery import run_iteration3_recovery


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Retrain frozen iteration-3 data from iteration 2 with the "
            "alternating Policy/Value recovery schedule."
        )
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3.json",
    )
    arguments = parser.parse_args()
    result = run_iteration3_recovery(arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
