from __future__ import annotations

import argparse
import json

from junqi_search_v3.ablation import run_iteration3_ablation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run frozen iteration-3 A/B/C/D training ablations."
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3.json",
    )
    arguments = parser.parse_args()
    result = run_iteration3_ablation(arguments.config)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

