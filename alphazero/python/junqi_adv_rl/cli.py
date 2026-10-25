from __future__ import annotations

import argparse
import sys

from .pipeline import AdvantageRLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fresh-rollout Ataraxos-style advantage RL for Junqi"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--iterations", type=int, default=25)
    arguments = parser.parse_args()
    pipeline = AdvantageRLPipeline(arguments.config)
    try:
        pipeline.train(arguments.iterations)
    except KeyboardInterrupt:
        print(
            "\n[adv-rl] interrupted; no partial game/rollout was committed. "
            "Resume from the last transaction-complete latest.pt.",
            flush=True,
        )
        sys.exit(130)
    finally:
        pipeline.close()
