from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_v4_runtime import ProcessLock


def main():
    # Spawned CPU workers execute this file during bootstrapping. Keep GPU and
    # model imports inside main so each worker stays small and CPU-only.
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import train_run

    parser = argparse.ArgumentParser(description="Repaired multi-layout learner/actor/champion training")
    parser.add_argument("--config", default="config/open_junqi_v4_repaired.json")
    parser.add_argument("--through-iteration", type=int, default=1)
    args = parser.parse_args()
    if args.through_iteration < 0:
        parser.error("through iteration must be nonnegative")
    root = Path(__file__).resolve().parents[1]
    config = OpenConfig.load((root / args.config).resolve())
    with ProcessLock(root / config.run_directory / "supervision/TRAINING.lock"):
        print(f"[runner] through_iteration={args.through_iteration} config={args.config}", flush=True)
        print(json.dumps(train_run(config, root, args.through_iteration), ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
