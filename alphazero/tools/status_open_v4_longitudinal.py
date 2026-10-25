"""Read-only status for a registered frozen historical-self assessment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from open_v4_runtime import lock_held, read_json

ROOT = Path(__file__).resolve().parents[1]


def status(plan_path):
    plan_path = (ROOT/plan_path).resolve()
    assert plan_path.is_relative_to(ROOT)
    plan = read_json(plan_path)
    run = (ROOT/plan["directory"]).resolve()
    assert run.is_relative_to(ROOT)
    saved = read_json(run/"STATUS.json") if (run/"STATUS.json").exists() else {"state":"not_started"}
    return saved | {
        "kind":"frozen_longitudinal_evaluation",
        "assessment_gpu_lock_held": lock_held(run/"supervision/TRAINING.lock") if run.exists() else False,
        "assessment_supervisor_lock_held": lock_held(run/"supervision/SUPERVISOR.lock") if run.exists() else False,
        "stop_requested": any((ROOT/path).exists() for path in plan["stop_paths"]),
        "training_active": False,
        "training_paused": True,
        "automatic_training_start": False,
        "results_committed": (run/"RESULTS.json").exists(),
    }


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",required=True)
    parser.add_argument("--status",action="store_true",required=True)
    args=parser.parse_args()
    print(json.dumps(status(args.plan),ensure_ascii=False,indent=2))
