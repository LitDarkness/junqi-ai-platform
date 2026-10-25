"""Continue a qualified branch while holding the original shared GPU locks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from open_v4_runtime import ProcessLock, atomic_json, config_run, read_json, utc_now
from supervise_open_v4 import supervise


def stalled_gate_count(run, first, last):
    count = 0
    for iteration in range(first + 1, last + 1):
        path = run / f"reports/iteration_{iteration:06d}.json"
        if not path.exists():
            continue
        gate = read_json(path)["gate"]
        if gate.get("status") == "gate not scheduled":
            continue
        count = 0 if gate["actor_accepted"] else count + 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--original-config", default="config/open_junqi_v4_repair_pilot.json")
    parser.add_argument("--stage-iterations", type=int, default=12)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    config_path, run, config = config_run(root, args.config)
    _, original, _ = config_run(root, args.original_config)
    if run == original or args.stage_iterations < 1:
        raise ValueError("independent branch and positive iteration budget required")
    original_stop = original / "supervision/STOP"
    if original_stop.exists() or (run / "supervision/STOP").exists():
        raise RuntimeError("STOP requested; no automatic clearing")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        start = read_json(run / "CURRENT.json")["iteration"]
        seed_iteration = read_json(run / "CONTINUATION.json")["seed_iteration"]
        status_path = run / "supervision/BRANCH_STATUS.json"
        status = dict(schema="junqi-continuation-wrapper-1", pid=os.getpid(), state="running", started_at=utc_now(),
            original_run=str(original), config=str(config_path), start_iteration=start,
            target_iteration=start + args.stage_iterations, completed_iteration=start)
        def update(**values):
            status.update(values, updated_at=utc_now())
            atomic_json(status_path, status)
        update()
        try:
            while status["completed_iteration"] < status["target_iteration"]:
                failures = stalled_gate_count(run, seed_iteration, status["completed_iteration"])
                if failures >= 3:
                    update(state="review_due", reason="three_consecutive_failed_development_gates", review_required=True, failed_gates=failures)
                    return
                remaining = status["target_iteration"] - status["completed_iteration"]
                result = supervise(root, config_path, stage_iterations=min(config["actor_gate_every"], remaining),
                    extra_stop_paths=(original_stop,))
                current = read_json(run / "CURRENT.json")["iteration"]
                update(completed_iteration=current, supervisor_result=result["state"])
                if result["state"] != "review_due" or result.get("reason") != "segment_complete":
                    update(state=result["state"], reason=result.get("reason"), review_required=result.get("review_required", False))
                    return
                failures = stalled_gate_count(run, seed_iteration, current)
                if failures >= 3:
                    update(state="review_due", reason="three_consecutive_failed_development_gates", review_required=True, failed_gates=failures)
                    break
            else:
                update(state="review_due", reason="segment_complete", review_required=True)
            print(json.dumps(status, ensure_ascii=False), flush=True)
        except BaseException as error:
            update(state="needs_attention", reason=repr(error), review_required=True)
            raise


if __name__ == "__main__":
    main()
