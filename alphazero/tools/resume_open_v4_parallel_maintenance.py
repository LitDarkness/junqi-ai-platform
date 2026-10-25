"""Clear only this verified agent maintenance marker, never a user's stop."""
from __future__ import annotations

import json
from pathlib import Path

from open_v4_runtime import ProcessLock, atomic_json, utc_now


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/open_junqi_v4_repair_pilot"
    folder = run / "supervision"
    report = root / "reports/parallel_20261003"
    with ProcessLock(folder / "SUPERVISOR.lock"), ProcessLock(folder / "TRAINING.lock"):
        recorded = (report / "maintenance_stop.json").read_bytes()
        stop_path = folder / "STOP"
        if stop_path.read_bytes() != recorded:
            raise ValueError("STOP changed; do not clear a newer request")
        stop = json.loads(recorded)
        if stop["source"] != "agent_maintenance" or not stop["resume_authorized"]:
            raise ValueError("not an authorized agent maintenance stop")
        migration = json.loads((run / "migrations/cpu_teacher_parallelism.json").read_text(encoding="utf-8"))
        contract = json.loads((run / "CONTRACT.json").read_text(encoding="utf-8"))
        if migration["new_source"] != contract["source"] or migration["checkpoints_modified"]:
            raise ValueError("verified concurrency migration is missing")
        plan = json.loads((folder / "PLAN.json").read_text(encoding="utf-8"))
        if plan["maintenance"]["stop_nonce"] != stop["nonce"]:
            raise ValueError("maintenance plan mismatch")
        plan["maintenance"].update(active=False, completed_at=utc_now(),
            report="reports/parallel_20261003/REPORT_zh-CN.md", migrated=True)
        plan["teacher_execution"] = "12 persistent CPU processes overlapping neural search; one GPU learner"
        plan["resume_from_iteration"] = 6
        atomic_json(folder / "PLAN.json", plan)
        record = {"at": utc_now(), "type": "verified_concurrency_maintenance",
            "checkpoint_iteration": 6, "effective_updates": 384, "tests_passed": 83,
            "collection_speedup": migration["collect"]["speedup"], "arena_speedup": migration["arena"]["speedup"],
            "trajectory_differences": 0, "checkpoints_modified": False,
            "decision": "Resume from iteration 6 for 14 more rounds to the original segment target 20",
            "report": "reports/parallel_20261003/REPORT_zh-CN.md"}
        with (folder / "REVIEWS.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        # Re-read immediately before removing this exact marker.
        if stop_path.read_bytes() != recorded:
            raise ValueError("STOP changed during review; preserve it")
        stop_path.unlink()
        print("Agent maintenance marker cleared; user stop requests remain protected")


if __name__ == "__main__":
    main()
