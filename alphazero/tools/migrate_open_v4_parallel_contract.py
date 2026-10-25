"""Migrate only the frozen, measured CPU concurrency optimization."""
from __future__ import annotations

import json
from pathlib import Path


def main():
    from open_v4_runtime import ProcessLock
    from junqi_open_v4.pipeline import source_contract, sha256
    from junqi_search_v3.dataset import atomic_json

    root = Path(__file__).resolve().parents[1]
    run = root / "runs/open_junqi_v4_repair_pilot"
    report = root / "reports/parallel_20261003"
    with ProcessLock(run / "supervision/SUPERVISOR.lock"), ProcessLock(run / "supervision/TRAINING.lock"):
        stop = json.loads((run / "supervision/STOP").read_text(encoding="utf-8"))
        recorded_stop = json.loads((report / "maintenance_stop.json").read_text(encoding="utf-8"))
        if stop != recorded_stop or stop["source"] != "agent_maintenance":
            raise ValueError("maintenance stop changed; preserve the newer stop request")
        old = json.loads((run / "CONTRACT.json").read_text(encoding="utf-8"))
        current = json.loads((run / "CURRENT.json").read_text(encoding="utf-8"))
        if current["contract_hash"] != sha256(run / "CONTRACT.json") or current["iteration"] != 6:
            raise ValueError("unexpected checkpoint or old contract")
        source = source_contract(root)
        snapshots = json.loads((report / "validated_source.json").read_text(encoding="utf-8"))
        if source != snapshots:
            raise ValueError("source changed since frozen validation")
        expected_changes = {str(Path("python/junqi_open_v4") / (name + ".py")) for name in ("games", "arena", "pipeline")}
        changed = {name for name in old["source"] if old["source"][name] != source.get(name)}
        added = set(source) - set(old["source"])
        if changed != expected_changes or added != {str(Path("python/junqi_open_v4/teachers.py"))}:
            raise ValueError("migration permits only teacher parallelism, overlap and worker shutdown")
        before = json.loads((report / "before_manifest.json").read_text(encoding="utf-8"))
        for row in before:
            if old["source"][str(Path(row["path"]))] != row["sha256"] or sha256(root / row["backup"]) != row["sha256"]:
                raise ValueError("original source snapshot mismatch")
        collect = json.loads((report / "collect_comparison.json").read_text(encoding="utf-8"))
        arena = json.loads((report / "arena_comparison.json").read_text(encoding="utf-8"))
        if collect["games"] != 64 or collect["trajectory_differences"] or not collect["sample_equivalence"]:
            raise ValueError("complete frozen collection equivalence failed")
        # Eight selected families have 18 surviving opening/midgame starts;
        # 18 * two teachers * both colors = 72 complete games.
        if arena["games"] != 72 or arena["trajectory_differences"]:
            raise ValueError("complete frozen development arena equivalence failed")
        if min(collect["speedup"], arena["speedup"]) <= 1:
            raise ValueError("concurrency did not improve end-to-end throughput")
        tests = json.loads((report / "tests.json").read_text(encoding="utf-8"))
        if not tests["passed"] or tests["exit_code"] != 0:
            raise ValueError("regression checks failed")
        for role in ("learner", "actor", "champion"):
            if sha256(run / current[role]["path"]) != current[role]["sha256"]:
                raise ValueError("checkpoint digest mismatch")
        if collect["checkpoint_sha256"] != current["actor"]["sha256"] or arena["checkpoint_sha256"] != current["actor"]["sha256"]:
            raise ValueError("benchmark was not run on the frozen current actor")
        migration = {"reason": "12 persistent spawned CPU teacher workers overlapping unchanged sequential neural MCTS",
            "previous_contract": old, "previous_current": current, "new_source": source,
            "collect": collect, "arena": arena, "tests": tests,
            "native_teacher_wall_time_caveat": "50 ms safety cap unchanged; frozen comparisons exact, arbitrary future trajectories are not guaranteed under differing system load",
            "checkpoints_modified": False, "optimizer_and_rng_modified": False}
        atomic_json(run / "migrations/cpu_teacher_parallelism.json", migration)
        old["source"] = source
        atomic_json(run / "CONTRACT.json", old)
        current["contract_hash"] = sha256(run / "CONTRACT.json")
        atomic_json(run / "CURRENT.json", current)
        print("Migrated measured CPU concurrency; all checkpoints, optimizer and RNG preserved")


if __name__ == "__main__":
    main()
