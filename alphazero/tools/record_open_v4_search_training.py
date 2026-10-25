"""Record the reviewed handoff; this does not launch or mutate training."""
import argparse
import hashlib
import json
from pathlib import Path
from open_v4_runtime import atomic_json, read_json, utc_now, lock_held


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch-record")
    parser.add_argument("--verify-running", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    original = root / "runs/open_junqi_v4_repair_pilot"
    branch = root / "runs/open_junqi_v4_fresh_continuation"
    plan_path = original / "supervision/PLAN.json"
    plan = read_json(plan_path)
    now = utc_now()
    confirmation = root / "runs/open_junqi_v4_search_confirmation_20261004"
    result = read_json(confirmation / "RESULTS.json")
    next_action = ("Follow PLAN optimization: read search-training STATUS/PROGRESS, variant complete stage checkpoints/reports, "
        "arena_primary/secondary and RESULTS plus logs. While original+fresh branch GPU locks are held and logs advance, "
        "read/record only: never start another GPU task or modify executed code/config. Six paired stages start from complete "
        "qualified26, each256 new train games/64 PPV updates; only collection search changes64/128 vs256/256. Primary selection "
        "uses matched256,full64 development families with positive teacher gains vs parent AND control, both direct wins, "
        "duplicate tolerance and unchanged champion gate. Secondary64/32 is diagnostic only. On success verify checkpoint/optimizer/RNG "
        "and fork qualified complete candidate into an independent256-search continuation with documented data seed namespace. "
        "On failure preserve all evidence and audit teacher-policy/terminal-WDL labels or design one new controlled factor; "
        "do not blind extend stalled35 or call runtime search gain trained-weight improvement. Respect all recorded STOPs; "
        "final layouts/development failures never enter replay. No automatic retry; inspect failure/locks/full progress before one explicit recovery.")
    old = plan["optimization"]
    if old["type"] != "paired_search_data_training":
        old.update(active=False, state="completed", completed_at=read_json(confirmation / "STATUS.json")["updated_at"],
            result=str((confirmation / "RESULTS.json").relative_to(root)), search_256_confirmed=True,
            review=str((confirmation / "REVIEW.json").relative_to(root)),
            conclusion="Full64:256 teacher score23.89% vs64 score14.84%,paired+9.05pp[3.78,14.39];direct65.04%[60.61,69.40]. Runtime search gain confirmed, not new trained weights; failed primary128 preserved.")
        plan.setdefault("optimization_history", []).append(old)
        plan["optimization"] = dict(active=True, type="paired_search_data_training", state="registered",
            experiment="config/experiments/open_v4_search_training_20261004.json",
            run_directory="runs/open_junqi_v4_search_training_20261004", registered_at=now,
            parent_iteration=26, frozen_unqualified_iteration=35, source_effective_updates=1664,
            stages=6, training_games_per_variant=1536, attempted_updates_per_variant=384,
            planned_primary_games=3564, planned_secondary_games=1716,
            reason="Frozen search256 passed full64 development recheck; test whether stronger search-data improves weights after64-search continuation stalled.",
            launcher="start_open_v4_search_training.ps1", report="reports/search_budgets_20261004/PLAN_zh-CN.md")
    active = plan["optimization"]
    active["next_action"] = next_action
    if args.launch_record:
        launch = read_json(root / args.launch_record)
        for key in ("pid", "stdout", "stderr"):
            value = launch[key]
            active[key] = str(Path(value).relative_to(root)) if key != "pid" else value
        active.update(state="launched", started_at=now)
    if args.verify_running:
        run = root / active["run_directory"]
        status, progress = read_json(run / "STATUS.json"), read_json(run / "PROGRESS.json")
        if status["pid"] != active["pid"] or status["state"] not in ("collecting", "training", "arena"):
            raise ValueError("expected running experiment missing; inspect failure before recovery")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        for frozen in (original, branch):
            if not all(lock_held(frozen / f"supervision/{name}.lock") for name in ("SUPERVISOR", "TRAINING")):
                raise ValueError("global GPU guard is not held")
        contract = read_json(run / "CONTRACT.json")
        if sha(root / "tools/run_open_v4_search_training.py") != contract["tool_sha256"]:
            raise ValueError("executing source changed")
        for record in progress["variants"].values():
            if sha(run / record["checkpoint"]) != record["sha256"]:
                raise ValueError("committed complete checkpoint changed")
        for record in progress["collections"].values():
            if sha(run / record["path"]) != record["sha256"] or sha(run / record["metadata"]) != record["metadata_sha256"]:
                raise ValueError("committed training shard changed")
            if read_json(run / record["metadata"])["layout_role"] != "train":
                raise ValueError("nontraining data in replay")
        seed_checks = {v: read_json(run / v / "SEED_VERIFICATION.json") for v in ("control", "candidate")}
        if not all(all(value is True for key, value in checks.items() if key != "source") for checks in seed_checks.values()):
            raise ValueError("full source state migration failed")
        log_path = root / active["stdout"]
        observation = {"at": now, "status": status, "progress": progress,
            "seed_checks": seed_checks, "four_global_gpu_locks_held": True,
            "committed_checkpoint_and_replay_digests_verified": True,
            "stdout_bytes": log_path.stat().st_size, "stderr_bytes": (root / active["stderr"]).stat().st_size,
            "external_process_inspection": "PID47864 is the one experiment;12 spawned CPU children, no other persistent Python/GPU training task; logs advance.",
            "decision": "Keep this one registered experiment running; do not modify executed code or start another GPU task."}
        atomic_json(run / "supervision/POSTLAUNCH_VERIFICATION.json", observation)
        active.update(state=status["state"], started_at=status["started_at"], last_review=observation)
    plan["next_action"] = next_action
    plan["next_training_hypothesis"] = "Qualified256 runtime search improved strength; stronger-search collection may improve weights versus64-search training at matched evaluation budgets. Actual controlled experiment, not assumed improvement."
    plan["continuation"].update(state="optimization_running", training_active=False, next_action=next_action)
    plan["automation"]["actual_heartbeat_received_at"] = "2026-10-03T17:11:17.791Z"
    atomic_json(plan_path, plan)
    branch_plan_path = branch / "supervision/PLAN.json"
    branch_plan = read_json(branch_plan_path)
    branch_plan.update(optimization=active, next_action=next_action)
    atomic_json(branch_plan_path, branch_plan)
    review = {"at": now, "heartbeat_received": "2026-10-03T17:11:17.791Z",
        "action": "verify_paired_experiment_running" if args.verify_running else ("launch_registered_paired_search_training" if args.launch_record else "confirm_search256_and_register_controlled_training"),
        "search_256_confirmed": True, "teacher_gain": result["teacher_gain_256_vs_64"],
        "teacher_score": result["summaries"][3]["family_score"], "direct_duel": result["summaries"][0]["family_score"],
        "old_roles_preserved": True, "complete_state_source": 26, "unqualified35_preserved": True,
        "optimization": active, "next_action": next_action}
    for run in (original, branch):
        with (run / "supervision/REVIEWS.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(review, ensure_ascii=False) + "\n")
    print(json.dumps({"recorded": str(plan_path), "state": active["state"]}))


if __name__ == "__main__":
    main()
