"""Register a new, authorized branch while preserving all stopped runs."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def main():
    from open_v4_runtime import atomic_json
    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    read = lambda p: json.loads(p.read_text(encoding="utf-8-sig"))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    directory = f"runs/open_junqi_v4_self_training_{stamp}"
    old = read(root / "config/experiments/open_v4_actor_refresh_20261004.json")
    progress = read(root / old["directory"] / "PROGRESS.json")
    participants = {"parent": old["source_checkpoint"]}
    for name in ("control", "candidate"):
        row = progress["variants"][name]
        if row["stage"] != 6: raise ValueError("actor-refresh training incomplete")
        participants[name] = {"path": str(Path(old["directory"]) / row["checkpoint"]),
                              "sha256": row["sha256"], "model_hash": row["model_hash"]}
    participants["repeat"] = participants["parent"]
    for row in participants.values():
        if sha(root / row["path"]) != row["sha256"]: raise ValueError("participant digest changed")
    replay = [old["replay_sources"][f"initial_{iteration:06d}"] for iteration in (25, 26)]
    sources = ["tools/run_open_v4_self_training.py", "tools/prepare_open_v4_self_training.py",
               "python/junqi_open_v4/config.py", "python/junqi_open_v4/games.py",
               "python/junqi_open_v4/training_replay.py", "python/junqi_open_v4/learner.py",
               "python/junqi_open_v4/arena.py", "python/junqi_open_v4/layouts.py",
               "python/junqi_open_v4/teachers.py", "python/junqi_search_v3/loop.py",
               "python/junqi_search_v3/search.py", "build/JunqiAZ.cp310-win_amd64.pyd",
               "tools/fork_open_v4_qualified.py", "tools/run_open_v4_search_training.py",
               "tools/open_v4_experiment_evidence.py", "tools/open_v4_runtime.py",
               "tools/evaluate_open_v4_search_budgets.py", "python/junqi_open_v4/pipeline.py"]
    frozen = []
    for name in ("open_junqi_v4_repair_pilot", "open_junqi_v4_fresh_continuation"):
        run = f"runs/{name}"
        frozen.append({"directory": run, "current_sha256": sha(root / run / "CURRENT.json")})
    plan = {"schema": "junqi-self-training-plan-1", "registered_at": datetime.now(timezone.utc).isoformat(),
        "directory": directory, "parent_config": old["parent_config"],
        "parent_config_sha256": sha(root / old["parent_config"]),
        "source_checkpoint": old["source_checkpoint"], "participants": participants,
        "source_counters": old["source_counters"],
        "suite": old["suite"], "teachers": old["teachers"],
        "layout_manifest_hash": old["layout_manifest_hash"], "initial_replay": replay,
        "frozen_runs": frozen, "source": {name: sha(root / name) for name in sources},
        "authorization": "User explicitly authorized evaluating all proposed work, fixing duplicate trajectories and starting training after the previous batch stop. New branch only; old STOP markers and checkpoints remain preserved.",
        "historical_stop_markers": [str(p.relative_to(root)) for p in (root / "runs").glob("*/**/STOP")],
        "stop_paths": [f"{directory}/STOP", f"{directory}/supervision/STOP"],
        "evaluation_seed": old["evaluation_seed"], "screen_families": 16, "full_families": 64,
        "evaluation_budget": 256, "search_diagnostic_positions": 48,
        "collection_start": 61, "stages_per_block": 6, "maximum_stages": 12,
        "games_per_stage": 256, "updates_per_stage": 64, "replay_stages": 3,
        "control": {"selfplay_fraction": .25, "teacher_selfplay_fraction": None,
                    "actor_policy_target": "teacher", "deduplicate": False, "neural_policy_only": False},
        "candidate": {"selfplay_fraction": .4, "teacher_selfplay_fraction": .1,
                      "actor_policy_target": "search", "deduplicate": True, "neural_policy_only": True},
        "collector": "Both experimental branches use their immediately preceding complete learner; no production role adoption from collection alone.",
        "hypothesis": "A self-improvement data package: all eligible Policy targets come from neural search, teacher turns are Value data, exact full trajectories deduplicated, more neural selfplay. Multiple package components vary; this experiment tests the package, not single-component causality.",
        "screen_rule": "Fixed16 families; no promotion. Expand only when both teacher mean deltas over parent and control are positive and both direct mean scores exceed0.5. Screening may miss a real gain; failed screening means unconfirmed, not proven regression.",
        "promotion_rule": "Unchanged full64 primary: both teacher paired CI95 lower>0 over parent and matched control, both direct CI95 lower>0.5, duplicate-parent point delta<=0.02, original full Champion gate. Never lower thresholds or use final layouts.",
        "failure_policy": "Stop for review at first unconfirmed block; no blind extension, retries or fallback weight adoption.",
        "validation_policy": "Training-family internal holdout monitoring only. Development/final trajectories never enter replay; final layouts unused."}
    path = root / f"config/experiments/open_v4_self_training_{stamp}.json"
    if path.exists() or (root / directory).exists(): raise ValueError("preserve existing branch")
    atomic_json(path, plan)
    print(json.dumps({"plan": str(path.relative_to(root)), "directory": directory}))


if __name__ == "__main__": main()
