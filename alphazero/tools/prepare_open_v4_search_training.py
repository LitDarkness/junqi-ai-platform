"""Verify the completed search qualification and register a new paired experiment."""
from __future__ import annotations

import hashlib
import json
from contextlib import ExitStack
from pathlib import Path


def main():
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, read_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from evaluate_open_v4_search_confirmation import full_budget_confirmed
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import source_contract, promotion_decision
    from run_open_v4_search_training import variant_config, verify_learning_state
    from fork_open_v4_qualified import fork_payload

    root = Path(__file__).resolve().parents[1]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    confirmation = root / "runs/open_junqi_v4_search_confirmation_20261004"
    proof = read_json(confirmation / "CONTRACT.json")
    config_path = root / proof["config"]
    base = OpenConfig.load(config_path)
    frozen_runs = proof["frozen_runs"]
    experiment = root / "config/experiments/open_v4_search_training_20261004.json"
    new_run = "runs/open_junqi_v4_search_training_20261004"
    stops = proof["stop_paths"] + [str((confirmation / "STOP").relative_to(root))]
    if any((root / p).exists() for p in stops):
        raise RuntimeError("STOP requested; no automatic clearing")
    if experiment.exists() or (root / new_run).exists():
        raise ValueError("preserve existing registration/experiment")
    with ExitStack() as locks:
        for record in frozen_runs:
            run = root / record["directory"]
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
            if read_json(run / "CURRENT.json") != record["current"]:
                raise ValueError("frozen roles changed")
            contract = read_json(run / "CONTRACT.json")
            if sha(run / "CONTRACT.json") != record["current"]["contract_hash"]:
                raise ValueError("frozen contract hash changed")
            if source_contract(root) != contract["source"] or contract["source"] != proof["source"]:
                raise ValueError("core/native source changed")
            for role in ("learner", "actor", "champion"):
                row = record["current"][role]
                if sha(run / row["path"]) != row["sha256"]:
                    raise ValueError("frozen checkpoint digest mismatch")
        if sha(root / "tools/evaluate_open_v4_search_confirmation.py") != proof["tool_sha256"]:
            raise ValueError("qualification tool changed")
        if sha(root / "config/experiments/open_v4_search_confirmation_20261004.json") != proof["plan_sha256"]:
            raise ValueError("qualification plan changed")
        for path, digest in proof["evidence_files"].items():
            if sha(root / path) != digest:
                raise ValueError("screening evidence/dependency changed")
        arena = read_json(confirmation / "arena.json")
        result = read_json(confirmation / "RESULTS.json")
        gain = teacher_deltas(arena, "baseline", "candidate")
        duplicate = teacher_deltas(arena, "baseline", "repeat")
        gate = promotion_decision(arena, base)
        if not result["search_256_confirmed"] or not full_budget_confirmed(gain, duplicate,
                arena["summaries"][0]["family_score"], gate, 64, .02):
            raise ValueError("full64 search qualification failed")
        if gain != result["teacher_gain_256_vs_64"] or duplicate != result["duplicate_control"] or gate != result["existing_gate"]:
            raise ValueError("raw arena/result gate discrepancy")
        source = proof["checkpoint"]
        payload = torch.load(root / source["path"], map_location="cpu", weights_only=False)
        if payload["config_hash"] != base.digest() or payload["counters"]["iteration"] != 26 or payload["counters"]["effective_updates"] != 1664:
            raise ValueError("qualified complete state mismatch")
        initial = []
        branch = root / base.run_directory
        for i in (24, 25, 26):
            path = branch / f"replay/iteration_{i:06d}.npz"
            metadata = read_json(path.with_suffix(".json"))
            if metadata["layout_role"] != "train" or metadata["config_hash"] != base.digest() or metadata["shard_sha256"] != sha(path):
                raise ValueError("source replay contract changed")
            initial.append({"iteration": i, "path": str(path.relative_to(root)), "sha256": sha(path),
                "metadata_sha256": sha(path.with_suffix(".json"))})
        evidence_files = dict(proof["evidence_files"])
        for path in [confirmation / name for name in ("RESULTS.json", "arena.json", "CONTRACT.json")]:
            evidence_files[str(path.relative_to(root))] = sha(path)
        for name in ("evaluate_open_v4_search_confirmation.py", "fork_open_v4_qualified.py", "run_open_v4_fresh_data.py"):
            path = root / "tools" / name
            evidence_files[str(path.relative_to(root))] = sha(path)
        checks = {}
        for variant in ("control", "candidate"):
            cfg = variant_config(base, new_run, source["path"], variant)
            checks[variant] = verify_learning_state(payload, fork_payload(payload, cfg, {"source": source}))
        plan = {"schema": "junqi-search-data-training-plan-1", "preregistered_at": utc_now(),
            "parent_config": proof["config"], "parent_config_hash": base.digest(), "directory": new_run,
            "frozen_runs": frozen_runs, "source": proof["source"], "teachers": proof["teachers"],
            "source_checkpoint": source, "source_counters": payload["counters"],
            "qualification_result": str((confirmation / "RESULTS.json").relative_to(root)),
            "layout_manifest_hash": proof["layout_manifest_hash"], "suite": proof["suite"],
            "initial_replay": initial, "initial_replay_iterations": [24, 25, 26],
            "first_collection_iteration": 45, "previous_max_collection_iteration": 44,
            "stages": 6, "games_per_variant_per_stage": 256, "updates_per_variant_per_stage": 64,
            "collection_budgets": {"control": [64, 128], "candidate": [256, 256]},
            "evaluation_budgets": {"primary": 256, "secondary": 64},
            "evaluation_families": {"primary": 64, "secondary": 32}, "evaluation_seed": 2026100403,
            "duplicate_control_tolerance": .02, "minimum_free_gib": 10,
            "evidence_files": evidence_files, "stop_paths": stops,
            "hypothesis": "Stronger search during collection may improve learning over64/128 data, with games, PPV updates, window and complete starting state fixed.",
            "changed_factor": "Collection search budget vector64/128 versus256/256; trajectories and selfplay visit targets may differ. Teacher hard labels stay unchanged.",
            "controlled": "Frozen qualified Actor26, identical paired train layout/color/mode/teacher/root-noise seeds,1536 games and384 updates per variant; common initial replay; fresh collection namespace45-50.",
            "primary_selection": "At matched256 evaluation on full64 development families: both teacher paired gain CI95 lower bounds >0 versus parent and control, both direct duels CI95 lower bounds >0.5, duplicate point deltas <=2pp, unchanged full champion gate. No automatic promotion.",
            "secondary": "Matched64 on32 families is diagnostic only, never selection or a fallback for a failed primary result.",
            "development_caveat": "Existing repeatedly used development corpus; exploratory optimization, no final layout use or independent final-validation claim.",
            "failure_policy": "No automatic retries. Explicit reviewed --resume only from hashed complete progress. Uncommitted shards/checkpoints require inspection; never overwrite silently.",
            "prepare_tool_sha256": sha(Path(__file__))}
        atomic_json(experiment, plan)
        review = {"at": utc_now(), "completed_at": read_json(confirmation / "STATUS.json")["updated_at"],
            "search_256_confirmed": True, "gate_recomputed_from_raw_arena": gate,
            "teacher_gain": gain, "duel": arena["summaries"][0]["family_score"],
            "teacher_summaries": [r for r in arena["summaries"] if r["opponent"] == "teacher0"],
            "frozen_checkpoint_and_core_digests_verified": True, "evidence_files": evidence_files,
            "source_complete_learning_state_migration_checks": checks,
            "decision": "Adopt qualified256 budget for independent paired collection experiment; preserve all old roles and failed primary128. No claim of new trained weights or monotonic scaling.",
            "experiment": str(experiment.relative_to(root))}
        atomic_json(confirmation / "REVIEW.json", review)
        atomic_json(root / "reports/search_budgets_20261004/FULL_CONFIRMATION_REVIEW.json", review)
        print(json.dumps({"registered": str(experiment), "search_256_confirmed": True, "gate": gate}, ensure_ascii=False))


if __name__ == "__main__":
    main()
