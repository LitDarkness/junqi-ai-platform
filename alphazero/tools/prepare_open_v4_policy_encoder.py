"""Review the completed frozen crossover and preregister one learning factor."""
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from prepare_open_v4_head_crossover import recompute_arena
from evaluate_open_v4_head_crossover import head_primary_signal


def main():
    import torch
    from open_v4_runtime import ProcessLock, read_json, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import source_contract

    root = Path(__file__).resolve().parents[1]
    prior = root / "runs/open_junqi_v4_head_crossover_20261004"
    proof = read_json(prior / "CONTRACT.json")
    result = read_json(prior / "RESULTS.json")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    experiment = root / "config/experiments/open_v4_policy_encoder_20261004.json"
    directory = "runs/open_junqi_v4_policy_encoder_20261004"
    if experiment.exists() or (root / directory).exists():
        raise ValueError("preserve existing registration or experiment")
    stops = proof["stop_paths"] + [str((prior / "STOP").relative_to(root))]
    if any((root / p).exists() for p in stops): raise RuntimeError("STOP requested; no clearing")
    with ExitStack() as locks:
        for row in proof["frozen_runs"]:
            run = root / row["directory"]
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
            assert read_json(run / "CURRENT.json") == row["current"]
            assert sha(run / "CONTRACT.json") == row["current"]["contract_hash"]
            for role in ("learner", "actor", "champion"):
                assert sha(run / row["current"][role]["path"]) == row["current"][role]["sha256"]
        assert source_contract(root) == proof["source"]
        for path, digest in proof["evidence_files"].items(): assert sha(root / path) == digest
        assert sha(root / "tools/evaluate_open_v4_head_crossover.py") == proof["tool_sha256"]
        assert sha(root / "config/experiments/open_v4_head_crossover_20261004.json") == proof["plan_sha256"]
        assert read_json(prior / "STATUS.json")["state"] == "review_due"
        arena = read_json(prior / "arena.json")
        assert len(arena["games"]) == 1920 and arena["role"] == "dev"
        assert len(arena["layout_families"]) == 32 and recompute_arena(arena)
        gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
            ("hybrid_vs_parent", "parent", "hybrid"), ("hybrid_vs_trained", "trained", "hybrid"),
            ("value_changed_vs_parent", "parent", "value_changed"), ("trained_vs_parent", "parent", "trained"),
            ("duplicate", "parent", "repeat"))}
        selected = head_primary_signal([gains["hybrid_vs_parent"], gains["hybrid_vs_trained"]], gains["duplicate"],
                                      [arena["summaries"][i]["family_score"] for i in (0, 1)], 32, .02)
        assert not selected and not result["primary_frozen_value_signal"]
        assert gains == result["teacher_deltas"] and arena["summaries"] == result["summaries"]
        complete = {}
        for name, item in proof["checkpoints"].items():
            path = root / item["path"]; assert sha(path) == item["sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            assert all(k in payload for k in ("model", "optimizer", "scheduler", "rng", "counters", "config", "config_hash", "model_hash", "leaf_value_scale"))
            h = hashlib.sha256()
            for key, tensor in sorted(payload["model"].items()):
                h.update(key.encode()); h.update(tensor.detach().contiguous().numpy().tobytes())
            assert h.hexdigest() == payload["model_hash"] == item["model_hash"]
            assert payload["counters"]["effective_updates"] == (1664 if name == "parent" else 2048)
            assert payload["counters"]["backtracking_rejections"] == 0
            assert payload["leaf_value_scale"] == .9076346023345448
            complete[name] = {"checkpoint": item, "counters": payload["counters"], "full_state_verified": True}
        old = read_json(root / "config/experiments/open_v4_teacher_targets_20261004.json")
        base = OpenConfig.load(root / proof["parent_config"])
        replay = dict(old["replay_sources"])
        games, rows = 0, 0
        for identifier, item in replay.items():
            if identifier.startswith("new_"):
                number = int(identifier.split("_")[1])
                path = root / f"runs/open_junqi_v4_search_training_20261004/control/replay/collection_{number:06d}.npz"
                meta = path.with_suffix(".json")
                item = dict(path=str(path.relative_to(root)), sha256=sha(path), metadata=str(meta.relative_to(root)), metadata_sha256=sha(meta))
                replay[identifier] = item
            assert sha(root / item["path"]) == item["sha256"] and sha(root / item["metadata"]) == item["metadata_sha256"]
            metadata = read_json(root / item["metadata"])
            assert metadata["layout_role"] == "train" and metadata["shard_sha256"] == item["sha256"]
            assert len(metadata["games"]) == 256
            games += len(metadata["games"])
            rows += sum(len(game["trace"]) for game in metadata["games"])
        evidence = dict(proof["evidence_files"])
        for name in ("RESULTS.json", "CONTRACT.json", "arena.json", "CALIBRATION.json"):
            path = prior / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("evaluate_open_v4_head_crossover.py", "prepare_open_v4_head_crossover.py",
                     "open_v4_policy_encoder_factor.py", "run_open_v4_policy_encoder.py", "prepare_open_v4_policy_encoder.py"):
            path = root / "tools" / name; evidence[str(path.relative_to(root))] = sha(path)
        path = root / "tests/test_open_v4_policy_encoder_factor.py"; evidence[str(path.relative_to(root))] = sha(path)
        parent = torch.load(root / proof["checkpoints"]["parent"]["path"], map_location="cpu", weights_only=False)
        plan = dict(schema="junqi-policy-encoder-plan-1", preregistered_at=utc_now(), directory=directory,
            parent_config=proof["parent_config"], parent_config_hash=base.digest(), source_checkpoint=proof["checkpoints"]["parent"],
            source_counters=parent["counters"], frozen_runs=proof["frozen_runs"], source=proof["source"], evidence_files=evidence,
            teachers=proof["teachers"], layout_manifest_hash=proof["layout_manifest_hash"], suite=proof["suite"],
            validation_replay=old["validation_replay"], replay_sources=replay, stop_paths=stops,
            stages=6, updates_per_stage=64, policy_encoder_lr_multipliers={"control": .25, "candidate": 0.},
            expected_control_model_hash=proof["checkpoints"]["trained"]["model_hash"],
            evaluation_families=64, evaluation_budget=256, evaluation_seed=2026100406, duplicate_control_tolerance=.02,
            hypothesis="Policy phases can alter shared-encoder Value despite fixed Value head. Test zero Policy encoder lr while keeping normal Value updates, against exact64-data learning control. Crossover did not support a static old Value replacement; this dynamic phase factor remains unproven.",
            changed_factor="Only encoder lr during Policy phases .25 -> 0. Encoder still receives gradients and Adam moment updates; global gradient clipping and all Value-phase learning/Policy KL protections unchanged. No architecture, labels, sampling, schedule, source state, Value scale or search change.",
            data_budget=dict(new_training_games=0, frozen_source_games=games, frozen_source_rows=rows, warm_games=512, existing64_data_games=1536, attempted_updates_per_variant=384),
            selection="Full64 matched256: both teacher paired CI lower>0 against BOTH parent and control, both duel lower>.5,duplicate point<=2pp,and unchanged full champion gate. Verify exact control tensor reproduction and paired batches before arena; no fallback adoption or automatic promotion.",
            negative_preserved="Frozen crossover primary false;hybrid08227.34%[20.70,34.51] vs fulltrained29.43%[23.31,35.42],paired-2.08pp[-9.38,5.34]. Both previous training route failures remain false; no unqualified32 or35 adopted.",
            failure_policy="No automatic retries or overwrite. Inspect STOPs,all locks,logs and committed complete state before one explicit reviewed recovery; completed result cannot be overwritten.",
            development_caveat="Repeated exploratory development corpus; no independent final claim. Final layouts and development failure trajectories never enter replay.")
        atomic_json(experiment, plan)
        review = dict(at=utc_now(), heartbeat_received="2026-10-03T21:42:21.910Z",
            completed_at=read_json(prior / "STATUS.json")["updated_at"], primary_frozen_value_signal=False,
            raw_games_bootstrap_recomputed=True, actual_games=1920, paired_teacher_deltas=gains,
            summaries=arena["summaries"], complete_checkpoints_verified=complete, source_roles_unchanged=True,
            compute=dict(arena_seconds=arena["elapsed_seconds"], search_metrics=arena["search_metrics"], peak_gpu_allocated_mib=result["peak_gpu_allocated_mib"]),
            new_experiment=str(experiment.relative_to(root)),
            decision="No confirmed gain from frozen old Value; preserve negative primary and qualified26. Isolate Policy-phase encoder lr using identical64-data trajectories and exact learning state; Value still learns, no static old Value adoption.")
        atomic_json(prior / "REVIEW.json", review)
        atomic_json(root / "reports/search_budgets_20261004/HEAD_CROSSOVER_RESULT_REVIEW.json", review)
        print(json.dumps(dict(registered=str(experiment), raw_games_recomputed=True, primary_frozen_value_signal=False, frozen_source_games=games, frozen_source_rows=rows)))


if __name__ == "__main__": main()
