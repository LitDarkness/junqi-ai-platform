"""Audit the completed Value-source result, then preregister actor refresh."""
import copy
import hashlib
import json
from contextlib import ExitStack
from pathlib import Path

from prepare_open_v4_head_crossover import recompute_arena
from prepare_open_v4_value_source import phase_audit
from run_open_v4_search_training import route_signal, verify_learning_state


def main():
    import torch
    from open_v4_runtime import ProcessLock, read_json, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import source_contract, promotion_decision

    root = Path(__file__).resolve().parents[1]
    prior = root / "runs/open_junqi_v4_value_source_20261004"
    proof = read_json(prior / "CONTRACT.json")
    result = read_json(prior / "RESULTS.json")
    progress = read_json(prior / "PROGRESS.json")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    experiment = root / "config/experiments/open_v4_actor_refresh_20261004.json"
    directory = "runs/open_junqi_v4_actor_refresh_20261004"
    if experiment.exists() or (root / directory).exists():
        raise ValueError("preserve existing registration or experiment")
    stops = proof["stop_paths"] + [str((prior / "STOP").relative_to(root)),
                                    str((prior / "supervision/STOP").relative_to(root))]
    if any((root / p).exists() for p in stops):
        raise RuntimeError("STOP requested; no clearing")
    with ExitStack() as locks:
        for frozen in proof["frozen_runs"]:
            run = root / frozen["directory"]
            for name in ("SUPERVISOR", "TRAINING"):
                locks.enter_context(ProcessLock(run / f"supervision/{name}.lock"))
            assert read_json(run / "CURRENT.json") == frozen["current"]
            assert sha(run / "CONTRACT.json") == frozen["current"]["contract_hash"]
            for role in ("learner", "actor", "champion"):
                item = frozen["current"][role]
                assert sha(run / item["path"]) == item["sha256"]
        assert source_contract(root) == proof["source"]
        for path, digest in proof["evidence_files"].items():
            assert sha(root / path) == digest
        assert sha(root / "tools/run_open_v4_value_source.py") == proof["tool_sha256"]
        assert sha(root / "config/experiments/open_v4_value_source_20261004.json") == proof["plan_sha256"]
        assert read_json(prior / "STATUS.json")["state"] == "review_due"
        assert sha(prior / "arena.json") == progress["arena"]["sha256"]
        arena = read_json(prior / "arena.json")
        assert len(arena["games"]) == 3240 and arena["role"] == "dev"
        assert len(arena["layout_families"]) == 64 and recompute_arena(arena)
        base = OpenConfig.load(root / proof["parent_config"])
        assert base.digest() == proof["parent_config_hash"]
        assert sha(root / base.layout_directory / "manifest.json") == proof["layout_manifest_hash"]
        assert sha(root / proof["suite"]["path"]) == proof["suite"]["sha256"]
        gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
            ("candidate_vs_parent", "parent", "candidate"),
            ("candidate_vs_control", "control", "candidate"), ("duplicate", "parent", "repeat"))}
        gate = promotion_decision(arena, base)
        selected = route_signal([gains["candidate_vs_parent"], gains["candidate_vs_control"]], gains["duplicate"],
                                [arena["summaries"][i]["family_score"] for i in (0, 1)], gate, 64, .02)
        assert not selected and not result["value_source_route_confirmed"]
        assert gains == result["teacher_deltas"] and gate == result["existing_gate"]
        assert arena["summaries"] == result["summaries"]
        checkpoints = {"parent": proof["source_checkpoint"], **{v: {
            "path": str((prior / row["checkpoint"]).relative_to(root)),
            "sha256": row["sha256"], "model_hash": row["model_hash"]} for v, row in progress["variants"].items()}}
        complete, payloads = {}, {}
        for name, item in checkpoints.items():
            path = root / item["path"]
            assert sha(path) == item["sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payloads[name] = payload
            assert all(k in payload for k in ("model", "optimizer", "scheduler", "rng", "counters",
                                              "config", "config_hash", "model_hash", "leaf_value_scale"))
            h = hashlib.sha256()
            for key, tensor in sorted(payload["model"].items()):
                h.update(key.encode()); h.update(tensor.detach().contiguous().numpy().tobytes())
            assert h.hexdigest() == payload["model_hash"] == item["model_hash"]
            assert payload["counters"]["effective_updates"] == (1664 if name == "parent" else 2048)
            assert payload["counters"]["backtracking_rejections"] == 0
            assert payload["leaf_value_scale"] == .9076346023345448
            complete[name] = {"checkpoint": item, "counters": payload["counters"], "full_state_verified": True}
        old = root / "runs/open_junqi_v4_search_training_20261004/control/checkpoints/learner_stage_000006.pt"
        exact = verify_learning_state(torch.load(old, map_location="cpu", weights_only=False), payloads["control"])
        for stage in range(1, 7):
            a = read_json(prior / "control" / f"reports/stage_{stage:06d}.json")
            b = read_json(prior / "candidate" / f"reports/stage_{stage:06d}.json")
            assert a["phase_batch_digests"]["policy"] == b["phase_batch_digests"]["policy"]
            assert a["sampling_key_digest"] == b["sampling_key_digest"]
        for v, row in progress["variants"].items():
            assert sha(prior / row["report"]) == row["report_sha256"]
        audit = phase_audit(prior)
        collectors = []
        for identifier, item in proof["replay_sources"].items():
            assert sha(root / item["path"]) == item["sha256"]
            assert sha(root / item["metadata"]) == item["metadata_sha256"]
            meta = read_json(root / item["metadata"])
            assert meta["layout_role"] == "train"
            if identifier.startswith("new_"):
                assert meta["actor_model_hash"] == proof["source_checkpoint"]["model_hash"]
                assert meta["game_count"] == 256
                collectors.append({"collection": meta["iteration"], "actor_model_hash": meta["actor_model_hash"],
                                   "games": meta["game_count"], "modes": meta["label_sources"]})
        assert len(collectors) == 6
        for item in proof["validation_replay"]:
            assert sha(root / item["path"]) == item["sha256"]
            assert sha((root / item["path"]).with_suffix(".json")) == item["metadata_sha256"]
        evidence = dict(proof["evidence_files"])
        for name in ("RESULTS.json", "CONTRACT.json", "PROGRESS.json", "arena.json"):
            path = prior / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("open_v4_actor_refresh_factor.py", "run_open_v4_actor_refresh.py", "prepare_open_v4_actor_refresh.py"):
            path = root / "tools" / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("tests/test_open_v4_actor_refresh_factor.py", "start_open_v4_actor_refresh.ps1"):
            path = root / name; evidence[str(path.relative_to(root))] = sha(path)
        plan = copy.deepcopy({k:v for k,v in proof.items() if k not in ("tool_sha256", "plan_sha256")})
        plan.update(schema="junqi-actor-refresh-plan-1", preregistered_at=utc_now(), directory=directory,
            evidence_files=evidence, stop_paths=stops, evaluation_seed=2026100409,
            baseline_control=checkpoints["control"], value_allowed_modes={"control": "all", "candidate": "all"},
            collector_refresh={"control": "qualified26", "candidate": "previous_complete_learner"},
            collector_staleness_audit=collectors,
            hypothesis="All critic/encoder factors unconfirmed. The six64-data collections use frozen26 throughout384 learning attempts. Test collecting network-dependent trajectories from immediately preceding full learner after each64 attempts, keeping production26 roles frozen. This tests data freshness relative to learning, not automatic production promotion; actor staleness is observed, its causal impact is unproven.",
            changed_factor="Only experimental collection actor follows previous complete learner, stage2-6. Stage1 and initial25/26 replay are identical frozen sources. Control reuses frozen64/128 trajectories; candidate generates five256-game collections at paired46-50 seeds/starts/colors/modes/teachers, same64/128 search, then64 PPV attempts. All original labels, learning rates, source modes, trust region, RNG keys, Value scale, replay window and budget unchanged. No new eligibility filter. Later actual batches intentionally differ because trajectories differ; only sampler keys remain paired.",
            data_budget={"source_games":2048, "control_new_games":0, "candidate_new_games":1280,
                         "borrowed_candidate_initial_games":768, "attempted_updates_each":384,
                         "first_stage_new_games":0, "later_candidate_games_per64_attempts":256},
            collection_number_policy="Explicit reuse of paired45-50 family/seed schedule in independent output directory; never overwrite old shards. Stage1 uses original45; candidate46-50 have new recorded actor checkpoint/model/counters and real terminal WDL. These are new trajectories at existing training layouts, not independent new layout families.",
            selection="Full64matched256: both teacher pairedCI lower>0 against BOTH parent/control,both direct CI lower>.5,duplicate<=2pp,and unchanged full champion gate. Control must reproduce ALL original learning fields and first-stage branches must be identical. No automatic promotion/fallback; raw results and complete states first.",
            compute_caveat="Frozen recorded control reused to avoid redundant games. Native teacher timing/nondeterminism can contribute collection noise; retain seeds and full trajectories, report duplicate evaluation and avoid claiming causal proof from one run. Equal learning-attempt and search budgets, not equal fresh-data compute. Separate source games, genuinely new games, throughput, time and memory.",
            negative_preserved="Value-source full64 route false, both original gates false. Candidate08223.70%[19.21,28.26],control28.13%[23.76,32.49],paired-4.43pp[-8.79,0.00];parent-0.07pp[-5.53,5.08]. Directparent51.37%[46.74,55.86],control46.09%[41.80,50.91]. Earlier failures and qualified26 remain unchanged.")
        review = dict(at=utc_now(), heartbeat_received="2026-10-04T00:42:24.741Z",
            completed_at=read_json(prior / "STATUS.json")["updated_at"], value_source_route_confirmed=False,
            raw_games_bootstrap_recomputed=True, actual_games=3240, paired_teacher_deltas=gains, existing_gate=gate,
            summaries=arena["summaries"], complete_checkpoints_verified=complete, source_roles_unchanged=True,
            original_control_learning_state_exact=exact, paired_policy_and_sampler_key_stages_verified=list(range(1,7)),
            value_batches_intentionally_different=True, phase_audit=audit, collector_staleness_audit=collectors,
            source_outcome_audit=proof["source_audit"],
            compute=dict(arena_seconds=arena["elapsed_seconds"], search_metrics=arena["search_metrics"],
                         peak_gpu_allocated_mib=result["peak_gpu_allocated_mib"]), new_experiment=str(experiment.relative_to(root)),
            decision="Value-source route false and both original gates false;no role adoption. Retain all complete checkpoints and raw negative evidence. Change one experimental collection-actor freshness factor at original64/128 budget, with original full learner and a frozen exact control;no blind extension of failed32/35.")
        atomic_json(prior / "REVIEW.json", review)
        atomic_json(root / "reports/search_budgets_20261004/VALUE_SOURCE_RESULT_REVIEW.json", review)
        evidence[str((prior / "REVIEW.json").relative_to(root))] = sha(prior / "REVIEW.json")
        atomic_json(experiment, plan)
        print(json.dumps(dict(registered=str(experiment), raw_games_recomputed=True,
            value_source_route_confirmed=False, collectors=collectors, paired_teacher_deltas=gains), ensure_ascii=False))


if __name__ == "__main__": main()
