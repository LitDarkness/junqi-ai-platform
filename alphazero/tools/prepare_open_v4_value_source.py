"""Review the failed Value-rate factor and register a critic-source factor."""
import copy
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

from prepare_open_v4_head_crossover import recompute_arena
from run_open_v4_search_training import route_signal, verify_learning_state
from open_v4_value_source_factor import audit_shard, ALLOWED


def phase_audit(run):
    result = {}
    for variant in ("control", "candidate"):
        reports = [json.loads((run / variant / f"reports/stage_{stage:06d}.json").read_text(encoding="utf-8")) for stage in range(1, 7)]
        result[variant] = {}
        for phase in ("policy", "value"):
            rows = [r for report in reports for r in report["training"] if r["phase"] == phase]
            result[variant][phase] = dict(updates=len(rows), accepted=sum(r["accepted"] for r in rows),
                attempts_histogram={str(i): sum(r["attempts"] == i for r in rows) for i in sorted({r["attempts"] for r in rows})},
                proposed_policy_kl_mean=statistics.mean(r["kl_proposed"] for r in rows),
                proposed_policy_kl_max=max(r["kl_proposed"] for r in rows),
                proposed_family_kl_max=max(r["family_kl_proposed"] for r in rows),
                gradient_norm_mean=statistics.mean(r["gradient_norm"] for r in rows),
                accepted_lr_scales=sorted({r["accepted_lr_scale"] for r in rows}))
    return result


def main():
    import torch
    from open_v4_runtime import ProcessLock, read_json, atomic_json, utc_now
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import source_contract, promotion_decision

    root = Path(__file__).resolve().parents[1]
    prior = root / "runs/open_junqi_v4_value_encoder_20261004"
    proof = read_json(prior / "CONTRACT.json"); result = read_json(prior / "RESULTS.json")
    progress = read_json(prior / "PROGRESS.json")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    experiment = root / "config/experiments/open_v4_value_source_20261004.json"
    directory = "runs/open_junqi_v4_value_source_20261004"
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
        assert sha(root / "tools/run_open_v4_value_encoder.py") == proof["tool_sha256"]
        assert sha(root / "config/experiments/open_v4_value_encoder_20261004.json") == proof["plan_sha256"]
        assert read_json(prior / "STATUS.json")["state"] == "review_due"
        assert sha(prior / "arena.json") == progress["arena"]["sha256"]
        arena = read_json(prior / "arena.json")
        assert len(arena["games"]) == 3240 and arena["role"] == "dev"
        assert len(arena["layout_families"]) == 64 and recompute_arena(arena)
        base = OpenConfig.load(root / proof["parent_config"])
        gains = {key: teacher_deltas(arena, before, after) for key, before, after in (
            ("candidate_vs_parent", "parent", "candidate"), ("candidate_vs_control", "control", "candidate"),
            ("duplicate", "parent", "repeat"))}
        gate = promotion_decision(arena, base)
        selected = route_signal([gains["candidate_vs_parent"], gains["candidate_vs_control"]], gains["duplicate"],
                               [arena["summaries"][i]["family_score"] for i in (0, 1)], gate, 64, .02)
        assert not selected and not result["value_encoder_route_confirmed"]
        assert gains == result["teacher_deltas"] and gate == result["existing_gate"] and arena["summaries"] == result["summaries"]
        checkpoints = {"parent": proof["source_checkpoint"], **{v: {"path": str((prior / row["checkpoint"]).relative_to(root)),
            "sha256": row["sha256"], "model_hash": row["model_hash"]} for v, row in progress["variants"].items()}}
        complete, payloads = {}, {}
        for name, item in checkpoints.items():
            path = root / item["path"]; assert sha(path) == item["sha256"]
            payload = torch.load(path, map_location="cpu", weights_only=False); payloads[name] = payload
            assert all(k in payload for k in ("model", "optimizer", "scheduler", "rng", "counters", "config", "config_hash", "model_hash", "leaf_value_scale"))
            h = hashlib.sha256()
            for key, tensor in sorted(payload["model"].items()):
                h.update(key.encode()); h.update(tensor.detach().contiguous().numpy().tobytes())
            assert h.hexdigest() == payload["model_hash"] == item["model_hash"]
            assert payload["counters"]["effective_updates"] == (1664 if name == "parent" else 2048)
            assert payload["counters"]["backtracking_rejections"] == 0
            assert payload["leaf_value_scale"] == .9076346023345448
            complete[name] = {"checkpoint": item, "counters": payload["counters"], "full_state_verified": True}
        old_control = torch.load(root / "runs/open_junqi_v4_search_training_20261004/control/checkpoints/learner_stage_000006.pt", map_location="cpu", weights_only=False)
        control_exact = verify_learning_state(old_control, payloads["control"])
        for stage in range(1, 7):
            a = read_json(prior / "control" / f"reports/stage_{stage:06d}.json")
            b = read_json(prior / "candidate" / f"reports/stage_{stage:06d}.json")
            assert a["batch_digest"] == b["batch_digest"]
        for v, row in progress["variants"].items():
            assert sha(prior / row["report"]) == row["report_sha256"]
        audit = phase_audit(prior)
        assert audit["control"]["value"]["updates"] == 128 and audit["control"]["value"]["accepted"] == 128
        assert audit["control"]["value"]["proposed_family_kl_max"] < base.value_policy_kl_limit
        from junqi_search_v3.loop import read_selfplay_shard
        from collections import Counter
        family_modes, source_audit = {}, {}
        for identifier, item in proof["replay_sources"].items():
            path = root / item["path"]; meta = root / item["metadata"]
            assert sha(path) == item["sha256"] and sha(meta) == item["metadata_sha256"]
            modes, summary = audit_shard(read_selfplay_shard(path), read_json(meta))
            for family, mode in modes.items():
                if family in family_modes: assert family_modes[family] == mode
                family_modes[family] = mode
            for mode, values in summary.items():
                row = source_audit.setdefault(mode, dict(games=0, samples=0, wdl=Counter(), actor_turn_wdl=Counter(),
                    teacher_turn_wdl=Counter(), actor_score_sum=0., terminal_reasons=Counter()))
                for key, value in values.items():
                    if isinstance(value, Counter): row[key].update(value)
                    else: row[key] += value
        for mode, row in source_audit.items(): row["actor_score"] = row.pop("actor_score_sum") / row["games"]
        assert sum(r["games"] for r in source_audit.values()) == 2048
        assert sum(r["samples"] for r in source_audit.values()) == 109350
        assert source_audit["teacher_vs_actor"]["actor_score"] < .03
        evidence = dict(proof["evidence_files"])
        for name in ("RESULTS.json", "CONTRACT.json", "PROGRESS.json", "arena.json"):
            path = prior / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("open_v4_value_source_factor.py", "run_open_v4_value_source.py", "prepare_open_v4_value_source.py"):
            path = root / "tools" / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("tests/test_open_v4_value_source_factor.py", "start_open_v4_value_source.ps1"):
            path = root / name; evidence[str(path.relative_to(root))] = sha(path)
        plan = copy.deepcopy({k:v for k,v in proof.items() if k not in ("tool_sha256", "plan_sha256", "value_encoder_lr_multipliers")})
        plan.update(schema="junqi-value-source-plan-1", preregistered_at=utc_now(), directory=directory, evidence_files=evidence,
            stop_paths=stops, value_allowed_modes={"control": "all", "candidate": list(ALLOWED)}, family_modes=family_modes, source_audit=source_audit, evaluation_seed=2026100408,
            hypothesis="All prior encoder factors unconfirmed. In1016 asymmetric teacher-vs-actor training games actor score2.657%;critic targets describe a much weaker behavior while Policy labels describe strong teacher decisions. Test Value only on symmetric selfplay/teacher_selfplay trajectories,keeping all Policy data and actual terminal WDL. Outcome labels are valid;off-policy behavior mismatch is a hypothesis,not a proven label bug.",
            changed_factor="Only Value sampling eligibility excludes teacher_vs_actor trajectories. All Policy batches and full-pool Policy anchors remain identical;Value samples change intentionally. Same256 random sampler keys per attempted update,PPV384 budget,base learning rates/encoder multipliers,trust region,source states and labels unchanged. No invented outcomes,changed trajectories or development replay.",
            phase_audit=audit,
            selection="Full64 matched256: both teacher paired CI lower>0 against BOTH parent and control,both duel lower>.5,duplicate point<=2pp,and unchanged full champion gate. Verify exact baseline control model and paired batches before arena;no fallback adoption or automatic promotion.",
            negative_preserved="Value-rate primary false;candidate08225.46%[20.64,30.60] vscontrol27.41%[23.11,31.84],paired-1.95pp[-6.32,2.21];parent+1.76pp[-4.75,8.72]. Original actor gate true but champion false/strict route false;no role adoption. Every earlier false result preserved.")
        atomic_json(experiment, plan)
        review = dict(at=utc_now(), heartbeat_received="2026-10-03T23:42:23.894Z",
            completed_at=read_json(prior / "STATUS.json")["updated_at"], value_encoder_route_confirmed=False,
            raw_games_bootstrap_recomputed=True, actual_games=3240, paired_teacher_deltas=gains, existing_gate=gate,
            summaries=arena["summaries"], complete_checkpoints_verified=complete, source_roles_unchanged=True,
            original_control_learning_state_exact=control_exact, paired_batch_stages_verified=[1,2,3,4,5,6], phase_audit=audit,
            compute=dict(arena_seconds=arena["elapsed_seconds"], search_metrics=arena["search_metrics"], peak_gpu_allocated_mib=result["peak_gpu_allocated_mib"]),
            new_experiment=str(experiment.relative_to(root)),
            source_audit=source_audit,
            decision="Value-rate full64 route false despite original loose Actor gate true;no adoption. Preserve qualified26 and complete learning states. Native terminal labels valid;train-only asymmetric actor score2.657% motivates one controlled Value-source eligibility factor. Keep all Policy data/full-pool anchors,original learning settings and paired sampler RNG;no blind extension.")
        atomic_json(prior / "REVIEW.json", review)
        atomic_json(root / "reports/search_budgets_20261004/VALUE_ENCODER_RESULT_REVIEW.json", review)
        print(json.dumps(dict(registered=str(experiment), raw_games_recomputed=True, value_encoder_route_confirmed=False,
                             source_audit=source_audit)))


if __name__ == "__main__": main()
