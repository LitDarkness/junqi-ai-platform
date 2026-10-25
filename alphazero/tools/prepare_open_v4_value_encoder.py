"""Review the failed Policy encoder factor and register a Value-rate factor."""
import copy
import hashlib
import json
import statistics
from contextlib import ExitStack
from pathlib import Path

from prepare_open_v4_head_crossover import recompute_arena
from run_open_v4_search_training import route_signal, verify_learning_state


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
    prior = root / "runs/open_junqi_v4_policy_encoder_20261004"
    proof = read_json(prior / "CONTRACT.json"); result = read_json(prior / "RESULTS.json")
    progress = read_json(prior / "PROGRESS.json")
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    experiment = root / "config/experiments/open_v4_value_encoder_20261004.json"
    directory = "runs/open_junqi_v4_value_encoder_20261004"
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
        assert sha(root / "tools/run_open_v4_policy_encoder.py") == proof["tool_sha256"]
        assert sha(root / "config/experiments/open_v4_policy_encoder_20261004.json") == proof["plan_sha256"]
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
        assert not selected and not result["policy_encoder_route_confirmed"]
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
        evidence = dict(proof["evidence_files"])
        for name in ("RESULTS.json", "CONTRACT.json", "PROGRESS.json", "arena.json"):
            path = prior / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("open_v4_value_encoder_factor.py", "run_open_v4_value_encoder.py", "prepare_open_v4_value_encoder.py"):
            path = root / "tools" / name; evidence[str(path.relative_to(root))] = sha(path)
        for name in ("tests/test_open_v4_value_encoder_factor.py", "start_open_v4_value_encoder.ps1"):
            path = root / name; evidence[str(path.relative_to(root))] = sha(path)
        plan = copy.deepcopy({k:v for k,v in proof.items() if k not in ("tool_sha256", "plan_sha256", "policy_encoder_lr_multipliers")})
        plan.update(schema="junqi-value-encoder-plan-1", preregistered_at=utc_now(), directory=directory, evidence_files=evidence,
            stop_paths=stops, value_encoder_lr_multipliers={"control": .1, "candidate": .5}, evaluation_seed=2026100407,
            hypothesis="Policy-encoder freeze did not improve external strength. With normal Policy representation learning retained, test whether too-slow Value encoder adaptation limits PPV: nominal encoder multiplier .1->.5, retaining exact existing Policy KL trust region/backtracking. Baseline128 Value steps accepted at full scale, meanKL.000110/max.000327 versus limit.001; this motivates a bounded rate factor, does not prove a Value bottleneck.",
            changed_factor="Only OpenConfig.value_encoder_lr_multiplier .1 -> .5. Global lr,Policy encoder .25,PPV schedule,Value head,architecture,labels,replay,batches,source optimizer/RNG and all existing Policy KL thresholds/backtracking unchanged. Actual accepted LR scales and rejections must be reported;no relaxed guard.",
            phase_audit=audit,
            selection="Full64 matched256: both teacher paired CI lower>0 against BOTH parent and control,both duel lower>.5,duplicate point<=2pp,and unchanged full champion gate. Verify exact baseline control model and paired batches before arena;no fallback adoption or automatic promotion.",
            negative_preserved="Policy encoder factor false;candidate08223.89%[19.60,28.26] vscontrol27.67%[23.37,32.23],paired-3.78pp[-8.40,.91],parent paired0[-4.88,5.08]. Earlier crossover/teacher-target/stronger-search-data false preserved;unqualified32/35 not adopted.")
        atomic_json(experiment, plan)
        review = dict(at=utc_now(), heartbeat_received="2026-10-03T22:42:22.989Z",
            completed_at=read_json(prior / "STATUS.json")["updated_at"], policy_encoder_route_confirmed=False,
            raw_games_bootstrap_recomputed=True, actual_games=3240, paired_teacher_deltas=gains, existing_gate=gate,
            summaries=arena["summaries"], complete_checkpoints_verified=complete, source_roles_unchanged=True,
            original_control_learning_state_exact=control_exact, paired_batch_stages_verified=[1,2,3,4,5,6], phase_audit=audit,
            compute=dict(arena_seconds=arena["elapsed_seconds"], search_metrics=arena["search_metrics"], peak_gpu_allocated_mib=result["peak_gpu_allocated_mib"]),
            new_experiment=str(experiment.relative_to(root)),
            decision="No confirmed external gain from zero Policy encoder lr. Keep normal Policy learning and qualified26;preserve failed full64 route. Run one bounded Value encoder rate factor from full26,exact same64-data trajectories/batches,existing KL trust region unchanged;no blind extension or adoption.")
        atomic_json(prior / "REVIEW.json", review)
        atomic_json(root / "reports/search_budgets_20261004/POLICY_ENCODER_RESULT_REVIEW.json", review)
        print(json.dumps(dict(registered=str(experiment), raw_games_recomputed=True, policy_encoder_route_confirmed=False,
                             baseline_value_audit=audit["control"]["value"])))


if __name__ == "__main__": main()
