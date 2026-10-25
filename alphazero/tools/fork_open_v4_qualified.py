"""Prepare a qualified continuation without resetting any learning state."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path


def fork_payload(payload, config, evidence):
    result = copy.deepcopy(payload)
    result.update(role="learner", config=config.as_dict(), config_hash=config.digest(),
        provenance={"prior_provenance": payload["provenance"], "full_state_continuation": evidence,
            "optimizer_reset": False, "scheduler_reset": False, "rng_reset": False},
        lineage={"continuation": evidence, "previous_lineage": payload["lineage"]})
    return result


def main():
    import numpy as np
    import torch
    from open_v4_runtime import ProcessLock, atomic_json, utc_now
    from evaluate_open_v4_confirmation import route_confirmed
    from open_v4_experiment_evidence import teacher_deltas
    from junqi_open_v4.config import OpenConfig
    from junqi_open_v4.pipeline import source_contract, promotion_decision
    from junqi_open_v4.learner import atomic_checkpoint
    from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard

    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmation", default="runs/open_junqi_v4_fresh_confirmation_20261003")
    parser.add_argument("--directory", default="runs/open_junqi_v4_fresh_continuation")
    parser.add_argument("--config", default="config/open_junqi_v4_fresh_continuation.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    confirmation = (root / args.confirmation).resolve()
    directory = (root / args.directory).resolve()
    config_path = (root / args.config).resolve()
    if any(not p.is_relative_to(root) or p == root for p in (confirmation, directory, config_path)):
        raise ValueError("continuation paths must remain in alphazero")
    if directory.exists() or config_path.exists():
        raise ValueError("preserve existing continuation; inspect it before any recovery")
    result = json.loads((confirmation / "RESULTS.json").read_text(encoding="utf-8"))
    plan = result["experiment"]
    original_config = OpenConfig.load(root / plan["parent_config"])
    original = root / original_config.run_directory
    prior = (root / plan["participants"]["candidate"]["path"]).parents[2]
    if any(p.exists() for p in (original / "supervision/STOP", confirmation / "STOP", prior / "STOP")):
        raise RuntimeError("STOP requested; no continuation")
    with ProcessLock(original / "supervision/SUPERVISOR.lock"), ProcessLock(original / "supervision/TRAINING.lock"):
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        original_current = json.loads((original / "CURRENT.json").read_text(encoding="utf-8"))
        original_contract = json.loads((original / "CONTRACT.json").read_text(encoding="utf-8"))
        proof_contract = json.loads((confirmation / "CONTRACT.json").read_text(encoding="utf-8"))
        if original_current != plan["parent_current"] or sha(original / "CONTRACT.json") != original_current["contract_hash"]:
            raise ValueError("original state changed")
        if source_contract(root) != plan["source"] or sha(root / "tools/evaluate_open_v4_confirmation.py") != proof_contract["tool_sha256"]:
            raise ValueError("evaluation or training source changed")
        for role in ("learner", "actor", "champion"):
            if sha(original / original_current[role]["path"]) != original_current[role]["sha256"]:
                raise ValueError("original role digest mismatch")
        for record in plan["participants"].values():
            if sha(root / record["path"]) != record["sha256"]:
                raise ValueError("participant changed after evaluation")
        arena = json.loads((confirmation / "arena.json").read_text(encoding="utf-8"))
        gate = promotion_decision(arena, original_config)
        summaries = arena["summaries"]
        deltas = {name: teacher_deltas(arena, before, after) for name, before, after in (
            ("parent", "parent", "candidate"), ("control", "control", "candidate"), ("duplicate", "parent", "parent_repeat"))}
        if not result["fresh_route_confirmed"] or not route_confirmed(deltas["parent"], deltas["control"],
                deltas["duplicate"], gate, 64, plan["duplicate_control_tolerance"], [summaries[i]["family_score"] for i in (1, 2)]):
            raise ValueError("full confirmation did not qualify the route")
        source_path = root / plan["participants"]["candidate"]["path"]
        payload = torch.load(source_path, map_location="cpu", weights_only=False)
        for field in ("model", "optimizer", "scheduler", "counters", "rng", "config", "config_hash", "leaf_value_scale"):
            if field not in payload:
                raise ValueError("incomplete source learning state")
        if payload["counters"]["iteration"] != 26 or payload["counters"]["effective_updates"] != 1664:
            raise ValueError("unexpected continuation counters")
        cfg = OpenConfig.load(prior / "fresh/config.json")
        if cfg.digest() != payload["config_hash"] or cfg.games_per_iteration != 256:
            raise ValueError("source configuration mismatch")
        cfg.run_directory = str(directory.relative_to(root))
        cfg.initial_checkpoint = str(source_path.relative_to(root))
        cfg.validate()
        changed = {k: {"before": payload["config"][k], "after": v} for k, v in cfg.as_dict().items() if payload["config"][k] != v}
        if set(changed) != {"run_directory", "initial_checkpoint"}:
            raise ValueError("unexpected numerical training change during continuation fork")
        progress = json.loads((prior / "PROGRESS.json").read_text(encoding="utf-8"))
        prior_contract = json.loads((prior / "CONTRACT.json").read_text(encoding="utf-8"))
        if sha(root / "tools/run_open_v4_fresh_data.py") != prior_contract["tool_sha256"]:
            raise ValueError("source data generation tool changed")
        directory.mkdir(parents=True)
        evidence = {"at": utc_now(), "source_checkpoint": plan["participants"]["candidate"],
            "confirmation": {"path": str((confirmation / "RESULTS.json").relative_to(root)), "sha256": sha(confirmation / "RESULTS.json")},
            "arena": {"path": str((confirmation / "arena.json").relative_to(root)), "sha256": sha(confirmation / "arena.json")},
            "gate": gate, "route_confirmed": True, "changed_config_fields": changed,
            "rule_and_training_source_unchanged": True, "old_roles_preserved": original_current,
            "collection_note": "Continuing standard production collector at 256 simultaneous games; qualifying experiment used four sequential 64-game chunks. Frozen actor is now the qualified candidate, with later actor updates gated.",
            "tool_sha256": sha(Path(__file__))}
        atomic_json(directory / "CONTINUATION.json", evidence | {"state": "preparing", "seed_iteration": 26})
        atomic_json(config_path, cfg.as_dict())
        atomic_json(directory / "effective_config.json", cfg.as_dict())
        contract = {"config_hash": cfg.digest(), "source": source_contract(root),
            "layouts": original_contract["layouts"], "teachers": original_contract["teachers"]}
        atomic_json(directory / "CONTRACT.json", contract)
        for stage in (4, 5, 6):
            samples, games, metadata_rows, sources = [], [], [], []
            for iteration in range(21 + (stage - 1) * 4, 21 + stage * 4):
                record = progress["collections"][str(iteration)]
                path = prior / record["path"]
                metadata_path = prior / record["metadata"]
                if sha(path) != record["sha256"] or sha(metadata_path) != record["metadata_sha256"]:
                    raise ValueError("imported training replay changed")
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata["layout_role"] != "train" or metadata["actor_model_hash"] != original_current["actor"]["model_hash"]:
                    raise ValueError("imported replay source mismatch")
                part = read_selfplay_shard(path)
                for sample in part:
                    sample.observation = sample.observation.astype(np.float16)
                samples.extend(part)
                games.extend(metadata["games"])
                metadata_rows.append(metadata)
                sources.append({"path": str(path.relative_to(root)), "sha256": record["sha256"],
                    "metadata_sha256": record["metadata_sha256"]})
            destination = directory / f"replay/iteration_{20 + stage:06d}.npz"
            write_selfplay_shard(destination, samples)
            elapsed = sum(m["elapsed_seconds"] for m in metadata_rows)
            plies = sum(m["total_plies"] for m in metadata_rows)
            neural = sum(m["neural_seconds"] for m in metadata_rows)
            simulations = sum(m["simulation_count"] for m in metadata_rows)
            metadata = {"iteration": 20 + stage, "games": games, "game_count": len(games),
                "layout_role": "train", "policy_samples": len(samples), "total_plies": plies,
                "config_hash": cfg.digest(), "shard_sha256": sha(destination),
                "actor_model_hash": original_current["actor"]["model_hash"], "teacher_provenance": contract["teachers"],
                "label_sources": dict(Counter(g["mode"] for g in games)), "reasons": dict(Counter(g["reason"] for g in games)),
                "elapsed_seconds": elapsed, "moves_per_second": plies / elapsed, "neural_seconds": neural,
                "simulation_count": simulations, "simulations_per_second": simulations / neural,
                "mean_inference_batch": None, "mean_inference_batch_note": "Per-source measured batches retained below; inference call counts unavailable for an exact pooled mean.",
                "source_collection_metrics": [{k: m[k] for k in ("iteration", "mean_inference_batch", "moves_per_second", "simulations_per_second")} for m in metadata_rows],
                "teacher_workers": cfg.workers, "teacher_execution": "four imported 64-game batches with 12 CPU workers",
                "generation_config_hash": metadata_rows[0]["config_hash"], "imported_sources": sources,
                "provenance_note": "Only replay namespace/config paths migrated; sparse observations, actions, visits and real terminal WDL copied from original training data; no new labels."}
            atomic_json(destination.with_suffix(".json"), metadata)
            del samples, part
        (directory / "development").mkdir()
        shutil.copy2(original / "development/suite.json", directory / "development/suite.json")
        (directory / "arena").mkdir()
        shutil.copy2(confirmation / "arena.json", directory / "arena/actor_000026.json")
        role_payload = fork_payload(payload, cfg, evidence)
        state = {"iteration": 26, "contract_hash": sha(directory / "CONTRACT.json")}
        for role in ("learner", "actor", "champion"):
            path = directory / f"checkpoints/{role}_000026.pt"
            role_payload["role"] = role
            atomic_checkpoint(path, role_payload)
            state[role] = {"path": str(path.relative_to(directory)), "sha256": sha(path), "model_hash": payload["model_hash"], "iteration": 26}
        source_report = json.loads((prior / "fresh/reports/stage_000006.json").read_text(encoding="utf-8"))
        report = {"iteration": 26, "collection": metadata, "training": source_report["training"],
            "internal_before": source_report["common_holdout_before"], "internal_after": source_report["common_holdout_after"],
            "gate": gate, "counters": payload["counters"], "roles": {k: state[k] for k in ("learner", "actor", "champion")},
            "replay_samples": source_report["training_samples"], "replay_file_bytes": destination.stat().st_size,
            "peak_gpu_allocated_mib": source_report["peak_gpu_allocated_mib"], "elapsed_seconds": source_report["training_seconds"],
            "imported_report": True, "timing_note": "Training time only; collection metrics pooled from four original chunks.", "continuation": evidence}
        atomic_json(directory / "reports/iteration_000026.json", report)
        atomic_json(directory / "supervision/PLAN.json", {"parent": str(original), "seed_iteration": 26,
            "authorization": "User authorized continued local training and evidence-based model promotion.", "continuation": evidence})
        atomic_json(directory / "CONTINUATION.json", evidence | {"state": "prepared", "seed_iteration": 26})
        # The unique committed entry is published after every required file.
        atomic_json(directory / "CURRENT.json", state)
        print(json.dumps({"directory": str(directory), "config": str(config_path), "iteration": 26,
            "effective_updates": payload["counters"]["effective_updates"], "promoted_roles": ["actor", "champion"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
