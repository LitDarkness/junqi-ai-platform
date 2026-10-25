from __future__ import annotations

import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

import JunqiAZ
from junqi_az.model import PolicyValueNet
from junqi_search_v3.dataset import atomic_json
from junqi_search_v3.loop import read_selfplay_shard, write_selfplay_shard, search_distill_batch
from junqi_search_v3.search import BatchedNeuralPUCT

from .arena import create_dev_suite, run_arena, select_families
from .config import OpenConfig
from .games import TeacherLeague, collect_training_games
from .layouts import MultiLayoutBackend, create_manifest, load_role
from .learner import Learner, atomic_checkpoint, model_hash


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_contract(root: Path) -> dict[str, str]:
    paths = list((root / "python/junqi_open_v4").glob("*.py")) + [
        root / "python/junqi_search_v3/search.py", root / "python/junqi_search_v3/recovery.py",
        root / "python/junqi_search_v3/loop.py", root / "python/junqi_az/model.py", Path(JunqiAZ.__file__),
    ]
    return {str(path.relative_to(root)): sha256(path) for path in paths}


@torch.no_grad()
def validation_metrics(model, samples, device) -> dict[str, Any]:
    if not samples:
        return {"samples": 0, "status": "insufficient paired training families"}
    model.eval()
    families = {}
    for begin in range(0, len(samples), 64):
        part = samples[begin:begin + 64]
        batch = search_distill_batch(part, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits, unused, aux, wdl, scalar = model.forward_heads(batch["observations"], batch["legal_mask"], batch["repetition"])
        policy_loss = -(batch["target"] * torch.log_softmax(logits.float(), -1)).sum(-1).cpu().numpy()
        value_loss = -(batch["wdl"] * torch.log_softmax(wdl.float(), -1)).sum(-1).cpu().numpy()
        for sample, first, second in zip(part, policy_loss, value_loss):
            families.setdefault(sample.game_id, []).append([float(first), float(second)])
    values = np.stack([np.mean(rows, axis=0) for rows in families.values()])
    return {"samples": len(samples), "families": len(families), "family_policy_ce": float(values[:, 0].mean()),
            "family_wdl_ce": float(values[:, 1].mean()), "role": "internal holdout of training layout pairs; monitoring only"}


def _search(model, config, device):
    return BatchedNeuralPUCT(model, device=device, cache_size=config.cache_size,
                            inference_batch_size=config.inference_batch_size, leaves_per_tree=config.leaves_per_tree)


def _load_model(path, config, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PolicyValueNet(config.model).to(device).eval()
    model.load_state_dict(payload["model"])
    model.search_v3_leaf_value_scale = payload.get("leaf_value_scale", 1.)
    return model


def promotion_decision(arena: dict[str, Any], config: OpenConfig) -> dict[str, Any]:
    if arena.get("role") != "dev":
        raise ValueError("final or training arenas cannot promote a model")
    duel = arena["summaries"][0]["family_score"]
    candidate, baseline = arena["summaries"][0]["candidate"], arena["summaries"][0]["opponent"]
    sentinel = []
    for row in arena["paired_deltas"]:
        first = arena["summaries"][row["from_matchup"]]
        second = arena["summaries"][row["to_matchup"]]
        if first["candidate"] == baseline and second["candidate"] == candidate and first["opponent"].startswith("teacher"):
            sentinel.append(row["delta"])
    if not sentinel:
        raise ValueError("promotion requires the paired teacher sentinel")
    enough = duel["families"] >= config.actor_gate_families
    actor = enough and duel["ci95"][0] >= .5 - config.actor_noninferiority_margin and all(
        row["families"] >= config.actor_gate_families and row["ci95"][0] >= -config.teacher_sentinel_tolerance for row in sentinel)
    champion = actor and duel["families"] >= config.champion_minimum_family_count and duel["ci95"][0] > .5
    return {"actor_accepted": bool(actor), "champion_accepted": bool(champion),
            "duel_ci95": duel["ci95"], "teacher_delta_ci95": [row["ci95"] for row in sentinel],
            "minimum_families_met": enough, "learner_rollback": False}


def train_run(config: OpenConfig, root: Path, through_iteration: int) -> dict[str, Any]:
    config.validate()
    with TeacherLeague(root, config.teacher_candidates, config.workers) as teachers:
        return _train_run(config, root, through_iteration, teachers)


def _train_run(config: OpenConfig, root: Path, through_iteration: int, teachers: TeacherLeague) -> dict[str, Any]:
    config.validate()
    run = (root / config.run_directory).resolve()
    run.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(config.workers)
    learner = Learner(config, root)
    layouts = (root / config.layout_directory).resolve()
    create_manifest(layouts, config.seed, train=64, dev=128, final=256)
    backend = MultiLayoutBackend(load_role(layouts, "train"), config.seed)
    contract = {"config_hash": config.digest(), "source": source_contract(root),
                "layouts": sha256(layouts / "manifest.json"), "teachers": teachers.provenance}
    contract_path = run / "CONTRACT.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != contract:
            raise ValueError("run contract changed; preserve this run and create a new run directory")
    else:
        atomic_json(contract_path, contract)
        atomic_json(run / "effective_config.json", config.as_dict())
    current = run / "CURRENT.json"
    if current.exists():
        state = json.loads(current.read_text(encoding="utf-8"))
        if state["contract_hash"] != sha256(contract_path):
            raise ValueError("current checkpoint contract mismatch")
        for role in ("learner", "actor", "champion"):
            if sha256(run / state[role]["path"]) != state[role]["sha256"]:
                raise ValueError("role checkpoint digest mismatch")
        learner.resume(run / state["learner"]["path"])
    else:
        state = {"iteration": 0, "contract_hash": sha256(contract_path)}
        payload = learner.payload(role="learner", lineage=state)
        for role in ("learner", "actor", "champion"):
            path = run / f"checkpoints/{role}_000000.pt"
            role_payload = copy.copy(payload)
            role_payload["role"] = role
            atomic_checkpoint(path, role_payload)
            state[role] = {"path": str(path.relative_to(run)), "sha256": sha256(path), "model_hash": payload["model_hash"], "iteration": 0}
        atomic_json(current, state)
    actor = _load_model(run / state["actor"]["path"], config, learner.device)
    summaries = []
    for iteration in range(state["iteration"] + 1, through_iteration + 1):
        started = time.perf_counter()
        data_path = run / f"replay/iteration_{iteration:06d}.npz"
        data_report = data_path.with_suffix(".json")
        if data_path.exists() and data_report.exists():
            metadata = json.loads(data_report.read_text(encoding="utf-8"))
            if metadata["actor_model_hash"] != model_hash(actor) or metadata["shard_sha256"] != sha256(data_path):
                raise ValueError("replay provenance mismatch")
        else:
            samples, metadata = collect_training_games(config, actor, learner.device, backend, teachers, iteration)
            write_selfplay_shard(data_path, samples)
            metadata.update(actor_model_hash=model_hash(actor), shard_sha256=sha256(data_path), config_hash=config.digest())
            atomic_json(data_report, metadata)
        samples = []
        for previous in range(max(1, iteration - config.replay_iterations + 1), iteration + 1):
            previous_path = run / f"replay/iteration_{previous:06d}.npz"
            previous_report = json.loads(previous_path.with_suffix(".json").read_text(encoding="utf-8"))
            if previous_report["config_hash"] != config.digest() or previous_report["shard_sha256"] != sha256(previous_path):
                raise ValueError("replay contract or digest mismatch")
            rows = read_selfplay_shard(previous_path)
            for sample in rows:
                sample.observation = sample.observation.astype(np.float16)
            samples.extend(rows)
        # Split entire paired games. Held-out trajectories never supply anchors.
        train_samples = [sample for sample in samples if (sample.game_id * 0x9E3779B1 ^ config.seed) % 10 != 0]
        validation = [sample for sample in samples if (sample.game_id * 0x9E3779B1 ^ config.seed) % 10 == 0]
        if len(train_samples) > config.maximum_replay_samples:
            indices = learner.generator.choice(len(train_samples), config.maximum_replay_samples, replace=False)
            train_samples = [train_samples[int(index)] for index in indices]
        validation = validation[:4096]
        before = validation_metrics(learner.model, validation, learner.device)
        training = learner.train(train_samples)
        after = validation_metrics(learner.model, validation, learner.device)
        learner.counters["iteration"] = iteration
        gate = {"actor_accepted": False, "champion_accepted": False, "status": "gate not scheduled", "learner_rollback": False}
        if iteration % config.actor_gate_every == 0:
            suite = create_dev_suite(run / "development", layouts, teachers, config.seed ^ 0xDE)
            selected = select_families(suite, config.actor_gate_families, config.seed ^ iteration)
            agents = {"learner": _search(learner.model, config, learner.device), "actor": _search(actor, config, learner.device)}
            matchups = [("learner", "actor")] + [pair for teacher_id in range(len(teachers.modules))
                for pair in (("actor", f"teacher{teacher_id}"), ("learner", f"teacher{teacher_id}"))]
            arena = run_arena(selected, matchups,
                              agents, teachers, {name: config.simulations for name in agents}, seed=config.seed ^ iteration)
            atomic_json(run / f"arena/actor_{iteration:06d}.json", arena)
            gate = promotion_decision(arena, config)
            # A champion is tested against the current champion on the full dev
            # corpus, never inferred from an actor whose lineage may have changed.
            gate["champion_accepted"] = False
            if gate["actor_accepted"]:
                champion = _load_model(run / state["champion"]["path"], config, learner.device)
                agents = {"learner": _search(learner.model, config, learner.device), "champion": _search(champion, config, learner.device)}
                matchups = [("learner", "champion")] + [pair for teacher_id in range(len(teachers.modules))
                    for pair in (("champion", f"teacher{teacher_id}"), ("learner", f"teacher{teacher_id}"))]
                full = run_arena(suite, matchups,
                                 agents, teachers, {name: config.simulations for name in agents}, seed=config.seed ^ iteration ^ 0xFF)
                atomic_json(run / f"arena/champion_{iteration:06d}.json", full)
                gate["champion_accepted"] = promotion_decision(full, config)["champion_accepted"]
        payload = learner.payload(role="learner", lineage={"parent": copy.deepcopy(state),
            "transition": {"iteration": iteration, "gate": gate}, "replay_sha256": sha256(data_path)})
        for role in ("learner", "actor", "champion"):
            if role != "learner" and not gate[role + "_accepted"]:
                continue
            path = run / f"checkpoints/{role}_{iteration:06d}.pt"
            role_payload = copy.copy(payload)
            role_payload["role"] = role
            atomic_checkpoint(path, role_payload)
            state[role] = {"path": str(path.relative_to(run)), "sha256": sha256(path), "model_hash": payload["model_hash"], "iteration": iteration}
        report = {"iteration": iteration, "collection": metadata, "training": training, "internal_before": before,
                  "internal_after": after, "gate": gate, "counters": learner.counters.copy(),
                  "roles": {role: state[role] for role in ("learner", "actor", "champion")},
                  "replay_samples": len(train_samples), "replay_file_bytes": data_path.stat().st_size,
                  "peak_gpu_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if learner.device.type == "cuda" else 0.,
                  "elapsed_seconds": time.perf_counter() - started}
        atomic_json(run / f"reports/iteration_{iteration:06d}.json", report)
        state["iteration"] = iteration
        atomic_json(current, state)
        if gate["actor_accepted"]:
            actor.load_state_dict(learner.model.state_dict())
            actor.search_v3_leaf_value_scale = learner.model.search_v3_leaf_value_scale
        summaries.append({"iteration": iteration, "effective_updates": learner.counters["effective_updates"], "gate": gate})
        print(f"[committed] iteration={iteration} learner={learner.counters['effective_updates']} actor={state['actor']['iteration']} champion={state['champion']['iteration']}", flush=True)
    return {"run": str(run), "completed_iteration": state["iteration"], "iterations": summaries,
            "roles": {role: state[role] for role in ("learner", "actor", "champion")}}
