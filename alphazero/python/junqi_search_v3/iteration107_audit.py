from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .alternating_lineage import _rng_snapshot
from .dataset import atomic_json
from .loop import SearchSample, SearchSelfPlay, read_selfplay_shard
from .recovery import RECOVERY_CONFIG
from .strength_continuation import StrengthContinuationLineage


AUDIT_ITERATION = 107
SOURCE_ITERATION = 106
FRACTIONS = (0.12, 0.24, 0.48)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _model_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _sample_key(sample: SearchSample) -> tuple[int, int, int]:
    return int(sample.game_id), int(sample.ply), int(sample.state_hash)


def _raw_samples(directory: Path) -> list[SearchSample]:
    result: list[SearchSample] = []
    for shard in sorted(directory.glob("games_*.npz")):
        result.extend(read_selfplay_shard(shard))
    return result


def _transition_action(
    state: Any,
    current: SearchSample,
    following: SearchSample,
) -> int:
    del current
    from_plane = np.flatnonzero(
        np.abs(np.asarray(following.observation[97]).reshape(-1)) > 0.5
    )
    to_plane = np.flatnonzero(
        np.abs(np.asarray(following.observation[98]).reshape(-1)) > 0.5
    )
    if len(from_plane) == 1 and len(to_plane) == 1:
        action = (59 - int(from_plane[0])) * 60 + (
            59 - int(to_plane[0])
        )
        if action in set(int(item) for item in state.legal_actions()):
            probe = state.clone()
            probe.apply(action)
            if int(probe.native_state.key()) == int(
                following.state_hash
            ):
                return action
    matches = []
    for action in state.legal_actions():
        probe = state.clone()
        probe.apply(int(action))
        if int(probe.native_state.key()) == int(following.state_hash):
            matches.append(int(action))
    if len(matches) != 1:
        raise RuntimeError(
            "cannot reconstruct a unique transition for "
            f"game={following.game_id} ply={following.ply}; "
            f"matches={matches}"
        )
    return matches[0]


def _reconstruct_selected_states(
    lineage: StrengthContinuationLineage,
    model: PolicyValueNet,
    raw: list[SearchSample],
    selected_keys: set[tuple[int, int, int]],
) -> tuple[list[Any], list[SearchSample]]:
    by_game: dict[int, list[SearchSample]] = defaultdict(list)
    for sample in raw:
        by_game[int(sample.game_id)].append(sample)
    selected_games = {key[0] for key in selected_keys}
    by_game = {
        game_id: sorted(rows, key=lambda item: int(item.ply))
        for game_id, rows in by_game.items()
    }
    # Per-block state caps remove shared opening states before the NPZ is
    # written (the first retained ply is as late as 8 in iteration 107).
    # Reproduce only that missing prefix with the frozen collection actor.
    prefix_search = SearchSelfPlay(
        lineage.config, lineage.stages.backend, model, lineage.device
    ).search
    first_states: dict[int, Any] = {}
    for block_begin in range(0, 1024, 256):
        block_games = list(range(block_begin, block_begin + 256))
        selected_in_block = [
            game_id
            for game_id in block_games
            if game_id in selected_games
        ]
        if not selected_in_block:
            continue
        maximum_first_ply = max(
            int(by_game[game_id][0].ply)
            for game_id in selected_in_block
        )
        cursors = {
            game_id: lineage.stages.backend.new_state(
                lineage.config.seed
                + AUDIT_ITERATION * 1_000_000
                + game_id
            )
            for game_id in block_games
        }
        game_rng = {
            game_id: random.Random(
                lineage.config.seed
                ^ (AUDIT_ITERATION * 0x9E3779B1)
                ^ game_id
            )
            for game_id in block_games
        }
        for ply in range(maximum_first_ply + 1):
            for game_id in selected_in_block:
                if int(by_game[game_id][0].ply) == ply:
                    first_states[game_id] = cursors[game_id].clone()
            if ply == maximum_first_ply:
                break
            groups: dict[int, list[int]] = defaultdict(list)
            for game_id in block_games:
                state = cursors[game_id]
                contexts = np.asarray(
                    state.repetition_contexts(), dtype=np.uint8
                )
                simulations = (
                    256
                    if int(state.native_state.repetition_count()) >= 2
                    or bool(np.any(contexts > 0))
                    else 64
                )
                groups[simulations].append(game_id)
            for simulations, active in groups.items():
                results, _ = prefix_search.search(
                    [cursors[game_id] for game_id in active],
                    simulations,
                    add_root_noise=True,
                    epsilon=lineage.config.puct.dirichlet_epsilon,
                    noise_seeds=[
                        lineage.config.seed
                        ^ game_id
                        ^ (ply * 0xA511E9B3)
                        for game_id in active
                    ],
                )
                for game_id, result in zip(active, results):
                    probability = result.visits.astype(np.float64)
                    probability /= probability.sum()
                    action = int(
                        game_rng[game_id].choices(
                            result.actions.tolist(),
                            weights=probability.tolist(),
                            k=1,
                        )[0]
                    )
                    cursors[game_id].apply(action)

    states: dict[tuple[int, int, int], Any] = {}
    samples: dict[tuple[int, int, int], SearchSample] = {}
    for game_id in sorted(by_game):
        if game_id not in selected_games:
            continue
        rows = by_game[game_id]
        if not rows:
            continue
        state = first_states[game_id]
        for index, sample in enumerate(rows):
            if int(state.native_state.key()) != int(sample.state_hash):
                raise RuntimeError(
                    "state reconstruction mismatch at "
                    f"game={game_id} ply={sample.ply}"
                )
            key = _sample_key(sample)
            if key in selected_keys:
                states[key] = state.clone()
                samples[key] = sample
            if index + 1 < len(rows):
                action = _transition_action(state, sample, rows[index + 1])
                state.apply(action)
    missing = selected_keys.difference(states)
    if missing:
        raise RuntimeError(
            f"{len(missing)} held-out states could not be reconstructed"
        )
    ordered_keys = sorted(selected_keys)
    return [states[key] for key in ordered_keys], [
        samples[key] for key in ordered_keys
    ]


def _legacy_search64_targets(
    lineage: StrengthContinuationLineage,
    model: PolicyValueNet,
    validation: list[SearchSample],
    output: Path,
    *,
    batch_states: int,
) -> tuple[
    dict[tuple[int, int, int], np.ndarray],
    dict[str, Any],
]:
    cache = output / "legacy_search64_targets.npz"
    keys = sorted(_sample_key(sample) for sample in validation)
    if cache.exists():
        with np.load(cache, allow_pickle=False) as data:
            cached_keys = [
                (int(game), int(ply), int(state_hash))
                for game, ply, state_hash in zip(
                    data["key_game_id"],
                    data["key_ply"],
                    data["key_state_hash"],
                )
            ]
            offsets = data["offsets"]
            visits = data["visits"]
            result = {
                key: visits[int(offsets[index]):int(offsets[index + 1])].copy()
                for index, key in enumerate(cached_keys)
            }
            exact = data["exact_normal_matches"].astype(np.int64)
        if cached_keys != keys:
            raise RuntimeError("legacy Search64 cache key mismatch")
        return result, {
            "ordinary_roots_exact_visit_matches": int(exact[0]),
            "ordinary_roots_compared": int(exact[1]),
            "ordinary_roots_exact_visit_match_rate": float(
                exact[0] / max(1, exact[1])
            ),
            "ordinary_targets_use_committed_visits": True,
            "cycle_targets_use_recomputed_search64": True,
        }

    replay = (
        lineage.output
        / "search_replay"
        / f"iteration_{AUDIT_ITERATION:06d}"
    )
    raw = _raw_samples(replay)
    states, ordered_samples = _reconstruct_selected_states(
        lineage, model, raw, set(keys)
    )
    search = SearchSelfPlay(
        lineage.config, lineage.stages.backend, model, lineage.device
    ).search
    target_map: dict[tuple[int, int, int], np.ndarray] = {}
    exact_normal_matches = 0
    exact_normal_total = 0
    for begin in range(0, len(states), batch_states):
        state_batch = states[begin:begin + batch_states]
        sample_batch = ordered_samples[begin:begin + batch_states]
        results, metric = search.search(
            state_batch,
            64,
            add_root_noise=True,
            epsilon=lineage.config.puct.dirichlet_epsilon,
            noise_seeds=[
                lineage.config.seed
                ^ int(sample.game_id)
                ^ (int(sample.ply) * 0xA511E9B3)
                for sample in sample_batch
            ],
        )
        print(
            f"[iteration107 audit] legacy-search64 "
            f"{min(begin + len(state_batch), len(states))}/{len(states)} "
            f"sps={metric['simulations_per_second']:.0f}",
            flush=True,
        )
        for sample, result in zip(sample_batch, results):
            if not np.array_equal(
                result.actions.astype(np.int32),
                sample.legal_actions.astype(np.int32),
            ):
                raise RuntimeError(
                    "legacy Search64 legal-action order mismatch at "
                    f"{_sample_key(sample)}"
                )
            if not bool(sample.cycle_entry):
                exact_normal_total += 1
                exact_normal_matches += int(
                    np.array_equal(result.visits, sample.visits)
                )
                # The committed value is the exact old Search64 target.
                target_map[_sample_key(sample)] = sample.visits.astype(
                    np.int32, copy=True
                )
            else:
                target_map[_sample_key(sample)] = result.visits.astype(
                    np.int32, copy=True
                )
    offsets = [0]
    flattened = []
    for key in keys:
        flattened.append(target_map[key])
        offsets.append(offsets[-1] + len(target_map[key]))
    temporary = cache.with_suffix(cache.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            key_game_id=np.asarray(
                [key[0] for key in keys], dtype=np.int32
            ),
            key_ply=np.asarray(
                [key[1] for key in keys], dtype=np.int16
            ),
            key_state_hash=np.asarray(
                [key[2] for key in keys], dtype=np.uint64
            ),
            offsets=np.asarray(offsets, dtype=np.int64),
            visits=np.concatenate(flattened).astype(np.int16),
            exact_normal_matches=np.asarray(
                [exact_normal_matches, exact_normal_total], dtype=np.int64
            ),
        )
    os.replace(temporary, cache)
    return target_map, {
        "ordinary_roots_exact_visit_matches": exact_normal_matches,
        "ordinary_roots_compared": exact_normal_total,
        "ordinary_roots_exact_visit_match_rate": (
            exact_normal_matches / max(1, exact_normal_total)
        ),
        "ordinary_targets_use_committed_visits": True,
        "cycle_targets_use_recomputed_search64": True,
    }


def _forward(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> tuple[list[np.ndarray], np.ndarray]:
    probabilities: list[np.ndarray] = []
    wdl_log_probability = []
    model.eval()
    microbatch = 256 if device.type == "cuda" else 16
    with torch.inference_mode():
        for begin in range(0, len(samples), microbatch):
            selected = samples[begin:begin + microbatch]
            observation = torch.from_numpy(
                np.stack([sample.observation for sample in selected])
            ).to(device=device, dtype=torch.float32)
            mask = np.zeros((len(selected), 3600), dtype=np.bool_)
            repetition = np.zeros((len(selected), 3600), dtype=np.uint8)
            for row, sample in enumerate(selected):
                legal = sample.legal_actions.astype(np.int64, copy=False)
                mask[row, legal] = True
                repetition[row, legal] = sample.repetition_contexts
            mask_tensor = torch.from_numpy(mask).to(device)
            repetition_tensor = torch.from_numpy(repetition).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                logits, _, _, wdl_logits, _ = model.forward_heads(
                    observation, mask_tensor, repetition_tensor
                )
            policy = torch.softmax(logits.float(), -1).cpu().numpy()
            wdl_log_probability.append(
                torch.log_softmax(wdl_logits.float(), -1).cpu().numpy()
            )
            for row, sample in enumerate(selected):
                probabilities.append(
                    policy[row, sample.legal_actions.astype(np.int64)].copy()
                )
    return probabilities, np.concatenate(wdl_log_probability)


def _is_tactical(sample: SearchSample) -> bool:
    occupied = np.any(
        np.asarray(sample.observation[14:38]) > 0.5, axis=0
    ).reshape(-1)
    return any(
        bool(occupied[int(action) % 60])
        for action in sample.legal_actions
    )


def _categories(samples: list[SearchSample]) -> dict[str, np.ndarray]:
    cycle = np.asarray([bool(sample.cycle_entry) for sample in samples])
    # A 256-simulation root contains 255 completed root visits.  The only
    # operation that lowers this sum is cycle-aware-v1 repeat scaling.
    adjusted = np.asarray(
        [
            bool(sample.cycle_entry) and int(sample.visits.sum()) < 255
            for sample in samples
        ]
    )
    endgame = np.asarray([bool(sample.endgame) for sample in samples])
    opening = np.asarray([int(sample.ply) < 24 for sample in samples])
    midgame = np.asarray(
        [
            int(sample.ply) >= 24 and not bool(sample.endgame)
            for sample in samples
        ]
    )
    tactical = np.asarray([_is_tactical(sample) for sample in samples])
    return {
        "all": np.ones(len(samples), dtype=np.bool_),
        "cycle_entry_adjusted": adjusted,
        "cycle_entry_unadjusted": cycle & ~adjusted,
        "ordinary_noncycle": ~cycle,
        "endgame": endgame,
        "opening": opening,
        "midgame": midgame,
        "tactical": tactical,
    }


def _bootstrap(
    values: np.ndarray,
    families: np.ndarray,
    seed: int,
) -> list[float]:
    family_values = []
    for family in np.unique(families):
        family_values.append(float(np.mean(values[families == family])))
    if not family_values:
        return [math.nan, math.nan]
    vector = np.asarray(family_values, dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(4000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = float(
            np.mean(rng.choice(vector, size=len(vector), replace=True))
        )
    return [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _target_metrics(
    samples: list[SearchSample],
    targets: list[np.ndarray],
    baseline_policy: list[np.ndarray],
    candidate_policy: list[np.ndarray],
    baseline_wdl_logp: np.ndarray,
    candidate_wdl_logp: np.ndarray,
    mask: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return {"samples": 0, "families": 0}
    baseline_excess = []
    candidate_excess = []
    candidate_top1 = []
    policy_kl = []
    baseline_wdl = []
    candidate_wdl = []
    family = []
    target_entropy = []
    for index in indices:
        target = targets[index].astype(np.float64)
        target /= max(1.0, float(target.sum()))
        entropy = -float(
            np.sum(target * np.log(np.clip(target, 1e-12, 1.0)))
        )
        before = np.clip(
            baseline_policy[index].astype(np.float64), 1e-12, 1.0
        )
        after = np.clip(
            candidate_policy[index].astype(np.float64), 1e-12, 1.0
        )
        baseline_excess.append(
            -float(np.sum(target * np.log(before))) - entropy
        )
        candidate_excess.append(
            -float(np.sum(target * np.log(after))) - entropy
        )
        candidate_top1.append(
            float(
                target[int(np.argmax(after))]
                >= float(np.max(target)) - 1e-7
            )
        )
        policy_kl.append(float(np.sum(before * np.log(before / after))))
        wdl = np.asarray(samples[index].wdl, dtype=np.float64)
        baseline_wdl.append(
            -float(np.sum(wdl * baseline_wdl_logp[index]))
        )
        candidate_wdl.append(
            -float(np.sum(wdl * candidate_wdl_logp[index]))
        )
        target_entropy.append(entropy)
        family.append(int(samples[index].game_id))
    before_excess = np.asarray(baseline_excess)
    after_excess = np.asarray(candidate_excess)
    before_wdl = np.asarray(baseline_wdl)
    after_wdl = np.asarray(candidate_wdl)
    families = np.asarray(family, dtype=np.int64)
    policy_delta = after_excess - before_excess
    wdl_delta = after_wdl - before_wdl
    unique_families = np.unique(families)
    policy_family_mean = float(
        np.mean(
            [
                np.mean(policy_delta[families == family])
                for family in unique_families
            ]
        )
    )
    wdl_family_mean = float(
        np.mean(
            [
                np.mean(wdl_delta[families == family])
                for family in unique_families
            ]
        )
    )
    return {
        "samples": int(len(indices)),
        "families": int(len(np.unique(families))),
        "target_entropy": float(np.mean(target_entropy)),
        "baseline_policy_excess": float(np.mean(before_excess)),
        "candidate_policy_excess": float(np.mean(after_excess)),
        "policy_excess_delta": float(np.mean(policy_delta)),
        "policy_excess_delta_family_mean": policy_family_mean,
        "policy_excess_delta_family_ci95": _bootstrap(
            policy_delta, families, seed
        ),
        "candidate_top1_agreement": float(np.mean(candidate_top1)),
        "baseline_wdl_ce": float(np.mean(before_wdl)),
        "candidate_wdl_ce": float(np.mean(after_wdl)),
        "wdl_delta": float(np.mean(wdl_delta)),
        "wdl_delta_family_mean": wdl_family_mean,
        "wdl_delta_family_ci95": _bootstrap(
            wdl_delta, families, seed ^ 0x57444C
        ),
        "baseline_to_candidate_policy_kl": float(np.mean(policy_kl)),
    }


def _candidate_snapshots(
    lineage: StrengthContinuationLineage,
    output: Path,
) -> tuple[Path, dict[float, Path], dict[str, Any]]:
    source_checkpoint = lineage._checkpoint(SOURCE_ITERATION)
    model, optimizer, scheduler, global_step = (
        lineage._load_training_state(SOURCE_ITERATION)
    )
    fresh = lineage.runner._iteration_samples(AUDIT_ITERATION)
    fresh_train = [
        sample for sample in fresh if int(sample.game_id) % 10 != 0
    ]
    recent_train = lineage._recent_samples(AUDIT_ITERATION)
    if not recent_train:
        recent_train = fresh_train
    pools = {
        "fresh_train": fresh_train,
        "fresh_validation": [
            sample for sample in fresh if int(sample.game_id) % 10 == 0
        ],
        "recent_train": recent_train,
        "oracle_train": lineage.runner._oracle_dataset().by_split["train"],
    }
    pools["cycle_train"] = [
        sample
        for sample in pools["oracle_train"]
        if np.any(sample.repetition_contexts > 0)
    ]
    epoch_steps = int(
        np.ceil(len(fresh_train) / lineage.config.train.batch_size)
    )
    target_steps = {
        fraction: int(round(epoch_steps * fraction))
        for fraction in FRACTIONS
    }
    if any(step % 2 for step in target_steps.values()):
        raise RuntimeError(
            f"diagnostic fractions do not align to Policy2:Value1: {target_steps}"
        )
    baseline_path = source_checkpoint
    paths: dict[float, Path] = {}
    lineage.seed = (
        lineage.config.seed ^ 0x414C5445 ^ AUDIT_ITERATION
    )
    policy_steps = 0
    value_steps = 0
    formal_validation_interval = max(
        1, int(round(epoch_steps * 0.25))
    )
    kl_weight = float(
        RECOVERY_CONFIG["value_policy_kl_initial_weight"]
    )
    training_rows = []
    while policy_steps < max(target_steps.values()):
        for _ in range(2):
            metric = lineage._policy_step(model, optimizer, pools)
            policy_steps += 1
            training_rows.append(
                {
                    "phase": "policy",
                    "policy_step": policy_steps,
                    "value_step": value_steps,
                    **metric,
                }
            )
        metric, kl_weight = lineage._value_phase(
            model, optimizer, pools, kl_weight, legacy_reference=True
        )
        value_steps += 1
        training_rows.append(
            {
                "phase": "value",
                "policy_step": policy_steps,
                "value_step": value_steps,
                **metric,
            }
        )
        if policy_steps % formal_validation_interval == 0:
            # The formal trainer selects 64 samples before computing the
            # shared-trunk Policy/Value gradient cosine at each 0.25 epoch
            # validation.  The diagnostic restores torch/model state, but the
            # alternating batch generator intentionally advances by 64 keys.
            # Consume those keys so later candidate batches remain identical.
            lineage._keys(64)
        for fraction, target in target_steps.items():
            if target != policy_steps:
                continue
            label = f"{fraction:.2f}".replace(".", "p")
            path = output / "candidates" / f"epoch_{label}.pt"
            payload = {
                "schema": "search-v3-iteration107-audit-candidate-v1",
                "source_iteration": SOURCE_ITERATION,
                "data_iteration": AUDIT_ITERATION,
                "fractional_epoch": fraction,
                "policy_steps": policy_steps,
                "value_steps": value_steps,
                "global_step": global_step + policy_steps + value_steps,
                "model": {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                },
                "optimizer": copy.deepcopy(optimizer.state_dict()),
                "scheduler": copy.deepcopy(scheduler),
                "rng": _rng_snapshot(lineage.generator, lineage.runner),
                "leaf_value_scale": float(
                    model.search_v3_leaf_value_scale
                ),
                "source_checkpoint_sha256": _sha256(
                    source_checkpoint
                ),
            }
            if path.exists():
                existing = torch.load(
                    path, map_location="cpu", weights_only=False
                )
                if _model_hash(existing["model"]) != _model_hash(
                    payload["model"]
                ):
                    raise RuntimeError(
                        f"candidate replay changed: {path}"
                    )
            else:
                _atomic_torch(path, payload)
            paths[fraction] = path
            print(
                f"[iteration107 audit] saved epoch={fraction:.2f} "
                f"policy={policy_steps} value={value_steps} "
                f"hash={_model_hash(payload['model'])[:12]}",
                flush=True,
            )
    metadata = {
        "epoch_steps": epoch_steps,
        "target_policy_steps": {
            str(key): value for key, value in target_steps.items()
        },
        "fresh_train_samples": len(fresh_train),
        "heldout_samples": len(pools["fresh_validation"]),
        "recent_train_samples": len(recent_train),
        "training_rows": training_rows,
    }
    return baseline_path, paths, metadata


def run_iteration107_audit(
    config_path: str | Path,
    *,
    cpu_threads: int = 12,
    search_batch_states: int = 256,
) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(cpu_threads)))
    lineage = StrengthContinuationLineage(
        config_path,
        training_workers=cpu_threads,
        arena_workers=1,
        arena_block_states=64,
    )
    output = (
        lineage.output
        / "diagnostics"
        / "iteration_000107_target_semantics_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    replay = (
        lineage.output
        / "search_replay"
        / f"iteration_{AUDIT_ITERATION:06d}"
    )
    evidence_paths = (
        sorted(replay.glob("games_*.npz"))
        + sorted(replay.glob("games_*.json"))
        + [
            replay / "summary.json",
            lineage.output
            / "reports"
            / f"iteration_{AUDIT_ITERATION:06d}.json",
            lineage._checkpoint(SOURCE_ITERATION),
            lineage._checkpoint(AUDIT_ITERATION),
        ]
    )
    manifest = {
        "schema": "search-v3-iteration107-frozen-evidence-v1",
        "iteration": AUDIT_ITERATION,
        "files": [
            {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in evidence_paths
            if path.exists()
        ],
        "training_process_was_not_started_by_audit": True,
        "formal_lineage_files_are_read_only": True,
    }
    atomic_json(output / "frozen_evidence_manifest.json", manifest)

    baseline_path, candidate_paths, replay_metadata = (
        _candidate_snapshots(lineage, output)
    )
    validation = [
        sample
        for sample in lineage.runner._iteration_samples(AUDIT_ITERATION)
        if int(sample.game_id) % 10 == 0
    ]
    baseline_payload = torch.load(
        baseline_path, map_location="cpu", weights_only=False
    )
    baseline = PolicyValueNet(lineage.config.model).to(lineage.device)
    baseline.load_state_dict(baseline_payload["model"])
    baseline.search_v3_leaf_value_scale = float(
        baseline_payload["leaf_value_scale"]
    )
    legacy_map, legacy_quality = _legacy_search64_targets(
        lineage,
        baseline,
        validation,
        output,
        batch_states=max(1, int(search_batch_states)),
    )
    aware_targets = [
        sample.visits.astype(np.float32, copy=True)
        for sample in validation
    ]
    legacy_targets = [
        legacy_map[_sample_key(sample)].astype(np.float32, copy=True)
        for sample in validation
    ]
    baseline_policy, baseline_wdl = _forward(
        baseline, validation, lineage.device
    )
    categories = _categories(validation)
    reports = {}
    for fraction, path in candidate_paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        candidate = PolicyValueNet(lineage.config.model).to(
            lineage.device
        )
        candidate.load_state_dict(payload["model"])
        candidate.search_v3_leaf_value_scale = float(
            payload["leaf_value_scale"]
        )
        candidate_policy, candidate_wdl = _forward(
            candidate, validation, lineage.device
        )
        semantics = {}
        for semantic_name, targets in (
            ("legacy_search64", legacy_targets),
            ("cycle_aware_v1", aware_targets),
        ):
            semantics[semantic_name] = {
                name: _target_metrics(
                    validation,
                    targets,
                    baseline_policy,
                    candidate_policy,
                    baseline_wdl,
                    candidate_wdl,
                    mask,
                    lineage.config.seed
                    ^ int(round(fraction * 1000))
                    ^ sum(ord(char) for char in name)
                    ^ sum(ord(char) for char in semantic_name),
                )
                for name, mask in categories.items()
            }
        reports[f"{fraction:.2f}"] = {
            "checkpoint": str(path.resolve()),
            "model_hash": _model_hash(payload["model"]),
            "policy_steps": int(payload["policy_steps"]),
            "value_steps": int(payload["value_steps"]),
            "value_step_peak_kl": max(
                (
                    float(row["policy_kl"])
                    for row in replay_metadata["training_rows"]
                    if row["phase"] == "value"
                    and int(row["policy_step"])
                    <= int(payload["policy_steps"])
                ),
                default=0.0,
            ),
            "candidate_final_kl": next(
                (
                    float(row["policy_kl"])
                    for row in reversed(
                        replay_metadata["training_rows"]
                    )
                    if row["phase"] == "value"
                    and int(row["policy_step"])
                    <= int(payload["policy_steps"])
                ),
                0.0,
            ),
            "candidate_final_family_kl": next(
                (
                    float(row["policy_family_kl"])
                    for row in reversed(
                        replay_metadata["training_rows"]
                    )
                    if row["phase"] == "value"
                    and int(row["policy_step"])
                    <= int(payload["policy_steps"])
                ),
                0.0,
            ),
            "semantics": semantics,
        }
        del candidate
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    stored_summary = json.loads(
        (replay / "summary.json").read_text(encoding="utf-8")
    )
    raw = _raw_samples(replay)
    raw_cycle = [sample for sample in raw if bool(sample.cycle_entry)]
    inferred_raw_adjusted = sum(
        int(sample.visits.sum()) < 255 for sample in raw_cycle
    )
    target_difference = np.asarray(
        [
            0.5
            * float(
                np.sum(
                    np.abs(
                        legacy.astype(np.float64)
                        / max(1.0, float(legacy.sum()))
                        - aware.astype(np.float64)
                        / max(1.0, float(aware.sum()))
                    )
                )
            )
            for legacy, aware in zip(legacy_targets, aware_targets)
        ]
    )
    formal_report = json.loads(
        (
            lineage.output
            / "reports"
            / f"iteration_{AUDIT_ITERATION:06d}.json"
        ).read_text(encoding="utf-8")
    )
    formal_by_steps = {
        int(candidate["policy_steps"]): candidate
        for candidate in formal_report["candidates"]
    }
    replay_fidelity = {}
    for fraction, candidate in reports.items():
        policy_steps = int(candidate["policy_steps"])
        if policy_steps not in formal_by_steps:
            continue
        formal = formal_by_steps[policy_steps]["validation"]
        audited = candidate["semantics"]["cycle_aware_v1"]["all"]
        replay_fidelity[fraction] = {
            "formal_policy_excess": float(formal["policy_excess"]),
            "audited_policy_excess": float(
                audited["candidate_policy_excess"]
            ),
            "absolute_policy_excess_error": abs(
                float(formal["policy_excess"])
                - float(audited["candidate_policy_excess"])
            ),
            "formal_wdl_ce": float(formal["wdl_ce"]),
            "audited_wdl_ce": float(audited["candidate_wdl_ce"]),
            "absolute_wdl_ce_error": abs(
                float(formal["wdl_ce"])
                - float(audited["candidate_wdl_ce"])
            ),
            "formal_top1": float(formal["top1"]),
            "audited_top1": float(
                audited["candidate_top1_agreement"]
            ),
        }
    result = {
        "schema": "search-v3-iteration107-target-semantics-audit-v1",
        "iteration": AUDIT_ITERATION,
        "source_iteration": SOURCE_ITERATION,
        "formal_training_data_modified": False,
        "formal_checkpoint_modified": False,
        "gate_code_path": (
            "_train_iteration validation -> _candidate -> "
            "_family_metrics(sample.visits)"
        ),
        "training_code_path": (
            "_policy_step -> _policy_ce -> "
            "search_distill_batch(sample.visits)"
        ),
        "code_path_semantics_match": True,
        "stored_replay_semantics": (
            "ordinary roots=Search64; cycle-sensitive roots=Search256; "
            "repeat visits scaled to 0.25 when Q_exit >= Q_repeat-0.02"
        ),
        "legacy_semantics": (
            "ordinary roots use their committed Search64 visits; "
            "cycle roots rerun Search64 with the original root-noise seed "
            "on the exact same full state and rules history"
        ),
        "legacy_search64_reconstruction_quality": legacy_quality,
        "cycle_adjustment_flag_inference": {
            "rule": "cycle_entry and sum(stored_visits)<255",
            "retained_cycle_roots": len(raw_cycle),
            "retained_inferred_adjusted": inferred_raw_adjusted,
            "pre_cap_committed_summary_adjusted": int(
                stored_summary["cycle_teacher"]["adjusted"]
            ),
            "pre_cap_committed_summary_roots": int(
                stored_summary["cycle_teacher"]["roots"]
            ),
            "count_layer_note": (
                "summary counts roots before the per-state replay cap; "
                "NPZ and gate see the retained post-cap subset"
            ),
            "inference_is_exact_by_writer_invariant": True,
        },
        "target_semantic_difference": {
            "mean_total_variation": float(np.mean(target_difference)),
            "changed_roots": int(np.sum(target_difference > 1e-12)),
            "roots": len(target_difference),
            "cycle_adjusted_mean_total_variation": float(
                np.mean(
                    target_difference[
                        categories["cycle_entry_adjusted"]
                    ]
                )
            ),
        },
        "category_definitions": {
            "cycle_entry_adjusted": (
                "cycle_entry=true and stored visit sum <255"
            ),
            "cycle_entry_unadjusted": (
                "cycle_entry=true and stored visit sum ==255"
            ),
            "ordinary_noncycle": "cycle_entry=false",
            "endgame": "stored endgame flag: live pieces<=24 or ply>=80",
            "opening": "ply<24",
            "midgame": "ply>=24 and endgame=false",
            "tactical": "at least one legal action captures an occupied square",
            "overlap_note": (
                "phase/tactical categories overlap cycle categories; "
                "cycle adjusted/unadjusted are disjoint"
            ),
        },
        "kl_direction": "KL(baseline_policy || candidate_policy)",
        "candidate_replay": replay_metadata,
        "candidate_replay_fidelity": replay_fidelity,
        "candidates": reports,
        "frozen_evidence_manifest": str(
            (output / "frozen_evidence_manifest.json").resolve()
        ),
    }
    atomic_json(output / "report.json", result)
    return result
