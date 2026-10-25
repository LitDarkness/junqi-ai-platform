from __future__ import annotations

import hashlib
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.evaluator import repetition_cycle_details

from .dataset import atomic_json
from .search import BatchedNeuralPUCT


CYCLE_TEACHER_SIMULATIONS = 256
CYCLE_TEACHER_EPSILON = 0.02
CYCLE_REPEAT_VISIT_SCALE = 0.25


@dataclass
class SearchSample:
    game_id: int
    state_hash: int
    observation: np.ndarray
    legal_actions: np.ndarray
    repetition_contexts: np.ndarray
    visits: np.ndarray
    player: int
    ply: int
    root_entropy: float
    raw_visits: np.ndarray | None = None
    legacy_search64_visits: np.ndarray | None = None
    legacy_search64_valid: bool = False
    cycle_teacher_eligible: bool = False
    cycle_teacher_adjusted: bool = False
    cycle_teacher_rational_repeat_preserved: bool = False
    position_key: int = 0
    repetition_count: int = 0
    cycle_entry: bool = False
    endgame: bool = False
    cycle_cleaned: bool = False
    wdl: np.ndarray | None = None


@dataclass
class SelfPlayGame:
    game_id: int
    state: Any
    rng: random.Random
    trace: list[int] = field(default_factory=list)
    samples: list[SearchSample] = field(default_factory=list)
    position_hashes: list[int] = field(default_factory=list)
    action_rows: list[dict[str, Any]] = field(default_factory=list)


def write_selfplay_shard(path: Path, samples: list[SearchSample]) -> None:
    offsets = [0]
    for sample in samples:
        offsets.append(offsets[-1] + len(sample.legal_actions))
    temporary = path.with_suffix(path.suffix + ".partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            game_id=np.asarray([sample.game_id for sample in samples], np.int64),
            state_hash=np.asarray(
                [sample.state_hash for sample in samples], np.uint64
            ),
            position_key=np.asarray(
                [sample.position_key for sample in samples], np.uint64
            ),
            observation=np.stack(
                [sample.observation.astype(np.float16) for sample in samples]
            ),
            offsets=np.asarray(offsets, np.int64),
            legal_actions=np.concatenate(
                [sample.legal_actions.astype(np.int16) for sample in samples]
            ),
            repetition_contexts=np.concatenate(
                [
                    sample.repetition_contexts.astype(np.uint8)
                    for sample in samples
                ]
            ),
            visits=np.concatenate(
                [sample.visits.astype(np.int16) for sample in samples]
            ),
            raw_visits=np.concatenate(
                [
                    (
                        sample.raw_visits
                        if sample.raw_visits is not None
                        else sample.visits
                    ).astype(np.int16)
                    for sample in samples
                ]
            ),
            legacy_search64_visits=np.concatenate(
                [
                    (
                        sample.legacy_search64_visits
                        if sample.legacy_search64_visits is not None
                        else sample.visits
                    ).astype(np.int16)
                    for sample in samples
                ]
            ),
            legacy_search64_valid=np.asarray(
                [sample.legacy_search64_valid for sample in samples],
                np.bool_,
            ),
            cycle_teacher_eligible=np.asarray(
                [sample.cycle_teacher_eligible for sample in samples],
                np.bool_,
            ),
            cycle_teacher_adjusted=np.asarray(
                [sample.cycle_teacher_adjusted for sample in samples],
                np.bool_,
            ),
            cycle_teacher_rational_repeat_preserved=np.asarray(
                [
                    sample.cycle_teacher_rational_repeat_preserved
                    for sample in samples
                ],
                np.bool_,
            ),
            player=np.asarray([sample.player for sample in samples], np.int8),
            ply=np.asarray([sample.ply for sample in samples], np.int16),
            root_entropy=np.asarray(
                [sample.root_entropy for sample in samples], np.float16
            ),
            repetition_count=np.asarray(
                [sample.repetition_count for sample in samples], np.uint8
            ),
            cycle_entry=np.asarray(
                [sample.cycle_entry for sample in samples], np.bool_
            ),
            endgame=np.asarray(
                [sample.endgame for sample in samples], np.bool_
            ),
            cycle_cleaned=np.asarray(
                [sample.cycle_cleaned for sample in samples], np.bool_
            ),
            wdl=np.stack([sample.wdl for sample in samples]).astype(np.float16),
        )
    os.replace(temporary, path)


def read_selfplay_shard(path: Path) -> list[SearchSample]:
    result = []
    with np.load(path, allow_pickle=False) as data:
        # np.load lazily decompresses members of an NPZ.  Accessing
        # data["observation"] inside the row loop decompresses the entire
        # member once per sample and quickly exhausts RAM.  Materialize each
        # member exactly once. Rows are copied out so reservoir sampling can
        # release an unselected shard; retaining one view must not pin the
        # entire decompressed NPZ member in RAM.
        game_id = data["game_id"]
        state_hash = data["state_hash"]
        position_key = (
            data["position_key"]
            if "position_key" in data.files
            else state_hash
        )
        observation = data["observation"]
        offsets = data["offsets"]
        legal_actions = data["legal_actions"]
        repetition_contexts = data["repetition_contexts"]
        visits = data["visits"]
        raw_visits = (
            data["raw_visits"] if "raw_visits" in data.files else None
        )
        legacy_search64_visits = (
            data["legacy_search64_visits"]
            if "legacy_search64_visits" in data.files
            else None
        )
        legacy_search64_valid = (
            data["legacy_search64_valid"]
            if "legacy_search64_valid" in data.files
            else None
        )
        cycle_teacher_eligible = (
            data["cycle_teacher_eligible"]
            if "cycle_teacher_eligible" in data.files
            else None
        )
        cycle_teacher_adjusted = (
            data["cycle_teacher_adjusted"]
            if "cycle_teacher_adjusted" in data.files
            else None
        )
        cycle_teacher_rational_repeat_preserved = (
            data["cycle_teacher_rational_repeat_preserved"]
            if "cycle_teacher_rational_repeat_preserved" in data.files
            else None
        )
        player = data["player"]
        ply = data["ply"]
        root_entropy = data["root_entropy"]
        repetition_count = (
            data["repetition_count"]
            if "repetition_count" in data.files
            else np.zeros(len(game_id), dtype=np.uint8)
        )
        cycle_entry = (
            data["cycle_entry"]
            if "cycle_entry" in data.files
            else None
        )
        endgame = (
            data["endgame"] if "endgame" in data.files else None
        )
        cycle_cleaned = (
            data["cycle_cleaned"]
            if "cycle_cleaned" in data.files
            else None
        )
        wdl = data["wdl"]
        for index in range(len(game_id)):
            lo, hi = int(offsets[index]), int(offsets[index + 1])
            result.append(
                SearchSample(
                    game_id=int(game_id[index]),
                    state_hash=int(state_hash[index]),
                    observation=observation[index].copy(),
                    legal_actions=legal_actions[lo:hi].copy(),
                    repetition_contexts=repetition_contexts[lo:hi].copy(),
                    visits=visits[lo:hi].copy(),
                    raw_visits=(
                        raw_visits[lo:hi].copy()
                        if raw_visits is not None
                        else visits[lo:hi].copy()
                    ),
                    legacy_search64_visits=(
                        legacy_search64_visits[lo:hi].copy()
                        if legacy_search64_visits is not None
                        else visits[lo:hi].copy()
                    ),
                    legacy_search64_valid=(
                        bool(legacy_search64_valid[index])
                        if legacy_search64_valid is not None
                        else not bool(
                            int(repetition_count[index]) >= 2
                            or np.any(repetition_contexts[lo:hi] > 0)
                        )
                    ),
                    cycle_teacher_eligible=(
                        bool(cycle_teacher_eligible[index])
                        if cycle_teacher_eligible is not None
                        else bool(
                            int(repetition_count[index]) >= 2
                            or np.any(repetition_contexts[lo:hi] > 0)
                        )
                    ),
                    cycle_teacher_adjusted=(
                        bool(cycle_teacher_adjusted[index])
                        if cycle_teacher_adjusted is not None
                        else bool(
                            (
                                int(repetition_count[index]) >= 2
                                or np.any(repetition_contexts[lo:hi] > 0)
                            )
                            and int(visits[lo:hi].sum()) < 255
                        )
                    ),
                    cycle_teacher_rational_repeat_preserved=(
                        bool(
                            cycle_teacher_rational_repeat_preserved[index]
                        )
                        if cycle_teacher_rational_repeat_preserved is not None
                        else False
                    ),
                    player=int(player[index]),
                    ply=int(ply[index]),
                    root_entropy=float(root_entropy[index]),
                    position_key=int(position_key[index]),
                    repetition_count=int(repetition_count[index]),
                    cycle_entry=(
                        bool(cycle_entry[index])
                        if cycle_entry is not None
                        else bool(
                            int(repetition_count[index]) > 0
                            or np.any(repetition_contexts[lo:hi] > 0)
                        )
                    ),
                    endgame=(
                        bool(endgame[index])
                        if endgame is not None
                        else bool(int(ply[index]) >= 80)
                    ),
                    cycle_cleaned=(
                        bool(cycle_cleaned[index])
                        if cycle_cleaned is not None
                        else False
                    ),
                    wdl=wdl[index].astype(np.float32),
                )
            )
    return result


def read_selfplay_state_hashes(path: Path) -> np.ndarray:
    """Read only the compact index used by diversity/replay accounting."""
    with np.load(path, allow_pickle=False) as data:
        return data["state_hash"].copy()


class SearchSelfPlay:
    def __init__(
        self,
        config: Any,
        backend: Any,
        model: PolicyValueNet,
        device: torch.device,
    ) -> None:
        self.config = config
        self.backend = backend
        self.search = BatchedNeuralPUCT(
            model.to(device).eval(),
            device=device,
            cpuct=config.puct.cpuct,
            inference_batch_size=config.puct.inference_batch_size,
            cache_size=int(getattr(config.puct, "cache_size", 0)),
            leaves_per_tree=int(getattr(config.puct, "leaves_per_tree", 1)),
        )

    @staticmethod
    def _target_entropy(visits: np.ndarray) -> float:
        probability = visits.astype(np.float64)
        if probability.sum() <= 0:
            return 0.0
        probability /= probability.sum()
        return float(
            -np.sum(
                probability
                * np.log(np.clip(probability, 1e-12, 1.0))
            )
        )

    @staticmethod
    def _cycle_teacher_visits(
        result: Any,
        contexts: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Soften avoidable repetition labels without changing game rules."""
        visits = result.visits.astype(np.int32, copy=True)
        repeated = contexts > 0
        non_repeated = ~repeated
        visited = visits > 0
        metric: dict[str, Any] = {
            "eligible": bool(np.any(repeated) and np.any(non_repeated)),
            "adjusted": False,
            "rational_repeat_preserved": False,
            "best_repeat_q": None,
            "best_exit_q": None,
        }
        if not metric["eligible"]:
            return visits, metric
        repeat_candidates = repeated & visited
        exit_candidates = non_repeated & visited
        if not np.any(repeat_candidates) or not np.any(exit_candidates):
            metric["insufficient_root_coverage"] = True
            return visits, metric
        repeat_q = float(np.max(result.q_values[repeat_candidates]))
        exit_q = float(np.max(result.q_values[exit_candidates]))
        metric["best_repeat_q"] = repeat_q
        metric["best_exit_q"] = exit_q
        metric["q_exit_minus_repeat"] = exit_q - repeat_q
        if exit_q >= repeat_q - CYCLE_TEACHER_EPSILON:
            scaled = np.rint(
                visits[repeated].astype(np.float64)
                * CYCLE_REPEAT_VISIT_SCALE
            ).astype(np.int32)
            scaled[(visits[repeated] > 0) & (scaled == 0)] = 1
            visits[repeated] = scaled
            metric["adjusted"] = True
        else:
            metric["rational_repeat_preserved"] = True
        return visits, metric

    @staticmethod
    def _trim_game_samples(
        game: SelfPlayGame,
    ) -> tuple[list[SearchSample], dict[str, int]]:
        """Keep pre-cycle play plus one full terminal cycle, then deduplicate."""
        samples = game.samples
        suffix_removed = 0
        if game.state.terminal_reason() == "REPETITION":
            cycle = repetition_cycle_details(
                game.position_hashes,
                game.action_rows,
                [None] * len(game.position_hashes),
                terminal_value=0.0,
                opponent_type="network_selfplay",
            )
            keep_count = int(cycle["first_cycle_ply"]) + int(
                cycle["cycle_length"]
            )
            keep_count = min(len(samples), max(0, keep_count))
            suffix_removed = len(samples) - keep_count
            samples = samples[:keep_count]
        unique = []
        seen: set[tuple[int, int, bytes]] = set()
        duplicates_removed = 0
        for sample in samples:
            key = (
                int(sample.state_hash),
                sample.observation.astype(np.float32, copy=False).tobytes(),
                sample.repetition_contexts.astype(
                    np.uint8, copy=False
                ).tobytes(),
            )
            if key in seen:
                duplicates_removed += 1
                continue
            seen.add(key)
            unique.append(sample)
        return unique, {
            "cycle_suffix_samples_removed": suffix_removed,
            "same_position_context_samples_removed": duplicates_removed,
        }

    def collect_block(
        self,
        iteration: int,
        first_game: int,
        count: int,
        progress_callback: Callable[[dict[str, int]], None] | None = None,
    ) -> tuple[list[SearchSample], dict[str, Any]]:
        games = []
        for index in range(count):
            state = self.backend.new_state(
                self.config.seed
                + iteration * 1_000_000
                + first_game + index
            )
            games.append(SelfPlayGame(
                game_id=first_game + index,
                state=state,
                rng=random.Random(
                    self.config.seed
                    ^ (iteration * 0x9E3779B1)
                    ^ (first_game + index)
                ),
                position_hashes=[int(state.native_state.position_key())],
            ))
        active = games[:]
        ply_round = 0
        search_seconds = 0.0
        simulation_count = 0
        legacy_search64_seconds = 0.0
        legacy_search64_simulation_count = 0
        while active:
            groups: dict[int, list[SelfPlayGame]] = {}
            for game in active:
                contexts = np.asarray(
                    game.state.repetition_contexts(), np.uint8
                )
                cycle_sensitive = (
                    int(game.state.native_state.repetition_count()) >= 2
                    or bool(np.any(contexts > 0))
                )
                simulations = (
                    self.config.puct.repeated_state_simulations
                    if cycle_sensitive
                    else self.config.puct.selfplay_simulations
                )
                groups.setdefault(simulations, []).append(game)
            result_by_game = {}
            legacy_search64_by_game = {}
            for simulations, selected in groups.items():
                if not selected:
                    continue
                results, metric = self.search.search(
                    [game.state for game in selected],
                    simulations,
                    add_root_noise=True,
                    epsilon=self.config.puct.dirichlet_epsilon,
                    noise_seeds=[
                        self.config.seed
                        ^ game.game_id
                        ^ (len(game.trace) * 0xA511E9B3)
                        for game in selected
                    ],
                )
                search_seconds += metric["elapsed_seconds"]
                simulation_count += len(selected) * simulations
                for game, result in zip(selected, results):
                    result_by_game[id(game)] = result
                    if simulations == self.config.puct.selfplay_simulations:
                        legacy_search64_by_game[id(game)] = result.visits
                legacy_selected = [
                    game
                    for game in selected
                    if simulations != self.config.puct.selfplay_simulations
                    and int(game.game_id) % 10 == 0
                    and getattr(self.config.puct, "legacy_cycle_diagnostic", False)
                ]
                if legacy_selected:
                    legacy_results, legacy_metric = self.search.search(
                        [game.state for game in legacy_selected],
                        self.config.puct.selfplay_simulations,
                        add_root_noise=True,
                        epsilon=self.config.puct.dirichlet_epsilon,
                        noise_seeds=[
                            self.config.seed
                            ^ game.game_id
                            ^ (len(game.trace) * 0xA511E9B3)
                            for game in legacy_selected
                        ],
                    )
                    legacy_search64_seconds += legacy_metric[
                        "elapsed_seconds"
                    ]
                    legacy_search64_simulation_count += (
                        len(legacy_selected)
                        * self.config.puct.selfplay_simulations
                    )
                    for game, legacy in zip(
                        legacy_selected, legacy_results
                    ):
                        current = result_by_game[id(game)]
                        if not np.array_equal(
                            current.actions, legacy.actions
                        ):
                            raise RuntimeError(
                                "cycle Search256/Search64 legal actions differ"
                            )
                        legacy_search64_by_game[id(game)] = legacy.visits
            next_active = []
            for game in active:
                result = result_by_game[id(game)]
                ply = len(game.trace)
                contexts = np.asarray(
                    game.state.repetition_contexts(), np.uint8
                )
                repetition_count = int(
                    game.state.native_state.repetition_count()
                )
                cycle_sensitive = (
                    repetition_count >= 2 or bool(np.any(contexts > 0))
                )
                teacher_visits = result.visits
                teacher_metric = {
                    "eligible": False,
                    "adjusted": False,
                    "rational_repeat_preserved": False,
                }
                if cycle_sensitive and getattr(self.config.puct, "cycle_target_mode", "raw") == "adjusted":
                    teacher_visits, teacher_metric = (
                        self._cycle_teacher_visits(result, contexts)
                    )
                observation = game.state.feature_planes()
                live_pieces = int(
                    np.count_nonzero(observation[14:38] > 0.5)
                )
                sample = SearchSample(
                    game_id=game.game_id,
                    state_hash=int(game.state.native_state.key()),
                    position_key=int(
                        game.state.native_state.position_key()
                    ),
                    observation=observation,
                    legal_actions=result.actions,
                    repetition_contexts=contexts,
                    visits=teacher_visits,
                    raw_visits=result.visits.copy(),
                    legacy_search64_visits=(
                        legacy_search64_by_game[id(game)].copy()
                        if id(game) in legacy_search64_by_game
                        else None
                    ),
                    legacy_search64_valid=(
                        id(game) in legacy_search64_by_game
                    ),
                    cycle_teacher_eligible=bool(
                        teacher_metric["eligible"]
                    ),
                    cycle_teacher_adjusted=bool(
                        teacher_metric["adjusted"]
                    ),
                    cycle_teacher_rational_repeat_preserved=bool(
                        teacher_metric["rational_repeat_preserved"]
                    ),
                    player=int(game.state.current_player()),
                    ply=ply,
                    root_entropy=self._target_entropy(teacher_visits),
                    repetition_count=repetition_count,
                    cycle_entry=cycle_sensitive,
                    endgame=(live_pieces <= 24 or ply >= 80),
                    cycle_cleaned=True,
                )
                game.samples.append(sample)
                if cycle_sensitive:
                    game.action_rows.append(
                        {
                            "cycle_teacher": teacher_metric,
                        }
                    )
                temperature = (
                    1.0 if ply < 24 or cycle_sensitive else 0.5
                )
                visits = teacher_visits.astype(np.float64)
                if visits.sum() <= 0:
                    probability = result.priors.astype(np.float64)
                else:
                    probability = np.power(
                        np.maximum(visits, 1e-12), 1.0 / temperature
                    )
                probability /= probability.sum()
                action = int(
                    game.rng.choices(
                        result.actions.tolist(),
                        weights=probability.tolist(),
                        k=1,
                    )[0]
                )
                mover = int(game.state.current_player())
                game.state.apply(action)
                game.trace.append(action)
                action_row = {
                    "action": action,
                    "mover_side": mover,
                }
                if cycle_sensitive:
                    action_row["cycle_teacher"] = teacher_metric
                    game.action_rows[-1] = action_row
                else:
                    game.action_rows.append(action_row)
                game.position_hashes.append(
                    int(game.state.native_state.position_key())
                )
                if not game.state.is_terminal():
                    next_active.append(game)
            active = next_active
            ply_round += 1
            if (
                progress_callback is not None
                and (ply_round == 1 or ply_round % 10 == 0 or not active)
            ):
                progress_callback(
                    {
                        "ply_round": ply_round,
                        "active_games": len(active),
                        "finished_games": count - len(active),
                        "simulation_count": simulation_count,
                    }
                )
        terminal_reasons = Counter()
        trajectory_hashes = Counter()
        prefixes = {8: Counter(), 16: Counter(), 24: Counter()}
        all_samples = []
        state_counts = Counter()
        cycle_types = Counter()
        cycle_signatures = Counter()
        trim_totals = Counter()
        cycle_teacher = Counter()
        for game in games:
            terminal_reasons[str(game.state.terminal_reason())] += 1
            encoded = np.asarray(game.trace, np.int16).tobytes()
            trajectory_hashes[hashlib.sha256(encoded).hexdigest()] += 1
            if game.state.terminal_reason() == "REPETITION":
                cycle = repetition_cycle_details(
                    game.position_hashes,
                    game.action_rows,
                    [None] * len(game.position_hashes),
                    terminal_value=0.0,
                    opponent_type="network_selfplay",
                )
                length = int(cycle["cycle_length"])
                cycle_types[
                    "2_ply" if length == 2
                    else ("4_ply" if length == 4 else "other")
                ] += 1
                cycle_signatures[
                    ">".join(cycle["position_hash_sequence"])
                ] += 1
            for length, counter in prefixes.items():
                prefix = np.asarray(game.trace[:length], np.int16).tobytes()
                counter[hashlib.sha256(prefix).hexdigest()] += 1
            trimmed, trim_metric = self._trim_game_samples(game)
            trim_totals.update(trim_metric)
            for row in game.action_rows:
                teacher = row.get("cycle_teacher")
                if not teacher:
                    continue
                cycle_teacher["roots"] += 1
                cycle_teacher["eligible"] += int(teacher["eligible"])
                cycle_teacher["adjusted"] += int(teacher["adjusted"])
                cycle_teacher["rational_repeat_preserved"] += int(
                    teacher["rational_repeat_preserved"]
                )
                cycle_teacher["insufficient_root_coverage"] += int(
                    teacher.get("insufficient_root_coverage", False)
                )
            for sample in trimmed:
                state_counts[sample.state_hash] += 1
                value = float(game.state.terminal_value(sample.player))
                sample.wdl = np.asarray(
                    [value < 0, value == 0, value > 0], np.float32
                )
                all_samples.append(sample)
        # Enforce the per-iteration Policy cap deterministically.  Value-only
        # copies may be added later through a separate stratified buffer.
        capped = []
        accepted = Counter()
        for sample in all_samples:
            if (
                accepted[sample.state_hash]
                >= self.config.loop.state_policy_cap_per_iteration
            ):
                continue
            accepted[sample.state_hash] += 1
            capped.append(sample)
        total_states = len(all_samples)
        total_plies = sum(len(game.trace) for game in games)
        report = {
            "games": len(games),
            "samples_before_state_cap": total_states,
            "policy_samples": len(capped),
            "replay_cycle_cleanup": dict(trim_totals),
            "cycle_teacher": {
                **dict(cycle_teacher),
                "simulations": CYCLE_TEACHER_SIMULATIONS,
                "epsilon": CYCLE_TEACHER_EPSILON,
                "repeat_visit_scale": CYCLE_REPEAT_VISIT_SCALE,
            },
            "unique_states": len(state_counts),
            "unique_state_ratio": len(state_counts) / max(1, total_states),
            "unique_trajectories": len(trajectory_hashes),
            "trajectory_counts": dict(trajectory_hashes),
            "maximum_trajectory_share": (
                max(trajectory_hashes.values()) / len(games)
            ),
            "unique_prefixes": {
                str(length): len(counter)
                for length, counter in prefixes.items()
            },
            "prefix_counts": {
                str(length): dict(counter)
                for length, counter in prefixes.items()
            },
            "maximum_prefix_share": {
                str(length): max(counter.values()) / len(games)
                for length, counter in prefixes.items()
            },
            "root_visit_entropy": float(
                np.mean([sample.root_entropy for sample in capped])
            ),
            "terminal_reasons": dict(terminal_reasons),
            "repetition_fraction": (
                terminal_reasons.get("REPETITION", 0) / len(games)
            ),
            "cycle_types": dict(cycle_types),
            "maximum_single_cycle_share_of_repetitions": (
                max(cycle_signatures.values(), default=0)
                / max(1, terminal_reasons.get("REPETITION", 0))
            ),
            "simulation_count": simulation_count,
            "search_seconds": search_seconds,
            "simulations_per_second": simulation_count / max(1e-9, search_seconds),
            "legacy_search64_diagnostic": {
                "heldout_cycle_roots": int(
                    legacy_search64_simulation_count
                    / max(1, self.config.puct.selfplay_simulations)
                ),
                "simulation_count": legacy_search64_simulation_count,
                "search_seconds": legacy_search64_seconds,
                "simulations_per_second": (
                    legacy_search64_simulation_count
                    / max(1e-9, legacy_search64_seconds)
                ),
                "does_not_change_training_target_or_action_sampling": True,
            },
            "total_plies": total_plies,
            "average_ply": total_plies / max(1, len(games)),
        }
        return capped, report


def search_distill_batch(
    samples: list[SearchSample],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    observations = torch.from_numpy(
        np.stack([sample.observation for sample in samples])
    ).to(device=device, dtype=torch.float32)
    legal_mask_cpu = np.zeros((len(samples), 3600), dtype=np.bool_)
    repetition_cpu = np.zeros((len(samples), 3600), dtype=np.uint8)
    target_cpu = np.zeros((len(samples), 3600), dtype=np.float32)
    for row, sample in enumerate(samples):
        legal = sample.legal_actions.astype(np.int64, copy=False)
        legal_mask_cpu[row, legal] = True
        repetition_cpu[row, legal] = sample.repetition_contexts
        visits = sample.visits.astype(np.float32)
        probability = visits / max(1.0, float(visits.sum()))
        target_cpu[row, legal] = probability
    legal_mask = torch.from_numpy(legal_mask_cpu).to(device)
    repetition = torch.from_numpy(repetition_cpu).to(device)
    target = torch.from_numpy(target_cpu).to(device)
    wdl = torch.from_numpy(
        np.stack([sample.wdl for sample in samples])
    ).to(device)
    return {
        "observations": observations,
        "legal_mask": legal_mask,
        "repetition": repetition,
        "target": target,
        "wdl": wdl,
    }
