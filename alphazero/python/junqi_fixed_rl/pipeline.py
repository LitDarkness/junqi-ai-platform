from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.config import ReplayConfig as DiskReplayConfig
from junqi_az.model import PolicyValueNet
from junqi_az.replay import ShardedReplayBuffer, SparseSample

from .config import FixedRLConfig
from .checkpoints import CheckpointManager
from .evaluator import (
    IndependentEvaluator,
    TERMINAL_CATEGORIES,
    repetition_cycle_details,
    repetition_severity,
    terminal_category,
)
from .layout import FixedSameLayoutBackend, load_fixed_layout
from .metrics import MetricsWriter
from .teachers import TeacherEnsemble, aggressive_policy, random_policy
from .trainer import FixedRLTrainer


@dataclass
class Position:
    observation: np.ndarray
    legal: np.ndarray
    repetitions: np.ndarray
    probabilities: np.ndarray
    action: int
    player: int
    actor: str
    ply: int
    predicted_value: float | None
    moved_piece: int


@dataclass
class Episode:
    state: Any
    game_key: int
    learner_side: int
    opponent: str
    opponent_category: str
    history_slot: int | None
    positions: list[Position] = field(default_factory=list)
    position_hashes: list[int] = field(default_factory=list)
    moves: int = 0


def _probabilities_to_visits(probabilities: np.ndarray) -> np.ndarray:
    scaled = np.maximum(1, np.rint(probabilities * 10000.0)).astype(np.uint16)
    return scaled


def _one_hot_visits(legal: np.ndarray, action: int) -> np.ndarray:
    visits = np.zeros(len(legal), dtype=np.uint16)
    positions = np.flatnonzero(legal == int(action))
    if len(positions) != 1:
        raise RuntimeError(f"selected action {action} is not uniquely legal")
    visits[int(positions[0])] = 1
    return visits


class FixedLayoutRLPipeline:
    def __init__(
        self,
        config_path: str | Path,
        *,
        overrides: dict[str, int | float | str] | None = None,
    ) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent.parent
        self.config = FixedRLConfig.load(self.config_path)
        for dotted, value in (overrides or {}).items():
            if "." in dotted:
                section_name, field_name = dotted.split(".", 1)
                setattr(getattr(self.config, section_name), field_name, value)
            else:
                setattr(self.config, dotted, value)
        self.config.validate(self.project_root)
        self._seed_everything(self.config.seed)
        self.run_directory = self._resolve(self.config.run_directory)
        self.run_directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints = CheckpointManager(self.run_directory, self.config)
        self.metrics = MetricsWriter(self.run_directory / "metrics.jsonl")

        self.layout = load_fixed_layout(
            self.config.layout.set_path,
            self.config.layout.layout_id,
            project_root=self.project_root,
        )
        self.backend = FixedSameLayoutBackend(
            self.layout, self.config.layout.mirror_probability
        )
        self.teacher = TeacherEnsemble(
            self.config.teacher,
            project_root=self.project_root,
            seed=self.config.seed + 17,
        )
        self.model = PolicyValueNet(self.config.model)
        self.trainer = FixedRLTrainer(self.config, self.model)
        self.teacher_replay = self._make_replay(
            self.config.replay.teacher_directory, self.config.seed + 101
        )
        self.selfplay_replay = self._make_replay(
            self.config.replay.selfplay_directory, self.config.seed + 202
        )
        self.iteration = 0
        self.games_generated = 0
        self._history_models: list[tuple[PolicyValueNet, str]] = []
        self._transaction_base_iteration: int | None = None
        self._transaction_payload: dict[str, Any] | None = None
        self._resume_and_recover()
        self._write_manifest(self.layout)
        if self.iteration > 0:
            migration = self.checkpoints.migrate_legacy_history(self.iteration)
            self.metrics.write("checkpoint_migration", **migration)
            if (
                self.iteration
                >= self.config.checkpoint.initial_champion_iteration
            ):
                self._initialize_champion()
            self._load_history_models()

    def _resolve(self, configured: str | Path) -> Path:
        path = Path(configured)
        return path if path.is_absolute() else (self.project_root / path).resolve()

    @staticmethod
    def _seed_everything(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _make_replay(self, directory: str, seed: int) -> ShardedReplayBuffer:
        replay_path = Path(directory)
        if not replay_path.is_absolute():
            replay_path = (self.run_directory / replay_path).resolve()
        config = DiskReplayConfig(
            directory=str(replay_path),
            capacity=self.config.replay.capacity,
            shard_size=self.config.replay.shard_size,
            observation_dtype=self.config.replay.observation_dtype,
            cache_shards=self.config.replay.cache_shards,
            sample_shards_per_batch=self.config.replay.sample_shards_per_batch,
            recent_fraction=0.75,
            recent_window_shards=16,
            require_game_provenance=True,
            require_repetition_context=True,
            validation_fraction=0.0,
            split_seed=self.config.seed,
        )
        return ShardedReplayBuffer(config, seed=seed)

    @staticmethod
    def _training_config(payload: dict[str, Any]) -> dict[str, Any]:
        result = copy.deepcopy(payload)
        result.pop("checkpoint", None)
        result.pop("evaluation", None)
        return result

    def _config_hash(self) -> str:
        return self.checkpoints.hash_payload(self.config.to_dict())

    def _resume_and_recover(self) -> None:
        latest = self.checkpoints.latest_path
        transaction = self.checkpoints.read_transaction()
        selected: Path | None = None
        recovery_required = False

        if transaction is not None and latest.is_file():
            latest_payload = torch.load(latest, map_location="cpu", weights_only=False)
            target = int(transaction.get("target_iteration", -1))
            if (
                self.checkpoints.is_complete_payload(latest_payload)
                and int(latest_payload.get("iteration", -2)) == target
            ):
                # Checkpoint replacement was durable; only deferred old shards
                # and the transaction marker survived the crash.
                removed = {
                    "teacher": self.teacher_replay.cleanup_orphans(),
                    "selfplay": self.selfplay_replay.cleanup_orphans(),
                }
                self.checkpoints.register_completed(target)
                self.checkpoints.clear_transaction()
                self.metrics.write(
                    "transaction_recovered_committed",
                    iteration=target,
                    orphan_shards_removed=removed,
                )
            else:
                base_path = Path(transaction["base_checkpoint"])
                selected = base_path
                recovery_required = True

        if selected is None and latest.is_file():
            payload = torch.load(latest, map_location="cpu", weights_only=False)
            if self.checkpoints.is_complete_payload(payload):
                selected = latest
            else:
                selected = self.checkpoints.find_last_complete_legacy()
                recovery_required = True
        if selected is None and not latest.is_file():
            return
        if selected is None:
            raise RuntimeError("no complete checkpoint is available for recovery")

        payload = torch.load(selected, map_location="cpu", weights_only=False)
        if recovery_required:
            teacher_report = self.teacher_replay.rollback_to_state(
                payload["teacher_replay"]
            )
            selfplay_report = self.selfplay_replay.rollback_to_state(
                payload["selfplay_replay"]
            )
            self.checkpoints.clear_transaction()
            self.metrics.write(
                "transaction_rollback",
                checkpoint=str(selected),
                iteration=int(payload["iteration"]),
                teacher=teacher_report,
                selfplay=selfplay_report,
            )
            print(
                "[fixed-rl recovery] restored complete checkpoint "
                f"iteration={payload['iteration']} from {selected}; "
                f"selfplay={selfplay_report}",
                flush=True,
            )
        self.load_checkpoint(selected, restore_replay=not recovery_required)
        if recovery_required or selected != latest or not payload.get(
            "iteration_completed", False
        ):
            self.save_checkpoint(
                completed_games=self.config.selfplay.games_per_iteration
                if self.iteration > 0
                else 0,
                register=False,
            )

    def _initialize_champion(self) -> None:
        metadata = self.checkpoints.initialize_champion(
            layout_id=self.layout.layout_id,
            layout_hash=hashlib.sha256(
                ",".join(str(value) for value in self.layout.setup).encode("ascii")
            ).hexdigest(),
            config_hash=self._config_hash(),
        )
        self.metrics.write(
            "champion_initialized",
            source_iteration=int(metadata["source_iteration"]),
            checkpoint=metadata["checkpoint"],
        )

    def _write_manifest(self, layout: Any) -> None:
        payload = {
            "schema": self.config.schema,
            "config": self.config.to_dict(),
            "config_path": str(self.config_path),
            "layout": {
                "id": layout.layout_id,
                "name_zh": layout.name_zh,
                "canonical_setup30": list(layout.setup),
                "source_sha256": layout.source_sha256,
                "same_for_both_sides": True,
            },
            "teacher_modules": self.teacher.identities,
            "network_parameters": sum(
                parameter.numel() for parameter in self.model.parameters()
            ),
        }
        target = self.run_directory / "manifest.json"
        target.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @property
    def temperature(self) -> float:
        selfplay = self.config.selfplay
        progress = min(
            1.0, self.games_generated / max(1, selfplay.temperature_decay_steps)
        )
        return max(
            selfplay.minimum_temperature,
            selfplay.policy_temperature
            + (selfplay.minimum_temperature - selfplay.policy_temperature)
            * progress,
        )

    def _new_episode(self, game_number: int, *, teacher_only: bool) -> Episode:
        seed = self.config.seed + game_number * 104729
        state = self.backend.new_state(seed)
        if teacher_only:
            episode = Episode(
                state=state,
                game_key=game_number + 1,
                learner_side=1,
                opponent="teacher_bootstrap",
                opponent_category="teacher_bootstrap",
                history_slot=None,
            )
            episode.position_hashes.append(
                int(state.native_state.position_key())
            )
            return episode
        keys = list(self.config.selfplay.opponent_weights)
        weights = [self.config.selfplay.opponent_weights[key] for key in keys]
        opponent = random.choices(keys, weights=weights, k=1)[0]
        history_slot = None
        if opponent == "historical" and self._history_models:
            history_slot = random.randrange(len(self._history_models))
            opponent_category = self._history_models[history_slot][1]
        elif opponent == "historical":
            opponent = "current"
            opponent_category = "current/latest"
        else:
            opponent_category = {
                "current": "current/latest",
                "heuristic_current": "084",
                "heuristic_historical": "082",
                "random": "random_exploiter",
                "aggressive": "aggressive_exploiter",
            }[opponent]
        episode = Episode(
            state=state,
            game_key=game_number + 1,
            learner_side=random.choice((1, -1)),
            opponent=opponent,
            opponent_category=opponent_category,
            history_slot=history_slot,
        )
        episode.position_hashes.append(
            int(state.native_state.position_key())
        )
        return episode

    def _actor_for(self, episode: Episode) -> tuple[str, int | None]:
        if episode.opponent == "teacher_bootstrap":
            # Mix the strongest and historical teacher across deterministic
            # positions without paying both alpha-beta searches at every root.
            actor = (
                "teacher_historical"
                if (episode.game_key + episode.moves) % 4 == 0
                else "teacher_current"
            )
            return actor, None
        if episode.state.current_player() == episode.learner_side:
            return "current", None
        if episode.opponent == "historical":
            return "historical", episode.history_slot
        return episode.opponent, None

    def _infer(
        self, states: list[Any], model: PolicyValueNet
    ) -> list[tuple[int, np.ndarray, np.ndarray, float]]:
        observations = np.stack([state.feature_planes() for state in states])
        masks = np.zeros(
            (len(states), self.config.model.action_size), dtype=np.bool_
        )
        repetitions = np.zeros_like(masks, dtype=np.uint8)
        legals: list[np.ndarray] = []
        for row, state in enumerate(states):
            legal = np.asarray(state.legal_actions(), dtype=np.int32)
            context = np.asarray(state.repetition_contexts(), dtype=np.uint8)
            legals.append(legal)
            masks[row, legal] = True
            repetitions[row, legal] = context
        device = self.trainer.device
        model.eval()
        with torch.inference_mode():
            probabilities, values = model.policy_value(
                torch.from_numpy(observations).to(device),
                torch.from_numpy(masks).to(device),
                torch.from_numpy(repetitions).to(device),
            )
        probabilities_np = probabilities.cpu().numpy()
        values_np = values.float().cpu().numpy().reshape(-1)
        result = []
        temperature = self.temperature
        for row, legal in enumerate(legals):
            legal_probabilities = probabilities_np[row, legal].astype(np.float64)
            if temperature != 1.0:
                legal_probabilities = np.power(
                    np.clip(legal_probabilities, 1e-12, 1.0),
                    1.0 / temperature,
                )
            legal_probabilities /= legal_probabilities.sum()
            action = random.choices(
                legal.tolist(), weights=legal_probabilities.tolist(), k=1
            )[0]
            result.append(
                (
                    int(action),
                    legal,
                    legal_probabilities.astype(np.float32),
                    float(values_np[row]),
                )
            )
        return result

    def _finish_episode(
        self, episode: Episode, replay: ShardedReplayBuffer, *, teacher_only: bool
    ) -> int:
        terminal = episode.state.is_terminal()
        samples = 0
        for position in episode.positions:
            outcome = (
                float(episode.state.terminal_value(position.player))
                if terminal
                else 0.0
            )
            if teacher_only:
                policy_weight = 1.0
                visits = _probabilities_to_visits(position.probabilities)
            elif position.actor == "current":
                # Outcome-weighted behaviour learning: reinforce sampled moves
                # that won, retain draws weakly, and do not clone losing moves.
                policy_weight = max(0.0, 1.0 + outcome)
                visits = _one_hot_visits(position.legal, position.action)
            elif position.actor.startswith("teacher_") or position.actor.startswith(
                "heuristic_"
            ):
                policy_weight = 0.35
                visits = _probabilities_to_visits(position.probabilities)
            else:
                policy_weight = 0.0
                visits = _one_hot_visits(position.legal, position.action)
            replay.append(
                SparseSample(
                    observation=position.observation,
                    legal_actions=position.legal,
                    visit_counts=visits,
                    outcome=outcome,
                    policy_weight=policy_weight,
                    repetition_contexts=position.repetitions,
                    game_key=episode.game_key,
                    ply_index=position.ply,
                )
            )
            samples += 1
        return samples

    def generate_games(
        self,
        games: int,
        *,
        teacher_only: bool = False,
        replay: ShardedReplayBuffer | None = None,
        progress: bool = True,
        game_number_start: int | None = None,
    ) -> dict[str, Any]:
        destination = replay or (
            self.teacher_replay if teacher_only else self.selfplay_replay
        )
        start = time.perf_counter()
        completed = 0
        samples = 0
        total_moves = 0
        opponent_categories = (
            "current/latest",
            "recent_checkpoint",
            "champion",
            "084",
            "082",
            "random_exploiter",
            "aggressive_exploiter",
        )
        opponent_stats = {
            category: {
                "games": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
                "moves": 0,
                "terminal_reasons": {
                    reason: 0 for reason in TERMINAL_CATEGORIES
                },
            }
            for category in opponent_categories
        }
        terminal_counts = {category: 0 for category in TERMINAL_CATEGORIES}
        first_game = (
            self.games_generated
            if game_number_start is None
            else int(game_number_start)
        )
        next_game = first_game
        peak_active = min(self.config.selfplay.parallel_games, games)
        report_interval = max(1, min(32, games))
        last_reported = -1
        active = [
            self._new_episode(next_game + index, teacher_only=teacher_only)
            for index in range(peak_active)
        ]
        next_game += len(active)
        while active:
            neural_groups: dict[tuple[str, int | None], list[int]] = {}
            decisions: dict[
                int,
                tuple[int, np.ndarray, np.ndarray, str, float | None],
            ] = {}
            teacher_indices: list[int] = []
            teacher_history: list[bool] = []
            for index, episode in enumerate(active):
                actor, slot = self._actor_for(episode)
                if actor in {"current", "historical"}:
                    neural_groups.setdefault((actor, slot), []).append(index)
                    continue
                if actor in {"teacher_current", "heuristic_current"}:
                    teacher_indices.append(index)
                    teacher_history.append(False)
                elif actor in {"teacher_historical", "heuristic_historical"}:
                    teacher_indices.append(index)
                    teacher_history.append(True)
                elif actor == "aggressive":
                    action, legal, probabilities = aggressive_policy(episode.state)
                    decisions[index] = (
                        action, legal, probabilities, actor, None
                    )
                elif actor == "random":
                    action, legal, probabilities = random_policy(episode.state)
                    decisions[index] = (
                        action, legal, probabilities, actor, None
                    )
                else:
                    raise RuntimeError(f"unknown population actor: {actor}")

            if teacher_indices:
                teacher_decisions = self.teacher.sample_actions(
                    [active[index].state for index in teacher_indices],
                    teacher_history,
                )
                for index, (action, legal, probabilities) in zip(
                    teacher_indices, teacher_decisions
                ):
                    actor, _ = self._actor_for(active[index])
                    decisions[index] = (
                        action, legal, probabilities, actor, None
                    )

            for (actor, slot), indices in neural_groups.items():
                model = (
                    self._history_models[int(slot)][0]
                    if actor == "historical" and slot is not None
                    else self.model
                )
                inferred = self._infer([active[index].state for index in indices], model)
                for index, (
                    action,
                    legal,
                    probabilities,
                    predicted_value,
                ) in zip(indices, inferred):
                    decisions[index] = (
                        action,
                        legal,
                        probabilities,
                        actor,
                        predicted_value,
                    )

            finished: list[int] = []
            for index, episode in enumerate(active):
                (
                    action,
                    legal,
                    probabilities,
                    actor,
                    predicted_value,
                ) = decisions[index]
                contexts = np.asarray(
                    episode.state.repetition_contexts(), dtype=np.uint8
                )
                source, _ = divmod(int(action), 60)
                moved_piece = int(episode.state.native_state.board()[source])
                episode.positions.append(
                    Position(
                        observation=np.asarray(
                            episode.state.feature_planes(), dtype=np.float16
                        ),
                        legal=legal,
                        repetitions=contexts,
                        probabilities=probabilities,
                        action=action,
                        player=episode.state.current_player(),
                        actor=actor,
                        ply=episode.moves,
                        predicted_value=predicted_value,
                        moved_piece=moved_piece,
                    )
                )
                episode.state.apply(action)
                episode.moves += 1
                episode.position_hashes.append(
                    int(episode.state.native_state.position_key())
                )
                if (
                    episode.state.is_terminal()
                    or episode.moves >= self.config.selfplay.max_moves
                ):
                    samples += self._finish_episode(
                        episode, destination, teacher_only=teacher_only
                    )
                    total_moves += episode.moves
                    completed += 1
                    if not teacher_only:
                        category = episode.opponent_category
                        row = opponent_stats[category]
                        reason = terminal_category(
                            episode.state,
                            episode.moves >= self.config.selfplay.max_moves,
                        )
                        row["games"] += 1
                        row["moves"] += episode.moves
                        row["terminal_reasons"][reason] += 1
                        if episode.state.is_terminal():
                            learner_value = episode.state.terminal_value(
                                episode.learner_side
                            )
                        else:
                            learner_value = 0.0
                        if learner_value > 0:
                            row["wins"] += 1
                        elif learner_value < 0:
                            row["losses"] += 1
                        else:
                            row["draws"] += 1
                        terminal_counts[reason] += 1
                        if reason == "repetition_draw":
                            trace = []
                            for position in episode.positions:
                                source, target = divmod(position.action, 60)
                                trace.append(
                                    {
                                        "ply": position.ply,
                                        "actor": position.actor,
                                        "mover_side": (
                                            "red"
                                            if position.player == 1
                                            else "black"
                                        ),
                                        "action": position.action,
                                        "from": source,
                                        "to": target,
                                        "moved_piece": position.moved_piece,
                                    }
                                )
                            cycle = repetition_cycle_details(
                                episode.position_hashes,
                                trace,
                                [
                                    position.predicted_value
                                    for position in episode.positions
                                ],
                                terminal_value=0.0,
                                opponent_type=episode.opponent_category,
                            )
                            self.metrics.write(
                                "selfplay_repetition_cycle",
                                iteration=self.iteration,
                                game_key=episode.game_key,
                                setup_seed=(
                                    self.config.seed
                                    + (episode.game_key - 1) * 104729
                                ),
                                learner_side=(
                                    "red"
                                    if episode.learner_side == 1
                                    else "black"
                                ),
                                layout_id=self.layout.layout_id,
                                **cycle,
                            )
                    finished.append(index)

            for index in reversed(finished):
                active.pop(index)
            while len(active) < peak_active and next_game < first_game + games:
                active.append(
                    self._new_episode(next_game, teacher_only=teacher_only)
                )
                next_game += 1
            if progress and (
                not active
                or (
                    completed > 0
                    and completed != last_reported
                    and completed % report_interval == 0
                )
            ):
                elapsed = time.perf_counter() - start
                print(
                    f"[fixed-rl {'teacher' if teacher_only else 'selfplay'}] "
                    f"{completed}/{games} games moves={total_moves} "
                    f"moves/s={total_moves / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )
                last_reported = completed
                if not teacher_only:
                    self._update_transaction_progress(completed, samples)
        destination.flush()
        elapsed = time.perf_counter() - start
        if not teacher_only:
            self.games_generated += games
        result = {
            "games": float(completed),
            "samples": float(samples),
            "moves": float(total_moves),
            "elapsed_seconds": elapsed,
            "moves_per_second": total_moves / max(elapsed, 1e-9),
            "games_per_second": completed / max(elapsed, 1e-9),
            "parallel_games": float(peak_active),
            "temperature": self.temperature,
        }
        if not teacher_only:
            detailed = {}
            implementation_labels = {
                "current/latest": "current_policy_network",
                "recent_checkpoint": "archived_policy_network",
                "champion": "champion_policy_network",
                "084": "wide_teacher_ensemble_current",
                "082": "wide_teacher_ensemble_historical",
                "random_exploiter": "random_legal_policy",
                "aggressive_exploiter": "capture_biased_policy",
            }
            for category, row in opponent_stats.items():
                games_in_category = int(row["games"])
                score = int(row["wins"]) + 0.5 * int(row["draws"])
                reason_distribution = {
                    reason: {
                        "count": int(count),
                        "fraction": (
                            int(count) / games_in_category
                            if games_in_category
                            else 0.0
                        ),
                    }
                    for reason, count in row["terminal_reasons"].items()
                }
                repetition_fraction = reason_distribution[
                    "repetition_draw"
                ]["fraction"]
                detailed[category] = {
                    **{
                        key: value
                        for key, value in row.items()
                        if key != "terminal_reasons"
                    },
                    "score_rate": (
                        score / games_in_category
                        if games_in_category
                        else 0.0
                    ),
                    # Categories are interleaved. Their rates use the same
                    # iteration wall clock and therefore sum to total moves/s.
                    "moves_per_second": int(row["moves"])
                    / max(elapsed, 1e-9),
                    "average_ply": (
                        int(row["moves"]) / games_in_category
                        if games_in_category
                        else 0.0
                    ),
                    "terminal_reasons": reason_distribution,
                    "repetition_fraction": repetition_fraction,
                    "repetition_severity": repetition_severity(
                        repetition_fraction
                    ),
                    "opponent_implementation": implementation_labels[category],
                }
                self.metrics.write(
                    "selfplay_opponent",
                    iteration=self.iteration,
                    opponent=category,
                    **detailed[category],
                )
            terminal_summary = {
                category: {
                    "count": count,
                    "fraction": count / max(1, completed),
                }
                for category, count in terminal_counts.items()
            }
            warnings = []
            aggregate_repetition_fraction = terminal_summary[
                "repetition_draw"
            ]["fraction"]
            aggregate_repetition_severity = repetition_severity(
                aggregate_repetition_fraction
            )
            if (
                aggregate_repetition_fraction
                > self.config.evaluation.repetition_warning_fraction
            ):
                warnings.append(
                    "repetition_draw fraction is unusually high: "
                    f"{aggregate_repetition_fraction:.1%} "
                    f"({aggregate_repetition_severity})"
                )
            for category, row in detailed.items():
                if row["repetition_fraction"] >= 0.15:
                    warnings.append(
                        f"{category} repetition_draw="
                        f"{row['repetition_fraction']:.1%} "
                        f"({row['repetition_severity']})"
                    )
            if (
                terminal_summary["max_ply_cutoff"]["fraction"]
                > self.config.evaluation.max_ply_warning_fraction
            ):
                warnings.append("max_ply_cutoff fraction is unusually high")
            result["opponents"] = detailed
            result["total_games"] = int(completed)
            result["terminal_reasons"] = terminal_summary
            result["repetition_fraction"] = aggregate_repetition_fraction
            result["repetition_severity"] = aggregate_repetition_severity
            result["warnings"] = warnings
            self.metrics.write(
                "selfplay_terminal_reasons",
                iteration=self.iteration,
                total_games=int(completed),
                reasons=terminal_summary,
                repetition_fraction=aggregate_repetition_fraction,
                repetition_severity=aggregate_repetition_severity,
                warnings=warnings,
            )
            for warning in warnings:
                print(f"[fixed-rl WARNING] {warning}", flush=True)
            self._update_transaction_progress(completed, samples)
        self.metrics.write(
            "teacher_generation" if teacher_only else "selfplay", **result
        )
        return result

    def bootstrap(self) -> None:
        summary = self.teacher_replay.provenance_summary()
        missing_games = max(
            0, self.config.teacher.bootstrap_games - summary["games"]
        )
        if missing_games:
            print(
                f"[fixed-rl] teacher replay has {summary['games']} completed games / "
                f"{summary['samples']} samples; generating {missing_games} more games",
                flush=True,
            )
            self.generate_games(
                missing_games,
                teacher_only=True,
                replay=self.teacher_replay,
                game_number_start=summary["max_game_key"],
            )
        if len(self.teacher_replay) < self.config.train.batch_size:
            raise RuntimeError(
                "teacher bootstrap produced fewer positions than one training "
                f"batch: replay={len(self.teacher_replay)} "
                f"batch={self.config.train.batch_size}"
            )
        if self.trainer.global_step == 0:
            self.trainer.set_anchor()
            self.trainer.train_steps(
                self.teacher_replay,
                self.selfplay_replay,
                self.config.teacher.bootstrap_train_steps,
                metric_writer=self.metrics,
            )
            self.save_checkpoint(completed_games=0, register=False)

    def _begin_iteration_transaction(self, target_iteration: int) -> None:
        if not self.checkpoints.latest_path.is_file():
            raise RuntimeError("cannot begin an iteration without latest.pt")
        self.teacher_replay.begin_transaction()
        self.selfplay_replay.begin_transaction()
        self._transaction_base_iteration = self.iteration
        self._transaction_payload = {
            "schema": "junqi-fixed-layout-iteration-transaction-v1",
            "base_checkpoint": str(self.checkpoints.latest_path.resolve()),
            "base_iteration": self.iteration,
            "target_iteration": int(target_iteration),
            "phase": "selfplay",
            "completed_games": 0,
            "committed_samples": 0,
            "started_at": time.time(),
            "teacher_replay": self.teacher_replay.transaction_status(),
            "selfplay_replay": self.selfplay_replay.transaction_status(),
        }
        self.checkpoints.write_transaction(self._transaction_payload)

    def _update_transaction_progress(
        self, completed_games: int, samples: int
    ) -> None:
        if self._transaction_payload is None:
            return
        self._transaction_payload.update(
            {
                "completed_games": int(completed_games),
                "committed_samples": int(samples),
                "updated_at": time.time(),
                "teacher_replay": self.teacher_replay.transaction_status(),
                "selfplay_replay": self.selfplay_replay.transaction_status(),
            }
        )
        self.checkpoints.write_transaction(self._transaction_payload)

    def _set_transaction_phase(self, phase: str) -> None:
        if self._transaction_payload is None:
            raise RuntimeError("iteration transaction is not active")
        self._transaction_payload["phase"] = phase
        self._transaction_payload["updated_at"] = time.time()
        self.checkpoints.write_transaction(self._transaction_payload)

    def abort_iteration(self, reason: str = "interrupted") -> dict[str, Any]:
        transaction = self._transaction_payload or self.checkpoints.read_transaction()
        if transaction is None:
            return {"active": False}
        teacher = self.teacher_replay.rollback_transaction()
        selfplay = self.selfplay_replay.rollback_transaction()
        base_checkpoint = Path(transaction["base_checkpoint"])
        self.load_checkpoint(base_checkpoint, restore_replay=False)
        self.checkpoints.clear_transaction()
        self._transaction_payload = None
        self._transaction_base_iteration = None
        report = {
            "active": True,
            "reason": reason,
            "base_iteration": int(transaction["base_iteration"]),
            "discarded_iteration": int(transaction["target_iteration"]),
            "completed_games_before_abort": int(
                transaction.get("completed_games", 0)
            ),
            "teacher": teacher,
            "selfplay": selfplay,
        }
        self.metrics.write("iteration_aborted", **report)
        return report

    def train(self, iterations: int) -> None:
        self.bootstrap()
        self._maybe_evaluate_champion()
        for _ in range(int(iterations)):
            target_iteration = self.iteration + 1
            self._begin_iteration_transaction(target_iteration)
            self.iteration = target_iteration
            try:
                self.trainer.set_anchor()
                generated = self.generate_games(
                    self.config.selfplay.games_per_iteration
                )
                self._set_transaction_phase("train")
                trained = self.trainer.train_steps(
                    self.teacher_replay,
                    self.selfplay_replay,
                    self.config.train.steps_per_iteration,
                    metric_writer=self.metrics,
                )
                if not all(
                    torch.isfinite(parameter).all().item()
                    for parameter in self.model.parameters()
                ):
                    raise RuntimeError("NaN/Inf detected in model parameters")
                if trained.get("entropy", 1.0) < 0.05:
                    print(
                        "[fixed-rl WARNING] policy entropy collapse detected; "
                        "checkpoint remains available for manual review",
                        flush=True,
                    )
                if trained.get("kl_to_anchor_rho", 0.0) > 1.0:
                    print(
                        "[fixed-rl WARNING] KL to anchor exploded above 1.0; "
                        "checkpoint remains available for manual review",
                        flush=True,
                    )
                self._set_transaction_phase("checkpoint")
                checkpoint = self.save_checkpoint(
                    completed_games=int(generated["games"]), register=True
                )
                self._set_transaction_phase("checkpoint_saved")
                self.teacher_replay.commit_transaction()
                self.selfplay_replay.commit_transaction()
                self.checkpoints.clear_transaction()
                self._transaction_payload = None
                self._transaction_base_iteration = None
                self._load_history_models()
            except BaseException:
                self.abort_iteration("exception_or_interrupt")
                raise
            print(
                f"[fixed-rl iteration] iteration={self.iteration} "
                f"games={int(generated['games'])} "
                f"moves/s={generated['moves_per_second']:.1f} "
                f"step={self.trainer.global_step} "
                f"loss={trained.get('loss', float('nan')):.4f} "
                f"checkpoint={checkpoint}",
                flush=True,
            )
            self._maybe_evaluate_champion()
            self._maybe_crossplay()

    def _rng_state(self) -> dict[str, Any]:
        state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    @staticmethod
    def _evaluation_metadata(result: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value
            for key, value in result.items()
            if key != "games_detail"
        }

    def _maybe_evaluate_champion(self) -> None:
        evaluation = self.config.evaluation
        if (
            not evaluation.enabled
            or self.iteration < self.config.checkpoint.initial_champion_iteration
            or self.iteration % evaluation.interval != 0
        ):
            return
        if not self.checkpoints.champion_path.is_file():
            metadata = self.checkpoints.initialize_champion(
                layout_id=self.layout.layout_id,
                layout_hash=hashlib.sha256(
                    ",".join(str(value) for value in self.layout.setup).encode(
                        "ascii"
                    )
                ).hexdigest(),
                config_hash=self._config_hash(),
            )
            if int(metadata["source_iteration"]) == self.iteration:
                # Iteration 10 establishes the initial champion. Its first
                # independent challenge is iteration 15.
                self._load_history_models()
                return
        final_prefix = (
            self.checkpoints.evaluation_directory
            / f"champion_iteration_{self.iteration:06d}"
        )
        if final_prefix.with_suffix(".json").is_file():
            return

        evaluator = IndependentEvaluator(
            self.config, self.layout, project_root=self.project_root
        )
        try:
            candidate = evaluator.agent(self.checkpoints.latest_path)
            baseline = evaluator.agent(self.checkpoints.champion_path)
            first = evaluator.evaluate(
                candidate,
                baseline,
                games=evaluation.initial_games,
                temperature=evaluation.temperature,
                mode=evaluation.mode,
                base_seed=evaluation.seed,
                output_prefix=final_prefix.with_name(final_prefix.name + "_stage512"),
            )
            if first["score_rate"] < evaluation.reject_score:
                result = first
                decision = "rejected_below_52pct"
                promoted = False
            elif first["score_rate"] > evaluation.direct_promote_score:
                result = first
                decision = "promoted_above_55pct"
                promoted = True
            else:
                additional_games = (
                    evaluation.extended_games - evaluation.initial_games
                )
                second = evaluator.evaluate(
                    candidate,
                    baseline,
                    games=additional_games,
                    temperature=evaluation.temperature,
                    mode=evaluation.mode,
                    base_seed=evaluation.seed,
                    seed_offset_pairs=evaluation.initial_games // 2,
                )
                rows = list(first["games_detail"]) + list(second["games_detail"])
                result = evaluator.summarize_rows(
                    rows,
                    agent_a=candidate.label,
                    agent_b=baseline.label,
                    temperature=evaluation.temperature,
                    mode=evaluation.mode,
                    base_seed=evaluation.seed,
                    elapsed_seconds=(
                        float(first["elapsed_seconds"])
                        + float(second["elapsed_seconds"])
                    ),
                )
                promoted = result["ci95_lower"] > 0.5
                decision = (
                    "promoted_extended_ci_lower_above_50pct"
                    if promoted
                    else "rejected_extended_ci_not_above_50pct"
                )
            result.update(
                {
                    "candidate_checkpoint": str(self.checkpoints.latest_path),
                    "champion_checkpoint_before": str(
                        self.checkpoints.champion_path
                    ),
                    "candidate_iteration": self.iteration,
                    "promoted": promoted,
                    "decision": decision,
                    "config_hash": self._config_hash(),
                }
            )
            evaluator.write_outputs(final_prefix, result)
            metadata = self._evaluation_metadata(result)
            if promoted:
                self.checkpoints.promote(
                    self.checkpoints.latest_path,
                    self.iteration,
                    metadata,
                )
                self._load_history_models()
            else:
                self.checkpoints.record_evaluation(metadata)
            self.metrics.write("champion_evaluation", **metadata)
            print(
                "[fixed-rl champion] "
                f"iteration={self.iteration} "
                f"W/D/L={result['wins']}/{result['draws']}/{result['losses']} "
                f"score={result['score_rate']:.4f} "
                f"CI95=[{result['ci95_lower']:.4f},{result['ci95_upper']:.4f}] "
                f"promoted={promoted} decision={decision}",
                flush=True,
            )
        finally:
            evaluator.close()

    def _maybe_crossplay(self) -> None:
        evaluation = self.config.evaluation
        if (
            not evaluation.enabled
            or evaluation.crossplay_interval < 1
            or self.iteration % evaluation.crossplay_interval != 0
        ):
            return
        target = (
            self.checkpoints.crossplay_directory
            / f"crossplay_iteration_{self.iteration:06d}.json"
        )
        if target.is_file():
            return
        evaluator = IndependentEvaluator(
            self.config, self.layout, project_root=self.project_root
        )
        try:
            agents = [
                ("latest", evaluator.agent(self.checkpoints.latest_path)),
                ("champion", evaluator.agent(self.checkpoints.champion_path)),
                ("084", evaluator.agent("084")),
                ("082", evaluator.agent("082")),
                ("random", evaluator.agent("random")),
                ("aggressive", evaluator.agent("aggressive")),
            ]
            comparisons = []
            pair_number = 0
            for left_index in range(len(agents)):
                for right_index in range(left_index + 1, len(agents)):
                    left_name, left = agents[left_index]
                    right_name, right = agents[right_index]
                    prefix = (
                        self.checkpoints.crossplay_directory
                        / f"iteration_{self.iteration:06d}_{left_name}_vs_{right_name}"
                    )
                    result = evaluator.evaluate(
                        left,
                        right,
                        games=evaluation.crossplay_games,
                        temperature=evaluation.temperature,
                        mode=evaluation.mode,
                        base_seed=evaluation.seed + pair_number * 100003,
                        output_prefix=prefix,
                    )
                    comparisons.append(self._evaluation_metadata(result))
                    pair_number += 1
            payload = {
                "schema": "junqi-fixed-layout-crossplay-v1",
                "iteration": self.iteration,
                "games_per_comparison": evaluation.crossplay_games,
                "temperature": evaluation.temperature,
                "mode": evaluation.mode,
                "comparisons": comparisons,
            }
            temporary = target.with_suffix(".json.tmp")
            temporary.write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(temporary, target)
            self.metrics.write(
                "crossplay_completed",
                iteration=self.iteration,
                comparisons=len(comparisons),
                games_per_comparison=evaluation.crossplay_games,
                output=str(target),
            )
        finally:
            evaluator.close()

    def save_checkpoint(
        self, *, completed_games: int, register: bool
    ) -> Path:
        payload = {
            "schema": self.config.schema,
            "checkpoint_kind": "latest",
            "phase": "completed",
            "iteration_completed": True,
            "completed_games_in_iteration": int(completed_games),
            "config": self.config.to_dict(),
            "trainer": self.trainer.state_dict(),
            "iteration": self.iteration,
            "games_generated": self.games_generated,
            "rng": self._rng_state(),
            "teacher_replay": self.teacher_replay.state_dict(),
            "selfplay_replay": self.selfplay_replay.state_dict(),
        }
        target = self.checkpoints.save_latest(payload)
        if register:
            paths = self.checkpoints.register_completed(self.iteration)
            self.metrics.write(
                "checkpoint_completed",
                iteration=self.iteration,
                global_step=self.trainer.global_step,
                phase="completed",
                completed_games=completed_games,
                latest=str(target),
                archive=paths.get("archive"),
                recent=paths.get("recent"),
                teacher_replay=self.teacher_replay.transaction_status(),
                selfplay_replay=self.selfplay_replay.transaction_status(),
            )
        return target

    def load_checkpoint(
        self, path: str | Path, *, restore_replay: bool = True
    ) -> None:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != self.config.schema:
            raise RuntimeError("checkpoint schema differs from active fixed-layout RL")
        if self._training_config(payload.get("config", {})) != self._training_config(
            self.config.to_dict()
        ):
            raise RuntimeError(
                "checkpoint training configuration differs from the active run"
            )
        if restore_replay:
            self.teacher_replay.load_state_dict(payload["teacher_replay"])
            self.selfplay_replay.load_state_dict(payload["selfplay_replay"])
        self.trainer.load_state_dict(payload["trainer"])
        self.iteration = int(payload.get("iteration", 0))
        self.games_generated = int(payload.get("games_generated", 0))
        rng = payload.get("rng", {})
        if "python" in rng:
            random.setstate(rng["python"])
        if "numpy" in rng:
            np.random.set_state(rng["numpy"])
        if "torch" in rng:
            torch.set_rng_state(rng["torch"])
        if torch.cuda.is_available() and "cuda" in rng:
            torch.cuda.set_rng_state_all(rng["cuda"])
        print(
            f"[fixed-rl] resumed iteration={self.iteration} "
            f"step={self.trainer.global_step} from {path}",
            flush=True,
        )

    def _load_history_models(self) -> None:
        self._history_models.clear()
        champion = self.checkpoints.champion_metadata()
        champion_iteration = (
            int(champion["source_iteration"]) if champion is not None else -1
        )
        by_iteration: dict[int, tuple[Path, str]] = {}
        for path in self.checkpoints.archive_directory.glob("iteration_*.pt"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            iteration = int(payload["iteration"])
            if iteration < self.iteration and iteration != champion_iteration:
                by_iteration[iteration] = (path, "recent_checkpoint")
        for path in self.checkpoints.recent_directory.glob("iteration_*.pt"):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            iteration = int(payload["iteration"])
            if iteration < self.iteration and iteration != champion_iteration:
                by_iteration[iteration] = (path, "recent_checkpoint")
        paths = [
            (iteration, path, category)
            for iteration, (path, category) in by_iteration.items()
        ]
        paths.sort(key=lambda row: row[0])
        keep = max(0, self.config.selfplay.historical_keep - 1)
        selected = paths[-keep:] if keep else []
        if self.checkpoints.champion_path.is_file():
            selected.append(
                (
                    champion_iteration,
                    self.checkpoints.champion_path,
                    "champion",
                )
            )
        for _, path, category in selected:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = PolicyValueNet(self.config.model)
            model.load_state_dict(payload["trainer"]["model"])
            model.to(self.trainer.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._history_models.append((model, category))
