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
import torch.nn.functional as F

from junqi_az.layout_policy import mirror_layout
from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.evaluator import (
    IndependentEvaluator,
    repetition_cycle_details,
    terminal_category,
)
from junqi_fixed_rl.layout import FixedLayout, FixedLayoutPairBackend, load_fixed_layout
from junqi_fixed_rl.metrics import MetricsWriter
from junqi_fixed_rl.config import FixedRLConfig
from junqi_fixed_rl.teachers import TeacherEnsemble, aggressive_policy, random_policy

from .checkpoints import (
    AdvantageCheckpointManager,
    restore_rng,
    rng_state,
)
from .config import AdvantageRLConfig
from .returns import compute_player_lambda_returns
from .rollout import (
    RolloutStep,
    atomic_save_rollout,
    load_rollout,
    remove_incomplete_rollouts,
)
from .trainer import AdvantageTrainer, load_policy_warm_start


@dataclass
class Episode:
    state: Any
    game_id: int
    learner_side: int
    opponent: str
    opponent_type: str
    layout: FixedLayout
    history_model: PolicyValueNet | None = None
    steps: list[RolloutStep] = field(default_factory=list)
    position_hashes: list[int] = field(default_factory=list)
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    moves: int = 0


class AdvantageRLPipeline:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.project_root = self.config_path.parent.parent
        self.config = AdvantageRLConfig.load(self.config_path)
        self._seed_all(self.config.seed)
        self.rng = np.random.default_rng(self.config.seed ^ 0xA7A2)
        self.run = self._resolve(self.config.run_directory)
        self.run.mkdir(parents=True, exist_ok=True)
        self.metrics = MetricsWriter(self.run / "metrics.jsonl")
        self.rollout_directory = self.run / "rollouts" / "value"
        self.teacher_path = self.run / "replay_teacher" / "bootstrap.pt"
        self.rollout_directory.mkdir(parents=True, exist_ok=True)
        self.teacher_path.parent.mkdir(parents=True, exist_ok=True)
        removed = remove_incomplete_rollouts(self.rollout_directory)
        self.layouts = {
            layout_id: load_fixed_layout(
                self.config.layouts.set_path,
                layout_id,
                project_root=self.project_root,
            )
            for layout_id in (
                self.config.layouts.training_ids
                + ([self.config.layouts.sealed_id] if self.config.layouts.sealed_id else [])
            )
        }
        self.training_layouts = [
            self.layouts[layout_id] for layout_id in self.config.layouts.training_ids
        ]
        self.teacher = TeacherEnsemble(
            self.config.teacher,
            project_root=self.project_root,
            seed=self.config.seed + 17,
        )
        self.model = PolicyValueNet(self.config.model)
        self.trainer = AdvantageTrainer(self.config, self.model)
        self.checkpoints = AdvantageCheckpointManager(
            self.run,
            self.config.checkpoints.archive_interval,
            self.config.checkpoints.recent_keep,
        )
        self.iteration = 0
        self.games_generated = 0
        self.phase = "initializing"
        self._history_models: list[PolicyValueNet] = []
        self.last_pi_old_hash = ""
        if self.checkpoints.latest.exists():
            self._resume()
        else:
            source = self._resolve(self.config.initialization_checkpoint)
            warm = load_policy_warm_start(self.model, str(source))
            self.trainer.reset_copies()
            self.metrics.write("warm_start", **warm)
            self._write_manifest(warm, removed)

    def _resolve(self, path: str | Path) -> Path:
        value = Path(path)
        return value if value.is_absolute() else (self.project_root / value).resolve()

    @staticmethod
    def _seed_all(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _write_manifest(self, warm: dict[str, Any], removed: list[str]) -> None:
        payload = {
            "schema": self.config.schema,
            "lineage": self.config.lineage,
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "initialization": warm,
            "isolation": {
                "v1_replay_loaded": False,
                "v1_optimizer_loaded": False,
                "optimizer_reset": True,
                "scheduler_reset": True,
                "ema_reset": True,
                "anchor_reset": True,
                "rng_seed": self.config.seed,
            },
            "training_layouts": self.config.layouts.training_ids,
            "sealed_layout": self.config.layouts.sealed_id or None,
            "sealed_excluded_from_training": bool(self.config.layouts.sealed_id),
            "network_parameters": sum(p.numel() for p in self.model.parameters()),
            "incomplete_rollouts_removed": removed,
        }
        (self.run / "manifest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    @property
    def temperature(self) -> float:
        c = self.config.selfplay
        progress = min(1.0, self.iteration / max(1, c.temperature_decay_iterations))
        return max(
            c.minimum_temperature,
            c.temperature + (c.minimum_temperature - c.temperature) * progress,
        )

    def _layout_for_game(self, game_number: int, teacher_only: bool) -> FixedLayout:
        if teacher_only and self.config.lineage == "v2b":
            # Exact stratification guarantees coverage of all four layouts and
            # both orientations in a fresh multi-layout teacher bootstrap.
            layout = self.training_layouts[game_number % len(self.training_layouts)]
            mirrored = (
                (game_number // len(self.training_layouts)) % 2 == 1
            )
        else:
            layout = self.training_layouts[
                random.Random(self.config.seed + game_number * 31337).randrange(
                    len(self.training_layouts)
                )
            ]
            mirrored = (
                random.Random(
                    self.config.seed + game_number * 104729 + 0x4C41594F
                ).random()
                < self.config.layouts.mirror_probability
            )
        if not mirrored:
            return layout
        return FixedLayout(
            layout_id=layout.layout_id + "__horizontal_mirror",
            name_zh=layout.name_zh + "（水平镜像）",
            setup=mirror_layout(layout.setup),
            source_sha256=layout.source_sha256,
        )

    def _new_episode(self, game_number: int, teacher_only: bool) -> Episode:
        layout = self._layout_for_game(game_number, teacher_only)
        state = FixedLayoutPairBackend(layout, layout).new_state(
            self.config.seed + game_number * 104729
        )
        if teacher_only:
            opponent = "wide_teacher_ensemble"
            learner_side = 1
        else:
            keys = list(self.config.selfplay.opponent_weights)
            opponent = random.choices(
                keys,
                weights=[self.config.selfplay.opponent_weights[key] for key in keys],
                k=1,
            )[0]
            if opponent == "historical" and not self._history_models:
                opponent = "current"
            learner_side = random.choice((1, -1))
        labels = {
            "current": "current/latest",
            "historical": "recent_checkpoint",
            "heuristic_current": "wide_teacher_ensemble_current",
            "heuristic_historical": "wide_teacher_ensemble_historical",
            "random": "random_exploiter",
            "aggressive": "aggressive_exploiter",
            "wide_teacher_ensemble": "wide_teacher_ensemble",
        }
        history = (
            random.choice(self._history_models)
            if opponent == "historical" and self._history_models
            else None
        )
        episode = Episode(
            state=state,
            game_id=game_number + 1,
            learner_side=learner_side,
            opponent=opponent,
            opponent_type=labels[opponent],
            layout=layout,
            history_model=history,
        )
        episode.position_hashes.append(int(state.native_state.position_key()))
        return episode

    def _actor(self, episode: Episode, teacher_only: bool) -> str:
        if teacher_only:
            return "wide_teacher_ensemble"
        if episode.opponent == "current":
            return "pi_old"
        if episode.state.current_player() == episode.learner_side:
            return "pi_old"
        return episode.opponent

    def _old_inference(
        self, episodes: list[Episode], model: PolicyValueNet
    ) -> list[dict[str, Any]]:
        observations = np.stack([episode.state.feature_planes() for episode in episodes])
        masks = np.zeros((len(episodes), 3600), dtype=np.bool_)
        repetitions = np.zeros_like(masks, dtype=np.uint8)
        legals = []
        for index, episode in enumerate(episodes):
            legal = np.asarray(episode.state.legal_actions(), np.int32)
            context = np.asarray(episode.state.repetition_contexts(), np.uint8)
            legals.append(legal)
            masks[index, legal] = True
            repetitions[index, legal] = context
        device = self.trainer.device
        model.eval()
        with torch.inference_mode():
            logits, values, _, wdl_logits, _ = model.forward_heads(
                torch.from_numpy(observations).to(device),
                torch.from_numpy(masks).to(device),
                torch.from_numpy(repetitions).to(device),
            )
            wdl = F.softmax(wdl_logits.float(), dim=-1)
        logits_np = logits.float().cpu().numpy()
        values_np = values.float().cpu().numpy()
        wdl_np = wdl.cpu().numpy()
        return [
            {
                "observation": observations[index].astype(np.float16),
                "legal": legal,
                "context": repetitions[index, legal],
                "raw_logits": logits_np[index, legal].astype(np.float32),
                "old_value": float(values_np[index]),
                "old_wdl": wdl_np[index].astype(np.float32),
            }
            for index, legal in enumerate(legals)
        ]

    def _sample_network(
        self, inference: dict[str, Any]
    ) -> tuple[int, float]:
        logits = inference["raw_logits"].astype(np.float64)
        probabilities = np.exp(
            np.clip(
                logits / max(self.temperature, 1e-6)
                - np.max(logits / max(self.temperature, 1e-6)),
                -60,
                0,
            )
        )
        probabilities /= probabilities.sum()
        mix = self.config.selfplay.uniform_mix
        if mix:
            probabilities = (1.0 - mix) * probabilities + mix / len(probabilities)
        position = random.choices(
            range(len(probabilities)), weights=probabilities.tolist(), k=1
        )[0]
        return (
            int(inference["legal"][position]),
            float(np.log(max(probabilities[position], 1e-30))),
        )

    def _historical_actions(
        self, rows: list[tuple[Episode, dict[str, Any]]]
    ) -> list[int]:
        if not rows:
            return []
        # Batch all games assigned to the same archive snapshot.  A per-state
        # GPU call reduced rollout throughput by about half once iteration 5
        # became the first historical archive.
        result = [0] * len(rows)
        groups: dict[int, tuple[PolicyValueNet, list[int]]] = {}
        for index, (episode, inference) in enumerate(rows):
            model = episode.history_model
            if model is None:
                result[index] = self._sample_network(inference)[0]
                continue
            key = id(model)
            if key not in groups:
                groups[key] = (model, [])
            groups[key][1].append(index)
        for model, indices in groups.values():
            historical = self._old_inference(
                [rows[index][0] for index in indices], model
            )
            for index, inference in zip(indices, historical):
                result[index] = self._sample_network(inference)[0]
        return result

    def generate(self, games: int, *, teacher_only: bool) -> tuple[list[RolloutStep], dict[str, Any]]:
        started = time.perf_counter()
        old = self.trainer.freeze_collection_policy()
        self.last_pi_old_hash = hashlib.sha256(
            b"".join(
                tensor.detach().cpu().numpy().tobytes()
                for tensor in old.state_dict().values()
            )
        ).hexdigest()
        first_game = self.games_generated
        active = [
            self._new_episode(first_game + index, teacher_only)
            for index in range(min(games, self.config.selfplay.parallel_games))
        ]
        next_game = first_game + len(active)
        completed: list[Episode] = []
        reason_counts: dict[str, int] = {}
        opponent_rows: dict[str, dict[str, float]] = {}
        cycle_signatures: dict[str, int] = {}
        report_every = max(1, min(32, games))
        while active:
            inferred = self._old_inference(active, old)
            actors = [self._actor(episode, teacher_only) for episode in active]
            decisions: dict[int, tuple[int, float, np.ndarray | None]] = {}
            teacher_indices = [
                index for index, actor in enumerate(actors)
                if actor in {"wide_teacher_ensemble", "heuristic_current", "heuristic_historical"}
            ]
            if teacher_indices:
                teacher_rows = self.teacher.policy_batch(
                    [active[index].state for index in teacher_indices],
                    [actors[index] == "heuristic_historical" for index in teacher_indices],
                )
                for index, (legal, probabilities) in zip(teacher_indices, teacher_rows):
                    position = random.choices(
                        range(len(legal)), weights=probabilities.tolist(), k=1
                    )[0]
                    decisions[index] = (
                        int(legal[position]),
                        float(np.log(max(float(probabilities[position]), 1e-30))),
                        probabilities,
                    )
            historical_indices = [
                index for index, actor in enumerate(actors) if actor == "historical"
            ]
            historical_actions = self._historical_actions(
                [(active[index], inferred[index]) for index in historical_indices]
            )
            for index, action in zip(historical_indices, historical_actions):
                decisions[index] = (action, float("nan"), None)
            for index, actor in enumerate(actors):
                if index in decisions:
                    continue
                if actor == "random":
                    action, _, probabilities = random_policy(active[index].state)
                    decisions[index] = (action, float("nan"), probabilities)
                elif actor == "aggressive":
                    action, _, probabilities = aggressive_policy(active[index].state)
                    decisions[index] = (action, float("nan"), probabilities)
                elif actor == "pi_old":
                    action, log_probability = self._sample_network(inferred[index])
                    decisions[index] = (action, log_probability, None)
                else:
                    raise RuntimeError(f"unknown actor {actor}")
            finished = []
            for index, episode in enumerate(active):
                actor = actors[index]
                action, behavior_log_prob, teacher_probabilities = decisions[index]
                row = inferred[index]
                side = int(episode.state.current_player())
                source, target = divmod(action, 60)
                moved_piece = int(episode.state.native_state.board()[source])
                episode.steps.append(
                    RolloutStep(
                        observation=row["observation"],
                        legal_actions=row["legal"],
                        repetition_contexts=row["context"],
                        selected_action=action,
                        old_raw_logits=row["raw_logits"],
                        behavior_log_prob=behavior_log_prob,
                        sampling_temperature=self.temperature,
                        uniform_mix=float(self.config.selfplay.uniform_mix),
                        acting_side=side,
                        opponent_type=episode.opponent_type,
                        actor_type=actor,
                        game_id=episode.game_id,
                        ply=episode.moves,
                        old_value=row["old_value"],
                        old_wdl=row["old_wdl"],
                        policy_eligible=(actor == "pi_old" and not teacher_only),
                        layout_id=episode.layout.layout_id,
                        moved_piece=moved_piece,
                        teacher_probabilities=(
                            np.asarray(teacher_probabilities, np.float32)
                            if teacher_probabilities is not None and teacher_only
                            else None
                        ),
                    )
                )
                episode.action_trace.append(
                    {
                        "ply": episode.moves,
                        "action": action,
                        "from": source,
                        "to": target,
                        "moved_piece": moved_piece,
                        "mover_side": "red" if side == 1 else "black",
                        "actor": actor,
                    }
                )
                episode.state.apply(action)
                episode.moves += 1
                episode.position_hashes.append(
                    int(episode.state.native_state.position_key())
                )
                cutoff = episode.moves >= self.config.selfplay.max_moves
                if episode.state.is_terminal() or cutoff:
                    reason = terminal_category(episode.state, cutoff)
                    terminal_red = (
                        float(episode.state.terminal_value(1))
                        if episode.state.is_terminal()
                        else 0.0
                    )
                    sides = np.asarray([step.acting_side for step in episode.steps])
                    values = np.asarray([step.old_value for step in episode.steps])
                    old_wdl = np.stack([step.old_wdl for step in episode.steps])
                    returns, advantages, xi = compute_player_lambda_returns(
                        sides,
                        values,
                        old_wdl,
                        terminal_red,
                        lambda_advantage=self.config.advantage.lambda_advantage,
                        lambda_outcome=self.config.advantage.lambda_outcome,
                    )
                    for step, scalar, advantage, vector in zip(
                        episode.steps, returns, advantages, xi
                    ):
                        step.outcome = terminal_red * step.acting_side
                        step.terminal_reason = reason
                        step.lambda_return = float(scalar)
                        step.advantage = float(advantage)
                        step.wdl_target = vector
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    stats = opponent_rows.setdefault(
                        episode.opponent_type,
                        {"games": 0, "wins": 0, "draws": 0, "losses": 0, "moves": 0, "repetitions": 0},
                    )
                    learner_outcome = terminal_red * episode.learner_side
                    stats["games"] += 1
                    stats["moves"] += episode.moves
                    stats["wins" if learner_outcome > 0 else "losses" if learner_outcome < 0 else "draws"] += 1
                    if reason == "repetition_draw":
                        stats["repetitions"] += 1
                        cycle = repetition_cycle_details(
                            episode.position_hashes,
                            episode.action_trace,
                            [step.old_value for step in episode.steps],
                            terminal_value=0.0,
                            opponent_type=episode.opponent_type,
                        )
                        signature = ">".join(cycle.get("position_hash_sequence", []))
                        cycle_signatures[signature] = cycle_signatures.get(signature, 0) + 1
                        self.metrics.write(
                            "repetition_cycle",
                            iteration=self.iteration + (0 if teacher_only else 1),
                            game_id=episode.game_id,
                            layout_id=episode.layout.layout_id,
                            **cycle,
                        )
                    completed.append(episode)
                    finished.append(index)
            for index in reversed(finished):
                active.pop(index)
            while len(active) < self.config.selfplay.parallel_games and next_game < first_game + games:
                active.append(self._new_episode(next_game, teacher_only))
                next_game += 1
            if len(completed) and len(completed) % report_every == 0:
                moves = sum(episode.moves for episode in completed)
                elapsed = time.perf_counter() - started
                print(
                    f"[adv-{self.config.lineage} {'teacher' if teacher_only else 'selfplay'}] "
                    f"{len(completed)}/{games} moves={moves} moves/s={moves/max(elapsed,1e-9):.1f}",
                    flush=True,
                )
        if len(completed) != games:
            raise RuntimeError("only complete games may be committed")
        all_steps = [step for episode in completed for step in episode.steps]
        elapsed = time.perf_counter() - started
        total_moves = len(all_steps)
        repetition_count = reason_counts.get("repetition_draw", 0)
        dominant_count = max(cycle_signatures.values(), default=0)
        detailed = {}
        for opponent, stats in opponent_rows.items():
            count = max(1, int(stats["games"]))
            detailed[opponent] = {
                **stats,
                "score_rate": (stats["wins"] + 0.5 * stats["draws"]) / count,
                "average_ply": stats["moves"] / count,
                "repetition_rate": stats["repetitions"] / count,
                "moves_per_second": stats["moves"] / max(elapsed, 1e-9),
            }
        summary = {
            "games": games,
            "samples": total_moves,
            "elapsed_seconds": elapsed,
            "moves_per_second": total_moves / max(elapsed, 1e-9),
            "temperature": self.temperature,
            "opponents": detailed,
            "terminal_reasons": {
                key: {"count": value, "fraction": value / games}
                for key, value in reason_counts.items()
            },
            "repetition_rate": repetition_count / games,
            "max_single_cycle_fraction": dominant_count / max(1, repetition_count),
            "dominant_cycle_count": dominant_count,
        }
        return all_steps, summary

    def _bootstrap(self) -> list[RolloutStep]:
        if self.teacher_path.exists():
            rows, metadata = load_rollout(self.teacher_path)
            return rows
        print(
            f"[adv-{self.config.lineage}] building independent teacher bootstrap "
            f"games={self.config.teacher.bootstrap_games}",
            flush=True,
        )
        rows, summary = self.generate(
            self.config.teacher.bootstrap_games, teacher_only=True
        )
        # Bootstrap value uses actual terminal WDL, while later value updates
        # use vector lambda-return.
        for row in rows:
            one_hot = np.zeros(3, np.float32)
            one_hot[2 if row.outcome > 0.5 else 0 if row.outcome < -0.5 else 1] = 1
            row.wdl_target = one_hot
        atomic_save_rollout(
            self.teacher_path,
            rows,
            metadata={
                "complete_games": self.config.teacher.bootstrap_games,
                "lineage": self.config.lineage,
                "layout_ids": sorted({row.layout_id for row in rows}),
                "teacher_kind": "wide_teacher_ensemble",
                "summary": summary,
            },
        )
        metrics = self.trainer.train_teacher_bootstrap(
            rows,
            steps=self.config.teacher.bootstrap_train_steps,
            rng=self.rng,
        )
        self.metrics.write("teacher_bootstrap", **summary, **metrics)
        self.phase = "bootstrap_completed"
        self._save(completed=True)
        self.checkpoints.initialize_champion()
        return rows

    def _checkpoint_payload(self, completed: bool) -> dict[str, Any]:
        return {
            "schema": "junqi-advantage-checkpoint-v2",
            "lineage": self.config.lineage,
            "phase": "completed" if completed else self.phase,
            "iteration_completed": bool(completed),
            "iteration": self.iteration,
            "games_generated": self.games_generated,
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "trainer": self.trainer.state_dict(),
            "evaluation_model": self.trainer.ema.state_dict(),
            "rng": rng_state(),
            "numpy_generator_state": copy.deepcopy(self.rng.bit_generator.state),
            "replay": {
                "teacher": str(self.teacher_path),
                "fresh_value_rollouts": sorted(
                    str(path) for path in self.rollout_directory.glob("iteration_*.pt")
                ),
                "policy_replay": "none: current iteration only",
            },
        }

    def _save(self, completed: bool) -> None:
        paths = self.checkpoints.save(
            self._checkpoint_payload(completed), self.iteration
        )
        self.metrics.write("checkpoint", iteration=self.iteration, **paths)

    def _commit_iteration_summary(
        self,
        *,
        selfplay: dict[str, Any],
        value: dict[str, Any],
        policy: dict[str, Any],
    ) -> None:
        directory = self.run / "rollouts" / "summaries"
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / f"iteration_{self.iteration:06d}.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "schema": "junqi-advantage-iteration-summary-v2",
                    "iteration": self.iteration,
                    "selfplay": selfplay,
                    "value": value,
                    "policy": policy,
                    "raw_rollout_retention": "latest_complete_iteration_only",
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _prune_old_raw_rollouts(self) -> list[str]:
        """Keep only the newest auditable fresh rollout.

        Policy data is intentionally one-iteration-only.  Permanent compact
        summaries preserve all diagnostic distributions without accumulating
        ~1.2 GiB of stale observations per iteration.
        """
        current = f"iteration_{self.iteration:06d}.pt"
        removed = []
        for path in self.rollout_directory.glob("iteration_*.pt"):
            if path.name != current:
                path.unlink()
                removed.append(str(path))
        return removed

    def _resume(self) -> None:
        payload = torch.load(
            self.checkpoints.latest, map_location="cpu", weights_only=False
        )
        if payload.get("schema") != "junqi-advantage-checkpoint-v2":
            raise ValueError("run directory contains a non-v2 checkpoint")
        if payload.get("config_hash") != self.config.hash():
            raise RuntimeError("v2 configuration hash changed; start a new lineage")
        if not payload.get("iteration_completed") or payload.get("phase") != "completed":
            raise RuntimeError("latest v2 checkpoint is not transaction-complete")
        self.trainer.load_state_dict(payload["trainer"])
        self.iteration = int(payload["iteration"])
        self.games_generated = int(payload["games_generated"])
        restore_rng(payload["rng"])
        self.rng.bit_generator.state = payload["numpy_generator_state"]
        self.phase = "completed"
        self._load_history()

    def _load_history(self) -> None:
        self._history_models.clear()
        candidates = sorted(self.checkpoints.archive.glob("iteration_*.pt"))[-8:]
        for path in candidates:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            model = PolicyValueNet(self.config.model)
            model.load_state_dict(payload["evaluation_model"])
            model.to(self.trainer.device).eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)
            self._history_models.append(model)

    def _evaluation_config(self) -> FixedRLConfig:
        config = FixedRLConfig()
        config.device = self.config.device
        config.model = copy.deepcopy(self.config.model)
        config.teacher = copy.deepcopy(self.config.teacher)
        config.selfplay.max_moves = self.config.selfplay.max_moves
        config.evaluation.temperature = self.config.evaluation.temperature
        config.evaluation.seed = self.config.evaluation.seed
        return config

    @staticmethod
    def _mirrored(layout: FixedLayout) -> FixedLayout:
        return FixedLayout(
            layout_id=layout.layout_id + "__horizontal_mirror",
            name_zh=layout.name_zh + "（水平镜像）",
            setup=mirror_layout(layout.setup),
            source_sha256=layout.source_sha256,
        )

    def _evaluate_match(
        self,
        layout: FixedLayout,
        agent_a: str,
        agent_b: str,
        label: str,
        seed_offset: int,
    ) -> dict[str, Any]:
        evaluator = IndependentEvaluator(
            self._evaluation_config(),
            layout,
            black_layout=layout,
            project_root=self.project_root,
        )
        output = self.run / "evaluations" / f"iteration_{self.iteration:06d}_{label}"
        try:
            return evaluator.evaluate(
                evaluator.agent(agent_a),
                evaluator.agent(agent_b),
                games=self.config.evaluation.games_per_match,
                temperature=self.config.evaluation.temperature,
                mode="sample",
                base_seed=self.config.evaluation.seed + seed_offset,
                output_prefix=output,
            )
        finally:
            evaluator.close()

    def evaluate_suite(self) -> dict[str, Any]:
        """Read-only EMA evaluation; the sealed layout is never returned to training."""
        latest = str(self.checkpoints.latest)
        champion = str(self.checkpoints.champion)
        previous = self.checkpoints.recent / f"iteration_{self.iteration - 1:06d}.pt"
        base = self.training_layouts[0]
        specifications: list[tuple[str, FixedLayout, str, str]] = [
            ("selfplay", base, latest, latest),
            (
                "recent_checkpoint",
                base,
                latest,
                str(previous if previous.exists() else self.checkpoints.champion),
            ),
            ("champion", base, latest, champion),
            ("exact_084", base, latest, "084"),
            ("exact_082", base, latest, "082"),
            ("random", base, latest, "random"),
            ("aggressive", base, latest, "aggressive"),
            ("training_layout_mirror", self._mirrored(base), latest, "084"),
        ]
        for index, layout in enumerate(self.training_layouts[1:], start=1):
            specifications.append(
                (f"training_layout_{index}_{layout.layout_id}", layout, latest, "084")
            )
        if self.config.layouts.sealed_id:
            sealed = self.layouts[self.config.layouts.sealed_id]
            specifications.extend(
                [
                    ("sealed_layout", sealed, latest, "084"),
                    ("sealed_layout_mirror", self._mirrored(sealed), latest, "084"),
                ]
            )
        results = {}
        for index, (label, layout, a, b) in enumerate(specifications):
            result = self._evaluate_match(layout, a, b, label, index * 10000)
            results[label] = {
                key: value for key, value in result.items() if key != "games_detail"
            }
            print(
                f"[adv-{self.config.lineage} eval] {label} "
                f"W/D/L={result['wins']}/{result['draws']}/{result['losses']} "
                f"score={result['score_rate']:.3f} "
                f"repeat={result['repetition_fraction']:.1%}",
                flush=True,
            )
        selfplay = results["selfplay"]
        collapsed = (
            selfplay["repetition_fraction"] >= 0.80
            and selfplay.get("max_single_cycle_fraction", 0.0) >= 0.50
            and results["exact_084"]["score_rate"] <= 0.02
        )
        summary = {
            "iteration": self.iteration,
            "ema_evaluation": True,
            "read_only": True,
            "results": results,
            "repetition_collapse": collapsed,
        }
        target = self.run / "evaluations" / f"iteration_{self.iteration:06d}_suite.json"
        target.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self.metrics.write("evaluation_suite", **summary)
        champion_result = results["champion"]
        if (
            champion_result["score_rate"] >= 0.52
            and champion_result["ci95_lower"] > 0.50
        ):
            self.checkpoints.promote_champion(self.iteration, champion_result)
            self.metrics.write(
                "champion_promoted",
                iteration=self.iteration,
                evaluation=champion_result,
            )
        return summary

    def train(self, iterations: int) -> None:
        teacher_rows = self._bootstrap()
        for _ in range(int(iterations)):
            target = self.iteration + 1
            self.phase = "selfplay"
            rows, selfplay = self.generate(
                self.config.selfplay.games_per_iteration, teacher_only=False
            )
            rollout_path = self.rollout_directory / f"iteration_{target:06d}.pt"
            atomic_save_rollout(
                rollout_path,
                rows,
                metadata={
                    "iteration": target,
                    "complete_games": self.config.selfplay.games_per_iteration,
                    "pi_old_hash": self.last_pi_old_hash,
                    "policy_fresh_only": True,
                    "summary": selfplay,
                },
            )
            self.phase = "train"
            value_metrics = self.trainer.train_value(rows, self.rng)
            policy_metrics = self.trainer.train_policy(
                rows, teacher_rows, iteration=target, rng=self.rng
            )
            if not all(
                np.isfinite(float(value))
                for value in {**value_metrics, **policy_metrics}.values()
                if isinstance(value, (int, float))
            ):
                raise FloatingPointError("non-finite v2 training metric")
            self.iteration = target
            self.games_generated += self.config.selfplay.games_per_iteration
            self.phase = "completed"
            self.metrics.write(
                "iteration",
                iteration=self.iteration,
                selfplay=selfplay,
                value=value_metrics,
                policy=policy_metrics,
                global_step=self.trainer.global_step,
            )
            self._commit_iteration_summary(
                selfplay=selfplay,
                value=value_metrics,
                policy=policy_metrics,
            )
            removed_rollouts = self._prune_old_raw_rollouts()
            if removed_rollouts:
                self.metrics.write(
                    "fresh_rollout_retention",
                    iteration=self.iteration,
                    removed=removed_rollouts,
                    policy="latest_complete_iteration_only",
                )
            self._save(completed=True)
            self._load_history()
            print(
                f"[adv-{self.config.lineage}] iteration={self.iteration} "
                f"repeat={selfplay['repetition_rate']:.1%} "
                f"cycle-max={selfplay['max_single_cycle_fraction']:.1%} "
                f"filter={policy_metrics.get('filtered_sample_ratio',0):.1%} "
                f"KL={policy_metrics.get('old_policy_reverse_kl',0):.4f} "
                f"entropy={policy_metrics.get('policy_entropy',0):.3f}",
                flush=True,
            )
            if (
                self.config.evaluation.enabled
                and self.iteration % self.config.evaluation.interval == 0
            ):
                self.evaluate_suite()

    def close(self) -> None:
        self.teacher.close()
