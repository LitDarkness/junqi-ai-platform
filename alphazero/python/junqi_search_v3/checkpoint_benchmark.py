from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .dataset import atomic_json
from .evaluation import SearchEvaluator
from .oracle import OracleHistory
from .strength import SearchV3Strength, _read_jsonl, _sha256


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------

_ITERATION_RE = re.compile(r"(?:iteration|iter|champion)[_-]?0*(\d+)", re.IGNORECASE)


def _safe_name(value: str) -> str:
    value = re.sub(r"[^0-9A-Za-z_.-]+", "_", value.strip())
    return value.strip("._") or "unnamed"


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def _model_state_dict(payload: dict[str, Any]) -> dict[str, torch.Tensor]:
    state_dict = payload.get("model")
    if state_dict is None:
        state_dict = payload.get("state_dict")
    if state_dict is None:
        raise RuntimeError(
            "checkpoint contains neither payload['model'] nor payload['state_dict']"
        )
    return state_dict


def _state_dict_hash(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        digest.update(name.encode("utf-8"))
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _extract_iteration(path: Path, payload: dict[str, Any]) -> int | None:
    for key in ("iteration", "incumbent_iteration", "champion_iteration"):
        value = payload.get(key)
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    for source in (path.stem, path.name, str(path.parent.name)):
        match = _ITERATION_RE.search(source)
        if match:
            return int(match.group(1))
    return None


def _bootstrap(values: np.ndarray, seed: int, samples: int = 5000) -> list[float]:
    if not len(values):
        return [0.0, 0.0]
    rng = np.random.default_rng(seed)
    sampled = values[
        rng.integers(0, len(values), size=(samples, len(values)))
    ].mean(axis=1)
    return list(map(float, np.quantile(sampled, [0.025, 0.975])))


def _score_to_elo(score: float) -> float | None:
    if not math.isfinite(score):
        return None
    score = min(max(float(score), 1e-6), 1.0 - 1e-6)
    return float(400.0 * math.log10(score / (1.0 - score)))


def _parse_csv_ints(value: str) -> list[int]:
    if not value.strip():
        return []
    result = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


# ---------------------------------------------------------------------------
# Frozen checkpoint inventory
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CheckpointInfo:
    iteration: int
    label: str
    path: Path
    checkpoint_sha256: str
    model_sha256: str
    global_step: int | None
    leaf_value_scale: float | None
    completed: bool | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "label": self.label,
            "path": str(self.path),
            "checkpoint_sha256": self.checkpoint_sha256,
            "model_sha256": self.model_sha256,
            "global_step": self.global_step,
            "leaf_value_scale": self.leaf_value_scale,
            "completed": self.completed,
        }


def scan_frozen_champions(directory: Path) -> tuple[list[CheckpointInfo], list[dict[str, Any]]]:
    directory = directory.resolve()
    if not directory.exists():
        raise FileNotFoundError(directory)

    discovered: list[CheckpointInfo] = []
    rejected: list[dict[str, Any]] = []

    for path in sorted(directory.rglob("*.pt")):
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(payload, dict):
                raise RuntimeError("checkpoint payload is not a dictionary")
            state_dict = _model_state_dict(payload)
            iteration = _extract_iteration(path, payload)
            if iteration is None:
                raise RuntimeError("iteration could not be inferred")
            discovered.append(
                CheckpointInfo(
                    iteration=iteration,
                    label=f"iteration_{iteration:06d}",
                    path=path.resolve(),
                    checkpoint_sha256=_sha256(path),
                    model_sha256=_state_dict_hash(state_dict),
                    global_step=(
                        int(payload["global_step"])
                        if payload.get("global_step") is not None
                        else None
                    ),
                    leaf_value_scale=(
                        float(payload["leaf_value_scale"])
                        if payload.get("leaf_value_scale") is not None
                        else None
                    ),
                    completed=(
                        bool(payload["completed"])
                        if payload.get("completed") is not None
                        else None
                    ),
                )
            )
        except Exception as error:  # inventory should report, not silently omit
            rejected.append({"path": str(path.resolve()), "error": repr(error)})

    if not discovered:
        raise RuntimeError(f"no usable .pt checkpoints found below {directory}")

    # Prefer a completed checkpoint and then the lexicographically newest path
    # if several files claim the same iteration. Preserve duplicate-model info.
    grouped: dict[int, list[CheckpointInfo]] = defaultdict(list)
    for item in discovered:
        grouped[item.iteration].append(item)

    selected = []
    duplicate_iterations = []
    for iteration in sorted(grouped):
        choices = sorted(
            grouped[iteration],
            key=lambda item: (
                item.completed is True,
                item.path.stat().st_mtime_ns,
                str(item.path),
            ),
            reverse=True,
        )
        selected.append(choices[0])
        if len(choices) > 1:
            duplicate_iterations.append(
                {
                    "iteration": iteration,
                    "selected": str(choices[0].path),
                    "alternatives": [str(item.path) for item in choices[1:]],
                }
            )

    rejected.extend(
        {"kind": "duplicate_iteration", **item}
        for item in duplicate_iterations
    )
    return selected, rejected


# ---------------------------------------------------------------------------
# Agent and game definitions
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentSpec:
    kind: str
    checkpoint: CheckpointInfo | None = None

    @property
    def label(self) -> str:
        if self.kind == "exact082":
            return "exact082"
        if self.checkpoint is None:
            raise AssertionError("model-backed agent lacks a checkpoint")
        return f"{self.checkpoint.label}_{self.kind}"

    @property
    def identity(self) -> tuple[str, str]:
        if self.kind == "exact082":
            return ("external", "exact082")
        if self.checkpoint is None:
            raise AssertionError("model-backed agent lacks a checkpoint")
        return (self.checkpoint.model_sha256, self.kind)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "label": self.label,
            "checkpoint": (
                self.checkpoint.to_dict() if self.checkpoint is not None else None
            ),
        }


@dataclass(frozen=True)
class MatchupSpec:
    primary: AgentSpec
    secondary: AgentSpec
    reason: str

    @property
    def label(self) -> str:
        return f"{self.primary.label}_vs_{self.secondary.label}"

    @property
    def cache_key(self) -> tuple[tuple[str, str], tuple[str, str]]:
        return (self.primary.identity, self.secondary.identity)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "reason": self.reason,
            "primary": self.primary.to_dict(),
            "secondary": self.secondary.to_dict(),
        }


@dataclass
class CheckpointContinuationGame:
    suite_id: int
    trajectory_family_id: int
    setup_seed: int
    primary_side: int
    primary_agent: AgentSpec
    secondary_agent: AgentSpec
    state: Any
    history: OracleHistory
    actions: list[int] = field(default_factory=list)
    position_occurrences: dict[int, list[int]] = field(default_factory=dict)
    cycle_events: list[dict[str, Any]] = field(default_factory=list)
    repetition_sensitive_roots: int = 0
    cycle_entry_actions: int = 0
    cycle_entry_with_legal_exit: int = 0
    cycle_entry_forced: int = 0
    max_repetition_context: int = 0

    @property
    def acting_agent(self) -> AgentSpec:
        return (
            self.primary_agent
            if int(self.state.current_player()) == self.primary_side
            else self.secondary_agent
        )


# ---------------------------------------------------------------------------
# Benchmark implementation
# ---------------------------------------------------------------------------

class CheckpointContinuationBenchmark:
    """Paired continuation benchmark across arbitrary frozen checkpoints."""

    def __init__(
        self,
        config_path: str | Path,
        champions_dir: str | Path,
        output: str | Path,
        *,
        workers: int = 16,
        block_states: int = 128,
    ) -> None:
        self.strength = SearchV3Strength(config_path, workers=workers)
        self.stages = self.strength.stages
        self.config = self.strength.config
        self.backend = self.stages.backend
        self.workers = max(1, int(workers))
        self.block_states = max(1, int(block_states))
        self.champions_dir = Path(champions_dir).resolve()
        self.output = Path(output).resolve()
        self.output.mkdir(parents=True, exist_ok=True)

        self.suite_path = (
            self.strength.run_dir
            / "single_layout_strength_suite"
            / "states.jsonl"
        )
        if not self.suite_path.exists():
            raise RuntimeError(
                f"single-layout strength suite is missing: {self.suite_path}"
            )
        self.suite = _read_jsonl(self.suite_path)
        self.suite_by_id = {
            int(row["suite_id"]): row for row in self.suite
        }
        self.inventory, self.inventory_notes = scan_frozen_champions(
            self.champions_dir
        )
        self.inventory_by_iteration = {
            item.iteration: item for item in self.inventory
        }
        atomic_json(
            self.output / "checkpoint_inventory.json",
            {
                "schema": "search-v3-frozen-checkpoint-inventory-v1",
                "champions_dir": str(self.champions_dir),
                "checkpoints": [item.to_dict() for item in self.inventory],
                "notes": self.inventory_notes,
            },
        )

    def _load_evaluator(self, checkpoint: CheckpointInfo) -> SearchEvaluator:
        payload = torch.load(
            checkpoint.path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = _model_state_dict(payload)
        actual_model_hash = _state_dict_hash(state_dict)
        if actual_model_hash != checkpoint.model_sha256:
            raise RuntimeError(
                f"checkpoint model changed after inventory: {checkpoint.path}"
            )

        model = PolicyValueNet(self.config.model)
        model.load_state_dict(state_dict)
        model.search_v3_leaf_value_scale = float(
            payload.get(
                "leaf_value_scale",
                checkpoint.leaf_value_scale
                if checkpoint.leaf_value_scale is not None
                else 1.0,
            )
        )
        model.eval()
        evaluator = SearchEvaluator(
            root=self.stages.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=self.strength.score_scale,
        )
        # Keep an explicit owner so the model cannot be collected while the
        # SearchEvaluator still refers to it.
        evaluator._checkpoint_benchmark_model_owner = model
        return evaluator

    def _state_history(
        self,
        row: dict[str, Any],
    ) -> tuple[Any, OracleHistory, dict[int, list[int]]]:
        state = self.backend.new_state(int(row["setup_seed"]))
        history = OracleHistory()
        positions: dict[int, list[int]] = defaultdict(list)
        positions[int(state.native_state.position_key())].append(
            int(state.native_state.turn_count())
        )
        for action in row["action_trace"]:
            history.observe(state)
            state.apply(int(action))
            positions[int(state.native_state.position_key())].append(
                int(state.native_state.turn_count())
            )
        history.observe(state)
        if int(state.native_state.key()) != int(row["state_key"]):
            raise RuntimeError(
                f"continuation start state mismatch at suite_id={row['suite_id']}"
            )
        return state, history, dict(positions)

    def _new_games(
        self,
        rows: list[dict[str, Any]],
        primary_agent: AgentSpec,
        secondary_agent: AgentSpec,
    ) -> list[CheckpointContinuationGame]:
        games = []
        for row in rows:
            for primary_side in (1, -1):
                state, history, positions = self._state_history(row)
                games.append(
                    CheckpointContinuationGame(
                        suite_id=int(row["suite_id"]),
                        trajectory_family_id=int(
                            row["trajectory_family_id"]
                        ),
                        setup_seed=int(row["setup_seed"]),
                        primary_side=primary_side,
                        primary_agent=primary_agent,
                        secondary_agent=secondary_agent,
                        state=state,
                        history=history,
                        position_occurrences=positions,
                    )
                )
        return games

    @staticmethod
    def _model_backed(agent: AgentSpec) -> bool:
        return agent.kind in {"bare", "search64", "search128", "search64r128"}

    @staticmethod
    def _repetition_sensitive(state: Any) -> bool:
        contexts = np.asarray(
            state.repetition_contexts(), dtype=np.uint8
        )
        return bool(
            (len(contexts) and int(contexts.max()) > 0)
            or int(state.native_state.repetition_count()) > 0
        )

    def _actions_for_agent(
        self,
        games: list[CheckpointContinuationGame],
        agent: AgentSpec,
        evaluators: dict[str, SearchEvaluator],
    ) -> tuple[list[int], dict[str, float]]:
        if not games:
            return [], {
                "search_simulations": 0.0,
                "search_seconds": 0.0,
            }

        if agent.kind == "exact082":
            # exact082 is model-independent; use either loaded evaluator.
            evaluator = next(iter(evaluators.values()))
            actions = []
            for game in games:
                state = game.state
                legal = np.asarray(state.legal_actions(), np.int32)
                actions.append(
                    int(
                        evaluator.exact082.select_action(
                            list(state.native_state.board()),
                            legal.tolist(),
                            1,
                            0.0,
                            0.0,
                        )
                    )
                )
            return actions, {
                "search_simulations": 0.0,
                "search_seconds": 0.0,
            }

        if agent.checkpoint is None:
            raise AssertionError("model-backed agent lacks checkpoint")
        evaluator = evaluators[agent.checkpoint.model_sha256]
        states = [game.state for game in games]

        if agent.kind == "bare":
            return (
                [int(result.action) for result in evaluator.search.bare(states)],
                {"search_simulations": 0.0, "search_seconds": 0.0},
            )

        if agent.kind in {"search64", "search128"}:
            budget = 64 if agent.kind == "search64" else 128
            results, metrics = evaluator.search.search(states, budget)
            return (
                [int(result.action) for result in results],
                {
                    "search_simulations": float(len(states) * budget),
                    "search_seconds": float(metrics["elapsed_seconds"]),
                },
            )

        if agent.kind == "search64r128":
            actions = [0] * len(games)
            simulations = 0.0
            seconds = 0.0
            grouped: dict[int, list[tuple[int, CheckpointContinuationGame]]] = {
                64: [],
                128: [],
            }
            for index, game in enumerate(games):
                budget = (
                    128
                    if self._repetition_sensitive(game.state)
                    else 64
                )
                grouped[budget].append((index, game))
            for budget, selected in grouped.items():
                if not selected:
                    continue
                results, metrics = evaluator.search.search(
                    [game.state for _, game in selected],
                    budget,
                )
                simulations += len(selected) * budget
                seconds += float(metrics["elapsed_seconds"])
                for (index, _game), result in zip(selected, results):
                    actions[index] = int(result.action)
            return actions, {
                "search_simulations": simulations,
                "search_seconds": seconds,
            }

        raise KeyError(agent.kind)

    def _record_action_context(
        self,
        game: CheckpointContinuationGame,
        action: int,
    ) -> None:
        state = game.state
        legal = list(map(int, state.legal_actions()))
        contexts = list(map(int, state.repetition_contexts()))
        if len(legal) != len(contexts):
            raise RuntimeError(
                f"repetition context mismatch at suite_id={game.suite_id}"
            )
        try:
            selected_index = legal.index(int(action))
        except ValueError as error:
            raise RuntimeError(
                f"illegal continuation action {action} at "
                f"suite_id={game.suite_id}"
            ) from error

        selected_context = int(contexts[selected_index])
        max_context = max(contexts, default=0)
        game.max_repetition_context = max(
            game.max_repetition_context, max_context
        )
        if (
            max_context > 0
            or int(state.native_state.repetition_count()) > 0
        ):
            game.repetition_sensitive_roots += 1

        if selected_context > 0:
            game.cycle_entry_actions += 1
            if any(context == 0 for context in contexts):
                game.cycle_entry_with_legal_exit += 1
            else:
                game.cycle_entry_forced += 1

        acting_side = int(state.current_player())
        before_turn = int(state.native_state.turn_count())
        state.apply(int(action))
        after_turn = int(state.native_state.turn_count())
        game.actions.append(int(action))

        new_key = int(state.native_state.position_key())
        previous = game.position_occurrences.get(new_key, [])
        if previous and len(game.cycle_events) < 128:
            last_seen = int(previous[-1])
            game.cycle_events.append(
                {
                    "continuation_ply": len(game.actions),
                    "absolute_turn_count": after_turn,
                    "last_seen_absolute_turn": last_seen,
                    "cycle_length": after_turn - last_seen,
                    "position_key": new_key,
                    "acting_side": acting_side,
                    "selected_action": int(action),
                    "selected_repetition_context": selected_context,
                    "legal_nonrepeat_exit_available": any(
                        context == 0 for context in contexts
                    ),
                    "before_turn_count": before_turn,
                }
            )
        game.position_occurrences.setdefault(new_key, []).append(after_turn)

        if not state.is_terminal():
            game.history.observe(state)

    def _play_block(
        self,
        rows: list[dict[str, Any]],
        matchup: MatchupSpec,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        checkpoints = {
            agent.checkpoint.model_sha256: agent.checkpoint
            for agent in (matchup.primary, matchup.secondary)
            if self._model_backed(agent) and agent.checkpoint is not None
        }
        if not checkpoints:
            raise RuntimeError("a matchup must contain at least one model-backed agent")

        evaluators = {
            model_hash: self._load_evaluator(checkpoint)
            for model_hash, checkpoint in checkpoints.items()
        }
        games = self._new_games(
            rows, matchup.primary, matchup.secondary
        )
        active = games[:]
        search_simulations = 0.0
        search_seconds = 0.0
        started = time.perf_counter()
        ply_round = 0

        try:
            while active:
                groups: dict[str, list[CheckpointContinuationGame]] = defaultdict(list)
                agents: dict[str, AgentSpec] = {}
                for game in active:
                    agent = game.acting_agent
                    groups[agent.label].append(game)
                    agents[agent.label] = agent

                action_by_game: dict[int, int] = {}
                for label, selected in groups.items():
                    actions, metrics = self._actions_for_agent(
                        selected,
                        agents[label],
                        evaluators,
                    )
                    search_simulations += metrics["search_simulations"]
                    search_seconds += metrics["search_seconds"]
                    for game, action in zip(selected, actions):
                        action_by_game[id(game)] = int(action)

                next_active = []
                for game in active:
                    action = action_by_game[id(game)]
                    self._record_action_context(game, action)
                    if not game.state.is_terminal():
                        next_active.append(game)
                active = next_active
                ply_round += 1

                if ply_round % 20 == 0 or not active:
                    completed = len(games) - len(active)
                    partial = []
                    repetitions = 0
                    for game in games:
                        if not game.state.is_terminal():
                            continue
                        value = float(
                            game.state.terminal_value(game.primary_side)
                        )
                        partial.append(
                            1.0 if value > 0 else (
                                0.5 if value == 0 else 0.0
                            )
                        )
                        if str(game.state.terminal_reason()) == "REPETITION":
                            repetitions += 1
                    print(
                        f"[checkpoint continuation] matchup={matchup.label} "
                        f"ply={ply_round} active={len(active)}/{len(games)} "
                        f"score="
                        f"{np.mean(partial) if partial else float('nan'):.3f} "
                        f"repetition={repetitions}/{max(1, len(partial))} "
                        f"games/s={completed / max(time.perf_counter() - started, 1e-9):.2f} "
                        f"search-sps={search_simulations / max(search_seconds, 1e-9):.0f}",
                        flush=True,
                    )
        finally:
            # SearchEvaluator owns GPU-side inference resources through its model.
            evaluators.clear()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        result_rows = []
        for game in games:
            value = float(
                game.state.terminal_value(game.primary_side)
            )
            score = 1.0 if value > 0 else (0.5 if value == 0 else 0.0)
            terminal_reason = str(game.state.terminal_reason())
            first_cycle = game.cycle_events[0] if game.cycle_events else None
            result_rows.append(
                {
                    "suite_id": game.suite_id,
                    "trajectory_family_id": game.trajectory_family_id,
                    "setup_seed": game.setup_seed,
                    "primary_side": game.primary_side,
                    "primary_agent": matchup.primary.label,
                    "secondary_agent": matchup.secondary.label,
                    "primary_model_sha256": (
                        matchup.primary.checkpoint.model_sha256
                        if matchup.primary.checkpoint is not None
                        else None
                    ),
                    "secondary_model_sha256": (
                        matchup.secondary.checkpoint.model_sha256
                        if matchup.secondary.checkpoint is not None
                        else None
                    ),
                    "score": score,
                    "outcome": (
                        "win" if value > 0 else (
                            "draw" if value == 0 else "loss"
                        )
                    ),
                    "continuation_plies": len(game.actions),
                    "terminal_reason": terminal_reason,
                    "terminal_repetition": int(
                        terminal_reason == "REPETITION"
                    ),
                    "terminal_no_capture": int(
                        terminal_reason in {
                            "NO_CAPTURE",
                            "NO_CAPTURE_LIMIT",
                            "NO_CAPTURE_PLY",
                        }
                    ),
                    "first_action": (
                        game.actions[0] if game.actions else None
                    ),
                    "trajectory_sha256": hashlib.sha256(
                        np.asarray(
                            game.actions, dtype=np.int16
                        ).tobytes()
                    ).hexdigest(),
                    "actions": game.actions,
                    "repetition_sensitive_roots": (
                        game.repetition_sensitive_roots
                    ),
                    "cycle_entry_actions": game.cycle_entry_actions,
                    "cycle_entry_with_legal_exit": (
                        game.cycle_entry_with_legal_exit
                    ),
                    "cycle_entry_forced": game.cycle_entry_forced,
                    "any_position_repeat": int(bool(game.cycle_events)),
                    "first_repeat_continuation_ply": (
                        int(first_cycle["continuation_ply"])
                        if first_cycle is not None
                        else None
                    ),
                    "first_cycle_length": (
                        int(first_cycle["cycle_length"])
                        if first_cycle is not None
                        else None
                    ),
                    "max_repetition_context": (
                        game.max_repetition_context
                    ),
                    "cycle_events": game.cycle_events,
                }
            )

        return result_rows, {
            "elapsed_seconds": time.perf_counter() - started,
            "search_simulations": int(search_simulations),
            "search_seconds": float(search_seconds),
            "search_simulations_per_second": (
                search_simulations / max(search_seconds, 1e-9)
            ),
        }

    def _summarize(
        self,
        rows: list[dict[str, Any]],
        matchup: MatchupSpec,
        performance: dict[str, Any],
    ) -> dict[str, Any]:
        scores = np.asarray(
            [float(row["score"]) for row in rows], np.float64
        )
        outcomes = Counter(row["outcome"] for row in rows)
        reasons = Counter(row["terminal_reason"] for row in rows)

        by_state: dict[int, list[float]] = defaultdict(list)
        state_repetition: dict[int, list[float]] = defaultdict(list)
        family_scores: dict[int, list[float]] = defaultdict(list)
        family_repetition: dict[int, list[float]] = defaultdict(list)

        for row in rows:
            suite_id = int(row["suite_id"])
            family_id = int(row["trajectory_family_id"])
            by_state[suite_id].append(float(row["score"]))
            state_repetition[suite_id].append(
                float(row["terminal_repetition"])
            )
            family_scores[family_id].append(float(row["score"]))
            family_repetition[family_id].append(
                float(row["terminal_repetition"])
            )

        if any(len(items) != 2 for items in by_state.values()):
            raise RuntimeError("paired role swap is incomplete")

        paired_scores = np.asarray(
            [np.mean(items) for items in by_state.values()],
            np.float64,
        )
        paired_repetition = np.asarray(
            [np.mean(items) for items in state_repetition.values()],
            np.float64,
        )
        family_score_values = np.asarray(
            [np.mean(items) for items in family_scores.values()],
            np.float64,
        )
        family_repetition_values = np.asarray(
            [np.mean(items) for items in family_repetition.values()],
            np.float64,
        )

        by_category = {}
        categories = sorted(
            {
                self.suite_by_id[int(row["suite_id"])]["primary_category"]
                for row in rows
            }
        )
        for category in categories:
            selected = [
                row
                for row in rows
                if self.suite_by_id[int(row["suite_id"])][
                    "primary_category"
                ]
                == category
            ]
            selected_scores = np.asarray(
                [float(row["score"]) for row in selected],
                np.float64,
            )
            category_repetition = sum(
                int(row["terminal_repetition"]) for row in selected
            )
            by_category[category] = {
                "games": len(selected),
                "score": float(selected_scores.mean()),
                "repetition_fraction": (
                    category_repetition / max(1, len(selected))
                ),
                "average_continuation_ply": float(
                    np.mean(
                        [row["continuation_plies"] for row in selected]
                    )
                ),
            }

        cycle_lengths = [
            int(event["cycle_length"])
            for row in rows
            for event in row["cycle_events"]
        ]
        first_repeat_plies = [
            int(row["first_repeat_continuation_ply"])
            for row in rows
            if row["first_repeat_continuation_ply"] is not None
        ]
        cycle_entry_total = sum(
            int(row["cycle_entry_actions"]) for row in rows
        )
        cycle_entry_exit = sum(
            int(row["cycle_entry_with_legal_exit"]) for row in rows
        )

        score = float(scores.mean())
        return {
            "schema": "search-v3-checkpoint-continuation-result-v1",
            "matchup": matchup.to_dict(),
            "games": len(rows),
            "paired_states": len(by_state),
            "trajectory_families": len(family_scores),
            "wins": outcomes["win"],
            "draws": outcomes["draw"],
            "losses": outcomes["loss"],
            "score": score,
            "elo_from_primary_perspective": _score_to_elo(score),
            "state_weighted_score": float(paired_scores.mean()),
            "family_weighted_mean_score": float(
                family_score_values.mean()
            ),
            "paired_state_ci95": _bootstrap(
                paired_scores,
                self.config.seed
                ^ sum(map(ord, matchup.label))
                ^ 0x53544154,
            ),
            "family_ci95": _bootstrap(
                family_score_values,
                self.config.seed
                ^ sum(map(ord, matchup.label))
                ^ 0xFA11,
            ),
            "terminal_reasons": dict(reasons),
            "repetition_fraction": float(
                np.mean(
                    [int(row["terminal_repetition"]) for row in rows]
                )
            ),
            "paired_state_repetition_ci95": _bootstrap(
                paired_repetition,
                self.config.seed
                ^ sum(map(ord, matchup.label))
                ^ 0x52505453,
            ),
            "family_repetition_mean": float(
                family_repetition_values.mean()
            ),
            "family_repetition_ci95": _bootstrap(
                family_repetition_values,
                self.config.seed
                ^ sum(map(ord, matchup.label))
                ^ 0x52505446,
            ),
            "any_position_repeat_fraction": float(
                np.mean(
                    [int(row["any_position_repeat"]) for row in rows]
                )
            ),
            "cycle_entry_actions": int(cycle_entry_total),
            "cycle_entry_with_legal_exit": int(cycle_entry_exit),
            "cycle_entry_forced": int(
                sum(int(row["cycle_entry_forced"]) for row in rows)
            ),
            "legal_exit_proxy_fraction": (
                cycle_entry_exit / max(1, cycle_entry_total)
            ),
            "repetition_sensitive_roots": int(
                sum(
                    int(row["repetition_sensitive_roots"])
                    for row in rows
                )
            ),
            "average_continuation_ply": float(
                np.mean([row["continuation_plies"] for row in rows])
            ),
            "first_repeat_continuation_ply": {
                "count": len(first_repeat_plies),
                "median": (
                    float(np.median(first_repeat_plies))
                    if first_repeat_plies
                    else None
                ),
                "p90": (
                    float(np.quantile(first_repeat_plies, 0.9))
                    if first_repeat_plies
                    else None
                ),
            },
            "cycle_length": {
                "count": len(cycle_lengths),
                "median": (
                    float(np.median(cycle_lengths))
                    if cycle_lengths
                    else None
                ),
                "p90": (
                    float(np.quantile(cycle_lengths, 0.9))
                    if cycle_lengths
                    else None
                ),
            },
            "unique_continuation_trajectories": len(
                {row["trajectory_sha256"] for row in rows}
            ),
            "by_primary_category": by_category,
            "performance": performance,
            "paired_role_swap_verified": True,
            "same_suite_seed_verified": True,
            "sampling": False,
            "dirichlet": False,
        }

    def _contract(
        self,
        matchup: MatchupSpec,
        states: int,
    ) -> dict[str, Any]:
        contract = {
            "schema": "search-v3-checkpoint-continuation-contract-v1",
            "suite": str(self.suite_path),
            "suite_sha256": _sha256(self.suite_path),
            "config_hash": self.config.hash(),
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "states": states,
            "block_states": self.block_states,
            "sampling": False,
            "dirichlet": False,
            "paired_role_swap": True,
            "statistical_unit": "suite state pair / trajectory family",
            "matchup": matchup.to_dict(),
        }
        rendered = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        contract["contract_hash"] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        return contract

    def run_matchup(
        self,
        matchup: MatchupSpec,
        *,
        states: int,
    ) -> dict[str, Any]:
        if states < 1 or states > len(self.suite):
            raise ValueError(
                f"states must be within [1, {len(self.suite)}]"
            )

        contract = self._contract(matchup, states)
        directory = (
            self.output
            / f"states_{states:04d}"
            / _safe_name(matchup.label)
        )
        directory.mkdir(parents=True, exist_ok=True)
        atomic_json(directory / "contract.json", contract)

        selected = self.suite[:states]
        all_rows = []
        performances = []

        for begin in range(0, states, self.block_states):
            end = min(states, begin + self.block_states)
            transaction = (
                directory / f"states_{begin:04d}_{end:04d}.json"
            )
            if transaction.exists():
                payload = json.loads(
                    transaction.read_text(encoding="utf-8")
                )
                if (
                    payload.get("completed") is not True
                    or payload.get("contract_hash")
                    != contract["contract_hash"]
                ):
                    raise RuntimeError(
                        f"invalid checkpoint transaction {transaction}"
                    )
                rows = payload["rows"]
                performance = payload["performance"]
            else:
                rows, performance = self._play_block(
                    selected[begin:end], matchup
                )
                atomic_json(
                    transaction,
                    {
                        "schema": (
                            "search-v3-checkpoint-continuation-transaction-v1"
                        ),
                        "completed": True,
                        "contract_hash": contract["contract_hash"],
                        "range": [begin, end],
                        "paired_games": 2 * (end - begin),
                        "rows": rows,
                        "performance": performance,
                    },
                )
            all_rows.extend(rows)
            performances.append(performance)

            partial_scores = np.asarray(
                [float(row["score"]) for row in all_rows],
                np.float64,
            )
            repetitions = sum(
                int(row["terminal_repetition"]) for row in all_rows
            )
            print(
                f"[checkpoint benchmark] matchup={matchup.label} "
                f"states={end}/{states} games={len(all_rows)} "
                f"score={partial_scores.mean():.4f} "
                f"repetition={repetitions / max(1, len(all_rows)):.2%}",
                flush=True,
            )

        performance = {
            "elapsed_seconds": float(
                sum(item["elapsed_seconds"] for item in performances)
            ),
            "search_simulations": int(
                sum(item["search_simulations"] for item in performances)
            ),
            "search_seconds": float(
                sum(item["search_seconds"] for item in performances)
            ),
        }
        performance["search_simulations_per_second"] = (
            performance["search_simulations"]
            / max(performance["search_seconds"], 1e-9)
        )

        summary = self._summarize(
            all_rows, matchup, performance
        )
        summary["contract_hash"] = contract["contract_hash"]
        summary["rows_path"] = str(directory / "games.jsonl")
        atomic_json(directory / "summary.json", summary)
        _atomic_jsonl(directory / "games.jsonl", all_rows)
        return summary

    def _selected_checkpoints(
        self,
        iterations: list[int],
    ) -> list[CheckpointInfo]:
        if not iterations:
            return list(self.inventory)
        missing = [
            iteration
            for iteration in iterations
            if iteration not in self.inventory_by_iteration
        ]
        if missing:
            raise RuntimeError(
                f"requested frozen iterations are missing: {missing}; "
                f"available={sorted(self.inventory_by_iteration)}"
            )
        return [
            self.inventory_by_iteration[iteration]
            for iteration in sorted(set(iterations))
        ]

    @staticmethod
    def _same_generation_matchups(
        checkpoints: list[CheckpointInfo],
        adaptive_search: bool,
    ) -> list[MatchupSpec]:
        search_kind = "search64r128" if adaptive_search else "search64"
        result = []
        for checkpoint in checkpoints:
            bare = AgentSpec("bare", checkpoint)
            search = AgentSpec(search_kind, checkpoint)
            exact = AgentSpec("exact082")
            result.extend(
                [
                    MatchupSpec(
                        bare,
                        exact,
                        reason="same_generation_absolute_bare",
                    ),
                    MatchupSpec(
                        search,
                        exact,
                        reason="same_generation_absolute_search",
                    ),
                    MatchupSpec(
                        search,
                        bare,
                        reason="same_generation_search_gain",
                    ),
                ]
            )
        return result

    @staticmethod
    def _cross_generation_matchups(
        newer: CheckpointInfo,
        older: CheckpointInfo,
        adaptive_search: bool,
        reason: str,
    ) -> list[MatchupSpec]:
        search_kind = "search64r128" if adaptive_search else "search64"
        newer_bare = AgentSpec("bare", newer)
        older_bare = AgentSpec("bare", older)
        newer_search = AgentSpec(search_kind, newer)
        older_search = AgentSpec(search_kind, older)
        return [
            MatchupSpec(
                newer_bare,
                older_bare,
                reason=f"{reason}_bare_vs_bare",
            ),
            MatchupSpec(
                newer_search,
                older_search,
                reason=f"{reason}_search_vs_search",
            ),
            MatchupSpec(
                newer_bare,
                older_search,
                reason=f"{reason}_new_bare_vs_old_search",
            ),
            MatchupSpec(
                newer_search,
                older_bare,
                reason=f"{reason}_new_search_vs_old_bare",
            ),
        ]

    def build_plan(
        self,
        *,
        iterations: list[int],
        anchors: list[int],
        adjacent: bool,
        full_matrix: bool,
        include_same_generation: bool,
        adaptive_search: bool,
    ) -> tuple[list[CheckpointInfo], list[MatchupSpec], list[dict[str, Any]]]:
        selected = self._selected_checkpoints(iterations)
        if len(selected) < 1:
            raise RuntimeError("no checkpoints selected")

        plan: list[MatchupSpec] = []
        notes: list[dict[str, Any]] = []

        if include_same_generation:
            plan.extend(
                self._same_generation_matchups(
                    selected, adaptive_search
                )
            )

        pair_reasons: dict[tuple[int, int], set[str]] = defaultdict(set)

        if adjacent:
            for older, newer in zip(selected, selected[1:]):
                pair_reasons[(newer.iteration, older.iteration)].add(
                    "adjacent"
                )

        selected_by_iteration = {
            item.iteration: item for item in selected
        }
        for anchor_iteration in anchors:
            if anchor_iteration not in selected_by_iteration:
                raise RuntimeError(
                    f"anchor {anchor_iteration} is not in selected iterations"
                )
            anchor = selected_by_iteration[anchor_iteration]
            for checkpoint in selected:
                if checkpoint.iteration == anchor.iteration:
                    continue
                newer, older = (
                    (checkpoint, anchor)
                    if checkpoint.iteration > anchor.iteration
                    else (anchor, checkpoint)
                )
                pair_reasons[(newer.iteration, older.iteration)].add(
                    f"anchor_{anchor_iteration}"
                )

        if full_matrix:
            for older_index, older in enumerate(selected):
                for newer in selected[older_index + 1:]:
                    pair_reasons[(newer.iteration, older.iteration)].add(
                        "full_matrix"
                    )

        for (newer_iteration, older_iteration), reasons in sorted(
            pair_reasons.items()
        ):
            newer = selected_by_iteration[newer_iteration]
            older = selected_by_iteration[older_iteration]
            plan.extend(
                self._cross_generation_matchups(
                    newer,
                    older,
                    adaptive_search,
                    reason="+".join(sorted(reasons)),
                )
            )

        # Deduplicate by actual model hash + agent kind, not checkpoint filename.
        # This prevents another "iteration 80 vs iteration 50 but same model"
        # benchmark while retaining search-vs-bare tests on one model.
        deduplicated: list[MatchupSpec] = []
        seen: dict[
            tuple[tuple[str, str], tuple[str, str]],
            MatchupSpec,
        ] = {}
        for matchup in plan:
            key = matchup.cache_key
            if key in seen:
                notes.append(
                    {
                        "kind": "duplicate_matchup_skipped",
                        "skipped_label": matchup.label,
                        "kept_label": seen[key].label,
                        "reason": matchup.reason,
                    }
                )
                continue
            seen[key] = matchup
            deduplicated.append(matchup)

        return selected, deduplicated, notes

    def run(
        self,
        *,
        iterations: list[int],
        anchors: list[int],
        adjacent: bool,
        full_matrix: bool,
        include_same_generation: bool,
        adaptive_search: bool,
        states: int,
    ) -> dict[str, Any]:
        selected, plan, plan_notes = self.build_plan(
            iterations=iterations,
            anchors=anchors,
            adjacent=adjacent,
            full_matrix=full_matrix,
            include_same_generation=include_same_generation,
            adaptive_search=adaptive_search,
        )

        manifest = {
            "schema": "search-v3-checkpoint-benchmark-manifest-v1",
            "created_unix": time.time(),
            "config": str(self.strength.stages.config_path)
            if hasattr(self.strength.stages, "config_path")
            else None,
            "config_hash": self.config.hash(),
            "suite": str(self.suite_path),
            "suite_sha256": _sha256(self.suite_path),
            "champions_dir": str(self.champions_dir),
            "states": states,
            "block_states": self.block_states,
            "workers": self.workers,
            "adaptive_search": adaptive_search,
            "selected_checkpoints": [
                item.to_dict() for item in selected
            ],
            "plan": [matchup.to_dict() for matchup in plan],
            "plan_notes": plan_notes,
        }
        atomic_json(self.output / "manifest.json", manifest)

        summaries = {}
        for index, matchup in enumerate(plan, start=1):
            print(
                f"[checkpoint benchmark] {index}/{len(plan)} "
                f"{matchup.label} reason={matchup.reason}",
                flush=True,
            )
            summaries[matchup.label] = self.run_matchup(
                matchup, states=states
            )

        report = {
            "schema": "search-v3-checkpoint-benchmark-report-v1",
            "manifest": manifest,
            "matchups": summaries,
        }
        atomic_json(self.output / "report.json", report)
        self._write_summary_csv(
            self.output / "summary.csv", summaries
        )
        self._write_pairwise_elo_csv(
            self.output / "pairwise_elo.csv", summaries
        )
        self._write_repetition_csv(
            self.output / "repetition_summary.csv", summaries
        )
        return report

    @staticmethod
    def _write_summary_csv(
        path: Path,
        summaries: dict[str, dict[str, Any]],
    ) -> None:
        fields = [
            "matchup",
            "reason",
            "primary",
            "secondary",
            "games",
            "score",
            "elo",
            "family_ci_low",
            "family_ci_high",
            "wins",
            "draws",
            "losses",
            "repetition_fraction",
            "family_repetition_ci_low",
            "family_repetition_ci_high",
            "average_continuation_ply",
            "search_simulations_per_second",
        ]
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label, summary in summaries.items():
                writer.writerow(
                    {
                        "matchup": label,
                        "reason": summary["matchup"]["reason"],
                        "primary": summary["primary_agent"]
                        if "primary_agent" in summary
                        else summary["matchup"]["primary"]["label"],
                        "secondary": summary["secondary_agent"]
                        if "secondary_agent" in summary
                        else summary["matchup"]["secondary"]["label"],
                        "games": summary["games"],
                        "score": summary["score"],
                        "elo": summary[
                            "elo_from_primary_perspective"
                        ],
                        "family_ci_low": summary["family_ci95"][0],
                        "family_ci_high": summary["family_ci95"][1],
                        "wins": summary["wins"],
                        "draws": summary["draws"],
                        "losses": summary["losses"],
                        "repetition_fraction": summary[
                            "repetition_fraction"
                        ],
                        "family_repetition_ci_low": summary[
                            "family_repetition_ci95"
                        ][0],
                        "family_repetition_ci_high": summary[
                            "family_repetition_ci95"
                        ][1],
                        "average_continuation_ply": summary[
                            "average_continuation_ply"
                        ],
                        "search_simulations_per_second": summary[
                            "performance"
                        ]["search_simulations_per_second"],
                    }
                )
        os.replace(temporary, path)

    @staticmethod
    def _write_pairwise_elo_csv(
        path: Path,
        summaries: dict[str, dict[str, Any]],
    ) -> None:
        fields = [
            "primary",
            "secondary",
            "score",
            "elo",
            "family_ci_low",
            "family_ci_high",
            "reason",
        ]
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for summary in summaries.values():
                primary = summary["matchup"]["primary"]["label"]
                secondary = summary["matchup"]["secondary"]["label"]
                if secondary == "exact082":
                    continue
                writer.writerow(
                    {
                        "primary": primary,
                        "secondary": secondary,
                        "score": summary["score"],
                        "elo": summary[
                            "elo_from_primary_perspective"
                        ],
                        "family_ci_low": summary["family_ci95"][0],
                        "family_ci_high": summary["family_ci95"][1],
                        "reason": summary["matchup"]["reason"],
                    }
                )
        os.replace(temporary, path)

    @staticmethod
    def _write_repetition_csv(
        path: Path,
        summaries: dict[str, dict[str, Any]],
    ) -> None:
        fields = [
            "matchup",
            "primary",
            "secondary",
            "games",
            "repetition_fraction",
            "family_repetition_mean",
            "family_repetition_ci_low",
            "family_repetition_ci_high",
            "any_position_repeat_fraction",
            "cycle_entry_actions",
            "cycle_entry_with_legal_exit",
            "cycle_entry_forced",
            "legal_exit_proxy_fraction",
            "repetition_sensitive_roots",
            "average_continuation_ply",
        ]
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label, summary in summaries.items():
                writer.writerow(
                    {
                        "matchup": label,
                        "primary": summary["matchup"]["primary"][
                            "label"
                        ],
                        "secondary": summary["matchup"]["secondary"][
                            "label"
                        ],
                        "games": summary["games"],
                        "repetition_fraction": summary[
                            "repetition_fraction"
                        ],
                        "family_repetition_mean": summary[
                            "family_repetition_mean"
                        ],
                        "family_repetition_ci_low": summary[
                            "family_repetition_ci95"
                        ][0],
                        "family_repetition_ci_high": summary[
                            "family_repetition_ci95"
                        ][1],
                        "any_position_repeat_fraction": summary[
                            "any_position_repeat_fraction"
                        ],
                        "cycle_entry_actions": summary[
                            "cycle_entry_actions"
                        ],
                        "cycle_entry_with_legal_exit": summary[
                            "cycle_entry_with_legal_exit"
                        ],
                        "cycle_entry_forced": summary[
                            "cycle_entry_forced"
                        ],
                        "legal_exit_proxy_fraction": summary[
                            "legal_exit_proxy_fraction"
                        ],
                        "repetition_sensitive_roots": summary[
                            "repetition_sensitive_roots"
                        ],
                        "average_continuation_ply": summary[
                            "average_continuation_ply"
                        ],
                    }
                )
        os.replace(temporary, path)


def run_checkpoint_benchmark(
    config_path: str | Path,
    champions_dir: str | Path,
    output: str | Path,
    *,
    iterations: list[int] | None = None,
    anchors: list[int] | None = None,
    adjacent: bool = True,
    full_matrix: bool = False,
    include_same_generation: bool = True,
    adaptive_search: bool = True,
    states: int = 256,
    workers: int = 16,
    block_states: int = 128,
) -> dict[str, Any]:
    benchmark = CheckpointContinuationBenchmark(
        config_path,
        champions_dir,
        output,
        workers=workers,
        block_states=block_states,
    )
    return benchmark.run(
        iterations=iterations or [],
        anchors=anchors or [],
        adjacent=adjacent,
        full_matrix=full_matrix,
        include_same_generation=include_same_generation,
        adaptive_search=adaptive_search,
        states=states,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark frozen Search-v3 checkpoints across generations using "
            "the immutable paired continuation arena."
        )
    )
    parser.add_argument(
        "--config",
        default="config/open_junqi_search_v3_strength.json",
    )
    parser.add_argument("--champions-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--iterations",
        default="",
        help="Comma-separated iterations; empty means every frozen checkpoint.",
    )
    parser.add_argument(
        "--anchors",
        default="",
        help="Comma-separated selected iterations used as fixed comparison anchors.",
    )
    parser.add_argument(
        "--adjacent",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--full-matrix", action="store_true")
    parser.add_argument(
        "--same-generation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--adaptive-search",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use Search64 normally and Search128 at repetition-sensitive roots. "
            "Disable to use plain Search64 everywhere."
        ),
    )
    parser.add_argument("--states", type=int, default=256)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--block-states", type=int, default=128)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    report = run_checkpoint_benchmark(
        arguments.config,
        arguments.champions_dir,
        arguments.output,
        iterations=_parse_csv_ints(arguments.iterations),
        anchors=_parse_csv_ints(arguments.anchors),
        adjacent=arguments.adjacent,
        full_matrix=arguments.full_matrix,
        include_same_generation=arguments.same_generation,
        adaptive_search=arguments.adaptive_search,
        states=arguments.states,
        workers=arguments.workers,
        block_states=arguments.block_states,
    )
    print(
        json.dumps(
            {
                "checkpoints": [
                    item["iteration"]
                    for item in report["manifest"][
                        "selected_checkpoints"
                    ]
                ],
                "matchups": len(report["matchups"]),
                "states_per_matchup": report["manifest"]["states"],
                "report": str(
                    Path(arguments.output).resolve() / "report.json"
                ),
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
