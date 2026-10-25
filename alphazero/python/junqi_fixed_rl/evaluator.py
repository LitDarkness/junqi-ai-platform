from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .config import FixedRLConfig
from .layout import FixedLayout, FixedLayoutPairBackend
from .teachers import TeacherEnsemble


TERMINAL_CATEGORIES = (
    "capture_flag",
    "no_legal_move",
    "repetition_draw",
    "no_capture_draw",
    "max_ply_cutoff",
    "other",
)


def terminal_category(state: Any, cutoff: bool) -> str:
    if cutoff and not state.is_terminal():
        return "max_ply_cutoff"
    reason = state.terminal_reason() if state.is_terminal() else ""
    return {
        "FLAG_CAPTURE": "capture_flag",
        "NO_LEGAL_MOVE": "no_legal_move",
        "REPETITION": "repetition_draw",
        "NO_CAPTURE_LIMIT": "no_capture_draw",
        "MAX_PLY": "max_ply_cutoff",
    }.get(reason, "other")


def score_statistics(scores: Sequence[float]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    if not len(values):
        raise ValueError("cannot summarize zero evaluation scores")
    mean = float(values.mean())
    standard_error = (
        float(values.std(ddof=1) / math.sqrt(len(values)))
        if len(values) > 1
        else 0.0
    )
    radius = 1.96 * standard_error
    return {
        "score_rate": mean,
        "standard_error": standard_error,
        "ci95_lower": max(0.0, mean - radius),
        "ci95_upper": min(1.0, mean + radius),
    }


def repetition_severity(fraction: float) -> str:
    if fraction < 0.15:
        return "normal"
    if fraction < 0.30:
        return "inspect_by_opponent"
    if fraction < 0.50:
        return "suspicious"
    return "likely_policy_degradation"


def repetition_cycle_details(
    position_hashes: Sequence[int],
    action_trace: Sequence[dict[str, Any]],
    position_values: Sequence[float | None],
    *,
    terminal_value: float,
    opponent_type: str,
) -> dict[str, Any]:
    if len(position_hashes) != len(action_trace) + 1:
        raise ValueError("position hashes must bracket every action")
    terminal_hash = int(position_hashes[-1])
    occurrences = [
        index
        for index, value in enumerate(position_hashes)
        if int(value) == terminal_hash
    ]
    if len(occurrences) < 3:
        raise RuntimeError(
            "repetition terminal does not contain three matching position hashes"
        )
    first, second, terminal = occurrences[-3:]
    intervals = [second - first, terminal - second]
    cycle_length = intervals[-1]
    repeated_by = (
        action_trace[second - 1].get("mover_side")
        if second > 0
        else "initial_position"
    )
    value_at_start = (
        position_values[first] if first < len(position_values) else None
    )
    return {
        "first_cycle_ply": first,
        "cycle_length": cycle_length,
        "cycle_repeat_intervals": intervals,
        "side_that_first_repeated": repeated_by,
        "position_hash_sequence": [
            f"0x{int(value):016x}"
            for value in position_hashes[first : second + 1]
        ],
        "value_at_cycle_start": (
            None if value_at_start is None else float(value_at_start)
        ),
        "terminal_value": float(terminal_value),
        "opponent_type": opponent_type,
        "cycle_actions": [
            dict(row) for row in action_trace[first:second]
        ],
    }


@dataclass(frozen=True)
class Agent:
    label: str
    kind: str
    checkpoint: Path | None = None


@dataclass
class EvaluationGame:
    state: Any
    seed: int
    pair_index: int
    a_side: int
    rng: random.Random
    moves: int = 0
    position_hashes: list[int] = field(default_factory=list)
    action_trace: list[dict[str, Any]] = field(default_factory=list)
    position_values: list[float | None] = field(default_factory=list)


class IndependentEvaluator:
    """Read-only fixed-layout evaluator; never touches optimizer/replay/train RNG."""

    def __init__(
        self,
        config: FixedRLConfig,
        layout: FixedLayout,
        *,
        project_root: Path,
        black_layout: FixedLayout | None = None,
    ) -> None:
        self.config = config
        self.red_layout = layout
        self.black_layout = black_layout or layout
        self.layout = layout
        self.project_root = Path(project_root)
        requested = config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.backend = FixedLayoutPairBackend(
            self.red_layout,
            self.black_layout,
        )
        self.teacher = TeacherEnsemble(
            config.teacher,
            project_root=self.project_root,
            seed=config.evaluation.seed ^ 0x4556414C,
        )
        self._models: dict[str, PolicyValueNet] = {}

    def close(self) -> None:
        self.teacher.close()
        self._models.clear()

    def agent(self, specification: str | Path) -> Agent:
        text = str(specification)
        aliases = {
            "084": ("heuristic084", "084"),
            "082": ("heuristic082", "082"),
            "random": ("random", "random_exploiter"),
            "aggressive": ("aggressive", "aggressive_exploiter"),
        }
        if text.lower() in aliases:
            kind, label = aliases[text.lower()]
            return Agent(label=label, kind=kind)
        path = Path(text).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"evaluation checkpoint is missing: {path}")
        return Agent(label=path.stem, kind="checkpoint", checkpoint=path)

    def _model(self, agent: Agent) -> PolicyValueNet:
        assert agent.checkpoint is not None
        key = str(agent.checkpoint)
        model = self._models.get(key)
        if model is not None:
            return model
        payload = torch.load(agent.checkpoint, map_location="cpu", weights_only=False)
        # v2 advantage checkpoints publish an EMA snapshot explicitly for
        # read-only evaluation.  Legacy v1 checkpoints continue to use the
        # online trainer model exactly as before.
        state = payload.get("evaluation_model")
        if state is None:
            state = payload.get("trainer", {}).get("model")
        if state is None:
            state = payload.get("model")
        if state is None:
            raise RuntimeError(f"checkpoint has no model state: {agent.checkpoint}")
        model = PolicyValueNet(self.config.model)
        model.load_state_dict(state)
        model.to(self.device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        self._models[key] = model
        return model

    def _checkpoint_decisions(
        self,
        games: list[EvaluationGame],
        indices: list[int],
        agent: Agent,
        *,
        temperature: float,
        mode: str,
    ) -> list[tuple[int, float]]:
        states = [games[index].state for index in indices]
        observations = np.stack([state.feature_planes() for state in states])
        masks = np.zeros(
            (len(states), self.config.model.action_size), dtype=np.bool_
        )
        repetitions = np.zeros_like(masks, dtype=np.uint8)
        legal_rows: list[np.ndarray] = []
        for row, state in enumerate(states):
            legal = np.asarray(state.legal_actions(), dtype=np.int32)
            legal_rows.append(legal)
            masks[row, legal] = True
            repetitions[row, legal] = state.repetition_contexts()
        with torch.inference_mode():
            policy, values = self._model(agent).policy_value(
                torch.from_numpy(observations).to(self.device),
                torch.from_numpy(masks).to(self.device),
                torch.from_numpy(repetitions).to(self.device),
            )
        policy = policy.cpu().numpy()
        values_np = values.float().cpu().numpy().reshape(-1)
        actions: list[tuple[int, float]] = []
        for source_index, legal, row, value in zip(
            indices, legal_rows, policy, values_np
        ):
            probabilities = row[legal].astype(np.float64)
            if mode == "greedy" or temperature <= 0:
                actions.append(
                    (
                        int(legal[int(np.argmax(probabilities))]),
                        float(value),
                    )
                )
                continue
            probabilities = np.power(
                np.clip(probabilities, 1e-12, 1.0), 1.0 / temperature
            )
            probabilities /= probabilities.sum()
            actions.append(
                (
                    int(
                    games[source_index].rng.choices(
                        legal.tolist(), weights=probabilities.tolist(), k=1
                    )[0]
                    ),
                    float(value),
                )
            )
        return actions

    @staticmethod
    def _exploiter_action(
        game: EvaluationGame, kind: str, mode: str
    ) -> int:
        legal = np.asarray(game.state.legal_actions(), dtype=np.int32)
        if kind == "random":
            weights = np.ones(len(legal), dtype=np.float64)
        else:
            board = np.asarray(game.state.native_state.board(), dtype=np.int32)
            weights = np.ones(len(legal), dtype=np.float64)
            for index, action in enumerate(legal):
                source, target = divmod(int(action), 60)
                if board[target] < 0:
                    weights[index] += 8.0 + math.log1p(abs(int(board[target])))
                weights[index] += max(
                    0, target // 5 - source // 5
                ) * 0.25
        if mode == "greedy":
            return int(legal[int(np.argmax(weights))])
        return int(
            game.rng.choices(legal.tolist(), weights=weights.tolist(), k=1)[0]
        )

    def _seed_list(self, games: int, base_seed: int) -> list[int]:
        if games % 2:
            raise ValueError("paired evaluation requires an even game count")
        rng = random.Random(int(base_seed))
        return [rng.getrandbits(63) for _ in range(games // 2)]

    def evaluate(
        self,
        agent_a: Agent,
        agent_b: Agent,
        *,
        games: int,
        temperature: float,
        mode: str,
        base_seed: int,
        seed_offset_pairs: int = 0,
        output_prefix: Path | None = None,
    ) -> dict[str, Any]:
        if mode not in {"sample", "greedy"}:
            raise ValueError("evaluation mode must be sample or greedy")
        all_seeds = self._seed_list(
            games + 2 * seed_offset_pairs, base_seed
        )
        seeds = all_seeds[seed_offset_pairs:]
        if len(seeds) != games // 2:
            raise RuntimeError("internal paired seed construction failed")

        python_rng = random.getstate()
        numpy_rng = np.random.get_state()
        torch_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        started = time.perf_counter()
        active: list[EvaluationGame] = []
        for pair_index, seed in enumerate(seeds, start=seed_offset_pairs):
            red_game = EvaluationGame(
                    state=self.backend.new_state(seed),
                    seed=seed,
                    pair_index=pair_index,
                    a_side=1,
                    rng=random.Random(seed ^ 0xA11CE001),
            )
            black_game = EvaluationGame(
                    state=self.backend.new_state(seed),
                    seed=seed,
                    pair_index=pair_index,
                    a_side=-1,
                    rng=random.Random(seed ^ 0xA11CE002),
            )
            red_game.position_hashes.append(
                int(red_game.state.native_state.position_key())
            )
            black_game.position_hashes.append(
                int(black_game.state.native_state.position_key())
            )
            active.extend((red_game, black_game))
        completed_rows: list[dict[str, Any]] = []
        try:
            while active:
                groups: dict[str, tuple[Agent, list[int]]] = {}
                actions: dict[int, int] = {}
                decision_values: dict[int, float | None] = {}
                actors: dict[int, Agent] = {}
                heuristic_groups: dict[str, list[int]] = {
                    "heuristic084": [],
                    "heuristic082": [],
                }
                for index, game in enumerate(active):
                    actor = (
                        agent_a
                        if game.state.current_player() == game.a_side
                        else agent_b
                    )
                    actors[index] = actor
                    if actor.kind == "checkpoint":
                        key = str(actor.checkpoint)
                        if key not in groups:
                            groups[key] = (actor, [])
                        groups[key][1].append(index)
                    elif actor.kind in heuristic_groups:
                        heuristic_groups[actor.kind].append(index)
                    else:
                        actions[index] = self._exploiter_action(
                            game, actor.kind, mode
                        )
                        decision_values[index] = None
                for actor, indices in groups.values():
                    decided = self._checkpoint_decisions(
                        active,
                        indices,
                        actor,
                        temperature=temperature,
                        mode=mode,
                    )
                    for index, (action, value) in zip(indices, decided):
                        actions[index] = action
                        decision_values[index] = value
                for kind, indices in heuristic_groups.items():
                    if not indices:
                        continue
                    decided = self.teacher.external_actions(
                        [active[index].state for index in indices],
                        historical=kind == "heuristic082",
                    )
                    actions.update(zip(indices, decided))
                    for index in indices:
                        decision_values[index] = None

                finished = []
                for index, game in enumerate(active):
                    action = actions[index]
                    mover = int(game.state.current_player())
                    source, target = divmod(int(action), 60)
                    board = game.state.native_state.board()
                    game.action_trace.append(
                        {
                            "ply": game.moves,
                            "actor": actors[index].label,
                            "mover_side": "red" if mover == 1 else "black",
                            "action": int(action),
                            "from": source,
                            "to": target,
                            "moved_piece": int(board[source]),
                        }
                    )
                    game.position_values.append(decision_values.get(index))
                    game.state.apply(action)
                    game.moves += 1
                    game.position_hashes.append(
                        int(game.state.native_state.position_key())
                    )
                    cutoff = game.moves >= self.config.selfplay.max_moves
                    if game.state.is_terminal() or cutoff:
                        reason = terminal_category(game.state, cutoff)
                        if game.state.is_terminal():
                            value = game.state.terminal_value(game.a_side)
                            score = 1.0 if value > 0 else 0.0 if value < 0 else 0.5
                        else:
                            score = 0.5
                        cycle = None
                        if reason == "repetition_draw":
                            cycle = repetition_cycle_details(
                                game.position_hashes,
                                game.action_trace,
                                game.position_values,
                                terminal_value=0.0,
                                opponent_type=agent_b.label,
                            )
                        completed_rows.append(
                            {
                                "pair_index": game.pair_index,
                                "seed": game.seed,
                                "a_side": "red" if game.a_side == 1 else "black",
                                "score": score,
                                "result": (
                                    "win" if score == 1.0
                                    else "loss" if score == 0.0
                                    else "draw"
                                ),
                                "moves": game.moves,
                                "terminal_reason": reason,
                                "repetition_cycle": cycle,
                            }
                        )
                        finished.append(index)
                for index in reversed(finished):
                    active.pop(index)
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
            torch.set_rng_state(torch_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)

        elapsed = time.perf_counter() - started
        result = self.summarize_rows(
            completed_rows,
            agent_a=agent_a.label,
            agent_b=agent_b.label,
            temperature=temperature,
            mode=mode,
            base_seed=base_seed,
            elapsed_seconds=elapsed,
        )
        result["agent_a_kind"] = agent_a.kind
        result["agent_b_kind"] = agent_b.kind
        if output_prefix is not None:
            self.write_outputs(output_prefix, result)
        return result

    def summarize_rows(
        self,
        rows: list[dict[str, Any]],
        *,
        agent_a: str,
        agent_b: str,
        temperature: float,
        mode: str,
        base_seed: int,
        elapsed_seconds: float,
    ) -> dict[str, Any]:
        scores = [float(row["score"]) for row in rows]
        statistics = score_statistics(scores)
        terminal = {
            category: sum(
                row["terminal_reason"] == category for row in rows
            )
            for category in TERMINAL_CATEGORIES
        }
        seeds = sorted({int(row["seed"]) for row in rows})
        seed_hash = hashlib.sha256(
            json.dumps(seeds, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        wins = sum(score == 1.0 for score in scores)
        draws = sum(score == 0.5 for score in scores)
        losses = sum(score == 0.0 for score in scores)
        distribution = {
            category: {
                "count": int(count),
                "fraction": count / max(1, len(rows)),
            }
            for category, count in terminal.items()
        }
        repetition_fraction = distribution["repetition_draw"]["fraction"]
        cycle_counts: dict[str, int] = {}
        for row in rows:
            cycle = row.get("repetition_cycle") or {}
            sequence = cycle.get("position_hash_sequence", [])
            if sequence:
                signature = ">".join(str(value) for value in sequence)
                cycle_counts[signature] = cycle_counts.get(signature, 0) + 1
        dominant_cycle_count = max(cycle_counts.values(), default=0)
        repeated_games = terminal["repetition_draw"]
        return {
            "schema": "junqi-fixed-layout-evaluation-v1",
            "agent_a": agent_a,
            "agent_b": agent_b,
            "games": len(rows),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            **statistics,
            "average_ply": float(np.mean([row["moves"] for row in rows])),
            "elapsed_seconds": elapsed_seconds,
            "moves_per_second": (
                sum(int(row["moves"]) for row in rows)
                / max(elapsed_seconds, 1e-9)
            ),
            "temperature": temperature,
            "mode": mode,
            "base_seed": base_seed,
            "seed_hash": seed_hash,
            "layout_id": (
                self.red_layout.layout_id
                if self.red_layout.layout_id == self.black_layout.layout_id
                else f"{self.red_layout.layout_id}__vs__{self.black_layout.layout_id}"
            ),
            "layout_hash": hashlib.sha256(
                (
                    ",".join(str(value) for value in self.red_layout.setup)
                    if self.red_layout.setup == self.black_layout.setup
                    else (
                        ",".join(str(value) for value in self.red_layout.setup)
                        + "|"
                        + ",".join(str(value) for value in self.black_layout.setup)
                    )
                ).encode("ascii")
            ).hexdigest(),
            "red_layout_id": self.red_layout.layout_id,
            "black_layout_id": self.black_layout.layout_id,
            "terminal_reasons": terminal,
            "terminal_distribution": distribution,
            "repetition_fraction": repetition_fraction,
            "dominant_cycle_count": dominant_cycle_count,
            "max_single_cycle_fraction": (
                dominant_cycle_count / max(1, repeated_games)
            ),
            "repetition_severity": repetition_severity(repetition_fraction),
            "capture_flag_rate": distribution["capture_flag"]["fraction"],
            "games_detail": rows,
        }

    @staticmethod
    def write_outputs(prefix: Path, result: dict[str, Any]) -> None:
        prefix.parent.mkdir(parents=True, exist_ok=True)
        json_path = prefix.with_suffix(".json")
        temporary = json_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, json_path)
        csv_path = prefix.with_suffix(".csv")
        temporary_csv = csv_path.with_suffix(".csv.tmp")
        with temporary_csv.open("w", encoding="utf-8-sig", newline="") as stream:
            fieldnames = [
                "pair_index",
                "seed",
                "a_side",
                "score",
                "result",
                "moves",
                "terminal_reason",
                "first_cycle_ply",
                "cycle_length",
                "side_that_first_repeated",
                "value_at_cycle_start",
                "terminal_value",
                "opponent_type",
                "position_hash_sequence",
                "cycle_actions",
            ]
            writer = csv.DictWriter(
                stream,
                fieldnames=fieldnames,
            )
            writer.writeheader()
            for source in result["games_detail"]:
                row = {
                    key: source.get(key)
                    for key in fieldnames
                    if key not in {
                        "first_cycle_ply",
                        "cycle_length",
                        "side_that_first_repeated",
                        "value_at_cycle_start",
                        "terminal_value",
                        "opponent_type",
                        "position_hash_sequence",
                        "cycle_actions",
                    }
                }
                cycle = source.get("repetition_cycle") or {}
                for key in (
                    "first_cycle_ply",
                    "cycle_length",
                    "side_that_first_repeated",
                    "value_at_cycle_start",
                    "terminal_value",
                    "opponent_type",
                ):
                    row[key] = cycle.get(key)
                row["position_hash_sequence"] = json.dumps(
                    cycle.get("position_hash_sequence", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                row["cycle_actions"] = json.dumps(
                    cycle.get("cycle_actions", []),
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                writer.writerow(row)
        os.replace(temporary_csv, csv_path)
