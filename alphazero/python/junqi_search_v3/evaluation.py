from __future__ import annotations

import importlib.util
import hashlib
import math
import random
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .oracle import OracleHistory, SupplementWideQuiescenceOracle
from .search import BatchedNeuralPUCT


@dataclass
class EvaluationGame:
    state: Any
    candidate_side: int
    red_agent: str
    black_agent: str
    history: OracleHistory = field(default_factory=OracleHistory)
    plies: int = 0
    actions: list[int] = field(default_factory=list)


def _load_exact082(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(
        "open_junqi_search_v3_eval_exact082", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_interval(scores: np.ndarray) -> tuple[float, float]:
    mean = float(scores.mean()) if len(scores) else 0.0
    if len(scores) < 2:
        return mean, mean
    standard_error = float(scores.std(ddof=1) / math.sqrt(len(scores)))
    return max(0.0, mean - 1.96 * standard_error), min(
        1.0, mean + 1.96 * standard_error
    )


class SearchEvaluator:
    def __init__(
        self,
        *,
        root: Path,
        backend: Any,
        config: Any,
        model: torch.nn.Module,
        score_scale: float,
    ) -> None:
        self.root = root
        self.backend = backend
        self.config = config
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.search = BatchedNeuralPUCT(
            model.to(self.device).eval(),
            device=self.device,
            cpuct=config.puct.cpuct,
            inference_batch_size=config.puct.inference_batch_size,
        )
        self.oracle = SupplementWideQuiescenceOracle(
            score_scale=score_scale,
            max_depth=config.oracle.max_depth,
            shallow_fraction=config.oracle.shallow_fraction,
        )
        exact_path = Path(config.data.exact082_candidate)
        if not exact_path.is_absolute():
            exact_path = (root / exact_path).resolve()
        self.exact082 = _load_exact082(exact_path)

    def _external_actions(
        self, games: list[EvaluationGame], opponent: str
    ) -> list[int]:
        def select(game: EvaluationGame) -> int:
            state = game.state
            legal = np.asarray(state.legal_actions(), np.int32)
            if opponent == "supplement_wide_quiescence":
                return int(
                    self.oracle.evaluate(
                        state, game.history, budget=500
                    )["root_best_action"]
                )
            if opponent == "exact082":
                return int(
                    self.exact082.select_action(
                        list(state.native_state.board()),
                        legal.tolist(),
                        1,
                        0.0,
                        0.0,
                    )
                )
            raise KeyError(opponent)

        with ThreadPoolExecutor(
            max_workers=min(
                int(
                    getattr(
                        self.config,
                        "runtime_workers",
                        self.config.data.workers,
                    )
                ),
                len(games),
            )
        ) as executor:
            return list(executor.map(select, games))

    @staticmethod
    def _state_signature(game: EvaluationGame) -> tuple[int, bytes]:
        # Native full_state_key includes board, side, draw counters and the
        # complete repetition-count table.  Recent-move planes are additionally
        # hashed because two transpositions with the same count table can still
        # expose different five-ply network history.
        features = np.ascontiguousarray(
            game.state.feature_planes(), dtype=np.float32
        )
        return (
            int(game.state.native_state.key()),
            hashlib.blake2b(features.tobytes(), digest_size=16).digest(),
        )

    def games(
        self,
        red_agent: str,
        black_agent: str,
        count: int,
        *,
        seed: int,
    ) -> dict[str, Any]:
        active = []
        for game_index in range(count):
            candidate_side = 1 if red_agent.startswith("candidate") else -1
            active.append(
                EvaluationGame(
                    state=self.backend.new_state(seed + game_index // 2),
                    candidate_side=candidate_side,
                    red_agent=red_agent,
                    black_agent=black_agent,
                )
            )
        completed: list[EvaluationGame] = []
        search_metrics = Counter()
        started = time.perf_counter()
        rounds = 0
        while active:
            rounds += 1
            for game in active:
                game.history.observe(game.state)
            groups: dict[str, list[EvaluationGame]] = {}
            for game in active:
                agent = (
                    game.red_agent
                    if game.state.current_player() == 1
                    else game.black_agent
                )
                groups.setdefault(agent, []).append(game)
            actions_by_game: dict[int, int] = {}
            for agent, games in groups.items():
                representatives: dict[
                    tuple[int, bytes], EvaluationGame
                ] = {}
                signatures: dict[int, tuple[int, bytes]] = {}
                for game in games:
                    signature = self._state_signature(game)
                    signatures[id(game)] = signature
                    representatives.setdefault(signature, game)
                unique_games = list(representatives.values())
                if agent in {
                    "supplement_wide_quiescence",
                    "exact082",
                }:
                    actions = self._external_actions(unique_games, agent)
                else:
                    simulations = int(agent.rsplit("_", 1)[-1])
                    leaf_value_mode = (
                        "zero"
                        if "_zero_" in agent
                        else "network"
                    )
                    if simulations == 0:
                        results = self.search.bare(
                            [game.state for game in unique_games]
                        )
                        metrics = {}
                    else:
                        results, metrics = self.search.search(
                            [game.state for game in unique_games],
                            simulations,
                            leaf_value_mode=leaf_value_mode,
                        )
                    actions = [result.action for result in results]
                    for key, value in metrics.items():
                        if isinstance(value, (int, float)):
                            search_metrics[key] += value
                action_by_signature = dict(
                    zip(representatives.keys(), actions)
                )
                search_metrics["logical_roots"] += len(games)
                search_metrics["unique_roots"] += len(unique_games)
                search_metrics["deduplicated_roots"] += (
                    len(games) - len(unique_games)
                )
                for game in games:
                    actions_by_game[id(game)] = int(
                        action_by_signature[signatures[id(game)]]
                    )
            next_active = []
            for game in active:
                action = actions_by_game[id(game)]
                game.actions.append(action)
                game.state.apply(action)
                game.plies += 1
                if game.state.is_terminal():
                    completed.append(game)
                else:
                    next_active.append(game)
            active = next_active
            if rounds % 50 == 0:
                print(
                    f"[search-v3 arena] red={red_agent} "
                    f"black={black_agent} ply={rounds} "
                    f"active={len(active)}/{count} "
                    f"unique_roots={int(search_metrics['unique_roots'])} "
                    f"logical_roots={int(search_metrics['logical_roots'])}",
                    flush=True,
                )
        scores = []
        reasons = Counter()
        wins = draws = losses = 0
        for game in completed:
            value = float(game.state.terminal_value(game.candidate_side))
            if value > 0:
                wins += 1
                scores.append(1.0)
            elif value < 0:
                losses += 1
                scores.append(0.0)
            else:
                draws += 1
                scores.append(0.5)
            reasons[str(game.state.terminal_reason())] += 1
        array = np.asarray(scores, np.float64)
        lower, upper = score_interval(array)
        trajectory_counts = Counter(
            hashlib.sha256(
                np.asarray(game.actions, np.int16).tobytes()
            ).hexdigest()
            for game in completed
        )
        return {
            "red_agent": red_agent,
            "black_agent": black_agent,
            "games": len(completed),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": float(array.mean()),
            "ci95": [lower, upper],
            "average_ply": float(
                np.mean([game.plies for game in completed])
            ),
            "terminal_reasons": dict(reasons),
            "repetition_fraction": (
                reasons.get("REPETITION", 0) / max(1, len(completed))
            ),
            "capture_flag_fraction": (
                reasons.get("FLAG_CAPTURE", 0) / max(1, len(completed))
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "unique_trajectories": len(trajectory_counts),
            "maximum_trajectory_share": (
                max(trajectory_counts.values(), default=0)
                / max(1, len(completed))
            ),
            "search_metrics": dict(search_metrics),
        }

    def paired_candidate_vs(
        self, candidate: str, opponent: str, games: int, seed: int
    ) -> dict[str, Any]:
        half = games // 2
        candidate_agent = (
            candidate
            if candidate.startswith("candidate_")
            else f"candidate_{candidate.rsplit('_',1)[-1]}"
        )
        red = self.games(
            candidate_agent,
            opponent,
            half,
            seed=seed,
        )
        black = self.games(
            opponent,
            candidate_agent,
            games - half,
            seed=seed,
        )
        wins = red["wins"] + black["wins"]
        draws = red["draws"] + black["draws"]
        losses = red["losses"] + black["losses"]
        score_values = np.asarray(
            [1.0] * wins + [0.5] * draws + [0.0] * losses
        )
        lower, upper = score_interval(score_values)
        reasons = Counter(red["terminal_reasons"])
        reasons.update(black["terminal_reasons"])
        return {
            "candidate": candidate,
            "opponent": opponent,
            "games": games,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "score": float(score_values.mean()),
            "ci95": [lower, upper],
            "terminal_reasons": dict(reasons),
            "repetition_fraction": reasons.get("REPETITION", 0) / games,
            "capture_flag_fraction": reasons.get("FLAG_CAPTURE", 0) / games,
            "average_ply": (
                red["average_ply"] * half
                + black["average_ply"] * (games - half)
            ) / games,
            "elapsed_seconds": (
                red["elapsed_seconds"] + black["elapsed_seconds"]
            ),
            "unique_trajectories": (
                red["unique_trajectories"] + black["unique_trajectories"]
            ),
            "deterministic_pseudoreplication_warning": (
                red["unique_trajectories"] + black["unique_trajectories"]
                < games
            ),
            "logical_games": games,
        }
