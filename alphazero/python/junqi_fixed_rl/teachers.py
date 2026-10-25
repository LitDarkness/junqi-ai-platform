from __future__ import annotations

import importlib
import importlib.util
import math
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import TeacherConfig


def _load_candidate(path: Path, index: int) -> Any:
    name = f"junqi_fixed_teacher_{index}_{path.parent.name}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load heuristic candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, "select_action", None)):
        raise TypeError(f"heuristic candidate has no select_action: {path}")
    return module


_WORKER_CANDIDATES: dict[str, Any] = {}


def _external_select_worker(
    path_text: str, board: tuple[int, ...], legal: tuple[int, ...]
) -> int:
    """Run one alpha-beta teacher in a persistent worker process."""
    module = _WORKER_CANDIDATES.get(path_text)
    if module is None:
        path = Path(path_text)
        module = _load_candidate(path, len(_WORKER_CANDIDATES))
        _WORKER_CANDIDATES[path_text] = module
    return int(module.select_action(list(board), list(legal), 1, 0.0, 0.0))


class TeacherEnsemble:
    """Broad policy teacher: successor scores, budgets/temperatures and votes."""

    def __init__(
        self, config: TeacherConfig, *, project_root: Path, seed: int
    ) -> None:
        self.config = config
        self.rng = random.Random(seed)
        self.native = importlib.import_module("JunqiAZ")
        self.external: list[str] = []
        self._executor: ProcessPoolExecutor | None = None
        for index, configured in enumerate(config.external_candidates):
            path = Path(configured)
            if not path.is_absolute():
                path = (project_root / path).resolve()
            if not path.is_file():
                print(f"[fixed-rl] teacher unavailable: {path}", flush=True)
                continue
            self.external.append(str(path))

    @property
    def identities(self) -> list[str]:
        return list(self.external)

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=max(1, int(self.config.workers))
            )
        return self._executor

    @staticmethod
    def _stable_softmax(values: np.ndarray, temperature: float) -> np.ndarray:
        scaled = values.astype(np.float64) / max(float(temperature), 1e-4)
        scaled -= scaled.max(initial=0.0)
        probabilities = np.exp(np.clip(scaled, -60.0, 0.0))
        return probabilities / probabilities.sum()

    def successor_scores(self, state: Any, legal: np.ndarray) -> np.ndarray:
        mover = int(state.current_player())
        boards: list[list[int]] = []
        terminal_scores = np.full(len(legal), np.nan, dtype=np.float64)
        live_indices: list[int] = []
        for index, action in enumerate(legal.tolist()):
            child = state.clone()
            child.apply(int(action))
            if child.is_terminal():
                terminal_scores[index] = 100.0 * float(child.terminal_value(mover))
            else:
                boards.append(list(child.native_state.board()))
                live_indices.append(index)
        scores = terminal_scores
        if boards:
            next_values = np.asarray(
                self.native.heuristic_084_raw_batch(
                    np.asarray(boards, dtype=np.int32), 1
                ),
                dtype=np.float64,
            )
            # Child boards are canonical for the opponent; negate back to the
            # player choosing the root action.
            for index, value in zip(live_indices, next_values):
                scores[index] = -value / max(self.config.heuristic_value_scale, 1e-6)
        if not np.isfinite(scores).all():
            raise RuntimeError("teacher produced a non-finite successor score")
        return scores

    def policy_batch(
        self, states: Sequence[Any], historical: Sequence[bool]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(states) != len(historical):
            raise ValueError("states and historical flags must align")
        rows: list[dict[str, Any]] = []
        jobs: list[tuple[int, str, tuple[int, ...], tuple[int, ...]]] = []
        for row_index, (state, use_history) in enumerate(zip(states, historical)):
            legal = np.asarray(state.legal_actions(), dtype=np.int32)
            if len(legal) == 0:
                raise RuntimeError(
                    "teacher requested policy for a state without legal moves"
                )
            scores = self.successor_scores(state, legal)
            probabilities = np.mean(
                [
                    self._stable_softmax(scores, temperature)
                    for temperature in self.config.softmax_temperatures
                ],
                axis=0,
            )
            rows.append(
                {
                    "legal": legal,
                    "probabilities": probabilities,
                    "vote": np.zeros(len(legal), dtype=np.float64),
                }
            )
            candidates = self.external[1:] if use_history else self.external[:1]
            if not candidates and self.external:
                candidates = self.external
            board = tuple(int(value) for value in state.native_state.board())
            legal_tuple = tuple(int(value) for value in legal)
            for path in candidates:
                jobs.append((row_index, path, board, legal_tuple))

        if jobs:
            actions = self._pool().map(
                _external_select_worker,
                [job[1] for job in jobs],
                [job[2] for job in jobs],
                [job[3] for job in jobs],
                chunksize=1,
            )
            for (row_index, _, _, _), action in zip(jobs, actions):
                legal = rows[row_index]["legal"]
                positions = np.flatnonzero(legal == int(action))
                if len(positions) != 1:
                    raise RuntimeError(
                        f"external heuristic returned illegal action {action}"
                    )
                rows[row_index]["vote"][int(positions[0])] += 1.0

        result: list[tuple[np.ndarray, np.ndarray]] = []
        for row in rows:
            probabilities = row["probabilities"]
            vote = row["vote"]
            if vote.sum() > 0:
                vote /= vote.sum()
                weight = float(self.config.external_vote_weight)
                probabilities = (1.0 - weight) * probabilities + weight * vote
            uniform_mix = float(self.config.uniform_mix)
            probabilities = (
                (1.0 - uniform_mix) * probabilities
                + uniform_mix / float(len(probabilities))
            )
            probabilities /= probabilities.sum()
            result.append(
                (row["legal"], probabilities.astype(np.float32))
            )
        return result

    def policy(
        self, state: Any, *, historical: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.policy_batch([state], [historical])[0]

    def sample_action(
        self, state: Any, *, historical: bool = False
    ) -> tuple[int, np.ndarray, np.ndarray]:
        legal, probabilities = self.policy(state, historical=historical)
        action = self.rng.choices(
            legal.tolist(), weights=probabilities.tolist(), k=1
        )[0]
        return int(action), legal, probabilities

    def sample_actions(
        self, states: Sequence[Any], historical: Sequence[bool]
    ) -> list[tuple[int, np.ndarray, np.ndarray]]:
        result = []
        for legal, probabilities in self.policy_batch(states, historical):
            action = self.rng.choices(
                legal.tolist(), weights=probabilities.tolist(), k=1
            )[0]
            result.append((int(action), legal, probabilities))
        return result

    def external_actions(
        self, states: Sequence[Any], historical: bool
    ) -> list[int]:
        """Return exact deterministic 084/082 actions for independent evaluation."""
        candidates = self.external[1:] if historical else self.external[:1]
        if not candidates:
            if not self.external:
                raise RuntimeError("requested external heuristic is unavailable")
            candidates = self.external
        path = candidates[0]
        boards = [
            tuple(int(value) for value in state.native_state.board())
            for state in states
        ]
        legals = [
            tuple(int(value) for value in state.legal_actions())
            for state in states
        ]
        return list(
            self._pool().map(
                _external_select_worker,
                [path] * len(states),
                boards,
                legals,
                chunksize=1,
            )
        )

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None


def random_policy(state: Any) -> tuple[int, np.ndarray, np.ndarray]:
    legal = np.asarray(state.legal_actions(), dtype=np.int32)
    probabilities = np.full(len(legal), 1.0 / len(legal), dtype=np.float32)
    return int(random.choice(legal.tolist())), legal, probabilities


def aggressive_policy(state: Any) -> tuple[int, np.ndarray, np.ndarray]:
    legal = np.asarray(state.legal_actions(), dtype=np.int32)
    board = np.asarray(state.native_state.board(), dtype=np.int32)
    weights = np.ones(len(legal), dtype=np.float64)
    for index, action in enumerate(legal):
        source, target = divmod(int(action), 60)
        if board[target] < 0:
            weights[index] += 8.0 + math.log1p(abs(int(board[target])))
        source_row, target_row = source // 5, target // 5
        weights[index] += max(0, target_row - source_row) * 0.25
    probabilities = (weights / weights.sum()).astype(np.float32)
    action = random.choices(legal.tolist(), weights=weights.tolist(), k=1)[0]
    return int(action), legal, probabilities
