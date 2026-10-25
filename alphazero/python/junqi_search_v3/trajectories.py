from __future__ import annotations

import hashlib
import importlib.util
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.teachers import aggressive_policy

from .dataset import OracleSample, family_split
from .oracle import OracleHistory, SupplementWideQuiescenceOracle


@dataclass
class PendingState:
    state: Any
    history: OracleHistory
    trace: list[int]
    ply: int
    player: int
    important: bool


def _load_candidate(path: Path) -> Any:
    name = "open_junqi_search_v3_exact082"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load exact082: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _weighted_choice(rng: random.Random, weights: dict[str, float]) -> str:
    names = list(weights)
    return rng.choices(names, [weights[name] for name in names], k=1)[0]


def _branch_count(rng: random.Random, weights: dict[str, float]) -> int:
    values = list(weights)
    return int(rng.choices(values, [weights[value] for value in values], k=1)[0])


def _branch_targets(rng: random.Random, count: int) -> set[int]:
    if count == 0:
        return set()
    # Every perturbed family changes its opening prefix.  Additional branch
    # points remain explicitly mid/late so diversity is not opening-only.
    windows = [(4, 12), (40, 88), (96, 176)]
    return {
        rng.randint(low, high)
        for low, high in windows[: min(count, len(windows))]
    }


class TrajectoryGenerator:
    def __init__(
        self,
        *,
        root: Path,
        backend: Any,
        config: Any,
        score_scale: float,
    ) -> None:
        self.root = root
        self.backend = backend
        self.config = config
        self.oracle = SupplementWideQuiescenceOracle(
            score_scale=score_scale,
            max_depth=config.oracle.max_depth,
            shallow_fraction=config.oracle.shallow_fraction,
        )
        candidate = Path(config.data.exact082_candidate)
        if not candidate.is_absolute():
            candidate = (root / candidate).resolve()
        self.exact082 = _load_candidate(candidate)
        checkpoint = Path(config.data.weak_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = (root / checkpoint).resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.weak = PolicyValueNet(config.model).eval()
        self.weak.load_state_dict(payload["model"])

    def _oracle_policy(
        self,
        state: Any,
        history: OracleHistory,
        budget: int,
    ) -> dict[str, Any]:
        return self.oracle.evaluate(state, history, budget=budget)

    def _sample_branch(
        self,
        result: dict[str, Any],
        rng: random.Random,
        *,
        force_alternative: bool,
    ) -> int:
        rows = result["actions"][: min(5, len(result["actions"]))]
        best = float(rows[0]["q"])
        # Retain near-best moves first.  A floor of three avoids deterministic
        # branches when the Q scale is very sharp.
        near = [row for row in rows if best - float(row["q"]) <= 0.20]
        rows = near if len(near) >= min(3, len(rows)) else rows
        if force_alternative and len(rows) > 1:
            rows = rows[1:]
        temperature = max(1e-4, self.config.data.trajectory_temperature)
        logits = np.asarray([float(row["q"]) for row in rows]) / temperature
        logits -= logits.max()
        probability = np.exp(logits)
        probability /= probability.sum()
        return int(rng.choices(
            [int(row["action"]) for row in rows],
            weights=probability.tolist(),
            k=1,
        )[0])

    def _weak_action(self, state: Any) -> int:
        observation = torch.from_numpy(state.feature_planes()).unsqueeze(0)
        legal = np.asarray(state.legal_actions(), np.int64)
        mask = torch.zeros((1, 3600), dtype=torch.bool)
        mask[0, torch.from_numpy(legal)] = True
        repetition = torch.zeros((1, 3600), dtype=torch.uint8)
        repetition[0, torch.from_numpy(legal)] = torch.from_numpy(
            state.repetition_contexts()
        )
        with torch.inference_mode():
            logits, _ = self.weak(observation, mask, repetition)
        return int(legal[int(torch.argmax(logits[0, legal]).item())])

    def _opponent_action(
        self,
        opponent: str,
        state: Any,
        history: OracleHistory,
        rng: random.Random,
    ) -> int:
        legal = np.asarray(state.legal_actions(), np.int32)
        if opponent == "supplement_wide_quiescence":
            return int(
                self._oracle_policy(state, history, 500)["root_best_action"]
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
        if opponent == "random":
            return int(legal[rng.randrange(len(legal))])
        if opponent == "aggressive":
            # Isolate Python's process-global RNG used by the legacy helper.
            previous = random.getstate()
            random.setstate(rng.getstate())
            try:
                action, _, _ = aggressive_policy(state)
                rng.setstate(random.getstate())
                return int(action)
            finally:
                random.setstate(previous)
        if opponent == "frozen_weak_network":
            return self._weak_action(state)
        raise KeyError(opponent)

    @staticmethod
    def _state_is_important(
        state: Any, legal: np.ndarray, next_action: int | None
    ) -> bool:
        repetition = np.asarray(state.repetition_contexts(), np.uint8)
        if (
            int(state.native_state.repetition_count()) >= 2
            or np.any(repetition >= 2)
        ):
            return True
        if next_action is not None:
            _, target = divmod(int(next_action), 60)
            board = np.asarray(state.native_state.board(), np.int16)
            if board[target] != 0:
                return True
        return state.native_state.turn_count() >= 180

    def generate_family(
        self, family_id: int
    ) -> tuple[list[OracleSample], dict[str, Any], list[dict[str, Any]]]:
        seed = self.config.seed + family_id * 1_000_003
        rng = random.Random(seed)
        opponent = _weighted_choice(rng, self.config.data.opponents)
        oracle_side = 1 if family_id % 2 == 0 else -1
        branches_requested = _branch_count(rng, self.config.data.branch_counts)
        branch_targets = _branch_targets(rng, branches_requested)
        state = self.backend.new_state(seed)
        history = OracleHistory()
        trace: list[int] = []
        pending: list[PendingState] = []
        branch_records: list[dict[str, Any]] = []
        sample_stride = rng.randint(2, 4)
        while not state.is_terminal():
            history.observe(state)
            ply = int(state.native_state.turn_count())
            legal = np.asarray(state.legal_actions(), np.int32)
            if len(legal) == 0:
                break
            branch_here = ply in branch_targets
            repetition = np.asarray(state.repetition_contexts(), np.uint8)
            force_exit = bool(np.any(repetition > 0) and family_id % 2 == 0)
            oracle_turn = state.current_player() == oracle_side
            result = None
            if branch_here or force_exit or oracle_turn:
                result = self._oracle_policy(state, history, 500)
            if branch_here:
                action = self._sample_branch(
                    result,
                    rng,
                    force_alternative=(family_id % 5 != 0),
                )
                branch_records.append(
                    {
                        "ply": ply,
                        "action": action,
                        "window_target": ply,
                        "kind": "topk_oracle_perturbation",
                    }
                )
            elif force_exit:
                noncycle = [
                    row for row in result["actions"]
                    if not row["enters_repetition"]
                ]
                action = int(
                    (noncycle[0] if noncycle else result["actions"][0])["action"]
                )
                branch_records.append(
                    {
                        "ply": ply,
                        "action": action,
                        "kind": "forced_best_noncycle",
                    }
                )
            elif oracle_turn:
                action = int(result["root_best_action"])
            else:
                action = self._opponent_action(
                    opponent, state, history, rng
                )
            important = self._state_is_important(state, legal, action)
            # A perturbation state is deliberately not labelled.  The reached
            # child may be sampled on the next ply and is independently
            # re-evaluated by the Oracle.
            if not branch_here and (ply % sample_stride == 0 or important):
                pending.append(
                    PendingState(
                        state.clone(),
                        OracleHistory(list(history.positions)),
                        list(trace),
                        ply,
                        int(state.current_player()),
                        important,
                    )
                )
            state.apply(action)
            trace.append(action)

        terminal_reason = (
            str(state.terminal_reason()) if state.is_terminal() else "other"
        )
        # Always retain the last nonterminal predecessor when available.
        terminal_player_values: dict[int, float] = {}
        for player in (-1, 1):
            terminal_player_values[player] = (
                float(state.terminal_value(player)) if state.is_terminal() else 0.0
            )
        samples: list[OracleSample] = []
        split = family_split(self.config.seed, family_id)
        for item in pending:
            budget = (
                self.config.data.important_budget
                if item.important
                else self.config.data.normal_budget
            )
            label = self._oracle_policy(item.state, item.history, budget)
            actions = np.asarray(
                [row["action"] for row in label["actions"]], np.int32
            )
            native_legal = np.asarray(item.state.legal_actions(), np.int32)
            native_repetition = np.asarray(
                item.state.repetition_contexts(), np.uint8
            )
            repetition_by_action = dict(
                zip(native_legal.tolist(), native_repetition.tolist())
            )
            q_values = np.asarray(
                [row["q"] for row in label["actions"]], np.float32
            )
            outcome = terminal_player_values[item.player]
            wdl = np.asarray(
                [outcome < 0, outcome == 0, outcome > 0], np.float32
            )
            samples.append(
                OracleSample(
                    family_id=family_id,
                    split=split,
                    state_hash=int(item.state.native_state.key()),
                    ply=item.ply,
                    player=item.player,
                    opponent=opponent,
                    terminal_reason=terminal_reason,
                    observation=item.state.feature_planes(),
                    legal_actions=actions,
                    repetition_contexts=np.asarray(
                        [
                            repetition_by_action[int(action)]
                            for action in actions
                        ],
                        np.uint8,
                    ),
                    q_values=q_values,
                    wdl=wdl,
                    important=item.important,
                    action_trace=item.trace,
                )
            )
        prefix = trace[:16]
        return samples, {
            "family_id": family_id,
            "seed": seed,
            "split": split,
            "opponent": opponent,
            "oracle_actor_side": oracle_side,
            "branches_requested": branches_requested,
            "branches_realized": len(
                [row for row in branch_records if row["kind"].startswith("topk")]
            ),
            "plies": len(trace),
            "terminal_reason": terminal_reason,
            "prefix16_sha256": hashlib.sha256(
                np.asarray(prefix, np.int16).tobytes()
            ).hexdigest(),
            "samples": len(samples),
        }, branch_records


_PROCESS_GENERATOR: TrajectoryGenerator | None = None


def initialize_trajectory_worker(
    config_path: str, score_scale: float
) -> None:
    global _PROCESS_GENERATOR
    from .config import SearchV3Config
    from junqi_fixed_rl.layout import (
        FixedSameLayoutBackend,
        load_fixed_layout,
    )

    torch.set_num_threads(1)
    config_file = Path(config_path).resolve()
    root = config_file.parent.parent
    config = SearchV3Config.load(config_file)
    layout = load_fixed_layout(
        config.boundary.layout_set,
        config.boundary.layout_id,
        project_root=root,
    )
    _PROCESS_GENERATOR = TrajectoryGenerator(
        root=root,
        backend=FixedSameLayoutBackend(layout),
        config=config,
        score_scale=score_scale,
    )


def generate_trajectory_worker(
    family_id: int,
) -> tuple[list[OracleSample], dict[str, Any], list[dict[str, Any]]]:
    if _PROCESS_GENERATOR is None:
        raise RuntimeError("trajectory process worker was not initialized")
    return _PROCESS_GENERATOR.generate_family(family_id)
