from __future__ import annotations

import math
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import JunqiOracle


def _canonical_pv(actions: list[int]) -> list[int]:
    """Convert supplement's fixed root frame to per-ply canonical actions."""
    result = []
    for ply, action in enumerate(actions):
        value = int(action)
        if ply % 2 == 1:
            source, target = divmod(value, 60)
            value = (59 - source) * 60 + (59 - target)
        result.append(value)
    return result


def oracle_transposition_key(
    state: Any, history: "OracleHistory"
) -> str:
    payload = history.payload(state)
    encoded = json.dumps(
        {
            "position": int(
                JunqiOracle.position_key(
                    list(state.native_state.board()), 1
                )
            ),
            "history": payload,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.blake2b(encoded, digest_size=16).hexdigest()


def _canonical_to_raw(board: tuple[int, ...], side: int) -> tuple[int, ...]:
    return board if side == 1 else tuple(-value for value in reversed(board))


def _raw_to_frame(
    raw: tuple[int, ...], frame_side: int
) -> tuple[int, ...]:
    return raw if frame_side == 1 else tuple(-value for value in reversed(raw))


@dataclass
class OracleHistory:
    """Board history in absolute provenance, rendered into each root frame."""

    positions: list[tuple[tuple[int, ...], int]] = field(default_factory=list)

    def observe(self, state: Any) -> None:
        self.positions.append(
            (
                tuple(int(value) for value in state.native_state.board()),
                int(state.current_player()),
            )
        )

    def payload(self, state: Any) -> dict[str, Any]:
        root_side = int(state.current_player())
        counts: Counter[int] = Counter()
        for canonical, side in self.positions:
            raw = _canonical_to_raw(canonical, side)
            framed = _raw_to_frame(raw, root_side)
            framed_player = 1 if side == root_side else -1
            counts[int(JunqiOracle.position_key(list(framed), framed_player))] += 1
        return {
            "turn_count": int(state.native_state.turn_count()),
            "no_capture_ply": int(state.native_state.no_capture_ply()),
            "repetition_limit": 3,
            "no_capture_ply_limit": 200,
            "max_ply": 1000,
            "repetition_counts": sorted(counts.items()),
        }


class SupplementWideQuiescenceOracle:
    name = "supplement_wide_quiescence"

    def __init__(
        self,
        *,
        score_scale: float,
        max_depth: int = 7,
        shallow_fraction: float = 0.30,
    ) -> None:
        if score_scale <= 0 or not math.isfinite(score_scale):
            raise ValueError("score_scale must be finite and positive")
        self.score_scale = float(score_scale)
        self.max_depth = int(max_depth)
        self.shallow_fraction = float(shallow_fraction)

    @staticmethod
    def raw_once(
        state: Any,
        legal: np.ndarray,
        history: OracleHistory,
        budget: int,
        max_depth: int,
    ) -> dict[str, Any]:
        return dict(
            JunqiOracle.search_once(
                list(state.native_state.board()),
                [int(action) for action in legal],
                1,
                int(budget),
                int(max_depth),
                history.payload(state),
            )
        )

    @staticmethod
    def _candidate_scores(
        payload: dict[str, Any], legal: np.ndarray
    ) -> dict[int, float]:
        result = {}
        for row in payload["candidates"]:
            action = int(row["action"])
            if action not in legal:
                continue
            result[action] = float(
                row["search_score"]
                if row["has_search_score"]
                else row["static_score"]
            )
        return result

    def evaluate(
        self,
        state: Any,
        history: OracleHistory,
        *,
        budget: int,
    ) -> dict[str, Any]:
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        if len(legal) == 0:
            raise ValueError("Oracle root has no legal action")
        shallow_budget = max(
            len(legal) * 2,
            min(int(budget * self.shallow_fraction), budget - 1),
        )
        shallow_budget = min(shallow_budget, budget)
        shallow = self.raw_once(
            state, legal, history, shallow_budget, min(3, self.max_depth)
        )
        raw_scores = self._candidate_scores(shallow, legal)
        pv_by_action = {
            int(row["action"]): _canonical_pv(row["pv"])
            for row in shallow["candidates"]
            if int(row["action"]) in legal
        }
        allocations = {int(action): 0 for action in legal}
        per_action_nodes = {int(action): 0 for action in legal}
        score_weights = {
            int(action): float(shallow_budget) / float(len(legal))
            for action in legal
        }
        remaining = max(0, budget - shallow_budget)
        contenders = sorted(
            [int(action) for action in legal],
            key=lambda action: raw_scores.get(action, -math.inf),
            reverse=True,
        )
        rounds = max(1, int(math.ceil(math.log2(max(1, len(contenders))))))
        for round_index in range(rounds):
            if len(contenders) <= 1 or remaining <= 0:
                break
            rounds_left = rounds - round_index
            # Front-load the common-depth round.  Junqi has many root moves
            # whose shallow ordering reverses after one tactical reply; an
            # equal split across log2 rounds eliminated those moves before
            # they received a meaningful search.  Later rounds still satisfy
            # successive halving, but refine rather than define the shortlist.
            if rounds_left == 1:
                round_budget = remaining
            else:
                reserve = max(1, rounds_left - 1)
                round_budget = max(
                    len(contenders) * 2,
                    int(math.floor(remaining * 0.75)),
                )
                round_budget = min(round_budget, remaining - reserve)
            round_budget = min(round_budget, remaining)
            # Search the whole surviving set in one call.  This makes every
            # score in a halving round come from the same completed depth;
            # separate forced calls can complete at different depths and make
            # their raw alpha-beta scores incomparable.
            restricted = np.asarray(contenders, dtype=np.int32)
            searched = self.raw_once(
                state,
                restricted,
                history,
                round_budget,
                self.max_depth,
            )
            round_scores = self._candidate_scores(searched, restricted)
            for row in searched["candidates"]:
                action = int(row["action"])
                if action in contenders and row.get("pv"):
                    candidate_pv = _canonical_pv(row["pv"])
                    if len(candidate_pv) >= len(
                        pv_by_action.get(action, [])
                    ):
                        pv_by_action[action] = candidate_pv
            share, remainder = divmod(round_budget, len(contenders))
            node_share, node_remainder = divmod(
                int(searched["nodes"]), len(contenders)
            )
            for index, action in enumerate(contenders):
                action_allocation = share + int(index < remainder)
                allocations[action] += action_allocation
                per_action_nodes[action] += (
                    node_share + int(index < node_remainder)
                )
                round_scores.setdefault(action, raw_scores[action])
                old_weight = score_weights[action]
                new_weight = float(action_allocation)
                raw_scores[action] = (
                    raw_scores[action] * old_weight
                    + round_scores[action] * new_weight
                ) / max(1.0, old_weight + new_weight)
                score_weights[action] = old_weight + new_weight
            remaining -= round_budget
            contenders = sorted(
                contenders,
                key=lambda action: raw_scores[action],
                reverse=True,
            )[: max(1, (len(contenders) + 1) // 2)]

        repetition = np.asarray(state.repetition_contexts(), dtype=np.uint8)
        action_rows = []
        for index, action_value in enumerate(legal):
            action = int(action_value)
            child = state.clone()
            child.apply(action)
            terminal = bool(child.is_terminal())
            terminal_value = (
                float(child.terminal_value(state.current_player()))
                if terminal
                else None
            )
            raw = float(raw_scores[action])
            q = (
                float(terminal_value)
                if terminal
                else float(np.tanh(raw / self.score_scale))
            )
            action_rows.append(
                {
                    "action": action,
                    "raw_score": raw,
                    "q": q,
                    "pv": pv_by_action.get(action, [action]),
                    "pv_complete": terminal,
                    "nodes": per_action_nodes[action],
                    "nodes_allocated": allocations[action],
                    "terminal_reason": (
                        str(child.terminal_reason()) if terminal else "NONE"
                    ),
                    "enters_repetition": bool(repetition[index] >= 2),
                }
            )
        action_rows.sort(key=lambda row: (-row["q"], row["action"]))
        total_nodes = int(shallow["nodes"]) + sum(per_action_nodes.values())
        return {
            "oracle": self.name,
            "budget": int(budget),
            "nodes": total_nodes,
            "nodes_allocated": budget - remaining,
            "shallow_nodes": int(shallow["nodes"]),
            "score_scale": self.score_scale,
            "root_best_action": int(action_rows[0]["action"]),
            "actions": action_rows,
            "root_key": int(
                JunqiOracle.position_key(
                    list(state.native_state.board()), 1
                )
            ),
            "transposition_key": oracle_transposition_key(state, history),
        }
