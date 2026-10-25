from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

import JunqiAZ
import JunqiOracle
from junqi_az.backend import CppAZStateAdapter
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout
from junqi_search_v3.oracle import (
    OracleHistory,
    SupplementWideQuiescenceOracle,
    oracle_transposition_key,
)


ROOT = Path(__file__).resolve().parents[1]


class SearchV3OracleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        layout = load_fixed_layout(
            ROOT / "config/layouts/fixed_layout_v1.json",
            "public_conservative_advance",
            project_root=ROOT,
        )
        cls.backend = FixedSameLayoutBackend(layout)

    def test_position_key_is_stable_and_side_sensitive(self) -> None:
        state = self.backend.new_state(7)
        board = list(state.native_state.board())
        self.assertEqual(
            JunqiOracle.position_key(board, 1),
            JunqiOracle.position_key(board, 1),
        )
        self.assertNotEqual(
            JunqiOracle.position_key(board, 1),
            JunqiOracle.position_key(board, -1),
        )

    def test_forced_root_action_is_respected(self) -> None:
        state = self.backend.new_state(11)
        history = OracleHistory()
        history.observe(state)
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        for action in (int(legal[0]), int(legal[-1])):
            payload = SupplementWideQuiescenceOracle.raw_once(
                state,
                np.asarray([action], dtype=np.int32),
                history,
                500,
                7,
            )
            self.assertEqual(int(payload["action"]), action)

    def test_two_stage_scores_every_legal_action_with_total_budget(self) -> None:
        state = self.backend.new_state(13)
        history = OracleHistory()
        history.observe(state)
        oracle = SupplementWideQuiescenceOracle(score_scale=100.0)
        result = oracle.evaluate(state, history, budget=2000)
        legal = set(int(action) for action in state.legal_actions())
        self.assertEqual({row["action"] for row in result["actions"]}, legal)
        self.assertLessEqual(result["nodes_allocated"], 2000)
        self.assertTrue(all(np.isfinite(row["q"]) for row in result["actions"]))
        best = result["actions"][0]
        self.assertGreaterEqual(len(best["pv"]), 2)
        cursor = state.clone()
        for action in best["pv"]:
            self.assertIn(action, cursor.legal_actions())
            cursor.apply(action)
            if cursor.is_terminal():
                break

    def test_black_canonical_view_is_searchable(self) -> None:
        state = self.backend.new_state(17)
        history = OracleHistory()
        history.observe(state)
        state.apply(int(state.legal_actions()[0]))
        history.observe(state)
        self.assertEqual(state.current_player(), -1)
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        payload = SupplementWideQuiescenceOracle.raw_once(
            state, legal[:1], history, 500, 7
        )
        self.assertEqual(int(payload["action"]), int(legal[0]))

    def test_capture_flag_terminal_value_and_reason(self) -> None:
        raw = JunqiAZ.JunqiAZState()
        board = [0] * 60
        board[1] = 3
        board[51] = 11
        board[56] = -3
        raw.load_board_state(board, 1)
        state = CppAZStateAdapter(raw)
        action = 51 * 60 + 56
        self.assertIn(action, set(int(value) for value in state.legal_actions()))
        history = OracleHistory()
        history.observe(state)
        oracle = SupplementWideQuiescenceOracle(score_scale=100.0)
        result = oracle.evaluate(state, history, budget=500)
        row = next(row for row in result["actions"] if row["action"] == action)
        self.assertEqual(row["q"], 1.0)
        self.assertEqual(row["terminal_reason"], "FLAG_CAPTURE")

    def test_repetition_history_uses_child_position_key(self) -> None:
        state = self.backend.new_state(19)
        action = int(state.legal_actions()[0])
        child = state.clone()
        child.apply(action)
        history = OracleHistory()
        history.observe(state)
        key_before = oracle_transposition_key(state, history)
        history.observe(child)
        history.observe(child)
        self.assertNotEqual(
            key_before, oracle_transposition_key(state, history)
        )
        payload = history.payload(state)
        child_raw = tuple(
            -value for value in reversed(tuple(child.native_state.board()))
        )
        expected_key = JunqiOracle.position_key(list(child_raw), -1)
        counts = dict(payload["repetition_counts"])
        self.assertGreaterEqual(counts.get(expected_key, 0), 2)
        forced = SupplementWideQuiescenceOracle.raw_once(
            state,
            np.asarray([action], dtype=np.int32),
            history,
            2000,
            7,
        )
        selected = next(
            row
            for row in forced["candidates"]
            if int(row["action"]) == action
        )
        score = (
            float(selected["search_score"])
            if selected["has_search_score"]
            else float(selected["static_score"])
        )
        self.assertAlmostEqual(score, 0.0, places=7)


if __name__ == "__main__":
    unittest.main()
