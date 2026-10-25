from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "build")]

import JunqiEnv
from JunqiEnv.junqi import PlayerId
from junqi_rl.bots import (
    HeuristicProductionBot,
    build_bot,
    choose_heuristic_action_v3_fast,
)


class HeuristicV3Tests(unittest.TestCase):
    def test_build_bot_and_default_use_production_v3(self):
        self.assertIsInstance(build_bot("heuristic_v3", torch.device("cpu")), HeuristicProductionBot)
        self.assertIsInstance(build_bot("heuristic", torch.device("cpu")), HeuristicProductionBot)

    def test_takes_flag_immediately(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[51] = 30
        board[56] = -3
        env.load_board_state(board, PlayerId.RED)
        legal = env.get_legal_actions(PlayerId.RED)
        action = choose_heuristic_action_v3_fast(board, legal, PlayerId.RED, 3.0, 15.0)
        self.assertEqual(action, 51 * 60 + 56)

    def test_stops_immediate_flag_capture(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[1] = 3
        board[7] = 40
        board[6] = -30
        board[58] = -3
        env.load_board_state(board, PlayerId.RED)
        legal = env.get_legal_actions(PlayerId.RED)
        defending_action = 7 * 60 + 6
        self.assertIn(defending_action, legal)
        action = choose_heuristic_action_v3_fast(board, legal, PlayerId.RED, 19.0, 17.0)
        self.assertEqual(action, defending_action)


if __name__ == "__main__":
    unittest.main()
