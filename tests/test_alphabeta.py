import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "build")
for path in (ROOT, BUILD):
    if path not in sys.path:
        sys.path.insert(0, path)

import torch
from JunqiEnv.junqi import PlayerId

from junqi_rl.bots import AlphaBetaBot, build_bot
from junqi_rl.env.env_wrapper import JunqiPPOEnv


class AlphaBetaTests(unittest.TestCase):
    def test_factory_exposes_bounded_search_variants(self):
        self.assertIsInstance(build_bot("ab2", torch.device("cpu")), AlphaBetaBot)
        self.assertEqual(build_bot("ab2", torch.device("cpu")).max_depth, 2)
        self.assertEqual(build_bot("ab", torch.device("cpu")).max_depth, 3)

    def test_selected_action_is_legal_and_has_analysis(self):
        env = JunqiPPOEnv(max_steps=100)
        env.reset()
        player = env.env.get_current_player()
        legal = set(env.env.get_legal_actions(player))
        bot = AlphaBetaBot(max_depth=2, time_limit_ms=100, node_limit=20_000)
        policy_action = bot.select_action(env, player)
        self.assertIsNotNone(policy_action)
        action = env.action_to_env(policy_action, player)
        self.assertIn(action, legal)
        self.assertGreaterEqual(bot.last_analysis.depth, 1)
        self.assertGreater(bot.last_analysis.nps, 0)

    def test_immediate_flag_capture_is_forced(self):
        # Only the tactical fact matters here: red's movable piece can capture
        # the black flag at 56 in one move.
        board = [0] * 60
        board[1] = 3
        board[51] = 40
        board[56] = -3
        board[58] = -40
        bot = AlphaBetaBot(max_depth=3, time_limit_ms=250, node_limit=50_000)
        analysis = bot.analyze(board, PlayerId.RED)
        self.assertEqual(analysis.best_action, 51 * 60 + 56)
        self.assertGreater(analysis.score, 90_000)


if __name__ == "__main__":
    unittest.main()
