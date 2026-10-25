from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "build")]

import JunqiEnv
from JunqiEnv.junqi import PlayerId
from junqi_rl import default_rule_profile
from junqi_rl.bots import CAMPS
from junqi_rl.model import JunqiActorCritic


class PlatformRegressionTests(unittest.TestCase):
    def test_rule_profile_is_stable(self):
        profile = default_rule_profile()
        self.assertEqual(profile.profile_id, "standard_open_v1")
        self.assertEqual(len(profile.sha256), 64)

    def test_open_observation_is_perfect_state(self):
        env = JunqiEnv.JunqiEnvironment()
        env.reset_random()
        self.assertEqual(env.get_observation(PlayerId.RED), env.get_perfect_state())
        self.assertEqual(env.get_observation(PlayerId.BLACK), env.get_perfect_state())

    def test_camp_topology_and_feature_plane_are_d2_symmetric(self):
        expected = {11, 13, 17, 21, 23, 36, 38, 42, 46, 48}
        self.assertEqual(CAMPS, expected)
        self.assertEqual({59 - position for position in CAMPS}, CAMPS)
        self.assertEqual(
            {row * 5 + (4 - col) for row, col in map(lambda p: divmod(p, 5), CAMPS)},
            CAMPS,
        )
        env = JunqiEnv.JunqiEnvironment()
        terrain = np.asarray(env.get_feature_planes(PlayerId.RED), dtype=np.float32).reshape(14, 60)[13]
        self.assertEqual(set(np.flatnonzero(terrain)), expected)

    def test_threefold_repetition(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[10] = 30
        board[49] = -30
        env.load_board_state(board, PlayerId.RED)
        for move in [(10, 15), (49, 44), (15, 10), (44, 49)] * 2:
            env.step(*move)
        self.assertTrue(env.is_game_over())
        self.assertEqual(env.get_winner(), PlayerId.NONE)
        self.assertEqual(env.get_terminal_reason(), "REPETITION")

    def test_flag_capture_counts_terminal_ply(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[51] = 30
        board[56] = -3
        env.load_board_state(board, PlayerId.RED)
        env.step(51, 56)
        self.assertEqual(env.get_winner(), PlayerId.RED)
        self.assertEqual(env.get_turn_count(), 1)
        self.assertEqual(env.get_terminal_reason(), "FLAG_CAPTURE")

    def test_cpp_legality_reason_does_not_mutate_state(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[10] = 1
        board[49] = -30
        env.load_board_state(board, PlayerId.RED)
        key_before = env.get_state_key()
        result = env.check_move(10, 15)
        self.assertFalse(result.legal)
        self.assertEqual(result.code, "PIECE_IMMOBILE")
        self.assertEqual(env.get_state_key(), key_before)
        self.assertEqual(env.get_turn_count(), 0)

    def test_cpp_legality_accepts_generated_move(self):
        env = JunqiEnv.JunqiEnvironment()
        board = [0] * 60
        board[10] = 30
        board[49] = -30
        env.load_board_state(board, PlayerId.RED)
        action = env.get_legal_actions(PlayerId.RED)[0]
        from_pos, to_pos = divmod(action, 60)
        result = env.check_move(from_pos, to_pos)
        self.assertTrue(result.legal)
        self.assertEqual(result.code, "OK")

    def test_factorized_policy_shape_and_size(self):
        model = JunqiActorCritic(channels=32, num_blocks=1)
        mask = torch.ones(2, 3600, dtype=torch.bool)
        logits, value = model(torch.zeros(2, 14, 12, 5), mask)
        self.assertEqual(tuple(logits.shape), (2, 3600))
        self.assertEqual(tuple(value.shape), (2, 1))
        policy_parameters = sum(
            p.numel() for name, p in model.named_parameters()
            if name.startswith(("from_", "to_"))
        )
        self.assertLess(policy_parameters, 10_000)


if __name__ == "__main__":
    unittest.main()
