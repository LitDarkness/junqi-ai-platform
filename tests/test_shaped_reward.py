import unittest

import numpy as np

from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv
from junqi_rl.training_config import RewardConfig


class _RewardBackend:
    def __init__(self):
        self.my_scores = np.asarray([15.0, 8.0], dtype=np.float32)
        self.opp_scores = np.asarray([10.0, 10.0], dtype=np.float32)

    def done_flags(self):
        return np.zeros(2, dtype=np.uint8)

    def winners(self):
        return np.zeros(2, dtype=np.int32)

    def turn_counts(self):
        return np.ones(2, dtype=np.int32)

    def material_scores(self, players):
        return self.my_scores if np.array_equal(players, np.asarray([1, 1])) else self.opp_scores


class ShapedRewardTests(unittest.TestCase):
    def test_material_delta_is_consumed_before_baseline_advances(self):
        env = CppSelfPlayVecEnv.__new__(CppSelfPlayVecEnv)
        env.num_envs = 2
        env.backend = _RewardBackend()
        env.learning_players = np.asarray([1, 1], dtype=np.int32)
        env.opponent_players = np.asarray([-1, -1], dtype=np.int32)
        env.prev_material_delta = np.asarray([1.0, -1.0], dtype=np.float32)
        env.reward_config = RewardConfig.preset("shaped")
        env.no_combat_steps = np.zeros(2, dtype=np.int32)
        env.max_steps = 400

        rewards, dones = env._compute_rewards()
        current_delta = np.asarray([5.0, -2.0], dtype=np.float32)
        expected = (current_delta - np.asarray([1.0, -1.0], dtype=np.float32)) * env.reward_config.material_scale
        expected += env.reward_config.step_penalty

        np.testing.assert_allclose(rewards, expected)
        np.testing.assert_allclose(env.prev_material_delta, current_delta)
        self.assertFalse(dones.any())


if __name__ == "__main__":
    unittest.main()
