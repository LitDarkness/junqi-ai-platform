import os
import sys
import unittest

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "build")
for path in (ROOT, BUILD):
    if path not in sys.path:
        sys.path.insert(0, path)

from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv
from junqi_rl.training_config import OpponentPoolConfig


class RolloutMaskCacheTests(unittest.TestCase):
    def test_cached_masks_match_backend_after_reset_and_step(self):
        # Random opponents keep this contract test quick and isolate mask
        # lifecycle from heuristic search performance.
        env = CppSelfPlayVecEnv(
            num_envs=8,
            opponent_config=OpponentPoolConfig(
                random_prob=1.0,
                latest_prob=0.0,
                history_prob=0.0,
                heuristic_prob=0.0,
            ),
        )
        _, infos = env.reset()
        returned = np.stack([info["action_mask"] for info in infos])
        np.testing.assert_array_equal(returned, env._learning_masks.astype(bool))

        actions = returned.argmax(axis=1).astype(np.int32)
        _, _, _, next_infos = env.step(actions)
        next_returned = np.stack([info["action_mask"] for info in next_infos])
        fresh = env._canonical_masks(env.learning_players).astype(bool)
        np.testing.assert_array_equal(next_returned, fresh)
        np.testing.assert_array_equal(env._learning_masks.astype(bool), fresh)
        env.close()


if __name__ == "__main__":
    unittest.main()
