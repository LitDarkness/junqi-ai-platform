from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "build")]

from junqi_rl.env.sharded_cpp_vec_env import ShardedCppSelfPlayVecEnv
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig


class ShardedEnvironmentTests(unittest.TestCase):
    def test_two_shards_reset_step_and_close(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as pool:
            env = ShardedCppSelfPlayVecEnv(
                num_envs=8,
                num_shards=2,
                pool_dir=pool,
                max_steps=80,
                reward_config=RewardConfig.preset("shaped"),
                opponent_config=OpponentPoolConfig(
                    random_prob=1.0,
                    latest_prob=0.0,
                    history_prob=0.0,
                    heuristic_prob=0.0,
                ),
                opponent_device=torch.device("cpu"),
                seed=17,
            )
            try:
                observations, infos = env.reset()
                self.assertEqual(observations.shape, (8, 14, 12, 5))
                self.assertEqual(len(infos), 8)
                actions = np.asarray([np.flatnonzero(info["action_mask"])[0] for info in infos])
                next_observations, rewards, dones, next_infos = env.step(actions)
                self.assertEqual(next_observations.shape, observations.shape)
                self.assertEqual(rewards.shape, (8,))
                self.assertEqual(dones.shape, (8,))
                self.assertEqual(len(next_infos), 8)
            finally:
                env.close()
            self.assertTrue(all(not process.is_alive() for process in env.processes))


if __name__ == "__main__":
    unittest.main()
