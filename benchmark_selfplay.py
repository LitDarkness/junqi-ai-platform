from __future__ import annotations

import argparse
import time

import numpy as np

from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch self-play env throughput.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--reward-preset", type=str, choices=["shaped", "hybrid", "terminal"], default="shaped")
    parser.add_argument(
        "--opponent",
        choices=["random", "heuristic", "mixed"],
        default="heuristic",
        help="Use an explicit opponent mix so the benchmark does not depend on pool contents.",
    )
    args = parser.parse_args()

    if args.opponent == "random":
        opponent_config = OpponentPoolConfig(random_prob=1.0, latest_prob=0.0, history_prob=0.0, heuristic_prob=0.0)
    elif args.opponent == "heuristic":
        opponent_config = OpponentPoolConfig(random_prob=0.0, latest_prob=0.0, history_prob=0.0, heuristic_prob=1.0)
    else:
        opponent_config = OpponentPoolConfig(random_prob=0.5, latest_prob=0.0, history_prob=0.0, heuristic_prob=0.5)

    env = CppSelfPlayVecEnv(
        num_envs=args.num_envs,
        reward_config=RewardConfig.preset(args.reward_preset),
        opponent_config=opponent_config,
    )
    obs, infos = env.reset()

    start = time.perf_counter()
    transitions = 0
    for _ in range(args.steps):
        masks = np.stack([info["action_mask"] for info in infos]).astype(np.uint8)
        actions = masks.argmax(axis=1).astype(np.int32)
        obs, rewards, dones, infos = env.step(actions)
        transitions += args.num_envs
    elapsed = time.perf_counter() - start

    print(f"opponent={args.opponent}")
    print(f"selfplay_tps={transitions / elapsed:.1f}")
    print(f"seconds={elapsed:.3f}")


if __name__ == "__main__":
    main()
