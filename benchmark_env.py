from __future__ import annotations

import argparse
import time

import numpy as np

from junqi_rl.env.cpp_batch_env import CppBatchJunqiEnv
from junqi_rl.env.env_wrapper import JunqiPPOEnv


def benchmark_python_loop(num_envs: int, steps: int):
    envs = [JunqiPPOEnv() for _ in range(num_envs)]
    for env in envs:
        env.reset()

    start = time.perf_counter()
    transitions = 0
    for _ in range(steps):
        for env in envs:
            player = env.env.get_current_player()
            legal_actions = env.env.get_legal_actions(player)
            action = legal_actions[0]
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
            transitions += 1
    elapsed = time.perf_counter() - start
    return transitions / elapsed


def benchmark_cpp_batch(num_envs: int, steps: int):
    env = CppBatchJunqiEnv(num_envs)
    env.reset_random_all()

    start = time.perf_counter()
    transitions = 0
    for _ in range(steps):
        players = env.current_players()
        masks = env.legal_action_masks(players)
        actions = masks.argmax(axis=1).astype(np.int32)
        env.step_actions(actions)
        done = np.flatnonzero(env.done_flags())
        if done.size > 0:
            env.reset_indices(done)
        transitions += num_envs
    elapsed = time.perf_counter() - start
    return transitions / elapsed


def main():
    parser = argparse.ArgumentParser(description="Benchmark Junqi env throughput.")
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=128)
    args = parser.parse_args()

    python_tps = benchmark_python_loop(args.num_envs, args.steps)
    cpp_tps = benchmark_cpp_batch(args.num_envs, args.steps)

    print(f"python_loop_tps={python_tps:.1f}")
    print(f"cpp_batch_tps={cpp_tps:.1f}")
    print(f"speedup={cpp_tps / python_tps:.2f}x")


if __name__ == "__main__":
    main()
