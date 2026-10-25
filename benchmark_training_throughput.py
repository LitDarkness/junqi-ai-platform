"""Compare PPO rollout throughput without starting a long training run."""
from __future__ import annotations

import argparse
import copy
import json
import random
import time

import numpy as np
import torch

from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv
from junqi_rl.env.sharded_cpp_vec_env import ShardedCppSelfPlayVecEnv
from junqi_rl.model import JunqiActorCritic
from junqi_rl.ppo import PPOConfig, VecPPOTrainer
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig


def measure(args, opponent_device: str) -> dict:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opponent_config = OpponentPoolConfig(
            random_prob=args.random_prob,
            heuristic_prob=args.heuristic_prob,
            latest_prob=args.latest_prob,
            history_prob=args.history_prob,
        ).normalized()
    common = dict(
        num_envs=args.num_envs, pool_dir=args.pool_dir, max_steps=args.max_steps,
        reward_config=RewardConfig.preset("shaped"), opponent_config=opponent_config,
        opponent_device=torch.device(opponent_device),
    )
    if args.shards > 1:
        env = ShardedCppSelfPlayVecEnv(
            **common, num_shards=args.shards, seed=args.seed,
        )
    else:
        env = CppSelfPlayVecEnv(**common)
    model = JunqiActorCritic().to(device)
    trainer = VecPPOTrainer(env, model, device, PPOConfig(update_epochs=1, minibatch_size=2048))
    trainer.collect_rollout(4)
    samples = []
    for _ in range(args.repeats):
        started = time.perf_counter()
        trainer.collect_rollout(args.num_steps)
        elapsed = time.perf_counter() - started
        samples.append(args.num_envs * args.num_steps / elapsed)
    env.close()
    return {
        "opponent_device": opponent_device,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "shards": args.shards,
        "opponent_mix": {
            "random": opponent_config.random_prob,
            "heuristic": opponent_config.heuristic_prob,
            "latest": opponent_config.latest_prob,
            "history": opponent_config.history_prob,
        },
        "samples_steps_per_second": samples,
        "median_steps_per_second": float(np.median(samples)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--env-grid", type=int, nargs="+", help="Benchmark several environment counts.")
    parser.add_argument("--num-steps", type=int, default=64)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--random-prob", type=float, default=0.1)
    parser.add_argument("--heuristic-prob", type=float, default=0.3)
    parser.add_argument("--latest-prob", type=float, default=0.6)
    parser.add_argument("--history-prob", type=float, default=0.0)
    parser.add_argument("--output", default="reports/training_throughput.json")
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], help="Opponent inference devices to test.")
    args = parser.parse_args()
    devices = args.devices or ["cpu"]
    if args.devices is None and torch.cuda.is_available():
        devices.append("cuda")
    results = []
    for num_envs in (args.env_grid or [args.num_envs]):
        variant = copy.copy(args)
        variant.num_envs = num_envs
        results.extend(measure(variant, device) for device in devices)
    payload = {"results": results, "note": "Learning transitions/s; opponent mix is held constant."}
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    for row in results:
        print(f"opponent={row['opponent_device']}: {row['median_steps_per_second']:.1f} learning steps/s")


if __name__ == "__main__":
    main()
