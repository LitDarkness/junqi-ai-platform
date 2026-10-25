from __future__ import annotations

import argparse
import csv
import json
import os
import time
import glob
import random
from collections import deque
from typing import Optional

def make_env(pool_dir: str, max_steps: int, reward_config: RewardConfig, opponent_config: OpponentPoolConfig, opponent_device: torch.device):
    def _factory():
        return JunqiSelfPlayEnv(
            pool_dir=pool_dir,
            max_steps=max_steps,
            reward_config=reward_config,
            opponent_config=opponent_config,
            opponent_device=opponent_device,
        )
    return _factory


def resolve_device(spec: str) -> torch.device:
    normalized = spec.lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def save_latest(model: torch.nn.Module, pool_dir: str):
    os.makedirs(pool_dir, exist_ok=True)
    latest_path = os.path.join(pool_dir, "latest.pth")
    save_model_state(model.state_dict(), latest_path)


def save_model_state(state_dict: dict, target_path: str):
    temp_path = f"{target_path}.tmp"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.save(state_dict, temp_path)
    os.replace(temp_path, target_path)


def save_model(model: torch.nn.Module, target_path: str):
    save_model_state(model.state_dict(), target_path)

def prune_model_pool(pool_dir: str, max_history: int) -> None:
    """Bound disk use while retaining the newest diverse opponents."""
    paths = glob.glob(os.path.join(pool_dir, "model_*.pth"))
    paths.sort(key=os.path.getmtime, reverse=True)
    for stale_path in paths[max(1, max_history):]:
        os.remove(stale_path)




def resolve_eval_opponent(eval_red: str, eval_black: str, latest_spec: str) -> Optional[str]:
    if eval_red == "latest" and eval_black != "latest":
        return latest_spec if eval_black == "latest" else eval_black
    if eval_black == "latest" and eval_red != "latest":
        return latest_spec if eval_red == "latest" else eval_red
    return None


def safe_score(wins: int, draws: int, games: int) -> float:
    return (wins + 0.5 * draws) / max(1, games)


def load_existing_score(path: str, fallback: float = float("-inf")) -> float:
    if not os.path.exists(path):
        return fallback
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return fallback
    return float(payload.get("score", fallback))


def save_score_metadata(path: str, payload: dict):
    temp_path = f"{path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temp_path, path)


def initialize_champion_score(
    champion_path: str,
    benchmark_opponent: Optional[str],
    games: int,
    max_steps: int,
    device: torch.device,
    reward_config: RewardConfig,
) -> float:
    if not os.path.exists(champion_path) or not benchmark_opponent:
        return float("-inf")
    _, _, result, _ = evaluate_symmetric_specs(
        primary_spec=champion_path,
        opponent_spec=benchmark_opponent,
        games=games,
        max_steps=max_steps,
        device=device,
        reward_config=reward_config,
    )
    return result.primary_score


def append_metrics(save_dir: str, row: dict):
    csv_path = os.path.join(save_dir, "metrics.csv")
    jsonl_path = os.path.join(save_dir, "metrics.jsonl")
    csv_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not csv_exists:
            writer.writeheader()
        writer.writerow(row)
    with open(jsonl_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    # Keep TensorBoard/TensorFlow out of Windows spawn workers.  Importing it
    # at module scope multiplied the private commit charge for every rollout
    # process even though only the trainer writes summaries.
    global np, torch, tqdm
    global evaluate_specs, evaluate_symmetric_specs
    global CppSelfPlayVecEnv, ShardedCppSelfPlayVecEnv, JunqiSelfPlayEnv, SubprocVecEnv
    global JunqiActorCritic, PPOConfig, VecPPOTrainer, OpponentPoolConfig, RewardConfig

    import numpy as np
    import torch
    from tqdm import tqdm
    from torch.utils.tensorboard import SummaryWriter

    from evaluate import evaluate_specs, evaluate_symmetric_specs
    from junqi_rl.env.cpp_selfplay_vec_env import CppSelfPlayVecEnv
    from junqi_rl.env.sharded_cpp_vec_env import ShardedCppSelfPlayVecEnv
    from junqi_rl.env.pool_env import JunqiSelfPlayEnv
    from junqi_rl.env.vec_env import SubprocVecEnv
    from junqi_rl.model import JunqiActorCritic
    from junqi_rl.ppo import PPOConfig, VecPPOTrainer
    from junqi_rl.training_config import OpponentPoolConfig, RewardConfig

    parser = argparse.ArgumentParser(description="Train PPO for open-information Junqi self-play.")
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=256)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--pool-dir", type=str, default="pool")
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--load-model", type=str, default=None)
    parser.add_argument("--seed", type=int, default=20260712)
    parser.add_argument("--max-history-models", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--torch-threads", type=int, default=1, help="Set PyTorch CPU thread count. Use 0 to keep framework default.")
    parser.add_argument("--opponent-device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--ppo-minibatch-size", type=int, default=1024)
    parser.add_argument("--ppo-update-epochs", type=int, default=4)
    parser.add_argument("--backend", type=str, choices=["python", "cpp-batch", "cpp-sharded"], default="cpp-batch")
    parser.add_argument("--rollout-shards", type=int, default=1, help="Worker processes for cpp-sharded rollout.")
    parser.add_argument("--reward-preset", type=str, choices=["shaped", "hybrid", "terminal"], default="shaped")
    parser.add_argument("--draw-reward", type=float, default=-2.0)
    parser.add_argument("--no-combat-limit", type=int, default=40, help="Consecutive MOVE_ONLY results before forcing a draw. Set 0 to disable.")
    parser.add_argument("--no-combat-draw-reward", type=float, default=-2.0)
    parser.add_argument("--turn-limit-mode", type=str, choices=["draw", "material"], default="draw")
    parser.add_argument("--opponent-latest-prob", type=float, default=0.4)
    parser.add_argument("--opponent-history-prob", type=float, default=0.4)
    parser.add_argument("--opponent-heuristic-prob", type=float, default=0.2)
    parser.add_argument("--opponent-random-prob", type=float, default=0.0)
    parser.add_argument("--eval-freq", type=int, default=50)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--eval-max-steps", type=int, default=200)
    parser.add_argument("--eval-red", type=str, default="latest", help="latest | random | greedy | path/to/model.pth")
    parser.add_argument("--eval-black", type=str, default="heuristic_v3", help="latest | random | greedy | heuristic_v1/v2/v3 | heuristic | path/to/model.pth")
    parser.add_argument("--eval-symmetric", action="store_true", default=True, help="Evaluate the candidate from both colors against the non-latest opponent.")
    parser.add_argument("--no-eval-symmetric", dest="eval_symmetric", action="store_false")
    parser.add_argument("--champion-games", type=int, default=40, help="Per-color games for challenger vs champion promotion matches.")
    parser.add_argument("--champion-gate", type=float, default=0.01, help="Minimum benchmark score improvement before challenging the champion.")
    parser.add_argument("--champion-margin", type=float, default=0.02, help="Required symmetric score margin over 50%% to promote a challenger.")
    parser.add_argument(
        "--champion-benchmark-slack",
        type=float,
        default=0.02,
        help="Maximum allowed benchmark score drop versus the current champion when promoting a head-to-head winner.",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.makedirs(args.pool_dir, exist_ok=True)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opponent_device = resolve_device(args.opponent_device)
    reward_config = RewardConfig.preset(args.reward_preset).with_overrides(
        draw_reward=args.draw_reward,
        no_combat_limit=args.no_combat_limit,
        no_combat_draw_reward=args.no_combat_draw_reward,
        turn_limit_mode=args.turn_limit_mode,
    )
    opponent_config = OpponentPoolConfig(
        random_prob=args.opponent_random_prob,
        latest_prob=args.opponent_latest_prob,
        history_prob=args.opponent_history_prob,
        heuristic_prob=args.opponent_heuristic_prob,
        max_history_models=args.max_history_models,
    ).normalized()
    if args.backend == "cpp-sharded":
        vec_env = ShardedCppSelfPlayVecEnv(
            num_envs=args.num_envs,
            num_shards=args.rollout_shards,
            pool_dir=args.pool_dir,
            max_steps=args.max_steps,
            reward_config=reward_config,
            opponent_config=opponent_config,
            opponent_device=opponent_device,
            seed=args.seed,
        )
    elif args.backend == "cpp-batch":
        vec_env = CppSelfPlayVecEnv(
            num_envs=args.num_envs,
            pool_dir=args.pool_dir,
            max_steps=args.max_steps,
            reward_config=reward_config,
            opponent_config=opponent_config,
            opponent_device=opponent_device,
        )
    else:
        env_fns = [make_env(args.pool_dir, args.max_steps, reward_config, opponent_config, opponent_device) for _ in range(args.num_envs)]
        vec_env = SubprocVecEnv(env_fns)
    model = JunqiActorCritic().to(device)

    if args.load_model:
        model.load_state_dict(torch.load(args.load_model, map_location=device))

    # Seed the opponent pool immediately so early self-play does not fall back to heuristic-only
    # just because `latest.pth` has not been written yet.
    save_latest(model, args.pool_dir)

    trainer = VecPPOTrainer(
        vec_env=vec_env,
        model=model,
        device=device,
        config=PPOConfig(
            learning_rate=args.lr,
            minibatch_size=args.ppo_minibatch_size,
            update_epochs=args.ppo_update_epochs,
        ),
    )

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    recent_returns = deque(maxlen=100)
    recent_lengths = deque(maxlen=100)
    benchmark_best_score = load_existing_score(os.path.join(args.save_dir, "benchmark_best_metrics.json"))
    latest_eval_target = os.path.join(args.pool_dir, "latest.pth")
    candidate_eval_target = os.path.join(args.save_dir, "candidate_eval.pth")
    champion_path = os.path.join(args.save_dir, "champion_model.pth")
    champion_metrics_path = os.path.join(args.save_dir, "champion_metrics.json")
    benchmark_best_path = os.path.join(args.save_dir, "benchmark_best_model.pth")
    best_model_path = os.path.join(args.save_dir, "best_model.pth")
    benchmark_opponent_spec = resolve_eval_opponent(args.eval_red, args.eval_black, latest_eval_target) if args.eval_symmetric else None
    champion_benchmark_score = load_existing_score(champion_metrics_path)
    if champion_benchmark_score == float("-inf"):
        champion_benchmark_score = initialize_champion_score(
            champion_path=champion_path,
            benchmark_opponent=benchmark_opponent_spec,
            games=args.eval_games,
            max_steps=args.eval_max_steps,
            device=device,
            reward_config=reward_config,
        )
    pbar = tqdm(range(1, args.iters + 1), desc="training", unit="iter")

    try:
        for iteration in pbar:
            rollout_start = time.perf_counter()
            last_values, ep_returns, ep_lengths = trainer.collect_rollout(args.num_steps)
            rollout_seconds = time.perf_counter() - rollout_start
            recent_returns.extend(ep_returns)
            recent_lengths.extend(ep_lengths)

            advantage_start = time.perf_counter()
            returns, advantages = trainer.compute_advantages(last_values)
            update_start = time.perf_counter()
            actor_loss, critic_loss, entropy = trainer.update(returns, advantages)
            update_seconds = time.perf_counter() - update_start
            advantage_seconds = update_start - advantage_start

            avg_return = float(np.mean(recent_returns)) if recent_returns else 0.0
            avg_length = float(np.mean(recent_lengths)) if recent_lengths else 0.0
            transitions = args.num_envs * args.num_steps
            rollout_tps = transitions / max(rollout_seconds, 1e-9)
            update_sps = transitions / max(update_seconds, 1e-9)

            writer.add_scalar("rollout/episode_return", avg_return, iteration)
            writer.add_scalar("rollout/episode_length", avg_length, iteration)
            writer.add_scalar("timing/rollout_seconds", rollout_seconds, iteration)
            writer.add_scalar("timing/advantage_seconds", advantage_seconds, iteration)
            writer.add_scalar("timing/update_seconds", update_seconds, iteration)
            writer.add_scalar("timing/rollout_tps", rollout_tps, iteration)
            writer.add_scalar("timing/update_samples_per_sec", update_sps, iteration)
            writer.add_scalar("loss/actor", actor_loss, iteration)
            writer.add_scalar("loss/critic", critic_loss, iteration)
            writer.add_scalar("loss/entropy", entropy, iteration)
            pbar.set_postfix(
                ep_ret=f"{avg_return:.2f}",
                ep_len=f"{avg_length:.1f}",
                actor=f"{actor_loss:.3f}",
                critic=f"{critic_loss:.3f}",
                roll=f"{rollout_seconds:.2f}s",
                upd=f"{update_seconds:.2f}s",
            )

            if iteration % 5 == 0:
                save_latest(model, args.pool_dir)

            if iteration % args.save_freq == 0:
                ckpt_path = os.path.join(args.save_dir, f"model_{iteration}.pth")
                pool_path = os.path.join(args.pool_dir, f"model_{iteration}.pth")
                save_model(model, ckpt_path)
                save_model(model, pool_path)
                prune_model_pool(args.pool_dir, opponent_config.max_history_models)

            row = {
                "iteration": iteration,
                "avg_return": avg_return,
                "avg_length": avg_length,
                "actor_loss": actor_loss,
                "critic_loss": critic_loss,
                "entropy": entropy,
                "rollout_seconds": rollout_seconds,
                "advantage_seconds": advantage_seconds,
                "update_seconds": update_seconds,
                "rollout_tps": rollout_tps,
                "update_samples_per_sec": update_sps,
                "torch_threads": torch.get_num_threads(),
                "device": str(device),
                "opponent_device": str(opponent_device),
                "ppo_minibatch_size": args.ppo_minibatch_size,
                "ppo_update_epochs": args.ppo_update_epochs,
                "draw_reward": args.draw_reward,
                "no_combat_limit": args.no_combat_limit,
                "no_combat_draw_reward": args.no_combat_draw_reward,
                "turn_limit_mode": args.turn_limit_mode,
                "opponent_latest_prob": opponent_config.latest_prob,
                "opponent_history_prob": opponent_config.history_prob,
                "opponent_heuristic_prob": opponent_config.heuristic_prob,
                "opponent_random_prob": opponent_config.random_prob,
                "backend": args.backend,
                "rollout_shards": args.rollout_shards,
                "reward_preset": args.reward_preset,
            }

            if args.eval_freq > 0 and iteration % args.eval_freq == 0:
                save_latest(model, args.pool_dir)
                save_model(model, candidate_eval_target)
                if args.eval_symmetric and benchmark_opponent_spec is not None:
                    eval_primary_bot, eval_opponent_bot, eval_result, eval_seconds = evaluate_symmetric_specs(
                        primary_spec=candidate_eval_target,
                        opponent_spec=benchmark_opponent_spec,
                        games=args.eval_games,
                        max_steps=args.eval_max_steps,
                        device=device,
                        reward_config=reward_config,
                    )
                    score = eval_result.primary_score
                    row.update({
                        "eval_mode": "symmetric",
                        "eval_primary": eval_primary_bot.name,
                        "eval_opponent": eval_opponent_bot.name,
                        "eval_games": eval_result.games,
                        "eval_primary_wins": eval_result.primary_wins,
                        "eval_opponent_wins": eval_result.opponent_wins,
                        "eval_draws": eval_result.draws,
                        "eval_primary_win_rate": eval_result.primary_win_rate,
                        "eval_primary_score": score,
                        "eval_avg_turns": eval_result.avg_turns,
                        "eval_as_red_wins": eval_result.as_red.red_wins,
                        "eval_as_red_losses": eval_result.as_red.black_wins,
                        "eval_as_red_draws": eval_result.as_red.draws,
                        "eval_as_black_wins": eval_result.as_black.black_wins,
                        "eval_as_black_losses": eval_result.as_black.red_wins,
                        "eval_as_black_draws": eval_result.as_black.draws,
                        "eval_seconds": eval_seconds,
                    })
                    writer.add_scalar("eval/primary_win_rate", eval_result.primary_win_rate, iteration)
                    writer.add_scalar("eval/primary_score", score, iteration)
                    writer.add_scalar("eval/avg_turns", eval_result.avg_turns, iteration)
                    pbar.write(
                        f"[eval {iteration}] {eval_primary_bot.name} vs {eval_opponent_bot.name} "
                        f"score={score:.3f} win_rate={eval_result.primary_win_rate:.3f} "
                        f"draws={eval_result.draws}/{eval_result.games} sec={eval_seconds:.2f}"
                    )
                    if score > benchmark_best_score:
                        benchmark_best_score = score
                        save_model(model, benchmark_best_path)
                        save_score_metadata(
                            os.path.join(args.save_dir, "benchmark_best_metrics.json"),
                            {
                                "iteration": iteration,
                                "score": score,
                                "win_rate": eval_result.primary_win_rate,
                                "draws": eval_result.draws,
                                "games": eval_result.games,
                                "opponent": eval_opponent_bot.name,
                            },
                        )
                        row["benchmark_best_updated"] = True
                    else:
                        row["benchmark_best_updated"] = False

                    champion_promoted = False
                    challenger_score = None
                    if not os.path.exists(champion_path):
                        save_model(model, champion_path)
                        save_model(model, best_model_path)
                        champion_benchmark_score = score
                        save_score_metadata(
                            champion_metrics_path,
                            {
                                "iteration": iteration,
                                "score": score,
                                "win_rate": eval_result.primary_win_rate,
                                "draws": eval_result.draws,
                                "games": eval_result.games,
                                "benchmark_opponent": eval_opponent_bot.name,
                                "promotion_reason": "bootstrap",
                            },
                        )
                        champion_promoted = True
                    else:
                        benchmark_gap = score - champion_benchmark_score
                        challenge_allowed = score >= (champion_benchmark_score - args.champion_benchmark_slack)
                        challenge_triggered = (
                            challenge_allowed and (
                                score >= champion_benchmark_score + args.champion_gate
                                or score >= benchmark_best_score - args.champion_benchmark_slack
                            )
                        )
                        row["champion_benchmark_gap"] = benchmark_gap
                        row["champion_challenge_allowed"] = challenge_allowed
                        row["champion_challenge_triggered"] = challenge_triggered
                    if os.path.exists(champion_path) and row.get("champion_challenge_triggered"):
                        _, _, champion_result, champion_seconds = evaluate_symmetric_specs(
                            primary_spec=candidate_eval_target,
                            opponent_spec=champion_path,
                            games=args.champion_games,
                            max_steps=args.eval_max_steps,
                            device=device,
                            reward_config=reward_config,
                        )
                        challenger_score = champion_result.primary_score
                        row.update({
                            "champion_match_games": champion_result.games,
                            "champion_match_wins": champion_result.primary_wins,
                            "champion_match_losses": champion_result.opponent_wins,
                            "champion_match_draws": champion_result.draws,
                            "champion_match_score": challenger_score,
                            "champion_match_avg_turns": champion_result.avg_turns,
                            "champion_match_seconds": champion_seconds,
                        })
                        writer.add_scalar("champion/match_score", challenger_score, iteration)
                        benchmark_ok = score >= (champion_benchmark_score - args.champion_benchmark_slack)
                        promoted_by_h2h = challenger_score > 0.5 + args.champion_margin
                        row["champion_benchmark_ok"] = benchmark_ok
                        row["champion_h2h_ok"] = promoted_by_h2h
                        if promoted_by_h2h and benchmark_ok:
                            save_model(model, champion_path)
                            save_model(model, best_model_path)
                            champion_benchmark_score = score
                            save_score_metadata(
                                champion_metrics_path,
                                {
                                    "iteration": iteration,
                                    "score": score,
                                    "win_rate": eval_result.primary_win_rate,
                                    "draws": eval_result.draws,
                                    "games": eval_result.games,
                                    "benchmark_opponent": eval_opponent_bot.name,
                                    "promotion_reason": "head_to_head",
                                    "champion_match_score": challenger_score,
                                },
                            )
                            champion_promoted = True
                        pbar.write(
                            f"[champion {iteration}] candidate vs champion "
                            f"score={challenger_score:.3f} benchmark={score:.3f} "
                            f"champion_benchmark={champion_benchmark_score:.3f} "
                            f"draws={champion_result.draws}/{champion_result.games} "
                            f"sec={champion_seconds:.2f}"
                        )
                    row["champion_promoted"] = champion_promoted
                    row["champion_benchmark_score"] = champion_benchmark_score
                    if challenger_score is not None:
                        row["challenger_score"] = challenger_score
                else:
                    red_spec = latest_eval_target if args.eval_red == "latest" else args.eval_red
                    black_spec = latest_eval_target if args.eval_black == "latest" else args.eval_black
                    eval_red_bot, eval_black_bot, eval_result, eval_seconds = evaluate_specs(
                        red_spec=red_spec,
                        black_spec=black_spec,
                        games=args.eval_games,
                        max_steps=args.eval_max_steps,
                        device=device,
                        reward_config=reward_config,
                    )
                    games = max(1, eval_result.games)
                    red_win_rate = eval_result.red_wins / games
                    row.update({
                        "eval_mode": "asymmetric",
                        "eval_red": eval_red_bot.name,
                        "eval_black": eval_black_bot.name,
                        "eval_games": eval_result.games,
                        "eval_red_wins": eval_result.red_wins,
                        "eval_black_wins": eval_result.black_wins,
                        "eval_draws": eval_result.draws,
                        "eval_red_win_rate": red_win_rate,
                        "eval_red_score": safe_score(eval_result.red_wins, eval_result.draws, eval_result.games),
                        "eval_avg_turns": eval_result.total_turns / games,
                        "eval_seconds": eval_seconds,
                    })
                    writer.add_scalar("eval/red_win_rate", red_win_rate, iteration)
                    writer.add_scalar("eval/avg_turns", eval_result.total_turns / games, iteration)
                    pbar.write(
                        f"[eval {iteration}] {eval_red_bot.name} vs {eval_black_bot.name} "
                        f"win_rate={red_win_rate:.3f} draws={eval_result.draws}/{games} sec={eval_seconds:.2f}"
                    )
                    if red_win_rate > benchmark_best_score:
                        benchmark_best_score = red_win_rate
                        save_model(model, best_model_path)
                        row["benchmark_best_updated"] = True
                    else:
                        row["benchmark_best_updated"] = False

            append_metrics(args.save_dir, row)
    except KeyboardInterrupt:
        save_model(model, os.path.join(args.save_dir, "model_interrupted.pth"))
    finally:
        vec_env.close()
        writer.close()


if __name__ == "__main__":
    main()
