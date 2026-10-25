from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

import torch
from tqdm import tqdm

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from JunqiEnv.junqi import PlayerId

from junqi_rl.bots import MatchResult, build_bot, configure_heuristic_lookahead
from junqi_rl.env.env_wrapper import JunqiPPOEnv
from junqi_rl.training_config import RewardConfig

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None


@dataclass
class SymmetricResult:
    as_red: MatchResult
    as_black: MatchResult

    @property
    def primary_wins(self) -> int:
        return self.as_red.red_wins + self.as_black.black_wins

    @property
    def opponent_wins(self) -> int:
        return self.as_red.black_wins + self.as_black.red_wins

    @property
    def draws(self) -> int:
        return self.as_red.draws + self.as_black.draws

    @property
    def games(self) -> int:
        return self.as_red.games + self.as_black.games

    @property
    def avg_turns(self) -> float:
        return (self.as_red.total_turns + self.as_black.total_turns) / max(1, self.games)

    @property
    def primary_win_rate(self) -> float:
        return self.primary_wins / max(1, self.games)

    @property
    def primary_score(self) -> float:
        return (self.primary_wins + 0.5 * self.draws) / max(1, self.games)


def play_game(env: JunqiPPOEnv, red_bot, black_bot) -> tuple[int, int]:
    env.reset()

    while True:
        current_player = env.env.get_current_player()
        current_bot = red_bot if current_player == PlayerId.RED else black_bot
        action = current_bot.select_action(env, current_player)

        if action is None:
            winner = PlayerId.BLACK if current_player == PlayerId.RED else PlayerId.RED
            return winner, env.env.get_turn_count()

        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            return env.env.get_winner(), env.env.get_turn_count()


def evaluate_pair(
    red_bot,
    black_bot,
    games: int,
    max_steps: int,
    reward_config: RewardConfig | None = None,
    show_progress: bool = True,
) -> MatchResult:
    env = JunqiPPOEnv(max_steps=max_steps, reward_config=reward_config)
    result = MatchResult()
    iterator = range(games)
    if show_progress:
        iterator = tqdm(iterator, desc=f"{red_bot.name} vs {black_bot.name}", unit="game")
    for _ in iterator:
        winner, turns = play_game(env, red_bot, black_bot)
        result.total_turns += turns
        if winner == PlayerId.RED:
            result.red_wins += 1
        elif winner == PlayerId.BLACK:
            result.black_wins += 1
        else:
            result.draws += 1

    return result


def evaluate_specs(
    red_spec: str,
    black_spec: str,
    games: int,
    max_steps: int,
    device: torch.device,
    reward_config: RewardConfig | None = None,
):
    red_bot = build_bot(red_spec, device)
    black_bot = build_bot(black_spec, device)
    start = time.perf_counter()
    result = evaluate_pair(red_bot, black_bot, games=games, max_steps=max_steps, reward_config=reward_config, show_progress=False)
    elapsed = time.perf_counter() - start
    return red_bot, black_bot, result, elapsed


def evaluate_symmetric_specs(
    primary_spec: str,
    opponent_spec: str,
    games: int,
    max_steps: int,
    device: torch.device,
    reward_config: RewardConfig | None = None,
):
    primary_bot = build_bot(primary_spec, device)
    opponent_bot = build_bot(opponent_spec, device)
    start = time.perf_counter()
    as_red = evaluate_pair(
        primary_bot,
        opponent_bot,
        games=games,
        max_steps=max_steps,
        reward_config=reward_config,
        show_progress=False,
    )
    as_black = evaluate_pair(
        opponent_bot,
        primary_bot,
        games=games,
        max_steps=max_steps,
        reward_config=reward_config,
        show_progress=False,
    )
    elapsed = time.perf_counter() - start
    return primary_bot, opponent_bot, SymmetricResult(as_red=as_red, as_black=as_black), elapsed


def print_report(red_bot, black_bot, result: MatchResult):
    games = max(1, result.games)
    rows = [[
        red_bot.name,
        black_bot.name,
        result.red_wins,
        result.black_wins,
        result.draws,
        f"{result.red_wins / games:.1%}",
        f"{result.total_turns / games:.1f}",
    ]]
    headers = ["red", "black", "red_wins", "black_wins", "draws", "red_win_rate", "avg_turns"]
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
        return

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(" | ".join(str(cell) for cell in row))


def print_symmetric_report(
    primary_bot,
    opponent_bot,
    as_red: MatchResult,
    as_black: MatchResult,
):
    red_games = max(1, as_red.games)
    black_games = max(1, as_black.games)
    total_games = red_games + black_games
    primary_wins = as_red.red_wins + as_black.black_wins
    opponent_wins = as_red.black_wins + as_black.red_wins
    draws = as_red.draws + as_black.draws
    avg_turns = (as_red.total_turns + as_black.total_turns) / max(1, total_games)

    rows = [
        [
            f"{primary_bot.name} as red",
            opponent_bot.name,
            as_red.red_wins,
            as_red.black_wins,
            as_red.draws,
            f"{as_red.red_wins / red_games:.1%}",
            f"{(as_red.red_wins + 0.5 * as_red.draws) / red_games:.1%}",
            f"{as_red.total_turns / red_games:.1f}",
        ],
        [
            f"{primary_bot.name} as black",
            opponent_bot.name,
            as_black.black_wins,
            as_black.red_wins,
            as_black.draws,
            f"{as_black.black_wins / black_games:.1%}",
            f"{(as_black.black_wins + 0.5 * as_black.draws) / black_games:.1%}",
            f"{as_black.total_turns / black_games:.1f}",
        ],
        [
            f"{primary_bot.name} combined",
            opponent_bot.name,
            primary_wins,
            opponent_wins,
            draws,
            f"{primary_wins / total_games:.1%}",
            f"{(primary_wins + 0.5 * draws) / total_games:.1%}",
            f"{avg_turns:.1f}",
        ],
    ]
    headers = ["primary", "opponent", "primary_wins", "opponent_wins", "draws", "primary_win_rate", "primary_score", "avg_turns"]
    if tabulate is not None:
        print(tabulate(rows, headers=headers, tablefmt="github"))
        return

    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for row in rows:
        print(" | ".join(str(cell) for cell in row))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Junqi agents.")
    parser.add_argument("--red", type=str, help="random | greedy | heuristic | path/to/model.pth")
    parser.add_argument("--black", type=str, help="random | greedy | heuristic | path/to/model.pth")
    parser.add_argument("--primary", type=str, help="Primary bot for symmetric evaluation.")
    parser.add_argument("--opponent", type=str, help="Opponent bot for symmetric evaluation.")
    parser.add_argument("--games", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--reward-preset", type=str, choices=["shaped", "hybrid", "terminal"], default="shaped")
    parser.add_argument("--draw-reward", type=float, default=-2.0)
    parser.add_argument("--no-combat-limit", type=int, default=40)
    parser.add_argument("--no-combat-draw-reward", type=float, default=-2.0)
    parser.add_argument("--turn-limit-mode", type=str, choices=["draw", "material"], default="draw")
    parser.add_argument("--heuristic-candidates", type=int, default=4, help="Lookahead heuristic candidate group count for external evaluation bots.")
    parser.add_argument("--heuristic-replies", type=int, default=3, help="Lookahead heuristic opponent reply group count for external evaluation bots.")
    parser.add_argument("--heuristic-counters", type=int, default=2, help="Lookahead heuristic counter-reply group count for external evaluation bots.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_heuristic_lookahead(
        candidates=args.heuristic_candidates,
        replies=args.heuristic_replies,
        counters=args.heuristic_counters,
    )
    reward_config = RewardConfig.preset(args.reward_preset).with_overrides(
        draw_reward=args.draw_reward,
        no_combat_limit=args.no_combat_limit,
        no_combat_draw_reward=args.no_combat_draw_reward,
        turn_limit_mode=args.turn_limit_mode,
    )
    if args.primary or args.opponent:
        if not args.primary or not args.opponent:
            raise SystemExit("--primary and --opponent must be provided together")
        primary_bot, opponent_bot, result, _ = evaluate_symmetric_specs(
            primary_spec=args.primary,
            opponent_spec=args.opponent,
            games=args.games,
            max_steps=args.max_steps,
            device=device,
            reward_config=reward_config,
        )
        print_symmetric_report(primary_bot, opponent_bot, result.as_red, result.as_black)
        return

    if not args.red or not args.black:
        raise SystemExit("Either provide --red/--black or --primary/--opponent")

    red_bot = build_bot(args.red, device)
    black_bot = build_bot(args.black, device)
    result = evaluate_pair(red_bot, black_bot, games=args.games, max_steps=args.max_steps, reward_config=reward_config)
    print_report(red_bot, black_bot, result)


if __name__ == "__main__":
    main()
