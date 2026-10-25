from __future__ import annotations

import os
import sys

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from JunqiEnv.junqi import PieceType, PlayerId

from junqi_rl.bots import HeuristicStrategicBot, HeuristicTacticalBot
from junqi_rl.env.env_wrapper import JunqiPPOEnv


def idx(row: int, col: int) -> int:
    return row * 5 + col


def red(piece: PieceType) -> int:
    return int(piece)


def black(piece: PieceType) -> int:
    return -int(piece)


def make_env(board: list[int], player: int) -> JunqiPPOEnv:
    env = JunqiPPOEnv(max_steps=200)
    env.env.load_board_state(board, player)
    env._current_player = player
    env._prev_material_delta = env._material_delta(player)
    return env


def test_heuristic_captures_flag():
    board = [0] * 60
    board[idx(5, 2)] = red(PieceType.CAPTAIN)
    board[idx(6, 2)] = black(PieceType.FLAG)
    env = make_env(board, PlayerId.RED)
    bot = HeuristicStrategicBot()
    action = bot.select_action(env, PlayerId.RED)
    env_action = env.action_to_env(action, PlayerId.RED)
    assert env_action == idx(5, 2) * 60 + idx(6, 2)


def test_heuristic_uses_sapper_on_mine():
    board = [0] * 60
    board[idx(5, 1)] = red(PieceType.SAPPER)
    board[idx(6, 1)] = black(PieceType.MINE)
    env = make_env(board, PlayerId.RED)
    bot = HeuristicStrategicBot()
    action = bot.select_action(env, PlayerId.RED)
    env_action = env.action_to_env(action, PlayerId.RED)
    assert env_action == idx(5, 1) * 60 + idx(6, 1)


def test_heuristic_avoids_non_sapper_into_mine_when_safe_move_exists():
    board = [0] * 60
    board[idx(5, 1)] = red(PieceType.CAPTAIN)
    board[idx(6, 1)] = black(PieceType.MINE)
    env = make_env(board, PlayerId.RED)
    bot = HeuristicStrategicBot()
    action = bot.select_action(env, PlayerId.RED)
    env_action = env.action_to_env(action, PlayerId.RED)
    assert env_action != idx(5, 1) * 60 + idx(6, 1)


def test_heuristic_prefers_safe_capture_over_bomb_trade():
    board = [0] * 60
    board[idx(5, 2)] = red(PieceType.GENERAL)
    board[idx(6, 2)] = black(PieceType.BOMB)
    board[idx(5, 3)] = black(PieceType.CAPTAIN)
    env = make_env(board, PlayerId.RED)
    bot = HeuristicStrategicBot()
    action = bot.select_action(env, PlayerId.RED)
    env_action = env.action_to_env(action, PlayerId.RED)
    assert env_action == idx(5, 2) * 60 + idx(5, 3)


def test_heuristic_v2_not_weaker_than_v1_on_safe_capture_case():
    board = [0] * 60
    board[idx(5, 2)] = red(PieceType.GENERAL)
    board[idx(6, 2)] = black(PieceType.BOMB)
    board[idx(5, 3)] = black(PieceType.LIEUTENANT)
    env = make_env(board, PlayerId.RED)
    v1 = HeuristicTacticalBot()
    v2 = HeuristicStrategicBot()
    action_v1 = env.action_to_env(v1.select_action(env, PlayerId.RED), PlayerId.RED)
    action_v2 = env.action_to_env(v2.select_action(env, PlayerId.RED), PlayerId.RED)
    assert action_v2 == idx(5, 2) * 60 + idx(5, 3)
    assert action_v1 in (
        idx(5, 2) * 60 + idx(5, 3),
        idx(5, 2) * 60 + idx(6, 2),
    )


def main():
    tests = [
        test_heuristic_captures_flag,
        test_heuristic_uses_sapper_on_mine,
        test_heuristic_avoids_non_sapper_into_mine_when_safe_move_exists,
        test_heuristic_prefers_safe_capture_over_bomb_trade,
        test_heuristic_v2_not_weaker_than_v1_on_safe_capture_case,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
