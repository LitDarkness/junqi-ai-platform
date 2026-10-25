from __future__ import annotations

import os
import sys

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import JunqiEnv
from JunqiEnv.junqi import CombatResult, PieceType, PlayerId


def idx(row: int, col: int) -> int:
    return row * 5 + col


def red(piece: PieceType) -> int:
    return int(piece)


def black(piece: PieceType) -> int:
    return -int(piece)


def fresh_env():
    env = JunqiEnv.JunqiEnvironment()
    board = [0] * 60
    env.load_board_state(board, PlayerId.RED)
    return env


def test_mine_is_immobile():
    env = fresh_env()
    board = [0] * 60
    board[idx(1, 0)] = red(PieceType.MINE)
    env.load_board_state(board, PlayerId.RED)
    assert not env.can_move(idx(1, 0), idx(2, 0))


def test_flag_is_immobile():
    env = fresh_env()
    board = [0] * 60
    board[idx(0, 1)] = red(PieceType.FLAG)
    env.load_board_state(board, PlayerId.RED)
    assert not env.can_move(idx(0, 1), idx(1, 1))


def test_sapper_clears_mine():
    env = fresh_env()
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.SAPPER)
    board[idx(6, 0)] = black(PieceType.MINE)
    env.load_board_state(board, PlayerId.RED)
    result = env.step(idx(5, 0), idx(6, 0))
    assert result == CombatResult.WIN


def test_non_sapper_loses_to_mine():
    env = fresh_env()
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.CAPTAIN)
    board[idx(6, 0)] = black(PieceType.MINE)
    env.load_board_state(board, PlayerId.RED)
    result = env.step(idx(5, 0), idx(6, 0))
    assert result == CombatResult.LOSS


def test_bomb_draws():
    env = fresh_env()
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.BOMB)
    board[idx(6, 0)] = black(PieceType.GENERAL)
    env.load_board_state(board, PlayerId.RED)
    result = env.step(idx(5, 0), idx(6, 0))
    assert result == CombatResult.DRAW


def test_equal_rank_draws():
    env = fresh_env()
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.CAPTAIN)
    board[idx(6, 0)] = black(PieceType.CAPTAIN)
    env.load_board_state(board, PlayerId.RED)
    result = env.step(idx(5, 0), idx(6, 0))
    assert result == CombatResult.DRAW


def test_higher_rank_wins():
    env = fresh_env()
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.GENERAL)
    board[idx(6, 0)] = black(PieceType.CAPTAIN)
    env.load_board_state(board, PlayerId.RED)
    result = env.step(idx(5, 0), idx(6, 0))
    assert result == CombatResult.WIN


def test_camp_blocks_attack_entry():
    env = fresh_env()
    board = [0] * 60
    board[idx(6, 2)] = red(PieceType.CAPTAIN)
    board[idx(7, 1)] = black(PieceType.LIEUTENANT)
    env.load_board_state(board, PlayerId.RED)
    assert not env.can_move(idx(6, 2), idx(7, 1))


def test_hq_piece_cannot_move_out():
    env = fresh_env()
    board = [0] * 60
    board[idx(0, 1)] = red(PieceType.CAPTAIN)
    env.load_board_state(board, PlayerId.RED)
    assert not env.can_move(idx(0, 1), idx(1, 1))


def test_camp_diagonal_move():
    env = fresh_env()
    board = [0] * 60
    board[idx(3, 2)] = red(PieceType.CAPTAIN)
    env.load_board_state(board, PlayerId.RED)
    assert env.can_move(idx(3, 2), idx(2, 1))


def main():
    tests = [
        test_mine_is_immobile,
        test_flag_is_immobile,
        test_sapper_clears_mine,
        test_non_sapper_loses_to_mine,
        test_bomb_draws,
        test_equal_rank_draws,
        test_higher_rank_wins,
        test_camp_blocks_attack_entry,
        test_hq_piece_cannot_move_out,
        test_camp_diagonal_move,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
