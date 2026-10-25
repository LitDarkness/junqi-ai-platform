from __future__ import annotations

import os
import sys

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from JunqiEnv.junqi import PieceType, PlayerId

from junqi_rl.env.env_wrapper import JunqiPPOEnv
from junqi_rl.training_config import RewardConfig


def idx(row: int, col: int) -> int:
    return row * 5 + col


def red(piece: PieceType) -> int:
    return int(piece)


def black(piece: PieceType) -> int:
    return -int(piece)


def test_no_combat_limit_triggers_draw():
    reward_config = RewardConfig.preset("shaped").with_overrides(
        draw_reward=-3.0,
        no_combat_limit=4,
        no_combat_draw_reward=-3.0,
        turn_limit_mode="draw",
    )
    env = JunqiPPOEnv(max_steps=100, reward_config=reward_config)
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.CAPTAIN)
    board[idx(6, 4)] = black(PieceType.CAPTAIN)
    env.env.load_board_state(board, PlayerId.RED)
    env._current_player = PlayerId.RED
    env._prev_material_delta = env._material_delta(PlayerId.RED)
    env._no_combat_steps = 0

    action_sequence = [
        (PlayerId.RED, idx(5, 0), idx(4, 0)),
        (PlayerId.BLACK, idx(6, 4), idx(7, 4)),
        (PlayerId.RED, idx(4, 0), idx(5, 0)),
        (PlayerId.BLACK, idx(7, 4), idx(6, 4)),
    ]

    last_reward = None
    last_truncated = None
    for player_id, from_pos, to_pos in action_sequence:
        policy_action = env.action_to_policy(from_pos * 60 + to_pos, player_id)
        _, reward, terminated, truncated, _ = env.step(policy_action)
        assert not terminated
        last_reward = reward
        last_truncated = truncated

    assert last_truncated is True
    assert last_reward == -3.0
    assert env._no_combat_steps == 4


def test_turn_limit_draw_reward():
    reward_config = RewardConfig.preset("shaped").with_overrides(
        draw_reward=-1.5,
        no_combat_limit=0,
        no_combat_draw_reward=-3.0,
        turn_limit_mode="draw",
    )
    env = JunqiPPOEnv(max_steps=2, reward_config=reward_config)
    board = [0] * 60
    board[idx(5, 0)] = red(PieceType.CAPTAIN)
    board[idx(6, 4)] = black(PieceType.CAPTAIN)
    env.env.load_board_state(board, PlayerId.RED)
    env._current_player = PlayerId.RED
    env._prev_material_delta = env._material_delta(PlayerId.RED)
    env._no_combat_steps = 0

    first = env.action_to_policy(idx(5, 0) * 60 + idx(4, 0), PlayerId.RED)
    _, _, terminated, truncated, _ = env.step(first)
    assert not terminated and not truncated

    second = env.action_to_policy(idx(6, 4) * 60 + idx(7, 4), PlayerId.BLACK)
    _, reward, terminated, truncated, _ = env.step(second)
    assert not terminated
    assert truncated is True
    assert reward == -1.5


def main():
    tests = [
        test_no_combat_limit_triggers_draw,
        test_turn_limit_draw_reward,
    ]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")


if __name__ == "__main__":
    main()
