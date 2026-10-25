from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "build"), str(Path(__file__).parent)]

import JunqiEnv
from JunqiEnv.junqi import PlayerId
import candidate


def compare() -> None:
    rng = random.Random(20260713)
    checked_positions = checked_transitions = 0
    for game in range(24):
        env = JunqiEnv.JunqiEnvironment()
        env.reset_random()
        for ply in range(48):
            board = list(env.get_perfect_state())
            player = int(env.get_current_player())
            expected = list(env.get_legal_actions(player))
            actual = candidate._MODULE.debug_legal_actions(board, player)
            if set(actual) != set(expected):
                raise AssertionError((game, ply, player, set(actual), set(expected)))
            checked_positions += 1
            for action in (expected[:1] + expected[len(expected) // 2:len(expected) // 2 + 1] + expected[-1:]):
                oracle = JunqiEnv.JunqiEnvironment()
                oracle.load_board_state(board, player)
                oracle.step(action // 60, action % 60)
                next_board, next_player, terminal, winner = candidate._MODULE.debug_transition(board, player, action)
                if (list(next_board) != list(oracle.get_perfect_state()) or next_player != int(oracle.get_current_player())
                        or terminal != oracle.is_game_over() or winner != int(oracle.get_winner())):
                    raise AssertionError((game, ply, action, next_player, int(oracle.get_current_player()), terminal, oracle.is_game_over(), winner, int(oracle.get_winner())))
                checked_transitions += 1
            if not expected or env.is_game_over():
                break
            action = rng.choice(expected)
            env.step(action // 60, action % 60)
    print(f"conformance passed: positions={checked_positions}, transitions={checked_transitions}")


if __name__ == "__main__":
    compare()
