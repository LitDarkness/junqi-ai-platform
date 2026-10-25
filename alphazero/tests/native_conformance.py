from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PLATFORM = ROOT.parent
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(PLATFORM / "build"))

import JunqiAZ  # type: ignore  # noqa: E402
import JunqiEnv  # type: ignore  # noqa: E402


def az_board_to_raw(state) -> list[int]:
    board = list(state.board())
    if state.current_player() == -1:
        return [-board[59 - position] for position in range(60)]
    return board


def canonical_to_raw(action: int, player: int) -> int:
    return JunqiAZ.JunqiAZState.flip_action(action) if player == -1 else action


def mirror_lr_position(position: int) -> int:
    row, col = divmod(position, 5)
    return row * 5 + (4 - col)


def mirror_lr_action(action: int) -> int:
    source, target = divmod(action, 60)
    return mirror_lr_position(source) * 60 + mirror_lr_position(target)


def run(games: int, plies: int, seed: int) -> tuple[int, int]:
    rng = random.Random(seed)
    transitions = 0
    for game in range(games):
        state = JunqiAZ.JunqiAZState()
        state.reset_random(seed + game)
        oracle = JunqiEnv.JunqiEnvironment()
        oracle.load_board_state(az_board_to_raw(state), 1)

        mirrored = JunqiAZ.JunqiAZState()
        raw_board = az_board_to_raw(state)
        mirrored_board = [0] * 60
        for position, piece in enumerate(raw_board):
            mirrored_board[mirror_lr_position(position)] = piece
        mirrored.load_board_state(mirrored_board, state.current_player())
        expected_actions = sorted(mirror_lr_action(int(a)) for a in state.legal_actions())
        assert sorted(int(a) for a in mirrored.legal_actions()) == expected_actions
        assert np.array_equal(
            np.asarray(mirrored.feature_planes()),
            np.flip(np.asarray(state.feature_planes()), axis=-1),
        )

        terrain = np.asarray(state.feature_planes())[13]
        assert int(terrain.sum()) == 10
        for _ in range(plies):
            assert state.current_player() == int(oracle.get_current_player())
            player = state.current_player()
            canonical = sorted(int(action) for action in state.legal_actions())
            raw_actions = sorted({canonical_to_raw(action, player) for action in canonical})
            # The legacy oracle may emit the same railway destination through
            # more than one path; action-set semantics are what the rules define.
            oracle_actions = sorted({int(action) for action in oracle.get_legal_actions(player)})
            if raw_actions != oracle_actions:
                raise AssertionError(f"legal action drift game={game} ply={transitions}")
            if not canonical:
                break
            action = canonical[rng.randrange(len(canonical))]
            raw_action = canonical_to_raw(action, player)
            source, target = divmod(raw_action, 60)
            az_result = int(state.apply(action))
            oracle_result = int(oracle.step(source, target))
            if az_result != oracle_result:
                raise AssertionError(
                    f"combat drift game={game} action={raw_action}: {az_result}!={oracle_result}"
                )
            if az_board_to_raw(state) != list(oracle.get_perfect_state()):
                raise AssertionError(f"board drift game={game} action={raw_action}")
            assert state.turn_count() == int(oracle.get_turn_count())
            assert state.is_terminal() == bool(oracle.is_game_over())
            assert state.winner() == int(oracle.get_winner())
            transitions += 1
            if state.is_terminal():
                break
    return games, transitions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--plies", type=int, default=80)
    parser.add_argument("--seed", type=int, default=20260713)
    args = parser.parse_args()
    games, transitions = run(args.games, args.plies, args.seed)
    print(f"NATIVE_CONFORMANCE_PASS games={games} transitions={transitions}")


if __name__ == "__main__":
    main()
