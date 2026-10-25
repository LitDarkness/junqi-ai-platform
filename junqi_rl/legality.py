from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LegalityResult:
    legal: bool
    code: str
    message_key: str
    blocking_square: int = -1
    path: tuple[int, ...] = ()


def check_move(env, from_pos: int, to_pos: int) -> LegalityResult:
    """Stable explanation adapter for the legacy C++ core.

    Detailed railway paths remain a C++ optimization concern; this adapter
    guarantees stable top-level reason codes for GUI, CLI and tests.
    """
    if hasattr(env, "check_move"):
        result = env.check_move(from_pos, to_pos)
        return LegalityResult(
            bool(result.legal),
            str(result.code),
            str(result.message_key),
            int(result.blocking_square),
            tuple(int(square) for square in result.path),
        )

    if env.is_game_over():
        return LegalityResult(False, "GAME_ALREADY_FINISHED", "move.game_already_finished")
    if not 0 <= from_pos < 60 or not 0 <= to_pos < 60:
        return LegalityResult(False, "SQUARE_OUT_OF_RANGE", "move.square_out_of_range")
    if from_pos == to_pos:
        return LegalityResult(False, "SAME_SOURCE_DESTINATION", "move.same_source_destination")
    source = env.get_piece_at(from_pos)
    target = env.get_piece_at(to_pos)
    if source.owner == 0:
        return LegalityResult(False, "NO_PIECE_AT_SOURCE", "move.no_piece_at_source")
    if source.owner != env.get_current_player():
        return LegalityResult(False, "NOT_SIDE_TO_MOVE", "move.not_side_to_move")
    if source.type in (1, 3):
        return LegalityResult(False, "PIECE_IMMOBILE", "move.piece_immobile")
    if target.owner == source.owner:
        return LegalityResult(False, "DEST_OWN_OCCUPIED", "move.dest_own_occupied")
    if env.can_move(from_pos, to_pos):
        return LegalityResult(True, "OK", "move.ok")
    return LegalityResult(False, "NO_BOARD_EDGE", "move.no_board_edge")
