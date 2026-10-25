from __future__ import annotations

import os
import random
import sys
import time
from dataclasses import dataclass, field

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import JunqiEnv
from JunqiEnv.junqi import PlayerId

BOARD_COLS = 5
BOARD_SIZE = 60
MATERIAL_VALUES = {
    90: 9.0,
    80: 8.0,
    70: 7.0,
    60: 6.0,
    50: 5.0,
    40: 4.0,
    30: 3.0,
    20: 2.0,
    10: 5.0,
    2: 8.0,
    1: 8.0,
    3: 15.0,
}
RANK_STRENGTH = {
    90: 9,
    80: 8,
    70: 7,
    60: 6,
    50: 5,
    40: 4,
    30: 3,
    20: 2,
    10: 1,
    2: 10,
    1: 0,
    3: -1,
}
RED_CAMPS = {11, 13, 17, 21, 23}
CAMPS = RED_CAMPS | {BOARD_SIZE - 1 - position for position in RED_CAMPS}
HQS = {
    PlayerId.RED: (1, 3),
    PlayerId.BLACK: (56, 58),
}
ORTHOGONAL_DIRS = (-5, 5, -1, 1)
LOOKAHEAD_CANDIDATES = 4
LOOKAHEAD_REPLIES = 3
LOOKAHEAD_COUNTERS = 2
HEURISTIC_V2_CANDIDATES = 12
HEURISTIC_V3_CANDIDATES = 6
HEURISTIC_V3_REPLY_WEIGHT = 0.55


def piece_value(piece: int) -> float:
    return MATERIAL_VALUES.get(abs(piece), 0.0)


def configure_heuristic_lookahead(
    candidates: int | None = None,
    replies: int | None = None,
    counters: int | None = None,
):
    global LOOKAHEAD_CANDIDATES, LOOKAHEAD_REPLIES, LOOKAHEAD_COUNTERS
    if candidates is not None:
        LOOKAHEAD_CANDIDATES = max(1, int(candidates))
    if replies is not None:
        LOOKAHEAD_REPLIES = max(1, int(replies))
    if counters is not None:
        LOOKAHEAD_COUNTERS = max(1, int(counters))


def row_of(pos: int) -> int:
    return pos // BOARD_COLS


def manhattan(a: int, b: int) -> int:
    ar, ac = divmod(a, BOARD_COLS)
    br, bc = divmod(b, BOARD_COLS)
    return abs(ar - br) + abs(ac - bc)


def combat_outcome(attacker: int, defender: int) -> int:
    attacker = abs(attacker)
    defender = abs(defender)
    if defender == 0:
        return 0
    if defender == 3:
        return 1
    if attacker == 2 or defender == 2:
        return 0
    if defender == 1:
        return 1 if attacker == 10 else -1
    if attacker == 1 or attacker == 3:
        return -1
    if attacker == defender:
        return 0
    return 1 if RANK_STRENGTH.get(attacker, -99) > RANK_STRENGTH.get(defender, -99) else -1


class BaseBot:
    def __init__(self, name: str):
        self.name = name
        self.is_human = False

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        raise NotImplementedError


class HumanBot(BaseBot):
    def __init__(self, name: str = "human"):
        super().__init__(name)
        self.is_human = True

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        return None


class RandomBot(BaseBot):
    def __init__(self, name: str = "random"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None
        return env_wrapper.action_to_policy(random.choice(legal_actions), player_id)


class GreedyMaterialBot(BaseBot):
    def __init__(self, name: str = "greedy"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None

        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        best_action = legal_actions[0]
        best_score = float("-inf")

        board = env_wrapper.env.get_perfect_state()
        for action in legal_actions:
            from_pos, to_pos = divmod(action, 60)
            attacker = abs(board[from_pos])
            defender = abs(board[to_pos])

            score = 0.0
            if defender > 0:
                score += defender * 2.0
                if attacker and defender >= attacker:
                    score -= attacker * 0.75
            if to_pos in (1, 3, 56, 58):
                score += 0.1
            if env_wrapper.env.get_material_score(player_id) >= env_wrapper.env.get_material_score(opponent):
                score += random.random() * 1e-3

            if score > best_score:
                best_score = score
                best_action = action

        return env_wrapper.action_to_policy(best_action, player_id)


class HeuristicTacticalBot(BaseBot):
    def __init__(self, name: str = "heuristic_v1"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None

        board = env_wrapper.env.get_perfect_state()
        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        my_material = env_wrapper.env.get_material_score(player_id)
        opp_material = env_wrapper.env.get_material_score(opponent)
        best_action = choose_heuristic_action(board, legal_actions, player_id, my_material, opp_material)
        return env_wrapper.action_to_policy(best_action, player_id)


class HeuristicStrategicBot(BaseBot):
    def __init__(self, name: str = "heuristic_v2"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None

        board = env_wrapper.env.get_perfect_state()
        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        my_material = env_wrapper.env.get_material_score(player_id)
        opp_material = env_wrapper.env.get_material_score(opponent)
        best_action = choose_heuristic_action_v2(board, legal_actions, player_id, my_material, opp_material)
        return env_wrapper.action_to_policy(best_action, player_id)


class HeuristicProductionBot(BaseBot):
    """Fast one-reply tactical guard built on the v2 positional evaluator."""

    def __init__(self, name: str = "heuristic_v3"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None
        board = env_wrapper.env.get_perfect_state()
        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        action = choose_heuristic_action_v3_fast(
            board,
            legal_actions,
            player_id,
            env_wrapper.env.get_material_score(player_id),
            env_wrapper.env.get_material_score(opponent),
        )
        return env_wrapper.action_to_policy(action, player_id)


class HeuristicTrialBot(BaseBot):
    """Explicit benchmark-only v3 configuration; never selected by default."""

    def __init__(
        self,
        name: str,
        *,
        candidate_count: int = 6,
        reply_weight: float = 0.55,
        baseline_credit: float = 0.0,
        reply_position_weight: float = 0.0,
    ):
        super().__init__(name)
        self.candidate_count = candidate_count
        self.reply_weight = reply_weight
        self.baseline_credit = baseline_credit
        self.reply_position_weight = reply_position_weight

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None
        board = env_wrapper.env.get_perfect_state()
        opponent = other_player(player_id)
        selected = choose_heuristic_action_v3_config(
            board,
            legal_actions,
            player_id,
            env_wrapper.env.get_material_score(player_id),
            env_wrapper.env.get_material_score(opponent),
            candidate_count=self.candidate_count,
            reply_weight=self.reply_weight,
            baseline_credit=self.baseline_credit,
            reply_position_weight=self.reply_position_weight,
        )
        return env_wrapper.action_to_policy(selected, player_id)


class HeuristicLookaheadBot(BaseBot):
    """Experimental multi-ply search bot; excluded from the production speed gate."""

    def __init__(self, name: str = "lookahead_experimental"):
        super().__init__(name)

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None

        board = env_wrapper.env.get_perfect_state()
        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        my_material = env_wrapper.env.get_material_score(player_id)
        opp_material = env_wrapper.env.get_material_score(opponent)
        best_action = choose_heuristic_action_lookahead(board, legal_actions, player_id, my_material, opp_material)
        return env_wrapper.action_to_policy(best_action, player_id)


def heuristic_forward_progress(from_pos: int, to_pos: int, player_id: int) -> int:
    from_row = row_of(from_pos)
    to_row = row_of(to_pos)
    if player_id == PlayerId.RED:
        return to_row - from_row
    return from_row - to_row


def score_heuristic_action(
    action: int,
    board: list[int] | tuple[int, ...],
    player_id: int,
    my_material: float,
    opp_material: float,
) -> float:
    opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
    material_edge = my_material - opp_material
    enemy_hqs = HQS[opponent]
    my_hqs = HQS[player_id]

    from_pos, to_pos = divmod(action, BOARD_SIZE)
    attacker = board[from_pos]
    defender = board[to_pos]
    attacker_type = abs(attacker)
    defender_type = abs(defender)
    attacker_value = piece_value(attacker)
    defender_value = piece_value(defender)

    score = 0.0

    if defender != 0:
        outcome = combat_outcome(attacker, defender)
        if defender_type == 3:
            return 10000.0
        if outcome > 0:
            score += 10.0 + defender_value * 5.0
            score -= attacker_value * 0.15
            if defender_type >= 70:
                score += 6.0
            if defender_type == 1 and attacker_type == 10:
                score += 5.0
        elif outcome == 0:
            score += defender_value * 1.6 - attacker_value * 0.8
            if attacker_type == 2 and defender_type >= 70:
                score += 6.0
            if material_edge > 6.0:
                score -= 6.0
        else:
            score -= attacker_value * 4.0
            if defender_type == 1 and attacker_type != 10:
                score -= 10.0
            if attacker_type >= 70 and defender_type <= 30:
                score -= 8.0
    else:
        score += 0.2

    forward = heuristic_forward_progress(from_pos, to_pos, player_id)
    score += forward * (0.6 if attacker_type >= 40 else 0.35)

    before_enemy_dist = min(manhattan(from_pos, hq) for hq in enemy_hqs)
    after_enemy_dist = min(manhattan(to_pos, hq) for hq in enemy_hqs)
    score += (before_enemy_dist - after_enemy_dist) * (0.28 if attacker_type >= 50 else 0.14)

    before_my_dist = min(manhattan(from_pos, hq) for hq in my_hqs)
    after_my_dist = min(manhattan(to_pos, hq) for hq in my_hqs)
    if material_edge > 0 and attacker_type >= 50:
        score += (after_my_dist - before_my_dist) * 0.10

    if to_pos in CAMPS:
        score += 0.8 if defender == 0 else 0.2
    if to_pos in enemy_hqs:
        score += 2.5
    if from_pos in my_hqs and attacker_type not in (1, 3):
        score += 0.6

    if material_edge > 5.0:
        if defender == 0 and attacker_type >= 70:
            score -= 0.7
        if defender != 0 and defender_value < attacker_value and combat_outcome(attacker, defender) == 0:
            score -= 2.0
    elif material_edge < -5.0:
        if defender != 0 and defender_value >= attacker_value:
            score += 1.5
        if attacker_type == 2 and defender_type >= 70:
            score += 2.0

    return score


def choose_heuristic_action(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
) -> int:
    best_action = legal_actions[0]
    best_score = float("-inf")
    for action in legal_actions:
        score = score_heuristic_action(action, board, player_id, my_material, opp_material)
        score += random.random() * 1e-4
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def in_bounds(pos: int) -> bool:
    return 0 <= pos < BOARD_SIZE


def same_side(piece: int, player_id: int) -> bool:
    return (player_id == PlayerId.RED and piece > 0) or (player_id == PlayerId.BLACK and piece < 0)


def opponent_side(piece: int, player_id: int) -> bool:
    return piece != 0 and not same_side(piece, player_id)


def orthogonal_neighbors(pos: int) -> list[int]:
    row, col = divmod(pos, BOARD_COLS)
    neighbors: list[int] = []
    if row > 0:
        neighbors.append(pos - BOARD_COLS)
    if row < 11:
        neighbors.append(pos + BOARD_COLS)
    if col > 0:
        neighbors.append(pos - 1)
    if col < BOARD_COLS - 1:
        neighbors.append(pos + 1)
    return neighbors


def adjacent_enemy_threat(board: list[int] | tuple[int, ...], pos: int, player_id: int, occupant_type: int) -> float:
    threat = 0.0
    occupant_value = piece_value(occupant_type)
    for neighbor in orthogonal_neighbors(pos):
        enemy_piece = board[neighbor]
        if not opponent_side(enemy_piece, player_id):
            continue
        enemy_type = abs(enemy_piece)
        outcome = combat_outcome(enemy_piece, occupant_type)
        if outcome > 0:
            threat += piece_value(enemy_type) * 0.8 + occupant_value * 0.6 + 1.0
        elif outcome == 0:
            threat += occupant_value * 0.4 + piece_value(enemy_type) * 0.2
    return threat


def adjacent_friendly_support(board: list[int] | tuple[int, ...], pos: int, player_id: int) -> float:
    support = 0.0
    for neighbor in orthogonal_neighbors(pos):
        piece = board[neighbor]
        if same_side(piece, player_id):
            support += 0.8 + piece_value(piece) * 0.03
    return support


def flag_guard_bonus(board: list[int] | tuple[int, ...], player_id: int) -> float:
    flag_value = 0.0
    flag_positions = [i for i, piece in enumerate(board) if same_side(piece, player_id) and abs(piece) == 3]
    if not flag_positions:
        return 0.0
    for flag_pos in flag_positions:
        mines = 0
        guards = 0
        for neighbor in orthogonal_neighbors(flag_pos):
            piece = board[neighbor]
            if same_side(piece, player_id):
                if abs(piece) == 1:
                    mines += 1
                elif abs(piece) != 3:
                    guards += 1
        flag_value += mines * 2.0 + guards * 0.6
    return flag_value


def board_material(board: list[int] | tuple[int, ...], player_id: int) -> float:
    total = 0.0
    for piece in board:
        if same_side(piece, player_id):
            total += piece_value(piece)
    return total


def strategic_board_score(board: list[int] | tuple[int, ...], player_id: int) -> float:
    opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
    score = board_material(board, player_id) - board_material(board, opponent)
    score += flag_guard_bonus(board, player_id)
    score -= flag_guard_bonus(board, opponent) * 0.6

    enemy_hqs = HQS[opponent]
    for pos, piece in enumerate(board):
        if not same_side(piece, player_id):
            continue
        piece_type = abs(piece)
        if piece_type in (1, 3):
            continue
        score += min(8, 14 - min(manhattan(pos, hq) for hq in enemy_hqs)) * (0.08 if piece_type >= 50 else 0.04)
        score -= adjacent_enemy_threat(board, pos, player_id, piece_type) * 0.08

    return score


def simulate_action(
    board: list[int] | tuple[int, ...],
    action: int,
) -> tuple[list[int], int]:
    after = list(board)
    from_pos, to_pos = divmod(action, BOARD_SIZE)
    attacker = after[from_pos]
    defender = after[to_pos]
    outcome = combat_outcome(attacker, defender)
    after[from_pos] = 0

    if defender == 0:
        after[to_pos] = attacker
    elif outcome > 0:
        after[to_pos] = attacker
    elif outcome == 0:
        after[to_pos] = 0
    else:
        pass
    return after, outcome


def score_heuristic_action_v2(
    action: int,
    board: list[int] | tuple[int, ...],
    player_id: int,
    my_material: float,
    opp_material: float,
    baseline_strategic: float | None = None,
) -> float:
    from_pos, to_pos = divmod(action, BOARD_SIZE)
    attacker = board[from_pos]
    attacker_type = abs(attacker)
    attacker_value = piece_value(attacker)
    defender = board[to_pos]
    defender_type = abs(defender)
    defender_value = piece_value(defender)
    material_edge = my_material - opp_material
    opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
    enemy_hqs = HQS[opponent]

    base = score_heuristic_action(action, board, player_id, my_material, opp_material)
    score = base

    pre_threat = adjacent_enemy_threat(board, from_pos, player_id, attacker_type)
    pre_support = adjacent_friendly_support(board, from_pos, player_id)

    after_board, outcome = simulate_action(board, action)
    post_piece = after_board[to_pos]
    post_type = abs(post_piece) if post_piece else 0
    post_threat = adjacent_enemy_threat(after_board, to_pos, player_id, post_type) if post_piece else 0.0
    post_support = adjacent_friendly_support(after_board, to_pos, player_id) if post_piece else 0.0

    score += min(8.0, pre_threat * 0.18)
    score -= post_threat * (0.35 if attacker_type >= 50 else 0.22)
    score += post_support * 0.45
    score -= pre_support * 0.10

    if defender == 0 and post_piece:
        before_enemy_dist = min(manhattan(from_pos, hq) for hq in enemy_hqs)
        after_enemy_dist = min(manhattan(to_pos, hq) for hq in enemy_hqs)
        score += (before_enemy_dist - after_enemy_dist) * (0.35 if attacker_type in (10, 20, 30, 40) else 0.18)

    if post_piece:
        future_capture_value = 0.0
        future_risk = 0.0
        for neighbor in orthogonal_neighbors(to_pos):
            target = after_board[neighbor]
            if opponent_side(target, player_id):
                follow_outcome = combat_outcome(post_piece, target)
                if follow_outcome > 0:
                    future_capture_value = max(future_capture_value, piece_value(target))
            elif same_side(target, player_id) and abs(target) != 3:
                future_risk += 0.2
        score += future_capture_value * 0.55
        score += future_risk * 0.3

    if attacker_type == 2:
        if defender_type >= 70 and outcome >= 0:
            score += 5.0
        if material_edge < 0 and defender != 0:
            score += defender_value * 0.5
    elif attacker_type == 1:
        if defender != 0:
            score -= 9.0
        if material_edge > 0:
            score += 0.8
    elif attacker_type == 3:
        score -= 100.0

    if attacker_type >= 70 and material_edge > 6.0 and post_threat > pre_threat:
        score -= 3.5

    if defender_type == 2 and attacker_type < 70:
        score += 1.0
    if defender_type == 1 and attacker_type != 10:
        score -= 6.0

    if baseline_strategic is None:
        baseline_strategic = strategic_board_score(board, player_id)
    score += (strategic_board_score(after_board, player_id) - baseline_strategic) * 0.25
    return score


def choose_heuristic_action_v2(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
) -> int:
    if len(legal_actions) > HEURISTIC_V2_CANDIDATES:
        candidates = sorted(
            legal_actions,
            key=lambda action: score_heuristic_action(
                action,
                board,
                player_id,
                my_material,
                opp_material,
            ),
            reverse=True,
        )[:HEURISTIC_V2_CANDIDATES]
    else:
        candidates = legal_actions

    best_action = candidates[0]
    best_score = float("-inf")
    baseline_strategic = strategic_board_score(board, player_id)
    for action in candidates:
        score = score_heuristic_action_v2(
            action,
            board,
            player_id,
            my_material,
            opp_material,
            baseline_strategic,
        )
        score += random.random() * 1e-4
        if score > best_score:
            best_score = score
            best_action = action
    return best_action


def other_player(player_id: int) -> int:
    return PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED


_SCRATCH_ENV: JunqiEnv.JunqiEnvironment | None = None


def get_scratch_env() -> JunqiEnv.JunqiEnvironment:
    global _SCRATCH_ENV
    if _SCRATCH_ENV is None:
        _SCRATCH_ENV = JunqiEnv.JunqiEnvironment()
    return _SCRATCH_ENV


def apply_action_with_env(
    board: list[int] | tuple[int, ...],
    acting_player: int,
    action: int,
):
    env = get_scratch_env()
    env.load_board_state(list(board), acting_player)
    from_pos, to_pos = divmod(action, BOARD_SIZE)
    env.step(from_pos, to_pos)
    opponent = other_player(acting_player)
    after_board = list(env.get_perfect_state())
    return {
        "board": after_board,
        "winner": env.get_winner(),
        "game_over": env.is_game_over(),
        "next_player": env.get_current_player(),
        "my_material": float(env.get_material_score(acting_player)),
        "opp_material": float(env.get_material_score(opponent)),
        "legal_actions_next": env.get_legal_actions(env.get_current_player()) if not env.is_game_over() else [],
    }


def top_scored_actions_v2(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
    limit: int,
) -> list[tuple[float, int]]:
    scored: list[tuple[float, int]] = []
    baseline_strategic = strategic_board_score(board, player_id)
    for action in legal_actions:
        score = score_heuristic_action_v2(
            action,
            board,
            player_id,
            my_material,
            opp_material,
            baseline_strategic,
        )
        scored.append((score, action))
    scored.sort(reverse=True)
    return scored[:limit]


def _strongest_reply_v3(
    board: list[int] | tuple[int, ...],
    player_id: int,
) -> tuple[float, int | None]:
    """Return the best immediate tactical score/action on an exact move set."""
    env = get_scratch_env()
    env.load_board_state(list(board), player_id)
    replies = env.get_legal_actions(player_id)
    if not replies:
        return -10000.0, None

    opponent = other_player(player_id)
    my_material = board_material(board, player_id)
    opp_material = board_material(board, opponent)
    strongest = float("-inf")
    strongest_action: int | None = None
    for action in replies:
        target = board[action % BOARD_SIZE]
        if target and abs(target) == 3:
            return 10000.0, action
        score = score_heuristic_action(
            action, board, player_id, my_material, opp_material
        )
        if score > strongest:
            strongest = score
            strongest_action = action
    return strongest, strongest_action


def _strongest_reply_score_v3(
    board: list[int] | tuple[int, ...],
    player_id: int,
) -> float:
    return _strongest_reply_v3(board, player_id)[0]


def choose_heuristic_action_v3_fast(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
) -> int:
    """Production heuristic with bounded one-reply tactical protection.

    v2 remains the positional/capture evaluator.  Only its six best candidates
    are examined against the opponent's exact C++-generated legal replies, so
    railway attacks and immediate flag captures are seen without an unbounded
    search tree.
    """
    return choose_heuristic_action_v3_config(
        board, legal_actions, player_id, my_material, opp_material,
        candidate_count=HEURISTIC_V3_CANDIDATES,
        reply_weight=HEURISTIC_V3_REPLY_WEIGHT,
    )


def choose_heuristic_action_v3_config(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
    *,
    candidate_count: int,
    reply_weight: float,
    baseline_credit: float = 0.0,
    reply_position_weight: float = 0.0,
) -> int:
    """Bounded v3 evaluator with explicit parameters for controlled trials."""
    if len(legal_actions) > HEURISTIC_V2_CANDIDATES:
        prescreened = sorted(
            legal_actions,
            key=lambda action: score_heuristic_action(
                action, board, player_id, my_material, opp_material
            ),
            reverse=True,
        )[:HEURISTIC_V2_CANDIDATES]
    else:
        prescreened = legal_actions
    ranked = top_scored_actions_v2(
        board,
        prescreened,
        player_id,
        my_material,
        opp_material,
        limit=min(candidate_count, len(legal_actions)),
    )
    baseline_reply = 0.0
    if baseline_credit > 0.0:
        baseline_reply = max(0.0, _strongest_reply_score_v3(board, other_player(player_id)))
    best_action = ranked[0][1]
    best_score = float("-inf")
    for static_score, action in ranked:
        _, to_pos = divmod(action, BOARD_SIZE)
        if board[to_pos] and abs(board[to_pos]) == 3:
            return action

        after_board, _ = simulate_action(board, action)
        reply_score, reply_action = _strongest_reply_v3(after_board, other_player(player_id))
        if reply_score >= 9999.0:
            candidate_score = -10000.0
        elif reply_score <= -9999.0:
            candidate_score = 10000.0
        else:
            effective_reply = max(0.0, reply_score - baseline_credit * baseline_reply)
            candidate_score = static_score - reply_weight * effective_reply
            if reply_position_weight and reply_action is not None:
                after_reply, _ = simulate_action(after_board, reply_action)
                positional_delta = strategic_board_score(after_reply, player_id) - strategic_board_score(after_board, player_id)
                candidate_score += reply_position_weight * max(-20.0, min(20.0, positional_delta))

        if candidate_score > best_score:
            best_score = candidate_score
            best_action = action
    return best_action


def choose_heuristic_action_lookahead(
    board: list[int] | tuple[int, ...],
    legal_actions: list[int],
    player_id: int,
    my_material: float,
    opp_material: float,
) -> int:
    opponent = other_player(player_id)
    current_eval = strategic_board_score(board, player_id)
    static_ranked = top_scored_actions_v2(
        board,
        legal_actions,
        player_id,
        my_material,
        opp_material,
        limit=len(legal_actions),
    )

    if len(static_ranked) >= 2:
        top_score, top_action = static_ranked[0]
        second_score, _ = static_ranked[1]
        _, top_to = divmod(top_action, BOARD_SIZE)
        if board[top_to] != 0 and top_score - second_score >= 8.0:
            return top_action

    candidate_count = min(LOOKAHEAD_CANDIDATES, len(static_ranked))
    best_action = static_ranked[0][1]
    best_score = float("-inf")

    for index, (static_score, action) in enumerate(static_ranked):
        if index >= candidate_count:
            break

        from_pos, to_pos = divmod(action, BOARD_SIZE)
        attacker = board[from_pos]
        defender = board[to_pos]
        attacker_type = abs(attacker)
        defender_type = abs(defender)
        outcome = combat_outcome(attacker, defender)
        transition = apply_action_with_env(board, player_id, action)
        if transition["game_over"]:
            if transition["winner"] == player_id:
                return action
            if transition["winner"] == PlayerId.NONE:
                candidate_score = static_score - 3.0
            else:
                candidate_score = -10000.0
        else:
            after_board = transition["board"]
            after_eval = strategic_board_score(after_board, player_id) - current_eval
            candidate_score = static_score + max(-4.0, min(4.0, after_eval * 0.25))

            if defender_type == 2 and outcome == 0 and attacker_type >= 70:
                candidate_score -= 8.0
            if defender_type == 1 and attacker_type != 10:
                candidate_score -= 10.0

            if transition["next_player"] == opponent:
                opp_my_material = transition["opp_material"]
                opp_opp_material = transition["my_material"]
                opp_scored = top_scored_actions_v2(
                    after_board,
                    transition["legal_actions_next"],
                    opponent,
                    opp_my_material,
                    opp_opp_material,
                    limit=LOOKAHEAD_REPLIES,
                )
                worst_reply_eval = float("inf")
                for reply_score, reply in opp_scored:
                    reply_transition = apply_action_with_env(after_board, opponent, reply)
                    if reply_transition["game_over"] and reply_transition["winner"] == opponent:
                        worst_reply_eval = -10000.0
                        break

                    reply_board = reply_transition["board"]
                    if reply_transition["next_player"] != player_id:
                        reply_eval = strategic_board_score(reply_board, player_id)
                        worst_reply_eval = min(worst_reply_eval, reply_eval)
                        continue

                    counter_my_material = reply_transition["my_material"]
                    counter_opp_material = reply_transition["opp_material"]
                    counter_actions = top_scored_actions_v2(
                        reply_board,
                        reply_transition["legal_actions_next"],
                        player_id,
                        counter_my_material,
                        counter_opp_material,
                        limit=LOOKAHEAD_COUNTERS,
                    )
                    best_counter_eval = -10000.0
                    for _, counter in counter_actions:
                        counter_transition = apply_action_with_env(reply_board, player_id, counter)
                        if counter_transition["game_over"]:
                            if counter_transition["winner"] == player_id:
                                best_counter_eval = 10000.0
                                break
                            if counter_transition["winner"] == PlayerId.NONE:
                                best_counter_eval = max(best_counter_eval, strategic_board_score(counter_transition["board"], player_id) - 2.0)
                            else:
                                best_counter_eval = max(best_counter_eval, -10000.0)
                        else:
                            best_counter_eval = max(best_counter_eval, strategic_board_score(counter_transition["board"], player_id))

                    if best_counter_eval == -10000.0:
                        best_counter_eval = strategic_board_score(reply_board, player_id) - reply_score * 0.10
                    worst_reply_eval = min(worst_reply_eval, best_counter_eval)

                if worst_reply_eval != float("inf"):
                    lookahead_adjust = (worst_reply_eval - current_eval) * 0.30
                    candidate_score += max(-6.0, min(6.0, lookahead_adjust))

        candidate_score += random.random() * 1e-4
        if candidate_score > best_score:
            best_score = candidate_score
            best_action = action

    return best_action


class _SearchAbort(Exception):
    pass


@dataclass
class AlphaBetaAnalysis:
    best_action: int | None = None
    score: float = 0.0
    depth: int = 0
    nodes: int = 0
    qnodes: int = 0
    tt_hits: int = 0
    cutoffs: int = 0
    elapsed_ms: float = 0.0
    nps: float = 0.0
    pv: tuple[int, ...] = ()
    root_scores: dict[int, float] = field(default_factory=dict)
    completed: bool = True


@dataclass
class _TTEntry:
    depth: int
    score: float
    flag: str
    best_action: int | None


class AlphaBetaBot(BaseBot):
    MATE_SCORE = 100_000.0

    def __init__(
        self,
        name: str = "alphabeta",
        max_depth: int = 3,
        time_limit_ms: int = 150,
        node_limit: int = 100_000,
        quiescence_depth: int = 1,
    ):
        super().__init__(name)
        self.max_depth = max(1, int(max_depth))
        self.time_limit_ms = max(1, int(time_limit_ms))
        self.node_limit = max(1, int(node_limit))
        self.quiescence_depth = max(0, int(quiescence_depth))
        self.last_analysis = AlphaBetaAnalysis()
        self._tt: dict[tuple[tuple[int, ...], int], _TTEntry] = {}
        self._deadline = 0.0
        self._nodes = 0
        self._qnodes = 0
        self._tt_hits = 0
        self._cutoffs = 0

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        legal_actions = env_wrapper.env.get_legal_actions(player_id)
        if not legal_actions:
            return None
        board = tuple(env_wrapper.env.get_perfect_state())
        analysis = self.analyze(board, player_id, legal_actions)
        if analysis.best_action is None:
            return None
        return env_wrapper.action_to_policy(analysis.best_action, player_id)

    def analyze(
        self,
        board: tuple[int, ...] | list[int],
        player_id: int,
        legal_actions: list[int] | None = None,
    ) -> AlphaBetaAnalysis:
        board = tuple(board)
        if legal_actions is None:
            legal_actions = self._legal_actions(board, player_id)
        if not legal_actions:
            self.last_analysis = AlphaBetaAnalysis(best_action=None, score=-self.MATE_SCORE)
            return self.last_analysis

        self._tt.clear()
        self._nodes = self._qnodes = self._tt_hits = self._cutoffs = 0
        started = time.perf_counter()
        self._deadline = started + self.time_limit_ms / 1000.0
        best_action = legal_actions[0]
        best_score = -self.MATE_SCORE
        completed_depth = 0
        completed = True
        root_scores: dict[int, float] = {}

        for depth in range(1, self.max_depth + 1):
            try:
                score, action, depth_scores = self._search_root(
                    board, player_id, legal_actions, depth
                )
            except _SearchAbort:
                completed = False
                break
            best_score, best_action = score, action
            root_scores = depth_scores
            completed_depth = depth

        elapsed = time.perf_counter() - started
        pv = self._principal_variation(board, player_id, best_action, completed_depth)
        self.last_analysis = AlphaBetaAnalysis(
            best_action=best_action,
            score=best_score,
            depth=completed_depth,
            nodes=self._nodes,
            qnodes=self._qnodes,
            tt_hits=self._tt_hits,
            cutoffs=self._cutoffs,
            elapsed_ms=elapsed * 1000.0,
            nps=(self._nodes + self._qnodes) / max(elapsed, 1e-9),
            pv=tuple(pv),
            root_scores=root_scores,
            completed=completed,
        )
        return self.last_analysis

    def score_actions(
        self,
        board: tuple[int, ...] | list[int],
        legal_actions: list[int],
        player_id: int,
    ) -> dict[int, float]:
        return dict(self.analyze(board, player_id, legal_actions).root_scores)

    def _check_limits(self) -> None:
        if self._nodes + self._qnodes >= self.node_limit:
            raise _SearchAbort
        if time.perf_counter() >= self._deadline:
            raise _SearchAbort

    def _legal_actions(self, board: tuple[int, ...], player_id: int) -> list[int]:
        env = get_scratch_env()
        env.load_board_state(list(board), player_id)
        return list(env.get_legal_actions(player_id))

    def _ordered_actions(
        self,
        board: tuple[int, ...],
        actions: list[int],
        player_id: int,
        tt_action: int | None = None,
    ) -> list[int]:
        my_material = board_material(board, player_id)
        opponent = other_player(player_id)
        opp_material = board_material(board, opponent)
        ranked = sorted(
            actions,
            key=lambda action: (
                action == tt_action,
                board[action % BOARD_SIZE] != 0,
                score_heuristic_action(action, board, player_id, my_material, opp_material),
            ),
            reverse=True,
        )
        return ranked

    def _terminal_score(self, winner: int, player_id: int, ply: int) -> float:
        if winner == PlayerId.NONE:
            return 0.0
        if winner == player_id:
            return self.MATE_SCORE - ply
        return -self.MATE_SCORE + ply

    def _search_root(
        self,
        board: tuple[int, ...],
        player_id: int,
        legal_actions: list[int],
        depth: int,
    ) -> tuple[float, int, dict[int, float]]:
        alpha = -self.MATE_SCORE
        beta = self.MATE_SCORE
        best_action = legal_actions[0]
        best_score = -self.MATE_SCORE
        root_scores: dict[int, float] = {}
        root_entry = self._tt.get((board, int(player_id)))
        tt_action = root_entry.best_action if root_entry else None
        for action in self._ordered_actions(board, legal_actions, player_id, tt_action):
            self._check_limits()
            transition = apply_action_with_env(board, player_id, action)
            if transition["game_over"]:
                score = self._terminal_score(transition["winner"], player_id, 1)
            else:
                score = -self._negamax(
                    tuple(transition["board"]),
                    transition["next_player"],
                    depth - 1,
                    -beta,
                    -alpha,
                    1,
                )
            root_scores[int(action)] = float(score)
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, score)
        self._tt[(board, int(player_id))] = _TTEntry(depth, best_score, "EXACT", best_action)
        return best_score, best_action, root_scores

    def _negamax(
        self,
        board: tuple[int, ...],
        player_id: int,
        depth: int,
        alpha: float,
        beta: float,
        ply: int,
    ) -> float:
        self._nodes += 1
        self._check_limits()
        if depth <= 0:
            return self._quiescence(board, player_id, alpha, beta, ply, self.quiescence_depth)

        key = (board, int(player_id))
        alpha_original = alpha
        entry = self._tt.get(key)
        if entry and entry.depth >= depth:
            self._tt_hits += 1
            if entry.flag == "EXACT":
                return entry.score
            if entry.flag == "LOWER":
                alpha = max(alpha, entry.score)
            elif entry.flag == "UPPER":
                beta = min(beta, entry.score)
            if alpha >= beta:
                return entry.score

        actions = self._legal_actions(board, player_id)
        if not actions:
            return -self.MATE_SCORE + ply

        best_score = -self.MATE_SCORE
        best_action: int | None = None
        tt_action = entry.best_action if entry else None
        for action in self._ordered_actions(board, actions, player_id, tt_action):
            transition = apply_action_with_env(board, player_id, action)
            if transition["game_over"]:
                score = self._terminal_score(transition["winner"], player_id, ply + 1)
            else:
                score = -self._negamax(
                    tuple(transition["board"]),
                    transition["next_player"],
                    depth - 1,
                    -beta,
                    -alpha,
                    ply + 1,
                )
            if score > best_score:
                best_score, best_action = score, action
            alpha = max(alpha, score)
            if alpha >= beta:
                self._cutoffs += 1
                break

        flag = "EXACT"
        if best_score <= alpha_original:
            flag = "UPPER"
        elif best_score >= beta:
            flag = "LOWER"
        self._tt[key] = _TTEntry(depth, best_score, flag, best_action)
        return best_score

    def _quiescence(
        self,
        board: tuple[int, ...],
        player_id: int,
        alpha: float,
        beta: float,
        ply: int,
        depth: int,
    ) -> float:
        self._qnodes += 1
        self._check_limits()
        stand_pat = strategic_board_score(board, player_id)
        if stand_pat >= beta:
            return beta
        alpha = max(alpha, stand_pat)
        if depth <= 0:
            return alpha

        captures = [
            action for action in self._legal_actions(board, player_id)
            if board[action % BOARD_SIZE] != 0
        ]
        for action in self._ordered_actions(board, captures, player_id):
            transition = apply_action_with_env(board, player_id, action)
            if transition["game_over"]:
                score = self._terminal_score(transition["winner"], player_id, ply + 1)
            else:
                score = -self._quiescence(
                    tuple(transition["board"]),
                    transition["next_player"],
                    -beta,
                    -alpha,
                    ply + 1,
                    depth - 1,
                )
            if score >= beta:
                self._cutoffs += 1
                return beta
            alpha = max(alpha, score)
        return alpha

    def _principal_variation(
        self,
        board: tuple[int, ...],
        player_id: int,
        first_action: int,
        depth: int,
    ) -> list[int]:
        pv: list[int] = []
        action: int | None = first_action
        for _ in range(depth):
            if action is None:
                break
            pv.append(action)
            transition = apply_action_with_env(board, player_id, action)
            if transition["game_over"]:
                break
            board = tuple(transition["board"])
            player_id = transition["next_player"]
            entry = self._tt.get((board, int(player_id)))
            action = entry.best_action if entry else None
        return pv


class PPOBot(BaseBot):
    def __init__(self, model_path: str, device: torch.device, name: str | None = None):
        import torch
        from junqi_rl.model import JunqiActorCritic

        resolved_name = name or os.path.basename(model_path)
        super().__init__(resolved_name)
        self.torch = torch
        self.device = device
        self.model = JunqiActorCritic().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def select_action(self, env_wrapper, player_id: int) -> int | None:
        obs = env_wrapper._observation(player_id)
        info = env_wrapper._info(player_id)

        with self.torch.no_grad():
            state = self.torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask = self.torch.from_numpy(info["action_mask"].astype("float32")).unsqueeze(0).to(self.device)
            action, _, _, _ = self.model.get_action_and_value(
                state,
                mask,
                deterministic=True,
            )
        return int(action.item())


@dataclass
class MatchResult:
    red_wins: int = 0
    black_wins: int = 0
    draws: int = 0
    total_turns: int = 0

    @property
    def games(self) -> int:
        return self.red_wins + self.black_wins + self.draws


def build_bot(spec: str, device: torch.device, name: str | None = None) -> BaseBot:
    lowered = spec.lower()
    if lowered == "human":
        return HumanBot(name or "human")
    if lowered == "random":
        return RandomBot(name or "random")
    if lowered == "greedy":
        return GreedyMaterialBot(name or "greedy")
    if lowered == "heuristic_v1":
        return HeuristicTacticalBot(name or "heuristic_v1")
    if lowered == "heuristic_v2":
        return HeuristicStrategicBot(name or "heuristic_v2")
    if lowered == "heuristic_v3":
        return HeuristicProductionBot(name or "heuristic_v3")
    if lowered == "heuristic_trial_r1":
        return HeuristicTrialBot(name or lowered, baseline_credit=0.5)
    if lowered == "heuristic_trial_r2":
        return HeuristicTrialBot(name or lowered, candidate_count=8, baseline_credit=0.5)
    if lowered == "heuristic_trial_r3":
        return HeuristicTrialBot(name or lowered, candidate_count=6, baseline_credit=0.5, reply_position_weight=0.20)
    if lowered == "heuristic":
        return HeuristicProductionBot(name or "heuristic")
    if lowered == "lookahead_experimental":
        return HeuristicLookaheadBot(name or "lookahead_experimental")
    if lowered in {"ab", "alphabeta", "ab3"}:
        return AlphaBetaBot(name or "alphabeta", max_depth=3)
    if lowered == "ab2":
        # Production search stays inside the same single-core 100 steps/s
        # interaction budget as the heuristic bots.  Use ab/ab3 for the
        # slower 150 ms analysis mode exposed by the GUI.
        return AlphaBetaBot(
            name or "alphabeta_d2_fast",
            max_depth=2,
            time_limit_ms=8,
            node_limit=10_000,
        )
    if not os.path.exists(spec):
        raise FileNotFoundError(f"Model not found: {spec}")
    return PPOBot(spec, device, name=name)
