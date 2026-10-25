from __future__ import annotations

import numpy as np


BOARD_ROWS = 12
BOARD_COLS = 5
BOARD_SIZE = BOARD_ROWS * BOARD_COLS
ACTION_SPACE_SIZE = BOARD_SIZE * BOARD_SIZE

FLIP_POSITIONS = np.arange(BOARD_SIZE - 1, -1, -1, dtype=np.int32)
FLIP_ACTIONS = np.empty(ACTION_SPACE_SIZE, dtype=np.int32)
for action in range(ACTION_SPACE_SIZE):
    src, dst = divmod(action, BOARD_SIZE)
    FLIP_ACTIONS[action] = int(FLIP_POSITIONS[src] * BOARD_SIZE + FLIP_POSITIONS[dst])


def flip_position(index: int) -> int:
    return int(FLIP_POSITIONS[index])


def flip_action(action: int) -> int:
    return int(FLIP_ACTIONS[action])


def canonicalize_features(features: np.ndarray, is_black: bool) -> np.ndarray:
    if not is_black:
        return features
    if features.ndim == 3:
        return np.flip(features, axis=(1, 2)).copy()
    if features.ndim == 4:
        return np.flip(features, axis=(2, 3)).copy()
    raise ValueError(f"Unsupported feature shape: {features.shape}")


def canonicalize_action_mask(mask: np.ndarray, is_black: bool) -> np.ndarray:
    if not is_black:
        return mask
    return mask[..., FLIP_ACTIONS].copy()


def policy_action_to_env(action: int, is_black: bool) -> int:
    return flip_action(action) if is_black else int(action)


def env_action_to_policy(action: int, is_black: bool) -> int:
    return flip_action(action) if is_black else int(action)


def policy_actions_to_env(actions: np.ndarray, is_black: np.ndarray) -> np.ndarray:
    mapped = np.asarray(actions, dtype=np.int32).copy()
    black_indices = np.flatnonzero(is_black)
    if black_indices.size > 0:
        mapped[black_indices] = FLIP_ACTIONS[mapped[black_indices]]
    return mapped


def canonicalize_action_masks_batch(masks: np.ndarray, players: np.ndarray, black_player: int) -> np.ndarray:
    canonical = np.asarray(masks)
    black_indices = np.flatnonzero(players == black_player)
    if black_indices.size > 0:
        canonical = canonical.copy()
        canonical[black_indices] = canonical[black_indices][:, FLIP_ACTIONS]
    return canonical


def canonicalize_features_batch(features: np.ndarray, players: np.ndarray, black_player: int) -> np.ndarray:
    canonical = np.asarray(features)
    black_indices = np.flatnonzero(players == black_player)
    if black_indices.size > 0:
        canonical = canonical.copy()
        canonical[black_indices] = canonical[black_indices][:, :, ::-1, ::-1]
    return canonical
