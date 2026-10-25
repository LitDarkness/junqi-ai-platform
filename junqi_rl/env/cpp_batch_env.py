from __future__ import annotations

import os
import sys

import numpy as np


BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import JunqiEnv


class CppBatchJunqiEnv:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.backend = JunqiEnv.BatchJunqiEnvironment(num_envs)

    def reset_random_all(self):
        self.backend.reset_random_all()

    def reset_indices(self, indices: np.ndarray | list[int]):
        if isinstance(indices, np.ndarray):
            payload = indices.astype(np.int32).tolist()
        else:
            payload = [int(index) for index in indices]
        self.backend.reset_indices(payload)

    def current_players(self) -> np.ndarray:
        return np.asarray(self.backend.get_current_players(), dtype=np.int32)

    def turn_counts(self) -> np.ndarray:
        return np.asarray(self.backend.get_turn_counts(), dtype=np.int32)

    def winners(self) -> np.ndarray:
        return np.asarray(self.backend.get_winners(), dtype=np.int32)

    def done_flags(self) -> np.ndarray:
        return np.asarray(self.backend.get_done_flags(), dtype=np.uint8)

    def perfect_states(self) -> np.ndarray:
        return np.asarray(
            self.backend.get_perfect_states_batch(),
            dtype=np.int32,
        )

    def feature_planes(self, player_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.backend.get_feature_planes_batch(player_ids.astype(np.int32).tolist()),
            dtype=np.float32,
        )

    def legal_action_masks(self, player_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.backend.get_legal_action_masks_batch(player_ids.astype(np.int32).tolist()),
            dtype=np.uint8,
        )

    def material_scores(self, player_ids: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.backend.get_material_scores_batch(player_ids.astype(np.int32).tolist()),
            dtype=np.float32,
        )

    def step_actions(self, action_indices: np.ndarray) -> np.ndarray:
        return np.asarray(
            self.backend.step_actions(action_indices.astype(np.int32).tolist()),
            dtype=np.int32,
        )

    def step_indexed_actions(self, indices: np.ndarray | list[int], action_indices: np.ndarray | list[int]) -> np.ndarray:
        if isinstance(indices, np.ndarray):
            index_payload = indices.astype(np.int32).tolist()
        else:
            index_payload = [int(index) for index in indices]
        if isinstance(action_indices, np.ndarray):
            action_payload = action_indices.astype(np.int32).tolist()
        else:
            action_payload = [int(action) for action in action_indices]
        return np.asarray(
            self.backend.step_indexed_actions(index_payload, action_payload),
            dtype=np.int32,
        )
