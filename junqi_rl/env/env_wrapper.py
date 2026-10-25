import os
import sys

import gymnasium as gym
import numpy as np
from gymnasium import spaces


BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import JunqiEnv
from JunqiEnv.junqi import CombatResult, PlayerId
from junqi_rl.perspective import (
    canonicalize_action_mask,
    canonicalize_features,
    env_action_to_policy,
    policy_action_to_env,
)
from junqi_rl.training_config import RewardConfig

FEATURE_CHANNELS = 14
ACTION_SPACE_SIZE = 3600


class JunqiBaseEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, max_steps: int = 400, reward_config: RewardConfig | None = None):
        super().__init__()
        self.env = JunqiEnv.JunqiEnvironment()
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()

        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(FEATURE_CHANNELS, 12, 5),
            dtype=np.float32,
        )

        self._current_player = PlayerId.RED
        self._prev_material_delta = 0.0
        self._no_combat_steps = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.env.reset_random()
        self._current_player = self.env.get_current_player()
        self._prev_material_delta = self._material_delta(self._current_player)
        self._no_combat_steps = 0
        return self._observation(self._current_player), self._info(self._current_player)

    def step(self, action: int):
        acting_player = self._current_player
        policy_action = int(action)
        env_action = policy_action_to_env(policy_action, acting_player == PlayerId.BLACK)
        from_pos, to_pos = divmod(env_action, 60)
        mask = self._mask(acting_player)

        if not mask[policy_action]:
            return (
                self._observation(acting_player),
                self.reward_config.invalid_action_penalty,
                True,
                False,
                self._info(acting_player),
            )

        result = self.env.step(from_pos, to_pos)
        if result == CombatResult.MOVE_ONLY:
            self._no_combat_steps += 1
        else:
            self._no_combat_steps = 0
        reward, terminated = self._transition_reward(acting_player)

        truncated = False
        if not terminated and self.reward_config.no_combat_limit > 0 and self._no_combat_steps >= self.reward_config.no_combat_limit:
            truncated = True
            reward = self.reward_config.no_combat_draw_reward
        elif not terminated and self.env.get_turn_count() >= self.max_steps:
            truncated = True
            reward = self._truncation_reward(acting_player)

        self._current_player = self.env.get_current_player()
        if not terminated and not truncated:
            self._prev_material_delta = self._material_delta(self._current_player)

        return (
            self._observation(self._current_player),
            reward,
            terminated,
            truncated,
            self._info(self._current_player),
        )

    def render(self):
        return self.env.render()

    def close(self):
        return None

    def _observation(self, player_id: int) -> np.ndarray:
        planes = np.asarray(self.env.get_feature_planes(player_id), dtype=np.float32)
        planes = planes.reshape(FEATURE_CHANNELS, 12, 5)
        return canonicalize_features(planes, player_id == PlayerId.BLACK)

    def _mask(self, player_id: int) -> np.ndarray:
        mask = np.asarray(self.env.get_legal_action_mask(player_id), dtype=np.bool_)
        return canonicalize_action_mask(mask, player_id == PlayerId.BLACK)

    def _info(self, player_id: int) -> dict:
        mask = self._mask(player_id)
        if not mask.any():
            mask = mask.copy()
            mask[0] = True
        return {"action_mask": mask}

    def action_to_env(self, action: int, player_id: int) -> int:
        return policy_action_to_env(action, player_id == PlayerId.BLACK)

    def action_to_policy(self, action: int, player_id: int) -> int:
        return env_action_to_policy(action, player_id == PlayerId.BLACK)

    def _material_delta(self, player_id: int) -> float:
        opponent = PlayerId.BLACK if player_id == PlayerId.RED else PlayerId.RED
        return self.env.get_material_score(player_id) - self.env.get_material_score(opponent)

    def _transition_reward(self, acting_player: int) -> tuple[float, bool]:
        if self.env.is_game_over():
            winner = self.env.get_winner()
            if winner == acting_player:
                return self.reward_config.win_reward, True
            if winner == PlayerId.NONE:
                return self.reward_config.draw_reward, True
            return self.reward_config.loss_reward, True

        material_delta = self._material_delta(acting_player)
        shaped = (material_delta - self._prev_material_delta) * self.reward_config.material_scale
        return shaped + self.reward_config.step_penalty, False

    def _truncation_reward(self, acting_player: int) -> float:
        if self.reward_config.turn_limit_mode == "draw":
            return self.reward_config.draw_reward
        material_delta = self._material_delta(acting_player)
        return (
            self.reward_config.truncation_penalty
            + self.reward_config.material_tiebreak_scale * np.sign(material_delta)
        )


class JunqiPPOEnv(JunqiBaseEnv):
    pass
