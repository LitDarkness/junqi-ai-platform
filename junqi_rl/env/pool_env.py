import glob
import os
import random

import numpy as np
import torch

from junqi_rl.bots import choose_heuristic_action_v3_fast
from junqi_rl.env.env_wrapper import JunqiBaseEnv
from junqi_rl.model import JunqiActorCritic
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig
from JunqiEnv.junqi import CombatResult, PlayerId


class JunqiSelfPlayEnv(JunqiBaseEnv):
    def __init__(
        self,
        pool_dir: str = "pool",
        max_steps: int = 400,
        reward_config: RewardConfig | None = None,
        opponent_device: torch.device | None = None,
        opponent_config: OpponentPoolConfig | None = None,
    ):
        super().__init__(max_steps=max_steps, reward_config=reward_config)
        self.pool_dir = pool_dir
        self.device = opponent_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.opponent_model = JunqiActorCritic().to(self.device)
        self.opponent_model.eval()
        self.opponent_config = (opponent_config or OpponentPoolConfig()).normalized()

        self.learning_player = PlayerId.RED
        self.opponent_player = PlayerId.BLACK
        self.opponent_kind = "heuristic"

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.learning_player = random.choice([PlayerId.RED, PlayerId.BLACK])
        self.opponent_player = PlayerId.BLACK if self.learning_player == PlayerId.RED else PlayerId.RED
        self._current_player = self.env.get_current_player()
        self._load_opponent()

        if self._current_player == self.opponent_player:
            self._opponent_turn()

        self._current_player = self.learning_player
        self._prev_material_delta = self._material_delta(self.learning_player)
        return self._observation(self.learning_player), self._info(self.learning_player)

    def step(self, action: int):
        self._current_player = self.learning_player
        mask = self._mask(self.learning_player)
        if not mask[action]:
            return (
                self._observation(self.learning_player),
                self.reward_config.invalid_action_penalty,
                True,
                False,
                self._info(self.learning_player),
            )

        env_action = self.action_to_env(int(action), self.learning_player)
        from_pos, to_pos = divmod(env_action, 60)
        result = self.env.step(from_pos, to_pos)
        if result == CombatResult.MOVE_ONLY:
            self._no_combat_steps += 1
        else:
            self._no_combat_steps = 0

        while not self.env.is_game_over() and self.env.get_current_player() == self.opponent_player:
            self._opponent_turn()

        reward, terminated = self._transition_reward(self.learning_player)
        truncated = False
        if not terminated and self.reward_config.no_combat_limit > 0 and self._no_combat_steps >= self.reward_config.no_combat_limit:
            truncated = True
            reward = self.reward_config.no_combat_draw_reward
        elif not terminated and self.env.get_turn_count() >= self.max_steps:
            truncated = True
            reward = self._truncation_reward(self.learning_player)

        self._current_player = self.learning_player
        if not terminated and not truncated:
            self._prev_material_delta = self._material_delta(self.learning_player)

        return (
            self._observation(self.learning_player),
            reward,
            terminated,
            truncated,
            self._info(self.learning_player),
        )

    def _load_opponent(self):
        os.makedirs(self.pool_dir, exist_ok=True)
        latest_path = os.path.join(self.pool_dir, "latest.pth")
        snapshot_paths = sorted(glob.glob(os.path.join(self.pool_dir, "model_*.pth")))
        roll = random.random()

        if roll < self.opponent_config.heuristic_prob:
            self.opponent_kind = "heuristic"
            return

        selected_path = None
        threshold_latest = self.opponent_config.heuristic_prob + self.opponent_config.latest_prob
        if roll < threshold_latest and os.path.exists(latest_path):
            selected_path = latest_path
        elif snapshot_paths:
            selected_path = random.choice(snapshot_paths)
        elif os.path.exists(latest_path):
            selected_path = latest_path

        if selected_path is None:
            self.opponent_kind = "heuristic"
            return

        try:
            state_dict = torch.load(selected_path, map_location=self.device)
            self.opponent_model.load_state_dict(state_dict)
            self.opponent_kind = "model"
        except Exception:
            self.opponent_kind = "heuristic"

    def _opponent_turn(self):
        legal_actions = self.env.get_legal_actions(self.opponent_player)
        if not legal_actions:
            return

        if self.opponent_kind == "heuristic":
            board = self.env.get_perfect_state()
            my_material = self.env.get_material_score(self.opponent_player)
            opp_material = self.env.get_material_score(self.learning_player)
            action = choose_heuristic_action_v3_fast(board, legal_actions, self.opponent_player, my_material, opp_material)
            result = self.env.step(action // 60, action % 60)
            if result == CombatResult.MOVE_ONLY:
                self._no_combat_steps += 1
            else:
                self._no_combat_steps = 0
            return

        obs = self._observation(self.opponent_player)
        mask = self._mask(self.opponent_player).astype(np.float32, copy=False)

        with torch.no_grad():
            state_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
            mask_tensor = torch.from_numpy(mask).unsqueeze(0).to(self.device)
            action, _, _, _ = self.opponent_model.get_action_and_value(
                state_tensor,
                mask_tensor,
                deterministic=False,
            )
        chosen = self.action_to_env(int(action.item()), self.opponent_player)
        result = self.env.step(chosen // 60, chosen % 60)
        if result == CombatResult.MOVE_ONLY:
            self._no_combat_steps += 1
        else:
            self._no_combat_steps = 0


JunqiPoolEnv = JunqiSelfPlayEnv
