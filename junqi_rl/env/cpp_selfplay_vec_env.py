from __future__ import annotations

import glob
import os
import random
import sys
from dataclasses import dataclass

import numpy as np

BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../build"))
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

from JunqiEnv.junqi import PlayerId

from junqi_rl.bots import choose_heuristic_action_v3_fast
from junqi_rl.env.cpp_batch_env import CppBatchJunqiEnv
from junqi_rl.perspective import (
    canonicalize_action_masks_batch,
    canonicalize_features_batch,
    policy_actions_to_env,
)
from junqi_rl.training_config import OpponentPoolConfig, RewardConfig


@dataclass
class OpponentSpec:
    kind: str
    key: str


class ModelCache:
    def __init__(self, device):
        self.device = device
        self.cache: dict[str, object] = {}
        self.mtimes: dict[str, float] = {}

    def get(self, model_path: str):
        import torch
        from junqi_rl.model import JunqiActorCritic
        current_mtime = os.path.getmtime(model_path)
        model = self.cache.get(model_path)
        cached_mtime = self.mtimes.get(model_path)
        if model is None or cached_mtime != current_mtime:
            model = JunqiActorCritic().to(self.device)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            self.cache[model_path] = model
            self.mtimes[model_path] = current_mtime
        return model


class CppSelfPlayVecEnv:
    def __init__(
        self,
        num_envs: int,
        pool_dir: str = "pool",
        max_steps: int = 400,
        reward_config: RewardConfig | None = None,
        opponent_config: OpponentPoolConfig | None = None,
        opponent_device: torch.device | None = None,
        opponent_inference=None,
    ):
        self.num_envs = num_envs
        self.pool_dir = pool_dir
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig.preset("shaped")
        self.opponent_config = (opponent_config or OpponentPoolConfig()).normalized()

        self.backend = CppBatchJunqiEnv(num_envs)
        if opponent_inference is None:
            import torch
            self.device = opponent_device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = None
        self.opponent_inference = opponent_inference
        self.model_cache = None if opponent_inference is not None else ModelCache(self.device)

        self.learning_players = np.full(num_envs, PlayerId.RED, dtype=np.int32)
        self.opponent_players = np.full(num_envs, PlayerId.BLACK, dtype=np.int32)
        self.prev_material_delta = np.zeros(num_envs, dtype=np.float32)
        self.episode_returns = np.zeros(num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(num_envs, dtype=np.int32)
        self.no_combat_steps = np.zeros(num_envs, dtype=np.int32)
        self.opponents: list[OpponentSpec] = [OpponentSpec("heuristic", "heuristic") for _ in range(num_envs)]
        # The mask returned in infos is the mask consumed by the next call to
        # step().  Cache it instead of asking C++ to regenerate the same 3600
        # entries at the beginning of every transition.
        self._learning_masks: np.ndarray | None = None

    def reset(self):
        self.backend.reset_random_all()
        all_indices = np.arange(self.num_envs, dtype=np.int32)
        self._prepare_indices(all_indices)

        obs = self._canonical_features(self.learning_players)
        self._learning_masks = self._canonical_masks(self.learning_players)
        infos = tuple(self._info_for_indices(all_indices, self._learning_masks))
        self.episode_returns.fill(0.0)
        self.episode_lengths.fill(0)
        self.no_combat_steps.fill(0)
        return obs, infos

    def step(self, actions: np.ndarray):
        actions = np.asarray(actions, dtype=np.int32)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.zeros(self.num_envs, dtype=np.bool_)
        infos = [None] * self.num_envs

        current_players = self.backend.current_players()
        if self._learning_masks is None:
            self._learning_masks = self._canonical_masks(self.learning_players)
        masks = self._learning_masks
        env_actions = policy_actions_to_env(actions, self.learning_players == PlayerId.BLACK)

        invalid = current_players != self.learning_players
        invalid |= masks[np.arange(self.num_envs), actions] == 0

        valid_indices = np.flatnonzero(~invalid)
        if valid_indices.size > 0:
            results = self.backend.step_indexed_actions(valid_indices, env_actions[valid_indices])
            self._update_no_combat(valid_indices, results)
            self._advance_until_learning_turn(valid_indices)

        term_reward, term_done = self._compute_rewards()
        rewards += term_reward
        dones |= term_done

        rewards[invalid] = self.reward_config.invalid_action_penalty
        dones[invalid] = True

        self.episode_returns += rewards
        self.episode_lengths += 1

        reset_indices = np.flatnonzero(dones)
        if reset_indices.size > 0:
            winners = self.backend.winners()
            turn_counts = self.backend.turn_counts()
            for env_index in reset_indices.tolist():
                infos[env_index] = {
                    "episode": {
                        "r": float(self.episode_returns[env_index]),
                        "l": int(self.episode_lengths[env_index]),
                    },
                    "terminal_winner": int(winners[env_index]),
                    "terminal_turns": int(turn_counts[env_index]),
                }

            self.backend.reset_indices(reset_indices)
            self._prepare_indices(reset_indices)
            self.episode_returns[reset_indices] = 0.0
            self.episode_lengths[reset_indices] = 0
            self.no_combat_steps[reset_indices] = 0

        obs = self._canonical_features(self.learning_players)
        self._learning_masks = self._canonical_masks(self.learning_players)
        refreshed_infos = self._info_for_indices(
            np.arange(self.num_envs, dtype=np.int32), self._learning_masks
        )
        for env_index in range(self.num_envs):
            payload = refreshed_infos[env_index]
            if infos[env_index] is not None:
                infos[env_index]["action_mask"] = payload["action_mask"]
            else:
                infos[env_index] = payload

        return obs, rewards, dones, tuple(infos)

    def close(self):
        return None

    def _configure_indices(self, indices: np.ndarray):
        for env_index in indices.tolist():
            learning_player = random.choice([PlayerId.RED, PlayerId.BLACK])
            opponent_player = PlayerId.BLACK if learning_player == PlayerId.RED else PlayerId.RED
            self.learning_players[env_index] = learning_player
            self.opponent_players[env_index] = opponent_player
            self.opponents[env_index] = self._sample_opponent()

        self.prev_material_delta[indices] = self._material_delta(indices)

    def _prepare_indices(self, indices: np.ndarray):
        pending = indices.astype(np.int32, copy=False)
        while pending.size > 0:
            self._configure_indices(pending)
            self._advance_until_learning_turn(pending)
            done = self.backend.done_flags().astype(bool)
            pending = pending[done[pending]]
            if pending.size > 0:
                self.backend.reset_indices(pending)
        # Opening opponent moves belong to setup, not to the learner's first
        # transition reward.  Freeze the baseline only after every requested
        # environment is ready for the learner.
        self.prev_material_delta[indices] = self._material_delta(indices)

    def _sample_opponent(self) -> OpponentSpec:
        os.makedirs(self.pool_dir, exist_ok=True)
        latest_path = os.path.join(self.pool_dir, self.opponent_config.latest_name)
        snapshots = sorted(glob.glob(os.path.join(self.pool_dir, f"{self.opponent_config.snapshot_prefix}*.pth")))
        roll = random.random()

        if roll < self.opponent_config.random_prob:
            return OpponentSpec("random", "random")

        threshold_heuristic = self.opponent_config.random_prob + self.opponent_config.heuristic_prob
        if roll < threshold_heuristic:
            return OpponentSpec("heuristic", "heuristic")

        threshold_latest = threshold_heuristic + self.opponent_config.latest_prob
        if roll < threshold_latest and os.path.exists(latest_path):
            return OpponentSpec("model", latest_path)

        historical = [path for path in snapshots if os.path.basename(path) != self.opponent_config.latest_name]
        if historical:
            return OpponentSpec("model", random.choice(historical))
        if os.path.exists(latest_path):
            return OpponentSpec("model", latest_path)
        return OpponentSpec("heuristic", "heuristic")

    def _advance_until_learning_turn(self, candidate_indices: np.ndarray):
        while True:
            done = self.backend.done_flags().astype(bool)
            current_players = self.backend.current_players()
            active = candidate_indices[
                (~done[candidate_indices]) &
                (current_players[candidate_indices] == self.opponent_players[candidate_indices])
            ]
            if active.size == 0:
                break
            self._run_opponent_turn(active)

    def _run_opponent_turn(self, indices: np.ndarray):
        all_masks = self._canonical_masks(self.opponent_players)

        random_indices = [idx for idx in indices.tolist() if self.opponents[idx].kind == "random"]
        if random_indices:
            random_actions = self._sample_random_actions(all_masks[random_indices])
            random_env_actions = policy_actions_to_env(
                random_actions,
                self.opponent_players[random_indices] == PlayerId.BLACK,
            )
            results = self.backend.step_indexed_actions(random_indices, random_env_actions)
            self._update_no_combat(np.asarray(random_indices, dtype=np.int32), results)

        heuristic_indices = [idx for idx in indices.tolist() if self.opponents[idx].kind == "heuristic"]
        if heuristic_indices:
            boards = self.backend.perfect_states()
            my_scores = self.backend.material_scores(self.opponent_players)
            opp_scores = self.backend.material_scores(self.learning_players)
            heuristic_actions = []
            for idx in heuristic_indices:
                legal_actions = np.flatnonzero(all_masks[idx]).astype(np.int32).tolist()
                heuristic_actions.append(
                    choose_heuristic_action_v3_fast(
                        boards[idx].tolist(),
                        legal_actions,
                        int(self.opponent_players[idx]),
                        float(my_scores[idx]),
                        float(opp_scores[idx]),
                    )
                )
            heuristic_actions = np.asarray(heuristic_actions, dtype=np.int32)
            heuristic_env_actions = policy_actions_to_env(
                heuristic_actions,
                self.opponent_players[heuristic_indices] == PlayerId.BLACK,
            )
            results = self.backend.step_indexed_actions(heuristic_indices, heuristic_env_actions)
            self._update_no_combat(np.asarray(heuristic_indices, dtype=np.int32), results)

        grouped: dict[str, list[int]] = {}
        for idx in indices.tolist():
            spec = self.opponents[idx]
            if spec.kind == "model":
                grouped.setdefault(spec.key, []).append(idx)

        if not grouped:
            return

        all_features = self._canonical_features(self.opponent_players)
        for model_path, model_indices in grouped.items():
            if self.opponent_inference is not None:
                selected_actions = np.asarray(
                    self.opponent_inference(
                        model_path,
                        all_features[model_indices],
                        all_masks[model_indices],
                    ),
                    dtype=np.int32,
                )
            else:
                import torch
                model = self.model_cache.get(model_path)
                state = torch.from_numpy(all_features[model_indices]).to(self.device)
                mask = torch.from_numpy(all_masks[model_indices].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    action, _, _, _ = model.get_action_and_value(state, mask, deterministic=False)
                selected_actions = action.cpu().numpy()
            env_actions = policy_actions_to_env(
                selected_actions,
                self.opponent_players[model_indices] == PlayerId.BLACK,
            )
            results = self.backend.step_indexed_actions(model_indices, env_actions)
            self._update_no_combat(np.asarray(model_indices, dtype=np.int32), results)

    def _sample_random_actions(self, masks: np.ndarray) -> np.ndarray:
        noise = np.random.random(masks.shape)
        scores = np.where(masks > 0, noise, -1.0)
        return scores.argmax(axis=1).astype(np.int32)

    def _material_delta(self, indices: np.ndarray) -> np.ndarray:
        my_scores = self.backend.material_scores(self.learning_players)
        opp_scores = self.backend.material_scores(self.opponent_players)
        return (my_scores - opp_scores)[indices]

    def _compute_rewards(self) -> tuple[np.ndarray, np.ndarray]:
        done = self.backend.done_flags().astype(bool)
        winners = self.backend.winners()
        turn_counts = self.backend.turn_counts()
        rewards = np.zeros(self.num_envs, dtype=np.float32)

        if self.reward_config.mode in {"shaped", "hybrid"}:
            current_delta = self.backend.material_scores(self.learning_players) - self.backend.material_scores(self.opponent_players)
            rewards += (current_delta - self.prev_material_delta) * self.reward_config.material_scale
            rewards += self.reward_config.step_penalty
            # Advance the baseline only after this transition's delta has been
            # consumed.  Reset environments overwrite their entries during
            # _prepare_indices later in step().
            self.prev_material_delta[:] = current_delta

        terminal = done.copy()
        if terminal.any():
            win_mask = terminal & (winners == self.learning_players)
            draw_mask = terminal & (winners == PlayerId.NONE)
            loss_mask = terminal & (~win_mask) & (~draw_mask)
            rewards[win_mask] = self.reward_config.win_reward
            rewards[draw_mask] = self.reward_config.draw_reward
            rewards[loss_mask] = self.reward_config.loss_reward

        no_combat_done = np.zeros(self.num_envs, dtype=np.bool_)
        if self.reward_config.no_combat_limit > 0:
            no_combat_done = (~done) & (self.no_combat_steps >= self.reward_config.no_combat_limit)
            if no_combat_done.any():
                terminal[no_combat_done] = True
                rewards[no_combat_done] = self.reward_config.no_combat_draw_reward

        truncated = (~terminal) & (turn_counts >= self.max_steps)
        if truncated.any():
            terminal[truncated] = True
            if self.reward_config.turn_limit_mode == "draw":
                rewards[truncated] = self.reward_config.draw_reward
            elif self.reward_config.mode in {"shaped", "hybrid"}:
                material_delta = self.backend.material_scores(self.learning_players) - self.backend.material_scores(self.opponent_players)
                rewards[truncated] += self.reward_config.truncation_penalty
                rewards[truncated] += self.reward_config.material_tiebreak_scale * np.sign(material_delta[truncated])
            else:
                rewards[truncated] = self.reward_config.draw_reward

        return rewards, terminal

    def _info_for_indices(self, indices: np.ndarray, masks: np.ndarray | None = None):
        if masks is None:
            masks = self._canonical_masks(self.learning_players)
        payload = []
        for env_index in indices.tolist():
            mask = masks[env_index].astype(np.bool_)
            if not mask.any():
                mask = mask.copy()
                mask[0] = True
            payload.append({"action_mask": mask})
        return payload

    def _canonical_features(self, players: np.ndarray) -> np.ndarray:
        features = self.backend.feature_planes(players)
        return canonicalize_features_batch(features, players, PlayerId.BLACK)

    def _canonical_masks(self, players: np.ndarray) -> np.ndarray:
        masks = self.backend.legal_action_masks(players)
        return canonicalize_action_masks_batch(masks, players, PlayerId.BLACK)

    def _update_no_combat(self, indices: np.ndarray, results: np.ndarray):
        if indices.size == 0:
            return
        move_only = results.astype(np.int32) == 0
        selected = indices.astype(np.int32, copy=False)
        self.no_combat_steps[selected[move_only]] += 1
        self.no_combat_steps[selected[~move_only]] = 0
