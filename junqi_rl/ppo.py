from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class PPOConfig:
    learning_rate: float = 2e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    update_epochs: int = 4
    minibatch_size: int = 1024


class VecPPOTrainer:
    def __init__(self, vec_env, model, device: torch.device, config: PPOConfig):
        self.vec_env = vec_env
        self.model = model
        self.device = device
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        self.buffer = {}
        self._rollout_obs = None
        self._rollout_infos = None

    def collect_rollout(self, num_steps: int):
        if self._rollout_obs is None or self._rollout_infos is None:
            obs, infos = self.vec_env.reset()
            self._rollout_obs = obs
            self._rollout_infos = infos
        else:
            obs, infos = self._rollout_obs, self._rollout_infos
        masks = np.stack([info["action_mask"] for info in infos]).astype(np.uint8)

        states = []
        action_masks = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []
        episodic_returns = []
        episodic_lengths = []

        for _ in range(num_steps):
            state_tensor = torch.from_numpy(obs).to(self.device)
            mask_tensor = torch.from_numpy(masks).to(self.device, dtype=torch.bool)

            with torch.no_grad():
                action, log_prob, _, value = self.model.get_action_and_value(state_tensor, mask_tensor)

            next_obs, reward, done, next_infos = self.vec_env.step(action.cpu().numpy())

            states.append(obs)
            action_masks.append(masks)
            actions.append(action.cpu().numpy())
            log_probs.append(log_prob.cpu().numpy())
            rewards.append(reward)
            dones.append(done.astype(np.float32))
            values.append(value.squeeze(-1).cpu().numpy())

            for info in next_infos:
                episode = info.get("episode")
                if episode is not None:
                    episodic_returns.append(float(episode["r"]))
                    episodic_lengths.append(float(episode["l"]))

            obs = next_obs
            masks = np.stack([info["action_mask"] for info in next_infos]).astype(np.uint8)
            infos = next_infos

        with torch.no_grad():
            final_values = self.model.get_action_and_value(
                torch.from_numpy(obs).to(self.device),
                torch.from_numpy(masks).to(self.device, dtype=torch.bool),
            )[3].squeeze(-1)

        self._rollout_obs = obs
        self._rollout_infos = infos

        self.buffer = {
            "states": np.asarray(states, dtype=np.float32),
            "action_masks": np.asarray(action_masks, dtype=np.uint8),
            "actions": np.asarray(actions, dtype=np.int64),
            "log_probs": np.asarray(log_probs, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
            "values": np.asarray(values, dtype=np.float32),
        }
        return final_values.cpu().numpy(), episodic_returns, episodic_lengths

    def compute_advantages(self, last_values: np.ndarray):
        rewards = self.buffer["rewards"]
        values = np.concatenate([self.buffer["values"], last_values[None, :]], axis=0)
        dones = self.buffer["dones"]

        advantages = np.zeros_like(rewards, dtype=np.float32)
        gae = np.zeros(rewards.shape[1], dtype=np.float32)

        for step in range(rewards.shape[0] - 1, -1, -1):
            next_non_terminal = 1.0 - dones[step]
            delta = rewards[step] + self.config.gamma * values[step + 1] * next_non_terminal - values[step]
            gae = delta + self.config.gamma * self.config.gae_lambda * next_non_terminal * gae
            advantages[step] = gae

        returns = advantages + self.buffer["values"]
        return (
            torch.from_numpy(returns).to(self.device),
            torch.from_numpy(advantages).to(self.device),
        )

    def update(self, returns: torch.Tensor, advantages: torch.Tensor):
        states = torch.from_numpy(self.buffer["states"]).reshape(-1, 14, 12, 5).to(self.device)
        action_masks = torch.from_numpy(self.buffer["action_masks"]).reshape(-1, 3600).to(self.device)
        actions = torch.from_numpy(self.buffer["actions"]).reshape(-1).to(self.device)
        old_log_probs = torch.from_numpy(self.buffer["log_probs"]).reshape(-1).to(self.device)
        returns = returns.reshape(-1)
        advantages = advantages.reshape(-1)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        indices = np.arange(batch_size)
        stats = {"actor_loss": 0.0, "critic_loss": 0.0, "entropy": 0.0, "updates": 0}

        for _ in range(self.config.update_epochs):
            np.random.shuffle(indices)
            for start in range(0, batch_size, self.config.minibatch_size):
                mb_idx = indices[start:start + self.config.minibatch_size]
                _, new_log_prob, entropy, value = self.model.get_action_and_value(
                    states[mb_idx],
                    action_masks[mb_idx],
                    action=actions[mb_idx],
                )

                ratio = (new_log_prob - old_log_probs[mb_idx]).exp()
                mb_advantages = advantages[mb_idx]
                policy_loss_1 = ratio * mb_advantages
                policy_loss_2 = torch.clamp(
                    ratio,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio,
                ) * mb_advantages
                actor_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                critic_loss = F.mse_loss(value.squeeze(-1), returns[mb_idx])
                entropy_loss = entropy.mean()
                loss = (
                    actor_loss
                    + self.config.value_coef * critic_loss
                    - self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.optimizer.step()

                stats["actor_loss"] += actor_loss.item()
                stats["critic_loss"] += critic_loss.item()
                stats["entropy"] += entropy_loss.item()
                stats["updates"] += 1

        updates = max(1, stats["updates"])
        return (
            stats["actor_loss"] / updates,
            stats["critic_loss"] / updates,
            stats["entropy"] / updates,
        )
