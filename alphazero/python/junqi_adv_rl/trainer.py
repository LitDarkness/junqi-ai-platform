from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from junqi_az.model import PolicyValueNet

from .config import AdvantageRLConfig
from .returns import filter_advantages, magnet_distribution
from .rollout import RolloutStep


POLICY_WARM_PREFIXES = (
    "encoder.",
    "from_head.",
    "to_head.",
    "from_bias.",
    "to_bias.",
    "action_bias",
    "repetition_policy_bias.",
)


def load_policy_warm_start(
    model: PolicyValueNet, checkpoint: str
) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload.get("trainer", {}).get("model", payload.get("model", {}))
    selected = {
        key: value
        for key, value in state.items()
        if key.startswith(POLICY_WARM_PREFIXES)
    }
    result = model.load_state_dict(selected, strict=False)
    unexpected = list(result.unexpected_keys)
    forbidden_missing = [
        key for key in result.missing_keys
        if key.startswith(POLICY_WARM_PREFIXES)
    ]
    if unexpected or forbidden_missing:
        raise RuntimeError(
            f"incompatible policy warm start: unexpected={unexpected} "
            f"missing_policy={forbidden_missing}"
        )
    return {
        "source": checkpoint,
        "loaded_tensors": len(selected),
        "source_iteration": int(payload.get("iteration", -1)),
        "value_head_reinitialized": True,
    }


class AdvantageTrainer:
    def __init__(self, config: AdvantageRLConfig, model: PolicyValueNet) -> None:
        self.config = config
        requested = config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            requested = "cpu"
        self.device = torch.device(requested)
        self.model = model.to(self.device)
        self.ema = copy.deepcopy(self.model).to(self.device).eval()
        self.anchor = copy.deepcopy(self.model).to(self.device).eval()
        for frozen in (self.ema, self.anchor):
            for parameter in frozen.parameters():
                parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, config.train.schedule_iterations),
            eta_min=config.train.minimum_learning_rate,
        )
        self.global_step = 0

    def reset_copies(self) -> None:
        self.ema.load_state_dict(self.model.state_dict())
        self.anchor.load_state_dict(self.model.state_dict())

    def freeze_collection_policy(self) -> PolicyValueNet:
        self.anchor.load_state_dict(self.model.state_dict())
        self.anchor.eval()
        return self.anchor

    def _amp(self):
        enabled = self.config.train.amp and self.device.type == "cuda"
        dtype = (
            torch.bfloat16
            if self.config.train.amp_dtype == "bfloat16"
            else torch.float16
        )
        return torch.autocast("cuda", dtype=dtype) if enabled else nullcontext()

    def _ema_update(self) -> None:
        smoothing = float(self.config.train.ema_smoothing)
        with torch.no_grad():
            for target, source in zip(self.ema.parameters(), self.model.parameters()):
                target.mul_(smoothing).add_(source, alpha=1.0 - smoothing)
            for target, source in zip(self.ema.buffers(), self.model.buffers()):
                target.copy_(source)

    def _dense_batch(
        self, rows: list[RolloutStep]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observations = np.stack([row.observation for row in rows]).astype(np.float32)
        masks = np.zeros((len(rows), self.config.model.action_size), dtype=np.bool_)
        repetitions = np.zeros_like(masks, dtype=np.uint8)
        for index, row in enumerate(rows):
            masks[index, row.legal_actions] = True
            repetitions[index, row.legal_actions] = row.repetition_contexts
        return (
            torch.from_numpy(observations).to(self.device, non_blocking=True),
            torch.from_numpy(masks).to(self.device, non_blocking=True),
            torch.from_numpy(repetitions).to(self.device, non_blocking=True),
        )

    def _iterate_once(self, rows: list[RolloutStep], rng: np.random.Generator):
        order = rng.permutation(len(rows))
        batch = int(self.config.train.batch_size)
        for start in range(0, len(rows), batch):
            yield [rows[int(index)] for index in order[start : start + batch]]

    def train_value(
        self, steps: list[RolloutStep], rng: np.random.Generator
    ) -> dict[str, float]:
        rows = [row for row in steps if row.wdl_target is not None]
        if not rows:
            return {"value_samples": 0}
        self.model.train()
        aggregate = {"loss": 0.0, "gradient_norm": 0.0, "batches": 0}
        for _ in range(self.config.train.value_epochs):
            for batch in self._iterate_once(rows, rng):
                observations, masks, repetitions = self._dense_batch(batch)
                targets = torch.from_numpy(
                    np.stack([row.wdl_target for row in batch]).astype(np.float32)
                ).to(self.device)
                self.optimizer.zero_grad(set_to_none=True)
                with self._amp():
                    _, _, _, wdl_logits, _ = self.model.forward_heads(
                        observations, masks, repetitions
                    )
                    if wdl_logits is None:
                        raise RuntimeError("WDL head is required")
                    log_wdl = F.log_softmax(wdl_logits.float(), dim=-1)
                    loss = -(targets * log_wdl).sum(dim=-1).mean()
                    weighted = self.config.train.value_weight * loss
                weighted.backward()
                grad = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.train.max_gradient_norm
                )
                self.optimizer.step()
                self._ema_update()
                self.global_step += 1
                aggregate["loss"] += float(loss.detach())
                aggregate["gradient_norm"] += float(grad)
                aggregate["batches"] += 1
        denominator = max(1, aggregate["batches"])
        return {
            "value_samples": len(rows),
            "value_wdl_loss": aggregate["loss"] / denominator,
            "value_gradient_norm": aggregate["gradient_norm"] / denominator,
            "value_batches": aggregate["batches"],
        }

    def _teacher_loss(
        self, rows: list[RolloutStep]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observations, masks, repetitions = self._dense_batch(rows)
        logits, _, _, _, _ = self.model.forward_heads(
            observations, masks, repetitions
        )
        log_policy = F.log_softmax(logits.float(), dim=-1)
        loss_rows = []
        for index, row in enumerate(rows):
            probabilities = torch.from_numpy(row.teacher_probabilities).to(self.device)
            loss_rows.append(
                -(probabilities * log_policy[index, row.legal_actions]).sum()
            )
        return torch.stack(loss_rows).mean(), log_policy

    def train_teacher_bootstrap(
        self,
        rows: list[RolloutStep],
        *,
        steps: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        if not rows:
            raise ValueError("empty teacher bootstrap")
        self.model.train()
        latest: dict[str, float] = {}
        for _ in range(int(steps)):
            indices = rng.integers(0, len(rows), size=self.config.train.batch_size)
            batch = [rows[int(index)] for index in indices]
            observations, masks, repetitions = self._dense_batch(batch)
            target_wdl = torch.from_numpy(
                np.stack([row.wdl_target for row in batch]).astype(np.float32)
            ).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with self._amp():
                logits, _, _, wdl_logits, _ = self.model.forward_heads(
                    observations, masks, repetitions
                )
                log_policy = F.log_softmax(logits.float(), dim=-1)
                policy_rows = []
                for index, row in enumerate(batch):
                    target = torch.from_numpy(row.teacher_probabilities).to(self.device)
                    policy_rows.append(
                        -(target * log_policy[index, row.legal_actions]).sum()
                    )
                policy_loss = torch.stack(policy_rows).mean()
                value_loss = -(
                    target_wdl * F.log_softmax(wdl_logits.float(), dim=-1)
                ).sum(dim=-1).mean()
                loss = policy_loss + self.config.train.value_weight * value_loss
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.max_gradient_norm
            )
            self.optimizer.step()
            self._ema_update()
            self.global_step += 1
            latest = {
                "teacher_imitation_loss": float(policy_loss.detach()),
                "teacher_value_loss": float(value_loss.detach()),
                "teacher_gradient_norm": float(grad),
                "teacher_bootstrap_step": self.global_step,
            }
        self.reset_copies()
        return latest

    def teacher_weight(self, iteration: int) -> float:
        progress = min(
            1.0, max(0, iteration - 1) / max(1, self.config.train.teacher_anneal_iterations)
        )
        return (
            self.config.train.teacher_weight_initial * (1.0 - progress)
            + self.config.train.teacher_weight_final * progress
        )

    def magnet_alpha(self, iteration: int) -> float:
        configured = self.config.advantage.magnet_alpha_override
        if configured is not None:
            return float(configured)
        return float(
            self.config.advantage.magnet_alpha_initial
            / ((iteration + 1) ** self.config.advantage.magnet_alpha_exponent)
        )

    def train_policy(
        self,
        all_steps: list[RolloutStep],
        teacher_rows: list[RolloutStep],
        *,
        iteration: int,
        rng: np.random.Generator,
    ) -> dict[str, float]:
        advantages = np.asarray([row.advantage for row in all_steps], np.float32)
        eligible = np.asarray([row.policy_eligible for row in all_steps], np.bool_)
        keep, diagnostics = filter_advantages(
            advantages,
            eligible,
            percentile=self.config.advantage.filter_percentile,
            minimum_absolute=self.config.advantage.filter_min_absolute,
        )
        rows = [row for row, retained in zip(all_steps, keep) if retained]
        if not rows:
            return {**diagnostics, "policy_batches": 0}
        old_kl_weight = float(self.config.advantage.old_policy_kl_weight)
        magnet_alpha = self.magnet_alpha(iteration)
        teacher_weight = self.teacher_weight(iteration)
        aggregate = {
            "pg": 0.0, "old_kl": 0.0, "magnet_kl": 0.0,
            "entropy": 0.0, "ratio": 0.0, "clip": 0.0,
            "teacher": 0.0, "grad": 0.0, "batches": 0,
        }
        self.model.train()
        # Exactly one pass over this iteration's retained policy rollout.
        for batch_index, batch in enumerate(self._iterate_once(rows, rng)):
            observations, masks, repetitions = self._dense_batch(batch)
            actions = torch.tensor(
                [row.selected_action for row in batch], device=self.device
            )
            old_log_prob = torch.tensor(
                [row.behavior_log_prob for row in batch],
                dtype=torch.float32,
                device=self.device,
            )
            adv = torch.tensor(
                [row.advantage for row in batch],
                dtype=torch.float32,
                device=self.device,
            )
            self.optimizer.zero_grad(set_to_none=True)
            with self._amp():
                logits, _, _, _, _ = self.model.forward_heads(
                    observations, masks, repetitions
                )
                log_policy = F.log_softmax(logits.float(), dim=-1)
                selected_log = log_policy.gather(1, actions[:, None]).squeeze(1)
                ratio = torch.exp((selected_log - old_log_prob).clamp(-20, 20))
                clipped_ratio = torch.clamp(
                    ratio,
                    1.0 - self.config.advantage.clip_epsilon,
                    1.0 + self.config.advantage.clip_epsilon,
                )
                pg = -torch.minimum(ratio * adv, clipped_ratio * adv).mean()
                old_kl_rows = []
                magnet_kl_rows = []
                entropy_rows = []
                for index, row in enumerate(batch):
                    legal = torch.from_numpy(row.legal_actions).to(self.device)
                    new_log = log_policy[index, legal]
                    new_prob = new_log.exp()
                    raw_old = torch.from_numpy(row.old_raw_logits).to(self.device).float()
                    behavior = F.softmax(
                        raw_old / max(row.sampling_temperature, 1e-6), dim=0
                    )
                    if row.uniform_mix:
                        behavior = (
                            (1.0 - row.uniform_mix) * behavior
                            + row.uniform_mix / len(row.legal_actions)
                        )
                    old_kl_rows.append(
                        (new_prob * (new_log - torch.log(behavior.clamp_min(1e-9)))).sum()
                    )
                    rho = torch.from_numpy(
                        magnet_distribution(row.legal_actions)
                    ).to(self.device)
                    magnet_kl_rows.append(
                        (new_prob * (new_log - torch.log(rho.clamp_min(1e-9)))).sum()
                    )
                    entropy_rows.append(-(new_prob * new_log).sum())
                old_kl = torch.stack(old_kl_rows).mean()
                magnet_kl = torch.stack(magnet_kl_rows).mean()
                entropy = torch.stack(entropy_rows).mean()
                teacher_loss = torch.zeros((), device=self.device)
                if teacher_rows and teacher_weight > 0:
                    teacher_indices = rng.integers(
                        0, len(teacher_rows), size=min(len(batch), len(teacher_rows))
                    )
                    teacher_batch = [
                        teacher_rows[int(index)] for index in teacher_indices
                    ]
                    teacher_loss, _ = self._teacher_loss(teacher_batch)
                loss = (
                    pg
                    + old_kl_weight * old_kl
                    + magnet_alpha * magnet_kl
                    + teacher_weight * teacher_loss
                )
            loss.backward()
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.max_gradient_norm
            )
            self.optimizer.step()
            self._ema_update()
            self.global_step += 1
            aggregate["pg"] += float(pg.detach())
            aggregate["old_kl"] += float(old_kl.detach())
            aggregate["magnet_kl"] += float(magnet_kl.detach())
            aggregate["entropy"] += float(entropy.detach())
            aggregate["ratio"] += float(ratio.mean().detach())
            aggregate["clip"] += float(
                ((ratio < 0.8) | (ratio > 1.2)).float().mean().detach()
            )
            aggregate["teacher"] += float(teacher_loss.detach())
            aggregate["grad"] += float(grad)
            aggregate["batches"] += 1
        self.scheduler.step()
        count = max(1, aggregate["batches"])
        return {
            **diagnostics,
            "policy_batches": aggregate["batches"],
            "policy_pg_loss": aggregate["pg"] / count,
            "old_policy_reverse_kl": aggregate["old_kl"] / count,
            "old_policy_kl_weight": old_kl_weight,
            "magnet_kl": aggregate["magnet_kl"] / count,
            "magnet_alpha": magnet_alpha,
            "policy_entropy": aggregate["entropy"] / count,
            "importance_ratio_mean": aggregate["ratio"] / count,
            "ppo_clip_fraction": aggregate["clip"] / count,
            "teacher_imitation_loss": aggregate["teacher"] / count,
            "teacher_auxiliary_weight": teacher_weight,
            "policy_gradient_norm": aggregate["grad"] / count,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "ema": self.ema.state_dict(),
            "anchor": self.anchor.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.model.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        self.ema.load_state_dict(payload["ema"])
        self.anchor.load_state_dict(payload["anchor"])
        self.global_step = int(payload["global_step"])
