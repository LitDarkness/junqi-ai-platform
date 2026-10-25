from __future__ import annotations

import copy
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from junqi_az.model import PolicyValueNet
from junqi_az.replay import ReplayBatch, ShardedReplayBuffer

from .config import FixedRLConfig


@dataclass(frozen=True)
class Schedule:
    progress: float
    learning_rate: float
    entropy: float
    kl: float
    teacher_fraction: float


def _interpolate(start: float, end: float, progress: float) -> float:
    return float(start + (end - start) * progress)


def _merge_batches(parts: list[ReplayBatch]) -> ReplayBatch:
    if len(parts) == 1:
        return parts[0]
    return ReplayBatch(
        observations=np.concatenate([part.observations for part in parts]),
        actions=[row for part in parts for row in part.actions],
        probabilities=[row for part in parts for row in part.probabilities],
        outcomes=np.concatenate([part.outcomes for part in parts]),
        policy_weights=np.concatenate([part.policy_weights for part in parts]),
        search_values=np.concatenate([part.search_values for part in parts]),
        repetition_contexts=[
            row for part in parts for row in part.repetition_contexts
        ],
        game_keys=np.concatenate([part.game_keys for part in parts]),
        ply_indices=np.concatenate([part.ply_indices for part in parts]),
    )


class FixedRLTrainer:
    def __init__(self, config: FixedRLConfig, model: PolicyValueNet) -> None:
        self.config = config
        requested = config.device
        if requested.startswith("cuda") and not torch.cuda.is_available():
            print("[fixed-rl] CUDA unavailable; falling back to CPU", flush=True)
            requested = "cpu"
        self.device = torch.device(requested)
        self.model = model.to(self.device)
        self.anchor = copy.deepcopy(model).to(self.device).eval()
        for parameter in self.anchor.parameters():
            parameter.requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.train.learning_rate,
            weight_decay=config.train.weight_decay,
        )
        self.global_step = 0

    def schedule(self) -> Schedule:
        train = self.config.train
        progress = min(1.0, self.global_step / max(1, train.schedule_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        learning_rate = train.minimum_learning_rate + (
            train.learning_rate - train.minimum_learning_rate
        ) * cosine
        return Schedule(
            progress=progress,
            learning_rate=learning_rate,
            entropy=_interpolate(train.entropy_start, train.entropy_end, progress),
            kl=_interpolate(train.kl_start, train.kl_end, progress),
            teacher_fraction=_interpolate(
                train.teacher_fraction_start,
                train.teacher_fraction_end,
                progress,
            ),
        )

    def set_anchor(self) -> None:
        self.anchor.load_state_dict(self.model.state_dict())
        self.anchor.eval()

    def _sample(
        self,
        teacher: ShardedReplayBuffer,
        selfplay: ShardedReplayBuffer,
        batch_size: int,
        teacher_fraction: float,
    ) -> ReplayBatch:
        teacher_count = min(
            len(teacher),
            int(round(batch_size * teacher_fraction)),
        )
        selfplay_count = batch_size - teacher_count
        if selfplay_count > len(selfplay):
            missing = selfplay_count - len(selfplay)
            teacher_count = min(len(teacher), teacher_count + missing)
            selfplay_count = batch_size - teacher_count
        if teacher_count + selfplay_count < batch_size:
            raise ValueError(
                f"replay too small: teacher={len(teacher)} selfplay={len(selfplay)} "
                f"batch={batch_size}"
            )
        parts: list[ReplayBatch] = []
        if teacher_count:
            parts.append(teacher.sample(teacher_count, split="all"))
        if selfplay_count:
            parts.append(selfplay.sample(selfplay_count, split="all"))
        merged = _merge_batches(parts)
        permutation = np.random.permutation(batch_size)
        return ReplayBatch(
            observations=merged.observations[permutation],
            actions=[merged.actions[index] for index in permutation],
            probabilities=[merged.probabilities[index] for index in permutation],
            outcomes=merged.outcomes[permutation],
            policy_weights=merged.policy_weights[permutation],
            search_values=merged.search_values[permutation],
            repetition_contexts=[
                merged.repetition_contexts[index] for index in permutation
            ],
            game_keys=merged.game_keys[permutation],
            ply_indices=merged.ply_indices[permutation],
        )

    def train_steps(
        self,
        teacher: ShardedReplayBuffer,
        selfplay: ShardedReplayBuffer,
        steps: int,
        *,
        metric_writer: Any | None = None,
    ) -> dict[str, float]:
        self.model.train()
        latest: dict[str, float] = {}
        for local_step in range(int(steps)):
            schedule = self.schedule()
            for group in self.optimizer.param_groups:
                group["lr"] = schedule.learning_rate
            batch = self._sample(
                teacher,
                selfplay,
                self.config.train.batch_size,
                schedule.teacher_fraction,
            )
            observations = torch.from_numpy(batch.observations).to(
                self.device, non_blocking=True
            )
            action_mask = torch.from_numpy(
                batch.action_mask(self.config.model.action_size)
            ).to(self.device, non_blocking=True)
            repetition = torch.from_numpy(
                batch.dense_repetition_context(self.config.model.action_size)
            ).to(self.device, non_blocking=True)
            target_policy = torch.from_numpy(
                batch.dense_policy(self.config.model.action_size)
            ).to(self.device, non_blocking=True)
            outcomes = torch.from_numpy(batch.outcomes).to(
                self.device, non_blocking=True
            )
            policy_weights = torch.from_numpy(batch.policy_weights).to(
                self.device, non_blocking=True
            )
            target_wdl = torch.where(
                outcomes > 0.5,
                2,
                torch.where(outcomes < -0.5, 0, 1),
            ).long()

            use_amp = self.config.train.amp and self.device.type == "cuda"
            amp_dtype = (
                torch.bfloat16
                if self.config.train.amp_dtype == "bfloat16"
                else torch.float16
            )
            amp_context = (
                torch.autocast("cuda", dtype=amp_dtype)
                if use_amp
                else nullcontext()
            )
            self.optimizer.zero_grad(set_to_none=True)
            with amp_context:
                logits, values, _, wdl_logits, _ = self.model.forward_heads(
                    observations, action_mask, repetition
                )
                if wdl_logits is None:
                    raise RuntimeError("fixed-layout trainer requires WDL logits")
                log_policy = F.log_softmax(logits.float(), dim=-1)
                probabilities = log_policy.exp()
                row_ce = -(target_policy * log_policy).sum(dim=-1)
                policy_weight_sum = policy_weights.sum().clamp_min(1.0)
                policy_loss = (row_ce * policy_weights).sum() / policy_weight_sum
                value_loss = F.cross_entropy(wdl_logits.float(), target_wdl)
                entropy = -(probabilities * log_policy).sum(dim=-1).mean()

                with torch.no_grad():
                    anchor_logits, _ = self.anchor(
                        observations, action_mask, repetition
                    )
                    anchor_policy = F.softmax(anchor_logits.float(), dim=-1)
                    legal_count = action_mask.sum(dim=-1, keepdim=True).clamp_min(1)
                    legal_uniform = action_mask.float() / legal_count
                    mix = self.config.train.rho_uniform_mix
                    rho = (1.0 - mix) * anchor_policy + mix * legal_uniform
                kl = (
                    probabilities
                    * (log_policy - torch.log(rho.clamp_min(1e-9)))
                ).sum(dim=-1).mean()
                loss = (
                    policy_loss
                    + self.config.train.value_weight * value_loss
                    - schedule.entropy * entropy
                    + schedule.kl * kl
                )

            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.train.gradient_clip
            )
            self.optimizer.step()
            self.global_step += 1
            with torch.no_grad():
                wdl_probabilities = F.softmax(wdl_logits.float(), dim=-1)
                expected = wdl_probabilities[:, 2] - wdl_probabilities[:, 0]
                value_mse = F.mse_loss(expected, outcomes)
                top1 = (
                    logits.argmax(dim=-1) == target_policy.argmax(dim=-1)
                ).float().mean()
                target_entropy = -(
                    target_policy * torch.log(target_policy.clamp_min(1e-9))
                ).sum(dim=-1).mean()
                uniform_ce = torch.log(
                    action_mask.sum(dim=-1).clamp_min(1).float()
                ).mean()
            latest = {
                "kind": "train",
                "step": float(self.global_step),
                "loss": float(loss.detach()),
                "policy_loss": float(policy_loss.detach()),
                "value_wdl_loss": float(value_loss.detach()),
                "value_mse": float(value_mse.detach()),
                "entropy": float(entropy.detach()),
                "target_entropy": float(target_entropy.detach()),
                "uniform_policy_loss": float(uniform_ce.detach()),
                "policy_excess": float((policy_loss - target_entropy).detach()),
                "policy_top1": float(top1.detach()),
                "kl_to_anchor_rho": float(kl.detach()),
                "entropy_coefficient": schedule.entropy,
                "kl_coefficient": schedule.kl,
                "teacher_fraction": schedule.teacher_fraction,
                "learning_rate": schedule.learning_rate,
                "gradient_norm": float(gradient_norm),
                "teacher_replay": float(len(teacher)),
                "selfplay_replay": float(len(selfplay)),
            }
            if metric_writer is not None:
                metric_writer.write(
                    "train", **{key: value for key, value in latest.items() if key != "kind"}
                )
            if (
                local_step == 0
                or (local_step + 1) % self.config.train.log_every == 0
                or local_step + 1 == steps
            ):
                print(
                    "[fixed-rl train] "
                    f"step={self.global_step} loss={latest['loss']:.4f} "
                    f"policy={latest['policy_loss']:.4f} "
                    f"excess={latest['policy_excess']:+.4f} "
                    f"wdl={latest['value_wdl_loss']:.4f} "
                    f"entropy={latest['entropy']:.3f} "
                    f"KL={latest['kl_to_anchor_rho']:.5f} "
                    f"teacher={latest['teacher_fraction']:.2f}",
                    flush=True,
                )
        return latest

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "anchor": self.anchor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        self.model.load_state_dict(payload["model"])
        self.anchor.load_state_dict(payload.get("anchor", payload["model"]))
        self.optimizer.load_state_dict(payload["optimizer"])
        self.global_step = int(payload.get("global_step", 0))
