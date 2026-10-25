from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class PolicyValueNet(nn.Module):
    """Junqi policy/value network; value is always bounded to [-1, 1]."""

    def __init__(self, config: ModelConfig | None = None):
        super().__init__()
        self.config = config or ModelConfig()
        c = self.config.channels
        pc = self.config.policy_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(self.config.input_channels, c, 3, padding=1, bias=False),
            nn.GroupNorm(8, c),
            nn.SiLU(),
            *[ResidualBlock(c) for _ in range(self.config.residual_blocks)],
        )
        self.from_head = nn.Conv2d(c, pc, 1)
        self.to_head = nn.Conv2d(c, pc, 1)
        self.from_bias = nn.Conv2d(c, 1, 1)
        self.to_bias = nn.Conv2d(c, 1, 1)
        self.action_bias: nn.Parameter | None = None
        if self.config.pairwise_action_bias:
            self.action_bias = nn.Parameter(torch.zeros(self.config.action_size))
        self.repetition_policy_bias: nn.Embedding | None = None
        self.repetition_value_adjustment: nn.Module | None = None
        if self.config.repetition_context:
            self.repetition_policy_bias = nn.Embedding(3, 1)
            nn.init.zeros_(self.repetition_policy_bias.weight)
            self.repetition_value_adjustment = nn.Sequential(
                nn.Linear(3, c),
                nn.SiLU(),
                nn.Linear(c, 1),
                nn.Tanh(),
            )
            nn.init.zeros_(self.repetition_value_adjustment[2].weight)
            nn.init.zeros_(self.repetition_value_adjustment[2].bias)
        self.value_head: nn.Module | None = None
        if self.config.value_head_type in {"scalar", "dual"}:
            self.value_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 1),
                nn.Tanh(),
            )
        self.wdl_head: nn.Module | None = None
        if self.config.value_head_type in {"wdl", "dual"}:
            self.wdl_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, 3),
            )
        self.auxiliary_head: nn.Module | None = None
        if self.config.auxiliary_targets:
            self.auxiliary_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(c, c),
                nn.SiLU(),
                nn.Linear(c, self.config.auxiliary_targets),
                nn.Tanh(),
            )

    def _policy_logits(
        self,
        features: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        repetition_context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = features.shape[0]
        from_emb = self.from_head(features).flatten(2).transpose(1, 2)
        to_emb = self.to_head(features).flatten(2)
        logits = torch.bmm(from_emb, to_emb) / (from_emb.shape[-1] ** 0.5)
        logits = logits + self.from_bias(features).flatten(2).transpose(1, 2)
        logits = logits + self.to_bias(features).flatten(2)
        logits = logits.reshape(batch, self.config.action_size)
        if self.action_bias is not None:
            logits = logits + self.action_bias
        if self.repetition_policy_bias is not None and repetition_context is not None:
            contexts = repetition_context.long().clamp_(0, 2)
            logits = logits + self.repetition_policy_bias(contexts).squeeze(-1)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), torch.finfo(logits.dtype).min)
        return logits

    def forward_all(
        self,
        observations: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        repetition_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        logits, values, auxiliary, _, _ = self.forward_heads(
            observations, action_mask, repetition_context
        )
        return logits, values, auxiliary

    def forward_heads(
        self,
        observations: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        repetition_context: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        """Return policy, search value, auxiliary, WDL logits, scalar value."""
        features = self.encoder(observations)
        auxiliary = (
            self.auxiliary_head(features)
            if self.auxiliary_head is not None
            else None
        )
        scalar_values = (
            self.value_head(features).squeeze(-1)
            if self.value_head is not None
            else None
        )
        wdl_logits = self.wdl_head(features) if self.wdl_head is not None else None
        if wdl_logits is not None:
            wdl_probabilities = torch.softmax(wdl_logits.float(), dim=-1)
            wdl_values = wdl_probabilities[:, 2] - wdl_probabilities[:, 0]
            if scalar_values is not None:
                mix = float(self.config.wdl_search_mix)
                values = (1.0 - mix) * scalar_values + mix * wdl_values
            else:
                values = wdl_values
        else:
            assert scalar_values is not None
            values = scalar_values
        if (
            self.repetition_value_adjustment is not None
            and repetition_context is not None
            and action_mask is not None
        ):
            legal = action_mask.bool()
            denominator = legal.sum(dim=-1, keepdim=True).clamp_min(1).float()
            histogram = torch.stack(
                [
                    ((repetition_context == bucket) & legal).sum(dim=-1).float()
                    / denominator.squeeze(-1)
                    for bucket in range(3)
                ],
                dim=-1,
            )
            adjustment = 0.25 * self.repetition_value_adjustment(histogram).squeeze(-1)
            values = (values + adjustment).clamp(-1.0, 1.0)
        return (
            self._policy_logits(features, action_mask, repetition_context),
            values,
            auxiliary,
            wdl_logits,
            scalar_values,
        )

    def policy_logits(self, observations: torch.Tensor, action_mask: torch.Tensor,
                      repetition_context: torch.Tensor | None = None) -> torch.Tensor:
        return self._policy_logits(self.encoder(observations), action_mask, repetition_context)

    def terminal_wdl_logits(self, observations: torch.Tensor) -> torch.Tensor:
        if self.wdl_head is None:
            raise ValueError("terminal WDL training requires a WDL head")
        return self.wdl_head(self.encoder(observations))

    def forward(
        self,
        observations: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        repetition_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values, _ = self.forward_all(
            observations, action_mask, repetition_context
        )
        return logits, values

    @torch.inference_mode()
    def policy_value(
        self,
        observations: torch.Tensor,
        action_masks: torch.Tensor,
        repetition_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, values = self(observations, action_masks, repetition_context)
        return torch.softmax(logits.float(), dim=-1), values.float()

    def encoder_parameters(self):
        """Iterate parameters shared by the policy and value heads."""
        yield from self.encoder.parameters()

    def policy_head_parameters(self):
        """Iterate parameters exclusive to the policy head."""
        yield from self.from_head.parameters()
        yield from self.to_head.parameters()
        yield from self.from_bias.parameters()
        yield from self.to_bias.parameters()
        if self.action_bias is not None:
            yield self.action_bias
        if self.repetition_policy_bias is not None:
            yield from self.repetition_policy_bias.parameters()

    def policy_parameters(self):
        """Backwards-compatible iterator over encoder and policy head."""
        yield from self.encoder_parameters()
        yield from self.policy_head_parameters()

    def value_parameters(self):
        """Iterate parameters exclusive to value and auxiliary supervision."""
        if self.value_head is not None:
            yield from self.value_head.parameters()
        if self.wdl_head is not None:
            yield from self.wdl_head.parameters()
        if self.auxiliary_head is not None:
            yield from self.auxiliary_head.parameters()
        if self.repetition_value_adjustment is not None:
            yield from self.repetition_value_adjustment.parameters()

    def load_compatible_state_dict(
        self, state: dict[str, torch.Tensor]
    ) -> dict[str, list[str]]:
        """Load checkpoints across the optional auxiliary-head boundary.

        Core policy/value tensors remain strict.  Only auxiliary tensors may be
        absent (old checkpoint -> new model) or unexpected (new -> old).
        """
        result = self.load_state_dict(state, strict=False)
        allowed_prefixes: tuple[str, ...] = (
            "auxiliary_head.",
            "repetition_policy_bias.",
            "repetition_value_adjustment.",
        )
        if self.config.value_head_type in {"wdl", "dual"}:
            allowed_prefixes += ("wdl_head.",)
        if self.config.value_head_type == "wdl":
            # A scalar source can warm-start the shared encoder/policy, but its
            # old scalar head is intentionally discarded by a WDL-only model.
            allowed_prefixes += ("value_head.",)
        missing = [
            key for key in result.missing_keys
            if not key.startswith(allowed_prefixes)
        ]
        unexpected = [
            key for key in result.unexpected_keys
            if not key.startswith(allowed_prefixes)
        ]
        if missing or unexpected:
            raise RuntimeError(
                "incompatible policy/value checkpoint: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return {
            "missing": sorted(result.missing_keys),
            "unexpected": sorted(result.unexpected_keys),
        }

    def set_training_phase(self, phase: str) -> None:
        """Select ``value_only`` bootstrap or normal ``joint`` training."""
        if phase not in {
            "value_only", "value_anchor", "value_distill", "policy_only", "joint"
        }:
            raise ValueError(
                "training phase must be 'value_only', 'value_anchor', "
                "'value_distill', 'policy_only' or 'joint'"
            )
        # ``value_anchor`` is entered after the trust-region limit is reached.
        # Freezing the encoder and policy head preserves the accepted boundary
        # policy while the exclusive value head uses the remaining steps.
        encoder_trainable = phase in {"joint", "value_distill", "policy_only"}
        policy_head_trainable = phase in {"joint", "policy_only"}
        value_head_trainable = phase != "policy_only"
        for parameter in self.encoder_parameters():
            parameter.requires_grad_(encoder_trainable)
        for parameter in self.policy_head_parameters():
            parameter.requires_grad_(policy_head_trainable)
        for parameter in self.value_parameters():
            parameter.requires_grad_(value_head_trainable)

    def warm_start_ppo(self, source: str | Path | dict[str, Any]) -> dict[str, list[str]]:
        """Load compatible PPO encoder/policy tensors, intentionally reset AZ value output.

        Checkpoints may contain ``model``, ``model_state_dict`` or a raw state dict.
        The PPO ``critic_head`` hidden layer is mapped to ``value_head`` when shapes match;
        its final layer is excluded because PPO shaped rewards have a different scale.
        """
        if isinstance(source, (str, Path)):
            payload = torch.load(source, map_location="cpu", weights_only=False)
        else:
            payload = source
        state = payload.get("model_state_dict", payload.get("model", payload))
        own = self.state_dict()
        mapped: dict[str, torch.Tensor] = {}
        skipped: list[str] = []
        for key, value in state.items():
            target = key
            if key.startswith("critic_head.2."):
                target = key.replace("critic_head.2.", "value_head.2.")
            elif key.startswith("critic_head"):
                skipped.append(key)
                continue
            if (
                target == "encoder.0.weight"
                and target in own
                and value.ndim == 4
                and value.shape[0] == own[target].shape[0]
                and value.shape[2:] == own[target].shape[2:]
                and value.shape[1] <= own[target].shape[1]
            ):
                expanded = own[target].clone()
                expanded[:, : value.shape[1]].copy_(value)
                mapped[target] = expanded
            elif target in own and own[target].shape == value.shape:
                mapped[target] = value
            else:
                skipped.append(key)
        result = self.load_state_dict(mapped, strict=False)
        # Keep the AlphaZero value output neutral at initialization.
        nn.init.zeros_(self.value_head[4].weight)
        nn.init.zeros_(self.value_head[4].bias)
        return {
            "loaded": sorted(mapped),
            "skipped": sorted(skipped),
            "missing": sorted(result.missing_keys),
        }
