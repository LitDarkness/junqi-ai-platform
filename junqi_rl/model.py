import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
        )
        self.activation = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class JunqiActorCritic(nn.Module):
    def __init__(self, in_channels: int = 14, channels: int = 96, num_blocks: int = 6, policy_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, channels),
            nn.SiLU(),
            *[ResidualBlock(channels) for _ in range(num_blocks)],
        )

        # A move is an ordered (from, to) pair. The old dense policy layer
        # dominated the checkpoint while ignoring this structure.
        self.from_head = nn.Conv2d(channels, policy_channels, kernel_size=1)
        self.to_head = nn.Conv2d(channels, policy_channels, kernel_size=1)
        self.from_bias = nn.Conv2d(channels, 1, kernel_size=1)
        self.to_bias = nn.Conv2d(channels, 1, kernel_size=1)
        self.critic_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, channels),
            nn.SiLU(),
            nn.Linear(channels, 1),
        )

    def forward(self, x: torch.Tensor, action_mask: torch.Tensor | None = None):
        features = self.encoder(x)
        batch = features.shape[0]
        from_emb = self.from_head(features).flatten(2).transpose(1, 2)
        to_emb = self.to_head(features).flatten(2)
        logits = torch.bmm(from_emb, to_emb) / (from_emb.shape[-1] ** 0.5)
        logits = logits + self.from_bias(features).flatten(2).transpose(1, 2)
        logits = logits + self.to_bias(features).flatten(2)
        logits = logits.reshape(batch, 3600)
        value = self.critic_head(features)

        if action_mask is not None:
            legal = action_mask > 0
            logits = logits.masked_fill(~legal, torch.finfo(logits.dtype).min)

        return logits, value

    def get_action_and_value(
        self,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        action: torch.Tensor | None = None,
        deterministic: bool = False,
    ):
        logits, value = self.forward(x, action_mask)
        dist = torch.distributions.Categorical(logits=logits)

        if action is None:
            action = torch.argmax(logits, dim=1) if deterministic else dist.sample()

        return action, dist.log_prob(action), dist.entropy(), value
