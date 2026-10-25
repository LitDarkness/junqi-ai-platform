"""Ataraxos-style fresh-rollout advantage training for open-information Junqi."""

from .config import AdvantageRLConfig
from .returns import compute_player_lambda_returns, magnet_distribution

__all__ = [
    "AdvantageRLConfig",
    "compute_player_lambda_returns",
    "magnet_distribution",
]
