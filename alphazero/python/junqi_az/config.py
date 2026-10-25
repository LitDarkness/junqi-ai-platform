from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModelConfig:
    input_channels: int = 107
    board_rows: int = 12
    board_cols: int = 5
    action_size: int = 3600
    channels: int = 192
    residual_blocks: int = 10
    policy_channels: int = 64
    auxiliary_targets: int = 0
    repetition_context: bool = True
    value_head_type: str = "wdl"
    wdl_search_mix: float = 1.0
    pairwise_action_bias: bool = True


@dataclass
class ReplayConfig:
    directory: str
    capacity: int = 500_000
    shard_size: int = 4096
    observation_dtype: str = "float16"
    cache_shards: int = 16
    sample_shards_per_batch: int = 16
    recent_fraction: float = 0.75
    recent_window_shards: int = 16
    require_game_provenance: bool = True
    require_repetition_context: bool = True
    validation_fraction: float = 0.0
    split_seed: int = 0
