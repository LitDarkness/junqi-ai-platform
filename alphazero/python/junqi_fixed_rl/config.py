from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from junqi_az.config import ModelConfig


@dataclass
class LayoutConfig:
    set_path: str = "config/layouts/fixed_layout_v1.json"
    layout_id: str = "public_conservative_advance"
    mirror_probability: float = 0.0


@dataclass
class TeacherConfig:
    bootstrap_games: int = 512
    bootstrap_train_steps: int = 400
    workers: int = 16
    softmax_temperatures: list[float] = field(
        default_factory=lambda: [0.7, 1.0, 1.5]
    )
    uniform_mix: float = 0.05
    external_vote_weight: float = 0.35
    heuristic_value_scale: float = 25.0
    external_candidates: list[str] = field(default_factory=list)


@dataclass
class SelfPlayConfig:
    games_per_iteration: int = 1024
    parallel_games: int = 256
    max_moves: int = 400
    policy_temperature: float = 0.9
    temperature_decay_steps: int = 100_000
    minimum_temperature: float = 0.25
    historical_keep: int = 8
    opponent_weights: dict[str, float] = field(
        default_factory=lambda: {
            "current": 0.25,
            "historical": 0.25,
            "heuristic_current": 0.20,
            "heuristic_historical": 0.10,
            "random": 0.10,
            "aggressive": 0.10,
        }
    )


@dataclass
class ReplayConfig:
    teacher_directory: str = "replay_teacher"
    selfplay_directory: str = "replay_selfplay"
    capacity: int = 500_000
    shard_size: int = 4096
    observation_dtype: str = "float16"
    cache_shards: int = 16
    sample_shards_per_batch: int = 16


@dataclass
class TrainConfig:
    batch_size: int = 1024
    steps_per_iteration: int = 200
    learning_rate: float = 2e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    value_weight: float = 1.0
    entropy_start: float = 0.02
    entropy_end: float = 0.002
    kl_start: float = 0.05
    kl_end: float = 0.005
    rho_uniform_mix: float = 0.10
    teacher_fraction_start: float = 0.70
    teacher_fraction_end: float = 0.10
    schedule_steps: int = 200_000
    gradient_clip: float = 5.0
    amp: bool = True
    amp_dtype: str = "bfloat16"
    checkpoint_every_iterations: int = 1
    log_every: int = 10


@dataclass
class CheckpointConfig:
    archive_interval: int = 5
    recent_keep: int = 5
    initial_champion_iteration: int = 10


@dataclass
class EvaluationConfig:
    enabled: bool = True
    interval: int = 5
    initial_games: int = 512
    extended_games: int = 2048
    reject_score: float = 0.52
    direct_promote_score: float = 0.55
    temperature: float = 0.25
    mode: str = "sample"
    seed: int = 2026072309
    crossplay_interval: int = 10
    crossplay_games: int = 128
    repetition_warning_fraction: float = 0.15
    max_ply_warning_fraction: float = 0.10


@dataclass
class FixedRLConfig:
    schema: str = "junqi-fixed-layout-rl-v1"
    seed: int = 2026072301
    device: str = "cuda"
    run_directory: str = "runs/fixed_layout_rl_v1"
    layout: LayoutConfig = field(default_factory=LayoutConfig)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(
            input_channels=107,
            channels=192,
            residual_blocks=10,
            policy_channels=64,
            value_head_type="wdl",
            wdl_search_mix=1.0,
            repetition_context=True,
            pairwise_action_bias=True,
        )
    )
    teacher: TeacherConfig = field(default_factory=TeacherConfig)
    selfplay: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    @classmethod
    def load(cls, path: str | Path) -> "FixedRLConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        config = cls(
            schema=str(raw.get("schema", "")),
            seed=int(raw["seed"]),
            device=str(raw.get("device", "cuda")),
            run_directory=str(raw["run_directory"]),
            layout=LayoutConfig(**raw["layout"]),
            model=ModelConfig(**raw["model"]),
            teacher=TeacherConfig(**raw["teacher"]),
            selfplay=SelfPlayConfig(**raw["selfplay"]),
            replay=ReplayConfig(**raw["replay"]),
            train=TrainConfig(**raw["train"]),
            checkpoint=CheckpointConfig(**raw.get("checkpoint", {})),
            evaluation=EvaluationConfig(**raw.get("evaluation", {})),
        )
        config.validate(source.parent.parent)
        return config

    def validate(self, project_root: Path | None = None) -> None:
        if self.schema != "junqi-fixed-layout-rl-v1":
            raise ValueError(f"unsupported config schema: {self.schema}")
        if self.model.input_channels != 107:
            raise ValueError("fixed-layout v1 requires 107 native feature planes")
        if self.model.value_head_type != "wdl":
            raise ValueError("fixed-layout v1 requires a WDL-only value head")
        if self.model.action_size != 3600 or not self.model.pairwise_action_bias:
            raise ValueError("policy head must use 3600 actions and pairwise b_ij")
        if not 0.0 <= self.layout.mirror_probability <= 1.0:
            raise ValueError("layout mirror_probability must be in [0,1]")
        if self.selfplay.parallel_games < 1 or self.selfplay.games_per_iteration < 1:
            raise ValueError("self-play game counts must be positive")
        if self.train.batch_size < 1 or self.train.steps_per_iteration < 1:
            raise ValueError("training batch/step counts must be positive")
        if self.checkpoint.archive_interval < 1 or self.checkpoint.recent_keep < 1:
            raise ValueError("checkpoint archive/recent retention must be positive")
        evaluation = self.evaluation
        if (
            evaluation.interval < 1
            or evaluation.initial_games < 2
            or evaluation.extended_games < evaluation.initial_games
            or evaluation.initial_games % 2 != 0
            or evaluation.extended_games % 2 != 0
            or evaluation.crossplay_games < 2
            or evaluation.crossplay_games % 2 != 0
            or evaluation.mode not in {"sample", "greedy"}
            or not 0.0 <= evaluation.temperature
        ):
            raise ValueError("invalid independent evaluation configuration")
        if not (
            0.5 <= evaluation.reject_score
            <= evaluation.direct_promote_score
            <= 1.0
        ):
            raise ValueError("invalid champion promotion thresholds")
        weights = self.selfplay.opponent_weights
        required = {
            "current",
            "historical",
            "heuristic_current",
            "heuristic_historical",
            "random",
            "aggressive",
        }
        if set(weights) != required or any(float(value) < 0 for value in weights.values()):
            raise ValueError("opponent_weights has missing/negative population entries")
        if sum(float(value) for value in weights.values()) <= 0:
            raise ValueError("opponent_weights must have positive total mass")
        if project_root is not None:
            layout_path = Path(self.layout.set_path)
            if not layout_path.is_absolute():
                layout_path = project_root / layout_path
            if not layout_path.is_file():
                raise FileNotFoundError(f"layout set is missing: {layout_path}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
