from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from junqi_az.config import ModelConfig

from . import CONTRACT


@dataclass
class OpenConfig:
    schema: str = CONTRACT
    seed: int = 2026100307
    device: str = "cuda"
    run_directory: str = "runs/open_junqi_v4_repaired"
    layout_directory: str = "config/layouts/open_v4"
    initial_checkpoint: str | None = "runs/open_junqi_search_v3_strength_continuation/checkpoints/loop_iteration_000150.pt"
    teacher_candidates: list[str] = field(default_factory=lambda: [
        "../teachers/082/candidate.py",
        "../teachers/084/candidate.py",
    ])
    workers: int = 12
    games_per_iteration: int = 64
    selfplay_fraction: float = .25
    teacher_selfplay_fraction: float | None = None
    actor_policy_target: str = "teacher"
    root_tactical_guard: bool = False
    training_greedy_after_ply: int | None = None
    simulations: int = 64
    repeated_simulations: int = 128
    inference_batch_size: int = 256
    cache_size: int = 0
    leaves_per_tree: int = 1
    replay_iterations: int = 3
    maximum_replay_samples: int = 100000
    updates_per_iteration: int = 64
    logical_batch_size: int = 256
    microbatch_size: int = 64
    learning_rate: float = 5e-5
    value_encoder_lr_multiplier: float = .1
    weight_decay: float = 1e-4
    gradient_clip: float = 1.
    value_policy_kl_limit: float = .001
    value_kl_weight: float = 1.
    value_backtracking_attempts: int = 5
    actor_gate_every: int = 2
    actor_gate_families: int = 32
    actor_noninferiority_margin: float = .05
    champion_minimum_family_count: int = 64
    teacher_sentinel_tolerance: float = .05
    model: ModelConfig = field(default_factory=lambda: ModelConfig(
        input_channels=107, channels=192, residual_blocks=10, policy_channels=64,
        value_head_type="wdl", repetition_context=True, pairwise_action_bias=True,
    ))

    @classmethod
    def load(cls, path: Path) -> "OpenConfig":
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
        raw["model"] = ModelConfig(**raw["model"])
        config = cls(**raw)
        config.validate()
        return config

    def validate(self) -> None:
        if self.schema != CONTRACT:
            raise ValueError("training contract changed; use a new run directory")
        if self.model.input_channels != 107 or self.model.action_size != 3600 or self.model.value_head_type != "wdl":
            raise ValueError("v4 requires the native 107-plane/3600-action WDL contract")
        if not self.model.repetition_context or not 1 <= self.leaves_per_tree <= 256:
            raise ValueError("invalid repetition context or leaf batching")
        for value in (self.games_per_iteration, self.updates_per_iteration, self.logical_batch_size,
                      self.microbatch_size, self.simulations, self.repeated_simulations, self.workers,
                      self.replay_iterations, self.maximum_replay_samples, self.actor_gate_every,
                      self.actor_gate_families, self.champion_minimum_family_count, self.value_backtracking_attempts):
            if value < 1:
                raise ValueError("positive training budgets required")
        if not 0 <= self.selfplay_fraction <= 1 or self.learning_rate <= 0 or self.value_policy_kl_limit <= 0:
            raise ValueError("invalid training weights")
        if self.teacher_selfplay_fraction is not None and not 0 <= self.teacher_selfplay_fraction <= 1 - self.selfplay_fraction:
            raise ValueError("training mode fractions must sum to at most one")
        if self.training_greedy_after_ply is not None and self.training_greedy_after_ply < 0:
            raise ValueError("greedy boundary must be a nonnegative ply or None")
        if self.actor_policy_target not in ("teacher", "search"):
            raise ValueError("actor policy target must be teacher or search")
        if self.initial_checkpoint is None and not self.teacher_candidates:
            raise ValueError("random initialization requires a teacher bootstrap")
        if not 0 <= self.actor_noninferiority_margin < .5:
            raise ValueError("invalid actor noninferiority margin")
        if self.simulations < 2 or self.repeated_simulations < 2:
            raise ValueError("search distillation needs at least two simulations for nonempty visit targets")
        if self.actor_gate_families < 2 or self.champion_minimum_family_count < 2:
            raise ValueError("promotion requires multiple independent layout families")
        if self.games_per_iteration % 2:
            raise ValueError("training games must retain paired actor colors")

    def as_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.as_dict(), sort_keys=True).encode()).hexdigest()
