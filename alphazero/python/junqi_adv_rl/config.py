from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from junqi_az.config import ModelConfig
from junqi_fixed_rl.config import TeacherConfig


@dataclass
class Layouts:
    set_path: str = "config/layouts/fixed_layout_evaluation_v1.json"
    training_ids: list[str] = field(
        default_factory=lambda: ["public_conservative_advance"]
    )
    sealed_id: str = ""
    mirror_probability: float = 0.0


@dataclass
class SelfPlay:
    games_per_iteration: int = 1024
    parallel_games: int = 256
    max_moves: int = 400
    temperature: float = 0.9
    minimum_temperature: float = 0.25
    temperature_decay_iterations: int = 100
    uniform_mix: float = 0.0
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
class Advantage:
    gamma: float = 1.0
    lambda_advantage: float = 0.5
    lambda_outcome: float = 0.8
    clip_epsilon: float = 0.2
    old_policy_kl_weight: float = 0.1
    magnet_alpha_initial: float = 0.05
    magnet_alpha_exponent: float = 0.3
    magnet_alpha_override: float | None = None
    filter_percentile: float = 75.0
    filter_min_absolute: float = 0.01
    policy_epochs: int = 1


@dataclass
class Train:
    batch_size: int = 1024
    learning_rate: float = 5e-5
    minimum_learning_rate: float = 5e-6
    weight_decay: float = 1e-4
    max_gradient_norm: float = 0.5
    schedule_iterations: int = 200
    ema_smoothing: float = 0.999
    value_epochs: int = 1
    value_weight: float = 1.0
    teacher_weight_initial: float = 0.30
    teacher_weight_final: float = 0.0
    teacher_anneal_iterations: int = 30
    amp: bool = True
    amp_dtype: str = "bfloat16"


@dataclass
class Evaluation:
    enabled: bool = True
    interval: int = 5
    games_per_match: int = 256
    temperature: float = 0.25
    seed: int = 2026072450


@dataclass
class Checkpoints:
    archive_interval: int = 5
    recent_keep: int = 5


@dataclass
class AdvantageRLConfig:
    schema: str = "junqi-advantage-rl-v2"
    lineage: str = "v2a"
    seed: int = 2026072401
    device: str = "cuda"
    run_directory: str = "runs/fixed_layout_rl_v2a_advantage_single_layout"
    initialization_checkpoint: str = (
        "runs/fixed_layout_rl_v1/checkpoints/archive/iteration_000040.pt"
    )
    layouts: Layouts = field(default_factory=Layouts)
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
    selfplay: SelfPlay = field(default_factory=SelfPlay)
    advantage: Advantage = field(default_factory=Advantage)
    train: Train = field(default_factory=Train)
    evaluation: Evaluation = field(default_factory=Evaluation)
    checkpoints: Checkpoints = field(default_factory=Checkpoints)

    @classmethod
    def load(cls, path: str | Path) -> "AdvantageRLConfig":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        result = cls(
            schema=str(raw["schema"]),
            lineage=str(raw["lineage"]),
            seed=int(raw["seed"]),
            device=str(raw.get("device", "cuda")),
            run_directory=str(raw["run_directory"]),
            initialization_checkpoint=str(raw["initialization_checkpoint"]),
            layouts=Layouts(**raw["layouts"]),
            model=ModelConfig(**raw["model"]),
            teacher=TeacherConfig(**raw["teacher"]),
            selfplay=SelfPlay(**raw["selfplay"]),
            advantage=Advantage(**raw["advantage"]),
            train=Train(**raw["train"]),
            evaluation=Evaluation(**raw.get("evaluation", {})),
            checkpoints=Checkpoints(**raw.get("checkpoints", {})),
        )
        result.validate(source.parent.parent)
        return result

    def validate(self, project_root: Path) -> None:
        if self.schema != "junqi-advantage-rl-v2":
            raise ValueError(f"unsupported schema: {self.schema}")
        if self.lineage not in {"v2a", "v2b"}:
            raise ValueError("lineage must be v2a or v2b")
        if self.model.input_channels != 107 or self.model.action_size != 3600:
            raise ValueError("native 107-plane / 3600-action contract is required")
        if self.model.value_head_type != "wdl" or not self.model.pairwise_action_bias:
            raise ValueError("v2 requires pairwise policy and WDL value heads")
        if not self.layouts.training_ids:
            raise ValueError("at least one training layout is required")
        if self.lineage == "v2a" and (
            len(self.layouts.training_ids) != 1
            or self.layouts.mirror_probability != 0.0
        ):
            raise ValueError("v2a must exactly preserve the v1 single layout")
        if self.lineage == "v2b" and (
            len(self.layouts.training_ids) != 4
            or not self.layouts.sealed_id
            or self.layouts.sealed_id in self.layouts.training_ids
        ):
            raise ValueError("v2b requires four training layouts and one sealed layout")
        if not 0.0 <= self.layouts.mirror_probability <= 1.0:
            raise ValueError("mirror_probability must be in [0,1]")
        if self.advantage.gamma != 1.0:
            raise ValueError("the approved experiment fixes gamma=1")
        if self.advantage.policy_epochs != 1:
            raise ValueError("fresh policy rollouts may be trained for one epoch only")
        if self.train.learning_rate > 1e-4:
            raise ValueError("initial learning_rate must be <= 1e-4")
        if self.train.max_gradient_norm <= 0:
            raise ValueError("max_gradient_norm must be positive")
        required = {
            "current", "historical", "heuristic_current",
            "heuristic_historical", "random", "aggressive",
        }
        if set(self.selfplay.opponent_weights) != required:
            raise ValueError("opponent population differs from the v1 contract")
        init_path = Path(self.initialization_checkpoint)
        if not init_path.is_absolute():
            init_path = project_root / init_path
        forbidden = ("000100", "000150", "000163", "000165")
        if any(token in init_path.name for token in forbidden):
            raise ValueError("forbidden collapsed checkpoint selected")
        if not init_path.is_file():
            raise FileNotFoundError(init_path)
        layout_path = Path(self.layouts.set_path)
        if not layout_path.is_absolute():
            layout_path = project_root / layout_path
        if not layout_path.is_file():
            raise FileNotFoundError(layout_path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def hash(self) -> str:
        canonical = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
