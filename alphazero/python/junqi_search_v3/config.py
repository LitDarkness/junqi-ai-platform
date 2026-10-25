from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from junqi_az.config import ModelConfig


@dataclass
class BoundaryConfig:
    layout_set: str = "config/layouts/fixed_layout_v1.json"
    layout_id: str = "public_conservative_advance"
    fixed_first_player: int = 1
    mirror: bool = False
    multiple_layouts: bool = False
    max_ply: int = 1000
    no_capture_ply_limit: int = 200
    repetition_limit: int = 3


@dataclass
class OracleConfig:
    name: str = "supplement_wide_quiescence"
    budgets: list[int] = field(default_factory=lambda: [500, 2000, 8000, 32000])
    max_depth: int = 7
    shallow_fraction: float = 0.30
    scale_calibration_states: int = 64
    scale_calibration_budget: int = 2000
    scale_quantile: float = 0.90
    stability_states: int = 32
    minimum_stability_8000_to_32000: float = 0.50
    minimum_top3_stability_8000_to_32000: float = 0.70
    maximum_mean_regret_8000_to_32000: float = 0.02


@dataclass
class DataConfig:
    trajectory_families: int = 1024
    workers: int = 16
    transaction_families: int = 16
    target_unique_states_min: int = 20000
    target_unique_states_max: int = 40000
    trajectory_temperature: float = 0.25
    oracle_actor_side_schedule: str = "balanced_by_family"
    branch_counts: dict[str, float] = field(
        default_factory=lambda: {"0": 0.10, "1": 0.40, "2": 0.35, "3": 0.15}
    )
    opponents: dict[str, float] = field(
        default_factory=lambda: {
            "supplement_wide_quiescence": 0.40,
            "exact082": 0.25,
            "random": 0.15,
            "aggressive": 0.10,
            "frozen_weak_network": 0.10,
        }
    )
    tau_labels: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.5])
    probe_states: int = 512
    normal_budget: int = 2000
    important_budget: int = 8000
    probe_budget: int = 32000
    exact082_candidate: str = "../teachers/082/candidate.py"
    weak_checkpoint: str = "runs/open_junqi_search_v3/checkpoints/initial_random.pt"


@dataclass
class TrainConfig:
    batch_size: int = 1024
    overfit_batch_size: int = 2048
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 1e-5
    weight_decay: float = 1e-4
    max_gradient_norm: float = 1.0
    terminal_wdl_weight: float = 1.0
    oracle_value_aux_weight: float = 0.1
    policy_weight: float = 1.0
    epochs: int = 20


@dataclass
class PUCTConfig:
    cpuct: float = 1.5
    inference_batch_size: int = 256
    simulations: list[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])
    selfplay_simulations: int = 64
    repeated_state_simulations: int = 128
    dirichlet_epsilon: float = 0.25
    search_gain_direct_games: int = 512
    search_gain_external_games: int = 64


@dataclass
class LoopConfig:
    iterations: int = 10
    games_per_iteration: int = 1024
    transaction_games: int = 64
    replay_iterations: int = 5
    maximum_replay_samples: int = 200000
    state_policy_cap_per_iteration: int = 4
    check_games: int = 128
    minimum_unique_state_ratio: float = 0.20
    maximum_prefix_share: float = 0.20
    consecutive_search_gain_failures: int = 2


@dataclass
class TrainingLogicConfig:
    schema: str = "late_stage_short_horizon_v1"
    source_iteration: int = 107
    policy_lr_multiplier: float = 0.5
    policy_candidate_steps: list[int] = field(
        default_factory=lambda: [1, 2, 3, 4, 6]
    )
    selected_policy_step: int = 3
    value_candidate_steps: list[int] = field(
        default_factory=lambda: [0, 1, 2]
    )
    selected_value_step: int = 2
    policy_shortlist_size: int = 2
    candidate_policy_kl_limit: float = 0.006
    value_final_kl_limit: float = 0.001
    value_peak_kl_limit: float = 0.0015
    bootstrap_expected_model_hash: str = (
        "4feaa5f7b6741c032b8f034e9d465b0391be74b011f354ea6c4979e89bf065b9"
    )
    adaptive_selection: bool = True


@dataclass
class SearchV3Config:
    schema: str = "open-junqi-search-v3"
    seed: int = 2026072501
    device: str = "cuda"
    run_directory: str = "runs/open_junqi_search_v3"
    initialization: str = "random"
    ablation_source: str = (
        "runs/fixed_layout_rl_v2b_advantage_multilayout/checkpoints/champion.pt"
    )
    boundary: BoundaryConfig = field(default_factory=BoundaryConfig)
    oracle: OracleConfig = field(default_factory=OracleConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    puct: PUCTConfig = field(default_factory=PUCTConfig)
    loop: LoopConfig = field(default_factory=LoopConfig)
    # Omitted from legacy serialization/hashes.  It is present only for the
    # separately versioned post-iteration-107 training logic.
    training_logic: TrainingLogicConfig | None = None
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

    @classmethod
    def load(cls, path: str | Path) -> "SearchV3Config":
        source = Path(path).resolve()
        raw = json.loads(source.read_text(encoding="utf-8-sig"))
        if "extends" in raw:
            base = (source.parent / str(raw["extends"])).resolve()
            merged = json.loads(base.read_text(encoding="utf-8-sig"))

            def update(target: dict[str, Any], override: dict[str, Any]) -> None:
                for key, value in override.items():
                    if key == "extends":
                        continue
                    if (
                        isinstance(value, dict)
                        and isinstance(target.get(key), dict)
                    ):
                        update(target[key], value)
                    else:
                        target[key] = value

            update(merged, raw)
            raw = merged
        result = cls(
            schema=raw["schema"],
            seed=int(raw["seed"]),
            device=raw["device"],
            run_directory=raw["run_directory"],
            initialization=raw.get("initialization", "random"),
            ablation_source=raw.get(
                "ablation_source",
                cls().ablation_source,
            ),
            boundary=BoundaryConfig(**raw["boundary"]),
            oracle=OracleConfig(**raw["oracle"]),
            data=DataConfig(**raw["data"]),
            train=TrainConfig(**raw["train"]),
            puct=PUCTConfig(**raw["puct"]),
            loop=LoopConfig(**raw["loop"]),
            training_logic=(
                TrainingLogicConfig(**raw["training_logic"])
                if raw.get("training_logic") is not None
                else None
            ),
            model=ModelConfig(**raw["model"]),
        )
        result.validate()
        return result

    def validate(self) -> None:
        if self.schema != "open-junqi-search-v3":
            raise ValueError("unsupported search-v3 schema")
        if self.initialization not in {
            "random",
            "v2b_trunk_ablation",
            "alternating_iteration3_recovery",
            "strength_incumbent",
        }:
            raise ValueError("invalid initialization")
        if self.boundary.mirror or self.boundary.multiple_layouts:
            raise ValueError("generation 1 is strictly single-layout and unmirrored")
        if self.boundary.fixed_first_player != 1:
            raise ValueError("generation 1 fixes RED as first player")
        if self.oracle.budgets != [500, 2000, 8000, 32000]:
            raise ValueError("approved Oracle budgets changed")
        if self.data.oracle_actor_side_schedule != "balanced_by_family":
            raise ValueError(
                "fixed-layout Oracle actor sides must be balanced by family"
            )
        if self.train.policy_weight < 0.5:
            raise ValueError("policy loss must retain a main-order weight")
        if self.train.oracle_value_aux_weight > 0.1:
            raise ValueError("Oracle value auxiliary exceeds 0.1")
        if self.model.action_size != 3600 or self.model.input_channels != 107:
            raise ValueError("native state/action contract changed")
        if not self.model.repetition_context:
            raise ValueError(
                "per-legal-action repetition context must reach the network"
            )
        if (
            self.boundary.repetition_limit != 3
            or self.boundary.no_capture_ply_limit != 200
            or self.boundary.max_ply != 1000
        ):
            raise ValueError(
                "107-plane normalized rule counters require the frozen "
                "3/200/1000 repetition/no-capture/max-ply thresholds"
            )
        if (
            self.puct.search_gain_direct_games < 512
            or self.puct.search_gain_direct_games % 2
            or self.puct.search_gain_external_games < 2
            or self.puct.search_gain_external_games % 2
        ):
            raise ValueError("invalid paired Search Gain game counts")
        if (
            self.loop.iterations < 5
            or self.loop.games_per_iteration != 1024
            or self.loop.transaction_games < 1
            or self.loop.state_policy_cap_per_iteration != 4
        ):
            raise ValueError("closed-loop generation-1 contract changed")
        if self.training_logic is not None:
            logic = self.training_logic
            if logic.schema != "late_stage_short_horizon_v1":
                raise ValueError("unsupported training logic schema")
            if logic.source_iteration != 107:
                raise ValueError("late-stage source iteration changed")
            if not 0.0 < logic.policy_lr_multiplier <= 1.0:
                raise ValueError("invalid late-stage Policy LR multiplier")
            if logic.policy_candidate_steps != [1, 2, 3, 4, 6]:
                raise ValueError("late-stage Policy candidate window changed")
            if logic.value_candidate_steps != [0, 1, 2]:
                raise ValueError("late-stage Value candidate window changed")
            if (
                logic.selected_policy_step
                not in logic.policy_candidate_steps
                or logic.selected_value_step
                not in logic.value_candidate_steps
            ):
                raise ValueError("bootstrap selection is outside its window")
            if logic.policy_shortlist_size not in {1, 2}:
                raise ValueError("late-stage Policy shortlist must be 1 or 2")
            if (
                logic.candidate_policy_kl_limit <= 0.0
                or logic.value_final_kl_limit <= 0.0
                or logic.value_peak_kl_limit
                < logic.value_final_kl_limit
            ):
                raise ValueError("invalid late-stage KL limits")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        if self.training_logic is None:
            # Preserve every legacy config hash exactly.  Old checkpoints and
            # iteration-107 evidence must remain byte-for-byte attributable to
            # the configuration that produced them.
            payload.pop("training_logic", None)
        return payload

    def hash(self) -> str:
        rendered = json.dumps(
            self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(rendered.encode("utf-8")).hexdigest()
