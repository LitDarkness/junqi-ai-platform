from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class RewardConfig:
    mode: str = "shaped"
    win_reward: float = 30.0
    loss_reward: float = -30.0
    draw_reward: float = -2.0
    invalid_action_penalty: float = -10.0
    step_penalty: float = -0.005
    material_scale: float = 0.5
    truncation_penalty: float = -5.0
    material_tiebreak_scale: float = 5.0
    turn_limit_mode: str = "draw"
    no_combat_limit: int = 40
    no_combat_draw_reward: float = -2.0

    @staticmethod
    def preset(name: str) -> "RewardConfig":
        normalized = name.lower()
        if normalized == "terminal":
            return RewardConfig(
                mode="terminal",
                draw_reward=-2.0,
                step_penalty=0.0,
                material_scale=0.0,
                truncation_penalty=0.0,
                material_tiebreak_scale=0.0,
                turn_limit_mode="draw",
                no_combat_limit=40,
                no_combat_draw_reward=-2.0,
            )
        if normalized == "hybrid":
            return RewardConfig(
                mode="hybrid",
                draw_reward=-2.0,
                step_penalty=-0.001,
                material_scale=0.15,
                truncation_penalty=-2.0,
                material_tiebreak_scale=2.0,
                turn_limit_mode="draw",
                no_combat_limit=40,
                no_combat_draw_reward=-2.0,
            )
        return RewardConfig(mode="shaped")

    def with_overrides(
        self,
        *,
        draw_reward: float | None = None,
        no_combat_limit: int | None = None,
        no_combat_draw_reward: float | None = None,
        turn_limit_mode: str | None = None,
    ) -> "RewardConfig":
        updated = self
        if draw_reward is not None:
            updated = replace(updated, draw_reward=draw_reward)
        if no_combat_limit is not None:
            updated = replace(updated, no_combat_limit=no_combat_limit)
        if no_combat_draw_reward is not None:
            updated = replace(updated, no_combat_draw_reward=no_combat_draw_reward)
        if turn_limit_mode is not None:
            updated = replace(updated, turn_limit_mode=turn_limit_mode)
        return updated


@dataclass(frozen=True)
class OpponentPoolConfig:
    random_prob: float = 0.0
    latest_prob: float = 0.4
    history_prob: float = 0.4
    heuristic_prob: float = 0.2
    snapshot_prefix: str = "model_"
    latest_name: str = "latest.pth"
    max_history_models: int = 8

    def normalized(self) -> "OpponentPoolConfig":
        total = self.random_prob + self.latest_prob + self.history_prob + self.heuristic_prob
        if total <= 0:
            return OpponentPoolConfig(0.0, 0.5, 0.5, 0.0, self.snapshot_prefix, self.latest_name, self.max_history_models)
        return OpponentPoolConfig(
            random_prob=self.random_prob / total,
            latest_prob=self.latest_prob / total,
            history_prob=self.history_prob / total,
            heuristic_prob=self.heuristic_prob / total,
            snapshot_prefix=self.snapshot_prefix,
            latest_name=self.latest_name,
            max_history_models=max(1, self.max_history_models),
        )
