"""One new policy-step-rate factor; frozen collector and Value objectives unchanged."""
from dataclasses import dataclass
from .root_explore_config import RootExploreConfig


@dataclass
class StablePolicyConfig(RootExploreConfig):
    policy_step_lr_multiplier: float = .25

    def validate(self):
        super().validate()
        if self.policy_step_lr_multiplier != .25:
            raise ValueError('registered Policy step multiplier must be .25')

