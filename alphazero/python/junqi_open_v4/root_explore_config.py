"""One preregistered root-exploration target factor."""
from dataclasses import dataclass
from .queued_config import QueuedConfig,root_noise_enabled
@dataclass
class RootExploreConfig(QueuedConfig):
    forced_root_coeff: float = 2.0
    exploration_target_pruning: bool = False
    def validate(self):
        super().validate()
        if self.forced_root_coeff!=2.0 or self.leaves_per_tree!=1 or self.actor_policy_target!='teacher':
            raise ValueError('registered k2,one leaf and original teacher labels required')
