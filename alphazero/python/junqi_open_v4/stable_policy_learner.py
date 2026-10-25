"""Reduce only Policy-phase LR; preserve Adam moments, Value LR and replay sampling."""
from types import MethodType


def stable_set_phase(self, phase, scale=1.):
    from .learner import Learner
    factor = self.config.policy_step_lr_multiplier if phase == 'policy' else 1.
    return Learner._set_phase(self, phase, scale * factor)


def make_stable_policy_learner(config, root):
    from .training_replay import make_self_training_learner
    learner = make_self_training_learner(config, root)
    learner._set_phase = MethodType(stable_set_phase, learner)
    return learner

