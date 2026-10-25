"""Isolated learning-rate factor; keep teacher child imports free of Torch."""
from pathlib import Path


def make_learner(config, root, factor):
    from junqi_open_v4.learner import Learner

    if factor not in (0., .25):
        raise ValueError("only registered policy encoder multipliers 0 and .25")

    class PolicyEncoderLearner(Learner):
        def _set_phase(self, phase, scale=1.):
            super()._set_phase(phase, scale)
            if phase == "policy":
                for group in self.optimizer.param_groups:
                    if group["name"] == "encoder":
                        group["lr"] = self.config.learning_rate * scale * factor

        def payload(self, *, role, lineage):
            result = super().payload(role=role, lineage=lineage)
            result["experiment_parameters"] = {"policy_encoder_lr_multiplier": factor}
            return result

        def resume(self, path):
            import torch
            saved = torch.load(Path(path), map_location="cpu", weights_only=False)
            if saved.get("experiment_parameters") != {"policy_encoder_lr_multiplier": factor}:
                raise ValueError("experiment learning-rate factor mismatch")
            return super().resume(path)

    return PolicyEncoderLearner(config, root)
