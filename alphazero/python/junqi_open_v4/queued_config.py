"""Explicit configuration for post-review online training branches."""
from dataclasses import dataclass
from .config import OpenConfig


@dataclass
class QueuedConfig(OpenConfig):
    root_noise_until_ply: int | None = None

    def validate(self):
        super().validate()
        if self.root_noise_until_ply not in (None,24):
            raise ValueError('only the registered late root-noise schedule is allowed')


def root_noise_enabled(config,ply):
    return config.root_noise_until_ply is None or ply<config.root_noise_until_ply
