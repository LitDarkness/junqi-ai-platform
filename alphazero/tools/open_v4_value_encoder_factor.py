"""Configuration-only Value encoder learning-rate factor."""
import copy
from pathlib import Path


def variant_config(base, directory, source, variant, factors):
    if factors != {"control": .1, "candidate": .5} or variant not in factors:
        raise ValueError("only registered .1/.5 Value encoder comparison")
    if base.value_encoder_lr_multiplier != .1:
        raise ValueError("unexpected original Value encoder multiplier")
    cfg = copy.deepcopy(base)
    cfg.run_directory = str(Path(directory) / variant)
    cfg.initial_checkpoint = None if source is None else str(source)
    cfg.value_encoder_lr_multiplier = factors[variant]
    cfg.validate()
    changed = {key for key, value in cfg.as_dict().items() if value != base.as_dict()[key]}
    if changed - {"run_directory", "initial_checkpoint", "value_encoder_lr_multiplier"}:
        raise ValueError("unregistered training factor")
    return cfg
