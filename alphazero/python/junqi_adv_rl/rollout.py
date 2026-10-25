from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass
class RolloutStep:
    observation: np.ndarray
    legal_actions: np.ndarray
    repetition_contexts: np.ndarray
    selected_action: int
    old_raw_logits: np.ndarray
    behavior_log_prob: float
    sampling_temperature: float
    uniform_mix: float
    acting_side: int
    opponent_type: str
    actor_type: str
    game_id: int
    ply: int
    old_value: float
    old_wdl: np.ndarray
    policy_eligible: bool
    layout_id: str
    moved_piece: int
    outcome: float = 0.0
    terminal_reason: str = ""
    lambda_return: float = 0.0
    advantage: float = 0.0
    wdl_target: np.ndarray | None = None
    teacher_probabilities: np.ndarray | None = None

    def payload(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "RolloutStep":
        return cls(**payload)


def atomic_save_rollout(
    target: str | Path,
    steps: list[RolloutStep],
    *,
    metadata: dict[str, Any],
) -> None:
    destination = Path(target)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    torch.save(
        {
            "schema": "junqi-advantage-fresh-rollout-v2",
            "metadata": metadata,
            "steps": [step.payload() for step in steps],
        },
        temporary,
    )
    # Windows requires a writable handle for FlushFileBuffers/os.fsync.
    with temporary.open("r+b") as stream:
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, destination)


def load_rollout(path: str | Path) -> tuple[list[RolloutStep], dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != "junqi-advantage-fresh-rollout-v2":
        raise ValueError(f"unsupported rollout file: {path}")
    return (
        [RolloutStep.from_payload(row) for row in payload["steps"]],
        dict(payload.get("metadata", {})),
    )


def remove_incomplete_rollouts(directory: str | Path) -> list[str]:
    root = Path(directory)
    removed: list[str] = []
    if root.is_dir():
        for path in root.glob("*.partial"):
            path.unlink()
            removed.append(str(path))
    return removed
