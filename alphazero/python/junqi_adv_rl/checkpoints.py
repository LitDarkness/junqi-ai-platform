from __future__ import annotations

import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(payload: dict[str, Any]) -> None:
    random.setstate(payload["python"])
    np.random.set_state(payload["numpy"])
    torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])


def atomic_torch_save(payload: dict[str, Any], target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, target)


def link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


class AdvantageCheckpointManager:
    def __init__(self, run: Path, archive_interval: int, recent_keep: int) -> None:
        self.run = run
        self.latest = run / "latest.pt"
        self.champion = run / "checkpoints" / "champion.pt"
        self.archive = run / "checkpoints" / "archive"
        self.recent = run / "checkpoints" / "recent"
        self.archive_interval = int(archive_interval)
        self.recent_keep = int(recent_keep)
        self.archive.mkdir(parents=True, exist_ok=True)
        self.recent.mkdir(parents=True, exist_ok=True)

    def save(self, payload: dict[str, Any], iteration: int) -> dict[str, str]:
        atomic_torch_save(payload, self.latest)
        result = {"latest": str(self.latest)}
        if iteration > 0:
            recent = self.recent / f"iteration_{iteration:06d}.pt"
            link_or_copy(self.latest, recent)
            result["recent"] = str(recent)
            if iteration % self.archive_interval == 0:
                archive = self.archive / f"iteration_{iteration:06d}.pt"
                if not archive.exists():
                    link_or_copy(self.latest, archive)
                result["archive"] = str(archive)
            rows = sorted(self.recent.glob("iteration_*.pt"))
            for stale in rows[:-self.recent_keep]:
                stale.unlink()
        return result

    def initialize_champion(self) -> None:
        if not self.champion.exists():
            link_or_copy(self.latest, self.champion)
            metadata = {
                "schema": "junqi-advantage-champion-v2",
                "source_iteration": 0,
                "checkpoint": str(self.champion),
                "note": "EMA bootstrap baseline; evaluation promotion is read-only",
                "history": [],
            }
            (self.run / "champion.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

    def promote_champion(
        self, iteration: int, evaluation: dict[str, Any]
    ) -> None:
        metadata_path = self.run / "champion.json"
        previous = (
            json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata_path.exists()
            else None
        )
        permanent = self.archive / f"iteration_{iteration:06d}.pt"
        if not permanent.exists():
            link_or_copy(self.latest, permanent)
        link_or_copy(permanent, self.champion)
        history = list((previous or {}).get("history", []))
        if previous:
            history.append(
                {
                    "source_iteration": previous.get("source_iteration"),
                    "checkpoint": previous.get("checkpoint"),
                    "promoted_at": previous.get("promoted_at"),
                }
            )
        payload = {
            "schema": "junqi-advantage-champion-v2",
            "source_iteration": iteration,
            "checkpoint": str(self.champion),
            "source_checkpoint": str(permanent),
            "promoted_at": time.time(),
            "evaluation": evaluation,
            "history": history,
        }
        temporary = metadata_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, metadata_path)
