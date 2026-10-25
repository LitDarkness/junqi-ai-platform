from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

import torch

from .config import FixedRLConfig


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    os.replace(temporary, target)


class CheckpointManager:
    """Three-tier latest/champion/archive lifecycle with recent rollback points."""

    def __init__(self, run_directory: Path, config: FixedRLConfig) -> None:
        self.run_directory = Path(run_directory)
        self.config = config
        self.latest_path = self.run_directory / "latest.pt"
        self.root = self.run_directory / "checkpoints"
        self.archive_directory = self.root / "archive"
        self.recent_directory = self.root / "recent"
        self.evaluation_directory = self.run_directory / "evaluations"
        self.crossplay_directory = self.run_directory / "crossplay"
        self.champion_path = self.root / "champion.pt"
        self.champion_metadata_path = self.run_directory / "champion.json"
        self.transaction_path = self.run_directory / "iteration_transaction.json"
        for directory in (
            self.archive_directory,
            self.recent_directory,
            self.evaluation_directory,
            self.crossplay_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def checkpoint_iteration(path: Path) -> int:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return int(payload.get("iteration", -1))

    @staticmethod
    def is_complete_payload(payload: dict[str, Any]) -> bool:
        if payload.get("iteration_completed") is not None:
            return bool(payload["iteration_completed"]) and payload.get("phase") == "completed"
        iteration = int(payload.get("iteration", -1))
        if iteration < 0:
            return False
        config = payload.get("config", {})
        teacher_steps = int(config.get("teacher", {}).get("bootstrap_train_steps", 0))
        per_iteration = int(config.get("train", {}).get("steps_per_iteration", 0))
        games_per_iteration = int(
            config.get("selfplay", {}).get("games_per_iteration", 0)
        )
        expected_step = teacher_steps + iteration * per_iteration
        expected_games = iteration * games_per_iteration
        actual_step = int(payload.get("trainer", {}).get("global_step", -1))
        actual_games = int(payload.get("games_generated", -1))
        return actual_step == expected_step and actual_games == expected_games

    def find_last_complete_legacy(self) -> Path | None:
        candidates = []
        legacy = self.run_directory / "history"
        candidates.extend(sorted(legacy.glob("iteration_*.pt")))
        candidates.extend(sorted(self.archive_directory.glob("iteration_*.pt")))
        candidates.extend(sorted(self.recent_directory.glob("iteration_*.pt")))
        complete: list[tuple[int, Path]] = []
        for path in candidates:
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                if self.is_complete_payload(payload):
                    complete.append((int(payload["iteration"]), path))
            except Exception:
                continue
        return max(complete, default=(None, None), key=lambda row: row[0])[1]

    def save_latest(self, payload: dict[str, Any]) -> Path:
        temporary = self.latest_path.with_suffix(".pt.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, self.latest_path)
        return self.latest_path

    def _archive_path(self, iteration: int) -> Path:
        return self.archive_directory / f"iteration_{iteration:06d}.pt"

    def _recent_path(self, iteration: int) -> Path:
        return self.recent_directory / f"iteration_{iteration:06d}.pt"

    def register_completed(self, iteration: int) -> dict[str, str]:
        if iteration < 1:
            return {}
        result: dict[str, str] = {}
        recent = self._recent_path(iteration)
        _link_or_copy(self.latest_path, recent)
        result["recent"] = str(recent)
        if iteration % self.config.checkpoint.archive_interval == 0:
            archive = self._archive_path(iteration)
            if not archive.exists():
                _link_or_copy(self.latest_path, archive)
            result["archive"] = str(archive)
        self.prune_recent()
        return result

    def prune_recent(self) -> None:
        rows = sorted(
            (
                (self.checkpoint_iteration(path), path)
                for path in self.recent_directory.glob("iteration_*.pt")
            ),
            key=lambda row: row[0],
        )
        keep = int(self.config.checkpoint.recent_keep)
        for _, path in rows[:-keep]:
            path.unlink(missing_ok=True)

    def ensure_permanent(self, iteration: int, source: Path) -> Path:
        target = self._archive_path(iteration)
        if not target.exists():
            _link_or_copy(source, target)
        return target

    def champion_metadata(self) -> dict[str, Any] | None:
        if not self.champion_metadata_path.is_file():
            return None
        return json.loads(self.champion_metadata_path.read_text(encoding="utf-8"))

    def initialize_champion(
        self,
        *,
        layout_id: str,
        layout_hash: str,
        config_hash: str,
    ) -> dict[str, Any]:
        existing = self.champion_metadata()
        if existing is not None and self.champion_path.is_file():
            return existing
        target_iteration = int(self.config.checkpoint.initial_champion_iteration)
        candidates: list[tuple[int, Path]] = []
        for directory in (
            self.archive_directory,
            self.recent_directory,
            self.run_directory / "history",
        ):
            for path in directory.glob("iteration_*.pt"):
                try:
                    payload = torch.load(path, map_location="cpu", weights_only=False)
                except Exception:
                    continue
                if self.is_complete_payload(payload):
                    candidates.append((int(payload["iteration"]), path))
        exact = [row for row in candidates if row[0] == target_iteration]
        if exact:
            iteration, source = exact[-1]
        else:
            multiples = [
                row
                for row in candidates
                if row[0] > 0
                and row[0] % self.config.checkpoint.archive_interval == 0
            ]
            if not multiples:
                raise RuntimeError(
                    "cannot initialize champion: iteration 10 and complete "
                    "5-multiple checkpoints are unavailable"
                )
            iteration, source = max(multiples, key=lambda row: row[0])
        permanent = self.ensure_permanent(iteration, source)
        _link_or_copy(permanent, self.champion_path)
        metadata = {
            "schema": "junqi-fixed-layout-champion-v1",
            "checkpoint": str(self.champion_path),
            "source_checkpoint": str(permanent),
            "source_iteration": iteration,
            "promoted_at": time.time(),
            "initial": True,
            "evaluation": None,
            "last_evaluation": None,
            "evaluation_history": [],
            "layout_id": layout_id,
            "layout_hash": layout_hash,
            "config_hash": config_hash,
            "previous_champion": None,
            "history": [],
        }
        _atomic_json(self.champion_metadata_path, metadata)
        return metadata

    def promote(
        self,
        candidate: Path,
        iteration: int,
        evaluation: dict[str, Any],
    ) -> dict[str, Any]:
        previous = self.champion_metadata()
        permanent = self.ensure_permanent(iteration, candidate)
        _link_or_copy(permanent, self.champion_path)
        previous_summary = (
            None
            if previous is None
            else {
                "source_iteration": previous.get("source_iteration"),
                "source_checkpoint": previous.get("source_checkpoint"),
                "promoted_at": previous.get("promoted_at"),
            }
        )
        history = list((previous or {}).get("history", []))
        evaluation_history = list(
            (previous or {}).get("evaluation_history", [])
        )
        evaluation_history.append(evaluation)
        if previous_summary is not None:
            history.append(previous_summary)
        metadata = {
            "schema": "junqi-fixed-layout-champion-v1",
            "checkpoint": str(self.champion_path),
            "source_checkpoint": str(permanent),
            "source_iteration": iteration,
            "promoted_at": time.time(),
            "initial": False,
            "evaluation": evaluation,
            "last_evaluation": evaluation,
            "evaluation_history": evaluation_history,
            "layout_id": evaluation["layout_id"],
            "layout_hash": evaluation["layout_hash"],
            "config_hash": evaluation["config_hash"],
            "previous_champion": previous_summary,
            "history": history,
        }
        _atomic_json(self.champion_metadata_path, metadata)
        return metadata

    def record_evaluation(self, evaluation: dict[str, Any]) -> dict[str, Any]:
        metadata = self.champion_metadata()
        if metadata is None:
            raise RuntimeError("cannot record evaluation before champion initialization")
        history = list(metadata.get("evaluation_history", []))
        history.append(evaluation)
        metadata["last_evaluation"] = evaluation
        metadata["evaluation_history"] = history
        _atomic_json(self.champion_metadata_path, metadata)
        return metadata

    def write_transaction(self, payload: dict[str, Any]) -> None:
        _atomic_json(self.transaction_path, payload)

    def read_transaction(self) -> dict[str, Any] | None:
        if not self.transaction_path.is_file():
            return None
        return json.loads(self.transaction_path.read_text(encoding="utf-8"))

    def clear_transaction(self) -> None:
        self.transaction_path.unlink(missing_ok=True)

    @staticmethod
    def hash_payload(payload: Any) -> str:
        rendered = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(rendered).hexdigest()

    def migrate_legacy_history(self, maximum_completed_iteration: int) -> dict[str, int]:
        legacy = self.run_directory / "history"
        if not legacy.is_dir():
            return {"archived": 0, "recent": 0, "deleted": 0}
        complete: dict[int, Path] = {}
        deleted = 0
        for path in sorted(legacy.glob("iteration_*.pt")):
            try:
                payload = torch.load(path, map_location="cpu", weights_only=False)
                iteration = int(payload.get("iteration", -1))
                if (
                    iteration <= maximum_completed_iteration
                    and self.is_complete_payload(payload)
                ):
                    complete[iteration] = path
                else:
                    path.unlink(missing_ok=True)
                    deleted += 1
            except Exception:
                path.unlink(missing_ok=True)
                deleted += 1
        recent_iterations = set(
            sorted(complete)[-int(self.config.checkpoint.recent_keep) :]
        )
        archived = 0
        recent = 0
        for iteration, source in complete.items():
            if iteration > 0 and iteration % self.config.checkpoint.archive_interval == 0:
                target = self._archive_path(iteration)
                if not target.exists():
                    _link_or_copy(source, target)
                archived += 1
            if iteration in recent_iterations and iteration > 0:
                target = self._recent_path(iteration)
                if not target.exists():
                    _link_or_copy(source, target)
                recent += 1
            source.unlink(missing_ok=True)
            deleted += 1
        try:
            legacy.rmdir()
        except OSError:
            pass
        self.prune_recent()
        return {"archived": archived, "recent": recent, "deleted": deleted}
