from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from .dataset import atomic_json


def _iteration(path: Path) -> int:
    return int(path.name.rsplit("_", 1)[-1])


class TrainingDataRetention:
    """Bound raw replay and console growth without touching reports."""

    def __init__(
        self,
        run: Path,
        *,
        recent_replay: int = 3,
        replay_archive_interval: int = 10,
        recent_console_logs: int = 5,
    ) -> None:
        self.run = run
        self.recent_replay = max(2, int(recent_replay))
        self.replay_archive_interval = max(
            1, int(replay_archive_interval)
        )
        self.recent_console_logs = max(1, int(recent_console_logs))
        self.manifest_path = run / "data_retention.json"

    def apply(
        self,
        current_iteration: int,
        *,
        champion_iterations: set[int],
    ) -> dict[str, Any]:
        replay_root = self.run / "search_replay"
        replay = {
            _iteration(path): path
            for path in replay_root.glob("iteration_*")
            if path.is_dir()
        }
        available = sorted(replay)
        recent = set(available[-self.recent_replay :])
        archive = {
            iteration
            for iteration in available
            if iteration % self.replay_archive_interval == 0
        }
        keep = recent | archive | set(champion_iterations)
        delete_replay = [
            {
                "iteration": iteration,
                "path": str(path),
                "bytes": sum(
                    item.stat().st_size
                    for item in path.rglob("*")
                    if item.is_file()
                ),
            }
            for iteration, path in replay.items()
            if iteration not in keep
        ]
        console = self.run / "console"
        logs = sorted(
            console.glob("*.log"),
            key=lambda path: path.stat().st_mtime,
        )
        delete_logs = logs[: -self.recent_console_logs]
        candidate_dev = []
        for path in sorted(
            (self.run / "candidate_dev").glob(
                "iteration_*/candidate.pt"
            )
        ):
            iteration = _iteration(path.parent)
            report = (
                self.run
                / "reports"
                / f"iteration_{iteration:06d}.json"
            )
            if iteration <= current_iteration and report.exists():
                candidate_dev.append(path)
        manifest = {
            "schema": "search-v3-training-data-retention-v1",
            "time": time.time(),
            "current_iteration": current_iteration,
            "policy": {
                "recent_full_replay": self.recent_replay,
                "full_replay_archive_interval": (
                    self.replay_archive_interval
                ),
                "champion_replay_permanent": True,
                "all_iteration_reports_permanent": True,
                "all_championship_suites_permanent": True,
                "recent_console_logs": self.recent_console_logs,
            },
            "protected_replay": {
                "recent": sorted(recent),
                "archive_every_10": sorted(archive),
                "championship": sorted(champion_iterations),
                "all": sorted(keep),
            },
            "delete_replay_manifest": delete_replay,
            "delete_console_manifest": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
                for path in delete_logs
            ],
            "delete_candidate_dev_manifest": [
                {
                    "iteration": _iteration(path.parent),
                    "path": str(path),
                    "bytes": path.stat().st_size,
                }
                for path in candidate_dev
            ],
            "bytes_reclaimed": sum(
                row["bytes"] for row in delete_replay
            )
            + sum(path.stat().st_size for path in delete_logs)
            + sum(path.stat().st_size for path in candidate_dev),
            "phase": "dry_run_manifest_written",
        }
        atomic_json(self.manifest_path, manifest)
        run_resolved = self.run.resolve()
        for row in delete_replay:
            path = Path(row["path"]).resolve()
            if run_resolved not in path.parents:
                raise RuntimeError(
                    f"replay retention escaped run directory: {path}"
                )
            shutil.rmtree(path)
        for path in delete_logs:
            resolved = path.resolve()
            if run_resolved not in resolved.parents:
                raise RuntimeError(
                    f"log retention escaped run directory: {resolved}"
                )
            resolved.unlink()
        for path in candidate_dev:
            resolved = path.resolve()
            if run_resolved not in resolved.parents:
                raise RuntimeError(
                    "candidate-dev retention escaped run directory: "
                    f"{resolved}"
                )
            resolved.unlink()
        manifest["phase"] = "completed"
        manifest["deleted_replay"] = delete_replay
        manifest["deleted_console_logs"] = [
            str(path) for path in delete_logs
        ]
        manifest["deleted_candidate_dev"] = [
            str(path) for path in candidate_dev
        ]
        atomic_json(self.manifest_path, manifest)
        return manifest


def trim_competitive_registry(
    path: Path,
    *,
    champion_iterations: set[int],
    maximum_non_champion: int = 8,
) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema": "search-v3-competitive-checkpoints-v1",
            "iterations": {},
        }
    registry = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        int(iteration): payload
        for iteration, payload in registry.get(
            "iterations", {}
        ).items()
    }
    non_champion = [
        (iteration, payload)
        for iteration, payload in rows.items()
        if iteration not in champion_iterations
    ]
    best = sorted(
        non_champion,
        key=lambda item: (
            float(
                item[1].get(
                    "search64_oracle_regret", float("inf")
                )
            ),
            -item[0],
        ),
    )[: max(0, maximum_non_champion // 2)]
    recent = sorted(
        non_champion, key=lambda item: item[0], reverse=True
    )[: max(0, maximum_non_champion // 2)]
    keep = set(champion_iterations)
    keep.update(
        iteration
        for iteration, payload in rows.items()
        if bool(payload.get("retention_lock", False))
    )
    keep.update(iteration for iteration, _ in best)
    keep.update(iteration for iteration, _ in recent)
    registry["iterations"] = {
        str(iteration): rows[iteration]
        for iteration in sorted(keep)
        if iteration in rows
    }
    registry["retention"] = {
        "maximum_non_champion": maximum_non_champion,
        "selection": "best half by Search64 regret plus latest half",
        "championship_permanent": True,
    }
    atomic_json(path, registry)
    return registry
