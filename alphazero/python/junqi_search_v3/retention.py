from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from .dataset import atomic_json


def _iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


class CheckpointRetention:
    """Transactional retention for official and candidate checkpoints."""

    def __init__(
        self,
        run: Path,
        *,
        recent: int = 5,
        archive_interval: int = 5,
        starting_incumbent: int = 10,
    ) -> None:
        self.run = run
        self.recent = int(recent)
        self.archive_interval = int(archive_interval)
        self.starting_incumbent = int(starting_incumbent)
        self.registry_path = run / "competitive.json"
        self.manifest_path = run / "retention.json"

    def _champion_iterations(self) -> set[int]:
        path = self.run / "championship.json"
        if not path.exists():
            return {self.starting_incumbent}
        payload = json.loads(path.read_text(encoding="utf-8"))
        result = {int(payload["current"]["iteration"])}
        result.update(
            int(row["iteration"])
            for row in payload.get("history", [])
        )
        return result

    def _competitive_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(
                self.registry_path.read_text(encoding="utf-8")
            )
        return {
            "schema": "search-v3-competitive-checkpoints-v1",
            "iterations": {
                str(self.starting_incumbent): {
                    "reason": "starting frozen championship incumbent",
                    "permanent": True,
                }
            },
        }

    def register(
        self, iteration: int, report: dict[str, Any]
    ) -> bool:
        chosen = report["chosen_candidate"]
        changed = bool(
            int(chosen["policy_steps"]) or int(chosen["value_steps"])
        )
        competitive = (
            changed
            and report["paired_policy"]["classification"]
            in {"improved", "neutral"}
            and report["paired_wdl"]["classification"]
            != "clearly_degraded"
            and report["post_probe"]["search64_oracle_regret"]
            < report["post_probe"]["bare_oracle_regret"]
            and report["value_phase_policy_kl"].get(
                "candidate_final_kl",
                report["value_phase_policy_kl"].get("maximum", 0.0),
            )
            <= 1e-3
            and report["value_phase_policy_kl"].get(
                "value_step_peak_kl",
                report["value_phase_policy_kl"].get("maximum", 0.0),
            )
            <= 1.5e-3
        )
        registry = self._competitive_registry()
        if competitive:
            registry["iterations"][str(iteration)] = {
                "reason": (
                    "accepted non-epoch0 update; Policy/WDL nondegraded; "
                    "Search64 beats bare on immutable probe; KL gate passed"
                ),
                "policy_classification": report["paired_policy"][
                    "classification"
                ],
                "wdl_classification": report["paired_wdl"][
                    "classification"
                ],
                "search64_oracle_regret": report["post_probe"][
                    "search64_oracle_regret"
                ],
                "permanent": True,
            }
        atomic_json(self.registry_path, registry)
        return competitive

    def apply(self, current_iteration: int) -> dict[str, Any]:
        official = {
            _iteration(path): path
            for path in (
                self.run / "checkpoints"
            ).glob("loop_iteration_*.pt")
        }
        candidates = {
            _iteration(path): path
            for path in (
                self.run / "candidates"
            ).glob("iteration_*.pt")
        }
        available = sorted(official)
        recent_iterations = set(available[-self.recent :])
        archive_iterations = {
            iteration
            for iteration in available
            if iteration % self.archive_interval == 0
        }
        champion_iterations = self._champion_iterations()
        registry = self._competitive_registry()
        competitive_iterations = {
            int(iteration)
            for iteration in registry.get("iterations", {})
        }
        keep_official = (
            recent_iterations
            | archive_iterations
            | champion_iterations
            | competitive_iterations
            | {max(available)}
        )
        # A passed iteration's official checkpoint is the same selected
        # model/optimizer/RNG payload with the completed marker committed.
        # Keeping its pre-commit candidate duplicates roughly 79 MiB without
        # adding rollback capability.  Candidates are retained only when no
        # official atomic checkpoint exists (failed/interrupted diagnostics).
        keep_candidates = {
            iteration
            for iteration in (
                recent_iterations
                | champion_iterations
                | competitive_iterations
            )
            if iteration not in official
        }
        delete = []
        for kind, mapping, protected in (
            ("official", official, keep_official),
            ("candidate", candidates, keep_candidates),
        ):
            for iteration, path in mapping.items():
                if iteration not in protected:
                    delete.append(
                        {
                            "kind": kind,
                            "iteration": iteration,
                            "path": str(path),
                            "bytes": path.stat().st_size,
                        }
                    )
        manifest = {
            "schema": "search-v3-checkpoint-retention-v1",
            "time": time.time(),
            "current_iteration": current_iteration,
            "policy": {
                "recent_iterations": self.recent,
                "archive_interval": self.archive_interval,
                "championship_permanent": True,
                "competitive_permanent": True,
                "accepted_candidate_replaced_by_official": True,
            },
            "protected": {
                "recent": sorted(recent_iterations),
                "archive_every_5": sorted(archive_iterations),
                "championship": sorted(champion_iterations),
                "competitive": sorted(competitive_iterations),
                "official": sorted(keep_official),
                "candidates": sorted(keep_candidates),
            },
            "delete_manifest": delete,
            "bytes_reclaimed": sum(row["bytes"] for row in delete),
            "phase": "dry_run_manifest_written",
        }
        atomic_json(self.manifest_path, manifest)
        deleted = []
        for row in delete:
            path = Path(row["path"]).resolve()
            if self.run.resolve() not in path.parents:
                raise RuntimeError(
                    f"retention path escaped run directory: {path}"
                )
            path.unlink()
            deleted.append(row)
        manifest["phase"] = "completed"
        manifest["deleted"] = deleted
        atomic_json(self.manifest_path, manifest)
        return manifest
