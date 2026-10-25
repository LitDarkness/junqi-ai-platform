from __future__ import annotations

import copy
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .alternating_lineage import (
    FormalAlternatingLineage,
    _sha256,
    _tree_hash,
)
from .closed_loop import ClosedLoopRunner
from .dataset import atomic_json
from .retention import CheckpointRetention
from .stages import SearchV3Stages


class ScaledTrainingLineage(FormalAlternatingLineage):
    """Successor training lineage rooted at frozen iteration 10."""

    def __init__(self, config_path: str | Path) -> None:
        self.stages = SearchV3Stages(config_path)
        self.config = self.stages.config
        self.root = self.stages.root
        self.output = self.stages.run
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.source_run = (
            self.root / "runs" / "open_junqi_search_v3_alternating"
        ).resolve()
        self.strength_run = (
            self.root / "runs" / "open_junqi_search_v3_strength"
        ).resolve()
        self.recovery_run = self.strength_run
        self.frozen = None
        self.frozen_path = self.output / "FROZEN.json"
        self.start_iteration = 11
        self.previous_data_iteration = 10
        self.required_confirmation_iterations: set[int] = set()
        self.status_filename = "TRAINING_STATUS.json"
        self._prepare_scaled_assets()
        scale = json.loads(
            (self.output / "oracle_scale.json").read_text(encoding="utf-8")
        )["score_scale"]
        self.runner = ClosedLoopRunner(
            root=self.root,
            run=self.output,
            backend=self.stages.backend,
            config=self.config,
            score_scale=float(scale),
            replay_trace=self.stages._replay_trace,
        )
        self.score_scale = float(scale)
        self.seed = self.config.seed ^ 0x5343414C
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self._initialize_scaled_lineage()
        self.retention = CheckpointRetention(
            self.output,
            recent=5,
            archive_interval=5,
            starting_incumbent=10,
        )

    def _copy_asset(self, relative: str) -> None:
        source = self.source_run / relative
        target = self.output / relative
        if source.is_dir():
            if not target.exists():
                shutil.copytree(source, target)
            if _tree_hash(source) != _tree_hash(target):
                raise RuntimeError(f"scaled asset mismatch: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if _sha256(source) != _sha256(target):
                raise RuntimeError(f"scaled asset mismatch: {relative}")

    def _prepare_scaled_assets(self) -> None:
        for relative in (
            "probe_manifest.json",
            "oracle_scale.json",
            "oracle_dataset",
            "search_replay/iteration_000009",
            "search_replay/iteration_000010",
        ):
            self._copy_asset(relative)

    def _initialize_scaled_lineage(self) -> None:
        source_checkpoint = (
            self.strength_run
            / "checkpoints"
            / "incumbent_iteration_000010.pt"
        )
        source_report = (
            self.source_run / "reports" / "iteration_000010.json"
        )
        frozen_manifest = self.source_run / "FROZEN.json"
        if not source_checkpoint.exists() or not source_report.exists():
            raise RuntimeError("strength incumbent iteration 10 is missing")
        source = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        report = json.loads(
            source_report.read_text(encoding="utf-8")
        )
        if (
            source.get("completed") is not True
            or int(source.get("iteration", -1)) != 10
            or report.get("passed") is not True
        ):
            raise RuntimeError("strength incumbent iteration 10 is invalid")
        manifest = {
            "schema": "open-junqi-search-v3-scaled-training-v1",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_report": str(source_report),
            "source_report_sha256": _sha256(source_report),
            "source_frozen_manifest": str(frozen_manifest),
            "source_frozen_manifest_sha256": _sha256(frozen_manifest),
            "start_iteration": 10,
            "next_iteration": 11,
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "retention": {
                "recent": 5,
                "archive_interval": 5,
                "championship_permanent": True,
                "competitive_permanent": True,
            },
        }
        atomic_json(self.output / "manifest.json", manifest)
        checkpoint = self._checkpoint(10)
        if not checkpoint.exists():
            payload = copy.deepcopy(source)
            payload.update(
                {
                    "schema": (
                        "open-junqi-search-v3-scaled-training-checkpoint"
                    ),
                    "config": self.config.to_dict(),
                    "config_hash": self.config.hash(),
                    "source_checkpoint": str(source_checkpoint),
                    "source_checkpoint_sha256": _sha256(
                        source_checkpoint
                    ),
                    "phase": "scaled_training_incumbent_import",
                }
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(payload, temporary)
            os.replace(temporary, checkpoint)
        else:
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                payload.get("config_hash") != self.config.hash()
                or payload.get("source_checkpoint_sha256")
                != _sha256(source_checkpoint)
            ):
                raise RuntimeError("stale scaled-training incumbent")
        championship = {
            "schema": "search-v3-championship-v1",
            "current": {
                "iteration": 10,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": _sha256(checkpoint),
                "source": "frozen strength incumbent",
                "continuation_report": str(
                    self.strength_run
                    / "continuation_arena"
                    / "report.json"
                ),
            },
            "history": [],
            "automatic_promotion": False,
            "promotion_policy": (
                "championship remains frozen until an independent paired "
                "continuation arena approves a successor"
            ),
        }
        championship_path = self.output / "championship.json"
        if not championship_path.exists():
            atomic_json(championship_path, championship)
        competitive_path = self.output / "competitive.json"
        if not competitive_path.exists():
            atomic_json(
                competitive_path,
                {
                    "schema": (
                        "search-v3-competitive-checkpoints-v1"
                    ),
                    "iterations": {
                        "10": {
                            "reason": (
                                "starting frozen championship incumbent"
                            ),
                            "permanent": True,
                        }
                    },
                },
            )

    def _after_successful_iteration(
        self,
        iteration: int,
        report: dict[str, Any],
        checkpoint: Path,
        candidate_checkpoint: Path,
    ) -> None:
        del checkpoint, candidate_checkpoint
        competitive = self.retention.register(iteration, report)
        retention = self.retention.apply(iteration)
        championship = json.loads(
            (self.output / "championship.json").read_text(
                encoding="utf-8"
            )
        )
        print(
            f"[retention] iteration={iteration} "
            f"recent={retention['protected']['recent']} "
            f"archive={retention['protected']['archive_every_5']} "
            f"competitive={'yes' if competitive else 'no'} "
            f"deleted={len(retention['deleted'])} "
            f"reclaimed={retention['bytes_reclaimed'] / (1024**2):.1f}MiB",
            flush=True,
        )
        print(
            f"[championship] frozen-champion="
            f"iteration_{championship['current']['iteration']:06d} "
            f"latest=iteration_{iteration:06d} "
            f"challenger-status="
            f"{'COMPETITIVE_SAVED' if competitive else 'LATEST_ONLY'}",
            flush=True,
        )


def run_scaled_training(
    config_path: str | Path, through_iteration: int
) -> dict[str, Any]:
    return ScaledTrainingLineage(config_path).run(through_iteration)
