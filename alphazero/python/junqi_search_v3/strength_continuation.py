from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .alternating_lineage import (
    FormalAlternatingLineage,
    _restore_rng,
    _rng_snapshot,
    _sha256,
    _tree_hash,
)
from .championship import TriggeredSealedChampionship
from .closed_loop import ClosedLoopRunner
from .dataset import atomic_json
from .retention import CheckpointRetention
from .sealed_continuation import MultiCheckpointContinuationEvaluator
from .stages import SearchV3Stages
from .training_retention import (
    TrainingDataRetention,
    trim_competitive_registry,
)


def _state_dict_hash(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def championship_trigger(
    state: dict[str, Any],
    scheduler: dict[str, Any],
) -> tuple[bool, dict[str, int]]:
    accepted = int(
        scheduler.get(
            "accepted_update_iterations",
            scheduler.get("accepted_iterations", 0),
        )
    )
    completed = int(scheduler.get("completed_iterations", 0))
    deltas = {
        "accepted_updates": (
            accepted
            - int(state["accepted_updates_at_last_evaluation"])
        ),
        "completed_rounds": (
            completed
            - int(state["completed_rounds_at_last_evaluation"])
        ),
    }
    return (
        deltas["accepted_updates"] >= 3
        or (
            deltas["completed_rounds"] >= 10
            and deltas["accepted_updates"] >= 1
        )
    ), deltas


class StrengthContinuationLineage(FormalAlternatingLineage):
    """Fixed-layout training rooted atomically at frozen champion 50."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        training_workers: int = 16,
        arena_workers: int = 16,
        arena_block_states: int = 128,
    ) -> None:
        self.stages = SearchV3Stages(config_path)
        self.config = self.stages.config
        self.training_logic = self.config.training_logic
        self.late_stage_short_horizon = (
            self.training_logic is not None
            and self.training_logic.schema
            == "late_stage_short_horizon_v1"
        )
        self.training_workers = max(1, int(training_workers))
        self.config.runtime_workers = self.training_workers
        self.root = self.stages.root
        self.output = self.stages.run
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.source_run = (
            self.root
            / "runs"
            / "open_junqi_search_v3_scaled_training"
        ).resolve()
        self.strength_run = (
            self.root / "runs" / "open_junqi_search_v3_strength"
        ).resolve()
        self.recovery_run = self.source_run
        self.frozen = None
        self.frozen_path = self.output / "FROZEN.json"
        self.start_iteration = (
            108 if self.late_stage_short_horizon else 51
        )
        self.previous_data_iteration = (
            107 if self.late_stage_short_horizon else 50
        )
        self.required_confirmation_iterations: set[int] = set()
        self.status_filename = "TRAINING_STATUS.json"
        self.arena_workers = max(1, int(arena_workers))
        self.arena_block_states = max(1, int(arena_block_states))
        self._prepare_assets()
        scale = json.loads(
            (self.output / "oracle_scale.json").read_text(
                encoding="utf-8"
            )
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
        self.seed = self.config.seed ^ 0x53545243
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self._initialize_lineage()
        self.retention = CheckpointRetention(
            self.output,
            recent=5,
            archive_interval=5,
            starting_incumbent=50,
        )
        self.data_retention = TrainingDataRetention(
            self.output,
            recent_replay=3,
            replay_archive_interval=10,
            recent_console_logs=5,
        )
        self.championship = TriggeredSealedChampionship(
            stages=self.stages,
            training_run=self.output,
            workers=self.arena_workers,
            block_states=self.arena_block_states,
        )

    def _candidate_continuation_dev(
        self,
        iteration: int,
        incumbent: Any,
        candidate: Any,
    ) -> dict[str, Any]:
        states = 64
        suite_path = (
            self.strength_run
            / "single_layout_strength_suite"
            / "states.jsonl"
        )
        seal_path = (
            self.strength_run
            / "single_layout_strength_suite"
            / "manifest.json"
        )
        if not suite_path.exists() or not seal_path.exists():
            raise RuntimeError("fixed continuation dev suite is missing")
        output = (
            self.output
            / "candidate_dev"
            / f"iteration_{iteration:06d}"
        )
        output.mkdir(parents=True, exist_ok=True)
        candidate_checkpoint = output / "candidate.pt"
        candidate_hash = _state_dict_hash(candidate.state_dict())
        if candidate_checkpoint.exists():
            existing = torch.load(
                candidate_checkpoint,
                map_location="cpu",
                weights_only=False,
            )
            if _state_dict_hash(existing["model"]) != candidate_hash:
                raise RuntimeError(
                    "candidate dev checkpoint changed for this iteration"
                )
        else:
            temporary = candidate_checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema": "search-v3-candidate-dev-checkpoint-v1",
                    "iteration": iteration,
                    "model_hash": candidate_hash,
                    "model": {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in candidate.state_dict().items()
                    },
                    "leaf_value_scale": float(
                        candidate.search_v3_leaf_value_scale
                    ),
                },
                temporary,
            )
            os.replace(temporary, candidate_checkpoint)
        incumbent_iteration = iteration - 1
        incumbent_checkpoint = self._checkpoint(incumbent_iteration)
        matchups = (
            (
                f"bare_{iteration}",
                f"bare_{incumbent_iteration}",
            ),
            (
                f"search64_{iteration}",
                f"search64_{incumbent_iteration}",
            ),
        )
        evaluator = MultiCheckpointContinuationEvaluator(
            stages=self.stages,
            models={
                incumbent_iteration: incumbent,
                iteration: candidate,
            },
            checkpoints={
                incumbent_iteration: incumbent_checkpoint,
                iteration: candidate_checkpoint,
            },
            suite_path=suite_path,
            seal=json.loads(seal_path.read_text(encoding="utf-8")),
            output=output / "arena",
            workers=self.arena_workers,
            block_states=min(self.arena_block_states, states),
            matchups=matchups,
        )
        report = evaluator.run(list(matchups), states=states)
        bare = report["matchups"][
            f"bare_{iteration}_vs_bare_{incumbent_iteration}"
        ]
        search = report["matchups"][
            f"search64_{iteration}_vs_search64_{incumbent_iteration}"
        ]
        bare_mean = float(bare["family_weighted_mean_score"])
        search_mean = float(search["family_weighted_mean_score"])
        bare_high = float(bare["family_ci95"][1])
        search_high = float(search["family_ci95"][1])
        accepted = (
            bare_high >= 0.5
            and search_high >= 0.5
            and (bare_mean + search_mean) / 2.0 >= 0.5
        )
        result = {
            "schema": "search-v3-candidate-continuation-dev-v1",
            "states": states,
            "games_per_matchup": states * 2,
            "suite": str(suite_path),
            "suite_sha256": _sha256(suite_path),
            "candidate_model_hash": candidate_hash,
            "incumbent_model_hash": _state_dict_hash(
                incumbent.state_dict()
            ),
            "bare_vs_incumbent": bare,
            "search64_vs_incumbent": search,
            "combined_family_mean_score": (
                bare_mean + search_mean
            )
            / 2.0,
            "accepted": accepted,
            "real_terminal_outcomes_are_selection_gate": True,
            "static_oracle_regret_is_diagnostic_only": True,
        }
        atomic_json(output / "decision.json", result)
        return result

    def _train_iteration(
        self,
        iteration: int,
        model: Any,
        optimizer: torch.optim.AdamW,
        fresh: list[Any],
        candidate_evaluator: Any = None,
        run_continuation_dev: bool = True,
    ) -> tuple[Any, torch.optim.AdamW, dict[str, Any], dict[str, Any]]:
        incumbent = type(model)(self.config.model).to(self.device)
        incumbent.load_state_dict(model.state_dict())
        incumbent.search_v3_leaf_value_scale = getattr(
            model, "search_v3_leaf_value_scale", 1.0
        )
        incumbent_optimizer = copy.deepcopy(optimizer.state_dict())
        incumbent_rng = _rng_snapshot(self.generator, self.runner)
        if self.late_stage_short_horizon:
            from .late_stage_training import train_short_horizon

            trained, optimizer, chosen, details = (
                train_short_horizon(
                    self,
                    iteration,
                    model,
                    optimizer,
                    fresh,
                    candidate_evaluator=candidate_evaluator,
                )
            )
        else:
            trained, optimizer, chosen, details = (
                super()._train_iteration(
                    iteration,
                    model,
                    optimizer,
                    fresh,
                    candidate_evaluator=candidate_evaluator,
                )
            )
        if not run_continuation_dev:
            return trained, optimizer, chosen, details
        if int(chosen["policy_steps"]) + int(chosen["value_steps"]) == 0:
            details["continuation_dev"] = None
            print(
                f"[candidate final] iteration={iteration} "
                f"best-nonzero={details['best_nonzero_candidate']} "
                f"why-epoch0-won={details['why_epoch0_won']}",
                flush=True,
            )
            return trained, optimizer, chosen, details
        dev = self._candidate_continuation_dev(
            iteration, incumbent, trained
        )
        details["continuation_dev"] = dev
        if not dev["accepted"]:
            rejected_index = int(details["chosen_candidate_index"])
            details["candidates"][rejected_index][
                "rejection_reason"
            ] = "continuation_dev_not_better_than_incumbent"
            details["candidates"][rejected_index][
                "eligible"
            ] = False
            details["candidates"][0]["rejection_reason"] = "none"
            trained.load_state_dict(incumbent.state_dict())
            trained.search_v3_leaf_value_scale = getattr(
                incumbent, "search_v3_leaf_value_scale", 1.0
            )
            optimizer.load_state_dict(incumbent_optimizer)
            _restore_rng(incumbent_rng, self.generator, self.runner)
            chosen = details["candidates"][0]
            details["chosen_candidate_index"] = 0
            details["accepted_rng"] = incumbent_rng
            details["why_epoch0_won"] = (
                f"candidate_{rejected_index}_failed_continuation_dev"
            )
        else:
            details["why_epoch0_won"] = (
                "not_applicable_nonzero_passed_continuation_dev"
            )
        print(
            f"[candidate final] iteration={iteration} "
            f"best-nonzero={details['best_nonzero_candidate']} "
            f"selected={details['chosen_candidate_index']} "
            f"dev-bare="
            f"{dev['bare_vs_incumbent']['family_weighted_mean_score']:.4f} "
            f"dev-search="
            f"{dev['search64_vs_incumbent']['family_weighted_mean_score']:.4f} "
            f"why-epoch0-won={details['why_epoch0_won']}",
            flush=True,
        )
        return trained, optimizer, chosen, details

    def _link_or_copy_file(self, source: Path, target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if _sha256(source) != _sha256(target):
            raise RuntimeError(f"asset mismatch: {target}")

    def _link_or_copy_asset(self, relative: str) -> None:
        source = self.source_run / relative
        target = self.output / relative
        if source.is_dir():
            for path in source.rglob("*"):
                if path.is_file():
                    self._link_or_copy_file(
                        path, target / path.relative_to(source)
                    )
            if _tree_hash(source) != _tree_hash(target):
                raise RuntimeError(f"asset tree mismatch: {relative}")
        else:
            self._link_or_copy_file(source, target)

    def _prepare_assets(self) -> None:
        for relative in (
            "probe_manifest.json",
            "oracle_scale.json",
            "oracle_dataset",
        ):
            self._link_or_copy_asset(relative)
        if not self._checkpoint(50).exists():
            for relative in (
                "search_replay/iteration_000049",
                "search_replay/iteration_000050",
            ):
                self._link_or_copy_asset(relative)

    def _freeze_copy(
        self, source: Path, iteration: int
    ) -> Path:
        target = (
            self.output
            / "frozen_champions"
            / f"iteration_{iteration:06d}.pt"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            temporary = target.with_suffix(".pt.tmp")
            try:
                os.link(source, temporary)
            except OSError:
                shutil.copy2(source, temporary)
            os.replace(temporary, target)
        if _sha256(target) != _sha256(source):
            raise RuntimeError(
                f"frozen champion {iteration} copy mismatch"
            )
        return target

    def _initialize_lineage(self) -> None:
        if self.late_stage_short_horizon:
            self._initialize_late_stage_logic()
            return
        source_checkpoint = (
            self.source_run
            / "checkpoints"
            / "loop_iteration_000050.pt"
        )
        source_report = (
            self.source_run / "reports" / "iteration_000050.json"
        )
        old_champion = (
            self.strength_run
            / "checkpoints"
            / "incumbent_iteration_000010.pt"
        )
        if not all(
            path.exists()
            for path in (
                source_checkpoint,
                source_report,
                old_champion,
            )
        ):
            raise RuntimeError(
                "iteration 10/50 source evidence is incomplete"
            )
        source = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        report = json.loads(
            source_report.read_text(encoding="utf-8")
        )
        required = {
            "model",
            "optimizer",
            "scheduler",
            "global_step",
            "rng",
            "replay",
            "leaf_value_calibration",
            "leaf_value_scale",
        }
        if (
            source.get("completed") is not True
            or int(source.get("iteration", -1)) != 50
            or report.get("passed") is not True
            or not required.issubset(source)
        ):
            raise RuntimeError(
                "iteration 50 is not a complete atomic checkpoint"
            )
        frozen50 = self._freeze_copy(source_checkpoint, 50)
        frozen10 = self._freeze_copy(old_champion, 10)
        checkpoint = self._checkpoint(50)
        if not checkpoint.exists():
            payload = copy.deepcopy(source)
            payload.update(
                {
                    "schema": (
                        "open-junqi-search-v3-strength-"
                        "continuation-checkpoint"
                    ),
                    "config": self.config.to_dict(),
                    "config_hash": self.config.hash(),
                    "phase": "strength_continuation_import",
                    "source_checkpoint": str(source_checkpoint),
                    "source_checkpoint_sha256": _sha256(
                        source_checkpoint
                    ),
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
                payload.get("source_checkpoint_sha256")
                != _sha256(source_checkpoint)
            ):
                raise RuntimeError(
                    "strength-continuation source checkpoint changed"
                )
            if payload.get("config_hash") != self.config.hash():
                started = (
                    self.output
                    / "search_replay"
                    / "iteration_000051"
                ).exists()
                if started:
                    raise RuntimeError(
                        "strength-continuation training configuration "
                        "changed after iteration 51 started"
                    )
                payload["config"] = self.config.to_dict()
                payload["config_hash"] = self.config.hash()
                payload["runtime_metadata_migration"] = (
                    "CpuThreads removed from semantic config hash "
                    "before iteration 51"
                )
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, checkpoint)
        imported_report = (
            self.output / "source" / "iteration_000050.json"
        )
        self._link_or_copy_file(source_report, imported_report)
        manifest = {
            "schema": "open-junqi-search-v3-strength-continuation-v1",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_report": str(source_report),
            "source_report_sha256": _sha256(source_report),
            "atomic_state_restored": sorted(required),
            "training_seed_checkpoint": str(checkpoint),
            "frozen_champion_50": str(frozen50),
            "frozen_champion_50_sha256": _sha256(frozen50),
            "historical_champion_10": str(frozen10),
            "historical_champion_10_sha256": _sha256(frozen10),
            "start_iteration": 50,
            "next_iteration": 51,
            "training_workers": self.training_workers,
            "arena_workers": self.arena_workers,
            "arena_block_states": self.arena_block_states,
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "multiple_layouts": False,
        }
        manifest_path = self.output / "manifest.json"
        if manifest_path.exists():
            existing = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            started = (
                self.output / "search_replay" / "iteration_000051"
            ).exists()
            if (
                existing.get("frozen_champion_50_sha256")
                != manifest["frozen_champion_50_sha256"]
                or (
                    existing.get("config_hash")
                    != manifest["config_hash"]
                    and started
                )
            ):
                raise RuntimeError(
                    "strength-continuation manifest changed"
                )
        atomic_json(manifest_path, manifest)
        championship_path = self.output / "championship.json"
        if not championship_path.exists():
            atomic_json(
                championship_path,
                {
                    "schema": "search-v3-championship-v2",
                    "current": {
                        "iteration": 50,
                        "checkpoint": str(frozen50),
                        "checkpoint_sha256": _sha256(frozen50),
                        "source": (
                            "user-designated formal frozen champion"
                        ),
                    },
                    "history": [
                        {
                            "iteration": 10,
                            "checkpoint": str(frozen10),
                            "checkpoint_sha256": _sha256(frozen10),
                            "source": "historical frozen champion",
                        }
                    ],
                    "promotion_policy": (
                        "fresh sealed continuation family CI gate"
                    ),
                },
            )
        state_path = self.output / "championship_state.json"
        if not state_path.exists():
            scheduler = source["scheduler"]
            atomic_json(
                state_path,
                {
                    "schema": "search-v3-championship-trigger-state-v1",
                    "current_champion_iteration": 50,
                    "last_evaluation_iteration": 50,
                    "accepted_updates_at_last_evaluation": int(
                        scheduler.get(
                            "accepted_update_iterations",
                            scheduler.get("accepted_iterations", 0),
                        )
                    ),
                    "completed_rounds_at_last_evaluation": int(
                        scheduler.get("completed_iterations", 0)
                    ),
                    "trigger": {
                        "accepted_updates": 3,
                        "completed_rounds": 10,
                        "first_to_reach": True,
                    },
                    "history": [],
                },
            )
        competitive_path = self.output / "competitive.json"
        if not competitive_path.exists():
            atomic_json(
                competitive_path,
                {
                    "schema": (
                        "search-v3-competitive-checkpoints-v1"
                    ),
                    "iterations": {
                        "50": {
                            "reason": (
                                "starting frozen championship incumbent"
                            ),
                            "search64_oracle_regret": report[
                                "post_probe"
                            ]["search64_oracle_regret"],
                            "permanent": True,
                        }
                    },
                },
            )

    def _initialize_late_stage_logic(self) -> None:
        """Version the post-107 logic without mutating iteration 107."""
        source_checkpoint = self._checkpoint(107)
        source_report = (
            self.output / "reports" / "iteration_000107.json"
        )
        if not source_checkpoint.exists() or not source_report.exists():
            raise RuntimeError(
                "late-stage logic requires the complete iteration 107 "
                "checkpoint and report"
            )
        payload = torch.load(
            source_checkpoint, map_location="cpu", weights_only=False
        )
        report = json.loads(
            source_report.read_text(encoding="utf-8")
        )
        required = {
            "model",
            "optimizer",
            "scheduler",
            "global_step",
            "rng",
            "replay",
            "leaf_value_calibration",
            "leaf_value_scale",
        }
        if (
            payload.get("completed") is not True
            or int(payload.get("iteration", -1)) != 107
            or report.get("passed") is not True
            or not required.issubset(payload)
        ):
            raise RuntimeError(
                "iteration 107 is not a complete atomic source"
            )
        if payload.get("config_hash") == self.config.hash():
            raise RuntimeError(
                "late-stage config must have a distinct semantic hash"
            )
        logic = self.training_logic
        assert logic is not None
        manifest = {
            "schema": logic.schema,
            "source_iteration": 107,
            "source_checkpoint": str(source_checkpoint.resolve()),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "source_report": str(source_report.resolve()),
            "source_report_sha256": _sha256(source_report),
            "source_config_hash": payload["config_hash"],
            "training_config": self.config.to_dict(),
            "training_config_hash": self.config.hash(),
            "policy_lr_multiplier": logic.policy_lr_multiplier,
            "policy_candidate_steps": logic.policy_candidate_steps,
            "selected_policy_step": logic.selected_policy_step,
            "value_candidate_steps": logic.value_candidate_steps,
            "selected_value_step": logic.selected_value_step,
            "adaptive_selection": logic.adaptive_selection,
            "iteration_107_immutable": True,
            "iteration_107_model_hash": _state_dict_hash(
                payload["model"]
            ),
            "bootstrap_expected_model_hash": (
                logic.bootstrap_expected_model_hash
            ),
            "bootstrap_checkpoint": str(
                (
                    self.output
                    / "checkpoints"
                    / "late_stage_bootstrap_pending.pt"
                ).resolve()
            ),
            "accepted_checkpoint": str(
                self._checkpoint(108).resolve()
            ),
            "next_selfplay_iteration": 109,
        }
        path = self.output / "late_stage_short_horizon_v1.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            immutable = (
                "source_checkpoint_sha256",
                "source_config_hash",
                "training_config_hash",
                "bootstrap_expected_model_hash",
            )
            if any(
                existing.get(key) != manifest.get(key)
                for key in immutable
            ):
                raise RuntimeError(
                    "late-stage training-logic manifest changed"
                )
        atomic_json(path, manifest)
        competitive_path = self.output / "competitive.json"
        competitive = json.loads(
            competitive_path.read_text(encoding="utf-8")
        )
        competitive.setdefault("iterations", {}).setdefault(
            "107",
            {
                "reason": (
                    "immutable safe no-op source for "
                    "late_stage_short_horizon_v1"
                ),
                "checkpoint": str(source_checkpoint.resolve()),
                "checkpoint_sha256": _sha256(source_checkpoint),
                "official_incumbent": True,
                "permanent": True,
                "retention_lock": True,
            },
        )
        atomic_json(competitive_path, competitive)
        accepted_bootstrap = self._checkpoint(108)
        if accepted_bootstrap.exists():
            accepted = torch.load(
                accepted_bootstrap,
                map_location="cpu",
                weights_only=False,
            )
            if (
                accepted.get("phase")
                == "late_stage_bootstrap_accepted"
            ):
                self.start_iteration = 109

    def _load_training_state(
        self, iteration: int
    ) -> tuple[Any, torch.optim.AdamW, dict[str, Any], int]:
        if (
            self.late_stage_short_horizon
            and iteration == 107
        ):
            seed = (
                self.output
                / "checkpoints"
                / "late_stage_seed_iteration_000107.pt"
            )
            if not seed.exists():
                raise RuntimeError(
                    "late-stage no-op migration seed is missing"
                )
            payload = torch.load(
                seed, map_location="cpu", weights_only=False
            )
            if (
                payload.get("completed") is not True
                or payload.get("phase")
                != "late_stage_schema_migration_noop"
                or payload.get("config_hash") != self.config.hash()
            ):
                raise RuntimeError(
                    "late-stage no-op migration seed is invalid"
                )
            from junqi_az.model import PolicyValueNet
            from .alternating_lineage import (
                _grouped_optimizer,
                _restore_rng,
            )

            model = PolicyValueNet(self.config.model).to(self.device)
            model.load_state_dict(payload["model"])
            model.search_v3_leaf_value_scale = float(
                payload["leaf_value_scale"]
            )
            optimizer = _grouped_optimizer(
                model, self.config, payload["optimizer"]
            )
            _restore_rng(
                payload["rng"], self.generator, self.runner
            )
            return (
                model,
                optimizer,
                copy.deepcopy(payload["scheduler"]),
                int(payload["global_step"]),
            )
        return super()._load_training_state(iteration)

    def _championship_payload(self) -> dict[str, Any]:
        return json.loads(
            (self.output / "championship.json").read_text(
                encoding="utf-8"
            )
        )

    def _champion_iterations(self) -> set[int]:
        payload = self._championship_payload()
        return {
            int(payload["current"]["iteration"]),
            *(
                int(row["iteration"])
                for row in payload.get("history", [])
            ),
        }

    def _freeze_promoted(
        self, checkpoint: Path, iteration: int
    ) -> Path:
        return self._freeze_copy(checkpoint, iteration)

    def _championship_due(
        self, checkpoint_payload: dict[str, Any]
    ) -> tuple[bool, dict[str, int]]:
        state = json.loads(
            (
                self.output / "championship_state.json"
            ).read_text(encoding="utf-8")
        )
        return championship_trigger(
            state, checkpoint_payload["scheduler"]
        )

    def _run_championship(
        self,
        iteration: int,
        checkpoint: Path,
        report: dict[str, Any],
        deltas: dict[str, int],
    ) -> dict[str, Any]:
        championship = self._championship_payload()
        pending_path = self.output / "championship_pending.json"
        existing_pending = (
            json.loads(pending_path.read_text(encoding="utf-8"))
            if pending_path.exists()
            else None
        )
        if existing_pending is not None:
            if int(
                existing_pending["challenger_iteration"]
            ) != iteration:
                raise RuntimeError(
                    "pending championship challenger changed"
                )
            champion_iteration = int(
                existing_pending["champion_iteration"]
            )
            champion_checkpoint = Path(
                existing_pending["champion_checkpoint"]
            )
        else:
            champion_iteration = int(
                championship["current"]["iteration"]
            )
            champion_checkpoint = Path(
                championship["current"]["checkpoint"]
            )
        challenger_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        champion_payload = torch.load(
            champion_checkpoint, map_location="cpu", weights_only=False
        )
        challenger_model_hash = _state_dict_hash(
            challenger_payload["model"]
        )
        champion_model_hash = _state_dict_hash(
            champion_payload["model"]
        )
        identical_models = (
            challenger_model_hash == champion_model_hash
        )
        pending = {
            "schema": "search-v3-championship-pending-v1",
            "challenger_iteration": iteration,
            "challenger_checkpoint": str(checkpoint),
            "challenger_checkpoint_sha256": _sha256(checkpoint),
            "champion_iteration": champion_iteration,
            "champion_checkpoint": str(champion_checkpoint),
            "champion_checkpoint_sha256": _sha256(
                champion_checkpoint
            ),
            "trigger_deltas": deltas,
            "phase": "arena",
            "challenger_model_hash": challenger_model_hash,
            "champion_model_hash": champion_model_hash,
        }
        if pending_path.exists():
            existing = json.loads(
                pending_path.read_text(encoding="utf-8")
            )
            keys = (
                "challenger_iteration",
                "challenger_checkpoint_sha256",
                "champion_iteration",
                "champion_checkpoint_sha256",
            )
            if any(
                existing.get(key) != pending.get(key)
                for key in keys
            ):
                raise RuntimeError(
                    "a different championship is already pending"
                )
        if not identical_models:
            atomic_json(pending_path, pending)
            print(
                f"[championship] FORMAL trigger challenger={iteration} "
                f"champion={champion_iteration} "
                f"accepted-since={deltas['accepted_updates']} "
                f"rounds-since={deltas['completed_rounds']} "
                f"suite=1024 states matchups=7 games=14336",
                flush=True,
            )
        else:
            print(
                f"[championship] SKIPPED_IDENTICAL_MODEL "
                f"challenger={iteration} champion={champion_iteration} "
                f"model={challenger_model_hash[:12]}",
                flush=True,
            )
        decision = self.championship.run(
            challenger_iteration=iteration,
            challenger_checkpoint=checkpoint,
            champion_iteration=champion_iteration,
            champion_checkpoint=champion_checkpoint,
            training_report=report,
        )
        if decision["promoted"]:
            if int(
                championship["current"]["iteration"]
            ) != iteration:
                frozen = self._freeze_promoted(
                    checkpoint, iteration
                )
                previous = dict(championship["current"])
                previous["replaced_by"] = iteration
                previous["championship_decision"] = decision[
                    "arena_report"
                ]
                championship.setdefault("history", []).append(
                    previous
                )
                championship["current"] = {
                    "iteration": iteration,
                    "checkpoint": str(frozen),
                    "checkpoint_sha256": _sha256(frozen),
                    "source": (
                        "triggered sealed continuation promotion"
                    ),
                    "championship_decision": decision[
                        "arena_report"
                    ],
                }
                atomic_json(
                    self.output / "championship.json",
                    championship,
                )
        checkpoint_payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        scheduler = checkpoint_payload["scheduler"]
        state_path = self.output / "championship_state.json"
        state = json.loads(state_path.read_text(encoding="utf-8"))
        state["current_champion_iteration"] = int(
            self._championship_payload()["current"]["iteration"]
        )
        state["last_evaluation_iteration"] = iteration
        state["accepted_updates_at_last_evaluation"] = int(
            scheduler.get(
                "accepted_update_iterations",
                scheduler.get("accepted_iterations", 0),
            )
        )
        state["completed_rounds_at_last_evaluation"] = int(
            scheduler.get("completed_iterations", 0)
        )
        state.setdefault("history", []).append(decision)
        atomic_json(state_path, state)
        pending["phase"] = "completed"
        pending["decision"] = decision
        challenge_record = (
            self.output
            / "championships"
            / (
                f"challenger_{iteration:06d}_vs_"
                f"champion_{champion_iteration:06d}"
            )
            / "trigger.json"
        )
        atomic_json(challenge_record, pending)
        pending_path.unlink(missing_ok=True)
        print(
            f"[championship] challenger={iteration} "
            f"status={decision.get('status', 'COMPLETED')} "
            f"promoted={decision['promoted']} "
            f"bare={decision['bare_vs_champion']} "
            f"search64={decision['search64_vs_champion']} "
            f"search-gain={decision['search64_vs_bare']}",
            flush=True,
        )
        return decision

    def _round_summary(
        self,
        iteration: int,
        report: dict[str, Any],
        checkpoint: Path,
    ) -> dict[str, Any]:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        chosen = report["chosen_candidate"]
        changed = bool(
            int(chosen["policy_steps"])
            or int(chosen["value_steps"])
        )
        terminal = report["data"]["terminal_reasons"]
        games = max(1, int(report["data"]["games"]))
        search_gain = (
            1.0
            - float(
                report["post_probe"]["search64_oracle_regret"]
            )
            / max(
                1e-12,
                float(
                    report["post_probe"]["bare_oracle_regret"]
                ),
            )
        )
        return {
            "completed_iteration": iteration,
            "selected_fractional_epoch": chosen[
                "fractional_epoch"
            ],
            "update": "accepted_update" if changed else "safe_no_op",
            "policy_steps": int(chosen["policy_steps"]),
            "value_steps": int(chosen["value_steps"]),
            "train_search_excess": report["fresh_train"][
                "policy_excess"
            ],
            "held_out_search_excess": report["held_out"][
                "policy_excess"
            ],
            "policy_family": report["paired_policy"],
            "wdl_ce": report["held_out"]["wdl_ce"],
            "wdl_family": report["paired_wdl"],
            "bare_oracle_regret": report["post_probe"][
                "bare_oracle_regret"
            ],
            "search64_oracle_regret": report["post_probe"][
                "search64_oracle_regret"
            ],
            "search64_gain_fraction": search_gain,
            "value_phase_policy_kl": report[
                "value_phase_policy_kl"
            ],
            "unique_state_ratio": report["data"][
                "unique_state_ratio"
            ],
            "unique_trajectories": report["data"][
                "unique_trajectories"
            ],
            "repetition_fraction": report["data"][
                "repetition_fraction"
            ],
            "avoidable_repetition_exit_rate": report[
                "post_probe"
            ]["search64_avoidable_exit_fraction"],
            "average_selfplay_ply": report["data"].get(
                "average_ply"
            ),
            "terminal_reason_fractions": {
                key: value / games for key, value in terminal.items()
            },
            "selfplay_simulations_per_second": report["data"][
                "simulations_per_second"
            ],
            "model_sha256": _state_dict_hash(payload["model"]),
            "checkpoint_sha256": _sha256(checkpoint),
            "global_step": int(payload["global_step"]),
            "atomic_checkpoint_fields": [
                "model",
                "optimizer",
                "scheduler",
                "global_step",
                "rng",
                "replay",
                "leaf_value_calibration",
            ],
        }

    def _diversity_reasons(
        self,
        data: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> list[str]:
        reasons = super()._diversity_reasons(data, previous)
        repetition = float(data["repetition_fraction"])
        if repetition >= 0.50:
            reasons.append(
                "self-play repetition reached the pathology threshold"
            )
        elif (
            previous is not None
            and repetition >= 0.15
            and repetition
            > float(previous.get("repetition_fraction", 0.0)) + 0.10
        ):
            reasons.append(
                "self-play repetition systematically increased"
            )
        return reasons

    def _maintenance(
        self,
        iteration: int,
        report: dict[str, Any],
        checkpoint: Path,
    ) -> None:
        summary = self._round_summary(
            iteration, report, checkpoint
        )
        report["round_summary"] = summary
        atomic_json(
            self.output
            / "reports"
            / f"iteration_{iteration:06d}.json",
            report,
        )
        competitive = self.retention.register(iteration, report)
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        due, deltas = self._championship_due(payload)
        pending_path = self.output / "championship_pending.json"
        if pending_path.exists():
            pending = json.loads(
                pending_path.read_text(encoding="utf-8")
            )
            due = True
            deltas = {
                key: int(value)
                for key, value in pending.get(
                    "trigger_deltas", deltas
                ).items()
            }
        decision = None
        if due:
            decision = self._run_championship(
                iteration, checkpoint, report, deltas
            )
        champion_iterations = self._champion_iterations()
        registry_path = self.output / "competitive.json"
        registry = json.loads(
            registry_path.read_text(encoding="utf-8")
        )
        for champion in champion_iterations:
            registry.setdefault("iterations", {}).setdefault(
                str(champion),
                {
                    "reason": "frozen championship history",
                    "permanent": True,
                },
            )
        atomic_json(registry_path, registry)
        trim_competitive_registry(
            registry_path,
            champion_iterations=champion_iterations,
            maximum_non_champion=8,
        )
        retention = self.retention.apply(iteration)
        data_retention = self.data_retention.apply(
            iteration,
            champion_iterations=champion_iterations,
        )
        championship_label = (
            decision.get("status", "COMPLETED")
            if decision
            else "NOT_DUE"
        )
        print(
            f"[round] iteration={iteration} "
            f"update={summary['update']} "
            f"epoch={summary['selected_fractional_epoch']:.2f} "
            f"policy-steps={summary['policy_steps']} "
            f"value-steps={summary['value_steps']} "
            f"heldout={summary['held_out_search_excess']:.5f} "
            f"wdl={summary['wdl_ce']:.5f} "
            f"search64-gain={summary['search64_gain_fraction']:.1%} "
            f"selfplay-sps="
            f"{summary['selfplay_simulations_per_second']:.0f} "
            f"global-step={summary['global_step']} "
            f"model={summary['model_sha256'][:12]}",
            flush=True,
        )
        atomic_json(
            self.output
            / "maintenance"
            / f"iteration_{iteration:06d}.json",
            {
                "schema": (
                    "search-v3-strength-continuation-maintenance-v1"
                ),
                "iteration": iteration,
                "checkpoint_sha256": _sha256(checkpoint),
                "round_summary_written": True,
                "championship_checked": True,
                "checkpoint_retention_completed": (
                    retention["phase"] == "completed"
                ),
                "data_retention_completed": (
                    data_retention["phase"] == "completed"
                ),
            },
        )
        print(
            f"[retention] checkpoint-deleted="
            f"{len(retention['deleted'])} "
            f"replay-deleted="
            f"{len(data_retention['deleted_replay'])} "
            f"reclaimed="
            f"{(retention['bytes_reclaimed'] + data_retention['bytes_reclaimed']) / (1024**2):.1f}MiB "
            f"competitive={'yes' if competitive else 'no'} "
            f"championship={championship_label}",
            flush=True,
        )

    def _after_successful_iteration(
        self,
        iteration: int,
        report: dict[str, Any],
        checkpoint: Path,
        candidate_checkpoint: Path,
    ) -> None:
        del candidate_checkpoint
        self._maintenance(iteration, report, checkpoint)

    def _resume_pending_or_missed(
        self, through_iteration: int
    ) -> None:
        completed = []
        for checkpoint in (
            self.output / "checkpoints"
        ).glob("loop_iteration_*.pt"):
            iteration = int(checkpoint.stem.rsplit("_", 1)[-1])
            if (
                self.start_iteration - 1
                < iteration
                <= through_iteration
            ):
                report_path = (
                    self.output
                    / "reports"
                    / f"iteration_{iteration:06d}.json"
                )
                if report_path.exists():
                    report = json.loads(
                        report_path.read_text(encoding="utf-8")
                    )
                    if report.get("passed") is True:
                        completed.append((iteration, checkpoint, report))
        if not completed:
            return
        iteration, checkpoint, report = max(
            completed, key=lambda item: item[0]
        )
        pending = self.output / "championship_pending.json"
        maintenance = (
            self.output
            / "maintenance"
            / f"iteration_{iteration:06d}.json"
        )
        due, _ = self._championship_due(
            torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
        )
        if (
            pending.exists()
            or due
            or "round_summary" not in report
            or not maintenance.exists()
        ):
            self._maintenance(iteration, report, checkpoint)

    def run(self, through_iteration: int = 100) -> dict[str, Any]:
        raise RuntimeError(
            "Historical v3 training is frozen after the repair: its data/search/training "
            "contract has changed. Use run_open_v4.ps1 with a new run directory. "
            "Frozen v3 checkpoints remain available for evaluation and initialization."
        )
        if self.late_stage_short_horizon:
            bootstrap = self._checkpoint(108)
            if bootstrap.exists() and self.start_iteration == 109:
                payload = torch.load(
                    bootstrap, map_location="cpu", weights_only=False
                )
                if (
                    payload.get("completed") is not True
                    or payload.get("phase")
                    != "late_stage_bootstrap_accepted"
                    or payload.get("config_hash") != self.config.hash()
                ):
                    raise RuntimeError(
                        "late-stage bootstrap checkpoint is not resumable"
                    )
            elif not (
                self.output
                / "checkpoints"
                / "late_stage_seed_iteration_000107.pt"
            ).exists():
                raise RuntimeError(
                    "late-stage bootstrap has not completed; run "
                    "tools/run_search_v3_late_stage_bootstrap.py first"
                )
        self._resume_pending_or_missed(through_iteration)
        return super().run(through_iteration)


def run_strength_continuation(
    config_path: str | Path,
    through_iteration: int,
    *,
    training_workers: int = 16,
    arena_workers: int = 16,
    arena_block_states: int = 128,
) -> dict[str, Any]:
    return StrengthContinuationLineage(
        config_path,
        training_workers=training_workers,
        arena_workers=arena_workers,
        arena_block_states=arena_block_states,
    ).run(through_iteration)
