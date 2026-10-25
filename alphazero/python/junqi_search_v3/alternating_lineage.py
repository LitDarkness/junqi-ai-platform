from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .closed_loop import ClosedLoopRunner
from .dataset import atomic_json
from .evaluation import EvaluationGame, SearchEvaluator
from .oracle import OracleHistory
from .recovery import (
    RECOVERY_CONFIG,
    Iteration3AlternatingRecovery,
    _family_metrics,
    _gradient_cosine,
    _select,
)
from .stages import SearchV3Stages


CANDIDATE_FINAL_KL_LIMIT = 1e-3
VALUE_STEP_PEAK_KL_HARD_LIMIT = 1.5e-3


def candidate_kl_eligible(final_kl: float, peak_kl: float) -> bool:
    return (
        float(final_kl) <= CANDIDATE_FINAL_KL_LIMIT
        and float(peak_kl) <= VALUE_STEP_PEAK_KL_HARD_LIMIT
    )


def candidate_rejection_reasons(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> list[str]:
    reasons = []
    if candidate["paired_policy"]["classification"] not in {
        "improved",
        "neutral",
    }:
        reasons.append("policy_family_not_improved_or_neutral")
    if candidate["paired_wdl"]["classification"] == "clearly_degraded":
        reasons.append("wdl_family_clearly_degraded")
    if candidate["candidate_final_kl"] > CANDIDATE_FINAL_KL_LIMIT:
        reasons.append("candidate_final_kl_exceeded")
    if (
        candidate["value_step_peak_kl"]
        > VALUE_STEP_PEAK_KL_HARD_LIMIT
    ):
        reasons.append("value_step_peak_kl_hard_limit_exceeded")
    probe = candidate["probe"]
    if (
        float(probe["search64_oracle_regret"])
        >= float(probe["bare_oracle_regret"])
    ):
        reasons.append("search64_not_better_than_bare")
    baseline_regret = float(
        baseline["probe"]["search64_oracle_regret"]
    )
    large_regression_limit = baseline_regret + max(
        0.005, baseline_regret * 0.50
    )
    if (
        float(probe["search64_oracle_regret"])
        > large_regression_limit
    ):
        reasons.append("static_search64_regret_large_regression")
    return reasons


def candidate_selection_rank(
    candidate: dict[str, Any],
) -> list[float]:
    if not candidate["eligible"]:
        return [-1.0, -1.0, -1e30, -1e30]
    policy_class = candidate["paired_policy"]["classification"]
    wdl_class = candidate["paired_wdl"]["classification"]
    evidence = float(policy_class == "improved") + float(
        wdl_class == "improved"
    )
    family_gain = -(
        float(candidate["paired_policy"]["relative_delta"])
        + float(candidate["paired_wdl"]["relative_delta"])
    )
    changed = float(
        int(candidate["policy_steps"]) + int(candidate["value_steps"]) > 0
    )
    return [
        changed,
        evidence,
        family_gain,
        -float(candidate["probe"]["search64_oracle_regret"]),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _rng_snapshot(
    generator: np.random.Generator,
    runner: ClosedLoopRunner,
) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
        "closed_loop_python": runner.rng.getstate(),
        "alternating_batch_generator": copy.deepcopy(
            generator.bit_generator.state
        ),
    }


def _restore_rng(
    payload: dict[str, Any],
    generator: np.random.Generator,
    runner: ClosedLoopRunner,
) -> None:
    if payload.get("python") is not None:
        random.setstate(payload["python"])
    if payload.get("numpy") is not None:
        np.random.set_state(payload["numpy"])
    if payload.get("torch") is not None:
        torch.set_rng_state(payload["torch"])
    if torch.cuda.is_available() and payload.get("cuda") is not None:
        torch.cuda.set_rng_state_all(payload["cuda"])
    if payload.get("closed_loop_python") is not None:
        runner.rng.setstate(payload["closed_loop_python"])
    if payload.get("alternating_batch_generator") is not None:
        generator.bit_generator.state = payload[
            "alternating_batch_generator"
        ]


def _grouped_optimizer(
    model: PolicyValueNet,
    config: Any,
    state: dict[str, Any],
) -> torch.optim.AdamW:
    optimizer = torch.optim.AdamW(
        [
            {
                "name": "trunk",
                "params": list(model.encoder_parameters()),
                "lr": config.train.learning_rate * 0.10,
                "weight_decay": config.train.weight_decay,
            },
            {
                "name": "policy_head",
                "params": list(model.policy_head_parameters()),
                "lr": 0.0,
                "weight_decay": config.train.weight_decay,
            },
            {
                "name": "wdl_head",
                "params": list(model.value_parameters()),
                "lr": config.train.learning_rate,
                "weight_decay": config.train.weight_decay,
            },
        ]
    )
    optimizer.load_state_dict(state)
    return optimizer


class FormalAlternatingLineage(Iteration3AlternatingRecovery):
    """Official iteration 4+ lineage using the accepted alternating schedule."""

    def __init__(self, config_path: str | Path) -> None:
        # Deliberately do not call the iteration-3 recovery constructor.
        self.stages = SearchV3Stages(config_path)
        self.config = self.stages.config
        self.root = self.stages.root
        self.output = self.stages.run
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.source_run = (
            self.root / "runs" / "open_junqi_search_v3"
        ).resolve()
        self.recovery_run = (
            self.root
            / "runs"
            / "open_junqi_search_v3_iteration3_recovery"
        ).resolve()
        self.frozen_path = self.output / "FROZEN.json"
        self.start_iteration = 4
        self.previous_data_iteration = 3
        self.required_confirmation_iterations = {4, 5}
        self.status_filename = "RECOVERY_STATUS.json"
        if self.frozen_path.exists():
            self.frozen = json.loads(
                self.frozen_path.read_text(encoding="utf-8")
            )
            # A frozen validation lineage is read-only.  In particular, do
            # not refresh its manifest or scheduler metadata merely because
            # an old command is invoked again.
            self.runner = None
            self.score_scale = 0.0
            self.seed = self.config.seed ^ 0x414C5445
            self.generator = np.random.Generator(
                np.random.PCG64(self.seed)
            )
            return
        self.frozen = None
        self._prepare_assets()
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
        self.seed = self.config.seed ^ 0x414C5445
        self.generator = np.random.Generator(np.random.PCG64(self.seed))
        self._initialize_lineage()
        self._migrate_scheduler_metadata()

    def _copy_asset(self, relative: str) -> None:
        source = self.source_run / relative
        target = self.output / relative
        if source.is_dir():
            if not target.exists():
                shutil.copytree(source, target)
            if _tree_hash(source) != _tree_hash(target):
                raise RuntimeError(f"alternating asset mismatch: {relative}")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                shutil.copy2(source, target)
            if _sha256(source) != _sha256(target):
                raise RuntimeError(f"alternating asset mismatch: {relative}")

    def _prepare_assets(self) -> None:
        for relative in (
            "probe_manifest.json",
            "oracle_scale.json",
            "oracle_dataset",
            "search_replay/iteration_000002",
            "search_replay/iteration_000003",
        ):
            self._copy_asset(relative)

    def _checkpoint(self, iteration: int) -> Path:
        return (
            self.output
            / "checkpoints"
            / f"loop_iteration_{iteration:06d}.pt"
        )

    def _initialize_lineage(self) -> None:
        checkpoint = self._checkpoint(3)
        source = (
            self.recovery_run
            / "checkpoints"
            / "iteration_000003_alternating.pt"
        )
        selection_path = (
            self.recovery_run / "reports" / "selection.json"
        )
        selection = json.loads(
            selection_path.read_text(encoding="utf-8")
        )
        failure_manifest = (
            self.source_run
            / "diagnostics"
            / "iteration_000003_joint_failure_original"
            / "manifest.json"
        )
        manifest = {
            "schema": "open-junqi-search-v3-formal-alternating-lineage",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "source_checkpoint": str(source),
            "source_checkpoint_sha256": _sha256(source),
            "source_selection": str(selection_path),
            "source_selection_sha256": _sha256(selection_path),
            "immutable_joint_failure_manifest": str(failure_manifest),
            "immutable_joint_failure_manifest_sha256": _sha256(
                failure_manifest
            ),
            "source_lineage_modified": False,
            "start_iteration": 3,
            "next_iterations": [4, 5],
            "training_contract": RECOVERY_CONFIG,
            "policy_replay": (
                "70% current fresh, 15% immediately previous 1-2 "
                "iterations, 10% Oracle, 5% cycle anchor"
            ),
            "external_arena_minimum_independent_trajectories": 32,
        }
        atomic_json(self.output / "manifest.json", manifest)
        if checkpoint.exists():
            existing = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                existing.get("config_hash") != self.config.hash()
                or existing.get("source_checkpoint_sha256")
                != manifest["source_checkpoint_sha256"]
            ):
                raise RuntimeError("stale formal alternating checkpoint")
            return
        payload = torch.load(source, map_location="cpu", weights_only=False)
        source_iteration2 = torch.load(
            self.source_run
            / "checkpoints"
            / "loop_iteration_000002.pt",
            map_location="cpu",
            weights_only=False,
        )
        # The recovery checkpoint already binds its selected model and grouped
        # Adam state.  Global RNG is copied byte-for-byte.  The original
        # one-off recovery had no dedicated scheduler or batch RNG, so the
        # formal lineage establishes and records both at this boundary.
        rng = copy.deepcopy(payload["rng"])
        rng["closed_loop_python"] = self.runner.rng.getstate()
        rng["alternating_batch_generator"] = copy.deepcopy(
            self.generator.bit_generator.state
        )
        chosen = selection["chosen_candidate"]
        accepted_updates = int(chosen["policy_steps"]) + 2 * int(
            chosen["value_steps"]
        )
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "open-junqi-search-v3-formal-alternating-checkpoint",
                "completed": True,
                "phase": "alternating",
                "iteration": 3,
                "global_step": int(
                    source_iteration2.get("global_step", 0)
                )
                + accepted_updates,
                "config": self.config.to_dict(),
                "config_hash": self.config.hash(),
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": _sha256(source),
                "model": payload["model"],
                "optimizer": payload["optimizer"],
                "scheduler": {
                    "kind": "constant",
                    "iteration": 3,
                    "accepted_iterations": 1,
                    "accepted_update_iterations": 1,
                    "completed_iterations": 1,
                    "learning_rate": self.config.train.learning_rate,
                },
                "leaf_value_scale": payload["leaf_value_scale"],
                "leaf_value_calibration": payload[
                    "leaf_value_calibration"
                ],
                "replay": {
                    "policy_iterations": [3],
                    "value_iterations": [2, 3],
                },
                "rng": rng,
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        atomic_json(
            self.output / "reports" / "iteration_000003_import.json",
            {
                "schema": "formal-alternating-import",
                "iteration": 3,
                "source_checkpoint": str(source),
                "source_checkpoint_sha256": _sha256(source),
                "selected_candidate": chosen,
                "scheduler_migration": (
                    "new constant scheduler; one-off recovery had no "
                    "scheduler object"
                ),
                "batch_rng_migration": (
                    "new dedicated PCG64 stream; source global RNG copied"
                ),
                "passed": True,
            },
        )

    def _migrate_scheduler_metadata(self) -> None:
        """Keep completed rounds distinct from rounds that changed weights."""
        changes = []
        accepted_update_iterations = 0
        completed_iterations = 0
        for iteration in range(3, 1_000_000):
            checkpoint = self._checkpoint(iteration)
            if not checkpoint.exists():
                break
            completed_iterations += 1
            if iteration == 3:
                changed_weights = True
            else:
                report_path = (
                    self.output
                    / "reports"
                    / f"iteration_{iteration:06d}.json"
                )
                if not report_path.exists():
                    raise RuntimeError(
                        "formal checkpoint has no matching report: "
                        f"{checkpoint}"
                    )
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                chosen = report["chosen_candidate"]
                changed_weights = bool(
                    int(chosen["policy_steps"])
                    or int(chosen["value_steps"])
                )
            accepted_update_iterations += int(changed_weights)
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            scheduler = dict(payload["scheduler"])
            expected = {
                "accepted_iterations": accepted_update_iterations,
                "accepted_update_iterations": accepted_update_iterations,
                "completed_iterations": completed_iterations,
            }
            if any(scheduler.get(key) != value for key, value in expected.items()):
                before = {
                    key: scheduler.get(key) for key in expected
                }
                scheduler.update(expected)
                payload["scheduler"] = scheduler
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save(payload, temporary)
                os.replace(temporary, checkpoint)
                changes.append(
                    {
                        "iteration": iteration,
                        "checkpoint": str(checkpoint),
                        "before": before,
                        "after": expected,
                        "model_optimizer_rng_unchanged": True,
                    }
                )
        atomic_json(
            self.output / "reports" / "scheduler_metadata.json",
            {
                "schema": "formal-alternating-scheduler-metadata",
                "changes": changes,
                "note": (
                    "An epoch-0 selection completes an iteration but is not "
                    "an accepted model update."
                ),
            },
        )

    def _load_training_state(
        self, iteration: int
    ) -> tuple[PolicyValueNet, torch.optim.AdamW, dict[str, Any], int]:
        payload = torch.load(
            self._checkpoint(iteration),
            map_location="cpu",
            weights_only=False,
        )
        if (
            payload.get("completed") is not True
            or payload.get("config_hash") != self.config.hash()
        ):
            raise RuntimeError(f"invalid alternating checkpoint {iteration}")
        model = PolicyValueNet(self.config.model).to(self.device)
        model.load_state_dict(payload["model"])
        model.search_v3_leaf_value_scale = float(
            payload["leaf_value_scale"]
        )
        optimizer = _grouped_optimizer(
            model, self.config, payload["optimizer"]
        )
        _restore_rng(payload["rng"], self.generator, self.runner)
        return (
            model,
            optimizer,
            copy.deepcopy(payload["scheduler"]),
            int(payload["global_step"]),
        )

    def _recent_samples(self, iteration: int) -> list[Any]:
        result = []
        for source_iteration in range(max(2, iteration - 2), iteration):
            source_directory = (
                self.output
                / "search_replay"
                / f"iteration_{source_iteration:06d}"
            )
            if not source_directory.exists():
                continue
            result.extend(
                sample
                for sample in self.runner._iteration_samples(
                    source_iteration
                )
                if (
                    sample.game_id % 10 != 0
                    and bool(getattr(sample, "cycle_cleaned", False))
                )
            )
        return result

    def _trace(
        self,
        evaluator: SearchEvaluator,
        opponent: str,
        simulations: int,
        candidate_side: int,
        seed: int,
    ) -> list[int]:
        state = self.stages.backend.new_state(seed)
        history = OracleHistory()
        actions = []
        while not state.is_terminal():
            history.observe(state)
            if state.current_player() == candidate_side:
                if simulations == 0:
                    action = evaluator.search.bare([state])[0].action
                else:
                    action = evaluator.search.search(
                        [state], simulations
                    )[0][0].action
            else:
                game = EvaluationGame(
                    state=state,
                    candidate_side=candidate_side,
                    red_agent="",
                    black_agent="",
                    history=history,
                )
                action = evaluator._external_actions(
                    [game], opponent
                )[0]
            actions.append(int(action))
            state.apply(int(action))
        return actions

    def _divergence_diagnostic(
        self,
        before: PolicyValueNet,
        after: PolicyValueNet,
        iteration: int,
    ) -> dict[str, Any]:
        first = SearchEvaluator(
            root=self.root,
            backend=self.stages.backend,
            config=self.config,
            model=before,
            score_scale=self.score_scale,
        )
        second = SearchEvaluator(
            root=self.root,
            backend=self.stages.backend,
            config=self.config,
            model=after,
            score_scale=self.score_scale,
        )
        rows = []
        for opponent in ("supplement_wide_quiescence", "exact082"):
            for simulations in (0, 64):
                for candidate_side in (1, -1):
                    seed = (
                        self.config.seed
                        ^ (iteration * 0x9E3779B1)
                        ^ (simulations * 0x85EBCA6B)
                        ^ (1 if candidate_side == 1 else 0xC2B2AE35)
                        ^ sum(ord(char) for char in opponent)
                    )
                    old = self._trace(
                        first,
                        opponent,
                        simulations,
                        candidate_side,
                        seed,
                    )
                    new = self._trace(
                        second,
                        opponent,
                        simulations,
                        candidate_side,
                        seed,
                    )
                    common = min(len(old), len(new))
                    divergence = next(
                        (
                            index
                            for index in range(common)
                            if old[index] != new[index]
                        ),
                        None,
                    )
                    row: dict[str, Any] = {
                        "opponent": opponent,
                        "candidate_simulations": simulations,
                        "candidate_side": candidate_side,
                        "seed": seed,
                        "before_plies": len(old),
                        "after_plies": len(new),
                        "first_divergence_ply": divergence,
                    }
                    if divergence is not None:
                        state = self.stages.backend.new_state(seed)
                        history = OracleHistory()
                        for action in old[:divergence]:
                            history.observe(state)
                            state.apply(action)
                        history.observe(state)
                        oracle = first.oracle.evaluate(
                            state, history, budget=32000
                        )
                        q = {
                            int(item["action"]): float(item["q"])
                            for item in oracle["actions"]
                        }
                        before_action = int(old[divergence])
                        after_action = int(new[divergence])
                        row.update(
                            {
                                "acting_side": int(
                                    state.current_player()
                                ),
                                "before_action": before_action,
                                "after_action": after_action,
                                "oracle32k_before_q": q[before_action],
                                "oracle32k_after_q": q[after_action],
                                "oracle32k_q_delta_after_minus_before": (
                                    q[after_action] - q[before_action]
                                ),
                                "oracle32k_best_action": int(
                                    oracle["root_best_action"]
                                ),
                                "oracle32k_nodes": int(oracle["nodes"]),
                            }
                        )
                    rows.append(row)
        return {
            "rows": rows,
            "divergent_trajectories": sum(
                row["first_divergence_ply"] is not None for row in rows
            ),
            "trajectory_count": len(rows),
        }

    def _train_iteration(
        self,
        iteration: int,
        model: PolicyValueNet,
        optimizer: torch.optim.AdamW,
        fresh: list[Any],
        candidate_evaluator: (
            Callable[
                [
                    int,
                    PolicyValueNet,
                    torch.optim.AdamW,
                    dict[str, Any],
                ],
                dict[str, Any],
            ]
            | None
        ) = None,
    ) -> tuple[
        PolicyValueNet,
        torch.optim.AdamW,
        dict[str, Any],
        dict[str, Any],
    ]:
        baseline = PolicyValueNet(self.config.model).to(self.device)
        baseline.load_state_dict(model.state_dict())
        baseline.search_v3_leaf_value_scale = getattr(
            model, "search_v3_leaf_value_scale", 1.0
        )
        fresh_train = [
            sample for sample in fresh if sample.game_id % 10 != 0
        ]
        recent_train = self._recent_samples(iteration)
        recent_fallback_to_fresh = not recent_train
        if recent_fallback_to_fresh:
            recent_train = fresh_train
        pools = {
            "fresh_train": fresh_train,
            "fresh_validation": [
                sample for sample in fresh if sample.game_id % 10 == 0
            ],
            "recent_train": recent_train,
            "oracle_train": self.runner._oracle_dataset().by_split["train"],
        }
        pools["cycle_train"] = [
            sample
            for sample in pools["oracle_train"]
            if np.any(sample.repetition_contexts > 0)
        ]
        validation = pools["fresh_validation"]
        baseline_family = _family_metrics(
            baseline, validation, self.device
        )
        epoch_steps = int(
            np.ceil(len(pools["fresh_train"]) / self.config.train.batch_size)
        )
        maximum_policy_steps = 2 * epoch_steps
        interval = max(1, int(round(epoch_steps * 0.25)))
        self.seed = self.config.seed ^ 0x414C5445 ^ iteration
        candidates = [
            self._candidate(
                model, baseline_family, validation, 0, 0
            )
        ]
        candidates[0]["fractional_epoch"] = 0.0
        candidates[0]["fresh_train"] = self.runner._evaluate_distill(
            model, pools["fresh_train"]
        )
        candidates[0]["value_step_peak_kl"] = 0.0
        candidates[0]["candidate_final_kl"] = 0.0
        candidates[0]["candidate_final_family_kl"] = 0.0
        candidates[0]["candidate_final_family_kl_max"] = 0.0
        candidates[0]["max_value_policy_kl"] = 0.0
        candidates[0]["rejection_reasons"] = candidate_rejection_reasons(
            candidates[0], candidates[0]
        )
        candidates[0]["eligible"] = not candidates[0][
            "rejection_reasons"
        ]
        candidates[0]["selection_rank"] = candidate_selection_rank(
            candidates[0]
        )
        candidates[0]["rejection_reason"] = "none"
        if candidate_evaluator is not None:
            candidates[0]["candidate_diagnostics"] = candidate_evaluator(
                0, model, optimizer, candidates[0]
            )
        best_index = 0
        best_rank = tuple(candidates[0]["selection_rank"])
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        best_rng = _rng_snapshot(self.generator, self.runner)
        policy_steps = value_steps = 0
        next_validation_step = interval
        kl_weight = float(
            RECOVERY_CONFIG["value_policy_kl_initial_weight"]
        )
        log = []
        cosines = []
        while policy_steps < maximum_policy_steps:
            for _ in range(2):
                if policy_steps >= maximum_policy_steps:
                    break
                metric = self._policy_step(model, optimizer, pools)
                policy_steps += 1
                log.append(
                    {
                        "phase": "policy",
                        "policy_step": policy_steps,
                        "value_step": value_steps,
                        **metric,
                    }
                )
            metric, kl_weight = self._value_phase(
                model, optimizer, pools, kl_weight
            )
            value_steps += 1
            log.append(
                {
                    "phase": "value",
                    "policy_step": policy_steps,
                    "value_step": value_steps,
                    **metric,
                }
            )
            if (
                policy_steps >= next_validation_step
                or policy_steps >= maximum_policy_steps
            ):
                while next_validation_step <= policy_steps:
                    next_validation_step += interval
                cosine = _gradient_cosine(
                    model,
                    _select(pools["fresh_train"], self._keys(64)),
                    self.device,
                )
                cosine["policy_step"] = policy_steps
                cosine["value_step"] = value_steps
                cosines.append(cosine)
                candidate = self._candidate(
                    model,
                    baseline_family,
                    validation,
                    policy_steps,
                    value_steps,
                )
                candidate["fractional_epoch"] = (
                    policy_steps / epoch_steps
                )
                candidate["fresh_train"] = (
                    self.runner._evaluate_distill(
                        model, pools["fresh_train"]
                    )
                )
                value_rows = [
                    row for row in log if row["phase"] == "value"
                ]
                candidate["value_step_peak_kl"] = max(
                    row["policy_kl"] for row in value_rows
                )
                candidate["candidate_final_kl"] = value_rows[-1][
                    "policy_kl"
                ]
                candidate["candidate_final_family_kl"] = value_rows[-1][
                    "policy_family_kl"
                ]
                candidate["candidate_final_family_kl_max"] = value_rows[-1][
                    "policy_family_kl_max"
                ]
                # Compatibility field for historical report readers.  It is
                # diagnostic only; eligibility uses final and peak separately.
                candidate["max_value_policy_kl"] = candidate[
                    "value_step_peak_kl"
                ]
                candidate["rejection_reasons"] = (
                    candidate_rejection_reasons(
                        candidate, candidates[0]
                    )
                )
                candidate["eligible"] = not candidate[
                    "rejection_reasons"
                ]
                candidate["selection_rank"] = candidate_selection_rank(
                    candidate
                )
                candidate["rejection_reason"] = (
                    "none"
                    if candidate["eligible"]
                    else ";".join(candidate["rejection_reasons"])
                )
                if candidate_evaluator is not None:
                    candidate["candidate_diagnostics"] = (
                        candidate_evaluator(
                            len(candidates), model, optimizer, candidate
                        )
                    )
                candidates.append(candidate)
                index = len(candidates) - 1
                rank = tuple(candidate["selection_rank"])
                if candidate["eligible"] and rank > best_rank:
                    best_index = index
                    best_rank = rank
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    best_optimizer = copy.deepcopy(
                        optimizer.state_dict()
                    )
                    best_rng = _rng_snapshot(
                        self.generator, self.runner
                    )
                print(
                    f"[alternating] iteration={iteration} "
                    f"epoch={candidate['fractional_epoch']:.2f}/2 "
                    f"train={candidate['fresh_train']['policy_excess']:.5f} "
                    f"heldout={candidate['validation']['policy_excess']:.5f} "
                    f"wdl={candidate['validation']['wdl_ce']:.5f} "
                    f"policy-family="
                    f"{candidate['paired_policy']['classification']} "
                    f"delta={candidate['paired_policy']['mean_delta']:+.6f} "
                    f"ci=[{candidate['paired_policy']['ci95'][0]:+.6f},"
                    f"{candidate['paired_policy']['ci95'][1]:+.6f}] "
                    f"wdl-family="
                    f"{candidate['paired_wdl']['classification']} "
                    f"delta={candidate['paired_wdl']['mean_delta']:+.6f} "
                    f"ci=[{candidate['paired_wdl']['ci95'][0]:+.6f},"
                    f"{candidate['paired_wdl']['ci95'][1]:+.6f}] "
                    f"kl-final={candidate['candidate_final_kl']:.6f} "
                    f"kl-family={candidate['candidate_final_family_kl']:.6f} "
                    f"kl-peak={candidate['value_step_peak_kl']:.6f} "
                    f"bare-regret="
                    f"{candidate['probe']['bare_oracle_regret']:.6f} "
                    f"search64-regret="
                    f"{candidate['probe']['search64_oracle_regret']:.6f} "
                    f"rank={candidate['selection_rank']} "
                    f"eligible={candidate['eligible']} "
                    f"rejection={candidate['rejection_reason']}",
                    flush=True,
                )
        best_nonzero = next(
            (
                (index, candidate)
                for index, candidate in sorted(
                    enumerate(candidates[1:], start=1),
                    key=lambda item: tuple(
                        item[1]["selection_rank"]
                    ),
                    reverse=True,
                )
                if candidate["eligible"]
            ),
            None,
        )
        for index, candidate in enumerate(candidates):
            if index == best_index:
                candidate["rejection_reason"] = "none"
            elif candidate["eligible"]:
                candidate["rejection_reason"] = (
                    f"lower_selection_rank_than_candidate_{best_index}"
                )
            print(
                f"[candidate decision] iteration={iteration} "
                f"candidate={index} epoch="
                f"{candidate['fractional_epoch']:.4f} "
                f"rank={candidate['selection_rank']} "
                f"eligible={candidate['eligible']} "
                f"rejection={candidate['rejection_reason']}",
                flush=True,
            )
        why_epoch0_won = (
            "no_eligible_nonzero_candidate"
            if best_index == 0 and best_nonzero is None
            else (
                "epoch0_has_higher_selection_rank"
                if best_index == 0
                else "not_applicable_nonzero_selected"
            )
        )
        model.load_state_dict(best_state)
        optimizer.load_state_dict(best_optimizer)
        _restore_rng(best_rng, self.generator, self.runner)
        chosen = candidates[best_index]
        model.search_v3_leaf_value_scale = chosen["value_calibration"][
            "leaf_value_scale"
        ]
        print(
            f"[candidate selection] iteration={iteration} "
            f"best-nonzero="
            f"{best_nonzero[0] if best_nonzero is not None else 'none'} "
            f"selected={best_index} "
            f"why-epoch0-won={why_epoch0_won}",
            flush=True,
        )
        return model, optimizer, chosen, {
            "candidates": candidates,
            "chosen_candidate_index": best_index,
            "training_log": log,
            "gradient_cosine": cosines,
            "accepted_rng": best_rng,
            "best_nonzero_candidate": (
                {
                    "index": best_nonzero[0],
                    "fractional_epoch": best_nonzero[1][
                        "fractional_epoch"
                    ],
                    "selection_rank": best_nonzero[1][
                        "selection_rank"
                    ],
                }
                if best_nonzero is not None
                else None
            ),
            "why_epoch0_won": why_epoch0_won,
            "recent_replay_cycle_cleaned_only": True,
            "recent_replay_fallback_to_current_fresh": (
                recent_fallback_to_fresh
            ),
        }

    def _diversity_reasons(
        self,
        data: dict[str, Any],
        previous: dict[str, Any] | None,
    ) -> list[str]:
        reasons = []
        if (
            data["unique_state_ratio"]
            < self.config.loop.minimum_unique_state_ratio
        ):
            reasons.append("unique state ratio collapsed")
        if data["maximum_trajectory_share"] > 0.02:
            reasons.append("one complete trajectory exceeds 2%")
        if any(
            value > self.config.loop.maximum_prefix_share
            for value in data["maximum_prefix_share"].values()
        ):
            reasons.append("one opening prefix exceeds 20%")
        if (
            previous is not None
            and data["unique_state_ratio"]
            < previous["unique_state_ratio"] - 0.20
        ):
            reasons.append("unique state ratio dropped by more than 20pp")
        return reasons

    def _after_successful_iteration(
        self,
        iteration: int,
        report: dict[str, Any],
        checkpoint: Path,
        candidate_checkpoint: Path,
    ) -> None:
        del iteration, report, checkpoint, candidate_checkpoint

    def run(self, through_iteration: int = 5) -> dict[str, Any]:
        raise RuntimeError(
            "Historical v3 training is frozen: use run_open_v4.ps1 in a new run "
            "directory. Its frozen models and diagnostic methods remain available."
        )
        if self.frozen is not None:
            frozen = self.frozen
            frozen_iteration = int(frozen["incumbent_iteration"])
            if through_iteration > frozen_iteration:
                raise RuntimeError(
                    "open_junqi_search_v3_alternating is frozen at "
                    f"iteration {frozen_iteration}; continue experiments "
                    "in the successor strength lineage"
                )
            reports = []
            for iteration in range(
                self.start_iteration, through_iteration + 1
            ):
                report_path = (
                    self.output
                    / "reports"
                    / f"iteration_{iteration:06d}.json"
                )
                checkpoint = self._checkpoint(iteration)
                if not report_path.exists():
                    raise RuntimeError(
                        "frozen alternating lineage is incomplete at "
                        f"iteration {iteration}"
                    )
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                if (
                    report.get("passed") is not True
                    or (
                        iteration == frozen_iteration
                        and not checkpoint.exists()
                    )
                ):
                    raise RuntimeError(
                        "frozen alternating lineage contains a failed "
                        f"iteration {iteration}"
                    )
                reports.append(report)
            return {
                "completed_iteration": through_iteration,
                "recovery_confirmed": through_iteration >= 5,
                "frozen": True,
                "reports": reports,
            }
        reports = []
        previous_data = json.loads(
            (
                self.output
                / "search_replay"
                / f"iteration_{self.previous_data_iteration:06d}"
                / "summary.json"
            ).read_text(encoding="utf-8")
        )
        for iteration in range(
            self.start_iteration, through_iteration + 1
        ):
            report_path = (
                self.output
                / "reports"
                / f"iteration_{iteration:06d}.json"
            )
            checkpoint = self._checkpoint(iteration)
            if report_path.exists():
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                reports.append(report)
                if report.get("passed") is not True:
                    return {
                        "completed_iteration": iteration - 1,
                        "paused_iteration": iteration,
                        "reports": reports,
                    }
                if checkpoint.exists():
                    previous_data = report["data"]
                    continue
                # Retention may prune an old official checkpoint after its
                # report and a newer resumable checkpoint are committed.
                if report.get("passed") is True:
                    previous_data = report["data"]
                    continue
                raise RuntimeError(f"failed report: {report_path}")
            actor, optimizer, scheduler, global_step = (
                self._load_training_state(iteration - 1)
            )
            print(
                f"[championship] incumbent=iteration_{iteration - 1:06d} "
                f"challenger=iteration_{iteration:06d} "
                f"accepted-updates="
                f"{scheduler.get('accepted_update_iterations', scheduler.get('accepted_iterations', 0))} "
                f"completed-rounds={scheduler.get('completed_iterations', iteration - 3)} "
                f"phase=selfplay",
                flush=True,
            )
            before = PolicyValueNet(self.config.model).to(self.device)
            before.load_state_dict(actor.state_dict())
            before.search_v3_leaf_value_scale = getattr(
                actor, "search_v3_leaf_value_scale", 1.0
            )
            pre_probe = self.runner._probe_metrics(before)
            data = self.runner._collect_iteration(iteration, actor)
            print(
                f"[throughput] iteration={iteration} "
                f"games={data['games']} samples={data['policy_samples']} "
                f"selfplay-sps={data['simulations_per_second']:.0f} "
                f"unique-state={data['unique_state_ratio']:.1%} "
                f"repetition={data['repetition_fraction']:.1%} "
                f"cycle-trim={data.get('replay_cycle_cleanup', {}).get('cycle_suffix_samples_removed', 0)} "
                f"cycle-teacher-adjusted={data.get('cycle_teacher', {}).get('adjusted', 0)}/"
                f"{data.get('cycle_teacher', {}).get('eligible', 0)}",
                flush=True,
            )
            diversity_reasons = self._diversity_reasons(
                data, previous_data
            )
            fresh = self.runner._iteration_samples(iteration)
            trained, optimizer, chosen, details = self._train_iteration(
                iteration, actor, optimizer, fresh
            )
            policy_rows = [
                row
                for row in details["training_log"]
                if row["phase"] == "policy"
            ]
            cycle_sampling = {
                source: {
                    "mean_fraction": float(
                        np.mean(
                            [
                                row.get(
                                    f"{source}_cycle_endgame_fraction",
                                    0.0,
                                )
                                for row in policy_rows
                            ]
                        )
                    )
                    if policy_rows
                    else 0.0,
                    "maximum_family_share": max(
                        (
                            float(
                                row.get(
                                    f"{source}_cycle_family_max_share",
                                    0.0,
                                )
                            )
                            for row in policy_rows
                        ),
                        default=0.0,
                    ),
                }
                for source in ("fresh_search", "recent_search")
            }
            print(
                f"[cycle sampling] iteration={iteration} "
                f"fresh={cycle_sampling['fresh_search']['mean_fraction']:.1%} "
                f"recent={cycle_sampling['recent_search']['mean_fraction']:.1%} "
                f"max-family={max(row['maximum_family_share'] for row in cycle_sampling.values()):.1%}",
                flush=True,
            )
            post_probe = chosen["probe"]
            reasons = list(diversity_reasons)
            if (
                chosen["paired_policy"]["classification"]
                == "clearly_degraded"
            ):
                reasons.append(
                    "held-out Search excess was not improved or neutral"
                )
            if (
                chosen["paired_wdl"]["classification"]
                == "clearly_degraded"
            ):
                reasons.append("held-out WDL CE was not improved or neutral")
            if (
                post_probe["search64_oracle_regret"]
                >= post_probe["bare_oracle_regret"]
            ):
                reasons.append("Search64 is not better than bare")
            if chosen["candidate_final_kl"] > CANDIDATE_FINAL_KL_LIMIT:
                reasons.append(
                    "candidate final Value-phase Policy KL exceeded 1e-3"
                )
            if (
                chosen["value_step_peak_kl"]
                > VALUE_STEP_PEAK_KL_HARD_LIMIT
            ):
                reasons.append(
                    "Value-phase Policy KL peak exceeded 1.5e-3"
                )
            if (
                post_probe["search64_avoidable_exit_fraction"] + 0.02
                < pre_probe["search64_avoidable_exit_fraction"]
            ):
                reasons.append(
                    "avoidable repetition exit rate systematically regressed"
                )
            divergence = self._divergence_diagnostic(
                before, trained, iteration
            )
            external = self.runner._arena_metrics(
                trained, iteration, "formal_alternating"
            )
            external["used_for_gate"] = all(
                row.get("unique_trajectories", 0) >= 32
                for row in external["external"].values()
            )
            if not external["used_for_gate"]:
                external["diagnostic_only_reason"] = (
                    "fewer than 32 independent trajectories per matchup"
                )
            accepted_updates = int(chosen["policy_steps"]) + int(
                chosen["value_steps"]
            )
            candidate_checkpoint = (
                self.output
                / "candidates"
                / f"iteration_{iteration:06d}.pt"
            )
            candidate_checkpoint.parent.mkdir(parents=True, exist_ok=True)
            scheduler = {
                **scheduler,
                "iteration": iteration,
                "completed_iterations": int(
                    scheduler.get("completed_iterations", 0)
                )
                + int(not reasons),
                "accepted_update_iterations": int(
                    scheduler.get(
                        "accepted_update_iterations",
                        scheduler.get("accepted_iterations", 0),
                    )
                )
                + int(not reasons and accepted_updates > 0),
            }
            scheduler["accepted_iterations"] = scheduler[
                "accepted_update_iterations"
            ]
            active_training_logic = getattr(
                self, "training_logic", None
            )
            checkpoint_payload = {
                "schema": "open-junqi-search-v3-formal-alternating-checkpoint",
                "completed": not reasons,
                "phase": (
                    "alternating" if not reasons else "paused_after_gate"
                ),
                "iteration": iteration,
                "global_step": global_step + accepted_updates,
                "config": self.config.to_dict(),
                "config_hash": self.config.hash(),
                "training_logic_schema": (
                    active_training_logic.schema
                    if active_training_logic is not None
                    else None
                ),
                "training_logic_selection": (
                    {
                        "policy_lr_multiplier": (
                            active_training_logic.policy_lr_multiplier
                        ),
                        "policy_candidate_steps": (
                            active_training_logic.policy_candidate_steps
                        ),
                        "selected_policy_step": int(
                            chosen["policy_steps"]
                        ),
                        "value_candidate_steps": (
                            active_training_logic.value_candidate_steps
                        ),
                        "selected_value_step": int(
                            chosen["value_steps"]
                        ),
                        "adaptive_selection": True,
                    }
                    if active_training_logic is not None
                    else None
                ),
                "model": trained.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler,
                "leaf_value_scale": trained.search_v3_leaf_value_scale,
                "leaf_value_calibration": chosen["value_calibration"],
                "replay": {
                    "policy_current_fraction": 0.70,
                    "policy_recent_fraction": 0.15,
                    "policy_recent_iterations": list(
                        range(max(2, iteration - 2), iteration)
                    ),
                    "value_current_fraction": 0.60,
                    "value_recent_fraction": 0.40,
                    "data_iteration": iteration,
                },
                "rng": details["accepted_rng"],
            }
            temporary = candidate_checkpoint.with_suffix(".pt.tmp")
            torch.save(checkpoint_payload, temporary)
            os.replace(temporary, candidate_checkpoint)
            report = {
                "schema": "open-junqi-search-v3-formal-alternating-iteration",
                "iteration": iteration,
                "passed": not reasons,
                "reasons": reasons,
                "data": data,
                "fresh_train": chosen["fresh_train"],
                "held_out": chosen["validation"],
                "paired_policy": chosen["paired_policy"],
                "paired_wdl": chosen["paired_wdl"],
                "value_phase_policy_kl": {
                    "final_limit": CANDIDATE_FINAL_KL_LIMIT,
                    "peak_hard_limit": VALUE_STEP_PEAK_KL_HARD_LIMIT,
                    "value_step_peak_kl": chosen[
                        "value_step_peak_kl"
                    ],
                    "candidate_final_kl": chosen[
                        "candidate_final_kl"
                    ],
                    "candidate_final_family_kl": chosen[
                        "candidate_final_family_kl"
                    ],
                    "candidate_final_family_kl_max": chosen[
                        "candidate_final_family_kl_max"
                    ],
                    "target": 1e-3,
                    "maximum": chosen["value_step_peak_kl"],
                    "mean_to_selected_candidate": float(
                        np.mean(
                            [
                                row["policy_kl"]
                                for row in details["training_log"]
                                if (
                                    row["phase"] == "value"
                                    and (
                                        getattr(
                                            self,
                                            "training_logic",
                                            None,
                                        )
                                        is None
                                        or int(
                                            row.get(
                                                "policy_step", -1
                                            )
                                        )
                                        == int(
                                            chosen["policy_steps"]
                                        )
                                    )
                                )
                            ][: int(chosen["value_steps"])]
                        )
                    )
                    if chosen["value_steps"]
                    else 0.0,
                },
                "pre_probe": pre_probe,
                "post_probe": post_probe,
                "avoidable_repetition": {
                    "before": pre_probe[
                        "search64_avoidable_exit_fraction"
                    ],
                    "after": post_probe[
                        "search64_avoidable_exit_fraction"
                    ],
                },
                "external_deterministic_divergence": divergence,
                "external_arena": external,
                "chosen_candidate": chosen,
                "chosen_candidate_index": details[
                    "chosen_candidate_index"
                ],
                "best_nonzero_candidate": details[
                    "best_nonzero_candidate"
                ],
                "why_epoch0_won": details["why_epoch0_won"],
                "candidate_continuation_dev": details.get(
                    "continuation_dev"
                ),
                "candidates": details["candidates"],
                "training_log": details["training_log"],
                "cycle_endgame_sampling": cycle_sampling,
                "recent_replay_cycle_cleaned_only": details[
                    "recent_replay_cycle_cleaned_only"
                ],
                "recent_replay_fallback_to_current_fresh": details[
                    "recent_replay_fallback_to_current_fresh"
                ],
                "gradient_cosine": details["gradient_cosine"],
                "training_logic": (
                    {
                        "schema": details.get(
                            "training_logic_schema"
                        ),
                        "policy_lr_multiplier": details.get(
                            "policy_lr_multiplier"
                        ),
                        "policy_candidate_steps": details.get(
                            "policy_candidate_steps"
                        ),
                        "value_candidate_steps": details.get(
                            "value_candidate_steps"
                        ),
                        "policy_shortlist": details.get(
                            "policy_shortlist"
                        ),
                        "selected_policy_step": int(
                            chosen["policy_steps"]
                        ),
                        "selected_value_step": int(
                            chosen["value_steps"]
                        ),
                        "policy_candidates": details.get(
                            "policy_candidates"
                        ),
                        "joint_candidates": details.get(
                            "joint_candidates"
                        ),
                    }
                    if active_training_logic is not None
                    else None
                ),
                "candidate_checkpoint": str(candidate_checkpoint),
                "official_checkpoint": (
                    str(checkpoint) if not reasons else None
                ),
                "failure_recovery": (
                    "reuse this committed iteration data; do not regenerate; "
                    "inspect policy mixture and value KL before any PUCT, "
                    "Dirichlet, temperature, or layout change"
                ),
            }
            if not reasons:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".pt.tmp")
                checkpoint_payload["completed"] = True
                torch.save(checkpoint_payload, temporary)
                os.replace(temporary, checkpoint)
            atomic_json(report_path, report)
            atomic_json(
                self.output / "progress.json",
                {
                    "schema": "formal-alternating-progress",
                    "iteration": iteration,
                    "phase": (
                        "completed" if not reasons else "paused_after_gate"
                    ),
                    "passed": not reasons,
                    "reasons": reasons,
                    "data_committed": True,
                    "candidate_checkpoint": str(candidate_checkpoint),
                    "official_checkpoint": (
                        str(checkpoint) if not reasons else None
                    ),
                },
            )
            reports.append(report)
            search_gain = (
                1.0
                - post_probe["search64_oracle_regret"]
                / max(1e-12, post_probe["bare_oracle_regret"])
            )
            championship_status = (
                "PAUSED"
                if reasons
                else (
                    "PROMOTED"
                    if accepted_updates > 0
                    else "INCUMBENT_RETAINED_EPOCH0"
                )
            )
            print(
                f"[championship] iteration={iteration} "
                f"status={championship_status} "
                f"policy={chosen['paired_policy']['classification']} "
                f"wdl={chosen['paired_wdl']['classification']} "
                f"search64-gain={search_gain:.1%} "
                f"value-kl-final={chosen['candidate_final_kl']:.6f} "
                f"value-kl-peak={chosen['value_step_peak_kl']:.6f} "
                f"external-independent="
                f"{'yes' if external['used_for_gate'] else 'no'}",
                flush=True,
            )
            if reasons:
                return {
                    "completed_iteration": iteration - 1,
                    "paused_iteration": iteration,
                    "reports": reports,
                }
            self._after_successful_iteration(
                iteration,
                report,
                checkpoint,
                candidate_checkpoint,
            )
            previous_data = data
        required = set(self.required_confirmation_iterations)
        success = all(
            report.get("passed") is True
            for report in reports
            if report.get("iteration") in required
        ) and required.issubset(
            {
                report.get("iteration")
                for report in reports
                if report.get("passed") is True
            }
        )
        if not required:
            success = all(
                report.get("passed") is True for report in reports
            )
        atomic_json(
            self.output / self.status_filename,
            {
                "schema": "formal-alternating-recovery-status",
                "required_iterations": sorted(required),
                "required_iteration_status": {
                    str(iteration): any(
                        row.get("iteration") == iteration
                        and row.get("passed")
                        for row in reports
                    )
                    for iteration in sorted(required)
                },
                "iteration_4_passed": any(
                    row.get("iteration") == 4 and row.get("passed")
                    for row in reports
                ),
                "iteration_5_passed": any(
                    row.get("iteration") == 5 and row.get("passed")
                    for row in reports
                ),
                "recovery_confirmed": success,
                "completed_through": through_iteration,
            },
        )
        return {
            "completed_iteration": through_iteration,
            "recovery_confirmed": success,
            "reports": reports,
        }


def run_formal_alternating(
    config_path: str | Path,
    through_iteration: int = 5,
) -> dict[str, Any]:
    return FormalAlternatingLineage(config_path).run(through_iteration)
