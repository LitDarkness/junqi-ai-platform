from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .continuation import SuiteContinuationEvaluator
from .dataset import atomic_json
from .evaluation import SearchEvaluator
from .oracle import OracleHistory
from .stages import SearchV3Stages
from .strength import (
    SUITE_SIZE,
    SingleLayoutStrengthSuite,
    _atomic_jsonl,
    _read_jsonl,
    _sha256,
)


DEFAULT_MATCHUPS = (
    ("bare_50", "bare_10"),
    ("search64_50", "search64_10"),
    ("bare_50", "exact082"),
    ("search64_50", "exact082"),
    ("search64_50", "bare_50"),
    ("bare_50", "bare_49"),
    ("search64_50", "search64_49"),
)


def _family_namespace(seed: int, source_family: int) -> int:
    digest = hashlib.sha256(
        f"sealed-v1:{seed}:{source_family}".encode("ascii")
    ).digest()
    return int.from_bytes(digest[:8], "little") & 0x7FFFFFFFFFFFFFFF


class SealedContinuationExperiment:
    """Build a decision-blind state bank, then run formal continuations."""

    def __init__(
        self,
        config_path: str | Path,
        *,
        workers: int = 16,
        block_states: int = 128,
    ) -> None:
        self.stages = SearchV3Stages(config_path)
        self.config = self.stages.config
        self.root = self.stages.root
        self.run_dir = self.stages.run
        self.workers = max(1, int(workers))
        self.block_states = max(1, int(block_states))
        self.scaled_run = (
            self.root
            / "runs"
            / "open_junqi_search_v3_scaled_training"
        ).resolve()
        self.strength_run = (
            self.root / "runs" / "open_junqi_search_v3_strength"
        ).resolve()
        self.checkpoints = {
            10: (
                self.strength_run
                / "checkpoints"
                / "incumbent_iteration_000010.pt"
            ),
            49: (
                self.scaled_run
                / "checkpoints"
                / "loop_iteration_000049.pt"
            ),
            50: (
                self.scaled_run
                / "checkpoints"
                / "loop_iteration_000050.pt"
            ),
        }
        self._validate_inputs()
        self.models = {
            iteration: self._load_model(iteration, path)
            for iteration, path in self.checkpoints.items()
        }

    def _validate_inputs(self) -> None:
        for iteration, checkpoint in self.checkpoints.items():
            if not checkpoint.exists():
                raise RuntimeError(
                    f"required iteration {iteration} checkpoint is missing: "
                    f"{checkpoint}"
                )
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if (
                payload.get("completed") is not True
                or int(payload.get("iteration", -1)) != iteration
                or "model" not in payload
            ):
                raise RuntimeError(
                    f"iteration {iteration} checkpoint is incomplete"
                )
            if iteration in {49, 50}:
                report_path = (
                    self.scaled_run
                    / "reports"
                    / f"iteration_{iteration:06d}.json"
                )
                if not report_path.exists():
                    raise RuntimeError(
                        f"iteration {iteration} report is missing"
                    )
                report = json.loads(
                    report_path.read_text(encoding="utf-8")
                )
                if report.get("passed") is not True:
                    raise RuntimeError(
                        f"iteration {iteration} did not pass its formal gate"
                    )

    def _load_model(
        self, iteration: int, checkpoint: Path
    ) -> PolicyValueNet:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        model = PolicyValueNet(self.config.model)
        model.load_state_dict(payload["model"])
        model.search_v3_leaf_value_scale = float(
            payload.get("leaf_value_scale", 1.0)
        )
        model.eval()
        model.sealed_iteration = iteration
        return model

    def _write_run_manifest(self) -> None:
        manifest = {
            "schema": "open-junqi-search-v3-sealed-continuation-v1",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "training_enabled": False,
            "replay_writes_enabled": False,
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "same_layout_both_sides": True,
            "greedy": True,
            "dirichlet": False,
            "checkpoints": {
                str(iteration): {
                    "path": str(path),
                    "sha256": _sha256(path),
                }
                for iteration, path in self.checkpoints.items()
            },
            "formal_matchups": [
                f"{primary}_vs_{secondary}"
                for primary, secondary in DEFAULT_MATCHUPS
            ],
        }
        path = self.run_dir / "manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("config_hash") != self.config.hash():
                raise RuntimeError("sealed continuation config changed")
            if existing.get("checkpoints") != manifest["checkpoints"]:
                raise RuntimeError("sealed continuation checkpoints changed")
        atomic_json(path, manifest)

    def _build_sealed_suite(self) -> tuple[Path, dict[str, Any]]:
        original_path = (
            self.strength_run
            / "single_layout_strength_suite"
            / "states.jsonl"
        )
        if not original_path.exists():
            raise RuntimeError("original 1024-state strength suite is missing")
        original_rows = _read_jsonl(original_path)
        original_keys = {
            int(row["state_key"]) for row in original_rows
        }
        raw_directory = self.run_dir / "suite_generation_raw"
        builder = SingleLayoutStrengthSuite(
            self.stages,
            self.models[10],
            raw_directory,
            excluded_state_keys=original_keys,
        )
        builder.build()
        raw_rows = _read_jsonl(builder.states_path)
        sealed_directory = (
            self.run_dir / "sealed_continuation_suite"
        )
        states_path = sealed_directory / "states.jsonl"
        seal_path = sealed_directory / "SEALED.json"
        sealed_directory.mkdir(parents=True, exist_ok=True)
        if not states_path.exists():
            rows = []
            for source in raw_rows:
                row = dict(source)
                source_family = int(row["trajectory_family_id"])
                row["source_trajectory_family_id"] = source_family
                row["trajectory_family_id"] = _family_namespace(
                    self.config.seed, source_family
                )
                row["sealed_family_key"] = (
                    f"sealed-v1-{self.config.seed}-"
                    f"{source_family:04d}"
                )
                rows.append(row)
            _atomic_jsonl(states_path, rows)
        rows = _read_jsonl(states_path)
        validation = builder.validate(rows)
        overlap = {
            int(row["state_key"]) for row in rows
        } & original_keys
        if overlap:
            raise RuntimeError(
                "new sealed suite overlaps the original strength suite"
            )
        family_count = len(
            {int(row["trajectory_family_id"]) for row in rows}
        )
        seal = {
            "schema": "open-junqi-search-v3-sealed-suite-v1",
            "sealed_before_evaluation": True,
            "created_unix": time.time(),
            "created_after_training_iteration": 50,
            "seed": self.config.seed,
            "states": len(rows),
            "trajectory_families": family_count,
            "states_path": str(states_path),
            "states_sha256": _sha256(states_path),
            "generation_checkpoint": str(self.checkpoints[10]),
            "generation_checkpoint_sha256": _sha256(
                self.checkpoints[10]
            ),
            "generation_policy": (
                "frozen iteration 10; sampled network/aggressive/random/"
                "cycle-seek trajectories; no matchup result is available "
                "before this seal is written"
            ),
            "original_suite": str(original_path),
            "original_suite_sha256": _sha256(original_path),
            "original_suite_full_state_overlap": 0,
            "layout_id": self.config.boundary.layout_id,
            "same_layout_both_sides": True,
            "mirror": False,
            "training_excluded": True,
            "replay_write_prohibited": True,
            "full_rule_context_retained": True,
            "validation": validation,
        }
        if len(rows) != SUITE_SIZE:
            raise RuntimeError(
                f"sealed suite has {len(rows)} states, expected {SUITE_SIZE}"
            )
        if seal_path.exists():
            existing = json.loads(
                seal_path.read_text(encoding="utf-8")
            )
            immutable_fields = (
                "seed",
                "states",
                "trajectory_families",
                "states_sha256",
                "generation_checkpoint_sha256",
                "original_suite_sha256",
                "original_suite_full_state_overlap",
            )
            if any(
                existing.get(key) != seal.get(key)
                for key in immutable_fields
            ):
                raise RuntimeError("sealed continuation suite changed")
            seal = existing
        else:
            atomic_json(seal_path, seal)
        return states_path, seal

    def run(self) -> dict[str, Any]:
        self._write_run_manifest()
        suite_path, seal = self._build_sealed_suite()
        evaluator = MultiCheckpointContinuationEvaluator(
            stages=self.stages,
            models=self.models,
            checkpoints=self.checkpoints,
            suite_path=suite_path,
            seal=seal,
            output=self.run_dir / "formal_continuation_arena",
            workers=self.workers,
            block_states=self.block_states,
        )
        return evaluator.run(list(DEFAULT_MATCHUPS), states=SUITE_SIZE)


class MultiCheckpointContinuationEvaluator(
    SuiteContinuationEvaluator
):
    """Read-only continuation evaluator for several frozen networks."""

    def __init__(
        self,
        *,
        stages: SearchV3Stages,
        models: dict[int, PolicyValueNet],
        checkpoints: dict[int, Path],
        suite_path: Path,
        seal: dict[str, Any],
        output: Path,
        workers: int,
        block_states: int,
        matchups: tuple[tuple[str, str], ...] = DEFAULT_MATCHUPS,
    ) -> None:
        self.stages = stages
        self.config = stages.config
        self.backend = stages.backend
        self.workers = max(1, int(workers))
        self.block_states = max(1, int(block_states))
        self.suite_path = suite_path
        self.suite = _read_jsonl(suite_path)
        self.suite_by_id = {
            int(row["suite_id"]): row for row in self.suite
        }
        self.models = models
        self.formal_matchups = tuple(matchups)
        self.evaluators = {
            iteration: SearchEvaluator(
                root=stages.root,
                backend=self.backend,
                config=self.config,
                model=model,
                score_scale=1.0,
            )
            for iteration, model in models.items()
        }
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        exact_path = Path(self.config.data.exact082_candidate)
        if not exact_path.is_absolute():
            exact_path = (stages.root / exact_path).resolve()
        self.contract = {
            "schema": (
                "open-junqi-search-v3-sealed-continuation-contract-v1"
            ),
            "suite": str(suite_path),
            "suite_sha256": _sha256(suite_path),
            "suite_seal": seal,
            "checkpoints": {
                str(iteration): {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "leaf_value_scale": float(
                        getattr(
                            models[iteration],
                            "search_v3_leaf_value_scale",
                            1.0,
                        )
                    ),
                }
                for iteration, path in checkpoints.items()
            },
            "exact082_path": str(exact_path),
            "exact082_sha256": _sha256(exact_path),
            "matchups": [
                f"{primary}_vs_{secondary}"
                for primary, secondary in self.formal_matchups
            ],
            "states_per_matchup": SUITE_SIZE,
            "games_per_matchup": SUITE_SIZE * 2,
            "layout_id": self.config.boundary.layout_id,
            "same_complete_start_state": True,
            "formal_repetition_history": True,
            "formal_rule_context": True,
            "paired_role_swap": True,
            "sampling": False,
            "greedy": True,
            "dirichlet": False,
            "search_simulations": 64,
            "block_states": self.block_states,
            "statistical_unit": "trajectory family",
        }
        rendered = json.dumps(
            self.contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.contract["contract_hash"] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        contract_path = self.output / "contract.json"
        if contract_path.exists():
            existing = json.loads(
                contract_path.read_text(encoding="utf-8")
            )
            if (
                existing.get("contract_hash")
                != self.contract["contract_hash"]
            ):
                raise RuntimeError(
                    "formal continuation contract changed; use a new run"
                )
        atomic_json(contract_path, self.contract)

    @staticmethod
    def _network_agent(agent: str) -> tuple[str, int]:
        mode, separator, raw_iteration = agent.rpartition("_")
        if not separator or mode not in {"bare", "search64"}:
            raise KeyError(agent)
        iteration = int(raw_iteration)
        return mode, iteration

    def _external_actions(
        self, games: list[Any], agent: str
    ) -> list[int]:
        if agent != "exact082":
            raise KeyError(agent)
        exact082 = self.evaluators[max(self.evaluators)].exact082

        def select(game: Any) -> int:
            state = game.state
            legal = np.asarray(state.legal_actions(), np.int32)
            return int(
                exact082.select_action(
                    list(state.native_state.board()),
                    legal.tolist(),
                    1,
                    0.0,
                    0.0,
                )
            )

        with ThreadPoolExecutor(
            max_workers=min(self.workers, len(games))
        ) as executor:
            return list(executor.map(select, games))

    def _play_block(
        self,
        rows: list[dict[str, Any]],
        primary_agent: str,
        secondary_agent: str,
        label: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        games = self._new_games(rows, primary_agent, secondary_agent)
        active = games[:]
        search_by_agent: dict[str, dict[str, float]] = defaultdict(
            lambda: {"simulations": 0.0, "seconds": 0.0}
        )
        started = time.perf_counter()
        ply_round = 0
        while active:
            groups: dict[str, list[Any]] = defaultdict(list)
            for game in active:
                groups[self._acting_agent(game)].append(game)
            action_by_game: dict[int, int] = {}
            for agent, selected in groups.items():
                if agent == "exact082":
                    actions = self._external_actions(selected, agent)
                else:
                    mode, iteration = self._network_agent(agent)
                    if iteration not in self.evaluators:
                        raise KeyError(
                            f"no evaluator loaded for {agent}"
                        )
                    evaluator = self.evaluators[iteration]
                    if mode == "bare":
                        results = evaluator.search.bare(
                            [game.state for game in selected]
                        )
                        actions = [result.action for result in results]
                    else:
                        results, metrics = evaluator.search.search(
                            [game.state for game in selected],
                            64,
                            add_root_noise=False,
                        )
                        actions = [result.action for result in results]
                        search_by_agent[agent]["simulations"] += (
                            len(selected) * 64
                        )
                        search_by_agent[agent]["seconds"] += float(
                            metrics["elapsed_seconds"]
                        )
                for game, action in zip(selected, actions):
                    action_by_game[id(game)] = int(action)
            next_active = []
            for game in active:
                action = action_by_game[id(game)]
                if action not in game.state.legal_actions():
                    raise RuntimeError(
                        f"illegal continuation action {action}"
                    )
                game.state.apply(action)
                game.actions.append(action)
                if not game.state.is_terminal():
                    game.history.observe(game.state)
                    next_active.append(game)
            active = next_active
            ply_round += 1
            if ply_round % 20 == 0 or not active:
                completed_games = [
                    game for game in games if game.state.is_terminal()
                ]
                scores = []
                for game in completed_games:
                    value = float(
                        game.state.terminal_value(game.primary_side)
                    )
                    scores.append(
                        1.0 if value > 0 else (
                            0.5 if value == 0 else 0.0
                        )
                    )
                wins = sum(score == 1.0 for score in scores)
                draws = sum(score == 0.5 for score in scores)
                losses = sum(score == 0.0 for score in scores)
                simulations = sum(
                    item["simulations"]
                    for item in search_by_agent.values()
                )
                search_seconds = sum(
                    item["seconds"]
                    for item in search_by_agent.values()
                )
                elapsed = time.perf_counter() - started
                print(
                    f"[sealed continuation] matchup={label} "
                    f"ply={ply_round} active={len(active)}/{len(games)} "
                    f"W/D/L={wins}/{draws}/{losses} "
                    f"score="
                    f"{np.mean(scores) if scores else float('nan'):.3f} "
                    f"games/s="
                    f"{len(completed_games) / max(elapsed, 1e-9):.2f} "
                    f"search64-sps="
                    f"{simulations / max(search_seconds, 1e-9):.0f}",
                    flush=True,
                )
        result_rows = []
        for game in games:
            value = float(game.state.terminal_value(game.primary_side))
            score = 1.0 if value > 0 else (0.5 if value == 0 else 0.0)
            result_rows.append(
                {
                    "suite_id": game.suite_id,
                    "trajectory_family_id": (
                        game.trajectory_family_id
                    ),
                    "setup_seed": game.setup_seed,
                    "primary_side": game.primary_side,
                    "primary_agent": primary_agent,
                    "secondary_agent": secondary_agent,
                    "score": score,
                    "outcome": (
                        "win"
                        if value > 0
                        else ("draw" if value == 0 else "loss")
                    ),
                    "continuation_plies": len(game.actions),
                    "terminal_reason": str(
                        game.state.terminal_reason()
                    ),
                    "first_action": (
                        game.actions[0] if game.actions else None
                    ),
                    "trajectory_sha256": hashlib.sha256(
                        np.asarray(
                            game.actions, dtype=np.int16
                        ).tobytes()
                    ).hexdigest(),
                    "actions": game.actions,
                }
            )
        elapsed = time.perf_counter() - started
        per_agent = {}
        for agent, item in search_by_agent.items():
            per_agent[agent] = {
                **item,
                "simulations_per_second": (
                    item["simulations"]
                    / max(item["seconds"], 1e-9)
                ),
            }
        simulations = sum(
            item["simulations"] for item in search_by_agent.values()
        )
        search_seconds = sum(
            item["seconds"] for item in search_by_agent.values()
        )
        return result_rows, {
            "elapsed_seconds": elapsed,
            "search64_simulations": int(simulations),
            "search64_seconds": float(search_seconds),
            "search64_simulations_per_second": (
                simulations / max(search_seconds, 1e-9)
            ),
            "by_search_agent": per_agent,
        }

    def _transaction_rows(
        self, label: str, states: int
    ) -> dict[tuple[int, int], dict[str, Any]]:
        directory = self.output / f"states_{states:04d}" / label
        result = {}
        for path in sorted(directory.glob("states_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["rows"]:
                result[
                    (
                        int(row["suite_id"]),
                        int(row["primary_side"]),
                    )
                ] = row
        return result

    def _paired_delta(
        self,
        first_label: str,
        second_label: str,
        *,
        states: int,
        seed: int,
    ) -> dict[str, Any]:
        first = self._transaction_rows(first_label, states)
        second = self._transaction_rows(second_label, states)
        if first.keys() != second.keys():
            raise RuntimeError(
                f"paired comparison differs: {first_label}, {second_label}"
            )
        by_family: dict[int, list[float]] = defaultdict(list)
        deltas = []
        for key in sorted(first):
            delta = float(first[key]["score"]) - float(
                second[key]["score"]
            )
            deltas.append(delta)
            by_family[
                int(first[key]["trajectory_family_id"])
            ].append(delta)
        family = np.asarray(
            [np.mean(items) for items in by_family.values()],
            np.float64,
        )
        return {
            "metric": f"{first_label} score minus {second_label} score",
            "paired_games": len(deltas),
            "trajectory_families": len(family),
            "mean_delta": float(np.mean(deltas)),
            "family_mean_delta": float(family.mean()),
            "family_ci95": self._bootstrap(family, seed),
        }

    def _paired_repetition_delta(
        self,
        first_label: str,
        second_label: str,
        *,
        states: int,
        seed: int,
    ) -> dict[str, Any]:
        """Paired family CI for repetition-rate change, first minus second."""
        first = self._transaction_rows(first_label, states)
        second = self._transaction_rows(second_label, states)
        if first.keys() != second.keys():
            raise RuntimeError(
                f"paired comparison differs: {first_label}, {second_label}"
            )
        by_family: dict[int, list[float]] = defaultdict(list)
        deltas = []
        for key in sorted(first):
            delta = float(
                first[key]["terminal_reason"] == "REPETITION"
            ) - float(second[key]["terminal_reason"] == "REPETITION")
            deltas.append(delta)
            by_family[
                int(first[key]["trajectory_family_id"])
            ].append(delta)
        family = np.asarray(
            [np.mean(items) for items in by_family.values()],
            np.float64,
        )
        return {
            "metric": (
                f"{first_label} repetition indicator minus "
                f"{second_label} repetition indicator"
            ),
            "paired_games": len(deltas),
            "trajectory_families": len(family),
            "mean_delta": float(np.mean(deltas)),
            "family_mean_delta": float(family.mean()),
            "family_ci95": self._bootstrap(family, seed),
            "unit": "absolute repetition-rate fraction",
        }

    def run(
        self,
        matchups: list[tuple[str, str]],
        *,
        states: int = SUITE_SIZE,
    ) -> dict[str, Any]:
        if tuple(matchups) != self.formal_matchups:
            raise RuntimeError(
                "sealed formal matchup plan may not be changed in-place"
            )
        summaries = {}
        for primary, secondary in matchups:
            label = f"{primary}_vs_{secondary}"
            summaries[label] = self.run_matchup(
                primary, secondary, states=states
            )
        challenger = max(self.models)
        search_exact = f"search64_{challenger}_vs_exact082"
        bare_exact = f"bare_{challenger}_vs_exact082"
        comparisons = {}
        if search_exact in summaries and bare_exact in summaries:
            comparisons[
                f"search64_minus_bare_{challenger}_vs_exact082"
            ] = self._paired_delta(
                search_exact,
                bare_exact,
                states=states,
                seed=self.config.seed ^ 0xC050 ^ challenger,
            )
        report = {
            "schema": (
                "open-junqi-search-v3-sealed-continuation-arena-v1"
            ),
            "contract": self.contract,
            "states_per_matchup": states,
            "games_per_matchup": states * 2,
            "total_matchups": len(matchups),
            "total_games": states * 2 * len(matchups),
            "matchups": summaries,
            "paired_comparisons": comparisons,
            "champion_replacement_automatic": False,
            "note": (
                "This read-only report supplies formal evidence; it does not "
                "modify championship.json."
            ),
        }
        report_path = self.output / "report.json"
        csv_path = self.output / "summary.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            fields = [
                "matchup",
                "games",
                "paired_states",
                "trajectory_families",
                "wins",
                "draws",
                "losses",
                "score",
                "state_weighted_score",
                "family_weighted_mean_score",
                "family_ci95_low",
                "family_ci95_high",
                "repetition_fraction",
                "capture_flag_fraction",
                "average_continuation_ply",
                "search64_simulations_per_second",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for label, summary in summaries.items():
                writer.writerow(
                    {
                        "matchup": label,
                        "games": summary["games"],
                        "paired_states": summary["paired_states"],
                        "trajectory_families": summary[
                            "trajectory_families"
                        ],
                        "wins": summary["wins"],
                        "draws": summary["draws"],
                        "losses": summary["losses"],
                        "score": summary["score"],
                        "state_weighted_score": summary[
                            "state_weighted_score"
                        ],
                        "family_weighted_mean_score": summary[
                            "family_weighted_mean_score"
                        ],
                        "family_ci95_low": summary["family_ci95"][0],
                        "family_ci95_high": summary["family_ci95"][1],
                        "repetition_fraction": summary[
                            "repetition_fraction"
                        ],
                        "capture_flag_fraction": summary[
                            "capture_flag_fraction"
                        ],
                        "average_continuation_ply": summary[
                            "average_continuation_ply"
                        ],
                        "search64_simulations_per_second": summary[
                            "performance"
                        ]["search64_simulations_per_second"],
                    }
                )
        os.replace(temporary, csv_path)
        report["report_path"] = str(report_path)
        report["summary_csv"] = str(csv_path)
        report["completed"] = True
        atomic_json(report_path, report)
        return report


def run_sealed_continuation(
    config_path: str | Path,
    *,
    workers: int = 16,
    block_states: int = 128,
) -> dict[str, Any]:
    return SealedContinuationExperiment(
        config_path,
        workers=workers,
        block_states=block_states,
    ).run()
