from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import atomic_json
from .evaluation import EvaluationGame
from .oracle import OracleHistory
from .strength import SearchV3Strength, _read_jsonl, _sha256


@dataclass
class _ContinuationGame:
    suite_id: int
    trajectory_family_id: int
    setup_seed: int
    primary_side: int
    primary_agent: str
    secondary_agent: str
    state: Any
    history: OracleHistory
    actions: list[int] = field(default_factory=list)


def _agent_label(agent: str) -> str:
    if agent == "bare":
        return "incumbent_bare"
    if agent == "search64":
        return "incumbent_search64"
    return agent


class SuiteContinuationEvaluator:
    """Continue paired games from immutable suite states to real terminals."""

    def __init__(
        self,
        strength: SearchV3Strength,
        *,
        workers: int = 16,
        block_states: int = 128,
        output_directory: str | Path | None = None,
        contract_extra: dict[str, Any] | None = None,
    ) -> None:
        self.strength = strength
        self.stages = strength.stages
        self.config = strength.config
        self.backend = self.stages.backend
        self.workers = max(1, int(workers))
        self.block_states = max(1, int(block_states))
        self.suite_path = (
            strength.run_dir
            / "single_layout_strength_suite"
            / "states.jsonl"
        )
        if not self.suite_path.exists():
            raise RuntimeError("single_layout_strength_suite is missing")
        self.suite = _read_jsonl(self.suite_path)
        self.suite_by_id = {
            int(row["suite_id"]): row for row in self.suite
        }
        self.evaluator = strength.model
        from .evaluation import SearchEvaluator

        self.search_evaluator = SearchEvaluator(
            root=self.stages.root,
            backend=self.backend,
            config=self.config,
            model=self.evaluator,
            score_scale=strength.score_scale,
        )
        self.output = (
            Path(output_directory)
            if output_directory is not None
            else strength.run_dir / "continuation_arena"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        exact_path = Path(self.config.data.exact082_candidate)
        if not exact_path.is_absolute():
            exact_path = (self.stages.root / exact_path).resolve()
        self.contract = {
            "schema": "single-layout-suite-continuation-contract-v1",
            "suite": str(self.suite_path),
            "suite_sha256": _sha256(self.suite_path),
            "incumbent_checkpoint": str(strength.incumbent),
            "incumbent_checkpoint_sha256": _sha256(
                strength.incumbent
            ),
            "exact082_path": str(exact_path),
            "exact082_sha256": _sha256(exact_path),
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "sampling": False,
            "dirichlet": False,
            "search_simulations": 64,
            "paired_role_swap": True,
            "statistical_unit": "suite state pair / trajectory family",
        }
        if contract_extra:
            self.contract.update(contract_extra)
        rendered = json.dumps(
            self.contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.contract["contract_hash"] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        atomic_json(self.output / "contract.json", self.contract)

    def _state_history(
        self, row: dict[str, Any]
    ) -> tuple[Any, OracleHistory]:
        state = self.backend.new_state(int(row["setup_seed"]))
        history = OracleHistory()
        for action in row["action_trace"]:
            history.observe(state)
            state.apply(int(action))
        history.observe(state)
        if int(state.native_state.key()) != int(row["state_key"]):
            raise RuntimeError("continuation start state mismatch")
        return state, history

    def _new_games(
        self,
        rows: list[dict[str, Any]],
        primary_agent: str,
        secondary_agent: str,
    ) -> list[_ContinuationGame]:
        games = []
        for row in rows:
            state, history = self._state_history(row)
            games.append(
                _ContinuationGame(
                    suite_id=int(row["suite_id"]),
                    trajectory_family_id=int(
                        row["trajectory_family_id"]
                    ),
                    setup_seed=int(row["setup_seed"]),
                    primary_side=1,
                    primary_agent=primary_agent,
                    secondary_agent=secondary_agent,
                    state=state,
                    history=history,
                )
            )
            second_state, second_history = self._state_history(row)
            games.append(
                _ContinuationGame(
                    suite_id=int(row["suite_id"]),
                    trajectory_family_id=int(
                        row["trajectory_family_id"]
                    ),
                    setup_seed=int(row["setup_seed"]),
                    primary_side=-1,
                    primary_agent=primary_agent,
                    secondary_agent=secondary_agent,
                    state=second_state,
                    history=second_history,
                )
            )
        return games

    @staticmethod
    def _acting_agent(game: _ContinuationGame) -> str:
        return (
            game.primary_agent
            if int(game.state.current_player()) == game.primary_side
            else game.secondary_agent
        )

    def _external_actions(
        self, games: list[_ContinuationGame], agent: str
    ) -> list[int]:
        def select(game: _ContinuationGame) -> int:
            state = game.state
            legal = np.asarray(state.legal_actions(), np.int32)
            if agent == "exact082":
                return int(
                    self.search_evaluator.exact082.select_action(
                        list(state.native_state.board()),
                        legal.tolist(),
                        1,
                        0.0,
                        0.0,
                    )
                )
            if agent.startswith("oracle"):
                budget = int(agent.rsplit("_", 1)[-1])
                return int(
                    self.search_evaluator.oracle.evaluate(
                        state, game.history, budget=budget
                    )["root_best_action"]
                )
            raise KeyError(agent)

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
        search_simulations = 0
        search_seconds = 0.0
        started = time.perf_counter()
        ply_round = 0
        while active:
            groups: dict[str, list[_ContinuationGame]] = defaultdict(list)
            for game in active:
                groups[self._acting_agent(game)].append(game)
            action_by_game: dict[int, int] = {}
            for agent, selected in groups.items():
                if agent == "bare":
                    results = self.search_evaluator.search.bare(
                        [game.state for game in selected]
                    )
                    actions = [result.action for result in results]
                elif agent == "search64":
                    results, metrics = (
                        self.search_evaluator.search.search(
                            [game.state for game in selected], 64
                        )
                    )
                    actions = [result.action for result in results]
                    search_simulations += len(selected) * 64
                    search_seconds += float(metrics["elapsed_seconds"])
                else:
                    actions = self._external_actions(selected, agent)
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
                completed = len(games) - len(active)
                elapsed = time.perf_counter() - started
                partial = []
                for game in games:
                    if game.state.is_terminal():
                        value = float(
                            game.state.terminal_value(
                                game.primary_side
                            )
                        )
                        partial.append(
                            1.0 if value > 0 else (
                                0.5 if value == 0 else 0.0
                            )
                        )
                wins = sum(score == 1.0 for score in partial)
                draws = sum(score == 0.5 for score in partial)
                losses = sum(score == 0.0 for score in partial)
                print(
                    f"[championship continuation] matchup={label} "
                    f"ply={ply_round} active={len(active)}/{len(games)} "
                    f"W/D/L={wins}/{draws}/{losses} "
                    f"score="
                    f"{np.mean(partial) if partial else float('nan'):.3f} "
                    f"games/s={completed / max(elapsed, 1e-9):.2f} "
                    f"search64-sps="
                    f"{search_simulations / max(search_seconds, 1e-9):.0f}",
                    flush=True,
                )
        results = []
        for game in games:
            value = float(
                game.state.terminal_value(game.primary_side)
            )
            score = 1.0 if value > 0 else (0.5 if value == 0 else 0.0)
            results.append(
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
        return results, {
            "elapsed_seconds": time.perf_counter() - started,
            "search64_simulations": search_simulations,
            "search64_seconds": search_seconds,
            "search64_simulations_per_second": (
                search_simulations / max(search_seconds, 1e-9)
            ),
        }

    @staticmethod
    def _bootstrap(
        values: np.ndarray, seed: int
    ) -> list[float]:
        if not len(values):
            return [0.0, 0.0]
        rng = np.random.default_rng(seed)
        samples = values[
            rng.integers(
                0, len(values), size=(5000, len(values))
            )
        ].mean(axis=1)
        return list(
            map(float, np.quantile(samples, [0.025, 0.975]))
        )

    def _summarize(
        self,
        rows: list[dict[str, Any]],
        primary_agent: str,
        secondary_agent: str,
        performance: dict[str, Any],
    ) -> dict[str, Any]:
        scores = np.asarray(
            [float(row["score"]) for row in rows], np.float64
        )
        outcome = Counter(row["outcome"] for row in rows)
        reasons = Counter(row["terminal_reason"] for row in rows)
        by_state: dict[int, list[float]] = defaultdict(list)
        family: dict[int, list[float]] = defaultdict(list)
        for row in rows:
            by_state[int(row["suite_id"])].append(float(row["score"]))
        if any(len(items) != 2 for items in by_state.values()):
            raise RuntimeError("paired role swap is incomplete")
        for suite_id, items in by_state.items():
            family_id = next(
                int(row["trajectory_family_id"])
                for row in rows
                if int(row["suite_id"]) == suite_id
            )
            family[family_id].append(float(np.mean(items)))
        paired_scores = np.asarray(
            [np.mean(items) for items in by_state.values()],
            np.float64,
        )
        family_scores = np.asarray(
            [np.mean(items) for items in family.values()],
            np.float64,
        )
        role = {}
        for side, name in ((1, "primary_red"), (-1, "primary_black")):
            selected = [
                row for row in rows if int(row["primary_side"]) == side
            ]
            role_scores = np.asarray(
                [row["score"] for row in selected], np.float64
            )
            counts = Counter(row["outcome"] for row in selected)
            role[name] = {
                "games": len(selected),
                "wins": counts["win"],
                "draws": counts["draw"],
                "losses": counts["loss"],
                "score": float(role_scores.mean()),
            }
        by_category = {}
        categories = sorted(
            {
                row["primary_category"]
                for row in self.suite_by_id.values()
            }
        )
        for category in categories:
            selected = [
                row
                for row in rows
                if self.suite_by_id[int(row["suite_id"])][
                    "primary_category"
                ]
                == category
            ]
            if not selected:
                continue
            selected_scores = np.asarray(
                [row["score"] for row in selected], np.float64
            )
            counts = Counter(row["outcome"] for row in selected)
            by_category[category] = {
                "games": len(selected),
                "wins": counts["win"],
                "draws": counts["draw"],
                "losses": counts["loss"],
                "score": float(selected_scores.mean()),
            }
        return {
            "schema": "single-layout-suite-continuation-result-v1",
            "primary_agent": primary_agent,
            "secondary_agent": secondary_agent,
            "games": len(rows),
            "paired_states": len(by_state),
            "trajectory_families": len(family),
            "wins": outcome["win"],
            "draws": outcome["draw"],
            "losses": outcome["loss"],
            "score": float(scores.mean()),
            "state_weighted_score": float(paired_scores.mean()),
            "family_weighted_mean_score": float(
                family_scores.mean()
            ),
            "paired_state_ci95": self._bootstrap(
                paired_scores,
                self.config.seed
                ^ sum(map(ord, primary_agent + secondary_agent)),
            ),
            "family_ci95": self._bootstrap(
                family_scores,
                self.config.seed
                ^ sum(map(ord, secondary_agent + primary_agent))
                ^ 0xFA11,
            ),
            "role_split": role,
            "by_primary_category": by_category,
            "terminal_reasons": dict(reasons),
            "repetition_fraction": (
                reasons.get("REPETITION", 0) / max(1, len(rows))
            ),
            "capture_flag_fraction": (
                reasons.get("FLAG_CAPTURE", 0) / max(1, len(rows))
            ),
            "average_continuation_ply": float(
                np.mean([row["continuation_plies"] for row in rows])
            ),
            "unique_continuation_trajectories": len(
                {row["trajectory_sha256"] for row in rows}
            ),
            "performance": performance,
            "paired_role_swap_verified": True,
            "same_suite_seed_verified": True,
            "sampling": False,
            "dirichlet": False,
        }

    def run_matchup(
        self,
        primary_agent: str,
        secondary_agent: str,
        *,
        states: int = 1024,
    ) -> dict[str, Any]:
        if states < 1 or states > len(self.suite):
            raise ValueError("invalid continuation suite state count")
        label = f"{primary_agent}_vs_{secondary_agent}"
        directory = (
            self.output / f"states_{states:04d}" / label
        )
        directory.mkdir(parents=True, exist_ok=True)
        selected = self.suite[:states]
        all_rows = []
        performances = []
        for begin in range(0, states, self.block_states):
            end = min(states, begin + self.block_states)
            transaction = (
                directory
                / f"states_{begin:04d}_{end:04d}.json"
            )
            if transaction.exists():
                payload = json.loads(
                    transaction.read_text(encoding="utf-8")
                )
                if (
                    payload.get("completed") is not True
                    or payload.get("contract_hash")
                    != self.contract["contract_hash"]
                ):
                    raise RuntimeError(
                        f"invalid continuation transaction {transaction}"
                    )
                rows = payload["rows"]
                performance = payload["performance"]
            else:
                rows, performance = self._play_block(
                    selected[begin:end],
                    primary_agent,
                    secondary_agent,
                    label,
                )
                atomic_json(
                    transaction,
                    {
                        "schema": (
                            "single-layout-suite-continuation-transaction-v1"
                        ),
                        "completed": True,
                        "contract_hash": self.contract["contract_hash"],
                        "range": [begin, end],
                        "paired_games": 2 * (end - begin),
                        "rows": rows,
                        "performance": performance,
                    },
                )
            all_rows.extend(rows)
            performances.append(performance)
            scores = np.asarray(
                [row["score"] for row in all_rows], np.float64
            )
            wins = int(np.sum(scores == 1.0))
            draws = int(np.sum(scores == 0.5))
            losses = int(np.sum(scores == 0.0))
            print(
                f"[championship] matchup={label} "
                f"states={end}/{states} games={len(all_rows)} "
                f"W/D/L={wins}/{draws}/{losses} "
                f"score={scores.mean():.4f}",
                flush=True,
            )
        performance = {
            "elapsed_seconds": float(
                sum(row["elapsed_seconds"] for row in performances)
            ),
            "search64_simulations": int(
                sum(row["search64_simulations"] for row in performances)
            ),
            "search64_seconds": float(
                sum(row["search64_seconds"] for row in performances)
            ),
        }
        performance["search64_simulations_per_second"] = (
            performance["search64_simulations"]
            / max(performance["search64_seconds"], 1e-9)
        )
        summary = self._summarize(
            all_rows, primary_agent, secondary_agent, performance
        )
        summary["contract_hash"] = self.contract["contract_hash"]
        summary["rows_path"] = str(directory / "games.csv")
        atomic_json(directory / "summary.json", summary)
        temporary = directory / "games.csv.tmp"
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            fields = [
                "suite_id",
                "trajectory_family_id",
                "setup_seed",
                "primary_side",
                "primary_agent",
                "secondary_agent",
                "outcome",
                "score",
                "continuation_plies",
                "terminal_reason",
                "first_action",
                "trajectory_sha256",
            ]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in all_rows:
                writer.writerow(
                    {key: row[key] for key in fields}
                )
        os.replace(temporary, directory / "games.csv")
        return summary

    def run(
        self,
        matchups: list[tuple[str, str]],
        *,
        states: int = 1024,
    ) -> dict[str, Any]:
        summaries = {}
        for primary, secondary in matchups:
            label = f"{primary}_vs_{secondary}"
            summaries[label] = self.run_matchup(
                primary, secondary, states=states
            )
        comparison = None
        if {
            "bare_vs_exact082",
            "search64_vs_exact082",
        }.issubset(summaries):
            run_directory = self.output / f"states_{states:04d}"
            first_rows = {}
            second_rows = {}
            for label, destination in (
                ("bare_vs_exact082", first_rows),
                ("search64_vs_exact082", second_rows),
            ):
                directory = run_directory / label
                for path in sorted(directory.glob("states_*.json")):
                    payload = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                    for row in payload["rows"]:
                        destination[
                            (
                                int(row["suite_id"]),
                                int(row["primary_side"]),
                            )
                        ] = row
            if first_rows.keys() != second_rows.keys():
                raise RuntimeError(
                    "bare/Search64 frozen-opponent pairs differ"
                )
            by_family = defaultdict(list)
            deltas = []
            for key in sorted(first_rows):
                delta = (
                    float(second_rows[key]["score"])
                    - float(first_rows[key]["score"])
                )
                deltas.append(delta)
                by_family[
                    int(first_rows[key]["trajectory_family_id"])
                ].append(delta)
            family_delta = np.asarray(
                [np.mean(value) for value in by_family.values()],
                np.float64,
            )
            comparison = {
                "metric": (
                    "Search64 score minus bare score versus the same "
                    "frozen exact082 opponent"
                ),
                "paired_games": len(deltas),
                "trajectory_families": len(family_delta),
                "mean_delta": float(np.mean(deltas)),
                "family_mean_delta": float(family_delta.mean()),
                "family_ci95": self._bootstrap(
                    family_delta, self.config.seed ^ 0xC011
                ),
            }
        report = {
            "schema": "single-layout-suite-continuation-arena-v1",
            "contract": self.contract,
            "states_per_matchup": states,
            "games_per_matchup": states * 2,
            "matchups": summaries,
            "search64_minus_bare_vs_exact082": comparison,
        }
        report_path = (
            self.output / f"report_states_{states:04d}.json"
        )
        atomic_json(report_path, report)
        if states == 1024:
            atomic_json(self.output / "report.json", report)
        return report


def run_suite_continuations(
    config_path: str | Path,
    *,
    workers: int = 16,
    block_states: int = 128,
    states: int = 1024,
    matchups: list[tuple[str, str]] | None = None,
) -> dict[str, Any]:
    strength = SearchV3Strength(config_path, workers=workers)
    evaluator = SuiteContinuationEvaluator(
        strength,
        workers=workers,
        block_states=block_states,
    )
    return evaluator.run(
        matchups
        or [
            ("bare", "exact082"),
            ("search64", "exact082"),
            ("search64", "bare"),
        ],
        states=states,
    )
