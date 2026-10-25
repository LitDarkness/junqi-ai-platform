from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .dataset import atomic_json
from .evaluation import SearchEvaluator
from .oracle import OracleHistory
from .stages import SearchV3Stages


SUITE_SIZE = 1024
SUITE_QUOTAS = {
    "opening": 160,
    "midgame": 320,
    "endgame": 160,
    "tactical": 320,
    "cycle_entry": 64,
}
ORACLE_BUDGETS = (500, 2000, 8000, 32000)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    row,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )
    os.replace(temporary, path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _inverse_action(action: int) -> int:
    source, target = divmod(int(action), 60)
    return target * 60 + source


@dataclass
class _SuiteGame:
    game_id: int
    seed: int
    state: Any
    rng: random.Random
    mode: str
    trace: list[int] = field(default_factory=list)
    position_keys: list[int] = field(default_factory=list)
    last_action_by_side: dict[int, int] = field(default_factory=dict)


class SingleLayoutStrengthSuite:
    """Immutable, reconstructable, training-excluded strength state suite."""

    def __init__(
        self,
        stages: SearchV3Stages,
        model: PolicyValueNet,
        output: Path,
        *,
        excluded_state_keys: set[int] | None = None,
    ) -> None:
        self.stages = stages
        self.config = stages.config
        self.backend = stages.backend
        self.model = model
        self.output = output
        self.excluded_state_keys = set(excluded_state_keys or ())
        self.states_path = output / "states.jsonl"
        self.manifest_path = output / "manifest.json"
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.evaluator = SearchEvaluator(
            root=stages.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=1.0,
        )

    @staticmethod
    def _tags(state: Any, ply: int) -> list[str]:
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        contexts = np.asarray(
            state.repetition_contexts(), dtype=np.uint8
        )
        board = tuple(int(value) for value in state.native_state.board())
        pieces = sum(value != 0 for value in board)
        tags = []
        if ply < 24:
            tags.append("opening")
        elif pieces <= 24 or ply >= 120:
            tags.append("endgame")
        else:
            tags.append("midgame")
        if any(board[int(action) % 60] != 0 for action in legal):
            tags.append("tactical")
        if len(contexts) and int(contexts.max()) > 0:
            tags.append("cycle_entry")
        return tags

    def _record(self, game: _SuiteGame) -> dict[str, Any]:
        state = game.state
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        contexts = np.asarray(
            state.repetition_contexts(), dtype=np.uint8
        )
        history = OracleHistory()
        reconstructed = self.backend.new_state(game.seed)
        for action in game.trace:
            history.observe(reconstructed)
            reconstructed.apply(int(action))
        history.observe(reconstructed)
        if int(reconstructed.native_state.key()) != int(
            state.native_state.key()
        ):
            raise RuntimeError("suite state reconstruction mismatch")
        features = np.ascontiguousarray(
            state.feature_planes(), dtype=np.float32
        )
        board = [int(value) for value in state.native_state.board()]
        history_payload = json.loads(
            json.dumps(history.payload(state), separators=(",", ":"))
        )
        return {
            "trajectory_family_id": game.game_id,
            "generation_mode": game.mode,
            "setup_seed": game.seed,
            "ply": len(game.trace),
            "action_trace": list(game.trace),
            "position_key_history": list(game.position_keys),
            "state_key": int(state.native_state.key()),
            "position_key": int(state.native_state.position_key()),
            "current_player": int(state.current_player()),
            "board": board,
            "piece_count": sum(value != 0 for value in board),
            "turn_count": int(state.native_state.turn_count()),
            "no_capture_ply": int(
                state.native_state.no_capture_ply()
            ),
            "repetition_count": int(
                state.native_state.repetition_count()
            ),
            "legal_actions": legal.tolist(),
            "repetition_contexts": contexts.tolist(),
            "oracle_history_payload": history_payload,
            "feature_sha256": hashlib.sha256(
                features.tobytes()
            ).hexdigest(),
            "tags": self._tags(state, len(game.trace)),
            "rules_context": {
                "repetition_limit": (
                    self.config.boundary.repetition_limit
                ),
                "no_capture_ply_limit": (
                    self.config.boundary.no_capture_ply_limit
                ),
                "max_ply": self.config.boundary.max_ply,
                "fixed_first_player": (
                    self.config.boundary.fixed_first_player
                ),
            },
            "history_encoding": (
                "fixed initial layout plus complete action_trace; "
                "position_key_history and repetition table retained"
            ),
        }

    @staticmethod
    def _weighted_action(
        game: _SuiteGame,
        result: Any,
    ) -> int:
        legal = result.actions.astype(np.int32, copy=False)
        contexts = np.asarray(
            game.state.repetition_contexts(), dtype=np.uint8
        )
        board = tuple(
            int(value) for value in game.state.native_state.board()
        )
        side = int(game.state.current_player())
        if game.mode == "cycle_seek":
            previous = game.last_action_by_side.get(side)
            if previous is not None:
                inverse = _inverse_action(previous)
                if inverse in legal:
                    return int(inverse)
            if len(contexts) and int(contexts.max()) > 0:
                choices = legal[
                    contexts == int(contexts.max())
                ].tolist()
                return int(game.rng.choice(choices))
        if game.mode == "aggressive":
            captures = [
                int(action)
                for action in legal
                if board[int(action) % 60] != 0
            ]
            if captures and game.rng.random() < 0.90:
                return int(game.rng.choice(captures))
        if game.mode == "random":
            return int(game.rng.choice(legal.tolist()))
        temperature = 1.25 if game.mode == "cycle_seek" else 0.80
        weights = np.power(
            np.clip(result.priors.astype(np.float64), 1e-12, 1.0),
            1.0 / temperature,
        )
        weights /= weights.sum()
        return int(
            game.rng.choices(
                legal.tolist(), weights=weights.tolist(), k=1
            )[0]
        )

    def _select(
        self, candidates: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        used: set[int] = set()
        rng = random.Random(self.config.seed ^ 0x53554954)
        selection_order = (
            "cycle_entry",
            "endgame",
            "opening",
            "tactical",
            "midgame",
        )
        for category in selection_order:
            quota = SUITE_QUOTAS[category]
            rows = list(candidates[category])
            rng.shuffle(rows)
            accepted = 0
            for row in rows:
                identity = int(row["state_key"])
                if identity in used or identity in self.excluded_state_keys:
                    continue
                row = copy.deepcopy(row)
                row["primary_category"] = category
                row["suite_id"] = len(selected)
                selected.append(row)
                used.add(identity)
                accepted += 1
                if accepted == quota:
                    break
            if accepted != quota:
                raise RuntimeError(
                    f"strength suite lacks {category}: "
                    f"{accepted}/{quota}"
                )
        if len(selected) != SUITE_SIZE:
            raise AssertionError(len(selected))
        selected.sort(key=lambda row: int(row["suite_id"]))
        return selected

    def validate(
        self, rows: list[dict[str, Any]]
    ) -> dict[str, Any]:
        if len(rows) != SUITE_SIZE:
            raise RuntimeError(
                f"strength suite size is {len(rows)}, expected {SUITE_SIZE}"
            )
        categories = Counter()
        state_keys = set()
        families = set()
        for row in rows:
            categories[row["primary_category"]] += 1
            state = self.backend.new_state(int(row["setup_seed"]))
            history = OracleHistory()
            for action in row["action_trace"]:
                history.observe(state)
                legal = state.legal_actions()
                if int(action) not in legal:
                    raise RuntimeError(
                        f"illegal suite trace at state {row['suite_id']}"
                    )
                state.apply(int(action))
            history.observe(state)
            if state.is_terminal():
                raise RuntimeError("terminal state entered strength suite")
            if int(state.native_state.key()) != int(row["state_key"]):
                raise RuntimeError("suite full-state key mismatch")
            actual_history = json.loads(
                json.dumps(
                    history.payload(state), separators=(",", ":")
                )
            )
            if actual_history != row["oracle_history_payload"]:
                raise RuntimeError("suite repetition history mismatch")
            if list(map(int, state.legal_actions())) != row["legal_actions"]:
                raise RuntimeError("suite legal-action mismatch")
            if list(map(int, state.repetition_contexts())) != row[
                "repetition_contexts"
            ]:
                raise RuntimeError("suite repetition-context mismatch")
            features = np.ascontiguousarray(
                state.feature_planes(), dtype=np.float32
            )
            if hashlib.sha256(features.tobytes()).hexdigest() != row[
                "feature_sha256"
            ]:
                raise RuntimeError("suite feature mismatch")
            state_keys.add(int(row["state_key"]))
            families.add(int(row["trajectory_family_id"]))
        overlap = state_keys & self.excluded_state_keys
        if overlap:
            raise RuntimeError(
                "strength suite overlaps an explicitly excluded state bank"
            )
        if dict(categories) != SUITE_QUOTAS:
            raise RuntimeError(
                f"strength suite category mismatch: {dict(categories)}"
            )
        if len(state_keys) != SUITE_SIZE:
            raise RuntimeError(
                "strength suite must contain 1024 unique full states"
            )
        return {
            "states": len(rows),
            "unique_full_state_keys": len(state_keys),
            "trajectory_families": len(families),
            "primary_categories": dict(categories),
            "training_excluded": True,
            "reconstruction_verified": True,
            "excluded_state_overlap": 0,
        }

    def build(self) -> dict[str, Any]:
        if self.states_path.exists():
            rows = _read_jsonl(self.states_path)
            validation = self.validate(rows)
            if self.manifest_path.exists():
                manifest = json.loads(
                    self.manifest_path.read_text(encoding="utf-8")
                )
                if manifest["states_sha256"] != _sha256(
                    self.states_path
                ):
                    raise RuntimeError("strength suite hash mismatch")
            else:
                manifest = {
                    "schema": "single-layout-strength-suite-v1",
                    "created_unix": time.time(),
                    "layout_id": self.config.boundary.layout_id,
                    "mirror": False,
                    "multiple_layouts": False,
                    "fixed_first_player": 1,
                    "states": SUITE_SIZE,
                    "quotas": SUITE_QUOTAS,
                    "states_path": str(self.states_path),
                    "states_sha256": _sha256(self.states_path),
                    "training_excluded": True,
                    "replay_write_prohibited": True,
                    "recovered_after_post_write_validation": True,
                    "validation": validation,
                }
                atomic_json(self.manifest_path, manifest)
            return {**manifest, "validation": validation}

        modes = (
            "network_sample",
            "network_sample",
            "aggressive",
            "random",
            "cycle_seek",
        )
        games = []
        generation_games = 512
        for game_id in range(generation_games):
            seed = (
                self.config.seed
                ^ 0x53545245
                ^ (game_id * 0x9E3779B1)
            ) & 0x7FFFFFFF
            state = self.backend.new_state(seed)
            games.append(
                _SuiteGame(
                    game_id=game_id,
                    seed=seed,
                    state=state,
                    rng=random.Random(seed ^ 0xA511E9B3),
                    mode=modes[game_id % len(modes)],
                    position_keys=[
                        int(state.native_state.position_key())
                    ],
                )
            )
        candidates: dict[str, list[dict[str, Any]]] = {
            category: [] for category in SUITE_QUOTAS
        }
        identities: dict[str, set[int]] = {
            category: set() for category in SUITE_QUOTAS
        }
        active = games
        started = time.perf_counter()
        maximum_generation_ply = 360
        while active:
            states = [game.state for game in active]
            results = self.evaluator.search.bare(states)
            next_active = []
            for game, result in zip(active, results):
                if len(game.trace) % 2 == 0:
                    row = self._record(game)
                    identity = int(row["state_key"])
                    for tag in row["tags"]:
                        if (
                            tag in candidates
                            and identity not in identities[tag]
                            and len(candidates[tag])
                            < SUITE_QUOTAS[tag] * 8
                        ):
                            candidates[tag].append(row)
                            identities[tag].add(identity)
                action = self._weighted_action(game, result)
                side = int(game.state.current_player())
                game.last_action_by_side[side] = action
                game.state.apply(action)
                game.trace.append(action)
                game.position_keys.append(
                    int(game.state.native_state.position_key())
                )
                if (
                    not game.state.is_terminal()
                    and len(game.trace) < maximum_generation_ply
                ):
                    next_active.append(game)
            active = next_active
            ply = max((len(game.trace) for game in games), default=0)
            if ply % 20 == 0 or not active:
                counts = " ".join(
                    f"{key}={len(value)}/{SUITE_QUOTAS[key]}"
                    for key, value in candidates.items()
                )
                elapsed = time.perf_counter() - started
                positions = sum(len(game.trace) for game in games)
                print(
                    f"[strength suite] ply={ply} active={len(active)}/"
                    f"{generation_games} {counts} "
                    f"positions/s={positions / max(elapsed, 1e-9):.1f}",
                    flush=True,
                )
        rows = self._select(candidates)
        _atomic_jsonl(self.states_path, rows)
        validation = self.validate(rows)
        manifest = {
            "schema": "single-layout-strength-suite-v1",
            "created_unix": time.time(),
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "multiple_layouts": False,
            "fixed_first_player": 1,
            "states": SUITE_SIZE,
            "quotas": SUITE_QUOTAS,
            "states_path": str(self.states_path),
            "states_sha256": _sha256(self.states_path),
            "generation_games": generation_games,
            "generation_modes": Counter(
                game.mode for game in games
            ),
            "maximum_generation_ply": maximum_generation_ply,
            "training_excluded": True,
            "replay_write_prohibited": True,
            "excluded_full_state_keys": len(self.excluded_state_keys),
            "validation": validation,
        }
        manifest["generation_modes"] = dict(
            manifest["generation_modes"]
        )
        atomic_json(self.manifest_path, manifest)
        return manifest


class StrengthEvaluator:
    def __init__(
        self,
        stages: SearchV3Stages,
        model: PolicyValueNet,
        suite_path: Path,
        output: Path,
        workers: int,
        score_scale: float,
    ) -> None:
        self.stages = stages
        self.config = stages.config
        self.backend = stages.backend
        self.model = model
        self.rows = _read_jsonl(suite_path)
        self.output = output
        self.workers = max(1, int(workers))
        self.evaluator = SearchEvaluator(
            root=stages.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=score_scale,
        )

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
            raise RuntimeError("strength evaluation reconstruction mismatch")
        return state, history

    def _network_actions(self) -> dict[str, Any]:
        path = self.output / "network_actions.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        states = [self._state_history(row)[0] for row in self.rows]
        bare = []
        search64 = []
        search_metrics = []
        started = time.perf_counter()
        batch = self.config.puct.inference_batch_size
        for begin in range(0, len(states), batch):
            selected = states[begin:begin + batch]
            bare.extend(
                result.action
                for result in self.evaluator.search.bare(selected)
            )
            results, metrics = self.evaluator.search.search(selected, 64)
            search64.extend(result.action for result in results)
            search_metrics.append(metrics)
            completed = min(len(states), begin + batch)
            elapsed = time.perf_counter() - started
            print(
                f"[strength network] states={completed}/{len(states)} "
                f"states/s={completed / max(elapsed, 1e-9):.1f} "
                f"search64-sps={metrics['simulations_per_second']:.0f}",
                flush=True,
            )
        payload = {
            "schema": "single-layout-strength-network-actions-v1",
            "states": len(states),
            "bare_actions": list(map(int, bare)),
            "search64_actions": list(map(int, search64)),
            "bare_search_agreement": float(
                np.mean(np.asarray(bare) == np.asarray(search64))
            ),
            "elapsed_seconds": time.perf_counter() - started,
            "search64_simulations_per_second": float(
                sum(
                    metric["simulations_per_second"]
                    * metric["elapsed_seconds"]
                    for metric in search_metrics
                )
                / max(
                    1e-9,
                    sum(
                        metric["elapsed_seconds"]
                        for metric in search_metrics
                    ),
                )
            ),
        }
        atomic_json(path, payload)
        return payload

    def _oracle_block(
        self,
        budget: int,
        rows: list[dict[str, Any]],
        bare: list[int],
        search64: list[int],
    ) -> list[dict[str, Any]]:
        def evaluate(
            args: tuple[dict[str, Any], int, int]
        ) -> dict[str, Any]:
            row, bare_action, search_action = args
            state, history = self._state_history(row)
            oracle = self.evaluator.oracle.evaluate(
                state, history, budget=budget
            )
            q = {
                int(item["action"]): float(item["q"])
                for item in oracle["actions"]
            }
            best_action = int(oracle["root_best_action"])
            best_q = q[best_action]
            return {
                "suite_id": int(row["suite_id"]),
                "trajectory_family_id": int(
                    row["trajectory_family_id"]
                ),
                "primary_category": row["primary_category"],
                "budget": budget,
                "nodes": int(oracle["nodes"]),
                "best_action": best_action,
                "best_q": best_q,
                "bare_action": int(bare_action),
                "bare_q": q[int(bare_action)],
                "bare_regret": best_q - q[int(bare_action)],
                "search64_action": int(search_action),
                "search64_q": q[int(search_action)],
                "search64_regret": best_q - q[int(search_action)],
            }

        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            return list(
                executor.map(evaluate, zip(rows, bare, search64))
            )

    def _oracle_budget(
        self,
        budget: int,
        network: dict[str, Any],
    ) -> list[dict[str, Any]]:
        directory = self.output / "transactions" / f"oracle_{budget:05d}"
        directory.mkdir(parents=True, exist_ok=True)
        block = 32
        result = []
        started = time.perf_counter()
        for begin in range(0, len(self.rows), block):
            end = min(len(self.rows), begin + block)
            path = directory / f"states_{begin:04d}_{end:04d}.json"
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("completed") is not True
                    or payload.get("budget") != budget
                ):
                    raise RuntimeError(f"invalid strength transaction {path}")
                rows = payload["rows"]
            else:
                rows = self._oracle_block(
                    budget,
                    self.rows[begin:end],
                    network["bare_actions"][begin:end],
                    network["search64_actions"][begin:end],
                )
                atomic_json(
                    path,
                    {
                        "schema": "strength-oracle-transaction-v1",
                        "completed": True,
                        "budget": budget,
                        "range": [begin, end],
                        "rows": rows,
                    },
                )
            result.extend(rows)
            if end % 128 == 0 or end == len(self.rows):
                elapsed = time.perf_counter() - started
                rate = len(result) / max(elapsed, 1e-9)
                eta = (len(self.rows) - end) / max(rate, 1e-9)
                print(
                    f"[strength oracle@{budget}] "
                    f"states={end}/{len(self.rows)} "
                    f"states/s={rate:.2f} eta={eta:.1f}s",
                    flush=True,
                )
        return result

    def _exact082(
        self, network: dict[str, Any]
    ) -> list[dict[str, Any]]:
        path = self.output / "transactions" / "exact082.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))["rows"]

        def select(row: dict[str, Any]) -> dict[str, Any]:
            state, unused = self._state_history(row)
            del unused
            legal = np.asarray(state.legal_actions(), np.int32)
            action = int(
                self.evaluator.exact082.select_action(
                    list(state.native_state.board()),
                    legal.tolist(),
                    1,
                    0.0,
                    0.0,
                )
            )
            return {
                "suite_id": int(row["suite_id"]),
                "trajectory_family_id": int(
                    row["trajectory_family_id"]
                ),
                "primary_category": row["primary_category"],
                "exact082_action": action,
            }

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=self.workers) as executor:
            rows = list(executor.map(select, self.rows))
        atomic_json(
            path,
            {
                "schema": "strength-exact082-transaction-v1",
                "completed": True,
                "rows": rows,
            },
        )
        print(
            f"[strength exact082] states={len(rows)} "
            f"states/s={len(rows) / max(time.perf_counter() - started, 1e-9):.1f}",
            flush=True,
        )
        return rows

    @staticmethod
    def _paired_family_interval(
        rows: list[dict[str, Any]], seed: int
    ) -> dict[str, Any]:
        family = defaultdict(list)
        for row in rows:
            family[int(row["trajectory_family_id"])].append(
                float(row["search64_regret"])
                - float(row["bare_regret"])
            )
        values = np.asarray(
            [np.mean(items) for items in family.values()], np.float64
        )
        rng = np.random.default_rng(seed)
        if len(values):
            samples = values[
                rng.integers(0, len(values), size=(2000, len(values)))
            ].mean(axis=1)
            interval = np.quantile(samples, [0.025, 0.975]).tolist()
        else:
            interval = [0.0, 0.0]
        return {
            "trajectory_families": len(values),
            "mean_search_minus_bare_regret": float(
                values.mean() if len(values) else 0.0
            ),
            "ci95": list(map(float, interval)),
            "unit": "trajectory_family",
        }

    def _oracle_summary(
        self, rows: list[dict[str, Any]], budget: int
    ) -> dict[str, Any]:
        def aggregate(selected: list[dict[str, Any]]) -> dict[str, Any]:
            bare = np.asarray(
                [row["bare_regret"] for row in selected], np.float64
            )
            search = np.asarray(
                [row["search64_regret"] for row in selected], np.float64
            )
            return {
                "states": len(selected),
                "bare": {
                    "mean_regret": float(bare.mean()),
                    "median_regret": float(np.median(bare)),
                    "top1_agreement": float(
                        np.mean(
                            [
                                row["bare_action"] == row["best_action"]
                                for row in selected
                            ]
                        )
                    ),
                },
                "search64": {
                    "mean_regret": float(search.mean()),
                    "median_regret": float(np.median(search)),
                    "top1_agreement": float(
                        np.mean(
                            [
                                row["search64_action"]
                                == row["best_action"]
                                for row in selected
                            ]
                        )
                    ),
                },
                "search64_regret_reduction": float(
                    1.0 - search.mean() / max(1e-12, bare.mean())
                ),
                "search_better_equal_worse": {
                    "better": int(np.sum(search < bare - 1e-12)),
                    "equal": int(np.sum(np.abs(search - bare) <= 1e-12)),
                    "worse": int(np.sum(search > bare + 1e-12)),
                },
            }

        return {
            "reference": f"supplement_wide_quiescence@{budget}",
            "overall": aggregate(rows),
            "by_primary_category": {
                category: aggregate(
                    [
                        row
                        for row in rows
                        if row["primary_category"] == category
                    ]
                )
                for category in SUITE_QUOTAS
            },
            "paired_family": self._paired_family_interval(
                rows, self.config.seed ^ budget
            ),
        }

    def run(self) -> dict[str, Any]:
        self.output.mkdir(parents=True, exist_ok=True)
        report_path = self.output / "report.json"
        if report_path.exists():
            return json.loads(report_path.read_text(encoding="utf-8"))
        started = time.perf_counter()
        network = self._network_actions()
        oracle_rows = {
            budget: self._oracle_budget(budget, network)
            for budget in ORACLE_BUDGETS
        }
        exact = self._exact082(network)
        exact_actions = np.asarray(
            [row["exact082_action"] for row in exact], np.int32
        )
        bare = np.asarray(network["bare_actions"], np.int32)
        search = np.asarray(network["search64_actions"], np.int32)
        exact_summary = {
            "reference": "exact082",
            "states": len(exact),
            "bare_top1_agreement": float(np.mean(bare == exact_actions)),
            "search64_top1_agreement": float(
                np.mean(search == exact_actions)
            ),
            "when_bare_search_disagree": {
                "states": int(np.sum(bare != search)),
                "exact082_matches_bare": int(
                    np.sum((bare != search) & (exact_actions == bare))
                ),
                "exact082_matches_search64": int(
                    np.sum((bare != search) & (exact_actions == search))
                ),
            },
            "note": (
                "exact082 exposes an action but no calibrated per-action Q; "
                "therefore this suite reports action agreement, not regret"
            ),
        }
        report = {
            "schema": "single-layout-strength-report-v1",
            "states": len(self.rows),
            "training_excluded": True,
            "network": network,
            "oracle": {
                str(budget): self._oracle_summary(
                    oracle_rows[budget], budget
                )
                for budget in ORACLE_BUDGETS
            },
            "exact082": exact_summary,
            "elapsed_seconds": time.perf_counter() - started,
        }
        atomic_json(report_path, report)

        csv_path = self.output / "per_state.csv"
        temporary = csv_path.with_suffix(".csv.tmp")
        with temporary.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            fields = [
                "suite_id",
                "trajectory_family_id",
                "primary_category",
                "bare_action",
                "search64_action",
                "exact082_action",
            ]
            for budget in ORACLE_BUDGETS:
                fields.extend(
                    [
                        f"oracle_{budget}_best_action",
                        f"bare_regret_{budget}",
                        f"search64_regret_{budget}",
                    ]
                )
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index, source in enumerate(self.rows):
                row = {
                    "suite_id": index,
                    "trajectory_family_id": source[
                        "trajectory_family_id"
                    ],
                    "primary_category": source["primary_category"],
                    "bare_action": int(bare[index]),
                    "search64_action": int(search[index]),
                    "exact082_action": int(exact_actions[index]),
                }
                for budget in ORACLE_BUDGETS:
                    item = oracle_rows[budget][index]
                    row[f"oracle_{budget}_best_action"] = item[
                        "best_action"
                    ]
                    row[f"bare_regret_{budget}"] = item["bare_regret"]
                    row[f"search64_regret_{budget}"] = item[
                        "search64_regret"
                    ]
                writer.writerow(row)
        os.replace(temporary, csv_path)
        return report


class SearchV3Strength:
    def __init__(
        self, config_path: str | Path, workers: int | None = None
    ) -> None:
        self.stages = SearchV3Stages(config_path)
        self.config = self.stages.config
        self.root = self.stages.root
        self.run_dir = self.stages.run
        self.source = (
            self.root / "runs" / "open_junqi_search_v3_alternating"
        ).resolve()
        self.workers = int(workers or self.config.data.workers)
        self.source_checkpoint = (
            self.source
            / "checkpoints"
            / "loop_iteration_000010.pt"
        )
        self.source_report = (
            self.source / "reports" / "iteration_000010.json"
        )
        self.incumbent = (
            self.run_dir
            / "checkpoints"
            / "incumbent_iteration_000010.pt"
        )
        self._freeze_and_import()
        payload = torch.load(
            self.incumbent, map_location="cpu", weights_only=False
        )
        self.model = PolicyValueNet(self.config.model)
        self.model.load_state_dict(payload["model"])
        self.model.search_v3_leaf_value_scale = float(
            payload["leaf_value_scale"]
        )
        self.score_scale = float(
            json.loads(
                (
                    self.source / "oracle_scale.json"
                ).read_text(encoding="utf-8")
            )["score_scale"]
        )

    def _freeze_and_import(self) -> None:
        if not self.source_checkpoint.exists() or not self.source_report.exists():
            raise RuntimeError(
                "iteration 10 is not a complete formal incumbent"
            )
        payload = torch.load(
            self.source_checkpoint, map_location="cpu", weights_only=False
        )
        report = json.loads(
            self.source_report.read_text(encoding="utf-8")
        )
        if (
            payload.get("completed") is not True
            or payload.get("iteration") != 10
            or report.get("passed") is not True
        ):
            raise RuntimeError(
                "iteration 10 did not pass the formal alternating gate"
            )
        later = list(
            (
                self.source / "checkpoints"
            ).glob("loop_iteration_*.pt")
        )
        if any(int(path.stem.rsplit("_", 1)[-1]) > 10 for path in later):
            raise RuntimeError(
                "alternating contains iterations later than the requested "
                "frozen incumbent"
            )
        frozen = {
            "schema": "open-junqi-search-v3-frozen-lineage-v1",
            "incumbent_iteration": 10,
            "checkpoint": str(self.source_checkpoint),
            "checkpoint_sha256": _sha256(self.source_checkpoint),
            "report": str(self.source_report),
            "report_sha256": _sha256(self.source_report),
            "reason": (
                "closed-loop validation complete; strength measurement moved "
                "to open_junqi_search_v3_strength"
            ),
            "mirror": False,
            "layout_id": self.config.boundary.layout_id,
        }
        frozen_path = self.source / "FROZEN.json"
        if frozen_path.exists():
            existing = json.loads(
                frozen_path.read_text(encoding="utf-8")
            )
            if (
                existing["checkpoint_sha256"]
                != frozen["checkpoint_sha256"]
            ):
                raise RuntimeError("alternating freeze manifest changed")
        else:
            atomic_json(frozen_path, frozen)
        self.incumbent.parent.mkdir(parents=True, exist_ok=True)
        if not self.incumbent.exists():
            temporary = self.incumbent.with_suffix(".pt.tmp")
            shutil.copy2(self.source_checkpoint, temporary)
            os.replace(temporary, self.incumbent)
        if _sha256(self.incumbent) != _sha256(self.source_checkpoint):
            raise RuntimeError("strength incumbent copy mismatch")
        source_manifest = json.loads(
            (self.source / "manifest.json").read_text(encoding="utf-8")
        )
        manifest = {
            "schema": "open-junqi-search-v3-strength-v1",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "source_lineage": str(self.source),
            "source_lineage_frozen": True,
            "incumbent_iteration": 10,
            "incumbent_checkpoint": str(self.incumbent),
            "incumbent_checkpoint_sha256": _sha256(self.incumbent),
            "incumbent_report_sha256": _sha256(self.source_report),
            "source_config_hash": source_manifest["config_hash"],
            "layout_id": self.config.boundary.layout_id,
            "mirror": False,
            "multiple_layouts": False,
            "training_enabled": False,
            "replay_writes_enabled": False,
        }
        atomic_json(self.run_dir / "manifest.json", manifest)

    def run(
        self,
        *,
        build_only: bool = False,
        evaluate_only: bool = False,
    ) -> dict[str, Any]:
        suite_directory = (
            self.run_dir / "single_layout_strength_suite"
        )
        suite = SingleLayoutStrengthSuite(
            self.stages, self.model, suite_directory
        )
        if evaluate_only and not suite.states_path.exists():
            raise RuntimeError("strength suite has not been built")
        suite_manifest = suite.build()
        if build_only:
            return {"suite": suite_manifest, "evaluation": None}
        evaluator = StrengthEvaluator(
            self.stages,
            self.model,
            suite.states_path,
            self.run_dir / "evaluation",
            self.workers,
            self.score_scale,
        )
        report = evaluator.run()
        return {"suite": suite_manifest, "evaluation": report}


def run_strength(
    config_path: str | Path,
    *,
    workers: int | None = None,
    build_only: bool = False,
    evaluate_only: bool = False,
) -> dict[str, Any]:
    return SearchV3Strength(config_path, workers=workers).run(
        build_only=build_only,
        evaluate_only=evaluate_only,
    )
