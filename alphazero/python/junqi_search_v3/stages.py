from __future__ import annotations

import hashlib
import json
import os
import random
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_az.backend import CppAZStateAdapter
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout

from .config import SearchV3Config
from .dataset import OracleSample
from .dataset import atomic_json as dataset_atomic_json
from .dataset import iter_shard, shard_state_hashes, write_shard
from .gates import StageGate
from .oracle import (
    OracleHistory,
    SupplementWideQuiescenceOracle,
    oracle_transposition_key,
)
from .trajectories import (
    generate_trajectory_worker,
    initialize_trajectory_worker,
)
from .supervised import SupervisedTrainer
from .evaluation import SearchEvaluator
from .closed_loop import ClosedLoopRunner


def _hash_files(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


class SearchV3Stages:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.root = self.config_path.parent.parent
        self.config = SearchV3Config.load(self.config_path)
        run = Path(self.config.run_directory)
        self.run = run if run.is_absolute() else (self.root / run).resolve()
        self.run.mkdir(parents=True, exist_ok=True)
        self.gates = StageGate(self.run)
        manifest_path = self.run / "manifest.json"
        self._manifest_config_mismatch = False
        if manifest_path.exists():
            manifest = json.loads(
                manifest_path.read_text(encoding="utf-8")
            )
            self._manifest_config_mismatch = (
                manifest.get("config_hash") != self.config.hash()
            )
        self.layout = load_fixed_layout(
            self.config.boundary.layout_set,
            self.config.boundary.layout_id,
            project_root=self.root,
        )
        self.backend = FixedSameLayoutBackend(self.layout, 0.0)
        self._seed(self.config.seed)

    def _require_manifest_config(self) -> None:
        if self._manifest_config_mismatch:
            raise RuntimeError(
                "active config hash differs from this lineage manifest; "
                "create a new run generation or rerun boundary before data"
            )

    @staticmethod
    def _seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed & 0xFFFFFFFF)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def boundary(self) -> dict[str, Any]:
        rule_names = [
            "junqi_board",
            "junqi_state",
            "setup_validator",
            "move_generator",
            "combat_resolver",
            "setup_generator",
            "junqi_environment",
            "az_state",
        ]
        rule_sources = [
            self.root / "cpp" / f"{name}.{extension}"
            for name in rule_names
            for extension in ("h", "cpp")
        ]
        rule_hash = _hash_files(rule_sources)
        oracle_hash = _hash_files(
            [
                self.root / "supplement/src/root_only_runtime.h",
                self.root / "supplement/src/root_only_runtime.cpp",
            ]
        )
        layout_hash = hashlib.sha256(
            ",".join(str(value) for value in self.layout.setup).encode("ascii")
        ).hexdigest()
        manifest = {
            "schema": "open-junqi-search-v3-manifest",
            "config": self.config.to_dict(),
            "config_hash": self.config.hash(),
            "boundary": {
                "layout_id": self.layout.layout_id,
                "layout_hash": layout_hash,
                "canonical_setup30": list(self.layout.setup),
                "same_layout_both_sides": True,
                "fixed_first_player": "red",
                "mirror": False,
                "multiple_layouts": False,
                "rules": {
                    "repetition_limit": self.config.boundary.repetition_limit,
                    "no_capture_ply_limit": self.config.boundary.no_capture_ply_limit,
                    "max_ply": self.config.boundary.max_ply,
                },
                "rule_hash": rule_hash,
                "oracle_source_hash": oracle_hash,
            },
            "disabled": [
                "mirror",
                "multilayout",
                "setup_network",
                "historical_population",
                "psro",
                "advantage",
                "ppo",
                "draw_penalty",
                "repetition_ban",
            ],
        }
        _atomic_json(self.run / "manifest.json", manifest)
        checkpoint = self.run / "checkpoints" / (
            "initial_random.pt"
            if self.config.initialization == "random"
            else "initial_v2b_trunk_ablation.pt"
        )
        if checkpoint.exists():
            existing = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            if existing.get("config_hash") != self.config.hash():
                if (
                    (self.run / "oracle_dataset" / "shards").exists()
                    or (self.gates.read("02_diversity") or {}).get("passed")
                ):
                    raise RuntimeError(
                        "active config changed after dataset generation; "
                        "start a new run generation"
                    )
                old_hash = existing.get("config_hash")
                existing["config"] = self.config.to_dict()
                existing["config_hash"] = self.config.hash()
                existing["phase"] = "boundary_config_metadata_refresh"
                temporary = checkpoint.with_suffix(".pt.tmp")
                torch.save(existing, temporary)
                os.replace(temporary, checkpoint)
                _atomic_json(
                    self.run / "config_migration.json",
                    {
                        "kind": "pre_data_config_metadata_refresh",
                        "model_weights_changed": False,
                        "optimizer_changed": False,
                        "rng_changed": False,
                        "old_config_hash": old_hash,
                        "new_config_hash": self.config.hash(),
                    },
                )
        else:
            model = PolicyValueNet(self.config.model)
            loaded_trunk = None
            if self.config.initialization == "v2b_trunk_ablation":
                source = Path(self.config.ablation_source)
                if not source.is_absolute():
                    source = (self.root / source).resolve()
                payload = torch.load(
                    source, map_location="cpu", weights_only=False
                )
                source_state = payload.get(
                    "model", payload.get("model_state_dict", payload)
                )
                target_state = model.state_dict()
                copied = []
                for key, value in source_state.items():
                    if (
                        key.startswith("encoder.")
                        and key in target_state
                        and target_state[key].shape == value.shape
                    ):
                        target_state[key] = value
                        copied.append(key)
                model.load_state_dict(target_state)
                if not copied:
                    raise RuntimeError("v2b ablation copied no trunk tensors")
                loaded_trunk = {
                    "source": str(source),
                    "tensor_count": len(copied),
                    "policy_head_reinitialized": True,
                    "wdl_head_reinitialized": True,
                }
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=self.config.train.learning_rate,
                weight_decay=self.config.train.weight_decay,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.config.train.epochs)
            )
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema": "open-junqi-search-v3-checkpoint",
                    "phase": "boundary_initialized",
                    "completed": True,
                    "config": self.config.to_dict(),
                    "config_hash": self.config.hash(),
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                    },
                    "trunk_ablation": loaded_trunk,
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
        metrics = {
            "manifest": str(self.run / "manifest.json"),
            "initial_checkpoint": str(checkpoint),
            "layout_hash": layout_hash,
            "rule_hash": rule_hash,
            "oracle_source_hash": oracle_hash,
            "network_parameters": sum(
                parameter.numel()
                for parameter in PolicyValueNet(self.config.model).parameters()
            ),
            "independent_replay": True,
            "independent_rng": True,
            "initialization": self.config.initialization,
            "ablation_source": (
                self.config.ablation_source
                if self.config.initialization == "v2b_trunk_ablation"
                else None
            ),
        }
        return self.gates.write("00_boundary", passed=True, metrics=metrics)

    def _calibration_states(self, count: int) -> list[tuple[Any, OracleHistory]]:
        result = []
        family = 0
        while len(result) < count:
            state = self.backend.new_state(self.config.seed + 10_000 + family)
            history = OracleHistory()
            rng = random.Random(self.config.seed + 90_000 + family)
            target_plies = sorted(
                set(rng.randint(4, 150) for _ in range(4))
            )
            for ply in range(max(target_plies) + 1):
                history.observe(state)
                if ply in target_plies and not state.is_terminal():
                    result.append((state.clone(), OracleHistory(list(history.positions))))
                    if len(result) >= count:
                        break
                if state.is_terminal():
                    break
                legal = np.asarray(state.legal_actions(), np.int32)
                state.apply(int(legal[rng.randrange(len(legal))]))
            family += 1
        return result

    def oracle(self) -> dict[str, Any]:
        self._require_manifest_config()
        self.gates.require_previous("01_oracle")
        cfg = self.config.oracle
        calibration = self._calibration_states(cfg.scale_calibration_states)
        raw_values = []
        raw_payloads = []
        for state, history in calibration:
            legal = np.asarray(state.legal_actions(), np.int32)
            payload = SupplementWideQuiescenceOracle.raw_once(
                state,
                legal,
                history,
                cfg.scale_calibration_budget,
                cfg.max_depth,
            )
            raw_payloads.append(payload)
            for row in payload["candidates"]:
                value = float(
                    row["search_score"]
                    if row["has_search_score"]
                    else row["static_score"]
                )
                if np.isfinite(value) and abs(value) < 100_000:
                    raw_values.append(abs(value))
        scale = max(
            1.0,
            float(np.quantile(raw_values, cfg.scale_quantile))
            if raw_values
            else 1.0,
        )
        scale_payload = {
            "schema": "supplement-wide-quiescence-scale-v1",
            "score_scale": scale,
            "quantile": cfg.scale_quantile,
            "calibration_states": len(calibration),
            "budget": cfg.scale_calibration_budget,
            "frozen": True,
            "state_source": "independent_random_calibration_seed_range_10000",
        }
        _atomic_json(self.run / "oracle_scale.json", scale_payload)
        oracle = SupplementWideQuiescenceOracle(
            score_scale=scale,
            max_depth=cfg.max_depth,
            shallow_fraction=cfg.shallow_fraction,
        )
        test_states = self._calibration_states(cfg.stability_states)
        rows = []
        budget_violations = 0
        coverage_violations = 0
        black_states = 0
        forced_root_pass = True
        tt_key_pass = True
        pv_legal_pass = True
        for state_index, (state, history) in enumerate(test_states):
            legal = {int(action) for action in state.legal_actions()}
            if int(state.current_player()) == -1:
                black_states += 1
            key_a = int(
                __import__("JunqiOracle").position_key(
                    list(state.native_state.board()), 1
                )
            )
            key_b = int(
                __import__("JunqiOracle").position_key(
                    list(state.native_state.board()), 1
                )
            )
            tt_key_pass &= key_a == key_b
            forced = SupplementWideQuiescenceOracle.raw_once(
                state,
                np.asarray([min(legal)], np.int32),
                history,
                500,
                cfg.max_depth,
            )
            forced_root_pass &= int(forced["action"]) == min(legal)
            budget_results = {}
            for budget in cfg.budgets:
                started = time.perf_counter()
                result = oracle.evaluate(state, history, budget=budget)
                result["elapsed_seconds"] = time.perf_counter() - started
                result_actions = {row["action"] for row in result["actions"]}
                if result["nodes_allocated"] > budget:
                    budget_violations += 1
                if result_actions != legal:
                    coverage_violations += 1
                for action_row in result["actions"]:
                    cursor = state.clone()
                    for pv_action in action_row["pv"]:
                        if int(pv_action) not in set(
                            int(value) for value in cursor.legal_actions()
                        ):
                            pv_legal_pass = False
                            break
                        cursor.apply(int(pv_action))
                        if cursor.is_terminal():
                            break
                budget_results[str(budget)] = {
                    "best_action": result["root_best_action"],
                    "best_q": result["actions"][0]["q"],
                    "top3_actions": [
                        action_row["action"]
                        for action_row in result["actions"][:3]
                    ],
                    "q_by_action": {
                        str(action_row["action"]): action_row["q"]
                        for action_row in result["actions"]
                    },
                    "nodes": result["nodes"],
                    "nodes_allocated": result["nodes_allocated"],
                    "elapsed_seconds": result["elapsed_seconds"],
                    "action_count": len(result["actions"]),
                    "pv_nonempty": all(bool(row["pv"]) for row in result["actions"]),
                }
            rows.append(
                {
                    "state_index": state_index,
                    "side": int(state.current_player()),
                    "key": key_a,
                    "budgets": budget_results,
                }
            )
        stable = sum(
            row["budgets"]["8000"]["best_action"]
            == row["budgets"]["32000"]["best_action"]
            for row in rows
        ) / max(1, len(rows))
        top3_stable = sum(
            row["budgets"]["8000"]["best_action"]
            in row["budgets"]["32000"]["top3_actions"]
            for row in rows
        ) / max(1, len(rows))
        regrets = []
        for row in rows:
            low_action = str(row["budgets"]["8000"]["best_action"])
            high = row["budgets"]["32000"]
            regrets.append(
                max(
                    0.0,
                    float(high["best_q"])
                    - float(high["q_by_action"][low_action]),
                )
            )
        mean_regret = float(np.mean(regrets)) if regrets else float("inf")
        pv_pass = all(
            value["pv_nonempty"]
            for row in rows
            for value in row["budgets"].values()
        )
        # Exact terminal sign in the root mover's perspective.
        native = __import__("JunqiAZ").JunqiAZState()
        terminal_board = [0] * 60
        terminal_board[1] = 3
        terminal_board[51] = 11
        terminal_board[56] = -3
        native.load_board_state(terminal_board, 1)
        terminal_state = CppAZStateAdapter(native)
        terminal_history = OracleHistory()
        terminal_history.observe(terminal_state)
        terminal_result = oracle.evaluate(
            terminal_state, terminal_history, budget=500
        )
        capture_action = 51 * 60 + 56
        capture_row = next(
            row for row in terminal_result["actions"]
            if row["action"] == capture_action
        )
        terminal_sign_pass = (
            capture_row["q"] == 1.0
            and capture_row["terminal_reason"] == "FLAG_CAPTURE"
        )
        # A third occurrence of the forced child is an exact draw, and its
        # history-sensitive key must differ from the same board without it.
        repeat_state, repeat_history = self._calibration_states(1)[0]
        repeat_action = int(repeat_state.legal_actions()[0])
        repeat_child = repeat_state.clone()
        repeat_child.apply(repeat_action)
        history_key_before = oracle_transposition_key(
            repeat_state, repeat_history
        )
        repeat_history.observe(repeat_child)
        repeat_history.observe(repeat_child)
        history_key_after = oracle_transposition_key(
            repeat_state, repeat_history
        )
        repeated = SupplementWideQuiescenceOracle.raw_once(
            repeat_state,
            np.asarray([repeat_action], np.int32),
            repeat_history,
            2000,
            cfg.max_depth,
        )
        repeated_row = next(
            row for row in repeated["candidates"]
            if int(row["action"]) == repeat_action
        )
        repeated_score = float(
            repeated_row["search_score"]
            if repeated_row["has_search_score"]
            else repeated_row["static_score"]
        )
        repetition_pass = abs(repeated_score) <= 1e-9
        history_key_pass = history_key_before != history_key_after
        reasons = []
        checks = {
            "module_available": True,
            "all_legal_actions_scored": coverage_violations == 0,
            "total_budget_respected": budget_violations == 0,
            "forced_root_action": forced_root_pass,
            "tt_key_stable": tt_key_pass,
            "black_view_covered": black_states > 0,
            "pv_nonempty": pv_pass,
            "pv_legal_in_canonical_frame": pv_legal_pass,
            "terminal_value_and_reason": terminal_sign_pass,
            "repetition_history_draw": repetition_pass,
            "history_sensitive_tt_key": history_key_pass,
            "budget_stability_8000_32000": stable,
            "budget_top3_stability_8000_32000": top3_stable,
            "budget_mean_regret_8000_to_32000": mean_regret,
        }
        if stable < cfg.minimum_stability_8000_to_32000:
            reasons.append(
                f"8000->32000 best-action stability {stable:.1%} "
                f"< {cfg.minimum_stability_8000_to_32000:.1%}"
            )
        if top3_stable < cfg.minimum_top3_stability_8000_to_32000:
            reasons.append(
                f"8000 best in 32000 top-3 {top3_stable:.1%} "
                f"< {cfg.minimum_top3_stability_8000_to_32000:.1%}"
            )
        if mean_regret > cfg.maximum_mean_regret_8000_to_32000:
            reasons.append(
                f"8000->32000 mean normalized regret {mean_regret:.4f} "
                f"> {cfg.maximum_mean_regret_8000_to_32000:.4f}"
            )
        for name, value in checks.items():
            if name.startswith("budget_"):
                continue
            if not value:
                reasons.append(f"Oracle correctness check failed: {name}")
        passed = not reasons
        report = {
            "oracle": "supplement_wide_quiescence",
            "score_scale": scale,
            "checks": checks,
            "states": rows,
            "budget_violations": budget_violations,
            "coverage_violations": coverage_violations,
            "source_hash": json.loads(
                (self.run / "manifest.json").read_text(encoding="utf-8")
            )["boundary"]["oracle_source_hash"],
        }
        _atomic_json(self.run / "oracle_validation.json", report)
        return self.gates.write(
            "01_oracle", passed=passed, metrics=report, reasons=reasons
        )

    def _replay_trace(
        self, trace: list[int]
    ) -> tuple[Any, OracleHistory]:
        state = self.backend.new_state(self.config.seed)
        history = OracleHistory()
        for action in trace:
            if state.is_terminal():
                raise RuntimeError("probe trace continues after terminal")
            history.observe(state)
            state.apply(int(action))
        history.observe(state)
        return state, history

    def diversity(self) -> dict[str, Any]:
        self._require_manifest_config()
        self.gates.require_previous("02_diversity")
        scale = json.loads(
            (self.run / "oracle_scale.json").read_text(encoding="utf-8")
        )["score_scale"]
        data_root = self.run / "oracle_dataset"
        shard_root = data_root / "shards"
        shard_root.mkdir(parents=True, exist_ok=True)
        seen: set[int] = set()
        family_rows: dict[int, dict[str, Any]] = {}
        for shard in sorted(shard_root.glob("block_*.npz")):
            seen.update(
                int(value) for value in shard_state_hashes(shard)
            )
        for path in sorted((data_root / "transactions").glob("block_*.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["families"]:
                family_rows[int(row["family_id"])] = row
        cfg = self.config.data
        block = max(1, int(cfg.transaction_families))
        transactions = data_root / "transactions"
        transactions.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        for begin in range(0, cfg.trajectory_families, block):
            end = min(cfg.trajectory_families, begin + block)
            ids = [
                family_id
                for family_id in range(begin, end)
                if family_id not in family_rows
            ]
            if not ids:
                continue
            results = {}
            with ProcessPoolExecutor(
                max_workers=min(cfg.workers, len(ids)),
                initializer=initialize_trajectory_worker,
                initargs=(str(self.config_path), float(scale)),
            ) as executor:
                futures = {
                    executor.submit(generate_trajectory_worker, family_id):
                    family_id for family_id in ids
                }
                for future in as_completed(futures):
                    family_id = futures[future]
                    results[family_id] = future.result()
            committed_samples = []
            branch_rows = []
            summaries = []
            for family_id in sorted(results):
                samples, summary, branches = results[family_id]
                unique_samples = []
                for sample in samples:
                    if sample.state_hash in seen:
                        continue
                    seen.add(sample.state_hash)
                    unique_samples.append(sample)
                summary["unique_samples_committed"] = len(unique_samples)
                committed_samples.extend(unique_samples)
                summaries.append(summary)
                branch_rows.extend(
                    [{"family_id": family_id, **row} for row in branches]
                )
            if committed_samples:
                write_shard(
                    shard_root / f"block_{begin:04d}_{end:04d}.npz",
                    committed_samples,
                )
            transaction = {
                "schema": "open-junqi-search-v3-data-transaction",
                "begin": begin,
                "end": end,
                "completed": True,
                "families": summaries,
                "branches": branch_rows,
                "unique_states_total": len(seen),
            }
            dataset_atomic_json(
                transactions / f"block_{begin:04d}_{end:04d}.json",
                transaction,
            )
            for row in summaries:
                family_rows[int(row["family_id"])] = row
            print(
                f"[search-v3 data] families={len(family_rows)}/"
                f"{cfg.trajectory_families} unique_states={len(seen)} "
                f"elapsed={time.perf_counter()-started:.1f}s",
                flush=True,
            )

        rows = [family_rows[index] for index in sorted(family_rows)]
        raw_total_samples = sum(
            int(row.get("unique_samples_committed", 0)) for row in rows
        )
        raw_samples = []
        for shard in sorted(shard_root.glob("block_*.npz")):
            raw_samples.extend(iter_shard(shard))
        important_samples = [row for row in raw_samples if row["important"]]
        normal_samples = [row for row in raw_samples if not row["important"]]
        selection_reason = None
        if len(important_samples) > cfg.target_unique_states_max:
            selected_rows = important_samples
            selection_reason = (
                f"important states alone total {len(important_samples)}, "
                f"above maximum {cfg.target_unique_states_max}"
            )
        else:
            remaining = cfg.target_unique_states_max - len(important_samples)
            normal_samples.sort(
                key=lambda row: hashlib.sha256(
                    f"{self.config.seed}:{row['state_hash']}".encode("ascii")
                ).digest()
            )
            selected_rows = important_samples + normal_samples[:remaining]
        selected_before_family_cap = len(selected_rows)
        # A single unusually long family must not dominate the loss.  Apply
        # the 2% rule to the actual selected set (not the larger raw pool),
        # iterating because removing rows slightly lowers the denominator.
        # Important/branch/combat rows win ties; the remainder is selected by
        # a stable state-hash ordering.
        while selected_rows:
            family_cap = max(1, int(len(selected_rows) * 0.02))
            counts = Counter(row["family_id"] for row in selected_rows)
            if max(counts.values(), default=0) <= family_cap:
                break
            by_family: dict[int, list[dict[str, Any]]] = {}
            for row in selected_rows:
                by_family.setdefault(row["family_id"], []).append(row)
            capped_rows = []
            for family_id, family_samples in by_family.items():
                family_samples.sort(
                    key=lambda row: (
                        not row["important"],
                        hashlib.sha256(
                            f"{self.config.seed}:{family_id}:"
                            f"{row['state_hash']}".encode("ascii")
                        ).digest(),
                    )
                )
                capped_rows.extend(family_samples[:family_cap])
            selected_rows = capped_rows
        family_cap_dropped = selected_before_family_cap - len(selected_rows)
        selected_important_count = sum(
            int(row["important"]) for row in selected_rows
        )
        training_root = data_root / "training_shards"
        training_root.mkdir(parents=True, exist_ok=True)
        for existing in training_root.glob("*.npz"):
            existing.unlink()
        converted = []
        for row in selected_rows:
            converted.append(
                OracleSample(
                    family_id=row["family_id"],
                    split=row["split"],
                    state_hash=row["state_hash"],
                    ply=row["ply"],
                    player=row["player"],
                    opponent=row["opponent"],
                    terminal_reason=row["terminal_reason"],
                    observation=row["observation"],
                    legal_actions=row["legal_actions"],
                    repetition_contexts=row["repetition_contexts"],
                    q_values=row["q_values"],
                    wdl=row["wdl"],
                    important=row["important"],
                    action_trace=row["action_trace"],
                )
            )
        for begin in range(0, len(converted), 4096):
            write_shard(
                training_root / f"training_{begin:06d}.npz",
                converted[begin:begin + 4096],
            )
        total_samples = len(converted)
        selected_family_counts = Counter(
            sample.family_id for sample in converted
        )
        max_family_share = max(
            (
                value / max(1, total_samples)
                for value in selected_family_counts.values()
            ),
            default=1.0,
        )
        prefix_counts = Counter(row["prefix16_sha256"] for row in rows)
        max_prefix_share = max(prefix_counts.values(), default=0) / max(
            1, len(rows)
        )
        split_counts = Counter(row["split"] for row in rows)
        opponent_counts = Counter(row["opponent"] for row in rows)
        oracle_actor_side_counts = Counter(
            str(row["oracle_actor_side"]) for row in rows
        )
        terminal_counts = Counter(row["terminal_reason"] for row in rows)
        branch_requested = Counter(
            str(row["branches_requested"]) for row in rows
        )
        branch_realized = Counter(
            str(row["branches_realized"]) for row in rows
        )

        # Select held-out diagnostic states.  The immutable 32k labels are
        # saved outside the training shards and never sampled by the trainer.
        probe_path = self.run / "probe_manifest.json"
        if not probe_path.exists():
            candidates = []
            for shard in sorted(training_root.glob("training_*.npz")):
                for row in iter_shard(shard):
                    if row["split"] in {"validation", "test"}:
                        candidates.append(row)
            probe_rng = random.Random(self.config.seed ^ 0x50524F4245)
            probe_rng.shuffle(candidates)
            chosen = candidates[: min(cfg.probe_states, len(candidates))]
            oracle = SupplementWideQuiescenceOracle(
                score_scale=float(scale),
                max_depth=self.config.oracle.max_depth,
                shallow_fraction=self.config.oracle.shallow_fraction,
            )
            probes = []
            for index, row in enumerate(chosen):
                state, history = self._replay_trace(row["action_trace"])
                label = oracle.evaluate(
                    state, history, budget=cfg.probe_budget
                )
                probes.append(
                    {
                        "probe_id": index,
                        "family_id": row["family_id"],
                        "split": row["split"],
                        "state_hash": row["state_hash"],
                        "ply": row["ply"],
                        "action_trace": row["action_trace"],
                        "root_best_action": label["root_best_action"],
                        "actions": label["actions"],
                    }
                )
            dataset_atomic_json(
                probe_path,
                {
                    "schema": "open-junqi-search-v3-probe-manifest",
                    "immutable": True,
                    "config_hash": self.config.hash(),
                    "budget": cfg.probe_budget,
                    "score_scale": scale,
                    "probes": probes,
                },
            )
        probe_count = len(
            json.loads(probe_path.read_text(encoding="utf-8"))["probes"]
        )
        reasons = []
        if len(rows) != cfg.trajectory_families:
            reasons.append(
                f"only {len(rows)}/{cfg.trajectory_families} families committed"
            )
        if not (
            cfg.target_unique_states_min
            <= total_samples
            <= cfg.target_unique_states_max
        ):
            reasons.append(
                f"unique states {total_samples} outside "
                f"[{cfg.target_unique_states_min},"
                f"{cfg.target_unique_states_max}]"
            )
        if selection_reason:
            reasons.append(selection_reason)
        if max_family_share > 0.02:
            reasons.append(
                f"one trajectory family contributes {max_family_share:.2%} > 2%"
            )
        if max_prefix_share > 0.20:
            reasons.append(
                f"one 16-ply prefix contributes {max_prefix_share:.2%} > 20%"
            )
        if abs(
            oracle_actor_side_counts.get("1", 0)
            - oracle_actor_side_counts.get("-1", 0)
        ) > 1:
            reasons.append(
                "Oracle actor sides are not balanced across trajectory families"
            )
        if probe_count < min(cfg.probe_states, 512):
            reasons.append(
                f"only {probe_count}/{min(cfg.probe_states,512)} probes"
            )
        report = {
            "families": len(rows),
            "raw_unique_states": raw_total_samples,
            "unique_states": total_samples,
            "important_states_available": len(important_samples),
            "important_states_selected": selected_important_count,
            "normal_states_selected": (
                total_samples - selected_important_count
            ),
            "family_cap_dropped_states": family_cap_dropped,
            "family_cap_final": (
                max(1, int(total_samples * 0.02))
                if total_samples
                else 0
            ),
            "selection_manifest": str(training_root),
            "target_unique_states": [
                cfg.target_unique_states_min,
                cfg.target_unique_states_max,
            ],
            "split_families": dict(split_counts),
            "opponents": dict(opponent_counts),
            "oracle_actor_sides": dict(oracle_actor_side_counts),
            "terminal_reasons": dict(terminal_counts),
            "branch_requested": dict(branch_requested),
            "branch_realized": dict(branch_realized),
            "maximum_family_sample_share": max_family_share,
            "maximum_prefix16_share": max_prefix_share,
            "probe_count": probe_count,
            "probe_manifest": str(probe_path),
            "elapsed_seconds": time.perf_counter() - started,
            "workers": cfg.workers,
        }
        dataset_atomic_json(data_root / "summary.json", report)
        return self.gates.write(
            "02_diversity",
            passed=not reasons,
            metrics=report,
            reasons=reasons,
        )

    def supervised(self) -> dict[str, Any]:
        self._require_manifest_config()
        self.gates.require_previous("03_supervised")
        trainer = SupervisedTrainer(self.config, self.run)
        overfit_path = self.run / "supervised_overfit.json"
        overfit = None
        if overfit_path.exists():
            cached = json.loads(overfit_path.read_text(encoding="utf-8"))
            phase = cached.get("policy_phase", {})
            if (
                cached.get("passed") is True
                and cached.get("samples")
                == self.config.train.overfit_batch_size
                and cached.get("tau") == 0.2
                and phase.get("value_loss_weight") == 0.0
                and phase.get("kl_weight") == 0.0
                and phase.get("weight_decay") == 0.0
                and phase.get("symmetry_augmentation") is False
                and phase.get("diagnostic_top_set_weight") == 0.05
            ):
                overfit = cached
                overfit["reused_after_formal_selection_fix"] = True
        if overfit is None:
            overfit_model, overfit = trainer.overfit(tau=0.2)
            del overfit_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            _atomic_json(overfit_path, overfit)
        if not overfit["passed"]:
            return self.gates.write(
                "03_supervised",
                passed=False,
                metrics={"overfit": overfit},
                reasons=[
                    "2048-sample fixed-batch overfit gate failed; "
                    "formal training is blocked"
                ],
            )
        ablations = []
        for tau in self.config.data.tau_labels:
            model, report = trainer.fit_variant(
                tau,
                min(3, self.config.train.epochs),
                save_name=f"tau_{str(tau).replace('.', '_')}_ablation.pt",
            )
            del model
            ablations.append(report)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        for row in ablations:
            validation = row["validation"]
            available_signal = max(
                1e-8,
                validation["uniform"] - validation["target_entropy"],
            )
            row["captured_label_signal_fraction"] = (
                validation["uniform"] - validation["policy"]
            ) / available_signal
        candidates = [
            row
            for row in ablations
            if row["validation"]["policy"]
            < row["validation"]["uniform"]
        ]
        if not candidates:
            return self.gates.write(
                "03_supervised",
                passed=False,
                metrics={
                    "overfit": overfit,
                    "tau_ablations": ablations,
                },
                reasons=[
                    "no tau ablation beats the legal-uniform Policy baseline"
                ],
            )
        selected = max(
            candidates,
            key=lambda row: (
                row["captured_label_signal_fraction"],
                row["validation"]["top1"],
                -row["validation"]["excess"],
            ),
        )
        tau = float(selected["tau"])
        model, formal = trainer.fit_variant(
            tau,
            self.config.train.epochs,
            save_name="supervised_latest.pt",
        )
        train_metric = trainer.evaluate(model, "train", tau)
        validation = formal["validation"]
        test = formal["test"]
        del model
        constant_wdl = trainer.constant_wdl_baseline("validation")
        history = formal["epochs"]
        reasons = []
        if validation["policy"] >= validation["uniform"]:
            reasons.append(
                "validation Policy CE is not below the legal-uniform baseline"
            )
        if history[-1]["validation"]["excess"] >= history[0]["validation"]["excess"]:
            reasons.append("validation policy excess did not decrease")
        if validation["wdl"] >= constant_wdl:
            reasons.append(
                "held-out WDL did not beat the constant-class predictor"
            )
        if validation["excess"] - train_metric["excess"] > 0.5:
            reasons.append("train/validation policy excess gap exceeds 0.5")
        if validation["illegal_probability"] > 1e-8:
            reasons.append("illegal action probability is nonzero")
        report = {
            "overfit": overfit,
            "tau_ablations": ablations,
            "selected_tau": tau,
            "tau_selection_rule": (
                "maximize captured Oracle-vs-uniform label signal among "
                "ablations whose Policy CE beats legal-uniform"
            ),
            "formal": formal,
            "train": train_metric,
            "validation": validation,
            "test": test,
            "constant_wdl_validation_ce": constant_wdl,
            "checkpoint": formal["checkpoint"],
            "policy_weight": self.config.train.policy_weight,
            "terminal_wdl_weight": self.config.train.terminal_wdl_weight,
            "oracle_auxiliary_weight": self.config.train.oracle_value_aux_weight,
            "oracle_auxiliary_coverage": 0.0,
        }
        _atomic_json(self.run / "supervised_report.json", report)
        return self.gates.write(
            "03_supervised",
            passed=not reasons,
            metrics=report,
            reasons=reasons,
        )

    def search_gain(self) -> dict[str, Any]:
        self._require_manifest_config()
        self.gates.require_previous("04_search_gain")
        checkpoint = torch.load(
            self.run / "checkpoints" / "supervised_latest.pt",
            map_location="cpu",
            weights_only=False,
        )
        model = PolicyValueNet(self.config.model)
        model.load_state_dict(checkpoint["model"])
        scale = json.loads(
            (self.run / "oracle_scale.json").read_text(encoding="utf-8")
        )["score_scale"]
        evaluator = SearchEvaluator(
            root=self.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=scale,
        )
        probe_payload = json.loads(
            (self.run / "probe_manifest.json").read_text(encoding="utf-8")
        )
        probes = probe_payload["probes"]
        states = [self._replay_trace(row["action_trace"])[0] for row in probes]
        variants: dict[str, Any] = {}
        bare = evaluator.search.bare(states)
        network_values = evaluator.search.values(states)
        search_results = {"bare": bare}
        performance = {}
        for simulations in self.config.puct.simulations:
            result, metric = evaluator.search.search(states, simulations)
            search_results[str(simulations)] = result
            performance[str(simulations)] = metric
        classifications = []
        for probe in probes:
            cycles = [
                row for row in probe["actions"] if row["enters_repetition"]
            ]
            exits = [
                row for row in probe["actions"] if not row["enters_repetition"]
            ]
            if not cycles or not exits:
                classifications.append("not_cycle_entry")
                continue
            cycle_q = max(float(row["q"]) for row in cycles)
            exit_q = max(float(row["q"]) for row in exits)
            if exit_q >= cycle_q + 0.05:
                classifications.append("avoidable_repetition")
            elif cycle_q >= exit_q + 0.05:
                classifications.append("rational_draw")
            else:
                classifications.append("unresolved")
        for name, results in search_results.items():
            regrets = []
            repeat_selected = 0
            exit_by_class = Counter()
            total_by_class = Counter()
            entropies = []
            actions = []
            for probe, result, classification in zip(
                probes, results, classifications
            ):
                by_action = {
                    int(row["action"]): float(row["q"])
                    for row in probe["actions"]
                }
                best = max(by_action.values())
                regrets.append(best - by_action[result.action])
                selected = next(
                    row for row in probe["actions"]
                    if int(row["action"]) == result.action
                )
                repeat_selected += int(selected["enters_repetition"])
                if classification != "not_cycle_entry":
                    total_by_class[classification] += 1
                    exit_by_class[classification] += int(
                        not selected["enters_repetition"]
                    )
                entropies.append(result.root_entropy)
                actions.append(result.action)
            variants[name] = {
                "oracle_regret_mean": float(np.mean(regrets)),
                "oracle_regret_median": float(np.median(regrets)),
                "root_entropy_mean": float(np.mean(entropies)),
                "repetition_action_fraction": repeat_selected / len(probes),
                "exit_fraction_by_class": {
                    key: exit_by_class[key] / max(1, value)
                    for key, value in total_by_class.items()
                },
                "actions": actions,
            }
        oracle_root_values = np.asarray(
            [max(float(row["q"]) for row in probe["actions"]) for probe in probes]
        )
        value_calibration = {
            "oracle_best_q_mse": float(
                np.mean((network_values - oracle_root_values) ** 2)
            ),
            "network_value_mean": float(np.mean(network_values)),
            "oracle_best_q_mean": float(np.mean(oracle_root_values)),
        }
        stability = {}
        sequence = ["bare"] + [
            str(value) for value in self.config.puct.simulations
        ]
        for first, second in zip(sequence, sequence[1:]):
            stability[f"{first}_to_{second}"] = float(
                np.mean(
                    np.asarray(variants[first]["actions"])
                    == np.asarray(variants[second]["actions"])
                )
            )
        direct = evaluator.paired_candidate_vs(
            "candidate_64",
            "baseline_0",
            self.config.puct.search_gain_direct_games,
            self.config.seed ^ 0x444952454354,
        )
        external = {}
        for opponent in ("supplement_wide_quiescence", "exact082"):
            for name in sequence:
                candidate = (
                    "candidate_0" if name == "bare"
                    else f"candidate_{name}"
                )
                external[f"{name}_vs_{opponent}"] = (
                    evaluator.paired_candidate_vs(
                        candidate,
                        opponent,
                        self.config.puct.search_gain_external_games,
                        self.config.seed
                        ^ sum(ord(char) for char in opponent)
                        ^ int(0 if name == "bare" else name),
                    )
                )
        score32 = np.mean(
            [
                external[f"32_vs_{opponent}"]["score"]
                for opponent in ("supplement_wide_quiescence", "exact082")
            ]
        )
        score64 = np.mean(
            [
                external[f"64_vs_{opponent}"]["score"]
                for opponent in ("supplement_wide_quiescence", "exact082")
            ]
        )
        bare_regret = variants["bare"]["oracle_regret_mean"]
        regret64 = variants["64"]["oracle_regret_mean"]
        avoid_bare = variants["bare"]["exit_fraction_by_class"].get(
            "avoidable_repetition", 0.0
        )
        avoid64 = variants["64"]["exit_fraction_by_class"].get(
            "avoidable_repetition", 0.0
        )
        reasons = []
        if not (
            direct["score"] > 0.52 or direct["ci95"][0] > 0.50
        ):
            reasons.append(
                "PUCT64 did not beat bare Policy at the 512-game gate"
            )
        if score64 < score32 - 0.02:
            reasons.append("PUCT64 external score is >2pp below PUCT32")
        if regret64 > bare_regret * 0.90:
            reasons.append(
                "PUCT64 Oracle regret did not improve bare by at least 10%"
            )
        if avoid64 < avoid_bare:
            reasons.append(
                "PUCT64 reduced the exit rate on avoidable repetitions"
            )
        report = {
            "probe_count": len(probes),
            "classification_counts": dict(Counter(classifications)),
            "probe_variants": variants,
            "root_action_stability": stability,
            "value_calibration": value_calibration,
            "direct_puct64_vs_bare": direct,
            "external": external,
            "external_score32": float(score32),
            "external_score64": float(score64),
            "performance": performance,
            "evaluation_noise": False,
            "evaluation_temperature": 0.0,
        }
        _atomic_json(self.run / "search_gain_report.json", report)
        return self.gates.write(
            "04_search_gain",
            passed=not reasons,
            metrics=report,
            reasons=reasons,
        )

    def closed_loop(self) -> dict[str, Any]:
        self._require_manifest_config()
        self.gates.require_previous("05_closed_loop")
        _atomic_json(
            self.run / "STATUS.json",
            {
                "schema": "open-junqi-search-v3-status",
                "current_stage": "05_closed_loop",
                "status": "running",
                "next_stage": None,
                "gate": str(self.run / "gates" / "05_closed_loop.json"),
            },
        )
        scale = json.loads(
            (self.run / "oracle_scale.json").read_text(encoding="utf-8")
        )["score_scale"]
        runner = ClosedLoopRunner(
            root=self.root,
            run=self.run,
            backend=self.backend,
            config=self.config,
            score_scale=float(scale),
            replay_trace=self._replay_trace,
        )
        passed, report, reasons = runner.run()
        _atomic_json(self.run / "closed_loop_report.json", report)
        return self.gates.write(
            "05_closed_loop",
            passed=passed,
            metrics=report,
            reasons=reasons,
        )
