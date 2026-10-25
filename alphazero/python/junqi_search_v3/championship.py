from __future__ import annotations

import json
import hashlib
import os
import time
from pathlib import Path
from typing import Any

import torch

from junqi_az.model import PolicyValueNet

from .dataset import atomic_json
from .sealed_continuation import (
    MultiCheckpointContinuationEvaluator,
    _family_namespace,
)
from .strength import (
    SUITE_SIZE,
    SingleLayoutStrengthSuite,
    _atomic_jsonl,
    _read_jsonl,
    _sha256,
)
from .stages import _hash_files


def _canonical_hash(payload: Any) -> str:
    rendered = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _model_hash(model: PolicyValueNet) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(tensor.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class TriggeredSealedChampionship:
    """A fresh, read-only continuation suite for one title challenge."""

    def __init__(
        self,
        *,
        stages: Any,
        training_run: Path,
        workers: int = 16,
        block_states: int = 128,
    ) -> None:
        self.stages = stages
        self.config = stages.config
        self.root = stages.root
        self.training_run = training_run
        self.workers = max(1, int(workers))
        self.block_states = max(1, int(block_states))

    def _load_model(
        self, checkpoint: Path, iteration: int
    ) -> PolicyValueNet:
        payload = torch.load(
            checkpoint, map_location="cpu", weights_only=False
        )
        if (
            payload.get("completed") is not True
            or int(payload.get("iteration", -1)) != iteration
        ):
            raise RuntimeError(
                f"incomplete championship checkpoint: {checkpoint}"
            )
        model = PolicyValueNet(self.config.model)
        model.load_state_dict(payload["model"])
        model.search_v3_leaf_value_scale = float(
            payload["leaf_value_scale"]
        )
        model.eval()
        return model

    def _rule_hash(self) -> str:
        names = (
            "junqi_board",
            "junqi_state",
            "setup_validator",
            "move_generator",
            "combat_resolver",
            "setup_generator",
            "junqi_environment",
            "az_state",
        )
        sources = [
            self.root / "cpp" / f"{name}.{extension}"
            for name in names
            for extension in ("h", "cpp")
        ]
        source_hash = _hash_files(sources)
        return _canonical_hash(
            {
                "source_hash": source_hash,
                "repetition_limit": (
                    self.config.boundary.repetition_limit
                ),
                "no_capture_ply_limit": (
                    self.config.boundary.no_capture_ply_limit
                ),
                "max_ply": self.config.boundary.max_ply,
            }
        )

    def _search_configuration_hash(self) -> str:
        return _canonical_hash(
            {
                "puct": self.config.to_dict()["puct"],
                "greedy": True,
                "sampling": False,
                "dirichlet": False,
                "candidate_search_simulations": 64,
                "leaf_value": "network_wdl",
                "paired_role_swap": True,
                "promotion_gate": (
                    "strength-plus-paired-family-repetition-v1"
                ),
                "repetition_noise_allowance": 0.02,
            }
        )

    def _write_decision(
        self,
        challenge_directory: Path,
        decision: dict[str, Any],
    ) -> None:
        path = challenge_directory / "promotion_decision.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing != decision:
                atomic_json(
                    challenge_directory
                    / (
                        "promotion_decision_"
                        f"{decision.get('status', 'updated').lower()}.json"
                    ),
                    decision,
                )
                return
        atomic_json(path, decision)

    def _excluded_state_bank(
        self, challenge_directory: Path
    ) -> tuple[set[int], list[dict[str, Any]]]:
        paths = [
            (
                self.root
                / "runs"
                / "open_junqi_search_v3_strength"
                / "single_layout_strength_suite"
                / "states.jsonl"
            ),
            (
                self.root
                / "runs"
                / "open_junqi_search_v3_sealed_continuation_v1"
                / "sealed_continuation_suite"
                / "states.jsonl"
            ),
        ]
        paths.extend(
            self.training_run.glob(
                "championships/*/sealed_suite/states.jsonl"
            )
        )
        current_states = (
            challenge_directory / "sealed_suite" / "states.jsonl"
        ).resolve()
        keys: set[int] = set()
        sources = []
        for path in paths:
            path = path.resolve()
            if path == current_states or not path.exists():
                continue
            rows = _read_jsonl(path)
            keys.update(int(row["state_key"]) for row in rows)
            sources.append(
                {
                    "path": str(path),
                    "sha256": _sha256(path),
                    "states": len(rows),
                }
            )
        return keys, sources

    def _build_suite(
        self,
        *,
        challenge_directory: Path,
        suite_seed: int,
        champion_model: PolicyValueNet,
        champion_checkpoint: Path,
        champion_iteration: int,
        challenger_iteration: int,
    ) -> tuple[Path, dict[str, Any]]:
        excluded, excluded_sources = self._excluded_state_bank(
            challenge_directory
        )
        original_seed = self.config.seed
        self.config.seed = suite_seed
        try:
            raw_directory = challenge_directory / "suite_generation_raw"
            builder = SingleLayoutStrengthSuite(
                self.stages,
                champion_model,
                raw_directory,
                excluded_state_keys=excluded,
            )
            builder.build()
            raw_rows = _read_jsonl(builder.states_path)
            sealed_directory = challenge_directory / "sealed_suite"
            states_path = sealed_directory / "states.jsonl"
            seal_path = sealed_directory / "SEALED.json"
            sealed_directory.mkdir(parents=True, exist_ok=True)
            if not states_path.exists():
                rows = []
                for source in raw_rows:
                    row = dict(source)
                    source_family = int(
                        row["trajectory_family_id"]
                    )
                    row["source_trajectory_family_id"] = source_family
                    row["trajectory_family_id"] = _family_namespace(
                        suite_seed, source_family
                    )
                    row["sealed_family_key"] = (
                        f"championship-{challenger_iteration:06d}-"
                        f"{source_family:04d}"
                    )
                    rows.append(row)
                _atomic_jsonl(states_path, rows)
            rows = _read_jsonl(states_path)
            validation = builder.validate(rows)
        finally:
            self.config.seed = original_seed
        overlap = {
            int(row["state_key"]) for row in rows
        } & excluded
        if overlap:
            raise RuntimeError(
                "championship suite overlaps a prior decision suite"
            )
        seal = {
            "schema": "search-v3-triggered-sealed-suite-v1",
            "sealed_before_evaluation": True,
            "created_unix": time.time(),
            "suite_seed": suite_seed,
            "states": len(rows),
            "trajectory_families": len(
                {
                    int(row["trajectory_family_id"])
                    for row in rows
                }
            ),
            "states_path": str(states_path),
            "states_sha256": _sha256(states_path),
            "generation_iteration": champion_iteration,
            "generation_checkpoint": str(champion_checkpoint),
            "generation_checkpoint_sha256": _sha256(
                champion_checkpoint
            ),
            "challenger_iteration": challenger_iteration,
            "excluded_state_keys": len(excluded),
            "excluded_suite_sources": excluded_sources,
            "excluded_full_state_overlap": 0,
            "layout_id": self.config.boundary.layout_id,
            "same_layout_both_sides": True,
            "mirror": False,
            "training_excluded": True,
            "replay_write_prohibited": True,
            "validation": validation,
        }
        if len(rows) != SUITE_SIZE:
            raise RuntimeError("championship suite is not 1024 states")
        if seal_path.exists():
            existing = json.loads(
                seal_path.read_text(encoding="utf-8")
            )
            for key in (
                "suite_seed",
                "states_sha256",
                "generation_checkpoint_sha256",
                "challenger_iteration",
                "excluded_full_state_overlap",
            ):
                if existing.get(key) != seal.get(key):
                    raise RuntimeError(
                        "sealed championship suite changed"
                    )
            seal = existing
        else:
            atomic_json(seal_path, seal)
        return states_path, seal

    @staticmethod
    def _direct_relation(summary: dict[str, Any]) -> str:
        low, high = map(float, summary["family_ci95"])
        if low > 0.5:
            return "significantly_improved"
        if high < 0.5:
            return "clearly_degraded"
        return "inconclusive"

    def run(
        self,
        *,
        challenger_iteration: int,
        challenger_checkpoint: Path,
        champion_iteration: int,
        champion_checkpoint: Path,
        training_report: dict[str, Any],
        suite_override: tuple[Path, dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        challenge_directory = (
            self.training_run
            / "championships"
            / (
                f"challenger_{challenger_iteration:06d}_vs_"
                f"champion_{champion_iteration:06d}"
            )
        )
        challenge_directory.mkdir(parents=True, exist_ok=True)
        suite_seed = (
            int(self.config.seed)
            ^ 0x4348414D
            ^ (challenger_iteration * 0x9E3779B1)
            ^ (champion_iteration * 0xA511E9B3)
        ) & 0x7FFFFFFF
        champion_model = self._load_model(
            champion_checkpoint, champion_iteration
        )
        challenger_model = self._load_model(
            challenger_checkpoint, challenger_iteration
        )
        challenger_model_hash = _model_hash(challenger_model)
        champion_model_hash = _model_hash(champion_model)
        rules_hash = self._rule_hash()
        search_configuration_hash = (
            self._search_configuration_hash()
        )
        if challenger_model_hash == champion_model_hash:
            decision = {
                "schema": "search-v3-championship-decision-v2",
                "status": "SKIPPED_IDENTICAL_MODEL",
                "challenger_iteration": challenger_iteration,
                "champion_iteration": champion_iteration,
                "challenger_model_hash": challenger_model_hash,
                "champion_model_hash": champion_model_hash,
                "rules_hash": rules_hash,
                "search_configuration_hash": (
                    search_configuration_hash
                ),
                "promoted": False,
                "bare_vs_champion": "not_run",
                "search64_vs_champion": "not_run",
                "search64_vs_bare": "not_run",
                "arena_report": None,
            }
            self._write_decision(challenge_directory, decision)
            return decision
        if suite_override is None:
            suite_path, seal = self._build_suite(
                challenge_directory=challenge_directory,
                suite_seed=suite_seed,
                champion_model=champion_model,
                champion_checkpoint=champion_checkpoint,
                champion_iteration=champion_iteration,
                challenger_iteration=challenger_iteration,
            )
        else:
            suite_path, seal = suite_override
            suite_path = suite_path.resolve()
            suite_seed = int(
                seal.get("suite_seed", seal.get("seed", suite_seed))
            )
            if (
                not suite_path.exists()
                or int(seal.get("states", -1)) != SUITE_SIZE
                or seal.get("states_sha256") != _sha256(suite_path)
            ):
                raise RuntimeError(
                    "sealed championship suite override is invalid"
                )
            atomic_json(
                challenge_directory / "sealed_suite_reuse.json",
                {
                    "schema": "search-v3-sealed-suite-reuse-v1",
                    "source": str(suite_path),
                    "source_sha256": _sha256(suite_path),
                    "seal": seal,
                    "reason": (
                        "iteration-107 late-stage bootstrap must use the "
                        "same previously sealed 1024-state suite"
                    ),
                    "training_excluded": True,
                    "read_only": True,
                },
            )
        cache_key_payload = {
            "challenger_model_hash": challenger_model_hash,
            "champion_model_hash": champion_model_hash,
            "suite_hash": seal["states_sha256"],
            "rules_hash": rules_hash,
            "search_configuration_hash": search_configuration_hash,
        }
        cache_key = _canonical_hash(cache_key_payload)
        cache_path = (
            self.training_run
            / "championship_cache"
            / f"{cache_key}.json"
        )
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached_decision = dict(cached["decision"])
            decision = dict(cached_decision)
            decision["status"] = "CACHED_RESULT"
            decision["cache_reused"] = True
            decision["cache_key"] = cache_key
            decision["cached_decision_status"] = cached_decision.get(
                "status", "COMPLETED"
            )
            self._write_decision(challenge_directory, decision)
            return decision
        matchups = (
            (
                f"bare_{challenger_iteration}",
                f"bare_{champion_iteration}",
            ),
            (
                f"search64_{challenger_iteration}",
                f"search64_{champion_iteration}",
            ),
            (f"bare_{challenger_iteration}", "exact082"),
            (f"search64_{challenger_iteration}", "exact082"),
            (
                f"search64_{challenger_iteration}",
                f"bare_{challenger_iteration}",
            ),
            (f"bare_{champion_iteration}", "exact082"),
            (f"search64_{champion_iteration}", "exact082"),
        )
        original_seed = self.config.seed
        self.config.seed = suite_seed
        try:
            evaluator = MultiCheckpointContinuationEvaluator(
                stages=self.stages,
                models={
                    champion_iteration: champion_model,
                    challenger_iteration: challenger_model,
                },
                checkpoints={
                    champion_iteration: champion_checkpoint,
                    challenger_iteration: challenger_checkpoint,
                },
                suite_path=suite_path,
                seal=seal,
                output=challenge_directory / "arena",
                workers=self.workers,
                block_states=self.block_states,
                matchups=matchups,
            )
            report = evaluator.run(
                list(matchups), states=SUITE_SIZE
            )
            bare_exact_delta = evaluator._paired_delta(
                f"bare_{challenger_iteration}_vs_exact082",
                f"bare_{champion_iteration}_vs_exact082",
                states=SUITE_SIZE,
                seed=suite_seed ^ 0xBAE0,
            )
            search_exact_delta = evaluator._paired_delta(
                f"search64_{challenger_iteration}_vs_exact082",
                f"search64_{champion_iteration}_vs_exact082",
                states=SUITE_SIZE,
                seed=suite_seed ^ 0x5EA2,
            )
            bare_repetition_delta = (
                evaluator._paired_repetition_delta(
                    f"bare_{challenger_iteration}_vs_exact082",
                    f"bare_{champion_iteration}_vs_exact082",
                    states=SUITE_SIZE,
                    seed=suite_seed ^ 0x7EBA,
                )
            )
            search_repetition_delta = (
                evaluator._paired_repetition_delta(
                    f"search64_{challenger_iteration}_vs_exact082",
                    f"search64_{champion_iteration}_vs_exact082",
                    states=SUITE_SIZE,
                    seed=suite_seed ^ 0x7E5A,
                )
            )
        finally:
            self.config.seed = original_seed
        labels = {
            "bare_vs_champion": (
                f"bare_{challenger_iteration}_vs_"
                f"bare_{champion_iteration}"
            ),
            "search_vs_champion": (
                f"search64_{challenger_iteration}_vs_"
                f"search64_{champion_iteration}"
            ),
            "search_vs_bare": (
                f"search64_{challenger_iteration}_vs_"
                f"bare_{challenger_iteration}"
            ),
        }
        bare_relation = self._direct_relation(
            report["matchups"][labels["bare_vs_champion"]]
        )
        search_relation = self._direct_relation(
            report["matchups"][labels["search_vs_champion"]]
        )
        search_gain_relation = self._direct_relation(
            report["matchups"][labels["search_vs_bare"]]
        )
        exact_search_not_degraded = (
            float(search_exact_delta["family_ci95"][1]) >= 0.0
        )
        repetition_noise_allowance = 0.02
        repetition_not_clearly_degraded = (
            float(bare_repetition_delta["family_ci95"][0])
            <= repetition_noise_allowance
            and float(search_repetition_delta["family_ci95"][0])
            <= repetition_noise_allowance
        )
        selfplay_repetition = float(
            training_report["data"]["repetition_fraction"]
        )
        repetition_pathology = (
            selfplay_repetition >= 0.80
            and float(
                training_report["data"].get(
                    "maximum_single_cycle_share_of_repetitions",
                    0.0,
                )
            )
            >= 0.50
        )
        terminal_reasons = training_report["data"][
            "terminal_reasons"
        ]
        games = max(1, int(training_report["data"]["games"]))
        nondecisive_fraction = (
            sum(
                int(terminal_reasons.get(key, 0))
                for key in (
                    "REPETITION",
                    "NO_CAPTURE_DRAW",
                    "MAX_PLY",
                    "MAX_PLY_CUTOFF",
                )
            )
            / games
        )
        terminal_structure_normal = nondecisive_fraction < 0.50
        diversity_normal = (
            training_report.get("passed") is True
            and float(
                training_report["data"]["unique_state_ratio"]
            )
            >= self.config.loop.minimum_unique_state_ratio
        )
        promoted = (
            (
                bare_relation == "significantly_improved"
                or search_relation == "significantly_improved"
            )
            and bare_relation != "clearly_degraded"
            and search_relation != "clearly_degraded"
            and exact_search_not_degraded
            and repetition_not_clearly_degraded
            and search_gain_relation == "significantly_improved"
            and diversity_normal
            and not repetition_pathology
            and terminal_structure_normal
        )
        decision = {
            "schema": "search-v3-championship-decision-v2",
            "status": "COMPLETED",
            "challenger_iteration": challenger_iteration,
            "champion_iteration": champion_iteration,
            "challenger_model_hash": challenger_model_hash,
            "champion_model_hash": champion_model_hash,
            "rules_hash": rules_hash,
            "search_configuration_hash": search_configuration_hash,
            "cache_key": cache_key,
            "cache_key_fields": cache_key_payload,
            "promoted": promoted,
            "bare_vs_champion": bare_relation,
            "search64_vs_champion": search_relation,
            "search64_vs_bare": search_gain_relation,
            "bare_exact082_family_delta": bare_exact_delta,
            "search64_exact082_family_delta": search_exact_delta,
            "bare_exact082_repetition_family_delta": (
                bare_repetition_delta
            ),
            "search64_exact082_repetition_family_delta": (
                search_repetition_delta
            ),
            "repetition_noise_allowance": (
                repetition_noise_allowance
            ),
            "repetition_not_clearly_degraded": (
                repetition_not_clearly_degraded
            ),
            "exact082_primary_metric_not_clearly_degraded": (
                exact_search_not_degraded
            ),
            "diversity_normal": diversity_normal,
            "repetition_pathology": repetition_pathology,
            "terminal_structure_normal": terminal_structure_normal,
            "nondecisive_terminal_fraction": nondecisive_fraction,
            "suite_seed": suite_seed,
            "suite_sha256": seal["states_sha256"],
            "arena_report": report["report_path"],
        }
        self._write_decision(challenge_directory, decision)
        report["championship_decision"] = decision
        atomic_json(
            challenge_directory / "arena" / "report.json",
            report,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(
            cache_path,
            {
                "schema": "search-v3-championship-cache-v1",
                "cache_key": cache_key,
                "cache_key_fields": cache_key_payload,
                "decision": decision,
                "report_path": report["report_path"],
            },
        )
        return decision
