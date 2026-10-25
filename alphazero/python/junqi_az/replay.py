from __future__ import annotations

import json
import os
import random
import copy
import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import ReplayConfig


@dataclass(frozen=True)
class SparseSample:
    observation: np.ndarray
    legal_actions: np.ndarray
    visit_counts: np.ndarray
    outcome: float
    # A zero policy weight keeps a valid WDL target while excluding an
    # opponent/exploiter action from behaviour learning.
    policy_weight: float = 1.0
    # Reserved optional search value. The no-search fixed-layout route stores NaN.
    search_value: float = math.nan
    # Per-legal-action count of how often the successor position has already
    # occurred in this game's history (0, 1, or 2).  A value of 2 means that
    # playing the action causes the third occurrence and therefore a draw.
    # None keeps old replay shards readable but must only be used by legacy
    # schema runs where the model has repetition_context disabled.
    repetition_contexts: np.ndarray | None = None
    # Stable game-level provenance. New schema runs require non-zero keys so
    # train/validation/test can be split without adjacent-position leakage.
    game_key: int = 0
    ply_index: int = -1

    def validate(self) -> None:
        actions = np.asarray(self.legal_actions)
        visits = np.asarray(self.visit_counts)
        if actions.ndim != 1 or visits.ndim != 1 or len(actions) != len(visits):
            raise ValueError("legal_actions and visit_counts must be equal-length vectors")
        if len(actions) == 0 or np.any(visits < 0) or float(visits.sum()) <= 0:
            raise ValueError("a sparse target needs at least one positive visit")
        if len(np.unique(actions)) != len(actions):
            raise ValueError("legal action indices must be unique")
        if not -1.0 <= float(self.outcome) <= 1.0:
            raise ValueError("outcome must be in [-1, 1]")
        if not math.isfinite(float(self.policy_weight)) or self.policy_weight < 0.0:
            raise ValueError("policy_weight must be finite and non-negative")
        if not math.isnan(float(self.search_value)) and not (
            math.isfinite(float(self.search_value))
            and -1.0 <= float(self.search_value) <= 1.0
        ):
            raise ValueError("search_value must be NaN or finite inside [-1, 1]")
        if self.repetition_contexts is not None:
            contexts = np.asarray(self.repetition_contexts)
            if contexts.ndim != 1 or len(contexts) != len(actions):
                raise ValueError(
                    "repetition_contexts must align with legal_actions"
                )
            if np.any(contexts < 0) or np.any(contexts > 2):
                raise ValueError("repetition_contexts must contain only 0, 1, or 2")
        if int(self.game_key) < 0 or int(self.game_key) > np.iinfo(np.uint64).max:
            raise ValueError("game_key must fit uint64")
        if int(self.ply_index) < -1 or int(self.ply_index) > np.iinfo(np.uint16).max:
            raise ValueError("ply_index must be -1 or fit uint16")


@dataclass
class ReplayBatch:
    observations: np.ndarray
    actions: list[np.ndarray]
    probabilities: list[np.ndarray]
    outcomes: np.ndarray
    policy_weights: np.ndarray
    search_values: np.ndarray
    repetition_contexts: list[np.ndarray]
    game_keys: np.ndarray
    ply_indices: np.ndarray

    def dense_policy(self, action_size: int) -> np.ndarray:
        dense = np.zeros((len(self.actions), action_size), dtype=np.float32)
        for row, (actions, probabilities) in enumerate(zip(self.actions, self.probabilities)):
            dense[row, actions] = probabilities
        return dense

    def action_mask(self, action_size: int) -> np.ndarray:
        mask = np.zeros((len(self.actions), action_size), dtype=np.bool_)
        for row, actions in enumerate(self.actions):
            mask[row, actions] = True
        return mask

    def dense_repetition_context(self, action_size: int) -> np.ndarray:
        dense = np.zeros((len(self.actions), action_size), dtype=np.uint8)
        for row, (actions, contexts) in enumerate(
            zip(self.actions, self.repetition_contexts)
        ):
            if len(actions) != len(contexts):
                raise ValueError("repetition contexts are not aligned with actions")
            dense[row, actions] = contexts
        return dense


class ShardedReplayBuffer:
    """Crash-safe, bounded replay using immutable NPZ shards.

    Targets stay sparse on disk (action id + visit count) and are densified only
    for the sampled training mini-batch. Old complete shards are removed when the
    configured capacity is exceeded.
    """

    INDEX_NAME = "index.json"

    def __init__(self, config: ReplayConfig, seed: int = 1337):
        self.config = config
        self.directory = Path(config.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._rng = random.Random(seed)
        self._pending: list[SparseSample] = []
        self._cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        # append/flush/sample are used by actor and learner threads in the
        # pipelined runner. RLock is intentional: append and sample may flush.
        self._lock = threading.RLock()
        self._index = self._read_index()
        self._transaction_base: dict | None = None
        self._deferred_delete: set[str] = set()

    def _read_index(self) -> dict:
        path = self.directory / self.INDEX_NAME
        if not path.exists():
            return {"schema_version": 1, "next_id": 0, "total": 0, "shards": []}
        data = json.loads(path.read_text(encoding="utf-8"))
        for shard in data.get("shards", []):
            if not (self.directory / shard["file"]).is_file():
                raise RuntimeError(f"replay index references missing shard: {shard['file']}")
        return data

    def __len__(self) -> int:
        with self._lock:
            return int(self._index["total"]) + len(self._pending)

    def append(self, sample: SparseSample) -> None:
        with self._lock:
            sample.validate()
            if self.config.require_game_provenance and (
                int(sample.game_key) == 0 or int(sample.ply_index) < 0
            ):
                raise ValueError("this replay requires game-level provenance")
            if (
                self.config.require_repetition_context
                and sample.repetition_contexts is None
            ):
                raise ValueError("this replay requires repetition contexts")
            self._pending.append(sample)
            if len(self._pending) >= self.config.shard_size:
                self.flush()

    def extend(self, samples: Iterable[SparseSample]) -> None:
        with self._lock:
            for sample in samples:
                self.append(sample)

    def flush(self) -> None:
        with self._lock:
            if not self._pending:
                return
            source_observations = np.stack(
                [np.asarray(s.observation, dtype=np.float32) for s in self._pending]
            )
            if self.config.observation_dtype == "uint8":
                observations = np.rint(np.clip(source_observations, 0.0, 1.0) * 255.0).astype(np.uint8)
            else:
                observations = source_observations.astype(self.config.observation_dtype)
            offsets = np.zeros(len(self._pending) + 1, dtype=np.int64)
            action_parts: list[np.ndarray] = []
            visit_dtype = np.uint16
            if any(np.asarray(sample.visit_counts).max(initial=0) > np.iinfo(np.uint16).max for sample in self._pending):
                visit_dtype = np.uint32
            visit_parts: list[np.ndarray] = []
            repetition_parts: list[np.ndarray] = []
            outcomes = np.empty(len(self._pending), dtype=np.float32)
            policy_weights = np.empty(len(self._pending), dtype=np.float32)
            search_values = np.empty(len(self._pending), dtype=np.float32)
            game_keys = np.empty(len(self._pending), dtype=np.uint64)
            ply_indices = np.empty(len(self._pending), dtype=np.uint16)
            for index, sample in enumerate(self._pending):
                action_parts.append(np.asarray(sample.legal_actions, dtype=np.int32))
                visit_parts.append(np.asarray(sample.visit_counts, dtype=visit_dtype))
                repetition_parts.append(
                    np.zeros(len(action_parts[-1]), dtype=np.uint8)
                    if sample.repetition_contexts is None
                    else np.asarray(sample.repetition_contexts, dtype=np.uint8)
                )
                offsets[index + 1] = offsets[index] + len(action_parts[-1])
                outcomes[index] = sample.outcome
                policy_weights[index] = sample.policy_weight
                search_values[index] = sample.search_value
                game_keys[index] = sample.game_key
                ply_indices[index] = max(0, sample.ply_index)
            shard_id = int(self._index["next_id"])
            filename = f"shard_{shard_id:08d}.npz"
            target = self.directory / filename
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    observations=observations,
                    offsets=offsets,
                    actions=np.concatenate(action_parts),
                    visits=np.concatenate(visit_parts),
                    outcomes=outcomes,
                    policy_weights=policy_weights,
                    search_values=search_values,
                    repetition_contexts=np.concatenate(repetition_parts),
                    game_keys=game_keys,
                    ply_indices=ply_indices,
                )
            os.replace(temporary, target)
            count = len(self._pending)
            self._index["next_id"] = shard_id + 1
            self._index["total"] += count
            self._index["shards"].append({"file": filename, "count": count})
            self._pending.clear()
            obsolete = self._trim_capacity()
            # Commit the new index before deleting files it no longer references.
            # A crash can then leave harmless orphans, never a broken live index.
            self._write_index()
            for filename in obsolete:
                self._cache.pop(filename, None)
                if self._transaction_base is not None:
                    self._deferred_delete.add(filename)
                else:
                    (self.directory / filename).unlink(missing_ok=True)

    def _trim_capacity(self) -> list[str]:
        obsolete: list[str] = []
        while (
            self._index["total"] > self.config.capacity
            and len(self._index["shards"]) > 1
        ):
            oldest = self._index["shards"].pop(0)
            self._index["total"] -= int(oldest["count"])
            obsolete.append(oldest["file"])
        return obsolete

    def trim_to_capacity(self) -> int:
        """Atomically enforce a newly lowered capacity and return files removed."""
        with self._lock:
            self.flush()
            obsolete = self._trim_capacity()
            if not obsolete:
                return 0
            self._write_index()
            for filename in obsolete:
                self._cache.pop(filename, None)
                if self._transaction_base is not None:
                    self._deferred_delete.add(filename)
                else:
                    (self.directory / filename).unlink(missing_ok=True)
            return len(obsolete)

    def _write_index(self) -> None:
        target = self.directory / self.INDEX_NAME
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(self._index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)

    def _load(self, filename: str) -> dict[str, np.ndarray]:
        cached = self._cache.pop(filename, None)
        if cached is not None:
            self._cache[filename] = cached
            return cached
        with np.load(self.directory / filename, allow_pickle=False) as archive:
            data = {key: archive[key] for key in archive.files}
        self._cache[filename] = data
        while len(self._cache) > self.config.cache_shards:
            self._cache.popitem(last=False)
        return data

    def sample(
        self,
        batch_size: int,
        *,
        recent_fraction: float | None = None,
        recent_window_shards: int | None = None,
        split: str = "train",
        minimum_policy_weight: float | None = None,
    ) -> ReplayBatch:
        with self._lock:
            if split not in {"train", "validation", "all"}:
                raise ValueError("split must be train, validation, or all")
            if split == "validation" and self.config.validation_fraction <= 0:
                raise ValueError("validation split is disabled")
            self.flush()
            total = int(self._index["total"])
            if total < batch_size:
                raise ValueError(f"replay has {total} samples, needs {batch_size}")
            shards = self._index["shards"]
            shard_budget = min(
                len(shards), max(1, int(self.config.sample_shards_per_batch))
            )
            active_recent_fraction = (
                self.config.recent_fraction
                if recent_fraction is None
                else float(recent_fraction)
            )
            active_recent_window = (
                self.config.recent_window_shards
                if recent_window_shards is None
                else int(recent_window_shards)
            )
            if not 0.0 <= active_recent_fraction <= 1.0:
                raise ValueError("recent_fraction override must be in [0, 1]")
            if active_recent_window < 1:
                raise ValueError("recent_window_shards override must be positive")
            recent_pool_start = max(0, len(shards) - active_recent_window)
            recent_indices = list(range(recent_pool_start, len(shards)))
            recent_budget = min(
                len(recent_indices),
                shard_budget,
                int(round(shard_budget * active_recent_fraction)),
            )
            selected_indices = self._rng.sample(recent_indices, recent_budget)
            remaining_pool = [
                index for index in range(len(shards)) if index not in set(selected_indices)
            ]
            selected_indices.extend(
                self._rng.sample(remaining_pool, shard_budget - len(selected_indices))
            )
            selected_shards = [shards[index] for index in selected_indices]
            loaded = {shard["file"]: self._load(shard["file"]) for shard in selected_shards}
            weights = [int(shard["count"]) for shard in selected_shards]
            rows: list[
                tuple[
                    np.ndarray,
                    np.ndarray,
                    np.ndarray,
                    float,
                    float,
                    float,
                    np.ndarray,
                    int,
                    int,
                ]
            ] = []
            attempts = 0
            maximum_attempts = max(batch_size * 100, 1000)
            while len(rows) < batch_size:
                attempts += 1
                if attempts > maximum_attempts:
                    raise ValueError(
                        f"could not sample {batch_size} rows from the {split} split"
                    )
                shard = self._rng.choices(selected_shards, weights=weights, k=1)[0]
                data = loaded[shard["file"]]
                local = self._rng.randrange(int(shard["count"]))
                lo, hi = int(data["offsets"][local]), int(data["offsets"][local + 1])
                actions = data["actions"][lo:hi].astype(np.int64, copy=False)
                visits = data["visits"][lo:hi].astype(np.float32, copy=False)
                # Shards written before schema 2 did not contain this field.
                policy_weight = (
                    float(data["policy_weights"][local])
                    if "policy_weights" in data
                    else 1.0
                )
                if (
                    minimum_policy_weight is not None
                    and policy_weight < minimum_policy_weight
                ):
                    continue
                search_value = (
                    float(data["search_values"][local])
                    if "search_values" in data
                    else math.nan
                )
                repetition_contexts = (
                    data["repetition_contexts"][lo:hi].astype(np.uint8, copy=False)
                    if "repetition_contexts" in data
                    else np.zeros(hi - lo, dtype=np.uint8)
                )
                game_key = (
                    int(data["game_keys"][local]) if "game_keys" in data else 0
                )
                if self.config.validation_fraction > 0:
                    if game_key == 0:
                        raise RuntimeError(
                            "game-level split requires non-zero replay game keys"
                        )
                    mixed_key = (
                        game_key ^ int(self.config.split_seed)
                    ) & np.iinfo(np.uint64).max
                    is_validation = (
                        mixed_key % 1_000_000
                        < int(self.config.validation_fraction * 1_000_000)
                    )
                    if split == "train" and is_validation:
                        continue
                    if split == "validation" and not is_validation:
                        continue
                ply_index = (
                    int(data["ply_indices"][local])
                    if "ply_indices" in data
                    else 0
                )
                rows.append(
                    (
                        data["observations"][local],
                        actions,
                        visits / visits.sum(),
                        data["outcomes"][local],
                        policy_weight,
                        search_value,
                        repetition_contexts,
                        game_key,
                        ply_index,
                    )
                )
            observations = np.stack([row[0] for row in rows]).astype(np.float32)
            if self.config.observation_dtype == "uint8":
                observations /= 255.0
            return ReplayBatch(
                observations=observations,
                actions=[row[1] for row in rows],
                probabilities=[row[2] for row in rows],
                outcomes=np.asarray([row[3] for row in rows], dtype=np.float32),
                policy_weights=np.asarray([row[4] for row in rows], dtype=np.float32),
                search_values=np.asarray([row[5] for row in rows], dtype=np.float32),
                repetition_contexts=[row[6] for row in rows],
                game_keys=np.asarray([row[7] for row in rows], dtype=np.uint64),
                ply_indices=np.asarray([row[8] for row in rows], dtype=np.uint16),
            )

    def state_dict(self) -> dict:
        with self._lock:
            self.flush()
            return {
                "rng": self._rng.getstate(),
                "index": copy.deepcopy(self._index),
            }

    def provenance_summary(self) -> dict[str, int]:
        """Return exact persisted game count/max key without loading observations."""
        with self._lock:
            self.flush()
            keys: set[int] = set()
            maximum = 0
            for shard in self._index["shards"]:
                data = self._load(shard["file"])
                if "game_keys" not in data:
                    continue
                for value in np.asarray(data["game_keys"], dtype=np.uint64):
                    key = int(value)
                    if key:
                        keys.add(key)
                        maximum = max(maximum, key)
            return {
                "samples": int(self._index["total"]),
                "games": len(keys),
                "max_game_key": maximum,
            }

    @staticmethod
    def _is_verified_append(expected: dict, current: dict) -> bool:
        """Accept an intact append, including capacity-driven prefix trimming.

        A crash can occur after new immutable shards and the bounded index were
        committed but before the next checkpoint.  In that case the current
        index is an ordered suffix of the checkpoint index followed by one or
        more new shards.  Missing/reordered middle shards remain invalid.
        """
        if expected.get("schema_version") != current.get("schema_version"):
            return False
        expected_shards = expected.get("shards", [])
        current_shards = current.get("shards", [])
        if not expected_shards or not current_shards:
            return False
        if int(current.get("next_id", -1)) <= int(expected.get("next_id", -1)):
            return False
        if int(expected.get("total", -1)) != sum(int(s["count"]) for s in expected_shards):
            return False
        if int(current.get("total", -1)) != sum(int(s["count"]) for s in current_shards):
            return False
        if len({s["file"] for s in current_shards}) != len(current_shards):
            return False
        for removed_prefix in range(len(expected_shards)):
            overlap = expected_shards[removed_prefix:]
            if len(current_shards) <= len(overlap):
                continue
            if current_shards[: len(overlap)] != overlap:
                continue
            appended = len(current_shards) - len(overlap)
            return int(current["next_id"]) == int(expected["next_id"]) + appended
        return False

    def load_state_dict(self, state: dict, allow_appended: bool = False) -> bool:
        """Restore sampling RNG; optionally retain verified post-checkpoint shards.

        Returns true when a strict append was recovered.  Any rewrite, missing
        shard, reordered shard, or non-growing index remains a hard failure.
        """
        with self._lock:
            self.flush()
            expected = state.get("index")
            if expected != self._index:
                if not allow_appended or not self._is_verified_append(expected, self._index):
                    raise RuntimeError(
                        "replay index does not match checkpoint; refusing an inexact resume"
                    )
                recovered = True
            else:
                recovered = False
            self._rng.setstate(state["rng"])
            return recovered

    def begin_transaction(self) -> dict:
        """Keep trimmed files until the enclosing iteration is committed."""
        with self._lock:
            if self._transaction_base is not None:
                raise RuntimeError("replay transaction is already active")
            base = self.state_dict()
            self._transaction_base = copy.deepcopy(base)
            self._deferred_delete.clear()
            return copy.deepcopy(base)

    def commit_transaction(self) -> None:
        """Finalize an iteration after its completed checkpoint is durable."""
        with self._lock:
            if self._transaction_base is None:
                return
            live = {row["file"] for row in self._index["shards"]}
            for filename in sorted(self._deferred_delete):
                if filename not in live:
                    self._cache.pop(filename, None)
                    (self.directory / filename).unlink(missing_ok=True)
            self._transaction_base = None
            self._deferred_delete.clear()

    def rollback_to_state(self, state: dict) -> dict[str, int]:
        """Restore a checkpoint index and delete every post-checkpoint shard.

        Capacity trimming may have removed a prefix from the checkpoint index.
        During new transactional runs those files are deferred and therefore
        recoverable. For legacy interrupted runs, a missing *prefix* is
        tolerated and reported; a missing middle/suffix remains corruption.
        """
        with self._lock:
            self._pending.clear()
            expected = copy.deepcopy(state["index"])
            expected_shards = list(expected.get("shards", []))
            exists = [
                (self.directory / row["file"]).is_file()
                for row in expected_shards
            ]
            first_existing = next(
                (index for index, present in enumerate(exists) if present),
                len(exists),
            )
            if any(not present for present in exists[first_existing:]):
                raise RuntimeError(
                    "cannot rollback replay: checkpoint has a missing non-prefix shard"
                )
            if expected_shards and first_existing == len(expected_shards):
                raise RuntimeError(
                    "cannot rollback replay: every checkpoint shard is missing"
                )
            kept = expected_shards[first_existing:]
            kept_names = {row["file"] for row in kept}
            removed_new = 0
            for path in self.directory.glob("shard_*.npz"):
                if path.name not in kept_names:
                    path.unlink(missing_ok=True)
                    removed_new += 1
            lost_prefix_samples = sum(
                int(row["count"]) for row in expected_shards[:first_existing]
            )
            self._index = {
                "schema_version": expected.get("schema_version", 1),
                "next_id": int(expected["next_id"]),
                "total": sum(int(row["count"]) for row in kept),
                "shards": kept,
            }
            self._cache.clear()
            self._rng.setstate(state["rng"])
            self._transaction_base = None
            self._deferred_delete.clear()
            self._write_index()
            return {
                "removed_post_checkpoint_shards": removed_new,
                "lost_trimmed_prefix_shards": first_existing,
                "lost_trimmed_prefix_samples": lost_prefix_samples,
                "restored_samples": int(self._index["total"]),
            }

    def rollback_transaction(self) -> dict[str, int]:
        with self._lock:
            if self._transaction_base is None:
                return {
                    "removed_post_checkpoint_shards": 0,
                    "lost_trimmed_prefix_shards": 0,
                    "lost_trimmed_prefix_samples": 0,
                    "restored_samples": int(self._index["total"]),
                }
            base = copy.deepcopy(self._transaction_base)
        return self.rollback_to_state(base)

    def cleanup_orphans(self) -> int:
        """Delete immutable shard files not referenced by the durable index."""
        with self._lock:
            live = {row["file"] for row in self._index["shards"]}
            removed = 0
            for path in self.directory.glob("shard_*.npz"):
                if path.name not in live:
                    self._cache.pop(path.name, None)
                    path.unlink(missing_ok=True)
                    removed += 1
            return removed

    def transaction_status(self) -> dict[str, int]:
        with self._lock:
            return {
                "persisted_samples": int(self._index["total"]),
                "pending_samples": len(self._pending),
                "next_shard_id": int(self._index["next_id"]),
                "shards": len(self._index["shards"]),
            }
