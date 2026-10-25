from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass
class OracleSample:
    family_id: int
    split: str
    state_hash: int
    ply: int
    player: int
    opponent: str
    terminal_reason: str
    observation: np.ndarray
    legal_actions: np.ndarray
    repetition_contexts: np.ndarray
    q_values: np.ndarray
    wdl: np.ndarray
    important: bool
    action_trace: list[int]


def family_split(seed: int, family_id: int) -> str:
    # Split whole trajectory families.  Integer mixing is stable across Python
    # versions, unlike hash(str).
    mixed = (
        (int(seed) ^ (int(family_id) * 0x9E3779B97F4A7C15))
        * 0xBF58476D1CE4E5B9
    ) & ((1 << 64) - 1)
    bucket = (mixed >> 32) % 100
    return "train" if bucket < 80 else ("validation" if bucket < 90 else "test")


def write_shard(path: Path, samples: list[OracleSample]) -> None:
    if not samples:
        raise ValueError("cannot write an empty Oracle shard")
    path.parent.mkdir(parents=True, exist_ok=True)
    legal_offsets = [0]
    trace_offsets = [0]
    for sample in samples:
        legal_offsets.append(legal_offsets[-1] + len(sample.legal_actions))
        trace_offsets.append(trace_offsets[-1] + len(sample.action_trace))
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            family_id=np.asarray([s.family_id for s in samples], np.int32),
            split=np.asarray([s.split for s in samples], "U10"),
            state_hash=np.asarray([s.state_hash for s in samples], np.uint64),
            ply=np.asarray([s.ply for s in samples], np.int16),
            player=np.asarray([s.player for s in samples], np.int8),
            opponent=np.asarray([s.opponent for s in samples], "U32"),
            terminal_reason=np.asarray(
                [s.terminal_reason for s in samples], "U32"
            ),
            observations=np.stack(
                [s.observation.astype(np.float16, copy=False) for s in samples]
            ),
            legal_offsets=np.asarray(legal_offsets, np.int64),
            legal_actions=np.concatenate(
                [s.legal_actions.astype(np.int16, copy=False) for s in samples]
            ),
            repetition_contexts=np.concatenate(
                [
                    s.repetition_contexts.astype(np.uint8, copy=False)
                    for s in samples
                ]
            ),
            q_values=np.concatenate(
                [s.q_values.astype(np.float16, copy=False) for s in samples]
            ),
            wdl=np.stack([s.wdl.astype(np.float16, copy=False) for s in samples]),
            important=np.asarray([s.important for s in samples], np.bool_),
            trace_offsets=np.asarray(trace_offsets, np.int64),
            action_trace=np.asarray(
                [a for sample in samples for a in sample.action_trace],
                np.int16,
            ),
        )
    os.replace(temporary, path)


def iter_shard(path: Path) -> Iterable[dict[str, Any]]:
    with np.load(path, allow_pickle=False) as data:
        # NpzFile.__getitem__ decompresses a member on every access.  Cache
        # each member once before iterating; indexing data["observations"] in
        # the loop turns shard loading into accidental O(rows^2) decompression.
        arrays = {name: data[name] for name in data.files}
        for index in range(len(arrays["family_id"])):
            lo = int(arrays["legal_offsets"][index])
            hi = int(arrays["legal_offsets"][index + 1])
            tlo = int(arrays["trace_offsets"][index])
            thi = int(arrays["trace_offsets"][index + 1])
            yield {
                "family_id": int(arrays["family_id"][index]),
                "split": str(arrays["split"][index]),
                "state_hash": int(arrays["state_hash"][index]),
                "ply": int(arrays["ply"][index]),
                "player": int(arrays["player"][index]),
                "opponent": str(arrays["opponent"][index]),
                "terminal_reason": str(arrays["terminal_reason"][index]),
                "observation": arrays["observations"][index].astype(np.float32),
                "legal_actions": arrays["legal_actions"][lo:hi].astype(np.int64),
                "repetition_contexts": arrays["repetition_contexts"][lo:hi]
                .astype(np.uint8),
                "q_values": arrays["q_values"][lo:hi].astype(np.float32),
                "wdl": arrays["wdl"][index].astype(np.float32),
                "important": bool(arrays["important"][index]),
                "action_trace": arrays["action_trace"][tlo:thi]
                .astype(np.int64)
                .tolist(),
            }


def shard_state_hashes(path: Path) -> np.ndarray:
    """Read only the compact state-hash member for transaction recovery."""
    with np.load(path, allow_pickle=False) as data:
        return data["state_hash"].copy()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
