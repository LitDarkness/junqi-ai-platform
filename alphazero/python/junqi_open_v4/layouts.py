from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

from junqi_az.layout_policy import CAMPS, HQS, INVENTORY, layout_hash, mirror_layout, validate_layout
from junqi_fixed_rl.layout import FixedLayout, FixedLayoutPairBackend
from junqi_search_v3.dataset import atomic_json


def random_layout(seed: int) -> tuple[int, ...]:
    rng = random.Random(seed)
    layout = [0] * 30
    free = sorted(set(range(30)) - CAMPS)
    flag = rng.choice(sorted(HQS))
    layout[flag] = 3
    free.remove(flag)
    for slot in rng.sample([slot for slot in free if slot < 10], 3):
        layout[slot] = 1
        free.remove(slot)
    for slot in rng.sample([slot for slot in free if slot < 25], 2):
        layout[slot] = 2
        free.remove(slot)
    pieces = [piece for piece, count in INVENTORY.items() if piece not in (1, 2, 3) for _ in range(count)]
    rng.shuffle(pieces)
    for slot, piece in zip(free, pieces):
        layout[slot] = piece
    result = tuple(layout)
    if validate_layout(result):
        raise RuntimeError("generated illegal layout")
    return result


def family_hash(layout: tuple[int, ...]) -> str:
    return min(layout_hash(layout), layout_hash(mirror_layout(layout)))


def create_manifest(directory: Path, seed: int, *, train: int = 64, dev: int = 64, final: int = 128) -> dict[str, Any]:
    directory = directory.resolve()
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing["seed"] != seed or existing["counts"] != dict(train=train, dev=dev, final=final):
            raise ValueError("layout manifest already exists with a different contract")
        return existing
    if min(train, dev, final) < 2:
        raise ValueError("each layout split needs at least two families")
    seen, files = set(), {}
    counter = 0
    for role, count in (("train", train), ("dev", dev), ("final", final)):
        rows = []
        while len(rows) < count:
            layout = random_layout(seed + counter)
            counter += 1
            family = family_hash(layout)
            if family in seen:
                continue
            seen.add(family)
            rows.append({"id": f"{role}_{len(rows):04d}", "family": family,
                         "layout_seed": seed + counter - 1, "setup": list(layout)})
        path = directory / f"{role}.json"
        atomic_json(path, {"schema": "junqi-layout-families-v4", "role": role, "rows": rows})
        files[role] = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest = {"schema": "junqi-layout-manifest-v4", "seed": seed,
                "counts": dict(train=train, dev=dev, final=final), "files": files,
                "family_unit": "layout and its horizontal reflection; disjoint across roles"}
    atomic_json(manifest_path, manifest)
    return manifest


def load_role(directory: Path, role: str, *, final_evaluation: bool = False) -> list[dict[str, Any]]:
    if role not in ("train", "dev", "final") or (role == "final" and not final_evaluation):
        raise ValueError("final layouts are forbidden for training, tuning, and promotion")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    entry = manifest["files"][role]
    path = directory / entry["file"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("immutable layout file hash mismatch")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["role"] != role:
        raise ValueError("layout role mismatch")
    rows = payload["rows"]
    if any(validate_layout(row["setup"]) or row["family"] != family_hash(tuple(row["setup"])) for row in rows):
        raise ValueError("layout legality or family hash mismatch")
    return rows


def backend_for(red: dict[str, Any], black: dict[str, Any], *, mirrored: bool = False) -> FixedLayoutPairBackend:
    def convert(row: dict[str, Any]) -> FixedLayout:
        setup = tuple(row["setup"])
        if mirrored:
            setup = mirror_layout(setup)
        return FixedLayout(row["id"], row["id"], setup, row["family"])
    return FixedLayoutPairBackend(convert(red), convert(black))


class MultiLayoutBackend:
    def __init__(self, rows: list[dict[str, Any]], seed: int):
        if len(rows) < 2:
            raise ValueError("multi-layout backend needs at least two families")
        self.rows, self.seed = rows, seed

    def new_state(self, seed: int | None = None):
        rng = random.Random(int(seed or 0) ^ self.seed)
        red, black = rng.sample(self.rows, 2)
        return backend_for(red, black, mirrored=bool(rng.randrange(2))).new_state(seed)
