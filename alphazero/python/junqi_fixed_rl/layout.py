from __future__ import annotations

import hashlib
import importlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from junqi_az.backend import CppAZStateAdapter
from junqi_az.layout_policy import layout_hash, mirror_layout, validate_layout


@dataclass(frozen=True)
class FixedLayout:
    layout_id: str
    name_zh: str
    setup: tuple[int, ...]
    source_sha256: str


def load_fixed_layout(
    path: str | Path, layout_id: str, *, project_root: Path
) -> FixedLayout:
    source = Path(path)
    if not source.is_absolute():
        source = project_root / source
    raw = source.read_bytes()
    payload = json.loads(raw.decode("utf-8-sig"))
    if payload.get("schema") != "junqi-fixed-layout-set-v1":
        raise ValueError("unsupported fixed layout set")
    matches = [row for row in payload.get("layouts", []) if row.get("id") == layout_id]
    if len(matches) != 1:
        raise KeyError(f"layout id must occur exactly once: {layout_id}")
    row = matches[0]
    setup = tuple(int(piece) for piece in row["canonical_setup30"])
    errors = validate_layout(setup)
    if errors:
        raise ValueError("invalid selected fixed layout: " + "; ".join(errors))
    return FixedLayout(
        layout_id=layout_id,
        name_zh=str(row.get("name_zh", layout_id)),
        setup=setup,
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


class FixedSameLayoutBackend:
    """Create every episode from one canonical layout for both players."""

    def __init__(
        self,
        layout: FixedLayout,
        mirror_probability: float = 0.0,
        module_name: str = "JunqiAZ",
    ) -> None:
        self.layout = layout
        self.mirror_probability = float(mirror_probability)
        module = importlib.import_module(module_name)
        self._state_type = getattr(module, "JunqiAZState", getattr(module, "AZState"))

    def setup_for_seed(self, seed: int) -> tuple[int, ...]:
        if (
            self.mirror_probability > 0.0
            and random.Random(int(seed) ^ 0x4C41594F5554).random()
            < self.mirror_probability
        ):
            return mirror_layout(self.layout.setup)
        return self.layout.setup

    def new_state(self, seed: int | None = None) -> CppAZStateAdapter:
        actual_seed = int(seed or 0)
        setup = self.setup_for_seed(actual_seed)
        raw = self._state_type()
        # BLACK receives the same canonical layout; native row order is the
        # 180-degree reverse of BLACK's own front-to-rear representation.
        raw.reset(list(setup), list(reversed(setup)))
        board = tuple(int(value) for value in raw.board())
        provenance: dict[str, Any] = {
            "layout_mode": "fixed_same_canonical",
            "layout_id": self.layout.layout_id,
            "layout_name_zh": self.layout.name_zh,
            "layout_source_sha256": self.layout.source_sha256,
            "setup_seed": actual_seed,
            "initial_board": list(board),
            "red_layout": list(setup),
            "black_layout": list(setup),
            "red_layout_hash": layout_hash(setup),
            "black_layout_hash": layout_hash(setup),
            "symmetric": True,
        }
        return CppAZStateAdapter(raw, provenance)


class FixedLayoutPairBackend:
    """Create episodes from explicit RED/BLACK canonical layouts."""

    def __init__(
        self,
        red_layout: FixedLayout,
        black_layout: FixedLayout,
        module_name: str = "JunqiAZ",
    ) -> None:
        self.red_layout = red_layout
        self.black_layout = black_layout
        module = importlib.import_module(module_name)
        self._state_type = getattr(module, "JunqiAZState", getattr(module, "AZState"))

    def new_state(self, seed: int | None = None) -> CppAZStateAdapter:
        actual_seed = int(seed or 0)
        red_setup = self.red_layout.setup
        black_setup = self.black_layout.setup
        raw = self._state_type()
        raw.reset(list(red_setup), list(reversed(black_setup)))
        board = tuple(int(value) for value in raw.board())
        provenance: dict[str, Any] = {
            "layout_mode": "fixed_explicit_pair",
            "red_layout_id": self.red_layout.layout_id,
            "black_layout_id": self.black_layout.layout_id,
            "red_layout_name_zh": self.red_layout.name_zh,
            "black_layout_name_zh": self.black_layout.name_zh,
            "setup_seed": actual_seed,
            "initial_board": list(board),
            "red_layout": list(red_setup),
            "black_layout": list(black_setup),
            "red_layout_hash": layout_hash(red_setup),
            "black_layout_hash": layout_hash(black_setup),
            "symmetric": red_setup == black_setup,
        }
        return CppAZStateAdapter(raw, provenance)
