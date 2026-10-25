from __future__ import annotations

import hashlib
from collections import Counter
from typing import Sequence


ROWS = 6
COLS = 5
SQUARES = 30
CAMPS = frozenset({11, 13, 17, 21, 23})
HQS = frozenset({1, 3})
REAR_ROWS = frozenset({0, 1})
FRONT_ROW = 5
MINE, BOMB, FLAG = 1, 2, 3
INVENTORY = {
    1: 3,
    2: 2,
    3: 1,
    10: 3,
    20: 3,
    30: 3,
    40: 2,
    50: 2,
    60: 2,
    70: 2,
    80: 1,
    90: 1,
}


def layout_hash(layout: Sequence[int]) -> str:
    raw = ",".join(str(int(value)) for value in layout).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def validate_layout(layout: Sequence[int]) -> list[str]:
    if len(layout) != SQUARES:
        return [f"layout length must be {SQUARES}, got {len(layout)}"]
    values = tuple(int(value) for value in layout)
    errors: list[str] = []
    if any(values[index] for index in CAMPS):
        errors.append("all five camps must be empty")
    if Counter(value for value in values if value) != Counter(INVENTORY):
        errors.append("piece inventory mismatch")
    flags = [index for index, value in enumerate(values) if value == FLAG]
    if len(flags) != 1 or flags[0] not in HQS:
        errors.append("flag must be in one headquarters")
    for index, value in enumerate(values):
        row = index // COLS
        if value == MINE and row not in REAR_ROWS:
            errors.append("mine outside rear two rows")
        if value == BOMB and row == FRONT_ROW:
            errors.append("bomb on front row")
    return errors


def mirror_layout(layout: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(layout[row * COLS + (COLS - 1 - column)])
        for row in range(ROWS)
        for column in range(COLS)
    )
