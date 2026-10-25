from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REQUIRED_TOP_LEVEL = {
    "rule_profile_id", "board_topology", "visibility", "camp",
    "headquarters", "railway", "combat_matrix", "terminal", "setup",
}


@dataclass(frozen=True)
class RuleProfile:
    data: dict[str, Any]
    canonical: bytes
    sha256: str

    @property
    def profile_id(self) -> str:
        return str(self.data["rule_profile_id"])


def load_rule_profile(path: str | Path) -> RuleProfile:
    source = Path(path)
    data = json.loads(source.read_text(encoding="utf-8"))
    unknown = set(data) - REQUIRED_TOP_LEVEL
    missing = REQUIRED_TOP_LEVEL - set(data)
    if unknown or missing:
        raise ValueError(f"invalid RuleProfile fields: missing={sorted(missing)}, unknown={sorted(unknown)}")
    if data["visibility"] != "full":
        raise ValueError("this platform build supports only the SRS full-information profile")
    terminal = data["terminal"]
    for field in ("repetition_count", "no_capture_ply_limit", "max_ply"):
        if not isinstance(terminal.get(field), int) or terminal[field] < 0:
            raise ValueError(f"terminal.{field} must be a non-negative integer")
    canonical = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return RuleProfile(data=data, canonical=canonical, sha256=hashlib.sha256(canonical).hexdigest())


def default_rule_profile() -> RuleProfile:
    root = Path(__file__).resolve().parents[1]
    return load_rule_profile(root / "config" / "rules" / "standard_open_v1.json")
