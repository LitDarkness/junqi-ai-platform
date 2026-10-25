from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .rule_profile import RuleProfile


@dataclass
class GameRecord:
    rules: RuleProfile
    initial_state: list[int]
    players: list[dict[str, str]]
    seed: int
    moves: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None

    def append(self, *, ply: int, from_pos: int, to_pos: int, key_before: str,
               key_after: str, event: dict[str, Any], analysis: dict[str, Any] | None = None) -> None:
        self.moves.append({
            "ply": ply, "from": from_pos, "to": to_pos,
            "key_before": key_before, "key_after": key_after,
            "event": event, "analysis": analysis,
        })

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "junqi-open-record-v1",
            "rule_profile_id": self.rules.profile_id,
            "rule_profile_sha256": self.rules.sha256,
            "rule_profile": self.rules.data,
            "initial_state": self.initial_state,
            "players": self.players,
            "seed": self.seed,
            "moves": self.moves,
            "result": self.result,
        }

    def save_atomic(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temp = target.with_suffix(target.suffix + ".tmp")
        temp.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, target)
