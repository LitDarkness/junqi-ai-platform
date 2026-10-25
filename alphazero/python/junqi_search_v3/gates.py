from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any


STAGES = [
    "00_boundary",
    "01_oracle",
    "02_diversity",
    "03_supervised",
    "04_search_gain",
    "05_closed_loop",
]


class StageGate:
    def __init__(self, run_directory: str | Path):
        self.run = Path(run_directory)
        self.root = self.run / "gates"
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, stage: str) -> Path:
        if stage not in STAGES:
            raise KeyError(stage)
        return self.root / f"{stage}.json"

    def require_previous(self, stage: str) -> None:
        index = STAGES.index(stage)
        if index == 0:
            return
        previous = self.read(STAGES[index - 1])
        if not previous or not previous.get("passed", False):
            raise RuntimeError(
                f"stage {stage} is blocked: {STAGES[index-1]} has not passed"
            )

    def read(self, stage: str) -> dict[str, Any] | None:
        path = self.path(stage)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def write(
        self,
        stage: str,
        *,
        passed: bool,
        metrics: dict[str, Any],
        reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "schema": "open-junqi-search-v3-stage-gate",
            "stage": stage,
            "passed": bool(passed),
            "paused": not passed,
            "time": time.time(),
            "metrics": metrics,
            "reasons": list(reasons or []),
        }
        target = self.path(stage)
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, target)
        status = self.run / "STATUS.json"
        status_tmp = status.with_suffix(".json.tmp")
        status_tmp.write_text(
            json.dumps(
                {
                    "schema": "open-junqi-search-v3-status",
                    "current_stage": stage,
                    "status": "passed" if passed else "paused_failed_gate",
                    "next_stage": (
                        STAGES[STAGES.index(stage) + 1]
                        if passed and STAGES.index(stage) + 1 < len(STAGES)
                        else None
                    ),
                    "gate": str(target),
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(status_tmp, status)
        return payload
