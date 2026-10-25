from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from junqi_search_v3.retention import CheckpointRetention
from junqi_search_v3.training_retention import TrainingDataRetention


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _iteration(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _latest_completed(run: Path) -> int:
    iterations = []
    for checkpoint in (run / "checkpoints").glob(
        "loop_iteration_*.pt"
    ):
        iteration = _iteration(checkpoint)
        report = run / "reports" / f"iteration_{iteration:06d}.json"
        if report.exists():
            payload = json.loads(report.read_text(encoding="utf-8"))
            if payload.get("passed") is True:
                iterations.append(iteration)
    if not iterations:
        raise RuntimeError("no completed strength-continuation checkpoint")
    return max(iterations)


def plan(run: Path) -> dict[str, Any]:
    run = run.resolve()
    latest = _latest_completed(run)
    rows: dict[Path, str] = {}
    # Any newer replay is an incomplete transaction.  It must be regenerated
    # because the cycle-aware target/replay schema starts after `latest`.
    for directory in (run / "search_replay").glob("iteration_*"):
        if _iteration(directory) > latest:
            for path in directory.rglob("*"):
                if path.is_file():
                    rows[path.resolve()] = (
                        "incomplete pre-cycle-aware replay transaction"
                    )
    # Passed candidates are superseded by their complete atomic checkpoint.
    for candidate in (run / "candidates").glob("iteration_*.pt"):
        iteration = _iteration(candidate)
        official = (
            run
            / "checkpoints"
            / f"loop_iteration_{iteration:06d}.pt"
        )
        if official.exists():
            rows[candidate.resolve()] = (
                "accepted candidate superseded by atomic official checkpoint"
            )
    # Continuation-dev tensors are temporary read-only adapters.  JSON/CSV
    # evidence remains; the selected official checkpoint is preserved.
    for candidate in (run / "candidate_dev").glob(
        "iteration_*/candidate.pt"
    ):
        iteration = _iteration(candidate.parent)
        report = run / "reports" / f"iteration_{iteration:06d}.json"
        if iteration <= latest and report.exists():
            rows[candidate.resolve()] = (
                "completed candidate-dev temporary model; reports retained"
            )
    for pattern in ("*.partial", "*.tmp"):
        for path in run.rglob(pattern):
            if path.is_file():
                rows[path.resolve()] = "uncommitted temporary file"
    files = [
        {
            "path": str(path),
            "relative": path.relative_to(run).as_posix(),
            "reason": reason,
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path, reason in sorted(rows.items())
    ]
    return {
        "schema": "search-v3-cycle-aware-cleanup-v1",
        "phase": "dry_run",
        "created_unix": time.time(),
        "run": str(run),
        "latest_completed_iteration": latest,
        "preserved": [
            "all official checkpoints selected by recent5/every5 policy",
            "all frozen champions and championship evidence",
            "bounded competitive checkpoints including requalified 62",
            "all iteration reports and failure evidence",
            "replay recent3/every10/champion after removing incomplete future",
        ],
        "files": files,
        "file_count": len(files),
        "bytes": sum(row["bytes"] for row in files),
    }


def apply(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("phase") != "dry_run":
        raise RuntimeError("cleanup manifest was already applied")
    run = Path(manifest["run"]).resolve()
    deleted = []
    for row in manifest["files"]:
        target = Path(row["path"]).resolve()
        if run not in target.parents:
            raise RuntimeError(f"cleanup escaped run: {target}")
        if not target.exists():
            continue
        stat = target.stat()
        if (
            stat.st_size != int(row["bytes"])
            or stat.st_mtime_ns != int(row["mtime_ns"])
        ):
            raise RuntimeError(f"cleanup target changed: {target}")
        target.unlink()
        deleted.append(row)
    for directory in sorted(
        (item for item in run.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        try:
            if not any(directory.iterdir()):
                directory.rmdir()
        except OSError:
            pass
    latest = int(manifest["latest_completed_iteration"])
    checkpoint = CheckpointRetention(
        run,
        recent=5,
        archive_interval=5,
        starting_incumbent=50,
    ).apply(latest)
    champion_payload = json.loads(
        (run / "championship.json").read_text(encoding="utf-8")
    )
    champions = {int(champion_payload["current"]["iteration"])}
    champions.update(
        int(row["iteration"])
        for row in champion_payload.get("history", [])
    )
    data = TrainingDataRetention(
        run,
        recent_replay=3,
        replay_archive_interval=10,
        recent_console_logs=5,
    ).apply(latest, champion_iterations=champions)
    manifest.update(
        {
            "phase": "completed",
            "completed_unix": time.time(),
            "deleted": deleted,
            "deleted_files": len(deleted),
            "direct_bytes_reclaimed": sum(
                int(row["bytes"]) for row in deleted
            ),
            "checkpoint_retention": checkpoint,
            "data_retention": data,
            "total_bytes_reclaimed": (
                sum(int(row["bytes"]) for row in deleted)
                + int(checkpoint["bytes_reclaimed"])
                + int(data["bytes_reclaimed"])
            ),
        }
    )
    _atomic_json(path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        default="runs/open_junqi_search_v3_strength_continuation",
    )
    parser.add_argument("--manifest")
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    if arguments.apply:
        if not arguments.manifest:
            parser.error("--apply requires --manifest")
        result = apply(Path(arguments.manifest).resolve())
    else:
        run = Path(arguments.run).resolve()
        result = plan(run)
        path = (
            Path(arguments.manifest).resolve()
            if arguments.manifest
            else run
            / "cleanup"
            / f"cycle_aware_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        _atomic_json(path, result)
        result["manifest"] = str(path)
    manifest_output = result.get("manifest")
    if manifest_output is None and arguments.manifest:
        manifest_output = str(Path(arguments.manifest).resolve())
    print(
        json.dumps(
            {
                "phase": result["phase"],
                "latest_completed_iteration": result[
                    "latest_completed_iteration"
                ],
                "files": result.get(
                    "deleted_files", result.get("file_count", 0)
                ),
                "gib": result.get(
                    "total_bytes_reclaimed", result.get("bytes", 0)
                )
                / (1024**3),
                "manifest": manifest_output,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
