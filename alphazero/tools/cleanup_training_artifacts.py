from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any


def _within(path: Path, root: Path) -> bool:
    resolved = path.resolve()
    return resolved == root or root in resolved.parents


def _files(root: Path, targets: list[Path]) -> list[Path]:
    result = set()
    for target in targets:
        if not target.exists():
            continue
        if not _within(target, root):
            raise RuntimeError(f"cleanup target escaped workspace: {target}")
        if target.is_file():
            result.add(target.resolve())
        else:
            result.update(
                path.resolve()
                for path in target.rglob("*")
                if path.is_file()
            )
    return sorted(result)


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def plan(root: Path) -> dict[str, Any]:
    runs = root / "runs"
    targets = [
        runs / "test_tmp",
        root / "tests" / "replay_tmp",
        runs
        / "fixed_layout_rl_v2a_advantage_single_layout"
        / "rollouts",
        runs
        / "fixed_layout_rl_v2a_advantage_single_layout"
        / "replay_teacher",
        runs
        / "open_junqi_search_v3_generation_1_asymmetric_value_failure"
        / "checkpoints",
        runs
        / "open_junqi_search_v3_generation_1_asymmetric_value_failure"
        / "oracle_dataset",
        runs
        / "open_junqi_search_v3_strength"
        / "continuation_arena"
        / "bare_vs_exact082",
        runs
        / "open_junqi_search_v3_strength"
        / "continuation_arena"
        / "search64_vs_exact082",
        runs
        / "open_junqi_search_v3_strength"
        / "continuation_arena"
        / "search64_vs_bare",
        runs
        / "open_junqi_search_v3_alternating"
        / "checkpoints"
        / "loop_iteration_000003.pt",
        runs
        / "open_junqi_search_v3_alternating"
        / "candidates"
        / "iteration_000005.pt",
    ]
    targets.extend(root.rglob("__pycache__"))
    targets.extend(root.rglob("*.partial"))
    targets.extend(root.rglob("*.tmp"))
    files = _files(root, targets)
    entries = [
        {
            "path": str(path),
            "relative": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        for path in files
    ]
    return {
        "schema": "search-v3-workspace-cleanup-v1",
        "created_unix": time.time(),
        "root": str(root),
        "phase": "dry_run",
        "preserved": {
            "fixed_layout_rl_v1": (
                "entire outcome-weighted failure baseline"
            ),
            "fixed_layout_rl_v2a": (
                "failure reports, metrics, final/latest/champion and "
                "retention checkpoints; only rollout/replay removed"
            ),
            "fixed_layout_rl_v2b": (
                "entire bootstrap state and final diagnostics"
            ),
            "open_junqi_search_v3": (
                "entire iteration-3 failure and causal recovery evidence"
            ),
            "open_junqi_search_v3_alternating": (
                "frozen iteration 10, recent 6-10, archive 5 and "
                "competitive iteration 4/7/8"
            ),
            "open_junqi_search_v3_strength": (
                "1024-state suite, full continuation arena and reports"
            ),
            "open_junqi_search_v3_scaled_training": (
                "new long-running lineage"
            ),
        },
        "deletion_reason": {
            "v2a_rollouts_replay": (
                "failed lineage is frozen; reports and final models retained"
            ),
            "asymmetric_value_failure_models": (
                "superseded failed-generation model/data; compact reports "
                "and diagnostics retained"
            ),
            "alternating_prune": (
                "apply recent5/every5/championship/competitive policy"
            ),
            "smoke_cache": "unreferenced smoke output and Python caches",
        },
        "files": entries,
        "file_count": len(entries),
        "bytes": sum(row["bytes"] for row in entries),
    }


def apply(manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(
        manifest_path.read_text(encoding="utf-8")
    )
    if (
        manifest.get("schema") != "search-v3-workspace-cleanup-v1"
        or manifest.get("phase") != "dry_run"
    ):
        raise RuntimeError("cleanup manifest is not an unapplied dry run")
    root = Path(manifest["root"]).resolve()
    deleted = []
    already_absent = []
    for row in manifest["files"]:
        path = Path(row["path"]).resolve()
        if not _within(path, root):
            raise RuntimeError(f"cleanup path escaped workspace: {path}")
        if not path.exists():
            already_absent.append(row)
            continue
        stat = path.stat()
        if (
            stat.st_size != int(row["bytes"])
            or stat.st_mtime_ns != int(row["mtime_ns"])
        ):
            raise RuntimeError(
                f"cleanup target changed since dry run: {path}"
            )
        path.unlink()
        deleted.append(row)
    def is_empty(directory: Path) -> bool:
        try:
            return not any(directory.iterdir())
        except (OSError, PermissionError):
            return False

    directories = sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_dir()
            and (
                path.name == "__pycache__"
                or is_empty(path)
            )
        ),
        key=lambda path: len(path.parts),
        reverse=True,
    )
    removed_directories = []
    for directory in directories:
        if _within(directory, root) and is_empty(directory):
            try:
                directory.rmdir()
                removed_directories.append(str(directory))
            except (OSError, PermissionError):
                pass
    manifest["phase"] = "completed"
    manifest["completed_unix"] = time.time()
    manifest["deleted"] = deleted
    manifest["already_absent_after_interrupted_apply"] = already_absent
    manifest["deleted_files"] = len(deleted)
    manifest["deleted_or_previously_applied_files"] = (
        len(deleted) + len(already_absent)
    )
    manifest["deleted_bytes"] = sum(
        int(row["bytes"]) for row in deleted + already_absent
    )
    manifest["removed_directories"] = removed_directories
    _write(manifest_path, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--manifest")
    parser.add_argument("--apply", action="store_true")
    arguments = parser.parse_args()
    root = Path(arguments.root).resolve()
    if arguments.apply:
        if not arguments.manifest:
            parser.error("--apply requires --manifest")
        result = apply(Path(arguments.manifest).resolve())
        print(
            json.dumps(
                {
                    "phase": result["phase"],
                    "deleted_files": result["deleted_files"],
                    "deleted_gib": result["deleted_bytes"] / (1024**3),
                    "manifest": str(
                        Path(arguments.manifest).resolve()
                    ),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    manifest = (
        Path(arguments.manifest).resolve()
        if arguments.manifest
        else root / "cleanup" / f"cleanup_{timestamp}.json"
    )
    result = plan(root)
    _write(manifest, result)
    print(
        json.dumps(
            {
                "phase": result["phase"],
                "files": result["file_count"],
                "gib": result["bytes"] / (1024**3),
                "manifest": str(manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
