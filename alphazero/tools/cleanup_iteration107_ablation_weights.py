from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _row(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def cleanup(root: Path, *, execute: bool) -> dict[str, Any]:
    root = root.resolve()
    run = (
        root
        / "runs"
        / "open_junqi_search_v3_strength_continuation"
    ).resolve()
    diagnostics = (run / "diagnostics").resolve()
    if diagnostics.parent != run or not diagnostics.exists():
        raise RuntimeError("unexpected diagnostics root")
    policy = (
        diagnostics
        / "iteration_000107_policy_lr_ablation"
        / "candidates"
    )
    value = (
        diagnostics
        / "iteration_000107_value_step_ablation"
        / "candidates"
    )
    keep_policy = {
        "lr_0p5_step_03.pt",
        "lr_0p25_step_06.pt",
    }
    keep_value = {"lr_0p5_policy_03_value_02.pt"}
    delete = [
        path
        for directory, keep in (
            (policy, keep_policy),
            (value, keep_value),
        )
        for path in directory.glob("*.pt")
        if path.name not in keep
    ]
    kept = [
        path
        for directory, keep in (
            (policy, keep_policy),
            (value, keep_value),
        )
        for path in directory.glob("*.pt")
        if path.name in keep
    ]
    evidence = [
        diagnostics
        / "iteration_000107_policy_lr_ablation"
        / name
        for name in (
            "report.json",
            "summary.csv",
            "fixed_batch_keys.npz",
        )
    ] + [
        diagnostics
        / "iteration_000107_value_step_ablation"
        / "report.json"
    ]
    evidence.extend(
        [
            root / "docs" / "ITERATION_107_POLICY_LR_ABLATION.md",
            root / "docs" / "ITERATION_107_VALUE_STEP_ABLATION.md",
            run / "checkpoints" / "loop_iteration_000107.pt",
            run
            / "checkpoints"
            / "late_stage_bootstrap_pending.pt",
            run
            / "diagnostics"
            / "iteration_000107_late_stage_bootstrap"
            / "internal_gate.json",
        ]
    )
    missing = [str(path) for path in evidence if not path.exists()]
    if missing:
        raise RuntimeError(
            "diagnostic evidence is incomplete: " + ", ".join(missing)
        )
    manifest = {
        "schema": "iteration107-ablation-weight-cleanup-v1",
        "created_unix": time.time(),
        "mode": "execute" if execute else "dry-run",
        "policy": (
            "delete only regenerable model-only ablation weights; retain "
            "reports, fixed batches, selected branch weights, full atomic "
            "iteration107, full reconstructed bootstrap, and all historical "
            "diagnostic/championship evidence"
        ),
        "delete": [_row(path, root) for path in delete],
        "keep_selected_model_only": [
            _row(path, root) for path in kept
        ],
        "preserved_evidence": [
            _row(path, root) for path in evidence
        ],
        "bytes_planned": sum(path.stat().st_size for path in delete),
        "deleted": [],
        "bytes_reclaimed": 0,
    }
    output = run / "cleanup"
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "iteration107_ablation_weights.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, manifest_path)
    if execute:
        deleted = []
        reclaimed = 0
        for path in delete:
            resolved = path.resolve()
            if diagnostics not in resolved.parents:
                raise RuntimeError(
                    f"refusing to delete outside diagnostics: {resolved}"
                )
            size = path.stat().st_size
            path.unlink()
            deleted.append(path.relative_to(root).as_posix())
            reclaimed += size
        manifest["deleted"] = deleted
        manifest["bytes_reclaimed"] = reclaimed
        manifest["phase"] = "completed"
        temporary.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
    )
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    result = cleanup(
        Path(arguments.root), execute=arguments.execute
    )
    print(
        json.dumps(
            {
                "mode": result["mode"],
                "files": len(result["delete"]),
                "planned_mib": round(
                    result["bytes_planned"] / (1024**2), 1
                ),
                "reclaimed_mib": round(
                    result["bytes_reclaimed"] / (1024**2), 1
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
