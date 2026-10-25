"""Record the validated inference optimization without rewriting checkpoints."""
from __future__ import annotations

import json
from pathlib import Path

from junqi_open_v4.pipeline import source_contract, sha256
from junqi_search_v3.dataset import atomic_json


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / "runs/open_junqi_v4_repair_pilot"
    old = json.loads((run / "CONTRACT.json").read_text(encoding="utf-8"))
    state = json.loads((run / "CURRENT.json").read_text(encoding="utf-8"))
    if state["contract_hash"] != sha256(run / "CONTRACT.json"):
        raise ValueError("old run contract mismatch")
    new_source = source_contract(root)
    key = str(Path("python/junqi_search_v3/search.py"))
    changed = [name for name in old["source"] if old["source"][name] != new_source.get(name)]
    if changed != [key] or old["source"][key] != sha256(root / "reports/repair_20261003/search_before_cuda_graphs.py"):
        raise ValueError("only the documented inference optimization may migrate")
    if new_source[key] != sha256(root / "reports/repair_20261003/search_after_cuda_graphs.py"):
        raise ValueError("current inference source differs from the validated snapshot")
    baseline = json.loads((root / "reports/repair_20261003/benchmark_baseline.json").read_text(encoding="utf-8"))
    optimized = json.loads((root / "reports/repair_20261003/benchmark_graphs.json").read_text(encoding="utf-8"))
    comparisons = 0
    for first, second in zip(baseline["rows"], optimized["rows"]):
        if first["actions"] != second["actions"] or any(row["cuda_graph_failed_shapes"] for row in second["reports"]):
            raise ValueError("frozen baseline/graph search equivalence did not pass")
        comparisons += first["roots"] * len(first["actions"])
    if comparisons != 1206:
        raise ValueError("incomplete frozen equivalence benchmark")
    for role in ("learner", "actor", "champion"):
        if sha256(run / state[role]["path"]) != state[role]["sha256"]:
            raise ValueError("role checkpoint digest mismatch")
    atomic_json(run / "migrations/cuda_graph_inference.json", {"reason": "same sequential search, CUDA graph inference for batch sizes 1/2/4/8",
        "previous_contract": old, "previous_current": state, "new_source": new_source,
        "equivalent_frozen_roots": comparisons, "checkpoints_modified": False, "optimizer_and_rng_modified": False})
    old["source"] = new_source
    atomic_json(run / "CONTRACT.json", old)
    state["contract_hash"] = sha256(run / "CONTRACT.json")
    atomic_json(run / "CURRENT.json", state)
    print("Migrated inference contract; all checkpoints preserved")


if __name__ == "__main__":
    main()
