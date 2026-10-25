"""Explicit source migration for the global phase-counter correction only."""
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
    changed = [key for key in old["source"] if old["source"][key] != new_source.get(key)]
    expected = str(Path("python/junqi_open_v4/learner.py"))
    if changed != [expected] or set(new_source) != set(old["source"]):
        raise ValueError("this migration permits only the phase-counter learner correction")
    if old["source"][expected] != "e6f24f78a5510d52a5be560b803445f45a313e8aba4ff44f98facf74ca7c2c33":
        raise ValueError("unexpected predecessor source")
    previous = (root / "reports/repair_20261003/learner_before_phase_fix.py").read_text(encoding="utf-8")
    corrected = previous.replace('phase = "value" if index % 3 == 2 else "policy"',
        'phase = "value" if self.counters["attempted_updates"] % 3 == 2 else "policy"')
    if (root / expected).read_text(encoding="utf-8") != corrected:
        raise ValueError("source changes exceed the declared phase correction")
    for role in ("learner", "actor", "champion"):
        if sha256(run / state[role]["path"]) != state[role]["sha256"]:
            raise ValueError("role checkpoint digest mismatch")
    migration = {"reason": "continue Policy, Policy, Value from global attempted_updates across iterations and resumes",
                 "previous_contract": old, "previous_current": state,
                 "changed_source": changed, "new_source": new_source,
                 "checkpoints_modified": False, "optimizer_and_rng_modified": False,
                 "prior_training": "first two pilot rounds reset phase within each round; retained as recorded"}
    atomic_json(run / "migrations/global_phase_counter.json", migration)
    old["source"] = new_source
    atomic_json(run / "CONTRACT.json", old)
    state["contract_hash"] = sha256(run / "CONTRACT.json")
    atomic_json(run / "CURRENT.json", state)
    print("Migrated source contract; every model/optimizer/RNG checkpoint is preserved")


if __name__ == "__main__":
    main()
