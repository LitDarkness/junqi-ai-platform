from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from junqi_open_v4.arena import select_families
from junqi_open_v4.config import OpenConfig
from junqi_open_v4.pipeline import promotion_decision, source_contract


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/repair_20261003"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def equivalence(first, second):
    count = 0
    for a, b in zip(first["rows"], second["rows"]):
        assert (a["root_type"], a["roots"]) == (b["root_type"], b["roots"])
        assert a["actions"] == b["actions"], "root actions or complete visits changed"
        count += sum(len(results) for results in a["actions"])
    assert count == 1206
    return count



def main():
    config = OpenConfig.load(ROOT / "config/open_junqi_v4_repair_pilot.json")
    run = ROOT / config.run_directory
    current, contract = read(run / "CURRENT.json"), read(run / "CONTRACT.json")
    assert contract["source"] == source_contract(ROOT)
    assert current["contract_hash"] == digest(run / "CONTRACT.json")
    for role in ("learner", "actor", "champion"):
        assert current[role]["sha256"] == digest(run / current[role]["path"])
    baseline_manifest = read(OUT / "baseline_manifest.json")
    changed, unchanged_cpp = [], []
    for row in baseline_manifest["files"]:
        path = ROOT / row["path"]
        if digest(path) != row["sha256"]:
            changed.append(row["path"])
        elif Path(row["path"]).parts[0] == "cpp":
            unchanged_cpp.append(row["path"])
    cpp_changed = {Path(path).name for path in changed if Path(path).parts[0] == "cpp"}
    assert cpp_changed == {"neural_puct.cpp", "neural_puct.h", "junqi_az_bindings.cpp"}

    old, graphs = read(OUT / "benchmark_baseline.json"), read(OUT / "benchmark_graphs.json")
    repeat_old, repeat_graphs = read(OUT / "benchmark_baseline_reproduced.json"), read(OUT / "benchmark_graphs_reproduced.json")
    checks = {"original_root_comparisons_exact": equivalence(old, graphs),
              "reproduced_root_comparisons_exact": equivalence(repeat_old, repeat_graphs),
              "source_checkpoint_contract": "PASS", "cpp_rule_files_unchanged": unchanged_cpp}
    trained, search, teachers = (read(OUT / name) for name in ("trained_arena.json", "search_arena.json", "teacher_arena.json"))
    teacher_pairs = defaultdict(list)
    for game in teachers["games"]:
        teacher_pairs[game["position"]].append(game)
    teacher_behavior = dict(paired_starts=len(teacher_pairs),
        identical_complete_trajectories_when_exchanging_teacher_roles=sum(
            len(rows) == 2 and rows[0]["trace"] == rows[1]["trace"] and rows[0]["reason"] == rows[1]["reason"]
            for rows in teacher_pairs.values()), scope="only this frozen development suite; not a proof of identical engines")
    selfplay, reps = read(OUT / "selfplay_benchmark.json"), read(OUT / "training_replications.json")
    rounds = [read(run / f"reports/iteration_{i:06d}.json") for i in (1, 2)]
    gate = promotion_decision(trained, config)
    summary = dict(
        scope="two-player open-information Junqi, repair validation; no final-corpus model selection",
        current=current, verification=checks, promotion=gate,
        strength=trained["summaries"], strength_paired_deltas=trained["paired_deltas"],
        search=search["summaries"], teacher=teachers["summaries"],
        teacher_behavior=teacher_behavior,
        development_games=sum(len(x["games"]) for x in (trained, search, teachers)),
        learner_effective_updates=rounds[-1]["counters"]["effective_updates"],
        neural_selfplay_speedup=selfplay["speedup"], selfplay_trajectories_exact=selfplay["trajectory_equivalence"],
        training_replicas=[{k: row[k] for k in ("seed", "before", "after", "counters", "checkpoint")} for row in reps["rows"]],
    )
    write(OUT / "RESULTS.json", summary)

    suite = select_families(read(run / "development/suite.json")["states"], 32, config.seed ^ 0xA2E)
    failures = sorted((g for g in trained["games"] if g["candidate"] == "candidate" and g["opponent"] == "teacher0" and g["score"] == 0),
                      key=lambda g: (len(g["trace"]), g["family"], g["id"]))[:12]
    write(OUT / "development_failure_cases.json", dict(
        role="dev", purpose="replay and diagnosis only; never add these development states to training replay",
        suite_digest=trained["suite_digest"], checkpoint=trained["checkpoints"]["candidate"],
        cases=[dict(start=suite[g["position"]], game=g) for g in failures],
    ))

    paths = []
    for folder in ("python", "cpp", "tests", "tools", "config"):
        paths.extend(p for p in (ROOT/folder).rglob("*") if p.is_file() and p.suffix in (".py", ".cpp", ".h", ".json"))
    paths.extend([ROOT/"run_open_v4.ps1", ROOT/"README.md", ROOT/"docs/OPEN_JUNQI_V4_zh-CN.md"])
    write(OUT / "final_source_manifest.json", dict(changed_baseline_files=changed, verification=checks,
          current_contract_hash=current["contract_hash"],
          files=[dict(path=str(p.relative_to(ROOT)), sha256=digest(p)) for p in sorted(set(paths))]))
    print(json.dumps({"development_games": summary["development_games"], "checks": checks,
                      "promotion": gate, "selfplay_speedup": selfplay["speedup"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
