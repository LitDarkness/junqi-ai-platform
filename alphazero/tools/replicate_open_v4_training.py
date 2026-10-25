from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from junqi_open_v4.config import OpenConfig
from junqi_open_v4.learner import Learner, atomic_checkpoint
from junqi_open_v4.pipeline import validation_metrics, source_contract
from junqi_search_v3.dataset import atomic_json
from junqi_search_v3.loop import read_selfplay_shard


def main():
    root = Path(__file__).resolve().parents[1]
    base = OpenConfig.load(root / "config/open_junqi_v4_repair_pilot.json")
    torch.set_num_threads(base.workers)
    paths = [root / base.run_directory / f"replay/iteration_{i:06d}.npz" for i in (1, 2)]
    provenance = [{"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]
    samples = [sample for path in paths for sample in read_selfplay_shard(path)]
    for sample in samples:
        sample.observation = sample.observation.astype("float16")
    training = [s for s in samples if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 != 0]
    validation = [s for s in samples if (s.game_id * 0x9E3779B1 ^ base.seed) % 10 == 0]
    rows = []
    for offset in (0, 1, 2):
        config = copy.deepcopy(base)
        config.seed += offset
        config.run_directory = f"runs/open_junqi_v4_fixed_data_replication/seed_{config.seed}"
        learner = Learner(config, root)
        before = validation_metrics(learner.model, validation, learner.device)
        log = learner.train(training, updates=64)
        after = validation_metrics(learner.model, validation, learner.device)
        path = root / config.run_directory / "learner.pt"
        atomic_checkpoint(path, learner.payload(role="learner", lineage={"fixed_data": provenance,
            "layout_seed": base.seed, "internal_holdout_seed": base.seed, "source": source_contract(root),
            "purpose": "three prespecified training seeds; no model selection or promotion"}))
        row = {"seed": config.seed, "before": before, "after": after, "counters": learner.counters,
               "checkpoint": str(path), "training": log}
        rows.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != "training"}, ensure_ascii=False), flush=True)
        del learner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    atomic_json(root / "reports/repair_20261003/training_replications.json", {
        "purpose": "fixed training replay, three prespecified RNG seeds; internal learning evidence only",
        "replay": provenance, "train_samples": len(training), "validation_samples": len(validation),
        "rows": rows})


if __name__ == "__main__":
    main()
