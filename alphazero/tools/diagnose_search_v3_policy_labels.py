from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "build"))

from junqi_search_v3.config import SearchV3Config  # noqa: E402
from junqi_search_v3.supervised import (  # noqa: E402
    MemoryOracleDataset,
    soft_target,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(ROOT / "config" / "open_junqi_search_v3.json"),
    )
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--samples", type=int, default=2048)
    args = parser.parse_args()

    config = SearchV3Config.load(Path(args.config))
    run = ROOT / config.run_directory
    dataset = MemoryOracleDataset(
        run / "oracle_dataset" / "training_shards"
    )
    source = dataset.by_split["train"]
    rng = random.Random(config.seed ^ 0x4F564552)
    indices = list(range(len(source)))
    rng.shuffle(indices)
    if len(indices) < args.samples:
        indices = (indices * math.ceil(args.samples / len(indices)))[
            : args.samples
        ]
    else:
        indices = indices[: args.samples]
    samples = [source[index] for index in indices]

    groups: dict[bytes, list[int]] = defaultdict(list)
    targets: list[np.ndarray] = []
    optimum_sets: list[np.ndarray] = []
    gaps: list[float] = []
    tie_sizes: list[int] = []
    entropies: list[float] = []
    for index, sample in enumerate(samples):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(sample.observation).tobytes())
        digest.update(np.ascontiguousarray(sample.legal_actions).tobytes())
        digest.update(
            np.ascontiguousarray(sample.repetition_contexts).tobytes()
        )
        groups[digest.digest()].append(index)

        q_values = sample.q_values.astype(np.float64)
        target = soft_target(q_values, args.tau).astype(np.float64)
        targets.append(target)
        maximum = float(q_values.max())
        optimum = np.flatnonzero(q_values >= maximum - 1e-12)
        optimum_sets.append(optimum)
        tie_sizes.append(int(len(optimum)))
        ordered = np.sort(q_values)
        gaps.append(
            float(ordered[-1] - ordered[-2])
            if len(ordered) > 1
            else float("inf")
        )
        entropies.append(
            float(-np.sum(target * np.log(np.clip(target, 1e-12, 1.0))))
        )

    achievable_correct = 0
    grouped_cross_entropy = 0.0
    per_sample_entropy = float(np.mean(entropies))
    collision_groups = 0
    collision_samples = 0
    for member_indices in groups.values():
        if len(member_indices) > 1:
            collision_groups += 1
            collision_samples += len(member_indices)
        legal = samples[member_indices[0]].legal_actions
        mean_target = np.mean(
            np.stack([targets[index] for index in member_indices]), axis=0
        )
        chosen_local = int(np.argmax(mean_target))
        achievable_correct += sum(
            chosen_local in optimum_sets[index] for index in member_indices
        )
        grouped_cross_entropy += sum(
            float(
                -np.sum(
                    targets[index]
                    * np.log(np.clip(mean_target, 1e-12, 1.0))
                )
            )
            for index in member_indices
        )
        assert all(
            np.array_equal(legal, samples[index].legal_actions)
            for index in member_indices
        )

    gap_array = np.asarray(gaps)
    report = {
        "samples": len(samples),
        "tau": args.tau,
        "unique_network_inputs": len(groups),
        "collision_groups": collision_groups,
        "collision_samples": collision_samples,
        "empirical_top1_achievable": achievable_correct / len(samples),
        "irreducible_excess": (
            grouped_cross_entropy / len(samples) - per_sample_entropy
        ),
        "optimal_tie_fraction": float(np.mean(np.asarray(tie_sizes) > 1)),
        "mean_optimal_set_size": float(np.mean(tie_sizes)),
        "q_top_gap": {
            "zero_fraction": float(np.mean(gap_array == 0)),
            "le_1e-4_fraction": float(np.mean(gap_array <= 1e-4)),
            "le_1e-3_fraction": float(np.mean(gap_array <= 1e-3)),
            "p10": float(np.quantile(gap_array, 0.10)),
            "p25": float(np.quantile(gap_array, 0.25)),
            "median": float(np.quantile(gap_array, 0.50)),
        },
    }
    destination = run / "diagnostics" / "policy_label_achievability.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
