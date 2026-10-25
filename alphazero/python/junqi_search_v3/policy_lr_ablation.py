from __future__ import annotations

import copy
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dataset import atomic_json
from .iteration107_audit import (
    AUDIT_ITERATION,
    _atomic_torch,
    _forward,
    _model_hash,
    _sha256,
)
from .recovery import (
    _phase_learning_rates,
    _policy_ce,
    _select,
    _stratified_search_select,
)
from .strength_continuation import StrengthContinuationLineage
from .supervised import collate as oracle_collate
from .supervised import losses as oracle_losses


LR_MULTIPLIERS = (1.0, 0.5, 0.25, 0.125)
SAVE_STEPS = (1, 2, 3, 6, 12)
POLICY_DEFINITIONS = (
    ("fresh_search", "fresh_train", 717, 0.70, False),
    ("recent_search", "recent_train", 154, 0.15, False),
    ("oracle_anchor", "oracle_train", 102, 0.10, True),
    ("cycle_anchor", "cycle_train", 51, 0.05, True),
)


def _canonical_digest(value: Any) -> str:
    digest = hashlib.sha256()

    def update(item: Any) -> None:
        if torch.is_tensor(item):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"T")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(item, np.ndarray):
            array = np.ascontiguousarray(item)
            digest.update(b"N")
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(str(array.shape).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(item, dict):
            digest.update(b"D")
            for key in sorted(item, key=lambda entry: str(entry)):
                update(str(key))
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"L")
            for child in item:
                update(child)
        else:
            digest.update(b"S")
            digest.update(repr(item).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _sample_digest(samples: list[Any]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        family = getattr(
            sample,
            "game_id",
            getattr(sample, "family_id", -1),
        )
        digest.update(
            (
                f"{int(family)}:"
                f"{int(getattr(sample, 'state_hash', 0))}:"
                f"{int(getattr(sample, 'ply', -1))}\n"
            ).encode("ascii")
        )
    return digest.hexdigest()


def _value_head_hash(model: Any) -> str:
    digest = hashlib.sha256()
    for parameter in model.value_parameters():
        digest.update(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def _make_key_plan(
    rng_state: dict[str, Any],
    output: Path,
) -> tuple[list[dict[str, np.ndarray]], str]:
    generator = np.random.Generator(np.random.PCG64())
    generator.bit_generator.state = copy.deepcopy(rng_state)
    plans = []
    arrays: dict[str, np.ndarray] = {}
    for step in range(1, max(SAVE_STEPS) + 1):
        row = {}
        for name, _, count, _, _ in POLICY_DEFINITIONS:
            keys = generator.integers(
                0,
                np.iinfo(np.uint64).max,
                size=count,
                dtype=np.uint64,
            )
            row[name] = keys
            arrays[f"step_{step:02d}_{name}"] = keys
        plans.append(row)
    path = output / "fixed_batch_keys.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)
    return plans, _sha256(path)


def _fixed_policy_step(
    lineage: StrengthContinuationLineage,
    model: Any,
    optimizer: torch.optim.Optimizer,
    pools: dict[str, list[Any]],
    keys: dict[str, np.ndarray],
    learning_rate: float,
) -> dict[str, float]:
    model.set_training_phase("policy_only")
    model.train()
    if any(parameter.requires_grad for parameter in model.value_parameters()):
        raise RuntimeError("Value head was not frozen in Policy-only phase")
    _phase_learning_rates(optimizer, "policy", learning_rate)
    optimizer.zero_grad(set_to_none=True)
    metrics = {}
    for name, pool_name, count, weight, oracle in POLICY_DEFINITIONS:
        pool = pools[pool_name]
        if oracle:
            selected = _select(pool, keys[name])
            loss = oracle_losses(
                model, oracle_collate(selected, 0.1, lineage.device)
            )["policy"]
        else:
            selected, _ = _stratified_search_select(
                pool, count, keys[name]
            )
            loss = _policy_ce(model, selected, lineage.device)
        (weight * loss).backward()
        metrics[name] = float(loss.detach().cpu())
    norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lineage.config.train.max_gradient_norm,
    )
    optimizer.step()
    metrics["gradient_norm"] = float(norm)
    return metrics


def _family_excess(
    samples: list[Any],
    probabilities: list[np.ndarray],
    mask: np.ndarray,
) -> tuple[float, dict[int, float]]:
    by_family: dict[int, list[float]] = {}
    for index in np.flatnonzero(mask):
        target = samples[index].visits.astype(np.float64)
        target /= max(1.0, float(target.sum()))
        probability = np.clip(
            probabilities[index].astype(np.float64), 1e-12, 1.0
        )
        entropy = -float(
            np.sum(target * np.log(np.clip(target, 1e-12, 1.0)))
        )
        excess = -float(np.sum(target * np.log(probability))) - entropy
        by_family.setdefault(int(samples[index].game_id), []).append(
            excess
        )
    family = {
        key: float(np.mean(values)) for key, values in by_family.items()
    }
    return float(np.mean(list(family.values()))), family


def _paired_ci(
    baseline: dict[int, float],
    candidate: dict[int, float],
    seed: int,
) -> tuple[float, list[float]]:
    families = sorted(set(baseline).intersection(candidate))
    delta = np.asarray(
        [candidate[key] - baseline[key] for key in families],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(4000, dtype=np.float64)
    for index in range(len(means)):
        means[index] = float(
            np.mean(rng.choice(delta, size=len(delta), replace=True))
        )
    return float(np.mean(delta)), [
        float(np.quantile(means, 0.025)),
        float(np.quantile(means, 0.975)),
    ]


def _policy_kl(
    before: list[np.ndarray],
    after: list[np.ndarray],
) -> float:
    values = []
    for first, second in zip(before, after):
        p = np.clip(first.astype(np.float64), 1e-12, 1.0)
        q = np.clip(second.astype(np.float64), 1e-12, 1.0)
        values.append(float(np.sum(p * np.log(p / q))))
    return float(np.mean(values))


def _metric(
    samples: list[Any],
    baseline_probability: list[np.ndarray],
    candidate_probability: list[np.ndarray],
    masks: dict[str, np.ndarray],
    baseline_excess: dict[str, tuple[float, dict[int, float]]],
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, name in enumerate(
        ("cycle_aware_all", "adjusted_cycle_roots", "ordinary_noncycle")
    ):
        candidate_mean, candidate_family = _family_excess(
            samples, candidate_probability, masks[name]
        )
        baseline_mean, baseline_family = baseline_excess[name]
        delta, ci = _paired_ci(
            baseline_family,
            candidate_family,
            seed ^ ((offset + 1) * 0x9E3779B1),
        )
        result[name] = {
            "samples": int(np.sum(masks[name])),
            "families": len(candidate_family),
            "baseline_family_policy_excess": baseline_mean,
            "candidate_family_policy_excess": candidate_mean,
            "family_delta": delta,
            "family_delta_ci95": ci,
        }
    result["candidate_total_policy_kl"] = _policy_kl(
        baseline_probability, candidate_probability
    )
    return result


def run_policy_lr_ablation(
    config_path: str | Path,
    *,
    cpu_threads: int = 12,
) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(cpu_threads)))
    lineage = StrengthContinuationLineage(
        config_path,
        training_workers=cpu_threads,
        arena_workers=1,
        arena_block_states=64,
    )
    source_checkpoint = lineage._checkpoint(AUDIT_ITERATION)
    source_payload = torch.load(
        source_checkpoint, map_location="cpu", weights_only=False
    )
    output = (
        lineage.output
        / "diagnostics"
        / "iteration_000107_policy_lr_ablation"
    )
    (output / "candidates").mkdir(parents=True, exist_ok=True)
    fresh = lineage.runner._iteration_samples(AUDIT_ITERATION)
    fresh_train = [
        sample for sample in fresh if int(sample.game_id) % 10 != 0
    ]
    validation = [
        sample for sample in fresh if int(sample.game_id) % 10 == 0
    ]
    recent_train = lineage._recent_samples(AUDIT_ITERATION)
    if not recent_train:
        raise RuntimeError("fixed recent replay pool is missing")
    pools = {
        "fresh_train": fresh_train,
        "recent_train": recent_train,
        "oracle_train": lineage.runner._oracle_dataset().by_split["train"],
    }
    pools["cycle_train"] = [
        sample
        for sample in pools["oracle_train"]
        if np.any(sample.repetition_contexts > 0)
    ]
    key_plan, key_plan_hash = _make_key_plan(
        source_payload["rng"]["alternating_batch_generator"],
        output,
    )
    baseline_model, _, _, _ = lineage._load_training_state(
        AUDIT_ITERATION
    )
    baseline_probability, _ = _forward(
        baseline_model, validation, lineage.device
    )
    adjusted = np.asarray(
        [
            bool(getattr(sample, "cycle_teacher_adjusted", False))
            or (
                bool(sample.cycle_entry)
                and int(sample.visits.sum()) < 255
            )
            for sample in validation
        ],
        dtype=np.bool_,
    )
    masks = {
        "cycle_aware_all": np.ones(len(validation), dtype=np.bool_),
        "adjusted_cycle_roots": adjusted,
        "ordinary_noncycle": np.asarray(
            [not bool(sample.cycle_entry) for sample in validation],
            dtype=np.bool_,
        ),
    }
    baseline_excess = {
        name: _family_excess(validation, baseline_probability, mask)
        for name, mask in masks.items()
    }
    source_value_hash = _value_head_hash(baseline_model)
    base_learning_rate = float(lineage.config.train.learning_rate)
    reports = {}
    for multiplier in LR_MULTIPLIERS:
        model, optimizer, scheduler, global_step = (
            lineage._load_training_state(AUDIT_ITERATION)
        )
        if _value_head_hash(model) != source_value_hash:
            raise RuntimeError("source Value head changed between LR runs")
        rows = []
        candidates = {}
        for step in range(1, max(SAVE_STEPS) + 1):
            step_metric = _fixed_policy_step(
                lineage,
                model,
                optimizer,
                pools,
                key_plan[step - 1],
                base_learning_rate * multiplier,
            )
            rows.append({"step": step, **step_metric})
            if _value_head_hash(model) != source_value_hash:
                raise RuntimeError(
                    f"Value head changed at multiplier={multiplier} step={step}"
                )
            if step not in SAVE_STEPS:
                continue
            candidate_probability, _ = _forward(
                model, validation, lineage.device
            )
            metric = _metric(
                validation,
                baseline_probability,
                candidate_probability,
                masks,
                baseline_excess,
                lineage.config.seed
                ^ int(round(multiplier * 1000))
                ^ step,
            )
            label = str(multiplier).replace(".", "p")
            checkpoint = (
                output
                / "candidates"
                / f"lr_{label}_step_{step:02d}.pt"
            )
            payload = {
                "schema": "search-v3-policy-lr-candidate-v1",
                "source_iteration": AUDIT_ITERATION,
                "source_checkpoint": str(source_checkpoint.resolve()),
                "source_checkpoint_sha256": _sha256(source_checkpoint),
                "source_model_hash": _model_hash(
                    source_payload["model"]
                ),
                "source_optimizer_hash": _canonical_digest(
                    source_payload["optimizer"]
                ),
                "source_rng_hash": _canonical_digest(
                    source_payload["rng"]
                ),
                "fixed_batch_key_sha256": key_plan_hash,
                "lr_multiplier": multiplier,
                "effective_policy_head_lr": (
                    base_learning_rate * multiplier
                ),
                "policy_steps": step,
                "value_steps": 0,
                "value_head_frozen": True,
                "value_head_hash": source_value_hash,
                "optimizer_state_saved": False,
                "scheduler_advanced": False,
                "global_step_advanced": False,
                "model": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "model_hash": _model_hash(model.state_dict()),
                "leaf_value_scale": float(
                    model.search_v3_leaf_value_scale
                ),
                "metrics": metric,
            }
            if checkpoint.exists():
                existing = torch.load(
                    checkpoint, map_location="cpu", weights_only=False
                )
                if existing["model_hash"] != payload["model_hash"]:
                    raise RuntimeError(
                        f"candidate changed on rerun: {checkpoint}"
                    )
            else:
                _atomic_torch(checkpoint, payload)
            candidates[str(step)] = {
                "checkpoint": str(checkpoint.resolve()),
                "model_hash": payload["model_hash"],
                **metric,
            }
            print(
                f"[policy-lr] multiplier={multiplier:g} step={step} "
                f"all={metric['cycle_aware_all']['family_delta']:+.6f} "
                f"cycle={metric['adjusted_cycle_roots']['family_delta']:+.6f} "
                f"ordinary={metric['ordinary_noncycle']['family_delta']:+.6f} "
                f"kl={metric['candidate_total_policy_kl']:.6f}",
                flush=True,
            )
        reports[str(multiplier)] = {
            "effective_policy_head_lr": base_learning_rate * multiplier,
            "training_rows": rows,
            "candidates": candidates,
            "final_optimizer_hash": _canonical_digest(
                optimizer.state_dict()
            ),
            "scheduler_unchanged": scheduler == source_payload["scheduler"],
            "global_step_unchanged": global_step
            == int(source_payload["global_step"]),
            "value_head_hash_unchanged": (
                _value_head_hash(model) == source_value_hash
            ),
        }
    report = {
        "schema": "search-v3-iteration107-policy-lr-ablation-v1",
        "source_iteration": AUDIT_ITERATION,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_model_hash": _model_hash(source_payload["model"]),
        "source_optimizer_hash": _canonical_digest(
            source_payload["optimizer"]
        ),
        "source_rng_hash": _canonical_digest(source_payload["rng"]),
        "replay_iteration": AUDIT_ITERATION,
        "replay_summary_sha256": _sha256(
            lineage.output
            / "search_replay"
            / f"iteration_{AUDIT_ITERATION:06d}"
            / "summary.json"
        ),
        "fixed_batch_key_file": str(
            (output / "fixed_batch_keys.npz").resolve()
        ),
        "fixed_batch_key_sha256": key_plan_hash,
        "fixed_inputs": {
            "model": True,
            "optimizer_state": True,
            "replay": True,
            "batch_order": True,
            "rng": True,
        },
        "policy_only": True,
        "value_frozen": True,
        "arena_run": False,
        "probe_run": False,
        "cycle_quota_changed": False,
        "repeat_scale_changed": False,
        "pool_digests": {
            name: {
                "samples": len(pool),
                "sha256": _sample_digest(pool),
            }
            for name, pool in pools.items()
        },
        "heldout": {
            "samples": len(validation),
            "families": len(
                {int(sample.game_id) for sample in validation}
            ),
            "adjusted_cycle_samples": int(np.sum(adjusted)),
            "ordinary_noncycle_samples": int(
                np.sum(masks["ordinary_noncycle"])
            ),
        },
        "baseline": {
            name: {
                "family_policy_excess": value[0],
                "families": len(value[1]),
            }
            for name, value in baseline_excess.items()
        },
        "lr_multipliers": list(LR_MULTIPLIERS),
        "save_steps": list(SAVE_STEPS),
        "results": reports,
    }
    atomic_json(output / "report.json", report)
    with (output / "summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "lr_multiplier",
                "policy_steps",
                "cycle_aware_family_excess",
                "cycle_aware_family_delta",
                "adjusted_cycle_family_excess",
                "adjusted_cycle_family_delta",
                "ordinary_noncycle_family_excess",
                "ordinary_noncycle_family_delta",
                "candidate_total_policy_kl",
                "checkpoint",
            ],
        )
        writer.writeheader()
        for multiplier, run in reports.items():
            for step, candidate in run["candidates"].items():
                writer.writerow(
                    {
                        "lr_multiplier": multiplier,
                        "policy_steps": step,
                        "cycle_aware_family_excess": candidate[
                            "cycle_aware_all"
                        ]["candidate_family_policy_excess"],
                        "cycle_aware_family_delta": candidate[
                            "cycle_aware_all"
                        ]["family_delta"],
                        "adjusted_cycle_family_excess": candidate[
                            "adjusted_cycle_roots"
                        ]["candidate_family_policy_excess"],
                        "adjusted_cycle_family_delta": candidate[
                            "adjusted_cycle_roots"
                        ]["family_delta"],
                        "ordinary_noncycle_family_excess": candidate[
                            "ordinary_noncycle"
                        ]["candidate_family_policy_excess"],
                        "ordinary_noncycle_family_delta": candidate[
                            "ordinary_noncycle"
                        ]["family_delta"],
                        "candidate_total_policy_kl": candidate[
                            "candidate_total_policy_kl"
                        ],
                        "checkpoint": candidate["checkpoint"],
                    }
                )
    return report
