from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .alternating_lineage import _restore_rng
from .dataset import atomic_json
from .iteration107_audit import (
    AUDIT_ITERATION,
    _atomic_torch,
    _forward,
    _model_hash,
    _sha256,
)
from .policy_lr_ablation import (
    _canonical_digest,
    _family_excess,
    _fixed_policy_step,
    _make_key_plan,
    _paired_ci,
    _policy_kl,
    _value_head_hash,
)
from .recovery import RECOVERY_CONFIG
from .strength_continuation import StrengthContinuationLineage


BRANCHES = (
    {
        "name": "lr_0p5_policy_03",
        "lr_multiplier": 0.5,
        "policy_steps": 3,
    },
    {
        "name": "lr_0p25_policy_06",
        "lr_multiplier": 0.25,
        "policy_steps": 6,
    },
)
VALUE_STEPS = (0, 1, 2, 3)


def _head_hash(parameters: Any) -> str:
    digest = hashlib.sha256()
    for parameter in parameters:
        digest.update(
            parameter.detach().cpu().contiguous().numpy().tobytes()
        )
    return digest.hexdigest()


def _wdl_family(
    samples: list[Any],
    wdl_log_probability: np.ndarray,
) -> tuple[float, dict[int, float]]:
    by_family: dict[int, list[float]] = {}
    for index, sample in enumerate(samples):
        target = np.asarray(sample.wdl, dtype=np.float64)
        ce = -float(np.sum(target * wdl_log_probability[index]))
        by_family.setdefault(int(sample.game_id), []).append(ce)
    family = {
        key: float(np.mean(values)) for key, values in by_family.items()
    }
    return float(np.mean(list(family.values()))), family


def _category_policy_metrics(
    samples: list[Any],
    probability: list[np.ndarray],
    masks: dict[str, np.ndarray],
    incumbent_excess: dict[str, tuple[float, dict[int, float]]],
    policy_start_excess: dict[str, tuple[float, dict[int, float]]],
    seed: int,
) -> dict[str, Any]:
    result = {}
    for offset, name in enumerate(
        ("cycle_aware_all", "adjusted_cycle_roots", "ordinary_noncycle")
    ):
        mean, family = _family_excess(samples, probability, masks[name])
        incumbent_delta, incumbent_ci = _paired_ci(
            incumbent_excess[name][1],
            family,
            seed ^ ((offset + 1) * 0x9E3779B1),
        )
        start_delta, start_ci = _paired_ci(
            policy_start_excess[name][1],
            family,
            seed ^ ((offset + 1) * 0x85EBCA6B),
        )
        result[name] = {
            "samples": int(np.sum(masks[name])),
            "families": len(family),
            "candidate_family_policy_excess": mean,
            "delta_vs_iteration107": incumbent_delta,
            "delta_vs_iteration107_ci95": incumbent_ci,
            "delta_vs_policy_start": start_delta,
            "delta_vs_policy_start_ci95": start_ci,
        }
    return result


def _classification(delta: float, ci: list[float], baseline: float) -> str:
    if float(ci[1]) < 0.0:
        return "improved"
    if float(ci[0]) > 0.0:
        return "clearly_degraded"
    if abs(float(delta)) > max(1e-12, .005 * abs(float(baseline))):
        return "inconclusive"
    return "neutral"


def run_value_step_ablation(
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
        / "iteration_000107_value_step_ablation"
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
        "fresh_validation": validation,
        "recent_train": recent_train,
        "oracle_train": lineage.runner._oracle_dataset().by_split["train"],
    }
    pools["cycle_train"] = [
        sample
        for sample in pools["oracle_train"]
        if np.any(sample.repetition_contexts > 0)
    ]
    policy_key_output = (
        lineage.output
        / "diagnostics"
        / "iteration_000107_policy_lr_ablation"
    )
    key_plan, key_plan_hash = _make_key_plan(
        source_payload["rng"]["alternating_batch_generator"],
        policy_key_output,
    )
    incumbent, _, _, _ = lineage._load_training_state(AUDIT_ITERATION)
    incumbent_probability, incumbent_wdl_logp = _forward(
        incumbent, validation, lineage.device
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
    incumbent_excess = {
        name: _family_excess(validation, incumbent_probability, mask)
        for name, mask in masks.items()
    }
    incumbent_wdl = _wdl_family(validation, incumbent_wdl_logp)
    base_learning_rate = float(lineage.config.train.learning_rate)
    source_value_rng = copy.deepcopy(source_payload["rng"])
    reports = {}
    for branch_index, branch in enumerate(BRANCHES):
        model, optimizer, scheduler, global_step = (
            lineage._load_training_state(AUDIT_ITERATION)
        )
        for step in range(1, int(branch["policy_steps"]) + 1):
            _fixed_policy_step(
                lineage,
                model,
                optimizer,
                pools,
                key_plan[step - 1],
                base_learning_rate * float(branch["lr_multiplier"]),
            )
        expected_path = (
            policy_key_output
            / "candidates"
            / (
                f"lr_{str(branch['lr_multiplier']).replace('.', 'p')}"
                f"_step_{int(branch['policy_steps']):02d}.pt"
            )
        )
        expected = torch.load(
            expected_path, map_location="cpu", weights_only=False
        )
        if _model_hash(model.state_dict()) != expected["model_hash"]:
            raise RuntimeError(
                f"Policy candidate reconstruction mismatch: {branch['name']}"
            )
        policy_head_hash = _head_hash(model.policy_head_parameters())
        policy_start_probability, policy_start_wdl_logp = _forward(
            model, validation, lineage.device
        )
        policy_start_excess = {
            name: _family_excess(
                validation, policy_start_probability, mask
            )
            for name, mask in masks.items()
        }
        policy_start_wdl = _wdl_family(
            validation, policy_start_wdl_logp
        )
        # Value batches must be identical across branches. Policy batches were
        # supplied from the external fixed key plan, so restore the exact
        # iteration-107 RNG before the first Value step in each branch.
        _restore_rng(source_value_rng, lineage.generator, lineage.runner)
        value_rng_before = _canonical_digest(
            lineage.generator.bit_generator.state
        )
        kl_weight = float(
            RECOVERY_CONFIG["value_policy_kl_initial_weight"]
        )
        value_rows = []
        value_anchor = lineage._value_anchor(model, pools)
        candidates = {}
        for value_step in VALUE_STEPS:
            if value_step > 0:
                metric, kl_weight = lineage._value_phase(
                    model, optimizer, pools, kl_weight, anchor=value_anchor
                )
                value_rows.append(
                    {
                        "value_step": value_step,
                        **metric,
                    }
                )
                if (
                    _head_hash(model.policy_head_parameters())
                    != policy_head_hash
                ):
                    raise RuntimeError(
                        "Policy head weights changed during Value phase"
                    )
            probability, wdl_logp = _forward(
                model, validation, lineage.device
            )
            policy = _category_policy_metrics(
                validation,
                probability,
                masks,
                incumbent_excess,
                policy_start_excess,
                lineage.config.seed
                ^ (branch_index * 0x9E3779B1)
                ^ value_step,
            )
            wdl_mean, wdl_family = _wdl_family(validation, wdl_logp)
            wdl_delta_start, wdl_ci_start = _paired_ci(
                policy_start_wdl[1],
                wdl_family,
                lineage.config.seed ^ 0x57444C ^ value_step,
            )
            wdl_delta_incumbent, wdl_ci_incumbent = _paired_ci(
                incumbent_wdl[1],
                wdl_family,
                lineage.config.seed ^ 0x57444C ^ 0x494E43 ^ value_step,
            )
            step_policy_kl = (
                float(value_rows[-1]["policy_kl"])
                if value_rows
                else 0.0
            )
            peak_policy_kl = max(
                (
                    float(row["policy_kl"])
                    for row in value_rows
                ),
                default=0.0,
            )
            wdl_class = _classification(
                wdl_delta_start,
                wdl_ci_start,
                policy_start_wdl[0],
            )
            wdl_incumbent_class = _classification(
                wdl_delta_incumbent,
                wdl_ci_incumbent,
                incumbent_wdl[0],
            )
            policy_classes = {
                name: _classification(
                    metric["delta_vs_policy_start"],
                    metric["delta_vs_policy_start_ci95"],
                    policy_start_excess[name][0],
                )
                for name, metric in policy.items()
            }
            policy_incumbent_classes = {
                name: _classification(
                    metric["delta_vs_iteration107"],
                    metric["delta_vs_iteration107_ci95"],
                    incumbent_excess[name][0],
                )
                for name, metric in policy.items()
            }
            value_integration_safe = (
                all(
                    value != "clearly_degraded"
                    for value in policy_classes.values()
                )
                and wdl_class != "clearly_degraded"
                and step_policy_kl <= 1e-3
                and peak_policy_kl <= 1.5e-3
            )
            formal_candidate_eligible = (
                all(
                    value != "clearly_degraded"
                    for value in policy_incumbent_classes.values()
                )
                and wdl_incumbent_class != "clearly_degraded"
                and step_policy_kl <= 1e-3
                and peak_policy_kl <= 1.5e-3
            )
            metrics = {
                **policy,
                "wdl": {
                    "candidate_family_wdl_ce": wdl_mean,
                    "delta_vs_policy_start": wdl_delta_start,
                    "delta_vs_policy_start_ci95": wdl_ci_start,
                    "delta_vs_iteration107": wdl_delta_incumbent,
                    "delta_vs_iteration107_ci95": wdl_ci_incumbent,
                    "classification_vs_policy_start": wdl_class,
                    "classification_vs_iteration107": (
                        wdl_incumbent_class
                    ),
                },
                "policy_classification_vs_policy_start": policy_classes,
                "policy_classification_vs_iteration107": (
                    policy_incumbent_classes
                ),
                "incremental_policy_kl_vs_policy_start": _policy_kl(
                    policy_start_probability, probability
                ),
                "candidate_total_policy_kl_vs_iteration107": _policy_kl(
                    incumbent_probability, probability
                ),
                "value_step_final_kl": step_policy_kl,
                "value_step_peak_kl": peak_policy_kl,
                "value_integration_safe": value_integration_safe,
                "formal_candidate_eligible": formal_candidate_eligible,
                # Backward-compatible name: safety relative to the Policy
                # branch start, not eligibility against the incumbent.
                "safe": value_integration_safe,
            }
            if value_step == 0:
                checkpoint = expected_path
            else:
                checkpoint = (
                    output
                    / "candidates"
                    / (
                        f"{branch['name']}_value_{value_step:02d}.pt"
                    )
                )
                payload = {
                    "schema": "search-v3-value-step-candidate-v1",
                    "source_iteration": AUDIT_ITERATION,
                    "policy_source_checkpoint": str(
                        expected_path.resolve()
                    ),
                    "policy_source_model_hash": expected["model_hash"],
                    "source_checkpoint_sha256": _sha256(
                        source_checkpoint
                    ),
                    "fixed_policy_batch_key_sha256": key_plan_hash,
                    "value_batch_rng_before_hash": value_rng_before,
                    "branch": branch,
                    "value_steps": value_step,
                    "policy_head_frozen": True,
                    "policy_head_hash": policy_head_hash,
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
                    "metrics": metrics,
                }
                if checkpoint.exists():
                    existing = torch.load(
                        checkpoint,
                        map_location="cpu",
                        weights_only=False,
                    )
                    if existing["model_hash"] != payload["model_hash"]:
                        raise RuntimeError(
                            f"Value candidate changed: {checkpoint}"
                        )
                else:
                    _atomic_torch(checkpoint, payload)
            candidates[str(value_step)] = {
                "checkpoint": str(checkpoint.resolve()),
                "model_hash": _model_hash(model.state_dict()),
                **metrics,
            }
            print(
                f"[value-ablation] branch={branch['name']} "
                f"value-steps={value_step} "
                f"policy-all="
                f"{policy['cycle_aware_all']['delta_vs_policy_start']:+.6f} "
                f"cycle="
                f"{policy['adjusted_cycle_roots']['delta_vs_policy_start']:+.6f} "
                f"ordinary="
                f"{policy['ordinary_noncycle']['delta_vs_policy_start']:+.6f} "
                f"wdl={wdl_delta_start:+.6f} "
                f"step-kl={step_policy_kl:.6f} "
                f"integration-safe={value_integration_safe} "
                f"formal-eligible={formal_candidate_eligible}",
                flush=True,
            )
        reports[branch["name"]] = {
            "branch": branch,
            "policy_source_checkpoint": str(expected_path.resolve()),
            "policy_source_model_hash": expected["model_hash"],
            "policy_head_hash": policy_head_hash,
            "value_batch_rng_before_hash": value_rng_before,
            "value_batch_rng_after_hash": _canonical_digest(
                lineage.generator.bit_generator.state
            ),
            "value_rows": value_rows,
            "candidates": candidates,
            "scheduler_unchanged": scheduler == source_payload["scheduler"],
            "global_step_unchanged": global_step
            == int(source_payload["global_step"]),
        }
    rng_after = {
        report["value_batch_rng_after_hash"]
        for report in reports.values()
    }
    if len(rng_after) != 1:
        raise RuntimeError("Value batch RNG diverged between branches")
    result = {
        "schema": "search-v3-iteration107-value-step-ablation-v1",
        "source_iteration": AUDIT_ITERATION,
        "source_checkpoint": str(source_checkpoint.resolve()),
        "source_checkpoint_sha256": _sha256(source_checkpoint),
        "source_model_hash": _model_hash(source_payload["model"]),
        "source_optimizer_hash": _canonical_digest(
            source_payload["optimizer"]
        ),
        "source_rng_hash": _canonical_digest(source_payload["rng"]),
        "fixed_policy_batch_key_sha256": key_plan_hash,
        "fixed_value_batch_rng": True,
        "value_batch_rng_after_identical": True,
        "value_steps": list(VALUE_STEPS),
        "arena_run": False,
        "probe_run": False,
        "puct_run": False,
        "fresh_only_tested": False,
        "cycle_quota_changed": False,
        "repeat_scale_changed": False,
        "baseline": {
            "iteration107_wdl_family_ce": incumbent_wdl[0],
            "iteration107_policy_family_excess": {
                name: value[0]
                for name, value in incumbent_excess.items()
            },
        },
        "results": reports,
    }
    atomic_json(output / "report.json", result)
    return result
