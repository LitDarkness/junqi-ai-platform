from __future__ import annotations

import copy
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .alternating_lineage import _restore_rng, _rng_snapshot
from .policy_lr_ablation import (
    _family_excess,
    _paired_ci,
    _policy_kl,
)
from .recovery import (
    RECOVERY_CONFIG,
    _family_metrics,
)
from .value_step_ablation import (
    _classification,
    _wdl_family,
)
from .iteration107_audit import _forward


def _masks(samples: list[Any]) -> dict[str, np.ndarray]:
    adjusted = np.asarray(
        [
            bool(getattr(sample, "cycle_teacher_adjusted", False))
            or (
                bool(sample.cycle_entry)
                and int(sample.visits.sum()) < 255
            )
            for sample in samples
        ],
        dtype=np.bool_,
    )
    return {
        "cycle_aware_all": np.ones(len(samples), dtype=np.bool_),
        "adjusted_cycle_roots": adjusted,
        "ordinary_noncycle": np.asarray(
            [not bool(sample.cycle_entry) for sample in samples],
            dtype=np.bool_,
        ),
    }


def _metrics(
    lineage: Any,
    model: PolicyValueNet,
    validation: list[Any],
    masks: dict[str, np.ndarray],
    baseline_probability: list[np.ndarray],
    baseline_excess: dict[str, tuple[float, dict[int, float]]],
    baseline_wdl: tuple[float, dict[int, float]],
    seed: int,
) -> dict[str, Any]:
    probability, wdl_logp = _forward(
        model, validation, lineage.device
    )
    categories = {}
    for offset, (name, mask) in enumerate(masks.items()):
        mean, family = _family_excess(validation, probability, mask)
        delta, ci = _paired_ci(
            baseline_excess[name][1],
            family,
            seed ^ ((offset + 1) * 0x9E3779B1),
        )
        categories[name] = {
            "family_policy_excess": mean,
            "family_delta": delta,
            "family_delta_ci95": ci,
            "classification": _classification(
                delta, ci, baseline_excess[name][0]
            ),
            "samples": int(np.sum(mask)),
            "families": len(family),
        }
    wdl_mean, wdl_family = _wdl_family(validation, wdl_logp)
    wdl_delta, wdl_ci = _paired_ci(
        baseline_wdl[1], wdl_family, seed ^ 0x57444C
    )
    return {
        "categories": categories,
        "wdl": {
            "family_wdl_ce": wdl_mean,
            "family_delta": wdl_delta,
            "family_delta_ci95": wdl_ci,
            "classification": _classification(
                wdl_delta, wdl_ci, baseline_wdl[0]
            ),
        },
        "candidate_level_policy_kl": _policy_kl(
            baseline_probability, probability
        ),
    }


def _policy_eligible(
    metrics: dict[str, Any],
    policy_kl_limit: float,
) -> tuple[bool, list[str]]:
    reasons = []
    for name in (
        "cycle_aware_all",
        "adjusted_cycle_roots",
        "ordinary_noncycle",
    ):
        if (
            metrics["categories"][name]["classification"]
            == "clearly_degraded"
        ):
            reasons.append(f"{name}_clearly_degraded")
    if metrics["candidate_level_policy_kl"] > policy_kl_limit:
        reasons.append("candidate_policy_kl_exceeded")
    return not reasons, reasons


def _joint_eligible(
    metrics: dict[str, Any],
    *,
    policy_kl_limit: float,
    final_kl: float,
    peak_kl: float,
    final_limit: float,
    peak_limit: float,
) -> tuple[bool, list[str]]:
    eligible, reasons = _policy_eligible(
        metrics, policy_kl_limit
    )
    if metrics["wdl"]["classification"] == "clearly_degraded":
        reasons.append("wdl_clearly_degraded")
    if final_kl > final_limit:
        reasons.append("value_final_kl_exceeded")
    if peak_kl > peak_limit:
        reasons.append("value_peak_kl_exceeded")
    return not reasons, reasons


def _policy_rank(row: dict[str, Any]) -> tuple[float, ...]:
    categories = row["late_stage_metrics"]["categories"]
    return (
        float(
            categories["cycle_aware_all"]["classification"]
            == "improved"
        ),
        -float(categories["cycle_aware_all"]["family_delta"]),
        float(
            categories["ordinary_noncycle"]["classification"]
            == "improved"
        ),
        -float(categories["ordinary_noncycle"]["family_delta"]),
        -float(
            row["late_stage_metrics"][
                "candidate_level_policy_kl"
            ]
        ),
    )


def _joint_rank(row: dict[str, Any]) -> tuple[float, ...]:
    metrics = row["late_stage_metrics"]
    categories = metrics["categories"]
    return (
        float(metrics["wdl"]["classification"] == "improved"),
        -float(metrics["wdl"]["family_delta"]),
        float(
            categories["cycle_aware_all"]["classification"]
            == "improved"
        ),
        -float(categories["cycle_aware_all"]["family_delta"]),
        float(
            categories["ordinary_noncycle"]["classification"]
            == "improved"
        ),
        -float(categories["ordinary_noncycle"]["family_delta"]),
        -float(metrics["candidate_level_policy_kl"]),
    )


def train_short_horizon(
    lineage: Any,
    iteration: int,
    model: PolicyValueNet,
    optimizer: torch.optim.AdamW,
    fresh: list[Any],
    candidate_evaluator: Any = None,
) -> tuple[
    PolicyValueNet,
    torch.optim.AdamW,
    dict[str, Any],
    dict[str, Any],
]:
    """Adaptive 1/2/3/4/6 Policy by 0/1/2 Value candidate grid."""
    logic = lineage.training_logic
    if logic is None:
        raise RuntimeError("late-stage training logic is missing")
    baseline = PolicyValueNet(lineage.config.model).to(lineage.device)
    baseline.load_state_dict(model.state_dict())
    baseline.search_v3_leaf_value_scale = getattr(
        model, "search_v3_leaf_value_scale", 1.0
    )
    fresh_train = [
        sample for sample in fresh if int(sample.game_id) % 10 != 0
    ]
    validation = [
        sample for sample in fresh if int(sample.game_id) % 10 == 0
    ]
    recent_train = lineage._recent_samples(iteration)
    recent_fallback_to_fresh = not recent_train
    if recent_fallback_to_fresh:
        recent_train = fresh_train
    pools = {
        "fresh_train": fresh_train,
        "fresh_validation": validation,
        "recent_train": recent_train,
        "oracle_train": (
            lineage.runner._oracle_dataset().by_split["train"]
        ),
    }
    pools["cycle_train"] = [
        sample
        for sample in pools["oracle_train"]
        if np.any(sample.repetition_contexts > 0)
    ]
    masks = _masks(validation)
    baseline_probability, baseline_wdl_logp = _forward(
        baseline, validation, lineage.device
    )
    baseline_excess = {
        name: _family_excess(
            validation, baseline_probability, mask
        )
        for name, mask in masks.items()
    }
    baseline_wdl = _wdl_family(validation, baseline_wdl_logp)
    baseline_family = _family_metrics(
        baseline, validation, lineage.device
    )
    baseline_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    baseline_optimizer = copy.deepcopy(optimizer.state_dict())
    baseline_rng = _rng_snapshot(lineage.generator, lineage.runner)
    policy_snapshots = []
    training_log = []
    maximum_policy_step = max(logic.policy_candidate_steps)
    for step in range(1, maximum_policy_step + 1):
        metric = lineage._policy_step(
            model,
            optimizer,
            pools,
            learning_rate_multiplier=logic.policy_lr_multiplier,
        )
        training_log.append(
            {
                "phase": "policy",
                "policy_step": step,
                "value_step": 0,
                **metric,
            }
        )
        if step not in logic.policy_candidate_steps:
            continue
        metrics = _metrics(
            lineage,
            model,
            validation,
            masks,
            baseline_probability,
            baseline_excess,
            baseline_wdl,
            lineage.config.seed ^ iteration ^ step,
        )
        eligible, reasons = _policy_eligible(
            metrics, logic.candidate_policy_kl_limit
        )
        row = {
            "policy_steps": step,
            "value_steps": 0,
            "late_stage_metrics": metrics,
            "policy_eligible": eligible,
            "policy_rejection_reasons": reasons,
            "policy_rank": list(
                _policy_rank(
                    {"late_stage_metrics": metrics}
                )
            ),
            "model": {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            },
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "rng": _rng_snapshot(lineage.generator, lineage.runner),
        }
        policy_snapshots.append(row)
        print(
            f"[short-policy] iteration={iteration} step={step} "
            f"all={metrics['categories']['cycle_aware_all']['family_delta']:+.6f}/"
            f"{metrics['categories']['cycle_aware_all']['classification']} "
            f"cycle={metrics['categories']['adjusted_cycle_roots']['family_delta']:+.6f}/"
            f"{metrics['categories']['adjusted_cycle_roots']['classification']} "
            f"ordinary={metrics['categories']['ordinary_noncycle']['family_delta']:+.6f}/"
            f"{metrics['categories']['ordinary_noncycle']['classification']} "
            f"kl={metrics['candidate_level_policy_kl']:.6f} "
            f"eligible={eligible}",
            flush=True,
        )
    shortlisted = sorted(
        (
            row for row in policy_snapshots
            if row["policy_eligible"]
        ),
        key=_policy_rank,
        reverse=True,
    )[: logic.policy_shortlist_size]
    joint_rows = []
    for policy_row in shortlisted:
        model.load_state_dict(policy_row["model"])
        optimizer.load_state_dict(policy_row["optimizer"])
        _restore_rng(
            policy_row["rng"], lineage.generator, lineage.runner
        )
        value_rows = []
        value_anchor = lineage._value_anchor(model, pools)
        kl_weight = float(
            RECOVERY_CONFIG["value_policy_kl_initial_weight"]
        )
        for value_step in logic.value_candidate_steps:
            if value_step:
                metric, kl_weight = lineage._value_phase(
                    model, optimizer, pools, kl_weight, anchor=value_anchor
                )
                value_rows.append(
                    {
                        "phase": "value",
                        "policy_step": policy_row["policy_steps"],
                        "value_step": value_step,
                        **metric,
                    }
                )
                training_log.append(value_rows[-1])
            metrics = _metrics(
                lineage,
                model,
                validation,
                masks,
                baseline_probability,
                baseline_excess,
                baseline_wdl,
                lineage.config.seed
                ^ iteration
                ^ (policy_row["policy_steps"] << 8)
                ^ value_step,
            )
            final_kl = (
                float(value_rows[-1]["policy_kl"])
                if value_rows
                else 0.0
            )
            peak_kl = max(
                (
                    float(item["policy_kl"])
                    for item in value_rows
                ),
                default=0.0,
            )
            eligible, reasons = _joint_eligible(
                metrics,
                policy_kl_limit=logic.candidate_policy_kl_limit,
                final_kl=final_kl,
                peak_kl=peak_kl,
                final_limit=logic.value_final_kl_limit,
                peak_limit=logic.value_peak_kl_limit,
            )
            row = {
                "policy_steps": policy_row["policy_steps"],
                "value_steps": value_step,
                "late_stage_metrics": metrics,
                "value_step_final_kl": final_kl,
                "value_step_peak_kl": peak_kl,
                "candidate_final_kl": final_kl,
                "candidate_final_family_kl": (
                    float(value_rows[-1]["policy_family_kl"])
                    if value_rows
                    else 0.0
                ),
                "candidate_final_family_kl_max": (
                    float(
                        value_rows[-1]["policy_family_kl_max"]
                    )
                    if value_rows
                    else 0.0
                ),
                "max_value_policy_kl": peak_kl,
                "eligible": eligible,
                "rejection_reasons": reasons,
                "rejection_reason": (
                    "none" if eligible else ";".join(reasons)
                ),
                "selection_rank": list(
                    _joint_rank(
                        {"late_stage_metrics": metrics}
                    )
                ),
                "model": {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                },
                "optimizer": copy.deepcopy(
                    optimizer.state_dict()
                ),
                "rng": _rng_snapshot(
                    lineage.generator, lineage.runner
                ),
            }
            joint_rows.append(row)
            print(
                f"[short-joint] iteration={iteration} "
                f"policy={row['policy_steps']} value={value_step} "
                f"all={metrics['categories']['cycle_aware_all']['family_delta']:+.6f}/"
                f"{metrics['categories']['cycle_aware_all']['classification']} "
                f"ordinary={metrics['categories']['ordinary_noncycle']['family_delta']:+.6f}/"
                f"{metrics['categories']['ordinary_noncycle']['classification']} "
                f"wdl={metrics['wdl']['family_delta']:+.6f}/"
                f"{metrics['wdl']['classification']} "
                f"kl-total={metrics['candidate_level_policy_kl']:.6f} "
                f"kl-final/peak={final_kl:.6f}/{peak_kl:.6f} "
                f"eligible={eligible}",
                flush=True,
            )
    eligible_joint = sorted(
        (row for row in joint_rows if row["eligible"]),
        key=_joint_rank,
        reverse=True,
    )
    selected_snapshot = eligible_joint[0] if eligible_joint else None
    if selected_snapshot is None:
        model.load_state_dict(baseline_state)
        optimizer.load_state_dict(baseline_optimizer)
        _restore_rng(
            baseline_rng, lineage.generator, lineage.runner
        )
        policy_steps = value_steps = 0
    else:
        model.load_state_dict(selected_snapshot["model"])
        optimizer.load_state_dict(selected_snapshot["optimizer"])
        _restore_rng(
            selected_snapshot["rng"],
            lineage.generator,
            lineage.runner,
        )
        policy_steps = int(selected_snapshot["policy_steps"])
        value_steps = int(selected_snapshot["value_steps"])
    # Run expensive calibration/probe only for epoch 0 and the selected joint
    # candidate, after the cheap adaptive grid.
    epoch0 = lineage._candidate(
        baseline,
        baseline_family,
        validation,
        0,
        0,
    )
    epoch0["fractional_epoch"] = 0.0
    epoch0["fresh_train"] = lineage.runner._evaluate_distill(
        baseline, fresh_train
    )
    epoch0.update(
        {
            "value_step_final_kl": 0.0,
            "value_step_peak_kl": 0.0,
            "candidate_final_kl": 0.0,
            "candidate_final_family_kl": 0.0,
            "candidate_final_family_kl_max": 0.0,
            "max_value_policy_kl": 0.0,
            "rejection_reasons": [],
            "rejection_reason": "none",
            "selection_rank": [0.0],
            "eligible": True,
        }
    )
    chosen = lineage._candidate(
        model,
        baseline_family,
        validation,
        policy_steps,
        value_steps,
    )
    chosen["fractional_epoch"] = float(policy_steps)
    chosen["fresh_train"] = lineage.runner._evaluate_distill(
        model, fresh_train
    )
    if selected_snapshot is None:
        chosen = epoch0
    else:
        for key in (
            "late_stage_metrics",
            "value_step_final_kl",
            "value_step_peak_kl",
            "candidate_final_kl",
            "candidate_final_family_kl",
            "candidate_final_family_kl_max",
            "max_value_policy_kl",
            "rejection_reasons",
            "rejection_reason",
            "selection_rank",
            "eligible",
        ):
            chosen[key] = selected_snapshot[key]
    if candidate_evaluator is not None:
        chosen["candidate_diagnostics"] = candidate_evaluator(
            0 if selected_snapshot is None else 1,
            model,
            optimizer,
            chosen,
        )
    # Strip large in-memory snapshots before persisting the diagnostic grid.
    compact_policy = [
        {
            key: value
            for key, value in row.items()
            if key not in {"model", "optimizer", "rng"}
        }
        for row in policy_snapshots
    ]
    compact_joint = [
        {
            key: value
            for key, value in row.items()
            if key not in {"model", "optimizer", "rng"}
        }
        for row in joint_rows
    ]
    best_nonzero = (
        {
            "policy_steps": policy_steps,
            "value_steps": value_steps,
            "selection_rank": chosen["selection_rank"],
        }
        if selected_snapshot is not None
        else None
    )
    why_epoch0_won = (
        "not_applicable_nonzero_selected"
        if selected_snapshot is not None
        else "no_joint_candidate_passed_short_horizon_gate"
    )
    candidates = (
        [epoch0, chosen] if selected_snapshot is not None else [epoch0]
    )
    chosen_index = 1 if selected_snapshot is not None else 0
    return model, optimizer, chosen, {
        "candidates": candidates,
        "chosen_candidate_index": chosen_index,
        "training_log": training_log,
        "gradient_cosine": [],
        "accepted_rng": _rng_snapshot(
            lineage.generator, lineage.runner
        ),
        "best_nonzero_candidate": best_nonzero,
        "why_epoch0_won": why_epoch0_won,
        "recent_replay_cycle_cleaned_only": True,
        "recent_replay_fallback_to_current_fresh": (
            recent_fallback_to_fresh
        ),
        "training_logic_schema": logic.schema,
        "policy_lr_multiplier": logic.policy_lr_multiplier,
        "policy_candidate_steps": logic.policy_candidate_steps,
        "value_candidate_steps": logic.value_candidate_steps,
        "policy_shortlist": [
            int(row["policy_steps"]) for row in shortlisted
        ],
        "policy_candidates": compact_policy,
        "joint_candidates": compact_joint,
    }
