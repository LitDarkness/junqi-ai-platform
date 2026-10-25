from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .alternating_lineage import (
    _grouped_optimizer,
    _restore_rng,
    _rng_snapshot,
    _sha256,
)
from .dataset import atomic_json
from .iteration107_audit import _forward, _model_hash
from .policy_lr_ablation import (
    _canonical_digest,
    _family_excess,
    _fixed_policy_step,
    _make_key_plan,
    _paired_ci,
    _policy_kl,
)
from .recovery import RECOVERY_CONFIG
from .strength_continuation import StrengthContinuationLineage
from .value_step_ablation import (
    _classification,
    _wdl_family,
)


SOURCE_ITERATION = 107
BOOTSTRAP_ITERATION = 108


def _atomic_torch(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _category_masks(samples: list[Any]) -> dict[str, np.ndarray]:
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


def _internal_gate(
    lineage: StrengthContinuationLineage,
    incumbent: PolicyValueNet,
    candidate: PolicyValueNet,
    validation: list[Any],
    value_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    before_probability, before_wdl_logp = _forward(
        incumbent, validation, lineage.device
    )
    after_probability, after_wdl_logp = _forward(
        candidate, validation, lineage.device
    )
    masks = _category_masks(validation)
    categories: dict[str, Any] = {}
    for offset, (name, mask) in enumerate(masks.items()):
        before_mean, before_family = _family_excess(
            validation, before_probability, mask
        )
        after_mean, after_family = _family_excess(
            validation, after_probability, mask
        )
        delta, ci = _paired_ci(
            before_family,
            after_family,
            lineage.config.seed
            ^ 0x4C415445
            ^ ((offset + 1) * 0x9E3779B1),
        )
        categories[name] = {
            "samples": int(np.sum(mask)),
            "families": len(after_family),
            "incumbent_family_policy_excess": before_mean,
            "candidate_family_policy_excess": after_mean,
            "family_delta": delta,
            "family_delta_ci95": ci,
            "classification": _classification(
                delta, ci, before_mean
            ),
        }
    before_wdl, before_wdl_family = _wdl_family(
        validation, before_wdl_logp
    )
    after_wdl, after_wdl_family = _wdl_family(
        validation, after_wdl_logp
    )
    wdl_delta, wdl_ci = _paired_ci(
        before_wdl_family,
        after_wdl_family,
        lineage.config.seed ^ 0x57444C31,
    )
    wdl = {
        "incumbent_family_wdl_ce": before_wdl,
        "candidate_family_wdl_ce": after_wdl,
        "family_delta": wdl_delta,
        "family_delta_ci95": wdl_ci,
        "classification": _classification(
            wdl_delta, wdl_ci, before_wdl
        ),
    }
    logic = lineage.training_logic
    assert logic is not None
    final_kl = float(value_rows[-1]["policy_kl"])
    peak_kl = max(float(row["policy_kl"]) for row in value_rows)
    total_kl = _policy_kl(before_probability, after_probability)
    reasons = []
    if (
        categories["ordinary_noncycle"]["classification"]
        != "improved"
    ):
        reasons.append("ordinary_policy_not_significantly_improved")
    for name in ("cycle_aware_all", "adjusted_cycle_roots"):
        if categories[name]["classification"] not in {
            "improved",
            "neutral",
        }:
            reasons.append(f"{name}_policy_clearly_degraded")
    if wdl["classification"] not in {"improved", "neutral"}:
        reasons.append("wdl_clearly_degraded")
    if final_kl > logic.value_final_kl_limit:
        reasons.append("value_final_kl_exceeded")
    if peak_kl > logic.value_peak_kl_limit:
        reasons.append("value_peak_kl_exceeded")
    if total_kl > logic.candidate_policy_kl_limit:
        reasons.append("candidate_policy_kl_exceeded")
    return {
        "schema": "late-stage-short-horizon-internal-gate-v1",
        "categories": categories,
        "wdl": wdl,
        "value_step_final_kl": final_kl,
        "value_step_peak_kl": peak_kl,
        "candidate_level_policy_kl": total_kl,
        "limits": {
            "value_final_kl": logic.value_final_kl_limit,
            "value_peak_kl": logic.value_peak_kl_limit,
            "candidate_policy_kl": logic.candidate_policy_kl_limit,
        },
        "passed": not reasons,
        "reasons": reasons,
    }


def _latest_sealed_suite(
    lineage: StrengthContinuationLineage,
) -> tuple[Path, dict[str, Any]]:
    challenge = (
        lineage.output
        / "championships"
        / "challenger_000100_vs_champion_000092"
        / "sealed_suite"
    )
    states = challenge / "states.jsonl"
    seal_path = challenge / "SEALED.json"
    if not states.exists() or not seal_path.exists():
        raise RuntimeError(
            "the approved existing 1024-state sealed suite is missing"
        )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if (
        int(seal.get("states", -1)) != 1024
        or seal.get("states_sha256") != _sha256(states)
    ):
        raise RuntimeError("the existing sealed suite changed")
    return states, seal


def reconstruct_iteration107_candidate(
    lineage: StrengthContinuationLineage,
) -> dict[str, Any]:
    logic = lineage.training_logic
    if logic is None:
        raise RuntimeError("late-stage training logic is not active")
    source_path = lineage._checkpoint(SOURCE_ITERATION)
    source = torch.load(
        source_path, map_location="cpu", weights_only=False
    )
    source_hash_before = _sha256(source_path)
    model = PolicyValueNet(lineage.config.model).to(lineage.device)
    model.load_state_dict(source["model"])
    model.search_v3_leaf_value_scale = float(
        source["leaf_value_scale"]
    )
    incumbent = PolicyValueNet(lineage.config.model).to(lineage.device)
    incumbent.load_state_dict(source["model"])
    incumbent.search_v3_leaf_value_scale = float(
        source["leaf_value_scale"]
    )
    optimizer = _grouped_optimizer(
        model, lineage.config, source["optimizer"]
    )
    _restore_rng(source["rng"], lineage.generator, lineage.runner)
    fresh = lineage.runner._iteration_samples(SOURCE_ITERATION)
    fresh_train = [
        sample for sample in fresh if int(sample.game_id) % 10 != 0
    ]
    validation = [
        sample for sample in fresh if int(sample.game_id) % 10 == 0
    ]
    recent_train = lineage._recent_samples(SOURCE_ITERATION)
    if not recent_train:
        raise RuntimeError("iteration-107 recent replay is missing")
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
    key_directory = (
        lineage.output
        / "diagnostics"
        / "iteration_000107_policy_lr_ablation"
    )
    key_plan, key_plan_hash = _make_key_plan(
        source["rng"]["alternating_batch_generator"],
        key_directory,
    )
    policy_rows = []
    for step in range(1, logic.selected_policy_step + 1):
        metric = _fixed_policy_step(
            lineage,
            model,
            optimizer,
            pools,
            key_plan[step - 1],
            lineage.config.train.learning_rate
            * logic.policy_lr_multiplier,
        )
        policy_rows.append({"policy_step": step, **metric})
    # Match the controlled ablation exactly: Value uses its own fixed stream,
    # independent of how many fixed Policy batches were replayed.
    _restore_rng(source["rng"], lineage.generator, lineage.runner)
    value_rows = []
    kl_weight = float(
        RECOVERY_CONFIG["value_policy_kl_initial_weight"]
    )
    for step in range(1, logic.selected_value_step + 1):
        metric, kl_weight = lineage._value_phase(
            model, optimizer, pools, kl_weight, legacy_reference=True
        )
        value_rows.append({"value_step": step, **metric})
    model_hash = _model_hash(model.state_dict())
    if model_hash != logic.bootstrap_expected_model_hash:
        raise RuntimeError(
            "full-state reconstruction model hash mismatch: "
            f"{model_hash}"
        )
    internal_gate = _internal_gate(
        lineage, incumbent, model, validation, value_rows
    )
    if not internal_gate["passed"]:
        raise RuntimeError(
            "late-stage internal gate failed: "
            + ";".join(internal_gate["reasons"])
        )
    calibration = lineage.runner._calibrate_leaf_value_scale(model)
    model.search_v3_leaf_value_scale = float(
        calibration["leaf_value_scale"]
    )
    rng = _rng_snapshot(lineage.generator, lineage.runner)
    pending_path = (
        lineage.output
        / "checkpoints"
        / "late_stage_bootstrap_pending.pt"
    )
    scheduler = copy.deepcopy(source["scheduler"])
    scheduler.update(
        {
            "iteration": BOOTSTRAP_ITERATION,
            "training_logic_schema": logic.schema,
            "policy_lr_multiplier": logic.policy_lr_multiplier,
            "policy_candidate_steps": logic.policy_candidate_steps,
            "selected_policy_step": logic.selected_policy_step,
            "value_candidate_steps": logic.value_candidate_steps,
            "selected_value_step": logic.selected_value_step,
            "bootstrap_acceptance": "pending_formal_championship",
        }
    )
    payload = {
        "schema": "open-junqi-search-v3-late-stage-bootstrap-v1",
        "completed": True,
        "phase": "pending_formal_championship",
        "iteration": BOOTSTRAP_ITERATION,
        "global_step": int(source["global_step"])
        + logic.selected_policy_step
        + logic.selected_value_step,
        "config": lineage.config.to_dict(),
        "config_hash": lineage.config.hash(),
        "training_logic_schema": logic.schema,
        "source_iteration": SOURCE_ITERATION,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_hash_before,
        "fixed_policy_batch_key_sha256": key_plan_hash,
        "model": model.state_dict(),
        "model_hash": model_hash,
        "optimizer": optimizer.state_dict(),
        "optimizer_hash": _canonical_digest(optimizer.state_dict()),
        "scheduler": scheduler,
        "leaf_value_scale": model.search_v3_leaf_value_scale,
        "leaf_value_calibration": calibration,
        "replay": {
            **copy.deepcopy(source["replay"]),
            "data_iteration": SOURCE_ITERATION,
            "bootstrap_reused_committed_iteration107_data": True,
            "new_selfplay_generated": False,
        },
        "rng": rng,
        "rng_hash": _canonical_digest(rng),
        "batch_order_restored_from_iteration107": True,
        "policy_rows": policy_rows,
        "value_rows": value_rows,
        "internal_gate": internal_gate,
    }
    if pending_path.exists():
        existing = torch.load(
            pending_path, map_location="cpu", weights_only=False
        )
        immutable = (
            "model_hash",
            "optimizer_hash",
            "rng_hash",
            "fixed_policy_batch_key_sha256",
        )
        if any(
            existing.get(key) != payload.get(key)
            for key in immutable
        ):
            raise RuntimeError(
                "late-stage bootstrap reconstruction changed"
            )
    else:
        _atomic_torch(pending_path, payload)
    if _sha256(source_path) != source_hash_before:
        raise RuntimeError("iteration 107 was modified")
    report = {
        "schema": "late-stage-short-horizon-bootstrap-replay-v1",
        "source_iteration": SOURCE_ITERATION,
        "bootstrap_iteration": BOOTSTRAP_ITERATION,
        "training_logic_schema": logic.schema,
        "source_checkpoint": str(source_path.resolve()),
        "source_checkpoint_sha256": source_hash_before,
        "source_config_hash": source["config_hash"],
        "new_config_hash": lineage.config.hash(),
        "policy_lr_multiplier": logic.policy_lr_multiplier,
        "policy_candidate_steps": logic.policy_candidate_steps,
        "selected_policy_step": logic.selected_policy_step,
        "value_candidate_steps": logic.value_candidate_steps,
        "selected_value_step": logic.selected_value_step,
        "model_hash": model_hash,
        "expected_model_hash": logic.bootstrap_expected_model_hash,
        "model_hash_reproduced": True,
        "optimizer_hash": payload["optimizer_hash"],
        "rng_hash": payload["rng_hash"],
        "fixed_policy_batch_key_sha256": key_plan_hash,
        "internal_gate": internal_gate,
        "pending_checkpoint": str(pending_path.resolve()),
        "pending_checkpoint_sha256": _sha256(pending_path),
        "source_iteration107_unchanged": True,
        "new_selfplay_generated": False,
    }
    atomic_json(
        lineage.output
        / "diagnostics"
        / "iteration_000107_late_stage_bootstrap"
        / "internal_gate.json",
        report,
    )
    return report


def run_late_stage_bootstrap(
    config_path: str | Path,
    *,
    cpu_threads: int = 12,
    arena_workers: int = 12,
    arena_block_states: int = 128,
    run_championship: bool = True,
) -> dict[str, Any]:
    torch.set_num_threads(max(1, int(cpu_threads)))
    lineage = StrengthContinuationLineage(
        config_path,
        training_workers=cpu_threads,
        arena_workers=arena_workers,
        arena_block_states=arena_block_states,
    )
    internal = reconstruct_iteration107_candidate(lineage)
    if not run_championship:
        return {
            "internal_gate": internal,
            "championship": None,
            "accepted": False,
        }
    pending_path = Path(internal["pending_checkpoint"])
    source_path = lineage._checkpoint(SOURCE_ITERATION)
    source_report = json.loads(
        (
            lineage.output
            / "reports"
            / "iteration_000107.json"
        ).read_text(encoding="utf-8")
    )
    suite = _latest_sealed_suite(lineage)
    decision = lineage.championship.run(
        challenger_iteration=BOOTSTRAP_ITERATION,
        challenger_checkpoint=pending_path,
        champion_iteration=SOURCE_ITERATION,
        champion_checkpoint=source_path,
        training_report=source_report,
        suite_override=suite,
    )
    acceptance = {
        "schema": "late-stage-short-horizon-bootstrap-acceptance-v1",
        "training_logic_schema": (
            lineage.training_logic.schema
            if lineage.training_logic is not None
            else None
        ),
        "source_iteration": SOURCE_ITERATION,
        "bootstrap_iteration": BOOTSTRAP_ITERATION,
        "internal_gate_passed": internal["internal_gate"]["passed"],
        "championship": decision,
        "accepted": bool(decision["promoted"]),
        "next_selfplay_iteration": (
            109 if decision["promoted"] else None
        ),
    }
    acceptance_path = (
        lineage.output
        / "diagnostics"
        / "iteration_000107_late_stage_bootstrap"
        / "acceptance.json"
    )
    if decision["promoted"]:
        pending = torch.load(
            pending_path, map_location="cpu", weights_only=False
        )
        scheduler = dict(pending["scheduler"])
        scheduler["accepted_update_iterations"] = int(
            scheduler.get(
                "accepted_update_iterations",
                scheduler.get("accepted_iterations", 0),
            )
        ) + 1
        scheduler["accepted_iterations"] = scheduler[
            "accepted_update_iterations"
        ]
        scheduler["completed_iterations"] = int(
            scheduler.get("completed_iterations", 0)
        ) + 1
        scheduler["bootstrap_acceptance"] = (
            "passed_formal_championship"
        )
        accepted = copy.deepcopy(pending)
        accepted.update(
            {
                "completed": True,
                "phase": "late_stage_bootstrap_accepted",
                "scheduler": scheduler,
                "championship_decision": decision,
            }
        )
        official = lineage._checkpoint(BOOTSTRAP_ITERATION)
        if official.exists():
            existing = torch.load(
                official, map_location="cpu", weights_only=False
            )
            if existing.get("model_hash") != accepted["model_hash"]:
                raise RuntimeError(
                    "official late-stage bootstrap checkpoint changed"
                )
        else:
            _atomic_torch(official, accepted)
        acceptance["official_checkpoint"] = str(official.resolve())
        acceptance["official_checkpoint_sha256"] = _sha256(official)
        report108 = {
            "schema": "late-stage-short-horizon-bootstrap-iteration-v1",
            "iteration": BOOTSTRAP_ITERATION,
            "passed": True,
            "accepted_update": True,
            "new_selfplay_generated": False,
            "reused_data_iteration": SOURCE_ITERATION,
            "training_logic_schema": accepted[
                "training_logic_schema"
            ],
            "chosen_candidate": {
                "policy_steps": int(
                    lineage.training_logic.selected_policy_step
                ),
                "value_steps": int(
                    lineage.training_logic.selected_value_step
                ),
                "model_hash": accepted["model_hash"],
            },
            "internal_gate": internal["internal_gate"],
            "championship": decision,
            "official_checkpoint": str(official.resolve()),
        }
        atomic_json(
            lineage.output
            / "reports"
            / "iteration_000108.json",
            report108,
        )
        atomic_json(
            lineage.output / "progress.json",
            {
                "schema": "formal-alternating-progress",
                "iteration": BOOTSTRAP_ITERATION,
                "phase": "late_stage_bootstrap_accepted",
                "passed": True,
                "next_iteration": 109,
                "training_logic_schema": accepted[
                    "training_logic_schema"
                ],
                "official_checkpoint": str(official.resolve()),
            },
        )
    else:
        acceptance["official_checkpoint"] = None
        acceptance["recovery"] = (
            "retain iteration 107 safe no-op; start adaptive iteration 108"
        )
        source = torch.load(
            source_path, map_location="cpu", weights_only=False
        )
        seed = copy.deepcopy(source)
        seed.update(
            {
                "schema": (
                    "open-junqi-search-v3-late-stage-noop-seed-v1"
                ),
                "completed": True,
                "phase": "late_stage_schema_migration_noop",
                "config": lineage.config.to_dict(),
                "config_hash": lineage.config.hash(),
                "training_logic_schema": (
                    lineage.training_logic.schema
                ),
                "source_checkpoint": str(source_path.resolve()),
                "source_checkpoint_sha256": _sha256(source_path),
                "model_hash": _model_hash(source["model"]),
            }
        )
        seed["scheduler"] = {
            **copy.deepcopy(source["scheduler"]),
            "training_logic_schema": (
                lineage.training_logic.schema
            ),
            "bootstrap_candidate_accepted": False,
        }
        seed_path = (
            lineage.output
            / "checkpoints"
            / "late_stage_seed_iteration_000107.pt"
        )
        if seed_path.exists():
            existing = torch.load(
                seed_path, map_location="cpu", weights_only=False
            )
            if (
                existing.get("model_hash") != seed["model_hash"]
                or existing.get("config_hash")
                != seed["config_hash"]
            ):
                raise RuntimeError(
                    "late-stage no-op migration seed changed"
                )
        else:
            _atomic_torch(seed_path, seed)
        acceptance["resumable_noop_seed"] = str(
            seed_path.resolve()
        )
        acceptance["resumable_noop_seed_sha256"] = _sha256(
            seed_path
        )
        acceptance["next_selfplay_iteration"] = 108
        atomic_json(
            lineage.output / "progress.json",
            {
                "schema": "formal-alternating-progress",
                "iteration": SOURCE_ITERATION,
                "phase": "late_stage_schema_migration_noop",
                "passed": True,
                "bootstrap_candidate_accepted": False,
                "next_iteration": 108,
                "training_logic_schema": (
                    lineage.training_logic.schema
                ),
                "resumable_checkpoint": str(seed_path.resolve()),
            },
        )
    atomic_json(acceptance_path, acceptance)
    return acceptance
