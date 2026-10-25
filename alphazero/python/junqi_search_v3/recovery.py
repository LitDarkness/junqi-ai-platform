from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .ablation import (
    Iteration3Ablation,
    _gradient_cosine,
    _model_optimizer,
    _policy_kl,
    _select,
    _set_global_rng,
)
from .dataset import atomic_json
from .loop import SearchSample, search_distill_batch
from .supervised import collate as oracle_collate
from .supervised import losses as oracle_losses


RECOVERY_CONFIG: dict[str, Any] = {
    "source_iteration": 2,
    "data_iteration": 3,
    "maximum_epochs": 2,
    "policy_steps_per_value_step": 2,
    "policy_mix": {
        "fresh_search": 0.70,
        "recent_search": 0.15,
        "oracle_anchor": 0.10,
        "cycle_anchor": 0.05,
    },
    "value_mix": {
        "fresh_terminal": 0.60,
        "recent_terminal": 0.40,
    },
    "policy_head_lr_multiplier": 1.0,
    "policy_trunk_lr_multiplier": 0.25,
    "value_head_lr_multiplier": 1.0,
    "value_trunk_lr_multiplier": 0.10,
    "oracle_scalar_value_weight": 0.05,
    "value_policy_kl_target": 1e-3,
    "value_policy_kl_initial_weight": 1.0,
    "validation_fraction_of_epoch": 0.25,
    "policy_relative_neutral_limit": 0.005,
    "clear_relative_regression_limit": 0.01,
    "bootstrap_resamples": 4000,
    "cycle_endgame_batch_fraction": 0.18,
    "cycle_family_batch_cap_fraction": 0.02,
}


def _is_cycle_or_endgame(sample: Any) -> bool:
    if bool(getattr(sample, "cycle_entry", False)):
        return True
    if bool(getattr(sample, "endgame", False)):
        return True
    contexts = np.asarray(
        getattr(sample, "repetition_contexts", ()), dtype=np.uint8
    )
    if contexts.size and bool(np.any(contexts > 0)):
        return True
    if int(getattr(sample, "repetition_count", 0)) >= 2:
        return True
    return int(getattr(sample, "ply", 0)) >= 80


def _family_balanced_select(
    pool: list[Any],
    count: int,
    keys: np.ndarray,
    *,
    family_cap_fraction: float,
) -> list[Any]:
    """Round-robin families and cap one family before deterministic fill."""
    if count <= 0 or not pool:
        return []
    seed = int(keys[0]) if len(keys) else 0
    rng = np.random.Generator(np.random.PCG64(seed))
    buckets: dict[int, list[Any]] = defaultdict(list)
    for sample in pool:
        buckets[int(getattr(sample, "game_id", 0))].append(sample)
    families = np.asarray(sorted(buckets), dtype=np.int64)
    rng.shuffle(families)
    for family in families:
        rng.shuffle(buckets[int(family)])
    cap = max(1, int(np.ceil(count * family_cap_fraction)))
    selected: list[Any] = []
    for offset in range(cap):
        for family in families:
            bucket = buckets[int(family)]
            if offset < len(bucket):
                selected.append(bucket[offset])
                if len(selected) >= count:
                    return selected
    # If too few families make the strict cap impossible, fill rather than
    # silently shrinking the optimizer batch; the actual share is logged.
    if len(selected) < count:
        selected.extend(
            _select(pool, keys[: count - len(selected)])
        )
    return selected[:count]


def _stratified_search_select(
    pool: list[Any],
    count: int,
    keys: np.ndarray,
) -> tuple[list[Any], dict[str, float]]:
    special = [sample for sample in pool if _is_cycle_or_endgame(sample)]
    ordinary = [sample for sample in pool if not _is_cycle_or_endgame(sample)]
    requested_special = int(
        round(count * RECOVERY_CONFIG["cycle_endgame_batch_fraction"])
    )
    special_count = min(requested_special, len(special))
    chosen_special = _family_balanced_select(
        special,
        special_count,
        keys[: max(1, special_count)],
        family_cap_fraction=RECOVERY_CONFIG[
            "cycle_family_batch_cap_fraction"
        ],
    )
    remainder = count - len(chosen_special)
    selected = chosen_special + _select(
        ordinary or pool,
        keys[special_count : special_count + remainder],
    )
    family_counts: dict[int, int] = defaultdict(int)
    for sample in chosen_special:
        family_counts[int(getattr(sample, "game_id", 0))] += 1
    return selected[:count], {
        "cycle_endgame_samples": float(len(chosen_special)),
        "cycle_endgame_fraction": len(chosen_special) / max(1, count),
        "cycle_family_max_share": (
            max(family_counts.values(), default=0) / max(1, count)
        ),
        "cycle_endgame_available": float(len(special)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _copy_optimizer_state(
    model: PolicyValueNet,
    source: torch.optim.Optimizer,
    base_lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    groups = [
        {
            "name": "trunk",
            "params": list(model.encoder_parameters()),
            "lr": base_lr * RECOVERY_CONFIG["policy_trunk_lr_multiplier"],
            "weight_decay": weight_decay,
        },
        {
            "name": "policy_head",
            "params": list(model.policy_head_parameters()),
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        {
            "name": "wdl_head",
            "params": list(model.value_parameters()),
            "lr": 0.0,
            "weight_decay": weight_decay,
        },
    ]
    optimizer = torch.optim.AdamW(groups)
    for parameter in model.parameters():
        if parameter in source.state:
            optimizer.state[parameter] = copy.deepcopy(
                source.state[parameter]
            )
    return optimizer


def _phase_learning_rates(
    optimizer: torch.optim.Optimizer,
    phase: str,
    base_lr: float,
) -> None:
    for group in optimizer.param_groups:
        name = group["name"]
        if phase == "policy":
            multiplier = {
                "trunk": RECOVERY_CONFIG["policy_trunk_lr_multiplier"],
                "policy_head": RECOVERY_CONFIG[
                    "policy_head_lr_multiplier"
                ],
                "wdl_head": 0.0,
            }[name]
        elif phase == "value":
            multiplier = {
                "trunk": RECOVERY_CONFIG["value_trunk_lr_multiplier"],
                "policy_head": 0.0,
                "wdl_head": RECOVERY_CONFIG[
                    "value_head_lr_multiplier"
                ],
            }[name]
        else:
            raise ValueError(f"unknown phase: {phase}")
        group["lr"] = base_lr * multiplier


@torch.inference_mode()
def _family_metrics(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> dict[int, dict[str, float]]:
    model.eval()
    totals: dict[int, list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0]
    )
    for begin in range(0, len(samples), 256):
        selected = samples[begin : begin + 256]
        batch = search_distill_batch(selected, device)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=device.type == "cuda",
        ):
            logits, _, _, wdl_logits, _ = model.forward_heads(
                batch["observations"],
                batch["legal_mask"],
                batch["repetition"],
            )
        logp = torch.log_softmax(logits.float(), -1)
        ce = -torch.sum(batch["target"] * logp, -1)
        entropy = -torch.sum(
            batch["target"]
            * torch.log(batch["target"].clamp_min(1e-12)),
            -1,
        )
        wdl = -torch.sum(
            batch["wdl"] * torch.log_softmax(wdl_logits.float(), -1),
            -1,
        )
        excess = (ce - entropy).cpu().numpy()
        value = wdl.cpu().numpy()
        for sample, first, second in zip(selected, excess, value):
            row = totals[int(sample.game_id)]
            row[0] += float(first)
            row[1] += float(second)
            row[2] += 1.0
    return {
        family: {
            "policy_excess": row[0] / row[2],
            "wdl_ce": row[1] / row[2],
            "samples": row[2],
        }
        for family, row in totals.items()
    }


def _paired_bootstrap(
    before: dict[int, dict[str, float]],
    after: dict[int, dict[str, float]],
    key: str,
    seed: int,
) -> dict[str, Any]:
    families = sorted(set(before) & set(after))
    delta = np.asarray(
        [after[family][key] - before[family][key] for family in families],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [before[family][key] for family in families], dtype=np.float64
    )
    mean = float(delta.mean()) if len(delta) else 0.0
    rng = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(RECOVERY_CONFIG["bootstrap_resamples"], np.float64)
    if not families:
        raise ValueError("paired bootstrap requires shared held-out families")
    for index in range(len(means)):
        selected = rng.integers(0, len(delta), size=len(delta))
        means[index] = float(delta[selected].mean())
    lower, upper = np.quantile(means, [0.025, 0.975])
    relative = mean / max(1e-12, float(baseline.mean()))
    if upper < 0.0:
        classification = "improved"
    elif lower > 0.0:
        classification = "clearly_degraded"
    elif (
        lower <= 0.0 <= upper
        and abs(relative)
        <= RECOVERY_CONFIG["policy_relative_neutral_limit"]
    ):
        classification = "neutral"
    else:
        classification = "inconclusive"
    return {
        "metric": key,
        "families": len(families),
        "mean_delta": mean,
        "relative_delta": relative,
        "ci95": [float(lower), float(upper)],
        "classification": classification,
        "practical_regression": relative > RECOVERY_CONFIG[
            "clear_relative_regression_limit"
        ],
        "unit": "held-out game/trajectory family",
    }


def _policy_ce(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> torch.Tensor:
    batch = search_distill_batch(samples, device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits, _, _, _, _ = model.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
    return -torch.sum(
        batch["target"] * torch.log_softmax(logits.float(), -1), -1
    ).mean()


def _wdl_ce(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> torch.Tensor:
    batch = search_distill_batch(samples, device)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model.terminal_wdl_logits(batch["observations"])
    return -torch.sum(
        batch["wdl"] * torch.log_softmax(logits.float(), -1), -1
    ).mean()


@torch.no_grad()
def _anchor_policy(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> list[torch.Tensor]:
    model.eval()
    output = []
    for begin in range(0, len(samples), 128):
        batch = search_distill_batch(samples[begin : begin + 128], device)
        logits, _, _, _, _ = model.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
        output.append(torch.softmax(logits.float(), -1).detach())
    return output


def _anchor_kl(
    model: PolicyValueNet,
    samples: list[SearchSample],
    old_probabilities: list[torch.Tensor],
    device: torch.device,
) -> torch.Tensor:
    return torch.cat(
        _anchor_kl_chunks(model, samples, old_probabilities, device)
    ).mean()


def _anchor_kl_chunks(
    model: PolicyValueNet,
    samples: list[SearchSample],
    old_probabilities: list[torch.Tensor],
    device: torch.device,
) -> list[torch.Tensor]:
    losses = []
    for chunk, begin in enumerate(range(0, len(samples), 128)):
        batch = search_distill_batch(samples[begin : begin + 128], device)
        logits, _, _, _, _ = model.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
        old = old_probabilities[chunk]
        new_log = torch.log_softmax(logits.float(), -1)
        term = torch.where(
            old > 0.0,
            old * (torch.log(old.clamp_min(1e-12)) - new_log),
            torch.zeros_like(old),
        )
        losses.append(torch.sum(term, -1))
    return losses


@dataclass
class PolicyAnchor:
    """Immutable training reference, shared by every step in a Value branch."""

    samples: list[SearchSample]
    probabilities: list[torch.Tensor]


def make_policy_anchor(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
    *,
    seed: int = 0,
    count: int = 256,
) -> PolicyAnchor:
    if not samples:
        raise ValueError("policy anchor requires training samples")
    keys = np.random.default_rng(seed).integers(0, 2**63 - 1, size=count)
    chosen = _family_balanced_select(
        samples, min(count, len(samples)), keys, family_cap_fraction=0.02
    )
    return PolicyAnchor(chosen, _anchor_policy(model, chosen, device))


class Iteration3AlternatingRecovery:
    def _value_anchor(self, model: PolicyValueNet, pools: dict[str, list[Any]]) -> PolicyAnchor:
        return make_policy_anchor(model, pools["fresh_train"], self.device, seed=self.config.seed)

    def __init__(self, config_path: str | Path) -> None:
        self.ablation = Iteration3Ablation(config_path)
        self.config = self.ablation.config
        self.runner = self.ablation.runner
        self.device = self.ablation.device
        self.source_run = self.ablation.run_dir
        self.output = (
            self.ablation.root
            / "runs"
            / "open_junqi_search_v3_iteration3_recovery"
        )
        (self.output / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output / "reports").mkdir(parents=True, exist_ok=True)
        self.seed = self.config.seed ^ 0x5245434F
        self.generator = np.random.Generator(np.random.PCG64(self.seed))

    def _keys(self, count: int) -> np.ndarray:
        return self.generator.integers(
            0, np.iinfo(np.uint64).max, size=count, dtype=np.uint64
        )

    def _policy_step(
        self,
        model: PolicyValueNet,
        optimizer: torch.optim.Optimizer,
        pools: dict[str, list[Any]],
        learning_rate_multiplier: float = 1.0,
    ) -> dict[str, float]:
        model.set_training_phase("policy_only")
        model.train()
        _phase_learning_rates(
            optimizer,
            "policy",
            self.config.train.learning_rate
            * float(learning_rate_multiplier),
        )
        optimizer.zero_grad(set_to_none=True)
        definitions = [
            ("fresh_search", pools["fresh_train"], 717, 0.70, False),
            ("recent_search", pools["recent_train"], 154, 0.15, False),
            ("oracle_anchor", pools["oracle_train"], 102, 0.10, True),
            ("cycle_anchor", pools["cycle_train"], 51, 0.05, True),
        ]
        output = {}
        for name, pool, count, weight, oracle in definitions:
            keys = self._keys(count)
            if oracle:
                selected = _select(pool, keys)
                composition = None
            else:
                selected, composition = _stratified_search_select(
                    pool, count, keys
                )
            if oracle:
                loss = oracle_losses(
                    model, oracle_collate(selected, 0.1, self.device)
                )["policy"]
            else:
                loss = _policy_ce(model, selected, self.device)
            (weight * loss).backward()
            output[name] = float(loss.detach().cpu())
            if composition is not None:
                for key, value in composition.items():
                    output[f"{name}_{key}"] = value
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            self.config.train.max_gradient_norm,
        )
        optimizer.step()
        output["gradient_norm"] = float(norm)
        return output

    def _value_phase(
        self,
        model: PolicyValueNet,
        optimizer: torch.optim.Optimizer,
        pools: dict[str, list[Any]],
        kl_weight: float,
        *,
        anchor: PolicyAnchor | None = None,
        legacy_reference: bool = False,
    ) -> tuple[dict[str, float], float]:
        model.set_training_phase("value_distill")
        _phase_learning_rates(
            optimizer, "value", self.config.train.learning_rate
        )
        if legacy_reference:
            # Only explicit historical reconstruction may use the old contract.
            anchor = PolicyAnchor(
                pools["fresh_validation"][:256],
                _anchor_policy(model, pools["fresh_validation"][:256], self.device),
            )
        elif anchor is None:
            anchor = make_policy_anchor(model, pools["fresh_train"], self.device)
        kl_samples, old_probabilities = anchor.samples, anchor.probabilities
        model.train()
        totals: dict[str, float] = defaultdict(float)
        # Exactly one optimizer update: the formal schedule is two Policy
        # updates followed by one Value update.  KL is measured immediately
        # afterwards and is a hard iteration gate at 1e-3.
        fresh = _select(pools["fresh_train"], self._keys(614))
        recent = _select(pools["recent_train"], self._keys(410))
        oracle = _select(pools["oracle_train"], self._keys(128))
        target = RECOVERY_CONFIG["value_policy_kl_target"]
        saved_model = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        saved_optimizer = copy.deepcopy(optimizer.state_dict())
        cpu_rng = torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        accepted = False
        attempts = 1 if legacy_reference else 5
        for attempt in range(attempts):
            if attempt:
                model.load_state_dict(saved_model)
                optimizer.load_state_dict(saved_optimizer)
                torch.set_rng_state(cpu_rng)
                if cuda_rng is not None:
                    torch.cuda.set_rng_state_all(cuda_rng)
            for group_index, group in enumerate(optimizer.param_groups):
                # saved LR is always the full phase LR, so retries are 1,1/2,...
                group["lr"] = saved_optimizer["param_groups"][
                    group_index
                ]["lr"] * (0.5 ** attempt)
            optimizer.zero_grad(set_to_none=True)
            totals.clear()
            if legacy_reference:
                fresh_loss = _wdl_ce(model, fresh, self.device)
                recent_loss = _wdl_ce(model, recent, self.device)
                scalar = oracle_losses(model, oracle_collate(oracle, 0.1, self.device))["oracle_value"]
                kl = _anchor_kl(model, kl_samples, old_probabilities, self.device)
                (0.60 * fresh_loss + 0.40 * recent_loss + 0.05 * scalar + kl_weight * kl).backward()
                totals.update(fresh_wdl=float(fresh_loss.detach()), recent_wdl=float(recent_loss.detach()),
                              oracle_scalar=float(scalar.detach()), kl_loss=float(kl.detach()))
            else:
                # Accumulate the same logical batch while bounding activation memory.
                for name, selected, weight in (("fresh_wdl", fresh, .60), ("recent_wdl", recent, .40)):
                    for begin in range(0, len(selected), 64):
                        part = selected[begin:begin + 64]
                        loss = _wdl_ce(model, part, self.device)
                        fraction = len(part) / len(selected)
                        (loss * weight * fraction).backward()
                        totals[name] += float(loss.detach()) * fraction
                for begin in range(0, len(oracle), 64):
                    part = oracle[begin:begin + 64]
                    scalar = oracle_losses(model, oracle_collate(part, .1, self.device))["oracle_value"]
                    fraction = len(part) / len(oracle)
                    (scalar * RECOVERY_CONFIG["oracle_scalar_value_weight"] * fraction).backward()
                    totals["oracle_scalar"] += float(scalar.detach()) * fraction
                for chunk, begin in enumerate(range(0, len(kl_samples), 128)):
                    part = kl_samples[begin:begin + 128]
                    kl = _anchor_kl(model, part, [old_probabilities[chunk]], self.device)
                    fraction = len(part) / len(kl_samples)
                    (kl_weight * kl * fraction).backward()
                    totals["kl_loss"] += float(kl.detach()) * fraction
            norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], self.config.train.max_gradient_norm
            )
            optimizer.step()
            observed = self._observed_anchor_kl(model, kl_samples, old_probabilities)
            accepted = bool(np.isfinite(float(norm)) and np.isfinite(observed["state_weighted"]) and
                            max(observed["state_weighted"], observed["family_weighted"]) <= target)
            if accepted or legacy_reference:
                break
        if not accepted and not legacy_reference:
            model.load_state_dict(saved_model)
            optimizer.load_state_dict(saved_optimizer)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
            observed = self._observed_anchor_kl(model, kl_samples, old_probabilities)
        totals["gradient_norm"] = float(norm)
        totals["update_accepted"] = float(accepted)
        totals["backtracking_attempts"] = float(attempt + 1)
        totals["accepted_lr_scale"] = float(0.5 ** attempt) if accepted else 0.0
        totals["anchor_from_training"] = float(not legacy_reference)
        if observed["state_weighted"] > target:
            kl_weight = min(100.0, max(1e-6, kl_weight * 2.0))
        elif observed["state_weighted"] < target * 0.5:
            kl_weight = max(1e-4, kl_weight * 0.8)
        totals["policy_kl"] = observed["state_weighted"]
        totals["policy_family_kl"] = observed["family_weighted"]
        totals["policy_family_kl_max"] = observed["family_maximum"]
        totals["policy_kl_families"] = observed["families"]
        totals["policy_kl_weight_next"] = kl_weight
        return dict(totals), kl_weight

    def _observed_anchor_kl(
        self,
        model: PolicyValueNet,
        samples: list[SearchSample],
        old_probabilities: list[torch.Tensor],
    ) -> dict[str, float | int]:
        with torch.inference_mode():
            values = torch.cat(
                _anchor_kl_chunks(
                    model, samples, old_probabilities, self.device
                )
            ).detach().cpu().numpy()
        family_values: dict[int, list[float]] = defaultdict(list)
        for sample, value in zip(samples, values):
            family_values[int(sample.game_id)].append(float(value))
        family_means = np.asarray(
            [
                float(np.mean(family_values[family]))
                for family in sorted(family_values)
            ],
            dtype=np.float64,
        )
        return {
            "state_weighted": float(np.mean(values)),
            "family_weighted": (
                float(np.mean(family_means)) if len(family_means) else 0.0
            ),
            "family_maximum": (
                float(np.max(family_means)) if len(family_means) else 0.0
            ),
            "families": int(len(family_means)),
        }

    def _candidate(
        self,
        model: PolicyValueNet,
        baseline_family: dict[int, dict[str, float]],
        validation: list[SearchSample],
        policy_steps: int,
        value_steps: int,
    ) -> dict[str, Any]:
        aggregate = self.runner._evaluate_distill(model, validation)
        family = _family_metrics(model, validation, self.device)
        policy = _paired_bootstrap(
            baseline_family,
            family,
            "policy_excess",
            self.seed + policy_steps * 17 + 1,
        )
        value = _paired_bootstrap(
            baseline_family,
            family,
            "wdl_ce",
            self.seed + policy_steps * 17 + 2,
        )
        calibration = self.runner._calibrate_leaf_value_scale(model)
        probe = self.runner._probe_metrics(model)
        eligible = (
            policy["classification"] in {"improved", "neutral"}
            and value["classification"] != "clearly_degraded"
            and probe["search64_oracle_regret"]
            < probe["bare_oracle_regret"]
        )
        return {
            "policy_steps": policy_steps,
            "value_steps": value_steps,
            "fractional_epoch": 0.0,
            "validation": aggregate,
            "paired_policy": policy,
            "paired_wdl": value,
            "value_calibration": calibration,
            "probe": probe,
            "eligible": eligible,
        }

    def run(self) -> dict[str, Any]:
        ablation_summary = (
            self.source_run
            / "diagnostics"
            / "iteration_000003_training_ablation"
            / "summary.json"
        )
        if not ablation_summary.exists():
            raise RuntimeError(
                "formal recovery is gated on completed A/B/C/D experiments"
            )
        ablation_payload = json.loads(
            ablation_summary.read_text(encoding="utf-8")
        )
        required = {
            "A_fresh_policy_only",
            "B_value_only",
            "C_joint_reproduction",
            "D_joint_fresh_policy",
        }
        if set(ablation_payload.get("results", {})) != required:
            raise RuntimeError("A/B/C/D ablation result set is incomplete")
        failure_manifest_path = (
            self.source_run
            / "diagnostics"
            / "iteration_000003_joint_failure_original"
            / "manifest.json"
        )
        failure_manifest = json.loads(
            failure_manifest_path.read_text(encoding="utf-8")
        )
        failure_root = failure_manifest_path.parent
        for relative, expected in failure_manifest["files"].items():
            actual = _sha256(failure_root / relative)
            if actual.upper() != expected.upper():
                raise RuntimeError(
                    f"immutable iteration-3 evidence changed: {relative}"
                )
        pools = self.ablation._load_pools()
        pools["recent_train"] = [
            sample
            for iteration in (1, 2)
            for sample in self.runner._iteration_samples(iteration)
            if sample.game_id % 10 != 0
        ]
        source_checkpoint = self.ablation.source_checkpoint_path
        contract = {
            "schema": "open-junqi-search-v3-iteration3-recovery-contract",
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": _sha256(source_checkpoint),
            "ablation_summary": str(ablation_summary),
            "ablation_summary_sha256": _sha256(ablation_summary),
            "immutable_failure_manifest": str(failure_manifest_path),
            "immutable_failure_manifest_sha256": _sha256(
                failure_manifest_path
            ),
            "data": str(
                self.source_run
                / "search_replay"
                / "iteration_000003"
            ),
            "no_new_selfplay": True,
            "split": "game_id % 10 == 0 is validation",
            "seed": self.seed,
            "configuration": RECOVERY_CONFIG,
            "pool_counts": {
                key: len(value) for key, value in pools.items()
            },
        }
        rendered = json.dumps(
            contract,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        contract["config_hash"] = hashlib.sha256(
            rendered.encode("utf-8")
        ).hexdigest()
        atomic_json(self.output / "manifest.json", contract)
        selection_path = self.output / "reports" / "selection.json"
        checkpoint_path = (
            self.output
            / "checkpoints"
            / "iteration_000003_alternating.pt"
        )
        if selection_path.exists():
            existing = json.loads(
                selection_path.read_text(encoding="utf-8")
            )
            if existing.get("config_hash") != contract["config_hash"]:
                raise RuntimeError(
                    "stale recovery result has a different contract hash"
                )
            if checkpoint_path.exists():
                return existing

        _set_global_rng(self.seed)
        model, source_optimizer = _model_optimizer(
            self.config,
            self.ablation.source_checkpoint,
            self.device,
        )
        baseline, unused = _model_optimizer(
            self.config,
            self.ablation.source_checkpoint,
            self.device,
        )
        del unused
        optimizer = _copy_optimizer_state(
            model,
            source_optimizer,
            self.config.train.learning_rate,
            self.config.train.weight_decay,
        )
        del source_optimizer
        validation = pools["fresh_validation"]
        baseline_family = _family_metrics(
            baseline, validation, self.device
        )
        epoch_steps = int(np.ceil(len(pools["fresh_train"]) / 1024))
        maximum_policy_steps = (
            RECOVERY_CONFIG["maximum_epochs"] * epoch_steps
        )
        interval = max(
            1,
            int(
                round(
                    epoch_steps
                    * RECOVERY_CONFIG["validation_fraction_of_epoch"]
                )
            ),
        )
        started = time.perf_counter()
        candidates = [
            self._candidate(
                model, baseline_family, validation, 0, 0
            )
        ]
        candidates[0]["fractional_epoch"] = 0.0
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        best_optimizer = copy.deepcopy(optimizer.state_dict())
        best_index = 0
        policy_steps = 0
        value_steps = 0
        kl_weight = RECOVERY_CONFIG[
            "value_policy_kl_initial_weight"
        ]
        training_log = []
        cosine_log = []
        while policy_steps < maximum_policy_steps:
            for _ in range(2):
                if policy_steps >= maximum_policy_steps:
                    break
                policy_metric = self._policy_step(
                    model, optimizer, pools
                )
                policy_steps += 1
                training_log.append(
                    {
                        "phase": "policy",
                        "policy_step": policy_steps,
                        "value_step": value_steps,
                        **policy_metric,
                    }
                )
            value_metric, kl_weight = self._value_phase(
                model, optimizer, pools, kl_weight
            )
            value_steps += 1
            training_log.append(
                {
                    "phase": "value",
                    "policy_step": policy_steps,
                    "value_step": value_steps,
                    **value_metric,
                }
            )
            if policy_steps % interval == 0 or (
                policy_steps >= maximum_policy_steps
            ):
                cosine = _gradient_cosine(
                    model,
                    _select(pools["fresh_train"], self._keys(64)),
                    self.device,
                )
                cosine.update(
                    {
                        "policy_step": policy_steps,
                        "value_step": value_steps,
                    }
                )
                cosine_log.append(cosine)
                candidate = self._candidate(
                    model,
                    baseline_family,
                    validation,
                    policy_steps,
                    value_steps,
                )
                candidate["fractional_epoch"] = (
                    policy_steps / epoch_steps
                )
                candidates.append(candidate)
                index = len(candidates) - 1
                if (
                    candidate["eligible"]
                    and candidate["probe"][
                        "search64_oracle_regret"
                    ]
                    < candidates[best_index]["probe"][
                        "search64_oracle_regret"
                    ]
                ):
                    best_index = index
                    best_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in model.state_dict().items()
                    }
                    best_optimizer = copy.deepcopy(
                        optimizer.state_dict()
                    )
                print(
                    "[search-v3 recovery] "
                    f"epoch={candidate['fractional_epoch']:.2f}/2 "
                    f"excess={candidate['validation']['policy_excess']:.6f} "
                    f"wdl={candidate['validation']['wdl_ce']:.6f} "
                    f"policy={candidate['paired_policy']['classification']} "
                    f"value={candidate['paired_wdl']['classification']} "
                    f"eligible={candidate['eligible']}",
                    flush=True,
                )
        model.load_state_dict(best_state)
        optimizer.load_state_dict(best_optimizer)
        chosen = candidates[best_index]
        model.search_v3_leaf_value_scale = chosen["value_calibration"][
            "leaf_value_scale"
        ]
        # Same label means exactly the same paired seeds for before/after.
        pre_external = self.runner._arena_metrics(baseline, 3, "recovery")
        post_external = self.runner._arena_metrics(model, 3, "recovery")
        checkpoint = checkpoint_path
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "open-junqi-search-v3-alternating-recovery",
                "completed": True,
                "source_iteration": 2,
                "data_iteration": 3,
                "config_hash": contract["config_hash"],
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "leaf_value_scale": model.search_v3_leaf_value_scale,
                "leaf_value_calibration": chosen["value_calibration"],
                "selected_candidate": chosen,
                "rng": {
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state(),
                    "cuda": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else None
                    ),
                },
            },
            temporary,
        )
        os.replace(temporary, checkpoint)
        report = {
            "schema": "open-junqi-search-v3-iteration3-recovery-report",
            "status": "candidate_selected_not_installed_in_source_lineage",
            "config_hash": contract["config_hash"],
            "no_new_selfplay": True,
            "elapsed_seconds": time.perf_counter() - started,
            "epoch_steps": epoch_steps,
            "maximum_policy_steps": maximum_policy_steps,
            "value_steps": value_steps,
            "validation_interval_policy_steps": interval,
            "candidates": candidates,
            "chosen_candidate_index": best_index,
            "chosen_candidate": chosen,
            "training_log": training_log,
            "gradient_cosine": cosine_log,
            "external_diagnostic": {
                "used_for_selection": False,
                "before": pre_external,
                "after": post_external,
            },
            "checkpoint": str(checkpoint),
        }
        atomic_json(selection_path, report)
        return report


def run_iteration3_recovery(config_path: str | Path) -> dict[str, Any]:
    return Iteration3AlternatingRecovery(config_path).run()
