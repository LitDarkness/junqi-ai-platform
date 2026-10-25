from __future__ import annotations

import math
import os
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from junqi_az.model import PolicyValueNet

from .dataset import iter_shard


@dataclass
class MemorySample:
    observation: np.ndarray
    legal_actions: np.ndarray
    repetition_contexts: np.ndarray
    q_values: np.ndarray
    wdl: np.ndarray
    opponent: str
    ply: int
    terminal_reason: str
    important: bool = False


class MemoryOracleDataset:
    def __init__(self, shard_root: Path) -> None:
        self.by_split: dict[str, list[MemorySample]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for shard in sorted(shard_root.glob("*.npz")):
            for row in iter_shard(shard):
                self.by_split[row["split"]].append(
                    MemorySample(
                        observation=row["observation"].astype(
                            np.float16, copy=False
                        ),
                        legal_actions=row["legal_actions"].astype(
                            np.int16, copy=False
                        ),
                        repetition_contexts=row["repetition_contexts"].astype(
                            np.uint8, copy=False
                        ),
                        q_values=row["q_values"].astype(
                            np.float16, copy=False
                        ),
                        wdl=row["wdl"].astype(np.float16, copy=False),
                        opponent=row["opponent"],
                        ply=row["ply"],
                        terminal_reason=row["terminal_reason"],
                        important=bool(row["important"]),
                    )
                )
        if not self.by_split["train"]:
            raise RuntimeError("Oracle training split is empty")


def soft_target(q_values: np.ndarray, tau: float) -> np.ndarray:
    values = q_values.astype(np.float64)
    values = (values - values.max(initial=0.0)) / max(float(tau), 1e-6)
    values = np.exp(np.clip(values, -80.0, 0.0))
    values /= values.sum()
    return values.astype(np.float32)


def collate(
    samples: list[MemorySample],
    tau: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    count = len(samples)
    observations = torch.from_numpy(
        np.stack([sample.observation for sample in samples])
    ).to(device=device, dtype=torch.float32)
    legal_mask_cpu = np.zeros((count, 3600), dtype=np.bool_)
    repetition_cpu = np.zeros((count, 3600), dtype=np.uint8)
    targets_cpu = np.zeros((count, 3600), dtype=np.float32)
    optimal_targets_cpu = np.zeros((count, 3600), dtype=np.float32)
    for index, sample in enumerate(samples):
        legal = sample.legal_actions.astype(np.int64, copy=False)
        target = soft_target(sample.q_values, tau)
        legal_mask_cpu[index, legal] = True
        repetition_cpu[index, legal] = sample.repetition_contexts
        targets_cpu[index, legal] = target
        q_values = sample.q_values.astype(np.float64)
        optimal = q_values >= float(q_values.max()) - 1e-12
        optimal_targets_cpu[index, legal] = (
            optimal.astype(np.float32) / max(1, int(optimal.sum()))
        )
    legal_mask = torch.from_numpy(legal_mask_cpu).to(device)
    repetition = torch.from_numpy(repetition_cpu).to(device)
    targets = torch.from_numpy(targets_cpu).to(device)
    optimal_targets = torch.from_numpy(optimal_targets_cpu).to(device)
    wdl = torch.from_numpy(
        np.stack([sample.wdl for sample in samples]).astype(np.float32)
    ).to(device)
    oracle_scalar = torch.from_numpy(
        np.asarray(
            [float(np.max(sample.q_values)) for sample in samples],
            dtype=np.float32,
        )
    ).to(device)
    return {
        "observations": observations,
        "legal_mask": legal_mask,
        "repetition": repetition,
        "policy_target": targets,
        "optimal_policy_target": optimal_targets,
        "wdl_target": wdl,
        "oracle_scalar_target": oracle_scalar,
    }


def _head_grad_norm(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> float:
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    total = sum(
        float(torch.sum(gradient.detach().float() ** 2))
        for gradient in gradients
        if gradient is not None
    )
    return math.sqrt(total)


def losses(
    model: PolicyValueNet,
    batch: dict[str, torch.Tensor],
    *,
    use_autocast: bool = True,
) -> dict[str, torch.Tensor]:
    with torch.autocast(
        device_type=batch["observations"].device.type,
        dtype=torch.bfloat16,
        enabled=batch["observations"].is_cuda and use_autocast,
    ):
        logits, _, _, wdl_logits, _ = model.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
    assert wdl_logits is not None
    log_policy = torch.log_softmax(logits.float(), dim=-1)
    policy_per = -torch.sum(batch["policy_target"] * log_policy, dim=-1)
    optimal_policy_per = -torch.sum(
        batch["optimal_policy_target"] * log_policy, dim=-1
    )
    target_entropy = -torch.sum(
        batch["policy_target"]
        * torch.log(batch["policy_target"].clamp_min(1e-12)),
        dim=-1,
    )
    wdl_per = -torch.sum(
        batch["wdl_target"] * torch.log_softmax(wdl_logits.float(), dim=-1),
        dim=-1,
    )
    wdl_probability = torch.softmax(wdl_logits.float(), dim=-1)
    scalar_value = wdl_probability[:, 2] - wdl_probability[:, 0]
    oracle_value_per = (
        scalar_value - batch["oracle_scalar_target"]
    ).square()
    predictions = torch.argmax(logits, dim=-1)
    predicted_target = batch["policy_target"].gather(
        1, predictions.unsqueeze(1)
    ).squeeze(1)
    maximum_target = batch["policy_target"].max(dim=-1).values
    illegal_probability = torch.sum(
        torch.softmax(logits.float(), dim=-1)
        * (~batch["legal_mask"]).float(),
        dim=-1,
    )
    return {
        "policy": policy_per.mean(),
        "optimal_policy": optimal_policy_per.mean(),
        "target_entropy": target_entropy.mean(),
        "excess": (policy_per - target_entropy).mean(),
        "uniform": torch.log(
            batch["legal_mask"].sum(dim=-1).float()
        ).mean(),
        "top1": (
            predicted_target >= maximum_target - 1e-7
        ).float().mean(),
        "top1_optimal_set_size": (
            batch["policy_target"]
            >= maximum_target.unsqueeze(1) - 1e-7
        ).float().sum(dim=-1).mean(),
        "wdl": wdl_per.mean(),
        "oracle_value": oracle_value_per.mean(),
        "wdl_accuracy": (
            torch.argmax(wdl_logits, dim=-1)
            == torch.argmax(batch["wdl_target"], dim=-1)
        ).float().mean(),
        "illegal_probability": illegal_probability.max(),
    }


def policy_label_achievability(
    samples: list[MemorySample],
    tau: float,
) -> dict[str, float]:
    """Measure the Bayes top-1 ceiling and the numerical difficulty of labels.

    A network cannot distinguish rows whose encoded observation, legal mask,
    and repetition context are identical.  The ceiling therefore chooses the
    best common action for each such equivalence class.  Gap statistics expose
    the common case where float16 Oracle Q labels are tied or separated by one
    quantisation unit; those labels require FP32 for a meaningful overfit test.
    """
    groups: dict[bytes, list[int]] = defaultdict(list)
    targets: list[np.ndarray] = []
    optimum_sets: list[np.ndarray] = []
    gaps: list[float] = []
    for index, sample in enumerate(samples):
        digest = hashlib.blake2b(digest_size=16)
        digest.update(np.ascontiguousarray(sample.observation).tobytes())
        digest.update(np.ascontiguousarray(sample.legal_actions).tobytes())
        digest.update(
            np.ascontiguousarray(sample.repetition_contexts).tobytes()
        )
        groups[digest.digest()].append(index)
        q_values = sample.q_values.astype(np.float64)
        targets.append(soft_target(q_values, tau).astype(np.float64))
        maximum = float(q_values.max())
        optimum_sets.append(
            np.flatnonzero(q_values >= maximum - 1e-12)
        )
        ordered = np.sort(q_values)
        gaps.append(
            float(ordered[-1] - ordered[-2])
            if len(ordered) > 1
            else float("inf")
        )

    correct = 0
    for indices in groups.values():
        mean_target = np.mean(
            np.stack([targets[index] for index in indices]), axis=0
        )
        selected = int(np.argmax(mean_target))
        correct += sum(selected in optimum_sets[index] for index in indices)
    gap_array = np.asarray(gaps)
    return {
        "top1_achievable": correct / max(1, len(samples)),
        "unique_network_inputs": float(len(groups)),
        "q_top_tie_fraction": float(np.mean(gap_array == 0)),
        "q_top_gap_le_1e3_fraction": float(
            np.mean(gap_array <= 1e-3)
        ),
        "q_top_gap_median": float(np.quantile(gap_array, 0.5)),
    }


class SupervisedTrainer:
    def __init__(self, config: Any, run: Path) -> None:
        self.config = config
        self.run = run
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.dataset = MemoryOracleDataset(
            run / "oracle_dataset" / "training_shards"
        )
        self.rng = random.Random(config.seed ^ 0x5355504552)
        self.initial = torch.load(
            run / "checkpoints" / "initial_random.pt",
            map_location="cpu",
            weights_only=False,
        )
        # The 6.9M network occupies well below 1 GiB at 64 rows on the target
        # 8 GiB GPU.  A 256-row microbatch removes 75% of Python launch and
        # gradient-accumulation overhead while retaining the configured 1024
        # effective batch.
        self.microbatch = 256 if self.device.type == "cuda" else 16

    def new_model(self) -> PolicyValueNet:
        model = PolicyValueNet(self.config.model)
        model.load_state_dict(self.initial["model"])
        return model.to(self.device)

    def _fixed_indices(self, split: str, count: int, salt: int) -> list[int]:
        source = self.dataset.by_split[split]
        rng = random.Random(self.config.seed ^ salt)
        indices = list(range(len(source)))
        rng.shuffle(indices)
        if len(indices) < count:
            indices = (indices * math.ceil(count / len(indices)))[:count]
        return indices[:count]

    @torch.inference_mode()
    def evaluate(
        self,
        model: PolicyValueNet,
        split: str,
        tau: float,
        maximum: int = 4096,
    ) -> dict[str, float]:
        model.eval()
        source = self.dataset.by_split[split]
        indices = self._fixed_indices(
            split, min(maximum, len(source)), 0x56414C + len(split)
        )
        totals: dict[str, float] = {}
        seen = 0
        for begin in range(0, len(indices), self.microbatch):
            selected = [
                source[index]
                for index in indices[begin:begin + self.microbatch]
            ]
            metric = losses(model, collate(selected, tau, self.device))
            size = len(selected)
            seen += size
            for name, value in metric.items():
                totals[name] = totals.get(name, 0.0) + float(value) * size
        return {name: value / max(1, seen) for name, value in totals.items()}

    def constant_wdl_baseline(self, split: str) -> float:
        train = np.stack(
            [sample.wdl.astype(np.float64) for sample in self.dataset.by_split["train"]]
        )
        prior = np.clip(train.mean(axis=0), 1e-8, 1.0)
        prior /= prior.sum()
        target = np.stack(
            [sample.wdl.astype(np.float64) for sample in self.dataset.by_split[split]]
        )
        return float(np.mean(-np.sum(target * np.log(prior), axis=-1)))

    def overfit(self, tau: float = 0.2) -> tuple[PolicyValueNet, dict[str, Any]]:
        model = self.new_model()
        model.set_training_phase("policy_only")
        model.train()
        indices = self._fixed_indices(
            "train", self.config.train.overfit_batch_size, 0x4F564552
        )
        fixed = [self.dataset.by_split["train"][index] for index in indices]
        achievability = policy_label_achievability(fixed, tau)
        required_top1 = max(
            0.0, achievability["top1_achievable"] - 0.05
        )
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=max(self.config.train.learning_rate, 3e-4),
            weight_decay=0.0,
        )
        policy_history = []
        maximum_steps = 1200
        for step in range(1, maximum_steps + 1):
            optimizer.zero_grad(set_to_none=True)
            aggregate = {
                "policy": 0.0,
                "optimal_policy": 0.0,
                "excess": 0.0,
                "top1": 0.0,
                "illegal_probability": 0.0,
                "top1_optimal_set_size": 0.0,
            }
            for begin in range(0, len(fixed), self.microbatch):
                selected = fixed[begin:begin + self.microbatch]
                batch = collate(selected, tau, self.device)
                # The Oracle Q table is stored compactly as float16.  More than
                # half of this fixed batch has a top gap <= 1e-3, so BF16 can
                # make an otherwise correct argmax test numerically impossible.
                # Formal training still uses AMP; this diagnostic intentionally
                # tests representational correctness in FP32.
                metric = losses(model, batch, use_autocast=False)
                fraction = len(selected) / len(fixed)
                # A very small diagnostic-only top-set term supplies a ranking
                # gradient for Q labels separated by one float16 quantisation
                # unit.  Formal Oracle fitting below remains pure soft-target
                # cross entropy.
                (
                    (
                        metric["policy"]
                        + 0.05 * metric["optimal_policy"]
                    )
                    * fraction
                ).backward()
                for name in aggregate:
                    aggregate[name] += float(metric[name].detach()) * fraction
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.config.train.max_gradient_norm
            )
            optimizer.step()
            if step == 1 or step % 25 == 0:
                row = {"step": step, **aggregate}
                policy_history.append(row)
                print(
                    f"[search-v3 overfit] step={step} "
                    f"excess={aggregate['excess']:.4f} "
                    f"top1={aggregate['top1']:.1%}",
                    flush=True,
                )
            if (
                aggregate["excess"] < 0.05
                and aggregate["top1"] >= required_top1
                and aggregate["illegal_probability"] <= 1e-8
            ):
                break
        policy_passed = (
            aggregate["excess"] < 0.05
            and aggregate["top1"] >= required_top1
            and aggregate["illegal_probability"] <= 1e-8
        )
        policy_final = dict(aggregate)
        if not policy_passed:
            return model, {
                "policy_steps": step,
                "value_steps": 0,
                "samples": len(fixed),
                "tau": tau,
                "label_achievability": achievability,
                "required_top1": required_top1,
                "policy_final": policy_final,
                "value_final": None,
                "policy_history": policy_history,
                "value_history": [],
                "policy_phase": {
                    "value_loss_weight": 0.0,
                    "weight_decay": 0.0,
                    "kl_weight": 0.0,
                    "symmetry_augmentation": False,
                    "trainable": "encoder+policy_head",
                    "soft_policy_weight": 1.0,
                    "diagnostic_top_set_weight": 0.05,
                },
                "value_phase": {"skipped": "policy_overfit_failed"},
                "passed": False,
            }

        # WDL capability is checked in a separate phase.  It cannot help or
        # hurt the Policy result above.
        model.set_training_phase("value_distill")
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=max(self.config.train.learning_rate, 3e-4),
            weight_decay=0.0,
        )
        value_history = []
        value_aggregate = {"wdl": float("inf"), "wdl_accuracy": 0.0}
        for value_step in range(1, 601):
            optimizer.zero_grad(set_to_none=True)
            value_aggregate = {"wdl": 0.0, "wdl_accuracy": 0.0}
            for begin in range(0, len(fixed), self.microbatch):
                selected = fixed[begin:begin + self.microbatch]
                batch = collate(selected, tau, self.device)
                metric = losses(model, batch, use_autocast=False)
                fraction = len(selected) / len(fixed)
                (metric["wdl"] * fraction).backward()
                value_aggregate["wdl"] += (
                    float(metric["wdl"].detach()) * fraction
                )
                value_aggregate["wdl_accuracy"] += (
                    float(metric["wdl_accuracy"].detach()) * fraction
                )
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                self.config.train.max_gradient_norm,
            )
            optimizer.step()
            if value_step == 1 or value_step % 25 == 0:
                row = {"step": value_step, **value_aggregate}
                value_history.append(row)
                print(
                    f"[search-v3 overfit-value] step={value_step} "
                    f"wdl={value_aggregate['wdl']:.4f} "
                    f"accuracy={value_aggregate['wdl_accuracy']:.1%}",
                    flush=True,
                )
            if (
                value_aggregate["wdl_accuracy"] >= 0.95
                and value_aggregate["wdl"] <= 0.15
            ):
                break
        value_passed = (
            value_aggregate["wdl_accuracy"] >= 0.95
            and value_aggregate["wdl"] <= 0.15
        )
        report = {
            "policy_steps": step,
            "value_steps": value_step,
            "samples": len(fixed),
            "tau": tau,
            "label_achievability": achievability,
            "required_top1": required_top1,
            "policy_final": policy_final,
            "value_final": value_aggregate,
            "policy_history": policy_history,
            "value_history": value_history,
            "policy_phase": {
                "value_loss_weight": 0.0,
                "weight_decay": 0.0,
                "kl_weight": 0.0,
                "symmetry_augmentation": False,
                "trainable": "encoder+policy_head",
                "soft_policy_weight": 1.0,
                "diagnostic_top_set_weight": 0.05,
            },
            "value_phase": {
                "policy_loss_weight": 0.0,
                "trainable": "encoder+wdl_head",
            },
            "passed": policy_passed and value_passed,
        }
        return model, report

    def fit_variant(
        self, tau: float, epochs: int, *, save_name: str
    ) -> tuple[PolicyValueNet, dict[str, Any]]:
        model = self.new_model()
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
            eta_min=self.config.train.minimum_learning_rate,
        )
        train = self.dataset.by_split["train"]
        epoch_rows = []
        best_state: dict[str, torch.Tensor] | None = None
        best_validation: dict[str, float] | None = None
        best_epoch = 0
        best_score = float("inf")
        for epoch in range(1, epochs + 1):
            model.train()
            indices = list(range(len(train)))
            self.rng.shuffle(indices)
            policy_head_norm = 0.0
            value_head_norm = 0.0
            policy_encoder_norm = 0.0
            value_encoder_norm = 0.0
            for batch_begin in range(
                0, len(indices), self.config.train.batch_size
            ):
                effective = indices[
                    batch_begin:
                    batch_begin + self.config.train.batch_size
                ]
                optimizer.zero_grad(set_to_none=True)
                for micro_begin in range(
                    0, len(effective), self.microbatch
                ):
                    selected = [
                        train[index]
                        for index in effective[
                            micro_begin:micro_begin + self.microbatch
                        ]
                    ]
                    batch = collate(selected, tau, self.device)
                    metric = losses(model, batch)
                    fraction = len(selected) / len(effective)
                    if batch_begin == 0 and micro_begin == 0:
                        encoder_parameters = list(
                            model.encoder.parameters()
                        )
                        policy_head_norm = _head_grad_norm(
                            metric["policy"],
                            list(model.policy_head_parameters()),
                        )
                        value_head_norm = _head_grad_norm(
                            metric["wdl"],
                            list(model.value_parameters()),
                        )
                        policy_encoder_norm = _head_grad_norm(
                            metric["policy"], encoder_parameters
                        )
                        value_encoder_norm = _head_grad_norm(
                            metric["wdl"], encoder_parameters
                        )
                    objective = (
                        self.config.train.policy_weight * metric["policy"]
                        + self.config.train.terminal_wdl_weight * metric["wdl"]
                        + self.config.train.oracle_value_aux_weight
                        * metric["oracle_value"]
                    )
                    (objective * fraction).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    self.config.train.max_gradient_norm,
                )
                optimizer.step()
            scheduler.step()
            validation = self.evaluate(model, "validation", tau)
            row = {
                "epoch": epoch,
                "learning_rate": scheduler.get_last_lr()[0],
                "validation": validation,
                "policy_loss": validation["policy"],
                "policy_weighted_contribution": (
                    self.config.train.policy_weight
                    * validation["policy"]
                ),
                "policy_head_gradient_norm": policy_head_norm,
                "policy_encoder_gradient_norm": policy_encoder_norm,
                "terminal_wdl_loss": validation["wdl"],
                "terminal_wdl_weighted_contribution": (
                    self.config.train.terminal_wdl_weight
                    * validation["wdl"]
                ),
                "terminal_wdl_head_gradient_norm": value_head_norm,
                "terminal_wdl_encoder_gradient_norm": value_encoder_norm,
                "oracle_auxiliary_loss": validation["oracle_value"],
                "oracle_auxiliary_weighted_contribution": (
                    self.config.train.oracle_value_aux_weight
                    * validation["oracle_value"]
                ),
                "oracle_auxiliary_gradient_norm": 0.0,
            }
            epoch_rows.append(row)
            # Select on held-out fit of both heads.  This prevents a late WDL
            # rebound from replacing a better search leaf evaluator merely
            # because the fixed epoch budget has not yet expired.
            selection_score = (
                validation["excess"] + validation["wdl"]
            )
            row["selection_score"] = selection_score
            if selection_score < best_score:
                best_score = selection_score
                best_epoch = epoch
                best_validation = dict(validation)
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
            print(
                f"[search-v3 supervised] tau={tau} epoch={epoch}/{epochs} "
                f"val_ce={validation['policy']:.4f} "
                f"excess={validation['excess']:.4f} "
                f"top1={validation['top1']:.1%} "
                f"wdl={validation['wdl']:.4f}",
                flush=True,
            )
        if best_state is None or best_validation is None:
            raise RuntimeError("supervised training produced no epoch")
        last_validation = dict(epoch_rows[-1]["validation"])
        model.load_state_dict(best_state)
        selected_validation = self.evaluate(model, "validation", tau)
        checkpoint = self.run / "checkpoints" / save_name
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temporary = checkpoint.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "open-junqi-search-v3-checkpoint",
                "phase": "supervised",
                "completed": True,
                "config": self.config.to_dict(),
                "config_hash": self.config.hash(),
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "tau_label": tau,
                "selected_epoch": best_epoch,
                "selection_score": best_score,
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
        return model, {
            "tau": tau,
            "effective_batch_size": self.config.train.batch_size,
            "microbatch_size": self.microbatch,
            "epochs": epoch_rows,
            "checkpoint": str(checkpoint),
            "validation": selected_validation,
            "last_validation": last_validation,
            "selected_epoch": best_epoch,
            "selection_score": best_score,
            "selection_criterion": "validation_excess_plus_wdl_ce",
            "test": self.evaluate(model, "test", tau),
        }
