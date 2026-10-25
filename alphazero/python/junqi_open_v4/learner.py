from __future__ import annotations

import copy
import hashlib
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet
from junqi_search_v3.loop import SearchSample, search_distill_batch
from junqi_search_v3.recovery import PolicyAnchor, _anchor_kl, _anchor_kl_chunks, make_policy_anchor, _family_balanced_select

from .config import OpenConfig


def model_hash(model: PolicyValueNet) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def rng_snapshot(generator: np.random.Generator) -> dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "batch_generator": copy.deepcopy(generator.bit_generator.state)}


def restore_rng(state: dict[str, Any], generator: np.random.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([row.cpu() for row in state["cuda"]])
    generator.bit_generator.state = copy.deepcopy(state["batch_generator"])


def atomic_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


class Learner:
    def __init__(self, config: OpenConfig, root: Path):
        config.validate()
        self.config, self.root = config, root
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        random.seed(config.seed)
        np.random.seed(config.seed & 0xffffffff)
        torch.manual_seed(config.seed)
        self.generator = np.random.default_rng(config.seed)
        self.model = PolicyValueNet(config.model).to(self.device)
        self.model.search_v3_leaf_value_scale = 1.
        self.provenance = {}
        if config.initial_checkpoint:
            path = (root / config.initial_checkpoint).resolve()
            payload = torch.load(path, map_location="cpu", weights_only=False)
            self.model.load_state_dict(payload["model"])
            self.model.search_v3_leaf_value_scale = float(payload.get("leaf_value_scale", 1.))
            self.provenance = {"initial_checkpoint": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                               "optimizer_reset": "new training and data contract"}
        self.optimizer = torch.optim.AdamW([
            {"name": "encoder", "params": list(self.model.encoder_parameters())},
            {"name": "policy", "params": list(self.model.policy_head_parameters())},
            {"name": "value", "params": list(self.model.value_parameters())},
        ], lr=config.learning_rate, weight_decay=config.weight_decay)
        self.scheduler = {"type": "constant", "base_learning_rate": config.learning_rate, "effective_updates": 0}
        self.counters = dict(iteration=0, attempted_updates=0, effective_updates=0, policy_updates=0,
                             value_updates=0, backtracking_rejections=0)

    def payload(self, *, role: str, lineage: dict[str, Any]) -> dict[str, Any]:
        return {"schema": self.config.schema, "role": role,
                "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                "optimizer": copy.deepcopy(self.optimizer.state_dict()), "scheduler": copy.deepcopy(self.scheduler),
                "rng": rng_snapshot(self.generator), "counters": dict(self.counters),
                "config": self.config.as_dict(), "config_hash": self.config.digest(),
                "leaf_value_scale": self.model.search_v3_leaf_value_scale,
                "model_hash": model_hash(self.model), "provenance": self.provenance,
                "lineage": copy.deepcopy(lineage)}

    def resume(self, path: Path) -> dict[str, Any]:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("schema") != self.config.schema or payload.get("config_hash") != self.config.digest() or payload.get("role") != "learner":
            raise ValueError("checkpoint contract or role mismatch; start a new run")
        self.model.load_state_dict(payload["model"])
        self.model.search_v3_leaf_value_scale = payload["leaf_value_scale"]
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler, self.counters, self.provenance = payload["scheduler"], payload["counters"], payload["provenance"]
        restore_rng(payload["rng"], self.generator)
        if model_hash(self.model) != payload["model_hash"]:
            raise ValueError("checkpoint tensor digest mismatch")
        return payload["lineage"]

    def _batch(self, samples: list[SearchSample]) -> list[SearchSample]:
        count = self.config.logical_batch_size
        keys = self.generator.integers(0, 2**63 - 1, size=count)
        return _family_balanced_select(samples, count, keys, family_cap_fraction=.02)

    def _set_phase(self, phase: str, scale: float = 1.) -> None:
        value = phase == "value"
        self.model.set_training_phase("value_distill" if value else "policy_only")
        for group in self.optimizer.param_groups:
            factor = ({"encoder": self.config.value_encoder_lr_multiplier, "policy": 0., "value": 1.} if value
                      else {"encoder": .25, "policy": 1., "value": 0.})[group["name"]]
            group["lr"] = self.config.learning_rate * scale * factor

    def _backward(self, selected: list[SearchSample], phase: str, anchor: PolicyAnchor | None) -> dict[str, float]:
        self.optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        self.model.train()
        for begin in range(0, len(selected), self.config.microbatch_size):
            part = selected[begin:begin + self.config.microbatch_size]
            batch = search_distill_batch(part, self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=self.device.type == "cuda"):
                logits = (self.model.terminal_wdl_logits(batch["observations"]) if phase == "value" else
                          self.model.policy_logits(batch["observations"], batch["legal_mask"], batch["repetition"]))
            target = batch["wdl"] if phase == "value" else batch["target"]
            loss = -(target * torch.log_softmax(logits.float(), -1)).sum(-1).mean()
            fraction = len(part) / len(selected)
            (loss * fraction).backward()
            loss_sum += float(loss.detach()) * fraction
        kl_sum = 0.
        if anchor is not None:
            # Match the reference chunks exactly; each backward frees its graph.
            for chunk, begin in enumerate(range(0, len(anchor.samples), 128)):
                part = anchor.samples[begin:begin + 128]
                loss = _anchor_kl(self.model, part, [anchor.probabilities[chunk]], self.device)
                fraction = len(part) / len(anchor.samples)
                (loss * fraction * self.config.value_kl_weight).backward()
                kl_sum += float(loss.detach()) * fraction
        norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.model.parameters() if p.requires_grad], self.config.gradient_clip)
        return dict(loss=loss_sum, anchor_kl_before=kl_sum, gradient_norm=float(norm))

    def step(self, selected: list[SearchSample], phase: str, *, anchor: PolicyAnchor | None = None) -> dict[str, Any]:
        if phase not in ("policy", "value") or not selected:
            raise ValueError("step requires a valid phase and nonempty training batch")
        if phase == "value" and anchor is None:
            raise ValueError("Value updates require an explicit frozen training anchor")
        self._set_phase(phase)
        saved_model = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        saved_optimizer = copy.deepcopy(self.optimizer.state_dict())
        saved_rng = rng_snapshot(self.generator)
        accepted = False
        attempts = self.config.value_backtracking_attempts if phase == "value" else 1
        for attempt in range(attempts):
            if attempt:
                self.model.load_state_dict(saved_model)
                self.optimizer.load_state_dict(saved_optimizer)
                restore_rng(saved_rng, self.generator)
            self._set_phase(phase, .5**attempt)
            metrics = self._backward(selected, phase, anchor)
            if np.isfinite(metrics["gradient_norm"]) and np.isfinite(metrics["loss"]):
                self.optimizer.step()
                with torch.no_grad():
                    family_kl = 0.
                    if anchor:
                        losses = torch.cat(_anchor_kl_chunks(self.model, anchor.samples, anchor.probabilities, self.device)).cpu().numpy()
                        kl = max(0., float(losses.mean()))
                        families = {}
                        for sample, loss in zip(anchor.samples, losses):
                            families.setdefault(sample.game_id, []).append(float(loss))
                        family_kl = max(0., float(np.mean([np.mean(rows) for rows in families.values()])))
                    else:
                        kl = 0.
                finite = bool(torch.stack([torch.isfinite(p).all() for p in self.model.parameters()]).all())
                accepted = finite and np.isfinite(kl) and np.isfinite(family_kl) and max(kl, family_kl) <= self.config.value_policy_kl_limit
            else:
                kl = float("inf")
                family_kl = float("inf")
            if accepted:
                break
        if not accepted:
            self.model.load_state_dict(saved_model)
            self.optimizer.load_state_dict(saved_optimizer)
            restore_rng(saved_rng, self.generator)
            self.counters["backtracking_rejections"] += 1
        self.counters["attempted_updates"] += 1
        self.counters["effective_updates"] += int(accepted)
        self.counters[phase + "_updates"] += int(accepted)
        self.scheduler["effective_updates"] = self.counters["effective_updates"]
        return {"phase": phase, **metrics, "accepted": accepted, "attempts": attempt + 1,
                "kl_proposed": kl, "family_kl_proposed": family_kl, "accepted_lr_scale": .5**attempt if accepted else 0.}

    def train(self, samples: list[SearchSample], *, updates: int | None = None) -> list[dict[str, Any]]:
        if not samples:
            raise ValueError("cannot train on empty replay")
        rows, anchor = [], None
        updates = self.config.updates_per_iteration if updates is None else updates
        for index in range(updates):
            phase = "value" if self.counters["attempted_updates"] % 3 == 2 else "policy"
            if phase == "value":
                anchor = make_policy_anchor(self.model, samples, self.device,
                                           seed=self.config.seed ^ self.counters["attempted_updates"], count=128)
            row = self.step(self._batch(samples), phase, anchor=anchor if phase == "value" else None)
            rows.append(row)
            if (index + 1) % 12 == 0 or index + 1 == updates:
                print(f"[learner] updates={index + 1}/{updates} effective={self.counters['effective_updates']} phase={phase} loss={row['loss']:.4f}", flush=True)
        self.model.eval()
        return rows
