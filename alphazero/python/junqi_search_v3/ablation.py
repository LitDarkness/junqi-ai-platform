from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .closed_loop import ClosedLoopRunner
from .config import SearchV3Config
from .dataset import atomic_json
from .loop import SearchSample, search_distill_batch
from .stages import SearchV3Stages
from .supervised import collate as oracle_collate
from .supervised import losses as oracle_losses


EXPERIMENTS = {
    "A_fresh_policy_only": "fresh_policy",
    "B_value_only": "value_only",
    "C_joint_reproduction": "joint_replay",
    "D_joint_fresh_policy": "joint_fresh_policy",
}


def _sample_digest(samples: list[SearchSample]) -> str:
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(
            f"{sample.game_id}:{sample.state_hash}:{sample.ply}\n".encode(
                "ascii"
            )
        )
    return digest.hexdigest()


def _select(pool: list[Any], keys: np.ndarray) -> list[Any]:
    size = len(pool)
    if size == 0:
        raise RuntimeError("ablation source pool is empty")
    return [pool[int(key % size)] for key in keys]


def _set_global_rng(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _model_optimizer(
    config: SearchV3Config,
    checkpoint: dict[str, Any],
    device: torch.device,
) -> tuple[PolicyValueNet, torch.optim.Optimizer]:
    model = PolicyValueNet(config.model).to(device)
    model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )
    optimizer.load_state_dict(copy.deepcopy(checkpoint["optimizer"]))
    return model, optimizer


def _gradient_cosine(
    model: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> dict[str, float]:
    was_training = model.training
    buffers = {
        name: value.detach().clone()
        for name, value in model.named_buffers()
    }
    torch_rng = torch.get_rng_state()
    cuda_rng = (
        torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else None
    )
    model.train()
    model.zero_grad(set_to_none=True)
    batch = search_distill_batch(samples, device)
    logits, _, _, wdl_logits, _ = model.forward_heads(
        batch["observations"],
        batch["legal_mask"],
        batch["repetition"],
    )
    policy = -torch.sum(
        batch["target"] * torch.log_softmax(logits.float(), -1), -1
    ).mean()
    wdl = -torch.sum(
        batch["wdl"] * torch.log_softmax(wdl_logits.float(), -1), -1
    ).mean()
    encoder = [
        parameter
        for parameter in model.encoder_parameters()
        if parameter.requires_grad
    ]
    search_grad = torch.autograd.grad(
        policy, encoder, retain_graph=True, allow_unused=True
    )
    value_grad = torch.autograd.grad(
        wdl, encoder, retain_graph=False, allow_unused=True
    )
    dot = torch.zeros((), device=device, dtype=torch.float64)
    search_sq = torch.zeros((), device=device, dtype=torch.float64)
    value_sq = torch.zeros((), device=device, dtype=torch.float64)
    for first, second in zip(search_grad, value_grad):
        if first is None or second is None:
            continue
        a = first.detach().double()
        b = second.detach().double()
        dot += torch.sum(a * b)
        search_sq += torch.sum(a * a)
        value_sq += torch.sum(b * b)
    denominator = torch.sqrt(search_sq * value_sq).clamp_min(1e-30)
    result = {
        "cosine": float((dot / denominator).cpu()),
        "search_norm": float(torch.sqrt(search_sq).cpu()),
        "value_norm": float(torch.sqrt(value_sq).cpu()),
    }
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        for name, value in model.named_buffers():
            value.copy_(buffers[name])
    torch.set_rng_state(torch_rng)
    if cuda_rng is not None:
        torch.cuda.set_rng_state_all(cuda_rng)
    model.train(was_training)
    return result


@torch.inference_mode()
def _policy_kl(
    before: PolicyValueNet,
    after: PolicyValueNet,
    samples: list[SearchSample],
    device: torch.device,
) -> float:
    before.eval()
    after.eval()
    total = 0.0
    seen = 0
    for begin in range(0, len(samples), 256):
        selected = samples[begin : begin + 256]
        batch = search_distill_batch(selected, device)
        old_logits, _, _, _, _ = before.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
        new_logits, _, _, _, _ = after.forward_heads(
            batch["observations"],
            batch["legal_mask"],
            batch["repetition"],
        )
        old_log = torch.log_softmax(old_logits.float(), -1)
        new_log = torch.log_softmax(new_logits.float(), -1)
        old_probability = torch.softmax(old_logits.float(), -1)
        kl = torch.sum(old_probability * (old_log - new_log), -1)
        total += float(kl.sum().cpu())
        seen += len(selected)
    return total / max(1, seen)


def _cosine_summary(rows: list[dict[str, float]]) -> dict[str, float]:
    values = np.asarray([row["cosine"] for row in rows], np.float64)
    return {
        "samples": int(len(values)),
        "mean": float(values.mean()) if len(values) else 0.0,
        "median": float(np.median(values)) if len(values) else 0.0,
        "p10": float(np.quantile(values, 0.10)) if len(values) else 0.0,
        "p90": float(np.quantile(values, 0.90)) if len(values) else 0.0,
        "negative_fraction": (
            float(np.mean(values < 0.0)) if len(values) else 0.0
        ),
    }


class Iteration3Ablation:
    def __init__(self, config_path: str | Path) -> None:
        self.config_path = Path(config_path).resolve()
        self.stages = SearchV3Stages(self.config_path)
        self.config = self.stages.config
        self.root = self.stages.root
        self.run_dir = self.stages.run
        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        score_scale = json.loads(
            (self.run_dir / "oracle_scale.json").read_text(encoding="utf-8")
        )["score_scale"]
        self.runner = ClosedLoopRunner(
            root=self.root,
            run=self.run_dir,
            backend=self.stages.backend,
            config=self.config,
            score_scale=float(score_scale),
            replay_trace=self.stages._replay_trace,
        )
        self.source_checkpoint_path = (
            self.run_dir / "checkpoints" / "loop_iteration_000002.pt"
        )
        self.source_checkpoint = torch.load(
            self.source_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        self.output = (
            self.run_dir
            / "diagnostics"
            / "iteration_000003_training_ablation"
        )
        self.output.mkdir(parents=True, exist_ok=True)
        self.plan_seed = self.config.seed ^ 0x41424344
        self.plan_rng_state: object | None = None
        self.steps = 350
        self.batch_size = self.config.train.batch_size
        self.validation_interval = max(
            1, math.ceil((self.steps / 2) / 4)
        )
        self.gradient_interval = 8

    def _load_pools(self) -> dict[str, list[Any]]:
        # Restore precisely the RNG boundary stored by iteration 2 before
        # constructing the same bounded replay population used by iteration 3.
        _, _, _, _ = self.runner._load_model_optimizer(2)
        fresh = self.runner._iteration_samples(3)
        replay = self.runner._replay_samples(3, fresh)
        self.plan_rng_state = self.runner.rng.getstate()
        oracle = self.runner._oracle_dataset()
        cycle = [
            sample
            for sample in oracle.by_split["train"]
            if np.any(sample.repetition_contexts > 0)
        ]
        return {
            "fresh_train": [
                sample for sample in fresh if sample.game_id % 10 != 0
            ],
            "fresh_validation": [
                sample for sample in fresh if sample.game_id % 10 == 0
            ],
            "replay_train": [
                sample for sample in replay if sample.game_id % 10 != 0
            ],
            "oracle_train": oracle.by_split["train"],
            "oracle_validation": oracle.by_split["validation"][:2048],
            "cycle_train": cycle,
            "cycle_validation": [
                sample
                for sample in oracle.by_split["validation"]
                if np.any(sample.repetition_contexts > 0)
            ][:1024],
        }

    def _build_plan(
        self, pools: dict[str, list[Any]]
    ) -> list[dict[str, Any]]:
        """Reconstruct the two selected _fit epochs from stored state.

        The main replay ordering, tail batches, Oracle cursor, cycle cursor,
        and reshuffles all start at the RNG boundary restored from iteration
        2.  Experiments A/B/D reuse these same indices against their
        causally different policy source pools.
        """
        if self.plan_rng_state is None:
            raise RuntimeError("ablation RNG boundary was not captured")
        rng = random.Random()
        rng.setstate(self.plan_rng_state)
        main_order = list(range(len(pools["replay_train"])))
        microbatch = 256 if self.device.type == "cuda" else 16
        plan: list[dict[str, Any]] = []
        for epoch in range(1, 3):
            rng.shuffle(main_order)
            oracle_order = list(range(len(pools["oracle_train"])))
            rng.shuffle(oracle_order)
            oracle_cursor = 0
            cycle_order = list(range(len(pools["cycle_train"])))
            rng.shuffle(cycle_order)
            cycle_cursor = 0
            for begin in range(0, len(main_order), self.batch_size):
                main = list(main_order[begin : begin + self.batch_size])
                anchors = []
                for _ in range(len(main)):
                    if oracle_cursor >= len(oracle_order):
                        rng.shuffle(oracle_order)
                        oracle_cursor = 0
                    anchors.append(oracle_order[oracle_cursor])
                    oracle_cursor += 1
                cycle_micro = []
                for micro in range(0, len(main), microbatch):
                    count = min(
                        len(cycle_order),
                        max(1, len(main[micro : micro + microbatch]) // 4),
                    )
                    selected = []
                    for _ in range(count):
                        if cycle_cursor >= len(cycle_order):
                            rng.shuffle(cycle_order)
                            cycle_cursor = 0
                        selected.append(cycle_order[cycle_cursor])
                        cycle_cursor += 1
                    cycle_micro.append(selected)
                plan.append(
                    {
                        "epoch": epoch,
                        "main": main,
                        "anchor": anchors,
                        "cycle_micro": cycle_micro,
                    }
                )
        if len(plan) != self.steps:
            raise RuntimeError(
                f"expected {self.steps} original steps, built {len(plan)}"
            )
        return plan

    def _contract(
        self,
        pools: dict[str, list[Any]],
        plan: list[dict[str, Any]],
    ) -> dict[str, Any]:
        checkpoint_hash = hashlib.sha256(
            self.source_checkpoint_path.read_bytes()
        ).hexdigest()
        replay_files = sorted(
            (
                self.run_dir
                / "search_replay"
                / "iteration_000003"
            ).glob("games_*.npz")
        )
        replay_digest = hashlib.sha256()
        for path in replay_files:
            replay_digest.update(path.read_bytes())
        plan_digest = hashlib.sha256()
        for row in plan:
            plan_digest.update(
                np.asarray(row["main"], dtype=np.int64).tobytes()
            )
            plan_digest.update(
                np.asarray(row["anchor"], dtype=np.int64).tobytes()
            )
            for cycle in row["cycle_micro"]:
                plan_digest.update(
                    np.asarray(cycle, dtype=np.int64).tobytes()
                )
        return {
            "schema": "open-junqi-search-v3-iteration3-ablation-contract",
            "source_iteration": 2,
            "data_iteration": 3,
            "source_checkpoint": str(self.source_checkpoint_path),
            "source_checkpoint_sha256": checkpoint_hash,
            "iteration3_replay_sha256": replay_digest.hexdigest(),
            "plan_seed": self.plan_seed,
            "rng_algorithm": "python.Random restored from iteration 2",
            "batch_plan_sha256": plan_digest.hexdigest(),
            "batch_plan": (
                "current _fit ordering reconstructed from the iteration-2 "
                "checkpoint RNG for selected epochs 1-2; A/B/D reuse "
                "identical indices"
            ),
            "steps": self.steps,
            "batch_size": self.batch_size,
            "validation_interval_steps": self.validation_interval,
            "gradient_cosine_interval_steps": self.gradient_interval,
            "split": "game_id % 10 == 0 is validation",
            "pool_counts": {
                key: len(value) for key, value in pools.items()
            },
            "fresh_train_digest": _sample_digest(
                pools["fresh_train"]
            ),
            "fresh_validation_digest": _sample_digest(
                pools["fresh_validation"]
            ),
            "replay_train_digest": _sample_digest(
                pools["replay_train"]
            ),
            "experiments": EXPERIMENTS,
        }

    def _evaluate(
        self,
        model: PolicyValueNet,
        baseline: PolicyValueNet,
        pools: dict[str, list[Any]],
        step: int,
    ) -> dict[str, Any]:
        fresh = self.runner._evaluate_distill(
            model, pools["fresh_validation"]
        )
        kl_samples = pools["fresh_validation"][:2048]
        return {
            "step": step,
            "fraction_of_original_selected_training": step / self.steps,
            "fresh": fresh,
            "policy_kl_from_iteration2": _policy_kl(
                baseline, model, kl_samples, self.device
            ),
        }

    def _run_one(
        self,
        name: str,
        mode: str,
        pools: dict[str, list[Any]],
        plan: list[dict[str, Any]],
        contract_hash: str,
    ) -> dict[str, Any]:
        directory = self.output / name
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "result.json"
        if result_path.exists():
            existing = json.loads(result_path.read_text(encoding="utf-8"))
            if existing.get("contract_hash") != contract_hash:
                raise RuntimeError(f"stale ablation result: {result_path}")
            return existing

        experiment_seed = self.plan_seed
        _set_global_rng(experiment_seed)
        model, optimizer = _model_optimizer(
            self.config, self.source_checkpoint, self.device
        )
        baseline, baseline_optimizer = _model_optimizer(
            self.config, self.source_checkpoint, self.device
        )
        del baseline_optimizer
        baseline.eval()
        if mode == "fresh_policy":
            model.set_training_phase("policy_only")
        elif mode == "value_only":
            model.set_training_phase("value_distill")
        else:
            model.set_training_phase("joint")

        checkpoints = [self._evaluate(model, baseline, pools, 0)]
        cosine_rows = []
        started = time.perf_counter()
        microbatch = 256 if self.device.type == "cuda" else 16
        anchor_weight = float(
            self.config.train.oracle_value_aux_weight
        )
        for step, step_plan in enumerate(plan, start=1):
            main_keys = np.asarray(step_plan["main"], dtype=np.uint64)
            fresh_batch = _select(pools["fresh_train"], main_keys)
            replay_batch = [
                pools["replay_train"][index]
                for index in step_plan["main"]
            ]
            anchor_batch_samples = [
                pools["oracle_train"][index]
                for index in step_plan["anchor"]
            ]

            if step == 1 or step % self.gradient_interval == 0:
                cosine = _gradient_cosine(
                    model, fresh_batch[:64], self.device
                )
                cosine["step"] = step
                cosine_rows.append(cosine)

            model.train()
            optimizer.zero_grad(set_to_none=True)
            effective_size = len(replay_batch)
            for micro_index, begin in enumerate(
                range(0, effective_size, microbatch)
            ):
                end = min(effective_size, begin + microbatch)
                fraction = (end - begin) / effective_size
                if mode == "fresh_policy":
                    fresh = search_distill_batch(
                        fresh_batch[begin:end], self.device
                    )
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        fresh_logits, _, _, _, _ = model.forward_heads(
                            fresh["observations"],
                            fresh["legal_mask"],
                            fresh["repetition"],
                        )
                    fresh_policy = -torch.sum(
                        fresh["target"]
                        * torch.log_softmax(fresh_logits.float(), -1),
                        -1,
                    ).mean()
                    loss = fresh_policy
                else:
                    replay = search_distill_batch(
                        replay_batch[begin:end], self.device
                    )
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        (
                            replay_logits,
                            _,
                            _,
                            replay_wdl_logits,
                            _,
                        ) = model.forward_heads(
                            replay["observations"],
                            replay["legal_mask"],
                            replay["repetition"],
                        )
                    replay_wdl = -torch.sum(
                        replay["wdl"]
                        * torch.log_softmax(
                            replay_wdl_logits.float(), -1
                        ),
                        -1,
                    ).mean()
                    anchor = oracle_losses(
                        model,
                        oracle_collate(
                            anchor_batch_samples[begin:end],
                            0.1,
                            self.device,
                        ),
                    )
                    if mode == "value_only":
                        loss = replay_wdl + anchor_weight * (
                            anchor["wdl"] + anchor["oracle_value"]
                        )
                    else:
                        if mode == "joint_replay":
                            main_policy = -torch.sum(
                                replay["target"]
                                * torch.log_softmax(
                                    replay_logits.float(), -1
                                ),
                                -1,
                            ).mean()
                        else:
                            fresh = search_distill_batch(
                                fresh_batch[begin:end], self.device
                            )
                            with torch.autocast(
                                device_type=self.device.type,
                                dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda",
                            ):
                                (
                                    fresh_logits,
                                    _,
                                    _,
                                    _,
                                    _,
                                ) = model.forward_heads(
                                    fresh["observations"],
                                    fresh["legal_mask"],
                                    fresh["repetition"],
                                )
                            main_policy = -torch.sum(
                                fresh["target"]
                                * torch.log_softmax(
                                    fresh_logits.float(), -1
                                ),
                                -1,
                            ).mean()
                        selected_cycles = [
                            pools["cycle_train"][index]
                            for index in step_plan["cycle_micro"][
                                micro_index
                            ]
                        ]
                        cycle = oracle_losses(
                            model,
                            oracle_collate(
                                selected_cycles,
                                0.1,
                                self.device,
                            ),
                        )
                        loss = (
                            main_policy
                            + replay_wdl
                            + anchor_weight
                            * (
                                anchor["policy"]
                                + anchor["wdl"]
                                + anchor["oracle_value"]
                            )
                            + 0.5 * cycle["policy"]
                        )
                (loss * fraction).backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                self.config.train.max_gradient_norm,
            )
            optimizer.step()

            if (
                step % self.validation_interval == 0
                or step == self.steps
            ):
                metric = self._evaluate(
                    model, baseline, pools, step
                )
                checkpoints.append(metric)
                print(
                    f"[search-v3 ablation] {name} step={step}/{self.steps} "
                    f"fresh-excess={metric['fresh']['policy_excess']:.6f} "
                    f"wdl={metric['fresh']['wdl_ce']:.6f} "
                    f"kl={metric['policy_kl_from_iteration2']:.6f}",
                    flush=True,
                )

        oracle = self.runner._evaluate_oracle_anchor(
            model, pools["oracle_validation"], 0.1
        )
        cycle = self.runner._evaluate_oracle_anchor(
            model, pools["cycle_validation"], 0.1
        )
        checkpoint_path = directory / "final.pt"
        temporary = checkpoint_path.with_suffix(".pt.tmp")
        torch.save(
            {
                "schema": "open-junqi-search-v3-iteration3-ablation",
                "experiment": name,
                "mode": mode,
                "contract_hash": contract_hash,
                "source_checkpoint": str(self.source_checkpoint_path),
                "steps": self.steps,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            temporary,
        )
        os.replace(temporary, checkpoint_path)
        result = {
            "schema": "open-junqi-search-v3-iteration3-ablation-result",
            "experiment": name,
            "mode": mode,
            "contract_hash": contract_hash,
            "steps": self.steps,
            "elapsed_seconds": time.perf_counter() - started,
            "checkpoints": checkpoints,
            "fresh_before": checkpoints[0]["fresh"],
            "fresh_after": checkpoints[-1]["fresh"],
            "policy_kl_after": checkpoints[-1][
                "policy_kl_from_iteration2"
            ],
            "oracle_after": oracle,
            "cycle_policy_excess_after": cycle["policy_excess"],
            "trunk_gradient_cosine": _cosine_summary(cosine_rows),
            "trunk_gradient_cosine_rows": cosine_rows,
            "checkpoint": str(checkpoint_path),
        }
        atomic_json(result_path, result)
        del model, baseline, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    def run(self) -> dict[str, Any]:
        pools = self._load_pools()
        plan = self._build_plan(pools)
        contract = self._contract(pools, plan)
        contract_rendered = json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        contract_hash = hashlib.sha256(
            contract_rendered.encode("utf-8")
        ).hexdigest()
        contract["contract_hash"] = contract_hash
        atomic_json(self.output / "contract.json", contract)
        results = {}
        for name, mode in EXPERIMENTS.items():
            results[name] = self._run_one(
                name, mode, pools, plan, contract_hash
            )
        archived = json.loads(
            (
                self.run_dir
                / "diagnostics"
                / "iteration_000003_joint_failure_original"
                / "iteration_000003.json"
            ).read_text(encoding="utf-8")
        )
        summary = {
            "schema": "open-junqi-search-v3-iteration3-ablation-summary",
            "contract_hash": contract_hash,
            "archived_original": {
                "fresh_before": archived["distill_before"],
                "fresh_after": archived["distill_after"],
                "chosen_epoch": archived.get("chosen_epoch"),
                "reasons": archived["reasons"],
            },
            "results": {
                name: {
                    "fresh_before": row["fresh_before"],
                    "fresh_after": row["fresh_after"],
                    "policy_kl_after": row["policy_kl_after"],
                    "trunk_gradient_cosine": row[
                        "trunk_gradient_cosine"
                    ],
                }
                for name, row in results.items()
            },
        }
        atomic_json(self.output / "summary.json", summary)
        return summary


def run_iteration3_ablation(config_path: str | Path) -> dict[str, Any]:
    return Iteration3Ablation(config_path).run()
