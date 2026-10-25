from __future__ import annotations

import copy
import json
import math
import os
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from junqi_az.model import PolicyValueNet

from .dataset import atomic_json
from .evaluation import SearchEvaluator
from .loop import (
    CYCLE_REPEAT_VISIT_SCALE,
    CYCLE_TEACHER_EPSILON,
    CYCLE_TEACHER_SIMULATIONS,
    SearchSelfPlay,
    SearchSample,
    read_selfplay_shard,
    read_selfplay_state_hashes,
    search_distill_batch,
    write_selfplay_shard,
)
from .supervised import (
    MemoryOracleDataset,
    collate as oracle_collate,
    losses as oracle_losses,
)


class ClosedLoopRunner:
    def __init__(
        self,
        *,
        root: Path,
        run: Path,
        backend: Any,
        config: Any,
        score_scale: float,
        replay_trace: Any,
    ) -> None:
        self.root = root
        # Keep the output path distinct from run(), the public loop entrypoint.
        # Using ``self.run`` here shadows the bound method on the instance and
        # turns ``runner.run()`` into an attempted WindowsPath call.
        self.run_dir = run
        self.backend = backend
        self.config = config
        self.score_scale = score_scale
        self.replay_trace = replay_trace
        self.device = torch.device(
            config.device if torch.cuda.is_available() else "cpu"
        )
        self.rng = random.Random(config.seed ^ 0x434C4F5345)
        self.probes = json.loads(
            (run / "probe_manifest.json").read_text(encoding="utf-8")
        )["probes"]
        self.probe_states = [
            replay_trace(row["action_trace"])[0] for row in self.probes
        ]
        self._oracle_dataset_cache: MemoryOracleDataset | None = None

    def _oracle_dataset(self) -> MemoryOracleDataset:
        if self._oracle_dataset_cache is None:
            self._oracle_dataset_cache = MemoryOracleDataset(
                self.run_dir / "oracle_dataset" / "training_shards"
            )
        return self._oracle_dataset_cache

    def _calibrate_leaf_value_scale(
        self, model: PolicyValueNet
    ) -> dict[str, float]:
        samples = self._oracle_dataset().by_split["validation"][:4096]
        predictions = []
        targets = []
        model.eval()
        microbatch = 256 if self.device.type == "cuda" else 16
        with torch.inference_mode():
            for begin in range(0, len(samples), microbatch):
                selected = samples[begin:begin + microbatch]
                batch = oracle_collate(selected, 0.1, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
                ):
                    _, values, _, _, _ = model.forward_heads(
                        batch["observations"],
                        batch["legal_mask"],
                        batch["repetition"],
                    )
                predictions.append(values.float().cpu().numpy())
                targets.append(
                    batch["oracle_scalar_target"].float().cpu().numpy()
                )
        predicted = np.concatenate(predictions).astype(np.float64)
        target = np.concatenate(targets).astype(np.float64)
        # These are only 4,096 scalars.  Avoid np.dot/corrcoef here: on the
        # Windows Conda stack they lazily load NumPy's BLAS OpenMP DLL after
        # PyTorch has loaded libiomp5md, causing OpenMP error #15.
        denominator = float(np.sum(predicted * predicted))
        unconstrained = (
            float(np.sum(predicted * target)) / denominator
            if denominator > 1e-12
            else 0.0
        )
        scale = float(np.clip(unconstrained, 0.0, 1.0))
        model.search_v3_leaf_value_scale = scale
        predicted_centered = predicted - float(np.mean(predicted))
        target_centered = target - float(np.mean(target))
        predicted_std = float(np.sqrt(np.mean(predicted_centered ** 2)))
        target_std = float(np.sqrt(np.mean(target_centered ** 2)))
        correlation = (
            float(np.mean(predicted_centered * target_centered))
            / (predicted_std * target_std)
            if predicted_std > 1e-12 and target_std > 1e-12
            else 0.0
        )
        return {
            "samples": float(len(target)),
            "unconstrained_scale": unconstrained,
            "leaf_value_scale": scale,
            "correlation": correlation,
            "unscaled_mse": float(np.mean((predicted - target) ** 2)),
            "scaled_mse": float(
                np.mean((scale * predicted - target) ** 2)
            ),
        }

    @staticmethod
    def _external_gate_reasons(
        pre_arena: dict[str, Any], post_arena: dict[str, Any]
    ) -> list[str]:
        reasons = []
        bare_regressed = (
            post_arena["external_bare_mean"] + 0.05
            < pre_arena["external_bare_mean"]
        )
        search_regressed = (
            post_arena["external_search64_mean"] + 0.05
            < pre_arena["external_search64_mean"]
        )
        # "The external system regressed" requires both its distilled bare
        # policy and its searched policy to move down.  With one fixed setup a
        # deterministic 128-game arena contains only the two swapped-side
        # trajectories, so an OR here mistakes a one-trajectory flip for
        # independent evidence even when Search improves strongly.
        if bare_regressed and search_regressed:
            reasons.append(
                "external wide_quiescence/exact082 system regressed"
            )
        # The intended closure compares the newly distilled bare policy with
        # the *previous searched* actor, not only with the previous bare actor.
        external_rows = []
        for arena in (pre_arena, post_arena):
            external_rows.extend(arena.get("external", {}).values())
        # A fixed layout + greedy policy produces only the two swapped-side
        # trajectories regardless of the nominal game count. A 5pp threshold
        # on those duplicated rows is not statistical evidence that the new
        # bare policy failed to imitate Search(pi_k). The fresh held-out visit
        # excess is the hard closure test until external evaluation has at
        # least 32 genuinely distinct trajectories per matchup.
        closure_has_independent_evidence = (
            not external_rows
            or min(
                int(row.get("unique_trajectories", 0))
                for row in external_rows
            )
            >= 32
        )
        if (
            closure_has_independent_evidence
            and
            post_arena["external_bare_mean"] + 0.05
            < pre_arena["external_search64_mean"]
        ):
            reasons.append(
                "bare pi_(k+1) did not approach external Search(pi_k) score"
            )
        return reasons

    def _checkpoint(self, iteration: int) -> Path:
        return self.run_dir / "checkpoints" / (
            "supervised_latest.pt"
            if iteration == 0
            else f"loop_iteration_{iteration:06d}.pt"
        )

    def _load_model_optimizer(
        self, iteration: int
    ) -> tuple[
        PolicyValueNet,
        torch.optim.Optimizer,
        torch.optim.lr_scheduler.LRScheduler,
        int,
    ]:
        payload = torch.load(
            self._checkpoint(iteration),
            map_location="cpu",
            weights_only=False,
        )
        model = PolicyValueNet(self.config.model).to(self.device)
        model.load_state_dict(payload["model"])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.weight_decay,
        )
        stored_scheduler = payload.get("scheduler")
        checkpoint_has_atomic_training_state = (
            isinstance(stored_scheduler, dict)
            and stored_scheduler.get("kind") == "constant"
        )
        if (
            iteration > 0
            and "optimizer" in payload
            and checkpoint_has_atomic_training_state
        ):
            optimizer.load_state_dict(payload["optimizer"])
        # Closed-loop currently uses a constant schedule, but it is a real,
        # independently checkpointed scheduler rather than an implicit/null
        # placeholder. This preserves exact phase state without introducing a
        # second learning-rate change into the causal experiment.
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda _: 1.0
        )
        if (
            iteration > 0
            and checkpoint_has_atomic_training_state
        ):
            scheduler.load_state_dict(stored_scheduler["state"])
        if iteration > 0:
            rng = payload.get("rng", {})
            if rng.get("python") is not None:
                random.setstate(rng["python"])
            if rng.get("numpy") is not None:
                np.random.set_state(rng["numpy"])
            if rng.get("torch") is not None:
                torch.set_rng_state(rng["torch"])
            if torch.cuda.is_available() and rng.get("cuda") is not None:
                torch.cuda.set_rng_state_all(rng["cuda"])
            if rng.get("closed_loop_python") is not None:
                self.rng.setstate(rng["closed_loop_python"])
        return (
            model,
            optimizer,
            scheduler,
            int(payload.get("global_step", 0)),
        )

    def _collect_iteration(
        self, iteration: int, actor: PolicyValueNet
    ) -> dict[str, Any]:
        directory = self.run_dir / "search_replay" / f"iteration_{iteration:06d}"
        directory.mkdir(parents=True, exist_ok=True)
        for partial in directory.glob("*.partial"):
            partial.unlink()
        collector = SearchSelfPlay(
            self.config, self.backend, actor, self.device
        )
        reports = []
        block = self.config.loop.transaction_games
        # Fill the configured neural inference batch.  After consolidating the
        # CUDA transfers in BatchedNeuralPUCT this improves the fixed-root
        # benchmark from 2,892 to 5,375 simulations/s at 256 roots.  Committed
        # 64-game shards from earlier runs remain valid and resume contiguously.
        collection_block = max(block, self.config.puct.inference_batch_size)
        completed = 0
        for transaction in sorted(directory.glob("games_*.json")):
            payload = json.loads(transaction.read_text(encoding="utf-8"))
            begin = int(payload.get("first_game", -1))
            count = int(payload.get("games", 0))
            end = begin + count
            shard = directory / f"games_{begin:04d}_{end:04d}.npz"
            if (
                payload.get("completed") is True
                and begin == completed
                and count > 0
                and shard.exists()
            ):
                reports.append(payload["metrics"])
                completed = end
            elif begin >= completed:
                # Never accept a gap or metadata without its committed shard.
                break
        atomic_json(
            self.run_dir / "loop" / "progress.json",
            {
                "schema": "open-junqi-search-v3-progress",
                "phase": "selfplay",
                "iteration": iteration,
                "completed_games": completed,
                "target_games": self.config.loop.games_per_iteration,
                "replay_shard_committed": completed > 0,
                "config_hash": self.config.hash(),
            },
        )
        while completed < self.config.loop.games_per_iteration:
            begin = completed
            end = min(
                self.config.loop.games_per_iteration,
                begin + collection_block,
            )
            shard = directory / f"games_{begin:04d}_{end:04d}.npz"
            transaction = directory / f"games_{begin:04d}_{end:04d}.json"
            def update_collection_progress(
                live: dict[str, int],
            ) -> None:
                atomic_json(
                    self.run_dir / "loop" / "progress.json",
                    {
                        "schema": "open-junqi-search-v3-progress",
                        "phase": "selfplay",
                        "iteration": iteration,
                        "completed_games": completed,
                        "target_games": self.config.loop.games_per_iteration,
                        "collection_window": [begin, end],
                        "window_finished_games": live["finished_games"],
                        "active_games": live["active_games"],
                        "ply_round": live["ply_round"],
                        "simulation_count": live["simulation_count"],
                        "replay_shard_committed": completed > 0,
                        "config_hash": self.config.hash(),
                    },
                )

            samples, metrics = collector.collect_block(
                iteration,
                begin,
                end - begin,
                progress_callback=update_collection_progress,
            )
            write_selfplay_shard(shard, samples)
            atomic_json(
                transaction,
                {
                    "schema": "open-junqi-search-v3-selfplay-transaction",
                    "completed": True,
                    "iteration": iteration,
                    "first_game": begin,
                    "games": end - begin,
                    "shard": str(shard),
                    "metrics": metrics,
                },
            )
            atomic_json(
                self.run_dir / "loop" / "progress.json",
                {
                    "schema": "open-junqi-search-v3-progress",
                    "phase": "selfplay",
                    "iteration": iteration,
                    "completed_games": end,
                    "target_games": self.config.loop.games_per_iteration,
                    "last_committed_shard": str(shard),
                    "replay_shard_committed": True,
                    "config_hash": self.config.hash(),
                    "rng": {
                        "closed_loop_python": self.rng.getstate(),
                    },
                },
            )
            reports.append(metrics)
            completed = end
            total_simulations = sum(
                row["simulation_count"] for row in reports
            )
            total_search_seconds = sum(
                row["search_seconds"] for row in reports
            )
            repetition_games = sum(
                row["terminal_reasons"].get("REPETITION", 0)
                for row in reports
            )
            print(
                f"[search-v3 loop] iteration={iteration} "
                f"games={end}/{self.config.loop.games_per_iteration} "
                f"states={sum(row['unique_states'] for row in reports)} "
                f"selfplay-sps="
                f"{total_simulations / max(1e-9, total_search_seconds):.0f} "
                f"last-block-sps={metrics['simulations_per_second']:.0f} "
                f"repetition={repetition_games}/{end}",
                flush=True,
            )
        games = sum(row["games"] for row in reports)
        state_counts = Counter()
        accepted_policy = Counter()
        samples = 0
        for shard in sorted(directory.glob("games_*.npz")):
            for raw_hash in read_selfplay_state_hashes(shard):
                state_hash = int(raw_hash)
                state_counts[state_hash] += 1
                if (
                    accepted_policy[state_hash]
                    < self.config.loop.state_policy_cap_per_iteration
                ):
                    accepted_policy[state_hash] += 1
                    samples += 1
        trajectories = Counter()
        prefix_counts = {8: Counter(), 16: Counter(), 24: Counter()}
        for row in reports:
            trajectories.update(row["trajectory_counts"])
            for length in (8, 16, 24):
                prefix_counts[length].update(
                    row["prefix_counts"][str(length)]
                )
        terminal = {}
        cycle_types = Counter()
        cycle_cleanup = Counter()
        cycle_teacher = Counter()
        legacy_search64 = Counter()
        for row in reports:
            for key, value in row["terminal_reasons"].items():
                terminal[key] = terminal.get(key, 0) + value
            cycle_types.update(row.get("cycle_types", {}))
            cycle_cleanup.update(row.get("replay_cycle_cleanup", {}))
            teacher = row.get("cycle_teacher", {})
            for key in (
                "roots",
                "eligible",
                "adjusted",
                "rational_repeat_preserved",
                "insufficient_root_coverage",
            ):
                cycle_teacher[key] += int(teacher.get(key, 0))
            legacy = row.get("legacy_search64_diagnostic", {})
            legacy_search64["heldout_cycle_roots"] += int(
                legacy.get("heldout_cycle_roots", 0)
            )
            legacy_search64["simulation_count"] += int(
                legacy.get("simulation_count", 0)
            )
            legacy_search64["search_seconds_micro"] += int(
                round(float(legacy.get("search_seconds", 0.0)) * 1e6)
            )
        summary = {
            "games": games,
            "policy_samples": samples,
            "unique_states": len(state_counts),
            "unique_state_ratio": len(state_counts) / max(1, samples),
            "unique_trajectories": len(trajectories),
            "unique_prefixes": {
                str(length): len(prefix_counts[length])
                for length in (8, 16, 24)
            },
            "maximum_trajectory_share": (
                max(trajectories.values()) / max(1, games)
            ),
            "maximum_prefix_share": {
                str(length): max(
                    prefix_counts[length].values(), default=0
                ) / max(1, games)
                for length in (8, 16, 24)
            },
            "root_visit_entropy": float(
                np.average(
                    [row["root_visit_entropy"] for row in reports],
                    weights=[row["policy_samples"] for row in reports],
                )
            ),
            "terminal_reasons": terminal,
            "repetition_fraction": terminal.get("REPETITION", 0) / games,
            "cycle_types": dict(cycle_types),
            "replay_cycle_cleanup": dict(cycle_cleanup),
            "cycle_teacher": {
                **dict(cycle_teacher),
                "simulations": CYCLE_TEACHER_SIMULATIONS,
                "epsilon": CYCLE_TEACHER_EPSILON,
                "repeat_visit_scale": CYCLE_REPEAT_VISIT_SCALE,
            },
            "legacy_search64_diagnostic": {
                "heldout_cycle_roots": int(
                    legacy_search64["heldout_cycle_roots"]
                ),
                "simulation_count": int(
                    legacy_search64["simulation_count"]
                ),
                "search_seconds": (
                    legacy_search64["search_seconds_micro"] / 1e6
                ),
                "simulations_per_second": (
                    legacy_search64["simulation_count"]
                    / max(
                        1e-9,
                        legacy_search64["search_seconds_micro"] / 1e6,
                    )
                ),
                "does_not_change_training_target_or_action_sampling": True,
            },
            "maximum_single_cycle_share_of_repetitions": max(
                (
                    row.get(
                        "maximum_single_cycle_share_of_repetitions", 0.0
                    )
                    for row in reports
                ),
                default=0.0,
            ),
            "simulations_per_second": (
                sum(row["simulation_count"] for row in reports)
                / max(1e-9, sum(row["search_seconds"] for row in reports))
            ),
            "average_ply": (
                sum(
                    row.get(
                        "total_plies",
                        row.get("average_ply", 0.0) * row["games"],
                    )
                    for row in reports
                )
                / max(1, games)
            ),
        }
        atomic_json(directory / "summary.json", summary)
        return summary

    def _replay_samples(
        self,
        iteration: int,
        current_samples: list[SearchSample] | None = None,
    ) -> list[SearchSample]:
        first = max(1, iteration - self.config.loop.replay_iterations + 1)
        maximum = self.config.loop.maximum_replay_samples
        result: list[SearchSample] = []
        eligible = 0

        def consider(sample: SearchSample) -> None:
            nonlocal eligible
            eligible += 1
            if len(result) < maximum:
                result.append(sample)
                return
            # Algorithm R: every eligible sample has exactly maximum/eligible
            # probability of surviving without first materializing the whole
            # five-iteration replay in memory.
            replacement = self.rng.randrange(eligible)
            if replacement < maximum:
                result[replacement] = sample

        for replay_iteration in range(first, iteration + 1):
            if replay_iteration == iteration and current_samples is not None:
                # Reuse the already decoded current rollout.  Loading it again
                # duplicated roughly 1.5 GiB for a 1,024-game iteration.
                for sample in current_samples:
                    consider(sample)
                continue
            accepted = Counter()
            directory = (
                self.run_dir
                / "search_replay"
                / f"iteration_{replay_iteration:06d}"
            )
            for shard in sorted(directory.glob("games_*.npz")):
                for sample in read_selfplay_shard(shard):
                    if (
                        accepted[sample.state_hash]
                        >= self.config.loop.state_policy_cap_per_iteration
                    ):
                        continue
                    accepted[sample.state_hash] += 1
                    consider(sample)
        self.last_replay_stats = {
            "first_iteration": first,
            "last_iteration": iteration,
            "eligible_samples": eligible,
            "retained_samples": len(result),
            "maximum_samples": maximum,
            "sampling": "uniform_reservoir_algorithm_r",
        }
        return result

    def _iteration_samples(self, iteration: int) -> list[SearchSample]:
        directory = (
            self.run_dir
            / "search_replay"
            / f"iteration_{iteration:06d}"
        )
        accepted = Counter()
        result = []
        for shard in sorted(directory.glob("games_*.npz")):
            for sample in read_selfplay_shard(shard):
                if (
                    accepted[sample.state_hash]
                    >= self.config.loop.state_policy_cap_per_iteration
                ):
                    continue
                accepted[sample.state_hash] += 1
                result.append(sample)
        return result

    def _fit(
        self,
        model: PolicyValueNet,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        samples: list[SearchSample],
        distill_validation: list[SearchSample],
    ) -> tuple[dict[str, float], int]:
        train = [
            sample for sample in samples if sample.game_id % 10 != 0
        ]
        validation = [
            sample for sample in samples if sample.game_id % 10 == 0
        ]
        oracle_dataset = self._oracle_dataset()
        oracle_train = oracle_dataset.by_split["train"]
        oracle_validation = oracle_dataset.by_split["validation"][:2048]
        cycle_train = [
            sample
            for sample in oracle_train
            if np.any(sample.repetition_contexts > 0)
        ]
        if not cycle_train:
            cycle_train = [
                sample for sample in oracle_train if sample.important
            ] or oracle_train
        cycle_validation = [
            sample
            for sample in oracle_dataset.by_split["validation"]
            if np.any(sample.repetition_contexts > 0)
        ][:1024]
        oracle_tau = 0.1
        anchor_weight = float(self.config.train.oracle_value_aux_weight)
        cycle_anchor_weight = 0.5
        maximum_epochs = min(3, int(self.config.train.epochs))
        global_steps = 0
        microbatch = 256 if self.device.type == "cuda" else 16
        best_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        best_optimizer_state = copy.deepcopy(optimizer.state_dict())
        best_scheduler_state = copy.deepcopy(scheduler.state_dict())
        best_epoch = 0
        best_global_steps = 0
        baseline_distill = self._evaluate_distill(
            model, distill_validation
        )
        baseline_oracle = self._evaluate_oracle_anchor(
            model, oracle_validation, oracle_tau
        )
        baseline_cycle = self._evaluate_oracle_anchor(
            model, cycle_validation, oracle_tau
        )
        baseline_oracle["cycle_policy_excess"] = baseline_cycle[
            "policy_excess"
        ]
        best_score = self._distill_anchor_score(
            baseline_distill, baseline_oracle
        )
        epoch_history = []
        for epoch in range(1, maximum_epochs + 1):
            self.rng.shuffle(train)
            oracle_indices = list(range(len(oracle_train)))
            self.rng.shuffle(oracle_indices)
            oracle_cursor = 0
            cycle_indices = list(range(len(cycle_train)))
            self.rng.shuffle(cycle_indices)
            cycle_cursor = 0
            model.train()
            for begin in range(0, len(train), self.config.train.batch_size):
                effective = train[begin:begin + self.config.train.batch_size]
                optimizer.zero_grad(set_to_none=True)
                for micro in range(0, len(effective), microbatch):
                    selected = effective[micro:micro + microbatch]
                    batch = search_distill_batch(selected, self.device)
                    with torch.autocast(
                        device_type=self.device.type,
                        dtype=torch.bfloat16,
                        enabled=self.device.type == "cuda",
                    ):
                        logits, _, _, wdl_logits, _ = model.forward_heads(
                            batch["observations"],
                            batch["legal_mask"],
                            batch["repetition"],
                        )
                    policy = -torch.sum(
                        batch["target"]
                        * torch.log_softmax(logits.float(), -1),
                        -1,
                    ).mean()
                    wdl = -torch.sum(
                        batch["wdl"]
                        * torch.log_softmax(wdl_logits.float(), -1),
                        -1,
                    ).mean()
                    anchor_indices = []
                    for _ in range(len(selected)):
                        if oracle_cursor >= len(oracle_indices):
                            self.rng.shuffle(oracle_indices)
                            oracle_cursor = 0
                        anchor_indices.append(oracle_indices[oracle_cursor])
                        oracle_cursor += 1
                    anchor_batch = oracle_collate(
                        [oracle_train[index] for index in anchor_indices],
                        oracle_tau,
                        self.device,
                    )
                    anchor = oracle_losses(model, anchor_batch)
                    cycle_count = min(
                        len(cycle_train), max(1, len(selected) // 4)
                    )
                    selected_cycles = []
                    for _ in range(cycle_count):
                        if cycle_cursor >= len(cycle_indices):
                            self.rng.shuffle(cycle_indices)
                            cycle_cursor = 0
                        selected_cycles.append(
                            cycle_train[cycle_indices[cycle_cursor]]
                        )
                        cycle_cursor += 1
                    cycle = oracle_losses(
                        model,
                        oracle_collate(
                            selected_cycles, oracle_tau, self.device
                        ),
                    )
                    fraction = len(selected) / len(effective)
                    (
                        self.config.train.policy_weight * policy
                        + self.config.train.terminal_wdl_weight * wdl
                        + anchor_weight
                        * (
                            anchor["policy"]
                            + anchor["wdl"]
                            + anchor["oracle_value"]
                        )
                        + cycle_anchor_weight * cycle["policy"]
                    ).mul(fraction).backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.train.max_gradient_norm
                )
                optimizer.step()
                global_steps += 1
            distill_metric = self._evaluate_distill(
                model, distill_validation
            )
            oracle_metric = self._evaluate_oracle_anchor(
                model, oracle_validation, oracle_tau
            )
            cycle_metric = self._evaluate_oracle_anchor(
                model, cycle_validation, oracle_tau
            )
            oracle_metric["cycle_policy_excess"] = cycle_metric[
                "policy_excess"
            ]
            score = self._distill_anchor_score(
                distill_metric, oracle_metric
            )
            epoch_history.append(
                {
                    "epoch": epoch,
                    "score": score,
                    "distill": distill_metric,
                    "oracle_anchor": oracle_metric,
                }
            )
            print(
                f"[search-v3 distill] epoch={epoch}/{maximum_epochs} "
                f"search-excess={distill_metric['policy_excess']:.4f} "
                f"oracle-excess={oracle_metric['policy_excess']:.4f} "
                f"search-wdl={distill_metric['wdl_ce']:.4f} "
                f"oracle-wdl={oracle_metric['wdl_ce']:.4f} "
                f"oracle-v={oracle_metric['oracle_value_mse']:.4f} "
                f"cycle-excess={oracle_metric['cycle_policy_excess']:.4f}",
                flush=True,
            )
            scheduler.step()
            if score < best_score:
                best_score = score
                best_epoch = epoch
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in model.state_dict().items()
                }
                best_optimizer_state = copy.deepcopy(
                    optimizer.state_dict()
                )
                best_scheduler_state = copy.deepcopy(
                    scheduler.state_dict()
                )
                best_global_steps = global_steps
        model.load_state_dict(best_state)
        optimizer.load_state_dict(best_optimizer_state)
        scheduler.load_state_dict(best_scheduler_state)
        validation_metric = self._evaluate_distill(model, validation)
        validation_metric.update(
            {
                "closed_loop_epochs": maximum_epochs,
                "chosen_epoch": best_epoch,
                "oracle_anchor_weight": anchor_weight,
                "oracle_anchor_tau": oracle_tau,
                "cycle_anchor_weight": cycle_anchor_weight,
                "cycle_anchor_samples": len(cycle_train),
                "epoch_history": epoch_history,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "attempted_optimizer_steps": global_steps,
                "accepted_optimizer_steps": best_global_steps,
            }
        )
        return validation_metric, best_global_steps

    def _evaluate_oracle_anchor(
        self,
        model: PolicyValueNet,
        samples: list[Any],
        tau: float,
    ) -> dict[str, float]:
        model.eval()
        total = {
            "policy_excess": 0.0,
            "wdl_ce": 0.0,
            "oracle_value_mse": 0.0,
        }
        seen = 0
        microbatch = 256 if self.device.type == "cuda" else 16
        with torch.inference_mode():
            for begin in range(0, len(samples), microbatch):
                selected = samples[begin:begin + microbatch]
                metric = oracle_losses(
                    model, oracle_collate(selected, tau, self.device)
                )
                size = len(selected)
                total["policy_excess"] += float(metric["excess"]) * size
                total["wdl_ce"] += float(metric["wdl"]) * size
                total["oracle_value_mse"] += (
                    float(metric["oracle_value"]) * size
                )
                seen += size
        return {
            key: value / max(1, seen) for key, value in total.items()
        }

    @staticmethod
    def _distill_anchor_score(
        distill: dict[str, float], oracle: dict[str, float]
    ) -> float:
        return float(
            distill["policy_excess"]
            + oracle["policy_excess"]
            + 0.25 * (distill["wdl_ce"] + oracle["wdl_ce"])
            + oracle["oracle_value_mse"]
            + 0.5 * oracle.get("cycle_policy_excess", 0.0)
        )

    def _evaluate_distill(
        self,
        model: PolicyValueNet,
        samples: list[SearchSample],
    ) -> dict[str, float]:
        model.eval()
        metrics = {
            "policy_ce": 0.0,
            "target_entropy": 0.0,
            "policy_excess": 0.0,
            "top1": 0.0,
            "wdl_ce": 0.0,
        }
        seen = 0
        microbatch = 256 if self.device.type == "cuda" else 16
        with torch.inference_mode():
            for begin in range(0, len(samples), microbatch):
                selected = samples[begin:begin + microbatch]
                if not selected:
                    continue
                batch = search_distill_batch(selected, self.device)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.bfloat16,
                    enabled=self.device.type == "cuda",
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
                    batch["wdl"]
                    * torch.log_softmax(wdl_logits.float(), -1),
                    -1,
                )
                size = len(selected)
                metrics["policy_ce"] += float(ce.sum())
                metrics["target_entropy"] += float(entropy.sum())
                metrics["policy_excess"] += float((ce - entropy).sum())
                metrics["top1"] += float(
                    (
                        batch["target"].gather(
                            1,
                            torch.argmax(logits, -1).unsqueeze(1),
                        ).squeeze(1)
                        >= batch["target"].max(-1).values - 1e-7
                    ).float().sum()
                )
                metrics["wdl_ce"] += float(wdl.sum())
                seen += size
        result = {
            key: value / max(1, seen) for key, value in metrics.items()
        }
        result["samples"] = float(seen)
        return result

    def _probe_metrics(
        self, model: PolicyValueNet
    ) -> dict[str, float]:
        evaluator = SearchEvaluator(
            root=self.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=self.score_scale,
        )
        bare = evaluator.search.bare(self.probe_states)
        searched, performance = evaluator.search.search(
            self.probe_states, 64
        )
        bare_regret = []
        search_regret = []
        avoidable_total = 0
        bare_avoidable_exits = 0
        search_avoidable_exits = 0
        action_agreement = 0
        for probe, first, second in zip(self.probes, bare, searched):
            values = {
                int(row["action"]): float(row["q"])
                for row in probe["actions"]
            }
            best = max(values.values())
            bare_regret.append(best - values[first.action])
            search_regret.append(best - values[second.action])
            action_agreement += int(first.action == second.action)
            cycles = [
                row for row in probe["actions"]
                if row["enters_repetition"]
            ]
            exits = [
                row for row in probe["actions"]
                if not row["enters_repetition"]
            ]
            if cycles and exits:
                cycle_q = max(float(row["q"]) for row in cycles)
                exit_q = max(float(row["q"]) for row in exits)
                if exit_q >= cycle_q + 0.05:
                    avoidable_total += 1
                    by_action = {
                        int(row["action"]): row
                        for row in probe["actions"]
                    }
                    bare_avoidable_exits += int(
                        not by_action[first.action]["enters_repetition"]
                    )
                    search_avoidable_exits += int(
                        not by_action[second.action]["enters_repetition"]
                    )
        return {
            "bare_oracle_regret": float(np.mean(bare_regret)),
            "search64_oracle_regret": float(np.mean(search_regret)),
            "search_gain_regret_fraction": (
                1.0
                - float(np.mean(search_regret))
                / max(1e-9, float(np.mean(bare_regret)))
            ),
            "search64_sps": performance["simulations_per_second"],
            "bare_search_action_agreement": (
                action_agreement / max(1, len(self.probes))
            ),
            "avoidable_repetition_probes": avoidable_total,
            "bare_avoidable_exit_fraction": (
                bare_avoidable_exits / max(1, avoidable_total)
            ),
            "search64_avoidable_exit_fraction": (
                search_avoidable_exits / max(1, avoidable_total)
            ),
        }

    def _arena_metrics(
        self, model: PolicyValueNet, iteration: int, label: str
    ) -> dict[str, Any]:
        evaluator = SearchEvaluator(
            root=self.root,
            backend=self.backend,
            config=self.config,
            model=model,
            score_scale=self.score_scale,
        )
        seed = (
            self.config.seed
            ^ (iteration * 0x9E3779B1)
            ^ sum(ord(char) for char in label)
        )
        direct = evaluator.paired_candidate_vs(
            "candidate_64",
            "baseline_0",
            self.config.loop.check_games,
            seed,
        )
        external = {}
        for opponent in ("supplement_wide_quiescence", "exact082"):
            for candidate in ("candidate_0", "candidate_64"):
                key = f"{candidate}_vs_{opponent}"
                external[key] = evaluator.paired_candidate_vs(
                    candidate,
                    opponent,
                    self.config.loop.check_games,
                    seed ^ sum(ord(char) for char in key),
                )
        return {
            "direct_search64_vs_bare": direct,
            "external": external,
            "external_bare_mean": float(
                np.mean(
                    [
                        external[f"candidate_0_vs_{opponent}"]["score"]
                        for opponent in (
                            "supplement_wide_quiescence",
                            "exact082",
                        )
                    ]
                )
            ),
            "external_search64_mean": float(
                np.mean(
                    [
                        external[f"candidate_64_vs_{opponent}"]["score"]
                        for opponent in (
                            "supplement_wide_quiescence",
                            "exact082",
                        )
                    ]
                )
            ),
        }

    def run(self) -> tuple[bool, dict[str, Any], list[str]]:
        failure_streak = 0
        previous_validation = None
        previous_data = None
        iteration_reports = []
        start_iteration = 1
        for iteration in range(1, self.config.loop.iterations + 1):
            checkpoint = self._checkpoint(iteration)
            report_path = (
                self.run_dir / "loop" / f"iteration_{iteration:06d}.json"
            )
            if not checkpoint.exists() or not report_path.exists():
                break
            payload = torch.load(
                checkpoint, map_location="cpu", weights_only=False
            )
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                payload.get("completed") is not True
                or payload.get("config_hash") != self.config.hash()
            ):
                break
            iteration_reports.append(report)
            stored_reasons = list(report.get("reasons", []))
            if (
                "external wide_quiescence/exact082 score regressed"
                in stored_reasons
                and report.get("pre_arena")
                and report.get("post_arena")
            ):
                # Migrate the first completed iteration from the legacy OR
                # predicate.  This is a gate-only correction: checkpoint,
                # optimizer and replay remain byte-for-byte unchanged.
                stored_reasons = [
                    reason
                    for reason in stored_reasons
                    if reason
                    != "external wide_quiescence/exact082 score regressed"
                ]
                stored_reasons.extend(
                    self._external_gate_reasons(
                        report["pre_arena"], report["post_arena"]
                    )
                )
                report["reasons"] = stored_reasons
                report["gate_logic_version"] = 2
                atomic_json(report_path, report)
                iteration_reports[-1] = report
            if (
                "bare pi_(k+1) did not approach external Search(pi_k) score"
                in stored_reasons
                and report.get("pre_arena")
                and report.get("post_arena")
            ):
                # Gate-v3 migration: the legacy rule treated 128 duplicated
                # copies of two deterministic trajectories as independent.
                # Recompute only the gate; checkpoint/replay stay immutable.
                stored_reasons = [
                    reason
                    for reason in stored_reasons
                    if reason
                    != "bare pi_(k+1) did not approach external Search(pi_k) score"
                ]
                stored_reasons.extend(
                    self._external_gate_reasons(
                        report["pre_arena"], report["post_arena"]
                    )
                )
                report["reasons"] = stored_reasons
                report["gate_logic_version"] = 3
                atomic_json(report_path, report)
                iteration_reports[-1] = report
            if stored_reasons:
                return False, {
                    "completed_iterations": iteration,
                    "iterations": iteration_reports,
                    "resumed_existing_pause": True,
                }, stored_reasons
            failure_streak = int(
                report.get("search_gain_failure_streak", 0)
            )
            previous_validation = report.get("validation", {}).get(
                "policy_excess"
            )
            previous_data = report.get("data")
            start_iteration = iteration + 1
        if start_iteration > self.config.loop.iterations:
            return True, {
                "completed_iterations": self.config.loop.iterations,
                "iterations": iteration_reports,
                "resumed_completed_run": True,
            }, []
        for iteration in range(
            start_iteration, self.config.loop.iterations + 1
        ):
            actor, optimizer, scheduler, global_step = (
                self._load_model_optimizer(iteration - 1)
            )
            actor.eval()
            pre_value_calibration = self._calibrate_leaf_value_scale(actor)
            print(
                f"[search-v3 value-calibration] iteration={iteration} "
                f"scale={pre_value_calibration['leaf_value_scale']:.4f} "
                f"corr={pre_value_calibration['correlation']:.4f} "
                f"mse={pre_value_calibration['unscaled_mse']:.4f}->"
                f"{pre_value_calibration['scaled_mse']:.4f}",
                flush=True,
            )
            atomic_json(
                self.run_dir / "loop" / "progress.json",
                {
                    "schema": "open-junqi-search-v3-progress",
                    "phase": "precheck",
                    "iteration": iteration,
                    "completed_games": 0,
                    "target_games": self.config.loop.games_per_iteration,
                    "pre_value_calibration": pre_value_calibration,
                    "config_hash": self.config.hash(),
                },
            )
            precheck_path = (
                self.run_dir
                / "loop"
                / f"precheck_{iteration:06d}.json"
            )
            if precheck_path.exists():
                precheck = json.loads(
                    precheck_path.read_text(encoding="utf-8")
                )
                if precheck.get("config_hash") != self.config.hash():
                    raise RuntimeError(
                        f"stale closed-loop precheck: {precheck_path}"
                    )
                cached_calibration = precheck.get(
                    "pre_value_calibration"
                )
                if (
                    cached_calibration is None
                    or abs(
                        float(cached_calibration["leaf_value_scale"])
                        - float(
                            pre_value_calibration["leaf_value_scale"]
                        )
                    )
                    > 1e-9
                ):
                    raise RuntimeError(
                        f"stale uncalibrated closed-loop precheck: {precheck_path}"
                    )
                pre_probe = precheck["pre_probe"]
                pre_arena = precheck["pre_arena"]
            else:
                pre_probe = self._probe_metrics(actor)
                pre_arena = self._arena_metrics(actor, iteration, "pre")
                atomic_json(
                    precheck_path,
                    {
                        "schema": "open-junqi-search-v3-precheck",
                        "iteration": iteration,
                        "config_hash": self.config.hash(),
                        "pre_value_calibration": pre_value_calibration,
                        "pre_probe": pre_probe,
                        "pre_arena": pre_arena,
                    },
                )
            data = self._collect_iteration(iteration, actor)
            reasons = []
            if (
                data["unique_state_ratio"]
                < self.config.loop.minimum_unique_state_ratio
            ):
                reasons.append("unique state ratio collapsed")
            if any(
                value > self.config.loop.maximum_prefix_share
                for value in data["maximum_prefix_share"].values()
            ):
                reasons.append("one opening prefix exceeds the diversity cap")
            if data["maximum_trajectory_share"] > 0.02:
                reasons.append("one complete trajectory exceeds the 2% cap")
            if (
                previous_data is not None
                and data["unique_state_ratio"] + 0.20
                < previous_data["unique_state_ratio"]
            ):
                reasons.append(
                    "unique state ratio dropped by more than 20pp in one iteration"
                )
            if reasons:
                paused = {
                    "iteration": iteration,
                    "data": data,
                    "phase": "paused_before_train",
                    "reasons": reasons,
                }
                atomic_json(
                    self.run_dir
                    / "loop"
                    / f"iteration_{iteration:06d}.json",
                    paused,
                )
                atomic_json(
                    self.run_dir / "loop" / "progress.json",
                    {
                        "schema": "open-junqi-search-v3-progress",
                        "phase": "paused_before_train",
                        "iteration": iteration,
                        "reasons": reasons,
                        "config_hash": self.config.hash(),
                    },
                )
                return False, {
                    "completed_iterations": iteration - 1,
                    "iterations": iteration_reports + [paused],
                }, reasons
            current_samples = self._iteration_samples(iteration)
            current_validation = [
                sample
                for sample in current_samples
                if sample.game_id % 10 == 0
            ]
            distill_before = self._evaluate_distill(
                actor, current_validation
            )
            samples = self._replay_samples(iteration, current_samples)
            # Validation and the bounded reservoir now own every row still
            # needed; release current-rollout rows rejected by the reservoir
            # before allocating dense training batches.
            del current_samples
            atomic_json(
                self.run_dir / "loop" / "progress.json",
                {
                    "schema": "open-junqi-search-v3-progress",
                    "phase": "train",
                    "iteration": iteration,
                    "completed_games": self.config.loop.games_per_iteration,
                    "target_games": self.config.loop.games_per_iteration,
                    "replay_flushed": True,
                    "policy_samples": len(samples),
                    "replay_sampling": self.last_replay_stats,
                    "config_hash": self.config.hash(),
                    "rng": {
                        "closed_loop_python": self.rng.getstate(),
                    },
                },
            )
            validation, steps = self._fit(
                actor,
                optimizer,
                scheduler,
                samples,
                current_validation,
            )
            post_value_calibration = self._calibrate_leaf_value_scale(actor)
            distill_after = self._evaluate_distill(
                actor, current_validation
            )
            post_probe = self._probe_metrics(actor)
            post_arena = self._arena_metrics(actor, iteration, "post")
            pre_search_gain_ok = (
                pre_probe["search64_oracle_regret"]
                < pre_probe["bare_oracle_regret"]
                and pre_arena["direct_search64_vs_bare"]["score"] >= 0.50
            )
            post_search_gain_ok = (
                post_probe["search64_oracle_regret"]
                < post_probe["bare_oracle_regret"]
                and post_arena["direct_search64_vs_bare"]["score"] >= 0.50
            )
            search_gain_ok = pre_search_gain_ok and post_search_gain_ok
            failure_streak = 0 if search_gain_ok else failure_streak + 1
            if (
                failure_streak
                >= self.config.loop.consecutive_search_gain_failures
            ):
                reasons.append("Search Gain failed in two consecutive windows")
            if (
                distill_after["policy_excess"]
                >= distill_before["policy_excess"]
            ):
                reasons.append(
                    "pi_(k+1) did not improve fit to Search(pi_k) visits"
                )
            if (
                post_probe["search64_avoidable_exit_fraction"] + 0.02
                < pre_probe["search64_avoidable_exit_fraction"]
            ):
                reasons.append(
                    "avoidable-repetition exit rate deteriorated"
                )
            reasons.extend(
                self._external_gate_reasons(pre_arena, post_arena)
            )
            if (
                previous_validation is not None
                and validation["policy_excess"]
                > previous_validation + 0.10
            ):
                reasons.append("validation policy excess deteriorated rapidly")
            previous_validation = validation["policy_excess"]
            previous_data = data
            checkpoint = self._checkpoint(iteration)
            temporary = checkpoint.with_suffix(".pt.tmp")
            torch.save(
                {
                    "schema": "open-junqi-search-v3-checkpoint",
                    "phase": "closed_loop",
                    "completed": True,
                    "iteration": iteration,
                    "global_step": global_step + steps,
                    "config": self.config.to_dict(),
                    "config_hash": self.config.hash(),
                    "model": actor.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "leaf_value_scale": post_value_calibration[
                        "leaf_value_scale"
                    ],
                    "leaf_value_calibration": post_value_calibration,
                    "scheduler": {
                        "kind": "constant",
                        "state": scheduler.state_dict(),
                    },
                    "replay": {
                        "first_iteration": max(
                            1,
                            iteration
                            - self.config.loop.replay_iterations
                            + 1,
                        ),
                        "last_iteration": iteration,
                        "samples_loaded": len(samples),
                    },
                    "rng": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                        "cuda": (
                            torch.cuda.get_rng_state_all()
                            if torch.cuda.is_available()
                            else None
                        ),
                        "closed_loop_python": self.rng.getstate(),
                    },
                },
                temporary,
            )
            os.replace(temporary, checkpoint)
            report = {
                "iteration": iteration,
                "data": data,
                "validation": validation,
                "chosen_epoch": validation.get("chosen_epoch", 0),
                "distill_before": distill_before,
                "distill_after": distill_after,
                "pre_probe": pre_probe,
                "post_probe": post_probe,
                "pre_arena": pre_arena,
                "post_arena": post_arena,
                "pre_value_calibration": pre_value_calibration,
                "post_value_calibration": post_value_calibration,
                "closed_loop_inequality": {
                    "pre_search_stronger_than_bare": pre_search_gain_ok,
                    "distillation_improved": (
                        distill_after["policy_excess"]
                        < distill_before["policy_excess"]
                    ),
                    "post_search_stronger_than_bare": post_search_gain_ok,
                },
                "search_gain_failure_streak": failure_streak,
                "gate_logic_version": 3,
                "checkpoint": str(checkpoint),
                "reasons": reasons,
            }
            atomic_json(
                self.run_dir / "loop" / f"iteration_{iteration:06d}.json",
                report,
            )
            # The iteration becomes completed only after both checkpoint and
            # immutable report have been atomically published.
            atomic_json(
                self.run_dir / "loop" / "progress.json",
                {
                    "schema": "open-junqi-search-v3-progress",
                    "phase": (
                        "paused_after_check" if reasons else "completed"
                    ),
                    "iteration": iteration,
                    "completed_games": self.config.loop.games_per_iteration,
                    "replay_flushed": True,
                    "checkpoint": str(checkpoint),
                    "report": str(
                        self.run_dir
                        / "loop"
                        / f"iteration_{iteration:06d}.json"
                    ),
                    "global_step": global_step + steps,
                    "reasons": reasons,
                    "config_hash": self.config.hash(),
                    "rng": {
                        "closed_loop_python": self.rng.getstate(),
                    },
                },
            )
            iteration_reports.append(report)
            if reasons:
                return False, {
                    "completed_iterations": iteration,
                    "iterations": iteration_reports,
                }, reasons
        return True, {
            "completed_iterations": self.config.loop.iterations,
            "iterations": iteration_reports,
        }, []
