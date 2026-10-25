from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from junqi_az.model import PolicyValueNet

from .config import FixedRLConfig
from .layout import FixedSameLayoutBackend, load_fixed_layout
from .evaluator import IndependentEvaluator
from .pipeline import FixedLayoutRLPipeline


def _overrides(arguments: argparse.Namespace) -> dict[str, int | float | str]:
    mapping = {
        "run_directory": arguments.run_directory,
        "teacher.bootstrap_games": arguments.teacher_games,
        "teacher.bootstrap_train_steps": arguments.bootstrap_steps,
        "selfplay.games_per_iteration": arguments.games_per_iteration,
        "selfplay.parallel_games": arguments.parallel_games,
        "train.steps_per_iteration": arguments.train_steps,
        "train.batch_size": arguments.batch_size,
    }
    return {key: value for key, value in mapping.items() if value is not None}


def train(arguments: argparse.Namespace) -> None:
    pipeline = FixedLayoutRLPipeline(
        arguments.config, overrides=_overrides(arguments)
    )
    try:
        pipeline.train(arguments.iterations)
    except KeyboardInterrupt:
        report = pipeline.abort_iteration("keyboard_interrupt")
        print(
            "\n[fixed-rl] interrupted; incomplete iteration discarded, "
            f"restored iteration={pipeline.iteration}: {report}",
            flush=True,
        )
        raise
    finally:
        pipeline.teacher.close()


def _load_benchmark_parts(config_path: Path):
    config = FixedRLConfig.load(config_path)
    project_root = config_path.resolve().parent.parent
    layout = load_fixed_layout(
        config.layout.set_path, config.layout.layout_id, project_root=project_root
    )
    backend = FixedSameLayoutBackend(layout, config.layout.mirror_probability)
    device = torch.device(
        config.device
        if not config.device.startswith("cuda") or torch.cuda.is_available()
        else "cpu"
    )
    model = PolicyValueNet(config.model).to(device).eval()
    return config, backend, model, device


def benchmark(arguments: argparse.Namespace) -> None:
    config_path = Path(arguments.config).resolve()
    config, backend, model, device = _load_benchmark_parts(config_path)
    random.seed(config.seed)
    np.random.seed(config.seed & 0xFFFFFFFF)

    state = backend.new_state(config.seed)
    transition_start = time.perf_counter()
    transitions = 0
    while transitions < arguments.transitions:
        if state.is_terminal():
            state = backend.new_state(config.seed + transitions)
        legal = state.legal_actions()
        state.apply(int(legal[transitions % len(legal)]))
        transitions += 1
    transition_elapsed = time.perf_counter() - transition_start

    samples = []
    state = backend.new_state(config.seed + 91)
    for index in range(64):
        if state.is_terminal():
            state = backend.new_state(config.seed + 91 + index)
        legal = np.asarray(state.legal_actions(), dtype=np.int32)
        mask = np.zeros(config.model.action_size, dtype=np.bool_)
        repetitions = np.zeros(config.model.action_size, dtype=np.uint8)
        mask[legal] = True
        repetitions[legal] = state.repetition_contexts()
        samples.append((state.feature_planes(), mask, repetitions))
        state.apply(int(legal[index % len(legal)]))

    inference_rows = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for batch_size in arguments.inference_batches:
        observations = np.stack(
            [samples[index % len(samples)][0] for index in range(batch_size)]
        )
        masks = np.stack(
            [samples[index % len(samples)][1] for index in range(batch_size)]
        )
        repetitions = np.stack(
            [samples[index % len(samples)][2] for index in range(batch_size)]
        )
        obs = torch.from_numpy(observations).to(device)
        action_mask = torch.from_numpy(masks).to(device)
        contexts = torch.from_numpy(repetitions).to(device)
        with torch.inference_mode():
            for _ in range(3):
                model.policy_value(obs, action_mask, contexts)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(arguments.inference_repeats):
                model.policy_value(obs, action_mask, contexts)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        inference_rows.append(
            {
                "batch": batch_size,
                "states_per_second": (
                    batch_size * arguments.inference_repeats / elapsed
                ),
                "milliseconds_per_batch": (
                    elapsed * 1000.0 / arguments.inference_repeats
                ),
            }
        )

    active = [
        backend.new_state(config.seed + 10000 + index)
        for index in range(arguments.rollout_games)
    ]
    active_plies = [0] * len(active)
    rollout_moves = 0
    inference_calls = 0
    rollout_start = time.perf_counter()
    while active:
        inference_calls += 1
        observations = []
        masks = np.zeros(
            (len(active), config.model.action_size), dtype=np.bool_
        )
        repetitions = np.zeros_like(masks, dtype=np.uint8)
        legal_rows = []
        for row, rollout_state in enumerate(active):
            legal = np.asarray(rollout_state.legal_actions(), dtype=np.int32)
            legal_rows.append(legal)
            masks[row, legal] = True
            repetitions[row, legal] = rollout_state.repetition_contexts()
            observations.append(rollout_state.feature_planes())
        with torch.inference_mode():
            policy, _ = model.policy_value(
                torch.from_numpy(np.stack(observations)).to(device),
                torch.from_numpy(masks).to(device),
                torch.from_numpy(repetitions).to(device),
            )
        policy = policy.cpu().numpy()
        finished = []
        for row, rollout_state in enumerate(active):
            legal = legal_rows[row]
            probabilities = policy[row, legal]
            probabilities = probabilities / probabilities.sum()
            action = random.choices(
                legal.tolist(), weights=probabilities.tolist(), k=1
            )[0]
            rollout_state.apply(int(action))
            rollout_moves += 1
            active_plies[row] += 1
            if (
                rollout_state.is_terminal()
                or active_plies[row] >= config.selfplay.max_moves
            ):
                finished.append(row)
        for row in reversed(finished):
            active.pop(row)
            active_plies.pop(row)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    rollout_elapsed = time.perf_counter() - rollout_start
    training_result: dict[str, float | str]
    try:
        training_batch = config.train.batch_size
        train_observations = np.stack(
            [samples[index % len(samples)][0] for index in range(training_batch)]
        )
        train_masks = np.stack(
            [samples[index % len(samples)][1] for index in range(training_batch)]
        )
        train_repetitions = np.stack(
            [samples[index % len(samples)][2] for index in range(training_batch)]
        )
        policy_targets = np.asarray(
            [
                np.flatnonzero(train_masks[index])[0]
                for index in range(training_batch)
            ],
            dtype=np.int64,
        )
        wdl_targets = np.arange(training_batch, dtype=np.int64) % 3
        train_obs = torch.from_numpy(train_observations).to(device)
        train_mask = torch.from_numpy(train_masks).to(device)
        train_context = torch.from_numpy(train_repetitions).to(device)
        policy_target = torch.from_numpy(policy_targets).to(device)
        wdl_target = torch.from_numpy(wdl_targets).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        model.train()

        def training_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            context = (
                torch.autocast("cuda", dtype=torch.bfloat16)
                if device.type == "cuda"
                else torch.autocast("cpu", enabled=False)
            )
            with context:
                logits, _, _, wdl_logits, _ = model.forward_heads(
                    train_obs, train_mask, train_context
                )
                loss = F.cross_entropy(logits.float(), policy_target)
                loss = loss + F.cross_entropy(wdl_logits.float(), wdl_target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            return float(loss.detach())

        training_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_start = time.perf_counter()
        training_loss = 0.0
        for _ in range(arguments.training_repeats):
            training_loss = training_step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        training_elapsed = time.perf_counter() - training_start
        training_result = {
            "batch": float(training_batch),
            "optimizer_steps_per_second": (
                arguments.training_repeats / training_elapsed
            ),
            "samples_per_second": (
                training_batch * arguments.training_repeats / training_elapsed
            ),
            "milliseconds_per_step": (
                1000.0 * training_elapsed / arguments.training_repeats
            ),
            "last_loss": training_loss,
        }
    except torch.cuda.OutOfMemoryError as error:
        training_result = {"error": f"CUDA out of memory: {error}"}
        optimizer = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "device": str(device),
        "network_parameters": sum(p.numel() for p in model.parameters()),
        "feature_channels": config.model.input_channels,
        "layout_id": config.layout.layout_id,
        "same_layout_for_both_sides": True,
        "cpp_transitions_per_second": transitions / transition_elapsed,
        "inference": inference_rows,
        "no_search_rollout": {
            "games": arguments.rollout_games,
            "moves": rollout_moves,
            "moves_per_second": rollout_moves / rollout_elapsed,
            "mean_inference_batch": rollout_moves / max(1, inference_calls),
        },
        "training": training_result,
        "gpu_peak_memory_mib": (
            torch.cuda.max_memory_allocated(device) / (1024 * 1024)
            if device.type == "cuda"
            else 0.0
        ),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def _resolve_evaluation_spec(specification: str, run_directory: Path) -> str:
    lowered = specification.lower()
    if lowered == "latest":
        return str(run_directory / "latest.pt")
    if lowered == "champion":
        return str(run_directory / "checkpoints/champion.pt")
    if lowered.startswith("iteration:"):
        iteration = int(lowered.split(":", 1)[1])
        filename = f"iteration_{iteration:06d}.pt"
        for directory in (
            run_directory / "checkpoints/archive",
            run_directory / "checkpoints/recent",
        ):
            candidate = directory / filename
            if candidate.is_file():
                return str(candidate)
        raise FileNotFoundError(f"iteration checkpoint is unavailable: {iteration}")
    return specification


def evaluate(arguments: argparse.Namespace) -> None:
    config_path = Path(arguments.config).resolve()
    config = FixedRLConfig.load(config_path)
    project_root = config_path.parent.parent
    run_directory = Path(arguments.run_directory or config.run_directory)
    if not run_directory.is_absolute():
        run_directory = (project_root / run_directory).resolve()
    layout_set = arguments.layout_set or config.layout.set_path
    shared_layout_id = arguments.layout_id or config.layout.layout_id
    red_layout = load_fixed_layout(
        layout_set,
        arguments.red_layout_id or shared_layout_id,
        project_root=project_root,
    )
    black_layout = load_fixed_layout(
        layout_set,
        arguments.black_layout_id or shared_layout_id,
        project_root=project_root,
    )
    evaluator = IndependentEvaluator(
        config,
        red_layout,
        black_layout=black_layout,
        project_root=project_root,
    )
    try:
        agent_a = evaluator.agent(
            _resolve_evaluation_spec(arguments.agent_a, run_directory)
        )
        agent_b = evaluator.agent(
            _resolve_evaluation_spec(arguments.agent_b, run_directory)
        )
        if arguments.output_prefix:
            output = Path(arguments.output_prefix)
            if not output.is_absolute():
                output = (project_root / output).resolve()
        else:
            output = (
                run_directory
                / "evaluations"
                / f"manual_{agent_a.label}_vs_{agent_b.label}_{int(time.time())}"
            )
        result = evaluator.evaluate(
            agent_a,
            agent_b,
            games=arguments.games,
            temperature=arguments.temperature,
            mode=arguments.mode,
            base_seed=arguments.seed,
            output_prefix=output,
        )
        summary = {
            key: value for key, value in result.items() if key != "games_detail"
        }
        summary["json"] = str(output.with_suffix(".json"))
        summary["csv"] = str(output.with_suffix(".csv"))
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    finally:
        evaluator.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fixed-same-layout Junqi policy/value RL"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    training = subcommands.add_parser("train", help="bootstrap/resume and train")
    training.add_argument("--config", required=True)
    training.add_argument("--iterations", type=int, default=100)
    training.add_argument("--run-directory")
    training.add_argument("--teacher-games", type=int)
    training.add_argument("--bootstrap-steps", type=int)
    training.add_argument("--games-per-iteration", type=int)
    training.add_argument("--parallel-games", type=int)
    training.add_argument("--train-steps", type=int)
    training.add_argument("--batch-size", type=int)
    training.set_defaults(func=train)

    speed = subcommands.add_parser("benchmark", help="native/GPU/rollout speed")
    speed.add_argument("--config", required=True)
    speed.add_argument("--transitions", type=int, default=100000)
    speed.add_argument(
        "--inference-batches", type=int, nargs="+", default=[64, 256, 1024]
    )
    speed.add_argument("--inference-repeats", type=int, default=20)
    speed.add_argument("--training-repeats", type=int, default=3)
    speed.add_argument("--rollout-games", type=int, default=256)
    speed.set_defaults(func=benchmark)

    evaluation = subcommands.add_parser(
        "evaluate", help="read-only checkpoint/heuristic evaluation"
    )
    evaluation.add_argument("--config", required=True)
    evaluation.add_argument("--run-directory")
    evaluation.add_argument("--a", dest="agent_a", required=True)
    evaluation.add_argument("--b", dest="agent_b", required=True)
    evaluation.add_argument("--games", type=int, default=512)
    evaluation.add_argument("--temperature", type=float, default=0.25)
    evaluation.add_argument(
        "--mode", choices=["sample", "greedy"], default="sample"
    )
    evaluation.add_argument("--seed", type=int, default=2026072309)
    evaluation.add_argument(
        "--layout-id",
        help="shared RED/BLACK layout id",
    )
    evaluation.add_argument("--red-layout-id")
    evaluation.add_argument("--black-layout-id")
    evaluation.add_argument(
        "--layout-set",
        help="layout-set JSON; defaults to the training config layout set",
    )
    evaluation.add_argument("--output-prefix")
    evaluation.set_defaults(func=evaluate)
    return parser


def main() -> None:
    parser = build_parser()
    arguments = parser.parse_args()
    try:
        arguments.func(arguments)
    except KeyboardInterrupt:
        print("\n[fixed-rl] interrupted; the last completed iteration is saved.", flush=True)
        sys.exit(130)
