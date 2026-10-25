from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable, Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD_DIR = str(ROOT / "build")
if BUILD_DIR not in sys.path:
    sys.path.insert(0, BUILD_DIR)

import JunqiEnv
from JunqiEnv.junqi import PlayerId

from junqi_rl.bots import (
    build_bot,
    choose_heuristic_action,
    choose_heuristic_action_v2,
    choose_heuristic_action_v3_fast,
)
from junqi_rl.env.env_wrapper import JunqiPPOEnv


MIN_HEURISTIC_STEPS_PER_SEC = 100.0
MIN_CPP_BACKEND_STEPS_PER_SEC = 50_000.0
PRODUCTION_HEURISTICS: dict[str, Callable] = {
    "heuristic_v1": choose_heuristic_action,
    "heuristic_v2": choose_heuristic_action_v2,
    "heuristic_v3": choose_heuristic_action_v3_fast,
}


@contextlib.contextmanager
def single_core_affinity() -> Iterator[int | None]:
    """Pin this process to one currently allowed logical CPU when supported."""
    if os.name == "nt":
        import ctypes

        kernel32 = ctypes.windll.kernel32
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.GetProcessAffinityMask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.POINTER(ctypes.c_size_t),
        ]
        kernel32.GetProcessAffinityMask.restype = ctypes.c_int
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        process = kernel32.GetCurrentProcess()
        process_mask = ctypes.c_size_t()
        system_mask = ctypes.c_size_t()
        if not kernel32.GetProcessAffinityMask(
            process,
            ctypes.byref(process_mask),
            ctypes.byref(system_mask),
        ):
            yield None
            return
        original = process_mask.value
        selected = original & -original
        if not kernel32.SetProcessAffinityMask(process, selected):
            yield None
            return
        try:
            yield selected.bit_length() - 1
        finally:
            kernel32.SetProcessAffinityMask(process, original)
        return

    if hasattr(os, "sched_getaffinity") and hasattr(os, "sched_setaffinity"):
        original = os.sched_getaffinity(0)
        selected = min(original)
        os.sched_setaffinity(0, {selected})
        try:
            yield selected
        finally:
            os.sched_setaffinity(0, original)
        return

    yield None


def run_steps(selector: Callable, steps: int) -> None:
    env = JunqiEnv.JunqiEnvironment()
    env.reset_random()
    for _ in range(steps):
        if env.is_game_over():
            env.reset_random()
        player = env.get_current_player()
        opponent = PlayerId.BLACK if player == PlayerId.RED else PlayerId.RED
        legal_actions = env.get_legal_actions(player)
        action = selector(
            env.get_perfect_state(),
            legal_actions,
            player,
            env.get_material_score(player),
            env.get_material_score(opponent),
        )
        env.step(*divmod(action, 60))


def run_cpp_backend_steps(steps: int) -> None:
    env = JunqiEnv.JunqiEnvironment()
    env.reset_random()
    for _ in range(steps):
        if env.is_game_over():
            env.reset_random()
        player = env.get_current_player()
        action = env.get_legal_actions(player)[0]
        env.step(*divmod(action, 60))


def measure(selector: Callable, steps: int, warmup: int, repeats: int) -> tuple[float, list[float]]:
    run_steps(selector, warmup)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.process_time()
        run_steps(selector, steps)
        elapsed = time.process_time() - start
        samples.append(steps / max(elapsed, 1e-9))
    return statistics.median(samples), samples


def measure_cpp_backend(steps: int, repeats: int) -> tuple[float, list[float]]:
    run_cpp_backend_steps(1_000)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.process_time()
        run_cpp_backend_steps(steps)
        elapsed = time.process_time() - start
        samples.append(steps / max(elapsed, 1e-9))
    return statistics.median(samples), samples


def run_agent_steps(spec: str, steps: int) -> None:
    import torch

    wrapper = JunqiPPOEnv(max_steps=400)
    bot = build_bot(spec, torch.device("cpu"))
    wrapper.reset()
    for _ in range(steps):
        if wrapper.env.is_game_over():
            wrapper.reset()
        player = wrapper.env.get_current_player()
        action = bot.select_action(wrapper, player)
        if action is None:
            wrapper.reset()
        else:
            wrapper.step(action)


def measure_agent(spec: str, steps: int, repeats: int) -> tuple[float, list[float]]:
    run_agent_steps(spec, 4)
    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_agent_steps(spec, steps)
        elapsed = time.perf_counter() - start
        samples.append(steps / max(elapsed, 1e-9))
    return statistics.median(samples), samples


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Single-core production Heuristic performance gate."
    )
    parser.add_argument("--min-steps-per-sec", type=float, default=MIN_HEURISTIC_STEPS_PER_SEC)
    parser.add_argument(
        "--min-backend-steps-per-sec",
        type=float,
        default=MIN_CPP_BACKEND_STEPS_PER_SEC,
    )
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    random.seed(args.seed)
    results: dict[str, dict] = {}
    with single_core_affinity() as cpu:
        backend_steps = max(20_000, args.steps * 20)
        backend_median, backend_samples = measure_cpp_backend(backend_steps, args.repeats)
        results["cpp_backend"] = {
            "median_steps_per_sec": backend_median,
            "samples": backend_samples,
            "minimum_steps_per_sec": args.min_backend_steps_per_sec,
            "passed": backend_median >= args.min_backend_steps_per_sec,
        }
        for name, selector in PRODUCTION_HEURISTICS.items():
            median, samples = measure(selector, args.steps, args.warmup, args.repeats)
            results[name] = {
                "median_steps_per_sec": median,
                "samples": samples,
                "minimum_steps_per_sec": args.min_steps_per_sec,
                "passed": median >= args.min_steps_per_sec,
            }
        search_steps = min(120, max(30, args.steps // 5))
        median, samples = measure_agent("ab2", search_steps, min(2, args.repeats))
        results["alphabeta_d2_fast"] = {
            "median_steps_per_sec": median,
            "samples": samples,
            "minimum_steps_per_sec": args.min_steps_per_sec,
            "passed": median >= args.min_steps_per_sec,
        }

    payload = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": sys.version,
        "single_core": True,
        "logical_cpu": cpu,
        "minimum_steps_per_sec": args.min_steps_per_sec,
        "steps_per_repeat": args.steps,
        "repeats": args.repeats,
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(temporary, args.output)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for name, result in results.items():
            status = "PASS" if result["passed"] else "FAIL"
            print(
                f"{status} {name}: {result['median_steps_per_sec']:.1f} steps/s "
                f"(minimum {result['minimum_steps_per_sec']:.1f})"
            )
        print("gate measured on one logical CPU")

    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
