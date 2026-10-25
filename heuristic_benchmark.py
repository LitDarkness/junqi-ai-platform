"""Deterministic, colour-balanced benchmark for Junqi heuristic bots.

The benchmark intentionally separates four questions:

* strength on a persisted corpus of opening layouts;
* red/black colour bias (both bots play both colours on every layout);
* single-core decision latency on unchanged positions;
* correctness on small, auditable tactical positions.

The C++ random layout generator is not seedable.  Consequently layouts are
generated once and persisted; the SHA-256 in every report identifies the exact
corpus and makes subsequent runs reproducible.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / "build"
for item in (str(ROOT), str(BUILD)):
    if item not in sys.path:
        sys.path.insert(0, item)

import JunqiEnv
from JunqiEnv.junqi import PlayerId

from junqi_rl.bots import build_bot
from junqi_rl.env.env_wrapper import JunqiPPOEnv

DEFAULT_CORPUS = ROOT / "config" / "benchmark_positions_v1.json"
DEFAULT_JSON = ROOT / "reports" / "heuristic_benchmark.json"
DEFAULT_MD = ROOT / "reports" / "heuristic_benchmark.md"


def action(src: int, dst: int) -> int:
    return src * 60 + dst


TACTICS = [
    {
        "name": "capture_flag_red",
        "player": int(PlayerId.RED),
        "pieces": {"1": 3, "27": 40, "32": -3, "58": -40},
        "expected": [action(27, 32)],
    },
    {
        "name": "capture_flag_black",
        "player": int(PlayerId.BLACK),
        "pieces": {"1": 40, "27": 3, "32": -40, "58": -3},
        "expected": [action(32, 27)],
    },
    {
        "name": "sapper_defuses_mine",
        "player": int(PlayerId.RED),
        "pieces": {"1": 3, "26": 10, "31": -1, "58": -3},
        "expected": [action(26, 31)],
    },
    {
        "name": "prefer_safe_capture_to_bomb",
        "player": int(PlayerId.RED),
        "pieces": {"1": 3, "27": 70, "28": -30, "32": -2, "58": -3},
        "expected": [action(27, 28)],
    },
]


def load_position(wrapper: JunqiPPOEnv, board: list[int], player: int = int(PlayerId.RED)) -> None:
    wrapper.env.load_board_state(board, player)
    wrapper._current_player = player
    wrapper._prev_material_delta = wrapper._material_delta(player)
    wrapper._no_combat_steps = 0


def generate_corpus(path: Path, count: int) -> None:
    backend = JunqiEnv.JunqiEnvironment()
    positions = []
    seen = set()
    while len(positions) < count:
        backend.reset_random()
        board = list(backend.get_perfect_state())
        fingerprint = tuple(board)
        if fingerprint not in seen:
            seen.add(fingerprint)
            positions.append(board)
    payload = {
        "format": "junqi-benchmark-positions-v1",
        "note": "Persisted layouts; generation itself uses the C++ random_device RNG.",
        "positions": positions,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def corpus_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def play_loaded(board: list[int], red_bot, black_bot, max_steps: int, seed: int) -> tuple[int, int]:
    random.seed(seed)
    env = JunqiPPOEnv(max_steps=max_steps)
    load_position(env, board)
    while True:
        player = env.env.get_current_player()
        bot = red_bot if player == PlayerId.RED else black_bot
        selected = bot.select_action(env, player)
        if selected is None:
            winner = PlayerId.BLACK if player == PlayerId.RED else PlayerId.RED
            return int(winner), int(env.env.get_turn_count())
        _, _, terminated, truncated, _ = env.step(selected)
        if terminated or truncated:
            return int(env.env.get_winner()), int(env.env.get_turn_count())


def bootstrap_ci(pair_scores: list[float], seed: int = 20260713, samples: int = 10000) -> list[float]:
    if not pair_scores:
        return [0.0, 1.0]
    rng = random.Random(seed)
    n = len(pair_scores)
    means = [sum(pair_scores[rng.randrange(n)] for _ in range(n)) / n for _ in range(samples)]
    means.sort()
    return [means[int(0.025 * samples)], means[int(0.975 * samples)]]


def result_score(winner: int, primary_colour: int) -> float:
    if winner == primary_colour:
        return 1.0
    if winner == int(PlayerId.NONE):
        return 0.5
    return 0.0


def benchmark_match(primary_spec: str, opponent_spec: str, positions: list[list[int]], max_steps: int) -> dict:
    primary = build_bot(primary_spec, torch.device("cpu"))
    opponent = build_bot(opponent_spec, torch.device("cpu"))
    rows = []
    start = time.perf_counter()
    for index, board in enumerate(positions):
        winner_red, turns_red = play_loaded(board, primary, opponent, max_steps, 10000 + index * 2)
        winner_black, turns_black = play_loaded(board, opponent, primary, max_steps, 10001 + index * 2)
        red_score = result_score(winner_red, int(PlayerId.RED))
        black_score = result_score(winner_black, int(PlayerId.BLACK))
        rows.append({
            "position": index,
            "as_red": red_score,
            "as_black": black_score,
            "pair_score": (red_score + black_score) / 2,
            "turns": turns_red + turns_black,
        })
    elapsed = time.perf_counter() - start
    flat = [value for row in rows for value in (row["as_red"], row["as_black"])]
    wins = sum(value == 1.0 for value in flat)
    draws = sum(value == 0.5 for value in flat)
    pair_scores = [row["pair_score"] for row in rows]
    turns = sum(row["turns"] for row in rows)
    return {
        "primary": primary_spec,
        "opponent": opponent_spec,
        "positions": len(rows),
        "games": len(flat),
        "wins": wins,
        "losses": len(flat) - wins - draws,
        "draws": draws,
        "score": sum(flat) / len(flat),
        "score_as_red": sum(row["as_red"] for row in rows) / len(rows),
        "score_as_black": sum(row["as_black"] for row in rows) / len(rows),
        "paired_bootstrap_ci95": bootstrap_ci(pair_scores),
        "avg_turns": turns / len(flat),
        "elapsed_seconds": elapsed,
        "game_steps_per_second": turns / max(elapsed, 1e-9),
        "pairs": rows,
    }


def benchmark_tactics(bot_specs: list[str]) -> list[dict]:
    results = []
    for spec in bot_specs:
        bot = build_bot(spec, torch.device("cpu"))
        cases = []
        for case in TACTICS:
            board = [0] * 60
            for square, piece in case["pieces"].items():
                board[int(square)] = piece
            env = JunqiPPOEnv(max_steps=100)
            load_position(env, board, case["player"])
            selected = bot.select_action(env, case["player"])
            chosen = None if selected is None else env.action_to_env(selected, case["player"])
            cases.append({"name": case["name"], "passed": chosen in case["expected"], "chosen": chosen,
                          "expected": case["expected"]})
        results.append({"bot": spec, "passed": sum(case["passed"] for case in cases),
                        "total": len(cases), "cases": cases})
    return results


def benchmark_latency(bot_specs: list[str], positions: list[list[int]], repeats: int) -> list[dict]:
    results = []
    for spec in bot_specs:
        bot = build_bot(spec, torch.device("cpu"))
        env = JunqiPPOEnv(max_steps=100)
        latencies = []
        # Warm up imports, scratch environments, and allocator paths.
        load_position(env, positions[0])
        bot.select_action(env, PlayerId.RED)
        for iteration in range(repeats):
            board = positions[iteration % len(positions)]
            load_position(env, board)
            started = time.perf_counter_ns()
            selected = bot.select_action(env, PlayerId.RED)
            latencies.append((time.perf_counter_ns() - started) / 1_000_000)
            if selected is None:
                raise RuntimeError(f"{spec} returned no action on corpus position")
        total_ms = sum(latencies)
        results.append({
            "bot": spec,
            "decisions": repeats,
            "decisions_per_second": repeats * 1000.0 / total_ms,
            "latency_ms_mean": statistics.fmean(latencies),
            "latency_ms_median": statistics.median(latencies),
            "latency_ms_p95": percentile(latencies, 0.95),
            "latency_ms_max": max(latencies),
        })
    return results


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# Heuristic benchmark",
        "",
        f"Generated: {payload['generated_at']}",
        f"Corpus SHA-256: `{payload['corpus_sha256']}` ({payload['position_count']} fixed layouts)",
        "",
        "Each layout is played twice with bot colours exchanged. Confidence intervals use a deterministic paired bootstrap over layouts.",
        "",
        "## Strength",
        "",
        "| Primary | Opponent | W-L-D | Score | Red | Black | Paired 95% CI | steps/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["matches"]:
        low, high = row["paired_bootstrap_ci95"]
        lines.append(f"| {row['primary']} | {row['opponent']} | {row['wins']}-{row['losses']}-{row['draws']} | "
                     f"{row['score']:.1%} | {row['score_as_red']:.1%} | {row['score_as_black']:.1%} | "
                     f"{low:.1%}-{high:.1%} | {row['game_steps_per_second']:.1f} |")
    lines += ["", "## Tactical suite", "", "| Bot | Passed |", "|---|---:|"]
    for row in payload["tactics"]:
        lines.append(f"| {row['bot']} | {row['passed']}/{row['total']} |")
    lines += ["", "## Single-process decision latency", "", "| Bot | decisions/s | mean ms | p95 ms | max ms |",
              "|---|---:|---:|---:|---:|"]
    for row in payload["latency"]:
        lines.append(f"| {row['bot']} | {row['decisions_per_second']:.1f} | {row['latency_ms_mean']:.3f} | "
                     f"{row['latency_ms_p95']:.3f} | {row['latency_ms_max']:.3f} |")
    lines += ["", "Caveat: the corpus is exactly reproducible, but Python/OS scheduling can perturb latency. "
              "The match sample is still too small for narrow strength estimates; use the paired interval, not only the point estimate.", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bots",
        nargs="+",
        default=["heuristic_v1", "heuristic_v2", "heuristic_v3"],
    )
    parser.add_argument("--opponents", nargs="+", default=["random", "greedy"])
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--positions", type=int, default=12, help="Number of persisted layouts to use.")
    parser.add_argument("--generate-corpus", type=int, metavar="N", help="Replace corpus with N new persisted layouts.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--latency-repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    args = parser.parse_args()
    if args.generate_corpus:
        generate_corpus(args.corpus, args.generate_corpus)
        print(f"Generated {args.generate_corpus} persisted positions at {args.corpus}")
    if not args.corpus.exists():
        raise SystemExit(f"Missing corpus: {args.corpus}; use --generate-corpus N once")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    positions = corpus["positions"][:args.positions]
    if len(positions) < args.positions:
        raise SystemExit(f"Corpus has only {len(positions)} positions, requested {args.positions}")

    matches = []
    pairs = [(bot, opponent) for bot in args.bots for opponent in args.opponents if bot != opponent]
    for index, (bot, opponent) in enumerate(pairs, 1):
        print(f"[match {index}/{len(pairs)}] {bot} vs {opponent}", flush=True)
        result = benchmark_match(bot, opponent, positions, args.max_steps)
        matches.append(result)
        print(f"  {result['wins']}-{result['losses']}-{result['draws']}, score={result['score']:.1%}, "
              f"CI={result['paired_bootstrap_ci95'][0]:.1%}-{result['paired_bootstrap_ci95'][1]:.1%}", flush=True)
    tested_specs = list(dict.fromkeys(args.bots + args.opponents))
    payload = {
        "schema": "junqi-heuristic-benchmark-v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "corpus": str(args.corpus.resolve()),
        "corpus_sha256": corpus_digest(args.corpus),
        "position_count": len(positions),
        "max_steps": args.max_steps,
        "matches": matches,
        "tactics": benchmark_tactics(tested_specs),
        "latency": benchmark_latency(tested_specs, positions, args.latency_repeats),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_markdown(args.markdown, payload)
    print(f"Wrote {args.output} and {args.markdown}")


if __name__ == "__main__":
    main()
