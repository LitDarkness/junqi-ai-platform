"""Reproducible symmetric strength/latency evaluation for every shipped agent."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime

import torch

from evaluate import evaluate_symmetric_specs


DEFAULT_BOTS = ["random", "greedy", "heuristic_v1", "heuristic_v2", "heuristic_v3", "ab2"]


def wilson_interval(successes: float, games: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval, treating a draw as half a success."""
    if games <= 0:
        return 0.0, 1.0
    p = successes / games
    denominator = 1.0 + z * z / games
    center = (p + z * z / (2.0 * games)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * games)) / games) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def safe_name(spec: str) -> str:
    return os.path.splitext(os.path.basename(spec))[0] if os.path.isfile(spec) else spec


def write_markdown(path: str, payload: dict) -> None:
    lines = [
        "# 军棋智能体能力实测",
        "",
        f"生成时间：{payload['generated_at']}",
        "",
        "所有配对均交换红黑双方；score = 胜率 + 0.5 × 和棋率。样本较小时请结合 95% Wilson 区间判断。",
        "",
        "| 主测方 | 对手 | 对局 | 胜 | 负 | 和 | score | 95% CI | 平均步数 | steps/s |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["matches"]:
        lines.append(
            f"| {row['primary']} | {row['opponent']} | {row['games']} | {row['wins']} | "
            f"{row['losses']} | {row['draws']} | {row['score']:.1%} | "
            f"{row['score_ci95'][0]:.1%}–{row['score_ci95'][1]:.1%} | "
            f"{row['avg_turns']:.1f} | {row['steps_per_second']:.1f} |"
        )
    lines += ["", "## 汇总", "", "| 智能体 | 累计 score | 配对数 |", "|---|---:|---:|"]
    for item in payload["ranking"]:
        lines.append(f"| {item['bot']} | {item['mean_score']:.1%} | {item['matches']} |")
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a symmetric Junqi bot evaluation suite.")
    parser.add_argument("--bots", nargs="+", default=DEFAULT_BOTS)
    parser.add_argument("--games-per-color", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--output", default="reports/evaluation_suite.json")
    parser.add_argument("--markdown", default="reports/evaluation_suite.md")
    args = parser.parse_args()
    if len(args.bots) < 2:
        raise SystemExit("--bots requires at least two agents")
    device_name = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    device = torch.device(device_name)
    matches: list[dict] = []
    total_pairs = len(args.bots) * (len(args.bots) - 1) // 2
    pair_index = 0
    for left in range(len(args.bots)):
        for right in range(left + 1, len(args.bots)):
            pair_index += 1
            primary, opponent = args.bots[left], args.bots[right]
            print(f"[评测 {pair_index}/{total_pairs}] {safe_name(primary)} vs {safe_name(opponent)}", flush=True)
            _, _, result, elapsed = evaluate_symmetric_specs(
                primary, opponent, args.games_per_color, args.max_steps, device
            )
            turns = result.as_red.total_turns + result.as_black.total_turns
            score = result.primary_score
            ci = wilson_interval(result.primary_wins + 0.5 * result.draws, result.games)
            row = {
                "primary": safe_name(primary), "opponent": safe_name(opponent),
                "primary_spec": primary, "opponent_spec": opponent,
                "games": result.games, "wins": result.primary_wins,
                "losses": result.opponent_wins, "draws": result.draws,
                "score": score, "score_ci95": list(ci), "avg_turns": result.avg_turns,
                "elapsed_seconds": elapsed, "steps_per_second": turns / max(elapsed, 1e-9),
            }
            matches.append(row)
            print(
                f"  W/L/D={row['wins']}/{row['losses']}/{row['draws']} "
                f"score={score:.1%}, {row['steps_per_second']:.1f} steps/s",
                flush=True,
            )

    scores: dict[str, list[float]] = {safe_name(bot): [] for bot in args.bots}
    for row in matches:
        scores[row["primary"]].append(row["score"])
        scores[row["opponent"]].append(1.0 - row["score"])
    ranking = sorted(
        ({"bot": bot, "mean_score": sum(values) / len(values), "matches": len(values)} for bot, values in scores.items()),
        key=lambda item: item["mean_score"], reverse=True,
    )
    payload = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "device": device_name, "games_per_color": args.games_per_color,
        "max_steps": args.max_steps, "matches": matches, "ranking": ranking,
        "warning": "C++ initial layouts currently use random_device; exact game sequences are not seed-reproducible.",
    }
    for path in (args.output, args.markdown):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    write_markdown(args.markdown, payload)
    print(f"完成：{args.output} / {args.markdown}")


if __name__ == "__main__":
    main()
