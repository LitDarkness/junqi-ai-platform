from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/repair_20261003"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_figure(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def main():
    old, graphs = read(OUT / "benchmark_baseline.json"), read(OUT / "benchmark_graphs.json")
    repeat_old, repeat_graphs = read(OUT / "benchmark_baseline_reproduced.json"), read(OUT / "benchmark_graphs_reproduced.json")
    trained, selfplay = read(OUT / "trained_arena.json"), read(OUT / "selfplay_benchmark.json")
    plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False, "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), constrained_layout=True)
    for ax, indices, title, reference in ((axes[0], [0, 5], "对旧 150 轮模型", 50),
                                          (axes[1], [1, 2, 6], "对强启发式 082", None)):
        for y, idx in enumerate(indices):
            row = trained["summaries"][idx]["family_score"]
            mean, (low, high) = row["mean"] * 100, [v * 100 for v in row["ci95"]]
            color = {0: "#2563eb", 5: "#d97706", 1: "#6b7280", 2: "#2563eb", 6: "#d97706"}[idx]
            ax.errorbar(mean, y, xerr=[[mean-low], [high-mean]], fmt="o", color=color, capsize=4)
            ax.text(high + .5, y, f"{mean:.1f}%", va="center", color=color)
        labels = ["连续训练 128 步", "固定数据预指定种子 64 步"] if reference else ["旧模型", "连续训练 128 步", "固定数据预指定种子 64 步"]
        ax.set_yticks(range(len(indices)), labels)
        ax.invert_yaxis()
        ax.set_ylim(len(indices)-.5, -.5)
        ax.set_title(title)
        ax.set_xlabel("布局对家族平均得分；95% 聚类区间")
        ax.grid(axis="x", alpha=.2)
        ax.set_xlim((40, 85) if reference else (0, 17))
        if reference: ax.axvline(reference, color="#9ca3af", linestyle="--")
    fig.suptitle("未见布局上的开发评测：每组 158 局 / 32 个独立布局对家族", fontsize=12)
    save_figure(fig, "strength")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    data = ([old["rows"][4]["elapsed_median"] * 1000, graphs["rows"][4]["elapsed_median"] * 1000],
            [row["elapsed_seconds"] for row in selfplay["rows"]])
    for ax, values, title, unit in zip(axes, data, ("单局串行 64 次模拟", "16 局完整网络自我对弈"), ("毫秒 / 每手", "秒 / 全部对局")):
        ax.bar([0, 1], values, color=["#6b7280", "#2563eb"], width=.55)
        ax.set_xticks([0, 1], ["常规推理", "CUDA Graph"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.set_ylim(0, max(values)*1.22)
        for i, val in enumerate(values): ax.text(i, val + max(values)*.03, f"{val:.1f}", ha="center")
        ax.text(.5, max(values)*1.12, f"速度 {values[0]/values[1]:.2f} 倍", ha="center", color="#2563eb")
    fig.suptitle("同模型、同模拟预算；落子 / 访问次数与完整自战轨迹一致", fontsize=12)
    save_figure(fig, "efficiency")

    fig, ax = plt.subplots(figsize=(7, 3.6), constrained_layout=True)
    for a, b, label in ((old, graphs, "首次测量"), (repeat_old, repeat_graphs, "独立重跑")):
        ratios = [x["elapsed_median"]/y["elapsed_median"] for x, y in zip(a["rows"][4:], b["rows"][4:])]
        ax.plot(range(4), ratios, marker="o", label=label)
    ax.axhline(1, color="#9ca3af", linestyle="--")
    ax.set_xticks(range(4), ["1", "8", "64", "128"])
    ax.set_xlabel("同时搜索的不同根局面数（每根 64 次模拟）")
    ax.set_ylabel("常规耗时 / 图推理耗时")
    ax.set_title("收益主要来自小批量；大批量收益接近测量波动")
    ax.legend(frameon=False)
    ax.grid(alpha=.2)
    save_figure(fig, "batch_speedup")



if __name__ == "__main__":
    main()
