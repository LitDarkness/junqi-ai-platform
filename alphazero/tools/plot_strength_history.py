"""Plot archived strength only. No model loading, games, training, or GPU work."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports/strength_history_20261005"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.facecolor": "#f8fafc",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#d1d5db",
    "text.color": "#172033",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
})
BARE, S64, S256, OTHER = "#475569", "#1976c9", "#7c3aed", "#dc7d25"
sources = {}
points = []


def source(path):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    rel = path.relative_to(ROOT).as_posix()
    if rel not in sources:
        sources[rel] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "bytes": path.stat().st_size}
    return path, rel


def read(path):
    path, rel = source(path)
    return json.loads(path.read_text(encoding="utf-8-sig")), rel


def family_mean_from_csv(row):
    path, _ = source(row["rows_path"])
    groups = defaultdict(list)
    with path.open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            groups[r["trajectory_family_id"]].append(float(r["score"]))
    assert len(groups) == row["trajectory_families"]
    assert sum(map(len, groups.values())) == row["games"]
    return float(np.mean([np.mean(v) for v in groups.values()]))


def add(**kwargs):
    p = {"phase": "", "iteration": None, "role": "", "budget": None,
         "opponent": "exact082", "score": None, "ci_low": None, "ci_high": None,
         "games": 0, "families": None, "unique_trajectories": None, "raw_game_score": None,
         "suite_digest": "", "model_hash": "", "checkpoint": "",
         "protocol": "", "source": "", "source_key": "",
         "main_curve": False, "notes": ""}
    p.update(kwargs)
    assert 0 <= p["score"] <= 1
    if p["ci_low"] is not None:
        assert p["ci_low"] <= p["score"] <= p["ci_high"]
    points.append(p)
    return p


def diagnostic_external(j, path, iteration, kind):
    for key, r in j["external"].items():
        if r["opponent"] not in ["exact082", "supplement_wide_quiescence"]:
            continue
        match = re.search(r"candidate_(\d+)", r["candidate"])
        budget = int(match.group(1)) if match else None
        add(phase="v3_diagnostic", iteration=iteration, role=kind, budget=budget,
            opponent=r["opponent"], score=r["score"], raw_game_score=r["score"], games=r["games"],
            unique_trajectories=r.get("unique_trajectories"),
            protocol="single initial layout; deterministic repeated trajectories",
            source=path, source_key="external/"+key,
            notes="Pseudo-replicated. Archived binomial CI deliberately omitted; not a strength gate.")


j, rel = read("runs/open_junqi_search_v3/search_gain_report.json")
diagnostic_external(j, rel, 0, "supervised_seed_before_closed_loop")
j, rel = read("runs/open_junqi_search_v3/closed_loop_report.json")
for item in j["iterations"]:
    diagnostic_external(item["post_arena"], rel, item["iteration"], "post_arena")
for run in ["open_junqi_search_v3_alternating", "open_junqi_search_v3_scaled_training",
            "open_junqi_search_v3_strength_continuation"]:
    for path in sorted((ROOT/"runs"/run/"reports").glob("iteration_*.json")):
        j, rel = read(path)
        if "external_arena" in j:
            diagnostic_external(j["external_arena"], rel, j["iteration"], "post_arena")

# Formal v3 archives used different sealed suites at different triggers.
# They are shown with dotted historical connectors, never as a stable-cohort test.
formal = []
formal_paths = [
    ROOT/"runs/open_junqi_search_v3_strength/continuation_arena/report.json",
    ROOT/"runs/open_junqi_search_v3_sealed_continuation_v1/formal_continuation_arena/report.json",
]
formal_paths += sorted((ROOT/"runs/open_junqi_search_v3_strength_continuation/championships").glob("*/arena/report.json"))
for index, path in enumerate(formal_paths):
    j, rel = read(path)
    contract = j["contract"]
    new_iteration = 10 if index == 0 else 50 if index == 1 else int(re.search(r"challenger_(\d+)", str(path)).group(1))
    for key, row in j["matchups"].items():
        if not key.endswith("vs_exact082"):
            continue
        prefix = row["primary_agent"]
        budget = 0 if prefix.startswith("bare") else 64
        found = re.search(r"_(\d+)$", prefix)
        iteration = int(found.group(1)) if found else 10
        mean = family_mean_from_csv(row)
        if "family_weighted_mean_score" in row:
            assert abs(mean-row["family_weighted_mean_score"]) < 1e-12, (rel, key)
        ci = row["family_ci95"]
        p = add(phase="v3_formal", iteration=iteration, role=prefix,
            budget=budget, score=mean, ci_low=ci[0], ci_high=ci[1],
            games=row["games"], families=row["trajectory_families"], raw_game_score=row["score"],
            suite_digest=contract["suite_sha256"], protocol="single-layout continuation; no sampling/no Dirichlet; family weighted",
            source=rel, source_key="matchups/"+key,
            main_curve=iteration == new_iteration,
            notes="Different triggers often regenerated the sealed continuation suite. No causal cross-suite comparison.")
        if p["main_curve"]:
            formal.append(p)


def arena_rows(path, phase, iteration_map=None, select=None, notes=""):
    j, rel = read(path)
    iteration_map = iteration_map or {}
    results = {}
    for row in j["summaries"]:
        if row["opponent"] != "teacher0":
            continue
        role = row["candidate"]
        metadata = j.get("checkpoints", {}).get(role, j.get("checkpoint", {}))
        budget = j.get("inference_contract", {}).get("budget")
        metric = j.get("search_metrics", {}).get(role, {})
        if metric.get("moves", 0):
            budget = int(round(metric.get("simulations", 0)/metric["moves"]))
        for variant in j.get("variants", []):
            if variant["name"] == role:
                budget = variant.get("simulations", budget)
        if budget is None and "frozen_completion" in str(path):
            budget = 256
        family = row["family_score"]
        p = add(phase=phase, iteration=iteration_map.get(role, metadata.get("iteration")),
            role=role, budget=budget, score=family["mean"],
            ci_low=family["ci95"][0], ci_high=family["ci95"][1], games=row["games"],
            families=family["families"], suite_digest=j["suite_digest"], raw_game_score=row["score"],
            model_hash=metadata.get("model_hash", ""), checkpoint=metadata.get("path", ""),
            source=rel, source_key="summaries/"+role+"_vs_teacher0",
            protocol="fixed16 BF16 physical32 search256" if j.get("inference_contract", {}).get("physical_batch") == 32 and family["families"]==16 else
                     "fixed64 BF16 physical32 search256" if j.get("inference_contract", {}).get("physical_batch") == 32 else "v4 independent layout families; dynamic inference batching",
            main_curve=role in (select or []), notes=notes)
        # Recompute plotted family means using the archived actual outcomes.
        groups = defaultdict(list)
        games = [g for g in j["games"] if g["candidate"] == role and g["opponent"] == "teacher0"]
        if games:
            assert len(games) == row["games"], (rel, role)
            for g in games:
                groups[str(g["family"])].append(float(g["score"]))
            assert len(groups) == family["families"], (rel, role)
            assert abs(float(np.mean([np.mean(v) for v in groups.values()]))-p["score"]) < 1e-12, (rel, role)
        results[role] = p
    return results


early = []
rows = arena_rows("reports/repair_20261003/trained_arena.json", "v4_early",
                  {"baseline150": 0, "candidate": 2}, ["baseline150", "candidate"],
                  "v4 iteration0 denotes imported v3 iteration150; it is not untrained weights.")
early += [rows["baseline150"], rows["candidate"]]
for path in sorted((ROOT/"runs/open_junqi_v4_repair_pilot/arena").glob("actor_*.json")):
    iteration = int(path.stem.split("_")[-1])
    rows = arena_rows(path, "v4_early", {"learner": iteration}, ["learner"],
                      "Arena actor is the pre-gate incumbent; do not label it as the committed actor of this iteration.")
    role = rows["learner"]
    report, _ = read(ROOT/"runs/open_junqi_v4_repair_pilot/reports"/f"iteration_{iteration:06d}.json")
    metadata = report["roles"]["learner"]
    role.update(model_hash=metadata["model_hash"], checkpoint=metadata["path"])
    early.append(role)
value_ablation = arena_rows("runs/open_junqi_v4_value_search_ablation_20261003/arena.json", "v4_value_ablation",
                           {k:20 for k in ["baseline", "repeat", "quarter", "zero", "bare"]}, ["baseline", "bare"],
                           "quarter/zero alter the Value scale, not the search budget; omit them from standard64 curves.")
early.append(value_ablation["baseline"])
budget_ablation = arena_rows("runs/open_junqi_v4_search_budgets_20261004/arena.json", "v4_budget_ablation",
                            {k:26 for k in ["baseline", "repeat", "sim32", "sim128", "sim256"]}, ["baseline"],
                            "Same frozen champion26. Dynamic physical batching, not identical to later fixed32 inference.")
early.append(budget_ablation["baseline"])
arena_rows("runs/open_junqi_v4_fresh_continuation/arena/actor_000026.json", "v4_full26_old",
           {"candidate":26, "control":26, "parent":20, "parent_repeat":20, "champion":9})
for iteration in [27,30,33]:
    rows = arena_rows(f"runs/open_junqi_v4_fresh_continuation/arena/actor_{iteration:06d}.json", "v4_early",
                      {"learner":iteration}, ["learner"])
    early.append(rows["learner"])

RUNS = [
    ("open_junqi_v4_long_behavior_20261004_081501", 38, "后期行为", "candidate"),
    ("open_junqi_v4_late_noise_factor_20261004_143905", 62, "后期噪声", "control"),
    ("open_junqi_v4_actor_policy_20261004_184157", 86, "Policy标签", "control"),
    ("open_junqi_v4_root_explore_20261004_214715", 110, "探索目标", "control"),
]
segments = []
for run, start, label, continuing_role in RUNS:
    blocks = {}
    for block in [12,24]:
        rows = arena_rows(f"runs/{run}/arena/block_{block:06d}_screen.json", "v4_fixed16",
                         {"champion":26, "parent":start, "repeat":start, "guard86":86,
                          "candidate":start+block, "control":start+block}, ["candidate", "control"],
                         "Exploratory developer screen; not a promotion or independent final test.")
        blocks[block] = rows
    segments.append({"label": label, "start": start, "continuing_role": continuing_role,
                     "parent": blocks[12]["parent"], "blocks": blocks})
fixed64 = []
for file, iteration in [("runs/open_junqi_v4_long_behavior_20261004_081501/arena/block_000012_full.json",50),
                        ("runs/open_junqi_v4_frozen_completion_20261004_131703/arena/block_000024_full.json",62)]:
    rows = arena_rows(file, "v4_fixed64", {"champion":26,"parent":38,"repeat":38,"candidate":iteration,"control":iteration})
    fixed64.append(rows)

# Retain additional v4 branches in a separate ledger, with unknown iteration left blank.
# This avoids assigning every independent stage6/iteration32 model to a fictitious serial line.
seen = {p["source"] for p in points}
for run in sorted((ROOT/"runs").glob("open_junqi_v4*")):
    for path in sorted(run.rglob("*.json")):
        if "arena" not in path.parts and path.name != "arena.json":
            continue
        if "REVIEWS" in path.parts:
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in seen:
            continue
        j, rel = read(path)
        if "summaries" not in j or "games" not in j:
            continue
        arena_rows(path, "v4_other_branch", notes="Not serially joined: independent branch, changed suite or changed inference context.")
        seen.add(rel)


def base_axis(ax, ymax=0.6):
    ax.set_ylim(0,ymax)
    ax.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    ax.grid(axis="y", color="#e2e8f0", lw=.7)
    ax.set_axisbelow(True)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_ylabel("对强启发式082的得分率")
    ax.axhline(.5,color="#b45309",ls="--",lw=1,alpha=.65)


def plot_points(ax, rows, color, label=None, ls="-", marker="o", alpha=1):
    rows = sorted(rows, key=lambda p:p["iteration"])
    x = np.array([p["iteration"] for p in rows]); y = np.array([p["score"] for p in rows])
    ax.plot(x,y,ls=ls,marker=marker,color=color,label=label,lw=2,ms=5,alpha=alpha)
    for p in rows:
        if p["ci_low"] is not None:
            ax.errorbar(p["iteration"],p["score"],yerr=[[p["score"]-p["ci_low"]],[p["ci_high"]-p["score"]]],
                        color=color,elinewidth=1,capsize=3,alpha=alpha*.7,zorder=2)
    return x,y


def save(fig,name):
    fig.savefig(OUT/f"{name}.png",dpi=175,facecolor=fig.get_facecolor())
    fig.savefig(OUT/f"{name}.svg",facecolor=fig.get_facecolor())
    plt.close(fig)


def v3_panel(ax):
    base_axis(ax,.56)
    for budget,color,label in [(0,BARE,"bare：只用网络Policy选着"),(64,S64,"64search：每步64次MCTS模拟")]:
        rows = [p for p in formal if p["budget"]==budget]
        # The iteration10 suite and later sealed suites are not identical.
        plot_points(ax,[p for p in rows if p["iteration"]==10],color)
        plot_points(ax,[p for p in rows if p["iteration"]>=50],color,label,":")
        best = max(rows,key=lambda p:p["score"])
        ax.annotate(f"{best['iteration']}轮 {best['score']:.1%}", (best["iteration"],best["score"]),
                    xytext=(-35,28 if budget==64 else -35),textcoords="offset points",fontsize=10,color=color,
                    arrowprops={"arrowstyle":"-","color":color,"alpha":.6})
    ax.axvspan(138,151,color="#e2e8f0",alpha=.7)
    ax.text(144.5,.055,"后期仅有\n少样本诊断",ha="center",fontsize=9,color="#64748b")
    ax.set_xlim(0,151)
    ax.set_xticks([0,10,50,60,70,80,92,108,117,137,150])
    ax.set_xlabel("旧V3迭代编号（与V4重新计数分开）")
    ax.set_title("A 旧V3 bare与64搜索历史  ·  每个正式点2048局，按轨迹家族统计",loc="left",pad=14)
    ax.legend(loc="upper left",frameon=False,fontsize=10,ncol=2)
    ax.text(.01,.85,"虚线仅连接历史记录：多数评测更换了封存续局集，跨点变化不全是模型变化。",
            transform=ax.transAxes,fontsize=9,color="#64748b",va="top")


def early_panel(ax):
    base_axis(ax,.5)
    plot_points(ax,early,S64,"64search 历史记录",":")
    plot_points(ax,[value_ablation["bare"]],BARE,"bare（仅20轮有此开发评测）",marker="s")
    ax.set_xlim(-1,35)
    ax.set_xticks([0,2,3,6,9,12,15,18,20,26,27,30,33])
    ax.set_xlabel("V4迭代编号（0 = 导入旧V3的150轮模型）")
    ax.set_title("B V4早期64搜索及保存的bare测量  ·  32家族开发集，历史动态推理batch",loc="left",pad=14)
    ax.legend(loc="upper left",frameon=False,fontsize=10)
    ax.text(.01,.69,"不同诊断的续局集／组批上下文有变化；不能与A或C直接拼接。\n后续34–134轮未保存标准bare／64开发曲线；缺失值未补测或插值。",
            transform=ax.transAxes,fontsize=9,color="#64748b",va="top")
    ax.annotate("bare 20轮 5.99%",(20,value_ablation["bare"]["score"]),xytext=(8,-17),textcoords="offset points",fontsize=9,color=BARE)


def recent_panel(ax, with_title=True):
    base_axis(ax,.64)
    reference = segments[0]["blocks"][12]["champion"]
    ax.axhline(reference["score"],color="#94a3b8",ls="--",lw=1)
    ax.text(27,reference["score"]+.012,"冻结生产26参照 22.92%",fontsize=9,color="#64748b")
    plot_points(ax,[reference],S256,marker="D")
    first = True
    for index,seg in enumerate(segments):
        start, role = seg["start"],seg["continuing_role"]
        rest = "candidate" if role=="control" else "control"
        selected = [seg["parent"],seg["blocks"][12][role],seg["blocks"][24][role]]
        alternate = [seg["parent"],seg["blocks"][12][rest],seg["blocks"][24][rest]]
        ax.axvspan(start,start+24,color="#f1f5f9" if index%2==0 else "#f5f3ff",alpha=.5,zorder=0)
        ax.text(start+12,.598,seg["label"],ha="center",fontsize=10,color="#64748b")
        plot_points(ax,selected,S256,"实际接续来源及当前对照（未晋级）" if first else None)
        plot_points(ax,alternate,OTHER,"另一实验支路" if first else None,"--",alpha=.85)
        ax.axvline(start,color="#cbd5e1",lw=.7,ls=":")
        first=False
    ax.set_xlim(24,145)
    ax.set_xticks([26,38,50,62,74,86,98,110,122,134])
    ax.set_xlabel("V4连续来源编号；每24段切换一次登记实验，颜色不代表正式晋级")
    ax.legend(loc="upper left",frameon=False,ncol=2,fontsize=10,bbox_to_anchor=(.01,.91))
    last = segments[-1]["blocks"][24]
    ax.text(135.5,last["control"]["score"]+.008,"对照134\n45.57%",fontsize=9,color=S256,va="bottom")
    ax.text(135.5,last["candidate"]["score"]-.018,"候选134\n41.93%",fontsize=9,color=OTHER,va="top")
    if with_title:
        ax.set_title("C V4近期256搜索  ·  同一固定16家族续局集，BF16物理batch32，误差条为95%区间",loc="left",pad=14)
    ax.text(.015,.025,"研究来源：38 → 后期行为候选62 → 降噪对照86 → 标签对照110 → 当前对照／候选134。\n16家族是探索筛查；这些研究分支没有替换生产26。50%只是点估计参考，超过强启发式还需区间下界>50%。",
            transform=ax.transAxes,fontsize=9,color="#64748b",va="bottom")


fig,axes=plt.subplots(3,1,figsize=(15,15))
fig.subplots_adjust(left=.085,right=.975,top=.885,bottom=.07,hspace=.51)
fig.suptitle("双人明棋AI 历史棋力曲线",x=.085,ha="left",y=.977,fontsize=24,fontweight="bold")
fig.text(.085,.925,"纵轴为对冻结强启发式082的得分率：胜=1、和=0.5、负=0；按家族等权。三个面板采用不同评测条件。",fontsize=12,color="#475569")
v3_panel(axes[0]);early_panel(axes[1]);recent_panel(axes[2])
fig.text(.085,.023,"截至2026年10月5日暂停状态。只读取保存记录，新增训练局0、学习更新0、评测局0。082与084相关，未合并成独立样本。",fontsize=10,color="#64748b")
save(fig,"strength_history_overview")

fig,ax=plt.subplots(figsize=(15,6.3))
fig.subplots_adjust(left=.085,right=.975,top=.77,bottom=.18)
fig.suptitle("近期同条件256搜索棋力曲线",x=.085,ha="left",y=.967,fontsize=23,fontweight="bold")
fig.text(.085,.848,"固定16家族开发筛查。紫线标明历史实际接续来源及当前对照，橙线为同起点的另一支。",fontsize=11,color="#475569")
recent_panel(ax,False)
save(fig,"recent_256_history")

fig,axes=plt.subplots(2,1,figsize=(15,9.2))
fig.subplots_adjust(left=.085,right=.975,top=.815,bottom=.12,hspace=.45)
fig.suptitle("早期逐轮记录 仅作诊断",x=.085,ha="left",y=.97,fontsize=23,fontweight="bold")
fig.text(.085,.875,"这些是从闭环开始到旧V3的150轮留下的bare／64记录：大量比赛仅重复两条确定性棋路，不能当作稳定棋力曲线。",fontsize=11,color="#475569")
for ax,opponent,title in [(axes[0],"exact082","D 对强启发式082"),(axes[1],"supplement_wide_quiescence","E 对早期补充启发式（不同对手，得分不可与082混合）")]:
    base_axis(ax,1.08)
    for budget,color,label in [(0,BARE,"bare"),(64,S64,"64search")]:
        rows=sorted([p for p in points if p["phase"]=="v3_diagnostic" and p["budget"]==budget and p["opponent"]==opponent],key=lambda p:p["iteration"])
        ax.plot([p["iteration"] for p in rows],[p["score"] for p in rows],color=color,lw=1,marker=".",ms=3,label=label,alpha=.8)
    ax.set_xlim(0,150);ax.set_xticks(np.arange(0,151,10));ax.set_title(title,loc="left",pad=10)
    ax.legend(loc="lower right",ncol=2,frameon=True,facecolor="white",edgecolor="#e2e8f0")
    ax.set_xlabel("旧V3闭环迭代编号（0为监督预训练后的种子模型）")
    ax.set_ylabel("原始对局得分率")
fig.text(.085,.025,"未绘制原档案将重复棋路当作独立对局计算的区间；也没有平滑或外推。这些高低跳动不能替代总览A中的家族评测。",fontsize=10,color="#64748b")
save(fig,"early_diagnostics")

fig,axes=plt.subplots(1,2,figsize=(14,6.6))
fig.subplots_adjust(left=.075,right=.975,top=.74,bottom=.23,wspace=.26)
fig.suptitle("保存权重的搜索预算对比",x=.075,ha="left",y=.968,fontsize=23,fontweight="bold")
fig.text(.075,.835,"横轴是每步搜索次数，不是训练轮次；这些是历史动态batch诊断，不能替代近期固定batch32的补测。",fontsize=11,color="#475569")
for ax in axes:base_axis(ax,.5)
ax=axes[0]
rows=[value_ablation["bare"],value_ablation["baseline"]]
for r in rows:
    ax.errorbar(r["budget"],r["score"],yerr=[[r["score"]-r["ci_low"]],[r["ci_high"]-r["score"]]],fmt="o",color=BARE if r["budget"]==0 else S64,capsize=4)
ax.plot([r["budget"] for r in rows],[r["score"] for r in rows],color=S64,ls=":")
ax.set_xticks([0,64],["bare (0)","64search"]);ax.set_xlim(-10,74)
ax.set_title("F 同一20轮权重",loc="left",pad=12)
for r in rows:ax.annotate(f"{r['score']:.2%}",(r["budget"],r["score"]),xytext=(0,32),textcoords="offset points",ha="center",fontsize=10)
ax.set_xlabel("Value尺度与标准搜索保持原档案设定")
ax=axes[1]
rows=[budget_ablation[k] for k in ["sim32","baseline","sim128","sim256"]]
x=[r["budget"] for r in rows];y=[r["score"] for r in rows]
ax.plot(x,y,marker="o",color=S64,lw=2)
for r in rows:
    ax.errorbar(r["budget"],r["score"],yerr=[[r["score"]-r["ci_low"]],[r["ci_high"]-r["score"]]],color=S64,capsize=4)
    ax.annotate(f"{r['score']:.2%}",(r["budget"],r["score"]),xytext=(0,29),textcoords="offset points",ha="center",fontsize=10)
ax.set_xticks([32,64,128,256]);ax.set_xlim(10,282)
ax.set_title("G 同一合格26轮权重（另有64复测）",loc="left",pad=12)
ax.set_xlabel("每步新增MCTS模拟次数")
fig.text(.075,.075,"F、G不是同一模型或同一套续局起点。G只描述已测预算：128没有明显优于64，256点估计更高；未宣称普遍单调提升。",fontsize=10,color="#64748b")
save(fig,"search_budget_diagnostics")

fig,ax=plt.subplots(figsize=(12,5.5))
fig.subplots_adjust(left=.1,right=.97,top=.745,bottom=.19)
base_axis(ax,.54)
rows0=fixed64[0];rows1=fixed64[1]
chosen=[rows0["champion"],rows0["parent"],rows0["candidate"],rows1["candidate"]]
plot_points(ax,chosen,S256,"研究候选／来源历史")
plot_points(ax,[rows0["control"],rows1["control"]],OTHER,"同期对照","--")
ax.set_xticks([26,38,50,62]);ax.set_xlim(22,67)
ax.set_xlabel("V4编号；同一完整64家族开发续局集")
ax.legend(loc="upper left",frameon=False)
fig.suptitle("完整64家族的256搜索复核记录",x=.1,ha="left",y=.965,fontsize=22,fontweight="bold")
fig.text(.1,.823,"BF16物理batch32。该图与固定16家族筛查分开；86／110／134没有对应的完整64结果。",fontsize=11,color="#475569")
fig.text(.1,.06,"26：24.15%；38：27.93%；候选50：34.38%；候选62：35.03%。62的额外路线未通过全部采用条件，生产模型仍26。",fontsize=10,color="#64748b")
save(fig,"full64_search256_history")

columns=list(points[0])
with (OUT/"strength_points.csv").open("w",encoding="utf-8-sig",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=columns)
    writer.writeheader();writer.writerows(points)
plan,_=read("runs/open_junqi_v4_repair_pilot/supervision/PLAN.json")
assert plan.get("paused_by_user") is True
assert (ROOT/"runs/open_junqi_v4_root_explore_20261004_214715/STOP").exists()
assert (ROOT/"runs/open_junqi_v4_root_explore_20261004_214715/supervision/STOP").exists()
evidence={"schema":"junqi-archived-strength-history-v1","date":"2026-10-05",
          "new_training_games":0,"new_updates":0,"new_evaluation_games":0,
          "paused_by_user_verified":True,"score_definition":"win1 draw0.5 loss0; family weighted except pseudo diagnostics",
          "point_count":len(points),"source_count":len(sources),"sources":sources,"points":points,
          "missing":"Latest134 has fixed16 standardsearch256 only; no current bare/64 evaluation. v3 and v4 counters reset.",
          "historical_suite_change":"v3 formal triggers usually changed sealed suites; dotted joins are descriptive only.",
          "strong_teachers_correlated":"Only teacher0 / frozen082 plotted. teacher1 is not treated as independent evidence."}
(OUT/"EVIDENCE.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding="utf-8")
qa={"archived_points":len(points),"sources":len(sources),"formal_v3_main_points":len(formal),
    "v4_early_search64_points":len(early),"fixed16_suite_digests":sorted({p["suite_digest"] for p in points if p["phase"]=="v4_fixed16"}),
    "fixed16_inference_contexts":sorted({p["protocol"] for p in points if p["phase"]=="v4_fixed16"}),
    "all_plotted_family_means_recomputed":True,"ci_score_consistency_verified":True,
    "pseudo_intervals_omitted":True,"training_pause_preserved":True,
    "figures":[p.name for p in OUT.glob("*.png")]}
(OUT/"QA.json").write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding="utf-8")
print(json.dumps(qa,ensure_ascii=False,indent=2))
