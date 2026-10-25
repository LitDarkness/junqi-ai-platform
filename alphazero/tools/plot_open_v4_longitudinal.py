"""CPU-only analysis of the frozen, paired historical-self evaluation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "old150": "旧 V3 · 150（V4 导入起点）",
    "champ26": "V4 · 生产参照 26",
    "source38": "V4 · 研究来源 38",
    "candidate62": "V4 · 行为候选 62",
    "control86": "V4 · 研究对照 86",
    "control110": "V4 · 研究对照 110",
    "control134": "当前对照 134",
    "candidate134": "当前候选 134",
}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def interval(values, seed, level=.95, count=20000):
    values = np.asarray(values, np.float64)
    rng = np.random.default_rng(seed)
    samples = np.empty(count)
    for begin in range(0, count, 256):
        n = min(256, count-begin)
        samples[begin:begin+n] = values[rng.integers(len(values), size=(n, len(values)))].mean(axis=1)
    alpha = (1-level)/2
    return list(map(float, np.quantile(samples, [alpha, 1-alpha])))


def family_values(games):
    grouped = defaultdict(list)
    for game in games:
        grouped[game["family"]].append(game["score"])
    return np.array([np.mean(grouped[key]) for key in sorted(grouped)])


def pct(value):
    return f"{100*value:.2f}%"


def score_text(mean, ci):
    return f"{100*mean:.2f}% [{100*ci[0]:.2f}, {100*ci[1]:.2f}]"


def build(run):
    result = read(run/"RESULTS.json")
    contract = read(run/"CONTRACT.json")
    status = read(run/"STATUS.json")
    assert status["state"] == "review_due"
    assert result["new_evaluation_games"] == contract["planned_games"] == 9072
    assert result["new_training_games"] == result["effective_updates"] == 0
    assert result["training_still_paused"] and result["weights_unchanged"]
    output = ROOT/"reports/longitudinal_self_20261005"
    output.mkdir(parents=True, exist_ok=True)
    rows, qa, arenas = [], {}, {}
    for budget in contract["budgets"]:
        arena_path = run/f"arena_{budget}.json"
        arena = read(arena_path)
        proof = read(run/f"verification_{budget}.json")
        arenas[budget] = arena
        assert sha(arena_path) == result["outputs"][str(budget)]["sha256"]
        assert len(arena["games"]) == proof["games"] == 4536
        assert proof["duplicate_exact_games"] == 324
        by_matchup = defaultdict(list)
        for game in arena["games"]:
            assert game["score"] == .5 + .5*game["value"]
            by_matchup[(game["candidate"], game["opponent"])].append(game)
        for summary in arena["summaries"]:
            games = by_matchup[(summary["candidate"], summary["opponent"])]
            values = family_values(games)
            assert len(games) == 324 and len(values) == 64
            assert abs(values.mean()-summary["family_score"]["mean"]) < 1e-14
            assert summary["wins"] == sum(g["value"] == 1 for g in games)
            assert summary["draws"] == sum(g["value"] == 0 for g in games)
            assert summary["losses"] == sum(g["value"] == -1 for g in games)
            if summary["candidate"] == "repeat134":
                continue
            opening = [g for g in games if g["start_ply"] == 0]
            assert len(opening) == 128
            row = dict(budget=budget, candidate=summary["candidate"], opponent=summary["opponent"],
                       games=len(games), families=len(values), wins=summary["wins"],
                       draws=summary["draws"], losses=summary["losses"],
                       family_score=summary["family_score"]["mean"],
                       ci95_lower=summary["family_score"]["ci95"][0],
                       ci95_upper=summary["family_score"]["ci95"][1],
                       raw_game_score=summary["score"],
                       opening_score=float(family_values(opening).mean()),
                       repetition_fraction=summary["repetition_fraction"],
                       mean_plies=float(np.mean([len(g["trace"]) for g in games])),
                       primary_ci975_lower=None, primary_ci975_upper=None)
            for primary in result["primary"]:
                if (budget, row["candidate"], row["opponent"]) == (64, primary["candidate"], primary["opponent"]):
                    row["primary_ci975_lower"], row["primary_ci975_upper"] = primary["ci975"]
            rows.append(row)
        def keyed(name):
            return {(g["position"],g["candidate_side"]):(g["trace"],g["reason"],g["score"])
                    for g in by_matchup[(name,"control86")]}
        assert keyed("candidate134") == keyed("repeat134")
        qa[str(budget)] = dict(native_games=proof["games"], native_plies=proof["native_plies"],
                               duplicate_exact_games=324, family_means_and_wdl_recomputed=True,
                               sha256=sha(arena_path))
    for index, primary in enumerate(result["primary"]):
        games = [g for g in arenas[64]["games"] if (g["candidate"],g["opponent"]) == (primary["candidate"],primary["opponent"])]
        ci = interval(family_values(games), contract["evaluation_seed"]+1000+index, .975)
        assert np.allclose(ci, primary["ci975"], rtol=0, atol=1e-14)
    archive_checks = dict(contract["historical_stop_preservation"])
    archive_checks.update({item["path"]:item["sha256"] for item in contract["participants"].values()})
    archive_checks.update(contract["frozen_sources"])
    archive_checks[contract["config"]]=contract["config_sha256"]
    archive_checks[contract["suite"]["path"]]=contract["suite"]["sha256"]
    assert all(sha(ROOT/path) == digest for path,digest in archive_checks.items())
    learning_integrity=read(run/"continuation_integrity_cpu.json")
    assert all(item["optimizer_finite"] and item["counters"]["iteration"]==134
               for item in learning_integrity.values())
    with (output/"scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer=csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    find = lambda b,c,o: next(r for r in rows if (r["budget"],r["candidate"],r["opponent"]) == (b,c,o))

    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    font = FontProperties(fname=str(font_path))
    plt.rcParams.update({"font.family":font.get_name(),"axes.unicode_minus":False,
                         "font.size":11,"savefig.facecolor":"#fbfcfe"})
    fig, axes = plt.subplots(1,2,figsize=(15,7.5),sharey=True,facecolor="#fbfcfe")
    history = contract["history"]
    colors = {"control134":"#425abb", "candidate134":"#dd8b25"}
    for axis,budget in zip(axes,contract["budgets"]):
        axis.set_facecolor("#fbfcfe")
        axis.axvspan(0,50,color="#edeff3",alpha=.6,zorder=0)
        axis.axvline(50,color="#6c7584",linestyle="--",linewidth=1.1)
        for candidate,offset in [("control134",-.13),("candidate134",.13)]:
            for i,opponent in enumerate(history):
                row=find(budget,candidate,opponent)
                mean=row["family_score"]*100
                lo,hi=row["ci95_lower"]*100,row["ci95_upper"]*100
                primary=budget==64 and opponent=="control86"
                if primary:
                    lo,hi=row["primary_ci975_lower"]*100,row["primary_ci975_upper"]*100
                axis.errorbar(mean,i+offset,xerr=[[mean-lo],[hi-mean]],fmt="o",capsize=3,
                              markersize=6,color=colors[candidate],elinewidth=2 if primary else 1.35,
                              label=LABELS[candidate] if i==0 else None)
                axis.text(min(hi+1.2,95),i+offset,f"{mean:.1f}%",va="center",fontsize=9,color=colors[candidate])
        axis.set_title("bare：双方都不搜索" if budget==0 else "64search：双方每步都做64模拟",pad=16,fontsize=14)
        axis.set_xlim(35,100);axis.set_xticks([40,50,60,70,80,90,100])
        axis.set_xlabel("当前134的对战得分率（%）",labelpad=12)
        axis.set_yticks(range(len(history)),[LABELS[name] for name in history])
        axis.grid(axis="x",alpha=.18);axis.spines[["top","right","left"]].set_visible(False)
        axis.tick_params(axis="y",length=0)
        axis.legend(loc="lower right",frameon=False,fontsize=10)
    axes[0].invert_yaxis()
    fig.suptitle("当前模型挑战历史上的自己",x=.06,ha="left",fontsize=21,fontweight="bold",y=.98)
    fig.text(.06,.912,"同一完整64家族开发集 · 每组324局 · 互换执色 · 胜1／和0.5／负0 · 50%为双方打平",fontsize=11,color="#4f5868")
    fig.text(.06,.048,"误差条：家族聚类95%；64search对86的两项主比较使用97.5%区间，控制两项比较的总体误报。",fontsize=10,color="#4f5868")
    fig.text(.06,.018,"这是对不同历史对手的横向对战图，不能将这些点连成训练曲线；重复开发评测也不等于最终布局泛化验证。",fontsize=10,color="#4f5868")
    fig.subplots_adjust(left=.205,right=.98,bottom=.18,top=.85,wspace=.10)
    for ext in ["png","svg"]:
        fig.savefig(output/f"historical_self_gauntlet.{ext}",dpi=190)
    plt.close(fig)

    old = read(ROOT/contract["existing256"]["path"])
    assert sha(ROOT/contract["existing256"]["path"]) == contract["existing256"]["sha256"]
    primary = result["primary"]
    signals = [p["candidate"] for p in primary if p["research_continuation_signal"]]
    duel = find(64,"candidate134","control134")
    conclusion = ("纵向比赛支持继续进行有保存边界与棋力检查的在线研究训练。" if signals else
                  "这次预注册的64搜索主比较尚不支持原样长期续训；应先诊断或预注册下一实验。")
    if "control134" in signals:
        source_recommendation = "建议优先以对照134的完整学习状态作为下一研究起点；该支满足本次续训信号，也避免默认采用未确认额外收益的目标修剪。候选134与冻结86完整保留。这是研究起点选择，生产26保持不变。"
    elif "candidate134" in signals:
        source_recommendation = "候选134满足本次续训信号，可作为后续独立研究起点；仍不能据此认定目标修剪通过原外部采用门禁。对照134、候选134和冻结86都完整保留，生产26保持不变。"
    else:
        source_recommendation = "目前不建议用编号更新代替棋力证据，把134原样无限续下去。保留两支134及冻结86，在新预注册实验里检查网络直接决策与搜索表现的分歧；不能用本次描述性子组补认原采用条件。生产26保持不变。"
    content = ["# 当前模型对历史版本的纵向交手报告", "",
               conclusion+"这与正式晋级、超过强启发式是三个不同判断；本次没有恢复训练，也没有改动生产模型。", "",
               "![当前模型挑战历史版本]("+(output/"historical_self_gauntlet.png").as_posix()+")", "",
               "## 最重要的结果", "",
               "| 预注册主比较 | 64搜索家族得分 | 支持续训的研究信号 |", "| --- | ---: | --- |"]
    for p in primary:
        content.append(f"| {LABELS[p['candidate']]} 对研究对照86 | {score_text(p['mean'],p['ci975'])}（97.5%区间） | {'满足' if p['research_continuation_signal'] else '未满足'} |")
    content += ["", "主比较在比赛前固定：最新两支各自对冻结86，区间下界须超过50%，并且同支对生产26的95%区间下界超过50%。两项主比较各用97.5%区间，通过Bonferroni方式控制总体95%的误报水平；家族仍是统计单位。它只是是否值得继续研究的信号，不替代原Champion门禁。", "",
                f"当前候选134直接对当前对照134，在64搜索下得分为 **{score_text(duel['family_score'],[duel['ci95_lower'],duel['ci95_upper']])}**。"+
                ("区间包含50%，本次不能确定哪支更强，也不能确认目标修剪的额外棋力收益。" if duel['ci95_lower']<=.5<=duel['ci95_upper'] else
                 "这项是描述性比较，不是经过多重比较校正的因素采用门禁。"), "",
                "## 定义", "",
                "- **得分率**：胜1分、和0.5分、负0分。先对同一家族的起点和双方执色平均，再对64家族等权平均。因此它可能与直接按324局汇总的比例不同。它不是纯胜率或Elo。",
                "- **bare**：直接选择网络在合法动作中Policy概率最大的动作，不用MCTS，也不用Value决定动作。",
                "- **64search**：同一保存权重每步运行原标准MCTS的64次模拟，含首次根展开。双方使用相同预算，关闭噪声、战术过滤、强制探索、目标修剪、缓存和树复用。",
                "- **当前对照134／候选134**：都从原标签研究对照110继续24段而来，每段有新采集及学习；两支训练时共同使用强制根探索，候选额外修剪真正自博弈Policy目标，对照保留真实访问目标。开发比赛中共同训练机制和修剪均关闭，比赛只读取学到的权重。‘对照’并非从项目最初一直不变的基线。",
                "- **这两支的训练机制**：每支24段、每段自身前段学习器生成256新局再做64次有效更新，共6144新局／1536更新；三段回放。数据模式保持25%网络自博弈、25%教师互战、50%网络对教师；普通与重复上下文搜索预算为64／128，24手后按最大访问选择行为。自博弈Policy使用搜索目标，其他模式仍保留原教师硬标签；Value学习实际终局的胜／和／负。这是混合蒸馏与自我训练，不能称纯自博弈RL。",
                "- **旧150**：旧V3编号，权重是V4导入起点；不能把150与134当成同一轮数体系比较。26、38、62、86、110、134是这里列出的特定完整检查点，并非编号下任意分支。",
                "- **纵向进步**：当前模型在同条件下对早期权重获得高于50%的交手得分。它与最近单一训练因素有效、每一段都进步、已经超过外部强启发式都不等价。", "",
                "## 完整交手表", "",
                "每格为家族得分及描述性95%区间。64搜索对86的续训判断请使用上面的97.5%主比较，不能改用这里较窄的95%区间。", "",
                "| 历史对手 | bare：对照134 | bare：候选134 | 64搜索：对照134 | 64搜索：候选134 |",
                "| --- | ---: | ---: | ---: | ---: |"]
    for opponent in history:
        cells = [score_text((r:=find(b,c,opponent))["family_score"],[r["ci95_lower"],r["ci95_upper"]])
                 for b,c in [(0,"control134"),(0,"candidate134"),(64,"control134"),(64,"candidate134")]]
        content.append("| "+LABELS[opponent]+" | "+" | ".join(cells)+" |")
    content += ["", "## 只从开局开始的检查", "",
                "完整协议包含第0、24、48手的保存开发起点（终局前仍存活者），前缀来自既有冻结开发集。为避免只看续局，本表另列每组128局的纯开局描述性得分。它是事后分层，不作为新的采用条件或独立样本。", "",
                "| 历史对手 | bare：对照134 | bare：候选134 | 64搜索：对照134 | 64搜索：候选134 |",
                "| --- | ---: | ---: | ---: | ---: |"]
    for opponent in history:
        cells=[pct(find(b,c,opponent)["opening_score"]) for b,c in [(0,"control134"),(0,"candidate134"),(64,"control134"),(64,"candidate134")]]
        content.append("| "+LABELS[opponent]+" | "+" | ".join(cells)+" |")
    content += ["", "## 与已有256搜索评测的关系", "",
                "以下是此前保存的固定16家族开发筛查，双方256搜索；它没有计入本次新增比赛，也不能与本次64家族协议拼成同一曲线。", "",
                "| 候选134挑战历史模型 | 256搜索得分及95%区间（旧16家族） |", "| --- | ---: |"]
    for opponent,label in [("champion","生产26"),("parent","来源110"),("control","同期对照134"),("guard86","冻结86")]:
        found=next((s for s in old["summaries"] if s["candidate"]=="candidate" and s["opponent"]==opponent),None)
        if found:
            content.append(f"| {label} | {score_text(found['family_score']['mean'],found['family_score']['ci95'])} |")
    content += ["", "已有同条件256搜索对冻结082的得分为：生产26 22.92% [16.15,29.69]；研究86 45.05% [38.54,51.82]；来源110 33.59% [25.00,42.97]；对照134 45.57% [36.20,55.21]；候选134 41.93% [31.51,51.56]。这是旧16家族外部筛查，并未确认134超过强启发式，也未确认修剪路线通过原完整门禁。082与084高度相关，不能合并成两份独立证据。", "",
                "## 能否继续训练", "",
                conclusion,
                "继续的意义应是让当前完整学习器生成新对局、学习后冻结权重交手验证；不能把反复重放旧分片或冻结评测叫作在线续训。建议仍按每段256新局／64次有效更新、自身前段学习权重采集、三段真实回放推进，在预设边界同时保留冻结86、当前134及生产26作为参照。暂停前各分支的模型、优化器、调度、计数、配置、全部RNG和Value尺度应完整继承，保留原权重供回退。",
                "本次结果不足以归因‘哪个修复带来全部提升’：存在共同训练机制、数据分布、训练长度等变化。最近标签、降噪、修剪因素未通过各自外部采用条件，并不表示从最初到现在没有学到东西；相反，纵向交手直接检验了这个问题。也不能因为打赢老模型就无边界运行相同因素。",
                source_recommendation+"恢复后的目标须兼顾历史自对战与冻结强启发式，避免只适应打败某一个旧网络。",
                "建议下一批先固定24段，在第12／24段保存完整权重并检查棋力；每段持续产生新经验。原样单支的工作量是6144新局／1536更新。如果要检验新的更新比例等因素，则另行预注册同期两支，明确唯一改动，不能把冻结134称作同期在线对照。最终边界外部增益仍未确认时结束复盘，不能按轮数盲目延长。这里是方案建议，尚未登记或启动新训练。",
                "本次只评估是否值得继续，**训练仍暂停，自动训练启动关闭**。后续新训练须使用独立登记目录，保护旧STOP和阴性结果；不能清除旧标记恢复旧实验。", "",
                "## 实际工作量与完整性", "",
                f"本次新增 **9072局冻结比赛、0局训练、0次学习更新**。其中648局是对同一候选134对86的重复校验；8424局为其余对比记录。每种预算14组交手、每组324局；相同布局／续局、互换执色、BF16物理batch32、cpuct1.5、原生规则及权重冻结。9072局不能当成9072份独立证据。",
                f"所有终局已经原生规则逐步回放，合计{sum(q['native_plies'] for q in qa.values()):,}步；两种预算各324局重复比赛完整棋路、终局原因、得分精确一致。原始W/D/L与家族均值重新核对，两项主比较区间独立重算一致，历史STOP、CURRENT角色和检查点摘要均保持一致。",
                "两个134完整状态都通过CPU续训完整性检查，模型和优化器数值有限，优化器、调度、计数、Python／NumPy／Torch／CUDA／批次抽样RNG及Value尺度齐全。保存时某一参数组学习率为0是Policy／Value分阶段更新的状态；每次更新会按其阶段重设各组学习率，不能由这一保存值判定Value永久停止学习。累计更新计数含继承历史，不能把它全部算作本次新在线训练。",
                "开发集曾多次用于研究，置信区间只刻画这次家族重采样的不确定性，不能消除历史选择偏差或证明最终布局泛化。最终集未打开，开发棋谱未写入训练replay。",
                "本次补齐的是bare／64搜索对历史网络的直接交手，不是对082的bare／64搜索复测；它不填补此前历史外部强度曲线中的那类缺失点。",
                f"bare实际比赛用时{result['outputs']['0']['arena_seconds']:.1f}秒；64搜索实际比赛用时{result['outputs']['64']['arena_seconds']:.1f}秒。时间不含原生回放、文件核验、初始化；完整墙钟区间另存EVIDENCE。两种预算同时提高了双方能力，交手比例的变化不是单个模型的搜索增益估计。", "",
                "## 可追溯材料", "",
                f"- [全部比赛得分和W/D/L]({(output/'scores.csv').as_posix()})",
                f"- [预注册与完整来源]({(run/'CONTRACT.json').as_posix()})",
                f"- [比赛结果与续训信号]({(run/'RESULTS.json').as_posix()})",
                f"- [完整学习状态的CPU核验]({(run/'continuation_integrity_cpu.json').as_posix()})",
                f"- [bare原始棋谱]({(run/'arena_0.json').as_posix()})",
                f"- [64搜索原始棋谱]({(run/'arena_64.json').as_posix()})",
                f"- [历史外部强度曲线]({(ROOT/'reports/strength_history_20261005/REPORT_zh-CN.md').as_posix()})", ""]
    spaced=[]
    for index,line in enumerate(content):
        spaced.append(line)
        if line and not line.startswith(("#","|","-","![")) and index+1<len(content):
            following=content[index+1]
            if following and not following.startswith(("#","|","-","![")):
                spaced.append("")
    (output/"REPORT_zh-CN.md").write_text("\n".join(spaced),encoding="utf-8")
    evidence = dict(created_at=datetime.now(timezone.utc).isoformat(),run=str(run.relative_to(ROOT)),
                    results_sha256=sha(run/"RESULTS.json"),primary=primary,verification=qa,
                    nonduplicate_comparison_games=8424,duplicate_calibration_games=648,
                    new_training_games=0,effective_updates=0,weights_and_archives_unchanged=True,
                    complete_wall_seconds=(datetime.fromisoformat(status["completed_at"])-datetime.fromisoformat(status["started_at"])).total_seconds(),
                    figure="historical_self_gauntlet.png",visual_qa=False,
                    files={name:sha(output/name) for name in ["scores.csv","REPORT_zh-CN.md","historical_self_gauntlet.png","historical_self_gauntlet.svg"]})
    (output/"EVIDENCE.json").write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"primary":primary,"latest_duel64":duel,
                      "verification":qa,"complete_wall_seconds":evidence["complete_wall_seconds"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--run",required=True)
    args=parser.parse_args()
    build((ROOT/args.run).resolve())
