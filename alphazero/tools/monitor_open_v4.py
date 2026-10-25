from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from open_v4_runtime import config_run, lock_held

ROOT = Path(__file__).resolve().parents[1]


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return fallback if fallback is not None else {}


def locked(path):
    if not path.exists():
        return False
    try:
        return lock_held(path)
    except OSError:
        return None


def tail(path, count=65536):
    try:
        with path.open("rb") as handle:
            handle.seek(max(0, path.stat().st_size - count))
            return handle.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def age(value):
    try:
        return max(0., (datetime.now(timezone.utc) - datetime.fromisoformat(value)).total_seconds())
    except (TypeError, ValueError):
        return None


def gpu_stats():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3,
                                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        if result.returncode:
            return []
        return [dict(zip(("name", "utilization", "memory_used_mib", "memory_total_mib", "temperature", "power_watts"),
                         [cell.strip() for cell in row])) for row in csv.reader(result.stdout.splitlines())]
    except (OSError, subprocess.TimeoutExpired):
        return []


def snapshot(run, config):
    status = read_json(run / "supervision/STATUS.json")
    current = read_json(run / "CURRENT.json")
    committed = int(current.get("iteration", 0))
    supervisor, training = locked(run / "supervision/SUPERVISOR.lock"), locked(run / "supervision/TRAINING.lock")
    active = status.get("iteration", committed) if supervisor or training else committed
    log = ""
    if status.get("log"):
        path = Path(status["log"]).resolve()
        if path.is_relative_to(run):
            log = tail(path)
    # Only the current child log provides uncommitted progress.
    live = log if (supervisor or training) and active > committed else ""
    learner = list(re.finditer(r"\[learner\] updates=(\d+)/(\d+) effective=(\d+) phase=(\w+) loss=([\d.eE+-]+)", live))
    collection = list(re.finditer(r"\[collect\] iteration=(\d+) round=(\d+) active=(\d+) completed=(\d+)/(\d+)", live))
    arena = list(re.finditer(r"\[arena\] round=(\d+) active=(\d+) completed=(\d+)", live))
    progress = dict(phase="准备 / 提交检查点", text="")
    if arena:
        rnd, remaining, complete = map(int, arena[-1].groups())
        progress = dict(phase="开发对局评测", text=f"已结束 {complete}/{complete+remaining} 局，剩余 {remaining} 局；续局 ply {rnd}")
    elif learner:
        done, total, effective, phase, loss = learner[-1].groups()
        progress = dict(phase="训练更新", text=f"本轮 {done}/{total} 步，{phase} loss={loss}")
    elif collection:
        iteration, rnd, remaining, complete, total = collection[-1].groups()
        progress = dict(phase="完整对局采集", text=f"已结束 {complete}/{total} 局，剩余 {remaining} 局；ply {rnd}")
    if learner:
        progress.update(uncommitted_effective_updates=int(learner[-1].group(3)), iteration_updates=int(learner[-1].group(1)),
                        iteration_update_target=int(learner[-1].group(2)))
    if not (supervisor or training):
        progress = dict(phase="未运行", text=status.get("reason", ""))
    optimization = read_json(run / "supervision/PLAN.json").get("optimization", {})
    experiment = {}
    if optimization.get("active") and optimization.get("run_directory"):
        experiment_run = (ROOT / optimization["run_directory"]).resolve()
        if experiment_run.is_relative_to(ROOT):
            experiment = read_json(experiment_run / "STATUS.json")
            if experiment and (supervisor or training):
                if optimization.get("stdout"):
                    experiment_log = (ROOT / optimization["stdout"]).resolve()
                    if experiment_log.is_relative_to(ROOT):
                        log = tail(experiment_log)
                        # A new variant may share the same collection number. Hide
                        # the preceding phase's last line until this phase writes.
                        try:
                            phase_started = datetime.fromisoformat(experiment.get("updated_at", "")).timestamp()
                            if experiment_log.stat().st_mtime < phase_started:
                                log = ""
                        except (OSError, ValueError):
                            pass
                status = dict(status, state="optimization_running")
                variant = {"control": "原方式对照", "policy": "仅策略更新", "wide": "扩大回放窗口", "fresh": "增加新对局", "candidate": "256 次搜索数据"}.get(experiment.get("variant"), "")
                if optimization.get("type") == "selfplay_teacher_policy_target":
                    variant = {"control": "原 MCTS 策略目标", "candidate": "教师策略目标"}.get(experiment.get("variant"), "")
                if optimization.get("type") == "policy_phase_encoder_lr":
                    variant = {"control": "原策略编码器学习率", "candidate": "策略阶段固定编码器"}.get(experiment.get("variant"), "")
                if optimization.get("type") == "value_phase_encoder_lr":
                    variant = {"control": "原 Value 编码器学习率", "candidate": "提高 Value 编码器学习率"}.get(experiment.get("variant"), "")
                if optimization.get("type") == "value_source_eligibility":
                    variant = {"control": "原全部 Value 数据", "candidate": "Value 仅用双方同策略对局"}.get(experiment.get("variant"), "")
                if optimization.get("type") == "experimental_actor_refresh":
                    variant = {"control": "固定第 26 轮采集对照", "candidate": "用上一段学习器采集新对局"}.get(experiment.get("variant"), "")
                if experiment.get("state") == "training":
                    stage = f"第 {experiment['stage']}/{experiment.get('stages','—')} 段；" if "stage" in experiment else ""
                    progress = dict(phase="训练对照实验", text=f"{stage}{variant}：{experiment.get('completed_updates',0)}/{experiment.get('updates','—')} 步，独立检查点已保存")
                elif experiment.get("state") == "collecting":
                    progress = dict(phase="采集新训练对局", text=f"第 {experiment.get('stage','—')}/{experiment.get('stages','—')} 段，第 {experiment.get('chunk','—')}/{experiment.get('chunk_total','—')} 批，{variant}；12 个 CPU 教师与 GPU 搜索并发")
                    latest_collection = re.search(r"\[collect\] iteration=(\d+) round=(\d+) active=(\d+) completed=(\d+)/(\d+)", log.splitlines()[-1] if log.splitlines() else "")
                    if latest_collection:
                        _, ply, remaining, complete, total = latest_collection.groups()
                        progress["text"] += f"；本批 {complete}/{total} 局，ply {ply}"
                elif experiment.get("state") == "labeling":
                    progress = dict(phase="重建训练轨迹与教师标注", text=f"分片 {experiment.get('shard','—')}；已完成 {experiment.get('label_shards_done',0)}/{experiment.get('label_shards_total','—')}；仅替换非循环自我对弈策略目标")
                    latest_labels = re.search(r"\[labels\] shard=(\S+) reconstructed_games=(\d+)/256 changed_policy_rows=(\d+)", log.splitlines()[-1] if log.splitlines() else "")
                    if latest_labels:
                        _, complete, changed = latest_labels.groups()
                        progress["text"] += f"；本片重建 {complete}/256 局，标注 {changed} 个局面"
                elif experiment.get("state") == "calibrating":
                    progress = dict(phase="训练数据 Value 校准审计", text=f"冻结网络；内部留出 {experiment.get('internal_samples','—')} 个局面；不更新权重")
                elif experiment.get("state") == "arena":
                    progress = dict(phase="实验配对棋力评测", text="比较对照、候选和冻结基线；原主线检查点保留")
                else:
                    progress = dict(phase="优化实验", text=experiment.get("state", "检查中"))
    reports = []
    for path in sorted((run / "reports").glob("iteration_*.json"), reverse=True):
        match = re.search(r"iteration_(\d+)", path.name)
        if match and int(match.group(1)) <= committed:
            report = read_json(path)
            if report:
                reports.append(report)
        if len(reports) == 6:
            break
    arenas = sorted((run / "arena").glob("actor_*.json"), reverse=True)
    latest_arena = {}
    if arenas:
        latest_arena = read_json(arenas[0])
        latest_arena["iteration"] = int(re.search(r"actor_(\d+)", arenas[0].name).group(1))
        latest_arena["path"] = str(arenas[0])
    return dict(run=str(run), config=config, status=status, current=current, committed_iteration=committed,
                active_iteration=active, supervisor_active=supervisor, training_active=training,
                stop_requested=(run / "supervision/STOP").exists(), heartbeat_age=age(status.get("last_heartbeat_at")),
                progress=progress, reports=reports, arena=latest_arena, experiment=experiment, last_log_lines=log.splitlines()[-4:],
                free_gib=shutil.disk_usage(run if run.exists() else ROOT).free/2**30)


def number(value, digits=3):
    return f"{value:.{digits}f}" if isinstance(value, (int, float)) else "—"


def pct(value):
    return f"{value*100:.1f}%" if isinstance(value, (int, float)) else "—"


def width(value):
    return sum(0 if unicodedata.combining(c) else (2 if unicodedata.east_asian_width(c) in "WF" else 1) for c in str(value))


def table(headers, rows):
    widths = [max(width(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    def line(row):
        return "  ".join(str(cell) + " " * (widths[i]-width(cell)) for i, cell in enumerate(row))
    return [line(headers), "  ".join("-" * size for size in widths)] + [line(row) for row in rows]


def render(data, gpus):
    height = shutil.get_terminal_size(fallback=(120, 45)).lines
    compact = height < 30
    st, current, progress = data["status"], data["current"], data["progress"]
    active = data["supervisor_active"] or data["training_active"]
    names = {"running": "运行中", "starting": "启动中", "review_due": "待分析并接续", "needs_attention": "需要检查",
             "optimization_running": "运行独立优化实验",
             "stopped_by_user": "已停止", "stopping_after_iteration": "保存本轮后停止", "retrying": "正在恢复"}
    state = names.get(st.get("state"), "尚未启动")
    if st.get("state") in ("running", "starting", "retrying", "stopping_after_iteration") and not active:
        state = "进程已退出（查看最后日志）"
    heartbeat = f"{data['heartbeat_age']:.0f} 秒前" if data["heartbeat_age"] is not None else "—"
    if data.get("experiment") and active:
        heartbeat = "已切换至独立实验"
    lines = ["明棋 AI · 训练监控    " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
             f"状态：{state}   当前轮：{data['active_iteration']}   已保存轮：{data['committed_iteration']}   本段目标：{st.get('target_iteration','—')}",
             f"进程：监督 {'活跃' if data['supervisor_active'] else '未运行'} / 训练 {'活跃' if data['training_active'] else '未运行'}   心跳：{heartbeat}   磁盘余量：{data['free_gib']:.1f} GiB",
             f"阶段：{progress['phase']}  {progress['text']}"]
    if data["stop_requested"]:
        lines.append("已请求停止：当前轮完整保存后退出。")
    if active and not data.get("experiment") and data["heartbeat_age"] is not None and data["heartbeat_age"] > 90:
        lines.append("注意：监督心跳超过 90 秒未更新。")
    latest = data["reports"][0] if data["reports"] else {}
    counters = latest.get("counters", {})
    lines.append(f"有效更新（已保存）：{counters.get('effective_updates','—')}   Policy：{counters.get('policy_updates','—')}   Value：{counters.get('value_updates','—')}   拒绝：{counters.get('backtracking_rejections','—')}")
    if data.get("experiment"):
        lines.append(f"实验状态：{data['experiment'].get('state','—')}；主线有效更新与晋级版本不受实验更新计数影响。")
    if "uncommitted_effective_updates" in progress:
        lines.append(f"本轮已执行：{progress['iteration_updates']}/{progress['iteration_update_target']}   暂存累计：{progress['uncommitted_effective_updates']}（评测后提交）")
    roles = []
    for role, label in (("learner", "训练模型"), ("actor", "出棋模型"), ("champion", "冠军")):
        row = current.get(role, {})
        roles.append(f"{label} 第{row.get('iteration','—')}轮")
    lines += ["已保存版本：" + " | ".join(roles), ""]
    for gpu in gpus:
        lines.append(f"GPU：利用率 {gpu['utilization']}%   显存 {gpu['memory_used_mib']}/{gpu['memory_total_mib']} MiB   {gpu['temperature']}°C   {gpu['power_watts']} W")
    if gpus:
        lines.append("")
    lines.append("最近完整轮次（CE 为内部验证损失，棋力看对局得分）")
    rows = []
    history_count = max(1, min(6, height - 28))
    for report in reversed(data["reports"][:history_count]):
        c, metric, gate = report.get("counters",{}), report.get("internal_after",{}), report.get("gate",{})
        promotion = "冠军晋级" if gate.get("champion_accepted") else ("出棋晋级" if gate.get("actor_accepted") else ("未安排" if gate.get("status")=="gate not scheduled" else "未晋级"))
        rows.append([str(report["iteration"]),str(c.get("effective_updates","—")), number(metric.get("family_policy_ce")),
                     number(metric.get("family_wdl_ce")),number(report.get("collection",{}).get("moves_per_second"),1),
                     number(report.get("elapsed_seconds"),0),promotion])
    if compact and rows:
        row=rows[-1]
        lines.append(f"第{row[0]}轮：Policy CE {row[2]}，WDL CE {row[3]}，{row[4]}步/秒，门禁 {row[6]}")
    else:
        lines += table(["轮次","有效更新","Policy CE","WDL CE","步/秒","本轮秒数","门禁"], rows) if rows else ["暂无已提交训练报告。"]
    arena = data["arena"]
    lines.append("")
    if arena.get("summaries"):
        note = "已提交轮次" if arena["iteration"] <= data["committed_iteration"] else "本轮未提交，暂不代表晋级"
        lines.append(f"最近开发评测：第 {arena['iteration']} 轮（{note}；得分=胜1/和0.5/负0）")
        labels = dict(learner="训练模型", actor="出棋模型", champion="冠军", candidate="训练模型",
                      parent="原学习器", control="小数据对照", parent_repeat="原学习器复测")
        for index, path in enumerate(data["config"].get("teacher_candidates", [])):
            name=Path(path).parent.name.split("_")[0]
            labels[f"teacher{index}"]=f"启发式{int(name):03d}" if name.isdigit() else f"教师{index}"
        rows=[]
        for row in arena["summaries"]:
            metric=row.get("family_score",{});ci=metric.get("ci95",[])
            rows.append([labels.get(row["candidate"],row["candidate"])+" vs "+labels.get(row["opponent"],row["opponent"]),
                         pct(metric.get("mean")), f"{pct(ci[0])}–{pct(ci[1])}" if len(ci)==2 else "—",
                         str(row.get("games","—")),str(metric.get("families","—"))])
        lines += table(["对阵","家族得分","95% 区间","局数","家族对"],rows)
    else:
        lines.append("暂无本次训练运行的开发评测结果，正在等待首个门禁。")
    if not compact:
        lines += ["", "最新日志："] + ["  "+line[:120] for line in data["last_log_lines"][-1:]]
    lines += ["", "Ctrl+C 退出监控；后台训练继续。"]
    return "\n".join(lines)


def enable_terminal():
    if not sys.stdout.isatty():
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel=ctypes.windll.kernel32
        kernel.GetStdHandle.argtypes=[wintypes.DWORD]
        kernel.GetStdHandle.restype=wintypes.HANDLE
        kernel.GetConsoleMode.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
        kernel.SetConsoleMode.argtypes=[wintypes.HANDLE,wintypes.DWORD]
        handle=kernel.GetStdHandle(-11)
        mode=wintypes.DWORD()
        return bool(kernel.GetConsoleMode(handle,ctypes.byref(mode)) and kernel.SetConsoleMode(handle,mode.value|4))
    return True


def main():
    parser=argparse.ArgumentParser(description="Live terminal dashboard; reads training state without importing Torch")
    parser.add_argument("--config",default="config/open_junqi_v4_repair_pilot.json")
    parser.add_argument("--refresh",type=float,default=5.)
    parser.add_argument("--once",action="store_true")
    parser.add_argument("--json",action="store_true")
    parser.add_argument("--no-gpu",action="store_true")
    args=parser.parse_args()
    if args.refresh<1:parser.error("refresh interval must be at least one second")
    if hasattr(sys.stdout,"reconfigure"):sys.stdout.reconfigure(encoding="utf-8",errors="replace")
    _,run,config=config_run(ROOT,args.config)
    terminal=enable_terminal() and not args.once and not args.json
    gpus, gpu_at=[],0.
    try:
        while True:
            if not args.no_gpu and time.monotonic()-gpu_at>=10:
                gpus,gpu_at=gpu_stats(),time.monotonic()
            continuation=read_json(run / "supervision/PLAN.json").get("continuation", {})
            if continuation.get("active") and continuation.get("config"):
                _,active_run,active_config=config_run(ROOT,continuation["config"])
            else:
                active_run,active_config=run,config
            data=snapshot(active_run,active_config)
            if args.json:
                data["gpus"]=gpus
                print(json.dumps(data,ensure_ascii=False,indent=2))
            else:
                print(("\033[H\033[2J" if terminal else "")+render(data,gpus),flush=True)
            if args.once or args.json:break
            time.sleep(args.refresh)
    except KeyboardInterrupt:
        print("\n监控已退出，后台训练继续。")


if __name__=="__main__":
    main()
