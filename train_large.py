"""Supervised long-running PPO training with live health/progress reporting."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROFILES = {
    "quick": dict(num_envs=64, num_steps=64, shards=4, iters=100, minibatch=1024, epochs=2, save_freq=25),
    "standard": dict(num_envs=192, num_steps=64, shards=12, iters=5000, minibatch=2048, epochs=4, save_freq=250),
    "large": dict(num_envs=256, num_steps=64, shards=16, iters=20000, minibatch=4096, epochs=4, save_freq=250),
}


def run_checked(label: str, command: list[str]) -> None:
    print(f"[预检] {label}", flush=True)
    completed = subprocess.run(command, cwd=ROOT)
    if completed.returncode:
        raise SystemExit(f"预检失败：{label} (exit={completed.returncode})")


def latest_row(metrics_path: Path) -> dict | None:
    try:
        with metrics_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            end = handle.tell()
            if end == 0:
                return None
            cursor = end - 1
            while cursor > 0:
                handle.seek(cursor)
                if handle.read(1) == b"\n" and cursor < end - 1:
                    break
                cursor -= 1
            handle.seek(cursor + (1 if cursor else 0))
            return json.loads(handle.readline().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def gpu_status() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            util, used, total, temp = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
            return f"GPU={util}% VRAM={used}/{total}MiB {temp}°C"
    except (OSError, subprocess.TimeoutExpired, ValueError):
        pass
    return "GPU=不可用"


def launch_evaluator(model: Path, report_dir: Path, games: int, max_steps: int) -> subprocess.Popen:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = report_dir / f"eval_{stamp}.json"
    markdown = report_dir / f"eval_{stamp}.md"
    log = (report_dir / f"eval_{stamp}.log").open("w", encoding="utf-8")
    command = [
        sys.executable, "-u", "evaluate_suite.py", "--bots", str(model), "random", "heuristic_v3",
        "--games-per-color", str(games), "--max-steps", str(max_steps), "--device", "cpu",
        "--output", str(output), "--markdown", str(markdown),
    ]
    print(f"[检测] 启动独立 CPU 评测进程 -> {output.name}", flush=True)
    return subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Large PPO training supervisor")
    parser.add_argument("--profile", choices=PROFILES, default="standard")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--resume", default=None, help="Model .pth to resume from (weights only).")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the resolved training command without launching it.")
    parser.add_argument("--monitor-seconds", type=int, default=10)
    parser.add_argument("--stall-minutes", type=int, default=15)
    parser.add_argument("--eval-interval-minutes", type=int, default=30, help="0 disables external evaluation.")
    parser.add_argument("--eval-games", type=int, default=5, help="Games per color and opponent.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[], help="Arguments appended to train.py.")
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    if not args.skip_preflight:
        run_checked("Python 回归测试", [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])
        run_checked("完整规则一致性", [sys.executable, "rule_sanity.py"])
        run_checked("单核性能门槛", [sys.executable, "performance_gate.py"])

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.run_dir).resolve() if args.run_dir else ROOT / "runs" / f"{args.profile}_{stamp}"
    save_dir, pool_dir, report_dir = run_dir / "checkpoints", run_dir / "pool", run_dir / "evaluations"
    command = [
        sys.executable, "-u", "train.py", "--backend", "cpp-sharded",
        "--rollout-shards", str(profile["shards"]),
        "--num-envs", str(profile["num_envs"]), "--num-steps", str(profile["num_steps"]),
        "--iters", str(profile["iters"]), "--ppo-minibatch-size", str(profile["minibatch"]),
        "--ppo-update-epochs", str(profile["epochs"]), "--save-freq", str(profile["save_freq"]),
        "--save-dir", str(save_dir), "--pool-dir", str(pool_dir), "--eval-freq", "0",
        # Rollout shards send model-opponent requests back to one centralized
        # CUDA cache; this avoids one Torch model/context per worker.
        "--opponent-device", "cuda", "--opponent-latest-prob", "0.35",
        "--opponent-history-prob", "0.25", "--opponent-heuristic-prob", "0.30",
        "--opponent-random-prob", "0.10",
    ]
    if args.resume:
        command += ["--load-model", str(Path(args.resume).resolve())]
    command += args.extra
    if args.dry_run:
        print("[DRY-RUN] " + subprocess.list2cmdline(command))
        print(f"[DRY-RUN] 运行目录：{run_dir}")
        return
    for directory in (save_dir, pool_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    (run_dir / "command.json").write_text(json.dumps(command, ensure_ascii=False, indent=2), encoding="utf-8")
    log_handle = (run_dir / "train.log").open("w", encoding="utf-8", buffering=1)
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(command, cwd=ROOT, stdout=log_handle, stderr=subprocess.STDOUT, creationflags=flags)
    print(f"[训练] PID={process.pid} profile={args.profile} 目录={run_dir}", flush=True)
    print(f"[训练] 日志={run_dir / 'train.log'}；TensorBoard={save_dir / 'logs'}", flush=True)
    metrics = save_dir / "metrics.jsonl"
    started = last_progress = time.time()
    last_iteration = 0
    # Do not compete with worker startup/model loading; first external
    # evaluation happens after a full configured interval.
    last_eval = time.time()
    eval_process: subprocess.Popen | None = None
    try:
        while process.poll() is None:
            row = latest_row(metrics)
            now = time.time()
            if row and int(row.get("iteration", 0)) > last_iteration:
                last_iteration = int(row["iteration"])
                last_progress = now
            if row:
                elapsed = now - started
                rate = last_iteration / max(elapsed, 1e-9)
                eta = (profile["iters"] - last_iteration) / max(rate, 1e-9)
                print(
                    f"[进度] {last_iteration}/{profile['iters']} ({last_iteration/profile['iters']:.1%}) "
                    f"ETA={eta/3600:.2f}h return={float(row.get('avg_return', 0)):.3f} "
                    f"rollout={float(row.get('rollout_tps', 0)):.0f} steps/s "
                    f"actor={float(row.get('actor_loss', 0)):.4f} critic={float(row.get('critic_loss', 0)):.4f} "
                    f"GPU-duty~={100*float(row.get('update_seconds', 0))/max(float(row.get('rollout_seconds', 0))+float(row.get('update_seconds', 0)), 1e-9):.0f}% "
                    f"{gpu_status()}", flush=True,
                )
            else:
                print(f"[等待] 首个 iteration 尚未完成；{gpu_status()}", flush=True)
            if now - last_progress > args.stall_minutes * 60:
                print(f"[警告] {args.stall_minutes} 分钟没有新指标；训练仍存活，检查 train.log。", flush=True)
                last_progress = now
            latest_model = pool_dir / "latest.pth"
            due = args.eval_interval_minutes > 0 and now - last_eval >= args.eval_interval_minutes * 60
            if due and latest_model.exists() and (eval_process is None or eval_process.poll() is not None):
                eval_process = launch_evaluator(latest_model, report_dir, args.eval_games, 200)
                last_eval = now
            time.sleep(max(1, args.monitor_seconds))
    except KeyboardInterrupt:
        print("[中断] 正在请求训练保存 model_interrupted.pth...", flush=True)
        if os.name == "nt":
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.terminate()
    finally:
        log_handle.close()
    code = process.wait()
    if code:
        raise SystemExit(f"训练异常退出：exit={code}，日志：{run_dir / 'train.log'}")
    print(f"[完成] 训练正常结束，模型与指标位于 {run_dir}")


if __name__ == "__main__":
    main()
