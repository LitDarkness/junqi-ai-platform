from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from open_v4_runtime import (ProcessLock, atomic_json, config_run, current_iteration,
                             lock_held, read_json, supervision_status, utc_now)

ROOT = Path(__file__).resolve().parents[1]


def log_tail(path):
    if not path.exists():
        return ""
    with path.open("rb") as handle:
        handle.seek(max(0, path.stat().st_size - 3000))
        return handle.read().decode("utf-8", errors="replace").strip()


def report_summary(run, iteration):
    report = read_json(run / f"reports/iteration_{iteration:06d}.json")
    return {"iteration": iteration, "counters": report["counters"], "gate": report["gate"],
            "internal_after": report["internal_after"], "elapsed_seconds": report["elapsed_seconds"],
            "moves_per_second": report["collection"]["moves_per_second"],
            "peak_gpu_allocated_mib": report["peak_gpu_allocated_mib"], "roles": report["roles"]}


def supervise(root, config_path, *, stage_iterations=18, minimum_free_gib=10., poll_seconds=15., runner=None, extra_stop_paths=()):
    root = Path(root).resolve()
    config_path, run, config = config_run(root, config_path)
    folder = run / "supervision"
    folder.mkdir(parents=True, exist_ok=True)
    stop, status_path = folder / "STOP", folder / "STATUS.json"
    def stop_requested():
        return stop.exists() or any(Path(path).exists() for path in extra_stop_paths)
    with ProcessLock(folder / "SUPERVISOR.lock"):
        start = current_iteration(run)
        status = dict(schema="open-junqi-background-training-1", state="starting", config=str(config_path),
                      run=str(run), supervisor_pid=os.getpid(), child_pid=None, started_at=utc_now(),
                      segment_start_iteration=start, target_iteration=start + stage_iterations,
                      completed_iteration=start, review_required=False)
        def update(**values):
            status.update(values)
            status["last_heartbeat_at"] = utc_now()
            atomic_json(status_path, status)
        update()
        for iteration in range(start + 1, start + stage_iterations + 1):
            if stop_requested():
                update(state="stopped_by_user", child_pid=None)
                return status
            free = shutil.disk_usage(run).free / 2**30
            if free < minimum_free_gib:
                update(state="needs_attention", review_required=True, reason="disk_space", free_gib=free)
                return status
            if lock_held(folder / "TRAINING.lock"):
                update(state="needs_attention", reason="another_training_process", review_required=True)
                return status
            for attempt in range(2):
                stamp = time.time_ns()
                path = folder / f"iteration_{iteration:06d}_attempt_{attempt + 1}_{stamp}.log"
                command = [sys.executable, "-X", "faulthandler", str(runner or root / "tools/run_open_v4.py"),
                           "--config", str(config_path), "--through-iteration", str(iteration)]
                env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1",
                           PYTHONPATH=os.pathsep.join([str(root / "build"), str(root / "python"), os.environ.get("PYTHONPATH", "")]))
                try:
                    with path.open("w", encoding="utf-8") as output:
                        child = subprocess.Popen(command, cwd=root, env=env, stdout=output, stderr=subprocess.STDOUT,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                        update(state="running", iteration=iteration, attempt=attempt + 1, child_pid=child.pid, log=str(path), free_gib=free)
                        while child.poll() is None:
                            update(state="stopping_after_iteration" if stop_requested() else "running", last_progress=log_tail(path)[-1000:])
                            time.sleep(poll_seconds)
                    code = child.returncode
                except Exception as error:
                    update(state="needs_attention", reason=repr(error), review_required=True)
                    raise
                committed = current_iteration(run)
                update(child_pid=None, child_exit_code=code, completed_iteration=committed, last_progress=log_tail(path)[-1000:])
                if code == 0 and committed == iteration:
                    last = report_summary(run, iteration)
                    update(last_committed=last)
                    print(json.dumps(last, ensure_ascii=False), flush=True)
                    if last["counters"]["effective_updates"] == 0 or sum(not x["accepted"] for x in read_json(run / f"reports/iteration_{iteration:06d}.json")["training"]) > config["updates_per_iteration"] / 4:
                        update(state="review_due", reason="ineffective_updates", review_required=True)
                        return status
                    break
                tail = log_tail(path)
                fatal = any(marker in tail for marker in ("contract changed", "contract mismatch", "digest mismatch", "provenance mismatch"))
                if stop_requested():
                    update(state="stopped_by_user")
                    return status
                if fatal or attempt == 1 or committed != iteration - 1:
                    update(state="needs_attention", reason="training_failed_or_unexpected_commit", review_required=True)
                    return status
                update(state="retrying", reason="retry_once_from_last_committed_checkpoint")
                time.sleep(min(5., poll_seconds))
        update(state="stopped_by_user" if stop_requested() else "review_due", child_pid=None,
               review_required=not stop_requested(), reason="segment_complete")
        return status


def main():
    parser = argparse.ArgumentParser(description="Checkpointed background Junqi training with single-writer locks")
    parser.add_argument("--config", default="config/open_junqi_v4_repair_pilot.json")
    parser.add_argument("--stage-iterations", type=int, default=18)
    parser.add_argument("--minimum-free-gib", type=float, default=10.)
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.stage_iterations < 1 or args.minimum_free_gib < 1:
        parser.error("positive stage and disk budgets required")
    _, run, _ = config_run(ROOT, args.config)
    if args.status:
        result = supervision_status(run)
    else:
        try:
            result = supervise(ROOT, args.config, stage_iterations=args.stage_iterations, minimum_free_gib=args.minimum_free_gib)
        except RuntimeError as error:
            if "another process owns" not in str(error):
                raise
            result = supervision_status(run)
            result["launch_result"] = "already_running"
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if result.get("state") == "needs_attention":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
