from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def config_run(root, config_path):
    root = Path(root).resolve()
    config_path = (root / config_path).resolve()
    config = read_json(config_path)
    run = (root / config["run_directory"]).resolve()
    if not config_path.is_relative_to(root) or not run.is_relative_to(root) or run == root:
        raise ValueError("configuration and run must stay inside alphazero")
    return config_path, run, config


class ProcessLock:
    """OS-managed lock: a process exit releases ownership, even after a crash."""

    def __init__(self, path):
        self.path, self.handle = Path(path), None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("a+b")
        self.handle.seek(0, os.SEEK_END)
        if self.handle.tell() == 0:
            self.handle.write(b"0")
            self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f"another process owns {self.path.name}") from None
        return self

    def __exit__(self, *unused):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def lock_held(path):
    try:
        with ProcessLock(path):
            return False
    except RuntimeError:
        return True


def current_iteration(run):
    path = Path(run) / "CURRENT.json"
    return int(read_json(path)["iteration"]) if path.exists() else 0


def supervision_status(run):
    run = Path(run)
    path = run / "supervision/STATUS.json"
    status = read_json(path) if path.exists() else {"state": "not_started"}
    status.update(supervisor_active=lock_held(run / "supervision/SUPERVISOR.lock"),
                  training_active=lock_held(run / "supervision/TRAINING.lock"),
                  stop_requested=(run / "supervision/STOP").exists(),
                  completed_iteration=current_iteration(run))
    return status
