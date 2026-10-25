"""CPU-only teacher workers; importing this module never imports Torch or CUDA."""
from __future__ import annotations

import hashlib
import importlib.util
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def load_candidate(path: Path):
    # Same exact-file loader as the previous evaluation helper, without Torch.
    spec = importlib.util.spec_from_file_location("open_junqi_search_v3_eval_exact082", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_WORKER_MODULES = []


def _initialize_worker(paths):
    global _WORKER_MODULES
    _WORKER_MODULES = [load_candidate(Path(path)) for path in paths]


def _select_snapshot(snapshot):
    board, legal, teacher_id = snapshot
    action = int(_WORKER_MODULES[teacher_id].select_action(list(board), list(legal), 1, 0., 0.))
    if action not in legal:
        raise RuntimeError("teacher returned an illegal action")
    return action, time.perf_counter()


class TeacherBatch:
    def __init__(self, futures, started):
        self.futures = futures
        self.started = started
        self.elapsed_seconds = 0.

    def result(self):
        rows = [future.result() for future in self.futures]
        # Completion timestamps exclude time spent waiting for GPU work before
        # the coordinator reads an already completed batch.
        self.elapsed_seconds = max((row[1] for row in rows), default=self.started) - self.started
        return [row[0] for row in rows]


class TeacherLeague:
    def __init__(self, root: Path, candidates: list[str], workers: int = 12):
        if workers < 1:
            raise ValueError("teacher workers must be positive")
        self.modules, self.provenance = [], []
        self.workers = workers
        paths = [(root / name).resolve() for name in candidates]
        for path in paths:
            module = load_candidate(path)
            binary = getattr(module, "_BINARY", None)
            self.modules.append(module)
            self.provenance.append({"name": path.parent.name, "candidate": str(path),
                "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "binary_sha256": hashlib.sha256(Path(binary).read_bytes()).hexdigest() if binary else None,
                "budget": "candidate's unchanged native budget; elapsed time measured"})
        if not self.modules:
            raise ValueError("teacher league is empty")
        self.executor = ProcessPoolExecutor(max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_initialize_worker, initargs=([str(path) for path in paths],))

    def submit_actions(self, states, teacher_ids):
        if len(states) != len(teacher_ids):
            raise ValueError("teacher ids and states must have equal lengths")
        # Snapshot before any search or state mutation; no native state or GPU
        # objects cross the process boundary. Results retain the input order.
        snapshots = [(tuple(map(int, state.native_state.board())),
                      tuple(map(int, state.legal_actions())),
                      int(teacher_id) % len(self.modules))
                     for state, teacher_id in zip(states, teacher_ids)]
        started = time.perf_counter()
        return TeacherBatch([self.executor.submit(_select_snapshot, row) for row in snapshots], started)

    def actions(self, states, teacher_ids):
        return self.submit_actions(states, teacher_ids).result()

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()
