"""One unchanged native teacher per CPU worker, with actual-file receipts.

Never load two identically named PyInit_fast_rule_ab modules in one process.
The legacy module remains frozen for reproducibility of historical evidence.
"""
from __future__ import annotations
import hashlib
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from importlib.machinery import EXTENSION_SUFFIXES
from pathlib import Path
from .teachers import load_candidate

_MODULE = None
_RECEIPT = None

def _initialize(path, expected, queue):
    global _MODULE, _RECEIPT
    import os
    _MODULE = load_candidate(Path(path))
    actual = Path(_MODULE._MODULE.__file__).resolve()
    requested = Path(_MODULE._BINARY).resolve()
    digest = hashlib.sha256(actual.read_bytes()).hexdigest()
    if actual != requested or str(actual) != expected['binary'] or digest != expected['binary_sha256']:
        raise RuntimeError('actual loaded native teacher differs from declared teacher')
    _RECEIPT = dict(pid=os.getpid(), candidate=str(Path(path).resolve()),
                    binary=str(actual), binary_sha256=digest)
    queue.put(_RECEIPT)

def _identity():
    return _RECEIPT

def _select(snapshot):
    board, legal = snapshot
    action = int(_MODULE.select_action(list(board), list(legal), 1, 0., 0.))
    if action not in legal: raise RuntimeError('teacher returned illegal action')
    return action, time.perf_counter(), _RECEIPT

class TeacherBatch:
    def __init__(self, owner, futures, started):
        self.owner, self.futures, self.started = owner, futures, started
        self.elapsed_seconds = 0.
    def result(self):
        rows = [f.result() for f in self.futures]
        for _, _, receipt in rows: self.owner._verify_receipt(receipt)
        self.elapsed_seconds = max((r[1] for r in rows), default=self.started)-self.started
        return [r[0] for r in rows]

class TeacherLeague:
    def __init__(self, root: Path, candidates: list[str], workers: int = 12):
        if not candidates or workers < len(candidates):
            raise ValueError('at least one isolated worker per teacher required')
        self.workers = workers
        self.modules, self.provenance, self.receipts, self.executors = [], [], {}, []
        context = multiprocessing.get_context('spawn')
        self.queue = context.Queue()
        paths = [(root/name).resolve() for name in candidates]
        self.expected = {}
        self.worker_counts = [workers//len(paths)+int(i<workers%len(paths)) for i in range(len(paths))]
        try:
            for i,path in enumerate(paths):
                binary = next((path.parent/f'fast_rule_ab{s}' for s in EXTENSION_SUFFIXES
                               if (path.parent/f'fast_rule_ab{s}').is_file()), None)
                if binary is None: raise ImportError('declared native teacher missing')
                expected = dict(binary=str(binary.resolve()), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
                self.expected[str(path)] = expected
                # Descriptors only. Parent process never imports a teacher binary.
                self.modules.append(dict(candidate=str(path),**expected))
                self.provenance.append(dict(name=path.parent.name,candidate=str(path),
                    source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    binary_sha256=expected['binary_sha256'],budget="candidate's unchanged native budget; elapsed time measured"))
                self.executors.append(ProcessPoolExecutor(max_workers=self.worker_counts[i],
                    mp_context=context,initializer=_initialize,initargs=(str(path),expected,self.queue)))
            warm=[pool.submit(_identity) for pool,n in zip(self.executors,self.worker_counts) for _ in range(n)]
            for f in warm: self._verify_receipt(f.result(timeout=60))
            # All workers initialize once; not just the fastest first worker.
            for _ in range(workers): self._verify_receipt(self.queue.get(timeout=60))
            if len(self.receipts)!=workers: raise RuntimeError('not all isolated workers verified')
        except BaseException:
            self.close(); raise
    def _verify_receipt(self, receipt):
        expected=self.expected[receipt['candidate']]
        if any(receipt[k]!=expected[k] for k in ('binary','binary_sha256')):
            raise RuntimeError('teacher worker identity mismatch')
        prior=self.receipts.get(receipt['pid'])
        if prior is not None and prior!=receipt: raise RuntimeError('worker changed native teacher')
        self.receipts[receipt['pid']]=dict(receipt)
    def identity_report(self):
        return dict(isolated=True,requested_workers=self.workers,worker_counts=self.worker_counts,
                    verified_workers=list(self.receipts.values()),all_workers_verified=len(self.receipts)==self.workers)
    def submit_actions(self, states, teacher_ids):
        if len(states)!=len(teacher_ids): raise ValueError('teacher request lengths differ')
        snapshots=[(tuple(map(int,s.native_state.board())),tuple(map(int,s.legal_actions()))) for s in states]
        started=time.perf_counter()
        futures=[self.executors[int(i)%len(self.executors)].submit(_select,snapshot)
                 for snapshot,i in zip(snapshots,teacher_ids)]
        return TeacherBatch(self,futures,started)
    def actions(self, states, teacher_ids): return self.submit_actions(states,teacher_ids).result()
    def close(self):
        for pool in self.executors: pool.shutdown(wait=True,cancel_futures=True)
        self.executors=[]
        self.queue.close(); self.queue.join_thread()
    def __enter__(self): return self
    def __exit__(self,*unused): self.close()
