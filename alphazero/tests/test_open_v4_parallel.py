from __future__ import annotations

import sys
import tempfile
import time
import unittest
from concurrent.futures import wait
from pathlib import Path

from junqi_open_v4.teachers import TeacherLeague


def worker_runtime():
    import os
    return os.getpid(), "torch" in sys.modules


class SnapshotState:
    def __init__(self, board, legal):
        self.values, self.legal = board, legal
        self.native_state = self

    def board(self):
        return self.values

    def legal_actions(self):
        return self.legal


class TeacherParallelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.path = self.root / "candidate.py"
        self.path.write_text("""import time
def select_action(board, legal, player, my, opp):
    time.sleep(board[0] / 1000)
    return -1 if board[1] else legal[board[2] % len(legal)]
""", encoding="utf-8")

    def tearDown(self):
        self.directory.cleanup()

    def test_inputs_snapshot_and_results_preserve_input_order(self):
        states = [SnapshotState([30, 0, 1], [10, 11]), SnapshotState([1, 0, 0], [20, 21])]
        with TeacherLeague(self.root, [str(self.path)], workers=2) as teachers:
            batch = teachers.submit_actions(states, [0, 0])
            states[0].values[:] = [0, 0, 0]
            states[0].legal[:] = [99]
            self.assertEqual(batch.result(), [11, 20])
            self.assertEqual(batch.result(), [11, 20])

    def test_completion_timing_excludes_delayed_result_read(self):
        state = SnapshotState([0, 0, 0], [7])
        with TeacherLeague(self.root, [str(self.path)], workers=1) as teachers:
            teachers.actions([state], [0])
            batch = teachers.submit_actions([state], [0])
            wait(batch.futures)
            time.sleep(.08)
            self.assertEqual(batch.result(), [7])
            self.assertGreater(time.perf_counter() - batch.started - batch.elapsed_seconds, .07)

    def test_failure_propagates_and_context_releases_all_workers(self):
        teachers = TeacherLeague(self.root, [str(self.path)], workers=2)
        with self.assertRaisesRegex(RuntimeError, "illegal"):
            with teachers:
                processes = None
                batch = teachers.submit_actions([SnapshotState([0, 1, 0], [7])], [0])
                processes = list(teachers.executor._processes.values())
                batch.result()
        self.assertTrue(all(not process.is_alive() for process in processes))

    def test_length_mismatch_and_empty_batches(self):
        with TeacherLeague(self.root, [str(self.path)], workers=1) as teachers:
            with self.assertRaisesRegex(ValueError, "equal lengths"):
                teachers.submit_actions([SnapshotState([0, 0, 0], [7])], [])
            batch = teachers.submit_actions([], [])
            self.assertEqual(batch.result(), [])
            self.assertEqual(batch.elapsed_seconds, 0.)

    def test_workers_do_not_import_torch(self):
        with TeacherLeague(self.root, [str(self.path)], workers=2) as teachers:
            teachers.actions([SnapshotState([0, 0, 0], [7])], [0])
            rows = [future.result() for future in [teachers.executor.submit(worker_runtime) for _ in range(4)]]
            self.assertTrue(rows)
            self.assertTrue(all(not loaded for pid, loaded in rows))


if __name__ == "__main__":
    unittest.main()
