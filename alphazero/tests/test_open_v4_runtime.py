from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
from open_v4_runtime import ProcessLock, atomic_json, config_run, lock_held, read_json
from supervise_open_v4 import supervise


FAKE_RUNNER = '''import argparse, json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--config');p.add_argument('--through-iteration',type=int);a=p.parse_args()
root=Path(a.config).parent; run=root/'run'; i=a.through_iteration
report=dict(counters=dict(effective_updates=i*4),gate=dict(actor_accepted=False,champion_accepted=False),internal_after={},elapsed_seconds=.01,collection=dict(moves_per_second=1.),peak_gpu_allocated_mib=0.,roles={},training=[dict(accepted=True)]*4)
(run/'reports').mkdir(exist_ok=True)
(run/f'reports/iteration_{i:06d}.json').write_text(json.dumps(report))
(run/'CURRENT.json').write_text(json.dumps(dict(iteration=i)))
'''


class OpenV4RuntimeTests(unittest.TestCase):
    def setup_run(self, root):
        config = root / "config.json"
        atomic_json(config, dict(run_directory="run", updates_per_iteration=4))
        atomic_json(root / "run/CURRENT.json", dict(iteration=1))
        runner = root / "fake.py"
        runner.write_text(FAKE_RUNNER, encoding="utf-8")
        return config, runner

    def test_lock_excludes_another_process_and_releases(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "writer.lock"
            command = [sys.executable, "-c", "import sys;sys.path.insert(0,sys.argv[1]);from open_v4_runtime import lock_held;print(int(lock_held(sys.argv[2])))", str(TOOLS), str(path)]
            with ProcessLock(path):
                self.assertTrue(lock_held(path))
                self.assertEqual(subprocess.check_output(command, text=True).strip(), "1")
            self.assertEqual(subprocess.check_output(command, text=True).strip(), "0")

    def test_segment_resumes_existing_iteration_and_requires_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, runner = self.setup_run(root)
            result = supervise(root, config, stage_iterations=2, minimum_free_gib=0, poll_seconds=.01, runner=runner)
            self.assertEqual(result["completed_iteration"], 3)
            self.assertEqual(result["state"], "review_due")
            self.assertTrue(result["review_required"])
            self.assertFalse(lock_held(root / "run/supervision/SUPERVISOR.lock"))

    def test_user_stop_is_respected_without_starting_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config, runner = self.setup_run(root)
            stop=root / "run/supervision/STOP";stop.parent.mkdir();stop.touch()
            with patch("supervise_open_v4.subprocess.Popen") as process:
                result=supervise(root,config,minimum_free_gib=0,runner=runner)
                process.assert_not_called()
            self.assertEqual(result["state"], "stopped_by_user")
            self.assertEqual(result["completed_iteration"],1)

    def test_original_stop_blocks_a_continuation_branch(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,runner=self.setup_run(root)
            original_stop=root/'original/STOP';original_stop.parent.mkdir();original_stop.touch()
            with patch('supervise_open_v4.subprocess.Popen') as process:
                result=supervise(root,config,minimum_free_gib=0,runner=runner,extra_stop_paths=(original_stop,))
                process.assert_not_called()
            self.assertEqual(result['state'],'stopped_by_user')
            self.assertTrue(original_stop.exists())

    def test_original_stop_during_iteration_prevents_next_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,runner=self.setup_run(root)
            original_stop=root/'original/STOP';original_stop.parent.mkdir()
            runner.write_text(FAKE_RUNNER+"\n(root/'original/STOP').touch()\n",encoding='utf-8')
            result=supervise(root,config,stage_iterations=3,minimum_free_gib=0,poll_seconds=.01,runner=runner,extra_stop_paths=(original_stop,))
            self.assertEqual(result['state'],'stopped_by_user')
            self.assertEqual(result['completed_iteration'],2)
            self.assertEqual(len(list((root/'run/supervision').glob('iteration_*.log'))),1)

    def test_failed_child_does_not_advance_checkpoint_and_retries_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,runner=self.setup_run(root)
            runner.write_text("raise SystemExit(7)\n",encoding="utf-8")
            result=supervise(root,config,minimum_free_gib=0,poll_seconds=.01,runner=runner)
            self.assertEqual(result["state"],"needs_attention")
            self.assertEqual(read_json(root/'run/CURRENT.json')["iteration"],1)
            self.assertEqual(len(list((root/'run/supervision').glob('iteration_*.log'))),2)

    def test_low_disk_space_prevents_next_iteration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);config,runner=self.setup_run(root)
            with patch("supervise_open_v4.shutil.disk_usage") as disk, patch("supervise_open_v4.subprocess.Popen") as process:
                disk.return_value.free=0
                result=supervise(root,config,runner=runner)
                process.assert_not_called()
            self.assertEqual(result["reason"],"disk_space")

    def test_run_path_cannot_escape_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            atomic_json(root/'config.json',dict(run_directory='../elsewhere'))
            with self.assertRaises(ValueError):config_run(root,root/'config.json')


if __name__ == "__main__":
    unittest.main()
