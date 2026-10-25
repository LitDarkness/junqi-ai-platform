import copy
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import test_open_v4 as fixtures
from fork_open_v4_qualified import fork_payload
from run_open_v4_search_training import (variant_config, verify_learning_state, replay_window,
                                        paired_schedule, route_signal)
from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash


class SearchTrainingTests(unittest.TestCase):
    def test_search_migration_preserves_exact_next_ppv_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = fixtures.small_config()
            base.simulations, base.repeated_simulations = 64, 128
            base.games_per_iteration = 256
            learner = Learner(base, root)
            sample = fixtures.OpenV4Tests()._sample(root)
            learner.train([sample], updates=4)
            payload = learner.payload(role="learner", lineage={"fixture": True})
            cfg = variant_config(base, "runs/new", None, "candidate")
            migrated = fork_payload(payload, cfg, {"fixture": True})
            self.assertTrue(all(verify_learning_state(payload, migrated).values()))
            path = root / "new.pt"
            atomic_checkpoint(path, migrated)
            learner.train([sample], updates=4)
            restored = Learner(cfg, root)
            restored.resume(path)
            restored.train([sample], updates=4)
            self.assertEqual(model_hash(learner.model), model_hash(restored.model))
            self.assertEqual(learner.counters, restored.counters)
            damaged = copy.deepcopy(migrated)
            damaged["counters"]["effective_updates"] += 1
            with self.assertRaisesRegex(ValueError, "full learning state"):
                verify_learning_state(payload, damaged)

    def test_replay_rolls_away_initial_data_with_fresh_namespace(self):
        self.assertEqual(replay_window(1, 3, [24, 25, 26], 45), [("initial", 25), ("initial", 26), ("new", 45)])
        self.assertEqual(replay_window(3, 3, [24, 25, 26], 45), [("new", 45), ("new", 46), ("new", 47)])
        self.assertEqual(replay_window(6, 3, [24, 25, 26], 45), [("new", 48), ("new", 49), ("new", 50)])

    def test_paired_schedules_allow_trajectory_but_reject_layout_changes(self):
        row = dict(index=0, family=45_000_000, mode="selfplay", actor_side=1,
                   teacher_id=0, provenance={"role": "train", "layout": 12}, trace=[1, 2])
        a, b = {"games": [row]}, {"games": [copy.deepcopy(row)]}
        b["games"][0]["trace"] = [3, 4]
        self.assertEqual(paired_schedule(a), paired_schedule(b))
        b["games"][0]["provenance"]["layout"] = 13
        self.assertNotEqual(paired_schedule(a), paired_schedule(b))

    def test_failed_primary_cannot_be_rescued_by_empty_or_partial_evidence(self):
        gain = [{"families": 64, "mean": .1, "ci95": [.01, .2]} for _ in range(2)]
        duplicate = [{"families": 64, "mean": 0., "ci95": [-.01, .01]} for _ in range(2)]
        duel = {"families": 64, "ci95": [.55, .7]}
        args = [[gain, gain], duplicate, [duel, duel], {"champion_accepted": True}, 64, .02]
        self.assertTrue(route_signal(*args))
        damaged = copy.deepcopy(args)
        damaged[0][1][0]["ci95"][0] = -.01
        self.assertFalse(route_signal(*damaged))
        damaged = copy.deepcopy(args)
        damaged[4] = 32
        self.assertFalse(route_signal(*damaged))
        damaged = copy.deepcopy(args)
        damaged[0] = []
        self.assertFalse(route_signal(*damaged))
        damaged = copy.deepcopy(args)
        damaged[3]["champion_accepted"] = False
        self.assertFalse(route_signal(*damaged))


if __name__ == "__main__":
    unittest.main()
