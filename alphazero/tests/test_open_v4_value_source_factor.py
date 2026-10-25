import copy
import tempfile
import unittest
from pathlib import Path

import torch
import test_open_v4 as fixtures
from junqi_open_v4.learner import Learner, atomic_checkpoint
from run_open_v4_search_training import verify_learning_state
from open_v4_value_source_factor import make_learner, audit_shard, ALLOWED


class ValueSourceFactorTests(unittest.TestCase):
    def pool(self, root):
        sample = fixtures.OpenV4Tests()._sample(root)
        rows = []
        for family in (1, 2, 3):
            row = copy.deepcopy(sample); row.game_id = family; rows.append(row)
        return rows, {1: "selfplay", 2: "teacher_selfplay", 3: "teacher_vs_actor"}

    def test_all_source_control_reproduces_full_learning_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = fixtures.small_config(); pool, modes = self.pool(root)
            original = Learner(cfg, root); original.train(pool, updates=6)
            control = make_learner(cfg, root, modes, "all"); control.train(pool, updates=6)
            self.assertTrue(all(verify_learning_state(original.payload(role="learner", lineage={}),
                                                    control.payload(role="learner", lineage={})).values()))

    def test_value_eligibility_keeps_policy_batch_and_rng_keys_paired(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = fixtures.small_config(); pool, modes = self.pool(root)
            control = make_learner(cfg, root, modes, "all")
            candidate = make_learner(cfg, root, modes, list(ALLOWED))
            for attempt in range(6):
                control.counters["attempted_updates"] = candidate.counters["attempted_updates"] = attempt
                a, b = control._batch(pool), candidate._batch(pool)
                if attempt % 3 == 2:
                    self.assertTrue(all(modes[s.game_id] in ALLOWED for s in b))
                else:
                    self.assertEqual([s.game_id for s in a], [s.game_id for s in b])
            self.assertEqual(control.sampling_key_digest.hexdigest(), candidate.sampling_key_digest.hexdigest())
            self.assertEqual(control.phase_batch_digests["policy"].hexdigest(), candidate.phase_batch_digests["policy"].hexdigest())
            self.assertEqual(candidate.value_pool_size, 2)
            unknown = copy.deepcopy(pool); unknown[0].game_id = 17
            with self.assertRaisesRegex(ValueError, "unregistered"):
                candidate._batch(unknown)

    def test_candidate_full_resume_and_label_audit_reject_wrong_role_or_sign(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = fixtures.small_config(); pool, modes = self.pool(root)
            learner = make_learner(cfg, root, modes, list(ALLOWED)); learner.train(pool, updates=2)
            path = root / "learner.pt"; atomic_checkpoint(path, learner.payload(role="learner", lineage={}))
            learner.train(pool, updates=4)
            resumed = make_learner(cfg, root, modes, list(ALLOWED)); resumed.resume(path); resumed.train(pool, updates=4)
            self.assertTrue(all(verify_learning_state(learner.payload(role="learner", lineage={}),
                                                    resumed.payload(role="learner", lineage={})).values()))
            with self.assertRaisesRegex(ValueError, "factor mismatch"):
                make_learner(cfg, root, modes, "all").resume(path)
            metadata = dict(layout_role="train", games=[dict(family=i, mode=modes[i], trace=[0],
                result_red=-1, actor_side=1, reason="FLAG_CAPTURE") for i in modes])
            actual, audit = audit_shard(pool, metadata); self.assertEqual(actual, modes)
            self.assertEqual(audit["teacher_vs_actor"]["actor_turn_wdl"][0], 1)
            with self.assertRaisesRegex(ValueError, "training trajectories"):
                audit_shard(pool, metadata | {"layout_role": "dev"})
            damaged = copy.deepcopy(metadata); damaged["games"][0]["result_red"] = 1
            with self.assertRaisesRegex(ValueError, "terminal WDL"):
                audit_shard(pool, damaged)


if __name__ == "__main__": unittest.main()
