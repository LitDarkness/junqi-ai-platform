import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from junqi_open_v4.policy_anchor_replay import policy_anchor_pools
from junqi_open_v4.training_replay import replay_pools, make_self_training_learner
from junqi_open_v4.learner import atomic_checkpoint
from run_open_v4_search_training import verify_learning_state
from run_open_v4_policy_anchor_factor import validate_factor_plan, legacy_guard, ordered_participants
from test_open_v4 import small_config
from test_open_v4_self_training import rows_and_metadata


class PolicyAnchorFactorTests(unittest.TestCase):
    def test_only_actual_teacher_turns_added_without_relabelling_actor_turns(self):
        shards = [rows_and_metadata(mode="teacher_vs_actor", contract="search", trace=(0, 1)),
                  rows_and_metadata(mode="teacher_vs_actor", contract="teacher", trace=(0, 1)),
                  rows_and_metadata(mode="selfplay", contract="teacher", trace=(0, 1)),
                  rows_and_metadata(mode="teacher_selfplay", contract="teacher", trace=(0, 1))]
        original_targets = [row.visits.copy() for samples, _ in shards for row in samples]
        values, pure, _ = policy_anchor_pools(shards, deduplicate=True,
            teacher_policy_anchor=False, holdout=lambda _: False)
        _, anchored, audit = policy_anchor_pools(shards, deduplicate=True,
            teacher_policy_anchor=True, holdout=lambda _: False)
        self.assertEqual(len(values), 8)
        self.assertEqual(len(pure), 3)
        self.assertEqual(len(anchored), 7)
        self.assertEqual(audit["policy_sources"], {"search": 3, "teacher": 4})
        self.assertEqual(audit["legacy_actor_imitation_rows_excluded"], 1)
        self.assertNotIn(id(shards[1][0][0]), {id(row) for row in anchored})
        self.assertTrue(all(row.training_teacher_owned_turn for row in anchored
                            if row.training_policy_source == "teacher"))
        for target, row in zip(original_targets, [row for samples, _ in shards for row in samples]):
            np.testing.assert_array_equal(target, row.visits)

    def test_neural_only_baseline_reproduces_complete_learning_state(self):
        shards = [rows_and_metadata(mode="teacher_vs_actor", contract="search", trace=(0, 1))]
        values, policy, _ = replay_pools(shards, deduplicate=True,
                                         neural_policy_only=True, holdout=lambda _: False)
        new_values, new_policy, _ = policy_anchor_pools(copy.deepcopy(shards), deduplicate=True,
            teacher_policy_anchor=False, holdout=lambda _: False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg = small_config()
            first = make_self_training_learner(cfg, root)
            path = root / "seed.pt"
            atomic_checkpoint(path, first.payload(role="learner", lineage={}))
            first.train_pools(values, policy, updates=5)
            expected = first.payload(role="learner", lineage={})
            second = make_self_training_learner(cfg, root); second.resume(path)
            second.train_pools(new_values, new_policy, updates=5)
            self.assertTrue(all(verify_learning_state(expected, second.payload(
                role="learner", lineage={})).values()))

    def test_legacy_guard_keeps_screen_separate_from_full_qualification(self):
        positive = dict(mean=.1, ci95=[-.01, .2], families=64)
        duel = dict(mean=.6, ci95=[.51, .7], families=64)
        self.assertTrue(legacy_guard([positive, positive], duel, full=False))
        self.assertFalse(legacy_guard([positive, positive], duel, full=True))
        positive["ci95"] = [.01, .2]
        self.assertTrue(legacy_guard([positive, positive], duel, full=True))
        for key, value in (("families", 16), ("ci95", [.5, .7])):
            changed = dict(duel, **{key: value})
            self.assertFalse(legacy_guard([positive, positive], changed, full=True))
        names = ["candidate", "repeat", "legacy", "control", "parent"]
        ordered = list(ordered_participants({name: {} for name in names}))
        self.assertLess(ordered.index("parent"), ordered.index("candidate"))
        self.assertLess(ordered.index("legacy"), ordered.index("candidate"))

    def test_rejects_mixed_factor_or_removed_legacy_guard(self):
        control = dict(selfplay_fraction=.4, teacher_selfplay_fraction=.1,
                       actor_policy_target="search", deduplicate=True, teacher_policy_anchor=False)
        plan = dict(experiment_type="teacher_owned_policy_anchor_factor", control=control,
                    candidate=dict(control, teacher_policy_anchor=True), maximum_stages=6,
                    stages_per_block=6, games_per_stage=256, updates_per_stage=64, replay_stages=3,
                    screen_families=16, full_families=64, evaluation_budget=256,
                    frozen_collections=[dict(stage=i, path=f"raw{i}.npz") for i in range(1, 7)],
                    reference_control=[dict(stage=i) for i in range(1, 7)],
                    legacy_reference={"path": "legacy.pt"}, legacy_guard_required=True)
        self.assertTrue(validate_factor_plan(plan))
        changed = copy.deepcopy(plan); changed["candidate"]["actor_policy_target"] = "teacher"
        with self.assertRaises(ValueError): validate_factor_plan(changed)
        changed = copy.deepcopy(plan); changed["legacy_guard_required"] = False
        with self.assertRaises(ValueError): validate_factor_plan(changed)


if __name__ == "__main__":
    unittest.main()
