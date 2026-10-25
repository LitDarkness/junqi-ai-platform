import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from junqi_open_v4.history_policy_replay import history_policy_pools
from junqi_open_v4.training_replay import replay_pools, make_self_training_learner
from junqi_open_v4.learner import atomic_checkpoint
from run_open_v4_search_training import verify_learning_state, route_signal
from run_open_v4_teacher_history_factor import validate_factor_plan, screening_signal
from test_open_v4 import small_config
from test_open_v4_self_training import rows_and_metadata


class TeacherHistoryFactorTests(unittest.TestCase):
    def test_history_eligibility_preserves_value_search_draws_and_original_targets(self):
        teacher, teacher_meta = rows_and_metadata(mode='teacher_selfplay', trace=(0, 1, 0, 1))
        teacher[0].repetition_count = 2
        teacher[1].repetition_contexts = np.array([0, 2], np.uint8)
        teacher[2].cycle_entry = True
        search, search_meta = rows_and_metadata(mode='selfplay', trace=(0, 1))
        search[0].repetition_count = 2
        search[1].repetition_contexts = np.array([1, 2], np.uint8)
        shards = [(teacher, teacher_meta), (search, search_meta)]
        saved_targets = [r.visits.copy() for rows, _ in shards for r in rows]
        full_values, full_policy, _ = history_policy_pools(shards,
            exclude_history_sensitive_teacher=False, holdout=lambda _: False)
        values, policy, audit = history_policy_pools(shards,
            exclude_history_sensitive_teacher=True, holdout=lambda _: False)
        self.assertEqual([id(r) for r in values], [id(r) for r in full_values])
        self.assertEqual(len(full_policy), 6)
        self.assertEqual([id(r) for r in policy], [id(teacher[3]), id(search[0]), id(search[1])])
        self.assertEqual(audit['excluded_policy_rows'], 3)
        self.assertEqual(audit['history_sensitive_search_rows_retained'], 2)
        self.assertEqual(audit['retained_terminal_reasons'], {'REPETITION': 2})
        for original, row in zip(saved_targets, [r for rows, _ in shards for r in rows]):
            np.testing.assert_array_equal(original, row.visits)
            np.testing.assert_array_equal(row.wdl, [0, 1, 0])
        self.assertEqual(audit.get('duplicate_games_removed', 0), 0)
        with self.assertRaisesRegex(ValueError, 'only training'):
            history_policy_pools([(teacher, teacher_meta | {'layout_role': 'dev'})],
                exclude_history_sensitive_teacher=True, holdout=lambda _: False)

    def test_legacy_control_full_state_and_filtered_continuation_reproduce(self):
        shards = [rows_and_metadata(mode='teacher_vs_actor', contract='teacher', trace=(0, 1)),
                  rows_and_metadata(mode='selfplay', trace=(0, 1))]
        shards[0][0][0].repetition_count = 2
        old_values, old_policy, _ = replay_pools(shards, deduplicate=False,
            neural_policy_only=False, holdout=lambda _: False)
        values, policy, _ = history_policy_pools(copy.deepcopy(shards),
            exclude_history_sensitive_teacher=False, holdout=lambda _: False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg = small_config()
            first = make_self_training_learner(cfg, root)
            seed = root / 'seed.pt'; atomic_checkpoint(seed, first.payload(role='learner', lineage={}))
            first.train_pools(old_values, old_policy, updates=5)
            expected = first.payload(role='learner', lineage={})
            second = make_self_training_learner(cfg, root); second.resume(seed)
            second.train_pools(values, policy, updates=5)
            self.assertTrue(all(verify_learning_state(expected, second.payload(role='learner', lineage={})).values()))
            v, p, _ = history_policy_pools(shards,
                exclude_history_sensitive_teacher=True, holdout=lambda _: False)
            second.train_pools(v, p, updates=2)
            middle = root / 'middle.pt'; atomic_checkpoint(middle, second.payload(role='learner', lineage={}))
            second.train_pools(v, p, updates=4)
            third = make_self_training_learner(cfg, root); third.resume(middle)
            third.train_pools(v, p, updates=4)
            self.assertTrue(all(verify_learning_state(second.payload(role='learner', lineage={}),
                third.payload(role='learner', lineage={})).values()))

    def test_rejects_other_factor_and_screen_never_qualifies(self):
        control = dict(selfplay_fraction=.25, teacher_selfplay_fraction=None,
            actor_policy_target='teacher', deduplicate=False, neural_policy_only=False,
            exclude_history_sensitive_teacher=False)
        plan = dict(experiment_type='teacher_history_policy_factor', control=control,
            candidate=dict(control, exclude_history_sensitive_teacher=True), maximum_stages=6,
            stages_per_block=6, games_per_stage=256, updates_per_stage=64, replay_stages=3,
            screen_families=16, full_families=64, evaluation_budget=256,
            frozen_collections=[dict(stage=i, path=f'raw{i}') for i in range(1, 7)],
            reference_control=[dict(stage=i) for i in range(1, 7)])
        self.assertTrue(validate_factor_plan(plan))
        for key, value in [('deduplicate', True), ('actor_policy_target', 'search'), ('selfplay_fraction', .4)]:
            changed = copy.deepcopy(plan); changed['candidate'][key] = value
            with self.assertRaises(ValueError): validate_factor_plan(changed)
        gain = dict(mean=.1, ci95=[.01, .2], families=16)
        duel = dict(mean=.6, ci95=[.51, .7], families=16)
        duplicate = dict(mean=0, ci95=[-.01, .01], families=16)
        self.assertTrue(screening_signal([[gain, gain], [gain, gain]], [duel, duel]))
        self.assertFalse(route_signal([[gain, gain], [gain, gain]], [duplicate, duplicate],
            [duel, duel], {'champion_accepted': True}, 16, .02))


if __name__ == '__main__': unittest.main()
