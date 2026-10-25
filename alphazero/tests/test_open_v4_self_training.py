import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from junqi_open_v4.games import training_mode, uses_search_target, collect_training_games
from junqi_open_v4.training_replay import replay_pools, make_self_training_learner
from junqi_open_v4.learner import atomic_checkpoint, model_hash
from junqi_search_v3.loop import SearchSample
from test_open_v4 import small_config


def rows_and_metadata(*, contract="search", mode="teacher_vs_actor", trace=(0,)):
    rows = []
    for ply in range(len(trace)):
        rows.append(SearchSample(1, 100 + ply, np.zeros((107, 12, 5), np.float16),
            np.array([0, 1]), np.array([0, 0]), np.array([2, 5]),
            1 if ply % 2 == 0 else -1, ply, .5, wdl=np.array([0., 1., 0.])))
    game = dict(family=1, mode=mode, actor_side=1, trace=list(trace), result_red=0,
                reason="REPETITION", provenance={"initial_board": [0] * 60})
    return rows, dict(layout_role="train", policy_target_contract=contract, games=[game])


class SelfTrainingTests(unittest.TestCase):
    def test_legacy_modes_and_targets_unchanged_new_actor_moves_use_search(self):
        cfg = small_config()
        self.assertEqual([training_mode(cfg, x) for x in (.1, .3, .7)],
                         ["selfplay", "teacher_selfplay", "teacher_vs_actor"])
        self.assertFalse(uses_search_target(cfg, "teacher_vs_actor", 1, 1))
        cfg.selfplay_fraction = .4; cfg.teacher_selfplay_fraction = .1
        cfg.actor_policy_target = "search"; cfg.validate()
        self.assertEqual([training_mode(cfg, x) for x in (.3, .45, .7)],
                         ["selfplay", "teacher_selfplay", "teacher_vs_actor"])
        self.assertTrue(uses_search_target(cfg, "teacher_vs_actor", 1, 1))
        self.assertFalse(uses_search_target(cfg, "teacher_vs_actor", -1, 1))
        cfg.teacher_selfplay_fraction = .7
        with self.assertRaises(ValueError): cfg.validate()

    def test_exact_duplicate_removed_but_draws_histories_and_targets_preserved(self):
        rows, meta = rows_and_metadata()
        values, policy, audit = replay_pools([(rows, meta), copy.deepcopy((rows, meta))],
            deduplicate=True, neural_policy_only=True, holdout=lambda _: False)
        self.assertEqual(len(values), 1); self.assertEqual(len(policy), 1)
        self.assertEqual(audit["duplicate_games_removed"], 1)
        self.assertEqual(audit["retained_terminal_reasons"], {"REPETITION": 1})
        changed_rows, changed_meta = copy.deepcopy((rows, meta))
        changed_meta["games"][0]["trace"] = [1]
        changed_rows[0].visits = np.array([3, 4])
        values, _, audit = replay_pools([(rows, meta), (changed_rows, changed_meta)],
            deduplicate=True, neural_policy_only=True, holdout=lambda _: False)
        self.assertEqual(len(values), 2)
        self.assertEqual(audit.get("duplicate_games_removed", 0), 0)
        for change in ("history", "target"):
            changed_rows, changed_meta = copy.deepcopy((rows, meta))
            if change == "history": changed_meta["games"][0]["trace"] = [1]
            else: changed_rows[0].visits = np.array([3, 4])
            values, _, _ = replay_pools([(rows, meta), (changed_rows, changed_meta)],
                deduplicate=True, neural_policy_only=True, holdout=lambda _: False)
            self.assertEqual(len(values), 2)
        with self.assertRaisesRegex(ValueError, "only training"):
            replay_pools([(rows, meta | {"layout_role": "dev"})],
                deduplicate=True, neural_policy_only=True, holdout=lambda _: False)

    def test_teacher_turns_and_legacy_imitation_excluded_only_from_policy(self):
        modern = rows_and_metadata(trace=(0, 1))
        legacy = rows_and_metadata(contract="teacher", trace=(0, 1))
        autonomous = rows_and_metadata(contract="teacher", mode="selfplay", trace=(0, 1))
        values, policy, audit = replay_pools([modern, legacy, autonomous],
            deduplicate=False, neural_policy_only=True, holdout=lambda _: False)
        self.assertEqual(len(values), 6)
        self.assertEqual(len(policy), 3)
        self.assertEqual(audit["policy_sources"], {"search": 3})
        bad = copy.deepcopy(modern); bad[0][0].wdl = np.array([1., 0., 0.])
        with self.assertRaisesRegex(ValueError, "terminal WDL"):
            replay_pools([bad], deduplicate=True, neural_policy_only=True, holdout=lambda _: False)

    def test_full_checkpoint_continuation_reproduces_split_pool_training(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary); cfg = small_config()
            rows, meta = rows_and_metadata(mode="selfplay", trace=(0, 1))
            values, policy, _ = replay_pools([(rows, meta)], deduplicate=True,
                neural_policy_only=True, holdout=lambda _: False)
            first = make_self_training_learner(cfg, root)
            first.train_pools(values, policy, updates=2)
            path = root / "learner.pt"
            atomic_checkpoint(path, first.payload(role="learner", lineage={}))
            first.train_pools(values, policy, updates=3)
            second = make_self_training_learner(cfg, root); second.resume(path)
            second.train_pools(values, policy, updates=3)
            self.assertEqual(model_hash(first.model), model_hash(second.model))
            self.assertEqual(first.counters, second.counters)
            self.assertEqual(first.scheduler, second.scheduler)
            self.assertEqual(second.policy_batch_sources, {"search": 8})
            self.assertTrue(torch.equal(first.payload(role="learner", lineage={})["rng"]["torch"],
                                        second.payload(role="learner", lineage={})["rng"]["torch"]))

    def test_collection_actor_target_is_search_and_skips_unused_teacher_job(self):
        from types import SimpleNamespace
        class State:
            native_state = property(lambda self: self)
            def key(self): return 1
            def position_key(self): return 1
            def repetition_count(self): return 0
            def current_player(self): return 1
            def legal_actions(self): return np.array([0, 1])
            def repetition_contexts(self): return np.array([0, 0])
            def feature_planes(self): return np.zeros((107, 12, 5))
            def provenance(self): return {"initial_board": [0] * 60}
            def apply(self, action): self.action = action
            def is_terminal(self): return hasattr(self, "action")
            def terminal_reason(self): return "FLAG_CAPTURE"
            def terminal_value(self, player): return player
        class Teachers:
            workers = 1; provenance = []
            modules = [None]
            calls = []
            def submit_actions(self, states, ids):
                self.calls.append(len(states))
                return SimpleNamespace(result=lambda: [0] * len(states), elapsed_seconds=0.)
        class Search:
            def __init__(self, *args, **kwargs): pass
            def search(self, states, *args, **kwargs):
                return [SimpleNamespace(action=1, visits=np.array([1, 9]), root_entropy=.3,
                    policy=np.array([.1, .9]), actions=np.array([0, 1])) for _ in states], dict(
                    elapsed_seconds=.01, inference_rows=len(states), inference_calls=1,
                    tactical_roots=0,tactical_native_transitions=0,tactical_immediate_wins=0,tactical_disabled_actions=0)
        cfg = small_config(); cfg.selfplay_fraction=0.; cfg.teacher_selfplay_fraction=0.
        cfg.actor_policy_target="search"
        teachers = Teachers()
        with patch("junqi_open_v4.games.BatchedNeuralPUCT", Search):
            samples, meta = collect_training_games(cfg, None, "cpu",
                SimpleNamespace(new_state=lambda seed: State()), teachers, 1, count=2)
        # Red actor's move is trained on visits; black actor's teacher move is a
        # teacher label retained only as Value data in neural-only replay.
        np.testing.assert_array_equal(samples[0].visits, [1, 9])
        np.testing.assert_array_equal(samples[1].visits, [1, 0])
        self.assertEqual(teachers.calls, [1])
        self.assertEqual(meta["policy_target_contract"], "search")

    def test_screening_is_only_a_futility_filter(self):
        from run_open_v4_self_training import screening_signal
        gains = [[{"mean": .01}, {"mean": .01}], [{"mean": .01}, {"mean": .01}]]
        self.assertTrue(screening_signal(gains, [{"mean": .51}, {"mean": .51}]))
        self.assertFalse(screening_signal(gains, [{"mean": .49}, {"mean": .51}]))
        self.assertFalse(screening_signal([[], []], [{"mean": .51}, {"mean": .51}]))

    def test_gate_receives_baseline_to_candidate_pairs_regardless_of_input_order(self):
        from run_open_v4_self_training import ordered_participants
        from junqi_open_v4.pipeline import promotion_decision
        names = list(ordered_participants({"candidate": 1, "parent": 2, "repeat": 3, "control": 4}))
        self.assertEqual(names, ["parent", "control", "candidate", "repeat"])
        summaries = [{"candidate": "candidate", "opponent": "parent",
                      "family_score": {"mean": .6, "ci95": [.55, .65], "families": 64}}]
        summaries.extend({"candidate": name, "opponent": "teacher0"} for name in names)
        arena = {"role": "dev", "summaries": summaries,
                 "paired_deltas": [{"from_matchup": 1, "to_matchup": 3,
                                   "delta": {"mean": .05, "ci95": [.01, .09], "families": 64}}]}
        self.assertTrue(promotion_decision(arena, small_config())["champion_accepted"])
        with self.assertRaises(ValueError): ordered_participants({"candidate": 1})


if __name__ == "__main__": unittest.main()
