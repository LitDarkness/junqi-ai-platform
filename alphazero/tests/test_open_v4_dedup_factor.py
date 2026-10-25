import copy
import unittest

from run_open_v4_dedup_factor import validate_factor_plan


def registered_plan():
    control = dict(selfplay_fraction=.25, teacher_selfplay_fraction=None,
                   actor_policy_target="teacher", deduplicate=False, neural_policy_only=False)
    return dict(experiment_type="exact_trajectory_dedup_factor", control=control,
                candidate=dict(control, deduplicate=True), maximum_stages=6,
                stages_per_block=6, games_per_stage=256, updates_per_stage=64,
                replay_stages=3, screen_families=16, full_families=64, evaluation_budget=256,
                frozen_collections=[dict(stage=i, path=f"raw_{i}.npz") for i in range(1, 7)],
                reference_control=[dict(stage=i) for i in range(1, 7)])


class DedupFactorContractTests(unittest.TestCase):
    def test_rejects_mixed_factors_and_missing_matched_control(self):
        plan = registered_plan()
        self.assertTrue(validate_factor_plan(plan))
        for key, value in (("selfplay_fraction", .4), ("actor_policy_target", "search"),
                           ("neural_policy_only", True)):
            changed = copy.deepcopy(plan)
            changed["candidate"][key] = value
            with self.assertRaisesRegex(ValueError, "only exact trajectory dedup"):
                validate_factor_plan(changed)
        changed = copy.deepcopy(plan)
        changed["reference_control"].pop()
        with self.assertRaisesRegex(ValueError, "six complete matched"):
            validate_factor_plan(changed)

    def test_rejects_duplicate_raw_sources_and_changed_gate(self):
        plan = registered_plan()
        changed = copy.deepcopy(plan)
        changed["frozen_collections"][1]["path"] = changed["frozen_collections"][0]["path"]
        with self.assertRaisesRegex(ValueError, "distinct raw"):
            validate_factor_plan(changed)
        for key, value in (("screen_families", 8), ("full_families", 32),
                           ("evaluation_budget", 64), ("maximum_stages", 12)):
            changed = copy.deepcopy(plan)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_factor_plan(changed)


if __name__ == "__main__":
    unittest.main()
