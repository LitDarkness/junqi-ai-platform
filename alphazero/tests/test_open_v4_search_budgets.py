import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_open_v4_search_budgets import StopAwareSearch, primary_signal, validated_budgets
from open_v4_experiment_evidence import teacher_deltas


class BudgetDiagnosticsTests(unittest.TestCase):
    def rows(self, lower=.01, mean=.03, families=32):
        return [dict(mean=mean, ci95=[lower, .1], families=families) for _ in range(2)]

    def test_primary_needs_two_positive_teacher_intervals_and_direct_win(self):
        duplicate = self.rows(mean=0)
        duel = dict(families=32, ci95=[.55, .65])
        self.assertEqual(primary_signal(self.rows(), duplicate, duel, 32, .02), (True, True))
        self.assertEqual(primary_signal(self.rows(lower=-.01), duplicate, duel, 32, .02), (True, False))
        self.assertEqual(primary_signal(self.rows()[:1], duplicate, duel, 32, .02), (True, False))
        self.assertEqual(primary_signal(self.rows(), duplicate, dict(families=32, ci95=[.49, .65]), 32, .02), (True, False))

    def test_unreliable_duplicate_or_missing_families_blocks_signal(self):
        duel = dict(families=32, ci95=[.55, .65])
        self.assertEqual(primary_signal(self.rows(), self.rows(mean=.03), duel, 32, .02), (False, False))
        self.assertEqual(primary_signal(self.rows(), self.rows(mean=0, families=16), duel, 32, .02), (False, False))

    def test_configuration_cannot_change_value_or_duplicate_budget(self):
        rows = [dict(name=name, simulations=budget) for name, budget in
                (("baseline", 64), ("repeat", 64), ("sim32", 32), ("sim128", 128), ("sim256", 256))]
        self.assertEqual(validated_budgets(rows, 64)["sim256"], 256)
        changed = [dict(row) for row in rows]
        changed[1]["simulations"] = 32
        with self.assertRaises(ValueError):
            validated_budgets(changed, 64)
        changed[1] = dict(rows[1], value_scale=0)
        with self.assertRaises(ValueError):
            validated_budgets(changed, 64)

    def test_duel_deltas_are_not_used_as_teacher_evidence(self):
        summaries = [dict(candidate="baseline", opponent="teacher0"),
                     dict(candidate="sim128", opponent="teacher0"),
                     dict(candidate="sim32", opponent="baseline"),
                     dict(candidate="sim128", opponent="baseline")]
        arena = dict(role="dev", summaries=summaries, paired_deltas=[
            dict(from_matchup=0, to_matchup=1, delta=dict(mean=.01)),
            dict(from_matchup=2, to_matchup=3, delta=dict(mean=.4))])
        self.assertEqual(teacher_deltas(arena, "baseline", "sim128"), [dict(mean=.01)])

    def test_stop_checked_before_any_neural_work(self):
        class Search:
            def search(self, *args, **kwargs):
                self.fail = True
                raise AssertionError("should not search after STOP")
        def stopped():
            raise RuntimeError("STOP")
        with self.assertRaisesRegex(RuntimeError, "STOP"):
            StopAwareSearch(Search(), stopped).search([], 256)


if __name__ == "__main__":
    unittest.main()
