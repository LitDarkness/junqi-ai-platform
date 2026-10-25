import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_open_v4_confirmation import route_confirmed


class ConfirmationTests(unittest.TestCase):
    def rows(self, families=64, lower=.01, mean=.05):
        return [{"families": families, "ci95": [lower, .1], "mean": mean} for _ in range(2)]

    def check(self, parent=None, control=None, duplicate=None, champion=True, duel_lower=.55):
        return route_confirmed(parent or self.rows(), control or self.rows(),
            duplicate or self.rows(mean=0), {"champion_accepted": champion}, 64, .02,
            self.rows(lower=duel_lower, mean=.6))

    def test_partial_suite_cannot_confirm_route(self):
        self.assertFalse(self.check(parent=self.rows(families=32)))
        self.assertTrue(self.check())

    def test_unresolved_parent_gain_remains_unconfirmed(self):
        self.assertFalse(self.check(parent=self.rows(lower=-.01)))

    def test_missing_teacher_unreliable_duplicate_or_failed_gate_blocks(self):
        self.assertFalse(self.check(control=self.rows()[:1]))
        self.assertFalse(self.check(duplicate=self.rows(mean=.03)))
        self.assertFalse(self.check(champion=False))
        self.assertFalse(self.check(duel_lower=.49))


if __name__ == "__main__":
    unittest.main()
