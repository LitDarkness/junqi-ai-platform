import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_open_v4_search_confirmation import full_budget_confirmed


class FullBudgetTests(unittest.TestCase):
    def check(self, families=64, gain_lower=.01, duplicate_mean=0, gate=True, duel_lower=.55):
        gain = [dict(families=families, mean=.1, ci95=[gain_lower, .2]) for _ in range(2)]
        duplicate = [dict(families=families, mean=duplicate_mean, ci95=[-.01, .01]) for _ in range(2)]
        return full_budget_confirmed(gain, duplicate, dict(families=families, ci95=[duel_lower,.65]),
                                     dict(champion_accepted=gate), families, .02)

    def test_secondary_screen_alone_cannot_qualify(self):
        self.assertFalse(self.check(families=32))
        self.assertTrue(self.check())

    def test_gain_gate_duplicate_and_direct_duel_all_required(self):
        self.assertFalse(self.check(gain_lower=-.01))
        self.assertFalse(self.check(duplicate_mean=.03))
        self.assertFalse(self.check(gate=False))
        self.assertFalse(self.check(duel_lower=.49))


if __name__ == "__main__":
    unittest.main()
