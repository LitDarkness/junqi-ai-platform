import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from open_v4_experiment_evidence import teacher_deltas


class ExperimentEvidenceTests(unittest.TestCase):
    def fixture(self):
        summaries = [{"candidate": name, "opponent": opponent} for opponent in ("parent", "teacher0", "teacher1") for name in ("control", "policy")]
        return {"role": "dev", "summaries": summaries,
            "paired_deltas": [{"from_matchup": i, "to_matchup": i+1, "delta": {"mean": i}} for i in (0, 2, 4)]}

    def test_network_duel_delta_excluded_from_teacher_evidence(self):
        self.assertEqual(teacher_deltas(self.fixture(), "control", "policy"), [{"mean": 2}, {"mean": 4}])

    def test_different_teacher_opponents_are_never_paired(self):
        arena = self.fixture()
        arena["paired_deltas"].append({"from_matchup": 2, "to_matchup": 5, "delta": {"mean": 99}})
        self.assertEqual(len(teacher_deltas(arena, "control", "policy")), 2)

    def test_unrequested_direction_is_excluded(self):
        self.assertEqual(teacher_deltas(self.fixture(), "policy", "control"), [])

    def test_final_arena_cannot_select_route(self):
        arena = self.fixture()
        arena["role"] = "final"
        with self.assertRaisesRegex(ValueError, "development"):
            teacher_deltas(arena, "control", "policy")
