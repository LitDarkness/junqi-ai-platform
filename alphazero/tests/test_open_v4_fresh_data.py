import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from run_open_v4_fresh_data import chunk_iterations, replay_groups, is_holdout


class FreshDataTests(unittest.TestCase):
    def groups(self, stage, variant):
        return replay_groups(stage, 3, [18, 19, 20], 21, 4, variant)

    def test_control_data_is_exact_subset_each_round(self):
        for stage in range(1, 7):
            control = self.groups(stage, "control")
            fresh = self.groups(stage, "fresh")
            self.assertEqual(len(control), 3)
            for small, large in zip(control, fresh):
                self.assertEqual(small, large[:1])

    def test_old_data_ages_out_after_three_rounds(self):
        self.assertEqual(self.groups(1, "control"), [[("original", 19)], [("original", 20)], [("fresh", 21)]])
        for variant in ("control", "fresh"):
            self.assertFalse(any(source == "original" for group in self.groups(3, variant) for source, _ in group))

    def test_chunks_never_repeat_or_overlap_parent_iterations(self):
        ids = [i for stage in range(1, 7) for i in chunk_iterations(stage, 21, 4)]
        self.assertEqual(ids, list(range(21, 45)))

    def test_paired_colors_cannot_cross_training_holdout(self):
        train, held = set(), set()
        for i in range(21_000_000, 21_000_032):
            for _color in (1, -1):
                (held if is_holdout(i, 2026100307) else train).add(i)
        self.assertTrue(train and held)
        self.assertFalse(train & held)

    def test_invalid_variant_rejected(self):
        with self.assertRaises(ValueError):
            self.groups(1, "final")


if __name__ == "__main__":
    unittest.main()
