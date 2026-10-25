from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from junqi_search_v3.dataset import (
    OracleSample,
    family_split,
    iter_shard,
    write_shard,
)
from junqi_search_v3.supervised import soft_target


ROOT = Path(__file__).resolve().parents[1]


class SearchV3DatasetTest(unittest.TestCase):
    def test_family_split_is_stable_and_whole(self) -> None:
        first = family_split(123, 77)
        self.assertEqual(first, family_split(123, 77))
        self.assertIn(first, {"train", "validation", "test"})

    def test_sparse_roundtrip_includes_repetition_context(self) -> None:
        sample = OracleSample(
            family_id=9,
            split="validation",
            state_hash=123456,
            ply=42,
            player=-1,
            opponent="exact082",
            terminal_reason="REPETITION",
            observation=np.zeros((107, 12, 5), np.float32),
            legal_actions=np.asarray([61, 122, 183], np.int32),
            repetition_contexts=np.asarray([0, 1, 2], np.uint8),
            q_values=np.asarray([-0.2, 0.0, 0.7], np.float32),
            wdl=np.asarray([0.0, 1.0, 0.0], np.float32),
            important=True,
            action_trace=[311, 311, 665, 665],
        )
        path = ROOT / "runs" / "_search_v3_dataset_test.npz"
        try:
            write_shard(path, [sample])
            self.assertFalse(path.with_suffix(".npz.partial").exists())
            row = list(iter_shard(path))[0]
        finally:
            path.unlink(missing_ok=True)
            path.with_suffix(".npz.partial").unlink(missing_ok=True)
        np.testing.assert_array_equal(
            row["legal_actions"], sample.legal_actions
        )
        np.testing.assert_array_equal(
            row["repetition_contexts"], sample.repetition_contexts
        )
        np.testing.assert_allclose(row["q_values"], sample.q_values, atol=2e-4)
        self.assertEqual(row["action_trace"], sample.action_trace)

    def test_oracle_soft_target_is_legal_normalized_and_not_one_hot(self) -> None:
        target = soft_target(np.asarray([-0.4, 0.1, 0.2]), 0.2)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=6)
        self.assertTrue(np.all(target > 0.0))
        self.assertLess(float(target.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
