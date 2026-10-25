from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

from junqi_search_v3.config import SearchV3Config
from junqi_search_v3.strength import (
    ORACLE_BUDGETS,
    SUITE_QUOTAS,
    SUITE_SIZE,
    _inverse_action,
)


class SearchV3StrengthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_strength_contract_is_fixed_and_unmirrored(self) -> None:
        config = SearchV3Config.load(
            self.root / "config" / "open_junqi_search_v3_strength.json"
        )
        self.assertEqual(config.initialization, "strength_incumbent")
        self.assertEqual(
            config.boundary.layout_id, "public_conservative_advance"
        )
        self.assertFalse(config.boundary.mirror)
        self.assertFalse(config.boundary.multiple_layouts)
        self.assertEqual(ORACLE_BUDGETS, (500, 2000, 8000, 32000))

    def test_suite_quota_and_action_inverse_contract(self) -> None:
        self.assertEqual(sum(SUITE_QUOTAS.values()), SUITE_SIZE)
        for action in (0, 59, 60, 721, 3599):
            self.assertEqual(
                _inverse_action(_inverse_action(action)), action
            )

    def test_materialized_suite_is_unique_and_training_excluded(self) -> None:
        directory = (
            self.root
            / "runs"
            / "open_junqi_search_v3_strength"
            / "single_layout_strength_suite"
        )
        manifest_path = directory / "manifest.json"
        states_path = directory / "states.jsonl"
        if not manifest_path.exists() or not states_path.exists():
            self.skipTest("strength suite has not been materialized")
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8")
        )
        rows = [
            json.loads(line)
            for line in states_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        self.assertEqual(len(rows), SUITE_SIZE)
        self.assertEqual(
            len({int(row["state_key"]) for row in rows}), SUITE_SIZE
        )
        self.assertTrue(manifest["training_excluded"])
        self.assertTrue(manifest["replay_write_prohibited"])
        self.assertEqual(
            hashlib.sha256(states_path.read_bytes()).hexdigest(),
            manifest["states_sha256"],
        )
        self.assertEqual(
            dict(
                sorted(
                    {
                        category: sum(
                            row["primary_category"] == category
                            for row in rows
                        )
                        for category in SUITE_QUOTAS
                    }.items()
                )
            ),
            dict(sorted(SUITE_QUOTAS.items())),
        )


if __name__ == "__main__":
    unittest.main()
