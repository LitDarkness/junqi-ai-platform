from __future__ import annotations

import json
import unittest
from pathlib import Path

from junqi_search_v3.config import SearchV3Config
from junqi_search_v3.sealed_continuation import (
    DEFAULT_MATCHUPS,
    MultiCheckpointContinuationEvaluator,
    SealedContinuationExperiment,
    _family_namespace,
)


class SearchV3ContinuationArenaTests(unittest.TestCase):
    def test_sealed_formal_plan_and_config(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = SearchV3Config.load(
            root
            / "config"
            / "open_junqi_search_v3_sealed_continuation_v1.json"
        )
        self.assertEqual(config.seed, 2026072701)
        self.assertFalse(config.boundary.mirror)
        self.assertFalse(config.boundary.multiple_layouts)
        self.assertEqual(len(DEFAULT_MATCHUPS), 7)
        self.assertEqual(
            set(DEFAULT_MATCHUPS),
            {
                ("bare_50", "bare_10"),
                ("search64_50", "search64_10"),
                ("bare_50", "exact082"),
                ("search64_50", "exact082"),
                ("search64_50", "bare_50"),
                ("bare_50", "bare_49"),
                ("search64_50", "search64_49"),
            },
        )
        self.assertEqual(
            MultiCheckpointContinuationEvaluator._network_agent(
                "search64_49"
            ),
            ("search64", 49),
        )
        self.assertNotEqual(
            _family_namespace(config.seed, 1),
            _family_namespace(config.seed, 2),
        )

    def test_sealed_experiment_run_method_is_not_shadowed(self) -> None:
        experiment = object.__new__(SealedContinuationExperiment)
        experiment.run_dir = Path("unused")
        self.assertTrue(callable(experiment.run))

    def test_materialized_full_arena_is_paired_and_complete(self) -> None:
        root = Path(__file__).resolve().parents[1]
        report_path = (
            root
            / "runs"
            / "open_junqi_search_v3_strength"
            / "continuation_arena"
            / "report.json"
        )
        if not report_path.exists():
            self.skipTest("full continuation arena has not been run")
        report = json.loads(
            report_path.read_text(encoding="utf-8")
        )
        self.assertEqual(report["states_per_matchup"], 1024)
        self.assertEqual(report["games_per_matchup"], 2048)
        self.assertEqual(
            set(report["matchups"]),
            {
                "bare_vs_exact082",
                "search64_vs_exact082",
                "search64_vs_bare",
            },
        )
        for result in report["matchups"].values():
            self.assertEqual(result["games"], 2048)
            self.assertEqual(result["paired_states"], 1024)
            self.assertEqual(
                result["wins"] + result["draws"] + result["losses"],
                2048,
            )
            self.assertTrue(result["paired_role_swap_verified"])
            self.assertTrue(result["same_suite_seed_verified"])
            self.assertEqual(
                result["role_split"]["primary_red"]["games"], 1024
            )
            self.assertEqual(
                result["role_split"]["primary_black"]["games"], 1024
            )


if __name__ == "__main__":
    unittest.main()
