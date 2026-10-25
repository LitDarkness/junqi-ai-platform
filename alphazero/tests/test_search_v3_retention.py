from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

from junqi_search_v3.championship import TriggeredSealedChampionship
from junqi_search_v3.config import SearchV3Config
from junqi_search_v3.alternating_lineage import (
    candidate_kl_eligible,
    candidate_rejection_reasons,
    candidate_selection_rank,
)
from junqi_search_v3.retention import CheckpointRetention
from junqi_search_v3.strength_continuation import (
    championship_trigger,
)
from junqi_search_v3.training_retention import (
    trim_competitive_registry,
)


class SearchV3RetentionTests(unittest.TestCase):
    def test_identical_model_championship_is_skipped_before_suite(self):
        run = Path(__file__).resolve().parent / "identical_title_test"
        object_under_test = TriggeredSealedChampionship.__new__(
            TriggeredSealedChampionship
        )
        object_under_test.training_run = run
        object_under_test.config = SimpleNamespace(seed=17)
        model = SimpleNamespace(
            state_dict=lambda: {"weight": torch.tensor([1.0])}
        )
        object_under_test._load_model = lambda *_: model
        object_under_test._rule_hash = lambda: "rules"
        object_under_test._search_configuration_hash = (
            lambda: "search"
        )
        decisions = []
        object_under_test._write_decision = (
            lambda _, decision: decisions.append(decision)
        )
        object_under_test._build_suite = lambda **_: self.fail(
            "identical models must not build a suite"
        )
        try:
            result = object_under_test.run(
                challenger_iteration=80,
                challenger_checkpoint=Path("challenger.pt"),
                champion_iteration=50,
                champion_checkpoint=Path("champion.pt"),
                training_report={},
            )
            self.assertEqual(
                result["status"], "SKIPPED_IDENTICAL_MODEL"
            )
            self.assertFalse(result["promoted"])
            self.assertEqual(len(decisions), 1)
        finally:
            if run.exists():
                for path in sorted(
                    run.rglob("*"), reverse=True
                ):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                run.rmdir()

    def test_candidate_kl_gate_does_not_reuse_early_peak_as_final(self):
        self.assertTrue(candidate_kl_eligible(0.0004, 0.0010287))
        self.assertFalse(candidate_kl_eligible(0.00101, 0.0011))
        self.assertFalse(candidate_kl_eligible(0.0004, 0.00151))

    def test_static_regret_is_safety_gate_not_epoch0_rank(self):
        baseline = {
            "policy_steps": 0,
            "value_steps": 0,
            "paired_policy": {
                "classification": "neutral",
                "relative_delta": 0.0,
            },
            "paired_wdl": {
                "classification": "neutral",
                "relative_delta": 0.0,
            },
            "candidate_final_kl": 0.0,
            "value_step_peak_kl": 0.0,
            "probe": {
                "bare_oracle_regret": 0.0143,
                "search64_oracle_regret": 0.00886,
            },
            "eligible": True,
        }
        candidate = {
            "policy_steps": 8,
            "value_steps": 4,
            "paired_policy": {
                "classification": "improved",
                "relative_delta": -0.08,
            },
            "paired_wdl": {
                "classification": "inconclusive",
                "relative_delta": -0.01,
            },
            "candidate_final_kl": 0.0004,
            "value_step_peak_kl": 0.0011,
            "probe": {
                "bare_oracle_regret": 0.016,
                "search64_oracle_regret": 0.012,
            },
        }
        candidate["rejection_reasons"] = candidate_rejection_reasons(
            candidate, baseline
        )
        candidate["eligible"] = not candidate["rejection_reasons"]
        self.assertTrue(candidate["eligible"])
        self.assertGreater(
            tuple(candidate_selection_rank(candidate)),
            tuple(candidate_selection_rank(baseline)),
        )

    def test_recent_archive_champion_and_competitive_are_preserved(
        self,
    ) -> None:
        run = (
            Path(__file__).resolve().parent
            / "retention_test_workspace"
        )
        checkpoints = run / "checkpoints"
        candidates = run / "candidates"
        checkpoints.mkdir(parents=True, exist_ok=True)
        candidates.mkdir(parents=True, exist_ok=True)
        metadata_paths = (
            run / "championship.json",
            run / "competitive.json",
            run / "retention.json",
        )
        for path in (*checkpoints.glob("*.pt"), *candidates.glob("*.pt")):
            path.unlink()
        for path in metadata_paths:
            path.unlink(missing_ok=True)
        try:
            for iteration in range(1, 13):
                (
                    checkpoints
                    / f"loop_iteration_{iteration:06d}.pt"
                ).write_bytes(bytes([iteration]))
                (
                    candidates / f"iteration_{iteration:06d}.pt"
                ).write_bytes(bytes([iteration]))
            (run / "championship.json").write_text(
                json.dumps(
                    {
                        "current": {"iteration": 2},
                        "history": [],
                    }
                ),
                encoding="utf-8",
            )
            (run / "competitive.json").write_text(
                json.dumps(
                    {
                        "iterations": {
                            "3": {"permanent": True}
                        }
                    }
                ),
                encoding="utf-8",
            )
            manager = CheckpointRetention(
                run,
                recent=5,
                archive_interval=5,
                starting_incumbent=2,
            )
            report = manager.apply(12)
            official = {
                int(path.stem.rsplit("_", 1)[-1])
                for path in checkpoints.glob("*.pt")
            }
            candidate = {
                int(path.stem.rsplit("_", 1)[-1])
                for path in candidates.glob("*.pt")
            }
            self.assertEqual(
                official, {2, 3, 5, 8, 9, 10, 11, 12}
            )
            self.assertEqual(candidate, set())
            self.assertEqual(report["phase"], "completed")
        finally:
            for path in (
                *checkpoints.glob("*.pt"),
                *candidates.glob("*.pt"),
            ):
                path.unlink()
            for path in metadata_paths:
                path.unlink(missing_ok=True)

    def test_scaled_config_remains_single_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = SearchV3Config.load(
            root
            / "config"
            / "open_junqi_search_v3_scaled_training.json"
        )
        self.assertEqual(config.loop.iterations, 100)
        self.assertFalse(config.boundary.mirror)
        self.assertFalse(config.boundary.multiple_layouts)
        self.assertEqual(
            config.run_directory,
            "runs/open_junqi_search_v3_scaled_training",
        )

    def test_strength_continuation_config_and_title_trigger(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = SearchV3Config.load(
            root
            / "config"
            / "open_junqi_search_v3_strength_continuation.json"
        )
        self.assertFalse(config.boundary.mirror)
        self.assertFalse(config.boundary.multiple_layouts)
        self.assertEqual(config.puct.selfplay_simulations, 64)
        self.assertEqual(config.puct.repeated_state_simulations, 128)
        self.assertEqual(config.loop.games_per_iteration, 1024)
        state = {
            "accepted_updates_at_last_evaluation": 5,
            "completed_rounds_at_last_evaluation": 48,
        }
        self.assertFalse(
            championship_trigger(
                state,
                {
                    "accepted_update_iterations": 7,
                    "completed_iterations": 57,
                },
            )[0]
        )
        self.assertTrue(
            championship_trigger(
                state,
                {
                    "accepted_update_iterations": 8,
                    "completed_iterations": 51,
                },
            )[0]
        )
        self.assertFalse(
            championship_trigger(
                state,
                {
                    "accepted_update_iterations": 5,
                    "completed_iterations": 58,
                },
            )[0]
        )
        self.assertTrue(
            championship_trigger(
                state,
                {
                    "accepted_update_iterations": 6,
                    "completed_iterations": 58,
                },
            )[0]
        )

    def test_late_stage_logic_is_separately_hashed_and_adaptive(
        self,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        legacy = SearchV3Config.load(
            root
            / "config"
            / "open_junqi_search_v3_strength_continuation.json"
        )
        late = SearchV3Config.load(
            root
            / "config"
            / "open_junqi_search_v3_strength_late_stage.json"
        )
        self.assertIsNone(legacy.training_logic)
        self.assertNotEqual(legacy.hash(), late.hash())
        logic = late.training_logic
        self.assertIsNotNone(logic)
        self.assertEqual(
            logic.schema, "late_stage_short_horizon_v1"
        )
        self.assertEqual(
            logic.policy_candidate_steps, [1, 2, 3, 4, 6]
        )
        self.assertEqual(logic.value_candidate_steps, [0, 1, 2])
        self.assertEqual(logic.selected_policy_step, 3)
        self.assertEqual(logic.selected_value_step, 2)
        self.assertTrue(logic.adaptive_selection)

    def test_retention_lock_survives_competitive_trimming(self) -> None:
        path = (
            Path(__file__).resolve().parent
            / "retention_lock_test.json"
        )
        try:
            path.write_text(
                json.dumps(
                    {
                        "schema": (
                            "search-v3-competitive-checkpoints-v1"
                        ),
                        "iterations": {
                            str(iteration): {
                                "search64_oracle_regret": float(
                                    iteration
                                ),
                                "retention_lock": iteration == 107,
                            }
                            for iteration in range(100, 112)
                        },
                    }
                ),
                encoding="utf-8",
            )
            result = trim_competitive_registry(
                path,
                champion_iterations=set(),
                maximum_non_champion=4,
            )
            self.assertIn("107", result["iterations"])
        finally:
            path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
