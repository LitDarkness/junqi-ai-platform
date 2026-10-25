from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

import numpy as np
import torch

from junqi_az.config import ModelConfig, ReplayConfig
from junqi_az.layout_policy import mirror_layout, validate_layout
from junqi_az.model import PolicyValueNet
from junqi_az.replay import ShardedReplayBuffer, SparseSample
from junqi_fixed_rl.checkpoints import CheckpointManager
from junqi_fixed_rl.config import FixedRLConfig
from junqi_fixed_rl.evaluator import (
    repetition_cycle_details,
    repetition_severity,
    score_statistics,
)
from junqi_fixed_rl.layout import (
    FixedLayoutPairBackend,
    FixedSameLayoutBackend,
    load_fixed_layout,
)


ROOT = Path(__file__).resolve().parents[1]


class FixedLayoutRLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.layout = load_fixed_layout(
            ROOT / "config/layouts/fixed_layout_v1.json",
            "public_conservative_advance",
            project_root=ROOT,
        )

    def test_layout_is_legal_and_same_for_both_sides(self) -> None:
        self.assertEqual(validate_layout(self.layout.setup), [])
        state = FixedSameLayoutBackend(self.layout).new_state(123)
        provenance = state.provenance()
        self.assertEqual(provenance["red_layout"], provenance["black_layout"])
        self.assertEqual(
            provenance["red_layout_hash"], provenance["black_layout_hash"]
        )
        self.assertTrue(provenance["symmetric"])

    def test_mirror_is_involution_and_legal(self) -> None:
        mirrored = mirror_layout(self.layout.setup)
        self.assertEqual(validate_layout(mirrored), [])
        self.assertEqual(mirror_layout(mirrored), self.layout.setup)

    def test_native_features_and_action_round_trip(self) -> None:
        state = FixedSameLayoutBackend(self.layout).new_state(1)
        features = state.feature_planes()
        self.assertEqual(features.shape, (107, 12, 5))
        # Fixed rule thresholds make these normalized counters equivalent to
        # the remaining distance to no-capture/repetition draws.
        self.assertTrue(np.allclose(features[40], 0.0))
        self.assertTrue(np.allclose(features[41], 1.0 / 3.0))
        for _ in range(20):
            if state.is_terminal():
                break
            legal = state.legal_actions()
            self.assertTrue(len(legal))
            self.assertEqual(len(legal), len(state.repetition_contexts()))
            action = int(legal[len(legal) // 2])
            import JunqiAZ

            self.assertEqual(
                JunqiAZ.JunqiAZState.flip_action(
                    JunqiAZ.JunqiAZState.flip_action(action)
                ),
                action,
            )
            state.apply(action)

    def test_policy_masks_illegal_actions_and_outputs_wdl(self) -> None:
        config = ModelConfig(
            input_channels=107,
            channels=16,
            residual_blocks=1,
            policy_channels=8,
            value_head_type="wdl",
            wdl_search_mix=1.0,
            repetition_context=True,
            pairwise_action_bias=True,
        )
        model = PolicyValueNet(config).eval()
        observation = torch.zeros(2, 107, 12, 5)
        mask = torch.zeros(2, 3600, dtype=torch.bool)
        mask[:, [1, 72, 3599]] = True
        repetitions = torch.zeros(2, 3600, dtype=torch.uint8)
        with torch.inference_mode():
            logits, values, _, wdl_logits, _ = model.forward_heads(
                observation, mask, repetitions
            )
        self.assertEqual(tuple(logits.shape), (2, 3600))
        self.assertEqual(tuple(wdl_logits.shape), (2, 3))
        self.assertTrue(torch.isfinite(values).all())
        self.assertTrue((logits[~mask] < -1e20).all())

    def test_sparse_float16_replay_round_trip(self) -> None:
        state = FixedSameLayoutBackend(self.layout).new_state(7)
        legal = state.legal_actions()
        probabilities = np.full(len(legal), 1.0 / len(legal))
        visits = np.maximum(1, np.rint(probabilities * 10000)).astype(np.uint16)
        directory = ROOT / "tests/replay_tmp"
        for path in directory.glob("*"):
            if path.name != ".gitkeep":
                path.unlink()
        try:
            replay = ShardedReplayBuffer(
                ReplayConfig(
                    directory=str(directory),
                    capacity=100,
                    shard_size=1,
                    observation_dtype="float16",
                    require_game_provenance=True,
                    require_repetition_context=True,
                )
            )
            replay.append(
                SparseSample(
                    observation=state.feature_planes(),
                    legal_actions=legal,
                    visit_counts=visits,
                    outcome=1.0,
                    repetition_contexts=state.repetition_contexts(),
                    game_key=1,
                    ply_index=0,
                )
            )
            batch = replay.sample(1, split="all")
            self.assertEqual(
                replay.provenance_summary(),
                {"samples": 1, "games": 1, "max_game_key": 1},
            )
            self.assertEqual(batch.observations.shape, (1, 107, 12, 5))
            self.assertAlmostEqual(float(batch.probabilities[0].sum()), 1.0, 5)
            dense = batch.dense_policy(3600)
            self.assertAlmostEqual(float(dense[0].sum()), 1.0, 5)
            illegal = np.ones(3600, dtype=np.bool_)
            illegal[legal] = False
            self.assertTrue((dense[0, illegal] == 0).all())
        finally:
            for path in directory.glob("*"):
                if path.name != ".gitkeep":
                    path.unlink()

    def test_replay_iteration_transaction_rolls_back_and_commits(self) -> None:
        state = FixedSameLayoutBackend(self.layout).new_state(17)
        legal = state.legal_actions()
        visits = np.ones(len(legal), dtype=np.uint16)

        def sample(game_key: int) -> SparseSample:
            return SparseSample(
                observation=state.feature_planes(),
                legal_actions=legal,
                visit_counts=visits,
                outcome=0.0,
                repetition_contexts=state.repetition_contexts(),
                game_key=game_key,
                ply_index=0,
            )

        directory = ROOT / "tests/replay_transaction_tmp"
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()
        try:
            replay = ShardedReplayBuffer(
                ReplayConfig(
                    directory=str(directory),
                    capacity=2,
                    shard_size=1,
                    observation_dtype="float16",
                    require_game_provenance=True,
                    require_repetition_context=True,
                )
            )
            replay.append(sample(1))
            replay.append(sample(2))
            first_shard = directory / "shard_00000000.npz"
            self.assertTrue(first_shard.is_file())

            replay.begin_transaction()
            replay.append(sample(3))
            self.assertTrue(
                first_shard.is_file(),
                "capacity-trimmed base shard must survive until commit",
            )
            report = replay.rollback_transaction()
            self.assertEqual(report["restored_samples"], 2)
            self.assertEqual(
                replay.provenance_summary(),
                {"samples": 2, "games": 2, "max_game_key": 2},
            )
            self.assertFalse((directory / "shard_00000002.npz").exists())

            replay.begin_transaction()
            replay.append(sample(3))
            replay.commit_transaction()
            self.assertFalse(first_shard.exists())
            self.assertEqual(
                replay.provenance_summary(),
                {"samples": 2, "games": 2, "max_game_key": 3},
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_checkpoint_tiers_and_initial_champion(self) -> None:
        config = FixedRLConfig.load(ROOT / "config/fixed_layout_rl_v1.json")
        directory = ROOT / "tests/checkpoint_manager_tmp"
        shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir()
        try:
            manager = CheckpointManager(directory, config)
            for iteration in range(1, 16):
                manager.save_latest(
                    {
                        "iteration": iteration,
                        "phase": "completed",
                        "iteration_completed": True,
                    }
                )
                manager.register_completed(iteration)

            self.assertEqual(
                sorted(path.name for path in manager.archive_directory.glob("*.pt")),
                [
                    "iteration_000005.pt",
                    "iteration_000010.pt",
                    "iteration_000015.pt",
                ],
            )
            self.assertEqual(
                sorted(path.name for path in manager.recent_directory.glob("*.pt")),
                [
                    "iteration_000011.pt",
                    "iteration_000012.pt",
                    "iteration_000013.pt",
                    "iteration_000014.pt",
                    "iteration_000015.pt",
                ],
            )
            champion = manager.initialize_champion(
                layout_id="layout",
                layout_hash="layout-hash",
                config_hash="config-hash",
            )
            self.assertEqual(champion["source_iteration"], 10)
            self.assertTrue(manager.champion_path.is_file())
            self.assertTrue(
                (manager.archive_directory / "iteration_000010.pt").is_file()
            )
        finally:
            shutil.rmtree(directory, ignore_errors=True)

    def test_evaluation_score_interval_keeps_draws(self) -> None:
        statistics = score_statistics([1.0, 0.5, 0.0, 0.5])
        self.assertAlmostEqual(statistics["score_rate"], 0.5)
        self.assertGreater(statistics["standard_error"], 0.0)
        self.assertLess(statistics["ci95_lower"], 0.5)
        self.assertGreater(statistics["ci95_upper"], 0.5)

    def test_repetition_cycle_diagnostic(self) -> None:
        hashes = [11, 22, 33, 44, 11, 22, 33, 44, 11]
        actions = [
            {
                "mover_side": "red" if index % 2 == 0 else "black",
                "action": index,
            }
            for index in range(8)
        ]
        cycle = repetition_cycle_details(
            hashes,
            actions,
            [0.25] * 8,
            terminal_value=0.0,
            opponent_type="current/latest",
        )
        self.assertEqual(cycle["first_cycle_ply"], 0)
        self.assertEqual(cycle["cycle_length"], 4)
        self.assertEqual(cycle["cycle_repeat_intervals"], [4, 4])
        self.assertEqual(len(cycle["position_hash_sequence"]), 5)
        self.assertEqual(cycle["value_at_cycle_start"], 0.25)
        self.assertEqual(repetition_severity(0.149), "normal")
        self.assertEqual(repetition_severity(0.15), "inspect_by_opponent")
        self.assertEqual(repetition_severity(0.30), "suspicious")
        self.assertEqual(
            repetition_severity(0.50), "likely_policy_degradation"
        )

    def test_explicit_different_layout_pair(self) -> None:
        # Source distributions omit historical evaluation corpora. Use a fresh
        # seeded legal layout while preserving the actual different-layout test.
        from dataclasses import replace
        from junqi_open_v4.layouts import random_layout
        other = replace(self.layout, layout_id="source_distribution_fixture",
                        setup=random_layout(2026102509), source_sha256="fresh-seeded-test-layout")
        state = FixedLayoutPairBackend(self.layout, other).new_state(99)
        provenance = state.provenance()
        self.assertEqual(
            provenance["red_layout_id"], "public_conservative_advance"
        )
        self.assertEqual(
            provenance["black_layout_id"], "source_distribution_fixture"
        )
        self.assertNotEqual(
            provenance["red_layout_hash"], provenance["black_layout_hash"]
        )
        self.assertFalse(provenance["symmetric"])

    def test_config_is_valid_json_and_utf8(self) -> None:
        payload = json.loads(
            (ROOT / "config/fixed_layout_rl_v1.json").read_text(encoding="utf-8")
        )
        self.assertEqual(payload["model"]["input_channels"], 107)
        self.assertEqual(payload["model"]["value_head_type"], "wdl")
        self.assertEqual(payload["layout"]["layout_id"], self.layout.layout_id)
        self.assertEqual(payload["checkpoint"]["archive_interval"], 5)
        self.assertEqual(payload["evaluation"]["initial_games"], 512)
        self.assertEqual(
            payload["evaluation"]["repetition_warning_fraction"], 0.15
        )


if __name__ == "__main__":
    unittest.main()
