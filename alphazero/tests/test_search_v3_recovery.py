from __future__ import annotations

import unittest

import torch

from junqi_az.config import ModelConfig
from junqi_az.model import PolicyValueNet
from junqi_search_v3.loop import SearchSample
from junqi_search_v3.recovery import (
    RECOVERY_CONFIG,
    _anchor_kl,
    _anchor_policy,
    _paired_bootstrap,
    _phase_learning_rates,
)


class SearchV3RecoveryTests(unittest.TestCase):
    def test_formal_schedule_contract(self) -> None:
        self.assertEqual(RECOVERY_CONFIG["maximum_epochs"], 2)
        self.assertEqual(
            RECOVERY_CONFIG["policy_steps_per_value_step"], 2
        )
        self.assertAlmostEqual(
            sum(RECOVERY_CONFIG["policy_mix"].values()), 1.0
        )
        self.assertAlmostEqual(
            sum(RECOVERY_CONFIG["value_mix"].values()), 1.0
        )
        self.assertEqual(
            RECOVERY_CONFIG["oracle_scalar_value_weight"], 0.05
        )

    def test_phase_learning_rate_groups(self) -> None:
        parameters = [torch.nn.Parameter(torch.zeros(())) for _ in range(3)]
        optimizer = torch.optim.SGD(
            [
                {"name": "trunk", "params": [parameters[0]]},
                {"name": "policy_head", "params": [parameters[1]]},
                {"name": "wdl_head", "params": [parameters[2]]},
            ],
            lr=1.0,
        )
        _phase_learning_rates(optimizer, "policy", 1e-4)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [2.5e-5, 1e-4, 0.0],
        )
        _phase_learning_rates(optimizer, "value", 1e-4)
        self.assertEqual(
            [group["lr"] for group in optimizer.param_groups],
            [1e-5, 0.0, 1e-4],
        )

    def test_paired_bootstrap_uses_family_units(self) -> None:
        before = {
            family: {
                "policy_excess": 1.0,
                "wdl_ce": 1.0,
            }
            for family in range(100)
        }
        improved = {
            family: {
                "policy_excess": 0.98 + (family % 2) * 0.002,
                "wdl_ce": 1.0,
            }
            for family in before
        }
        result = _paired_bootstrap(
            before, improved, "policy_excess", 123
        )
        self.assertEqual(result["families"], 100)
        self.assertEqual(
            result["unit"], "held-out game/trajectory family"
        )
        self.assertEqual(result["classification"], "improved")
        self.assertLess(result["ci95"][1], 0.0)

    def test_small_uncertain_change_is_neutral(self) -> None:
        before = {
            family: {
                "policy_excess": 1.0,
                "wdl_ce": 1.0,
            }
            for family in range(100)
        }
        after = {
            family: {
                "policy_excess": (
                    1.010 if family % 2 else 0.992
                ),
                "wdl_ce": 1.0,
            }
            for family in before
        }
        result = _paired_bootstrap(
            before, after, "policy_excess", 456
        )
        self.assertEqual(result["classification"], "neutral")
        self.assertLess(abs(result["relative_delta"]), 0.005)
        self.assertLessEqual(result["ci95"][0], 0.0)
        self.assertGreaterEqual(result["ci95"][1], 0.0)

    def test_frozen_policy_anchor_supports_backward(self) -> None:
        model = PolicyValueNet(
            ModelConfig(
                input_channels=1,
                channels=8,
                residual_blocks=0,
                policy_channels=8,
                action_size=3600,
                value_head_type="wdl",
                repetition_context=True,
                pairwise_action_bias=True,
            )
        )
        sample = SearchSample(
            game_id=0,
            state_hash=1,
            observation=torch.zeros(1, 12, 5).numpy(),
            legal_actions=torch.tensor([0, 1]).numpy(),
            repetition_contexts=torch.tensor([0, 0]).numpy(),
            visits=torch.tensor([1, 1]).numpy(),
            player=1,
            ply=0,
            root_entropy=0.0,
            wdl=torch.tensor([0.0, 1.0, 0.0]).numpy(),
        )
        anchors = _anchor_policy(model, [sample], torch.device("cpu"))
        model.set_training_phase("value_distill")
        loss = _anchor_kl(
            model, [sample], anchors, torch.device("cpu")
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
