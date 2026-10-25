from __future__ import annotations

import numpy as np
import unittest

from junqi_adv_rl.returns import (
    compute_player_lambda_returns,
    filter_advantages,
    magnet_distribution,
)


class AdvantageReturnTests(unittest.TestCase):
    def test_magnet_is_uniform_over_piece_then_destination(self) -> None:
        legal = np.asarray([0 * 60 + 1, 0 * 60 + 2, 5 * 60 + 6], np.int32)
        rho = magnet_distribution(legal)
        np.testing.assert_allclose(rho, [0.25, 0.25, 0.5], atol=1e-7)
        self.assertAlmostEqual(float(rho.sum()), 1.0)


    def test_lambda_return_uses_same_player_sequence_and_terminal_sign(self) -> None:
    # Alternating plies: RED states are indices 0,2; BLACK states are 1,3.
        sides = np.asarray([1, -1, 1, -1], np.int8)
        old_values = np.asarray([0.2, -0.4, 0.6, -0.8], np.float32)
        old_wdl = np.asarray(
        [
            [0.2, 0.4, 0.4],
            [0.5, 0.4, 0.1],
            [0.1, 0.2, 0.7],
            [0.8, 0.1, 0.1],
        ],
        np.float32,
    )
        returns, advantage, xi = compute_player_lambda_returns(
        sides,
        old_values,
        old_wdl,
        terminal_outcome_red=1.0,
        lambda_advantage=0.5,
        lambda_outcome=0.8,
    )
    # Last action by each side bootstraps directly to its terminal outcome.
        np.testing.assert_allclose(returns[[2, 3]], [1.0, -1.0])
    # Earlier RED uses RED's next state V=0.6, never adjacent BLACK V=-0.4.
        np.testing.assert_allclose(returns[0], 0.5 * 0.6 + 0.5 * 1.0)
    # Earlier BLACK uses BLACK's next state V=-0.8 and BLACK terminal loss.
        np.testing.assert_allclose(returns[1], 0.5 * -0.8 + 0.5 * -1.0)
        np.testing.assert_allclose(advantage, returns - old_values)
        np.testing.assert_allclose(xi.sum(axis=1), 1.0)
        np.testing.assert_allclose(xi[2], [0.0, 0.0, 1.0])
        np.testing.assert_allclose(xi[3], [1.0, 0.0, 0.0])


    def test_draw_terminal_is_zero_and_draw_one_hot_for_both_sides(self) -> None:
        sides = np.asarray([1, -1], np.int8)
        values = np.asarray([0.7, -0.7], np.float32)
        old_wdl = np.asarray([[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]], np.float32)
        returns, advantage, xi = compute_player_lambda_returns(
            sides, values, old_wdl, 0.0
        )
        np.testing.assert_allclose(returns, [0.0, 0.0])
        np.testing.assert_allclose(advantage, [-0.7, 0.7])
        np.testing.assert_allclose(xi, [[0, 1, 0], [0, 1, 0]])


    def test_advantage_filter_requires_percentile_and_absolute_floor(self) -> None:
        advantage = np.asarray([-0.5, -0.02, 0.0, 0.03, 0.8], np.float32)
        eligible = np.asarray([True, True, True, True, False])
        keep, metrics = filter_advantages(
            advantage, eligible, percentile=75, minimum_absolute=0.01
        )
        np.testing.assert_array_equal(keep, [True, False, False, False, False])
        self.assertEqual(metrics["policy_samples_before"], 4)
        self.assertEqual(metrics["policy_samples_after"], 1)
