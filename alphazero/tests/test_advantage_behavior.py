from __future__ import annotations

import numpy as np
import unittest


def final_behavior(raw_logits: np.ndarray, temperature: float, mix: float) -> np.ndarray:
    scaled = raw_logits / temperature
    probabilities = np.exp(scaled - scaled.max())
    probabilities /= probabilities.sum()
    return (1.0 - mix) * probabilities + mix / len(probabilities)


class AdvantageBehaviorTests(unittest.TestCase):
    def test_behavior_log_prob_is_post_temperature_and_post_mixture(self) -> None:
        raw = np.asarray([0.0, 1.0, -1.0], np.float64)
        behavior = final_behavior(raw, 0.5, 0.2)
        selected = 1
        stored = float(np.log(behavior[selected]))
        self.assertFalse(np.isclose(stored, raw[selected]))
        reconstructed = final_behavior(raw, 0.5, 0.2)
        np.testing.assert_allclose(np.exp(stored), reconstructed[selected])
        np.testing.assert_allclose(reconstructed.sum(), 1.0)
