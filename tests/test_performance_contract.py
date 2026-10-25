from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "build")]

from junqi_rl.bots import HeuristicLookaheadBot, HeuristicProductionBot, build_bot
from performance_gate import (
    MIN_CPP_BACKEND_STEPS_PER_SEC,
    MIN_HEURISTIC_STEPS_PER_SEC,
    PRODUCTION_HEURISTICS,
)


class PerformanceContractTests(unittest.TestCase):
    def test_gate_is_at_least_one_hundred_steps_per_second(self):
        self.assertGreaterEqual(MIN_HEURISTIC_STEPS_PER_SEC, 100.0)

    def test_cpp_backend_has_a_strict_gate(self):
        self.assertGreaterEqual(MIN_CPP_BACKEND_STEPS_PER_SEC, 50_000.0)

    def test_all_production_versions_are_registered(self):
        self.assertEqual(
            set(PRODUCTION_HEURISTICS),
            {"heuristic_v1", "heuristic_v2", "heuristic_v3"},
        )

    def test_default_heuristic_is_production_v3(self):
        bot = build_bot("heuristic", torch.device("cpu"))
        self.assertIsInstance(bot, HeuristicProductionBot)

    def test_production_alphabeta_uses_fast_budget(self):
        bot = build_bot("ab2", torch.device("cpu"))
        self.assertLessEqual(bot.time_limit_ms, 8)

    def test_lookahead_requires_explicit_experimental_name(self):
        bot = build_bot("lookahead_experimental", torch.device("cpu"))
        self.assertIsInstance(bot, HeuristicLookaheadBot)


if __name__ == "__main__":
    unittest.main()
