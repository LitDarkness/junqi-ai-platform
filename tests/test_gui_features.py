from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "build")]

from GUI import JunqiWorkbench, _deployment_errors  # noqa: E402
from junqi_rl.bots import BaseBot, HumanBot, RandomBot  # noqa: E402


class ScoredBot(BaseBot):
    def __init__(self):
        super().__init__("scored-test-engine")

    def select_action(self, env_wrapper, player_id):
        actions = list(env_wrapper.env.get_legal_actions(player_id))
        return None if not actions else env_wrapper.action_to_policy(actions[0], player_id)

    def score_actions(self, board, legal_actions, player_id):
        return {int(action): float(index) for index, action in enumerate(legal_actions)}


class GuiFeatureTests(unittest.TestCase):
    def test_heatmap_is_immediate_engine_backed_and_engines_switch(self):
        scored = ScoredBot()
        random_bot = RandomBot()
        catalog = {"scored": scored, "random": random_bot}
        gui = JunqiWorkbench(
            HumanBot(),
            scored,
            engine_options=("scored", "random"),
            engine_resolver=catalog.__getitem__,
            red_engine_id="human",
            black_engine_id="scored",
        )
        gui._handle_action("heat")
        self.assertTrue(gui.heatmap_enabled)
        self.assertTrue(gui.heat_action_scores)
        self.assertTrue(gui.heat_ranks)
        self.assertEqual(gui.heat_source, scored.name)
        self.assertIsNone(gui.selected_pos)
        gui._handle_action("engine_black")
        self.assertEqual(gui.black_engine_id, "random")
        self.assertIs(gui.black_bot, random_bot)
        gui.run(max_frames=1)

    def test_layout_editor_swaps_only_rule_conformant_pieces_and_round_trips(self):
        gui = JunqiWorkbench(HumanBot(), RandomBot())
        original = list(gui.env.get_perfect_state())
        self.assertEqual(_deployment_errors(original), [])
        gui.toggle_layout_edit()
        self.assertTrue(gui.layout_edit_mode)

        valid_pair = None
        occupied = [index for index in range(30) if original[index] > 0]
        for first in occupied:
            for second in occupied:
                if first >= second or original[first] == original[second]:
                    continue
                candidate = original.copy()
                candidate[first], candidate[second] = candidate[second], candidate[first]
                if not _deployment_errors(candidate):
                    valid_pair = (first, second, candidate)
                    break
            if valid_pair:
                break
        self.assertIsNotNone(valid_pair)
        first, second, expected = valid_pair
        gui._select_or_swap_layout(first)
        gui._select_or_swap_layout(second)
        self.assertEqual(list(gui.env.get_perfect_state()), expected)

        saved = gui.save_layout()
        self.assertIsNotNone(saved)
        try:
            gui._select_or_swap_layout(first)
            gui._select_or_swap_layout(second)
            gui.load_layout(saved)
            self.assertEqual(list(gui.env.get_perfect_state()), expected)
        finally:
            saved.unlink(missing_ok=True)
        gui.run(max_frames=1)


if __name__ == "__main__":
    unittest.main()
