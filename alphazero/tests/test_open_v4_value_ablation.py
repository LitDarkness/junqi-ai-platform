import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from evaluate_open_v4_value_ablation import ScaledSearch


class FakeSearch:
    reuse_roots = False
    def search(self, states, simulations, **kwargs):
        return states, simulations, kwargs
    def clear_trees(self):
        return "cleared"
    def bare(self, states):
        return states


class ValueAblationTests(unittest.TestCase):
    def test_scale_override_preserves_budget_and_root_ids(self):
        wrapper = ScaledSearch(FakeSearch(), .25)
        states, budget, options = wrapper.search(["state"], 64, tree_ids=[5], add_root_noise=False)
        self.assertEqual((states, budget), (["state"], 64))
        self.assertEqual(options, {"leaf_value_scale": .25, "tree_ids": [5], "add_root_noise": False})

    def test_bare_and_tree_operations_delegate(self):
        wrapper = ScaledSearch(FakeSearch(), 0.)
        self.assertFalse(wrapper.reuse_roots)
        self.assertEqual(wrapper.clear_trees(), "cleared")
        self.assertEqual(wrapper.bare(["state"]), ["state"])
