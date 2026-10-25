from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

import JunqiAZ
from junqi_az.backend import CppAZStateAdapter
from junqi_az.config import ModelConfig
from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout
from junqi_search_v3.search import BatchedNeuralPUCT


ROOT = Path(__file__).resolve().parents[1]


class SearchV3PUCTTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        layout = load_fixed_layout(
            ROOT / "config/layouts/fixed_layout_v1.json",
            "public_conservative_advance",
            project_root=ROOT,
        )
        cls.backend = FixedSameLayoutBackend(layout)

    def test_selection_expansion_and_root_visits(self) -> None:
        state = self.backend.new_state(1)
        tree = JunqiAZ.NeuralPUCTTree(state.native_state, 1.5)
        root = tree.select_leaf()
        self.assertFalse(root["terminal"])
        legal = root["legal_actions"]
        tree.backup([1.0 / len(legal)] * len(legal), 0.25)
        self.assertEqual(tree.root_visits(), 1)
        child = tree.select_leaf()
        self.assertFalse(child["terminal"])
        child_legal = child["legal_actions"]
        tree.backup([1.0 / len(child_legal)] * len(child_legal), -0.5)
        self.assertEqual(tree.root_visits(), 2)
        self.assertEqual(sum(row["visits"] for row in tree.root_edges()), 1)

    def test_nonterminal_leaf_value_flips_to_parent_loss(self) -> None:
        state = self.backend.new_state(8)
        tree = JunqiAZ.NeuralPUCTTree(state.native_state, 1.5)
        root = tree.select_leaf()
        legal = list(root["legal_actions"])
        priors = [1.0 if index == 0 else 0.0 for index in range(len(legal))]
        tree.backup(priors, 0.0)
        child = tree.select_leaf()
        self.assertEqual(child["player"], -state.current_player())
        child_legal = list(child["legal_actions"])
        tree.backup([1.0 / len(child_legal)] * len(child_legal), 1.0)
        edge = next(row for row in tree.root_edges() if row["action"] == legal[0])
        self.assertEqual(edge["q"], -1.0)

    def test_independent_dirichlet_noise_is_normalized(self) -> None:
        state = self.backend.new_state(9)
        trees = [
            JunqiAZ.NeuralPUCTTree(state.native_state, 1.5)
            for _ in range(2)
        ]
        priors_after = []
        for seed, tree in enumerate(trees, start=1):
            root = tree.select_leaf()
            count = len(root["legal_actions"])
            tree.backup([1.0 / count] * count, 0.0)
            tree.add_root_dirichlet(0.3, 0.25, seed)
            priors_after.append(
                np.asarray([row["prior"] for row in tree.root_edges()])
            )
        self.assertAlmostEqual(float(priors_after[0].sum()), 1.0, places=6)
        self.assertFalse(np.allclose(priors_after[0], priors_after[1]))

    def test_forced_flag_win_backup_has_positive_root_q(self) -> None:
        raw = JunqiAZ.JunqiAZState()
        board = [0] * 60
        board[1] = 3
        board[51] = 11
        board[56] = -3
        raw.load_board_state(board, 1)
        tree = JunqiAZ.NeuralPUCTTree(raw, 1.5)
        request = tree.select_leaf()
        legal = list(request["legal_actions"])
        capture = 51 * 60 + 56
        priors = [1.0 if action == capture else 0.0 for action in legal]
        tree.backup(priors, 0.0)
        terminal = tree.select_leaf()
        self.assertTrue(terminal["terminal"])
        self.assertEqual(terminal["terminal_value"], 1.0)
        tree.backup([], terminal["terminal_value"])
        edge = next(row for row in tree.root_edges() if row["action"] == capture)
        self.assertEqual(edge["visits"], 1)
        self.assertEqual(edge["q"], 1.0)

    def test_black_view_and_state_keys(self) -> None:
        state = self.backend.new_state(2)
        state.apply(int(state.legal_actions()[0]))
        self.assertEqual(state.current_player(), -1)
        tree = JunqiAZ.NeuralPUCTTree(state.native_state, 1.5)
        request = tree.select_leaf()
        self.assertEqual(request["player"], -1)
        self.assertEqual(request["state_key"], state.native_state.key())
        self.assertEqual(
            request["position_key"], state.native_state.position_key()
        )

    def test_repetition_history_is_present_in_transposition_key(self) -> None:
        initial = self.backend.new_state(1)
        repeated = initial.clone()
        for action in (311, 311, 665, 665):
            self.assertIn(action, repeated.legal_actions())
            repeated.apply(action)
        self.assertEqual(
            repeated.native_state.position_key(),
            initial.native_state.position_key(),
        )
        self.assertNotEqual(
            repeated.native_state.key(), initial.native_state.key()
        )
        self.assertEqual(repeated.native_state.repetition_count(), 2)
        request = JunqiAZ.NeuralPUCTTree(
            repeated.native_state, 1.5
        ).select_leaf()
        self.assertEqual(request["state_key"], repeated.native_state.key())

    def test_batched_inference_matches_identical_roots(self) -> None:
        torch.manual_seed(3)
        config = ModelConfig(
            input_channels=107,
            channels=32,
            residual_blocks=1,
            policy_channels=16,
            value_head_type="wdl",
            repetition_context=True,
            pairwise_action_bias=True,
        )
        model = PolicyValueNet(config).eval()
        model.search_v3_leaf_value_scale = 0.25
        states = [self.backend.new_state(4), self.backend.new_state(4)]
        search = BatchedNeuralPUCT(
            model, device=torch.device("cpu"), cpuct=1.5
        )
        results, metrics = search.search(states, 8)
        self.assertEqual(results[0].action, results[1].action)
        np.testing.assert_array_equal(results[0].visits, results[1].visits)
        self.assertEqual(metrics["mean_inference_batch"], 2.0)
        self.assertEqual(metrics["leaf_value_scale"], 0.25)


if __name__ == "__main__":
    unittest.main()
