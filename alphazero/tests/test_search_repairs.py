from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import torch

import JunqiAZ
from junqi_az.config import ModelConfig
from junqi_az.model import PolicyValueNet
from junqi_fixed_rl.layout import FixedSameLayoutBackend, load_fixed_layout
from junqi_search_v3.loop import SearchSample
from junqi_search_v3.recovery import _anchor_kl, make_policy_anchor, _paired_bootstrap
from junqi_search_v3.search import BatchedNeuralPUCT

ROOT = Path(__file__).resolve().parents[1]


class SearchRepairTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(13)
        self.model = PolicyValueNet(ModelConfig(input_channels=107, channels=16, residual_blocks=0,
            policy_channels=8, value_head_type="wdl", repetition_context=True, pairwise_action_bias=True))
        self.backend = FixedSameLayoutBackend(load_fixed_layout(
            ROOT / "config/layouts/fixed_layout_v1.json", "public_conservative_advance", project_root=ROOT))

    def test_bulk_backup_is_atomic_and_cancellable(self):
        state = self.backend.new_state(1)
        tree = JunqiAZ.NeuralPUCTTree(state.native_state)
        root = tree.select_leaves(4)
        self.assertEqual(len(root), 1)
        tree.backup_leaves([[1.] * len(root[0]["legal_actions"])], [0.])
        leaves = tree.select_leaves(4)
        self.assertEqual(len(leaves), 4)
        priors = [[1.] * len(row["legal_actions"]) for row in leaves]
        with self.assertRaises(ValueError):
            tree.backup_leaves(priors, [0., 0., float("nan"), 0.])
        self.assertEqual(tree.root_visits(), 1)
        self.assertTrue(tree.has_pending_leaf())
        tree.cancel_pending()
        again = tree.select_leaves(4)
        self.assertEqual([r["state_key"] for r in leaves], [r["state_key"] for r in again])
        tree.backup_leaves(priors, [0.] * 4)
        self.assertEqual(tree.root_visits(), 5)
        self.assertEqual(sum(r["visits"] for r in tree.root_edges()), 4)

    def test_cached_sequential_search_preserves_visits_and_invalidates_weights(self):
        states = [self.backend.new_state(1)] * 3
        baseline = BatchedNeuralPUCT(self.model, device=torch.device("cpu"))
        cached = BatchedNeuralPUCT(self.model, device=torch.device("cpu"), cache_size=64)
        before, _ = baseline.search(states, 12)
        after, metrics = cached.search(states, 12)
        for a, b in zip(before, after):
            np.testing.assert_array_equal(a.visits, b.visits)
        self.assertGreater(metrics["deduplicated_rows"], 0)
        _, warm = cached.search(states, 12)
        self.assertEqual(warm["inference_rows"], 0)
        with torch.no_grad():
            self.model.action_bias.add_(torch.arange(3600) / 1000.)
        _, changed = cached.search(states, 12)
        self.assertGreater(changed["inference_rows"], 0)

    def test_reused_root_retains_visits_and_full_history(self):
        state = self.backend.new_state(2)
        search = BatchedNeuralPUCT(self.model, device=torch.device("cpu"), reuse_roots=True)
        result, _ = search.search([state], 20, tree_ids=["game"])
        state.apply(result[0].action)
        self.assertTrue(search.advance_tree("game", result[0].action, state))
        _, metrics = search.search([state], 8, tree_ids=["game"])
        self.assertEqual(metrics["reused_roots"], 1)
        self.assertGreater(metrics["inherited_root_visits"], 0)
        _, reset = search.search([self.backend.new_state(2)], 8, tree_ids=["game"])
        self.assertEqual(reset["reused_roots"], 0)

    def test_frozen_anchor_has_restoring_gradient_after_encoder_update(self):
        state = self.backend.new_state(3)
        legal = state.legal_actions()
        sample = SearchSample(1, 1, state.feature_planes(), legal, state.repetition_contexts(),
                              np.ones(len(legal), np.int32), 1, 0, 0., wdl=np.array([0.,1.,0.],np.float32))
        anchor = make_policy_anchor(self.model, [sample], torch.device("cpu"))
        reference = [p.clone() for p in anchor.probabilities]
        with torch.no_grad():
            next(self.model.encoder.parameters()).add_(torch.randn_like(next(self.model.encoder.parameters())) * .02)
        self.model.set_training_phase("value_distill")
        loss = _anchor_kl(self.model, anchor.samples, anchor.probabilities, torch.device("cpu"))
        loss.backward()
        self.assertGreater(float(loss), 1e-5)
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.encoder_parameters()), 1e-3)
        for a, b in zip(reference, anchor.probabilities):
            torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_uncertain_practical_regression_is_not_statistical_proof(self):
        before = {i: {"policy_excess": 1.} for i in range(8)}
        after = {i: {"policy_excess": 1.15 if i < 4 else .89} for i in range(8)}
        result = _paired_bootstrap(before, after, "policy_excess", 12)
        self.assertTrue(result["practical_regression"])
        self.assertEqual(result["classification"], "inconclusive")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA graph test needs CUDA")
    def test_cuda_graphs_preserve_search_and_invalidate_after_weight_update(self):
        model = self.model.cuda().eval()
        plain = BatchedNeuralPUCT(model, device=torch.device("cuda"), cuda_graphs=False)
        graphs = BatchedNeuralPUCT(model, device=torch.device("cuda"), cuda_graphs=True)
        for count in (1, 2, 4, 8):
            states = [self.backend.new_state(index) for index in range(count)]
            a, unused = plain.search(states, 8)
            b, metrics = graphs.search(states, 8)
            for first, second in zip(a, b):
                np.testing.assert_array_equal(first.visits, second.visits)
                np.testing.assert_array_equal(first.q_values, second.q_values)
            self.assertEqual(metrics["cuda_graph_failed_shapes"], [])
            self.assertGreater(metrics["cuda_graph_replays"], 0)
        old = graphs._graphs[1][0]
        with torch.no_grad():
            model.action_bias.add_(torch.arange(3600, device="cuda") / 1000.)
        state = self.backend.new_state(2)
        a, unused = plain.search([state], 8)
        b, metrics = graphs.search([state], 8)
        np.testing.assert_array_equal(a[0].visits, b[0].visits)
        self.assertIsNot(graphs._graphs[1][0], old)


if __name__ == "__main__":
    unittest.main()
