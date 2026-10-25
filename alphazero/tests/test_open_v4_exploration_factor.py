import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import JunqiAZ

from junqi_az.model import PolicyValueNet
from junqi_open_v4.layouts import create_manifest, load_role, backend_for
from junqi_open_v4.learner import model_hash
from junqi_search_v3.search import BatchedNeuralPUCT
from run_open_v4_exploration_factor import validate_plan
from run_open_v4_search_training import route_signal
from test_open_v4 import small_config


class ExplorationFactorTests(unittest.TestCase):
    def test_rejects_weight_budget_or_adoption_changes(self):
        parent = dict(path='parent.pt', sha256='a', model_hash='b')
        legacy = dict(path='legacy.pt', sha256='c', model_hash='d')
        plan = dict(experiment_type='frozen_puct_exploration_factor',
            search_variants={n: dict(cpuct=c, simulations=256) for n, c in
                [('parent', 1.5), ('control', 1.5), ('candidate', .75), ('repeat', 1.5)]},
            participants={'parent': parent, 'control': legacy, 'candidate': parent, 'repeat': parent},
            stronger_legacy_reference=legacy, screen_families=16, full_families=64, diagnostic_positions=48,
            training_updates=0, new_training_games=0, automatic_adoption=False)
        self.assertTrue(validate_plan(plan))
        for mutation in ('budget', 'weights', 'adoption', 'reference'):
            changed = copy.deepcopy(plan)
            if mutation == 'budget': changed['search_variants']['candidate']['simulations'] = 64
            if mutation == 'weights': changed['participants']['candidate'] = legacy
            if mutation == 'adoption': changed['automatic_adoption'] = True
            if mutation == 'reference': changed['participants']['control'] = parent
            with self.assertRaises(ValueError): validate_plan(changed)

    def test_actual_native_parameter_budget_and_history_are_preserved(self):
        torch.set_num_threads(1); torch.manual_seed(710)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); create_manifest(root, 710, train=2, dev=2, final=2)
            state = backend_for(*load_role(root, 'train')).new_state(13)
            # Search a full-history successor, rather than only a fresh board.
            for _ in range(4): state.apply(int(state.legal_actions()[0]))
            before = (int(state.native_state.key()), state.feature_planes().copy(), state.repetition_contexts().copy())
            model = PolicyValueNet(small_config().model).eval(); digest = model_hash(model)
            native = JunqiAZ.NeuralPUCTTree; seen = []
            def capture(root_state, cpuct):
                seen.append(cpuct); return native(root_state, cpuct)
            with patch('junqi_search_v3.search.JunqiAZ.NeuralPUCTTree', side_effect=capture):
                for coefficient in (1.5, .75):
                    search = BatchedNeuralPUCT(model, device=torch.device('cpu'), cpuct=coefficient)
                    results, metrics = search.search([state], 16)
                    self.assertEqual(results[0].simulations, 16)
                    self.assertEqual(int(results[0].visits.sum()), 15)
                    self.assertIn(results[0].action, state.legal_actions())
                    self.assertGreater(metrics['inference_calls'], 0)
            self.assertEqual(seen, [1.5, .75])
            self.assertEqual(model_hash(model), digest)
            self.assertEqual(int(state.native_state.key()), before[0])
            np.testing.assert_array_equal(state.feature_planes(), before[1])
            np.testing.assert_array_equal(state.repetition_contexts(), before[2])

    def test_fixed16_positive_result_still_cannot_qualify(self):
        gain = dict(mean=.1, ci95=[.01, .2], families=16)
        duel = dict(mean=.6, ci95=[.51, .7], families=16)
        repeat = dict(mean=0, ci95=[-.01, .01], families=16)
        self.assertFalse(route_signal([[gain, gain], [gain, gain]], [repeat, repeat], [duel, duel],
            {'champion_accepted': True}, 16, .02))


if __name__ == '__main__': unittest.main()
