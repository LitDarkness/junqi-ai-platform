import copy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'build-experiments/93c0594487ee8e5defbfdc0b0b883a43917f354fb24359808304aa76fc6ce504'))
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_open_v4 import recipe50_games
from open_v4_recipe50_to800_contract import validate_source_config, fork_stable_payload, verify_stable_seed
from run_open_v4_recipe50_to800 import confirmed_route

SOURCE = ROOT / 'runs/open_junqi_v4_recipe50_to600_20261007_034300/learner/checkpoints/learner_stage_000176.pt'


class RecipeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.source = torch.load(SOURCE, map_location='cpu', weights_only=False)

    def config(self):
        cfg = RootExploreConfig.load(SOURCE.parents[1] / 'config.json')
        cfg.run_directory = 'runs/recipe50_test/learner'
        cfg.initial_checkpoint = str(SOURCE.relative_to(ROOT))
        cfg.selfplay_fraction = .5
        cfg.teacher_selfplay_fraction = 0.
        return cfg

    def test_complete600_learning_state_is_exactly_preserved(self):
        cfg = self.config()
        cfg.validate()
        self.assertEqual(self.source['counters']['iteration'], 600)
        migrated = fork_stable_payload(self.source, cfg, {'authorized_recipe': True})
        self.assertTrue(all(verify_stable_seed(self.source, migrated).values()))
        self.assertEqual(set(validate_source_config(self.source, cfg)),
                         {'run_directory', 'initial_checkpoint'})
        self.assertFalse(torch.cuda.is_initialized())

    def test_unapproved_rate_search_ratio_label_or_replay_change_rejected(self):
        for field, value in [('learning_rate', .0001), ('simulations', 128),
                             ('updates_per_iteration', 256), ('replay_iterations', 4),
                             ('actor_policy_target', 'search'), ('workers', 6)]:
            cfg = self.config()
            setattr(cfg, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_source_config(self.source, cfg)

    def test_real_collector_first_requests_only082_on_seeded256_schedule(self):
        cfg = self.config()
        for iteration in (1542, 1543, 1544):
            expected = sum(np.random.default_rng(cfg.seed ^ (iteration * 1000000 + i // 2)).random() >= .5
                           for i in range(256))
            self.assertGreater(expected, 0)
            self.assertLess(expected, 256)
            class EndProbe(Exception): pass
            calls = []
            def submit(states, ids):
                calls.append((len(states), list(ids)))
                raise EndProbe()
            teachers = SimpleNamespace(modules=[object(), object()], submit_actions=submit)
            backend = SimpleNamespace(new_state=lambda seed: SimpleNamespace(provenance=lambda: {'setup_seed': seed}, current_player=lambda: 1))
            with patch.object(recipe50_games, 'RootExploreSearch', return_value=object()), self.assertRaises(EndProbe):
                recipe50_games.collect_training_games(cfg, None, 'cpu', backend, teachers, iteration)
            self.assertEqual(calls, [(expected, [0] * expected)])
        for key, expected in [(0., 'selfplay'), (.499999, 'selfplay'), (.5, 'teacher_vs_actor'), (.999999, 'teacher_vs_actor')]:
            self.assertEqual(recipe50_games.training_mode(cfg, key), expected)

    def test_original_strict_formal_protections_not_weakened(self):
        row = dict(mean=.1, ci95=[.01, .2], families=64)
        gains = [[copy.deepcopy(row), copy.deepcopy(row)] for _ in range(4)]
        duplicate = [dict(mean=0., families=64)] * 2
        duels = [dict(mean=.6, ci95=[.51, .7], families=64)] * 4
        gate = {'champion_accepted': True}
        self.assertTrue(confirmed_route(gains, duplicate, duels, gate, 64, .02))
        gains[0][1]['ci95'][0] = 0.
        self.assertFalse(confirmed_route(gains, duplicate, duels, gate, 64, .02))
        gains[0][1]['ci95'][0] = .01
        duels[2]['ci95'][0] = .5
        self.assertFalse(confirmed_route(gains, duplicate, duels, gate, 64, .02))
        duels[2]['ci95'][0] = .51
        self.assertFalse(confirmed_route(gains, duplicate, duels, gate, 16, .02))
        self.assertFalse(confirmed_route(gains, [dict(mean=.02001, families=64)] * 2, duels, gate, 64, .02))


    def test_horizon_preserves400_qualification_and_monitors084(self):
        import json
        from open_v4_recipe50_to800_contract import check_single_parameters
        p=json.loads((ROOT/'config/experiments/open_v4_recipe50_20261007_012129.json').read_text(encoding='utf-8'))
        p.update(schema='junqi-recipe50-600to800-plan-1',route='recipe50_selfplay_vs082_to800',source_checkpoint=dict(iteration=600),target_iteration=800,maximum_stages=200,screen_boundaries=[6]+list(range(12,201,12))+[200],monitor084_boundaries=list(range(24,201,24))+[200],qualification_reference400=copy.deepcopy(p['participants']['parent']))
        check_single_parameters(p)
        for field,value in [('target_iteration',801),('maximum_stages',201),('monitor084_boundaries',[200])]:
            bad=copy.deepcopy(p);bad[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):check_single_parameters(bad)
        bad=copy.deepcopy(p);bad['participants']['parent']=dict(iteration=600)
        with self.assertRaises(ValueError):check_single_parameters(bad)

    def test_saved084_monitor_is_independently_reconstructed(self):
        import json
        from open_v4_monitor_evidence import reconstruct_monitor
        from junqi_open_v4.arena import select_families
        p=json.loads((ROOT/'config/experiments/open_v4_recipe50_20261007_012129.json').read_text(encoding='utf-8'))
        arena=json.loads((ROOT/p['directory']/'arena/iteration424_true084.json').read_text(encoding='utf-8'))
        selected=select_families(json.loads((ROOT/p['suite']['path']).read_text(encoding='utf-8'))['states'],16,p['evaluation_seed'])
        r=reconstruct_monitor(arena,selected,p,ROOT)
        self.assertTrue(r['all_checks_passed']);self.assertTrue(r['monitor_only']);self.assertFalse(r['promotion'])
        self.assertEqual(r['native_verification'],dict(games=258,native_plies=9879))
        self.assertLess(r['paired_deltas'][0]['delta']['ci95'][1],0)


if __name__ == '__main__': unittest.main()
