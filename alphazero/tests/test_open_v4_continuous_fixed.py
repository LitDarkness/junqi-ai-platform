import copy
import unittest
from pathlib import Path
import torch
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_az.config import ModelConfig
from open_v4_continuous_fixed_contract import (
    fork_stable_payload,verify_stable_seed,small_screen_review,
    continuation_action,is_screen_stage,validate_source_config,
)


class ContinuousFixedTests(unittest.TestCase):
    def test_negative_uncertain_or_confirmed_strength_never_stops(self):
        for stage in (6,12,24,36,120,10000):
            for confirmed in (False,True):
                for negative in (False,True):
                    self.assertEqual(continuation_action(stage,confirmed,negative), 'continue_monitor_only')
        games=[]
        for family in range(16):
            for matchup,score in enumerate((0.2,0.7,0.1,0.7)):
                games.append(dict(family=family,matchup=matchup,score=score))
        result=small_screen_review(dict(role='dev',summaries=[{}]*4,games=games),9,1-.05/6,.7)
        self.assertLess(result['paired']['mean'],0)
        self.assertTrue(result['observed_negative_external_point'])
        self.assertTrue(result['historical_stop_would_have_fired'])
        self.assertFalse(result['early_stop'])
        self.assertFalse(result['positive_signal'])

    def test_screen_schedule_continues_after_old_horizon(self):
        p=dict(screen_boundaries=[6,12,24],periodic_screen_stages=12)
        self.assertEqual([i for i in range(1,73) if is_screen_stage(i,p)],[6,12,24,36,48,60,72])

    def test_continuation_does_not_relax_formal_adoption(self):
        from run_open_v4_continuous_fixed import confirmed_route
        positive=dict(families=64,mean=.1,ci95=[.01,.2])
        gains=[[copy.deepcopy(positive) for _ in range(2)] for _ in range(4)]
        duplicate=[dict(families=64,mean=0.,ci95=[0.,0.]) for _ in range(2)]
        duels=[dict(families=64,mean=.6,ci95=[.51,.69]) for _ in range(4)]
        gate=dict(champion_accepted=True)
        self.assertTrue(confirmed_route(gains,duplicate,duels,gate,64,.02))
        self.assertFalse(confirmed_route(gains,duplicate,duels,gate,16,.02))
        self.assertFalse(confirmed_route(gains,duplicate,duels,dict(champion_accepted=False),64,.02))
        duels[0]['ci95'][0]=.49
        self.assertFalse(confirmed_route(gains,duplicate,duels,gate,64,.02))
        duels[0]['ci95'][0]=.51
        gains[3][1]['ci95'][0]=-.01
        self.assertFalse(confirmed_route(gains,duplicate,duels,gate,64,.02))

    def test_latest_full170_weights_optimizer_counters_rng_exact(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'runs/open_junqi_v4_mechanism_reversion_20261005_113636/learner/checkpoints/learner_stage_000006.pt'
        source=torch.load(path,map_location='cpu',weights_only=False)
        data=copy.deepcopy(source['config'])
        cfg=RootExploreConfig(**{k:v for k,v in data.items() if k!='model'},model=ModelConfig(**data['model']))
        cfg.run_directory='runs/continuous_fixed_contract_test/learner'
        cfg.initial_checkpoint=str(path.relative_to(root))
        cfg.validate()
        result=fork_stable_payload(source,cfg,{'user_continuous_authorization':True})
        self.assertTrue(all(verify_stable_seed(source,result).values()))
        self.assertEqual(source['optimizer']['param_groups'],result['optimizer']['param_groups'])
        self.assertEqual(source['counters'],result['counters'])
        self.assertEqual(source['model_hash'],result['model_hash'])
        changed={k for k in source['config'] if source['config'][k]!=result['config'][k]}
        self.assertEqual(changed,{'run_directory','initial_checkpoint'})
        bad=copy.deepcopy(cfg);bad.updates_per_iteration=256
        with self.assertRaises(ValueError):validate_source_config(source,bad)
        bad=copy.deepcopy(cfg);bad.learning_rate*=.25
        with self.assertRaises(ValueError):validate_source_config(source,bad)


if __name__=='__main__':unittest.main()
