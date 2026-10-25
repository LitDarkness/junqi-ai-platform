import unittest
from pathlib import Path
import copy
import torch
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_open_v4.training_replay import make_self_training_learner
from junqi_az.config import ModelConfig
from open_v4_mechanism_reversion_contract import fork_stable_payload,small_screen_review,validate_source_config


class ReversionTests(unittest.TestCase):
    def test_original_phase_rates(self):
        cfg=RootExploreConfig(device='cpu',initial_checkpoint=None,model=ModelConfig(input_channels=107,channels=8,residual_blocks=1,policy_channels=4,value_head_type='wdl',repetition_context=True,pairwise_action_bias=True))
        learner=make_self_training_learner(cfg,Path('.'))
        learner._set_phase('policy');self.assertEqual({g['name']:g['lr'] for g in learner.optimizer.param_groups},dict(encoder=1.25e-5,policy=5e-5,value=0.))
        learner._set_phase('value');self.assertEqual({g['name']:g['lr'] for g in learner.optimizer.param_groups},dict(encoder=5e-6,policy=0.,value=5e-5))

    def test_actual_current_weights_optimizer_and_rng_preserved(self):
        root=Path(__file__).resolve().parents[1]
        path=root/'runs/open_junqi_v4_stable_policy_20261005_110305/learner/checkpoints/learner_stage_000006.pt'
        source=torch.load(path,map_location='cpu',weights_only=False)
        data=copy.deepcopy(source['config']);data.pop('policy_step_lr_multiplier');data['updates_per_iteration']=64
        # Use the public dataclass loader to keep nested model typing exact.
        from dataclasses import fields
        cfg=RootExploreConfig(**{k:v for k,v in data.items() if k!='model'},model=ModelConfig(**data['model']))
        cfg.validate();result=fork_stable_payload(source,cfg,{'user_mechanism_reversion':True})
        self.assertEqual(result['model_hash'],source['model_hash'])
        self.assertEqual(result['counters'],source['counters'])
        self.assertEqual(result['config']['updates_per_iteration'],64)
        self.assertEqual(result['optimizer']['param_groups'][1]['lr'],5e-5)
        bad=copy.deepcopy(cfg);bad.simulations=128
        with self.assertRaises(ValueError):validate_source_config(source,bad)

    def test_point_decline_does_not_wait_for_significance(self):
        games=[]
        for family in range(16):
            for matchup,score in enumerate((.5,.5,.5+(-.5 if family==0 else 0),.5)):
                games.append(dict(family=family,matchup=matchup,score=score))
        result=small_screen_review(dict(role='dev',summaries=[{}]*4,games=games),9,1-.05/6,.5)
        self.assertLess(result['paired']['mean'],0)
        self.assertGreaterEqual(result['paired95']['interval'][1],0)
        self.assertTrue(result['early_stop'])


if __name__=='__main__':unittest.main()
