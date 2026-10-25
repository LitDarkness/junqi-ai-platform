"""Meaningful phase/migration and prospective-stop regressions; no neural training."""
import copy
import unittest
from pathlib import Path
import numpy as np
import torch
from junqi_az.config import ModelConfig
from junqi_open_v4.stable_policy_config import StablePolicyConfig
from junqi_open_v4.stable_policy_learner import make_stable_policy_learner
from open_v4_stable_policy_contract import fork_stable_payload,validate_source_config,continuation_action,small_screen_review


class StablePolicyTests(unittest.TestCase):
    def test_real_factory_phase_and_backtracking_rates(self):
        cfg=StablePolicyConfig(device='cpu',initial_checkpoint=None,model=ModelConfig(input_channels=107,channels=8,residual_blocks=1,policy_channels=4,value_head_type='wdl',repetition_context=True,pairwise_action_bias=True))
        learner=make_stable_policy_learner(cfg,Path('.'))
        def rates(phase,scale=1.):
            learner._set_phase(phase,scale)
            return {g['name']:g['lr'] for g in learner.optimizer.param_groups}
        self.assertEqual(rates('policy'),dict(encoder=3.125e-6,policy=1.25e-5,value=0.))
        self.assertEqual(rates('value'),dict(encoder=5e-6,policy=0.,value=5e-5))
        self.assertEqual(rates('policy'),dict(encoder=3.125e-6,policy=1.25e-5,value=0.))
        self.assertEqual(rates('value',.5),dict(encoder=2.5e-6,policy=0.,value=2.5e-5))

    def test_full_seed_migration_and_extra_factor_rejection(self):
        cfg=StablePolicyConfig(device='cpu',initial_checkpoint=None,model=ModelConfig(input_channels=107,channels=8,residual_blocks=1,policy_channels=4,value_head_type='wdl',repetition_context=True,pairwise_action_bias=True))
        learner=make_stable_policy_learner(cfg,Path('.'))
        source=learner.payload(role='learner',lineage={})
        source['config'].pop('policy_step_lr_multiplier');source['counters']['attempted_updates']=10112
        source['optimizer']['param_groups'][0]['lr']=1.25e-5
        source['optimizer']['param_groups'][1]['lr']=5e-5
        source['optimizer']['param_groups'][2]['lr']=0.
        # A real stored Adam state must survive, not just empty metadata.
        source['optimizer']['state'][0]=dict(step=torch.tensor(7.),exp_avg=torch.tensor([.123]),exp_avg_sq=torch.tensor([.456]))
        cfg.updates_per_iteration=256
        result=fork_stable_payload(source,cfg,{'test':True})
        self.assertEqual(result['optimizer']['param_groups'][0]['lr'],3.125e-6)
        self.assertEqual(result['optimizer']['param_groups'][1]['lr'],1.25e-5)
        self.assertTrue(torch.equal(result['optimizer']['state'][0]['exp_avg'],source['optimizer']['state'][0]['exp_avg']))
        self.assertEqual(result['scheduler'],source['scheduler'])
        bad=copy.deepcopy(cfg);bad.value_encoder_lr_multiplier=.2
        with self.assertRaises(ValueError):validate_source_config(source,bad)

    def test_external_early_stop_does_not_require_internal_collapse(self):
        def arena(delta):
            games=[]
            for family in range(16):
                for matchup,score in enumerate((.5,.75,.75+delta,.75)):
                    games.append(dict(family=family,matchup=matchup,score=score))
            return dict(role='dev',summaries=[{}]*4,games=games)
        negative=small_screen_review(arena(-.125),32,1-.05/6)
        self.assertTrue(negative['early_stop'])
        self.assertEqual(negative['early_stop_reason'],'external_negative')
        self.assertFalse(small_screen_review(arena(0),32,1-.05/6)['early_stop'])
        self.assertEqual(continuation_action(6,False,False,24),'continue')
        self.assertEqual(continuation_action(12,False,False,24),'review_due')
        self.assertEqual(continuation_action(12,True,False,24),'auto_next_segment')
        self.assertEqual(continuation_action(6,False,True,24),'review_due')
        self.assertEqual(continuation_action(24,True,False,24),'segment_complete')

    def test_saved_negative_arena_recomputes_correct_paired_direction(self):
        import json
        path=Path(__file__).resolve().parents[1]/'runs/open_junqi_v4_single_ratio_20261005_070831/arena/block_000024_screen.json'
        if not path.exists():self.skipTest('local saved research evidence unavailable')
        old=json.loads(path.read_text(encoding='utf-8'))
        wanted=[('candidate','parent'),('parent','teacher0'),('candidate','teacher0'),('repeat','teacher0')]
        indices=[next(i for i,r in enumerate(old['summaries']) if (r['candidate'],r['opponent'])==pair) for pair in wanted]
        mapping={n:i for i,n in enumerate(indices)}
        arena=dict(role='dev',summaries=[old['summaries'][n] for n in indices],games=[dict(r,matchup=mapping[r['matchup']]) for r in old['games'] if r['matchup'] in mapping])
        review=small_screen_review(arena,2026100307,1-.05/6)
        self.assertAlmostEqual(review['paired']['mean'],-.1640625)
        self.assertTrue(review['source_drift_ok'])
        self.assertEqual(len(arena['games']),344)
        self.assertLess(review['paired95']['interval'][1],0)
        # Accounting for both endpoints and all three possible looks is
        # deliberately more conservative than the descriptive95% result.
        self.assertGreaterEqual(review['paired']['interval'][1],0)
        self.assertFalse(review['early_stop'])


if __name__=='__main__':unittest.main()
