"""Validate update clocks, exact initial RNG and unchanged protected gates."""
import unittest,copy,tempfile
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import numpy as np,torch
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_az.config import ModelConfig
from junqi_open_v4.training_replay import make_self_training_learner
from fork_open_v4_qualified import fork_payload
from run_open_v4_search_training import verify_learning_state
from open_v4_update_ratio_contract import check_factor_parameters,exposure_summary
from run_open_v4_update_ratio import confirmed_route,severe_regression

def plan():
    branch={'selfplay_fraction':.25,'teacher_selfplay_fraction':None,'actor_policy_target':'teacher','root_tactical_guard':False,
        'training_greedy_after_ply':24,'root_noise_until_ply':None,'forced_root_coeff':2.,'exploration_target_pruning':False,'deduplicate':False,'neural_policy_only':False}
    return {'control':dict(branch,updates_per_stage=64),'candidate':dict(branch,updates_per_stage=256),'source_checkpoint':{'iteration':134},'participants':{'guard86':{}},
        'updates_per_stage':{'control':64,'candidate':256},'registered_updates':7680,'maximum_stages':24,'stages_per_block':12,'games_per_stage':256,'replay_stages':3,'collection_start':1000}

class RatioTests(unittest.TestCase):
    def test_single_factor_and_counts_reject_hidden_changes(self):
        p=plan();p['participants']['guard86']={'iteration':86};check_factor_parameters(p)
        for field,value in [('forced_root_coeff',0.),('exploration_target_pruning',True),('actor_policy_target','search'),('updates_per_stage',128),('deduplicate',True)]:
            q=copy.deepcopy(p);q['candidate'][field]=value
            with self.assertRaises(ValueError):check_factor_parameters(q)
        q=copy.deepcopy(p);q['registered_updates']=3072
        with self.assertRaises(ValueError):check_factor_parameters(q)

    def test_config256_roundtrip_and_complete_state_fork(self):
        torch.set_num_threads(2)
        cfg=RootExploreConfig(device='cpu',model=ModelConfig(channels=16,residual_blocks=1,policy_channels=8),updates_per_iteration=64)
        with tempfile.TemporaryDirectory() as temp:
            learner=make_self_training_learner(cfg,Path(temp));learner.counters.update(iteration=134,attempted_updates=8576,effective_updates=8576,policy_updates=5718,value_updates=2858)
            learner.scheduler['effective_updates']=8576
            original=learner.payload(role='learner',lineage={});newcfg=copy.deepcopy(cfg);newcfg.updates_per_iteration=256;newcfg.run_directory='new'
            newcfg.validate();path=Path(temp)/'config.json';newcfg.save(path)
            self.assertEqual(RootExploreConfig.load(path).updates_per_iteration,256)
            migrated=fork_payload(original,newcfg,{'optimizer_reset':False});self.assertTrue(all(verify_learning_state(original,migrated).values()))
            self.assertEqual(migrated['leaf_value_scale'],original['leaf_value_scale'])

    def test_same_initial64_batches_then_fourfold_ppv_clock_rng(self):
        cfg=RootExploreConfig(device='cpu',model=ModelConfig(channels=16,residual_blocks=1,policy_channels=8))
        pool=[SimpleNamespace(game_id=family,identity=(family,ply),training_policy_source='teacher') for family in range(100) for ply in range(5)]
        snapshots=[]
        for count in (64,256):
            learner=make_self_training_learner(cfg,Path('.'));learner.policy_pool=pool;learner.policy_batch_sources=Counter();learner.counters['attempted_updates']=8576
            draws=[];phases=Counter()
            for i in range(count):
                phase='value' if learner.counters['attempted_updates']%3==2 else 'policy';selected=learner._batch(pool)
                draws.append([s.identity for s in selected]);phases[phase]+=1;learner.counters['attempted_updates']+=1
            snapshots.append((draws,phases,copy.deepcopy(learner.generator.bit_generator.state)))
        self.assertEqual(snapshots[0][0],snapshots[1][0][:64]);self.assertNotEqual(snapshots[0][2],snapshots[1][2])
        self.assertEqual(snapshots[0][1],{'value':22,'policy':42});self.assertEqual(snapshots[1][1],{'value':86,'policy':170})

    def test_example_accounting_keeps_distinct_histories(self):
        a=('shard',1,2,1,9);b=('shard',1,2,1,10)
        result=exposure_summary({'policy':Counter({a:3,b:1}),'value':Counter({a:1})},{'policy':8,'value':8})
        self.assertEqual(result['policy']['sampled_rows'],4);self.assertEqual(result['policy']['unique_examples'],2);self.assertEqual(result['value']['unique_pool_fraction'],.125)

    def test_full_gate_and_guard_required_uncertainty_not_severe_stop(self):
        gains=[[{'families':64,'ci95':[.01,.2],'mean':.1}]*2 for _ in range(4)];duels=[{'families':64,'ci95':[.51,.7],'mean':.6}]*4;repeat=[{'families':64,'ci95':[0,0],'mean':0}]*2
        self.assertTrue(confirmed_route(gains,repeat,duels,{'champion_accepted':True},64,.02))
        self.assertFalse(confirmed_route(gains[:3],repeat,duels[:3],{'champion_accepted':True},64,.02))
        self.assertFalse(confirmed_route(gains,repeat,duels,{'champion_accepted':True},16,.02))
        a={'review':{'gains':{'parent':[{'ci95':[-.2,.01]}]*2}},'summaries':[{}, {'family_score':{'ci95':[.3,.6]}}]};self.assertFalse(severe_regression(a))
        a['review']['gains']['parent']=[{'ci95':[-.3,-.11]}]*2;a['summaries'][1]['family_score']['ci95']=[.1,.39];self.assertTrue(severe_regression(a))
if __name__=='__main__':unittest.main()
