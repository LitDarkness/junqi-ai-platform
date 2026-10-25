import unittest
from types import SimpleNamespace
import numpy as np
import torch
from junqi_open_v4.config import OpenConfig
from junqi_open_v4.games import behavior_action
from junqi_search_v3.search import BatchedNeuralPUCT
from run_open_v4_long_behavior import severe_regression, confirmed_route
from test_open_v4_tactical_guard import BiasedNetwork, TacticalGuardTests


class BehaviorTests(unittest.TestCase):
    def result(self):
        return SimpleNamespace(actions=np.array([10,20]),policy=np.array([.9,.1]),action=10,root_entropy=.33)

    def test_default_exact_sampling_and_rng(self):
        for ply in (0,23,24,80):
            a=np.random.default_rng(411);b=np.random.default_rng(411)
            p=self.result().policy.astype(float)
            if ply>=24:p=p**2
            p/=p.sum();expected=int(b.choice([10,20],p=p))
            result,metrics=behavior_action(OpenConfig(),self.result(),ply,a)
            self.assertEqual(expected,result);self.assertEqual(a.bit_generator.state,b.bit_generator.state)
            self.assertEqual(metrics['greedy_decisions'],0)

    def test_late_greedy_changes_behavior_keeps_targets_and_consumes_draw(self):
        cfg=OpenConfig(training_greedy_after_ply=24);r=self.result();before=r.policy.copy()
        greedy=np.random.default_rng(9);old=np.random.default_rng(9)
        got=[]
        for _ in range(200):
            action,metrics=behavior_action(cfg,r,24,greedy);got.append(action)
            behavior_action(OpenConfig(),r,24,old)
            self.assertEqual(metrics['greedy_decisions'],1)
        self.assertEqual(set(got),{10});np.testing.assert_array_equal(before,r.policy)
        self.assertEqual(greedy.bit_generator.state,old.bit_generator.state)

    def test_early_sampling_unchanged(self):
        r=self.result();a=np.random.default_rng(3);b=np.random.default_rng(3)
        x,_=behavior_action(OpenConfig(training_greedy_after_ply=24),r,23,a)
        y,_=behavior_action(OpenConfig(),r,23,b);self.assertEqual(x,y)

    def test_only_clear_regression_stops_early(self):
        arena={'review':{'gains':{'parent':[{'ci95':[-.3,-.11]}]*2}},'summaries':[{}, {'family_score':{'ci95':[.2,.39]}}]}
        self.assertTrue(severe_regression(arena))
        arena['review']['gains']['parent'][0]={'ci95':[-.2,.01]}
        self.assertFalse(severe_regression(arena))

    def test_zero_visit_search_selects_highest_prior(self):
        TacticalGuardTests.setUpClass();case=TacticalGuardTests.audit['variants']['control']['cases'][0]
        s=TacticalGuardTests().state(case);action=int(s.legal_actions()[-1])
        search=BatchedNeuralPUCT(BiasedNetwork(action),device=torch.device('cpu'),cuda_graphs=False)
        results,_=search.search([s],1)
        self.assertEqual(results[0].action,action);self.assertEqual(results[0].visits.sum(),0)

    def test_fixed_chunks_pad_and_preserve_original_row_mapping(self):
        class Fake(BatchedNeuralPUCT):
            def _infer_dynamic(self,requests):
                self.sizes.append(len(requests));return [(r,0.) for r in requests]
        search=Fake(BiasedNetwork(1),device=torch.device('cpu'),fixed_inference_batch_size=4)
        search.sizes=[];rows=list(range(7));result=search._infer(rows)
        self.assertEqual(search.sizes,[4,4]);self.assertEqual([r[0] for r in result],rows)


if __name__=='__main__':unittest.main()
