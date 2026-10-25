import unittest
import numpy as np
import torch
from junqi_az.model import PolicyValueNet
from junqi_az.config import ModelConfig
from evaluate_open_v4_head_crossover import make_cross_model,calibration_summary,head_primary_signal


class HeadCrossoverTests(unittest.TestCase):
    def test_exact_policy_and_value_sources_with_history(self):
        cfg=ModelConfig(channels=8,residual_blocks=0,policy_channels=8,value_head_type='wdl',repetition_context=True)
        torch.manual_seed(35);a=PolicyValueNet(cfg).eval();b=PolicyValueNet(cfg).eval()
        a.search_v3_leaf_value_scale=b.search_v3_leaf_value_scale=.9
        x=torch.randn(2,cfg.input_channels,12,5);mask=torch.zeros(2,3600,dtype=torch.bool);mask[:,[0,13,250]]=True
        rep=torch.zeros(2,3600,dtype=torch.uint8);rep[:,13]=2
        before={k:v.clone() for k,v in a.state_dict().items()}
        with torch.inference_mode():
            pa,va=a.policy_value(x,mask,rep);pb,vb=b.policy_value(x,mask,rep)
            pc,vc=make_cross_model(a,b).policy_value(x,mask,rep)
            torch.testing.assert_close(pc,pa,rtol=0,atol=0);torch.testing.assert_close(vc,vb,rtol=0,atol=0)
            pi,vi=make_cross_model(a,a).policy_value(x,mask,rep)
            torch.testing.assert_close(pi,pa,rtol=0,atol=0);torch.testing.assert_close(vi,va,rtol=0,atol=0)
        self.assertEqual(float(pc[:,~mask[0]].sum()),0.)
        self.assertTrue(all(torch.equal(v,before[k]) for k,v in a.state_dict().items()))
        b.search_v3_leaf_value_scale=.8
        with self.assertRaises(ValueError):make_cross_model(a,b)

    def test_calibration_perfect_labels_and_family_balance(self):
        y=np.eye(3)[[0,0,2]];r=calibration_summary(y,y,np.array([-1,-1,1]),[1,1,2])
        self.assertEqual(r['family_brier'],0.);self.assertEqual(r['family_scaled_search_value_mse'],0.)
        self.assertEqual(r['training_game_families'],2);self.assertEqual(r['row_weighted_confidence_ece'],0.)
        with self.assertRaises(ValueError):calibration_summary(y,y*.5,np.zeros(3),[1,1,2])

    def test_partial_or_duplicate_failed_evidence_cannot_select(self):
        row=dict(mean=.1,ci95=[.01,.2],families=32,uncertainty_estimable=True)
        duel=dict(mean=.6,ci95=[.51,.7],families=32,uncertainty_estimable=True)
        dup=dict(mean=0.,ci95=[-.01,.01],families=32,uncertainty_estimable=True)
        self.assertTrue(head_primary_signal([[row,row],[row,row]],[dup,dup],[duel,duel],32,.02))
        self.assertFalse(head_primary_signal([[row],[row,row]],[dup,dup],[duel,duel],32,.02))
        self.assertFalse(head_primary_signal([[row,row],[row,row]],[dup|{'mean':.03},dup],[duel,duel],32,.02))


if __name__=='__main__':unittest.main()
