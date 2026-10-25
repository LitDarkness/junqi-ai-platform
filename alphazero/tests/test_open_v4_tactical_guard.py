from __future__ import annotations
import json
import unittest
from pathlib import Path
import numpy as np
import torch
import JunqiAZ
from junqi_az.backend import CppAZStateAdapter
from junqi_open_v4.layouts import backend_for, load_role
from junqi_search_v3.search import BatchedNeuralPUCT

ROOT = Path(__file__).resolve().parents[1]


class BiasedNetwork:
    config = None
    def __init__(self, action): self.action = action
    def eval(self): return self
    def parameters(self): return []
    def policy_value(self, obs, masks, repetition):
        scores = torch.full(masks.shape, -30., device=obs.device)
        scores[:, self.action] = 30.
        return torch.softmax(scores.masked_fill(~masks, -1e30), -1), torch.zeros(len(obs), device=obs.device)


class TacticalGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.audit = json.loads((ROOT/'tests/fixtures/open_v4_tactical_cases.json').read_text())
        cls.layouts = {r['id']:r for r in load_role(ROOT/'config/layouts/open_v4', 'train')}

    def state(self, case):
        p = case['provenance']; red=self.layouts[p['red_layout_id']]; black=self.layouts[p['black_layout_id']]
        s=backend_for(red,black,mirrored=list(red['setup'])!=p['red_layout']).new_state(p['setup_seed'])
        for action in case['prefix']: s.apply(action)
        return s

    def test_all_recorded_avoidable_losses_receive_zero_actual_search_visits(self):
        count = 0
        for item in self.audit['variants'].values():
            for case in item['cases']:
                s=self.state(case); before=s.native_state.key(); legal=s.legal_actions().copy()
                model=BiasedNetwork(case['action'])
                old=BatchedNeuralPUCT(model,device=torch.device('cpu'),cuda_graphs=False)
                new=BatchedNeuralPUCT(model,device=torch.device('cpu'),cuda_graphs=False,root_tactical_guard=True)
                a,_=old.search([s],8)
                b,metrics=new.search([s],8,add_root_noise=True,noise_seeds=[31])
                self.assertEqual(a[0].action,case['action'])
                self.assertIn(b[0].action,case['safe_actions'])
                self.assertEqual(sum(b[0].visits),7)
                np.testing.assert_array_equal(b[0].actions,legal)
                self.assertEqual(int(b[0].visits[np.where(legal==case['action'])[0][0]]),0)
                self.assertGreater(metrics['tactical_disabled_actions'],0)
                self.assertEqual(s.native_state.key(),before)
                count+=1
        self.assertEqual(count,9)

    def test_immediate_native_win_overrides_bad_prior_and_root_noise(self):
        raw=JunqiAZ.JunqiAZState()
        board=[0]*60;board[1]=3;board[58]=-3;board[53]=10;board[6]=-20
        raw.load_board_state(board,1);s=CppAZStateAdapter(raw)
        win=53*60+58;self.assertIn(win,s.legal_actions())
        other=next(int(a) for a in s.legal_actions() if int(a)!=win)
        search=BatchedNeuralPUCT(BiasedNetwork(other),device=torch.device('cpu'),root_tactical_guard=True)
        result,metrics=search.search([s],8,add_root_noise=True,noise_seeds=[19])
        self.assertEqual(result[0].action,win)
        one,_=search.search([s],1)
        self.assertEqual(one[0].action,win)
        empty,empty_metrics=search.search([],8)
        self.assertEqual(empty,[])
        self.assertEqual(empty_metrics['tactical_roots'],0)
        self.assertEqual(int(result[0].visits.sum()),7)
        self.assertGreater(metrics['tactical_immediate_wins'],0)

    def test_all_losing_roots_remain_searchable_and_reuse_is_rejected(self):
        raw=JunqiAZ.JunqiAZState();board=[0]*60
        board[1]=3;board[58]=-3;board[2]=-20;board[40]=10
        raw.load_board_state(board,1);s=CppAZStateAdapter(raw)
        # Every action by the distant sapper leaves the flag capturable.
        search=BatchedNeuralPUCT(BiasedNetwork(int(s.legal_actions()[0])),device=torch.device('cpu'),root_tactical_guard=True)
        result,metrics=search.search([s],8)
        self.assertIn(result[0].action,s.legal_actions())
        self.assertEqual(metrics['tactical_disabled_actions'],0)
        with self.assertRaisesRegex(ValueError,'fresh roots'):
            BatchedNeuralPUCT(BiasedNetwork(0),device=torch.device('cpu'),root_tactical_guard=True,reuse_roots=True)


if __name__=='__main__': unittest.main()
