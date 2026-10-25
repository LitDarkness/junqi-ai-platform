"""Native history/backup conformance plus real collector raw/target separation."""
import os,sys,unittest,random,json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/os.environ['JUNQI_ROOT_EXPLORE_BUILD']))
import JunqiAZ,JunqiAZRootExplore
from test_open_v4_immediate import FakeState
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_open_v4.root_explore_games import collect_training_games
from open_v4_root_explore_contract import verify_collection
from run_open_v4_root_explore import confirmed_route,screening_signal

def compare_requests(a,b):
    for k in a:
        if k=='features':np.testing.assert_array_equal(a[k],b[k])
        else:assert a[k]==b[k],k

def backup(tree,request):
    if request['terminal']:tree.backup([],request['terminal_value'])
    else:
        prior=[float(1+(a%19)) for a in request['legal_actions']]
        tree.backup(prior,float((request['state_key']%17)-8)/10)

class NativeTests(unittest.TestCase):
    def test_default_off_exact_every_request_edge_noise_and_history(self):
        rng=random.Random(2026100501)
        for seed in range(12):
            s=JunqiAZ.JunqiAZState();s.reset_random(seed+491)
            for _ in range(seed*2):
                if s.terminal():break
                s.apply(rng.choice(s.legal_actions()))
            if s.terminal():continue
            original_key=s.key();features=np.asarray(s.features()).copy()
            for budget in (1,2,64,128):
                old=JunqiAZ.NeuralPUCTTree(s,1.5,False);new=JunqiAZRootExplore.RootExploreTree(s,1.5,False,0.)
                for i in range(budget):
                    a=old.select_leaf();b=new.select_leaf();compare_requests(a,b);backup(old,a);backup(new,b)
                    if i==0 and seed%2==0:
                        old.add_root_dirichlet(.3,.25,seed);new.add_root_dirichlet(.3,.25,seed)
                    self.assertEqual(old.root_edges(),new.root_edges());self.assertEqual(old.root_visits(),new.root_visits())
                self.assertEqual(new.target_visits(),[e['visits'] for e in new.root_edges()])
            self.assertEqual(s.key(),original_key);np.testing.assert_array_equal(s.features(),features)

    def test_forced_budget_target_best_protection_cancel_and_invalid_backup(self):
        for seed in range(12):
            s=JunqiAZ.JunqiAZState();s.reset_random(seed+890)
            for budget in (64,128):
                t=JunqiAZRootExplore.RootExploreTree(s,1.5,False,2.)
                first=t.select_leaf();t.cancel_pending();self.assertFalse(t.has_pending_leaf());self.assertEqual(t.root_visits(),0)
                compare_requests(first,t.select_leaf())
                with self.assertRaises(ValueError):t.backup([1],float('nan'))
                self.assertEqual(t.root_visits(),0);t.cancel_pending()
                for i in range(budget):
                    r=t.select_leaf();backup(t,r)
                    if i==0:t.add_root_dirichlet(.3,.25,seed)
                edges=t.root_edges();before=json.dumps(edges,sort_keys=True);target=t.target_visits()
                raw=np.array([e['visits'] for e in edges]);q=np.array([e['q'] for e in edges]);actions=np.array([e['action'] for e in edges])
                best=np.lexsort((actions,-q,-raw))[0]
                self.assertEqual(raw.sum(),budget-1);self.assertEqual(t.root_visits(),budget)
                self.assertEqual(target[best],raw[best]);self.assertGreater(sum(target),0)
                self.assertTrue(np.all(np.array(target)<=raw));self.assertTrue(np.all(np.array(target)>=0))
                self.assertEqual(before,json.dumps(t.root_edges(),sort_keys=True))
                self.assertGreater(t.forced_decisions(),0)
                with self.assertRaises(ValueError):t.select_leaves(2)

    def test_native_flag_terminal_and_nonterminal_player_value_sign(self):
        board=[0]*60;board[3]=3;board[58]=-3;board[53]=10
        s=JunqiAZ.JunqiAZState();s.load_board_state(board,1)
        win=53*60+58;self.assertIn(win,s.legal_actions())
        t=JunqiAZRootExplore.RootExploreTree(s,1.5,False,2.)
        r=t.select_leaf();t.backup([1. if a==win else 0. for a in r['legal_actions']],0.)
        r=t.select_leaf();self.assertTrue(r['terminal']);t.backup([],r['terminal_value'])
        edge=next(e for e in t.root_edges() if e['action']==win);self.assertEqual(edge['q'],1.)
        s.reset_random(473);t=JunqiAZRootExplore.RootExploreTree(s,1.5,False,0.)
        r=t.select_leaf();t.backup([1.]*len(r['legal_actions']),0.);r=t.select_leaf()
        self.assertFalse(r['terminal']);self.assertEqual(r['player'],-s.current_player());t.backup([1.]*len(r['legal_actions']),.5)
        self.assertEqual(next(e['q'] for e in t.root_edges() if e['visits']),-.5)

class FixedSearch:
    def __init__(self,*a,**kw):self.pruned=kw['target_pruning']
    def search(self,states,budget,**kw):
        results=[]
        for s in states:
            raw=np.array([10,budget-11],np.int32);target=np.array([3,budget-11],np.int32) if self.pruned else raw.copy()
            results.append(SimpleNamespace(actions=np.array([0,1]),visits=raw,pruned_visits=np.array([3,budget-11]),
                target_visits=target,target_policy=target/target.sum(),policy=raw/raw.sum(),action=1,root_entropy=.2,
                forced_decisions=4,priors=np.array([.1,.9]),q_values=np.array([-.3,.4]),simulations=budget,
                target_type='root_explore_integer_pruned' if self.pruned else 'root_explore_raw'))
        return results,dict(elapsed_seconds=.001,inference_rows=len(states),physical_inference_rows=len(states),inference_calls=1,
            tactical_roots=0,tactical_native_transitions=0,tactical_immediate_wins=0,tactical_disabled_actions=0)
class Teachers:
    modules=[None];workers=1;provenance={}
    def submit_actions(self,states,ids):return SimpleNamespace(result=lambda:[0]*len(states),elapsed_seconds=0.)
class CollectorTests(unittest.TestCase):
    def test_only_selfplay_targets_change_raw_behavior_wdl_and_teacher_hard_preserved(self):
        for mode in ('selfplay','teacher_selfplay','teacher_vs_actor'):
            fraction={'selfplay':(1.,0.),'teacher_selfplay':(0.,1.),'teacher_vs_actor':(0.,0.)}[mode]
            saved=[]
            for prune in (False,True):
                cfg=RootExploreConfig(selfplay_fraction=fraction[0],teacher_selfplay_fraction=fraction[1],
                    training_greedy_after_ply=24,games_per_iteration=2,exploration_target_pruning=prune)
                with patch('junqi_open_v4.root_explore_games.RootExploreSearch',FixedSearch):
                    rows,meta=collect_training_games(cfg,None,'cpu',SimpleNamespace(new_state=lambda seed:FakeState()),Teachers(),990)
                proof=verify_collection(rows,meta,cfg);self.assertTrue(proof['all_passed']);saved.append((rows,meta))
            a,b=saved;self.assertEqual([g['trace'] for g in a[1]['games']],[g['trace'] for g in b[1]['games']])
            self.assertEqual(a[1]['teacher_query_audit'],b[1]['teacher_query_audit'])
            for x,y in zip(a[0],b[0]):
                np.testing.assert_array_equal(x.raw_visits,y.raw_visits);np.testing.assert_array_equal(x.wdl,y.wdl)
                if mode=='selfplay':self.assertLess(y.visits.sum(),x.visits.sum())
                else:np.testing.assert_array_equal(x.visits,y.visits)

    def test_old_gate_plus_stronger86_cannot_be_bypassed(self):
        rows=[[{'families':64,'mean':.1,'ci95':[.01,.2]}]*2 for _ in range(4)]
        duels=[{'families':64,'mean':.6,'ci95':[.51,.7]}]*4
        duplicate=[{'families':64,'mean':0,'ci95':[0,0]}]*2
        self.assertTrue(confirmed_route(rows,duplicate,duels,{'champion_accepted':True},64,.02))
        bad=[*rows[:3],[{'families':64,'mean':.1,'ci95':[-.01,.2]}]*2]
        self.assertFalse(confirmed_route(bad,duplicate,duels,{'champion_accepted':True},64,.02))
        self.assertFalse(confirmed_route(rows[:3],duplicate,duels[:3],{'champion_accepted':True},64,.02))
        self.assertFalse(confirmed_route(rows,duplicate,duels,{'champion_accepted':True},16,.02))

if __name__=='__main__':unittest.main()
