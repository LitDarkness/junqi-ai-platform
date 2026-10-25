import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from junqi_open_v4 import arena, resumable_arena

class State:
    def __init__(self,seed): self.seed=seed;self.steps=0;self.native_state=self
    def apply(self,a): self.steps+=1
    def legal_actions(self): return [0,1]
    def current_player(self): return 1 if self.steps%2==0 else -1
    def key(self): return self.seed*100+self.steps
    def is_terminal(self): return self.steps>=self.seed
    def terminal_value(self,side): return side if self.seed%2 else 0
    def terminal_reason(self): return 'FLAG' if self.seed%2 else 'REPETITION'

class Agent:
    reuse_roots=False
    def clear_trees(self): pass
    def search(self,states,budget,**kwargs):
        return [SimpleNamespace(action=s.steps%2) for s in states],dict(
            elapsed_seconds=.1,inference_calls=1,inference_rows=len(states),
            cache_hits=0,deduplicated_rows=0,reused_roots=0,inherited_root_visits=0)

class Teachers:
    workers=2
    def submit_actions(self,states,ids):
        return SimpleNamespace(result=lambda:[s.steps%2 for s in states],elapsed_seconds=.01)

class ArenaEngineeringTests(unittest.TestCase):
    def setUp(self):
        self.suite=[dict(red={},black={},setup_seed=seed,state_key=seed*100,trace=[],family=i,start_ply=0) for i,seed in enumerate((3,8,17))]
        self.args=(self.suite,[('a','teacher0'),('a','b')],{'a':Agent(),'b':Agent()},Teachers(),{'a':256,'b':256})
        self.backend=lambda r,b:SimpleNamespace(new_state=lambda seed:State(seed))
        self.patches=[patch.object(module,'backend_for',self.backend) for module in (arena,resumable_arena)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
    def assert_same(self,a,b):
        self.assertEqual(a['summaries'],b['summaries']);self.assertEqual(a['paired_deltas'],b['paired_deltas'])
        def rows(result): return [{k:v for k,v in g.items() if not k.endswith('_seconds')} for g in result['games']]
        self.assertEqual(rows(a),rows(b))
        self.assertEqual(a['search_metrics'],b['search_metrics'])
    def test_original_schedule_stats_and_journal_resume_exact(self):
        original=arena.run_arena(*self.args,seed=17,progress=False)
        with tempfile.TemporaryDirectory() as d:
            uninterrupted=resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={'model':'fixed'},checkpoint_rounds=4)
            self.assert_same(original,uninterrupted)
            completed=resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={'model':'fixed'},checkpoint_rounds=4)
            self.assert_same(original,completed)
        with tempfile.TemporaryDirectory() as d:
            calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==7:raise InterruptedError('simulated interruption')
            with self.assertRaises(InterruptedError):resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={'model':'fixed'},check_stop=stop,checkpoint_rounds=4)
            manifest=json.loads((Path(d)/'MANIFEST.json').read_text())
            self.assertGreater(manifest['completed_games'],0);self.assertGreater(manifest['active_games'],0)
            resumed=resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={'model':'fixed'},checkpoint_rounds=4)
            self.assert_same(original,resumed)
            with self.assertRaisesRegex(ValueError,'contract'):resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={'model':'changed'})
    def test_corruption_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={},checkpoint_rounds=4)
            manifest=json.loads((Path(d)/'MANIFEST.json').read_text())
            path=Path(d)/manifest['chunks'][0]['path'];path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'checksum'):resumable_arena.run_resumable_arena(*self.args,seed=17,progress=False,checkpoint_directory=d,binding={})

class NativeJournalTests(unittest.TestCase):
    def test_saved_dev_trajectories_native_reconstruction_and_stats(self):
        root=Path(__file__).resolve().parents[1]
        plan=json.loads((root/'config/experiments/open_v4_until400_20261006_041458.json').read_text())
        suite=json.loads((root/plan['suite']['path']).read_text())['states'][:3]
        class FirstLegal(Agent):
            def search(self,states,budget,**kwargs):
                results,report=super().search([SimpleNamespace(steps=0) for _ in states],budget,**kwargs)
                for r,s in zip(results,states):r.action=int(s.legal_actions()[0])
                return results,report
        class NativeTeachers(Teachers):
            def submit_actions(self,states,ids):
                return SimpleNamespace(result=lambda:[int(s.legal_actions()[0]) for s in states],elapsed_seconds=.01)
        args=(suite,[('a','teacher0'),('a','b')],{'a':FirstLegal(),'b':FirstLegal()},NativeTeachers(),{'a':256,'b':256})
        original=arena.run_arena(*args,seed=17,progress=False)
        with tempfile.TemporaryDirectory() as d:
            calls=0
            def stop():
                nonlocal calls
                calls+=1
                if calls==7:raise InterruptedError('native resume test')
            with self.assertRaises(InterruptedError):resumable_arena.run_resumable_arena(*args,seed=17,progress=False,checkpoint_directory=d,binding={'real_native':True},check_stop=stop,checkpoint_rounds=4)
            resumed=resumable_arena.run_resumable_arena(*args,seed=17,progress=False,checkpoint_directory=d,binding={'real_native':True},checkpoint_rounds=4)
            self.assertEqual(original['summaries'],resumed['summaries']);self.assertEqual(original['paired_deltas'],resumed['paired_deltas'])
            strip=lambda rows:[{k:v for k,v in g.items() if not k.endswith('_seconds')} for g in rows]
            self.assertEqual(strip(original['games']),strip(resumed['games']))
            from evaluate_open_v4_longitudinal import verify_native
            proof=verify_native(resumed,suite,lambda:None)
            self.assertEqual(proof['games'],12)
if __name__=='__main__':unittest.main()
