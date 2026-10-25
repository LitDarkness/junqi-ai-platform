"""CPU checks for automatic handoff, STOP preservation and actual noise sampling."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np

from open_v4_queue_contract import queue_decision,verify_training_prerequisite,verify_boundary_audit
from open_v4_runtime import ProcessLock,atomic_json
from run_open_v4_immediate import request_stop
from junqi_open_v4.queued_config import QueuedConfig,root_noise_enabled
from junqi_open_v4.queued_games import collect_training_games


class FakeState:
    def __init__(self):self.ply=0;self.native_state=self
    def provenance(self):return {'training_fixture':True}
    def current_player(self):return 1 if self.ply%2==0 else -1
    def repetition_contexts(self):return np.zeros(2,dtype=np.int32)
    def legal_actions(self):return np.array([0,1],dtype=np.int32)
    def feature_planes(self):return np.zeros((107,12,5),dtype=np.float32)
    def key(self):return 100+self.ply
    def position_key(self):return self.key()
    def repetition_count(self):return 1
    def apply(self,action):assert action in (0,1);self.ply+=1
    def is_terminal(self):return self.ply==30
    def terminal_value(self,player):return 0
    def terminal_reason(self):return 'independent_draw_fixture'


class FakeSearch:
    noise_calls=[]
    def __init__(self,*args,**kwargs):pass
    def search(self,states,budget,*,add_root_noise,noise_seeds):
        type(self).noise_calls.append((states[0].ply,add_root_noise,len(states)))
        visits=np.array([63,0] if add_root_noise else [0,63],dtype=np.int32)
        results=[SimpleNamespace(actions=np.array([0,1],dtype=np.int32),visits=visits.copy(),
                  policy=visits/visits.sum(),action=int(np.argmax(visits)),root_entropy=0.) for state in states]
        metrics=dict(elapsed_seconds=.001,inference_rows=len(states),physical_inference_rows=32,inference_calls=1,
                     tactical_roots=0,tactical_native_transitions=0,tactical_immediate_wins=0,tactical_disabled_actions=0)
        return results,metrics


class FakeTeachers:
    modules=[None];workers=1;provenance={'fixture':'cpu'}
    def submit_actions(self,states,ids):
        return SimpleNamespace(result=lambda:[0]*len(states),elapsed_seconds=0.)


class ImmediateTests(unittest.TestCase):
    def test_live_or_still_locked_review_never_launches(self):
        for state in ('checking','arena','review_due'):
            self.assertEqual(queue_decision(state,stopped=False,locked=True,complete=True),'wait')
        self.assertEqual(queue_decision('review_due',stopped=False,locked=False,complete=True),'launch')

    def test_missing_commit_error_or_user_stop_never_launches(self):
        for state,complete in (('arena',False),('review_due',False),('needs_attention',True)):
            self.assertEqual(queue_decision(state,stopped=False,locked=False,complete=complete),'fail')
        self.assertEqual(queue_decision('review_due',stopped=True,locked=False,complete=True),'stop')
        self.assertEqual(queue_decision('stopped_by_user',stopped=False,locked=True,complete=True),'stop')

    def test_os_lock_excludes_duplicate_waiter_and_preserves_user_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)
            with ProcessLock(path/'NEXT_STAGE.lock'):
                with self.assertRaises(RuntimeError):
                    with ProcessLock(path/'NEXT_STAGE.lock'):pass
            stop=path/'STOP';stop.write_bytes(b'original user instruction')
            request_stop(stop);self.assertEqual(stop.read_bytes(),b'original user instruction')
            request_stop(path/'new/STOP');self.assertTrue((path/'new/STOP').exists())

    def test_config_noise_boundary_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cfg.json';cfg=QueuedConfig(root_noise_until_ply=24);cfg.validate()
            atomic_json(path,cfg.as_dict());saved=QueuedConfig.load(path)
            self.assertEqual(saved.digest(),cfg.digest())
            self.assertTrue(root_noise_enabled(saved,23));self.assertFalse(root_noise_enabled(saved,24))
            self.assertTrue(root_noise_enabled(QueuedConfig(),100))
            with self.assertRaises(ValueError):QueuedConfig(root_noise_until_ply=25).validate()

    def test_real_collector_changes_noise_and_visit_targets_only_at_registered_boundary(self):
        metadata={};samples={}
        for schedule in (None,24):
            cfg=QueuedConfig(selfplay_fraction=1.,teacher_selfplay_fraction=0.,training_greedy_after_ply=24,
                             root_noise_until_ply=schedule,games_per_iteration=2)
            FakeSearch.noise_calls=[]
            with patch('junqi_open_v4.queued_games.BatchedNeuralPUCT',FakeSearch):
                rows,meta=collect_training_games(cfg,None,'cpu',SimpleNamespace(new_state=lambda seed:FakeState()),FakeTeachers(),500)
            metadata[schedule]=meta;samples[schedule]=rows
            self.assertEqual(meta['game_count'],2);self.assertEqual(meta['total_plies'],60)
            self.assertTrue(all(np.array_equal(row.wdl,[0,1,0]) for row in rows))
            self.assertEqual([r[1] for r in FakeSearch.noise_calls],[schedule is None or ply<24 for ply in range(30)])
            self.assertEqual(meta['physical_inference_rows'],960);self.assertEqual(meta['padding_rows'],900)
        self.assertEqual(metadata[None]['behavior_contract']['noise_decisions'],{'with_noise':60})
        self.assertEqual(metadata[24]['behavior_contract']['noise_decisions'],{'with_noise':48,'without_noise':12})
        self.assertEqual([g['trace'] for g in metadata[None]['games']],[[0]*30]*2)
        self.assertEqual([g['trace'] for g in metadata[24]['games']],[[0]*24+[1]*6]*2)
        for a,b in zip(samples[None],samples[24]):
            self.assertEqual(np.array_equal(a.visits,b.visits),a.ply<24)

    def test_actual_latest_full_learning_state_survives_config_fork(self):
        import torch
        from fork_open_v4_qualified import fork_payload
        from run_open_v4_search_training import verify_learning_state
        root=Path(__file__).resolve().parents[1]
        saved=torch.load(root/'runs/open_junqi_v4_long_behavior_20261004_081501/candidate/checkpoints/learner_stage_000024.pt',map_location='cpu',weights_only=False)
        raw=dict(saved['config'],root_noise_until_ply=24)
        from junqi_az.config import ModelConfig
        raw['model']=ModelConfig(**raw['model']);cfg=QueuedConfig(**raw)
        cfg.run_directory='runs/test_unused';cfg.validate()
        self.assertTrue(all(verify_learning_state(saved,fork_payload(saved,cfg,{'test':'cpu'})).values()))
        self.assertEqual(saved['counters']['iteration'],62)

    def test_runner_rejects_unqualified_route_even_with_saved_native_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);proof={'arena':{'sha256':'fixture'},'review':{'route_confirmed':False},'native_games_checked':0}
            atomic_json(root/'queue/COMPLETION_REVIEW.json',dict(proof,native_games_checked=4212))
            plan={'queue_directory':'queue','completion_plan':{},'source':{},'prerequisite':{'native_all_games_replayed':4212,'required_route_confirmed':True}}
            with patch('open_v4_queue_contract.validate_finished_arena',return_value=proof):
                with self.assertRaisesRegex(ValueError,'route'):verify_training_prerequisite(root,plan)
            plan['prerequisite']['required_route_confirmed']=False
            with patch('open_v4_queue_contract.validate_finished_arena',return_value=proof):
                with self.assertRaises(FileNotFoundError):verify_training_prerequisite(root,plan)

    def test_empty_or_unbound_diagnostic_cannot_trigger_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);queue={'directory':'queue','source':{},'completion_plan':{'path':'completion.json'}}
            atomic_json(root/'queue/BOUNDARY_AUDIT.json',{'complete':True,'queue_identity':{}})
            with self.assertRaisesRegex(ValueError,'committed'):verify_boundary_audit(root,queue)


if __name__=='__main__':unittest.main()
