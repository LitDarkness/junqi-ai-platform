"""CPU coverage of actual collector labels, shadow workload and old replay."""
import copy,unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from test_open_v4_immediate import FakeState
from junqi_open_v4.queued_config import QueuedConfig
from junqi_open_v4.actor_policy_games import collect_training_games
from junqi_open_v4.training_replay import replay_pools
from run_open_v4_search_training import paired_schedule,verify_learning_state
from open_v4_actor_policy_contract import check_factor_parameters


class FixedSearch:
    def __init__(self,*args,**kwargs):pass
    def search(self,states,budget,**kwargs):
        visits=np.array([0,budget-1],np.int32)
        return [SimpleNamespace(actions=np.array([0,1]),visits=visits.copy(),policy=np.array([0.,1.]),action=1,root_entropy=0.) for s in states],dict(elapsed_seconds=.001,inference_rows=len(states),physical_inference_rows=len(states),inference_calls=1,tactical_roots=0,tactical_native_transitions=0,tactical_immediate_wins=0,tactical_disabled_actions=0)


class FixedTeachers:
    modules=[None];workers=1;provenance={}
    def __init__(self):self.requests=[]
    def submit_actions(self,states,ids):
        self.requests.append(len(states));return SimpleNamespace(result=lambda:[0]*len(states),elapsed_seconds=0.)


class ActorPolicyTests(unittest.TestCase):
    def collect(self,mode,target):
        fractions={'selfplay':(1.,0.),'teacher_selfplay':(0.,1.),'teacher_vs_actor':(0.,0.)}[mode]
        cfg=QueuedConfig(selfplay_fraction=fractions[0],teacher_selfplay_fraction=fractions[1],actor_policy_target=target,training_greedy_after_ply=24,games_per_iteration=2)
        teacher=FixedTeachers()
        with patch('junqi_open_v4.actor_policy_games.BatchedNeuralPUCT',FixedSearch):
            rows,meta=collect_training_games(cfg,None,'cpu',SimpleNamespace(new_state=lambda seed:FakeState()),teacher,800)
        return rows,meta,teacher.requests

    def test_actual_actor_targets_change_but_behavior_terminal_and_teacher_workload_do_not(self):
        old,old_meta,old_requests=self.collect('teacher_vs_actor','teacher')
        new,new_meta,new_requests=self.collect('teacher_vs_actor','search')
        self.assertEqual(paired_schedule(old_meta),paired_schedule(new_meta));self.assertEqual(old_requests,new_requests)
        self.assertEqual([g['trace'] for g in old_meta['games']],[g['trace'] for g in new_meta['games']])
        self.assertEqual(old_meta['teacher_query_audit'],new_meta['teacher_query_audit'])
        self.assertEqual(new_meta['policy_label_audit']['actor_owned_search_targets'],30)
        for game_index in range(2):
            actor_side=new_meta['games'][game_index]['actor_side']
            for before,after in zip(old[game_index*30:(game_index+1)*30],new[game_index*30:(game_index+1)*30]):
                np.testing.assert_array_equal(before.wdl,after.wdl);np.testing.assert_array_equal(after.wdl,[0,1,0])
                np.testing.assert_array_equal(after.raw_visits,after.visits)
                np.testing.assert_array_equal(after.visits,[0,63] if after.player==actor_side else [1,0])
                np.testing.assert_array_equal(before.visits,[1,0])

    def test_selfplay_and_real_teacher_modes_unchanged(self):
        for mode in ('selfplay','teacher_selfplay'):
            a,ma,qa=self.collect(mode,'teacher');b,mb,qb=self.collect(mode,'search');self.assertEqual(qa,qb)
            for old,new in zip(a,b):np.testing.assert_array_equal(old.visits,new.visits);np.testing.assert_array_equal(old.wdl,new.wdl)

    def test_old_hard_actor_labels_are_not_reinterpreted_and_independent_draws_retained(self):
        old,old_meta,_=self.collect('teacher_vs_actor','teacher');new,new_meta,_=self.collect('teacher_vs_actor','search')
        values,policy,audit=replay_pools([(new,new_meta),(old,old_meta)],deduplicate=False,neural_policy_only=False,holdout=lambda group:False)
        self.assertEqual(len(values),120);self.assertEqual(len(policy),120);self.assertEqual(audit['policy_sources'],{'search':30,'teacher':90})
        self.assertTrue(all(np.array_equal(r.wdl,[0,1,0]) for r in values));self.assertEqual([r.training_policy_source for r in old],['teacher']*60)

    def test_only_registered_target_factor_is_allowed(self):
        control=dict(selfplay_fraction=.25,teacher_selfplay_fraction=.25,actor_policy_target='teacher',root_tactical_guard=False,training_greedy_after_ply=24,root_noise_until_ply=None,deduplicate=False,neural_policy_only=False)
        plan={'control':control,'candidate':dict(control,actor_policy_target='search')};check_factor_parameters(plan)
        implicit=copy.deepcopy(plan)
        for branch in implicit.values():branch['teacher_selfplay_fraction']=None
        check_factor_parameters(implicit)
        bad=copy.deepcopy(plan);bad['candidate']['selfplay_fraction']=.4
        with self.assertRaises(ValueError):check_factor_parameters(bad)

    def test_latest86_full_learning_state_preserved_in_both_config_forks(self):
        import torch
        from pathlib import Path
        from junqi_az.config import ModelConfig
        from fork_open_v4_qualified import fork_payload
        root=Path(__file__).resolve().parents[1];saved=torch.load(root/'runs/open_junqi_v4_late_noise_factor_20261004_143905/control/checkpoints/learner_stage_000024.pt',map_location='cpu',weights_only=False)
        for target in ('teacher','search'):
            raw=dict(saved['config'],actor_policy_target=target);raw['model']=ModelConfig(**raw['model']);cfg=QueuedConfig(**raw);cfg.run_directory='runs/test_not_started';cfg.initial_checkpoint='latest86';cfg.validate()
            self.assertTrue(all(verify_learning_state(saved,fork_payload(saved,cfg,{'cpu_test':True})).values()))
        self.assertEqual(saved['counters']['iteration'],86);self.assertEqual(saved['counters']['effective_updates'],5504)


if __name__=='__main__':unittest.main()
