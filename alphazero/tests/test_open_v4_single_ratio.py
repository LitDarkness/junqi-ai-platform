"""One evolving158 learner, complete-state migration and unchanged protected gates."""
import copy
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from junqi_open_v4.root_explore_config import RootExploreConfig
from junqi_az.config import ModelConfig
from junqi_open_v4.training_replay import make_self_training_learner
from fork_open_v4_qualified import fork_payload
from run_open_v4_search_training import verify_learning_state
from open_v4_single_ratio_contract import check_single_parameters, continuation_action, exposure_summary
from run_open_v4_single_ratio import confirmed_route, ordered_participants


def valid_plan():
    source=dict(iteration=158,model_hash='source')
    return dict(schema='junqi-single-ratio-plan-1',route='single_learning_update_ratio',training_variants=['learner'],
        maximum_stages=96,segment_stages=24,stages_per_block=12,games_per_stage=256,updates_per_stage=256,replay_stages=3,
        directory='runs/new',stop_paths=['runs/new/STOP','runs/new/supervision/STOP'],source_checkpoint=source,
        participants=dict(parent=source,control=dict(iteration=134),guard86=dict(iteration=86)),
        registered_first_segment_updates=6144,registered_first_segment_new_games=6144,
        learner=dict(selfplay_fraction=.25,teacher_selfplay_fraction=None,actor_policy_target='teacher',root_tactical_guard=False,
            training_greedy_after_ply=24,root_noise_until_ply=None,forced_root_coeff=2.,exploration_target_pruning=False,deduplicate=False,neural_policy_only=False))


class SingleRatioTests(unittest.TestCase):
    def test_rejects_second_learner_hidden_factor_or_old_source(self):
        p=valid_plan();check_single_parameters(p)
        for change in [dict(training_variants=['control','candidate']),dict(updates_per_stage=64),dict(source_checkpoint=dict(iteration=134))]:
            q=copy.deepcopy(p);q.update(change)
            with self.assertRaises(ValueError):check_single_parameters(q)
        for key,value in [('actor_policy_target','search'),('root_noise_until_ply',24),('exploration_target_pruning',True),('deduplicate',True)]:
            q=copy.deepcopy(p);q['learner'][key]=value
            with self.assertRaises(ValueError):check_single_parameters(q)

    def test_archival_stops_never_become_new_authorization(self):
        p=valid_plan();p['stop_paths'].append('runs/old/STOP')
        with self.assertRaises(ValueError):check_single_parameters(p)

    def test_full_learning_state_and_configuration_migration(self):
        torch.set_num_threads(2)
        cfg=RootExploreConfig(device='cpu',initial_checkpoint=None,model=ModelConfig(channels=16,residual_blocks=1,policy_channels=8),updates_per_iteration=64)
        with tempfile.TemporaryDirectory() as temp:
            learner=make_self_training_learner(cfg,Path(temp));learner.counters.update(iteration=158,attempted_updates=10112,effective_updates=10112,policy_updates=6742,value_updates=3370)
            learner.scheduler['effective_updates']=10112
            original=learner.payload(role='learner',lineage={});newcfg=copy.deepcopy(cfg);newcfg.updates_per_iteration=256;newcfg.run_directory='new'
            newcfg.validate();migrated=fork_payload(original,newcfg,{'optimizer_reset':False})
            self.assertTrue(all(verify_learning_state(original,migrated).values()))
            self.assertEqual(migrated['config']['updates_per_iteration'],256)
            self.assertEqual(migrated['rng']['batch_generator'],original['rng']['batch_generator'])

    def test_exact_first64_batches_and_continuous_longer_clock(self):
        cfg=RootExploreConfig(device='cpu',initial_checkpoint=None,model=ModelConfig(channels=16,residual_blocks=1,policy_channels=8))
        pool=[SimpleNamespace(game_id=f,identity=(f,ply),training_policy_source='teacher') for f in range(64) for ply in range(4)]
        snapshots=[]
        for count in (64,256):
            learner=make_self_training_learner(cfg,Path('.'));learner.counters['attempted_updates']=10112
            learner.policy_pool=pool;learner.policy_batch_sources=Counter()
            draws=[];phases=Counter()
            for _ in range(count):
                phase='value' if learner.counters['attempted_updates']%3==2 else 'policy';selected=learner._batch(pool)
                draws.append([s.identity for s in selected]);phases[phase]+=1;learner.counters['attempted_updates']+=1
            snapshots.append((draws,phases,copy.deepcopy(learner.generator.bit_generator.state)))
        self.assertEqual(snapshots[0][0],snapshots[1][0][:64]);self.assertNotEqual(snapshots[0][2],snapshots[1][2])
        self.assertEqual(snapshots[1][1],{'policy':170,'value':86})

    def test_continuation_is_conditional_and_no_failed_resume(self):
        self.assertEqual(continuation_action(12,False,False),'continue_registered_segment')
        self.assertEqual(continuation_action(24,False,False),'review_due')
        self.assertEqual(continuation_action(24,True,False),'auto_next_segment')
        self.assertEqual(continuation_action(12,True,True),'review_due')

    def test_protected_full_gate_cannot_adopt_weak_only_win(self):
        gains=[[dict(mean=.1,ci95=[.01,.2],families=64)]*2 for _ in range(4)]
        duels=[dict(mean=.6,ci95=[.51,.7],families=64) for _ in range(4)];repeat=[dict(mean=0,ci95=[0,0],families=64)]*2
        self.assertTrue(confirmed_route(gains,repeat,duels,dict(champion_accepted=True),64,.02))
        duels[2]['ci95'][0]=.5
        self.assertFalse(confirmed_route(gains,repeat,duels,dict(champion_accepted=True),64,.02))
        self.assertFalse(confirmed_route(gains,repeat,duels,dict(champion_accepted=True),16,.02))
        names=('candidate','repeat','control','parent','champion','guard86')
        self.assertLess(list(ordered_participants(dict.fromkeys(names))).index('champion'),list(ordered_participants(dict.fromkeys(names))).index('candidate'))

    def test_exposure_retains_distinct_saved_game_examples(self):
        a=('shard',1,0,2,1,9);b=('shard',1,1,2,1,9)
        rows=exposure_summary({'policy':Counter({a:3,b:1}),'value':Counter({a:1})},{'policy':8,'value':8})
        self.assertEqual(rows['policy']['unique_examples'],2);self.assertEqual(rows['policy']['sampled_rows'],4)
        self.assertEqual(rows['value']['unique_pool_fraction'],.125)

if __name__=='__main__':unittest.main()
