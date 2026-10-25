"""Only one learner, no gate bypass, bounded continuation and archival protection."""
import copy
import unittest
from open_v4_single_contract import check_single_parameters, continuation_action
from run_open_v4_single import confirmed_route, screening_signal, ordered_participants


def valid_plan():
    source=dict(iteration=134,model_hash='source')
    return dict(schema='junqi-single-online-plan-1',route='single_online_continuation',
        training_variants=['learner'],maximum_stages=96,segment_stages=24,stages_per_block=12,
        games_per_stage=256,updates_per_stage=64,replay_stages=3,directory='runs/new',
        stop_paths=['runs/new/STOP','runs/new/supervision/STOP'],source_checkpoint=source,
        participants=dict(parent=source,control=dict(model_hash='other134')),
        learner=dict(selfplay_fraction=.25,teacher_selfplay_fraction=None,actor_policy_target='teacher',
            root_tactical_guard=False,training_greedy_after_ply=24,root_noise_until_ply=None,
            forced_root_coeff=2.0,exploration_target_pruning=False,deduplicate=False,neural_policy_only=False))


class SingleContractTests(unittest.TestCase):
    def test_one_branch(self):
        plan=valid_plan(); check_single_parameters(plan)
        plan['training_variants']=['control','candidate']
        with self.assertRaises(ValueError): check_single_parameters(plan)

    def test_no_numeric_reset_or_pruning(self):
        for key,val in [('exploration_target_pruning',True),('root_noise_until_ply',24),('deduplicate',True)]:
            plan=valid_plan();plan['learner'][key]=val
            with self.assertRaises(ValueError): check_single_parameters(plan)

    def test_old_stop_is_archival_not_new_scope(self):
        plan=valid_plan(); plan['stop_paths'].append('runs/old/STOP')
        with self.assertRaises(ValueError): check_single_parameters(plan)

    def test_no_blind_handoff(self):
        self.assertEqual(continuation_action(12,False,False),'continue_registered_segment')
        self.assertEqual(continuation_action(24,False,False),'review_due')
        self.assertEqual(continuation_action(24,True,False),'auto_next_segment')
        self.assertEqual(continuation_action(12,True,True),'review_due')
        self.assertEqual(continuation_action(96,True,False),'registered_horizon_complete')

    def test_original_and_protective_gates_are_mandatory(self):
        gains=[[dict(mean=.1,ci95=[.01,.2],families=64)]*2 for _ in range(4)]
        duels=[dict(mean=.6,ci95=[.51,.7],families=64) for _ in range(4)]
        repeat=[dict(mean=0,ci95=[0,0],families=64)]*2
        gate=dict(champion_accepted=True)
        self.assertTrue(confirmed_route(gains,repeat,duels,gate,64,.02))
        self.assertFalse(confirmed_route(gains,repeat,duels,dict(champion_accepted=False),64,.02))
        for i in range(4):
            bad=copy.deepcopy(gains);bad[i][0]['ci95'][0]=0
            self.assertFalse(confirmed_route(bad,repeat,duels,gate,64,.02))
        self.assertFalse(confirmed_route(gains,repeat,duels,gate,16,.02))
        bad=copy.deepcopy(repeat);bad[0]['mean']=.021
        self.assertFalse(confirmed_route(gains,bad,duels,gate,64,.02))

    def test_reference_order_preserves_original_promotion_delta(self):
        names=('candidate','repeat','control','parent','champion','guard86')
        ordered=ordered_participants(dict.fromkeys(names))
        self.assertLess(list(ordered).index('champion'),list(ordered).index('candidate'))
        self.assertTrue(screening_signal([[dict(mean=.01)]*2 for _ in range(4)],[dict(mean=.51)]*4))
        self.assertFalse(screening_signal([[dict(mean=.01)]*2 for _ in range(4)],[dict(mean=.5)]*4))


if __name__=='__main__': unittest.main()
