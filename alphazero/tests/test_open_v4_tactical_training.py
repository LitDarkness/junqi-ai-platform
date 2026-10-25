import copy
import unittest
from run_open_v4_tactical_training import screening_signal, confirmed_route, ordered_participants


class RouteTests(unittest.TestCase):
    def setUp(self):
        self.gains = [[dict(mean=.1, ci95=[.01,.2], families=64) for _ in range(2)] for _ in range(3)]
        self.repeat = [dict(mean=.0, ci95=[-.02,.02], families=64) for _ in range(2)]
        self.duels = [dict(mean=.6, ci95=[.51,.7], families=64) for _ in range(3)]
        self.gate = dict(champion_accepted=True)

    def test_full_gate_and_production_champion_protection(self):
        self.assertTrue(confirmed_route(self.gains,self.repeat,self.duels,self.gate,64,.02))
        for group in range(3):
            changed=copy.deepcopy(self.gains); changed[group][0]['ci95'][0]=-.01
            self.assertFalse(confirmed_route(changed,self.repeat,self.duels,self.gate,64,.02))
        self.assertFalse(confirmed_route(self.gains,self.repeat,self.duels,self.gate,16,.02))
        self.repeat[0]['mean']=.021
        self.assertFalse(confirmed_route(self.gains,self.repeat,self.duels,self.gate,64,.02))

    def test_exploratory_screen_cannot_be_promotion(self):
        self.assertTrue(screening_signal(self.gains,self.duels))
        self.duels[2]['mean']=.49
        self.assertFalse(screening_signal(self.gains,self.duels))

    def test_baselines_precede_candidate_for_paired_deltas(self):
        ordered=ordered_participants(dict(repeat=1,candidate=2,control=3,parent=4,champion=5))
        self.assertEqual(list(ordered),['champion','parent','control','candidate','repeat'])
        with self.assertRaises(ValueError): ordered_participants({'parent':1})


if __name__ == '__main__': unittest.main()
