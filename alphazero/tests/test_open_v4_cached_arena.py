import copy
import json
import unittest
from pathlib import Path

from junqi_open_v4.arena import select_families
from junqi_open_v4.config import OpenConfig
from junqi_open_v4.pipeline import promotion_decision
from open_v4_cached_arena import assemble, compare_repeat, index_games, matchups
from open_v4_experiment_evidence import teacher_deltas


class CachedArenaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[1]
        cls.plan=json.loads((root/'config/experiments/open_v4_long_behavior_20261004_081501.json').read_text(encoding='utf-8-sig'))
        cls.cached=json.loads((root/cls.plan['directory']/'arena/block_000012_full.json').read_text(encoding='utf-8-sig'))
        cls.suite=select_families(json.loads((root/cls.plan['suite']['path']).read_text(encoding='utf-8-sig'))['states'],64,cls.plan['evaluation_seed'])
        cls.base=OpenConfig.load(root/cls.plan['parent_config'])
        cls.cached_pairs=[(role,f'teacher{i}') for i in range(2) for role in ('champion','parent')]
        cls.fresh=[row for row in cls.cached['games'] if (row['candidate'],row['opponent']) not in cls.cached_pairs]

    def test_real_full64_reaggregation_matches_every_original_statistic_and_gate(self):
        result=assemble(self.cached,[{'games':self.fresh}],self.suite,self.plan['evaluation_seed'],reuse=True)
        self.assertEqual(result['summaries'],self.cached['summaries'])
        self.assertEqual(result['paired_deltas'],self.cached['paired_deltas'])
        self.assertEqual(promotion_decision(result,self.base),promotion_decision(self.cached,self.base))
        for role in ('champion','parent','control','repeat'):
            self.assertEqual(teacher_deltas(result,role,'candidate'),teacher_deltas(self.cached,role,'candidate'))
        self.assertEqual((result['fresh_games'],result['reused_games']),(2916,1296))
        self.assertEqual([r['id'] for r in result['games']],[r['id'] for r in self.cached['games']])
        self.assertNotIn('elapsed_seconds',result)
        self.assertNotIn('search_metrics',result)

    def test_reordered_partial_output_keeps_canonical_seeded_intervals(self):
        result=assemble(self.cached,[{'games':list(reversed(self.fresh))}],self.suite,self.plan['evaluation_seed'],reuse=True)
        self.assertEqual(result['summaries'],self.cached['summaries'])
        self.assertEqual(result['paired_deltas'],self.cached['paired_deltas'])

    def test_cache_can_be_disabled_with_complete_fresh_fixtures(self):
        result=assemble(self.cached,[{'games':self.cached['games']}],self.suite,self.plan['evaluation_seed'],reuse=False)
        self.assertEqual((result['fresh_games'],result['reused_games']),(4212,0))
        self.assertEqual(result['summaries'],self.cached['summaries'])

    def test_missing_duplicate_and_malformed_fixture_rejected(self):
        rows=copy.deepcopy(self.cached['games'])
        with self.assertRaises(ValueError):index_games(rows[:-1],self.suite,matchups())
        with self.assertRaises(ValueError):index_games(rows+[rows[0]],self.suite,matchups())
        mutations={'position':-1,'candidate_side':0,'family':-1,'start_ply':-1,'score':.123,'value':5,'trace':[],'opponent':'unknown'}
        for field,value in mutations.items():
            with self.subTest(field=field):
                bad=copy.deepcopy(rows);bad[0][field]=value
                with self.assertRaises(ValueError):index_games(bad,self.suite,matchups())

    def test_same_result_different_trace_does_not_validate_cache(self):
        repeat={'games':copy.deepcopy([r for r in self.cached['games'] if r['candidate']=='repeat'])}
        proof=compare_repeat(self.cached,repeat,self.suite)
        self.assertTrue(proof['all_exact']);self.assertEqual(proof['games'],648)
        repeat['games'][0]['trace'][0]+=1
        proof=compare_repeat(self.cached,repeat,self.suite)
        self.assertFalse(proof['all_exact']);self.assertEqual(proof['exact_matches'],647)
        self.assertEqual(proof['first_differences'][0]['fields'],['trace'])


if __name__=='__main__':unittest.main()
