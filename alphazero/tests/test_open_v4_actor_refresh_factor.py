import copy
import unittest

from open_v4_actor_refresh_factor import collection_source, verify_collection_pair


class ActorRefreshTests(unittest.TestCase):
    def test_only_candidate_refreshes_from_immediate_complete_checkpoint(self):
        qualified = dict(path="qualified26.pt", sha256="parent", model_hash="model26")
        previous = dict(stage=1, checkpoint="candidate/learner1.pt", sha256="full1", model_hash="model1")
        self.assertEqual(collection_source(1, "candidate", dict(stage=0), qualified), qualified)
        self.assertEqual(collection_source(2, "control", previous, qualified), qualified)
        chosen = collection_source(2, "candidate", previous, qualified)
        self.assertEqual(chosen["sha256"], "full1")
        self.assertEqual(chosen["model_hash"], "model1")
        self.assertEqual(previous["stage"], 1)
        with self.assertRaisesRegex(ValueError, "immediately preceding"):
            collection_source(3, "candidate", previous, qualified)

    def test_pairs_keep_layout_color_mode_teacher_but_allow_new_outcomes(self):
        games = [dict(index=i, family=46000000+i//2, mode="selfplay", actor_side=1 if i%2==0 else -1,
                      teacher_id=(i//2)%2, provenance={"layout": i//2}, trace=[0], result_red=-1) for i in range(256)]
        frozen = dict(layout_role="train", game_count=256, iteration=46, games=games, actor_model_hash="parent")
        fresh = copy.deepcopy(frozen); fresh["actor_model_hash"]="learner"
        fresh["games"][0].update(trace=[1,2], result_red=1)
        self.assertTrue(verify_collection_pair(frozen, fresh, "learner"))
        bad = copy.deepcopy(fresh); bad["games"][0]["mode"]="teacher_selfplay"
        with self.assertRaisesRegex(ValueError, "paired starts"):
            verify_collection_pair(frozen, bad, "learner")
        with self.assertRaisesRegex(ValueError, "training layouts"):
            verify_collection_pair(frozen, fresh | {"layout_role": "dev"}, "learner")
        with self.assertRaisesRegex(ValueError, "recorded learner"):
            verify_collection_pair(frozen, fresh, "parent")

    def test_sampler_keys_remain_paired_when_refreshed_trajectory_lengths_differ(self):
        import tempfile
        from pathlib import Path
        import test_open_v4 as fixtures
        from open_v4_value_source_factor import make_learner
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); cfg = fixtures.small_config()
            sample = fixtures.OpenV4Tests()._sample(root)
            old, fresh = [], []
            for family in (1, 2, 3):
                row = copy.deepcopy(sample); row.game_id=family; old.append(row)
                for ply in range(3):
                    row = copy.deepcopy(sample); row.game_id=family; row.ply=ply; row.state_hash+=ply+1
                    fresh.append(row)
            modes={1:"selfplay",2:"teacher_selfplay",3:"teacher_vs_actor"}
            control=make_learner(cfg,root,modes,"all"); candidate=make_learner(cfg,root,modes,"all")
            for attempt in range(6):
                control.counters["attempted_updates"]=candidate.counters["attempted_updates"]=attempt
                control._batch(old); candidate._batch(fresh)
            self.assertEqual(control.sampling_key_digest.hexdigest(),candidate.sampling_key_digest.hexdigest())
            self.assertNotEqual(control.phase_batch_digests["policy"].hexdigest(),candidate.phase_batch_digests["policy"].hexdigest())
            self.assertEqual(control.value_pool_size,len(old))
            self.assertEqual(candidate.value_pool_size,len(fresh))


if __name__ == "__main__": unittest.main()
