import copy
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
from run_open_v4_teacher_targets import history_safe,hard_policy_target,relabel_shard
from junqi_open_v4.layouts import create_manifest,load_role,MultiLayoutBackend
from junqi_search_v3.loop import SearchSample


class Teachers:
    def actions(self,states,ids):return [int(s.legal_actions()[-1]) for s in states]


class TeacherTargetTests(unittest.TestCase):
    def fixture(self,root):
        create_manifest(root,133,train=4,dev=4,final=4)
        backend=MultiLayoutBackend(load_role(root,"train"),133);state=backend.new_state(178)
        provenance=state.provenance();samples=[];trace=[];rng=np.random.default_rng(49)
        while not state.is_terminal():
            legal=state.legal_actions();visits=np.zeros(len(legal),np.int32);visits[0]=10
            context=state.repetition_contexts()
            samples.append(SearchSample(42,int(state.native_state.key()),state.feature_planes(),legal,context,visits,
                state.current_player(),len(trace),.1,raw_visits=visits.copy(),position_key=int(state.native_state.position_key()),
                repetition_count=int(state.native_state.repetition_count()),cycle_entry=bool(np.any(context))))
            action=int(rng.choice(legal));trace.append(action);state.apply(action)
            self.assertLess(len(trace),1000)
        for row in samples:row.wdl=np.eye(3,dtype=np.float32)[int(state.terminal_value(row.player))+1]
        game=dict(family=42,mode="selfplay",teacher_id=0,provenance=provenance,trace=trace,reason=state.terminal_reason(),result_red=state.terminal_value(1))
        return backend,samples,{"layout_role":"train","games":[game]}

    def test_teacher_target_changes_only_sparse_visits_on_exact_native_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            backend,rows,metadata=self.fixture(Path(directory));original=copy.deepcopy(rows)
            audit=relabel_shard(rows,metadata,backend,Teachers(),lambda:None,lambda *args:None)
            self.assertEqual(audit["native_rows_verified"],len(rows));self.assertGreater(audit["changed_selfplay_policy_rows"],0)
            for before,after in zip(original,rows):
                for field in ("observation","wdl","legal_actions","repetition_contexts","raw_visits"):
                    np.testing.assert_array_equal(getattr(before,field),getattr(after,field))
                self.assertEqual(before.state_hash,after.state_hash)
                if history_safe(before):self.assertEqual(int(after.visits.sum()),1)
                else:np.testing.assert_array_equal(before.visits,after.visits)

    def test_nonselfplay_rows_and_history_sensitive_labels_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            backend,rows,metadata=self.fixture(Path(directory));original=copy.deepcopy(rows)
            metadata["games"][0]["mode"]="teacher_selfplay"
            audit=relabel_shard(rows,metadata,backend,Teachers(),lambda:None,lambda *args:None)
            self.assertEqual(audit["changed_selfplay_policy_rows"],0)
            for a,b in zip(original,rows):np.testing.assert_array_equal(a.visits,b.visits)
            rows[0].repetition_count=2;self.assertFalse(history_safe(rows[0]))
            rows[0].repetition_count=1;rows[0].cycle_entry=True;self.assertFalse(history_safe(rows[0]))

    def test_trace_corruption_and_nontraining_source_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            backend,rows,metadata=self.fixture(Path(directory));rows[0].state_hash^=1
            with self.assertRaisesRegex(ValueError,"state mismatch"):
                relabel_shard(rows,metadata,backend,Teachers(),lambda:None,lambda *args:None)
            metadata["layout_role"]="dev"
            with self.assertRaisesRegex(ValueError,"only training"):
                relabel_shard(rows,metadata,backend,Teachers(),lambda:None,lambda *args:None)
            with self.assertRaisesRegex(ValueError,"not uniquely legal"):hard_policy_target(rows[0],-1)


if __name__=="__main__":unittest.main()
