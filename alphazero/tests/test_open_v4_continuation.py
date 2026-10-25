import copy
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))

from fork_open_v4_qualified import fork_payload
from supervise_open_v4_branch import stalled_gate_count
from open_v4_runtime import atomic_json
from junqi_open_v4.learner import Learner,atomic_checkpoint,model_hash
from test_open_v4 import OpenV4Tests,small_config


class ContinuationTests(unittest.TestCase):
    def test_fork_preserves_optimizer_rng_counters_and_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);cfg=small_config()
            learner=Learner(cfg,root)
            sample=OpenV4Tests()._sample(root)
            learner.train([sample],updates=4)
            payload=learner.payload(role='learner',lineage={'source':'fixture'})
            migrated=copy.deepcopy(cfg);migrated.run_directory='runs/new';migrated.games_per_iteration=256
            forked=fork_payload(payload,migrated,{'proof':'verified'})
            self.assertEqual(forked['counters'],payload['counters'])
            self.assertEqual(forked['scheduler'],payload['scheduler'])
            self.assertEqual(forked['rng']['batch_generator'],payload['rng']['batch_generator'])
            path=root/'new.pt';atomic_checkpoint(path,forked)
            learner.train([sample],updates=4);expected=model_hash(learner.model)
            restored=Learner(migrated,root);restored.resume(path);restored.train([sample],updates=4)
            self.assertEqual(model_hash(restored.model),expected)
            self.assertEqual(restored.counters,learner.counters)

    def test_gate_stall_ignores_unscheduled_rounds_and_survives_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for i,gate in ((27,{'actor_accepted':False}),(28,{'status':'gate not scheduled'}),
                (30,{'actor_accepted':False}),(33,{'actor_accepted':False})):
                atomic_json(root/f'reports/iteration_{i:06d}.json',{'gate':gate})
            self.assertEqual(stalled_gate_count(root,26,33),3)
            atomic_json(root/'reports/iteration_000036.json',{'gate':{'actor_accepted':True}})
            self.assertEqual(stalled_gate_count(root,26,36),0)
