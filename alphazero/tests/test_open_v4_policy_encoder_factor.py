import tempfile
import unittest
from pathlib import Path

import torch
import test_open_v4 as fixtures
from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
from junqi_search_v3.loop import search_distill_batch
from junqi_search_v3.recovery import make_policy_anchor
from open_v4_policy_encoder_factor import make_learner
from run_open_v4_search_training import verify_learning_state


class PolicyEncoderFactorTests(unittest.TestCase):
    def test_original_multiplier_reproduces_complete_ppv_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixtures.small_config()
            sample = fixtures.OpenV4Tests()._sample(root)
            original = Learner(config, root)
            original.train([sample], updates=5)
            actual = make_learner(config, root, .25)
            actual.train([sample], updates=5)
            self.assertTrue(all(verify_learning_state(
                original.payload(role="learner", lineage={}),
                actual.payload(role="learner", lineage={})).values()))

    def test_policy_preserves_encoder_and_value_but_value_can_learn(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixtures.small_config()
            learner = make_learner(config, root, 0.)
            sample = fixtures.OpenV4Tests()._sample(root)
            encoder = [p.detach().clone() for p in learner.model.encoder_parameters()]
            policy = [p.detach().clone() for p in learner.model.policy_head_parameters()]
            batch = search_distill_batch([sample], learner.device)
            learner.model.eval()
            with torch.no_grad(): before = learner.model.terminal_wdl_logits(batch["observations"]).clone()
            result = learner.step([sample], "policy")
            self.assertTrue(result["accepted"])
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(encoder,learner.model.encoder_parameters())))
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(policy,learner.model.policy_head_parameters())))
            learner.model.eval()
            with torch.no_grad(): after = learner.model.terminal_wdl_logits(batch["observations"])
            torch.testing.assert_close(before, after, rtol=0, atol=0)
            # Adam's encoder moments intentionally remain active with lr=0.
            self.assertTrue(all(p in learner.optimizer.state for p in learner.model.encoder_parameters()))
            anchor = make_policy_anchor(learner.model, [sample], learner.device, count=1)
            result = learner.step([sample], "value", anchor=anchor)
            self.assertTrue(result["accepted"])
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(encoder,learner.model.encoder_parameters())))

    def test_resume_preserves_full_next_ppv_and_rejects_wrong_factor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); config = fixtures.small_config()
            sample = fixtures.OpenV4Tests()._sample(root)
            learner = make_learner(config, root, 0.)
            learner.train([sample], updates=2)
            path = root / "learner.pt"
            atomic_checkpoint(path, learner.payload(role="learner", lineage={"fixture": True}))
            learner.train([sample], updates=4)
            resumed = make_learner(config, root, 0.)
            self.assertEqual(resumed.resume(path), {"fixture": True})
            resumed.train([sample], updates=4)
            self.assertEqual(model_hash(learner.model), model_hash(resumed.model))
            self.assertTrue(all(verify_learning_state(
                learner.payload(role="learner", lineage={}),
                resumed.payload(role="learner", lineage={})).values()))
            with self.assertRaisesRegex(ValueError, "factor mismatch"):
                make_learner(config, root, .25).resume(path)


if __name__ == "__main__": unittest.main()
