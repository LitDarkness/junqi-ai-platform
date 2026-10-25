import tempfile
import unittest
from pathlib import Path

import torch
import test_open_v4 as fixtures
from fork_open_v4_qualified import fork_payload
from junqi_open_v4.learner import Learner, atomic_checkpoint
from junqi_search_v3.recovery import make_policy_anchor
from open_v4_value_encoder_factor import variant_config
from run_open_v4_search_training import verify_learning_state


class ValueEncoderFactorTests(unittest.TestCase):
    def test_complete_state_control_reproduces_next_ppv_and_factor_is_config_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixtures.small_config()
            sample = fixtures.OpenV4Tests()._sample(root)
            learner = Learner(base, root); learner.train([sample], updates=4)
            saved = learner.payload(role="learner", lineage={})
            factors = {"control": .1, "candidate": .5}
            cfg = variant_config(base, "runs/test", None, "control", factors)
            migrated = fork_payload(saved, cfg, {"fixture": True})
            self.assertTrue(all(verify_learning_state(saved, migrated).values()))
            path = root / "control.pt"; atomic_checkpoint(path, migrated)
            learner.train([sample], updates=5)
            control = Learner(cfg, root); control.resume(path); control.train([sample], updates=5)
            self.assertTrue(all(verify_learning_state(learner.payload(role="learner", lineage={}),
                                                    control.payload(role="learner", lineage={})).values()))
            candidate_cfg = variant_config(base, "runs/test", None, "candidate", factors)
            self.assertNotEqual(candidate_cfg.digest(), cfg.digest())
            with self.assertRaisesRegex(ValueError, "contract or role mismatch"):
                Learner(candidate_cfg, root).resume(path)

    def test_only_value_phase_encoder_rate_changes_and_policy_guard_remains(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); base = fixtures.small_config()
            cfg = variant_config(base, "runs/test", None, "candidate", {"control": .1, "candidate": .5})
            changed = {k for k,v in cfg.as_dict().items() if v != base.as_dict()[k]}
            self.assertEqual(changed, {"run_directory", "value_encoder_lr_multiplier"})
            sample = fixtures.OpenV4Tests()._sample(root)
            learner = Learner(cfg, root)
            learner._set_phase("policy")
            rates = {g["name"]:g["lr"] for g in learner.optimizer.param_groups}
            self.assertEqual(rates["encoder"], base.learning_rate * .25)
            learner._set_phase("value")
            rates = {g["name"]:g["lr"] for g in learner.optimizer.param_groups}
            self.assertEqual(rates["encoder"], base.learning_rate * .5)
            with self.assertRaisesRegex(ValueError, "explicit frozen"):
                learner.step([sample], "value")
            anchor = make_policy_anchor(learner.model, [sample], learner.device, count=1)
            result = learner.step([sample], "value", anchor=anchor)
            self.assertTrue(result["accepted"])
            self.assertLessEqual(result["kl_proposed"], cfg.value_policy_kl_limit)
            self.assertLessEqual(result["family_kl_proposed"], cfg.value_policy_kl_limit)
            # Overlarge proposal still restores all learning state except attempt/rejection counts.
            cfg.learning_rate = .1; cfg.value_policy_kl_limit = 1e-12; cfg.value_backtracking_attempts = 1
            anchor = make_policy_anchor(learner.model, [sample], learner.device, count=1)
            learner._set_phase("value")
            before = learner.payload(role="learner", lineage={})
            result = learner.step([sample], "value", anchor=anchor)
            self.assertFalse(result["accepted"])
            after = learner.payload(role="learner", lineage={})
            after["counters"] = before["counters"]
            self.assertTrue(all(verify_learning_state(before, after).values()))


if __name__ == "__main__": unittest.main()
