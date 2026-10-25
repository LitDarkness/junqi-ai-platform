from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from junqi_az.config import ModelConfig
from junqi_az.layout_policy import mirror_layout
from junqi_open_v4.config import OpenConfig
from junqi_open_v4.layouts import backend_for, create_manifest, load_role
from junqi_open_v4.learner import Learner, atomic_checkpoint, model_hash
from junqi_open_v4.pipeline import promotion_decision
from junqi_search_v3.loop import SearchSample
from junqi_search_v3.recovery import make_policy_anchor


def small_config():
    return OpenConfig(device="cpu", initial_checkpoint=None, logical_batch_size=4, microbatch_size=2, teacher_candidates=["fixture"],
        model=ModelConfig(input_channels=107, channels=16, residual_blocks=0, policy_channels=8,
                          value_head_type="wdl", repetition_context=True, pairwise_action_bias=True))


class OpenV4Tests(unittest.TestCase):
    def test_layout_partitions_and_reflection_preserve_rules_and_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            create_manifest(root, 177, train=4, dev=4, final=4)
            train, dev = load_role(root, "train"), load_role(root, "dev")
            final = load_role(root, "final", final_evaluation=True)
            self.assertEqual(len({row["family"] for row in train + dev + final}), 12)
            with self.assertRaises(ValueError):
                load_role(root, "final")
            rng = np.random.default_rng(299)
            for red, black in zip(train, train[1:] + train[:1]):
                state = backend_for(red, black).new_state(1)
                reflected = backend_for(red, black, mirrored=True).new_state(1)
                for ply in range(160):
                    np.testing.assert_array_equal(state.feature_planes()[:, :, ::-1], reflected.feature_planes())
                    legal = state.legal_actions()
                    def mirror_action(action):
                        source, target = divmod(int(action), 60)
                        return (source // 5 * 5 + 4 - source % 5) * 60 + target // 5 * 5 + 4 - target % 5
                    mapped = [mirror_action(action) for action in legal]
                    actual = dict(zip(reflected.legal_actions(), reflected.repetition_contexts()))
                    self.assertEqual(set(mapped), set(actual))
                    self.assertEqual(list(map(int, state.repetition_contexts())), [int(actual[a]) for a in mapped])
                    if state.is_terminal():
                        break
                    action = int(rng.choice(legal))
                    state.apply(action)
                    reflected.apply(mirror_action(action))
                    self.assertEqual(state.is_terminal(), reflected.is_terminal())
                    self.assertEqual(state.terminal_reason(), reflected.terminal_reason())
                if state.is_terminal():
                    self.assertEqual(state.terminal_value(1), reflected.terminal_value(1))

    def _sample(self, root):
        create_manifest(root / "layouts", 133, train=2, dev=2, final=2)
        rows = load_role(root / "layouts", "train")
        state = backend_for(*rows).new_state(1)
        legal = state.legal_actions()
        visits = np.zeros(len(legal), np.int32)
        visits[0] = 1
        return SearchSample(1, int(state.native_state.key()), state.feature_planes(), legal,
                            state.repetition_contexts(), visits, 1, 0, 0., wdl=np.array([1.,0.,0.], np.float32))

    def test_rejected_value_update_restores_weights_optimizer_and_counters(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            config.learning_rate = .1
            config.value_policy_kl_limit = 1e-12
            config.value_backtracking_attempts = 1
            learner = Learner(config, root)
            sample = self._sample(root)
            anchor = make_policy_anchor(learner.model, [sample], learner.device, count=1)
            before = model_hash(learner.model)
            result = learner.step([sample], "value", anchor=anchor)
            self.assertFalse(result["accepted"])
            self.assertEqual(before, model_hash(learner.model))
            self.assertEqual(len(learner.optimizer.state), 0)
            self.assertEqual(learner.counters["effective_updates"], 0)
            self.assertEqual(learner.counters["backtracking_rejections"], 1)

    def test_full_checkpoint_resume_reproduces_next_update(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            learner = Learner(config, root)
            sample = self._sample(root)
            learner.step([sample], "policy")
            path = root / "learner.pt"
            atomic_checkpoint(path, learner.payload(role="learner", lineage={"actor": 0}))
            learner.step(learner._batch([sample]), "policy")
            expected = model_hash(learner.model)
            resumed = Learner(config, root)
            self.assertEqual(resumed.resume(path), {"actor": 0})
            resumed.step(resumed._batch([sample]), "policy")
            self.assertEqual(model_hash(resumed.model), expected)
            self.assertEqual(resumed.counters, learner.counters)

    def test_microbatch_preserves_full_logical_batch_gradients(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            first = Learner(config, root)
            full_config = copy.deepcopy(config)
            full_config.microbatch_size = 4
            full = Learner(full_config, root)
            sample = self._sample(root)
            samples = []
            for index in range(4):
                row = copy.deepcopy(sample)
                row.visits = np.roll(row.visits, index)
                row.wdl = np.eye(3, dtype=np.float32)[index % 3]
                samples.append(row)
            for phase in ("policy", "value"):
                first._set_phase(phase)
                full._set_phase(phase)
                a = first._backward(samples, phase, None)
                b = full._backward(samples, phase, None)
                self.assertAlmostEqual(a["loss"], b["loss"], places=5)
                for left, right in zip(first.model.parameters(), full.model.parameters()):
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)

    def test_phase_schedule_continues_across_checkpoint_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = small_config()
            learner = Learner(config, root)
            sample = self._sample(root)
            first = learner.train([sample], updates=2)
            path = root / "phase.pt"
            atomic_checkpoint(path, learner.payload(role="learner", lineage={}))
            resumed = Learner(config, root)
            resumed.resume(path)
            second = resumed.train([sample], updates=2)
            self.assertEqual([row["phase"] for row in first + second], ["policy", "policy", "value", "policy"])

    def test_gate_separates_actor_champion_and_forbids_final(self):
        config = small_config()
        arena = {"role": "dev", "summaries": [
            {"candidate": "learner", "opponent": "actor", "family_score": {"families": 32, "ci95": [.47,.60]}},
            {"candidate": "actor", "opponent": "teacher0"}, {"candidate": "learner", "opponent": "teacher0"}],
            "paired_deltas": [{"from_matchup": 1, "to_matchup": 2, "delta": {"families": 32, "ci95": [-.01,.03]}}]}
        decision = promotion_decision(arena, config)
        self.assertTrue(decision["actor_accepted"])
        self.assertFalse(decision["champion_accepted"])
        self.assertFalse(decision["learner_rollback"])
        arena["role"] = "final"
        with self.assertRaises(ValueError):
            promotion_decision(arena, config)


if __name__ == "__main__":
    unittest.main()
