"""History-safe trajectory deduplication and explicit self-improvement targets.

Raw shards remain untouched. No draw is discarded or relabelled because it is
a draw; only byte-identical complete training examples share one replay entry.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import numpy as np


def complete_games(samples, metadata):
    if metadata.get("layout_role") != "train":
        raise ValueError("only training trajectories can enter replay")
    offset = 0
    for game in metadata["games"]:
        if game["mode"] not in ("selfplay", "teacher_selfplay", "teacher_vs_actor"):
            raise ValueError("unknown game source")
        rows = samples[offset:offset + len(game["trace"])]
        offset += len(rows)
        if len(rows) != len(game["trace"]) or not rows:
            raise ValueError("complete nonempty trajectories required")
        for ply, row in enumerate(rows):
            if row.game_id != game["family"] or row.ply != ply:
                raise ValueError("trajectory/sample alignment changed")
            expected = np.eye(3, dtype=np.float32)[int(game["result_red"] * row.player) + 1]
            if not np.array_equal(row.wdl, expected):
                raise ValueError("terminal WDL labels changed")
        yield game, rows
    if offset != len(samples):
        raise ValueError("unassigned replay samples")


def trajectory_digest(game, rows, contract):
    # Include full move history, initial board, behavior, labels and targets.
    # Identical boards with different histories or search targets are distinct.
    body = {"board": game["provenance"]["initial_board"], "trace": game["trace"],
            "mode": game["mode"], "contract": contract,
            "result_red": game["result_red"], "reason": game["reason"]}
    if game["mode"] == "teacher_vs_actor":
        body["actor_side"] = game["actor_side"]
    h = hashlib.sha256(json.dumps(body, sort_keys=True).encode())
    for row in rows:
        for value, dtype in ((row.observation, np.float16), (row.legal_actions, np.int16),
                             (row.repetition_contexts, np.uint8), (row.visits, np.int32),
                             (row.wdl, np.float32)):
            data = np.ascontiguousarray(value, dtype=dtype)
            h.update(str(data.shape).encode()); h.update(data.tobytes())
        h.update(str((row.player, row.ply, row.repetition_count)).encode())
    return h.hexdigest()


def replay_pools(shards, *, deduplicate, neural_policy_only, holdout):
    values, policy, seen = [], [], set()
    counts, reasons, sources = Counter(), Counter(), Counter()
    # Input is newest first so an identical older trajectory cannot replace it.
    for samples, metadata in shards:
        contract = metadata.get("policy_target_contract", "teacher")
        if contract not in ("teacher", "search"):
            raise ValueError("unknown policy target contract")
        for game, rows in complete_games(samples, metadata):
            counts["raw_games"] += 1; counts["raw_rows"] += len(rows)
            if holdout(int(game["family"])):
                counts["internal_holdout_games"] += 1
                continue
            digest = trajectory_digest(game, rows, contract) if deduplicate else None
            if deduplicate and digest in seen:
                counts["duplicate_games_removed"] += 1
                counts["duplicate_rows_removed"] += len(rows)
                continue
            if deduplicate: seen.add(digest)
            counts["retained_games"] += 1
            reasons[game["reason"]] += 1
            values.extend(rows)
            for row in rows:
                neural = game["mode"] == "selfplay" or (game["mode"] == "teacher_vs_actor"
                    and row.player == game["actor_side"] and contract == "search")
                source = "search" if neural else "teacher"
                row.training_policy_source = source
                if not neural_policy_only or neural:
                    policy.append(row); sources[source] += 1
    if not values or not policy:
        raise ValueError("empty Value or Policy training pool")
    return values, policy, {**dict(counts), "value_rows": len(values), "policy_rows": len(policy),
                           "policy_sources": dict(sources), "retained_terminal_reasons": dict(reasons)}


def make_self_training_learner(config, root):
    from .learner import Learner
    from junqi_search_v3.recovery import _family_balanced_select

    class SelfTrainingLearner(Learner):
        policy_pool = None

        def _batch(self, samples):
            phase = "value" if self.counters["attempted_updates"] % 3 == 2 else "policy"
            pool = self.policy_pool if phase == "policy" else samples
            if pool is None: raise ValueError("explicit Policy pool required")
            keys = self.generator.integers(0, 2**63 - 1, size=self.config.logical_batch_size)
            selected = _family_balanced_select(pool, len(keys), keys, family_cap_fraction=.02)
            if phase == "policy":
                self.policy_batch_sources.update(getattr(s, "training_policy_source", "unknown") for s in selected)
            return selected

        def train_pools(self, values, policy, updates=None):
            self.policy_pool = policy
            self.policy_batch_sources = Counter()
            try:
                return self.train(values, updates=updates)
            finally:
                self.policy_pool = None

    return SelfTrainingLearner(config, root)
