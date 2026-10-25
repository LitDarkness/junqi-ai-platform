"""Keep neural search targets and optionally add actual teacher-owned turns.

Legacy actor turns labelled by a teacher stay Value-only. Adding an anchor
never converts those imitation targets into the actor's own search targets.
"""
from collections import Counter

from .training_replay import complete_games, replay_pools


def policy_anchor_pools(shards, *, deduplicate, teacher_policy_anchor, holdout):
    for samples, metadata in shards:
        for game, rows in complete_games(samples, metadata):
            for row in rows:
                row.training_teacher_owned_turn = (
                    game["mode"] == "teacher_selfplay" or
                    (game["mode"] == "teacher_vs_actor" and row.player != game["actor_side"]))
    values, policy, audit = replay_pools(shards, deduplicate=deduplicate,
                                        neural_policy_only=True, holdout=holdout)
    if teacher_policy_anchor:
        policy = [row for row in values if row.training_policy_source == "search"
                  or row.training_teacher_owned_turn]
        audit.update(policy_rows=len(policy),
                     policy_sources=dict(Counter(row.training_policy_source for row in policy)))
    audit["teacher_policy_anchor"] = bool(teacher_policy_anchor)
    audit["legacy_actor_imitation_rows_excluded"] = sum(
        row.training_policy_source == "teacher" and not row.training_teacher_owned_turn for row in values)
    return values, policy, audit
