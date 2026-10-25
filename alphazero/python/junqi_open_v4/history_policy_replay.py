"""One Policy eligibility factor; real terminal Value and neural targets stay."""
from collections import Counter

import numpy as np

from .training_replay import replay_pools


def teacher_history_safe(row):
    # Successor contexts encode a previously seen position (1) or repetition
    # terminal (2). No teacher receives this actual history through its API.
    return (not row.cycle_entry and row.repetition_count <= 1
            and not np.any(row.repetition_contexts))


def history_policy_pools(shards, *, exclude_history_sensitive_teacher, holdout):
    values, policy, audit = replay_pools(shards, deduplicate=False,
        neural_policy_only=False, holdout=holdout)
    sensitive = [r for r in policy if r.training_policy_source == 'teacher'
                 and not teacher_history_safe(r)]
    if exclude_history_sensitive_teacher:
        policy = [r for r in policy if r.training_policy_source == 'search'
                  or teacher_history_safe(r)]
    if not policy:
        raise ValueError('empty history-qualified Policy pool')
    audit.update(exclude_history_sensitive_teacher=bool(exclude_history_sensitive_teacher),
        history_sensitive_teacher_rows=len(sensitive),
        excluded_policy_rows=len(sensitive) if exclude_history_sensitive_teacher else 0,
        excluded_teacher_wdl_classes=dict(Counter(int(np.argmax(r.wdl)) for r in sensitive))
            if exclude_history_sensitive_teacher else {},
        policy_rows=len(policy), policy_sources=dict(Counter(r.training_policy_source for r in policy)),
        history_sensitive_search_rows_retained=sum(r.training_policy_source == 'search'
            and not teacher_history_safe(r) for r in policy),
        value_pool_unchanged=True, deduplicate=False)
    return values, policy, audit
