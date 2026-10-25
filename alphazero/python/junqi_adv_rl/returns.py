from __future__ import annotations

import numpy as np


def magnet_distribution(legal_actions: np.ndarray) -> np.ndarray:
    """Uniform movable piece, then uniform legal destination for that piece."""
    actions = np.asarray(legal_actions, dtype=np.int64)
    if actions.ndim != 1 or len(actions) == 0:
        raise ValueError("legal_actions must be a non-empty vector")
    sources = actions // 60
    unique, counts = np.unique(sources, return_counts=True)
    source_mass = 1.0 / float(len(unique))
    count_by_source = dict(zip(unique.tolist(), counts.tolist()))
    rho = np.asarray(
        [source_mass / count_by_source[int(source)] for source in sources],
        dtype=np.float64,
    )
    rho /= rho.sum()
    return rho.astype(np.float32)


def compute_player_lambda_returns(
    acting_sides: np.ndarray,
    old_values: np.ndarray,
    old_wdl: np.ndarray,
    terminal_outcome_red: float,
    *,
    lambda_advantage: float = 0.5,
    lambda_outcome: float = 0.8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return scalar G, advantage, and WDL xi on each player's own timeline.

    WDL vectors are ordered (loss, draw, win), always from the acting side.
    """
    sides = np.asarray(acting_sides, dtype=np.int8)
    values = np.asarray(old_values, dtype=np.float64)
    wdl = np.asarray(old_wdl, dtype=np.float64)
    if sides.ndim != 1 or values.shape != sides.shape or wdl.shape != (len(sides), 3):
        raise ValueError("lambda-return arrays do not align")
    if not np.isin(sides, (-1, 1)).all():
        raise ValueError("acting sides must be RED(+1) or BLACK(-1)")
    scalar_returns = np.empty(len(sides), dtype=np.float64)
    vector_returns = np.empty((len(sides), 3), dtype=np.float64)
    for side in (1, -1):
        indices = np.flatnonzero(sides == side)
        if not len(indices):
            continue
        outcome = float(terminal_outcome_red) * float(side)
        terminal_wdl = np.zeros(3, dtype=np.float64)
        terminal_wdl[2 if outcome > 0.5 else 0 if outcome < -0.5 else 1] = 1.0
        next_g = outcome
        next_xi = terminal_wdl
        for reverse_position, index in enumerate(indices[::-1]):
            if reverse_position == 0:
                current_g = outcome
                current_xi = terminal_wdl
            else:
                next_index = indices[len(indices) - reverse_position]
                current_g = (
                    (1.0 - lambda_advantage) * values[next_index]
                    + lambda_advantage * next_g
                )
                current_xi = (
                    (1.0 - lambda_outcome) * wdl[next_index]
                    + lambda_outcome * next_xi
                )
            scalar_returns[index] = current_g
            vector_returns[index] = current_xi / current_xi.sum()
            next_g = current_g
            next_xi = current_xi
    advantages = scalar_returns - values
    return (
        scalar_returns.astype(np.float32),
        advantages.astype(np.float32),
        vector_returns.astype(np.float32),
    )


def filter_advantages(
    advantages: np.ndarray,
    policy_eligible: np.ndarray,
    *,
    percentile: float = 75.0,
    minimum_absolute: float = 0.01,
) -> tuple[np.ndarray, dict[str, float]]:
    advantages = np.asarray(advantages, dtype=np.float32)
    eligible = np.asarray(policy_eligible, dtype=np.bool_)
    pool = np.abs(advantages[eligible])
    threshold = max(
        float(minimum_absolute),
        float(np.percentile(pool, percentile)) if len(pool) else float("inf"),
    )
    retained = eligible & (np.abs(advantages) >= threshold)
    kept = advantages[retained]
    quantiles = (
        np.percentile(pool, [0, 25, 50, 75, 90, 95, 99, 100]).tolist()
        if len(pool)
        else [0.0] * 8
    )
    return retained, {
        "policy_samples_before": int(eligible.sum()),
        "policy_samples_after": int(retained.sum()),
        "filtered_sample_ratio": float(retained.sum() / max(1, eligible.sum())),
        "advantage_positive_fraction": float((kept > 0).mean()) if len(kept) else 0.0,
        "advantage_negative_fraction": float((kept < 0).mean()) if len(kept) else 0.0,
        "advantage_threshold": threshold,
        "advantage_abs_q0": quantiles[0],
        "advantage_abs_q25": quantiles[1],
        "advantage_abs_q50": quantiles[2],
        "advantage_abs_q75": quantiles[3],
        "advantage_abs_q90": quantiles[4],
        "advantage_abs_q95": quantiles[5],
        "advantage_abs_q99": quantiles[6],
        "advantage_abs_q100": quantiles[7],
    }
