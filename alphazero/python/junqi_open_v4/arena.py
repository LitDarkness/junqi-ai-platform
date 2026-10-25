from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from junqi_search_v3.dataset import atomic_json

from .games import TeacherLeague
from .layouts import backend_for, load_role


def paired_interval(values, seed: int, *, resamples: int = 4000) -> dict[str, Any]:
    values = np.asarray(values, np.float64)
    if not len(values):
        raise ValueError("interval requires family observations")
    if not np.isfinite(values).all():
        raise ValueError("nonfinite family observations")
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(resamples)
    for start in range(0, resamples, 256):
        count = min(256, resamples - start)
        bootstrap[start:start + count] = values[rng.integers(0, len(values), (count, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": list(map(float, np.quantile(bootstrap, [.025, .975]))),
            "families": len(values), "unit": "disjoint pair of layout families; all starts and both colors clustered",
            "uncertainty_estimable": len(values) >= 2}


def create_dev_suite(directory: Path, layout_directory: Path, teachers: TeacherLeague,
                     seed: int, *, prefixes: tuple[int, ...] = (0, 24, 48)) -> list[dict[str, Any]]:
    path = directory / "suite.json"
    layout_manifest_hash = hashlib.sha256((layout_directory / "manifest.json").read_bytes()).hexdigest()
    contract = {"seed": seed, "layout_manifest_hash": layout_manifest_hash,
                "teacher_provenance": teachers.provenance, "prefixes": list(prefixes), "role": "dev"}
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["contract"] != contract:
            raise ValueError("development suite contract changed")
        if hashlib.sha256(json.dumps(payload["states"], sort_keys=True).encode()).hexdigest() != payload["states_hash"]:
            raise ValueError("development suite digest mismatch")
        return payload["states"]
    layouts = load_role(layout_directory, "dev")
    if len(layouts) % 2:
        raise ValueError("development suite needs disjoint pairs of families")
    states = []
    active = []
    for family in range(len(layouts) // 2):
        red, black = layouts[family * 2:family * 2 + 2]
        state = backend_for(red, black).new_state(seed + family)
        active.append(dict(family=family, state=state, trace=[], red=red, black=black,
                           setup_seed=seed + family))
    for ply in range(max(prefixes) + 1):
        if ply in prefixes:
            for game in active:
                states.append({k: v for k, v in game.items() if k != "state"} | {
                    "trace": game["trace"].copy(), "start_ply": ply,
                    "state_key": int(game["state"].native_state.key())})
        if ply == max(prefixes) or not active:
            break
        actions = teachers.actions([g["state"] for g in active], [g["family"] % len(teachers.modules) for g in active])
        next_active = []
        for game, action in zip(active, actions):
            game["state"].apply(action)
            game["trace"].append(action)
            if not game["state"].is_terminal():
                next_active.append(game)
        active = next_active
    states_hash = hashlib.sha256(json.dumps(states, sort_keys=True).encode()).hexdigest()
    atomic_json(path, {"schema": "junqi-development-suite-v4", "contract": contract,
                       "states_hash": states_hash, "states": states})
    return states


def select_families(suite: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    families = sorted({row["family"] for row in suite})
    rng = np.random.default_rng(seed)
    chosen = set(map(int, rng.permutation(families)[:count]))
    return [row for row in suite if row["family"] in chosen]


def run_arena(suite: list[dict[str, Any]], matchups: list[tuple[str, str]],
              agents: dict[str, Any], teachers: TeacherLeague, budgets: dict[str, int],
              *, seed: int, progress: bool = True) -> dict[str, Any]:
    if not suite or any(row.get("role", "dev") == "final" for row in suite):
        raise ValueError("promotion arena requires a nonempty development suite")
    for search in agents.values():
        search.clear_trees()
    active, completed = [], []
    for position, descriptor in enumerate(suite):
        for matchup, (candidate, opponent) in enumerate(matchups):
            for side in (1, -1):
                state = backend_for(descriptor["red"], descriptor["black"]).new_state(descriptor["setup_seed"])
                for action in descriptor["trace"]:
                    state.apply(int(action))
                if state.is_terminal() or int(state.native_state.key()) != descriptor["state_key"]:
                    raise ValueError("suite reconstruction or history mismatch")
                active.append(dict(id=len(active), position=position, family=descriptor["family"],
                    start_ply=descriptor["start_ply"], matchup=matchup, candidate=candidate, opponent=opponent,
                    candidate_side=side, state=state, trace=[], candidate_seconds=0., opponent_seconds=0.))
    metrics = {name: defaultdict(float) for name in agents}
    started = time.perf_counter()
    rounds = 0
    while active:
        groups = defaultdict(list)
        for game in active:
            name = game["candidate"] if game["state"].current_player() == game["candidate_side"] else game["opponent"]
            groups[name].append(game)
        teacher_batches = {}
        for name, games in groups.items():
            if name.startswith("teacher"):
                teacher_id = int(name.removeprefix("teacher"))
                teacher_batches[name] = teachers.submit_actions([g["state"] for g in games], [teacher_id] * len(games))
        # Neural groups keep their original order and sequential MCTS semantics.
        # CPU jobs are already running, and every result resolves before apply.
        ordered_groups = [(name, games) for name, games in groups.items() if name not in teacher_batches]
        ordered_groups += [(name, games) for name, games in groups.items() if name in teacher_batches]
        for name, games in ordered_groups:
            before = time.perf_counter()
            states = [g["state"] for g in games]
            if name.startswith("teacher"):
                actions = teacher_batches[name].result()
            else:
                search = agents[name]
                if budgets[name]:
                    results, report = search.search(states, budgets[name], tree_ids=[g["id"] for g in games])
                    for key in ("elapsed_seconds", "inference_calls", "inference_rows", "cache_hits", "deduplicated_rows", "reused_roots", "inherited_root_visits"):
                        metrics[name][key] += report[key]
                    metrics[name]["simulations"] += budgets[name] * len(games)
                else:
                    results = search.bare(states)
                actions = [r.action for r in results]
                metrics[name]["moves"] += len(games)
            elapsed = time.perf_counter() - before
            if name in teacher_batches:
                elapsed = teacher_batches[name].elapsed_seconds
            for game, action in zip(games, actions):
                game["selected"] = action
                role = "candidate" if game["state"].current_player() == game["candidate_side"] else "opponent"
                game[role + "_seconds"] += elapsed / len(games)
        next_active = []
        for game in active:
            state = game["state"]
            action = game.pop("selected")
            if action not in state.legal_actions():
                raise RuntimeError("arena selected an illegal action")
            state.apply(action)
            game["trace"].append(action)
            for name in {game["candidate"], game["opponent"]}:
                if name in agents and agents[name].reuse_roots:
                    agents[name].advance_tree(game["id"], action, state)
            if not state.is_terminal():
                if game["start_ply"] + len(game["trace"]) >= 1000:
                    raise RuntimeError("maximum-ply terminal contract failed")
                next_active.append(game)
            else:
                row = {k: v for k, v in game.items() if k != "state"}
                row.update(value=state.terminal_value(game["candidate_side"]), reason=state.terminal_reason())
                row["score"] = .5 + .5 * row["value"]
                completed.append(row)
        active = next_active
        rounds += 1
        if progress and (rounds % 20 == 0 or not active):
            print(f"[arena] round={rounds} active={len(active)} completed={len(completed)}", flush=True)
    summaries, scores = [], {}
    for matchup, names in enumerate(matchups):
        games = [row for row in completed if row["matchup"] == matchup]
        families = defaultdict(list)
        starts = defaultdict(list)
        for row in games:
            families[row["family"]].append(row["score"])
            starts[row["start_ply"]].append(row["score"])
        scores[matchup] = {family: float(np.mean(rows)) for family, rows in families.items()}
        interval = paired_interval(list(scores[matchup].values()), seed ^ matchup)
        summaries.append({"candidate": names[0], "opponent": names[1], "games": len(games),
            "wins": sum(row["value"] > 0 for row in games), "draws": sum(row["value"] == 0 for row in games),
            "losses": sum(row["value"] < 0 for row in games), "score": float(np.mean([r["score"] for r in games])),
            "family_score": interval, "score_by_start_ply": {str(k): float(np.mean(v)) for k, v in starts.items()},
            "repetition_fraction": float(np.mean([r["reason"] == "REPETITION" for r in games]))})
    pairs = []
    for first in range(len(matchups)):
        for second in range(first + 1, len(matchups)):
            if matchups[first][1] != matchups[second][1]:
                continue
            common = sorted(scores[first].keys() & scores[second].keys())
            pairs.append({"from_matchup": first, "to_matchup": second,
                          "delta": paired_interval([scores[second][f] - scores[first][f] for f in common], seed ^ (first << 8) ^ second)})
    for name, report in metrics.items():
        report["simulations_per_second"] = report["simulations"] / max(report["elapsed_seconds"], 1e-9)
        report["mean_inference_batch"] = report["inference_rows"] / max(report["inference_calls"], 1)
    return {"schema": "junqi-development-arena-v4", "role": "dev", "seed": seed,
        "suite_digest": hashlib.sha256(json.dumps(suite, sort_keys=True).encode()).hexdigest(),
        "layout_families": sorted({row["family"] for row in suite}), "budget_comparison": "fixed added simulations; wall time also reported",
        "elapsed_seconds": time.perf_counter() - started, "summaries": summaries, "paired_deltas": pairs,
        "teacher_workers": teachers.workers, "teacher_execution": "spawned CPU processes overlapping neural search",
        "search_metrics": {k: dict(v) for k, v in metrics.items()}, "games": completed}
