"""Isolated collector; original games.py remains immutable during evaluation."""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from junqi_search_v3.loop import SearchSample
from junqi_search_v3.search import BatchedNeuralPUCT

from .queued_config import QueuedConfig as OpenConfig, root_noise_enabled
from .layouts import MultiLayoutBackend
from .teachers import TeacherLeague


def training_mode(config: OpenConfig, key: float) -> str:
    teacher_fraction = ((1 - config.selfplay_fraction) / 3 if config.teacher_selfplay_fraction is None
                        else config.teacher_selfplay_fraction)
    return "selfplay" if key < config.selfplay_fraction else (
        "teacher_selfplay" if key < config.selfplay_fraction + teacher_fraction else "teacher_vs_actor")


def uses_search_target(config: OpenConfig, mode: str, player: int, actor_side: int) -> bool:
    return mode == "selfplay" or (mode == "teacher_vs_actor" and player == actor_side
                                  and config.actor_policy_target == "search")


def behavior_action(config, result, ply, rng):
    """Change behavior only; keep every search visit target intact.

    Consume the original stochastic draw in both branches for paired RNG keys.
    Greedy still comes from a noisy search, and can still be a mistaken action.
    """
    probability = result.policy.astype(np.float64)
    if ply >= 24:
        probability = np.square(probability)
    probability /= probability.sum()
    sampled = int(rng.choice(result.actions, p=probability))
    greedy = config.training_greedy_after_ply is not None and ply >= config.training_greedy_after_ply
    action = int(result.action) if greedy else sampled
    return action, dict(neural_decisions=1, greedy_decisions=int(greedy),
                       stochastic_decisions=int(not greedy),
                       sample_greedy_disagreements=int(sampled != result.action),
                       actual_non_greedy_decisions=int(action != result.action),
                       behavior_entropy=float(-np.sum(probability*np.log(np.maximum(probability,1e-12)))) if not greedy else 0.,
                       search_entropy=float(result.root_entropy))


def collect_training_games(config: OpenConfig, model, device, backend: MultiLayoutBackend,
                           teachers: TeacherLeague, iteration: int, *, count: int | None = None,
                           check_stop=None):
    count = config.games_per_iteration if count is None else count
    search = BatchedNeuralPUCT(model, device=device, inference_batch_size=config.inference_batch_size,
                              cache_size=config.cache_size, leaves_per_tree=config.leaves_per_tree,
                              root_tactical_guard=config.root_tactical_guard)
    active, completed, samples = [], [], []
    for index in range(count):
        # Both actor colors share the same layout pair, and stay in one split.
        pair_id = iteration * 1_000_000 + index // 2
        state = backend.new_state(config.seed + pair_id)
        rng = np.random.default_rng(config.seed ^ (iteration << 20) ^ index)
        pair_rng = np.random.default_rng(config.seed ^ pair_id)
        mode_key = pair_rng.random()
        mode = training_mode(config, mode_key)
        active.append(dict(index=index, family=pair_id, state=state, rng=rng, mode=mode,
                           actor_side=1 if index % 2 == 0 else -1, teacher_id=(index // 2) % len(teachers.modules),
                           samples=[], trace=[], provenance=state.provenance()))
    started = time.perf_counter()
    simulations = inference_rows = inference_calls = 0
    neural_seconds = teacher_seconds = teacher_wait_seconds = 0.
    tactical = Counter()
    behavior = Counter()
    noise_decisions=Counter();physical_rows=padding_rows=0
    ply = 0
    while active:
        if check_stop is not None:
            check_stop()
        teacher_games = [game for game in active if game["mode"] != "selfplay" and not (
            config.actor_policy_target == "search" and game["mode"] == "teacher_vs_actor"
            and game["state"].current_player() == game["actor_side"])]
        teacher_batch = teachers.submit_actions([g["state"] for g in teacher_games], [g["teacher_id"] for g in teacher_games])
        neural_games = [game for game in active if game["mode"] == "selfplay" or (
            game["mode"] == "teacher_vs_actor" and game["state"].current_player() == game["actor_side"])]
        result_map = {}
        for budget in sorted(set((config.simulations, config.repeated_simulations))):
            selected = [game for game in neural_games if (
                config.repeated_simulations if np.any(game["state"].repetition_contexts() > 0) else config.simulations
            ) == budget]
            if not selected:
                continue
            results, metrics = search.search([g["state"] for g in selected], budget, add_root_noise=root_noise_enabled(config,ply),
                noise_seeds=[config.seed ^ (iteration << 20) ^ g["index"] ^ (ply * 65537) for g in selected])
            result_map.update((id(g), r) for g, r in zip(selected, results))
            simulations += budget * len(selected)
            neural_seconds += metrics["elapsed_seconds"]
            inference_rows += metrics["inference_rows"]
            inference_calls += metrics["inference_calls"]
            physical_rows+=metrics["physical_inference_rows"]
            padding_rows+=metrics["physical_inference_rows"]-metrics["inference_rows"]
            noise_decisions["with_noise" if root_noise_enabled(config,ply) else "without_noise"]+=len(selected)
            for name in ("tactical_roots", "tactical_native_transitions", "tactical_immediate_wins", "tactical_disabled_actions"):
                tactical[name] += metrics[name]
        before = time.perf_counter()
        teacher_actions = teacher_batch.result()
        teacher_wait_seconds += time.perf_counter() - before
        teacher_seconds += teacher_batch.elapsed_seconds
        teacher_map = {id(game): action for game, action in zip(teacher_games, teacher_actions)}
        next_active = []
        for game in active:
            state = game["state"]
            legal = state.legal_actions()
            if uses_search_target(config, game["mode"], state.current_player(), game["actor_side"]):
                result = result_map[id(game)]
                target_visits = result.visits
                entropy = result.root_entropy
            else:
                # Teacher decisions are sparse hard labels, never invented Q values.
                target_visits = (legal == teacher_map[id(game)]).astype(np.int32)
                entropy = 0.
            contexts = state.repetition_contexts()
            game["samples"].append(SearchSample(game["family"], int(state.native_state.key()),
                state.feature_planes().astype(np.float16), legal, contexts, target_visits.copy(),
                state.current_player(), len(game["trace"]), entropy,
                raw_visits=target_visits.copy(), position_key=int(state.native_state.position_key()),
                repetition_count=int(state.native_state.repetition_count()),
                cycle_entry=bool(np.any(contexts)), endgame=len(game["trace"]) >= 80))
            if id(game) in result_map:
                result = result_map[id(game)]
                action, behavior_row = behavior_action(config, result, len(game["trace"]), game["rng"])
                behavior.update(behavior_row)
            else:
                action = teacher_map[id(game)]
            state.apply(action)
            game["trace"].append(action)
            if not state.is_terminal():
                if len(game["trace"]) >= 1000:
                    raise RuntimeError("native maximum-ply terminal contract failed")
                next_active.append(game)
            else:
                for sample in game["samples"]:
                    sample.wdl = np.eye(3, dtype=np.float32)[int(state.terminal_value(sample.player)) + 1]
                samples.extend(game["samples"])
                completed.append({k: v for k, v in game.items() if k not in ("state", "rng", "samples")})
                completed[-1].update(reason=state.terminal_reason(), result_red=state.terminal_value(1),
                                     terminal_key=int(state.native_state.key()))
        active = next_active
        ply += 1
        if ply % 20 == 0 or not active:
            print(f"[collect] iteration={iteration} round={ply} active={len(active)} completed={len(completed)}/{count}", flush=True)
    elapsed = time.perf_counter() - started
    plies = sum(len(g["trace"]) for g in completed)
    return samples, {"iteration": iteration, "games": completed, "game_count": count,
        "label_sources": dict(Counter(g["mode"] for g in completed)), "teacher_provenance": teachers.provenance,
        "layout_role": "train", "policy_samples": len(samples), "total_plies": plies,
        "policy_target_contract": config.actor_policy_target,
        "neural_search_contract": "native_root_tactical_guard" if config.root_tactical_guard else "standard_puct",
        "tactical_metrics": dict(tactical),
        "behavior_contract": {"greedy_after_ply": config.training_greedy_after_ply,
                              "root_noise_until_ply": config.root_noise_until_ply, "noise_decisions":dict(noise_decisions),
                              "policy_targets_changed_by_noise_schedule":config.root_noise_until_ply is not None,
                              "policy_targets_changed_by_behavior": False},
        "behavior_metrics": dict(behavior),
        "elapsed_seconds": elapsed, "moves_per_second": plies / max(elapsed, 1e-9),
        "physical_inference_rows":physical_rows,"padding_rows":padding_rows,
        "simulation_count": simulations, "neural_seconds": neural_seconds, "teacher_seconds": teacher_seconds,
        "teacher_wait_seconds": teacher_wait_seconds, "teacher_workers": teachers.workers,
        "teacher_execution": "spawned CPU processes overlapping neural search",
        "simulations_per_second": simulations / max(neural_seconds, 1e-9),
        "mean_inference_batch": inference_rows / max(1, inference_calls),
        "reasons": dict(Counter(g["reason"] for g in completed))}
