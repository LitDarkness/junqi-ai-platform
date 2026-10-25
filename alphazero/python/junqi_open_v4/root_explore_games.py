"""Isolated collector; original games.py remains immutable during evaluation."""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from junqi_search_v3.loop import SearchSample
from junqi_search_v3.root_explore_search import RootExploreSearch

from .root_explore_config import RootExploreConfig as OpenConfig, root_noise_enabled
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
    search = RootExploreSearch(model, device=device, inference_batch_size=config.inference_batch_size,
                              cache_size=config.cache_size, leaves_per_tree=config.leaves_per_tree,
                              root_tactical_guard=config.root_tactical_guard,forced_coeff=config.forced_root_coeff,
                              target_pruning=config.exploration_target_pruning)
    active, completed, samples = [], [], []
    exploration=Counter();target_examples=[]
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
    label_audit = Counter(); teacher_query_audit = Counter(); actor_target_examples = []
    noise_decisions=Counter();physical_rows=padding_rows=0
    ply = 0
    while active:
        if check_stop is not None:
            check_stop()
        # Keep teacher query schedules identical before learned trajectories diverge.
        # Candidate actor-owned targets use true neural visits; its unused teacher
        # query is retained to avoid confounding the teacher's time-capped workload.
        teacher_games = [game for game in active if game["mode"] != "selfplay"]
        shadows=sum(g["mode"]=="teacher_vs_actor" and g["state"].current_player()==g["actor_side"] for g in teacher_games)
        teacher_query_audit['shadow_actor_owned_requests']+=shadows
        teacher_query_audit['actual_teacher_turn_requests']+=len(teacher_games)-shadows
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
            for r in results:
                exploration['all_neural_roots']+=1
                exploration['all_forced_decisions']+=r.forced_decisions
                exploration['visited_edges']+=int(np.count_nonzero(r.visits))
                exploration['legal_edges']+=len(r.actions)
                exploration['unmet_visited_quotas']+=int(np.count_nonzero((r.visits>0)&(r.visits<np.sqrt(2*r.priors*r.visits.sum()))))
                exploration['potential_pruned_counts']+=int(r.visits.sum()-r.pruned_visits.sum())
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
                if not np.array_equal(result.actions, legal) or np.any(result.visits < 0) or result.visits.sum() <= 0:
                    raise ValueError('true sparse search target must match legal action order and positive real visits')
                target_visits = result.target_visits
                target_p=result.target_policy
                entropy=float(-np.sum(target_p*np.log(np.maximum(target_p,1e-12))))
                exploration['search_target_rows']+=1
                exploration['raw_target_visits']+=int(result.visits.sum())
                exploration['saved_target_visits']+=int(target_visits.sum())
                exploration['deleted_visits']+=int(result.visits.sum()-target_visits.sum())
                exploration['forced_decisions']+=result.forced_decisions
                exploration['changed_target_rows']+=int(not np.array_equal(target_visits,result.visits))
                exploration['raw_entropy_sum']+=result.root_entropy
                exploration['target_entropy_sum']+=entropy
                if len(target_examples)<12:
                    target_examples.append({'family':game['family'],'index':game['index'],'ply':len(game['trace']),
                        'actions':legal.tolist(),'raw_visits':result.visits.tolist(),'target_visits':target_visits.tolist(),
                        'priors':result.priors.tolist(),'q':result.q_values.tolist(),'greedy_action':int(result.action),
                        'budget':result.simulations,'forced_decisions':result.forced_decisions,'target_type':result.target_type})
            else:
                # Teacher decisions are sparse hard labels, never invented Q values.
                target_visits = (legal == teacher_map[id(game)]).astype(np.int32)
                entropy = 0.
            own=game["mode"]=="teacher_vs_actor" and state.current_player()==game["actor_side"]
            if own:
                label_audit['actor_owned_rows']+=1
                label_audit['actor_owned_search_targets' if config.actor_policy_target=='search' else 'actor_owned_teacher_targets']+=1
                label_audit['shadow_teacher_differs_from_neural_greedy']+=int(teacher_map[id(game)]!=result_map[id(game)].action)
            contexts = state.repetition_contexts()
            game["samples"].append(SearchSample(game["family"], int(state.native_state.key()),
                state.feature_planes().astype(np.float16), legal, contexts, target_visits.copy(),
                state.current_player(), len(game["trace"]), entropy,
                raw_visits=(result_map[id(game)].visits.copy() if uses_search_target(config,game['mode'],state.current_player(),game['actor_side'])
                            else target_visits.copy()), position_key=int(state.native_state.position_key()),
                repetition_count=int(state.native_state.repetition_count()),
                cycle_entry=bool(np.any(contexts)), endgame=len(game["trace"]) >= 80))
            if id(game) in result_map:
                result = result_map[id(game)]
                action, behavior_row = behavior_action(config, result, len(game["trace"]), game["rng"])
                behavior.update(behavior_row)
            else:
                action = teacher_map[id(game)]
            if own and len(actor_target_examples) < 8:
                native_result = result_map[id(game)]
                actor_target_examples.append({'family': game['family'], 'index': game['index'], 'ply': len(game['trace']),
                    'teacher_action': teacher_map[id(game)], 'played_action': action, 'neural_greedy_action': int(native_result.action),
                    'legal_actions': legal.tolist(), 'real_neural_visits': native_result.visits.tolist(),
                    'training_target_visits': target_visits.tolist(), 'target_type': config.actor_policy_target})
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
        "selfplay_target_contract":"root_explore_integer_pruned" if config.exploration_target_pruning else "root_explore_raw",
        "root_explore_contract":{"forced_coeff":2.0,"singleton_cutoff":False,"raw_behavior":True,"additional_simulations":0},
        "root_explore_audit":dict(exploration),"root_explore_target_examples":target_examples,
        "policy_label_audit":dict(label_audit), "teacher_query_audit":dict(teacher_query_audit),
        "actor_owned_target_examples":actor_target_examples,
        "teacher_query_contract": "all non-selfplay turns, including unused candidate actor-owned teacher advice",
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
