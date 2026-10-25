from __future__ import annotations

import math
import hashlib
from collections import OrderedDict
import time
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

import JunqiAZ


@dataclass
class SearchResult:
    action: int
    actions: np.ndarray
    visits: np.ndarray
    priors: np.ndarray
    q_values: np.ndarray
    policy: np.ndarray
    root_entropy: float
    simulations: int


class BatchedNeuralPUCT:
    """C++ traversal/state transitions with batched Python neural inference."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        device: torch.device,
        cpuct: float = 1.5,
        inference_batch_size: int = 256,
        cache_size: int = 0,
        leaves_per_tree: int = 1,
        reuse_roots: bool = False,
        max_retained_trees: int = 256,
        cuda_graphs: bool = True,
        root_tactical_guard: bool = False,
        fixed_inference_batch_size: int = 0,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.cpuct = float(cpuct)
        self.inference_batch_size = max(1, int(inference_batch_size))
        if not 1 <= leaves_per_tree <= 256 or cache_size < 0 or max_retained_trees < 1:
            raise ValueError("invalid search resource limits")
        self.cache_size = int(cache_size)
        self.leaves_per_tree = int(leaves_per_tree)
        self.reuse_roots = bool(reuse_roots)
        self.root_tactical_guard = bool(root_tactical_guard)
        if not 0 <= fixed_inference_batch_size <= 256:
            raise ValueError("fixed inference batch size must be0(default dynamic) or1..256")
        self.fixed_inference_batch_size = int(fixed_inference_batch_size)
        self._physical_inference_rows = 0
        if self.root_tactical_guard and self.reuse_roots:
            raise ValueError("tactical guarded searches require fresh roots")
        self.max_retained_trees = int(max_retained_trees)
        self._cache = OrderedDict()
        self._trees = OrderedDict()
        self._revision = None
        self._tree_contract = None
        self.cuda_graphs = bool(cuda_graphs and device.type == "cuda")
        self._graphs = {}
        self._graph_failures = set()
        self._graph_replays = 0

    def _refresh_revision(self) -> None:
        revision = (repr(getattr(self.model, "config", None)),
                    tuple((id(p), p._version, str(p.device), str(p.dtype)) for p in self.model.parameters()))
        if revision != self._revision:
            self._cache.clear()
            self._trees.clear()
            self._graphs.clear()
            self._graph_failures.clear()
            self._revision = revision
        self.model.eval()

    @staticmethod
    def _request_key(request: dict[str, Any]) -> bytes:
        digest = hashlib.blake2b(digest_size=32)
        digest.update(int(request["state_key"]).to_bytes(8, "little"))
        digest.update(np.asarray(request["features"], np.float32).tobytes())
        digest.update(np.asarray(request["legal_actions"], np.int32).tobytes())
        digest.update(np.asarray(request["repetition_contexts"], np.uint8).tobytes())
        return digest.digest()

    def _infer(self, requests: list[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
        if not self.fixed_inference_batch_size:
            return self._infer_dynamic(requests)
        predictions = []
        size = self.fixed_inference_batch_size
        for begin in range(0, len(requests), size):
            part = requests[begin:begin + size]
            count = len(part)
            # GroupNorm has no cross-example statistics. Fixed physical shapes
            # remove the observed batch-shape rounding difference in evaluation.
            padded = part + [part[-1]] * (size - count)
            predictions.extend(self._infer_dynamic(padded)[:count])
        return predictions

    def _infer_dynamic(self, requests: list[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
        self._physical_inference_rows += len(requests)
        observations = np.stack([np.asarray(r["features"], np.float32) for r in requests])
        masks = np.zeros((len(requests), 3600), np.bool_)
        repetition = np.zeros((len(requests), 3600), np.uint8)
        legals = []
        for row, request in enumerate(requests):
            legal = np.asarray(request["legal_actions"], np.int64)
            contexts = np.asarray(request["repetition_contexts"], np.uint8)
            if not len(legal) or len(legal) != len(contexts):
                raise ValueError("nonterminal state must have legal actions and matching contexts")
            masks[row, legal] = True
            repetition[row, legal] = contexts
            legals.append(legal)
        inputs = (torch.from_numpy(observations).to(self.device),
                  torch.from_numpy(masks).to(self.device), torch.from_numpy(repetition).to(self.device))
        size = len(requests)
        if self.cuda_graphs and size in (1, 2, 4, 8) and size not in self._graph_failures:
            if size not in self._graphs:
                try:
                    static = tuple(value.clone() for value in inputs)
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        for unused in range(3):
                            self._forward_inference(static)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = self._forward_inference(static)
                    self._graphs[size] = (graph, static, output)
                except RuntimeError:
                    # Unsupported graph capture falls back to the same forward.
                    # Do not keep retrying an unsupported shape every leaf.
                    self._graph_failures.add(size)
            if size in self._graphs:
                graph, static, output = self._graphs[size]
                for target, source in zip(static, inputs):
                    target.copy_(source)
                graph.replay()
                policy, values = output
                self._graph_replays += 1
            else:
                policy, values = self._forward_inference(inputs)
        else:
            policy, values = self._forward_inference(inputs)
        policy_cpu = policy.float().cpu().numpy()
        values_cpu = values.float().cpu().numpy()
        return [(policy_cpu[row, legal].copy(), float(values_cpu[row]))
                for row, legal in enumerate(legals)]

    def _forward_inference(self, inputs):
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                            enabled=self.device.type == "cuda", cache_enabled=False):
            return self.model.policy_value(*inputs)

    def advance_tree(self, tree_id: Any, action: int, state_after: Any) -> bool:
        self._refresh_revision()
        entry = self._trees.get(tree_id)
        if entry is None:
            return False
        tree, unused = entry
        if not tree.advance_root(int(action)) or int(tree.root_key()) != int(state_after.native_state.key()):
            self._trees.pop(tree_id, None)
            return False
        self._trees[tree_id] = (tree, self._state_key(state_after))
        return True

    @classmethod
    def _state_key(cls, state: Any) -> bytes:
        return cls._request_key({"state_key": int(state.native_state.key()),
            "features": state.feature_planes(), "legal_actions": state.legal_actions(),
            "repetition_contexts": state.repetition_contexts()})

    def clear_trees(self) -> None:
        self._trees.clear()

    @torch.inference_mode()
    def search(
        self,
        states: Sequence[Any],
        simulations: int,
        *,
        add_root_noise: bool = False,
        epsilon: float = 0.25,
        noise_seeds: Sequence[int] | None = None,
        leaf_value_mode: str = "network",
        leaf_value_scale: float | None = None,
        tree_ids: Sequence[Any] | None = None,
    ) -> tuple[list[SearchResult], dict[str, Any]]:
        if simulations < 1:
            raise ValueError("simulations must be positive")
        if leaf_value_mode not in {"network", "zero"}:
            raise ValueError("leaf_value_mode must be network or zero")
        if leaf_value_scale is None:
            leaf_value_scale = float(getattr(self.model, "search_v3_leaf_value_scale", 1.0))
        if not 0.0 <= float(leaf_value_scale) <= 1.0:
            raise ValueError("leaf_value_scale must be in [0, 1]")
        if noise_seeds is not None and len(noise_seeds) != len(states):
            raise ValueError("noise seeds must match states")
        if tree_ids is not None and (len(tree_ids) != len(states) or len(set(tree_ids)) != len(tree_ids)):
            raise ValueError("tree ids must be unique and match states")
        if any(state.is_terminal() for state in states):
            raise ValueError("search requires nonterminal states")
        self._refresh_revision()
        contract = (leaf_value_mode, float(leaf_value_scale))
        if self._tree_contract != contract:
            self._trees.clear()
            self._tree_contract = contract
        started = time.perf_counter()
        physical_rows_before = self._physical_inference_rows
        trees, reused, root_keys = [], 0, []
        for index, state in enumerate(states):
            key = self._state_key(state) if self.reuse_roots and tree_ids is not None and not add_root_noise else None
            entry = self._trees.get(tree_ids[index]) if key is not None else None
            if entry is not None and entry[1] == key:
                tree = entry[0]
                reused += 1
            else:
                tree = JunqiAZ.NeuralPUCTTree(state.native_state, self.cpuct, self.root_tactical_guard)
            trees.append(tree)
            root_keys.append(key)
        remaining = [simulations] * len(trees)
        inherited = sum(tree.root_visits() for tree in trees)
        noise_added = [False] * len(trees)
        noise_seeds = list(range(len(trees))) if noise_seeds is None else noise_seeds
        inference_calls = inference_rows = cache_hits = deduplicated = 0
        graph_replays_before = self._graph_replays
        try:
            while any(remaining):
                waves = [list(tree.select_leaves(min(self.leaves_per_tree, left)))
                         if left else [] for tree, left in zip(trees, remaining)]
                predictions, misses, duplicates = {}, [], {}
                for index, requests in enumerate(waves):
                    for leaf, request in enumerate(requests):
                        slot = (index, leaf)
                        if request["terminal"]:
                            predictions[slot] = (np.empty(0, np.float32), float(request["terminal_value"]))
                            continue
                        key = self._request_key(request) if self.cache_size else None
                        if key is not None and key in self._cache:
                            predictions[slot] = self._cache[key]
                            self._cache.move_to_end(key)
                            cache_hits += 1
                        elif key is not None and key in duplicates:
                            duplicates[key].append(slot)
                            deduplicated += 1
                        else:
                            misses.append((slot, key, request))
                            if key is not None:
                                duplicates[key] = [slot]
                for begin in range(0, len(misses), self.inference_batch_size):
                    chunk = misses[begin:begin + self.inference_batch_size]
                    inferred = self._infer([row[2] for row in chunk])
                    inference_calls += 1
                    inference_rows += len(chunk)
                    for (slot, key, unused), prediction in zip(chunk, inferred):
                        for target in duplicates.get(key, [slot]) if key is not None else [slot]:
                            predictions[target] = prediction
                        if key is not None:
                            self._cache[key] = prediction
                            while len(self._cache) > self.cache_size:
                                self._cache.popitem(last=False)
                for index, requests in enumerate(waves):
                    if not requests:
                        continue
                    priors, values = [], []
                    for leaf, request in enumerate(requests):
                        prior, value = predictions[(index, leaf)]
                        priors.append(prior.tolist())
                        values.append(value if request["terminal"] else (
                            value * float(leaf_value_scale) if leaf_value_mode == "network" else 0.0))
                    trees[index].backup_leaves(priors, values)
                    remaining[index] -= len(requests)
                    if add_root_noise and not noise_added[index]:
                        count = len(requests[0]["legal_actions"])
                        trees[index].add_root_dirichlet(float(np.clip(10.0 / max(1, count), .03, 1.0)),
                                                       epsilon, int(noise_seeds[index]))
                        noise_added[index] = True
        except Exception:
            for tree in trees:
                tree.cancel_pending()
            self.clear_trees()
            raise
        results = []
        for index, tree in enumerate(trees):
            edges = list(tree.root_edges())
            actions = np.asarray([row["action"] for row in edges], np.int32)
            visits = np.asarray([row["visits"] for row in edges], np.int32)
            priors = np.asarray([row["prior"] for row in edges], np.float32)
            q_values = np.asarray([row["q"] for row in edges], np.float32)
            policy = visits.astype(np.float64) / visits.sum() if visits.sum() else priors / priors.sum()
            eligible = np.flatnonzero(priors > 0) if self.root_tactical_guard else np.arange(len(actions))
            if not visits.sum():
                best = int(eligible[np.lexsort((actions[eligible], -priors[eligible]))[0]])
            else:
                best = int(eligible[np.lexsort((actions[eligible], -q_values[eligible], -visits[eligible]))[0]])
            results.append(SearchResult(int(actions[best]), actions, visits, priors, q_values,
                policy.astype(np.float32), float(-np.sum(policy * np.log(np.clip(policy, 1e-12, 1.0)))), simulations))
            if root_keys[index] is not None:
                self._trees[tree_ids[index]] = (tree, root_keys[index])
                self._trees.move_to_end(tree_ids[index])
                while len(self._trees) > self.max_retained_trees:
                    self._trees.popitem(last=False)
        elapsed = time.perf_counter() - started
        tactics = np.sum([tree.root_tactical_stats() for tree in trees], axis=0).tolist() if trees else [0] * 4
        return results, {"elapsed_seconds": elapsed,
            "fixed_inference_batch_size": self.fixed_inference_batch_size,
            "physical_inference_rows": self._physical_inference_rows - physical_rows_before,
            "tactical_roots": tactics[0], "tactical_native_transitions": tactics[1],
            "tactical_immediate_wins": tactics[2], "tactical_disabled_actions": tactics[3],
            "simulations_per_second": len(states) * simulations / max(elapsed, 1e-9),
            "inference_calls": inference_calls, "inference_rows": inference_rows,
            "mean_inference_batch": inference_rows / max(1, inference_calls),
            "cache_hits": cache_hits, "deduplicated_rows": deduplicated,
            "cache_entries": len(self._cache), "reused_roots": reused,
            "inherited_root_visits": inherited, "leaves_per_tree": self.leaves_per_tree,
            "cuda_graph_replays": self._graph_replays - graph_replays_before,
            "cuda_graph_shapes": sorted(self._graphs), "cuda_graph_failed_shapes": sorted(self._graph_failures),
            "leaf_value_mode": leaf_value_mode, "leaf_value_scale": float(leaf_value_scale)}

    @torch.inference_mode()
    def bare(self, states: Sequence[Any]) -> list[SearchResult]:
        if not states:
            return []
        if any(state.is_terminal() for state in states):
            raise ValueError("bare policy requires nonterminal states")
        self._refresh_revision()
        observations = torch.from_numpy(
            np.stack([state.feature_planes() for state in states])
        ).to(self.device)
        masks_cpu = np.zeros((len(states), 3600), dtype=np.bool_)
        repetition_cpu = np.zeros((len(states), 3600), dtype=np.uint8)
        legals = []
        for row, state in enumerate(states):
            legal = np.asarray(state.legal_actions(), np.int64)
            contexts = np.asarray(state.repetition_contexts(), np.uint8)
            masks_cpu[row, legal] = True
            repetition_cpu[row, legal] = contexts
            legals.append(legal)
        masks = torch.from_numpy(masks_cpu).to(self.device)
        repetition = torch.from_numpy(repetition_cpu).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            policy, _ = self.model.policy_value(
                observations, masks, repetition
            )
        policy_cpu = policy.float().cpu().numpy()
        results = []
        for row, legal in enumerate(legals):
            probabilities = policy_cpu[row, legal]
            action_index = int(np.argmax(probabilities))
            entropy = float(
                -np.sum(
                    probabilities
                    * np.log(np.clip(probabilities, 1e-12, 1.0))
                )
            )
            results.append(
                SearchResult(
                    action=int(legal[action_index]),
                    actions=legal.astype(np.int32),
                    visits=np.zeros(len(legal), np.int32),
                    priors=probabilities,
                    q_values=np.zeros(len(legal), np.float32),
                    policy=probabilities,
                    root_entropy=entropy,
                    simulations=0,
                )
            )
        return results

    @torch.inference_mode()
    def values(self, states: Sequence[Any]) -> np.ndarray:
        if not states:
            return np.empty(0, np.float32)
        self._refresh_revision()
        observations = torch.from_numpy(
            np.stack([state.feature_planes() for state in states])
        ).to(self.device)
        masks_cpu = np.zeros((len(states), 3600), np.bool_)
        repetition_cpu = np.zeros((len(states), 3600), np.uint8)
        for row, state in enumerate(states):
            legal = np.asarray(state.legal_actions(), np.int64)
            masks_cpu[row, legal] = True
            repetition_cpu[row, legal] = np.asarray(state.repetition_contexts(), np.uint8)
        masks = torch.from_numpy(masks_cpu).to(self.device)
        repetition = torch.from_numpy(repetition_cpu).to(self.device)
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            _, values = self.model(
                observations, masks, repetition
            )
        return values.float().cpu().numpy()
