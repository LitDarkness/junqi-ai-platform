"""Exact-shape inference scheduling; no changes to traversal or targets.

The original search implementations remain available as the reference oracle.
Capture uses the same forward, BF16 context and physical batch shape. No shape
bucketing, inference-result cache, tree reuse or additional simulations.
"""
from collections import OrderedDict
import numpy as np
import torch
from .search import BatchedNeuralPUCT
from .root_explore_search import RootExploreSearch


class GraphInference:
    def __init__(self, *args, graph_slots=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.graph_slots = int(graph_slots)
        self._graphs = OrderedDict()
        self.graph_capture_seconds = 0.
        self._shape_uses = {}

    def _refresh_revision(self):
        previous=self._revision
        super()._refresh_revision()
        if previous!=self._revision: self._shape_uses.clear()

    def _infer_dynamic(self, requests):
        import time
        self._physical_inference_rows += len(requests)
        observations = np.stack([np.asarray(r['features'], np.float32) for r in requests])
        masks = np.zeros((len(requests), 3600), np.bool_)
        repetition = np.zeros((len(requests), 3600), np.uint8)
        legals = []
        for row, request in enumerate(requests):
            legal = np.asarray(request['legal_actions'], np.int64)
            contexts = np.asarray(request['repetition_contexts'], np.uint8)
            if not len(legal) or len(legal) != len(contexts):
                raise ValueError('nonterminal state must have matching legal/context arrays')
            masks[row, legal] = True
            repetition[row, legal] = contexts
            legals.append(legal)
        inputs = (torch.from_numpy(observations).to(self.device),
                  torch.from_numpy(masks).to(self.device), torch.from_numpy(repetition).to(self.device))
        size = len(requests)
        self._shape_uses[size]=self._shape_uses.get(size,0)+1
        # Large dynamic collector batches are compute-bound. Their short-lived
        # shapes failed the capture-inclusive benchmark; retain eager inference.
        # Evaluation remains exactly physical32 and amortizes one capture.
        admitted=(bool(self.fixed_inference_batch_size) or (size<=64 and self._shape_uses[size]>=16))
        if self.cuda_graphs and admitted and size not in self._graph_failures:
            if size not in self._graphs:
                started = time.perf_counter()
                try:
                    static = tuple(value.clone() for value in inputs)
                    stream = torch.cuda.Stream(device=self.device)
                    stream.wait_stream(torch.cuda.current_stream(self.device))
                    with torch.cuda.stream(stream):
                        for _ in range(3):
                            self._forward_inference(static)
                    torch.cuda.current_stream(self.device).wait_stream(stream)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        output = self._forward_inference(static)
                    self._graphs[size] = (graph, static, output)
                    while len(self._graphs) > self.graph_slots:
                        self._graphs.popitem(last=False)
                except RuntimeError:
                    self._graph_failures.add(size)
                self.graph_capture_seconds += time.perf_counter() - started
            if size in self._graphs:
                self._graphs.move_to_end(size)
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
        # Preserve original conversion and indexing order exactly.
        policy_cpu = policy.float().cpu().numpy()
        values_cpu = values.float().cpu().numpy()
        return [(policy_cpu[row, legal].copy(), float(values_cpu[row]))
                for row, legal in enumerate(legals)]


class GraphNeuralPUCT(GraphInference, BatchedNeuralPUCT):
    pass


class GraphRootExplore(GraphInference, RootExploreSearch):
    pass
