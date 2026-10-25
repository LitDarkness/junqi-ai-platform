import json
import os
import sys
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BUILD = os.path.join(ROOT, "build")
for path in (ROOT, BUILD):
    if path not in sys.path:
        sys.path.insert(0, path)

from heuristic_benchmark import DEFAULT_CORPUS, benchmark_tactics, bootstrap_ci


VALIDATION_CORPUS = os.path.join(ROOT, "config", "benchmark_validation_positions_v1.json")


class HeuristicBenchmarkTests(unittest.TestCase):
    def test_persisted_corpus_has_valid_shape_and_piece_signs(self):
        payload = json.loads(DEFAULT_CORPUS.read_text(encoding="utf-8"))
        self.assertEqual(payload["format"], "junqi-benchmark-positions-v1")
        self.assertGreaterEqual(len(payload["positions"]), 16)
        for board in payload["positions"]:
            self.assertEqual(len(board), 60)
            self.assertTrue(any(piece > 0 for piece in board))
            self.assertTrue(any(piece < 0 for piece in board))

    def test_production_heuristics_pass_reference_tactics(self):
        results = benchmark_tactics(["heuristic_v1", "heuristic_v2"])
        for result in results:
            self.assertEqual(result["passed"], result["total"], result)

    def test_validation_corpus_is_independent_and_has_64_unique_layouts(self):
        training = json.loads(DEFAULT_CORPUS.read_text(encoding="utf-8"))["positions"]
        with open(VALIDATION_CORPUS, encoding="utf-8") as handle:
            validation = json.load(handle)["positions"]
        self.assertEqual(len(validation), 64)
        validation_keys = {tuple(board) for board in validation}
        self.assertEqual(len(validation_keys), 64)
        self.assertTrue(validation_keys.isdisjoint({tuple(board) for board in training}))

    def test_paired_bootstrap_is_repeatable(self):
        scores = [0.0, 0.5, 1.0, 1.0]
        self.assertEqual(bootstrap_ci(scores), bootstrap_ci(scores))


if __name__ == "__main__":
    unittest.main()
