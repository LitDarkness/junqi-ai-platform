from __future__ import annotations

import unittest
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from junqi_search_v3.closed_loop import ClosedLoopRunner
from junqi_search_v3.loop import (
    SearchSample,
    SearchSelfPlay,
    read_selfplay_shard,
    read_selfplay_state_hashes,
    write_selfplay_shard,
)
from junqi_search_v3.supervised import MemorySample, collate as oracle_collate

ROOT = Path(__file__).resolve().parents[1]


class ClosedLoopRunnerTest(unittest.TestCase):
    def test_selfplay_reports_uncommitted_window_progress(self) -> None:
        class Native:
            @staticmethod
            def repetition_count() -> int:
                return 1

            @staticmethod
            def position_key() -> int:
                return 11

            @staticmethod
            def key() -> int:
                return 12

        class State:
            def __init__(self) -> None:
                self.native_state = Native()
                self.done = False

            @staticmethod
            def repetition_contexts() -> list[int]:
                return [0]

            @staticmethod
            def current_player() -> int:
                return 1

            @staticmethod
            def feature_planes() -> np.ndarray:
                return np.zeros((107, 12, 5), np.float32)

            def apply(self, action: int) -> None:
                self.done = True

            def is_terminal(self) -> bool:
                return self.done

            @staticmethod
            def terminal_reason() -> str:
                return "FLAG_CAPTURE"

            @staticmethod
            def terminal_value(player: int) -> float:
                return 1.0

        class Backend:
            @staticmethod
            def new_state(seed: int) -> State:
                return State()

        class Search:
            @staticmethod
            def search(states, simulations, **kwargs):
                result = SimpleNamespace(
                    actions=np.asarray([1], np.int32),
                    visits=np.asarray([1], np.int32),
                    priors=np.asarray([1.0], np.float32),
                    root_entropy=0.0,
                )
                return [result for _ in states], {
                    "elapsed_seconds": 1.0,
                }

        collector = SearchSelfPlay.__new__(SearchSelfPlay)
        collector.config = SimpleNamespace(
            seed=7,
            puct=SimpleNamespace(
                repeated_state_simulations=128,
                selfplay_simulations=64,
                dirichlet_epsilon=0.25,
            ),
            loop=SimpleNamespace(state_policy_cap_per_iteration=4),
        )
        collector.backend = Backend()
        collector.search = Search()
        updates = []
        samples, metrics = collector.collect_block(
            1, 0, 2, progress_callback=updates.append
        )
        self.assertEqual(metrics["games"], 2)
        self.assertEqual(len(samples), 2)
        self.assertEqual(updates[-1]["finished_games"], 2)
        self.assertEqual(updates[-1]["active_games"], 0)

    def test_output_directory_does_not_shadow_run_entrypoint(self) -> None:
        runner = ClosedLoopRunner.__new__(ClosedLoopRunner)
        runner.run_dir = Path("runs/open_junqi_search_v3")
        self.assertTrue(callable(runner.run))

    def test_external_gate_checks_system_and_distillation_separately(self) -> None:
        current = ClosedLoopRunner._external_gate_reasons(
            {"external_bare_mean": 0.50, "external_search64_mean": 0.25},
            {"external_bare_mean": 0.25, "external_search64_mean": 0.75},
        )
        self.assertEqual(current, [])
        system_regression = ClosedLoopRunner._external_gate_reasons(
            {"external_bare_mean": 0.60, "external_search64_mean": 0.70},
            {"external_bare_mean": 0.40, "external_search64_mean": 0.45},
        )
        self.assertIn(
            "external wide_quiescence/exact082 system regressed",
            system_regression,
        )
        self.assertIn(
            "bare pi_(k+1) did not approach external Search(pi_k) score",
            system_regression,
        )

    def test_external_closure_does_not_gate_on_pseudoreplication(self) -> None:
        external = {
            "a": {
                "unique_trajectories": 2,
                "deterministic_pseudoreplication_warning": True,
            }
        }
        reasons = ClosedLoopRunner._external_gate_reasons(
            {
                "external_bare_mean": 0.50,
                "external_search64_mean": 0.375,
                "external": external,
            },
            {
                "external_bare_mean": 0.25,
                "external_search64_mean": 0.625,
                "external": external,
            },
        )
        self.assertEqual(reasons, [])

    def test_selfplay_shard_is_materialized_once_and_roundtrips(self) -> None:
        sample = SearchSample(
            game_id=7,
            state_hash=123456,
            observation=np.arange(107 * 12 * 5, dtype=np.float32).reshape(
                107, 12, 5
            ),
            legal_actions=np.asarray([1, 99], np.int32),
            repetition_contexts=np.asarray([0, 2], np.uint8),
            visits=np.asarray([3, 5], np.int32),
            raw_visits=np.asarray([12, 5], np.int32),
            legacy_search64_visits=np.asarray([2, 6], np.int32),
            legacy_search64_valid=True,
            cycle_teacher_eligible=True,
            cycle_teacher_adjusted=True,
            cycle_teacher_rational_repeat_preserved=False,
            player=-1,
            ply=11,
            root_entropy=0.5,
            position_key=654321,
            repetition_count=2,
            cycle_entry=True,
            endgame=True,
            cycle_cleaned=True,
            wdl=np.asarray([0, 1, 0], np.float32),
        )
        temporary_root = ROOT / "runs" / "open_junqi_search_v3" / "test_io"
        temporary_root.mkdir(parents=True, exist_ok=True)
        path = temporary_root / "games.npz"
        try:
            write_selfplay_shard(path, [sample])
            loaded = read_selfplay_shard(path)
            hashes = read_selfplay_state_hashes(path)
        finally:
            path.unlink(missing_ok=True)
        self.assertEqual(hashes.tolist(), [123456])
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].legal_actions.tolist(), [1, 99])
        self.assertEqual(loaded[0].visits.tolist(), [3, 5])
        self.assertEqual(loaded[0].raw_visits.tolist(), [12, 5])
        self.assertEqual(
            loaded[0].legacy_search64_visits.tolist(), [2, 6]
        )
        self.assertTrue(loaded[0].legacy_search64_valid)
        self.assertTrue(loaded[0].cycle_teacher_eligible)
        self.assertTrue(loaded[0].cycle_teacher_adjusted)
        self.assertFalse(
            loaded[0].cycle_teacher_rational_repeat_preserved
        )
        self.assertEqual(loaded[0].wdl.tolist(), [0.0, 1.0, 0.0])
        self.assertEqual(loaded[0].position_key, 654321)
        self.assertEqual(loaded[0].repetition_count, 2)
        self.assertTrue(loaded[0].cycle_entry)
        self.assertTrue(loaded[0].endgame)
        self.assertTrue(loaded[0].cycle_cleaned)
        self.assertIsNone(loaded[0].observation.base)
        self.assertIsNone(loaded[0].legal_actions.base)

    def test_cycle_teacher_only_softens_avoidable_repetition(self) -> None:
        result = SimpleNamespace(
            visits=np.asarray([100, 80, 30], np.int32),
            q_values=np.asarray([0.20, 0.19, -0.4], np.float32),
        )
        visits, metric = SearchSelfPlay._cycle_teacher_visits(
            result, np.asarray([1, 0, 0], np.uint8)
        )
        self.assertTrue(metric["adjusted"])
        self.assertEqual(visits.tolist(), [25, 80, 30])
        result.q_values[0] = 0.30
        visits, metric = SearchSelfPlay._cycle_teacher_visits(
            result, np.asarray([1, 0, 0], np.uint8)
        )
        self.assertTrue(metric["rational_repeat_preserved"])
        self.assertEqual(visits.tolist(), [100, 80, 30])

    def test_terminal_cycle_suffix_is_trimmed_after_one_cycle(self) -> None:
        def sample(ply: int, position: int) -> SearchSample:
            return SearchSample(
                game_id=1,
                state_hash=100 + ply,
                position_key=position,
                observation=np.zeros((107, 12, 5), np.float16),
                legal_actions=np.asarray([ply], np.int16),
                repetition_contexts=np.asarray([min(2, ply // 2)], np.uint8),
                visits=np.asarray([1], np.int16),
                player=1,
                ply=ply,
                root_entropy=0.0,
            )

        state = SimpleNamespace(terminal_reason=lambda: "REPETITION")
        game = SimpleNamespace(
            state=state,
            # Cycle is positions 11 -> 12 -> 11, repeated three times.
            position_hashes=[10, 11, 12, 11, 12, 11],
            action_rows=[
                {"mover_side": 1},
                {"mover_side": -1},
                {"mover_side": 1},
                {"mover_side": -1},
                {"mover_side": 1},
            ],
            samples=[
                sample(0, 10),
                sample(1, 11),
                sample(2, 12),
                sample(3, 11),
                sample(4, 12),
            ],
        )
        kept, metric = SearchSelfPlay._trim_game_samples(game)
        self.assertEqual([row.ply for row in kept], [0, 1, 2])
        self.assertEqual(metric["cycle_suffix_samples_removed"], 2)

    def test_replay_reservoir_caps_memory_before_materializing_all(self) -> None:
        root = ROOT / "runs" / "open_junqi_search_v3" / "test_replay"
        for iteration in (1, 2):
            directory = root / "search_replay" / f"iteration_{iteration:06d}"
            directory.mkdir(parents=True, exist_ok=True)
            samples = []
            for offset in range(3):
                state_hash = iteration * 10 + offset
                samples.append(
                    SearchSample(
                        game_id=state_hash,
                        state_hash=state_hash,
                        observation=np.zeros((107, 12, 5), np.float16),
                        legal_actions=np.asarray([1], np.int16),
                        repetition_contexts=np.asarray([0], np.uint8),
                        visits=np.asarray([1], np.int16),
                        player=1,
                        ply=0,
                        root_entropy=0.0,
                        wdl=np.asarray([0, 0, 1], np.float32),
                    )
                )
            write_selfplay_shard(directory / "games_0000_0003.npz", samples)
        runner = ClosedLoopRunner.__new__(ClosedLoopRunner)
        runner.run_dir = root
        runner.rng = random.Random(123)
        runner.config = SimpleNamespace(
            loop=SimpleNamespace(
                replay_iterations=5,
                maximum_replay_samples=4,
                state_policy_cap_per_iteration=4,
            )
        )
        try:
            retained = runner._replay_samples(2)
        finally:
            for path in sorted(
                root.rglob("*"), key=lambda item: len(item.parts), reverse=True
            ):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    path.rmdir()
        self.assertEqual(len(retained), 4)
        self.assertEqual(runner.last_replay_stats["eligible_samples"], 6)
        self.assertEqual(
            runner.last_replay_stats["sampling"],
            "uniform_reservoir_algorithm_r",
        )

    def test_oracle_scalar_value_target_uses_best_root_q(self) -> None:
        sample = MemorySample(
            observation=np.zeros((107, 12, 5), np.float32),
            legal_actions=np.asarray([5, 9], np.int16),
            repetition_contexts=np.asarray([0, 0], np.uint8),
            q_values=np.asarray([-0.2, 0.7], np.float32),
            wdl=np.asarray([0, 0, 1], np.float32),
            opponent="test",
            ply=0,
            terminal_reason="",
            important=True,
        )
        batch = oracle_collate([sample], 0.1, torch.device("cpu"))
        self.assertAlmostEqual(
            float(batch["oracle_scalar_target"][0]), 0.7, places=6
        )


if __name__ == "__main__":
    unittest.main()
