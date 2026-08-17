#!/usr/bin/env python3
"""Differential matrix: Rust sampler-step planning vs the legacy oracle."""

from __future__ import annotations

import itertools
import random
import unittest

from tllm import _core

from test.core.oracles.sampler_plan import (
    oracle_partition_rows,
    oracle_plan_sampler_step,
)

_BACKENDS = ["pre_filter_dense", "post_filter_exact", "post_filter_dense_cache"]
_TEMP_MODES = ["all_greedy", "all_random", "mixed"]


class SamplerPlanOracleMatrixUnitTest(unittest.TestCase):
    def test_full_decision_matrix(self) -> None:
        for backend, temp_mode, greedy_avail, candidate in itertools.product(
            _BACKENDS, _TEMP_MODES, [False, True], [False, True]
        ):
            for n_rows, n_affected, n_greedy in [
                (4, 4, 0),
                (4, 4, 4),
                (4, 4, 2),
                (4, 2, 0),
                (4, 2, 2),
                (1, 1, 1),
                (1, 1, 0),
            ]:
                if n_greedy and not greedy_avail:
                    continue  # greedy rows only exist when greedy_sampled is available
                with self.subTest(
                    backend=backend, temp=temp_mode, rows=n_rows,
                    affected=n_affected, greedy=n_greedy,
                    avail=greedy_avail, candidate=candidate,
                ):
                    oracle = oracle_plan_sampler_step(
                        backend=backend,
                        temperature_mode=temp_mode,
                        n_rows=n_rows,
                        n_affected=n_affected,
                        n_greedy=n_greedy,
                        greedy_sampled_available=greedy_avail,
                        candidate_eligible=candidate,
                    )
                    rust = _core.plan_sampler_step(
                        backend, temp_mode, n_rows, n_affected, n_greedy, greedy_avail, candidate
                    )
                    self.assertEqual(rust.path, oracle.path)
                    self.assertEqual(rust.use_candidate_kernel, oracle.use_candidate_kernel)
                    self.assertEqual(rust.dense_inplace_applied, oracle.dense_inplace_applied)
                    self.assertEqual(rust.base, oracle.base)

    def test_partition_rows_matrix(self) -> None:
        rng = random.Random(20260817)
        for trial in range(120):
            n_rows = rng.randint(1, 12)
            affected = tuple(sorted(rng.sample(range(n_rows), rng.randint(0, n_rows))))
            greedy_avail = rng.random() < 0.7
            greedy = (
                tuple(sorted(rng.sample(range(n_rows), rng.randint(0, n_rows))))
                if greedy_avail
                else ()
            )
            with self.subTest(trial=trial):
                oracle = oracle_partition_rows(
                    n_rows=n_rows,
                    affected_rows=affected,
                    greedy_rows=greedy,
                    greedy_sampled_available=greedy_avail,
                )
                rust = _core.partition_rows(n_rows, list(affected), list(greedy), greedy_avail)
                self.assertEqual(tuple(rust.affected_greedy), oracle.affected_greedy)
                self.assertEqual(tuple(rust.affected_random), oracle.affected_random)
                self.assertEqual(tuple(rust.unaffected_greedy), oracle.unaffected_greedy)
                self.assertEqual(tuple(rust.unaffected_random), oracle.unaffected_random)


class DecodeLocalizationOracleUnitTest(unittest.TestCase):
    def test_against_legacy_examples_and_random(self) -> None:
        rng = random.Random(7)
        cases = [
            ([False, True, True, False], [4, 7, 11, 15], 4, 16, 0),
            ([True], [0], 1, 1, 0),
            ([False, False], [3, 4], 2, 8, 0),
            ([True, True, True], [2, 5, 9], 3, 16, 2),
        ]
        for _ in range(80):
            n = rng.randint(1, 10)
            cases.append(
                (
                    [rng.random() < 0.6 for _ in range(n)],
                    sorted(rng.sample(range(3 * n), n)),
                    n,
                    3 * n,
                    rng.choice([0, 0, n // 2]),
                )
            )
        for is_decode, logits_indices, req_count, num_tokens, cap in cases:
            with self.subTest(case=(is_decode, logits_indices, cap)):
                loc = _core.localize_decode_rows(is_decode, logits_indices, req_count, num_tokens, cap)
                expected_positions = [
                    i for i, d in enumerate(is_decode) if d and i < len(logits_indices) and i < req_count
                ]
                if cap > 0:
                    expected_positions = expected_positions[:cap]
                self.assertEqual(loc.positions, expected_positions)
                self.assertEqual(loc.row_idx, [logits_indices[p] for p in expected_positions])
                if expected_positions:
                    first = expected_positions[0]
                    self.assertEqual(
                        loc.contiguous,
                        expected_positions == list(range(first, first + len(expected_positions))),
                    )

    def test_out_of_range_error_matches_legacy(self) -> None:
        with self.assertRaisesRegex(
            RuntimeError,
            r"logits_indices out of range in prepare_inputs: min=0 max=20 num_actual_tokens=16",
        ):
            _core.localize_decode_rows([True, True], [0, 20], 2, 16, 0)


if __name__ == "__main__":
    unittest.main()
