"""Differential tests for Rust packed prefill row planning."""

from __future__ import annotations

import random
import unittest

import pytest

from tllm import _core
from test.core.oracles.prefill_row_plan import oracle_plan_packed_prefill_rows


def _as_tuple(value: object) -> tuple[object, ...]:
    return tuple(value)  # type: ignore[arg-type]


class PrefillRowPlanOracleUnitTest(unittest.TestCase):
    def test_rust_classifies_complete_scheduler_batch_phase(self) -> None:
        cases = (
            (([], [], []), _core.SchedulerBatchPhase.Empty),
            (([2], [4], [0]), _core.SchedulerBatchPhase.Prefill),
            (([2], [4], [4]), _core.SchedulerBatchPhase.Decode),
            (([2], [4], [3]), _core.SchedulerBatchPhase.Mixed),
            (([1, 1], [2, 3], [0, 3]), _core.SchedulerBatchPhase.Mixed),
        )
        for (scheduled, prompt, computed), expected in cases:
            with self.subTest(scheduled=scheduled, prompt=prompt, computed=computed):
                actual = _core.classify_scheduler_batch_phase(
                    scheduled, prompt, computed
                )
                self.assertEqual(actual, expected)

        with pytest.raises(ValueError, match="equal lengths"):
            _core.classify_scheduler_batch_phase([1], [], [0])

    def _assert_matches(self, scheduled: list[int], prompt: list[int], computed: list[int]) -> None:
        expected = oracle_plan_packed_prefill_rows(scheduled, prompt, computed)
        actual = _core.plan_packed_prefill_rows(scheduled, prompt, computed)
        self.assertEqual(_as_tuple(actual.request_ordinals), expected.request_ordinals)
        self.assertEqual(_as_tuple(actual.packed_row_indices), expected.packed_row_indices)
        self.assertEqual(_as_tuple(actual.token_offsets), expected.token_offsets)
        self.assertEqual(_as_tuple(actual.phases), expected.phases)
        self.assertEqual(actual.total_rows, expected.total_rows)
        self.assertEqual(len(actual.request_spans), len(expected.request_spans))
        for rust_span, oracle_span in zip(actual.request_spans, expected.request_spans):
            for field in (
                "request_ordinal",
                "packed_row_start",
                "packed_row_end",
                "token_offset_start",
                "token_offset_end",
                "prefill_row_count",
                "decode_row_count",
            ):
                self.assertEqual(getattr(rust_span, field), getattr(oracle_span, field))

    def test_multiple_requests_chunked_and_mixed_phases(self) -> None:
        self._assert_matches([4, 2, 3], [7, 2, 5], [1, 2, 5])

    def test_zero_scheduled_request_keeps_an_empty_span(self) -> None:
        self._assert_matches([0, 2, 0], [5, 4, 0], [0, 2, 0])

    def test_decode_can_advance_beyond_prompt_length(self) -> None:
        self._assert_matches([2, 1], [3, 4], [5, 4])

    def test_random_differential_matrix(self) -> None:
        rng = random.Random(20260822)
        for trial in range(100):
            request_count = rng.randint(0, 8)
            prompt = [rng.randint(0, 12) for _ in range(request_count)]
            computed = [rng.randint(0, value + 8) for value in prompt]
            scheduled = [rng.randint(0, 8) for _ in range(request_count)]
            with self.subTest(trial=trial):
                self._assert_matches(scheduled, prompt, computed)

    def test_expected_total_rows_is_a_boundary_check(self) -> None:
        actual = _core.plan_packed_prefill_rows([2, 1], [3, 1], [0, 1], 3)
        self.assertEqual(actual.total_rows, 3)
        with pytest.raises(ValueError, match="packed row count mismatch"):
            _core.plan_packed_prefill_rows([2, 1], [3, 1], [0, 1], 4)

    def test_invalid_counts_fail_closed(self) -> None:
        cases = [
            ([1], [1, 1], [0], "equal lengths"),
            ([-1], [1], [0], "non-negative"),
            ([1], [-1], [0], "non-negative"),
            ([1], [1], [-1], "non-negative"),
        ]
        for scheduled, prompt, computed, message in cases:
            with self.subTest(scheduled=scheduled, prompt=prompt, computed=computed):
                with pytest.raises(ValueError, match=message):
                    _core.plan_packed_prefill_rows(scheduled, prompt, computed)


if __name__ == "__main__":
    unittest.main()
