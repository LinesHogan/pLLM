#!/usr/bin/env python3
"""Unit tests for retry-control and update-barrier runtime contracts."""

from __future__ import annotations

import unittest


class RetryUpdateBarrierRuntimeUnitTest(unittest.TestCase):
    def test_retry_control_entry_rejects_implicit_identity_and_action_coercion(self) -> None:
        from tllm.runtime.retry_control_runtime import RetryControlEntry

        with self.assertRaisesRegex(TypeError, "request_id must be str"):
            RetryControlEntry(request_id=7, prompt_idx=0, sample_idx=0, action="retry")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "prompt_idx must be int"):
            RetryControlEntry(request_id="req-a", prompt_idx=True, sample_idx=0, action="retry")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "action must be str"):
            RetryControlEntry(request_id="req-a", prompt_idx=0, sample_idx=0, action=1)  # type: ignore[arg-type]
        with self.assertRaisesRegex(ValueError, "unsupported retry action"):
            RetryControlEntry(request_id="req-a", prompt_idx=0, sample_idx=0, action="again")

    def test_retry_control_store_appends_reads_and_drains(self) -> None:
        from tllm.runtime.retry_control_runtime import (
            RetryControlBatch,
            RetryControlEntry,
            read_retry_controls,
            write_retry_controls,
        )

        runtime = type("Runtime", (), {})()
        batch = RetryControlBatch(
            entries=(
                RetryControlEntry(
                    request_id="req-a",
                    prompt_idx=1,
                    sample_idx=2,
                    output_index=0,
                    group_id="round-a",
                    action="retry",
                    reason="needs feedback",
                    retry_after_update=True,
                ),
            )
        )

        self.assertEqual(write_retry_controls(runtime, batch), 1)
        self.assertIs(runtime.last_retry_controls, batch)
        self.assertEqual(read_retry_controls(runtime).entries, batch.entries)
        self.assertEqual(read_retry_controls(runtime, drain=True).entries, batch.entries)
        self.assertEqual(read_retry_controls(runtime).entries, ())

    def test_retry_control_read_rejects_non_bool_drain(self) -> None:
        from tllm.runtime.retry_control_runtime import RetryControlBatch, RetryControlEntry, read_retry_controls, write_retry_controls

        runtime = type("Runtime", (), {})()
        batch = RetryControlBatch(entries=(RetryControlEntry(request_id="req-a", prompt_idx=0, sample_idx=0),))
        write_retry_controls(runtime, batch)

        with self.assertRaisesRegex(TypeError, "drain must be bool"):
            read_retry_controls(runtime, drain="False")  # type: ignore[arg-type]
        self.assertEqual(read_retry_controls(runtime).entries, batch.entries)

    def test_update_barrier_entry_rejects_implicit_coercion(self) -> None:
        from tllm.runtime.update_barrier_runtime import UpdateBarrierEntry

        with self.assertRaisesRegex(TypeError, "barrier_id must be str"):
            UpdateBarrierEntry(barrier_id=1, group_id="round-a", consumer_id="ttrl", state="ready")  # type: ignore[arg-type]
        with self.assertRaisesRegex(TypeError, "request_ids entries must be str"):
            UpdateBarrierEntry(
                barrier_id="barrier-a",
                group_id="round-a",
                consumer_id="ttrl",
                state="ready",
                request_ids=("req-a", 2),  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(TypeError, "ready_count must be int"):
            UpdateBarrierEntry(
                barrier_id="barrier-a",
                group_id="round-a",
                consumer_id="ttrl",
                state="ready",
                ready_count=True,  # type: ignore[arg-type]
            )
        with self.assertRaisesRegex(ValueError, "unsupported barrier state"):
            UpdateBarrierEntry(barrier_id="barrier-a", group_id="round-a", consumer_id="ttrl", state="done")

    def test_update_barrier_store_appends_reads_and_drains(self) -> None:
        from tllm.runtime.update_barrier_runtime import (
            UpdateBarrierBatch,
            UpdateBarrierEntry,
            read_update_barriers,
            write_update_barriers,
        )

        runtime = type("Runtime", (), {})()
        batch = UpdateBarrierBatch(
            entries=(
                UpdateBarrierEntry(
                    barrier_id="barrier-a",
                    group_id="round-a",
                    consumer_id="ttrl",
                    state="ready",
                    request_ids=("req-a", "req-b"),
                    ready_count=2,
                    reason="all rewards available",
                ),
            )
        )

        self.assertEqual(write_update_barriers(runtime, batch), 1)
        self.assertIs(runtime.last_update_barriers, batch)
        self.assertEqual(read_update_barriers(runtime).entries, batch.entries)
        self.assertEqual(read_update_barriers(runtime, drain=True).entries, batch.entries)
        self.assertEqual(read_update_barriers(runtime).entries, ())

    def test_update_barrier_read_rejects_non_bool_drain(self) -> None:
        from tllm.runtime.update_barrier_runtime import UpdateBarrierBatch, UpdateBarrierEntry, read_update_barriers, write_update_barriers

        runtime = type("Runtime", (), {})()
        batch = UpdateBarrierBatch(entries=(UpdateBarrierEntry(barrier_id="barrier-a", group_id="", consumer_id="ttrl"),))
        write_update_barriers(runtime, batch)

        with self.assertRaisesRegex(TypeError, "drain must be bool"):
            read_update_barriers(runtime, drain="False")  # type: ignore[arg-type]
        self.assertEqual(read_update_barriers(runtime).entries, batch.entries)


if __name__ == "__main__":
    unittest.main()
