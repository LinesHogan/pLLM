#!/usr/bin/env python3
"""Unit tests for retry-control and update-barrier runtime contracts."""

from __future__ import annotations

import unittest


class RetryUpdateBarrierRuntimeUnitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
