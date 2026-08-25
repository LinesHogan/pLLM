#!/usr/bin/env python3
"""MRv2 publishes packed fast-weight row metadata at prepare-inputs time."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from tllm.backends.vllm import v027_mrv1
from tllm.backends.vllm import v027_mrv2
from tllm.runtime.token_batch_metadata import RuntimeTokenBatchMeta


class VllmMRv2FastWeightMetadataUnitTest(unittest.TestCase):
    @staticmethod
    def _rows(runtime: object) -> list[tuple[object, ...]]:
        metadata = runtime.fast_weight_current_token_batch_meta
        assert isinstance(metadata, RuntimeTokenBatchMeta)
        return [
            (
                row.key.request_id,
                row.key.prompt_idx,
                row.key.sample_idx,
                row.row.packed_row_idx,
                row.row.logical_row_idx,
                row.row.token_offset,
                row.phase,
                row.engine_step_id,
            )
            for row in metadata.rows
        ]

    def test_prepare_inputs_refreshes_metadata_before_forward(self) -> None:
        runtime = SimpleNamespace(
            config=SimpleNamespace(fast_weight_enabled=True),
            event_step_id=12,
            reqid_to_promptidx={"req-a": 21, "req-b": 22},
            reqid_to_sampleidx={"req-a": 0, "req-b": 1},
            fast_weight_phase="prefill",
        )
        input_batch = SimpleNamespace(
            req_ids=["req-a", "req-b"],
            num_reqs=2,
            # These are the request-ordered fields exposed by MRv2. The
            # runner's legacy input_batch below is intentionally stale.
            prefill_len_np=[4, 3],
            num_computed_tokens_np=[0, 3],
            num_scheduled_tokens=[2, 1],
        )
        runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["stale"], num_reqs=1))
        orig = mock.Mock(return_value=input_batch)

        def assert_metadata_published(*, runtime: object, runner: object) -> int:
            _ = runner
            self.assertIsInstance(runtime.fast_weight_current_token_batch_meta, RuntimeTokenBatchMeta)
            return 0

        with (
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=False),
            mock.patch.object(
                v027_mrv2._fast_weight,
                "maybe_install_fast_weight_wrappers",
                side_effect=assert_metadata_published,
            ) as install,
        ):
            result = v027_mrv2._wrapped_prepare_inputs(
                core=SimpleNamespace(RUNTIME=runtime),
                runner=runner,
                args=(),
                kwargs={},
                orig=orig,
            )
        install.assert_called_once_with(runtime=runtime, runner=runner)

        self.assertIs(result, input_batch)
        metadata = runtime.fast_weight_current_token_batch_meta
        self.assertIsInstance(metadata, RuntimeTokenBatchMeta)
        assert isinstance(metadata, RuntimeTokenBatchMeta)
        self.assertEqual(runtime.fast_weight_phase, "decode")
        self.assertEqual(
            self._rows(runtime),
            [
                ("req-a", 21, 0, 0, 0, 0, "prefill", 12),
                ("req-a", 21, 0, 1, 1, 1, "prefill", 12),
                ("req-b", 22, 1, 2, 2, 3, "decode", 12),
            ],
        )

    def test_mrv1_and_mrv2_publish_identical_packed_rows(self) -> None:
        def runtime() -> SimpleNamespace:
            return SimpleNamespace(
                config=SimpleNamespace(fast_weight_enabled=True),
                event_step_id=12,
                reqid_to_promptidx={"req-a": 21, "req-b": 22},
                reqid_to_sampleidx={"req-a": 0, "req-b": 1},
                fast_weight_phase="prefill",
            )

        mrv1_runtime = runtime()
        mrv1_runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                req_ids=["req-a", "req-b"],
                num_reqs=2,
                req_id_to_index={"req-a": 0, "req-b": 1},
                num_prompt_tokens=[4, 3],
                num_computed_tokens_cpu=[0, 3],
            )
        )
        with (
            mock.patch.object(v027_mrv1.active_targets, "runtime_has_active_targets", return_value=False),
            mock.patch.object(v027_mrv1._fast_weight, "maybe_install_fast_weight_wrappers"),
        ):
            v027_mrv1._wrapped_prepare_inputs(
                core=SimpleNamespace(RUNTIME=mrv1_runtime),
                runner=mrv1_runner,
                scheduler_output=SimpleNamespace(),
                num_scheduled_tokens=np.asarray([2, 1], dtype=np.int32),
                orig=mock.Mock(
                    return_value=(
                        np.asarray([0, 1, 2], dtype=np.int64),
                        None,
                    )
                ),
            )

        mrv2_runtime = runtime()
        mrv2_batch = SimpleNamespace(
            req_ids=["req-a", "req-b"],
            num_reqs=2,
            prefill_len_np=[4, 3],
            num_computed_tokens_np=[0, 3],
            num_scheduled_tokens=np.asarray([2, 1], dtype=np.int32),
        )
        mrv2_runner = SimpleNamespace(input_batch=SimpleNamespace(req_ids=["stale"], num_reqs=1))
        with (
            mock.patch.object(v027_mrv2.active_targets, "runtime_has_active_targets", return_value=False),
            mock.patch.object(v027_mrv2._fast_weight, "maybe_install_fast_weight_wrappers"),
        ):
            v027_mrv2._wrapped_prepare_inputs(
                core=SimpleNamespace(RUNTIME=mrv2_runtime),
                runner=mrv2_runner,
                args=(),
                kwargs={},
                orig=mock.Mock(return_value=mrv2_batch),
            )

        self.assertEqual(self._rows(mrv1_runtime), self._rows(mrv2_runtime))


if __name__ == "__main__":
    unittest.main()
