#!/usr/bin/env python3
"""Unit tests for graph policy and strict spec-decode row mapping."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tllm.runtime.torch_adapter import CudaLongHostStaging


class GraphPolicyUnitTest(unittest.TestCase):
    def test_plan_compatibility_enforcement(self) -> None:
        from tllm.backends.protocol import GraphSafety
        from tllm.runtime.graph_policy import (
            GraphMode,
            GraphPolicyError,
            assert_plan_graph_compatible,
        )

        producer = SimpleNamespace(graph_safety=GraphSafety.EAGER_ONLY)
        planned = SimpleNamespace(
            entry=SimpleNamespace(producer_id="p.eager", producer=producer)
        )
        plan = SimpleNamespace(producers=(planned,))
        assert_plan_graph_compatible(plan, GraphMode.EAGER)  # no raise
        with self.assertRaises(GraphPolicyError):
            assert_plan_graph_compatible(plan, GraphMode.PIECEWISE)


class _FakeCore:
    class _Runtime:
        pass

    RUNTIME = _Runtime()

    @staticmethod
    def _resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
        return (7, 0)


class SpecRowMappingUnitTest(unittest.TestCase):
    def _input_batch(self, *, num_drafts: int) -> SimpleNamespace:
        num_logits = [1, 1 + num_drafts]
        cu = np.array([0, num_logits[0], num_logits[0] + num_logits[1]], dtype=np.int32)
        return SimpleNamespace(
            num_reqs=2,
            req_ids=["reqA", "reqB"],
            is_prefilling_np=np.array([False, False]),
            cu_num_logits_np=cu,
            logits_indices=torch.tensor([10, 20, 21, 22, 23][: 2 + num_drafts], dtype=torch.long),
        )

    def _runtime_with_scratch(self, rows: int = 8) -> SimpleNamespace:
        return SimpleNamespace(
            decode_row_idx=torch.zeros(rows, dtype=torch.long),
            decode_valid_mask=torch.zeros(rows, dtype=torch.bool),
            decode_prompt_idx_buf=torch.zeros(rows, dtype=torch.long),
            decode_sample_idx_buf=torch.zeros(rows, dtype=torch.long),
            decode_identity_staging=CudaLongHostStaging(
                rows, torch.device("cpu"), channels=2
            ),
            decode_logit_row_selector_buf=torch.empty(rows, dtype=torch.long),
            decode_logit_row_selector_staging=CudaLongHostStaging(
                rows, torch.device("cpu")
            ),
            decode_count=0,
            event_step_id=1,
            workload_execution=None,
        )

    def test_multi_row_request_yields_strict_mapping(self) -> None:
        from tllm.backends.vllm.mrv2_localization import prepare_decode_localization_mrv2

        core = _FakeCore()
        core.RUNTIME = self._runtime_with_scratch()
        spec_rows = prepare_decode_localization_mrv2(
            core=core, runner=None, input_batch=self._input_batch(num_drafts=3)
        )
        self.assertEqual(len(spec_rows), 4)
        self.assertTrue(all(r.request_id == "reqB" for r in spec_rows))
        self.assertEqual([r.sampled_index for r in spec_rows], [0, 1, 2, 3])
        self.assertEqual([r.logit_row for r in spec_rows], [1, 2, 3, 4])
        self.assertEqual([r.hidden_row for r in spec_rows], [20, 21, 22, 23])
        self.assertTrue(all(r.prompt_idx == 7 for r in spec_rows))
        self.assertEqual(core.RUNTIME.decode_count, 2)


if __name__ == "__main__":
    unittest.main()
