#!/usr/bin/env python3
"""Unit tests for graph policy and strict spec-decode row mapping."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch


class GraphPolicyUnitTest(unittest.TestCase):
    def test_eager_runner_resolves_eager(self) -> None:
        from tllm.runtime.graph_policy import GraphMode, may_launch_from_hooks, resolve_graph_mode

        runner = SimpleNamespace(enforce_eager=True)
        self.assertEqual(resolve_graph_mode(runner), GraphMode.EAGER)
        self.assertTrue(may_launch_from_hooks(GraphMode.EAGER))

    def test_piecewise_and_full_modes(self) -> None:
        from tllm.runtime.graph_policy import GraphMode, may_launch_from_hooks, resolve_graph_mode

        piecewise = SimpleNamespace(
            enforce_eager=False,
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="PIECEWISE"), level=3),
        )
        full = SimpleNamespace(
            enforce_eager=False,
            compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name="FULL"), level=3),
        )
        self.assertEqual(resolve_graph_mode(piecewise), GraphMode.PIECEWISE)
        self.assertEqual(resolve_graph_mode(full), GraphMode.FULL)
        self.assertFalse(may_launch_from_hooks(GraphMode.PIECEWISE))
        self.assertFalse(may_launch_from_hooks(GraphMode.FULL))

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

    def test_legacy_predicate_delegates(self) -> None:
        from tllm.runtime import residual_runtime as rr

        eager = SimpleNamespace(enforce_eager=True, use_cuda_graph=False, compilation_config=None)
        graphs = SimpleNamespace(
            enforce_eager=False,
            use_cuda_graph=True,
            compilation_config=None,
        )
        self.assertFalse(rr.runner_uses_compilation_or_cudagraph(eager))
        self.assertTrue(rr.runner_uses_compilation_or_cudagraph(graphs))


class _FakeCore:
    class _Runtime:
        pass

    RUNTIME = _Runtime()

    @staticmethod
    def _resolve_prompt_sample_for_req_id(req_id: str) -> tuple[int, int]:
        return (7, 0)


class SpecRowMappingUnitTest(unittest.TestCase):
    def _input_batch(self, *, num_drafts: int) -> SimpleNamespace:
        # Two decode requests; request 1 carries `num_drafts` draft tokens.
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
            decode_count=0,
        )

    def test_single_row_requests_yield_empty_spec_mapping(self) -> None:
        from tllm.backends.vllm.mrv2_localization import prepare_decode_localization_mrv2

        core = _FakeCore()
        core.RUNTIME = self._runtime_with_scratch()
        spec_rows = prepare_decode_localization_mrv2(
            core=core, runner=None, input_batch=self._input_batch(num_drafts=0)
        )
        self.assertEqual(spec_rows, ())
        self.assertEqual(core.RUNTIME.decode_count, 2)

    def test_multi_row_request_yields_strict_mapping(self) -> None:
        from tllm.backends.vllm.mrv2_localization import prepare_decode_localization_mrv2

        core = _FakeCore()
        core.RUNTIME = self._runtime_with_scratch()
        spec_rows = prepare_decode_localization_mrv2(
            core=core, runner=None, input_batch=self._input_batch(num_drafts=3)
        )
        # Only request B has multiple rows.
        self.assertEqual(len(spec_rows), 4)
        self.assertTrue(all(r.request_id == "reqB" for r in spec_rows))
        self.assertEqual([r.sampled_index for r in spec_rows], [0, 1, 2, 3])
        self.assertEqual([r.logit_row for r in spec_rows], [1, 2, 3, 4])
        self.assertEqual([r.hidden_row for r in spec_rows], [20, 21, 22, 23])
        self.assertTrue(all(r.prompt_idx == 7 for r in spec_rows))
        # Training scratch keeps the first-row view (both requests, 2 rows).
        self.assertEqual(core.RUNTIME.decode_count, 2)

    def test_mrv1_spec_rows_from_metadata(self) -> None:
        from tllm.backends.vllm.v027_mrv1 import _build_spec_rows_mrv1
        from tllm.runtime.step_session import StepSession

        session = StepSession(
            step_id=1,
            backend_id="vllm-0.27-mrv1",
            request_ids=("reqA", "reqB"),
            request_index={"reqA": 0, "reqB": 1},
        )
        metadata = SimpleNamespace(num_draft_tokens=[0, 2])
        view = SimpleNamespace(
            spec_decode_metadata=metadata,
            logits_indices=torch.tensor([10, 20, 21, 22], dtype=torch.long),
        )
        rows = _build_spec_rows_mrv1(core=_FakeCore(), view=view, session=session)
        self.assertEqual([r.request_id for r in rows], ["reqB", "reqB", "reqB"])
        self.assertEqual([r.sampled_index for r in rows], [0, 1, 2])
        self.assertEqual([r.hidden_row for r in rows], [20, 21, 22])

    def test_mrv1_spec_rows_empty_without_metadata(self) -> None:
        from tllm.backends.vllm.v027_mrv1 import _build_spec_rows_mrv1
        from tllm.runtime.step_session import StepSession

        session = StepSession(step_id=1, backend_id="vllm-0.27-mrv1")
        view = SimpleNamespace(spec_decode_metadata=None, logits_indices=None)
        self.assertEqual(_build_spec_rows_mrv1(core=_FakeCore(), view=view, session=session), ())


if __name__ == "__main__":
    unittest.main()
