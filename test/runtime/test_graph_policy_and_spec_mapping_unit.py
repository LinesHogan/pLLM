#!/usr/bin/env python3
"""Unit tests for graph policy and strict spec-decode row mapping."""

from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np
import torch

from tllm.contracts.request_identity import resolve_prompt_sample
from tllm.runtime.request_trainable_state import RequestIdentityBatch
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

    def _resolve_prompt_sample_for_req_id(self, req_id: str) -> tuple[int, int]:
        return resolve_prompt_sample(
            req_id,
            prompt_indices=self.RUNTIME.reqid_to_promptidx,
            sample_indices=self.RUNTIME.reqid_to_sampleidx,
        )


class _NativeIdentityRecorder:
    requires_request_identities = True

    def __init__(self) -> None:
        self.sampler_identity: (
            tuple[int, tuple[str, ...], tuple[int, ...], torch.Tensor | None]
            | None
        ) = None
        self.request_identities: tuple[int, RequestIdentityBatch] | None = None

    def stage_sampler_identity(
        self,
        step_id: int,
        request_ids: tuple[str, ...],
        sample_indices: tuple[int, ...],
        sample_index_tensor: torch.Tensor | None,
    ) -> None:
        self.sampler_identity = (
            step_id,
            request_ids,
            sample_indices,
            sample_index_tensor,
        )

    def stage_request_identities(
        self, step_id: int, batch: RequestIdentityBatch
    ) -> None:
        self.request_identities = (step_id, batch)


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
            reqid_to_promptidx={"reqA": 7, "reqB": 7},
            reqid_to_sampleidx={},
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

    def test_mrv1_spec_decode_restores_first_rows_and_stages_parent_identities(
        self,
    ) -> None:
        from tllm.backends.vllm.v027_mrv1 import (
            _build_spec_rows_mrv1,
            _restore_first_row_localization_mrv1,
        )
        from tllm.runtime.step_session import StepSession

        parent_id = "parent_with_underscores"
        child_ids = (
            f"0_{parent_id}",
            f"1_{parent_id}",
        )
        execution = _NativeIdentityRecorder()
        core = _FakeCore()
        core.RUNTIME = self._runtime_with_scratch()
        core.RUNTIME.reqid_to_promptidx = {parent_id: 7}
        core.RUNTIME.decode_request_ids = list(child_ids)
        core.RUNTIME.decode_prompt_idxs = [7, 7]
        core.RUNTIME.decode_sample_idxs = [0, 1]
        core.RUNTIME.workload_execution = execution

        logits_indices = torch.tensor([10, 11, 12, 20, 21], dtype=torch.long)
        core.RUNTIME.decode_row_idx[:2].copy_(logits_indices[:2])
        view = SimpleNamespace(
            logits_indices=logits_indices,
            spec_decode_metadata=SimpleNamespace(num_draft_tokens=(2, 1)),
        )
        session = StepSession(
            step_id=1,
            backend_id="vllm-0.27-mrv1",
            request_ids=child_ids,
            request_index={
                request_id: index
                for index, request_id in enumerate(child_ids)
            },
        )

        spec_rows = _build_spec_rows_mrv1(core=core, view=view, session=session)
        _restore_first_row_localization_mrv1(
            core=core,
            view=view,
            session=session,
        )

        self.assertEqual(
            [
                (
                    row.request_id,
                    row.prompt_idx,
                    row.sample_idx,
                    row.sampled_index,
                    row.logit_row,
                    row.hidden_row,
                )
                for row in spec_rows
            ],
            [
                (child_ids[0], 7, 0, 0, 0, 10),
                (child_ids[0], 7, 0, 1, 1, 11),
                (child_ids[0], 7, 0, 2, 2, 12),
                (child_ids[1], 7, 1, 0, 3, 20),
                (child_ids[1], 7, 1, 1, 4, 21),
            ],
        )
        self.assertEqual(core.RUNTIME.decode_row_idx[:2].tolist(), [10, 20])
        self.assertEqual(core.RUNTIME.decode_count, 2)
        assert execution.sampler_identity is not None
        sampler_step, sampler_requests, sampler_indices, sampler_index_tensor = (
            execution.sampler_identity
        )
        self.assertEqual(
            (sampler_step, sampler_requests, sampler_indices),
            (1, child_ids, (0, 1)),
        )
        self.assertIs(sampler_index_tensor, core.RUNTIME.decode_sample_idx_tensor)
        assert execution.request_identities is not None
        identity_step, identity_batch = execution.request_identities
        self.assertEqual(identity_step, 1)
        self.assertEqual(
            tuple(
                (
                    identity.request_id,
                    identity.prompt_idx,
                    identity.sample_idx,
                )
                for identity in identity_batch.identities
            ),
            ((parent_id, 7, 0), (parent_id, 7, 1)),
        )


if __name__ == "__main__":
    unittest.main()
