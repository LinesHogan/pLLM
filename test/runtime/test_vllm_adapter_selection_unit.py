#!/usr/bin/env python3
"""Unit tests for vLLM prepare-inputs adapter selection and parsing."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest import mock

import torch


class _Runner:
    def __init__(self) -> None:
        self.input_batch = SimpleNamespace(
            req_ids=["reqA", "reqB", None],
            num_reqs=2,
            req_id_to_index={"reqA": 0, "reqB": 1},
            num_prompt_tokens=[4, 5],
            num_computed_tokens_cpu=[4, 3],
        )


class VllmAdapterSelectionUnitTest(unittest.TestCase):
    def test_inprocess_runner_binding_unwraps_engine_core_client(self) -> None:
        from tllm.runtime.vllm_adapter import bind_inprocess_runner

        class Runner:
            pass

        runner = Runner()
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                engine_core=SimpleNamespace(
                    engine_core=SimpleNamespace(
                        model_executor=SimpleNamespace(
                            driver_worker=SimpleNamespace(model_runner=runner)
                        )
                    )
                )
            )
        )
        binding = object()
        with mock.patch(
            "tllm.backends.protocol.runner_binding_for", return_value=binding
        ):
            result = bind_inprocess_runner(llm, object())

        self.assertIsNotNone(result)
        assert result is not None
        self.assertIs(result.binding, binding)

    def test_engine_core_topology_distinguishes_uniproc_from_mp_transport(self) -> None:
        """collective_rpc exists in both topologies; client type is decisive."""

        from vllm.v1.engine.core_client import InprocClient, SyncMPClient

        from tllm.runtime.vllm_adapter import (
            EngineCoreTopology,
            engine_core_topology,
        )

        def llm_for(client: object) -> SimpleNamespace:
            return SimpleNamespace(llm_engine=SimpleNamespace(engine_core=client))

        self.assertIs(
            engine_core_topology(llm_for(object.__new__(InprocClient))),
            EngineCoreTopology.INPROCESS,
        )
        self.assertIs(
            engine_core_topology(llm_for(object.__new__(SyncMPClient))),
            EngineCoreTopology.MULTIPROCESS,
        )
        self.assertIsNone(engine_core_topology(object()))

    def test_unknown_major_fails_fast(self) -> None:
        from tllm.runtime.vllm_patch.adapters import select_prepare_inputs_adapter

        with self.assertRaises(RuntimeError):
            select_prepare_inputs_adapter("1.0.0")

    def test_v027_adapter_parses_logits_indices_first_tuple(self) -> None:
        from tllm.runtime.vllm_patch.adapters import select_prepare_inputs_adapter

        adapter = select_prepare_inputs_adapter("0.27.1")
        runner = _Runner()
        logits_indices = torch.tensor([1, 3], dtype=torch.long)
        spec_decode_metadata = SimpleNamespace(kind="spec")
        out = (logits_indices, spec_decode_metadata)

        import numpy as np

        view = adapter.unpack_prepare_inputs_output(
            runner=runner,
            scheduler_output=SimpleNamespace(num_scheduled_tokens={"reqA": 1, "reqB": 2}),
            out=out,
            num_scheduled_tokens=np.array([1, 2], dtype=np.int32),
        )

        self.assertIsNone(view.attn_metadata)
        self.assertTrue(torch.equal(view.logits_indices, logits_indices))
        self.assertIsNone(view.spec_decode_common)
        self.assertEqual(view.num_actual_tokens, 3)


if __name__ == "__main__":
    unittest.main()
