"""Contracts for the typed LoReFT workload."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.reft.workload import ReftConfig, ReftWorkload
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState

_HIDDEN = 8
_RANK = 2


def _spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=_HIDDEN,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )


def _timeline() -> object:
    capabilities = BackendCapabilities(
        backend_id="vllm-0.27-mrv2",
        runner_family="mrv2",
        sampler_stages=frozenset(),
        native_min_p=False,
        tuple_sampler_return=False,
        split_execute_and_sample=True,
        graph_safety=GraphSafety.EAGER_ONLY,
        hidden_locators=frozenset(),
    )
    return export_main_workload_timeline(
        capabilities, graph_mode="eager", static_headroom_bytes=1 << 20
    )


def _compile(workload: ReftWorkload) -> tuple[TllmRuntime, object]:
    runtime = TllmRuntime(workloads=(workload,), workload_memory_budget_bytes=1 << 20)
    runtime.cold_specialize_workloads(_spec())
    return runtime, runtime.compile_workloads(_timeline())


def _drive(runtime: TllmRuntime, compilation: object) -> int:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(7)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        return execution.state_versions.observed_version("intervention", None)


class ReftWorkloadUnitTest(unittest.TestCase):
    def test_publish_is_generated_and_advances_signed_state(self) -> None:
        workload = ReftWorkload(ReftConfig(layers=(1,), rank=_RANK))
        runtime, compilation = _compile(workload)
        self.assertEqual(
            tuple(region.tasks for region in compilation.decision.execution_regions),
            (("publish__intervention",),),
        )
        source_w = torch.randn(1, _RANK, _HIDDEN)
        source_b = torch.randn(1, _RANK)
        rotate = torch.randn(1, _HIDDEN, _RANK)
        runtime.publish(
            ReftWorkload.intervention,
            {"source_w": source_w, "source_b": source_b, "rotate": rotate},
        )
        self.assertEqual(_drive(runtime, compilation), 1)
        self.assertTrue(torch.equal(workload.intervention.source_w, source_w))
        self.assertTrue(torch.equal(workload.intervention.rotate, rotate))

    def test_missing_publish_skips_without_version_change(self) -> None:
        workload = ReftWorkload(ReftConfig(layers=(1,), rank=_RANK))
        runtime, compilation = _compile(workload)
        self.assertEqual(_drive(runtime, compilation), 0)

    def test_loreft_math_stays_in_consumer(self) -> None:
        workload = ReftWorkload(ReftConfig(layers=(1,), rank=_RANK))
        workload.cold_specialize(_spec())
        rows = torch.randn(4, _HIDDEN)
        state = workload.intervention
        expected = (
            torch.nn.functional.linear(rows, state.source_w[0], state.source_b[0])
            - rows @ state.rotate[0]
        ) @ state.rotate[0].T
        self.assertTrue(torch.allclose(workload.loreft_delta(0, rows), expected))


if __name__ == "__main__":
    unittest.main()
