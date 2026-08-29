"""Contracts for the typed LoReFT workload."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.reft.workload import ReftConfig, ReftWorkload
from tllm.consumers.reft.vllm_patch import (
    _prompt_rows,
    install_reft_patch_on_llm,
)
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
    def test_llm_patch_resolves_active_workload_after_epoch_replacement(self) -> None:
        class _Layer(torch.nn.Module):
            def forward(
                self,
                positions: torch.Tensor,
                hidden: torch.Tensor,
                residual: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor]:
                del positions, residual
                return (hidden,)

        model = torch.nn.Module()
        body = torch.nn.Module()
        body.layers = torch.nn.ModuleList([_Layer()])
        model.model = body
        llm = SimpleNamespace(
            llm_engine=SimpleNamespace(
                model_executor=SimpleNamespace(
                    driver_worker=SimpleNamespace(
                        model_runner=SimpleNamespace(model=model)
                    )
                )
            )
        )
        initial = ReftWorkload(ReftConfig(layers=(0,), rank=_RANK))
        replacement = ReftWorkload(ReftConfig(layers=(0,), rank=_RANK))
        runtime = mock.create_autospec(TllmRuntime, instance=True)
        runtime.local_workload.side_effect = [initial, initial, replacement]

        handle = install_reft_patch_on_llm(llm, runtime)
        try:
            layer = body.layers[0]
            positions = torch.arange(2)
            hidden = torch.randn(2, _HIDDEN)
            _ = layer(positions, hidden)
            _ = layer(positions, hidden)
        finally:
            handle.restore()
        self.assertEqual(runtime.local_workload.call_count, 3)
        runtime.local_workload.assert_called_with(initial)

    def test_config_constraints_are_rust_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "constraint"):
            ReftConfig(rank=0)
        with self.assertRaisesRegex(ValueError, "constraint"):
            ReftConfig(last_tokens=-1)

    def test_cold_specialization_initializes_loreft_parameters(self) -> None:
        workload = ReftWorkload(ReftConfig(layers=(1, 3), rank=_RANK))
        workload.cold_specialize(_spec())
        state = workload.intervention
        self.assertEqual(tuple(state.source_w.shape), (2, _RANK, _HIDDEN))
        self.assertEqual(tuple(state.source_b.shape), (2, _RANK))
        self.assertEqual(tuple(state.rotate.shape), (2, _HIDDEN, _RANK))
        gram = state.rotate[0].T @ state.rotate[0]
        self.assertTrue(torch.allclose(gram, torch.eye(_RANK), atol=1e-5))

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

    def test_prompt_row_rule_stays_in_consumer_integration(self) -> None:
        rows = _prompt_rows(
            positions=torch.tensor([0, 1, 2, 0, 1]),
            query_start=torch.tensor([0, 3, 5]),
            sequence_lengths=torch.tensor([3, 2]),
            computed=torch.tensor([0, 0]),
            first_tokens=1,
            last_tokens=1,
        )
        self.assertEqual(rows.tolist(), [0, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
