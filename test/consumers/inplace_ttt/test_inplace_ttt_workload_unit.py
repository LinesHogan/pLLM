"""Native exact In-Place-TTT and overlay lifecycle contracts."""

from __future__ import annotations

from typing import cast
import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.inplace_ttt.workload import (
    InPlaceTTTConfig,
    InPlaceTTTFinishBatch,
    InPlaceTTTKey,
    InPlaceTTTMetadata,
    InPlaceTTTPhase,
    InPlaceTTTWorkload,
)
from tllm.contracts.request_identity import RequestIdentity
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_bindings import LocalRuntimeServices
from tllm.schedule import MainWorkloadTimeline, WorkloadCompilation


def _timeline() -> MainWorkloadTimeline:
    return export_main_workload_timeline(
        BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
        ),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )


def _build(
    config: InPlaceTTTConfig,
) -> tuple[InPlaceTTTWorkload, TllmRuntime[InPlaceTTTWorkload], WorkloadCompilation]:
    workload = InPlaceTTTWorkload(config)
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(2, 3, torch.float64, torch.device("cpu"))
    )
    return workload, runtime, runtime.compile_workloads(_timeline())


def _drive(
    runtime: TllmRuntime[InPlaceTTTWorkload],
    compilation: WorkloadCompilation,
    step_id: int,
) -> tuple[int, int]:
    state = ResidualRuntimeState()
    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation, bindings=runtime.workload_execution_bindings
        )
        residual_runtime.prepare_physical_workgraph_step(step_id)
        for region in compilation.decision.execution_regions:
            residual_runtime.advance_workload_anchor(region.placement_anchor)
        execution = state.workload_execution
        assert execution is not None
        return (
            execution.state_versions.observed_version("fast_weights", None),
            execution.state_versions.observed_version("overlay_versions", None),
        )


class InPlaceTTTWorkloadUnitTest(unittest.TestCase):
    def test_exact_chunk_output_precedes_update_and_publishes_owned_weight(self) -> None:
        config = InPlaceTTTConfig(
            layers=(0,), chunk_size=2, learning_rate=0.3
        )
        workload, runtime, compilation = _build(config)
        request = RequestIdentity("req-0", 0, 0)
        hidden = torch.tensor(
            [[1.0, -0.5, 0.25], [0.2, 0.4, -0.3]], dtype=torch.float64
        )
        target = torch.tensor(
            [[0.5, -0.25], [0.1, 0.75]], dtype=torch.float64
        )
        base = torch.tensor(
            [[0.2, -0.3, 0.5], [0.4, 0.1, -0.2]], dtype=torch.float64
        )
        expected_output = hidden @ base.transpose(0, 1)
        expected_weight = base + 0.3 * (target.transpose(0, 1) @ hidden)
        runtime.stage_pair(
            InPlaceTTTWorkload.batch_tensors,
            {"hidden": hidden, "target": target, "base_weight": base},
            InPlaceTTTWorkload.batch_metadata,
            InPlaceTTTMetadata(InPlaceTTTKey(request, 0), InPlaceTTTPhase.PREFILL),
        )

        self.assertEqual(_drive(runtime, compilation, 1), (1, 1))
        step = runtime.drain(InPlaceTTTWorkload.steps)[0]
        torch.testing.assert_close(step.output, expected_output)
        torch.testing.assert_close(step.weight, expected_weight)
        publication = runtime.drain(InPlaceTTTWorkload.publications)[0]
        assert publication.version is not None
        materialized = workload.overlay_versions.store.materialize(
            publication.version
        )
        torch.testing.assert_close(materialized["weight"], expected_weight)
        materialized["weight"].zero_()
        torch.testing.assert_close(
            workload.overlay_versions.store.materialize(publication.version)["weight"],
            expected_weight,
        )
        self.assertEqual(workload.update_count, 1)

    def test_tail_and_decode_disabled_are_conditional_state_transitions(self) -> None:
        config = InPlaceTTTConfig(chunk_size=4, update_decode=False)
        workload, runtime, compilation = _build(config)
        request = RequestIdentity("req-tail", 1, 0)
        hidden = torch.ones((2, 3), dtype=torch.float64)
        target = torch.ones((2, 2), dtype=torch.float64)
        base = torch.ones((2, 3), dtype=torch.float64)
        runtime.stage_pair(
            InPlaceTTTWorkload.batch_tensors,
            {"hidden": hidden, "target": target, "base_weight": base},
            InPlaceTTTWorkload.batch_metadata,
            InPlaceTTTMetadata(InPlaceTTTKey(request, 0), InPlaceTTTPhase.PREFILL),
        )
        self.assertEqual(_drive(runtime, compilation, 1), (1, 0))
        self.assertEqual(runtime.drain(InPlaceTTTWorkload.steps)[0].tail_rows, 2)
        self.assertIsNone(runtime.drain(InPlaceTTTWorkload.publications)[0].version)

        decode_request = RequestIdentity("req-decode", 2, 0)
        runtime.stage_pair(
            InPlaceTTTWorkload.batch_tensors,
            {"hidden": hidden, "target": target, "base_weight": base},
            InPlaceTTTWorkload.batch_metadata,
            InPlaceTTTMetadata(
                InPlaceTTTKey(decode_request, 0), InPlaceTTTPhase.DECODE
            ),
        )
        self.assertEqual(_drive(runtime, compilation, 2), (1, 0))
        self.assertFalse(runtime.drain(InPlaceTTTWorkload.steps)[0].state_changed)
        self.assertEqual(workload.update_count, 0)

    def test_finished_request_releases_tail_and_overlay_lease(self) -> None:
        workload, runtime, compilation = _build(
            InPlaceTTTConfig(chunk_size=2)
        )
        request = RequestIdentity("req-finish", 0, 0)
        runtime.stage_pair(
            InPlaceTTTWorkload.batch_tensors,
            {
                "hidden": torch.ones((2, 3), dtype=torch.float64),
                "target": torch.ones((2, 2), dtype=torch.float64),
                "base_weight": torch.ones((2, 3), dtype=torch.float64),
            },
            InPlaceTTTWorkload.batch_metadata,
            InPlaceTTTMetadata(InPlaceTTTKey(request, 0), InPlaceTTTPhase.PREFILL),
        )
        _drive(runtime, compilation, 1)
        self.assertIsNotNone(workload.overlay_versions.store.pinned(request))
        runtime.stage(
            InPlaceTTTWorkload.finished_requests,
            InPlaceTTTFinishBatch((request,)),
        )

        self.assertEqual(_drive(runtime, compilation, 2), (2, 2))
        receipt = runtime.drain(InPlaceTTTWorkload.cleanup)[0]
        self.assertEqual(receipt.cleared_slots, 1)
        self.assertEqual(receipt.released_versions, 1)
        self.assertIsNone(workload.overlay_versions.store.pinned(request))

    def test_config_contains_algorithm_semantics_not_scheduler_controls(self) -> None:
        self.assertEqual(
            InPlaceTTTConfig.field_names(),
            (
                "enabled",
                "layers",
                "chunk_size",
                "learning_rate",
                "update_prefill",
                "update_decode",
            ),
        )
        services = cast(
            LocalRuntimeServices,
            InPlaceTTTWorkload().native_workload_resources().services,
        )
        self.assertEqual(services.diagnostics()["update_count"], 0)


if __name__ == "__main__":
    unittest.main()
