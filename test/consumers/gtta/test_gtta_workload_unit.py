"""Native GTTA state, queue, and signed-task contracts."""

from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.consumers.gtta import (
    GTTAConfig,
    GTTAEnvironmentService,
    GTTAQuery,
    GTTATrajectoryBatch,
    GTTAWorkload,
    TrajectoryFeedback,
)
from tllm.runtime import residual_runtime
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
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


def _build() -> tuple[GTTAWorkload, TllmRuntime[GTTAWorkload], WorkloadCompilation]:
    workload = GTTAWorkload(GTTAConfig(score_momentum=0.25))
    runtime = TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    )
    runtime.cold_specialize_workloads(
        RuntimeSpecializationSpec(1, 1, torch.float32, torch.device("cpu"))
    )
    return workload, runtime, runtime.compile_workloads(_timeline())


def _drive(
    runtime: TllmRuntime[GTTAWorkload], compilation: WorkloadCompilation
) -> int:
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
        return execution.state_versions.observed_version("dynamics", None)


class GTTAWorkloadUnitTest(unittest.TestCase):
    def test_environment_service_preserves_bounded_fifo(self) -> None:
        service = GTTAEnvironmentService(GTTAConfig(max_queue_size=1))
        first = TrajectoryFeedback(
            task_id="maps",
            action="open directions",
            previous_observation="map results",
            next_observation="directions panel",
            reward=0.75,
        )
        second = TrajectoryFeedback(
            task_id="maps",
            action="clear",
            previous_observation="map results",
            reward=-1.0,
        )
        self.assertTrue(service.enqueue_feedback(first))
        self.assertFalse(service.enqueue_feedback(second))
        self.assertEqual(service.dropped_trajectories, 1)
        self.assertEqual(service.drain().entries[0].action, "open directions")

    def test_signed_update_matches_sequential_ema_and_query_order(self) -> None:
        workload, runtime, compilation = _build()
        service = GTTAEnvironmentService(GTTAConfig(max_queue_size=8))
        for reward in (1.0, 0.5):
            service.enqueue_feedback(
                TrajectoryFeedback(
                    task_id="maps",
                    action="open directions",
                    previous_observation="Map   Results",
                    next_observation="directions panel",
                    reward=reward,
                )
            )
        service.enqueue_feedback(
            TrajectoryFeedback(
                task_id="maps",
                action="clear",
                previous_observation="Map Results",
                next_observation="empty page",
                reward=-0.25,
            )
        )
        runtime.stage(GTTAWorkload.trajectories, service.drain())
        runtime.stage(
            GTTAWorkload.queries,
            GTTAQuery(
                prompt_idx=4,
                task_id="maps",
                observation="map results",
                candidate_actions=("clear", "unknown", "open directions"),
            ),
        )

        self.assertEqual(_drive(runtime, compilation), 1)
        receipt = runtime.drain(GTTAWorkload.updates)[0]
        self.assertEqual(receipt.processed, 3)
        self.assertEqual(receipt.state_version, 1)
        result = runtime.drain(GTTAWorkload.query_results)[0]
        self.assertEqual(
            result.ranked_actions,
            (("open directions", 0.5625), ("unknown", 0.0), ("clear", -0.1875)),
        )
        self.assertEqual(result.prompt_context.prompt_idx, 4)
        self.assertIn('"action_taken": "open directions"', result.prompt_context.text)
        self.assertEqual(workload.state_version, 1)

    def test_empty_batch_is_a_signed_noop_transition(self) -> None:
        _workload, runtime, compilation = _build()
        runtime.stage(GTTAWorkload.trajectories, GTTATrajectoryBatch(()))

        self.assertEqual(_drive(runtime, compilation), 0)
        self.assertEqual(runtime.drain(GTTAWorkload.updates)[0].processed, 0)

    def test_config_has_no_workload_scheduler_controls(self) -> None:
        self.assertEqual(
            GTTAConfig.field_names(),
            ("enabled", "max_queue_size", "score_momentum"),
        )
        with self.assertRaisesRegex(ValueError, "constraint"):
            GTTAConfig(score_momentum=1.1)


if __name__ == "__main__":
    unittest.main()
