from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.backends.protocol import BackendCapabilities, GraphSafety
from tllm.backends.timeline import export_main_workload_timeline
from tllm.contracts.hidden_locator import HiddenLocatorCapability
from tllm.consumers.override.config import OverRIDEConsumerConfig
from tllm.consumers.override.region_workload import OverRIDERegionWorkload
from tllm.ports.residual_stream import ResidualLocator
from tllm.runtime import residual_runtime
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.region_cost_probe import RegionCostProbe
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.sampler_bridge.types import SamplerStepView
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState


class _FakeEvent:
    def __init__(self, *, complete: bool = False, elapsed_ms: float = 0.0) -> None:
        self._complete = complete
        self._elapsed_ms = elapsed_ms
        self.synchronized = False

    def query(self) -> bool:
        return self._complete

    def synchronize(self) -> None:
        self._complete = True
        self.synchronized = True

    def elapsed_time(self, _end: "_FakeEvent") -> float:
        return self._elapsed_ms


class RegionCostProbeUnitTest(unittest.TestCase):
    def test_ewma_update_math(self) -> None:
        probe = RegionCostProbe(alpha=0.5)
        probe.record_elapsed_ns("region-a", "execute", "default", 100)
        reading = probe.record_elapsed_ns("region-a", "execute", "default", 300)
        self.assertEqual(reading.samples, 2)
        self.assertAlmostEqual(reading.ewma_ns, 200.0)
        self.assertEqual(reading.last_ns, 300)
        reading = probe.record_elapsed_ns("region-a", "execute", "default", 0)
        self.assertEqual(reading.samples, 3)
        self.assertAlmostEqual(reading.ewma_ns, 100.0)

    def test_alpha_validation(self) -> None:
        with self.assertRaises(ValueError):
            RegionCostProbe(alpha=0.0)
        with self.assertRaises(ValueError):
            RegionCostProbe(alpha=1.5)

    def test_event_harvest_defers_in_flight_pairs(self) -> None:
        probe = RegionCostProbe()
        pending_start = _FakeEvent(complete=False, elapsed_ms=0.25)
        pending_end = _FakeEvent(complete=False)
        probe.submit_events("region-a", "replay", "default", pending_start, pending_end)
        self.assertEqual(probe.pending_count, 1)
        self.assertEqual(dict(probe.readings()), {})

        ready_start = _FakeEvent(complete=True, elapsed_ms=0.5)
        ready_end = _FakeEvent(complete=True)
        probe.submit_events("region-b", "execute", "side_forward", ready_start, ready_end)
        # The in-flight pair blocks the FIFO; the ready pair queues behind it.
        self.assertEqual(probe.pending_count, 2)

        harvested = probe.harvest(block=True)
        self.assertEqual(harvested, 2)
        self.assertEqual(probe.pending_count, 0)
        readings = probe.readings()
        self.assertEqual(readings["region-a/replay"].last_ns, 250_000)
        self.assertEqual(readings["region-b/execute"].last_ns, 500_000)
        self.assertEqual(readings["region-b/execute"].stream, "side_forward")

    def test_report_estimates_only_default_stream_main_path_phases(self) -> None:
        probe = RegionCostProbe(alpha=1.0)
        probe.record_elapsed_ns("region-a", "execute", "default", 1000)
        probe.record_elapsed_ns("region-b", "replay", "default", 2000)
        probe.record_elapsed_ns("region-c", "execute", "side_forward", 9000)
        probe.record_elapsed_ns("region-a", "capture", "default", 50000)

        report = probe.report(signed_main_path_extension_ns=0)
        self.assertEqual(report.estimated_main_path_extension_ns, 3000)
        self.assertEqual(report.signed_main_path_extension_ns, 0)
        self.assertEqual(len(report.readings), 4)
        self.assertTrue(any("read-only" in line for line in report.explanations))
        self.assertTrue(any("side-stream" in line for line in report.explanations))
        self.assertTrue(any("never" in line for line in report.explanations))

    def test_measure_uses_perf_counter_when_cuda_unavailable(self) -> None:
        probe = RegionCostProbe()
        with mock.patch("torch.cuda.is_available", return_value=False):
            with probe.measure("region-a", "execute", "default"):
                pass
        readings = probe.readings()
        self.assertEqual(readings["region-a/execute"].samples, 1)


class RegionCostBackfillIntegrationTest(unittest.TestCase):
    """Drive one signed native step and read back observed region costs."""

    def test_signed_step_backfills_readonly_observed_costs(self) -> None:
        workload = OverRIDERegionWorkload(
            OverRIDEConsumerConfig(
                implementation="region",
                hidden_size=4,
                vocab_size=7,
                rank=2,
                num_iterations=2,
            )
        )
        runtime = TllmRuntime(
            workloads=(workload,), workload_memory_budget_bytes=1 << 20
        )
        runtime.cold_specialize_workloads(
            RuntimeSpecializationSpec(2, 4, torch.float32, torch.device("cpu"))
        )
        capabilities = BackendCapabilities(
            backend_id="vllm-0.27-mrv2",
            runner_family="mrv2",
            sampler_stages=frozenset(),
            native_min_p=False,
            tuple_sampler_return=False,
            split_execute_and_sample=True,
            graph_safety=GraphSafety.EAGER_ONLY,
            hidden_locators=frozenset({HiddenLocatorCapability.BLOCK_OUTPUT_DECODE}),
        )
        compilation = runtime.compile_workloads(
            export_main_workload_timeline(
                capabilities, graph_mode="eager", static_headroom_bytes=1 << 20
            )
        )
        signed_before = compilation.decision.cost

        state = ResidualRuntimeState()
        locator = ResidualLocator(0, "block_output", "decode")
        state.residual_bindings = {"source": ResidualPathBinding(locator, "source")}
        state.tap_decode_hidden = {"source": torch.randn(2, 4)}
        state.decode_count = 2
        state.decode_request_ids = ["0_a", "1_b"]
        state.decode_prompt_idxs = [0, 1]
        state.decode_sample_idxs = [0, 1]

        with mock.patch.object(residual_runtime, "RUNTIME", state):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=runtime.workload_execution_bindings,
            )
            residual_runtime.prepare_physical_workgraph_step(7)
            regions = compilation.decision.execution_regions
            residual_runtime.advance_workload_anchor(regions[0].placement_anchor)
            # Stage the sampler-boundary training rows exactly as the bridge
            # does between the reweight and train region advances.
            provider = workload.sampler_provider
            view = SamplerStepView(
                engine_step_id=7,
                phase="decode",
                logits=torch.randn(2, 7),
                sampling_metadata=object(),
                decode_count=2,
                request_ids=("0_a", "1_b"),
                prompt_idxs=(0, 1),
                sample_idxs=(0, 1),
                prompt_idx_tensor=None,
                sample_idx_tensor=None,
                source_hidden=state.tap_decode_hidden["source"],
                device=torch.device("cpu"),
                model=object(),
                runner=object(),
            )
            modifier = provider.prepare_step(view)
            assert modifier is not None
            provider.after_sample(view, torch.tensor([2, 4], dtype=torch.long))
            for region in regions[1:]:
                residual_runtime.advance_workload_anchor(region.placement_anchor)
            execution = state.workload_execution
            assert execution is not None
            report = execution.region_cost_report(block=True)
            execution.finish_step(7)

        # Every signed region phase was sampled at least once.
        for region in compilation.decision.execution_regions:
            phases = [
                key
                for key in report.readings
                if key.startswith(f"{region.region_id}/")
            ]
            self.assertTrue(phases, f"no cost reading for {region.region_id}")
            for key in phases:
                self.assertGreaterEqual(report.readings[key].samples, 1)
        self.assertGreaterEqual(report.estimated_main_path_extension_ns, 0)
        # Read-only guarantee: the signed decision cost is byte-identical.
        self.assertEqual(compilation.decision.cost, signed_before)
        self.assertEqual(report.signed_main_path_extension_ns, 0)
        self.assertTrue(any("read-only" in line for line in report.explanations))


if __name__ == "__main__":
    unittest.main()
