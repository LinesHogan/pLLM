from __future__ import annotations

from types import SimpleNamespace
import sys
import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.workflows import esamp_support
from tllm.workflows.benchmarks import per_request_esamp_benchmark


class Phase2WorkGraphExecutionToggleUnitTest(unittest.TestCase):
    def test_disabled_config_skips_cold_specialization_and_keeps_legacy_engine_path(self) -> None:
        default = ESampConsumerConfig()
        self.assertIs(default.enable_workgraph_execution, True)

        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_workgraph_execution=False),
            engine=engine,
        )
        spec = RuntimeSpecializationSpec(
            rows=2,
            hidden_size=4,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        with mock.patch.object(consumer, "specialize_workgraph") as specialize:
            consumer.cold_specialize_workgraph(spec)
        specialize.assert_not_called()
        self.assertEqual(consumer.workgraph_captures(), ())
        self.assertEqual(consumer.workgraph_attachments(), ())
        self.assertIsNone(consumer._workgraph_executor)
        engine.launch_step.assert_not_called()

    def test_workflow_configure_passes_workgraph_execution_toggle_to_consumer_config(self) -> None:
        runtime = esamp_support.runtime
        saved_consumer = runtime.RUNTIME.consumer
        try:
            runtime.RUNTIME.consumer = None
            with (
                mock.patch.object(runtime, "clear_dispatch_consumers"),
                mock.patch.object(runtime, "set_runtime_consumer"),
                mock.patch.object(runtime, "configure_runtime"),
            ):
                consumer = esamp_support.configure_esamp_runtime(
                    enable_workgraph_execution=False,
                )
            self.assertFalse(consumer.config.enable_workgraph_execution)
        finally:
            runtime.RUNTIME.consumer = saved_consumer

    def test_per_request_cli_exposes_toggle_and_case_stats_report_contract(self) -> None:
        with mock.patch.object(
            sys,
            "argv",
            ["per_request_esamp_benchmark", "--no-workgraph-execution"],
        ):
            args = per_request_esamp_benchmark._parse_args()
        self.assertFalse(args.workgraph_execution)

        stats = SimpleNamespace(
            active_plan_epoch=1,
            prepare_count=2,
            execute_count=3,
            finish_count=4,
            replay_count=5,
            eager_count=6,
            fallback_count=7,
            fence_wait_count=8,
            fence_record_count=9,
            bucket_miss_count=10,
            specialization_request_count=11,
            active_leases=12,
            retired_plans=13,
            deadline_flush_count=14,
        )
        fields = per_request_esamp_benchmark._workgraph_runtime_stats_fields(stats)
        self.assertEqual(set(fields), set(vars(stats)))


if __name__ == "__main__":
    unittest.main()
