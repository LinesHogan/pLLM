from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.runtime.specialization import RuntimeSpecializationSpec


class WorkGraphWorkGraphExecutionToggleUnitTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
