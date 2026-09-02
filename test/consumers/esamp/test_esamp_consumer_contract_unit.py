#!/usr/bin/env python3
"""Unit tests for the current ESamp public consumer contract."""

from __future__ import annotations

from dataclasses import replace
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tllm.consumers.esamp import ESampConsumer, ESampConsumerConfig
from tllm.consumers.esamp.engine import ESampTrainEngine
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.address import port_address_desc
from tllm.ports.base import PortKind
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime import residual_runtime as esamp_runtime
from tllm.workgraph.frontend import (
    CaptureStatus,
    CapturedFlow,
    PortBindingDesc,
    PortBindingDirection,
)
from tllm.workgraph.hierarchy import (
    CaptureAttachment,
)
from tllm.workgraph.ir import (
    NodeKey,
)


class ESampConsumerContractUnitTest(unittest.TestCase):
    def test_training_flow_is_compiler_movable_to_sampler_bubble(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=8))
        flow = consumer.flows()[0]

        self.assertEqual(str(flow.timing.mobility), "deferrable")
        self.assertEqual(str(flow.timing.deadline), "next_prepare")
        self.assertEqual(flow.timing.max_defer_steps, 1)
        self.assertEqual(flow.lease_lifetime, "step")

    def test_first_per_prompt_capacity_drives_static_training_graph_rows(self) -> None:
        consumer = ESampConsumer(
            ESampConsumerConfig(
                graph_scratch_rows=128,
                model_bank_slots=8,
                training_row_selection="first_per_prompt",
            )
        )

        flow = consumer.flows()[0]

        self.assertEqual(flow.max_bundle_rows, 8)
        self.assertEqual(consumer._static_training_rows(), 8)

    def tearDown(self) -> None:
        esamp_runtime.clear_dispatch_consumers()

    @staticmethod
    def _ctx(*, runner_max_num_reqs: int = 8, model=None) -> RuntimeContext:
        return RuntimeContext(
            runner=SimpleNamespace(max_num_reqs=runner_max_num_reqs),
            model=model,
            device=torch.device("cpu"),
            main_stream=None,
            is_compiling=False,
            uses_cudagraph=False,
            event_name="flow:decode",
        )

    def test_flows_cover_source_target_and_request_metadata(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        flows = consumer.flows()

        self.assertEqual(len(flows), 1)
        flow = flows[0]
        self.assertEqual(flow.window, "out_of_band")
        self.assertEqual(
            [read.kind for read in flow.reads],
            [PortKind.RESIDUAL_STREAM, PortKind.RESIDUAL_STREAM, PortKind.REQUEST_META],
        )
        self.assertEqual([read.role for read in flow.reads[:2]], ["source", "target"])
        self.assertEqual(flow.bundle_key, ("engine_step_id", "phase"))
        self.assertEqual(flow.delivery, "device_lease")
        self.assertEqual(flow.ownership, "runtime_lease")

    def test_flows_are_empty_when_esamp_is_disabled(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(enable_esamp_training=False), engine=mock.Mock())

        self.assertEqual(tuple(consumer.flows()), ())

    def test_cold_path_specialization_captures_identity_residual_handoff(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(), engine=engine)
        source_example = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_example = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

        captured, attachment = consumer.specialize_workgraph(
            source_example,
            target_example,
        )

        expected_reads = (
            ResidualStream.read(
                layer=0,
                site="block_output",
                phase="decode",
                role="source",
            ),
            ResidualStream.read(
                layer=-1,
                site="block_output",
                phase="decode",
                role="target",
            ),
        )
        self.assertEqual(tuple(consumer.flows()[0].reads[:2]), expected_reads)
        expected_bindings = tuple(
            PortBindingDesc(
                port=port_address_desc(read),
                direction=PortBindingDirection.INPUT,
                value_path=(index,),
            )
            for index, read in enumerate(consumer.flows()[0].reads[:2])
        )
        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertIsInstance(captured, CapturedFlow)
        self.assertIsInstance(attachment, CaptureAttachment)
        self.assertEqual(captured.inputs, expected_bindings)
        self.assertEqual(attachment.bindings, expected_bindings)
        self.assertEqual(
            attachment.node_key,
            NodeKey.consumer_flow(consumer.consumer_id, 0),
        )
        self.assertEqual(attachment.artifact_key, captured.artifact_key)
        self.assertIsNotNone(captured.leaf_artifact)
        assert captured.leaf_artifact is not None
        self.assertEqual(captured.leaf_artifact.operators, ())
        self.assertEqual(
            captured.leaf_artifact.input_values,
            captured.leaf_artifact.output_values,
        )
        engine.assert_not_called()

    def test_consume_bundle_hot_path_does_not_capture_residual_handoff(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        target_hidden = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "device_lease": DeviceTensorLease(
                    entries={"source": source_hidden, "target": target_hidden},
                    active_rows=2,
                ),
                "request_meta": RowBatchMeta(
                    request_ids=("reqA", "reqB"),
                    prompt_idxs=(7, 9),
                    sample_idxs=(0, 1),
                    phase="decode",
                    engine_step_id=1,
                ),
            },
        )

        with mock.patch(
            "tllm.workgraph.capture.capture_flow",
            side_effect=AssertionError("consume_bundle must not enter cold-path capture"),
        ) as capture_flow, mock.patch.object(
            consumer,
            "_ensure_runtime_resources",
            return_value=None,
        ):
            consumer.consume_bundle(bundle, self._ctx())

        capture_flow.assert_not_called()
        engine.launch_step.assert_called_once_with(source_hidden, target_hidden)

    def test_disabled_esamp_does_not_publish_workgraph_specialization(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=False),
            engine=engine,
        )
        source_example = torch.ones((2, 2))
        target_example = torch.zeros((2, 2))

        with self.assertRaisesRegex(RuntimeError, "disabled ESamp"):
            consumer.specialize_workgraph(source_example, target_example)

        self.assertEqual(consumer.workgraph_captures(), ())
        self.assertEqual(consumer.workgraph_attachments(), ())
        engine.assert_not_called()

    def test_specialize_workgraph_rejects_rank_layout_and_device_mismatch(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(), engine=engine)
        cases = (
            (
                "rank",
                torch.ones((2, 2)),
                torch.ones((2,)),
                "matching rank",
            ),
            (
                "layout",
                torch.ones((2, 2)),
                torch.sparse_coo_tensor(
                    indices=torch.tensor([[0, 1], [0, 1]]),
                    values=torch.ones((2,)),
                    size=(2, 2),
                ),
                "matching tensor layouts",
            ),
            (
                "device",
                torch.ones((2, 2)),
                torch.empty((2, 2), device="meta"),
                "same device",
            ),
        )

        for label, source_example, target_example, message in cases:
            with self.subTest(kind=label):
                with self.assertRaisesRegex(ValueError, message):
                    consumer.specialize_workgraph(source_example, target_example)

        engine.assert_not_called()

    def test_equivalent_workgraph_specializations_keep_artifact_identity_stable(self) -> None:
        consumer = ESampConsumer(ESampConsumerConfig(), engine=mock.Mock())
        first_captured, first_attachment = consumer.specialize_workgraph(
            torch.ones((2, 2)),
            torch.zeros((2, 2)),
        )
        second_captured, second_attachment = consumer.specialize_workgraph(
            torch.full((2, 2), 3.0),
            torch.full((2, 2), 4.0),
        )

        self.assertEqual(first_captured.artifact_key, second_captured.artifact_key)
        self.assertEqual(first_captured.source_span, second_captured.source_span)
        self.assertEqual(first_attachment.artifact_key, second_attachment.artifact_key)
        self.assertEqual(
            consumer.workgraph_captures(),
            (second_captured,),
        )
        self.assertEqual(
            consumer.workgraph_attachments(),
            (second_attachment,),
        )

    def test_disabled_workgraph_metadata_accessors_remain_empty(self) -> None:
        consumer = ESampConsumer(
            ESampConsumerConfig(enable_esamp_training=False),
            engine=mock.Mock(),
        )

        self.assertEqual(consumer.workgraph_captures(), ())
        self.assertEqual(consumer.workgraph_attachments(), ())

    def test_consume_bundle_passes_device_lease_to_engine_owned_step_stage(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        source_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
        target_hidden = torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float32)
        ctx = self._ctx()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "device_lease": DeviceTensorLease(
                    entries={"source": source_hidden, "target": target_hidden},
                    active_rows=2,
                ),
                "request_meta": RowBatchMeta(
                    request_ids=("reqA", "reqB"),
                    prompt_idxs=(7, 9),
                    sample_idxs=(0, 1),
                    phase="decode",
                    engine_step_id=1,
                ),
            },
        )

        with mock.patch.object(consumer, "_ensure_runtime_resources", return_value=None) as p_prepare:
            consumer.consume_bundle(bundle, ctx)

        p_prepare.assert_called_once()
        self.assertIs(p_prepare.call_args.kwargs["rows_hidden"], source_hidden)
        engine.launch_step.assert_called_once()
        self.assertIs(engine.launch_step.call_args.args[0], source_hidden)
        self.assertIs(engine.launch_step.call_args.args[1], target_hidden)
        engine.launch_forward.assert_not_called()
        engine.launch_target.assert_not_called()

    def test_step_lease_retirement_fences_engine_inputs(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(), engine=engine)

        consumer.retire_device_lease(self._ctx())

        engine.retire_step_inputs.assert_called_once_with()

    def test_sampling_parameter_read_fence_precedes_delayed_training(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(), engine=engine)
        read_done = mock.Mock()

        consumer._last_active_rows = 1
        consumer._last_prompt_idxs = (7,)
        consumer.on_step_end(
            replace(self._ctx(), predecessor_events=(read_done,))
        )

        engine.launch_delayed_backward.assert_called_once_with(
            1,
            prompt_idxs=(7,),
            wait_events=(read_done,),
        )

    def test_fusion_group_handle_uses_dedicated_group_entry_without_public_fallback(self) -> None:
        engine = mock.Mock()
        engine.fusion_replay_count.side_effect = (4, 5)
        consumer = ESampConsumer(
            ESampConsumerConfig(graph_scratch_rows=4),
            engine=engine,
        )
        handle = consumer.fusion_group_handle(("esamp#0",))
        assert handle is not None
        bundle = PortBundle(
            key=BundleKey(
                engine_step_id=1,
                phase="decode",
                request_id="reqA",
                sample_idx=0,
            ),
            entries={
                "device_lease": DeviceTensorLease(
                    entries={
                        "source": torch.ones((1, 2)),
                        "target": torch.ones((1, 2)),
                    },
                    active_rows=1,
                ),
                "request_meta": RowBatchMeta(
                    request_ids=("reqA",),
                    prompt_idxs=(0,),
                    sample_idxs=(0,),
                    phase="decode",
                    engine_step_id=1,
                ),
            },
        )
        ctx = self._ctx()

        with mock.patch.object(consumer, "_ensure_runtime_resources"), mock.patch.object(
            consumer,
            "consume_bundle",
            side_effect=AssertionError("fused execution called public consume_bundle"),
        ) as public_consume, mock.patch.object(
            consumer,
            "on_step_end",
            side_effect=AssertionError("fused execution called public on_step_end"),
        ) as public_step_end:
            handle.execute({"esamp#0": bundle}, ctx)

        public_consume.assert_not_called()
        public_step_end.assert_not_called()
        engine.launch_step.assert_called_once()
        engine.launch_delayed_backward.assert_called_once()
        self.assertEqual(handle.replay_count(), 4)
        self.assertEqual(handle.replay_count(), 5)

    def test_consume_bundle_rejects_mismatched_source_target_row_counts(self) -> None:
        engine = mock.Mock()
        consumer = ESampConsumer(ESampConsumerConfig(graph_scratch_rows=4), engine=engine)
        ctx = self._ctx()
        bundle = PortBundle(
            key=BundleKey(engine_step_id=1, phase="decode", request_id="reqA", sample_idx=0),
            entries={
                "source": torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32),
                "target": torch.tensor([[5.0, 6.0]], dtype=torch.float32),
                "request_meta": [{"prompt_idx": 7, "sample_idx": 0}],
            },
        )

        with self.assertRaisesRegex(RuntimeError, "row count"):
            consumer.consume_bundle(bundle, ctx)

    def test_direct_runtime_installs_esamp_consumer(self) -> None:
        consumer = ESampConsumer(
            ESampConsumerConfig(
                graph_scratch_rows=64,
                source_layer_path="model.model.layers[0]",
                target_layer_path="model.model.layers[-1]",
                enable_esamp_training=True,
                distiller_hidden_dim=128,
                distiller_lr=1e-3,
            )
        )
        esamp_runtime.set_runtime_consumer(consumer)
        esamp_runtime.configure_runtime(
            graph_scratch_rows=64,
            tap_layer_paths=("model.model.layers[0]", "model.model.layers[-1]"),
            source_layer_path="model.model.layers[0]",
            target_layer_path="model.model.layers[-1]",
            adaptation_pipeline_slots=4,
            adaptation_stream_mode="dual",
            adaptation_stream_priority=0,
            compact_capture_lane=False,
        )
        self.assertIsInstance(esamp_runtime.RUNTIME.consumer, ESampConsumer)
        self.assertIsNotNone(esamp_runtime.RUNTIME.dispatch_plan)
        self.assertEqual(
            esamp_runtime.RUNTIME.residual_raw_paths,
            {
                ResidualLocator(layer=0, site="block_output", phase="decode"): "model.model.layers[0]",
                ResidualLocator(layer=-1, site="block_output", phase="decode"): "model.model.layers[-1]",
            },
        )
        targets = esamp_runtime.RUNTIME.dispatch_plan.flow_targets()
        self.assertTrue(targets)
        self.assertIs(targets[0].consumer, esamp_runtime.RUNTIME.consumer)


if __name__ == "__main__":
    unittest.main()
