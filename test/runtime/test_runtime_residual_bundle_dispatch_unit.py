#!/usr/bin/env python3
"""Unit tests for residual bundle dispatch helpers extracted from runtime hooks."""

from __future__ import annotations

import unittest

import torch

from tllm.consumers.base import BaseConsumer
from tllm.contracts.gpu_stage import DeviceTensorLease
from tllm.contracts.port_bundle import BundleKey, PortBundle
from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.contracts.runtime_context import RuntimeContext
from tllm.ports.base import ConsumerFlow, TimingContract
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualLocator, ResidualStream
from tllm.runtime.dispatch_plan import FlowDispatchTarget
from tllm.runtime.ports.residual_bindings import ResidualPathBinding
from tllm.runtime.ports import residual_bundle_dispatch
from tllm.runtime.ports import residual_write
from tllm.runtime.ports.residual_write import ResidualWriteLease, WriteLeaseKey
from tllm.runtime.scheduling.deferred import DeferredWorkQueue
from tllm.ports.base import WriteMode, WriteRows


class RuntimeResidualBundleDispatchUnitTest(unittest.TestCase):
    def _core(self):
        runtime = type("Runtime", (), {})()
        runtime.tap_decode_hidden = {
            "layers.0": torch.tensor([[1.0, 2.0], [3.0, 4.0], [99.0, 99.0]], dtype=torch.float32),
            "layers.1": torch.tensor([[5.0, 6.0], [7.0, 8.0], [88.0, 88.0]], dtype=torch.float32),
        }
        runtime.tap_decode_hidden_compact = {}
        runtime.decode_compact_count = 0
        runtime.decode_compact_row_ids = ()
        runtime.decode_count = 2
        runtime.decode_prompt_idxs = [10, 11]
        runtime.decode_sample_idxs = [0, 1]
        runtime.decode_request_ids = ["reqA", "reqB"]
        runtime.residual_bindings = {
            "layers.0": ResidualPathBinding(
                locator=ResidualLocator(layer=0, site="block_output", phase="decode"),
                resolved_path="layers.0",
                include_request_meta=True,
            ),
            "layers.1": ResidualPathBinding(
                locator=ResidualLocator(layer=-1, site="block_output", phase="decode"),
                resolved_path="layers.1",
                include_request_meta=False,
            ),
        }
        runtime.event_step_id = 9
        return type("Core", (), {"RUNTIME": runtime})()

    def test_build_step_scope_port_bundle_uses_residual_binding_table(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertTrue(torch.equal(bundle.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(bundle.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))

    def test_build_device_lease_step_scope_bundle_wraps_residual_entries_in_lease(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        self.assertIn("device_lease", bundle.entries)
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertTrue(torch.equal(lease.entries["source"], torch.tensor([[1.0, 2.0], [3.0, 4.0]])))
        self.assertTrue(torch.equal(lease.entries["target"], torch.tensor([[5.0, 6.0], [7.0, 8.0]])))
        self.assertEqual(lease.active_rows, 2)
        self.assertEqual(lease.ownership, "runtime_lease")
        self.assertEqual(lease.lifetime, "consume_call")
        self.assertIn("request_meta", bundle.entries)
        self.assertIsInstance(bundle.entries["request_meta"], RowBatchMeta)
        meta = bundle.entries["request_meta"]
        self.assertEqual(meta.prompt_idxs, (10, 11))
        self.assertEqual(meta.sample_idxs, (0, 1))
        self.assertEqual(meta.request_ids, ("reqA", "reqB"))
        self.assertEqual(meta.phase, "decode")
        self.assertEqual(meta.engine_step_id, 9)

    def test_step_lifetime_is_preserved_in_device_lease(self) -> None:
        core = self._core()
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            lease_lifetime="step",
            timing="pinned",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.lifetime, "step")

    def test_declared_write_output_stages_graph_resident_effect(self) -> None:
        core = self._core()
        key = WriteLeaseKey(layer_path="layers.0", writer_id="test.writer", order=0)
        write_lease = ResidualWriteLease(
            key=key,
            mode=WriteMode.ADD,
            rows=WriteRows.DECODE_ONLY,
            max_rows=3,
            hidden_size=2,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        core.RUNTIME.write_leases = {key: write_lease}
        flow = ConsumerFlow(
            reads=(RequestMeta.read(),),
            writes=(
                ResidualStream.write(
                    layer=0,
                    site="block_output",
                    phase="decode",
                    role="update",
                    mode="add",
                    fusion_candidate=True,
                ),
            ),
            window="same_step",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            max_bundle_rows=3,
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(
            core=core,
            flow=flow,
            consumer_id="test.writer",
        )

        self.assertIsNotNone(bundle)
        assert bundle is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        lease.output("update").stage(
            delta=torch.tensor([[3.0, 4.0]]),
            row_idx=torch.tensor([1]),
        )
        hidden = torch.zeros((3, 2))
        residual_write.apply_pending_writes(core.RUNTIME.write_leases, "layers.0", hidden)
        self.assertTrue(torch.equal(hidden[1], torch.tensor([3.0, 4.0])))

    def test_first_per_prompt_compaction_shapes_device_lease_and_metadata(self) -> None:
        core = self._core()
        core.RUNTIME.decode_count = 3
        core.RUNTIME.decode_prompt_idxs = [10, 10, 11]
        core.RUNTIME.decode_sample_idxs = [0, 1, 0]
        core.RUNTIME.decode_request_ids = ["reqA-0", "reqA-1", "reqB-0"]
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        bundle = residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)

        self.assertIsNotNone(bundle)
        assert bundle is not None
        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.active_rows, 2)
        self.assertTrue(torch.equal(lease.entries["source"], torch.tensor([[1.0, 2.0], [99.0, 99.0]])))
        self.assertTrue(torch.equal(lease.entries["target"], torch.tensor([[5.0, 6.0], [88.0, 88.0]])))
        self.assertEqual(lease.entries["source"].untyped_storage().data_ptr(), core.RUNTIME.tap_decode_hidden["layers.0"].untyped_storage().data_ptr())
        self.assertEqual(lease.entries["target"].untyped_storage().data_ptr(), core.RUNTIME.tap_decode_hidden["layers.1"].untyped_storage().data_ptr())
        meta = bundle.entries["request_meta"]
        self.assertIsInstance(meta, RowBatchMeta)
        self.assertEqual(meta.request_ids, ("reqA-0", "reqB-0"))
        self.assertEqual(meta.prompt_idxs, (10, 11))
        self.assertEqual(meta.sample_idxs, (0, 0))
        self.assertEqual(meta.row_compaction, "first_per_prompt")
        self.assertEqual(meta.row_ids, (0, 2))

    def test_deferred_device_lease_target_rejects_legacy_frame_bundle(self) -> None:
        flow = ConsumerFlow(
            reads=(RequestMeta.read(),),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            lease_lifetime="step",
            timing=TimingContract(
                mobility="deferrable",
                deadline="next_prepare",
                max_defer_steps=1,
            ),
        )
        consumer = type(
            "Consumer",
            (BaseConsumer,),
            {
                "consumer_id": property(lambda self: "test.consumer"),
                "consume_bundle": lambda self, bundle, ctx: None,
                "flows": lambda self: (flow,),
            },
        )()
        target = FlowDispatchTarget(
            consumer=consumer,
            flow=flow,
            flow_id="test.consumer#0",
            placement_hook="sample_tokens.pre",
        )
        legacy_bundle = PortBundle(
            key=BundleKey(
                engine_step_id=1,
                phase="decode",
                request_id="",
                sample_idx=0,
            ),
            entries={"request_meta": []},
            version=1,
        )

        with self.assertRaisesRegex(RuntimeError, "device_lease flow.*legacy frame bundle"):
            residual_bundle_dispatch._enqueue_flow_target(
                queue=DeferredWorkQueue(),
                runtime=object(),  # type: ignore[arg-type] - validation fails before runtime use
                target=target,
                bundle=legacy_bundle,
                ctx=RuntimeContext(
                    runner=None,
                    model=None,
                    device=torch.device("cpu"),
                    main_stream=None,
                    is_compiling=False,
                    uses_cudagraph=False,
                    event_name="test",
                ),
                step_id=1,
            )

    def test_first_per_prompt_rejects_active_rows_without_prompt_identity(self) -> None:
        core = self._core()
        core.RUNTIME.decode_prompt_idxs = [-1, -1]
        core.RUNTIME.decode_sample_idxs = [-1, -1]
        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            bundle_key=("engine_step_id", "phase"),
            delivery="device_lease",
            ownership="runtime_lease",
            row_compaction="first_per_prompt",
        )

        with self.assertRaisesRegex(RuntimeError, "first_per_prompt.*request identity"):
            residual_bundle_dispatch.build_step_scope_port_bundle(core=core, flow=flow)


if __name__ == "__main__":
    unittest.main()
