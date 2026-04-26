#!/usr/bin/env python3
"""Unit tests for phase-aware step-scope bundle helpers."""

from __future__ import annotations

import unittest

import torch

from tllm.contracts.request_meta_view import RowBatchMeta
from tllm.ports.base import ConsumerFlow
from tllm.ports.request_meta import RequestMeta
from tllm.ports.residual_stream import ResidualStream


class StepScopeBundleUnitTest(unittest.TestCase):
    def test_builds_prefill_bundle_from_explicit_entries(self) -> None:
        from tllm.runtime.ports.residual_bundle_dispatch import build_explicit_step_scope_port_bundle

        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="prefill", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="prefill", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="background",
            bundle_key=("engine_step_id", "phase"),
        )
        meta = RowBatchMeta(
            request_ids=("reqA", "reqB"),
            prompt_idxs=(0, 1),
            sample_idxs=(0, 0),
            phase="prefill",
            engine_step_id=42,
        )
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)

        bundle = build_explicit_step_scope_port_bundle(
            flow=flow,
            phase="prefill",
            engine_step_id=42,
            row_meta=meta,
            entries={"source": source, "target": target},
        )

        self.assertEqual(bundle.key.phase, "prefill")
        self.assertEqual(bundle.key.engine_step_id, 42)
        self.assertIs(bundle.entries["source"], source)
        self.assertIs(bundle.entries["target"], target)
        self.assertIs(bundle.entries["request_meta"], meta)

    def test_builds_device_lease_bundle_from_explicit_entries(self) -> None:
        from tllm.contracts.gpu_stage import DeviceTensorLease
        from tllm.runtime.ports.residual_bundle_dispatch import build_explicit_step_scope_port_bundle

        flow = ConsumerFlow(
            reads=(
                ResidualStream.read(layer=0, site="block_output", phase="decode", role="source"),
                ResidualStream.read(layer=-1, site="block_output", phase="decode", role="target"),
                RequestMeta.read(),
            ),
            writes=(),
            window="out_of_band",
            delivery="device_lease",
            ownership="runtime_lease",
            bundle_key=("engine_step_id", "phase"),
        )
        meta = RowBatchMeta(
            request_ids=("reqA", "reqB"),
            prompt_idxs=(0, 1),
            sample_idxs=(0, 0),
            phase="decode",
            engine_step_id=43,
        )
        source = torch.randn(4, 4)
        target = torch.randn(4, 4)

        bundle = build_explicit_step_scope_port_bundle(
            flow=flow,
            phase="decode",
            engine_step_id=43,
            row_meta=meta,
            entries={"source": source, "target": target},
        )

        lease = bundle.entries["device_lease"]
        self.assertIsInstance(lease, DeviceTensorLease)
        self.assertEqual(lease.active_rows, 2)
        self.assertIs(lease.entries["source"], source)
        self.assertIs(lease.entries["target"], target)
        self.assertIs(bundle.entries["request_meta"], meta)


if __name__ == "__main__":
    unittest.main()
