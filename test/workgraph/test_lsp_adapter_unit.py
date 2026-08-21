from __future__ import annotations

import json
import unittest
from dataclasses import FrozenInstanceError
from unittest import mock

import torch

from tllm.producers.registry import ExecutionPlan
from tllm.runtime.dispatch_plan import DispatchPlan
from tllm.workgraph.shadow import compile_shadow_snapshot
from tllm.workgraph.frontend import (
    CaptureStatus,
    DynamicDim,
    Effect,
    FlowCaptureCode,
    ShapeGuardKind,
    flow,
)
from tllm.workgraph.hierarchy import PartitionBoundary, RegionKind
from tllm.workgraph.lsp import (
    LspDiagnostic,
    LspRegionDescriptor,
    LspSnapshot,
    build_lsp_snapshot,
)


class WorkGraphLspAdapterTest(unittest.TestCase):
    def test_adapter_is_exported_from_workgraph_control_plane(self) -> None:
        from tllm.workgraph import lsp as lsp_module

        self.assertIs(lsp_module.build_lsp_snapshot, build_lsp_snapshot)

    def test_successful_capture_exposes_typed_hover_and_regions(self) -> None:
        @flow(dynamic_dims=(DynamicDim(value_path=(0,), dimension=0, upper_bound=4),))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones((2, 3)))
        report = build_lsp_snapshot(captured)

        self.assertIsInstance(report, LspSnapshot)
        self.assertEqual(report.diagnostics, ())
        self.assertEqual(report.artifact_keys, (captured.artifact_key,))
        self.assertEqual(
            tuple(region.kind for region in report.regions),
            (RegionKind.ROOT, RegionKind.TASK, RegionKind.LEAF),
        )
        self.assertTrue(all(isinstance(region, LspRegionDescriptor) for region in report.regions))
        root, task, leaf = report.regions
        self.assertIsNone(root.parent)
        self.assertEqual(root.input_values, captured.leaf_artifact.input_values)
        self.assertEqual(root.output_values, captured.leaf_artifact.output_values)
        self.assertEqual(task.parent, root.key)
        self.assertEqual(leaf.parent, task.key)
        self.assertEqual(task.operator_ordinals, leaf.operator_ordinals)

        self.assertEqual(report.shape_guards, (captured.shape_guard,))
        guard = report.shape_guards[0].expressions[0]
        self.assertEqual(guard.kind, ShapeGuardKind.RANGE)
        value_hover = next(hover for hover in report.hovers if hover.value is not None)
        assert value_hover.value is not None
        self.assertEqual(value_hover.value.shape, captured.values[0].shape)
        self.assertEqual(value_hover.value.dtype.name, "float32")
        self.assertEqual(value_hover.value.device.kind.value, "cpu")
        self.assertEqual(value_hover.value.layout.value, "strided")
        self.assertEqual(value_hover.value.stride, ("3", "1"))
        self.assertEqual(value_hover.value.effect, Effect.PURE)
        self.assertEqual(value_hover.source_span, captured.source_span)

        payload = json.loads(report.to_json())
        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["regions"][0]["kind"], "root")
        self.assertEqual(payload["regions"][2]["kind"], "leaf")
        self.assertEqual(payload["hovers"][0]["shape_guard"], payload["shape_guards"][0])
        self.assertNotIn("commands", payload)
        self.assertNotIn("code_actions", payload)
        self.assertNotIn("recompile", payload)

        with self.assertRaises(FrozenInstanceError):
            report.regions = ()  # type: ignore[misc]

    def test_opaque_capture_exposes_diagnostic_and_opaque_region(self) -> None:
        @flow
        def host_io(value: torch.Tensor) -> torch.Tensor:
            print(value)
            return value + 1

        captured = host_io.capture(torch.ones(2))
        self.assertEqual(captured.status, CaptureStatus.OPAQUE_BARRIER)
        report = build_lsp_snapshot(captured)

        self.assertEqual(len(report.diagnostics), 1)
        diagnostic = report.diagnostics[0]
        self.assertIsInstance(diagnostic, LspDiagnostic)
        self.assertEqual(diagnostic.code, FlowCaptureCode.HOST_IO)
        self.assertEqual(diagnostic.message, captured.diagnostics[0].message)
        self.assertEqual(diagnostic.source_span, captured.diagnostics[0].source_span)
        self.assertEqual(
            tuple(region.kind for region in report.regions),
            (RegionKind.ROOT, RegionKind.TASK, RegionKind.OPAQUE),
        )
        self.assertEqual(report.hovers[0].effects, (Effect.HOST_IO, Effect.OPAQUE))
        self.assertEqual(report.hovers[0].value, None)
        self.assertEqual(report.regions[-1].operator_ordinals, ())

    def test_rng_partition_is_preserved_in_read_only_regions(self) -> None:
        @flow
        def sample(value: torch.Tensor) -> torch.Tensor:
            return value + torch.rand_like(value)

        captured = sample.capture(torch.ones(2))
        report = build_lsp_snapshot(captured)

        tasks = tuple(region for region in report.regions if region.kind == RegionKind.TASK)
        self.assertEqual(len(tasks), 2)
        self.assertEqual(tuple(task.boundary for task in tasks),
                         (PartitionBoundary.ROOT, PartitionBoundary.EFFECT))
        self.assertEqual(tasks[1].dependencies, (tasks[0].key,))
        covered_ordinals = tuple(
            ordinal
            for region in report.regions
            if region.kind in {RegionKind.LEAF, RegionKind.OPAQUE}
            for ordinal in region.operator_ordinals
        )
        self.assertEqual(
            covered_ordinals,
            tuple(operator.ordinal for operator in captured.leaf_artifact.operators),
        )
        self.assertEqual(
            tuple(task.effects for task in tasks),
            tuple(operator.effects for operator in captured.leaf_artifact.operators),
        )

    def test_phase1_plan_without_capture_is_adapted_without_runtime_access(self) -> None:
        plan = compile_shadow_snapshot(
            consumers=(),
            dispatch_plan=DispatchPlan.build([]),
            execution_plan=ExecutionPlan(producers=(), port_sources={}),
        )

        with mock.patch(
            "tllm.workgraph.capture.capture_flow",
            side_effect=AssertionError("LSP adapter must not capture"),
        ), mock.patch(
            "tllm.workgraph.shadow.compile_shadow_parity",
            side_effect=AssertionError("LSP adapter must not compile"),
        ):
            report = build_lsp_snapshot(plan)

        self.assertIsInstance(report, LspSnapshot)
        self.assertEqual(report.regions, ())
        self.assertEqual(report.hovers, ())
        self.assertEqual(report.artifact_keys, ())
        self.assertEqual(json.loads(report.to_json())["regions"], [])

    def test_invalid_snapshot_type_is_rejected(self) -> None:
        with self.assertRaises(TypeError):
            build_lsp_snapshot(object())


if __name__ == "__main__":
    unittest.main()
