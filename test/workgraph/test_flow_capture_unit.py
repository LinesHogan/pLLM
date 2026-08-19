from __future__ import annotations

import unittest
from unittest import mock

import torch

from tllm.workgraph import (
    BeforeNextUse,
    CaptureStatus,
    CapturedFlow,
    DynamicDim,
    Effect,
    FlowCaptureCode,
    SourceSpan,
    TensorDeviceDesc,
    TensorDTypeDesc,
    flow,
)
from tllm.workgraph.ir import GraphSafetyKind, NodeKind, TimingMobilityKind


def _assert_opaque_barrier(
    testcase: unittest.TestCase,
    captured: CapturedFlow,
    code: FlowCaptureCode,
) -> None:
    testcase.assertIsInstance(captured, CapturedFlow)
    testcase.assertEqual(captured.status, CaptureStatus.OPAQUE_BARRIER)
    testcase.assertEqual(captured.node_kind, NodeKind.BARRIER)
    testcase.assertEqual(captured.graph_safety, GraphSafetyKind.EAGER_ONLY)
    testcase.assertIs(captured.mobility, TimingMobilityKind.PINNED)
    testcase.assertIn(Effect.OPAQUE, captured.effects)
    testcase.assertEqual(captured.values, ())
    diagnostics = [diagnostic for diagnostic in captured.diagnostics if diagnostic.code == code]
    testcase.assertEqual(len(diagnostics), 1)
    testcase.assertIsInstance(diagnostics[0].source_span, SourceSpan)


class WorkGraphFlowCaptureTest(unittest.TestCase):
    def test_capture_uses_strict_export_as_the_only_export_path(self) -> None:
        @flow
        def affine(value: torch.Tensor) -> torch.Tensor:
            return value * 2 + 1

        export = torch.export.export
        strict_values: list[object] = []

        def record_export(*args: object, **kwargs: object) -> object:
            strict_values.append(kwargs.get("strict"))
            return export(*args, **kwargs)

        with mock.patch.object(torch.export, "export", side_effect=record_export):
            captured = affine.capture(torch.ones(2))

        self.assertEqual(strict_values, [True])
        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertEqual(captured.diagnostics, ())

    def test_strict_export_failure_has_no_non_strict_or_partial_fallback(self) -> None:
        @flow
        def affine(value: torch.Tensor) -> torch.Tensor:
            return value * 2 + 1

        with mock.patch.object(
            torch.export,
            "export",
            side_effect=RuntimeError("strict capture failed"),
        ) as export:
            captured = affine.capture(torch.ones(2))

        export.assert_called_once()
        self.assertIs(export.call_args.kwargs["strict"], True)
        _assert_opaque_barrier(self, captured, FlowCaptureCode.STRICT_EXPORT_FAILURE)
        self.assertIsNone(captured.leaf_artifact)

    def test_simple_tensor_graph_retains_inferred_value_facts(self) -> None:
        @flow
        def add(value: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            return value + residual

        captured = add.capture(
            torch.ones((2, 3), dtype=torch.float32),
            torch.zeros((2, 3), dtype=torch.float32),
        )

        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertGreaterEqual(len(captured.values), 3)
        for value in captured.values:
            with self.subTest(value=getattr(value, "value_id", "unknown")):
                self.assertIsInstance(value.shape, tuple)
                self.assertTrue(value.shape)
                self.assertTrue(value.dtype)
                self.assertIsInstance(value.dtype, TensorDTypeDesc)
                self.assertTrue(value.device)
                self.assertIsInstance(value.device, TensorDeviceDesc)
                self.assertTrue(value.layout)
                self.assertIsInstance(value.stride, tuple)
                self.assertTrue(value.stride)
                self.assertIsInstance(value.version, int)
                self.assertGreaterEqual(value.version, 1)
                self.assertTrue(value.use_def)
                self.assertTrue(value.alias_set)
                self.assertIsInstance(value.effect, Effect)
        self.assertIsNotNone(captured.leaf_artifact)
        self.assertTrue(captured.leaf_artifact.operators)
        self.assertTrue(all(operator.namespace == "aten" for operator in captured.leaf_artifact.operators))

    def test_view_alias_and_leaf_aten_nodes_are_retained_without_runtime_objects(self) -> None:
        @flow
        def flatten(value: torch.Tensor) -> torch.Tensor:
            return value.view(-1)

        captured = flatten.capture(torch.ones((2, 3)))

        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        self.assertEqual(len(captured.values), 2)
        self.assertEqual(captured.values[0].alias_set, captured.values[1].alias_set)
        self.assertEqual(
            tuple((operator.namespace, operator.name) for operator in captured.leaf_artifact.operators),
            (("aten", "view"),),
        )
        self.assertFalse(any(isinstance(value, torch.Tensor) for value in captured.values))

    def test_bounded_dynamic_shape_builds_range_guard(self) -> None:
        @flow(dynamic_dims=(DynamicDim(value_path=(0,), dimension=0, upper_bound=4),))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        captured = add_one.capture(torch.ones((2, 3)))

        self.assertEqual(captured.status, CaptureStatus.CAPTURED)
        batch_guard = next(
            expression
            for expression in captured.shape_guard.expressions
            if expression.value_path == (0,) and expression.dimension == 0
        )
        self.assertEqual(batch_guard.kind.value, "range")
        self.assertEqual((batch_guard.lower_bound, batch_guard.upper_bound), (1, 4))

    def test_unbounded_dynamic_shape_is_rejected_before_export(self) -> None:
        @flow(dynamic_dims=(DynamicDim(value_path=(0,), dimension=0),))
        def add_one(value: torch.Tensor) -> torch.Tensor:
            return value + 1

        with mock.patch.object(torch.export, "export") as export:
            captured = add_one.capture(torch.ones((2, 3)))

        export.assert_not_called()
        _assert_opaque_barrier(self, captured, FlowCaptureCode.UNBOUNDED_DYNAMIC_SHAPE)

    def test_item_control_flow_is_a_typed_opaque_barrier(self) -> None:
        @flow
        def data_dependent_branch(value: torch.Tensor) -> torch.Tensor:
            if value.item() > 0:
                return value + 1
            return value - 1

        _assert_opaque_barrier(
            self,
            data_dependent_branch.capture(torch.tensor(1)),
            FlowCaptureCode.DATA_DEPENDENT_CONTROL_FLOW,
        )

    def test_host_io_is_a_typed_opaque_barrier(self) -> None:
        @flow
        def host_io(value: torch.Tensor) -> torch.Tensor:
            print(value)
            return value + 1

        with mock.patch.object(torch.export, "export") as export:
            captured = host_io.capture(torch.ones(1))
        export.assert_not_called()
        _assert_opaque_barrier(self, captured, FlowCaptureCode.HOST_IO)
        self.assertIn(Effect.HOST_IO, captured.effects)

    def test_host_io_alias_is_rejected_before_export(self) -> None:
        emit = print

        @flow
        def host_io_alias(value: torch.Tensor) -> torch.Tensor:
            emit(value)
            return value + 1

        with mock.patch.object(torch.export, "export") as export:
            captured = host_io_alias.capture(torch.ones(1))
        export.assert_not_called()
        _assert_opaque_barrier(self, captured, FlowCaptureCode.HOST_IO)

    def test_unregistered_closure_state_is_a_typed_opaque_barrier(self) -> None:
        class MutableState:
            def __init__(self) -> None:
                self.offset = 1

        state = MutableState()

        @flow
        def closure_state(value: torch.Tensor) -> torch.Tensor:
            return value + state.offset

        _assert_opaque_barrier(
            self,
            closure_state.capture(torch.ones(1)),
            FlowCaptureCode.UNREGISTERED_CLOSURE_STATE,
        )
        self.assertIn(Effect.READ_STATE, closure_state.capture(torch.ones(1)).effects)

    def test_unprovable_mutation_is_a_typed_opaque_barrier(self) -> None:
        @flow
        def resize_input(value: torch.Tensor) -> torch.Tensor:
            return value.resize_(value.numel())

        captured = resize_input.capture(torch.ones((2, 2)))
        _assert_opaque_barrier(
            self,
            captured,
            FlowCaptureCode.UNSUPPORTED_MUTATION,
        )
        self.assertIn(Effect.WRITE_STATE, captured.effects)

    def test_operator_schema_mutation_is_rejected_even_when_ast_looks_like_a_call(self) -> None:
        @flow
        def schema_mutation(value: torch.Tensor) -> torch.Tensor:
            return torch.ops.aten.add_.Tensor(value, 1)

        _assert_opaque_barrier(
            self,
            schema_mutation.capture(torch.ones(2)),
            FlowCaptureCode.UNSUPPORTED_MUTATION,
        )

    def test_capture_freshness_uses_the_typed_singleton(self) -> None:
        @flow(freshness=BeforeNextUse)
        def identity(value: torch.Tensor) -> torch.Tensor:
            return value

        self.assertIs(identity.flow_freshness, BeforeNextUse)


if __name__ == "__main__":
    unittest.main()
