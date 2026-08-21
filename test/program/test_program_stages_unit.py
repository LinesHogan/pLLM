#!/usr/bin/env python3
"""Stage-slicing tests (plan Phase 2).

Covers availability-based slicing into hook stages, pure-chain recompute,
RNG pinning with owned cross-stage values, bounded dynamic-dim buckets,
fail-closed PG130 errors, staged-eager vs Python reference equivalence, and
the physical planner's Program-only multi-task partition (WG421 unchanged
for legacy direct captures).
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from tllm.program.stages import FlowStagePlan, StagedFlowExecutor, slice_flow_stages
from tllm.workgraph.frontend import (
    DynamicDim,
    ShapeGuardKind,
    extern,
    flow,
    hidden,
    local,
    result,
    sink,
)
from tllm.workgraph.hierarchy import (
    CaptureAttachment,
    CaptureDataCentricIR,
    CaptureHierarchy,
    build_capture_root,
)
from tllm.workgraph.ir import (
    CompiledWorkGraph,
    DataCentricIR,
    NodeKey,
    RuntimePlanBinding,
    SemanticGraph,
)
from tllm.workgraph.lowering import WorkGraphError
from tllm.workgraph.physical import compile_physical_capture_region
from tllm.workgraph.shadow import ShadowParityPlan


class _Distiller(torch.nn.Module):
    """The ESamp shape: source-only prediction prefix + target-dependent loss."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)

    @flow(
        inputs={
            "source": hidden(layer=0, site="block_output", phase="decode"),
            "target": hidden(layer=-1, site="block_output", phase="decode"),
        },
        outputs={"prediction": sink("guidance"), "loss": local()},
        state="distiller",
    )
    def forward(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction = torch.relu(self.proj(source))
        loss = F.mse_loss(prediction.float(), target.detach().float())
        return prediction, loss


def _slice(
    instance: torch.nn.Module,
    method: str = "forward",
    example_args: tuple[torch.Tensor, ...] = (torch.randn(2, 4), torch.randn(2, 4)),
    dynamic_dims: tuple[DynamicDim, ...] = (),
) -> tuple[FlowStagePlan, StagedFlowExecutor]:
    function = getattr(instance, method)
    return slice_flow_stages(
        flow=method,
        flow_ordinal=0,
        decl=function.flow_program,
        function=function,
        example_args=example_args,
        dynamic_dims=dynamic_dims,
    )


class StageSlicingTest(unittest.TestCase):
    def test_source_only_and_mixed_ops_split_into_two_stages(self) -> None:
        plan, _executor = _slice(_Distiller())
        self.assertEqual(plan.hook_count, 2)
        stage0, stage1 = plan.stages
        self.assertEqual(stage0.flow_outputs, ("prediction",))
        self.assertEqual(stage1.flow_outputs, ("loss",))
        self.assertNotEqual(stage0.operator_ordinals, ())
        self.assertNotEqual(stage1.operator_ordinals, ())
        # No operator may sit in a stage before its inputs arrive.
        self.assertTrue(all(level == 0 for level in (stage0.hook_level,)))
        self.assertEqual(stage1.hook_level, 1)

    def test_pure_prefix_is_recomputed_not_threaded(self) -> None:
        plan, _executor = _slice(_Distiller())
        stage1 = plan.stages[1]
        # model(source) is needed by the training stage but is pure, so the
        # chain is recomputed there instead of becoming a cross-stage value.
        self.assertNotEqual(stage1.recomputed_ordinals, ())
        self.assertEqual(plan.cross_stage_values, ())

    def test_staged_eager_matches_python_reference(self) -> None:
        instance = _Distiller()
        plan, executor = _slice(instance)
        source = torch.randn(3, 4)
        target = torch.randn(3, 4)
        ref_prediction, ref_loss = instance.forward(source, target)
        out = executor.run(source, target)
        self.assertTrue(torch.allclose(out["prediction"], ref_prediction))
        self.assertTrue(torch.allclose(out["loss"], ref_loss))
        self.assertEqual(tuple(out), plan.output_names)

    def test_three_hook_levels(self) -> None:
        class ThreeHook(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

            @flow(
                inputs={
                    "early": hidden(layer=0, site="block_output", phase="decode"),
                    "middle": hidden(layer=8, site="block_output", phase="decode"),
                    "late": hidden(layer=-1, site="block_output", phase="decode"),
                },
                outputs={"total": local()},
                state="distiller",
            )
            def forward(
                self, early: torch.Tensor, middle: torch.Tensor, late: torch.Tensor
            ) -> torch.Tensor:
                a = self.proj(early)  # level 0 only
                b = a * middle.sum()  # level 1
                c = b + late  # level 2
                return c.sum()

        plan, executor = _slice(
            ThreeHook(), example_args=(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
        )
        self.assertEqual(plan.hook_count, 3)
        self.assertEqual(tuple(stage.hook_level for stage in plan.stages), (0, 1, 2))
        self.assertTrue(all(stage.operator_ordinals for stage in plan.stages))
        instance = ThreeHook()
        _, executor = _slice(instance, example_args=(torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4)))
        args = (torch.randn(2, 4), torch.randn(2, 4), torch.randn(2, 4))
        out = executor.run(*args)
        self.assertTrue(torch.allclose(out["total"], instance.forward(*args)))


class RngPinningTest(unittest.TestCase):
    def test_rng_op_is_pinned_and_crosses_as_owned_value(self) -> None:
        class Noisy(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.proj = torch.nn.Linear(4, 4)

            @flow(
                inputs={
                    "source": hidden(layer=0, site="block_output", phase="decode"),
                    "target": hidden(layer=-1, site="block_output", phase="decode"),
                },
                outputs={"noisy": sink("guidance"), "loss": local()},
                state="distiller",
            )
            def forward(
                self, source: torch.Tensor, target: torch.Tensor
            ) -> tuple[torch.Tensor, torch.Tensor]:
                noise = torch.randn_like(source)
                noisy = torch.relu(self.proj(source)) + noise
                loss = F.mse_loss(noisy.float(), target.detach().float()) + noise.sum()
                return noisy, loss

        instance = Noisy()
        plan, executor = _slice(instance)
        # The RNG value cannot be recomputed in stage 1: it crosses as an
        # owned value and no RNG operator is duplicated.
        self.assertNotEqual(plan.cross_stage_values, ())
        for stage in plan.stages[1:]:
            self.assertEqual(stage.recomputed_ordinals, ())
        source = torch.randn(2, 4)
        target = torch.randn(2, 4)
        torch.manual_seed(0)
        ref_noisy, ref_loss = instance.forward(source, target)
        torch.manual_seed(0)
        out = executor.run(source, target)
        self.assertTrue(torch.allclose(out["noisy"], ref_noisy))
        self.assertTrue(torch.allclose(out["loss"], ref_loss))


class DynamicDimTest(unittest.TestCase):
    def test_bounded_dynamic_dim_produces_range_guard(self) -> None:
        instance = _Distiller()
        dims = (
            DynamicDim(value_path=(0,), dimension=0, lower_bound=1, upper_bound=8),
            DynamicDim(value_path=(1,), dimension=0, lower_bound=1, upper_bound=8),
        )
        plan, executor = _slice(
            instance,
            example_args=(torch.randn(2, 4), torch.randn(2, 4)),
            dynamic_dims=dims,
        )
        kinds = {expression.kind for expression in plan.shape_guard.expressions}
        self.assertIn(ShapeGuardKind.RANGE, kinds)
        # The same specialization serves any in-bucket batch size.
        source = torch.randn(5, 4)
        target = torch.randn(5, 4)
        ref_prediction, ref_loss = instance.forward(source, target)
        out = executor.run(source, target)
        self.assertTrue(torch.allclose(out["prediction"], ref_prediction))
        self.assertTrue(torch.allclose(out["loss"], ref_loss))


class FailClosedTest(unittest.TestCase):
    def test_unexportable_flow_fails_pg130(self) -> None:
        class Cheater(torch.nn.Module):
            @flow(
                inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
                outputs={"loss": local()},
                state="distiller",
            )
            def forward(self, source: torch.Tensor) -> torch.Tensor:
                return source.sum() + float(source.sum().item())

        with self.assertRaises(WorkGraphError) as ctx:
            _slice(Cheater(), example_args=(torch.randn(2, 4),))
        self.assertEqual(ctx.exception.code, "PG130")

    def test_output_count_mismatch_fails_pg130(self) -> None:
        class Mismatch(torch.nn.Module):
            @flow(
                inputs={"source": hidden(layer=0, site="block_output", phase="decode")},
                outputs={"only": local()},
                state="distiller",
            )
            def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return source, source.sum()

        with self.assertRaises(WorkGraphError) as ctx:
            _slice(Mismatch(), example_args=(torch.randn(2, 4),))
        self.assertEqual(ctx.exception.code, "PG130")

    def test_non_module_function_fails_pg130(self) -> None:
        instance = _Distiller()
        decl = instance.forward.flow_program

        def plain(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            return source

        with self.assertRaises(WorkGraphError) as ctx:
            slice_flow_stages(
                flow="plain",
                flow_ordinal=0,
                decl=decl,
                function=plain,
                example_args=(torch.randn(2, 4), torch.randn(2, 4)),
            )
        self.assertEqual(ctx.exception.code, "PG130")


class _TwoModule(torch.nn.Module):
    """Two submodules in sequence: module boundary yields two capture tasks."""

    def __init__(self) -> None:
        super().__init__()
        self.first = torch.nn.Linear(4, 4)
        self.second = torch.nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.second(torch.relu(self.first(value)))


def _two_module_hierarchy() -> tuple[ShadowParityPlan, object]:
    from tllm.workgraph.capture import capture_module_flow

    instance = _TwoModule()
    captured, exported = capture_module_flow(
        instance, _TwoModule.forward, (torch.randn(2, 4),), {}
    )
    assert exported is not None
    root = build_capture_root(captured)
    hierarchy = CaptureHierarchy(
        roots=(root,),
        attachments=(
            CaptureAttachment(
                node_key=NodeKey.consumer_flow("program-test", 0),
                artifact_key=captured.artifact_key,
            ),
        ),
        data_centric=CaptureDataCentricIR(),
    )
    plan = ShadowParityPlan(
        compiled=CompiledWorkGraph(
            semantic=SemanticGraph(nodes=()),
            data_centric=DataCentricIR(nodes=()),
            epoch=1,
        ),
        binding=RuntimePlanBinding(
            dispatch_flow_ids=(),
            dispatch_hooks=(),
            execution_producer_ids=(),
            producer_hooks=(),
        ),
        hierarchy=hierarchy,
    )
    return plan, captured.artifact_key


class PhysicalPartitionTest(unittest.TestCase):
    def test_legacy_path_still_rejects_multi_task(self) -> None:
        plan, artifact_key = _two_module_hierarchy()
        with self.assertRaises(WorkGraphError) as ctx:
            compile_physical_capture_region(plan, artifact_key)
        self.assertEqual(ctx.exception.code, "WG421_UNSUPPORTED_CAPTURE_PARTITION")

    def test_program_partition_lowers_one_node_per_task(self) -> None:
        plan, artifact_key = _two_module_hierarchy()
        hierarchy = plan.hierarchy
        assert hierarchy is not None
        task_count = len(hierarchy.roots[0].children)
        self.assertGreaterEqual(task_count, 2)
        region = compile_physical_capture_region(
            plan, artifact_key, allow_task_partition=True
        )
        self.assertEqual(len(region.plan.nodes), task_count)
        node_keys = [node.node_key for node in region.plan.nodes]
        self.assertEqual(len(set(node_keys)), task_count)
        # Stage dependencies become fences in topological order.
        later = region.plan.nodes[1]
        self.assertNotEqual(later.fences, ())
        self.assertEqual(later.fences[0].producer, region.plan.nodes[0].node_key)


if __name__ == "__main__":
    unittest.main()
