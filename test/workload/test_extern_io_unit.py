from __future__ import annotations

import unittest

import torch

from tllm.runtime.workload_execution import _plan_resources
from tllm.schedule import (
    CompileOptions,
    MainWorkloadTimeline,
    TimelineAnchor,
    WorkloadCompilerCoordinator,
)
from tllm.workload import (
    AdamWSpec,
    Between,
    State,
    StateScope,
    declare_workload,
    extern_io,
    provided,
    read,
    sink,
    task,
    workload,
)
from tllm.workload.ir import EffectKind
from tllm.workload.lowering import lower_schedule_graph


def _timeline() -> MainWorkloadTimeline:
    return MainWorkloadTimeline(
        backend_id="vllm",
        backend_version="0.27.1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor("model.execute", 0),
            TimelineAnchor("sampler.pre_draw", 1),
            TimelineAnchor("step.end", 2),
        ),
        regions=(),
        static_headroom_bytes=64,
    )


class ExternIODeclarationUnitTest(unittest.TestCase):
    def test_extern_io_signs_a_raw_channel_resource(self) -> None:
        effect = extern_io("trace:file")
        self.assertIs(effect.kind, EffectKind.EXTERN_IO)
        self.assertEqual(effect.resource_name, "trace:file")
        self.assertIsNone(effect.state)

    def test_extern_io_rejects_empty_and_state_bound_forms(self) -> None:
        with self.assertRaises((TypeError, ValueError)):
            extern_io("")
        with self.assertRaises((TypeError, ValueError)):
            extern_io(None)  # type: ignore[arg-type]
        state = State(
            scope=StateScope.REPLICA,
            parameters="weights",
            optimizer=AdamWSpec(lr=1e-3),
        )
        from tllm.workload.frontend import StateEffect

        with self.assertRaisesRegex(ValueError, "external resource"):
            StateEffect(EffectKind.EXTERN_IO, state)
        with self.assertRaisesRegex(ValueError, "raw resource"):
            StateEffect(EffectKind.READ, state, "trace:file")

    def test_same_channel_extern_io_serializes_in_declaration_order(self) -> None:
        @workload(id="extern-io.ordered")
        class Ordered:
            @task(
                inputs={"value": provided("value")},
                outputs={"first_out": sink("first")},
                effects=(extern_io("trace:file"),),
                timing=Between("model.execute", "step.end"),
            )
            def first(self, value: object) -> object:
                return value

            @task(
                inputs={"value": provided("value")},
                outputs={"second_out": sink("second")},
                effects=(extern_io("trace:file"),),
                timing=Between("model.execute", "step.end"),
            )
            def second(self, value: object) -> object:
                return value

        declaration = declare_workload(Ordered())
        effect_edges = [
            (edge.source, edge.target)
            for edge in declaration.ir.edges
            if edge.kind.value == "effect"
        ]
        self.assertEqual(effect_edges, [("first", "second")])
        schedule = [entry for entry in declaration.ir.schedule]
        self.assertLess(schedule.index("first"), schedule.index("second"))


class ExternIOCompileUnitTest(unittest.TestCase):
    def _declaration(self):
        @workload(id="extern-io.mixed")
        class Mixed:
            bank = State(
                scope=StateScope.REPLICA,
                parameters="weights",
                optimizer=AdamWSpec(lr=1e-3),
            )

            @task(
                inputs={"value": provided("value")},
                outputs={"pure_out": sink("pure")},
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def pure(self, value: object) -> object:
                return value

            @task(
                inputs={"value": provided("value")},
                outputs={"read_out": sink("read")},
                effects=(read(bank),),
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def read_state(self, value: object) -> object:
                return value

            @task(
                inputs={"value": provided("value")},
                outputs={"io_out": sink("io")},
                effects=(extern_io("trace:file"),),
                timing=Between("model.execute", "sampler.pre_draw"),
            )
            def io(self, value: object) -> object:
                return value

        return declare_workload(Mixed())

    def test_lowering_maps_extern_io_to_the_signed_effect_kind(self) -> None:
        graph = lower_schedule_graph((self._declaration().ir,))
        effects = {
            (effect.kind.value, effect.resource)
            for task in graph.tasks
            for effect in task.effects
        }
        self.assertIn(("state_read", "bank"), effects)
        self.assertIn(("extern_io", "trace:file"), effects)
        # extern_io forces capture off at the graph level.
        io_task = next(task for task in graph.tasks if task.task_id == "io")
        self.assertFalse(io_task.capture_safe)

    def test_extern_io_region_is_never_merged_and_never_side_stream(self) -> None:
        compilation = WorkloadCompilerCoordinator(
            (self._declaration(),),
            options=CompileOptions(
                memory_budget_bytes=1 << 20, allow_side_stream=True
            ),
        ).compile(_timeline())
        decision = compilation.decision
        io_regions = [
            region
            for region in decision.execution_regions
            if "io" in region.tasks
        ]
        self.assertEqual(len(io_regions), 1)
        io_region = io_regions[0]
        # The extern_io region stands alone even though `pure` shares its
        # window, and it keeps the extern_io effect on the signed region.
        self.assertEqual(io_region.tasks, ("io",))
        self.assertEqual(
            [effect.kind.value for effect in io_region.effects], ["extern_io"]
        )
        self.assertFalse(io_region.capture_safe)
        self.assertEqual(io_region.stream.value, "default")
        # The graph declares a state while the extern channel is not a state:
        # this compiled, which previously failed with "unknown typed state".
        self.assertIn("bank", {state.resource for state in compilation.graph.states})


if __name__ == "__main__":
    unittest.main()
