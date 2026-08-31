"""Selection evidence stays explicit from graph lowering to Region effects."""

from __future__ import annotations

from dataclasses import replace
import json
import unittest

from tllm.contracts.request_identity import RequestKeyField
from tllm.schedule.contracts import (
    ExecutionRegion,
    MemoryPlan,
    RegionEffect,
    RegionImplementationKind,
    RegionInputBinding,
    RegionInputKind,
    RegionPort,
    RegionTaskInvocation,
    ScheduleDecision,
    ScheduleEffect,
    ScheduleEffectKind,
    ScheduleGraph,
    ScheduleInput,
    ScheduleInputKind,
    ScheduleState,
    ScheduleTask,
    ScheduleTree,
    TaskPlacement,
    TaskTiming,
)
from tllm.workload.ir import (
    EffectIR,
    EffectKind,
    ImplementationIR,
    ImplementationKind,
    StateIR,
    StateScope,
    TaskIR,
    TaskInputIR,
    TimingIR,
    ValueSourceKind,
    WorkloadIR,
)
from tllm.workload.lowering import lower_schedule_graph


def _request_state(resource: str) -> ScheduleState:
    return ScheduleState(
        resource=resource,
        scope="request",
        request_key=(RequestKeyField.REQUEST_ID,),
    )


def _selection_task(
    resource: str,
    *,
    selection_port: str = "selection",
) -> ScheduleTask:
    return ScheduleTask(
        task_id="update",
        timing=TaskTiming("model.execute", "step.end"),
        implementation_capabilities=((),),
        inputs=(
            ScheduleInput(
                name="selection",
                kind=ScheduleInputKind.STATE_SELECTION,
                source=resource,
            ),
        ),
        effects=(
            ScheduleEffect(
                ScheduleEffectKind.STATE_TRANSITION,
                resource,
                selection_port,
            ),
        ),
    )


def _selection_region() -> ExecutionRegion:
    return ExecutionRegion(
        region_id="region-selection",
        tasks=("update",),
        after_anchor="model.execute",
        before_anchor="step.end",
        placement_anchor="model.execute",
        implementation_kind=RegionImplementationKind.EXTERN,
        implementation_ids=("test:update@1",),
        input_bindings=(
            RegionInputBinding(
                binding_id="selection-binding",
                kind=RegionInputKind.STATE_SELECTION,
                source="weights",
                consumers=(RegionPort("update", "selection"),),
            ),
        ),
        task_invocations=(RegionTaskInvocation("update"),),
        effects=(
            RegionEffect(
                ScheduleEffectKind.STATE_TRANSITION,
                "weights",
                "selection-binding",
            ),
        ),
        compile_safe=False,
        artifact_key="0" * 64,
    )


def _selection_decision() -> ScheduleDecision:
    return ScheduleDecision(
        graph_key="graph",
        timeline_key="timeline",
        execution_order=("update",),
        placements=(TaskPlacement("update", "model.execute", "test:update@1"),),
        schedule_tree=ScheduleTree("sequence", ("update",)),
        memory_plan=MemoryPlan(0, 0, 0, ()),
        execution_regions=(_selection_region(),),
        completion_anchor="step.end",
        schema_version=6,
    )


class ScheduleSelectionContractUnitTest(unittest.TestCase):
    def test_graph_v4_round_trips_request_selection_exactly(self) -> None:
        graph = ScheduleGraph(
            workload_key="selection",
            tasks=(_selection_task("weights"),),
            states=(_request_state("weights"),),
            schema_version=4,
        )

        encoded = json.loads(graph.to_json())
        self.assertEqual(encoded["schema_version"], 4)
        self.assertEqual(
            encoded["tasks"][0]["inputs"][0]["kind"],
            ScheduleInputKind.STATE_SELECTION.value,
        )
        self.assertEqual(
            encoded["tasks"][0]["effects"][0]["selection_port"],
            "selection",
        )
        self.assertEqual(ScheduleGraph.from_json(graph.to_json()), graph)

        missing = json.loads(graph.to_json())
        del missing["tasks"][0]["effects"][0]["selection_port"]
        with self.assertRaisesRegex(ValueError, "selection_port"):
            ScheduleGraph.from_json(json.dumps(missing))

        mismatched = json.loads(graph.to_json())
        mismatched["tasks"][0]["inputs"][0]["source"] = "other"
        with self.assertRaisesRegex(ValueError, "request state resource"):
            ScheduleGraph.from_json(json.dumps(mismatched))

    def test_graph_v1_to_v3_reject_selection_evidence(self) -> None:
        with self.assertRaisesRegex(ValueError, "before v4"):
            ScheduleGraph(
                workload_key="selection",
                tasks=(_selection_task("weights"),),
                states=(_request_state("weights"),),
                schema_version=3,
            )

        legacy_graph = ScheduleGraph(
            workload_key="legacy-no-selection",
            tasks=(
                ScheduleTask(
                    task_id="read",
                    timing=TaskTiming("model.execute", "step.end"),
                    implementation_capabilities=((),),
                    effects=(
                        ScheduleEffect(
                            ScheduleEffectKind.STATE_READ,
                            "weights",
                        ),
                    ),
                ),
            ),
            states=(ScheduleState("weights", "replica"),),
            schema_version=3,
        )
        explicit_empty = json.loads(legacy_graph.to_json())
        explicit_empty["tasks"][0]["effects"][0]["selection_port"] = ""
        with self.assertRaisesRegex(ValueError, "before v4.*selection_port"):
            ScheduleGraph.from_json(json.dumps(explicit_empty))

    def test_graph_v4_rejects_effects_on_unknown_state_resources(self) -> None:
        task = ScheduleTask(
            task_id="read",
            timing=TaskTiming("model.execute", "step.end"),
            implementation_capabilities=((),),
            effects=(
                ScheduleEffect(
                    ScheduleEffectKind.STATE_READ,
                    "missing",
                    "",
                ),
            ),
        )
        with self.assertRaisesRegex(ValueError, "unknown state resource"):
            ScheduleGraph(
                workload_key="unknown-state",
                tasks=(task,),
                states=(),
                schema_version=4,
            )

    def test_lowering_resolves_selection_source_with_the_state_resource(self) -> None:
        request_workload = WorkloadIR(
            workload_id="request",
            states=(
                StateIR(
                    name="weights",
                    scope=StateScope.REQUEST,
                    request_key=(RequestKeyField.REQUEST_ID,),
                    parameters=("weights",),
                    optimizer=None,
                ),
            ),
            tasks=(
                TaskIR(
                    name="update",
                    ordinal=0,
                    inputs=(
                        TaskInputIR(
                            "selection",
                            ValueSourceKind.STATE_SELECTION,
                            "weights",
                        ),
                    ),
                    outputs=(),
                    effects=(
                        EffectIR(
                            EffectKind.TRANSITION,
                            "weights",
                            selection_port="selection",
                        ),
                    ),
                    timing=TimingIR("model.execute", "step.end"),
                    implementations=(
                        ImplementationIR(
                            ImplementationKind.OPTIMIZER_TRANSITION,
                            "test:update@1",
                        ),
                    ),
                ),
            ),
            values=(),
            edges=(),
            schedule=("update",),
            schema_version=3,
        )
        empty_workload = WorkloadIR(
            workload_id="empty",
            states=(),
            tasks=(),
            values=(),
            edges=(),
            schedule=(),
            schema_version=3,
        )

        graph = lower_schedule_graph((request_workload, empty_workload))

        self.assertEqual(graph.schema_version, 4)
        task = graph.tasks[0]
        self.assertEqual(task.inputs[0].source, "request/weights")
        self.assertEqual(task.effects[0].resource, "request/weights")
        self.assertEqual(task.effects[0].selection_port, "selection")

        with self.assertRaisesRegex(ValueError, "bind exactly one effect"):
            WorkloadIR(
                workload_id=request_workload.workload_id,
                states=request_workload.states,
                tasks=(replace(request_workload.tasks[0], effects=()),),
                values=(),
                edges=(),
                schedule=("update",),
                schema_version=3,
            )
        duplicated_effect = request_workload.tasks[0].effects[0]
        with self.assertRaisesRegex(ValueError, "bind exactly one effect"):
            WorkloadIR(
                workload_id=request_workload.workload_id,
                states=request_workload.states,
                tasks=(
                    replace(
                        request_workload.tasks[0],
                        effects=(duplicated_effect, duplicated_effect),
                    ),
                ),
                values=(),
                edges=(),
                schedule=("update",),
                schema_version=3,
            )

    def test_legacy_request_effects_stay_on_graph_v3_and_cannot_mix_native(self) -> None:
        legacy = WorkloadIR(
            workload_id="legacy",
            states=(
                StateIR(
                    name="weights",
                    scope=StateScope.REQUEST,
                    request_key=(RequestKeyField.REQUEST_ID,),
                    parameters=("weights",),
                    optimizer=None,
                ),
            ),
            tasks=(
                TaskIR(
                    name="update",
                    ordinal=0,
                    inputs=(),
                    outputs=(),
                    effects=(EffectIR(EffectKind.TRANSITION, "weights"),),
                    timing=TimingIR("model.execute", "step.end"),
                    implementations=(
                        ImplementationIR(
                            ImplementationKind.OPTIMIZER_TRANSITION,
                            "legacy:update@1",
                        ),
                    ),
                ),
            ),
            values=(),
            edges=(),
            schedule=("update",),
            schema_version=2,
        )
        native_without_request_effect = WorkloadIR(
            workload_id="native",
            states=(),
            tasks=(),
            values=(),
            edges=(),
            schedule=(),
            schema_version=3,
        )

        self.assertEqual(lower_schedule_graph((legacy,)).schema_version, 3)
        with self.assertRaisesRegex(ValueError, "native WorkloadIR v3"):
            lower_schedule_graph((legacy, native_without_request_effect))

        legacy_without_effect = WorkloadIR(
            workload_id="legacy-empty",
            states=legacy.states,
            tasks=(),
            values=(),
            edges=(),
            schedule=(),
            schema_version=2,
        )
        with self.assertRaisesRegex(ValueError, "native WorkloadIR v3"):
            lower_schedule_graph(
                (legacy_without_effect, native_without_request_effect)
            )

    def test_decision_v6_binds_region_effect_to_state_selection_input(self) -> None:
        decision = _selection_decision()
        encoded = json.loads(decision.to_json())
        effect = encoded["execution_regions"][0]["effects"][0]
        self.assertEqual(effect["selection_binding_id"], "selection-binding")
        self.assertEqual(ScheduleDecision.from_json(decision.to_json()), decision)

        missing = json.loads(decision.to_json())
        del missing["execution_regions"][0]["effects"][0]["selection_binding_id"]
        with self.assertRaisesRegex(ValueError, "selection_binding_id"):
            ScheduleDecision.from_json(json.dumps(missing))

        mismatched = json.loads(decision.to_json())
        mismatched["execution_regions"][0]["effects"][0][
            "selection_binding_id"
        ] = "other"
        with self.assertRaisesRegex(ValueError, "selection_binding_id"):
            ScheduleDecision.from_json(json.dumps(mismatched))

        stale = json.loads(decision.to_json())
        stale["schema_version"] = 5
        with self.assertRaisesRegex(ValueError, "schema v5 is stale.*v7"):
            ScheduleDecision.from_json(json.dumps(stale))

        with self.assertRaisesRegex(ValueError, "schema v4 cannot carry"):
            ScheduleDecision(
                graph_key=decision.graph_key,
                timeline_key=decision.timeline_key,
                execution_order=decision.execution_order,
                placements=decision.placements,
                schedule_tree=decision.schedule_tree,
                memory_plan=decision.memory_plan,
                execution_regions=decision.execution_regions,
                completion_anchor=decision.completion_anchor,
                schema_version=4,
            )


if __name__ == "__main__":
    unittest.main()
