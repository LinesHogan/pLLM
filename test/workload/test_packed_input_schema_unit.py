"""Signed schema contracts for packed request rows and hidden snapshots."""

from __future__ import annotations

from dataclasses import replace
import json

import pytest

from tllm import _core
from tllm.contracts.backend_capability import BackendCapability
from tllm.contracts.hidden_locator import HiddenPhase, HiddenSite
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedSamplerRowBatch,
)
from tllm.contracts.request_identity import RequestKeyField
from tllm.schedule.contracts import (
    ExecutionRegion,
    BackendObservableEvent,
    CompileOptions,
    MainWorkloadTimeline,
    HiddenLayerKind as ScheduleHiddenLayerKind,
    MemoryPlan,
    PackedHiddenResourceBinding,
    RegionEffect,
    RegionCompilerAdapter,
    RegionImplementationKind,
    RegionInputBinding,
    RegionInputKind,
    RegionPort,
    ResidualResourceBinding,
    RegionTaskInvocation,
    RegionTaskPortBinding,
    ScheduleDecision,
    ScheduleCapability,
    ScheduleEffectKind,
    ScheduleInputKind,
    ScheduleTree,
    TaskPlacement,
    TimelineAnchor,
)
from tllm.workload.ir import (
    EffectIR,
    EffectKind,
    HiddenLayerKind as WorkloadHiddenLayerKind,
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
from tllm.workload import (
    Between,
    declare_workload,
    packed_request_rows,
    packed_sampler_rows,
    task,
    workload,
)


def _implementation() -> tuple[ImplementationIR, ...]:
    return (
        ImplementationIR(
            ImplementationKind.CAPTURED,
            "test:packed@1",
            capabilities=("torch.export",),
        ),
    )


def _packed_inputs() -> tuple[TaskInputIR, ...]:
    return (
        TaskInputIR(
            "rows",
            ValueSourceKind.PACKED_REQUEST_ROWS,
            "runtime.packed_phase_rows",
        ),
        TaskInputIR(
            "source",
            ValueSourceKind.PACKED_HIDDEN,
            "source",
            layer="0",
            layer_kind=WorkloadHiddenLayerKind.INDEX.value,
            site=HiddenSite.BLOCK_OUTPUT,
            packed_rows_port="rows",
        ),
        TaskInputIR(
            "target",
            ValueSourceKind.PACKED_HIDDEN,
            "target",
            layer="-1",
            layer_kind=WorkloadHiddenLayerKind.INDEX.value,
            site=HiddenSite.BLOCK_OUTPUT,
            packed_rows_port="rows",
        ),
        TaskInputIR(
            "selection",
            ValueSourceKind.STATE_SELECTION,
            "weights",
            phase=HiddenPhase.DECODE,
            packed_rows_port="rows",
        ),
    )


def _packed_workload() -> WorkloadIR:
    return WorkloadIR(
        workload_id="packed",
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
                name="train",
                ordinal=0,
                inputs=_packed_inputs(),
                outputs=(),
                effects=(
                    EffectIR(
                        EffectKind.READ,
                        "weights",
                        selection_port="selection",
                    ),
                ),
                timing=TimingIR("model.execute", "step.end"),
                implementations=_implementation(),
            ),
        ),
        values=(),
        edges=(),
        schedule=("train",),
        schema_version=4,
    )


def test_workload_v4_signs_one_packed_row_relation_per_task() -> None:
    workload = _packed_workload()
    encoded = json.loads(workload.to_json())

    assert encoded["schema_version"] == 4
    assert tuple(item["kind"] for item in encoded["tasks"][0]["inputs"]) == (
        "packed_request_rows",
        "packed_hidden",
        "packed_hidden",
        "state_selection",
    )
    assert tuple(
        item["packed_rows_port"] for item in encoded["tasks"][0]["inputs"]
    ) == ("", "rows", "rows", "rows")
    assert (
        encoded["tasks"][0]["inputs"][1]["layer"],
        encoded["tasks"][0]["inputs"][1]["layer_kind"],
        encoded["tasks"][0]["inputs"][1]["site"],
        encoded["tasks"][0]["inputs"][1]["phase"],
    ) == ("0", "index", "block_output", "")
    assert (
        ScheduleCapability.PACKED_PHASE_ROWS.value
        == BackendCapability.PACKED_PHASE_ROWS.value
    )
    assert (
        ScheduleCapability.PACKED_HIDDEN_BLOCK_OUTPUT.value
        == BackendCapability.PACKED_HIDDEN_BLOCK_OUTPUT.value
    )


def test_workload_v4_rejects_ambiguous_or_mixed_packed_inputs() -> None:
    workload = _packed_workload()
    task = workload.tasks[0]
    duplicate_rows = TaskInputIR(
        "other_rows",
        ValueSourceKind.PACKED_REQUEST_ROWS,
        "runtime.packed_phase_rows",
    )
    with pytest.raises(ValueError, match="one packed_request_rows"):
        replace(
            workload,
            tasks=(replace(task, inputs=(*task.inputs, duplicate_rows)),),
        )

    bad_hidden = replace(task.inputs[1], packed_rows_port="missing")
    with pytest.raises(ValueError, match="packed_rows_port"):
        replace(
            workload,
            tasks=(replace(task, inputs=(task.inputs[0], bad_hidden)),),
        )

    with pytest.raises(ValueError, match="locator"):
        replace(task.inputs[1], layer="", layer_kind="", site=HiddenSite.NONE)
    with pytest.raises(ValueError, match="locator"):
        replace(
            task.inputs[1],
            layer_kind=WorkloadHiddenLayerKind.NONE.value,
        )
    with pytest.raises(ValueError, match="phase NONE"):
        replace(task.inputs[1], phase=HiddenPhase.PREFILL)
    with pytest.raises(ValueError, match="runtime.packed_phase_rows"):
        replace(task.inputs[0], source="arbitrary")
    with pytest.raises(ValueError, match="must not declare packed_rows_port"):
        replace(task.inputs[0], packed_rows_port="rows")
    unbound_selection = replace(
        task.inputs[-1],
        phase=HiddenPhase.NONE,
        packed_rows_port="",
    )
    with pytest.raises(ValueError, match="state_selection.*packed_rows_port"):
        replace(
            workload,
            tasks=(
                replace(
                    task,
                    inputs=(*task.inputs[:-1], unbound_selection),
                ),
            ),
        )
    with pytest.raises(ValueError, match="packed_rows_port.*packed_request_rows"):
        replace(
            workload,
            tasks=(
                replace(
                    task,
                    inputs=(task.inputs[-1],),
                ),
            ),
        )

    ordinary_hidden = TaskInputIR(
        "ordinary",
        ValueSourceKind.HIDDEN,
        "hidden:0:block_output:prefill",
        layer="0",
        layer_kind=WorkloadHiddenLayerKind.INDEX.value,
        site=HiddenSite.BLOCK_OUTPUT,
        phase=HiddenPhase.PREFILL,
    )
    with pytest.raises(ValueError, match="ordinary hidden"):
        replace(
            workload,
            tasks=(replace(task, inputs=(*task.inputs, ordinary_hidden)),),
        )

    with pytest.raises(ValueError, match="schema v3"):
        replace(workload, schema_version=3)

    prefill_task = replace(
        task,
        name="prefill-train",
        ordinal=1,
        inputs=(
            *task.inputs[:-1],
            replace(task.inputs[-1], phase=HiddenPhase.PREFILL),
        ),
    )
    with pytest.raises(ValueError, match="cannot mix row phases"):
        replace(
            workload,
            tasks=(task, prefill_task),
            schedule=("train", "prefill-train"),
        )


def test_lowering_emits_graph_v5_and_preserves_packed_port_relation() -> None:
    graph = lower_schedule_graph((_packed_workload(),))
    assert graph.schema_version == 5
    task = graph.tasks[0]
    assert not task.capture_safe
    assert task.required_capabilities == (
        "hidden/block_output/packed@1",
        "runtime/packed_phase_rows@1",
    )
    assert tuple(item.kind for item in task.inputs) == (
        ScheduleInputKind.PACKED_REQUEST_ROWS,
        ScheduleInputKind.PACKED_HIDDEN,
        ScheduleInputKind.PACKED_HIDDEN,
        ScheduleInputKind.STATE_SELECTION,
    )
    assert tuple(item.packed_rows_port for item in task.inputs) == (
        "",
        "rows",
        "rows",
        "rows",
    )
    assert task.inputs[-1].source == "weights"
    assert task.inputs[-1].phase is HiddenPhase.DECODE
    assert graph == type(graph).from_json(graph.to_json())


def _packed_region() -> ExecutionRegion:
    return ExecutionRegion(
        region_id="region-packed",
        tasks=("train",),
        after_anchor="model.execute",
        before_anchor="step.end",
        placement_anchor="model.execute",
        implementation_kind=RegionImplementationKind.EXTERN,
        implementation_ids=("test:packed@1",),
        input_bindings=(
            RegionInputBinding(
                binding_id="rows-binding",
                kind=RegionInputKind.PACKED_REQUEST_ROWS,
                source="runtime.packed_phase_rows",
                consumers=(RegionPort("train", "rows"),),
            ),
            RegionInputBinding(
                binding_id="source-binding",
                kind=RegionInputKind.PACKED_HIDDEN,
                source="source",
                consumers=(RegionPort("train", "source"),),
                layer="0",
                layer_index=0,
                layer_kind=ScheduleHiddenLayerKind.INDEX,
                site=HiddenSite.BLOCK_OUTPUT,
                packed_rows_binding_id="rows-binding",
            ),
            RegionInputBinding(
                binding_id="target-binding",
                kind=RegionInputKind.PACKED_HIDDEN,
                source="target",
                consumers=(RegionPort("train", "target"),),
                layer="-1",
                layer_index=-1,
                layer_kind=ScheduleHiddenLayerKind.INDEX,
                site=HiddenSite.BLOCK_OUTPUT,
                packed_rows_binding_id="rows-binding",
            ),
            RegionInputBinding(
                binding_id="selection-binding",
                kind=RegionInputKind.STATE_SELECTION,
                source="weights",
                consumers=(RegionPort("train", "selection"),),
                phase=HiddenPhase.DECODE,
                packed_rows_binding_id="rows-binding",
            ),
        ),
        task_invocations=(
            RegionTaskInvocation(
                "train",
                inputs=(
                    RegionTaskPortBinding("rows", "rows-binding"),
                    RegionTaskPortBinding("source", "source-binding"),
                    RegionTaskPortBinding("target", "target-binding"),
                    RegionTaskPortBinding("selection", "selection-binding"),
                ),
            ),
        ),
        effects=(
            RegionEffect(
                ScheduleEffectKind.STATE_READ,
                "weights",
                "selection-binding",
            ),
        ),
        compile_safe=False,
        capture_safe=False,
        artifact_key="0" * 64,
    )


def test_decision_v7_signs_and_tamper_checks_packed_binding_relation() -> None:
    region = _packed_region()
    decision = ScheduleDecision(
        graph_key="graph",
        timeline_key="timeline",
        execution_order=("train",),
        placements=(TaskPlacement("train", "model.execute", "test:packed@1"),),
        schedule_tree=ScheduleTree("sequence", ("train",)),
        memory_plan=MemoryPlan(0, 0, 0, ()),
        execution_regions=(region,),
        completion_anchor="step.end",
        schema_version=7,
    )
    encoded = json.loads(decision.to_json())
    bindings = encoded["execution_regions"][0]["input_bindings"]
    assert tuple(binding["packed_rows_binding_id"] for binding in bindings) == (
        "",
        "rows-binding",
        "rows-binding",
        "rows-binding",
    )
    assert ScheduleDecision.from_json(decision.to_json()) == decision

    missing = json.loads(decision.to_json())
    del missing["execution_regions"][0]["input_bindings"][1][
        "packed_rows_binding_id"
    ]
    with pytest.raises(ValueError, match="packed_rows_binding_id"):
        ScheduleDecision.from_json(json.dumps(missing))

    mismatched = replace(
        region.input_bindings[1], packed_rows_binding_id="selection-binding"
    )
    with pytest.raises(ValueError, match="packed rows"):
        replace(
            decision,
            execution_regions=(
                replace(
                    region,
                    input_bindings=(
                        region.input_bindings[0],
                        mismatched,
                        *region.input_bindings[2:],
                    ),
                ),
            ),
        )

    unbound_selection = replace(
        region.input_bindings[-1],
        phase=HiddenPhase.NONE,
        packed_rows_binding_id="",
    )
    with pytest.raises(ValueError, match="same-task packed rows"):
        replace(
            region,
            input_bindings=(
                *region.input_bindings[:-1],
                unbound_selection,
            ),
        )

    with pytest.raises(ValueError, match="schema v6"):
        replace(decision, schema_version=6)


def test_region_rejects_cross_task_packed_rows_binding_substitution() -> None:
    region = _packed_region()
    other_rows = RegionInputBinding(
        binding_id="other-rows-binding",
        kind=RegionInputKind.PACKED_REQUEST_ROWS,
        source="runtime.packed_phase_rows",
        consumers=(RegionPort("other", "rows"),),
    )
    substituted = replace(
        region.input_bindings[1],
        packed_rows_binding_id="other-rows-binding",
    )
    with pytest.raises(ValueError, match="consumer task"):
        replace(
            region,
            tasks=("train", "other"),
            implementation_kind=RegionImplementationKind.COMPILED,
            implementation_ids=("test:packed@1", "test:other@1"),
            input_bindings=(
                region.input_bindings[0],
                substituted,
                *region.input_bindings[2:],
                other_rows,
            ),
            task_invocations=(
                region.task_invocations[0],
                RegionTaskInvocation(
                    "other",
                    inputs=(
                        RegionTaskPortBinding(
                            "rows", "other-rows-binding"
                        ),
                    ),
                ),
            ),
        )


def test_core_compiler_signs_v7_packed_relations_and_rejects_tamper() -> None:
    graph = lower_schedule_graph((_packed_workload(),))
    timeline = MainWorkloadTimeline(
        backend_id="test",
        backend_version="1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor(
                "model.execute",
                0,
                backend_events=(
                    BackendObservableEvent.PREPARE_INPUTS_POST,
                    BackendObservableEvent.EXECUTE_MODEL_POST,
                ),
            ),
            TimelineAnchor(
                "step.end",
                1,
                backend_events=(
                    BackendObservableEvent.SAMPLE_TOKENS_POST,
                ),
            ),
        ),
        regions=(),
        static_headroom_bytes=64,
        capabilities=tuple(
            sorted(
                (
                    BackendCapability.PACKED_HIDDEN_BLOCK_OUTPUT.value,
                    BackendCapability.PACKED_PHASE_ROWS.value,
                )
            )
        ),
        schema_version=2,
    )
    options = CompileOptions(
        packed_hidden_resource_manifest=(
            PackedHiddenResourceBinding(
                layer="-1",
                layer_kind=ScheduleHiddenLayerKind.INDEX,
                site=HiddenSite.BLOCK_OUTPUT,
                module_path="model.layers.-1",
            ),
            PackedHiddenResourceBinding(
                layer="0",
                layer_kind=ScheduleHiddenLayerKind.INDEX,
                site=HiddenSite.BLOCK_OUTPUT,
                module_path="model.layers.0",
            ),
        )
    )
    compiled = _core.compile_schedule(
        graph.to_json(), timeline.to_json(), options.to_json()
    )
    decision = ScheduleDecision.from_json(compiled)
    assert decision.schema_version == 7
    region = decision.execution_regions[0]
    rows = next(
        binding
        for binding in region.input_bindings
        if binding.kind is RegionInputKind.PACKED_REQUEST_ROWS
    )
    assert all(
        binding.packed_rows_binding_id == rows.binding_id
        for binding in region.input_bindings
        if binding.kind
        in (RegionInputKind.PACKED_HIDDEN, RegionInputKind.STATE_SELECTION)
    )

    with pytest.raises(RuntimeError, match="packed hidden resource manifest coverage"):
        _core.compile_schedule(
            graph.to_json(),
            timeline.to_json(),
            replace(
                options,
                packed_hidden_resource_manifest=(
                    options.packed_hidden_resource_manifest[0],
                ),
            ).to_json(),
        )

    with pytest.raises(RuntimeError, match="unexpected"):
        _core.compile_schedule(
            graph.to_json(),
            timeline.to_json(),
            replace(
                options,
                packed_hidden_resource_manifest=(
                    *options.packed_hidden_resource_manifest,
                    PackedHiddenResourceBinding(
                        layer="source",
                        layer_kind=ScheduleHiddenLayerKind.PATH,
                        site=HiddenSite.BLOCK_OUTPUT,
                        module_path="model.layers.1",
                    ),
                ),
            ).to_json(),
        )

    with pytest.raises(RuntimeError, match="residual resource manifest coverage"):
        _core.compile_schedule(
            graph.to_json(),
            timeline.to_json(),
            replace(
                options,
                residual_resource_manifest=(
                    ResidualResourceBinding(
                        layer_index=-1,
                        site=HiddenSite.BLOCK_OUTPUT,
                        phase=HiddenPhase.DECODE,
                        module_path="model.layers.-1",
                    ),
                    ResidualResourceBinding(
                        layer_index=0,
                        site=HiddenSite.BLOCK_OUTPUT,
                        phase=HiddenPhase.DECODE,
                        module_path="model.layers.0",
                    ),
                ),
                packed_hidden_resource_manifest=(),
            ).to_json(),
        )

    tampered = json.loads(compiled)
    tampered["execution_regions"][0]["input_bindings"][1][
        "packed_rows_binding_id"
    ] = "wrong"
    corrupted = json.dumps(
        tampered, separators=(",", ":"), ensure_ascii=True
    )
    with pytest.raises(RuntimeError, match="signature is invalid"):
        _core.validate_schedule(
            graph.to_json(),
            timeline.to_json(),
            options.to_json(),
            corrupted,
        )


def test_packed_sampler_rows_signs_same_task_relation_capability_and_eager_gate() -> None:
    rows = packed_request_rows()
    sampler_rows = packed_sampler_rows(rows=rows)

    @workload(id="packed.sampler_rows")
    class PackedSamplerRowsWorkload:
        @task(
            inputs={"rows": rows, "sampler_rows": sampler_rows},
            timing=Between("model.execute", "step.end"),
        )
        def train(
            self,
            rows: PackedPhaseRowBatch,
            sampler_rows: PackedSamplerRowBatch,
        ) -> None:
            _ = rows, sampler_rows

    declaration = declare_workload(PackedSamplerRowsWorkload())
    assert tuple(item.kind for item in declaration.ir.tasks[0].inputs) == (
        ValueSourceKind.PACKED_REQUEST_ROWS,
        ValueSourceKind.PACKED_SAMPLER_ROWS,
    )
    assert tuple(
        item.packed_rows_port for item in declaration.ir.tasks[0].inputs
    ) == ("", "rows")

    graph = lower_schedule_graph((declaration.ir,))
    assert graph.tasks[0].required_capabilities == (
        BackendCapability.PACKED_PHASE_ROWS.value,
        BackendCapability.PACKED_SAMPLER_ROWS.value,
    )
    timeline = MainWorkloadTimeline(
        backend_id="test",
        backend_version="1",
        runner_family="mrv2",
        graph_mode="eager",
        anchors=(
            TimelineAnchor(
                "model.execute",
                0,
                backend_events=(BackendObservableEvent.PREPARE_INPUTS_POST,),
            ),
            TimelineAnchor(
                "step.end",
                1,
                backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
            ),
        ),
        regions=(),
        static_headroom_bytes=64,
        capabilities=(
            BackendCapability.PACKED_PHASE_ROWS.value,
            BackendCapability.PACKED_SAMPLER_ROWS.value,
        ),
        schema_version=2,
    )
    decision = ScheduleDecision.from_json(
        _core.compile_schedule(
            graph.to_json(), timeline.to_json(), CompileOptions().to_json()
        )
    )
    region = decision.execution_regions[0]
    rows_binding = next(
        binding
        for binding in region.input_bindings
        if binding.kind is RegionInputKind.PACKED_REQUEST_ROWS
    )
    sampler_binding = next(
        binding
        for binding in region.input_bindings
        if binding.kind is RegionInputKind.PACKED_SAMPLER_ROWS
    )
    assert sampler_binding.source == "runtime.packed_sampler_rows"
    assert sampler_binding.packed_rows_binding_id == rows_binding.binding_id
    assert region.compile_safe is False
    assert region.compile_command is not None
    assert region.compile_command.adapter is RegionCompilerAdapter.EAGER

    with pytest.raises(RuntimeError, match="runtime/packed_sampler_rows@1"):
        _core.compile_schedule(
            graph.to_json(),
            replace(
                timeline,
                capabilities=(BackendCapability.PACKED_PHASE_ROWS.value,),
            ).to_json(),
            CompileOptions().to_json(),
        )
