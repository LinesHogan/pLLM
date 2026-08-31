"""CPU execution contracts for signed packed Region inputs."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
import torch

from tllm.contracts.hidden_locator import HiddenSite
from tllm.contracts.packed_hidden import (
    PackedHiddenSnapshot,
    PackedHiddenSnapshotBatch,
)
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
)
from tllm.contracts.physical_plan import (
    ExecutorKey,
    NodeKey,
    PhysicalFlowPlan,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.contracts.request_identity import RequestKeyField
from tllm.contracts.workload_ports import SKIPPED
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging
from tllm.runtime.request_trainable_state import (
    RequestStateRuntimeService,
    RequestStateSelectionBatch,
    RequestTrainableStateStore,
)
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutor,
    PhysicalExecutionContext,
    PhysicalExecutionKind,
)
from tllm.runtime.scheduling.plan_store import PlanResources, PlanSnapshotStore
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import (
    NoExecutionLifecycle,
    NativeWorkloadExecution,
    StateVersionRegistry,
    _RegionValueStore,
    _SerialStream,
    _SignedMaterializationBindings,
)
from tllm.schedule import (
    CompileOptions,
    HiddenLayerKind,
    MainWorkloadTimeline,
    MemoryPlan,
    PackedHiddenResourceBinding,
    RegionInputBinding,
    RegionInputKind,
    RegionPort,
    ScheduleDecision,
    ScheduleGraph,
    ScheduleTree,
    TimelineAnchor,
    WorkloadCompilation,
)
from tllm.workload.region_compiler import RegionCallableArtifact


@dataclass(frozen=True, slots=True)
class _ObservedPackedInputs:
    rows: PackedPhaseRowBatch
    layer_zero: torch.Tensor
    layer_minus_one: torch.Tensor
    selection: RequestStateSelectionBatch


def _packed_resources() -> tuple[PackedHiddenResourceBinding, ...]:
    return (
        PackedHiddenResourceBinding(
            layer="-1",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="model.layers.-1",
        ),
        PackedHiddenResourceBinding(
            layer="0",
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            module_path="model.layers.0",
        ),
    )


def _packed_rows(
    *, plan_epoch: int = 4, step_id: int = 7
) -> PackedPhaseRowBatch:
    return PackedPhaseRowStaging().stage(
        plan_epoch,
        step_id,
        PackedRowPlanInputs(
            request_ids=("request-a", "request-b"),
            scheduled_tokens=(2, 1),
            prompt_tokens=(2, 1),
            computed_tokens=(0, 1),
        ),
        prompt_indices={"request-a": 3, "request-b": 5},
        sample_indices={},
    )


def _snapshot_batch(
    rows: PackedPhaseRowBatch,
    resources: tuple[PackedHiddenResourceBinding, ...],
    hidden: tuple[torch.Tensor, ...],
) -> PackedHiddenSnapshotBatch:
    return PackedHiddenSnapshotBatch(
        rows,
        tuple(
            PackedHiddenSnapshot(resource, tensor)
            for resource, tensor in zip(resources, hidden, strict=True)
        ),
    )


def _request_state_service() -> RequestStateRuntimeService[torch.nn.Module]:
    def module_factory() -> torch.nn.Module:
        return torch.nn.Linear(2, 2, bias=False)

    def optimizer_factory(
        module: torch.nn.Module,
    ) -> torch.optim.Optimizer:
        return torch.optim.SGD(module.parameters(), lr=0.1)

    return RequestStateRuntimeService(
        RequestTrainableStateStore(
            resource="weights",
            key_fields=(RequestKeyField.REQUEST_ID,),
            capacity=2,
            epoch=1,
            device=torch.device("cpu"),
            module_factory=module_factory,
            optimizer_factory=optimizer_factory,
        )
    )


def _packed_artifact(
    observed: list[_ObservedPackedInputs],
) -> RegionCallableArtifact:
    bindings = (
        RegionInputBinding(
            binding_id="rows",
            kind=RegionInputKind.PACKED_REQUEST_ROWS,
            source="runtime.packed_phase_rows",
            consumers=(RegionPort("train", "rows"),),
        ),
        RegionInputBinding(
            binding_id="layer_zero",
            kind=RegionInputKind.PACKED_HIDDEN,
            source="packed_hidden:0:block_output",
            consumers=(RegionPort("train", "layer_zero"),),
            layer="0",
            layer_index=0,
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            packed_rows_binding_id="rows",
        ),
        RegionInputBinding(
            binding_id="layer_minus_one",
            kind=RegionInputKind.PACKED_HIDDEN,
            source="packed_hidden:-1:block_output",
            consumers=(RegionPort("train", "layer_minus_one"),),
            layer="-1",
            layer_index=-1,
            layer_kind=HiddenLayerKind.INDEX,
            site=HiddenSite.BLOCK_OUTPUT,
            packed_rows_binding_id="rows",
        ),
        RegionInputBinding(
            binding_id="selection",
            kind=RegionInputKind.STATE_SELECTION,
            source="weights",
            consumers=(RegionPort("train", "selection"),),
            packed_rows_binding_id="rows",
        ),
    )

    def execute(
        rows: PackedPhaseRowBatch,
        layer_zero: torch.Tensor,
        layer_minus_one: torch.Tensor,
        selection: RequestStateSelectionBatch,
    ) -> tuple[()]:
        observed.append(
            _ObservedPackedInputs(
                rows,
                layer_zero,
                layer_minus_one,
                selection,
            )
        )
        return ()

    return RegionCallableArtifact(
        region_id="packed-inputs",
        artifact_key="0" * 64,
        input_keys=tuple(binding.binding_id for binding in bindings),
        output_keys=(),
        callable=execute,
        compiled=False,
        input_bindings=bindings,
        output_bindings=(),
        effects=(),
    )


def _context(*, plan_epoch: int = 4, step_id: int = 7) -> PhysicalExecutionContext:
    return PhysicalExecutionContext(
        step_id=step_id,
        plan_epoch=plan_epoch,
        node=PhysicalNodePlan(
            node_key=NodeKey("test", "packed-inputs"),
            executor=ExecutorKey("test", "packed-inputs"),
            stream=StreamClass.DEFAULT,
        ),
        kind=PhysicalExecutionKind.EAGER,
        stream=_SerialStream(),
        tensors=(),
        leases=(),
    )


def _value_store(
    artifact: RegionCallableArtifact,
    service: RequestStateRuntimeService[torch.nn.Module],
    *,
    resources: tuple[PackedHiddenResourceBinding, ...] | None = None,
) -> _RegionValueStore:
    return _RegionValueStore(
        runtime=ResidualRuntimeState(),
        providers={},
        sinks={},
        artifacts=(artifact,),
        materializations=_SignedMaterializationBindings(()),
        state_versions=StateVersionRegistry((), {}, ()),
        lifecycle=NoExecutionLifecycle(),
        request_state_services={"weights": service},
        packed_hidden_resources=(
            _packed_resources() if resources is None else resources
        ),
    )


def _native_execution(store: _RegionValueStore) -> NativeWorkloadExecution:
    physical = PhysicalFlowPlan(plan_epoch=4, nodes=())
    compilation = WorkloadCompilation(
        cache_key="0" * 64,
        graph=ScheduleGraph(
            workload_key="packed-runtime-test",
            tasks=(),
            schema_version=5,
        ),
        timeline=MainWorkloadTimeline(
            backend_id="test",
            backend_version="1",
            runner_family="mrv2",
            graph_mode="eager",
            anchors=(
                TimelineAnchor("model.execute", 0),
                TimelineAnchor("step.end", 1),
            ),
            regions=(),
            static_headroom_bytes=0,
            capabilities=(),
            schema_version=1,
        ),
        options=CompileOptions(),
        decision=ScheduleDecision(
            graph_key="1" * 64,
            timeline_key="2" * 64,
            execution_order=(),
            placements=(),
            schedule_tree=ScheduleTree("sequence", ()),
            memory_plan=MemoryPlan(0, 0, 0, ()),
            schema_version=1,
        ),
        physical=physical,
    )
    executor = PhysicalExecutor(
        snapshots=PlanSnapshotStore(PlanResources(physical)),
        handlers={},
    )
    return NativeWorkloadExecution(
        compilation=compilation,
        executor=executor,
        completion_anchor="step.end",
        _owned_consumers=(),
        _value_store=store,
    )


def test_packed_inputs_resolve_one_epoch_step_and_complete_row_identity() -> None:
    observed: list[_ObservedPackedInputs] = []
    artifact = _packed_artifact(observed)
    service = _request_state_service()
    store = _value_store(artifact, service)
    rows = _packed_rows()
    minus_one_hidden = torch.full((3, 2), -1.0)
    zero_hidden = torch.full((3, 2), 2.0)
    snapshots = _snapshot_batch(
        rows,
        _packed_resources(),
        (minus_one_hidden, zero_hidden),
    )

    assert store.requires_packed_rows
    assert not store.requires_request_identities
    store.prepare_step(7)
    store.stage_packed_rows(7, rows)
    store.stage_packed_hidden_snapshots(7, snapshots)
    store.execute(artifact, _context())

    assert len(observed) == 1
    result = observed[0]
    assert result.rows is rows
    torch.testing.assert_close(result.layer_zero, zero_hidden)
    torch.testing.assert_close(result.layer_minus_one, minus_one_hidden)
    assert tuple(row.identity for row in result.selection.rows) == rows.identities
    assert (len(rows.prefill_rows), len(rows.decode_rows)) == (2, 1)
    assert len(result.selection.rows) == 3
    assert len(result.selection.leases) == 2
    assert tuple(version.version for version in service.versions()) == (0, 0)
    store.finish_step(7)


def test_native_execution_stages_only_its_prepared_epoch_and_step() -> None:
    artifact = _packed_artifact([])
    store = _value_store(artifact, _request_state_service())
    execution = _native_execution(store)
    rows = _packed_rows()
    snapshots = _snapshot_batch(
        rows,
        _packed_resources(),
        (torch.zeros((3, 2)), torch.ones((3, 2))),
    )

    assert execution.requires_packed_rows
    assert not execution.requires_request_identities
    execution.prepare_step(7)
    execution.stage_packed_rows(rows)
    execution.stage_packed_hidden_snapshots(snapshots)
    execution.finish_step(7)

    execution.prepare_step(8)
    with pytest.raises(RuntimeError, match="another native execution epoch"):
        execution.stage_packed_rows(_packed_rows(plan_epoch=5, step_id=8))
    with pytest.raises(RuntimeError, match="prepared native step"):
        execution.stage_packed_rows(_packed_rows(step_id=9))
    execution.finish_step(8)


def test_native_execution_owns_and_retires_the_canonical_packed_row_batch() -> None:
    artifact = _packed_artifact([])
    execution = _native_execution(
        _value_store(artifact, _request_state_service())
    )
    columns = PackedRowPlanInputs(
        request_ids=("parent",),
        scheduled_tokens=(1,),
        prompt_tokens=(1,),
        computed_tokens=(1,),
    )

    execution.prepare_step(7)
    rows = execution.stage_packed_row_plan(
        7,
        columns,
        {"parent": 3},
        {"parent": 9},
    )
    assert execution.packed_rows_for_step(7) is rows
    execution.finish_step(7)
    with pytest.raises(RuntimeError, match="were not staged"):
        execution.packed_rows_for_step(7)

    execution.prepare_step(8)
    next_rows = execution.stage_packed_row_plan(
        8,
        columns,
        {"parent": 3},
        {"parent": 9},
    )
    assert next_rows.step_id == 8
    execution.finish_step(8)


def test_packed_inputs_reject_missing_or_mismatched_runtime_provenance() -> None:
    artifact = _packed_artifact([])
    store = _value_store(artifact, _request_state_service())
    rows = _packed_rows()
    snapshots = _snapshot_batch(
        rows,
        _packed_resources(),
        (torch.zeros((3, 2)), torch.ones((3, 2))),
    )
    store.prepare_step(7)
    store.stage_packed_rows(7, rows)

    with pytest.raises(RuntimeError, match="must be staged"):
        store.execute(artifact, _context())
    with pytest.raises(RuntimeError, match="plan epoch or step"):
        store.resolve_inputs(artifact, _context(plan_epoch=5))

    changed_rows = replace(
        rows,
        rows=(replace(rows.rows[0], token_offset=9), *rows.rows[1:]),
    )
    with pytest.raises(RuntimeError, match="differ from the staged rows"):
        store.stage_packed_hidden_snapshots(
            7,
            _snapshot_batch(
                changed_rows,
                _packed_resources(),
                (torch.zeros((3, 2)), torch.ones((3, 2))),
            ),
        )

    drifted_resources = (
        replace(
            _packed_resources()[0],
            module_path="model.layers.drifted",
        ),
        _packed_resources()[1],
    )
    with pytest.raises(RuntimeError, match="signed local resources"):
        store.stage_packed_hidden_snapshots(
            7,
            _snapshot_batch(
                rows,
                drifted_resources,
                (torch.zeros((3, 2)), torch.ones((3, 2))),
            ),
        )

    store.stage_packed_hidden_snapshots(7, snapshots)
    with pytest.raises(RuntimeError, match="already staged"):
        store.stage_packed_hidden_snapshots(
            7,
            _snapshot_batch(
                rows,
                _packed_resources(),
                (torch.zeros((3, 2)), torch.ones((3, 2))),
            ),
        )
    store.finish_step(7)


def test_empty_packed_rows_are_the_only_skipped_selection_batch() -> None:
    artifact = _packed_artifact([])
    service = _request_state_service()
    store = _value_store(artifact, service)
    rows = PackedPhaseRowStaging().stage(
        4,
        7,
        PackedRowPlanInputs(
            request_ids=("request-a",),
            scheduled_tokens=(0,),
            prompt_tokens=(1,),
            computed_tokens=(0,),
        ),
        prompt_indices={"request-a": 3},
        sample_indices={},
    )
    snapshots = _snapshot_batch(
        rows,
        _packed_resources(),
        (torch.empty((0, 2)), torch.empty((0, 2))),
    )

    store.prepare_step(7)
    store.stage_packed_rows(7, rows)
    store.stage_packed_hidden_snapshots(7, snapshots)
    inputs = store.resolve_inputs(artifact, _context())

    assert inputs["rows"] is rows
    assert inputs["selection"] is SKIPPED
    assert service.versions() == ()
    store.finish_step(7)


def test_snapshot_executor_accepts_generic_exact_resource_arity() -> None:
    rows_binding = RegionInputBinding(
        binding_id="rows",
        kind=RegionInputKind.PACKED_REQUEST_ROWS,
        source="runtime.packed_phase_rows",
        consumers=(RegionPort("single", "rows"),),
    )
    hidden_binding = RegionInputBinding(
        binding_id="hidden",
        kind=RegionInputKind.PACKED_HIDDEN,
        source="packed_hidden:-1:block_output",
        consumers=(RegionPort("single", "hidden"),),
        layer="-1",
        layer_index=-1,
        layer_kind=HiddenLayerKind.INDEX,
        site=HiddenSite.BLOCK_OUTPUT,
        packed_rows_binding_id="rows",
    )
    observed: list[torch.Tensor] = []

    def execute_single(
        rows: PackedPhaseRowBatch, hidden: torch.Tensor
    ) -> tuple[()]:
        _ = rows
        observed.append(hidden)
        return ()

    single_artifact = RegionCallableArtifact(
        region_id="single-packed-hidden",
        artifact_key="3" * 64,
        input_keys=("rows", "hidden"),
        output_keys=(),
        callable=execute_single,
        compiled=False,
        input_bindings=(rows_binding, hidden_binding),
    )
    single_resource = (_packed_resources()[0],)
    single_store = _value_store(
        single_artifact,
        _request_state_service(),
        resources=single_resource,
    )
    rows = _packed_rows()
    single_hidden = torch.full((3, 2), 7.0)
    single_store.prepare_step(7)
    single_store.stage_packed_rows(7, rows)
    single_store.stage_packed_hidden_snapshots(
        7,
        _snapshot_batch(rows, single_resource, (single_hidden,)),
    )
    single_store.execute(single_artifact, _context())
    torch.testing.assert_close(observed[0], single_hidden)
    single_store.finish_step(7)

    teacher = PackedHiddenResourceBinding(
        layer="teacher",
        layer_kind=HiddenLayerKind.PATH,
        site=HiddenSite.BLOCK_OUTPUT,
        module_path="model.teacher",
    )
    teacher_binding = RegionInputBinding(
        binding_id="teacher",
        kind=RegionInputKind.PACKED_HIDDEN,
        source="packed_hidden:teacher:block_output",
        consumers=(RegionPort("train", "teacher"),),
        layer="teacher",
        layer_kind=HiddenLayerKind.PATH,
        site=HiddenSite.BLOCK_OUTPUT,
        packed_rows_binding_id="rows",
    )
    triple_resources = (*_packed_resources(), teacher)
    base_artifact = _packed_artifact([])
    triple_artifact = replace(
        base_artifact,
        input_keys=(*base_artifact.input_keys, teacher_binding.binding_id),
        input_bindings=(*base_artifact.input_bindings, teacher_binding),
    )
    triple_store = _value_store(
        triple_artifact,
        _request_state_service(),
        resources=triple_resources,
    )
    teacher_hidden = torch.full((3, 2), 11.0)
    triple_store.prepare_step(7)
    triple_store.stage_packed_rows(7, rows)
    triple_store.stage_packed_hidden_snapshots(
        7,
        _snapshot_batch(
            rows,
            triple_resources,
            (
                torch.full((3, 2), -1.0),
                torch.full((3, 2), 2.0),
                teacher_hidden,
            ),
        ),
    )
    inputs = triple_store.resolve_inputs(triple_artifact, _context())
    assert inputs["teacher"] is teacher_hidden
    triple_store.finish_step(7)


def test_request_state_resource_cannot_mix_packed_and_ordinary_identity() -> None:
    artifact = _packed_artifact([])
    ordinary_selection = RegionInputBinding(
        binding_id="ordinary_selection",
        kind=RegionInputKind.STATE_SELECTION,
        source="weights",
        consumers=(RegionPort("ordinary", "selection"),),
    )
    mixed = replace(
        artifact,
        input_keys=(*artifact.input_keys, ordinary_selection.binding_id),
        input_bindings=(*artifact.input_bindings, ordinary_selection),
    )

    with pytest.raises(RuntimeError, match="cannot mix ordinary and packed"):
        _value_store(mixed, _request_state_service())
