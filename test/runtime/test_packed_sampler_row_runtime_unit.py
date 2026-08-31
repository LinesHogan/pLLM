"""Runtime provenance contract for signed packed sampler rows."""

from __future__ import annotations

from dataclasses import dataclass, replace

import pytest
from tllm.contracts.packed_rows import (
    PackedPhaseRowBatch,
    PackedRowPlanInputs,
    PackedSamplerRowBatch,
    PackedSamplerRowLink,
)
from tllm.contracts.physical_plan import (
    ExecutorKey,
    NodeKey,
    PhysicalNodePlan,
    StreamClass,
)
from tllm.runtime.packed_phase_row_staging import PackedPhaseRowStaging
from tllm.runtime.scheduling.workgraph_executor import (
    PhysicalExecutionContext,
    PhysicalExecutionKind,
)
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_execution import (
    NoExecutionLifecycle,
    StateVersionRegistry,
    _RegionValueStore,
    _SerialStream,
    _SignedMaterializationBindings,
)
from tllm.schedule import RegionInputBinding, RegionInputKind, RegionPort
from tllm.workload.region_compiler import RegionCallableArtifact


@dataclass(frozen=True, slots=True)
class _ObservedSamplerRows:
    rows: PackedPhaseRowBatch
    sampler_rows: PackedSamplerRowBatch


def test_runtime_resolves_sampler_links_from_one_exact_packed_row_batch() -> None:
    rows = PackedPhaseRowStaging().stage(
        4,
        7,
        PackedRowPlanInputs(
            request_ids=("0_parent", "1_parent"),
            scheduled_tokens=(2, 1),
            prompt_tokens=(2, 1),
            computed_tokens=(0, 1),
        ),
        prompt_indices={"parent": 3},
        sample_indices={},
    )
    sampler_rows = PackedSamplerRowBatch(
        row_batch=rows,
        links=(
            PackedSamplerRowLink(0, 2),
            PackedSamplerRowLink(1, 0),
        ),
    )
    observed: list[_ObservedSamplerRows] = []
    bindings = (
        RegionInputBinding(
            binding_id="rows",
            kind=RegionInputKind.PACKED_REQUEST_ROWS,
            source="runtime.packed_phase_rows",
            consumers=(RegionPort("guide", "rows"),),
        ),
        RegionInputBinding(
            binding_id="sampler_rows",
            kind=RegionInputKind.PACKED_SAMPLER_ROWS,
            source="runtime.packed_sampler_rows",
            consumers=(RegionPort("guide", "sampler_rows"),),
            packed_rows_binding_id="rows",
        ),
    )

    def execute(
        rows: PackedPhaseRowBatch,
        sampler_rows: PackedSamplerRowBatch,
    ) -> tuple[()]:
        observed.append(_ObservedSamplerRows(rows, sampler_rows))
        return ()

    artifact = RegionCallableArtifact(
        region_id="packed-sampler-rows",
        artifact_key="0" * 64,
        input_keys=("rows", "sampler_rows"),
        output_keys=(),
        callable=execute,
        compiled=False,
        input_bindings=bindings,
    )
    store = _RegionValueStore(
        runtime=ResidualRuntimeState(),
        providers={},
        sinks={},
        artifacts=(artifact,),
        materializations=_SignedMaterializationBindings(()),
        state_versions=StateVersionRegistry((), {}, ()),
        lifecycle=NoExecutionLifecycle(),
    )
    context = PhysicalExecutionContext(
        step_id=7,
        plan_epoch=4,
        node=PhysicalNodePlan(
            node_key=NodeKey("test", "packed-sampler-rows"),
            executor=ExecutorKey("test", "packed-sampler-rows"),
            stream=StreamClass.DEFAULT,
        ),
        kind=PhysicalExecutionKind.EAGER,
        stream=_SerialStream(),
        tensors=(),
        leases=(),
    )

    store.prepare_step(7)
    store.stage_packed_rows(7, rows)
    with pytest.raises(RuntimeError, match="differ from the staged rows"):
        store.stage_packed_sampler_rows(
            7,
            replace(sampler_rows, row_batch=replace(rows)),
        )
    store.stage_packed_sampler_rows(7, sampler_rows)
    store.execute(artifact, context)

    assert observed == [_ObservedSamplerRows(rows, sampler_rows)]
    assert sampler_rows.identities == (
        rows.rows[2].identity,
        rows.rows[0].identity,
    )
    store.finish_step(7)
