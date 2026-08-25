"""Native backend-binding Region for vLLM Recirculation prefill."""

from __future__ import annotations

from dataclasses import replace
import json
from unittest import mock

import pytest

from tllm.backends.protocol import OrderedRowFusionCapabilities
from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.consumers.recirculation import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationConfig,
    VllmRecirculationRegionWorkload,
)
from tllm.contracts.backend_capability import BackendCapability
from tllm.runtime import residual_runtime
from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import (
    WorkloadWorkerSpec,
    static_backend_binding_manifest_wire,
)
from tllm.schedule import BackendObservableEvent, RegionImplementationKind


def _runtime() -> tuple[TllmRuntime, VllmRecirculationRegionWorkload]:
    workload = VllmRecirculationRegionWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2)
    )
    return TllmRuntime(
        workloads=(workload,), workload_memory_budget_bytes=1 << 20
    ), workload


def _timeline():
    return export_main_workload_timeline(
        V027MRv2Driver().capabilities(),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )


def test_vllm_recirculation_is_one_signed_prepare_post_binder_region() -> None:
    runtime, workload = _runtime()
    compilation = runtime.compile_workloads(_timeline())

    assert runtime.consumers == ()
    assert len(compilation.decision.execution_regions) == 1
    region = compilation.decision.execution_regions[0]
    assert region.tasks == ("bind_prefill",)
    assert region.placement_anchor == "prepare.post"
    assert region.implementation_kind is RegionImplementationKind.EXTERN
    assert region.implementation_ids == (PREFILL_BINDER_IMPLEMENTATION_ID,)
    assert BackendCapability.BACKEND_REGION_BINDER.value in region.required_capabilities
    assert (
        BackendCapability.RECIRCULATION_PREFILL_BINDING.value
        in region.required_capabilities
    )
    binder_stages = tuple(
        stage
        for advance in compilation.decision.runtime_advance_plan
        for stage in advance.backend_region_binder_stages
    )
    assert len(binder_stages) == 1
    assert binder_stages[0].backend_event is BackendObservableEvent.PREPARE_INPUTS_POST
    assert binder_stages[0].region_id == region.region_id
    assert binder_stages[0].implementation_id == PREFILL_BINDER_IMPLEMENTATION_ID
    assert runtime.workload_execution_bindings is not None
    assert runtime.workload_execution_bindings.backend_region_binders == {
        PREFILL_BINDER_IMPLEMENTATION_ID: workload.binder
    }
    assert static_backend_binding_manifest_wire(
        runtime.workload_execution_bindings.static_binding_entries
    ) == compilation.options.static_binding_manifest


def test_local_recirculation_static_resources_must_match_signed_plan() -> None:
    runtime, workload = _runtime()
    compilation = runtime.compile_workloads(_timeline())
    bindings = runtime.workload_execution_bindings
    assert bindings is not None
    resources = workload.native_workload_resources()
    spec = resources.worker_spec
    assert spec is not None

    changed_config = WorkloadWorkerSpec(
        factory=spec.factory,
        config={**spec.config, "source_layer": 2},
    )
    changed_specialization = replace(
        resources,
        specialization_key={**resources.specialization_key, "alpha": float(0.3).hex()},
    )
    changed_binder_id = "tllm.recirculation:other@1"
    changed_binder = replace(
        resources,
        backend_region_binders={
            changed_binder_id: BackendRegionBinderSlot(changed_binder_id)
        },
    )
    for label, entry in (
        (
            "config",
            ("recirculation.vllm-prefill", resources, changed_config),
        ),
        (
            "specialization",
            ("recirculation.vllm-prefill", changed_specialization, spec),
        ),
        (
            "binder",
            ("recirculation.vllm-prefill", changed_binder, spec),
        ),
    ):
        drifted_bindings = replace(bindings, static_binding_entries=(entry,))
        with (
            pytest.raises(RuntimeError, match="static backend binding"),
            mock.patch.object(residual_runtime, "RUNTIME", ResidualRuntimeState()),
        ):
            residual_runtime.publish_workload_compilation(
                compilation,
                bindings=drifted_bindings,
            )


def test_fusion_policy_is_signed_into_native_region_artifact_identity() -> None:
    runtime, _workload = _runtime()
    wavefront = runtime.compile_workloads(_timeline())
    capabilities = V027MRv2Driver().capabilities()
    model_region = capabilities.model_region
    assert model_region is not None
    changed_policy_timeline = export_main_workload_timeline(
        replace(
            capabilities,
            model_region=replace(
                model_region,
                ordered_row_fusion=OrderedRowFusionCapabilities(
                    internalized_edges=frozenset({"kv_overwrite_read"}),
                    max_rows=3,
                ),
            ),
        ),
        graph_mode="eager",
        static_headroom_bytes=1 << 20,
    )
    changed_policy = runtime.compile_workloads(changed_policy_timeline)

    assert wavefront.timeline.capabilities != changed_policy.timeline.capabilities
    assert wavefront.decision.signature != changed_policy.decision.signature
    assert (
        wavefront.decision.execution_regions[0].artifact_key
        != changed_policy.decision.execution_regions[0].artifact_key
    )


def test_signed_backend_binder_stage_rejects_tampering_before_execution() -> None:
    runtime, _workload = _runtime()
    compilation = runtime.compile_workloads(_timeline())
    tampered = json.loads(compilation.decision.to_json())
    stage = next(
        stage
        for advance in tampered["runtime_advance_plan"]
        for stage in advance["backend_region_binder_stages"]
    )
    stage["implementation_id"] = "test.backend:tampered@1"

    with pytest.raises(ValueError, match="signed backend binder Region"):
        type(compilation.decision).from_json(json.dumps(tampered))

    stage["implementation_id"] = PREFILL_BINDER_IMPLEMENTATION_ID
    stage["region_id"] = "region-0000-tampered"
    with pytest.raises(ValueError, match="RuntimeAdvancePlan binder stages"):
        type(compilation.decision).from_json(json.dumps(tampered))


def test_signed_region_consumes_staged_plan_and_invokes_backend_once() -> None:
    runtime, workload = _runtime()
    compilation = runtime.compile_workloads(_timeline())
    state = ResidualRuntimeState()
    calls: list[tuple[object, object, int]] = []
    plan = "canonical-rust-prefill-plan"
    workload.binder.stage(
        plan,
        lambda staged, inputs, step: calls.append(
            (staged, inputs["config"], step)
        ),
        7,
    )

    with mock.patch.object(residual_runtime, "RUNTIME", state):
        residual_runtime.publish_workload_compilation(
            compilation,
            bindings=runtime.workload_execution_bindings,
        )
        residual_runtime.prepare_physical_workgraph_step(7)
        assert residual_runtime.advance_workload_anchor("prepare.post")
        execution = state.workload_execution
        assert execution is not None
        assert execution.executor.stats.execute_count == 1
        assert execution.retained_value_count == 0

    assert calls == [(plan, workload.binding_config, 7)]
