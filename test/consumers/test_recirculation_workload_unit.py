"""Native backend binding for vLLM Recirculation prefill."""

from __future__ import annotations

from dataclasses import replace
import json
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from tllm.backends.vllm import runner_adapter
from tllm.backends.protocol import OrderedRowFusionCapabilities
from tllm.backends.timeline import export_main_workload_timeline
from tllm.backends.vllm.v027_mrv2 import V027MRv2Driver
from tllm.consumers.recirculation import (
    PREFILL_BINDER_IMPLEMENTATION_ID,
    RecirculationConfig,
    RecirculationWorkload,
)
from tllm.contracts.backend_capability import BackendCapability
from tllm.runtime import residual_runtime
from tllm.runtime.backend_region_binders import (
    BackendRegionBinderSlot,
    BackendRegionStageContext,
)
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.specialization import RuntimeSpecializationSpec
from tllm.runtime.state import ResidualRuntimeState
from tllm.runtime.workload_plugins import (
    WorkloadWorkerSpec,
    static_backend_binding_manifest_wire,
)
from tllm.runtime.workload_bindings import (
    BackendRuntimeBinding,
    LocalRuntimeServices,
    WorkerRuntimeBinding,
)
from tllm.schedule import BackendObservableEvent, RegionImplementationKind
from tllm.consumers.recirculation import native_backend
from tllm.consumers.recirculation import stage_definition


def _runtime() -> tuple[TllmRuntime, RecirculationWorkload]:
    workload = RecirculationWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2, beta=0.8)
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


def _cold_spec() -> RuntimeSpecializationSpec:
    return RuntimeSpecializationSpec(
        rows=2,
        hidden_size=8,
        dtype=torch.float32,
        device=torch.device("cpu"),
        model_layer_count=3,
        max_prefill_tokens=2,
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
        BackendCapability.TYPED_LAYER_REPLAY.value
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
        PREFILL_BINDER_IMPLEMENTATION_ID: workload.backend.binder
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
    assert isinstance(resources.worker, WorkerRuntimeBinding)
    spec = resources.worker.spec

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
        backend=BackendRuntimeBinding(
            ports=resources.backend.ports,
            binders={changed_binder_id: BackendRegionBinderSlot(changed_binder_id)},
        ),
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
    plan = "canonical-typed-recurrence-stage"
    workload.backend.binder.stage(
        plan,
        lambda staged, inputs, step: calls.append((staged, inputs, step)),
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

    assert calls == [(plan, {}, 7)]


def test_cold_artifact_native_stage_binds_without_graph_or_plan_compilation() -> None:
    """The hook may bind signed facts, but cannot reconstruct a recurrence."""

    workload = RecirculationWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2, beta=0.8)
    )
    services = workload.native_workload_resources().services
    assert isinstance(services, LocalRuntimeServices)
    services.cold_specialize(_cold_spec())

    columns = runner_adapter.PackedRowPlanInputs(
        request_ids=("request-0",),
        scheduled_tokens=(2,),
        prompt_tokens=(2,),
        computed_tokens=(0,),
    )
    handle = SimpleNamespace(
        prepare=mock.Mock(),
    )
    driver = SimpleNamespace(
        install_typed_layer_replay=lambda **_: handle,
        restore_typed_layer_replay=lambda **_: 0,
        packed_row_plan_inputs=lambda **_: columns,
    )
    binding = SimpleNamespace(active=True, driver=driver)
    context = BackendRegionStageContext(
        implementation_id=PREFILL_BINDER_IMPLEMENTATION_ID,
        backend_event="prepare_inputs.post",
        region_id="region-test",
        slot=workload.backend.binder,
        runtime=object(),
        runner=object(),
        step_id=9,
        num_scheduled_tokens=(2,),
        input_batch=object(),
    )

    with (
        mock.patch.object(
            stage_definition,
            "build_recirculation_graph",
            side_effect=AssertionError("hook rebuilt a Recirculation graph"),
        ),
        mock.patch.object(
            native_backend._core,
            "compile_typed_recurrence_plan",
            side_effect=AssertionError("hook compiled a typed recurrence plan"),
        ),
        mock.patch.object(
            native_backend,
            "runner_binding_for",
            return_value=binding,
        ),
    ):
        workload.backend.binder.stage_from_backend(context)
        workload.backend.binder.invoke(9)

    prepared = handle.prepare.call_args.args[0]
    assert prepared.total_rows == 2
    assert tuple(request.request_id for request in prepared.requests) == ("request-0",)


def test_worker_reconstruction_loads_signed_artifact_without_any_compile() -> None:
    coordinator = RecirculationWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2, beta=0.8)
    )
    services = coordinator.native_workload_resources().services
    assert isinstance(services, LocalRuntimeServices)
    services.cold_specialize(_cold_spec())
    worker = coordinator.native_workload_resources().worker
    assert isinstance(worker, WorkerRuntimeBinding)
    spec = worker.spec
    expected_signature = coordinator.native_workload_resources().specialization_key[
        "stage_artifact_bundle"
    ]

    with (
        mock.patch.object(
            native_backend,
            "build_stage_definitions",
            side_effect=AssertionError("worker rebuilt stage definitions"),
        ),
        mock.patch.object(
            native_backend._core,
            "compile_typed_recurrence_plan",
            side_effect=AssertionError("worker compiled a typed recurrence plan"),
        ),
        mock.patch.object(
            native_backend._core,
            "compile_typed_recurrence_stage_artifact_json",
            side_effect=AssertionError("worker compiled a stage artifact"),
        ),
    ):
        rebuilt = RecirculationWorkload.reconstruct(spec.config)

    assert rebuilt.specialization_key["stage_artifact_bundle"] == expected_signature


def test_runtime_row_shape_miss_fails_closed_without_hook_compilation() -> None:
    workload = RecirculationWorkload(
        RecirculationConfig(source_layer=1, destination_layer=0, alpha=0.2, beta=0.8)
    )
    services = workload.native_workload_resources().services
    assert isinstance(services, LocalRuntimeServices)
    services.cold_specialize(_cold_spec())
    columns = runner_adapter.PackedRowPlanInputs(
        request_ids=("request-0",),
        scheduled_tokens=(3,),
        prompt_tokens=(3,),
        computed_tokens=(0,),
    )
    handle = SimpleNamespace()
    driver = SimpleNamespace(
        install_typed_layer_replay=lambda **_: handle,
        restore_typed_layer_replay=lambda **_: 0,
        packed_row_plan_inputs=lambda **_: columns,
    )
    binding = SimpleNamespace(active=True, driver=driver)
    context = BackendRegionStageContext(
        implementation_id=PREFILL_BINDER_IMPLEMENTATION_ID,
        backend_event="prepare_inputs.post",
        region_id="region-test",
        slot=workload.backend.binder,
        runtime=object(),
        runner=object(),
        step_id=10,
        num_scheduled_tokens=(3,),
        input_batch=object(),
    )

    with (
        mock.patch.object(
            stage_definition,
            "build_recirculation_graph",
            side_effect=AssertionError("shape miss rebuilt a Recirculation graph"),
        ),
        mock.patch.object(
            native_backend._core,
            "compile_typed_recurrence_plan",
            side_effect=AssertionError("shape miss compiled a typed recurrence plan"),
        ),
        mock.patch.object(
            native_backend,
            "runner_binding_for",
            return_value=binding,
        ),
        pytest.raises(RuntimeError, match="specialization-required"),
    ):
        workload.backend.binder.stage_from_backend(context)


def test_worker_reconstruction_rejects_missing_signed_stage_artifact() -> None:
    with pytest.raises(KeyError, match="typed_stage_artifact_bundle"):
        RecirculationWorkload.reconstruct(
            {
                "source_layer": 1,
                "destination_layer": 0,
                "alpha": 0.2,
                "beta": 0.8,
                "ramp_tokens": 0,
                "enabled": True,
            }
        )
