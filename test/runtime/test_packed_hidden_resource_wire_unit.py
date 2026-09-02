"""Process-local packed hidden resources stay equal to the signed manifest."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace

import pytest
import torch

from tllm import _core
from tllm.contracts.hidden_locator import HiddenSite
from tllm.contracts.packed_rows import PackedPhaseRowBatch
from tllm.runtime.runtime import TllmRuntime
from tllm.runtime.workload_bindings import (
    NoWorkerBinding,
    WorkerRuntimeBinding,
    WorkloadRuntimeBinding,
)
from tllm.runtime.workload_plugins import (
    NativeWorkload,
    WorkloadWorkerSpec,
    reconstruct_worker_bindings,
)
from tllm.backends.protocol import BackendObservableEvent
from tllm.schedule import (
    HiddenLayerKind,
    MainWorkloadTimeline,
    PackedHiddenResourceBinding,
    ScheduleCapability,
    TimelineAnchor,
)
from tllm.schedule.contracts import canonical_json
from tllm.workload import (
    Between,
    packed_hidden,
    packed_request_rows,
    task,
    workload,
)


_ROWS = packed_request_rows()
_PACKED_BINDING = PackedHiddenResourceBinding(
    layer="0",
    layer_kind=HiddenLayerKind.INDEX,
    site=HiddenSite.BLOCK_OUTPUT,
    module_path="model.layers.0",
)


@workload(id="runtime.packed-hidden-resource-wire")
class _PackedHiddenResourceWorkload(NativeWorkload):
    def __init__(self, module_path: str) -> None:
        binding = replace(_PACKED_BINDING, module_path=module_path)
        spec = WorkloadWorkerSpec(
            factory=f"{__name__}:_reconstruct_packed_hidden_resource_workload",
            config={"module_path": module_path},
        )
        self._resources = WorkloadRuntimeBinding(
            owner=self,
            worker=WorkerRuntimeBinding(spec),
            packed_hidden_resources=(binding,),
        )

    def native_workload_resources(self) -> WorkloadRuntimeBinding:
        return self._resources

    @task(
        timing=Between("model.execute", "step.end"),
        inputs={
            "rows": _ROWS,
            "hidden": packed_hidden(0, rows=_ROWS),
        },
    )
    def train(self, rows: PackedPhaseRowBatch, hidden: torch.Tensor) -> None:
        _ = rows, hidden


def _reconstruct_packed_hidden_resource_workload(
    raw: Mapping[str, object],
) -> WorkloadRuntimeBinding:
    module_path = raw.get("module_path")
    if type(module_path) is not str:
        raise TypeError("packed hidden worker module_path must be a string")
    return _PackedHiddenResourceWorkload(module_path).native_workload_resources()


def _timeline() -> MainWorkloadTimeline:
    return MainWorkloadTimeline(
        backend_id="packed-hidden-resource-wire",
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
                backend_events=(BackendObservableEvent.SAMPLE_TOKENS_POST,),
            ),
        ),
        regions=(),
        static_headroom_bytes=64,
        capabilities=tuple(
            capability.value
            for capability in (
                ScheduleCapability.PACKED_HIDDEN_BLOCK_OUTPUT,
                ScheduleCapability.PACKED_PHASE_ROWS,
            )
        ),
        schema_version=2,
    )


def test_parent_and_worker_mechanically_rebuild_packed_resource_manifest() -> None:
    workload_instance = _PackedHiddenResourceWorkload("model.layers.0")
    runtime = TllmRuntime(
        workloads=(workload_instance,),
        workload_memory_budget_bytes=64,
    )

    assert runtime.workload_compile_options.packed_hidden_resource_manifest == (
        _PACKED_BINDING,
    )

    compilation = runtime.compile_workloads(_timeline())
    local_bindings = runtime.workload_execution_bindings
    assert local_bindings is not None
    assert local_bindings.packed_hidden_resources == (_PACKED_BINDING,)
    resources = workload_instance.native_workload_resources()
    match resources.worker:
        case WorkerRuntimeBinding(spec=spec):
            pass
        case NoWorkerBinding():
            raise AssertionError("packed worker resources require a reconstruction spec")
    reconstructed = reconstruct_worker_bindings(compilation, (spec,))

    assert reconstructed.packed_hidden_resources == (_PACKED_BINDING,)


def test_rust_reconstruction_gates_packed_coverage_and_exact_module_path() -> None:
    workload_instance = _PackedHiddenResourceWorkload("model.layers.0")
    runtime = TllmRuntime(
        workloads=(workload_instance,),
        workload_memory_budget_bytes=64,
    )
    compilation = runtime.compile_workloads(_timeline())
    physical = compilation.physical
    assert physical is not None
    plan_epoch = physical.plan_epoch
    signed_static_manifest = compilation.options.static_binding_manifest
    signed_packed_manifest = canonical_json((_PACKED_BINDING,))

    _core.validate_worker_reconstruction_wire(
        compilation.cache_key,
        compilation.graph.to_json(),
        compilation.timeline.to_json(),
        compilation.options.to_json(),
        compilation.decision.to_json(),
        plan_epoch,
        compilation.graph.to_json(),
        "null",
        "[]",
        signed_packed_manifest,
        signed_static_manifest,
    )
    _core.validate_local_execution_binding_wire(
        compilation.cache_key,
        compilation.graph.to_json(),
        compilation.timeline.to_json(),
        compilation.options.to_json(),
        compilation.decision.to_json(),
        plan_epoch,
        compilation.graph.to_json(),
        "[]",
        signed_packed_manifest,
        signed_static_manifest,
    )

    drifted_manifest = canonical_json(
        (replace(_PACKED_BINDING, module_path="model.layers.drifted"),)
    )
    with pytest.raises(
        RuntimeError,
        match="packed hidden resource manifest does not match",
    ):
        _core.validate_worker_reconstruction_wire(
            compilation.cache_key,
            compilation.graph.to_json(),
            compilation.timeline.to_json(),
            compilation.options.to_json(),
            compilation.decision.to_json(),
            plan_epoch,
            compilation.graph.to_json(),
            "null",
            "[]",
            drifted_manifest,
            signed_static_manifest,
        )

    with pytest.raises(RuntimeError, match="packed hidden resource manifest coverage"):
        _core.validate_local_execution_binding_wire(
            compilation.cache_key,
            compilation.graph.to_json(),
            compilation.timeline.to_json(),
            compilation.options.to_json(),
            compilation.decision.to_json(),
            plan_epoch,
            compilation.graph.to_json(),
            "[]",
            "[]",
            signed_static_manifest,
        )
