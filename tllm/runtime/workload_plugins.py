"""Spawn-safe reconstruction of process-local workload Region bindings."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from tllm.runtime.workload_bindings import (
    DiagnosticsRuntimeServices,
    LocalRuntimeServices,
    NoRuntimeServices,
    NoSamplerBinding,
    OptimizerTransitionService,
    RuntimeInputBinding,
    RuntimeOutputBinding,
    RuntimePortRegistry,
    SamplerRuntimeBinding,
    WorkloadRuntimeBinding,
)
from tllm.runtime.request_trainable_state import RequestStateRuntimeService

if TYPE_CHECKING:
    import torch

    from tllm.contracts.packed_hidden import PackedHiddenResourceBinding
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
    from tllm.runtime.specialization import RuntimeSpecializationSpec
    from tllm.runtime.workload_execution import WorkloadExecutionBindings
    from tllm.schedule import (
        ResidualResourceBinding,
        WorkloadCompilation,
    )


@dataclass(frozen=True, slots=True)
class WorkloadWorkerSpec:
    """Pure-data recipe for rebuilding one workload inside a worker."""

    factory: str
    config: Mapping[str, object] = field(default_factory=lambda: MappingProxyType({}))

    def __post_init__(self) -> None:
        if ":" not in self.factory:
            raise ValueError("WorkloadWorkerSpec.factory must be 'module:qualname'")
        module_name, qualname = self.factory.split(":", 1)
        if not module_name or not qualname or any(
            not part or part == "<locals>" for part in qualname.split(".")
        ):
            raise ValueError("WorkloadWorkerSpec.factory must be importable")
        normalized = dict(self.config)
        try:
            encoded = json.dumps(
                normalized,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise TypeError("WorkloadWorkerSpec.config must be pure JSON data") from exc
        object.__setattr__(self, "config", MappingProxyType(_json_object(encoded)))

    def to_wire(self) -> dict[str, object]:
        return {"factory": self.factory, "config": dict(self.config)}

    @classmethod
    def from_wire(cls, raw: object) -> "WorkloadWorkerSpec":
        if type(raw) is not dict:
            raise ValueError(
                "WorkloadWorkerSpec wire must contain exactly factory and config"
            )
        wire = cast(dict[str, object], raw)
        if set(wire) != {"factory", "config"}:
            raise ValueError(
                "WorkloadWorkerSpec wire must contain exactly factory and config"
            )
        factory = wire["factory"]
        config = wire["config"]
        if type(factory) is not str:
            raise TypeError("WorkloadWorkerSpec wire factory must be a string")
        if type(config) is not dict:
            raise TypeError("WorkloadWorkerSpec wire config must be a JSON object")
        return cls(factory=factory, config=cast(dict[str, object], config))


def _json_object(encoded: str) -> dict[str, object]:
    decoded = cast(object, json.loads(encoded))
    if type(decoded) is not dict:
        raise TypeError("WorkloadWorkerSpec.config must encode a JSON object")
    result: dict[str, object] = {}
    for name, value in cast(dict[object, object], decoded).items():
        if type(name) is not str:
            raise TypeError("WorkloadWorkerSpec.config must use string keys")
        result[name] = value
    return result


@dataclass(frozen=True, slots=True)
class NoWorkloadRuntimeBinding:
    """A pure declaration with no process-local runtime binding."""


StaticRuntimeBinding = NoWorkloadRuntimeBinding | WorkloadRuntimeBinding
StaticBackendBindingEntry = tuple[
    str, StaticRuntimeBinding, WorkloadWorkerSpec
]


def static_backend_binding_manifest_wire(
    entries: tuple[StaticBackendBindingEntry, ...],
) -> str:
    """Transport live static backend semantics for Rust comparison."""

    # Rust owns the closed schema, canonical ordering requirements, and the
    # equality check against the coordinator-signed CompileOptions payload.
    records: list[dict[str, object]] = []
    for workload_id, resource, spec in entries:
        match resource:
            case WorkloadRuntimeBinding() as binding:
                specialization_key = binding.specialization_key.items()
                implementation_ids = binding.backend.binders
            case NoWorkloadRuntimeBinding():
                specialization_key = ()
                implementation_ids = ()
        records.append(
            {
                "workload_id": workload_id,
                "factory": spec.factory,
                "config": dict(spec.config),
                "specialization_key": [
                    [name, value]
                    for name, value in specialization_key
                ],
                "backend_region_implementation_ids": sorted(
                    implementation_ids
                ),
            }
        )
    records.sort(key=lambda record: (record["workload_id"], record["factory"]))
    return json.dumps(
        records,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


class NativeWorkload(ABC):
    """Nominal marker for declarations that own complete native resources."""

    @abstractmethod
    def native_workload_resources(self) -> WorkloadRuntimeBinding:
        """Return immutable bindings without inspecting task bodies at runtime."""


def residual_resource_manifest_from_paths(
    paths: Mapping["ResidualLocator", str],
) -> tuple["ResidualResourceBinding", ...]:
    """Mechanically encode INDEX hidden locators as signed resource bindings."""

    from tllm.schedule import HiddenPhase, HiddenSite, ResidualResourceBinding

    entries: list[ResidualResourceBinding] = []
    for locator, module_path in paths.items():
        entries.append(
            ResidualResourceBinding(
                layer_index=locator.layer,
                site=HiddenSite(locator.site),
                phase=HiddenPhase(locator.phase),
                module_path=module_path,
            )
        )
    return tuple(sorted(entries, key=lambda entry: entry.locator_key))


def _residual_resource_manifest_wire(
    resources: tuple[WorkloadRuntimeBinding, ...],
) -> str:
    """Transport worker resource identities without deciding their validity."""

    entries = tuple(
        entry
        for item in resources
        for entry in residual_resource_manifest_from_paths(item.residual_paths)
    )
    # Sorting is serialization canonicalization only.  Rust still owns the
    # coverage and duplicate-locator checks against the signed graph.
    return json.dumps(
        [
            {
                "layer_index": entry.layer_index,
                "site": entry.site.value,
                "phase": entry.phase.value,
                "module_path": entry.module_path,
            }
            for entry in sorted(entries, key=lambda entry: entry.locator_key)
        ],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _resolve_factory(path: str) -> Callable[[Mapping[str, object]], WorkloadRuntimeBinding]:
    module_name, qualname = path.split(":", 1)
    value: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"workload worker factory {path!r} is not callable")
    return cast(
        Callable[[Mapping[str, object]], WorkloadRuntimeBinding],
        value,
    )


def reconstruct_worker_bindings(
    compilation: "WorkloadCompilation",
    specs: tuple[WorkloadWorkerSpec, ...],
    *,
    specialization: "RuntimeSpecializationSpec | None" = None,
) -> "WorkloadExecutionBindings":
    """Rebuild local resources; Rust proves the signed graph and facts."""

    from tllm.runtime.workload_execution import (
        CompositeExecutionLifecycleBinding,
        ExecutionLifecycleBinding,
        NoExecutionLifecycle,
        StateVersionBinding,
        WorkloadExecutionBindings,
    )
    from tllm.workload import StateScope, declare_workload
    from tllm.workload.lowering import lower_schedule_graph

    if not specs:
        raise ValueError("native worker reconstruction requires at least one spec")
    from tllm import _core

    plan_epoch = 1 if compilation.physical is None else compilation.physical.plan_epoch
    # Validate the complete signed compilation and the exact factory/config
    # recipes before importing or invoking any worker-local factory.
    _core.validate_worker_reconstruction_specs(
        compilation.cache_key,
        compilation.graph.to_json(),
        compilation.timeline.to_json(),
        compilation.options.to_json(),
        compilation.decision.to_json(),
        plan_epoch,
        json.dumps(
            [spec.to_wire() for spec in specs],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
    )
    reconstructed: list[WorkloadRuntimeBinding] = []
    for spec in specs:
        reconstructed.append(_resolve_factory(spec.factory)(spec.config))
    declarations = tuple(
        declare_workload(item.owner) for item in reconstructed
    )
    # Replica state advertises a snapshot before specialization, but a pure
    # request-state workload cannot: its typed service (and therefore its
    # snapshots) is created by cold specialization itself.  Use the already
    # reconstructed declaration to recognize that generic capability rather
    # than teaching this worker path about any individual workload.
    cold_specializers = tuple(
        services.cold_specialize
        for declaration, item in zip(declarations, reconstructed, strict=True)
        if isinstance(services := item.services, LocalRuntimeServices)
        and (
            item.states.snapshots
            or item.request_state_services
            or item.backend.binders
            or any(
                state.scope is StateScope.REQUEST
                for state in declaration.ir.states
            )
        )
    )
    if cold_specializers and specialization is None:
        raise RuntimeError(
            "native worker reconstruction requires cold specialization facts"
        )
    expected_graph = lower_schedule_graph(
        tuple(declaration.ir for declaration in declarations),
        persistent_state_bytes=compilation.graph.persistent_state_bytes,
    )
    residual_resource_manifest_json = _residual_resource_manifest_wire(
        tuple(reconstructed)
    )
    packed_hidden_resources: tuple["PackedHiddenResourceBinding", ...] = tuple(
        sorted(
            (
                binding
                for item in reconstructed
                for binding in item.packed_hidden_resources
            ),
            key=lambda binding: binding.locator_key,
        )
    )
    static_binding_manifest_json = static_backend_binding_manifest_wire(
        tuple(
            (declaration.ir.workload_id, resource, spec)
            for declaration, resource, spec in zip(
                declarations, reconstructed, specs, strict=True
            )
        )
    )
    if specialization is None:
        specialization_facts_json = "null"
    else:
        specialization_facts_json = json.dumps(
            {
                "device": str(specialization.device),
                "dtype": str(specialization.dtype),
                "hidden_size": specialization.hidden_size,
                "rows": specialization.rows,
                **(
                    {"max_prefill_tokens": specialization.max_prefill_tokens}
                    if specialization.max_prefill_tokens is not None
                    else {}
                ),
                **(
                    {"model_layer_count": specialization.model_layer_count}
                    if specialization.model_layer_count is not None
                    else {}
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    # This is deliberately before callback/resource binding.  Rust validates
    # the original signed wire, re-runs graph derivation for the local graph,
    # and owns all signed specialization-fact comparisons.
    _core.validate_worker_reconstruction_wire(
        compilation.cache_key,
        compilation.graph.to_json(),
        compilation.timeline.to_json(),
        compilation.options.to_json(),
        compilation.decision.to_json(),
        1 if compilation.physical is None else compilation.physical.plan_epoch,
        expected_graph.to_json(),
        specialization_facts_json,
        residual_resource_manifest_json,
        json.dumps(
            [
                {
                    "layer": binding.layer,
                    "layer_kind": binding.layer_kind.value,
                    "site": binding.site.value,
                    "module_path": binding.module_path,
                }
                for binding in packed_hidden_resources
            ],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ),
        static_binding_manifest_json,
    )
    if specialization is not None:
        for specialize in cold_specializers:
            specialize(specialization)
        # A request-state service is allocated by cold specialization, so its
        # pre-specialization binding cannot carry the service itself. Refresh
        # every owner binding only after the signed graph/facts gates above
        # have accepted the reconstruction; this is still local resource
        # binding, never a compiler decision.
        reconstructed = [
            item.owner.native_workload_resources() for item in reconstructed
        ]

    def merge_inputs() -> dict[str, RuntimeInputBinding[object]]:
        merged: dict[str, RuntimeInputBinding[object]] = {}
        for item in reconstructed:
            for name, callback in item.ports.inputs.items():
                if name in merged:
                    raise RuntimeError(
                        f"worker reconstruction has duplicate input binding {name!r}"
                    )
                merged[name] = callback
        return merged

    def merge_outputs() -> dict[str, RuntimeOutputBinding[object]]:
        merged: dict[str, RuntimeOutputBinding[object]] = {}
        for item in reconstructed:
            for name, callback in item.ports.outputs.items():
                if name in merged:
                    raise RuntimeError(
                        f"worker reconstruction has duplicate output binding {name!r}"
                    )
                merged[name] = callback
        return merged

    multiple = len(declarations) > 1
    schedule_states = {state.resource: state for state in compilation.graph.states}
    state_versions: dict[str, StateVersionBinding] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        for local_name, snapshot in item.states.snapshots.items():
            resource = (
                f"{declaration.ir.workload_id}/{local_name}"
                if multiple
                else local_name
            )
            if resource in state_versions:
                raise RuntimeError(
                    f"worker reconstruction has duplicate state binding {resource!r}"
                )
            try:
                state = schedule_states[resource]
            except KeyError as exc:
                raise RuntimeError(
                    f"worker reconstruction references unknown state {resource!r}"
                ) from exc
            state_versions[resource] = StateVersionBinding(state, snapshot)
    request_state_services: dict[
        str, RequestStateRuntimeService["torch.nn.Module"]
    ] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        for local_name, service in item.request_state_services.items():
            resource = (
                f"{declaration.ir.workload_id}/{local_name}"
                if multiple
                else local_name
            )
            if resource in request_state_services:
                raise RuntimeError(
                    "worker reconstruction has duplicate request-state service "
                    f"binding {resource!r}"
                )
            try:
                state = schedule_states[resource]
            except KeyError as exc:
                raise RuntimeError(
                    "worker reconstruction references unknown request state "
                    f"{resource!r}"
                ) from exc
            if state.scope != "request":
                raise RuntimeError(
                    "worker reconstruction request-state service is not bound to "
                    f"a request resource {resource!r}"
                )
            request_state_services[resource] = service.for_signed_resource(resource)
    sampler_bindings: list[SamplerRuntimeBinding] = []
    for item in reconstructed:
        match item.sampler:
            case SamplerRuntimeBinding() as sampler:
                sampler_bindings.append(sampler)
            case NoSamplerBinding():
                pass
    if len(sampler_bindings) > 1:
        raise RuntimeError("native worker reconstruction has multiple sampler providers")
    residual_paths: dict["ResidualLocator", str] = {}
    for item in reconstructed:
        overlap = set(residual_paths).intersection(item.residual_paths)
        if overlap:
            raise RuntimeError(
                "native worker reconstruction has duplicate residual path bindings"
            )
        residual_paths.update(item.residual_paths)
    backend_region_binders: dict[str, "BackendRegionBinderSlot"] = {}
    from tllm.runtime.parameter_overlay import OverlayChannel, OverlayVersionStore

    parameter_overlays: dict[OverlayChannel, OverlayVersionStore] = {}
    optimizer_transitions: dict[str, OptimizerTransitionService] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        overlap = set(backend_region_binders).intersection(item.backend.binders)
        if overlap:
            raise RuntimeError(
                "native worker reconstruction has duplicate backend Region binder "
                "binding(s): "
                + ", ".join(sorted(overlap))
            )
        backend_region_binders.update(item.backend.binders)
        overlay_overlap = set(parameter_overlays).intersection(
            item.parameter_overlays
        )
        if overlay_overlap:
            raise RuntimeError(
                "native worker reconstruction has duplicate parameter overlay channels"
            )
        parameter_overlays.update(item.parameter_overlays)
        for local_name, transition_service in item.optimizer_transitions.items():
            resource = (
                f"{declaration.ir.workload_id}/{local_name}"
                if multiple
                else local_name
            )
            if resource in optimizer_transitions:
                raise RuntimeError(
                    "native worker reconstruction has duplicate optimizer "
                    f"transition binding {resource!r}"
                )
            optimizer_transitions[resource] = transition_service

    diagnostics_readers: dict[str, Callable[[], Mapping[str, int | float]]] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        match item.services:
            case LocalRuntimeServices(diagnostics=reader) | DiagnosticsRuntimeServices(
                diagnostics=reader
            ):
                pass
            case NoRuntimeServices():
                continue
        workload_id = declaration.ir.workload_id
        if workload_id in diagnostics_readers:
            raise RuntimeError(
                "worker reconstruction has duplicate diagnostics workload "
                f"binding {workload_id!r}"
            )
        diagnostics_readers[workload_id] = reader

    lifecycle_service_list: list[LocalRuntimeServices] = []
    for item in reconstructed:
        match item.services:
            case LocalRuntimeServices() as services:
                lifecycle_service_list.append(services)
            case DiagnosticsRuntimeServices() | NoRuntimeServices():
                pass
    lifecycle_services = tuple(lifecycle_service_list)

    return WorkloadExecutionBindings(
        declarations=declarations,
        ports=RuntimePortRegistry(
            inputs=merge_inputs(),
            outputs=merge_outputs(),
        ),
        state_versions=state_versions,
        request_state_services=request_state_services,
        optimizer_transitions=optimizer_transitions,
        sampler=(NoSamplerBinding() if not sampler_bindings else sampler_bindings[0]),
        diagnostics_readers=diagnostics_readers,
        lifecycle=(
            NoExecutionLifecycle()
            if not lifecycle_services
            else (
                ExecutionLifecycleBinding(
                    stats=lifecycle_services[0].stats,
                    is_enabled=lifecycle_services[0].is_enabled,
                    synchronize=lifecycle_services[0].synchronize,
                    per_request_stats=lifecycle_services[0].per_request_stats,
                )
                if len(lifecycle_services) == 1
                else CompositeExecutionLifecycleBinding(
                    tuple(
                        ExecutionLifecycleBinding(
                            stats=service.stats,
                            is_enabled=service.is_enabled,
                            synchronize=service.synchronize,
                            per_request_stats=service.per_request_stats,
                        )
                        for service in lifecycle_services
                    )
                )
            )
        ),
        residual_paths=residual_paths,
        packed_hidden_resources=packed_hidden_resources,
        backend_region_binders=backend_region_binders,
        parameter_overlays=parameter_overlays,
        static_binding_entries=tuple(
            (declaration.ir.workload_id, resource, spec)
            for declaration, resource, spec in zip(
                declarations, reconstructed, specs, strict=True
            )
        ),
    )


__all__ = [
    "NativeWorkload",
    "NoWorkloadRuntimeBinding",
    "StaticBackendBindingEntry",
    "WorkloadWorkerSpec",
    "residual_resource_manifest_from_paths",
    "static_backend_binding_manifest_wire",
    "reconstruct_worker_bindings",
]
