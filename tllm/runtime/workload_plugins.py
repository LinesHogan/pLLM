"""Spawn-safe reconstruction of process-local workload Region bindings."""

from __future__ import annotations

import importlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
    from tllm.contracts.consumer_stats import TrainingStats
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.specialization import RuntimeSpecializationSpec
    from tllm.runtime.workload_execution import WorkloadExecutionBindings
    from tllm.schedule import ResidualResourceBinding, WorkloadCompilation


@dataclass(frozen=True, slots=True)
class WorkloadWorkerSpec:
    """Pure-data recipe for rebuilding one workload inside a worker."""

    factory: str
    config: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.factory, str) or ":" not in self.factory:
            raise ValueError("WorkloadWorkerSpec.factory must be 'module:qualname'")
        module_name, qualname = self.factory.split(":", 1)
        if not module_name or not qualname or any(
            not part or part == "<locals>" for part in qualname.split(".")
        ):
            raise ValueError("WorkloadWorkerSpec.factory must be importable")
        if not isinstance(self.config, Mapping):
            raise TypeError("WorkloadWorkerSpec.config must be a mapping")
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
        decoded = json.loads(encoded)
        if not isinstance(decoded, dict):
            raise TypeError("WorkloadWorkerSpec.config must encode a JSON object")
        object.__setattr__(self, "config", MappingProxyType(decoded))

    def to_wire(self) -> dict[str, object]:
        return {"factory": self.factory, "config": dict(self.config)}

    @classmethod
    def from_wire(cls, raw: Mapping[str, object]) -> "WorkloadWorkerSpec":
        if type(raw) is not dict or set(raw) != {"factory", "config"}:
            raise ValueError(
                "WorkloadWorkerSpec wire must contain exactly factory and config"
            )
        factory = raw["factory"]
        config = raw["config"]
        if type(factory) is not str:
            raise TypeError("WorkloadWorkerSpec wire factory must be a string")
        if type(config) is not dict:
            raise TypeError("WorkloadWorkerSpec wire config must be a JSON object")
        return cls(factory=factory, config=config)


@dataclass(frozen=True, slots=True)
class NativeWorkloadResources:
    """Explicit process-local resources for one native workload declaration."""

    workload: object
    provided_inputs: Mapping[str, Callable[[int], object]] = field(default_factory=dict)
    output_sinks: Mapping[str, Callable[[object, int], None]] = field(default_factory=dict)
    state_snapshots: Mapping[
        str, Callable[[int], Mapping[Hashable, int]]
    ] = field(default_factory=dict)
    cold_specializer: Callable[["RuntimeSpecializationSpec"], None] | None = None
    sampler_provider: "SamplerModifierProvider | None" = None
    diagnostics_reader: Callable[[], Mapping[str, int | float]] | None = None
    stats_reader: Callable[[bool], "TrainingStats"] | None = None
    per_request_stats_reader: Callable[[bool], Mapping[int, "TrainingStats"]] | None = None
    enabled_setter: Callable[[bool], None] | None = None
    enabled_predicate: Callable[[], bool] | None = None
    synchronizer: Callable[[], None] | None = None
    specialization_key: Mapping[str, str] = field(default_factory=dict)
    residual_paths: Mapping[object, str] = field(default_factory=dict)
    backend_region_binders: Mapping[str, "BackendRegionBinderSlot"] = field(
        default_factory=dict
    )
    worker_spec: WorkloadWorkerSpec | None = None

    def __post_init__(self) -> None:
        for label, bindings in (
            ("provided_inputs", self.provided_inputs),
            ("output_sinks", self.output_sinks),
            ("state_snapshots", self.state_snapshots),
        ):
            if not isinstance(bindings, Mapping) or any(
                not isinstance(name, str) or not name or not callable(callback)
                for name, callback in bindings.items()
            ):
                raise TypeError(f"{label} must map non-empty names to callbacks")
            object.__setattr__(self, label, MappingProxyType(dict(bindings)))
        for label in (
            "cold_specializer",
            "diagnostics_reader",
            "stats_reader",
            "per_request_stats_reader",
            "enabled_setter",
            "enabled_predicate",
            "synchronizer",
        ):
            callback = getattr(self, label)
            if callback is not None and not callable(callback):
                raise TypeError(f"{label} must be callable or None")
        provider = self.sampler_provider
        if provider is not None:
            from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider

            if not isinstance(provider, SamplerModifierProvider):
                raise TypeError(
                    "sampler_provider must implement SamplerModifierProvider"
                )
        if not isinstance(self.specialization_key, Mapping) or any(
            not isinstance(name, str)
            or not name
            or not isinstance(value, str)
            or not value
            for name, value in self.specialization_key.items()
        ):
            raise TypeError("specialization_key must map non-empty str keys to values")
        object.__setattr__(
            self,
            "specialization_key",
            MappingProxyType(dict(sorted(self.specialization_key.items()))),
        )
        from tllm.ports.residual_stream import ResidualLocator

        if not isinstance(self.residual_paths, Mapping) or any(
            not isinstance(locator, ResidualLocator)
            or not isinstance(path, str)
            or not path
            for locator, path in self.residual_paths.items()
        ):
            raise TypeError(
                "residual_paths must map ResidualLocator values to non-empty paths"
            )
        object.__setattr__(
            self, "residual_paths", MappingProxyType(dict(self.residual_paths))
        )
        from tllm.runtime.backend_region_binders import BackendRegionBinderSlot

        if not isinstance(self.backend_region_binders, Mapping) or any(
            not isinstance(implementation_id, str)
            or not implementation_id
            or not isinstance(slot, BackendRegionBinderSlot)
            or slot.implementation_id != implementation_id
            for implementation_id, slot in self.backend_region_binders.items()
        ):
            raise TypeError(
                "backend_region_binders must map implementation IDs to matching "
                "BackendRegionBinderSlot values"
            )
        object.__setattr__(
            self,
            "backend_region_binders",
            MappingProxyType(dict(self.backend_region_binders)),
        )
        if self.worker_spec is not None and not isinstance(
            self.worker_spec, WorkloadWorkerSpec
        ):
            raise TypeError("worker_spec must be WorkloadWorkerSpec or None")


# Historical name retained at the worker-reconstruction API boundary.
ReconstructedWorkload = NativeWorkloadResources


def static_backend_binding_manifest_wire(
    entries: tuple[object, ...],
) -> str:
    """Transport live static backend semantics for Rust comparison."""

    # Rust owns the closed schema, canonical ordering requirements, and the
    # equality check against the coordinator-signed CompileOptions payload.
    records: list[dict[str, object]] = []
    for entry in entries:
        if not isinstance(entry, tuple) or len(entry) != 3:
            raise TypeError("static backend binding entries must be triples")
        workload_id, resource, spec = entry
        if not isinstance(workload_id, str):
            raise TypeError("static backend binding workload ID must be a string")
        if resource is not None and not isinstance(resource, NativeWorkloadResources):
            raise TypeError(
                "static backend binding resource must be NativeWorkloadResources or None"
            )
        if not isinstance(spec, WorkloadWorkerSpec):
            raise TypeError("static backend binding spec must be WorkloadWorkerSpec")
        records.append(
            {
                "workload_id": workload_id,
                "factory": spec.factory,
                "config": dict(spec.config),
                "specialization_key": [
                    [name, value]
                    for name, value in (
                        () if resource is None else resource.specialization_key.items()
                    )
                ],
                "backend_region_implementation_ids": sorted(
                    () if resource is None else resource.backend_region_binders
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
    def native_workload_resources(self) -> NativeWorkloadResources:
        """Return immutable bindings without inspecting task bodies at runtime."""


def residual_resource_manifest_from_paths(
    paths: Mapping[object, str],
) -> tuple["ResidualResourceBinding", ...]:
    """Mechanically encode INDEX hidden locators as signed resource bindings."""

    from tllm.ports.residual_stream import ResidualLocator
    from tllm.schedule import HiddenPhase, HiddenSite, ResidualResourceBinding

    if not isinstance(paths, Mapping):
        raise TypeError("residual resource paths must be a mapping")
    entries: list[ResidualResourceBinding] = []
    for locator, module_path in paths.items():
        if not isinstance(locator, ResidualLocator):
            raise TypeError("residual resource paths must use ResidualLocator keys")
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
    resources: tuple[NativeWorkloadResources, ...],
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


def _resolve_factory(path: str) -> Callable[[Mapping[str, object]], ReconstructedWorkload]:
    module_name, qualname = path.split(":", 1)
    value: object = importlib.import_module(module_name)
    for part in qualname.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError(f"workload worker factory {path!r} is not callable")
    return cast(
        Callable[[Mapping[str, object]], ReconstructedWorkload],
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
        StateVersionBinding,
        WorkloadExecutionBindings,
    )
    from tllm.workload import declare_workload
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
    reconstructed: list[NativeWorkloadResources] = []
    for spec in specs:
        result = _resolve_factory(spec.factory)(spec.config)
        if not isinstance(result, NativeWorkloadResources):
            raise TypeError(
                f"workload worker factory {spec.factory!r} must return NativeWorkloadResources"
            )
        reconstructed.append(result)
    cold_specializers = tuple(
        item.cold_specializer
        for item in reconstructed
        if item.cold_specializer is not None
    )
    if cold_specializers and specialization is None:
        raise RuntimeError(
            "native worker reconstruction requires cold specialization facts"
        )
    declarations = tuple(
        declare_workload(item.workload) for item in reconstructed
    )
    expected_graph = lower_schedule_graph(
        tuple(declaration.ir for declaration in declarations),
        persistent_state_bytes=compilation.graph.persistent_state_bytes,
    )
    residual_resource_manifest_json = _residual_resource_manifest_wire(
        tuple(reconstructed)
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
        static_binding_manifest_json,
    )
    if specialization is not None:
        for specialize in cold_specializers:
            specialize(specialization)

    def merge_callbacks(attribute: str) -> dict[str, Callable[..., object]]:
        merged: dict[str, Callable[..., object]] = {}
        for item in reconstructed:
            for name, callback in getattr(item, attribute).items():
                if name in merged:
                    raise RuntimeError(
                        f"worker reconstruction has duplicate {attribute} binding {name!r}"
                    )
                merged[name] = callback
        return merged

    multiple = len(declarations) > 1
    schedule_states = {state.resource: state for state in compilation.graph.states}
    state_versions: dict[str, StateVersionBinding] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        for local_name, snapshot in item.state_snapshots.items():
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
    providers = tuple(
        item.sampler_provider
        for item in reconstructed
        if item.sampler_provider is not None
    )
    if len(providers) > 1:
        raise RuntimeError("native worker reconstruction has multiple sampler providers")
    residual_paths: dict[object, str] = {}
    for item in reconstructed:
        overlap = set(residual_paths).intersection(item.residual_paths)
        if overlap:
            raise RuntimeError(
                "native worker reconstruction has duplicate residual path bindings"
            )
        residual_paths.update(item.residual_paths)
    backend_region_binders: dict[str, "BackendRegionBinderSlot"] = {}
    for item in reconstructed:
        overlap = set(backend_region_binders).intersection(item.backend_region_binders)
        if overlap:
            raise RuntimeError(
                "native worker reconstruction has duplicate backend Region binder "
                "binding(s): "
                + ", ".join(sorted(overlap))
            )
        backend_region_binders.update(item.backend_region_binders)

    diagnostics_readers: dict[str, Callable[[], Mapping[str, int | float]]] = {}
    for declaration, item in zip(declarations, reconstructed, strict=True):
        reader = item.diagnostics_reader
        if reader is None:
            continue
        workload_id = declaration.ir.workload_id
        if workload_id in diagnostics_readers:
            raise RuntimeError(
                "worker reconstruction has duplicate diagnostics workload "
                f"binding {workload_id!r}"
            )
        diagnostics_readers[workload_id] = reader

    def one_callback(attribute: str) -> Callable[..., object] | None:
        callbacks = tuple(
            callback
            for item in reconstructed
            if (callback := getattr(item, attribute)) is not None
        )
        if len(callbacks) > 1:
            raise RuntimeError(
                f"native worker reconstruction has multiple {attribute} callbacks"
            )
        return None if not callbacks else callbacks[0]

    return WorkloadExecutionBindings(
        declarations=declarations,
        provided_inputs=cast(
            Mapping[str, Callable[[int], object]],
            merge_callbacks("provided_inputs"),
        ),
        output_sinks=cast(
            Mapping[str, Callable[[object, int], None]],
            merge_callbacks("output_sinks"),
        ),
        state_versions=state_versions,
        sampler_provider=None if not providers else providers[0],
        diagnostics_readers=diagnostics_readers,
        stats_reader=cast(
            Callable[[bool], "TrainingStats"] | None,
            one_callback("stats_reader"),
        ),
        per_request_stats_reader=cast(
            Callable[[bool], Mapping[int, "TrainingStats"]] | None,
            one_callback("per_request_stats_reader"),
        ),
        enabled_setter=cast(
            Callable[[bool], None] | None,
            one_callback("enabled_setter"),
        ),
        enabled_predicate=cast(
            Callable[[], bool] | None,
            one_callback("enabled_predicate"),
        ),
        synchronizer=cast(
            Callable[[], None] | None,
            one_callback("synchronizer"),
        ),
        residual_paths=residual_paths,
        backend_region_binders=backend_region_binders,
        static_binding_entries=tuple(
            (declaration.ir.workload_id, resource, spec)
            for declaration, resource, spec in zip(
                declarations, reconstructed, specs, strict=True
            )
        ),
    )


__all__ = [
    "NativeWorkload",
    "NativeWorkloadResources",
    "ReconstructedWorkload",
    "WorkloadWorkerSpec",
    "residual_resource_manifest_from_paths",
    "static_backend_binding_manifest_wire",
    "reconstruct_worker_bindings",
]
