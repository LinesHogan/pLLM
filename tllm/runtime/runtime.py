#!/usr/bin/env python3
"""Instance-scoped tLLM runtime.

``TllmRuntime`` is the recommended entry point:

    runtime = TllmRuntime(workloads=[EsampConsumer(...)])
    llm = runtime.make_llm(model_name=..., dtype=..., ...)

Module-level ``tllm.make_llm`` / ``tllm.register_consumer`` remain as a thin
façade over a process-wide default runtime for backward compatibility.

Current limitation (tracked by the runtime-instancing migration): backend
hooks are installed on engine classes once per process, and the active
runtime's state is what those hooks see. Creating two engines with
*different* runtimes in one process is not supported yet; the runner ->
runtime routing table (:func:`bind_runner_runtime`) is the seam where
per-runner dispatch will land.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Mapping
from threading import RLock
from typing import TYPE_CHECKING, Protocol, Sequence, TypeVar, cast
import warnings
from uuid import uuid4

from tllm.runtime.declaration_adapter import (
    consumer_spec,
    legacy_consumer,
    register_legacy_consumer,
    workload_definition,
)
from tllm.runtime.workload_bindings import (
    DiagnosticsRuntimeServices,
    LocalRuntimeServices,
    NoPerRequestStats,
    NoRuntimeServices,
    NoSamplerBinding,
    NoWorkerBinding,
    PerRequestStats,
    SamplerRuntimeBinding,
    WorkerRuntimeBinding,
)
from tllm.runtime.vllm_adapter import (
    VllmHandle,
    bind_inprocess_runner,
    bind_runner_owner,
    create_llm,
    install_generate_poll,
)
from tllm.runtime.torch_adapter import runtime_torch_values

if TYPE_CHECKING:
    import torch
    from tllm.contracts.consumer_stats import TrainingStats
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
    from tllm.runtime.mp.control import ControlPlane
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.workload_execution import WorkloadExecutionBindings
    from tllm.runtime.specialization import RuntimeSpecializationSpec
    from tllm.runtime.workload_plugins import (
        NativeWorkloadResources,
        WorkloadWorkerSpec,
    )
    from tllm.workload.staging import StagedPort
    from tllm.workload.frontend import PublishedModule
    from tllm.workload.host import HostOutput
    from tllm.schedule import (
        CompileOptions,
        MainWorkloadTimeline,
        WorkloadCompilation,
        WorkloadCompilerCoordinator,
    )


# The runtime the process-wide class hooks currently dispatch to.
_active_runtime: "TllmRuntime | None" = None
StagedT = TypeVar("StagedT")
HostOutputT = TypeVar("HostOutputT")
RuntimeWorkloadT = TypeVar("RuntimeWorkloadT")
PublishedModuleT = TypeVar("PublishedModuleT", bound="torch.nn.Module")


class _StagedWorkload(Protocol[StagedT]):
    def stage_port(
        self, port: "StagedPort[StagedT]", value: StagedT
    ) -> int: ...

    def prepare_port(self, key: str, transaction: str, value: object) -> None: ...

    def commit_port(self, key: str, transaction: str) -> int: ...

    def abort_port(self, key: str, transaction: str) -> None: ...


class _HostOutputWorkload(Protocol[HostOutputT]):
    def drain_port(
        self, port: "HostOutput[HostOutputT]"
    ) -> tuple[HostOutputT, ...]: ...


def bind_runner_runtime(runner: object, runtime: "TllmRuntime") -> None:
    bind_runner_owner(runner, runtime)



def active_runtime() -> "TllmRuntime":
    if _active_runtime is None:
        return default_runtime()
    return _active_runtime


def _runtime_specialization_spec_from_wire(
    raw: Mapping[str, object],
) -> "RuntimeSpecializationSpec":
    """Map Rust-validated wire facts to the two concrete Torch value types."""

    from tllm.runtime.specialization import RuntimeSpecializationSpec

    required = {"device", "dtype", "hidden_size", "rows"}
    optional = {"model_layer_count", "max_prefill_tokens"}
    keys = set(raw)
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise RuntimeError("worker returned an invalid runtime specialization fact set")
    device_wire = raw["device"]
    dtype_wire = raw["dtype"]
    if type(device_wire) is not str or type(dtype_wire) is not str:
        raise RuntimeError("worker runtime specialization device/dtype must be strings")
    numeric_values = {
        name: raw[name]
        for name in ("rows", "hidden_size", "model_layer_count", "max_prefill_tokens")
        if name in raw
    }
    if any(type(value) is not int for value in numeric_values.values()):
        raise RuntimeError("worker runtime specialization numeric facts must be exact ints")
    device, dtype = runtime_torch_values(device_wire, dtype_wire)
    return RuntimeSpecializationSpec(
        rows=cast(int, numeric_values["rows"]),
        hidden_size=cast(int, numeric_values["hidden_size"]),
        dtype=dtype,
        device=device,
        model_layer_count=cast("int | None", numeric_values.get("model_layer_count")),
        max_prefill_tokens=cast("int | None", numeric_values.get("max_prefill_tokens")),
    )


class TllmRuntime:
    def local_workload(
        self, workload_type: type[RuntimeWorkloadT]
    ) -> RuntimeWorkloadT:
        """Return the process-local instance that executes this workload type."""

        self._require_workload_live()
        from tllm.runtime import residual_runtime

        compilation = self._workload_compilation
        expected_epoch = (
            compilation.physical.plan_epoch
            if compilation is not None and compilation.physical is not None
            else 1
        )
        store = residual_runtime.RUNTIME.workload_execution_store
        if (
            _active_runtime is self
            and store is not None
            and compilation is not None
        ):
            candidate = store.next_execution
            if (
                candidate.compilation.decision.signature
                == compilation.decision.signature
                and candidate.plan_epoch == expected_epoch
            ):
                return candidate.workload_instance(workload_type)
        execution = residual_runtime.RUNTIME.workload_execution
        if (
            _active_runtime is self
            and execution is not None
            and compilation is not None
            and execution.compilation.decision.signature
            == compilation.decision.signature
            and execution.plan_epoch == expected_epoch
        ):
            return execution.workload_instance(workload_type)
        try:
            workload = self._workloads_by_type[workload_type]
        except KeyError as exc:
            raise LookupError(workload_type.__qualname__) from exc
        return cast(RuntimeWorkloadT, workload)

    def _current_local_workload(
        self, workload_type: type[RuntimeWorkloadT]
    ) -> RuntimeWorkloadT:
        from tllm.runtime import residual_runtime

        store = residual_runtime.RUNTIME.workload_execution_store
        if _active_runtime is self and store is not None:
            return store.current.workload_instance(workload_type)
        execution = residual_runtime.RUNTIME.workload_execution
        if _active_runtime is self and execution is not None:
            return execution.workload_instance(workload_type)
        try:
            workload = self._workloads_by_type[workload_type]
        except KeyError as exc:
            raise LookupError(workload_type.__qualname__) from exc
        return cast(RuntimeWorkloadT, workload)

    def stage(self, port: "StagedPort[StagedT]", value: StagedT) -> int:
        """Stage one typed payload for the next signed workload step."""

        self._require_workload_live()
        workload = cast(
            "_StagedWorkload[StagedT]", self.local_workload(port.owner)
        )
        from tllm.runtime.mp.control import active_control_plane

        plane = active_control_plane()
        if plane is None:
            return workload.stage_port(port, value)

        from tllm.runtime.mp.port_wire import encode_port_payload

        self._port_transaction_sequence += 1
        transaction = (
            f"{self._port_transaction_owner}:{self._port_transaction_sequence}"
        )
        quorum = plane.worker_quorum()
        encoded = encode_port_payload(value)
        local_prepared = False
        remote_started = False
        sequence_mismatch = False
        try:
            workload.prepare_port(port.key, transaction, value)
            local_prepared = True
            remote_started = True
            plane.prepare_workload_port(
                port.key,
                transaction,
                encoded.wire,
                quorum=quorum,
            )
            remote_commits = plane.commit_workload_port(
                port.key,
                transaction,
                quorum=quorum,
            )
            local_sequence = workload.commit_port(port.key, transaction)
            if local_sequence != remote_commits[0].sequence:
                sequence_mismatch = True
                raise RuntimeError(
                    "workload port parent/worker sequence quorum mismatch"
                )
            return local_sequence
        except Exception as exc:
            rollback_failures: list[Exception] = []
            if local_prepared:
                try:
                    workload.abort_port(port.key, transaction)
                except Exception as rollback_error:
                    rollback_failures.append(rollback_error)
            if remote_started:
                try:
                    plane.abort_workload_port(
                        port.key,
                        transaction,
                        quorum=quorum,
                    )
                except Exception as rollback_error:
                    rollback_failures.append(rollback_error)
            if rollback_failures:
                self._fail_stop_workloads(
                    "workload port transaction rollback did not reach its full quorum"
                )
                raise RuntimeError(
                    "workload port publication failed and rollback quorum was not confirmed"
                ) from exc
            if sequence_mismatch:
                self._fail_stop_workloads(
                    "workload port parent and worker transaction histories diverged"
                )
            raise
        finally:
            try:
                encoded.release()
            except Exception as cleanup_error:
                warnings.warn(
                    f"staged port shared-memory cleanup failed: {cleanup_error}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def drain(self, port: "HostOutput[HostOutputT]") -> tuple[HostOutputT, ...]:
        """Drain typed outputs published by signed workload tasks."""

        workload = cast(
            "_HostOutputWorkload[HostOutputT]",
            self._current_local_workload(port.owner),
        )
        return workload.drain_port(port)

    def publish(
        self,
        state: "PublishedModule[PublishedModuleT]",
        values: "Mapping[str, torch.Tensor]",
    ) -> int:
        """Publish a complete typed State payload for the next safe point."""

        return self.stage(state.port, values)

    def __init__(
        self,
        *,
        backend: str = "auto",
        producers: Sequence[object] = (),
        consumers: Sequence[object] = (),
        workloads: Sequence[object] = (),
        workload_memory_budget_bytes: int = 0,
        workload_allow_serial_fallback: bool = True,
        workload_allow_side_stream: bool = False,
        workload_persistent_state_bytes: int = 0,
        workload_decision_cache_directory: str | None = None,
        workload_provided_inputs: Mapping[str, Callable[[int], object]] | None = None,
        workload_output_sinks: Mapping[
            str, Callable[[object, int], None]
        ] | None = None,
        workload_state_snapshots: Mapping[
            str, Callable[[int], Mapping[Hashable, int]]
        ] | None = None,
        workload_worker_specs: Sequence["WorkloadWorkerSpec"] = (),
    ) -> None:
        if backend != "auto":
            raise ValueError(
                f"backend={backend!r}: only 'auto' is supported; backend selection "
                "is structural (tllm.backends)"
            )
        self.producers: tuple[object, ...] = tuple(producers)
        self._workloads: list[object] = list(workloads)
        self._workloads_by_type: dict[type[object], object] = {
            type(workload): workload for workload in self._workloads
        }
        self._consumers: list[object] = list(consumers)
        self._unadapted_workload_ids: list[str] = []
        self._native_workload_resources: dict[int, "NativeWorkloadResources"] = {}
        from tllm.schedule import CompileOptions

        self._workload_compile_options = CompileOptions(
            memory_budget_bytes=workload_memory_budget_bytes,
            allow_serial_fallback=workload_allow_serial_fallback,
            allow_side_stream=workload_allow_side_stream,
        )
        if workload_persistent_state_bytes < 0:
            raise ValueError("workload_persistent_state_bytes must be a non-negative int")
        self._workload_persistent_state_bytes = workload_persistent_state_bytes
        if workload_decision_cache_directory == "":
            raise TypeError(
                "workload_decision_cache_directory must be a non-empty str or None"
            )
        self._workload_decision_cache_directory = workload_decision_cache_directory
        self._workload_provided_inputs = dict(workload_provided_inputs or {})
        self._workload_output_sinks = dict(workload_output_sinks or {})
        self._workload_state_snapshots = dict(workload_state_snapshots or {})
        self._workload_worker_specs: tuple["WorkloadWorkerSpec", ...] = tuple(
            workload_worker_specs
        )
        self._workload_compiler: "WorkloadCompilerCoordinator | None" = None
        from tllm.schedule import PlanEpochAllocator

        self._workload_epoch_allocator = PlanEpochAllocator()
        self._workload_compilation: WorkloadCompilation | None = None
        self._pending_workload_finalizations: dict[
            tuple[str, int], tuple["ControlPlane", "WorkloadCompilation", int]
        ] = {}
        self._workload_fail_stop_reason: str | None = None
        self._port_transaction_owner = uuid4().hex
        self._port_transaction_sequence = 0
        self._shape_specialization_poll_lock = RLock()
        for workload_instance in self._workloads:
            definition = workload_definition(workload_instance)
            consumer = legacy_consumer(workload_instance, definition)
            if consumer is None:
                if definition is None:
                    raise RuntimeError("native workload has no definition")
                self._record_native_resources(workload_instance)
                self._unadapted_workload_ids.append(definition.workload_id)
            elif consumer not in self._consumers:
                self._consumers.append(consumer)
        resource_worker_specs: list["WorkloadWorkerSpec"] = []
        for resources in self._native_workload_resources.values():
            match resources.worker:
                case WorkerRuntimeBinding(spec=spec):
                    resource_worker_specs.append(spec)
                case NoWorkerBinding():
                    pass
        if resource_worker_specs:
            if self._workload_worker_specs:
                raise RuntimeError(
                    "native workload worker specs cannot be supplied both by "
                    "resources and TllmRuntime"
                )
            self._workload_worker_specs = tuple(resource_worker_specs)
        if self._unadapted_workload_ids and self._consumers:
            raise RuntimeError(
                "native Region authority cannot be coinstalled with legacy "
                "dispatch consumers"
            )
        self._refresh_resource_specialization_options()
        self._rebuild_workload_compiler()

    def _refresh_resource_specialization_options(self) -> None:
        from dataclasses import replace
        from tllm.runtime.workload_plugins import (
            residual_resource_manifest_from_paths,
            static_backend_binding_manifest_wire,
        )

        entries: list[tuple[str, str]] = []
        native_resource_entries: list[
            tuple[str, "NativeWorkloadResources | None"]
        ] = []
        residual_paths: dict["ResidualLocator", str] = {}
        for workload_instance in self._workloads:
            if not self._is_native_workload_declaration(workload_instance):
                continue
            resources = self._native_workload_resources.get(id(workload_instance))
            definition = workload_definition(workload_instance)
            if definition is None:
                continue
            workload_id = definition.workload_id
            if resources is not None:
                entries.extend(
                    (f"{workload_id}/{name}", value)
                    for name, value in resources.specialization_key.items()
                )
            native_resource_entries.append((workload_id, resources))
            if resources is None:
                continue
            overlap = set(residual_paths).intersection(resources.residual_paths)
            if overlap:
                raise RuntimeError(
                    "native workloads declare duplicate residual locator bindings"
                )
            residual_paths.update(resources.residual_paths)
        runtime_entries = (
            (name, value)
            for name, value in self._workload_compile_options.shape_specialization
            if name.startswith("runtime/")
        )
        normalized = tuple(sorted((*entries, *runtime_entries)))
        if len({name for name, _value in normalized}) != len(normalized):
            raise RuntimeError("native workload specialization keys must be unique")
        static_binding_entries: tuple[
            tuple[
                str,
                "NativeWorkloadResources | None",
                "WorkloadWorkerSpec",
            ],
            ...,
        ] = ()
        if self._workload_worker_specs:
            if len(self._workload_worker_specs) != len(native_resource_entries):
                raise RuntimeError(
                    "native workload worker specs must cover every native workload"
                )
            static_binding_entries = tuple(
                (workload_id, resources, spec)
                for (workload_id, resources), spec in zip(
                    native_resource_entries,
                    self._workload_worker_specs,
                    strict=True,
                )
            )
        self._workload_compile_options = replace(
            self._workload_compile_options,
            shape_specialization=normalized,
            residual_resource_manifest=residual_resource_manifest_from_paths(
                residual_paths
            ),
            static_binding_manifest=static_backend_binding_manifest_wire(
                static_binding_entries
            ),
        )

    def _refresh_native_worker_specs(self) -> None:
        """Capture fresh immutable resource snapshots for a new worker epoch."""

        if not self._unadapted_workload_ids:
            return
        specs: list["WorkloadWorkerSpec"] = []
        resource_owned_count = 0
        for workload_instance in self._workloads:
            if id(workload_instance) not in self._native_workload_resources:
                continue
            resource_owned_count += 1
            self._record_native_resources(workload_instance)
            resources = self._native_workload_resources[id(workload_instance)]
            match resources.worker:
                case WorkerRuntimeBinding(spec=spec):
                    specs.append(spec)
                case NoWorkerBinding():
                    # A caller-supplied immutable spec remains authoritative for
                    # workloads which do not publish a resource-owned recipe.
                    return
        if resource_owned_count == 0:
            return
        fresh_specs = tuple(specs)
        if fresh_specs == self._workload_worker_specs:
            return
        self._workload_worker_specs = fresh_specs
        self._refresh_resource_specialization_options()
        self._rebuild_workload_compiler()

    def _record_native_resources(self, workload_instance: object) -> None:
        """Bind nominal native resources once, outside every execution path."""

        from tllm.runtime.workload_plugins import (
            NativeWorkload,
        )

        if not isinstance(workload_instance, NativeWorkload):
            return
        resources = workload_instance.native_workload_resources()
        if resources.owner is not workload_instance:
            raise RuntimeError("native workload resources belong to another instance")
        self._native_workload_resources[id(workload_instance)] = resources

    @staticmethod
    def _is_native_workload_declaration(workload_instance: object) -> bool:
        definition = workload_definition(workload_instance)
        if definition is None:
            return False
        from tllm.workload import WorkloadAuthority

        return definition.authority is WorkloadAuthority.NATIVE_REGION

    @property
    def workloads(self) -> tuple[object, ...]:
        return tuple(self._workloads)

    @property
    def consumers(self) -> tuple[object, ...]:
        return tuple(self._consumers)

    @property
    def workload_compilation(self) -> "WorkloadCompilation | None":
        return self._workload_compilation

    @property
    def workload_compile_options(self) -> "CompileOptions":
        return self._workload_compile_options

    @property
    def workload_execution_bindings(self) -> "WorkloadExecutionBindings | None":
        """Return local callable bindings only for fully native declarations."""

        if not self._unadapted_workload_ids:
            return None
        compiler = self._workload_compiler
        if compiler is None:
            raise RuntimeError("native workload bindings require a compiler")
        declarations = compiler.declarations
        if len(self._unadapted_workload_ids) != len(declarations):
            raise RuntimeError(
                "one compilation cannot mix native Region authority with legacy adapters"
            )
        from tllm.runtime.workload_execution import (
            StateVersionBinding,
            WorkloadExecutionBindings,
        )

        state_by_resource = {state.resource: state for state in compiler.graph.states}
        static_binding_entries: tuple[object, ...] = ()
        if self._workload_worker_specs:
            if len(self._workload_worker_specs) != len(declarations):
                raise RuntimeError(
                    "native workload worker specs must cover every local binding"
                )
            static_binding_entries = tuple(
                (
                    declaration.ir.workload_id,
                    self._native_workload_resources.get(id(declaration.instance)),
                    spec,
                )
                for declaration, spec in zip(
                    declarations,
                    self._workload_worker_specs,
                    strict=True,
                )
            )
        provided_inputs = dict(self._workload_provided_inputs)
        output_sinks = dict(self._workload_output_sinks)
        snapshots = dict(self._workload_state_snapshots)
        sampler_providers: list["SamplerModifierProvider"] = []
        diagnostics_readers: dict[str, Callable[[], Mapping[str, int | float]]] = {}
        residual_paths: dict["ResidualLocator", str] = {}
        backend_region_binders: dict[str, "BackendRegionBinderSlot"] = {}
        stats_readers: list[Callable[[bool], "TrainingStats"]] = []
        per_request_stats_readers: list[
            Callable[[bool], Mapping[int, "TrainingStats"]]
        ] = []
        enabled_setters: list[Callable[[bool], None]] = []
        enabled_predicates: list[Callable[[], bool]] = []
        synchronizers: list[Callable[[], None]] = []
        multiple = len(declarations) > 1
        for declaration in declarations:
            resources = self._native_workload_resources.get(id(declaration.instance))
            if resources is None:
                continue
            provided_overlap = sorted(
                set(provided_inputs).intersection(resources.ports.inputs)
            )
            if provided_overlap:
                raise RuntimeError(
                    "duplicate native workload provided input binding(s): "
                    + ", ".join(provided_overlap)
                )
            provided_inputs.update(resources.ports.inputs)
            sink_overlap = sorted(
                set(output_sinks).intersection(resources.ports.outputs)
            )
            if sink_overlap:
                raise RuntimeError(
                    "duplicate native workload output sink binding(s): "
                    + ", ".join(sink_overlap)
                )
            output_sinks.update(resources.ports.outputs)
            for local_name, snapshot in resources.states.snapshots.items():
                resource = (
                    f"{declaration.ir.workload_id}/{local_name}"
                    if multiple
                    else local_name
                )
                if resource in snapshots:
                    raise RuntimeError(
                        f"duplicate native workload state binding {resource!r}"
                    )
                snapshots[resource] = snapshot
            match resources.sampler:
                case SamplerRuntimeBinding(provider=provider):
                    sampler_providers.append(provider)
                case NoSamplerBinding():
                    pass

            match resources.services:
                case LocalRuntimeServices() as services:
                    diagnostic_callbacks = (services.diagnostics,)
                    stats_readers.append(services.stats)
                    enabled_setters.append(services.set_enabled)
                    enabled_predicates.append(services.is_enabled)
                    synchronizers.append(services.synchronize)
                    match services.per_request_stats:
                        case PerRequestStats(read=read):
                            per_request_stats_readers.append(read)
                        case NoPerRequestStats():
                            pass
                case DiagnosticsRuntimeServices(diagnostics=diagnostics):
                    diagnostic_callbacks = (diagnostics,)
                case NoRuntimeServices():
                    diagnostic_callbacks = ()
            for diagnostics in diagnostic_callbacks:
                workload_id = declaration.ir.workload_id
                if workload_id in diagnostics_readers:
                    raise RuntimeError(
                        "duplicate native workload diagnostics binding "
                        f"{workload_id!r}"
                    )
                diagnostics_readers[workload_id] = diagnostics
            overlap = set(residual_paths).intersection(resources.residual_paths)
            if overlap:
                raise RuntimeError(
                    "native workloads declare duplicate residual locator bindings"
                )
            residual_paths.update(resources.residual_paths)
            binder_overlap = set(backend_region_binders).intersection(
                resources.backend.binders
            )
            if binder_overlap:
                raise RuntimeError(
                    "native workloads declare duplicate backend Region binder "
                    "binding(s): "
                    + ", ".join(sorted(binder_overlap))
                )
            backend_region_binders.update(resources.backend.binders)
        if len(sampler_providers) > 1:
            raise RuntimeError("native Region plan supports at most one sampler provider")
        duplicates = tuple(
            attribute
            for attribute, count in (
                ("stats_reader", len(stats_readers)),
                ("per_request_stats_reader", len(per_request_stats_readers)),
                ("enabled_setter", len(enabled_setters)),
                ("enabled_predicate", len(enabled_predicates)),
                ("synchronizer", len(synchronizers)),
            )
            if count > 1
        )
        if duplicates:
            raise RuntimeError(
                "native Region plan has ambiguous lifecycle callbacks: "
                + ", ".join(duplicates)
            )
        state_versions: dict[str, StateVersionBinding] = {}
        for resource, snapshot in snapshots.items():
            try:
                state = state_by_resource[resource]
            except KeyError as exc:
                raise RuntimeError(
                    f"workload_state_snapshots references unknown state {resource!r}"
                ) from exc
            state_versions[resource] = StateVersionBinding(state, snapshot)

        return WorkloadExecutionBindings(
            declarations=declarations,
            provided_inputs=provided_inputs,
            output_sinks=output_sinks,
            state_versions=state_versions,
            sampler_provider=(
                None if not sampler_providers else sampler_providers[0]
            ),
            diagnostics_readers=diagnostics_readers,
            stats_reader=None if not stats_readers else stats_readers[0],
            per_request_stats_reader=(
                None
                if not per_request_stats_readers
                else per_request_stats_readers[0]
            ),
            enabled_setter=None if not enabled_setters else enabled_setters[0],
            enabled_predicate=(
                None if not enabled_predicates else enabled_predicates[0]
            ),
            synchronizer=None if not synchronizers else synchronizers[0],
            residual_paths=residual_paths,
            backend_region_binders=backend_region_binders,
            static_binding_entries=static_binding_entries,
        )

    def cold_specialize_workloads(self, spec: "RuntimeSpecializationSpec") -> None:
        """Materialize native state before Region artifacts are compiled."""

        import json
        from tllm import _core
        from tllm.schedule import CompileOptions
        facts_json = json.dumps(
            {
                "device": str(spec.device),
                "dtype": str(spec.dtype),
                "hidden_size": spec.hidden_size,
                "rows": spec.rows,
                **(
                    {"max_prefill_tokens": spec.max_prefill_tokens}
                    if spec.max_prefill_tokens is not None
                    else {}
                ),
                **(
                    {"model_layer_count": spec.model_layer_count}
                    if spec.model_layer_count is not None
                    else {}
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        specialized_options = CompileOptions.from_json(
            _core.normalize_runtime_specialization_options(
                self._workload_compile_options.to_json(), facts_json
            )
        )
        if specialized_options == self._workload_compile_options:
            return
        # Publish the runtime facts before workload cold specializers rebuild
        # immutable, shape-specific static resources. Their refreshed worker
        # specs and specialization keys must participate in this same epoch.
        self._workload_compile_options = specialized_options
        refreshed_resources = False
        for resources in self._native_workload_resources.values():
            match resources.services:
                case LocalRuntimeServices(cold_specialize=specialize):
                    specialize(spec)
                    refreshed_resources = True
                case NoRuntimeServices() | DiagnosticsRuntimeServices():
                    pass
        if refreshed_resources:
            self._refresh_native_worker_specs()
        self._rebuild_workload_compiler()

    @property
    def has_compilable_workloads(self) -> bool:
        return self._workload_compiler is not None

    def _rebuild_workload_compiler(self) -> None:
        from tllm.schedule import WorkloadCompilerCoordinator
        from tllm.workload import declare_workload

        declarations = tuple(
            declare_workload(workload_instance)
            for workload_instance in self._workloads
            if self._is_native_workload_declaration(workload_instance)
        )
        self._workload_compiler = (
            None
            if not declarations
            else WorkloadCompilerCoordinator(
                declarations,
                options=self._workload_compile_options,
                persistent_state_bytes=self._workload_persistent_state_bytes,
                epoch_allocator=self._workload_epoch_allocator,
                cache_directory=self._workload_decision_cache_directory,
            )
        )
        self._workload_compilation = None

    def compile_workloads(
        self, timeline: "MainWorkloadTimeline"
    ) -> "WorkloadCompilation":
        self._require_workload_live()
        compiler = self._workload_compiler
        if compiler is None:
            raise RuntimeError("runtime has no @workload declarations to compile")
        compilation = compiler.compile(timeline)
        self._workload_compilation = compilation
        return compilation

    @property
    def has_native_workloads(self) -> bool:
        """Whether this runtime owns fully native Region declarations."""

        return bool(self._unadapted_workload_ids)

    def transition_native_workload_enabled(
        self,
        enabled: bool,
        *,
        plane: "ControlPlane | None" = None,
    ) -> bool:
        """Serialize an enable transition with all other cold epoch changes."""

        with self._shape_specialization_poll_lock:
            if self._pending_workload_finalizations:
                raise RuntimeError(
                    "cannot change native enable semantics while a prior workload "
                    "epoch awaits finalization"
                )
            from tllm.runtime import residual_runtime

            store = residual_runtime.RUNTIME.workload_execution_store
            if store is not None and store.staged is not None:
                raise RuntimeError(
                    "cannot change native enable semantics while a workload epoch "
                    "is already staged"
                )
            # Worker reconstruction creates fresh ESamp model/optimizer state.
            # Until that state has an explicit epoch-to-epoch migration contract,
            # MP must reject a live semantic toggle rather than reset training.
            if plane is not None and self._workload_compilation is not None:
                raise RuntimeError(
                    "live native enable transitions under MP require explicit "
                    "state migration and are not supported"
                )
            return self._transition_native_workload_enabled_locked(enabled)

    def _transition_native_workload_enabled_locked(
        self,
        enabled: bool,
    ) -> bool:
        """Publish enable semantics as a new coordinator-owned Region epoch.

        ``training_enabled`` participates in both the static resource binding
        and the signed compile options. Changing it cannot mutate the local
        callbacks of a live execution: the old epoch remains current until a
        replacement has compiled, validated, and reached the normal safe-point
        publication protocol.
        """

        self._require_workload_live()
        if not self._unadapted_workload_ids:
            return False
        mutable_services: list[LocalRuntimeServices] = []
        for resources in self._native_workload_resources.values():
            match resources.services:
                case LocalRuntimeServices() as services:
                    mutable_services.append(services)
                case NoRuntimeServices() | DiagnosticsRuntimeServices():
                    pass
        if len(mutable_services) != 1:
            raise RuntimeError(
                "native enable transition requires exactly one declared "
                "enable-state resource"
            )
        services = mutable_services[0]
        current_enabled = services.is_enabled()
        if current_enabled == enabled:
            return False

        previous_options = self._workload_compile_options
        previous_compiler = self._workload_compiler
        previous_compilation = self._workload_compilation
        previous_resources = dict(self._native_workload_resources)
        previous_specs = self._workload_worker_specs
        setter = services.set_enabled
        try:
            # The declaration owns the semantic mutation. Immediately replace
            # its local resource snapshot before rebuilding the signed inputs.
            setter(enabled)
            self._refresh_native_worker_specs()
            current = previous_compilation
            if current is None:
                return True
            replacement = self.compile_workloads(current.timeline)
            if replacement.decision.signature == current.decision.signature:
                raise RuntimeError(
                    "native enable transition did not change the signed decision"
                )
            if replacement.cache_key == current.cache_key:
                raise RuntimeError(
                    "native enable transition did not change the compilation key"
                )
            from tllm.runtime import residual_runtime

            residual_runtime.publish_workload_compilation(
                replacement,
                bindings=self.workload_execution_bindings,
            )
        except Exception:
            # No partial semantic update may survive a failed replacement.
            setter(current_enabled)
            self._native_workload_resources = previous_resources
            self._workload_worker_specs = previous_specs
            self._workload_compile_options = previous_options
            self._workload_compiler = previous_compiler
            self._workload_compilation = previous_compilation
            raise
        return True

    def _require_workload_live(self) -> None:
        reason = self._workload_fail_stop_reason
        if reason is not None:
            raise RuntimeError(f"tLLM workload runtime is fail-stopped: {reason}")

    def _fail_stop_workloads(self, reason: str) -> None:
        if not reason:
            raise TypeError("workload fail-stop reason must be a non-empty string")
        if self._workload_fail_stop_reason is None:
            self._workload_fail_stop_reason = reason

    def apply_region_shape_specializations(
        self,
        requests: Sequence[object],
        *,
        publish: bool = True,
    ) -> bool:
        """Compile and stage one signed shape epoch at an in-process safe point.

        Workers without the owning coordinator return ``False`` so the JSON
        requests remain queued for the parent MP control plane.
        """

        self._require_workload_live()

        from tllm.contracts.shape_specialization import (
            RegionShapeSpecializationRequest,
        )

        typed = tuple(requests)
        if not typed or any(
            not isinstance(request, RegionShapeSpecializationRequest)
            for request in typed
        ):
            raise TypeError(
                "shape specializations must contain RegionShapeSpecializationRequest values"
            )
        typed_requests = tuple(
            cast("RegionShapeSpecializationRequest", request) for request in typed
        )
        current = self._workload_compilation
        if current is None or self._workload_compiler is None:
            return False
        physical = current.physical
        if physical is None:
            raise RuntimeError("active workload compilation has no physical epoch")
        import json

        from tllm import _core
        from tllm.schedule import CompileOptions

        requests_json = json.dumps(
            [request.to_wire() for request in typed_requests],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        try:
            options_json = _core.apply_region_shape_specializations(
                current.cache_key,
                current.graph.to_json(),
                current.timeline.to_json(),
                current.options.to_json(),
                current.decision.to_json(),
                physical.plan_epoch,
                requests_json,
            )
        except ValueError as exc:
            # Preserve the runtime method's established public failure type;
            # the closed semantic validation itself is Rust-owned.
            raise RuntimeError(str(exc)) from exc
        specialized_options = CompileOptions.from_json(options_json)
        if specialized_options == current.options:
            raise RuntimeError(
                "shape specialization request did not produce a new immutable epoch"
            )

        previous_options = self._workload_compile_options
        previous_compiler = self._workload_compiler
        previous_compilation = self._workload_compilation
        try:
            self._workload_compile_options = specialized_options
            self._rebuild_workload_compiler()
            replacement = self.compile_workloads(current.timeline)
            if publish:
                from tllm.runtime import residual_runtime

                residual_runtime.publish_workload_compilation(
                    replacement,
                    bindings=self.workload_execution_bindings,
                )
        except Exception:
            self._workload_compile_options = previous_options
            self._workload_compiler = previous_compiler
            self._workload_compilation = previous_compilation
            raise
        return True

    def apply_pending_worker_shape_specializations(self, plane: "ControlPlane") -> bool:
        """Install one parent-compiled replacement for queued worker misses.

        Shape misses are immutable requests for the *current* worker epoch.
        This method deliberately does not retry the failed execution.  It
        prepares the next epoch, installs it through the normal MP epoch
        protocol, then removes exactly the requests consumed by that epoch.
        """

        self._require_workload_live()
        with self._shape_specialization_poll_lock:
            return self._apply_pending_worker_shape_specializations_locked(plane)

    def _apply_pending_worker_shape_specializations_locked(
        self, plane: "ControlPlane"
    ) -> bool:
        """Run one serialized parent-side shape poll under the runtime lock."""

        current = self._workload_compilation
        if current is None or current.physical is None:
            return False
        replies = plane.peek_shape_specialization_requests()
        if len(replies) != 1:
            raise RuntimeError(
                "shape specialization currently requires exactly one TP=1 worker reply"
            )
        reply = replies[0]
        requests = reply.requests
        if not requests:
            # A globally activated replacement is promoted by the worker's
            # next execution safe point.  Until then a poll can legitimately
            # observe the prior current epoch; with no request to consume,
            # that stale-but-typed envelope is harmless.
            return False
        expected_signature = current.decision.signature
        expected_epoch = current.physical.plan_epoch
        if (
            reply.signature != expected_signature
            or reply.plan_epoch != expected_epoch
        ):
            raise RuntimeError(
                "worker shape-specialization envelope does not match the active signed epoch"
            )
        previous_options = self._workload_compile_options
        previous_compiler = self._workload_compiler
        previous_compilation = current
        dedup_keys = tuple(request.dedup_key for request in requests)
        try:
            if not self.apply_region_shape_specializations(requests, publish=False):
                raise RuntimeError(
                    "parent runtime cannot compile a worker shape-specialization epoch"
                )
            replacement = self._workload_compilation
            if replacement is None or replacement is previous_compilation:
                raise RuntimeError(
                    "shape-specialization request did not produce a replacement epoch"
                )
            self._install_workload_compilation_on_workers(plane, replacement)
            acknowledgements = plane.ack_shape_specialization_requests(
                expected_signature,
                expected_epoch,
                dedup_keys,
            )
            if len(acknowledgements) != 1 or acknowledgements[0] != dedup_keys:
                raise RuntimeError(
                    "workers did not acknowledge the exact shape-specialization requests"
                )
        except Exception:
            # The failed current execution remains failed.  Restoring this
            # parent-only coordinator state avoids a silent retry or a partial
            # cache-key switch; requests stay queued until a later safe point.
            if self._workload_fail_stop_reason is None:
                self._workload_compile_options = previous_options
                self._workload_compiler = previous_compiler
                self._workload_compilation = previous_compilation
            raise
        return True

    def register_consumer(self, consumer: object) -> None:
        if self._unadapted_workload_ids:
            raise RuntimeError(
                "native Region authority cannot be coinstalled with legacy "
                "dispatch consumers"
            )
        plane = None
        spec = None
        if active_runtime() is self:
            from tllm.runtime.mp.control import active_control_plane

            plane = active_control_plane()
            if plane is not None:
                spec = consumer_spec(consumer)
        self._consumers.append(consumer)
        if active_runtime() is self:
            from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec

            register_legacy_consumer(consumer)
            refresh_exported_bootstrap_spec()
            if plane is not None and spec is not None:
                plane.register_consumer(spec.name, spec.config)

    def register_workload(self, workload_instance: object) -> None:
        definition = workload_definition(workload_instance)
        consumer = legacy_consumer(workload_instance, definition)
        if consumer is None:
            if definition is None:
                raise RuntimeError("native workload has no definition")
            if self._consumers:
                raise RuntimeError(
                    "native Region authority cannot be coinstalled with legacy "
                    "dispatch consumers"
                )
            self._workloads.append(workload_instance)
            self._record_native_resources(workload_instance)
            self._refresh_resource_specialization_options()
            self._rebuild_workload_compiler()
            self._unadapted_workload_ids.append(definition.workload_id)
            return
        self._workloads.append(workload_instance)
        self._rebuild_workload_compiler()
        self.register_consumer(consumer)

    def clear_consumers(self) -> None:
        self._consumers.clear()
        if active_runtime() is self:
            from tllm.runtime import residual_runtime
            from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec
            from tllm.runtime.mp.control import active_control_plane

            residual_runtime.clear_dispatch_consumers()
            refresh_exported_bootstrap_spec()
            plane = active_control_plane()
            if plane is not None:
                plane.clear_consumers()

    def activate(self) -> None:
        """Make this runtime the one global hooks dispatch to.

        A runtime with no consumers of its own leaves any consumers already
        registered through the legacy façade untouched (backward compat).
        """
        global _active_runtime
        _active_runtime = self
        from tllm.runtime import residual_runtime

        residual_runtime.publish_workload_compilation(None)
        residual_runtime.RUNTIME.native_workloads_pending = bool(
            self._unadapted_workload_ids
        )
        residual_runtime.RUNTIME.native_schedule_graph = (
            None if self._workload_compiler is None else self._workload_compiler.graph
        )
        native_residual_paths: dict["ResidualLocator", str] = {}
        for resources in self._native_workload_resources.values():
            overlap = set(native_residual_paths).intersection(resources.residual_paths)
            if overlap:
                raise RuntimeError(
                    "native workloads declare duplicate residual locator bindings"
                )
            native_residual_paths.update(resources.residual_paths)
        if native_residual_paths:
            residual_runtime.RUNTIME.residual_raw_paths = native_residual_paths
        self._apply_consumers()

    def _apply_consumers(self) -> None:
        if not self._consumers:
            if self._unadapted_workload_ids:
                from tllm.runtime import residual_runtime

                residual_runtime.clear_dispatch_consumers()
            return
        from tllm.runtime import residual_runtime

        residual_runtime.clear_dispatch_consumers()
        for consumer in self._consumers:
            register_legacy_consumer(consumer)

    def make_llm(self, *args: object, **kwargs: object) -> VllmHandle:
        from tllm.runtime.mp.control import set_active_control_plane

        # Resolve mixed-authority errors before constructing an engine. Pure
        # native declarations are bound on the backend cold-compilation path.
        _ = self.workload_execution_bindings
        set_active_control_plane(None)
        self.activate()
        llm = create_llm(args, dict(kwargs))
        self._bind_engine_runners(llm)
        return llm

    def _bind_engine_runners(self, llm: object) -> None:
        """Record runner -> runtime routing for every live worker runner."""
        from tllm.runtime.mp.control import ControlPlane, set_active_control_plane

        plane = ControlPlane.try_bind(llm)
        if plane is not None:
            try:
                plane.call("tllm_bind_runner_runtime")
            except Exception:
                set_active_control_plane(None)
            else:
                if self.has_compilable_workloads:
                    self._coordinate_worker_compilation(plane)
                set_active_control_plane(plane)
                self._install_shape_specialization_poll(llm, plane)
                return
        bind_inprocess_runner(llm, self)

    def _install_shape_specialization_poll(
        self, llm: object, plane: "ControlPlane"
    ) -> None:
        """Poll completed worker misses before the next user request.

        A shape miss intentionally fails its current worker step.  The parent
        has no callback from an EngineCore child, so the next public generate
        boundary is the safe place to install the cold specialization without
        replaying that failed step.
        """

        if not self.has_compilable_workloads:
            return
        def before_generate() -> None:
            # Keep the lock through the underlying call: after staging a
            # replacement, this request must reach the worker safe point
            # before another caller observes the parent-only next epoch.
            with self._shape_specialization_poll_lock:
                self._require_workload_live()
                self.apply_pending_worker_shape_specializations(plane)
                self._require_workload_live()

        install_generate_poll(llm, self, before_generate)

    def _coordinate_worker_compilation(self, plane: "ControlPlane") -> None:
        """Compile once in the parent from worker-reported pure timeline facts."""

        self._require_workload_live()
        import json

        from tllm.schedule import MainWorkloadTimeline

        self.drain_pending_workload_finalizations()
        if self._pending_workload_finalizations:
            raise RuntimeError(
                "cannot install another workload epoch before the committed epoch drains"
            )

        self._refresh_native_worker_specs()

        local_bindings = self.workload_execution_bindings
        if local_bindings is not None and not self._workload_worker_specs:
            raise RuntimeError(
                "native compiled workloads require an explicit worker reconstruction "
                "spec; callable transport over the JSON control plane is forbidden"
            )
        if local_bindings is None and self._workload_worker_specs:
            raise RuntimeError(
                "worker reconstruction specs require fully native Region authority"
            )

        replies = plane.workload_timelines(
            configured_budget=int(
                self._workload_compile_options.memory_budget_bytes
            )
        )
        if len(replies) != 1:
            raise RuntimeError(
                "workload compiler currently requires exactly one TP=1 worker timeline"
            )
        from tllm import _core
        from tllm.schedule import CompileOptions

        timeline_payloads: list[Mapping[str, object]] = []
        normalized_options_json: list[str] = []
        current_options_json = self._workload_compile_options.to_json()
        for reply in replies:
            timeline_payloads.append(reply.timeline)
            facts_json = json.dumps(
                reply.specialization,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
            normalized_options_json.append(
                _core.normalize_runtime_specialization_options(
                    current_options_json,
                    facts_json,
                )
            )
        if len(set(normalized_options_json)) != 1:
            raise RuntimeError(
                "TP=1 workload compilation requires identical worker specializations"
            )
        specialized_options = CompileOptions.from_json(normalized_options_json[0])
        specialization = None
        if timeline_payloads and replies[0].specialization is not None:
            specialization = _runtime_specialization_spec_from_wire(
                replies[0].specialization
            )
        if specialization is not None:
            # The coordinator owns compilation of typed cold artifacts.  The
            # subsequent worker reconstruction receives only the immutable,
            # Rust-signed bundle through its refreshed worker spec.
            self.cold_specialize_workloads(specialization)
        elif specialized_options != self._workload_compile_options:
            self._workload_compile_options = specialized_options
            self._rebuild_workload_compiler()
        timelines = tuple(
            MainWorkloadTimeline.from_json(
                json.dumps(payload, sort_keys=True, separators=(",", ":"))
            )
            for payload in timeline_payloads
        )
        timeline_keys = {timeline.timeline_key for timeline in timelines}
        if len(timeline_keys) != 1:
            raise RuntimeError(
                "TP=1 workload compilation requires identical worker timelines"
            )
        compilation = self.compile_workloads(timelines[0])
        self._install_workload_compilation_on_workers(plane, compilation)

    def _install_workload_compilation_on_workers(
        self, plane: "ControlPlane", compilation: "WorkloadCompilation"
    ) -> None:
        """Install an already compiled immutable epoch on every TP=1 worker."""

        self._require_workload_live()
        from tllm.runtime import residual_runtime

        self.drain_pending_workload_finalizations()
        if self._pending_workload_finalizations:
            raise RuntimeError(
                "cannot install another workload epoch before the committed epoch drains"
            )

        local_bindings = self.workload_execution_bindings
        if local_bindings is not None and not self._workload_worker_specs:
            raise RuntimeError(
                "native compiled workloads require an explicit worker reconstruction "
                "spec; callable transport over the JSON control plane is forbidden"
            )
        if local_bindings is None and self._workload_worker_specs:
            raise RuntimeError(
                "worker reconstruction specs require fully native Region authority"
            )
        expected_artifacts = tuple(
            (region.region_id, region.artifact_key)
            for region in compilation.decision.execution_regions
        )
        expected_epoch = (
            compilation.physical.plan_epoch
            if compilation.physical is not None
            else 1
        )

        def validate_reports(
            reports: Sequence[Mapping[str, object]], *, committed: bool
        ) -> None:
            from tllm.runtime.mp.control import workload_install_report_from_wire

            typed_reports = tuple(
                workload_install_report_from_wire(report, committed=committed)
                for report in reports
            )
            if not typed_reports or any(
                report["signature"] != compilation.decision.signature
                for report in typed_reports
            ):
                raise RuntimeError(
                    "worker rejected or rewrote the signed workload decision"
                )
            if any(
                report["plan_epoch"] != expected_epoch for report in typed_reports
            ):
                raise RuntimeError("worker reported an invalid workload plan epoch")
            if local_bindings is None:
                return
            for report in typed_reports:
                raw_artifacts = report["artifacts"]
                if report["native"] is not True:
                    raise RuntimeError(
                        "worker did not materialize native Region artifacts"
                    )
                actual_artifacts = tuple(
                    (item["region_id"], item["artifact_key"])
                    for item in cast("list[dict[str, str]]", raw_artifacts)
                )
                if actual_artifacts != expected_artifacts:
                    raise RuntimeError(
                        "worker Region artifact report does not match the signed decision"
                    )

        prepared = False
        activation_started = False
        try:
            # A collective prepare may fail after earlier ranks have staged
            # successfully, so arm abort before issuing the RPC.
            prepared = True
            prepared_reports = plane.prepare_workload_compilation(
                compilation,
                reconstruction_specs=self._workload_worker_specs,
            )
            validate_reports(prepared_reports, committed=False)
            committed_reports = plane.commit_workload_compilation(compilation)
            if len(committed_reports) != len(prepared_reports):
                raise RuntimeError("worker commit quorum reply count mismatch")
            validate_reports(committed_reports, committed=True)
            # Activation is a third phase.  Do not let an individual worker's
            # safe point expose this epoch until every commit vote is present.
            activation_started = True
            activation_reports = plane.activate_workload_compilation(compilation)
            if (
                not activation_reports
                or len(activation_reports) != len(committed_reports)
                or any(
                    report.signature != compilation.decision.signature
                    or report.plan_epoch != expected_epoch
                    or report.activated is not True
                    for report in activation_reports
                )
            ):
                raise RuntimeError("worker returned an invalid workload activation report")
            # Worker commits retain their old NativeWorkloadExecution until a
            # later successful replacement.  This makes an abort after a
            # partial collective commit deterministic instead of relying on a
            # best-effort current-step eager fallback.
            installed_store = residual_runtime.RUNTIME.workload_execution_store
            installed_execution = None
            if installed_store is not None:
                for candidate in (
                    installed_store.current,
                    installed_store.staged,
                ):
                    if (
                        candidate is not None
                        and candidate.plan_epoch == expected_epoch
                        and candidate.compilation.decision.signature
                        == compilation.decision.signature
                    ):
                        installed_execution = candidate
                        break
            if installed_execution is None:
                residual_runtime.publish_workload_compilation(
                    compilation,
                    bindings=local_bindings,
                )
            else:
                if installed_execution.artifact_report != expected_artifacts:
                    raise RuntimeError(
                        "in-process native Region artifact report does not match "
                        "the signed decision"
                    )
            if installed_store is not None and installed_execution is installed_store.current:
                # In-process collective RPC and the parent share one runtime.
                # Keep the worker-materialized execution and publish only the
                # parent compilation identity after exact signature matching.
                residual_runtime.RUNTIME.workload_compilation = compilation
        except Exception as exc:
            if prepared and not activation_started:
                try:
                    plane.abort_workload_compilation(compilation)
                except Exception:
                    # Preserve the original failure: the worker RPC already
                    # reports which rank failed and callers can retry abort.
                    pass
            if activation_started:
                self._fail_stop_workloads(
                    "worker activation started but did not complete with a global quorum"
                )
                raise RuntimeError(
                    "workload activation failed after commit quorum; workers must "
                    "fail-stop rather than abort an epoch that may be active"
                ) from exc
            raise
        # Finalize is cleanup, not another publication point.  A partial
        # finalization must leave the new epoch active; its RPC is idempotent
        # for already-finalized workers and can therefore be retried by the
        # caller without rolling back a globally committed plan.
        try:
            finalized_reports = plane.finalize_workload_compilation(compilation)
            if (
                len(finalized_reports) != len(prepared_reports)
                or any(
                    report.get("signature") != compilation.decision.signature
                    or report.get("plan_epoch") != expected_epoch
                    for report in finalized_reports
                )
            ):
                raise RuntimeError(
                    "worker returned an invalid workload finalization report"
                )
            finalized = tuple(
                report.get("finalized") is True for report in finalized_reports
            )
            if all(finalized):
                return
            if any(finalized) or any(
                report.get("drained") is not False for report in finalized_reports
            ):
                raise RuntimeError("workers diverged while finalizing a workload epoch")
            self._pending_workload_finalizations[
                (compilation.decision.signature, expected_epoch)
            ] = (plane, compilation, len(prepared_reports))
        except Exception as exc:
            self._fail_stop_workloads(
                "worker activation completed but finalization evidence is uncertain"
            )
            raise RuntimeError(
                "workload finalization failed after activation; runtime is fail-stopped"
            ) from exc

    def drain_pending_workload_finalizations(self) -> int:
        """Finalize committed worker epochs only after every rank has drained."""

        self._require_workload_live()
        completed: list[tuple[str, int]] = []
        for identity, (plane, compilation, worker_count) in tuple(
            self._pending_workload_finalizations.items()
        ):
            signature, epoch = identity
            try:
                reports = plane.drain_workload_epochs(signature, epoch)
                if (
                    len(reports) != worker_count
                    or any(
                        report.get("signature") != signature
                        or report.get("plan_epoch") != epoch
                        for report in reports
                    )
                ):
                    raise RuntimeError("worker returned an invalid workload drain report")
                drained = tuple(report.get("drained") is True for report in reports)
                if not all(drained):
                    continue
                finalized = plane.finalize_workload_compilation(compilation)
                if (
                    len(finalized) != worker_count
                    or any(
                        report.get("signature") != signature
                        or report.get("plan_epoch") != epoch
                        or report.get("finalized") is not True
                        or report.get("drained") is not True
                        for report in finalized
                    )
                ):
                    raise RuntimeError("worker failed to finalize a drained workload epoch")
                completed.append(identity)
            except Exception as exc:
                self._fail_stop_workloads(
                    "worker finalization drain evidence is uncertain after activation"
                )
                raise RuntimeError(
                    "workload epoch drain failed after activation; runtime is fail-stopped"
                ) from exc
        for identity in completed:
            del self._pending_workload_finalizations[identity]
        return len(completed)


_default_runtime: "TllmRuntime | None" = None


def default_runtime() -> TllmRuntime:
    global _default_runtime
    if _default_runtime is None:
        _default_runtime = TllmRuntime()
    return _default_runtime
