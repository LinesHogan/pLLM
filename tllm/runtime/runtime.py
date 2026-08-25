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
from typing import TYPE_CHECKING, Any, Sequence, cast
import weakref

if TYPE_CHECKING:
    from tllm.contracts.consumer_stats import TrainingStats
    from tllm.ports.residual_stream import ResidualLocator
    from tllm.runtime.backend_region_binders import BackendRegionBinderSlot
    from tllm.runtime.sampler_bridge.provider import SamplerModifierProvider
    from tllm.runtime.workload_execution import WorkloadExecutionBindings
    from tllm.runtime.workload_plugins import (
        NativeWorkloadResources,
        WorkloadWorkerSpec,
    )
    from tllm.schedule import (
        CompileOptions,
        MainWorkloadTimeline,
        WorkloadCompilation,
        WorkloadCompilerCoordinator,
        PlanEpochAllocator,
    )


# runner -> owning TllmRuntime. Weak on both sides: runners and runtimes are
# collected freely; stale entries vanish.
_RUNNER_RUNTIMES: "weakref.WeakKeyDictionary[Any, TllmRuntime]" = weakref.WeakKeyDictionary()

# The runtime the process-wide class hooks currently dispatch to.
_ACTIVE_RUNTIME: "TllmRuntime | None" = None


def bind_runner_runtime(runner: Any, runtime: "TllmRuntime") -> None:
    _RUNNER_RUNTIMES[runner] = runtime



def active_runtime() -> "TllmRuntime":
    if _ACTIVE_RUNTIME is None:
        return default_runtime()
    return _ACTIVE_RUNTIME


class TllmRuntime:
    def __init__(
        self,
        *,
        backend: str = "auto",
        producers: Sequence[Any] = (),
        consumers: Sequence[Any] = (),
        workloads: Sequence[Any] = (),
        workload_memory_budget_bytes: int = 0,
        workload_allow_serial_fallback: bool = True,
        workload_persistent_state_bytes: int = 0,
        workload_provided_inputs: Mapping[str, Callable[[int], object]] | None = None,
        workload_output_sinks: Mapping[
            str, Callable[[object, int], None]
        ] | None = None,
        workload_state_snapshots: Mapping[
            str, Callable[[int], Mapping[Hashable, int]]
        ] | None = None,
        workload_worker_specs: Sequence[Any] = (),
    ) -> None:
        if backend != "auto":
            raise ValueError(
                f"backend={backend!r}: only 'auto' is supported; backend selection "
                "is structural (tllm.backends)"
            )
        self.producers = tuple(producers)
        self._workloads: list[Any] = list(workloads)
        self._consumers: list[Any] = list(consumers)
        self._unadapted_workload_ids: list[str] = []
        self._native_workload_resources: dict[int, "NativeWorkloadResources"] = {}
        from tllm.schedule import CompileOptions

        self._workload_compile_options = CompileOptions(
            memory_budget_bytes=workload_memory_budget_bytes,
            allow_serial_fallback=workload_allow_serial_fallback,
        )
        if (
            not isinstance(workload_persistent_state_bytes, int)
            or isinstance(workload_persistent_state_bytes, bool)
            or workload_persistent_state_bytes < 0
        ):
            raise ValueError("workload_persistent_state_bytes must be a non-negative int")
        self._workload_persistent_state_bytes = workload_persistent_state_bytes
        self._workload_provided_inputs = dict(workload_provided_inputs or {})
        self._workload_output_sinks = dict(workload_output_sinks or {})
        self._workload_state_snapshots = dict(workload_state_snapshots or {})
        from tllm.runtime.workload_plugins import WorkloadWorkerSpec

        if any(
            not isinstance(spec, WorkloadWorkerSpec)
            for spec in workload_worker_specs
        ):
            raise TypeError(
                "workload_worker_specs must contain WorkloadWorkerSpec values"
            )
        self._workload_worker_specs: tuple["WorkloadWorkerSpec", ...] = tuple(
            workload_worker_specs
        )
        self._workload_compiler: "WorkloadCompilerCoordinator | None" = None
        from tllm.schedule import PlanEpochAllocator

        self._workload_epoch_allocator = PlanEpochAllocator()
        self._workload_compilation: WorkloadCompilation | None = None
        self._pending_workload_finalizations: dict[
            tuple[str, int], tuple[Any, "WorkloadCompilation", int]
        ] = {}
        self._workload_fail_stop_reason: str | None = None
        self._shape_specialization_poll_lock = RLock()
        for workload_instance in self._workloads:
            consumer = self._legacy_consumer_for(workload_instance)
            if consumer is None:
                self._record_native_resources(workload_instance)
                definition = self._workload_definition(workload_instance)
                workload_id = (
                    type(workload_instance).__qualname__
                    if definition is None
                    else definition.workload_id
                )
                self._unadapted_workload_ids.append(str(workload_id))
            elif consumer not in self._consumers:
                self._consumers.append(consumer)
        resource_worker_specs = tuple(
            resources.worker_spec
            for resources in self._native_workload_resources.values()
            if resources.worker_spec is not None
        )
        if resource_worker_specs:
            if self._workload_worker_specs:
                raise RuntimeError(
                    "native workload worker specs cannot be supplied both by "
                    "resources and TllmRuntime"
                )
            self._workload_worker_specs = resource_worker_specs
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
        native_resource_entries = []
        residual_paths: dict[object, str] = {}
        for workload_instance in self._workloads:
            resources = self._native_workload_resources.get(id(workload_instance))
            definition = self._workload_definition(workload_instance)
            if definition is None:
                continue
            workload_id = (
                type(workload_instance).__qualname__
                if definition is None
                else definition.workload_id
            )
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
        static_binding_entries = ()
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
        specs: list[Any] = []
        resource_owned_count = 0
        for workload_instance in self._workloads:
            if id(workload_instance) not in self._native_workload_resources:
                continue
            resource_owned_count += 1
            self._record_native_resources(workload_instance)
            resources = self._native_workload_resources[id(workload_instance)]
            if resources.worker_spec is None:
                # A caller-supplied immutable spec remains authoritative for
                # workloads which do not publish a resource-owned recipe.
                return
            specs.append(resources.worker_spec)
        if resource_owned_count == 0:
            return
        fresh_specs = tuple(specs)
        if fresh_specs == self._workload_worker_specs:
            return
        self._workload_worker_specs = fresh_specs
        self._refresh_resource_specialization_options()
        self._rebuild_workload_compiler()

    def _record_native_resources(self, workload_instance: Any) -> None:
        """Bind nominal native resources once, outside every execution path."""

        from tllm.runtime.workload_plugins import (
            NativeWorkload,
            NativeWorkloadResources,
        )

        if not isinstance(workload_instance, NativeWorkload):
            return
        resources = workload_instance.native_workload_resources()
        if not isinstance(resources, NativeWorkloadResources):
            raise TypeError(
                "NativeWorkload.native_workload_resources() must return "
                "NativeWorkloadResources"
            )
        if resources.workload is not workload_instance:
            raise RuntimeError("native workload resources belong to another instance")
        self._native_workload_resources[id(workload_instance)] = resources

    @staticmethod
    def _workload_definition(workload_instance: Any) -> Any | None:
        try:
            return type(workload_instance).__tllm_workload__
        except AttributeError:
            return None

    @staticmethod
    def _legacy_consumer_for(workload_instance: Any) -> Any | None:
        """Resolve the migration adapter without treating declarations as executable."""

        if TllmRuntime._workload_definition(workload_instance) is None:
            return workload_instance
        try:
            adapter = workload_instance.legacy_consumer
        except AttributeError:
            return None
        return adapter() if callable(adapter) else adapter

    @property
    def workloads(self) -> tuple[Any, ...]:
        return tuple(self._workloads)

    @property
    def consumers(self) -> tuple[Any, ...]:
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
                set(provided_inputs).intersection(resources.provided_inputs)
            )
            if provided_overlap:
                raise RuntimeError(
                    "duplicate native workload provided input binding(s): "
                    + ", ".join(provided_overlap)
                )
            provided_inputs.update(resources.provided_inputs)
            sink_overlap = sorted(
                set(output_sinks).intersection(resources.output_sinks)
            )
            if sink_overlap:
                raise RuntimeError(
                    "duplicate native workload output sink binding(s): "
                    + ", ".join(sink_overlap)
                )
            output_sinks.update(resources.output_sinks)
            for local_name, snapshot in resources.state_snapshots.items():
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
            if resources.sampler_provider is not None:
                sampler_providers.append(resources.sampler_provider)
            if resources.diagnostics_reader is not None:
                workload_id = declaration.ir.workload_id
                if workload_id in diagnostics_readers:
                    raise RuntimeError(
                        "duplicate native workload diagnostics binding "
                        f"{workload_id!r}"
                    )
                diagnostics_readers[workload_id] = resources.diagnostics_reader
            overlap = set(residual_paths).intersection(resources.residual_paths)
            if overlap:
                raise RuntimeError(
                    "native workloads declare duplicate residual locator bindings"
                )
            residual_paths.update(
                cast("Mapping[ResidualLocator, str]", resources.residual_paths)
            )
            binder_overlap = set(backend_region_binders).intersection(
                resources.backend_region_binders
            )
            if binder_overlap:
                raise RuntimeError(
                    "native workloads declare duplicate backend Region binder "
                    "binding(s): "
                    + ", ".join(sorted(binder_overlap))
                )
            backend_region_binders.update(resources.backend_region_binders)
            if resources.stats_reader is not None:
                stats_readers.append(resources.stats_reader)
            if resources.per_request_stats_reader is not None:
                per_request_stats_readers.append(resources.per_request_stats_reader)
            if resources.enabled_setter is not None:
                enabled_setters.append(resources.enabled_setter)
            if resources.enabled_predicate is not None:
                enabled_predicates.append(resources.enabled_predicate)
            if resources.synchronizer is not None:
                synchronizers.append(resources.synchronizer)
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
        state_versions = {}
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
            residual_paths=cast("Mapping[object, str]", residual_paths),
            backend_region_binders=backend_region_binders,
            static_binding_entries=static_binding_entries,
        )

    def cold_specialize_workloads(self, spec: Any) -> None:
        """Materialize native state before Region artifacts are compiled."""

        import json
        from dataclasses import replace

        from tllm import _core
        from tllm.runtime.specialization import RuntimeSpecializationSpec
        from tllm.schedule import CompileOptions

        if not isinstance(spec, RuntimeSpecializationSpec):
            raise TypeError("native workload specialization requires RuntimeSpecializationSpec")
        facts_json = json.dumps(
            {
                "device": str(spec.device),
                "dtype": str(spec.dtype),
                "hidden_size": spec.hidden_size,
                "rows": spec.rows,
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
        for resources in self._native_workload_resources.values():
            if resources.cold_specializer is not None:
                resources.cold_specializer(spec)
        if specialized_options != self._workload_compile_options:
            self._workload_compile_options = specialized_options
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
            if self._workload_definition(workload_instance) is not None
        )
        self._workload_compiler = (
            None
            if not declarations
            else WorkloadCompilerCoordinator(
                declarations,
                options=self._workload_compile_options,
                persistent_state_bytes=self._workload_persistent_state_bytes,
                epoch_allocator=self._workload_epoch_allocator,
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
        plane: Any | None = None,
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
        if type(enabled) is not bool:
            raise TypeError("native workload enabled state must be a bool")

        mutable_resources = tuple(
            resources
            for resources in self._native_workload_resources.values()
            if resources.enabled_setter is not None
        )
        if len(mutable_resources) != 1:
            raise RuntimeError(
                "native enable transition requires exactly one declared "
                "enable-state resource"
            )
        resource = mutable_resources[0]
        predicate = resource.enabled_predicate
        if predicate is None:
            raise RuntimeError(
                "native enable transition requires an immutable enabled-state snapshot"
            )
        current_enabled = bool(predicate())
        if current_enabled == enabled:
            return False

        previous_options = self._workload_compile_options
        previous_compiler = self._workload_compiler
        previous_compilation = self._workload_compilation
        previous_resources = dict(self._native_workload_resources)
        previous_specs = self._workload_worker_specs
        setter = resource.enabled_setter
        assert setter is not None
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
        if not isinstance(reason, str) or not reason:
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

    def apply_pending_worker_shape_specializations(self, plane: Any) -> bool:
        """Install one parent-compiled replacement for queued worker misses.

        Shape misses are immutable requests for the *current* worker epoch.
        This method deliberately does not retry the failed execution.  It
        prepares the next epoch, installs it through the normal MP epoch
        protocol, then removes exactly the requests consumed by that epoch.
        """

        self._require_workload_live()
        with self._shape_specialization_poll_lock:
            return self._apply_pending_worker_shape_specializations_locked(plane)

    def _apply_pending_worker_shape_specializations_locked(self, plane: Any) -> bool:
        """Run one serialized parent-side shape poll under the runtime lock."""

        from tllm.runtime.mp.control import WorkerShapeSpecializationReply

        current = self._workload_compilation
        if current is None or current.physical is None:
            return False
        replies = plane.peek_shape_specialization_requests()
        if len(replies) != 1:
            raise RuntimeError(
                "shape specialization currently requires exactly one TP=1 worker reply"
            )
        reply = replies[0]
        if not isinstance(reply, WorkerShapeSpecializationReply):
            raise TypeError("control plane returned an untyped shape-specialization reply")
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

    def register_consumer(self, consumer: Any) -> None:
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
                from tllm.runtime.consumer_plugins import spec_for_consumer

                spec = spec_for_consumer(consumer)
        self._consumers.append(consumer)
        if active_runtime() is self:
            from tllm.runtime import residual_runtime
            from tllm.runtime.mp.bootstrap import refresh_exported_bootstrap_spec

            residual_runtime.register_dispatch_consumer(consumer)
            refresh_exported_bootstrap_spec()
            if plane is not None and spec is not None:
                plane.register_consumer(spec.name, spec.config)

    def register_workload(self, workload_instance: Any) -> None:
        consumer = self._legacy_consumer_for(workload_instance)
        if consumer is None:
            if self._consumers:
                raise RuntimeError(
                    "native Region authority cannot be coinstalled with legacy "
                    "dispatch consumers"
                )
            self._workloads.append(workload_instance)
            self._record_native_resources(workload_instance)
            self._refresh_resource_specialization_options()
            self._rebuild_workload_compiler()
            definition = self._workload_definition(workload_instance)
            workload_id = (
                type(workload_instance).__qualname__
                if definition is None
                else definition.workload_id
            )
            self._unadapted_workload_ids.append(str(workload_id))
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
        global _ACTIVE_RUNTIME
        _ACTIVE_RUNTIME = self
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
            native_residual_paths.update(
                cast("Mapping[ResidualLocator, str]", resources.residual_paths)
            )
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
            residual_runtime.register_dispatch_consumer(consumer)

    def make_llm(self, *args: Any, **kwargs: Any) -> Any:
        from tllm.runtime import residual_runtime
        from tllm.runtime.mp.control import set_active_control_plane

        # Resolve mixed-authority errors before constructing an engine. Pure
        # native declarations are bound on the backend cold-compilation path.
        _ = self.workload_execution_bindings
        set_active_control_plane(None)
        self.activate()
        llm = residual_runtime.make_llm(*args, **kwargs)
        self._bind_engine_runners(llm)
        return llm

    def _bind_engine_runners(self, llm: Any) -> None:
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
        # Fallback for engines without a collective_rpc path (test doubles):
        # walk the in-process executor directly.
        def _bind(worker: Any) -> None:
            runner = getattr(worker, "model_runner", None)
            if runner is not None:
                bind_runner_runtime(runner, self)

        engine = getattr(llm, "llm_engine", None)
        core = getattr(engine, "engine_core", None)
        executor = getattr(core, "model_executor", None)
        driver = getattr(executor, "driver_worker", None)
        if driver is not None:
            _bind(driver)

    def _install_shape_specialization_poll(self, llm: Any, plane: Any) -> None:
        """Poll completed worker misses before the next user request.

        A shape miss intentionally fails its current worker step.  The parent
        has no callback from an EngineCore child, so the next public generate
        boundary is the safe place to install the cold specialization without
        replaying that failed step.
        """

        if not self.has_compilable_workloads:
            return
        try:
            owner = llm._tllm_shape_specialization_poll_owner
        except AttributeError:
            owner = None
        if owner is self:
            return
        if owner is not None:
            raise RuntimeError(
                "LLM already has a shape-specialization poll owned by another runtime"
            )
        try:
            generate = llm.generate
        except AttributeError:
            return
        if not callable(generate):
            return

        def generate_after_shape_poll(*call_args: Any, **call_kwargs: Any) -> Any:
            # Keep the lock through the underlying call: after staging a
            # replacement, this request must reach the worker safe point
            # before another caller observes the parent-only next epoch.
            with self._shape_specialization_poll_lock:
                self._require_workload_live()
                self.apply_pending_worker_shape_specializations(plane)
                self._require_workload_live()
                return generate(*call_args, **call_kwargs)

        llm._tllm_shape_specialization_poll_owner = self
        llm.generate = generate_after_shape_poll

    def _coordinate_worker_compilation(self, plane: Any) -> None:
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
                "native Region workloads require an explicit worker reconstruction "
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
            timeline_raw = reply.get("timeline")
            specialization_raw = reply.get("specialization")
            if not isinstance(timeline_raw, Mapping) or (
                specialization_raw is not None and type(specialization_raw) is not dict
            ):
                raise RuntimeError("worker returned an invalid workload timeline envelope")
            timeline_payloads.append(timeline_raw)
            facts_json = json.dumps(
                specialization_raw,
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
        if specialized_options != self._workload_compile_options:
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
        self, plane: Any, compilation: "WorkloadCompilation"
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
                "native Region workloads require an explicit worker reconstruction "
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
            from tllm.runtime.mp.control import _workload_install_report_from_wire

            typed_reports = tuple(
                _workload_install_report_from_wire(report, committed=committed)
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


_DEFAULT_RUNTIME: "TllmRuntime | None" = None


def default_runtime() -> TllmRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None:
        _DEFAULT_RUNTIME = TllmRuntime()
    return _DEFAULT_RUNTIME
