"""Instance-scoped cold compiler for auxiliary workload declarations."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from tllm.schedule.compiler import (
    compile_schedule,
    effective_compile_options,
    validate_compile_environment,
    validate_schedule,
)
from tllm.schedule.contracts import (
    CompileOptions,
    MainWorkloadTimeline,
    ScheduleDecision,
    ScheduleGraph,
)
from tllm.schedule.physical import lower_physical_flow_plan
from tllm.contracts.physical_plan import PhysicalFlowPlan
from tllm.workload.frontend import WorkloadDeclaration
from tllm.workload.lowering import lower_schedule_graph

_PERSISTED_COMPILATION_SCHEMA_VERSION = 3


class WorkloadDecisionCacheError(RuntimeError):
    """A persisted workload decision cache entry failed validation."""


@dataclass(frozen=True, slots=True)
class WorkloadCompilation:
    cache_key: str
    graph: ScheduleGraph
    timeline: MainWorkloadTimeline
    options: CompileOptions
    decision: ScheduleDecision
    physical: PhysicalFlowPlan | None


class PlanEpochAllocator:
    """Allocate monotonic physical-plan epochs independently of cache entries."""

    __slots__ = ("_next", "_lock")

    def __init__(self, *, initial_epoch: int = 1) -> None:
        if (
            not isinstance(initial_epoch, int)
            or isinstance(initial_epoch, bool)
            or initial_epoch < 1
        ):
            raise ValueError("initial_epoch must be a positive int")
        self._next = initial_epoch
        self._lock = Lock()

    def allocate(self) -> int:
        with self._lock:
            epoch = self._next
            self._next += 1
            return epoch


class WorkloadCompilerCoordinator:
    """Compile and cache immutable decisions without retaining runtime payloads."""

    def __init__(
        self,
        declarations: tuple[WorkloadDeclaration, ...],
        *,
        options: CompileOptions = CompileOptions(),
        persistent_state_bytes: int = 0,
        epoch_allocator: PlanEpochAllocator | None = None,
        cache_directory: str | Path | None = None,
    ) -> None:
        if not declarations or any(
            not isinstance(declaration, WorkloadDeclaration)
            for declaration in declarations
        ):
            raise TypeError(
                "declarations must be a non-empty tuple[WorkloadDeclaration, ...]"
            )
        if not isinstance(options, CompileOptions):
            raise TypeError("options must be CompileOptions")
        self._declarations = declarations
        self._options = effective_compile_options(options)
        self._graph = lower_schedule_graph(
            tuple(declaration.ir for declaration in declarations),
            persistent_state_bytes=persistent_state_bytes,
        )
        self._cache: dict[str, WorkloadCompilation] = {}
        if cache_directory is not None:
            if isinstance(cache_directory, Path):
                pass
            elif isinstance(cache_directory, str) and cache_directory:
                cache_directory = Path(cache_directory)
            else:
                raise TypeError("cache_directory must be a non-empty str, Path, or None")
        self._cache_directory = cache_directory
        if epoch_allocator is not None and not isinstance(epoch_allocator, PlanEpochAllocator):
            raise TypeError("epoch_allocator must be PlanEpochAllocator or None")
        self._epoch_allocator = (
            PlanEpochAllocator() if epoch_allocator is None else epoch_allocator
        )

    @property
    def declarations(self) -> tuple[WorkloadDeclaration, ...]:
        return self._declarations

    @property
    def graph(self) -> ScheduleGraph:
        return self._graph

    @property
    def options(self) -> CompileOptions:
        return self._options

    def _cache_key(self, timeline: MainWorkloadTimeline) -> str:
        return workload_compilation_cache_key(
            self._graph, timeline, self._options
        )

    def _persisted_path(self, cache_key: str) -> Path:
        assert self._cache_directory is not None
        return self._cache_directory / f"{cache_key}.json"

    def _load_persisted(self, cache_key: str) -> WorkloadCompilation | None:
        """Load a disk-cached decision, failing closed on any mismatch.

        Only absence is a cache miss.  A present entry is untrusted input:
        the embedded payloads must re-derive the requested key and pass the
        Rust wire validation (signature + canonical Region plan), or the
        load raises instead of falling back to a silently recompiled plan.
        """

        path = self._persisted_path(cache_key)
        if not path.is_file():
            return None
        try:
            payload = json.loads(path.read_text(encoding="ascii"))
            if not isinstance(payload, dict) or payload.get("schema_version") != (
                _PERSISTED_COMPILATION_SCHEMA_VERSION
            ):
                raise ValueError("unsupported workload decision cache schema")
            if payload.get("cache_key") != cache_key:
                raise ValueError("workload decision cache key does not match its entry")
            fields: dict[str, str] = {}
            for name in ("graph", "timeline", "options", "decision"):
                value = payload.get(name)
                if not isinstance(value, str):
                    raise ValueError("workload decision cache payloads must be strings")
                fields[name] = value
            compilation = install_workload_compilation(
                cache_key=cache_key,
                graph_json=fields["graph"],
                timeline_json=fields["timeline"],
                options_json=fields["options"],
                decision_json=fields["decision"],
                plan_epoch=self._epoch_allocator.allocate(),
            )
        except Exception as exc:
            raise WorkloadDecisionCacheError(
                f"persisted workload decision {cache_key} failed validation: {exc}"
            ) from exc
        return compilation

    def _persist(self, compilation: WorkloadCompilation) -> None:
        """Write one signed decision atomically; write errors surface."""

        path = self._persisted_path(compilation.cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": _PERSISTED_COMPILATION_SCHEMA_VERSION,
            "cache_key": compilation.cache_key,
            "graph": compilation.graph.to_json(),
            "timeline": compilation.timeline.to_json(),
            "options": compilation.options.to_json(),
            "decision": compilation.decision.to_json(),
        }
        temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="ascii",
        )
        os.replace(temporary, path)

    def compile(self, timeline: MainWorkloadTimeline) -> WorkloadCompilation:
        if not isinstance(timeline, MainWorkloadTimeline):
            raise TypeError("timeline must be MainWorkloadTimeline")
        cache_key = self._cache_key(timeline)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        if self._cache_directory is not None:
            persisted = self._load_persisted(cache_key)
            if persisted is not None:
                self._cache[cache_key] = persisted
                return persisted

        decision = compile_schedule(self._graph, timeline, self._options)
        validate_schedule(self._graph, timeline, decision, self._options)
        plan_epoch = self._epoch_allocator.allocate()
        physical = lower_physical_flow_plan(
            self._graph,
            timeline,
            decision,
            self._options,
            plan_epoch=plan_epoch,
        )
        compilation = WorkloadCompilation(
            cache_key=cache_key,
            graph=self._graph,
            timeline=timeline,
            options=self._options,
            decision=decision,
            physical=physical,
        )
        if self._cache_directory is not None:
            self._persist(compilation)
        self._cache[cache_key] = compilation
        return compilation


def workload_compilation_cache_key(
    graph: ScheduleGraph,
    timeline: MainWorkloadTimeline,
    options: CompileOptions,
) -> str:
    options = effective_compile_options(options)
    payload = "\0".join(
        (
            "workload-compilation-v7",
            graph.to_json(),
            timeline.to_json(),
            options.to_json(),
        )
    )
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def install_workload_compilation(
    *,
    cache_key: str,
    graph_json: str,
    timeline_json: str,
    options_json: str,
    decision_json: str,
    plan_epoch: int = 1,
) -> WorkloadCompilation:
    """Validate and lower a coordinator-signed wire plan on one worker."""

    from tllm import _core

    # Rust validates the raw payload before Python can normalize it through a
    # dataclass.  It owns the signed JSON and cache-key contracts.
    _core.validate_workload_compilation_wire(
        cache_key,
        graph_json,
        timeline_json,
        options_json,
        decision_json,
        plan_epoch,
    )
    options = CompileOptions.from_json(options_json)
    validate_compile_environment(options)
    graph = ScheduleGraph.from_json(graph_json)
    timeline = MainWorkloadTimeline.from_json(timeline_json)
    decision = ScheduleDecision.from_json(decision_json)
    physical = lower_physical_flow_plan(
        graph,
        timeline,
        decision,
        options,
        plan_epoch=plan_epoch,
    )
    return WorkloadCompilation(
        cache_key=cache_key,
        graph=graph,
        timeline=timeline,
        options=options,
        decision=decision,
        physical=physical,
    )


__all__ = [
    "PlanEpochAllocator",
    "WorkloadCompilation",
    "WorkloadCompilerCoordinator",
    "WorkloadDecisionCacheError",
    "install_workload_compilation",
    "workload_compilation_cache_key",
]
