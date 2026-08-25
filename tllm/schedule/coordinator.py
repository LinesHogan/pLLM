"""Instance-scoped cold compiler for auxiliary workload declarations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from threading import Lock

from tllm.schedule.compiler import (
    compile_schedule,
    effective_compile_options,
    validate_schedule,
)
from tllm.schedule.contracts import (
    CompileOptions,
    MainWorkloadTimeline,
    ScheduleDecision,
    ScheduleGraph,
)
from tllm.schedule.physical import lower_physical_flow_plan
from tllm.workgraph.physical import PhysicalFlowPlan
from tllm.workload.frontend import WorkloadDeclaration
from tllm.workload.lowering import lower_schedule_graph


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

    def compile(self, timeline: MainWorkloadTimeline) -> WorkloadCompilation:
        if not isinstance(timeline, MainWorkloadTimeline):
            raise TypeError("timeline must be MainWorkloadTimeline")
        cache_key = self._cache_key(timeline)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

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
            "workload-compilation-v2",
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
    graph = ScheduleGraph.from_json(graph_json)
    timeline = MainWorkloadTimeline.from_json(timeline_json)
    options = CompileOptions.from_json(options_json)
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
    "install_workload_compilation",
    "workload_compilation_cache_key",
]
