"""Clean-room GTTA declaration and CPU specialization."""

from __future__ import annotations

from dataclasses import dataclass

from tllm.canonical.scheduler import compile_schedule, lower_physical_plan
from tllm.canonical.workloads.base import CanonicalWorkload
from tllm.canonical.workloads.gtta.execution import GTTAExecution
from tllm.canonical.workloads.gtta.tasks import (
    GTTA_PUBLISH_IMPLEMENTATION,
    GTTA_QUERY_IMPLEMENTATION,
    GTTA_UPDATE_IMPLEMENTATION,
)
from tllm.canonical.workloads.gtta.types import GTTAConfig
from tllm.contracts.canonical_compilation import CanonicalScheduleBundle
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    ExecutionKind,
    HostPayloadSchemaId,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    SchedulerTimeline,
    SemanticEdge,
    SemanticEffect,
    SemanticGraph,
    SemanticInput,
    SemanticOutput,
    SemanticOutputKind,
    SemanticState,
    SemanticTask,
    StateEffectKind,
    StateResource,
    StateScope,
    TaskImplementationSpec,
    TaskResultInputSource,
    TaskSymbol,
    TimingConstraint,
    WorkloadId,
)


_DYNAMICS_STATE = StateResource("dynamics")
_PUBLISH_TASK = TaskSymbol("publish")
_QUERY_TASK = TaskSymbol("query")
_UPDATE_TASK = TaskSymbol("update")


@dataclass(frozen=True, slots=True)
class GTTACompileOptions:
    timeline: SchedulerTimeline
    plan_epoch: int = 1

    def __post_init__(self) -> None:
        if self.plan_epoch < 1:
            raise ValueError("GTTA compile options are not canonical")


@dataclass(frozen=True, slots=True)
class GTTACompilation:
    execution: GTTAExecution
    schedule_bundle: CanonicalScheduleBundle
    execution_parameters_digest: str

    def __post_init__(self) -> None:
        if (
            self.execution.plan is not self.schedule_bundle.schedule_physical
            or self.execution.execution_parameters_digest
            != self.execution_parameters_digest
            or self.schedule_bundle.semantic_graph.execution_parameters_digest
            != self.execution_parameters_digest
        ):
            raise TypeError("GTTA compilation is not execution-aligned")


class GTTAWorkload(CanonicalWorkload):
    __slots__ = ("config",)

    def __init__(self, config: GTTAConfig = GTTAConfig()) -> None:
        self.config = config

    def semantic_graph(self) -> SemanticGraph:
        cpu = ("compute/cpu@1",)
        host = ("host/service@1",)
        return SemanticGraph(
            workload_id=WorkloadId("gtta.dynamics"),
            tasks=(
                SemanticTask(
                    symbol=_PUBLISH_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.output.publish"),
                        AnchorSymbol("host.next"),
                    ),
                    inputs=(
                        SemanticInput(
                            "query_result",
                            TaskResultInputSource(_QUERY_TASK, "query_result"),
                        ),
                        SemanticInput(
                            "update_receipt",
                            TaskResultInputSource(_UPDATE_TASK, "update_receipt"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "query_result",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.gtta.query_result@1"),
                        ),
                        SemanticOutput(
                            "update_receipt",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.gtta.update_receipt@1"),
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            GTTA_PUBLISH_IMPLEMENTATION,
                            ExecutionKind.HOST_SERVICE,
                            host,
                        ),
                    ),
                    effects=(SemanticEffect(_DYNAMICS_STATE, StateEffectKind.READ),),
                ),
                SemanticTask(
                    symbol=_QUERY_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.compute.end"),
                        AnchorSymbol("host.output.publish"),
                    ),
                    inputs=(
                        SemanticInput(
                            "query",
                            HostServiceInputSource(
                                HostServiceId("gtta.environment"),
                                HostServicePort("query"),
                                HostPayloadSchemaId("canonical.gtta.query@1"),
                            ),
                        ),
                        SemanticInput(
                            "state_version",
                            TaskResultInputSource(_UPDATE_TASK, "state_version"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput("query_result", SemanticOutputKind.TEMPORARY),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            GTTA_QUERY_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                    ),
                    effects=(SemanticEffect(_DYNAMICS_STATE, StateEffectKind.READ),),
                ),
                SemanticTask(
                    symbol=_UPDATE_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.input.ready"),
                        AnchorSymbol("host.compute.end"),
                    ),
                    inputs=(
                        SemanticInput(
                            "trajectories",
                            HostServiceInputSource(
                                HostServiceId("gtta.environment"),
                                HostServicePort("trajectories"),
                                HostPayloadSchemaId(
                                    "canonical.gtta.trajectory_batch@1"
                                ),
                            ),
                        ),
                    ),
                    outputs=(
                        SemanticOutput("state_version", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("update_receipt", SemanticOutputKind.TEMPORARY),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            GTTA_UPDATE_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                    ),
                    effects=(
                        SemanticEffect(
                            _DYNAMICS_STATE,
                            StateEffectKind.TRANSITION_MAYBE,
                        ),
                    ),
                ),
            ),
            execution_parameters_digest=self.config.execution_parameters_digest(),
            states=(SemanticState(_DYNAMICS_STATE, StateScope.INSTANCE, ()),),
            edges=(
                SemanticEdge(_QUERY_TASK, _PUBLISH_TASK),
                SemanticEdge(_UPDATE_TASK, _PUBLISH_TASK),
                SemanticEdge(_UPDATE_TASK, _QUERY_TASK),
            ),
        )

    def compile(self, options: GTTACompileOptions) -> GTTACompilation:
        graph = self.semantic_graph()
        decision = compile_schedule(graph, options.timeline)
        physical = lower_physical_plan(
            graph,
            options.timeline,
            decision,
            plan_epoch=options.plan_epoch,
        )
        bundle = CanonicalScheduleBundle(graph, options.timeline, decision, physical)
        execution = GTTAExecution(self.config, bundle)
        return GTTACompilation(
            execution,
            bundle,
            self.config.execution_parameters_digest(),
        )


__all__ = ["GTTACompilation", "GTTACompileOptions", "GTTAWorkload"]
