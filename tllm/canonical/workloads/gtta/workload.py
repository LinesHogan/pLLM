"""Clean-room GTTA declaration and CPU specialization."""

from __future__ import annotations

import torch

from tllm.canonical.host_runtime import HostExecution
from tllm.canonical.workloads.base import CanonicalWorkload
from tllm.canonical.workloads.gtta.state import (
    GTTADynamicsState, GTTADynamicsSnapshot, GTTAStepContext,
    GTTAPublishTask, GTTAQueryTask, GTTAUpdateTask,
    GTTA_PUBLISH_IMPLEMENTATION,
    GTTA_QUERY_IMPLEMENTATION,
    GTTA_UPDATE_IMPLEMENTATION,
)
from tllm.canonical.workloads.gtta.types import GTTAConfig, GTTAHostTransaction, GTTAResult
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

    def compile(
        self, timeline: SchedulerTimeline, *, plan_epoch: int = 1,
    ) -> HostExecution[GTTAHostTransaction, GTTAStepContext, GTTADynamicsSnapshot, GTTAResult]:
        return HostExecution(
            self.semantic_graph(), timeline, GTTADynamicsState(),
            (GTTAPublishTask(), GTTAQueryTask(), GTTAUpdateTask(self.config)),
            torch.device("cpu"), plan_epoch=plan_epoch,
        )


__all__ = ["GTTAWorkload"]
