"""Clean-room FTTT declaration and CPU specialization."""

from __future__ import annotations

from dataclasses import replace

from tllm.canonical.host_runtime import HostExecution
from tllm.canonical.workloads.base import CanonicalWorkload
from tllm.canonical.workloads.fttt.tasks import (
    FTTT_CUDA_LOSS_IMPLEMENTATION,
    FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
    FTTT_CUDA_PREDICT_IMPLEMENTATION,
    FTTT_CUDA_PREPARE_IMPLEMENTATION,
    FTTT_LOSS_IMPLEMENTATION,
    FTTT_OPTIMIZE_IMPLEMENTATION,
    FTTT_PREDICT_IMPLEMENTATION,
    FTTT_PREPARE_IMPLEMENTATION,
    FTTT_PUBLISH_IMPLEMENTATION,
    FTTTEditState, FTTTStepContext, initialize_edit_state,
    FTTTLossTask, FTTTOptimizeTask, FTTTPredictTask, FTTTPrepareTask, FTTTPublishTask,
)
from tllm.canonical.workloads.fttt.types import (
    FTTTConfig, FTTTHostTransaction, FTTTResult, FTTTSpecialization,
)
from tllm.contracts.semantic_graph import (
    AnchorSymbol,
    ExecutionKind,
    HostServiceId,
    HostServiceInputSource,
    HostServicePort,
    HostPayloadSchemaId,
    SchedulerTimeline,
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


_EDIT_STATE = StateResource("edit_adapter")
_LOSS_TASK = TaskSymbol("loss")
_OPTIMIZE_TASK = TaskSymbol("optimize")
_PREDICT_TASK = TaskSymbol("predict")
_PREPARE_TASK = TaskSymbol("prepare")
_PUBLISH_TASK = TaskSymbol("publish")


class FTTTWorkload(CanonicalWorkload):
    """Declare FTTT math/dataflow/effects without physical scheduling policy."""

    __slots__ = ("config",)

    def __init__(self, config: FTTTConfig = FTTTConfig()) -> None:
        if type(config) is not FTTTConfig:
            raise TypeError("FTTT workload requires exact config")
        self.config = config

    def semantic_graph(self) -> SemanticGraph:
        cpu = ("compute/cpu@1",)
        cuda = ("compute/cuda@1",)
        host = ("host/service@1",)
        return SemanticGraph(
            workload_id=WorkloadId("fttt.feedback"),
            tasks=(
                SemanticTask(
                    symbol=_LOSS_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.input.ready"),
                        AnchorSymbol("host.compute.end"),
                    ),
                    inputs=(
                        SemanticInput(
                            "edit_input",
                            TaskResultInputSource(_PREPARE_TASK, "edit_input"),
                        ),
                        SemanticInput(
                            "target",
                            TaskResultInputSource(_PREPARE_TASK, "target"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput("gradients", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("loss", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("edit_input", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("intent", SemanticOutputKind.TEMPORARY),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            FTTT_LOSS_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                        TaskImplementationSpec(
                            FTTT_CUDA_LOSS_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda,
                        ),
                    ),
                    effects=(SemanticEffect(_EDIT_STATE, StateEffectKind.READ),),
                ),
                SemanticTask(
                    symbol=_OPTIMIZE_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.input.ready"),
                        AnchorSymbol("host.compute.end"),
                    ),
                    inputs=(
                        SemanticInput(
                            "gradients",
                            TaskResultInputSource(_LOSS_TASK, "gradients"),
                        ),
                        SemanticInput(
                            "edit_input",
                            TaskResultInputSource(_LOSS_TASK, "edit_input"),
                        ),
                        SemanticInput(
                            "intent",
                            TaskResultInputSource(_LOSS_TASK, "intent"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput("state_version", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("edit_input", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("intent", SemanticOutputKind.TEMPORARY),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            FTTT_OPTIMIZE_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                        TaskImplementationSpec(
                            FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda,
                        ),
                    ),
                    effects=(
                        SemanticEffect(_EDIT_STATE, StateEffectKind.TRANSITION),
                    ),
                ),
                SemanticTask(
                    symbol=_PREDICT_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.compute.end"),
                        AnchorSymbol("host.output.publish"),
                    ),
                    inputs=(
                        SemanticInput(
                            "edit_input",
                            TaskResultInputSource(_OPTIMIZE_TASK, "edit_input"),
                        ),
                        SemanticInput(
                            "state_version",
                            TaskResultInputSource(_OPTIMIZE_TASK, "state_version"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "edit_delta",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.fttt.edit_delta@1"),
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            FTTT_PREDICT_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                        TaskImplementationSpec(
                            FTTT_CUDA_PREDICT_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda,
                        ),
                    ),
                    effects=(SemanticEffect(_EDIT_STATE, StateEffectKind.READ),),
                ),
                SemanticTask(
                    symbol=_PREPARE_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.input.ready"),
                        AnchorSymbol("host.compute.end"),
                    ),
                    inputs=(
                        SemanticInput(
                            "transaction",
                            HostServiceInputSource(
                                HostServiceId("fttt.feedback"),
                                HostServicePort("training_transaction"),
                                HostPayloadSchemaId(
                                    "canonical.fttt.host_transaction@1"
                                ),
                            ),
                        ),
                    ),
                    outputs=(
                        SemanticOutput("edit_input", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("target", SemanticOutputKind.TEMPORARY),
                        SemanticOutput("intent", SemanticOutputKind.TEMPORARY),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            FTTT_PREPARE_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cpu,
                        ),
                        TaskImplementationSpec(
                            FTTT_CUDA_PREPARE_IMPLEMENTATION,
                            ExecutionKind.NATIVE_TASK,
                            cuda,
                        ),
                    ),
                ),
                SemanticTask(
                    symbol=_PUBLISH_TASK,
                    timing=TimingConstraint(
                        AnchorSymbol("host.output.publish"),
                        AnchorSymbol("host.next"),
                    ),
                    inputs=(
                        SemanticInput(
                            "intent",
                            TaskResultInputSource(_OPTIMIZE_TASK, "intent"),
                        ),
                        SemanticInput(
                            "state_version",
                            TaskResultInputSource(_OPTIMIZE_TASK, "state_version"),
                        ),
                    ),
                    outputs=(
                        SemanticOutput(
                            "reward_outcomes",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.fttt.reward_outcomes@1"),
                        ),
                        SemanticOutput(
                            "retry_controls",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.fttt.retry_controls@1"),
                        ),
                        SemanticOutput(
                            "update_barriers",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.fttt.update_barriers@1"),
                        ),
                        SemanticOutput(
                            "closed_loop",
                            SemanticOutputKind.HOST,
                            HostPayloadSchemaId("canonical.fttt.closed_loop@1"),
                        ),
                    ),
                    implementations=(
                        TaskImplementationSpec(
                            FTTT_PUBLISH_IMPLEMENTATION,
                            ExecutionKind.HOST_SERVICE,
                            host,
                        ),
                    ),
                    effects=(SemanticEffect(_EDIT_STATE, StateEffectKind.READ),),
                ),
            ),
            states=(
                SemanticState(_EDIT_STATE, StateScope.INSTANCE, ()),
            ),
            execution_parameters_digest=self.config.execution_parameters_digest(),
        )

    def compile(
        self,
        specialization: FTTTSpecialization,
        timeline: SchedulerTimeline,
        *,
        plan_epoch: int = 1,
    ) -> HostExecution[FTTTHostTransaction, FTTTStepContext, FTTTEditState, FTTTResult]:
        if self.config.rank > specialization.hidden_size:
            raise ValueError("FTTT rank exceeds the specialized hidden size")
        execution_parameters_digest = specialization.execution_parameters_digest(
            self.config
        )
        graph = replace(
            self.semantic_graph(),
            execution_parameters_digest=execution_parameters_digest,
        )
        state = initialize_edit_state(self.config, specialization)
        cpu = specialization.device.type == "cpu"
        return HostExecution(
            graph, timeline, state,
            (
                FTTTLossTask(FTTT_LOSS_IMPLEMENTATION if cpu else FTTT_CUDA_LOSS_IMPLEMENTATION),
                FTTTOptimizeTask(
                    self.config, FTTT_OPTIMIZE_IMPLEMENTATION if cpu else FTTT_CUDA_OPTIMIZE_IMPLEMENTATION,
                ),
                FTTTPredictTask(FTTT_PREDICT_IMPLEMENTATION if cpu else FTTT_CUDA_PREDICT_IMPLEMENTATION),
                FTTTPrepareTask(
                    self.config, FTTT_PREPARE_IMPLEMENTATION if cpu else FTTT_CUDA_PREPARE_IMPLEMENTATION,
                ),
                FTTTPublishTask(),
            ),
            specialization.device, plan_epoch=plan_epoch,
        )


__all__ = [
    "FTTTWorkload",
]
