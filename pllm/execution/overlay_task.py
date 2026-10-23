"""Drive in-forward overlay segments at model projections through the existing state task runner."""

from collections.abc import Mapping
from typing import Generic, Protocol

import torch

from pllm.authoring import TensorStepInput
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerRowBatch
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.state_execution import DataT, ResultT, StateExecution, StateTaskContext


class ProjectionTransition(Protocol):
    """The output boundary consumed by a model projection."""

    def run(
        self, operands: tuple[CapturedParameterOperand, ...]
    ) -> Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor]: ...


class ProjectionTask(Generic[DataT, ResultT]):
    """One invocation's projection segments; private writes remain owned by StateExecution."""

    def __init__(
        self,
        execution: StateExecution[TensorStepInput, DataT, ResultT],
        invocation: TaskGraphInvocation[StateTaskContext[DataT]],
        rows: PackedSamplerRowBatch,
        task: TaskId,
        targets: tuple[ParameterTarget, ...],
    ) -> None:
        self.execution, self.invocation, self.rows, self.task = execution, invocation, rows, task
        self.pending = set(targets)

    def run(
        self, operands: tuple[CapturedParameterOperand, ...]
    ) -> Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor]:
        """Execute available target segments and hand their residuals to the model reader."""
        targets = {operand.target for operand in operands}
        assert targets and targets <= self.pending
        self.pending.difference_update(targets)
        self.execution.execute_part(
            self.invocation,
            TensorStepInput(self.rows.row_batch, self.rows, operands=operands),
            self.task,
            complete=not self.pending,
        )
        return self.invocation.context.projection_outputs
