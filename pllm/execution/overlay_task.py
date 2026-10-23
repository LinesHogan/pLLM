"""Bind chunk domains and drive projection segments through the existing state task runner."""

import ast
import inspect
import textwrap
from collections.abc import Mapping
from types import FunctionType, MethodType
from typing import Generic, Protocol

import torch

from pllm.authoring import TensorStepInput
from pllm.contracts.parameter_overlay import CapturedParameterOperand, ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerRowBatch
from pllm.contracts.semantic_graph import TaskId
from pllm.execution.execution import TaskGraphInvocation
from pllm.execution.scan import scan
from pllm.execution.state_execution import DataT, ResultT, StateExecution, StateTaskContext


def overlay_chunk(function: FunctionType | MethodType) -> int:
    """Resolve the single scan's constant chunk extent from the author's bound source."""
    source = ast.parse(textwrap.dedent(inspect.getsource(function)))
    definition = function.__func__ if isinstance(function, MethodType) else function
    values: dict[str, object] = dict(definition.__globals__)
    if isinstance(function, MethodType):
        values["self"] = function.__self__
    chunks: list[int] = []
    for node in ast.walk(source):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            values[node.targets[0].id] = _constant(node.value, values)
        if isinstance(node, ast.Call) and _constant(node.func, values) is scan:
            chunk = next((keyword.value for keyword in node.keywords if keyword.arg == "chunk"), None)
            size = None if chunk is None else _constant(chunk, values)
            if type(size) is not int or size < 1:
                raise ValueError("Exact overlay needs a constant positive scan chunk size")
            chunks.append(size)
    if len(chunks) != 1:
        raise ValueError("Exact overlay needs one identifiable chunk-state transition")
    return chunks[0]


def _constant(node: ast.expr, values: Mapping[str, object]) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return values.get(node.id)
    if isinstance(node, ast.Attribute):
        owner = _constant(node.value, values)
        return None if owner is None else object.__getattribute__(owner, node.attr)
    return None


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
