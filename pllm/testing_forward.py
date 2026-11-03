"""A deterministic CPU decoder for testing.check's custom-forward schedule comparisons.

It exercises real state bindings and token programs; its pointwise layers do not model attention or KV caches.
"""

from collections.abc import Callable
from dataclasses import replace

import torch

from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.execution.custom_forward import CustomForwardTask, SerialCustomForward
from pllm.execution.state_execution import StateExecution
from pllm.execution.step_input import StepInput
from pllm.execution.tensor_binding import TensorOutput
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.token_program import KV


class CheckForward(SerialCustomForward):
    """Run authored token math against a small deterministic stack, with production state reads."""

    def __init__(self, execution: StateExecution[StepInput, TensorFrame, TensorOutput]) -> None:
        task = execution.arena.plan.semantic_graph.custom_forward_task
        assert task is not None
        scheduled = execution.arena.plan.scheduled(task.symbol)
        binding = CustomForwardTask[TensorFrame].from_implementation(
            execution.implementation(scheduled.implementation_id)
        )
        execution.rebind_implementation(replace(binding, primitive=self))
        self._rows: tuple[PackedPhaseRow, ...] = ()
        self.inputs = torch.empty(0)
        self.output = torch.empty(0)

    def prepare(self, value: StepInput, inputs: torch.Tensor) -> None:
        """Set this step's independently generated token embeddings."""
        self._rows, self.inputs = value.rows.rows, inputs

    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]:
        return self._rows

    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([row.token_offset]), self.inputs[row.packed_row_index : row.packed_row_index + 1]

    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv: KV = "commit",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        stream = hidden if residual is None else hidden + residual
        return (stream.float() * 0.5 + (ordinal + 1) * 0.01).tanh().to(hidden.dtype), torch.zeros_like(hidden)

    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return hidden + residual

    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None:
        record()

    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None:
        self.output = torch.cat(outputs)
