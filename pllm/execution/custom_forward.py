"""Bind custom-forward execution to the owning state invocation and its trace."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.semantic_graph import ImplementationId, SemanticTask, TaskId, TokenProgramEffect
from pllm.execution.overlay_task import OverlayTransition, ProgramOverlay
from pllm.execution.position_state import PositionBuffers
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import TaskOutput, TensorBatch, TensorOperation, read_batch
from pllm.execution.token_program import KV, TokenProgram
from pllm.execution.token_program_executor import execute_token_program, ordinary_pass
from pllm.execution.trace import TraceEventKind

if TYPE_CHECKING:
    from pllm.execution.execution import TaskGraphImplementation
    from pllm.execution.state_execution import ProjectionResiduals, StateTaskContext

DataT = TypeVar("DataT")
CUSTOM_FORWARD_IMPLEMENTATION = ImplementationId("custom_forward@1")


class CustomForwardPrimitive(ABC):
    """A backend's executor of a step's token programs over its packed rows."""

    overlay: ProgramOverlay | None = None

    @property
    @abstractmethod
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

    @abstractmethod
    def run(
        self, program: dict[str, TokenProgram], effect: TokenProgramEffect, observe: Callable[[int, torch.Tensor], None]
    ) -> None: ...


class SerialCustomForward(CustomForwardPrimitive):
    """Serial reference executor: one row at a time through model-owned layer calls (CPU test models)."""

    position: PositionBuffers | None = None

    @abstractmethod
    def begin(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def layer(
        self,
        row: PackedPhaseRow,
        ordinal: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        residual: torch.Tensor | None,
        kv: KV = "commit",
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None: ...

    @abstractmethod
    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None: ...

    def run(
        self, program: dict[str, TokenProgram], effect: TokenProgramEffect, observe: Callable[[int, torch.Tensor], None]
    ) -> None:
        """Run every row's program, or its ordinary pass when the program skips its phase."""
        assert all(item.over == "token" for item in program.values()), (
            'over="span" programs need a backend span executor'
        )
        outputs: list[torch.Tensor] = []
        for index, row in enumerate(self.rows):
            positions, seed = self.begin(row)
            selected = effect.rows == "all" or row.phase.value == effect.rows
            if self.overlay is not None:
                self.overlay.stage(index, selected)
                self.overlay.reset()

            def layer(
                ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None, kv: KV
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.layer(row, ordinal, positions, hidden, residual, kv)

            if selected:
                entries = () if self.position is None else self.position.entry(())[1:]
                hidden, saved = execute_token_program(
                    program[row.phase.value],
                    seed,
                    positions,
                    torch.full((seed.shape[0], 1), row.phase.value == "decode", device=seed.device),
                    entries,
                    layer,
                    (lambda: None) if self.overlay is None else self.overlay.apply_updates,
                )
                if self.position is not None:
                    self.position.exit(saved)
            else:
                hidden = ordinary_pass(seed, effect.layer_count, layer)
                if self.position is not None:
                    self.position.exit(self.position.entry(())[1:])
            normal = self.normalize(row, hidden, torch.zeros_like(hidden))
            outputs.append(normal)
            if self.overlay is not None:
                self.overlay.collect(tuple(self.overlay.results), self.overlay.captures())
            self.commit(row, lambda: observe(index, hidden) if selected else None)
        self.finish(tuple(outputs))


@dataclass(frozen=True, slots=True)
class CustomForwardTask(Generic[DataT]):
    """Disjoint role programs and their publication-owned optional overlay transition."""

    task: SemanticTask
    program: dict[str, TokenProgram]
    domain: CausalModelDomain
    implementation_id: ImplementationId = CUSTOM_FORWARD_IMPLEMENTATION
    operation: TensorOperation[DataT] | None = None
    overlay_math: OverlayTransition | None = None
    primitive: CustomForwardPrimitive | None = None

    @classmethod
    def from_implementation(
        cls, implementation: TaskGraphImplementation[StateTaskContext[DataT], ProjectionResiduals]
    ) -> CustomForwardTask[DataT]:
        if not isinstance(implementation, cls):
            raise ValueError("a custom forward needs its native task binding")
        return implementation

    def execute(self, context: StateTaskContext[DataT]) -> ProjectionResiduals:
        if self.operation is not None:
            return replace(self.operation, body=self._overlay_body).execute(context)
        self.run(context)
        return {}

    def _overlay_body(self, data: DataT, batch: TensorBatch, context: StateTaskContext[DataT] | None) -> TaskOutput:
        assert context is not None and isinstance(data, TensorFrame) and self.overlay_math is not None
        primitive = self.primitive
        assert primitive is not None
        primitive.overlay = ProgramOverlay(self.overlay_math, batch, read_batch(context), data)
        try:
            self.run(context)
            return primitive.overlay.finish()
        finally:
            primitive.overlay = None

    def run(self, context: StateTaskContext[DataT]) -> None:
        primitive, effect = self.primitive, self.task.custom_forward
        assert primitive is not None and effect is not None
        task_id = TaskId(context.arena.plan.semantic_graph.tasks.index(self.task))

        def observe(index: int, hidden: torch.Tensor) -> None:
            row = primitive.rows[index]
            context.trace.observe(context.invocation_id, task_id, "custom_forward.hidden", hidden, row=row, iteration=0)
            context.trace.record(
                TraceEventKind.CUSTOM_FORWARD, context.invocation_id, task_id=task_id, row=row, iteration=0
            )

        with torch.no_grad(), nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
            primitive.run(self.program, effect, observe)
