"""Bind direct route execution to the owning state invocation and its trace."""

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
from pllm.execution.overlay_task import OverlayTransition, RouteOverlay
from pllm.execution.position_state import PositionBuffers
from pllm.execution.route_executor import execute_route, ordinary_pass
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import TaskOutput, TensorBatch, TensorOperation, _read  # pyright: ignore[reportPrivateUsage]
from pllm.execution.token_program import TokenProgram
from pllm.execution.trace import TracePhase

if TYPE_CHECKING:
    from pllm.execution.execution import TaskGraphImplementation
    from pllm.execution.state_execution import StateTaskContext

DataT = TypeVar("DataT")
TOKENWISE_REPLAY_IMPLEMENTATION = ImplementationId("canonical.tokenwise_layer_replay@1")


class TokenwiseReplayPrimitive(ABC):
    """Model-owned layer calls; CPU models execute the same direct route steps."""

    overlay: RouteOverlay | None = None
    position: PositionBuffers | None = None

    @property
    @abstractmethod
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

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
    ) -> tuple[torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def normalize(self, row: PackedPhaseRow, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor: ...

    @abstractmethod
    def commit(self, row: PackedPhaseRow, record: Callable[[], None]) -> None: ...

    @abstractmethod
    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None: ...

    def run(
        self, program: TokenProgram, effect: TokenProgramEffect, observe: Callable[[int, torch.Tensor], None]
    ) -> None:
        """Serial CPU reference using the model's layer interface."""
        outputs: list[torch.Tensor] = []
        for index, row in enumerate(self.rows):
            positions, seed = self.begin(row)
            selected = effect.rows == "all" or row.phase.value == effect.rows
            if self.overlay is not None:
                self.overlay.stage(index, selected)
                self.overlay.reset()

            def layer(
                ordinal: int, hidden: torch.Tensor, residual: torch.Tensor | None
            ) -> tuple[torch.Tensor, torch.Tensor]:
                return self.layer(row, ordinal, positions, hidden, residual)

            if selected:
                entries = () if self.position is None else self.position.entry(())[1:]
                hidden, saved = execute_route(
                    program,
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
class TokenwiseReplayTask(Generic[DataT]):
    """One traced program and its publication-owned optional overlay transition."""

    task: SemanticTask
    program: TokenProgram
    domain: CausalModelDomain
    implementation_id: ImplementationId = TOKENWISE_REPLAY_IMPLEMENTATION
    operation: TensorOperation[DataT] | None = None
    overlay_math: OverlayTransition | None = None

    @classmethod
    def from_implementation(
        cls, implementation: TaskGraphImplementation[StateTaskContext[DataT]]
    ) -> TokenwiseReplayTask[DataT]:
        if not isinstance(implementation, cls):
            raise ValueError("tokenwise replay requires its native task binding")
        return implementation

    def execute(self, context: StateTaskContext[DataT]) -> None:
        if self.operation is not None:
            replace(self.operation, body=self._overlay_body).execute(context)
        else:
            self.run(context)

    def _overlay_body(self, data: DataT, batch: TensorBatch, context: StateTaskContext[DataT] | None) -> TaskOutput:
        assert context is not None and isinstance(data, TensorFrame) and self.overlay_math is not None
        primitive = context.replay
        assert isinstance(primitive, TokenwiseReplayPrimitive)
        primitive.overlay = RouteOverlay(self.overlay_math, batch, _read(context), data)
        try:
            self.run(context)
            return primitive.overlay.finish()
        finally:
            primitive.overlay = None

    def run(self, context: StateTaskContext[DataT]) -> None:
        primitive, effect = context.replay, self.task.replay
        assert isinstance(primitive, TokenwiseReplayPrimitive) and isinstance(effect, TokenProgramEffect)
        task_id = TaskId(context.arena.plan.semantic_graph.tasks.index(self.task))

        def observe(index: int, hidden: torch.Tensor) -> None:
            row = primitive.rows[index]
            context.trace.observe(context.invocation_id, task_id, "replay.hidden", hidden, row=row, iteration=0)
            context.trace.record(TracePhase.REPLAY, context.invocation_id, task_id=task_id, row=row, iteration=0)

        with torch.no_grad(), nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
            primitive.run(self.program, effect, observe)
