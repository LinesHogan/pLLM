"""Bounded replay in the original signed state invocation, without a version owner."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar

import torch

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_composition import RegionFunctionBody
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import BorrowedResource, TensorType
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.semantic_graph import (
    ImplementationId,
    SemanticTask,
    TaskId,
    TaskSourceRegion,
    TokenProgramEffect,
)
from pllm.execution.decoder_region import DecoderActivation
from pllm.execution.host_borrow import TraceResourceView
from pllm.execution.overlay_task import OverlayTransition, RouteOverlay, RouteOverlayStorage
from pllm.execution.region_execution import RegionInvocation
from pllm.execution.region_native import RegionFunction
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
    TensorResourceBinding,
)
from pllm.execution.region_values import Data
from pllm.execution.tensor_frame import TensorFrame
from pllm.execution.tensor_program import TaskOutput, TensorBatch, TensorOperation, _read  # pyright: ignore[reportPrivateUsage]
from pllm.execution.token_program import update_function
from pllm.execution.tokenwise_region import (
    TokenwiseRegion,
    compile_tokenwise_source,
)
from pllm.execution.trace import TracePhase

if TYPE_CHECKING:
    from pllm.execution.execution import TaskGraphImplementation
    from pllm.execution.state_execution import StateTaskContext

DataT = TypeVar("DataT")
TOKENWISE_REPLAY_IMPLEMENTATION = ImplementationId("canonical.tokenwise_layer_replay@1")


class TokenwiseReplayPrimitive(ABC):
    overlay: RouteOverlay | None = None
    overlay_storage: RouteOverlayStorage | None = None
    token_capture = False

    def run_region(self, invocation: RegionInvocation) -> tuple[torch.Tensor, ...]:
        """CPU models use the source interpreter with fresh invocation bindings."""
        return invocation.run()

    @property
    def function_bodies(self) -> tuple[RegionFunctionBody, ...]:
        return ()

    def model_endpoints(self) -> tuple[ExternalFunction, ...]:
        return ()

    @property
    def kernel_imports(self) -> tuple[NativeKernelImport, ...]:
        """Locally verified execution contracts supplied by this backend."""
        return ()

    @property
    @abstractmethod
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

    @abstractmethod
    def bind_kv(self, resource: BorrowedResource, owner: RegionPageOwner) -> PagedResourceBinding: ...

    @abstractmethod
    def bind_weights(self, resource: BorrowedResource, owner: RegionPageOwner) -> ReadonlyResourceBinding: ...

    @abstractmethod
    def bind_context(self, resource: BorrowedResource, owner: RegionPageOwner) -> HostResourceBinding: ...

    def bind_activation(self, resource: BorrowedResource, layer: int, owner: RegionPageOwner) -> PinnedResourceBinding:
        raise ValueError("model replay has no signed activation binding")

    def activation_selected(self, row: int, layer: int) -> bool:
        raise ValueError("model replay has no prepared activation selection")

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


@dataclass(frozen=True, slots=True)
class TokenwiseReplayTask(Generic[DataT]):
    task: SemanticTask
    region: TokenwiseRegion
    implementation_id: ImplementationId = TOKENWISE_REPLAY_IMPLEMENTATION
    policy: RegionSelectionPolicy = RegionSelectionPolicy()
    operation: TensorOperation[DataT] | None = None
    overlay_math: OverlayTransition | None = None

    @classmethod
    def from_implementation(
        cls, implementation: TaskGraphImplementation[StateTaskContext[DataT]]
    ) -> TokenwiseReplayTask[DataT]:
        if not isinstance(implementation, cls):
            raise ValueError("tokenwise replay requires its original native task binding")
        return implementation

    @classmethod
    def bind(
        cls,
        task: SemanticTask,
        source: TaskSourceRegion,
        cells: Mapping[str, RegionFunction],
        tensor: TensorType,
        max_rows: int,
        domain: CausalModelDomain,
        implementation_id: ImplementationId,
        policy: RegionSelectionPolicy,
        payload_budget: int,
        *,
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activations: tuple[DecoderActivation, ...] = (),
        operation: TensorOperation[DataT] | None = None,
        overlay_math: OverlayTransition | None = None,
    ) -> TokenwiseReplayTask[DataT]:
        effect = task.replay
        assert isinstance(effect, TokenProgramEffect)
        region = compile_tokenwise_source(
            tensor,
            max_rows,
            domain,
            effect,
            source,
            source_imports=source_imports,
            function_bodies=function_bodies,
            activations=activations,
        ).within_payload_budget(payload_budget)
        region = region.bind_functions(
            {
                name: cell
                for name, cell in cells.items()
                if name not in region.functions and name in {function.name for function in region.program.functions}
            }
        )
        return cls(task, region, implementation_id, policy, operation, overlay_math)

    def execute(self, context: StateTaskContext[DataT]) -> None:
        if self.operation is not None:
            replace(self.operation, body=self._overlay_body).execute(context)
        else:
            self.run(context)

    def _overlay_body(self, data: DataT, batch: TensorBatch, context: StateTaskContext[DataT] | None) -> TaskOutput:
        assert context is not None and isinstance(data, TensorFrame) and self.overlay_math is not None
        primitive = context.replay
        assert isinstance(primitive, TokenwiseReplayPrimitive)
        resource = next(resource for resource in self.region.program.resources if resource.name == "overlay")
        assert isinstance(resource, BorrowedResource)
        if primitive.overlay_storage is None:
            primitive.overlay_storage = RouteOverlayStorage(resource, batch.roots[0])
        primitive.overlay = RouteOverlay(self.overlay_math, batch, _read(context), primitive.overlay_storage, data)
        try:
            self.run(context)
            return primitive.overlay.finish()
        finally:
            primitive.overlay = None

    def run(self, context: StateTaskContext[DataT]) -> None:
        assert self.policy == context.arena.plan.timeline.region_policy
        primitive, effect = context.replay, self.task.replay
        assert isinstance(primitive, TokenwiseReplayPrimitive) and isinstance(effect, TokenProgramEffect)
        rows = primitive.rows
        task_id = TaskId(context.arena.plan.semantic_graph.tasks.index(self.task))

        overlay = primitive.overlay

        def begin(index: int) -> tuple[torch.Tensor, torch.Tensor]:
            if overlay is not None and not primitive.token_capture:
                overlay.stage(index, effect.rows == "all" or rows[index].phase.value == effect.rows)
            return primitive.begin(rows[index])

        def layer(
            index: int,
            request: int,
            token: int,
            ordinal: int,
            positions: torch.Tensor,
            hidden: torch.Tensor,
            residual: torch.Tensor | None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return primitive.layer(rows[index], ordinal, positions, hidden, residual)

        def normalize(index: int, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            return primitive.normalize(rows[index], hidden, residual)

        def observe(index: int, hidden: torch.Tensor) -> None:
            context.trace.observe(context.invocation_id, task_id, "replay.hidden", hidden, row=rows[index], iteration=0)

        def commit(index: int, record: Callable[[], None]) -> None:
            if overlay is not None and not primitive.token_capture:
                overlay.collect(tuple(overlay.results), overlay.captures())
            primitive.commit(rows[index], record)

        def apply_updates(inputs: tuple[Data, ...]) -> tuple[Data, ...]:
            assert overlay is not None
            overlay.apply_updates()
            return ()

        def record(index: int) -> None:
            context.trace.record(
                TracePhase.REPLAY, context.invocation_id, task_id=task_id, row=rows[index], iteration=0
            )

        with torch.no_grad(), nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
            outputs = self.region.run(
                context,
                tuple(effect.rows == "all" or row.phase.value == effect.rows for row in rows),
                tuple(row.request_ordinal for row in rows),
                tuple(row.token_offset for row in rows),
                kv=primitive.bind_kv(self.region.kv, context),
                weights=primitive.bind_weights(self.region.weights, context),
                context=primitive.bind_context(self.region.context, context),
                trace=HostResourceBinding(TraceResourceView(self.region.trace, context.trace), context),
                begin=begin,
                layer=layer,
                normalize=normalize,
                observe=observe,
                commit=commit,
                record=record,
                run_region=primitive.run_region,
                kernel_imports=primitive.kernel_imports,
                function_bodies=primitive.function_bodies,
                model_endpoints=(
                    *primitive.model_endpoints(),
                    *(
                        ()
                        if overlay is None
                        or not any(
                            function.name == "overlay.apply_updates@1" for function in self.region.program.functions
                        )
                        else (ExternalFunction(update_function(overlay.storage.resource), apply_updates),)
                    ),
                ),
                overlay_borrows=() if overlay is None else (TensorResourceBinding(overlay.storage, context),),
                activation_borrows=tuple(
                    primitive.bind_activation(resource, activation.binding.layer, context)
                    for resource, activation in zip(
                        self.region.activation_resources, self.region.activations, strict=True
                    )
                ),
                activation_selected=primitive.activation_selected,
            )
            primitive.finish(outputs)
