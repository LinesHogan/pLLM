"""Bounded replay in the original signed state invocation, without a version owner."""

from __future__ import annotations

import struct
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch

from pllm.contracts.backend_attachment import CausalModelDomain
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.region_composition import RegionFunctionBody
from pllm.contracts.region_imports import NativeKernelImport, RegionTensorImport
from pllm.contracts.region_program import BorrowedResource, TensorType
from pllm.contracts.region_selection import RegionSelectionPolicy
from pllm.contracts.semantic_graph import (
    ConditionalLayerReplay,
    ImplementationId,
    SemanticTask,
    TaskId,
    TokenwiseLayerReplay,
)
from pllm.contracts.state_ref import StateRef
from pllm.execution.decoder_region import DecoderActivation
from pllm.execution.host_borrow import TraceResourceView
from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.region_execution import RegionInvocation
from pllm.execution.region_replay import ConditionalReplayRegion
from pllm.execution.region_resources import (
    ExternalFunction,
    HostResourceBinding,
    PagedResourceBinding,
    PinnedResourceBinding,
    ReadonlyResourceBinding,
    RegionPageOwner,
)
from pllm.execution.region_streams import RegionStreamBindings
from pllm.execution.tokenwise_region import (
    TokenwiseRegion,
    compile_tokenwise_region,
    mix_layer_outputs as mix_layer_outputs,
)
from pllm.execution.trace import TracePhase

if TYPE_CHECKING:
    from pllm.execution.execution import TaskGraphImplementation
    from pllm.execution.state_execution import StateTaskContext

DataT = TypeVar("DataT")
REPLAY_IMPLEMENTATION = ImplementationId("canonical.layer_replay@1")
TOKENWISE_REPLAY_IMPLEMENTATION = ImplementationId("canonical.tokenwise_layer_replay@1")


@dataclass(frozen=True, slots=True)
class Recurrence:
    source_layer: int = -1
    destination_layer: int = 0
    alpha: float = 0.15
    beta: float | None = 0.85
    ramp_tokens: int = 0
    enabled: bool = True

    def __post_init__(self) -> None:
        if (
            not isfinite(self.alpha)
            or not 0 <= self.alpha <= 1
            or self.ramp_tokens < 0
            or self.beta is not None
            and (not isfinite(self.beta) or not 0 <= self.beta <= 1)
        ):
            raise ValueError("recurrence weights must be finite unit-interval scalars with nonnegative ramp")

    def effect(self, layer_count: int) -> TokenwiseLayerReplay:
        source = self.source_layer if self.source_layer >= 0 else layer_count + self.source_layer
        destination = self.destination_layer if self.destination_layer >= 0 else layer_count + self.destination_layer
        return TokenwiseLayerReplay(
            source,
            destination,
            layer_count,
            struct.unpack("!Q", struct.pack("!d", self.alpha))[0],
            None if self.beta is None else struct.unpack("!Q", struct.pack("!d", self.beta))[0],
            self.ramp_tokens,
            self.enabled,
        )


class TokenwiseReplayPrimitive(ABC):
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

    def borrow_streams(self, owner: RegionPageOwner, lanes: int) -> RegionStreamBindings:
        """Backend capability requested only by a selected physical plan."""
        raise ValueError("this model backend has no auxiliary stream binding")

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

    @classmethod
    def from_implementation(
        cls, implementation: TaskGraphImplementation[StateTaskContext[DataT]]
    ) -> TokenwiseReplayTask[DataT]:
        if not isinstance(implementation, cls):
            raise ValueError("tokenwise tuning requires its original native task binding")
        return implementation

    @classmethod
    def bind(
        cls,
        task: SemanticTask,
        tensor: TensorType,
        max_rows: int,
        domain: CausalModelDomain,
        implementation_id: ImplementationId,
        policy: RegionSelectionPolicy,
        payload_budget: int,
        *,
        stream_lanes: int = 1,
        token_capture: bool = False,
        source_imports: tuple[RegionTensorImport, ...] = (),
        function_bodies: tuple[RegionFunctionBody, ...] = (),
        activations: tuple[DecoderActivation, ...] = (),
    ) -> TokenwiseReplayTask[DataT]:
        effect = task.replay
        if not isinstance(effect, TokenwiseLayerReplay):
            raise ValueError("tokenwise region requires its declared recurrence")
        region = compile_tokenwise_region(
            tensor,
            max_rows,
            domain,
            effect,
            policy,
            stream_lanes=stream_lanes,
            token_capture=token_capture,
            source_imports=source_imports,
            function_bodies=function_bodies,
            activations=activations,
        ).within_payload_budget(payload_budget)
        if set(region.functions) == {function.name for function in region.program.functions}:
            region = region.bind_functions({})
        return cls(task, region, implementation_id, policy)

    def execute(self, context: StateTaskContext[DataT]) -> None:
        if self.policy != context.arena.plan.timeline.region_policy:
            raise ValueError("tokenwise schedule policy differs from the original signed timeline")
        primitive, effect = context.replay, self.task.replay
        if not isinstance(primitive, TokenwiseReplayPrimitive) or not isinstance(effect, TokenwiseLayerReplay):
            raise RuntimeError("tokenwise replay requires its original model invocation")
        rows = primitive.rows
        task_id = TaskId(context.arena.plan.semantic_graph.tasks.index(self.task))

        def begin(index: int) -> tuple[torch.Tensor, torch.Tensor]:
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
            if (request, token) != (rows[index].request_ordinal, rows[index].token_offset):
                raise ValueError("layer call differs from its original packed KV coordinates")
            return primitive.layer(rows[index], ordinal, positions, hidden, residual)

        def normalize(index: int, hidden: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            return primitive.normalize(rows[index], hidden, residual)

        def observe(index: int, hidden: torch.Tensor) -> None:
            context.trace.observe(context.invocation_id, task_id, "replay.hidden", hidden, row=rows[index], iteration=0)

        def commit(index: int, record: Callable[[], None]) -> None:
            primitive.commit(rows[index], record)

        def record(index: int) -> None:
            context.trace.record(
                TracePhase.REPLAY, context.invocation_id, task_id=task_id, row=rows[index], iteration=0
            )

        physical = self.region.stream_plan
        streams = None if physical is None else primitive.borrow_streams(context, physical.lanes)
        with torch.no_grad(), nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
            outputs = self.region.run(
                context,
                tuple(effect.enabled and row.phase is HiddenPhase.PREFILL for row in rows),
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
                streams=streams,
                run_region=primitive.run_region,
                kernel_imports=primitive.kernel_imports,
                function_bodies=primitive.function_bodies,
                model_endpoints=primitive.model_endpoints(),
                activation_borrows=tuple(
                    primitive.bind_activation(resource, activation.binding.layer, context)
                    for resource, activation in zip(
                        self.region.activation_resources, self.region.activations, strict=True
                    )
                ),
                activation_selected=primitive.activation_selected,
            )
            primitive.finish(outputs)


class LayerReplayPrimitive(Protocol):
    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

    def seed(self, row: PackedPhaseRow) -> torch.Tensor: ...

    def step(
        self, row: PackedPhaseRow, carry: torch.Tensor, arena: PageBundleArena, pin: PagePin, ref: StateRef
    ) -> torch.Tensor: ...

    def restore(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReplayStateTask(Generic[DataT]):
    task: SemanticTask
    region: ConditionalReplayRegion
    implementation_id: ImplementationId = REPLAY_IMPLEMENTATION

    def execute(self, context: StateTaskContext[DataT]) -> None:
        # Conditional replay is explicitly eager and already observes GPU convergence.
        if context.updated_mask is not None:
            selected = tuple(bool(value) for value in context.updated_mask.cpu().unbind())
            context.updated_refs = tuple(
                ref for ref, active in zip(context.updated_refs, selected, strict=True) if active
            )
        publication = context.publication
        if publication is None or not context.updated_refs:
            return
        effect, primitive = self.task.replay, context.replay
        if (
            not isinstance(effect, ConditionalLayerReplay)
            or primitive is None
            or isinstance(primitive, TokenwiseReplayPrimitive)
        ):
            raise RuntimeError("signed replay requires its bound model primitive")
        if context.replay_started:
            raise RuntimeError("published replay cannot be retried; settle the original invocation")
        context.replay_started = True
        pin = context.borrow_published(context.updated_refs)
        try:
            with (
                torch.no_grad(),
                nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream),
            ):
                for row in primitive.rows:
                    before = context.selection.ref_for(row.identity)
                    if before not in publication.before_refs:
                        continue
                    ref = publication.after_refs[publication.before_refs.index(before)]
                    if ref not in context.updated_refs:
                        continue

                    def step(carry: torch.Tensor) -> torch.Tensor:
                        return primitive.step(row, carry, context.arena, pin, ref)

                    def observe(result: torch.Tensor, iteration: int) -> None:
                        context.trace.observe(
                            context.invocation_id,
                            TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)),
                            "replay.carry",
                            result,
                            row=row,
                            iteration=iteration,
                            refs=(ref,),
                        )

                    def record(iteration: int, relative: float | None) -> None:
                        context.trace.record(
                            TracePhase.REPLAY,
                            context.invocation_id,
                            task_id=TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)),
                            refs=(ref,),
                            operation_id=publication.operation_id,
                            pin_id=pin.pin_id,
                            row=row,
                            iteration=iteration,
                            relative_l2=relative,
                        )

                    self.region.run(context, primitive.seed(row).detach(), step=step, observe=observe, record=record)
                if context.cancel_requested.is_set():
                    raise RuntimeError("conditional replay cancelled before completion")
        except Exception:
            with nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
                primitive.restore()
            raise
