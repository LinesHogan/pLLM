"""Bounded replay in the original signed state invocation, without a version owner."""

from __future__ import annotations

from contextlib import nullcontext
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isfinite
import struct
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import torch

from pllm.execution.ptts.page_bundle import PageBundleArena, PagePin
from pllm.execution.trace import TracePhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.semantic_graph import ConditionalLayerReplay, ImplementationId, SemanticTask, TaskId, TokenwiseLayerReplay
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.state_ref import StateRef

if TYPE_CHECKING:
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
        if (not isfinite(self.alpha) or not 0 <= self.alpha <= 1 or self.ramp_tokens < 0
            or self.beta is not None and (not isfinite(self.beta) or not 0 <= self.beta <= 1)):
            raise ValueError("recurrence weights must be finite unit-interval scalars with nonnegative ramp")

    def effect(self, layer_count: int) -> TokenwiseLayerReplay:
        source = self.source_layer if self.source_layer >= 0 else layer_count + self.source_layer
        destination = self.destination_layer if self.destination_layer >= 0 else layer_count + self.destination_layer
        return TokenwiseLayerReplay(source, destination, layer_count,
            struct.unpack("!Q", struct.pack("!d", self.alpha))[0],
            None if self.beta is None else struct.unpack("!Q", struct.pack("!d", self.beta))[0], self.ramp_tokens, self.enabled)


class TokenwiseReplayPrimitive(ABC):
    @property
    @abstractmethod
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

    @abstractmethod
    def normal(self, row: PackedPhaseRow) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...

    @abstractmethod
    def replay(self, row: PackedPhaseRow, hidden: torch.Tensor) -> None: ...

    @abstractmethod
    def restore(self) -> None: ...

    @abstractmethod
    def finish(self, outputs: tuple[torch.Tensor, ...]) -> None: ...


def mix_layer_outputs(effect: TokenwiseLayerReplay, source: torch.Tensor,
                      destination: torch.Tensor, position: int) -> torch.Tensor:
    alpha = struct.unpack("!d", struct.pack("!Q", effect.alpha_bits))[0]
    beta = None if effect.beta_bits is None else struct.unpack("!d", struct.pack("!Q", effect.beta_bits))[0]
    if effect.ramp_tokens:
        alpha *= min(position, effect.ramp_tokens) / effect.ramp_tokens
    source_float, destination_float = source.float(), destination.float()
    normalized = source_float / source_float.square().sum(-1, keepdim=True).sqrt().clamp_min(torch.finfo(torch.float32).tiny)
    normalized *= destination_float.square().sum(-1, keepdim=True).sqrt()
    return ((1.0 - alpha if beta is None else beta) * destination_float + alpha * normalized).to(destination.dtype)


@dataclass(frozen=True, slots=True)
class TokenwiseReplayTask(Generic[DataT]):
    task: SemanticTask
    implementation_id: ImplementationId = TOKENWISE_REPLAY_IMPLEMENTATION

    def execute(self, context: StateTaskContext[DataT]) -> None:
        primitive, effect = context.replay, self.task.replay
        if not isinstance(primitive, TokenwiseReplayPrimitive) or not isinstance(effect, TokenwiseLayerReplay):
            raise RuntimeError("tokenwise replay requires its original model invocation")
        outputs: list[torch.Tensor] = []
        with torch.no_grad(), (nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream)):
            for row in primitive.rows:
                if context.cancel_requested.is_set():
                    raise RuntimeError("tokenwise replay cancelled between tokens")
                try:
                    normal, source, destination = primitive.normal(row)
                    outputs.append(normal.detach())
                    if effect.enabled and row.phase is HiddenPhase.PREFILL:
                        hidden = mix_layer_outputs(effect, source, destination, row.token_offset)
                        context.trace.observe(context.invocation_id,
                            TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)),
                            "replay.hidden", hidden, row=row, iteration=0)
                        primitive.replay(row, hidden.detach())
                        context.trace.record(TracePhase.REPLAY, context.invocation_id,
                            task_id=TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)), row=row, iteration=0)
                except Exception:
                    primitive.restore()
                    raise
            primitive.finish(tuple(outputs))


class LayerReplayPrimitive(Protocol):
    @property
    def rows(self) -> tuple[PackedPhaseRow, ...]: ...

    def seed(self, row: PackedPhaseRow) -> torch.Tensor: ...

    def step(self, row: PackedPhaseRow, carry: torch.Tensor, arena: PageBundleArena,
             pin: PagePin, ref: StateRef) -> torch.Tensor: ...

    def restore(self) -> None: ...


@dataclass(frozen=True, slots=True)
class ReplayStateTask(Generic[DataT]):
    task: SemanticTask
    implementation_id: ImplementationId = REPLAY_IMPLEMENTATION

    def execute(self, context: StateTaskContext[DataT]) -> None:
        publication = context.publication
        if publication is None or not context.updated_refs:
            return
        effect, primitive = self.task.replay, context.replay
        if (not isinstance(effect, ConditionalLayerReplay) or primitive is None
            or isinstance(primitive, TokenwiseReplayPrimitive)):
            raise RuntimeError("signed replay requires its bound model primitive")
        if context.replay_started:
            raise RuntimeError("published replay cannot be retried; settle the original invocation")
        context.replay_started = True
        pin = context.borrow_published(context.updated_refs)
        try:
            with torch.no_grad(), (nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream)):
                for row in primitive.rows:
                    before = context.selection.ref_for(row.identity)
                    if before not in publication.before_refs:
                        continue
                    ref = publication.after_refs[publication.before_refs.index(before)]
                    if ref not in context.updated_refs:
                        continue
                    carry = primitive.seed(row).detach()
                    previous: torch.Tensor | None = None
                    for iteration in range(effect.max_iterations):
                        if context.cancel_requested.is_set():
                            raise RuntimeError("conditional replay cancelled between iterations")
                        result = primitive.step(row, carry, context.arena, pin, ref).detach()
                        context.trace.observe(context.invocation_id,
                            TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)),
                            "replay.carry", result, row=row, iteration=iteration, refs=(ref,))
                        relative = None if previous is None else float(((result.float() - previous.float()).square().sum().sqrt()
                            / previous.float().square().sum().sqrt().clamp_min(1e-12)).item())
                        if not torch.isfinite(result).all().item():
                            raise RuntimeError("conditional replay produced nonfinite carry")
                        context.trace.record(TracePhase.REPLAY, context.invocation_id,
                            task_id=TaskId(context.arena.plan.semantic_graph.tasks.index(self.task)),
                            refs=(ref,), operation_id=publication.operation_id, pin_id=pin.pin_id,
                            row=row, iteration=iteration, relative_l2=relative)
                        carry = previous = result
                        if relative is not None and relative <= effect.relative_l2_ppm / 1_000_000:
                            break
                if context.cancel_requested.is_set():
                    raise RuntimeError("conditional replay cancelled before completion")
        except Exception:
            with nullcontext() if context.pin.stream is None else torch.cuda.stream(context.pin.stream):
                primitive.restore()
            raise
