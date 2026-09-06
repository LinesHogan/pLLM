"""One signed invocation, sampler lease and settlement lifecycle for tensor state."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from enum import Enum, auto
from threading import RLock
from typing import Generic, Protocol, TypeVar

import torch

from tllm.canonical.cuda_stream import CudaStreamReadiness
from tllm.canonical.execution import TaskGraphExecution, TaskGraphImplementation, TaskGraphInvocation
from tllm.canonical.ptts.page_bundle import PageBundleArena, PagePin, PagePublication, PageRelease, PageWrite
from tllm.canonical.ptts.request_state import RequestStateDirectory, RequestStateSelection, RequestStateTransitionLease
from tllm.contracts.request_identity import RequestIdentity
from tllm.contracts.request_state import IndexedRequestIdentityBatch, RequestStateKey
from tllm.contracts.sampler import PackedSamplerGuidanceBatch
from tllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan, StateScope
from tllm.contracts.state_ref import StateRef
from tllm.contracts.state_representation import StatePhysicalPlan


InputT = TypeVar("InputT")
InputContra = TypeVar("InputContra", contravariant=True)
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
ResultCo = TypeVar("ResultCo", covariant=True)


@dataclass(slots=True)
class StateTaskContext(Generic[DataT]):
    selection: RequestStateSelection
    transition_lease: RequestStateTransitionLease
    arena: PageBundleArena
    pin: PagePin
    value: DataT | None = None
    write: PageWrite | None = None
    publication: PagePublication | None = None
    release: PageRelease | None = None
    publication_settled: bool = False
    guidance: PackedSamplerGuidanceBatch | None = None
    guidance_ready: CudaStreamReadiness | None = None
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0

    def data(self) -> DataT:
        if self.value is None:
            raise RuntimeError("state invocation requires its input")
        return self.value

    def publish(self) -> None:
        if self.write is not None:
            if self.publication is not None:
                raise RuntimeError("state invocation already owns its publication receipt")
            self.publication = self.arena.publish(self.write)
            self.write = None


class StateMath(Protocol[InputContra, DataT, ResultCo]):
    """Input validation and tensor mathematics, never invocation ownership."""

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...]) -> None: ...

    def rows(self, value: InputContra) -> IndexedRequestIdentityBatch: ...

    def prepare(self, value: InputContra, context: StateTaskContext[DataT]) -> DataT: ...

    def output(self, context: StateTaskContext[DataT]) -> ResultCo: ...


@dataclass(frozen=True, slots=True)
class PublishStateTask(Generic[DataT]):
    implementation_id: ImplementationId

    def execute(self, context: StateTaskContext[DataT]) -> None:
        context.publish()


class _SamplerPhase(Enum):
    PENDING = auto()
    BORROWED = auto()
    COMPLETE = auto()


@dataclass(slots=True, eq=False)
class StateInvocation(Generic[DataT]):
    invocation_id: int
    cursor: TaskGraphInvocation[StateTaskContext[DataT]]
    sampler_phase: _SamplerPhase = _SamplerPhase.PENDING
    sampler_stream: torch.cuda.Stream | None = None


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    active_lineages: int
    loss_count: int
    update_count: int
    trained_rows: int
    completed_sampler_pins: int
    pending_state_releases: int
    fallback_count: int


class StateSettlementError(RuntimeError, Generic[DataT]):
    def __init__(self, invocation: StateInvocation[DataT], receipt: PagePublication) -> None:
        super().__init__("state publication committed; settle the original invocation without replay")
        self.invocation = invocation
        self.receipt = receipt


class StateExecution(Generic[InputT, DataT, ResultT]):
    def __init__(self, arena: PageBundleArena, directory: RequestStateDirectory,
                 math: StateMath[InputT, DataT, ResultT],
                 implementations: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...],
                 sampler_anchor_id: AnchorId) -> None:
        self.arena, self.directory, self.math = arena, directory, math
        self.sampler_anchor_id = sampler_anchor_id
        self._runner = TaskGraphExecution(arena.plan.schedule_physical, implementations)
        self._active: dict[int, StateInvocation[DataT]] = {}
        self._next_invocation = 1
        self._loss_count = self._update_count = self._trained_rows = self._sampler_count = 0
        self._lock = RLock()

    @property
    def plan(self) -> PhysicalPlan:
        return self.arena.plan.schedule_physical

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self.arena.plan.state_physical

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def admit(self, identities: tuple[RequestIdentity, ...],
              stream: torch.cuda.Stream | None = None) -> tuple[StateRef, ...]:
        with self._lock:
            batch = self.directory.prepare_admission(identities, stream)
            try:
                with nullcontext() if stream is None else torch.cuda.stream(stream):
                    self.math.initialize(self.arena.writable_roots(batch.write))
                receipt = self.directory.commit_admission(batch)
            except Exception:
                self.directory.abort_admission(batch)
                raise
            self.arena.settle(receipt)
            return receipt.after_refs

    def begin(self, rows: IndexedRequestIdentityBatch,
              stream: torch.cuda.Stream | None = None) -> StateInvocation[DataT]:
        with self._lock:
            self.arena.collect_completed()
            selection = self.directory.select(rows)
            lease = self.directory.acquire_transition_lease(selection)
            try:
                pin = self.arena.pin_exact(selection.refs, stream)
            except Exception:
                self.directory.release_transition_lease(lease)
                raise
            context = StateTaskContext[DataT](selection, lease, self.arena, pin)
            invocation = StateInvocation(self._next_invocation, self._runner.start(context))
            self._next_invocation += 1
            self._active[invocation.invocation_id] = invocation
            return invocation

    def start_step(self, value: InputT, stream: torch.cuda.Stream | None = None) -> StateInvocation[DataT]:
        with self._lock:
            invocation = self.begin(self.math.rows(value), stream)
            self.submit_input(invocation, value)
            return invocation

    def step(self, value: InputT, stream: torch.cuda.Stream | None = None) -> ResultT:
        """Run without an external sampler borrower using the same staged lifecycle."""
        with self._lock:
            invocation = self.start_step(value, stream)
            for anchor in self.arena.plan.timeline.anchors:
                self.execute_anchor(invocation, AnchorId(anchor.ordinal))
                if anchor.ordinal == self.sampler_anchor_id:
                    self.complete_sampler(invocation, stream)
            return self.finish_step(invocation)

    def submit_input(self, invocation: StateInvocation[DataT], value: InputT,
                     readiness: CudaStreamReadiness | None = None) -> None:
        """Consume original inputs, ordering an external CUDA producer when supplied."""
        with self._lock:
            context = self._context(invocation)
            if context.value is not None:
                raise RuntimeError("state invocation input is already supplied")
            try:
                if readiness is not None:
                    if context.pin.stream is None:
                        raise ValueError("CUDA input readiness requires a CUDA invocation")
                    readiness.order_on(context.pin.stream)
                if self.directory.select(self.math.rows(value)) != context.selection:
                    raise ValueError("state invocation input differs from its selected rows")
                context.value = self.math.prepare(value, context)
            except Exception:
                self.cancel_step(invocation)
                raise

    def execute_anchor(self, invocation: StateInvocation[DataT], anchor_id: AnchorId) -> None:
        with self._lock:
            context = self._context(invocation)
            if anchor_id > self.sampler_anchor_id and invocation.sampler_phase is not _SamplerPhase.COMPLETE:
                raise RuntimeError("later anchors require sampler completion")
            try:
                self._runner.execute_anchor(invocation.cursor, anchor_id)
            except Exception as exc:
                self._failed(invocation, context, exc)

    def sampler_guidance(self, invocation: StateInvocation[DataT],
                         stream: torch.cuda.Stream | None = None) -> PackedSamplerGuidanceBatch:
        with self._lock:
            context = self._context(invocation)
            if context.guidance is None:
                raise RuntimeError("signed tasks have not produced sampler guidance")
            if invocation.sampler_phase is _SamplerPhase.COMPLETE or (context.pin.stream is None) != (stream is None):
                raise RuntimeError("sampler guidance stream or phase differs from the invocation")
            if stream is not None:
                if stream.device != self.device:
                    raise RuntimeError("sampler stream or phase differs from the invocation")
                if invocation.sampler_phase is _SamplerPhase.BORROWED and invocation.sampler_stream != stream:
                    raise RuntimeError("guidance is already borrowed by another sampler stream")
                if context.guidance_ready is None:
                    raise RuntimeError("CUDA guidance requires its original readiness")
                context.guidance_ready.order_on(stream)
                invocation.sampler_phase, invocation.sampler_stream = _SamplerPhase.BORROWED, stream
            return context.guidance

    def complete_sampler(self, invocation: StateInvocation[DataT],
                         stream: torch.cuda.Stream | None = None) -> None:
        with self._lock:
            context = self._context(invocation)
            if invocation.cursor.last_anchor_id < self.sampler_anchor_id:
                raise RuntimeError("sampler completion precedes its signed anchor")
            if invocation.sampler_phase is _SamplerPhase.COMPLETE:
                raise RuntimeError("sampler completion is already recorded")
            if stream is not None and stream.device != self.device:
                raise RuntimeError("sampler stream differs from the signed device")
            if context.pin.stream is not None and stream is None:
                raise RuntimeError("CUDA sampler completion requires its original stream")
            if invocation.sampler_phase is _SamplerPhase.BORROWED and invocation.sampler_stream != stream:
                raise RuntimeError("sampler completion differs from the borrowed stream")
            self._order_sampler(invocation, stream)
            invocation.sampler_phase = _SamplerPhase.COMPLETE
            self._sampler_count += 1

    def finish_step(self, invocation: StateInvocation[DataT]) -> ResultT:
        with self._lock:
            context = self._context(invocation)
            if invocation.sampler_phase is not _SamplerPhase.COMPLETE:
                raise RuntimeError("state invocation requires sampler completion")
            try:
                result = self.math.output(context)
                if not invocation.cursor.finished:
                    self._runner.finish(invocation.cursor)
                self._release(invocation, context)
            except Exception as exc:
                self._failed(invocation, context, exc)
                raise
            return result

    def cancel_step(self, invocation: StateInvocation[DataT]) -> None:
        with self._lock:
            context = self._context(invocation)
            self._order_sampler(invocation, invocation.sampler_stream)
            if context.write is not None:
                self.arena.abort(context.write)
                context.write = None
            self._release(invocation, context)

    def cancel_active(self) -> None:
        with self._lock:
            for invocation in tuple(self._active.values()):
                self.cancel_step(invocation)

    def _failed(self, invocation: StateInvocation[DataT], context: StateTaskContext[DataT], exc: Exception) -> None:
        if context.publication is not None:
            raise StateSettlementError(invocation, context.publication) from exc
        self.cancel_step(invocation)
        raise exc

    def _release(self, invocation: StateInvocation[DataT], context: StateTaskContext[DataT]) -> None:
        if context.release is None:
            context.release = self.arena.release_pin(context.pin)
        if context.publication is not None and not context.publication_settled:
            self.arena.settle(context.publication)
            context.publication_settled = True
        self.directory.release_transition_lease(context.transition_lease)
        self._active.pop(invocation.invocation_id)
        if context.publication is not None:
            self._loss_count += context.loss_count
            self._update_count += context.update_count
            self._trained_rows += context.trained_rows

    def _order_sampler(self, invocation: StateInvocation[DataT], stream: torch.cuda.Stream | None) -> None:
        consumer = invocation.cursor.context.pin.stream
        if stream is not None and consumer is not None:
            CudaStreamReadiness(stream).order_on(consumer)

    def _context(self, invocation: StateInvocation[DataT]) -> StateTaskContext[DataT]:
        if self._active.get(invocation.invocation_id) is not invocation:
            raise RuntimeError("state invocation is not registered as active")
        return invocation.cursor.context

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            if self.directory.state.scope is StateScope.INSTANCE and identities:
                raise ValueError("instance retirement is not a request lifecycle action")
            keys = ({None} if self.directory.state.scope is StateScope.INSTANCE else
                {RequestStateKey.project(identity, self.directory.state.key_fields) for identity in identities})
            for invocation in tuple(self._active.values()):
                if keys.intersection(invocation.cursor.context.selection.keys):
                    self.cancel_step(invocation)
            self.arena.collect_completed(wait=True)
            return self.directory.retire(identities)

    def snapshot(self, ref: StateRef, stream: torch.cuda.Stream | None = None) -> tuple[torch.Tensor, ...]:
        with self._lock:
            pin = self.arena.pin_exact((ref,), stream)
            try:
                return self.arena.snapshot(pin, ref)
            finally:
                self.arena.wait_release(self.arena.release_pin(pin))

    def diagnostics(self) -> StateDiagnostics:
        with self._lock:
            self.arena.collect_completed()
            return StateDiagnostics(self.directory.active_count, self._loss_count, self._update_count,
                self._trained_rows, self._sampler_count, self.arena.pending_pin_releases, self.plan.fallback_count)
