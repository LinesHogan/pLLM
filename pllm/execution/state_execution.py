"""One signed invocation, sampler lease and settlement lifecycle for tensor state."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from enum import Enum, auto
from threading import Event, RLock
from typing import Generic, Protocol, TypeVar

import torch

from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.execution import TaskGraphExecution, TaskGraphImplementation, TaskGraphInvocation
from pllm.execution.ptts.page_bundle import KernelCapture, PageBundleArena, PageBundleUsage, PagePin, PagePublication, PageRelease, PageWrite
from pllm.execution.ptts.request_state import RequestStateDirectory, RequestStateSelection
from pllm.execution.replay import LayerReplayPrimitive, TokenwiseReplayPrimitive
from pllm.execution.trace import ExecutionTrace, TracePhase
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan, StateScope, SamplerFeedbackInputSource, TaskSymbol
from pllm.contracts.state_ref import StateRef
from pllm.contracts.state_representation import StatePhysicalPlan


InputT = TypeVar("InputT", contravariant=True)
InputContra = TypeVar("InputContra", contravariant=True)
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
ResultCo = TypeVar("ResultCo", covariant=True)


class _SamplerPhase(Enum):
    PENDING = auto()
    BORROWED = auto()
    COMPLETE = auto()


@dataclass(slots=True)
class StateTaskContext(Generic[DataT]):
    selection: RequestStateSelection
    arena: PageBundleArena
    pin: PagePin
    trace: ExecutionTrace
    invocation_id: int
    value: DataT | None = None
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]())
    write: PageWrite | None = None
    publication: PagePublication | None = None
    release: PageRelease | None = None
    publication_settled: bool = False
    guidance: PackedSamplerGuidanceBatch | None = None
    loss: torch.Tensor | None = None
    guidance_ready: CudaStreamReadiness | None = None
    sampler_feedback: SamplerFeedback | None = None
    sampler_phase: _SamplerPhase = _SamplerPhase.PENDING
    sampler_stream: torch.cuda.Stream | None = None
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0
    replay: LayerReplayPrimitive | TokenwiseReplayPrimitive | None = None
    updated_refs: tuple[StateRef, ...] = ()
    published_pin: PagePin | None = None
    published_release: PageRelease | None = None
    replay_started: bool = False
    cancel_requested: Event = field(default_factory=Event)

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
            self.trace.record(TracePhase.PUBLISH, self.invocation_id, refs=self.publication.after_refs,
                operation_id=self.publication.operation_id)

    def borrow_published(self, refs: tuple[StateRef, ...]) -> PagePin:
        if self.publication is None or not set(refs).issubset(self.publication.after_refs):
            raise ValueError("published borrow requires this invocation's original receipt")
        if self.published_pin is None:
            self.published_pin = self.arena.pin_exact(self.publication.after_refs, self.pin.stream)
        elif not set(refs).issubset(self.published_pin.refs):
            raise ValueError("invocation already borrowed a different publication selection")
        return self.published_pin


class StateMath(Protocol[InputContra, DataT, ResultCo]):
    """Input validation and tensor mathematics, never invocation ownership."""

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...], /) -> None: ...

    def rows(self, value: InputContra, /) -> tuple[PackedPhaseRow, ...]: ...

    def prepare(self, value: InputContra, context: StateTaskContext[DataT]) -> DataT: ...

    def output(self, context: StateTaskContext[DataT]) -> ResultCo: ...


@dataclass(frozen=True, slots=True)
class MathBinding(Generic[InputT, DataT, ResultT]):
    """Local native implementations, separate from the closed data contracts."""

    math: StateMath[InputT, DataT, ResultT]
    tasks: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...]
    captures: tuple[KernelCapture, ...] = ()

    def bind(self, arena: PageBundleArena, sampler_anchor_id: AnchorId) -> StateExecution[InputT, DataT, ResultT]:
        # Reject any underfunded kernel before allocating/capturing the first one's inputs.
        for capture in self.captures:
            capture.check_inputs(arena.plan, capture.inputs(torch.device("meta")))
        for capture in self.captures:
            capture.bind(arena)
        states = arena.plan.semantic_graph.states
        if len(states) > 1:
            raise ValueError("tensor execution requires one logical state with any number of fields")
        return StateExecution(arena, RequestStateDirectory(arena, states[0] if states else None),
            self.math, self.tasks, sampler_anchor_id)


@dataclass(frozen=True, slots=True)
class PublishStateTask(Generic[DataT]):
    implementation_id: ImplementationId

    def execute(self, context: StateTaskContext[DataT]) -> None:
        context.publish()


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    active_lineages: int
    inference_pins: int
    loss_count: int
    update_count: int
    trained_rows: int
    completed_sampler_pins: int
    pending_state_releases: int
    fallback_count: int
    state_usage: PageBundleUsage


class StateSettlementError(RuntimeError, Generic[DataT]):
    def __init__(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], receipt: PagePublication) -> None:
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
        self.trace = ExecutionTrace(arena.plan.schedule_physical)
        self._values = tuple((task.symbol, output.name) for task in arena.plan.semantic_graph.tasks for output in task.outputs)
        self._runner = TaskGraphExecution(arena.plan.schedule_physical, implementations, self.trace, self._release_values)
        self._active: dict[int, TaskGraphInvocation[StateTaskContext[DataT]]] = {}
        self._next_invocation = 1
        self._loss_count = self._update_count = self._trained_rows = self._sampler_count = 0
        self._lock = RLock()

    @property
    def plan(self) -> PhysicalPlan:
        return self.arena.plan.schedule_physical

    def _release_values(self, context: StateTaskContext[DataT], values: tuple[int, ...]) -> None:
        for value in values:
            context.results.pop(self._values[value], None)

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self.arena.plan.state_physical

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def admit(self, identities: tuple[RequestIdentity, ...],
              stream: torch.cuda.Stream | None = None) -> tuple[StateRef, ...]:
        with self._lock:
            if self.directory.state is None:
                return ()
            if self.directory.scope is StateScope.INSTANCE and identities:
                refs = self.directory.admit_inference(identities, stream)
                self.trace.record(TracePhase.ADMIT, 0, identities=identities, refs=refs)
                return refs
            batch = self.directory.prepare_admission(identities, stream)
            try:
                if batch.write is not None:
                    with nullcontext() if stream is None else torch.cuda.stream(stream):
                        self.math.initialize(self.arena.writable_roots(batch.write))
                receipt = self.directory.commit_admission(batch)
            except Exception:
                self.directory.abort_admission(batch)
                raise
            if receipt is not None:
                self.arena.settle(receipt)
            # One visible ref per unique admitted key, whether created or joined.
            refs = self.directory.latest_refs(identities)
            if receipt is None:
                self.trace.record(TracePhase.ADMIT, 0, identities=identities, refs=refs)
            else:
                self.trace.record(TracePhase.ADMIT, 0, identities=identities, refs=refs,
                    operation_id=receipt.operation_id)
            return refs

    def begin(self, rows: tuple[PackedPhaseRow, ...],
              stream: torch.cuda.Stream | None = None) -> TaskGraphInvocation[StateTaskContext[DataT]]:
        with self._lock:
            self.arena.collect_completed()
            selection = self.directory.select(rows)
            self.directory.acquire_transition_lease(selection)
            try:
                pin = self.arena.pin_exact(selection.refs, stream)
            except Exception:
                self.directory.release_transition_lease(selection)
                raise
            context = StateTaskContext[DataT](selection, self.arena, pin, self.trace, self._next_invocation)
            invocation = TaskGraphInvocation(self._runner, context, self._next_invocation)
            self._next_invocation += 1
            self._active[invocation.invocation_id] = invocation
            self.trace.record(TracePhase.BEGIN, invocation.invocation_id, refs=pin.refs, pin_id=pin.pin_id,
                identities=tuple(row.identity for row in selection.rows))
            return invocation

    def start_step(self, value: InputT, stream: torch.cuda.Stream | None = None) -> TaskGraphInvocation[StateTaskContext[DataT]]:
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

    def submit_input(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], value: InputT,
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
                self.trace.record(TracePhase.FAILED, invocation.invocation_id)
                self.cancel_step(invocation)
                raise

    def execute_anchor(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], anchor_id: AnchorId) -> None:
        with self._lock:
            context = self._context(invocation)
            if anchor_id > self.sampler_anchor_id and context.sampler_phase is not _SamplerPhase.COMPLETE:
                raise RuntimeError("later anchors require sampler completion")
            try:
                invocation.execute_anchor(anchor_id, context.pin.stream)
            except Exception as exc:
                self._failed(invocation, context, exc)

    def sampler_guidance(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]],
                         stream: torch.cuda.Stream | None = None) -> PackedSamplerGuidanceBatch:
        with self._lock:
            context = self._context(invocation)
            if context.guidance is None:
                raise RuntimeError("signed tasks have not produced sampler guidance")
            if context.sampler_phase is _SamplerPhase.COMPLETE or (context.pin.stream is None) != (stream is None):
                raise RuntimeError("sampler guidance stream or phase differs from the invocation")
            if stream is not None:
                if stream.device != self.device:
                    raise RuntimeError("sampler stream or phase differs from the invocation")
                if context.sampler_phase is _SamplerPhase.BORROWED and context.sampler_stream != stream:
                    raise RuntimeError("guidance is already borrowed by another sampler stream")
                if context.guidance_ready is None:
                    raise RuntimeError("CUDA guidance requires its original readiness")
                context.guidance_ready.order_on(stream)
                context.sampler_phase, context.sampler_stream = _SamplerPhase.BORROWED, stream
            return context.guidance

    def complete_sampler(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]],
                         stream: torch.cuda.Stream | None = None, *, feedback: SamplerFeedback | None = None) -> None:
        with self._lock:
            context = self._context(invocation)
            if invocation.last_anchor_id < self.sampler_anchor_id:
                raise RuntimeError("sampler completion precedes its signed anchor")
            if context.sampler_phase is _SamplerPhase.COMPLETE:
                raise RuntimeError("sampler completion is already recorded")
            if stream is not None and stream.device != self.device:
                raise RuntimeError("sampler stream differs from the signed device")
            if context.pin.stream is not None and stream is None:
                raise RuntimeError("CUDA sampler completion requires its original stream")
            if context.sampler_phase is _SamplerPhase.BORROWED and context.sampler_stream != stream:
                raise RuntimeError("sampler completion differs from the borrowed stream")
            required = any(isinstance(item.source, SamplerFeedbackInputSource)
                for task in self.arena.plan.semantic_graph.tasks for item in task.inputs)
            if required != (feedback is not None):
                raise ValueError("sampler feedback differs from the signed input contract")
            if feedback is not None and (context.guidance is None
                or feedback.sampler_rows is not context.guidance.sampler_rows
                or feedback.base_logits.device != self.device):
                raise ValueError("sampler feedback requires original guidance rows and device")
            self._order_sampler(invocation, stream)
            context.sampler_feedback = feedback
            context.sampler_phase = _SamplerPhase.COMPLETE
            self._sampler_count += 1
            self.trace.record(TracePhase.SAMPLER_COMPLETE, invocation.invocation_id, pin_id=context.pin.pin_id)

    def finish_step(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> ResultT:
        with self._lock:
            context = self._context(invocation)
            if context.sampler_phase is not _SamplerPhase.COMPLETE:
                raise RuntimeError("state invocation requires sampler completion")
            try:
                result = self.math.output(context)
                if not invocation.finished:
                    invocation.finish()
                self._release(invocation, context)
            except Exception as exc:
                self._failed(invocation, context, exc)
                raise
            return result

    def cancel_step(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> None:
        invocation.context.cancel_requested.set()
        with self._lock:
            context = self._context(invocation)
            self._order_sampler(invocation, context.sampler_stream)
            if context.write is not None:
                self.arena.abort(context.write)
                context.write = None
            self.trace.record(TracePhase.ABORT, invocation.invocation_id)
            self._release(invocation, context)

    def cancel_active(self) -> None:
        with self._lock:
            for invocation in tuple(self._active.values()):
                self.cancel_step(invocation)

    def _failed(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT], exc: Exception) -> None:
        self.trace.record(TracePhase.FAILED, invocation.invocation_id,
            operation_id=None if context.publication is None else context.publication.operation_id)
        if context.publication is not None:
            raise StateSettlementError(invocation, context.publication) from exc
        self.cancel_step(invocation)
        raise exc

    def _release(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT]) -> None:
        if context.published_pin is not None and context.published_release is None:
            context.published_release = self.arena.release_pin(context.published_pin)
        if context.release is None:
            context.release = self.arena.release_pin(context.pin)
        if context.publication is not None and not context.publication_settled:
            self.arena.settle(context.publication)
            context.publication_settled = True
        self.directory.release_transition_lease(context.selection)
        self._active.pop(invocation.invocation_id)
        context.value = None
        context.results.clear()
        context.loss = None
        context.guidance = context.sampler_feedback = None
        context.guidance_ready = None
        self.trace.record(TracePhase.SETTLED, invocation.invocation_id, pin_id=context.pin.pin_id,
            release_id=context.release.release_id,
            operation_id=None if context.publication is None else context.publication.operation_id)
        if context.publication is not None or invocation.finished:
            self._loss_count += context.loss_count
            self._update_count += context.update_count
            self._trained_rows += context.trained_rows

    def _order_sampler(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], stream: torch.cuda.Stream | None) -> None:
        consumer = invocation.context.pin.stream
        if stream is not None and consumer is not None:
            CudaStreamReadiness(stream).order_on(consumer)

    def _context(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> StateTaskContext[DataT]:
        if (self._active.get(invocation.invocation_id) is not invocation
            or invocation.execution is not self._runner):
            raise RuntimeError("state invocation is not registered as active")
        return invocation.context

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            if self.directory.state is None:
                if self._active:
                    raise RuntimeError("stateless requests cannot retire during an invocation")
                self.arena.collect_completed(wait=True)
                self.trace.record(TracePhase.RETIRE, 0, identities=identities)
                return ()
            if self.directory.scope is StateScope.INSTANCE and identities:
                refs = self.directory.retire_inference(identities)
                self.trace.record(TracePhase.RETIRE, 0, identities=identities, refs=refs)
                return refs
            if self.directory.inference_pin_count:
                raise RuntimeError("instance retirement requires all inference requests to finish")
            if self.directory.scope is StateScope.INSTANCE:
                for invocation in tuple(self._active.values()):
                    if None in invocation.context.selection.keys:
                        self.cancel_step(invocation)
            else:
                retiring = set(identities)
                # Only the retiring requests' own steps cancel; requests sharing a
                # route lineage keep running, and the lineage outlives the last
                # holder until dormant retirement.
                for invocation in tuple(self._active.values()):
                    if any(row.identity in retiring for row in invocation.context.selection.rows):
                        self.cancel_step(invocation)
            self.arena.collect_completed(wait=True)
            refs = self.directory.retire(identities)
            self.trace.record(TracePhase.RETIRE, 0, identities=identities, refs=refs)
            return refs

    def retire_dormant(self) -> tuple[StateRef, ...]:
        with self._lock:
            if self._active:
                raise RuntimeError("dormant retirement requires no active invocation")
            self.arena.collect_completed(wait=True)
            refs = self.directory.retire_dormant()
            self.trace.record(TracePhase.RETIRE, 0, identities=(), refs=refs)
            return refs

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
            return StateDiagnostics(self.directory.active_count, self.directory.inference_pin_count, self._loss_count, self._update_count,
                self._trained_rows, self._sampler_count, self.arena.pending_pin_releases, self.plan.fallback_count, self.arena.usage())
