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
from pllm.execution.ptts.page_bundle import PageBundleArena, PageBundleUsage, PagePin, PagePublication, PageRelease, PageWrite
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_capture import KernelCapture
from pllm.execution.ptts.request_state import RequestStateDirectory, RequestStateSelection
from pllm.execution.replay import LayerReplayPrimitive, TokenwiseReplayPrimitive
from pllm.execution.trace import ExecutionTrace, TracePhase
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import AnchorId, ImplementationId, PhysicalPlan, SamplerFeedbackInputSource, StateEffectKind, StateScope, TaskSymbol
from pllm.contracts.compilation import PlanBundle
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.contracts.state_representation import StatePhysicalPlan


InputT = TypeVar("InputT", contravariant=True)
InputContra = TypeVar("InputContra", contravariant=True)
DataT = TypeVar("DataT")
ResultT = TypeVar("ResultT")
ResultCo = TypeVar("ResultCo", covariant=True)


@dataclass(frozen=True, slots=True)
class HeldInput:
    reason: str
    step_id: int
    successful: tuple[tuple[RequestIdentity, tuple[StateRef, ...], int], ...]


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
    # These collections replace the single-write lifecycle. Every entry is an
    # original arena-issued capability; no independently mutable latest map.
    page_writes: dict[PageWrite, None] = field(default_factory=lambda: dict[PageWrite, None]())
    page_publications: dict[PagePublication, bool] = field(default_factory=lambda: dict[PagePublication, bool]())
    page_pins: dict[PagePin, PageRelease | None] = field(default_factory=lambda: dict[PagePin, PageRelease | None]())
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
    replay_started: bool = False
    cancel_requested: Event = field(default_factory=Event)
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    held_input: HeldInput | None = None
    injected_refs: dict[RequestIdentity, set[StateRef]] = field(default_factory=lambda: dict[RequestIdentity, set[StateRef]]())
    input_step_id: int = -1

    def data(self) -> DataT:
        if self.value is None:
            raise RuntimeError("state invocation requires its input")
        return self.value

    def __post_init__(self) -> None:
        self.page_pins[self.pin] = None

    @property
    def write(self) -> PageWrite | None:
        if len(self.page_writes) > 1:
            raise RuntimeError("legacy write access requires a single private transition")
        return next(iter(self.page_writes), None)

    @write.setter
    def write(self, token: PageWrite | None) -> None:
        if token is None:
            pending = self.write
            if pending is not None:
                del self.page_writes[pending]
        else:
            if self.page_writes:
                raise RuntimeError("legacy write access cannot replace a private transition")
            self.page_writes[token] = None

    @property
    def publication(self) -> PagePublication | None:
        return next(reversed(self.page_publications), None)

    @property
    def release(self) -> PageRelease | None:
        return self.page_pins[self.pin]

    @property
    def published_pin(self) -> PagePin | None:
        return next((pin for pin in reversed(self.page_pins) if pin is not self.pin), None)

    @property
    def published_release(self) -> PageRelease | None:
        pin = self.published_pin
        return None if pin is None else self.page_pins[pin]

    def current_owned_ref(self, original: StateRef) -> StateRef:
        if original not in self.pin.refs:
            raise ValueError("state lineage is outside this invocation's original pin")
        current = original
        for receipt in self.page_publications:
            for before, after in zip(receipt.before_refs, receipt.after_refs, strict=True):
                if before == current:
                    current = after
        return current

    def check_page_execution(self) -> None:
        if self.release is not None:
            raise RuntimeError("state invocation has already released its original pin")
        if self.cancel_requested.is_set():
            raise RuntimeError("state invocation was cancelled")

    def prepare_owned_write(self, sources: tuple[StateRef, ...],
                            fields: tuple[tuple[LayoutFieldId, ...], ...]) -> PageWrite:
        self.check_page_execution()
        authorized = {self.current_owned_ref(ref) for ref in self.pin.refs}
        if not set(sources) <= authorized:
            raise ValueError("transition must extend this invocation's exact owned versions")
        token = self.arena.prepare_transition(sources, self.pin.stream, fields=fields, copy_fields=())
        self.page_writes[token] = None
        return token

    def publish_owned_write(self, token: PageWrite) -> PagePublication:
        self.check_page_execution()
        if token not in self.page_writes:
            raise ValueError("publication requires this invocation's original private write")
        receipt = self.arena.publish(token)
        self.page_publications[receipt] = False
        del self.page_writes[token]
        self.trace.record(TracePhase.PUBLISH, self.invocation_id, refs=receipt.after_refs,
            operation_id=receipt.operation_id)
        return receipt

    def settle_owned_publication(self, receipt: PagePublication) -> None:
        if receipt not in self.page_publications:
            raise ValueError("settlement requires this invocation's original receipt")
        if not self.page_publications[receipt]:
            self.arena.settle(receipt)
            self.page_publications[receipt] = True

    def abort_owned_writes(self) -> None:
        for token in tuple(self.page_writes):
            self.arena.abort(token)
            del self.page_writes[token]

    def release_owned_pin(self, pin: PagePin) -> None:
        if pin not in self.page_pins:
            raise ValueError("release requires this invocation's original pin")
        if self.page_pins[pin] is None:
            self.page_pins[pin] = self.arena.release_pin(pin)

    def publish(self) -> None:
        token = self.write
        if token is not None:
            if self.publication is not None:
                raise RuntimeError("legacy state invocation already owns its publication receipt")
            self.publish_owned_write(token)

    def borrow_published(self, refs: tuple[StateRef, ...]) -> PagePin:
        self.check_page_execution()
        if not refs or not all(any(ref in receipt.after_refs for receipt in self.page_publications) for ref in refs):
            raise ValueError("published borrow requires this invocation's original receipts")
        for pin, release in self.page_pins.items():
            if release is None and set(refs) <= set(pin.refs):
                return pin
        pin = self.arena.pin_exact(refs, self.pin.stream)
        self.page_pins[pin] = None
        return pin


class StateMath(Protocol[InputContra, DataT, ResultCo]):
    """Input validation and tensor mathematics, never invocation ownership."""

    def initialize(self, roots: tuple[tuple[torch.Tensor, ...], ...], resources: tuple[str, ...], /) -> None: ...

    def rows(self, value: InputContra, /) -> tuple[PackedPhaseRow, ...]: ...

    def prepare(self, value: InputContra, context: StateTaskContext[DataT]) -> DataT: ...

    def output(self, context: StateTaskContext[DataT]) -> ResultCo: ...


@dataclass(frozen=True, slots=True)
class MathBinding(Generic[InputT, DataT, ResultT]):
    """Local native implementations, separate from the closed data contracts."""

    math: StateMath[InputT, DataT, ResultT]
    tasks: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...]
    captures: tuple[KernelCapture, ...] = ()
    borrowed_resources: tuple[str, ...] = ()

    def local_resources(self, plan: PlanBundle) -> tuple[str, ...]:
        states = plan.semantic_graph.states
        borrowed = set(self.borrowed_resources)
        if (len(borrowed) != len(self.borrowed_resources) or not borrowed <= {state.resource for state in states}
            or any(state.scope is not StateScope.INSTANCE for state in states if state.resource in borrowed)
            or any(effect.kind is not StateEffectKind.READ for task in plan.semantic_graph.tasks
                for effect in task.effects if effect.resource in borrowed)):
            raise ValueError("borrowed bindings require distinct read-only instance resources from the original plan")
        return tuple(str(state.resource) for state in states if state.resource not in borrowed)

    def bind(self, arena: PageBundleArena, sampler_anchor_id: AnchorId) -> StateExecution[InputT, DataT, ResultT]:
        # Reject any underfunded kernel before allocating/capturing the first one's inputs.
        for capture in self.captures:
            capture.check_inputs(arena.plan, capture.inputs(torch.device("meta")))
        for capture in self.captures:
            capture.bind(arena)
        states = arena.plan.semantic_graph.states
        for group in arena.plan.semantic_graph.input_groups:
            if set(group.borrowed_resources) != set(self.borrowed_resources):
                raise ValueError("conditional input must cover the complete signed borrowed resource group")
            guarded = {task.implementation_id for task in arena.plan.schedule_physical.tasks
                if arena.plan.semantic_graph.tasks[task.task_id].symbol in group.tasks}
            if guarded != {task.implementation_id for task in self.tasks if isinstance(task, ConditionalProducerTask)}:
                raise ValueError("conditional producer bindings differ from the signed task closure")
        if set(arena.owned_resources) != set(self.local_resources(arena.plan)):
            raise ValueError("local and borrowed resources must partition the original plan")
        # A borrowed binding owns only its local directory; test_video verifies no remote lineage.
        if self.borrowed_resources:
            states = tuple(state for state in states if state.resource in arena.owned_resources)
        execution = StateExecution(arena, RequestStateDirectory(arena, states), self.math, self.tasks, sampler_anchor_id,
            _borrowed_resources=self.borrowed_resources)
        return execution


@dataclass(frozen=True, slots=True)
class ConditionalProducerTask(Generic[DataT]):
    original: TaskGraphImplementation[StateTaskContext[DataT]]

    @property
    def implementation_id(self) -> ImplementationId:
        return self.original.implementation_id

    def execute(self, context: StateTaskContext[DataT]) -> None:
        context.check_page_execution()
        if context.held_input is None:
            self.original.execute(context)
        else:
            context.trace.record(TracePhase.INPUT_SKIPPED, context.invocation_id)


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
    hold_count: int = 0
    last_hold: HeldInput | None = None


class StateSettlementError(RuntimeError, Generic[DataT]):
    def __init__(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], receipt: PagePublication) -> None:
        super().__init__("state publication committed; settle the original invocation without replay")
        self.invocation = invocation
        self.receipt = receipt


class StateExecution(Generic[InputT, DataT, ResultT]):
    def __init__(self, arena: PageBundleArena, directory: RequestStateDirectory,
                 math: StateMath[InputT, DataT, ResultT],
                 implementations: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...],
                 sampler_anchor_id: AnchorId, *, _borrowed_resources: tuple[str, ...] = ()) -> None:
        self.arena, self.directory, self.math = arena, directory, math
        self.sampler_anchor_id = sampler_anchor_id
        self.trace = ExecutionTrace(arena.plan.schedule_physical)
        self._values = tuple((task.symbol, output.name) for task in arena.plan.semantic_graph.tasks for output in task.outputs)
        self._runner = TaskGraphExecution(arena.plan.schedule_physical, implementations, self.trace, self._release_values)
        self._active: dict[int, TaskGraphInvocation[StateTaskContext[DataT]]] = {}
        self._next_invocation = 1
        self._loss_count = self._update_count = self._trained_rows = self._sampler_count = 0
        self._lock = RLock()
        self._borrowed_resources = _borrowed_resources
        self._pending_borrows: tuple[BorrowedStateRoot, ...] | None = None
        self._borrow_failed = False
        self._pending_hold: HeldInput | None = None
        self._successful_inputs: dict[RequestIdentity, tuple[tuple[StateRef, ...], int]] = {}
        self._hold_count = 0
        self._last_hold: HeldInput | None = None

    def _require_borrow_live(self) -> None:
        if self._borrow_failed:
            raise RuntimeError("borrowed execution is fail-stopped with resources retained")

    def stage_readonly(self, roots: tuple[BorrowedStateRoot, ...]) -> None:
        """Internal transport handoff; no change to workload authoring signatures."""
        with self._lock:
            self._require_borrow_live()
            if not self._borrowed_resources or self._active or self._pending_borrows is not None or self._pending_hold is not None:
                raise RuntimeError("read-only staging requires an idle borrowed execution")
            self._pending_borrows = roots  # Retain even invalid incoming mappings until the session terminates.
            try:
                if (len(roots) != len(self._borrowed_resources)
                    or {root.resource for root in roots} != set(self._borrowed_resources)
                    or any(root.signature != self.arena.plan.schedule_decision.signature for root in roots)):
                    raise ValueError("received roots differ from the required signed resources")
                for root in roots:
                    root.claim(self.arena.allocation_ranges)
            except Exception:
                self._borrow_failed = True
                raise

    def stage_hold(self, identities: tuple[RequestIdentity, ...], reason: str, *, step_id: int) -> None:
        """Internal signed branch handoff; exact versions come only from settlement."""
        with self._lock:
            self._require_borrow_live()
            try:
                if self.arena.plan.semantic_graph.input_groups and not self.arena.plan.semantic_graph.input_groups[0].allow_absent:
                    raise RuntimeError("missing activation input: signed STOP policy stops the instance")
                if (not self.arena.plan.semantic_graph.input_groups
                    or not self.arena.plan.semantic_graph.input_groups[0].allow_absent or not self._borrowed_resources
                    or self._active or self._pending_borrows is not None or self._pending_hold is not None
                    or reason not in ("absent", "end") or step_id < 0 or not identities
                    or any(identity not in self._successful_inputs for identity in identities)):
                    raise RuntimeError("HOLD requires a signed idle input branch and a first successful injection")
                self._pending_hold = HeldInput(reason, step_id,
                    tuple((identity, *self._successful_inputs[identity]) for identity in identities))
            except Exception:
                self._borrow_failed = True
                raise

    @property
    def plan(self) -> PhysicalPlan:
        return self.arena.plan.schedule_physical

    def _release_values(self, context: StateTaskContext[DataT], values: tuple[int, ...]) -> None:
        for value in values:
            context.results.pop(self._values[value], None)

    @property
    def state_plan(self) -> StatePhysicalPlan:
        return self.arena.plan.state_physical

    def implementation(self, identity: ImplementationId) -> TaskGraphImplementation[StateTaskContext[DataT]]:
        matches = tuple(item for item in self._runner.implementations if item.implementation_id == identity)
        if len(matches) != 1:
            raise ValueError("implementation must name one original physical task")
        return matches[0]

    def rebind_implementation(self, implementation: TaskGraphImplementation[StateTaskContext[DataT]]) -> None:
        """Install a locally verified lowering while no original invocation is active."""
        with self._lock:
            if self._active:
                raise RuntimeError("cannot rebind a physical task during an active invocation")
            original = self.implementation(implementation.implementation_id)
            if isinstance(original, ConditionalProducerTask) != isinstance(implementation, ConditionalProducerTask):
                raise ValueError("rebinding cannot remove or add a signed conditional producer guard")
            self._runner.implementations = tuple(implementation if item is original else item
                                                for item in self._runner.implementations)

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def admit(self, identities: tuple[RequestIdentity, ...],
              stream: torch.cuda.Stream | None = None) -> tuple[StateRef, ...]:
        with self._lock:
            # Only borrowing executions have a terminal transport state; test_video covers it.
            if self._borrowed_resources:
                self._require_borrow_live()
            if not self.directory.states:
                return ()
            batch = self.directory.prepare_admission(identities, stream)
            try:
                if batch.write is not None:
                    with nullcontext() if stream is None else torch.cuda.stream(stream):
                        self.math.initialize(self.arena.writable_roots(batch.write),
                            tuple(str(key.resource) for key in batch.keys))
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
            # Complete exact input before any local pin/write; test_video rejects missing staging.
            if self._borrowed_resources:
                self._require_borrow_live()
                if self._active or (self._pending_hold is None and self._pending_borrows is None) or any(
                    root.stream != stream for root in self._pending_borrows or ()):
                    self._borrow_failed = True
                    raise RuntimeError("borrowed step requires complete staged roots on the receiving stream")
            self.arena.collect_completed()
            selection = self.directory.select(rows)
            self.directory.acquire_transition_lease(selection)
            try:
                pin = self.arena.pin_exact(selection.refs, stream)
            except Exception:
                self.directory.release_transition_lease(selection)
                raise
            context = StateTaskContext[DataT](selection, self.arena, pin, self.trace, self._next_invocation)
            context.held_input, self._pending_hold = self._pending_hold, None
            invocation = TaskGraphInvocation(self._runner, context, self._next_invocation)
            self._next_invocation += 1
            self._active[invocation.invocation_id] = invocation
            if self._borrowed_resources and context.held_input is None:
                assert self._pending_borrows is not None
                context.borrowed_roots = self._pending_borrows
                self._pending_borrows = None
                try:
                    for root in context.borrowed_roots:
                        root.bind(pin)
                except Exception as exc:
                    self._failed(invocation, context, exc)
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
            except Exception as exc:
                self.trace.record(TracePhase.FAILED, invocation.invocation_id)
                # Failed preparation can leave aliases in traceback frames; test_video retains grants.
                if context.borrowed_roots:
                    self._failed(invocation, context, exc)
                self.cancel_step(invocation)
                raise

    def execute_anchor(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], anchor_id: AnchorId) -> None:
        with self._lock:
            context = self._context(invocation)
            if anchor_id > self.sampler_anchor_id and context.sampler_phase is not _SamplerPhase.COMPLETE:
                raise RuntimeError("later anchors require sampler completion")
            try:
                context.check_page_execution()
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
                if context.held_input is not None and (context.borrowed_roots or context.page_writes
                    or context.page_publications or context.update_count
                    or set(context.injected_refs) != {identity for identity, _, _ in context.held_input.successful}
                    or any(not context.injected_refs[identity] <= set(refs)
                        for identity, refs, _ in context.held_input.successful)):
                    raise RuntimeError("HOLD settlement requires actual pinned injection without producer effects")
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
            # Cancellation cannot prove that remote aliases were discarded; test_video checks no ACK.
            if context.borrowed_roots or context.held_input is not None:
                self._failed(invocation, context, RuntimeError("borrowed invocation cancelled; resources retained"))
            self._order_sampler(invocation, context.sampler_stream)
            context.abort_owned_writes()
            self.trace.record(TracePhase.ABORT, invocation.invocation_id)
            self._release(invocation, context)

    def cancel_active(self) -> None:
        with self._lock:
            for invocation in tuple(self._active.values()):
                self.cancel_step(invocation)

    def _failed(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT], exc: Exception) -> None:
        context.cancel_requested.set()
        self.trace.record(TracePhase.FAILED, invocation.invocation_id,
            operation_id=None if context.publication is None else context.publication.operation_id)
        # Quarantine borrowed failures, including exception-frame aliases; test_video covers both publication outcomes.
        if context.borrowed_roots or context.held_input is not None:
            self._borrow_failed = True
            if context.publication is None:
                try:
                    context.abort_owned_writes()
                except Exception as cleanup:
                    exc.add_note(f"local abort also failed; resources retained: {cleanup}")
                raise exc
        if context.publication is not None:
            raise StateSettlementError(invocation, context.publication) from exc
        self.cancel_step(invocation)
        raise exc

    def _release(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT]) -> None:
        # Only borrowed steps clear aliases and ACK before settlement; test_video checks ACK/failure ordering.
        if context.borrowed_roots:
            context.value = None
            context.results.clear()
            context.loss = None
            context.guidance = context.sampler_feedback = None
            context.guidance_ready = None
            for root in context.borrowed_roots:
                root.close()
        for pin in context.page_pins:
            context.release_owned_pin(pin)
        for receipt in context.page_publications:
            context.settle_owned_publication(receipt)
        self.directory.release_transition_lease(context.selection)
        self._active.pop(invocation.invocation_id)
        context.value = None
        context.results.clear()
        context.loss = None
        context.guidance = context.sampler_feedback = None
        context.guidance_ready = None
        release = context.release
        if release is None:
            raise RuntimeError("state settlement requires the original pin release")
        self.trace.record(TracePhase.SETTLED, invocation.invocation_id, pin_id=context.pin.pin_id,
            release_id=release.release_id,
            operation_id=None if context.publication is None else context.publication.operation_id)
        if context.publication is not None or invocation.finished:
            self._loss_count += context.loss_count
            self._update_count += context.update_count
            self._trained_rows += context.trained_rows
        if invocation.finished and self.arena.plan.semantic_graph.input_groups and context.held_input is None:
            for identity, refs in context.injected_refs.items():
                self._successful_inputs[identity] = tuple(sorted(refs)), context.input_step_id
        if invocation.finished and context.held_input is not None:
            self._hold_count += 1
            self._last_hold = context.held_input

    def _order_sampler(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], stream: torch.cuda.Stream | None) -> None:
        consumer = invocation.context.pin.stream
        if stream is not None and consumer is not None:
            CudaStreamReadiness(stream).order_on(consumer)

    def _context(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> StateTaskContext[DataT]:
        if self._borrowed_resources:
            self._require_borrow_live()
        if (self._active.get(invocation.invocation_id) is not invocation
            or invocation.execution is not self._runner):
            raise RuntimeError("state invocation is not registered as active")
        return invocation.context

    def retire(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        with self._lock:
            for identity in identities:
                self._successful_inputs.pop(identity, None)
            if not self.directory.states:
                if self._active:
                    raise RuntimeError("stateless requests cannot retire during an invocation")
                self.arena.collect_completed(wait=True)
                self.trace.record(TracePhase.RETIRE, 0, identities=identities)
                return ()
            if not identities and self.directory.inference_pin_count:
                raise RuntimeError("instance retirement requires all inference requests to finish")
            if not identities:
                for invocation in tuple(self._active.values()):
                    if any(address.key is None for address in invocation.context.selection.addresses):
                        self.cancel_step(invocation)
            elif self.directory.request_states:
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
                self._trained_rows, self._sampler_count, self.arena.pending_pin_releases, self.plan.fallback_count, self.arena.usage(),
                self._hold_count, self._last_hold)
