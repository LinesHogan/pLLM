"""Own invocation pages, sampler handoff and settlement for tensor execution and vLLM sessions."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from threading import Event, RLock
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar

import numpy as np
import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow
from pllm.contracts.parameter_overlay import ParameterTarget
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    PhysicalPlan,
    TaskId,
    TaskSymbol,
)
from pllm.contracts.state_feed import FeedBinding
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.admission import WindowBackpressure
from pllm.execution.cuda_stream import CudaStreamReadiness, DeviceCounters
from pllm.execution.execution import TaskGraphExecution, TaskGraphImplementation, TaskGraphInvocation
from pllm.execution.memory import MemoryUsage
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import (
    PageBundleArena,
    PageBundleUsage,
    PagePin,
    PagePublication,
    PageRelease,
    PageWrite,
)
from pllm.execution.ptts.request_state import RequestStateDirectory, RequestStateSelection, StateAddress
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.trace import ExecutionTrace, TraceEvent, TracePhase

if TYPE_CHECKING:
    from pllm.execution.replay import TokenwiseReplayPrimitive
    from pllm.execution.tensor_program import TensorBatch

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


@dataclass(slots=True)
class StateTaskContext(Generic[DataT]):
    selection: RequestStateSelection
    arena: PageBundleArena
    pin: PagePin
    trace: ExecutionTrace
    invocation_id: int
    value: DataT | None = None
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]()
    )
    # These collections replace the single-write lifecycle. Every entry is an
    # original arena-issued capability; no independently mutable latest map.
    page_writes: dict[PageWrite, None] = field(default_factory=lambda: dict[PageWrite, None]())
    page_publications: dict[PagePublication, bool] = field(default_factory=lambda: dict[PagePublication, bool]())
    page_pins: dict[PagePin, PageRelease | None] = field(default_factory=lambda: dict[PagePin, PageRelease | None]())
    guidance: PackedSamplerGuidanceBatch | None = None
    loss: torch.Tensor | None = None
    guidance_ready: CudaStreamReadiness | None = None
    sampler_feedback: SamplerFeedback | None = None
    sampler_stream: torch.cuda.Stream | None = None
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0
    replay: TokenwiseReplayPrimitive | None = None
    overlay_counters: torch.Tensor | None = None
    cancel_requested: Event = field(default_factory=Event)
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    held_input: HeldInput | None = None
    injected_refs: dict[RequestIdentity, set[StateRef]] = field(
        default_factory=lambda: dict[RequestIdentity, set[StateRef]]()
    )
    input_step_id: int = -1
    projection_outputs: Mapping[tuple[RequestIdentity, ParameterTarget], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[RequestIdentity, ParameterTarget], torch.Tensor]()
    )

    def data(self) -> DataT:
        assert self.value is not None
        return self.value

    def __post_init__(self) -> None:
        self.page_pins[self.pin] = None

    @property
    def publication(self) -> PagePublication | None:
        return next(reversed(self.page_publications), None)

    @property
    def release(self) -> PageRelease | None:
        return self.page_pins[self.pin]

    def current_owned_ref(self, original: StateRef) -> StateRef:
        return self.current_owned_refs((original,))[0]

    def current_owned_refs(self, originals: Sequence[StateRef]) -> tuple[StateRef, ...]:
        """Follow this invocation's publications from each ref, walking the receipts once for the whole batch."""
        successor: dict[StateRef, StateRef] = {}
        for receipt in self.page_publications:
            for before, after in zip(receipt.before_refs, receipt.after_refs, strict=True):
                successor.setdefault(before, after)

        def latest(ref: StateRef) -> StateRef:
            while ref in successor:
                ref = successor[ref]
            return ref

        return tuple(map(latest, originals))

    def check_page_execution(self) -> None:
        if self.cancel_requested.is_set():
            raise RuntimeError("state invocation was cancelled")

    def prepare_owned_write(
        self,
        sources: Sequence[StateRef],
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        *,
        copy_fields: tuple[LayoutFieldId, ...] | None = (),
    ) -> PageWrite:
        """Prepare and retain a private transition, returning its explicit write token."""
        self.check_page_execution()
        authorized = set(self.current_owned_refs(self.pin.refs))
        assert set(sources) <= authorized
        token = self.arena.prepare_transition(sources, self.pin.stream, fields=fields, copy_fields=copy_fields)
        self.page_writes[token] = None
        return token

    def publish_owned_write(self, token: PageWrite) -> PagePublication:
        self.check_page_execution()
        assert token in self.page_writes
        receipt = self.arena.publish(token)
        self.page_publications[receipt] = False
        del self.page_writes[token]
        self.trace.record(
            TracePhase.PUBLISH, self.invocation_id, refs=receipt.after_refs, operation_id=receipt.operation_id
        )
        return receipt

    def settle_owned_publication(self, receipt: PagePublication) -> None:
        assert receipt in self.page_publications
        if not self.page_publications[receipt]:
            self.arena.settle(receipt)
            self.page_publications[receipt] = True

    def abort_owned_writes(self) -> None:
        for token in tuple(self.page_writes):
            self.arena.abort(token)
            del self.page_writes[token]

    def release_owned_pin(self, pin: PagePin) -> None:
        assert pin in self.page_pins
        if self.page_pins[pin] is None:
            self.page_pins[pin] = self.arena.release_pin(pin)

    def borrow_published(self, refs: Sequence[StateRef]) -> PagePin:
        self.check_page_execution()
        refs = StateRefs.of(refs)
        authorized = np.zeros(len(refs), dtype=np.bool_)
        for receipt in self.page_publications:
            authorized |= receipt.after_refs.positions(refs) >= 0
        assert refs and authorized.all()
        for pin, release in self.page_pins.items():
            if release is None and (pin.refs.positions(refs) >= 0).all():
                return pin
        pin = self.arena.pin_exact(refs, self.pin.stream)
        self.page_pins[pin] = None
        return pin


class StateMath(Protocol[InputContra, DataT, ResultCo]):
    """Input validation and tensor mathematics, never invocation ownership."""

    @property
    def parameter_phases(self) -> tuple[HiddenPhase, ...]:
        """Authored training phases used to select pinned projection captures."""
        ...

    def initialize(
        self, roots: tuple[tuple[torch.Tensor, ...], ...], addresses: tuple[StateAddress, ...], /
    ) -> None: ...

    def rows(self, value: InputContra, /) -> tuple[PackedPhaseRow, ...]: ...

    def prepare(self, value: InputContra, context: StateTaskContext[DataT]) -> DataT: ...

    def output(self, context: StateTaskContext[DataT]) -> ResultCo: ...


@dataclass(frozen=True, slots=True)
class MathBinding(Generic[InputT, DataT, ResultT]):
    """Local native implementations, separate from the closed data contracts."""

    math: StateMath[InputT, DataT, ResultT]
    tasks: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...]
    borrowed_resources: tuple[str, ...] = ()
    feeds: tuple[FeedBinding, ...] = ()
    on_export: Callable[[TraceEvent], None] | None = None

    def local_resources(self, plan: PlanBundle) -> tuple[str, ...]:
        states = plan.semantic_graph.states
        borrowed = set(self.borrowed_resources)
        return tuple(str(state.resource) for state in states if state.resource not in borrowed)

    def bind(self, arena: PageBundleArena, sampler_anchor_id: AnchorId) -> StateExecution[InputT, DataT, ResultT]:
        states = arena.plan.semantic_graph.states
        # A borrowed binding owns only its local directory; test_video verifies no remote lineage.
        if self.borrowed_resources:
            states = tuple(state for state in states if state.resource in arena.owned_resources)
        execution = StateExecution(
            arena,
            RequestStateDirectory(arena, states),
            self.math,
            self.tasks,
            sampler_anchor_id,
            _borrowed_resources=self.borrowed_resources,
            feeds=self.feeds,
            on_export=self.on_export,
        )
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

    def memory(self, data: DataT, batch: TensorBatch) -> MemoryUsage:
        """Publication changes ownership metadata; PTTS already funds its tensor pages."""
        return MemoryUsage(0, 0)

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return ()

    def resident(self, capture_bytes: int) -> int:
        return 0

    def execute(self, context: StateTaskContext[DataT]) -> None:
        if context.page_writes:
            assert len(context.page_writes) == 1
            (write,) = context.page_writes
            context.publish_owned_write(write)


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
    def __init__(
        self,
        arena: PageBundleArena,
        directory: RequestStateDirectory,
        math: StateMath[InputT, DataT, ResultT],
        implementations: tuple[TaskGraphImplementation[StateTaskContext[DataT]], ...],
        sampler_anchor_id: AnchorId,
        *,
        _borrowed_resources: tuple[str, ...] = (),
        feeds: tuple[FeedBinding, ...] = (),
        on_export: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.arena, self.directory, self.math = arena, directory, math
        self.sampler_anchor_id = sampler_anchor_id
        self.trace = ExecutionTrace(arena.plan.schedule_physical, on_export=on_export)
        self._values = tuple(
            (task.symbol, output.name) for task in arena.plan.semantic_graph.tasks for output in task.outputs
        )
        self._runner = TaskGraphExecution(
            arena.plan.schedule_physical, implementations, self.trace, self._release_values
        )
        self._active: dict[int, TaskGraphInvocation[StateTaskContext[DataT]]] = {}
        self._next_invocation = 1
        self._loss_count = self._update_count = self._trained_rows = self._sampler_count = 0
        self._overlay_counters = (
            DeviceCounters(self.device)
            if any(
                task.visibility is not None and task.visibility.scope == "overlay"
                for task in arena.plan.semantic_graph.tasks
            )
            else None
        )
        self._lock = RLock()
        self._borrowed_resources = _borrowed_resources
        self.feeds = feeds
        self._pending_borrows: tuple[BorrowedStateRoot, ...] | None = None
        self._borrow_failed = False
        self._pending_hold: HeldInput | None = None
        self._successful_inputs: dict[RequestIdentity, tuple[tuple[StateRef, ...], int]] = {}
        self._hold_count = 0
        self._last_hold: HeldInput | None = None

    def close(self) -> None:
        """Finish asynchronous exports when direct ownership or a backend session ends."""
        self.trace.close()

    def discard_borrows(self) -> None:
        """Retain failed grants until all GPU work ends, then discard their aliases."""
        roots = list(self._pending_borrows or ())
        for invocation in self._active.values():
            roots.extend(invocation.context.borrowed_roots)
            invocation.context.value = None
            invocation.context.results.clear()
        if roots and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for root in roots:
            root.discard()
        self._pending_borrows = None
        self._borrow_failed = True

    def _require_borrow_live(self) -> None:
        if self._borrow_failed:
            raise RuntimeError("borrowed execution is fail-stopped with resources retained")

    def stage_readonly(self, roots: tuple[BorrowedStateRoot, ...]) -> None:
        """Internal transport handoff; no change to workload authoring signatures."""
        with self._lock:
            self._require_borrow_live()
            self._pending_borrows = roots  # Retain even invalid incoming mappings until the session terminates.
            try:
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
                if (
                    self.arena.plan.semantic_graph.input_groups
                    and not self.arena.plan.semantic_graph.input_groups[0].allow_absent
                ):
                    raise RuntimeError("missing activation input: signed STOP policy stops the instance")
                self._pending_hold = HeldInput(
                    reason, step_id, tuple((identity, *self._successful_inputs[identity]) for identity in identities)
                )
            except Exception:
                self._borrow_failed = True
                raise

    @property
    def plan(self) -> PhysicalPlan:
        return self.arena.plan.schedule_physical

    def _release_values(self, context: StateTaskContext[DataT], values: tuple[int, ...]) -> None:
        for value in values:
            context.results.pop(self._values[value], None)

    def implementation(self, identity: ImplementationId) -> TaskGraphImplementation[StateTaskContext[DataT]]:
        return next(item for item in self._runner.implementations if item.implementation_id == identity)

    def rebind_implementation(self, implementation: TaskGraphImplementation[StateTaskContext[DataT]]) -> None:
        """Install a locally verified lowering while no original invocation is active."""
        with self._lock:
            original = self.implementation(implementation.implementation_id)
            self._runner.implementations = tuple(
                implementation if item is original else item for item in self._runner.implementations
            )

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def admit(
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None = None
    ) -> tuple[StateRef, ...]:
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
                        self.math.initialize(self.arena.writable_roots(batch.write), batch.keys)
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
                self.trace.record(
                    TracePhase.ADMIT, 0, identities=identities, refs=refs, operation_id=receipt.operation_id
                )
            return refs

    def begin(
        self,
        rows: tuple[PackedPhaseRow, ...],
        stream: torch.cuda.Stream | None = None,
        *,
        history: tuple[StateRefs, ...] = (),
    ) -> TaskGraphInvocation[StateTaskContext[DataT]]:
        with self._lock:
            if self._borrowed_resources:
                self._require_borrow_live()
            self.arena.collect_completed()
            selection = self.directory.select(rows, history)
            pin = self.arena.pin_exact(selection.refs, stream)
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
            self.trace.record(
                TracePhase.BEGIN,
                invocation.invocation_id,
                refs=pin.refs,
                pin_id=pin.pin_id,
                identities=tuple(row.identity for row in selection.rows),
            )
            return invocation

    def start_step(
        self,
        value: InputT,
        stream: torch.cuda.Stream | None = None,
        *,
        history: tuple[StateRefs, ...] = (),
    ) -> TaskGraphInvocation[StateTaskContext[DataT]]:
        with self._lock:
            invocation = self.begin(self.math.rows(value), stream, history=history)
            self.submit_input(invocation, value)
            return invocation

    def resume_latest(
        self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], stream: torch.cuda.Stream
    ) -> None:
        """Move an invocation to its signed writer stream without losing exact read holds.

        The caller orders producers and copies reusable inputs first. Old reader pages remain
        pinned on the new stream until settlement, including aliases consumed by later tasks.
        Equal reader and writer refs share the new pin; retained history owns its own hold.
        """
        with self._lock:
            context = self._context(invocation)
            assert not context.page_writes and not context.page_publications and not context.borrowed_roots
            reader = context.pin
            context.selection = self.directory.select(context.selection.rows)
            context.pin = self.arena.pin_exact(context.selection.refs, stream)
            context.page_pins[context.pin] = None
            if reader.refs != context.pin.refs:
                held = self.arena.pin_exact(reader.refs, stream)
                context.page_pins[held] = None
            context.release_owned_pin(reader)

    def step(self, value: InputT, stream: torch.cuda.Stream | None = None) -> ResultT:
        """Run without an external sampler borrower using the same staged lifecycle."""
        with self._lock:
            invocation = self.start_step(value, stream)
            try:
                for anchor in self.arena.plan.timeline.anchors:
                    self.execute_anchor(invocation, AnchorId(anchor.ordinal))
                    if anchor.ordinal == self.sampler_anchor_id:
                        self.complete_sampler(invocation, stream)
                return self.finish_step(invocation)
            except WindowBackpressure:
                # The convenience call exposes no cursor; staged callers keep it through execute_anchor.
                self.cancel_step(invocation)
                raise

    def submit_input(
        self,
        invocation: TaskGraphInvocation[StateTaskContext[DataT]],
        value: InputT,
        readiness: CudaStreamReadiness | None = None,
    ) -> None:
        """Consume original inputs, ordering an external CUDA producer when supplied."""
        with self._lock:
            context = self._context(invocation)
            try:
                if readiness is not None:
                    assert context.pin.stream is not None
                    readiness.order_on(context.pin.stream)
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
            try:
                context.check_page_execution()
                invocation.execute_anchor(anchor_id, context.pin.stream)
            except WindowBackpressure:
                # No cell of the pending task issued; its original cursor and pins remain usable.
                raise
            except Exception as exc:
                self._failed(invocation, context, exc)

    def execute_part(
        self,
        invocation: TaskGraphInvocation[StateTaskContext[DataT]],
        value: InputT,
        task_id: TaskId,
        *,
        complete: bool,
    ) -> None:
        """Supply actual projection operands to one segment of a bound in-forward task."""
        with self._lock:
            context = self._context(invocation)
            try:
                context.check_page_execution()
                context.value = self.math.prepare(value, context)
                invocation.execute_part(task_id, complete=complete, stream=context.pin.stream)
            except Exception as exc:
                self._failed(invocation, context, exc)

    def sampler_guidance(
        self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], stream: torch.cuda.Stream | None = None
    ) -> PackedSamplerGuidanceBatch:
        with self._lock:
            context = self._context(invocation)
            assert context.guidance is not None
            if stream is not None:
                assert context.guidance_ready is not None
                context.guidance_ready.order_on(stream)
                context.sampler_stream = stream
            return context.guidance

    def complete_sampler(
        self,
        invocation: TaskGraphInvocation[StateTaskContext[DataT]],
        stream: torch.cuda.Stream | None = None,
        *,
        feedback: SamplerFeedback | None = None,
    ) -> None:
        with self._lock:
            context = self._context(invocation)
            self._order_sampler(invocation, stream)
            context.sampler_feedback = feedback
            self._sampler_count += 1
            self.trace.record(TracePhase.SAMPLER_COMPLETE, invocation.invocation_id, pin_id=context.pin.pin_id)

    def finish_step(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> ResultT:
        with self._lock:
            context = self._context(invocation)
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

    def _failed(
        self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT], exc: Exception
    ) -> None:
        context.cancel_requested.set()
        self.trace.record(
            TracePhase.FAILED,
            invocation.invocation_id,
            operation_id=None if context.publication is None else context.publication.operation_id,
        )
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

    def _release(
        self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], context: StateTaskContext[DataT]
    ) -> None:
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
        self._active.pop(invocation.invocation_id)
        context.value = None
        context.results.clear()
        context.loss = None
        context.guidance = context.sampler_feedback = None
        context.guidance_ready = None
        release = context.release
        assert release is not None
        self.trace.record(
            TracePhase.SETTLED,
            invocation.invocation_id,
            pin_id=context.pin.pin_id,
            release_id=release.release_id,
            operation_id=None if context.publication is None else context.publication.operation_id,
        )
        if context.publication is not None or invocation.finished:
            self._loss_count += context.loss_count
            self._update_count += context.update_count
            self._trained_rows += context.trained_rows
            if context.overlay_counters is not None:
                assert self._overlay_counters is not None
                self._overlay_counters.add(context.overlay_counters, context.pin.stream)
        if invocation.finished and self.arena.plan.semantic_graph.input_groups and context.held_input is None:
            for identity, refs in context.injected_refs.items():
                self._successful_inputs[identity] = tuple(sorted(refs)), context.input_step_id
        if invocation.finished and context.held_input is not None:
            self._hold_count += 1
            self._last_hold = context.held_input

    def _order_sampler(
        self, invocation: TaskGraphInvocation[StateTaskContext[DataT]], stream: torch.cuda.Stream | None
    ) -> None:
        consumer = invocation.context.pin.stream
        if stream is not None and consumer is not None:
            CudaStreamReadiness(stream).order_on(consumer)

    def _context(self, invocation: TaskGraphInvocation[StateTaskContext[DataT]]) -> StateTaskContext[DataT]:
        if self._borrowed_resources:
            self._require_borrow_live()
        return invocation.context

    def retire(self, identities: tuple[RequestIdentity, ...], *, wait: bool = False) -> tuple[StateRef, ...]:
        with self._lock:
            for identity in identities:
                self._successful_inputs.pop(identity, None)
            if not self.directory.states:
                self.arena.collect_completed(wait=wait)
                self.trace.record(TracePhase.RETIRE, 0, identities=identities)
                return ()
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
            self.arena.collect_completed(wait=wait)
            refs = self.directory.retire(identities)
            self.trace.record(TracePhase.RETIRE, 0, identities=identities, refs=refs)
            return refs

    def retire_dormant(self, *, wait: bool = False) -> tuple[StateRef, ...]:
        with self._lock:
            self.arena.collect_completed(wait=wait)
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
            updates, rows = (0, 0) if self._overlay_counters is None else self._overlay_counters.read()
            return StateDiagnostics(
                self.directory.active_count,
                self.directory.inference_pin_count,
                self._loss_count,
                self._update_count + updates,
                self._trained_rows + rows,
                self._sampler_count,
                self.arena.pending_pin_releases,
                self.plan.fallback_count,
                self.arena.usage(),
                self._hold_count,
                self._last_hold,
            )
