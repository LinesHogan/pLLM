"""Own invocation pages, sampler handoff and commit for tensor execution and vLLM sessions.

A StateExecution belongs to the one thread that runs the model step; it has no lock. The trace export thread
reads only ExecutionTrace, which has its own Condition. Calling diagnostics()/command() from another thread
while generate is running is unsupported.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, ClassVar, Literal, NoReturn, Protocol, Self

import numpy as np
import torch

from pllm.contracts.compilation import PlanBundle
from pllm.contracts.packed_rows import PackedRows
from pllm.contracts.parameter_overlay import Projection
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.sampler import PackedSamplerGuidanceBatch, SamplerFeedback
from pllm.contracts.semantic_graph import (
    AnchorId,
    ImplementationId,
    PhysicalTask,
    ScheduleDecision,
    StateResource,
    TaskId,
    TaskResultInputSource,
    TaskSymbol,
)
from pllm.contracts.state_feed import FeedBinding
from pllm.contracts.state_ref import StateRef
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.execution.admission import WindowBackpressure
from pllm.execution.cuda_stream import CudaStreamReadiness, DeviceCounters
from pllm.execution.ptts.page_borrow import BorrowedStateRoot
from pllm.execution.ptts.page_bundle import (
    FieldNumericHealth,
    PageBundleArena,
    PageBundleUsage,
    PagePin,
    PagePublication,
    PageRelease,
    PageWrite,
)
from pllm.execution.ptts.request_state import RequestStateDirectory, RequestStateSelection
from pllm.execution.ptts.slot_pools import SlotPools, Slots
from pllm.execution.ptts.state_refs import StateRefs
from pllm.execution.torch_private import MemoryUsage
from pllm.execution.trace import ExecutionTrace, TraceEvent, TraceEventKind

if TYPE_CHECKING:
    from pllm.execution.post_sample import PostSampleOverlap
    from pllm.execution.step_input import StepInput
    from pllm.execution.tensor_declaration import TensorOutput
    from pllm.execution.tensor_frame import TensorFrame
    from pllm.execution.tensor_program import FunctionMath, TensorBatch

ProjectionResiduals = Mapping[tuple[RequestIdentity, Projection], torch.Tensor]
_PagePhase = Literal["abort", "release"]


@dataclass(frozen=True, slots=True)
class _WriteAbort:
    write: PageWrite
    phase: ClassVar[_PagePhase] = "abort"

    def complete(self, arena: PageBundleArena) -> None:
        arena.abort(self.write)


@dataclass(frozen=True, slots=True)
class _PinRelease:
    pin: PagePin
    phase: ClassVar[_PagePhase] = "release"

    def complete(self, arena: PageBundleArena) -> PageRelease:
        return arena.release_pin(self.pin)


@dataclass(frozen=True, slots=True, eq=False)
class _SlotRelease:
    allocations: tuple[tuple[SlotPools, Slots], ...]
    stream: torch.cuda.Stream | None
    phase: ClassVar[_PagePhase] = "release"

    def complete(self, arena: PageBundleArena) -> None:
        completion = None
        if self.stream is not None:
            completion = torch.cuda.Event()
            completion.record(self.stream)
        for pools, slots in self.allocations:
            pools.release(slots, completion)


_PageObligation = _WriteAbort | _PinRelease | _SlotRelease


@dataclass(frozen=True, slots=True)
class HeldInput:
    reason: str
    step_id: int
    successful: tuple[tuple[RequestIdentity, tuple[StateRef, ...], int], ...]


@dataclass(slots=True)
class StateTaskContext:
    """Per-step cursor, pages and sampler handoff driven by StateExecution."""

    selection: RequestStateSelection
    arena: PageBundleArena
    pin: PagePin
    trace: ExecutionTrace
    invocation_id: int
    next_task: int = 0
    last_anchor_id: int = -1
    finished: bool = False
    handed_off: bool = False
    prefetch_stream: torch.cuda.Stream | None = None
    value: TensorFrame | None = None
    results: dict[tuple[TaskSymbol, str], torch.Tensor] = field(
        default_factory=lambda: dict[tuple[TaskSymbol, str], torch.Tensor]()
    )
    _obligations: list[_PageObligation] = field(default_factory=lambda: list[_PageObligation]())
    # Receipts are ordered history; only outstanding obligations drive cleanup.
    publications: list[PagePublication] = field(default_factory=lambda: list[PagePublication]())
    releases: list[PageRelease] = field(default_factory=lambda: list[PageRelease]())
    guidance: tuple[PackedSamplerGuidanceBatch, ...] = ()
    loss: torch.Tensor | None = None
    guidance_ready: CudaStreamReadiness | None = None
    sampler_feedback: SamplerFeedback | None = None
    sampler_stream: torch.cuda.Stream | None = None
    loss_count: int = 0
    update_count: int = 0
    trained_rows: int = 0
    # Device [applied updates, trained rows] summed over the step's training tasks.
    training_counters: torch.Tensor | None = None
    cancel_requested: bool = False
    borrowed_roots: tuple[BorrowedStateRoot, ...] = ()
    held_input: HeldInput | None = None
    injected_refs: dict[RequestIdentity, set[StateRef]] = field(
        default_factory=lambda: dict[RequestIdentity, set[StateRef]]()
    )
    input_step_id: int = -1
    retain_gradients: bool = True

    def data(self) -> TensorFrame:
        assert self.value is not None
        return self.value

    def __post_init__(self) -> None:
        self.register_pin(self.pin)

    def register_pin(self, pin: PagePin) -> None:
        """Retain responsibility for releasing an acquired pin."""
        self._obligations.append(_PinRelease(pin))

    def hold_slots(self, allocations: tuple[tuple[SlotPools, Slots], ...]) -> None:
        """Retain transient inputs until this reader stream completes, also on cancellation."""
        for pools, slots in allocations:
            pools.hold(slots)
        self._obligations.append(_SlotRelease(allocations, self.pin.stream))

    def move_slots(self) -> None:
        """Transfer input holds after ordering the current pin's stream behind their prior readers."""
        for index, obligation in enumerate(self._obligations):
            if isinstance(obligation, _SlotRelease):
                if (
                    obligation.stream is not None
                    and self.pin.stream is not None
                    and obligation.stream != self.pin.stream
                ):
                    self.pin.stream.wait_stream(obligation.stream)
                self._obligations[index] = _SlotRelease(obligation.allocations, self.pin.stream)

    def pending_write(self, resource: StateResource) -> PageWrite | None:
        """The invocation's pending private write of `resource`, which the tasks transitioning it share."""
        (write,) = [
            entry.write
            for entry in self._obligations
            if isinstance(entry, _WriteAbort) and self.arena.access_resource(entry.write) == resource
        ] or (None,)
        return write

    @property
    def writing(self) -> bool:
        """Whether any private write is pending."""
        return any(isinstance(entry, _WriteAbort) for entry in self._obligations)

    @property
    def publication(self) -> PagePublication | None:
        """The invocation's latest publication, of any state."""
        return next(reversed(self.publications), None)

    def publication_of(self, resource: StateResource) -> PagePublication | None:
        """The invocation's latest publication of `resource`."""
        return next(
            (
                receipt
                for receipt in reversed(self.publications)
                if self.arena.ref_resource(receipt.after_refs) == resource
            ),
            None,
        )

    def current_owned_ref(self, original: StateRef) -> StateRef:
        return self.current_owned_refs(StateRefs.of((original,)))[0]

    def current_owned_refs(self, refs: StateRefs) -> StateRefs:
        """Follow this invocation's publications in order without materializing individual identities."""
        for receipt in self.publications:
            refs = refs.replace_versions(receipt.before_refs, receipt.after_refs)
        return refs

    def check_page_execution(self) -> None:
        if self.cancel_requested:
            raise RuntimeError("state invocation was cancelled")

    def prepare_owned_write(
        self,
        sources: StateRefs,
        fields: tuple[tuple[LayoutFieldId, ...], ...] | None = None,
        *,
        copy_fields: tuple[LayoutFieldId, ...] | None = (),
    ) -> PageWrite:
        """Prepare and retain a private transition, returning its explicit write token."""
        self.check_page_execution()
        authorized = self.current_owned_refs(self.pin.refs)
        assert authorized == sources or (authorized.positions(sources) >= 0).all()
        token = self.arena.prepare_transition(sources, self.pin.stream, fields=fields, copy_fields=copy_fields)
        self._obligations.append(_WriteAbort(token))
        return token

    def owned_write_slots(self, sources: StateRefs, *, copy_fields: tuple[LayoutFieldId, ...] | None) -> Slots:
        """Prepare a private transition of `sources` and return its destination slots: a row per source, a column
        per field."""
        write = self.prepare_owned_write(sources, copy_fields=copy_fields)
        return self.arena.slot_matrix(write, write.after_refs)[1]

    def publish_owned_write(self, token: PageWrite) -> PagePublication:
        self.check_page_execution()
        obligation = _WriteAbort(token)
        assert obligation in self._obligations
        receipt = self.arena.publish(token)
        self.publications.append(receipt)
        self._obligations.remove(obligation)
        self.trace.record(
            TraceEventKind.PUBLISH, self.invocation_id, refs=receipt.after_refs, operation_id=receipt.operation_id
        )
        return receipt

    def finish_pages(self, phases: tuple[_PagePhase, ...] = ("abort", "release")) -> None:
        """Discharge each phase in registration order, retaining failures for retry."""
        for phase in phases:
            for obligation in tuple(self._obligations):
                if obligation.phase == phase:
                    self._discharge(obligation)

    def _discharge(self, obligation: _PageObligation) -> None:
        assert obligation in self._obligations
        receipt = obligation.complete(self.arena)
        if receipt is not None:
            self.releases.append(receipt)
        self._obligations.remove(obligation)

    def move_owned_pin(self, pin: PagePin, stream: torch.cuda.Stream) -> PagePin:
        """Move a retained reader and its release obligation without reacquiring its pages."""
        index = self._obligations.index(_PinRelease(pin))
        moved = self.arena.move_pin(pin, stream)
        self._obligations[index] = _PinRelease(moved)
        return moved

    def release_owned_pin(self, pin: PagePin) -> None:
        """Discharge an early release through the same operation as terminal cleanup."""
        self._discharge(_PinRelease(pin))

    def read_versions(
        self,
        identities: Sequence[RequestIdentity],
        resource: StateResource,
        directory: RequestStateDirectory,
        *,
        latest: bool = False,
    ) -> tuple[tuple[PagePin, StateRef], ...]:
        """Per request, the version of `resource` a model-side read sees and the pin that keeps it readable.

        Without `latest`, an unkeyed (instance) resource is read at each request's admitted inference pin.
        Otherwise a read sees the newest version this invocation published from the selected ref, under one
        pin borrowing every such publication, or the selected ref itself under the invocation pin.
        """
        group = self.selection.for_resource(resource)
        if group.instance and not latest:
            pins = tuple(directory.inference_pin(identity) for identity in identities)
            return tuple(
                (pin, next(ref for ref in pin.refs if self.arena.ref_resource(StateRefs.of((ref,))) == resource))
                for pin in pins
            )
        if group.instance:
            entry = group.refs.take((0,) * len(identities))
        else:
            # One index for the whole batch: decode replays resolve every request before each graph launch.
            index = dict(zip((row.identity for row in self.selection.rows), group.state_indices, strict=True))
            entry = self.selection.refs.take(tuple(index[identity] for identity in identities))
        refs = self.current_owned_refs(entry)
        current = self.current_owned_refs(group.refs)
        updated = current.take(np.flatnonzero(current.versions != group.refs.versions))
        borrowed = self.borrow_published(updated) if updated else self.pin
        return tuple(
            (self.pin if ref == selected else borrowed, ref) for ref, selected in zip(refs, entry, strict=True)
        )

    def borrow_published(self, refs: StateRefs) -> PagePin:
        self.check_page_execution()
        authorized = np.zeros(len(refs), dtype=np.bool_)
        for receipt in self.publications:
            authorized |= receipt.after_refs.positions(refs) >= 0
        assert refs and authorized.all()
        return self.pin_for(refs)

    def pin_for(self, refs: StateRefs) -> PagePin:
        """Reuse a live invocation pin covering these exact refs, or acquire one."""
        self.check_page_execution()
        for entry in self._obligations:
            if isinstance(entry, _PinRelease) and (entry.pin.refs.positions(refs) >= 0).all():
                return entry.pin
        pin = self.arena.pin_exact(refs, self.pin.stream)
        self.register_pin(pin)
        return pin


class TaskGraphImplementation(Protocol):
    """One planned task's body, run by StateExecution in Rust's task order."""

    @property
    def implementation_id(self) -> ImplementationId: ...

    def execute(self, context: StateTaskContext) -> ProjectionResiduals: ...


@dataclass(frozen=True, slots=True)
class MathBinding:
    """Local native implementations, separate from the closed data contracts."""

    math: FunctionMath
    tasks: tuple[TaskGraphImplementation, ...]
    feeds: tuple[FeedBinding, ...] = ()
    on_export: Callable[[TraceEvent], None] | None = None

    def local_resources(self, plan: PlanBundle) -> tuple[str, ...]:
        """States this process owns; a fed resource is borrowed from its producer."""
        borrowed = {feed.resource for feed in self.feeds}
        return tuple(str(state.resource) for state in plan.semantic_graph.states if state.resource not in borrowed)

    def bind(self, arena: PageBundleArena, sampler_anchor_id: AnchorId) -> StateExecution:
        states = arena.plan.semantic_graph.states
        # A borrowed binding owns only its local directory; test_video verifies no remote lineage.
        if self.feeds:
            states = tuple(state for state in states if state.resource in arena.owned_resources)
        execution = StateExecution(
            arena,
            RequestStateDirectory(arena, states),
            self.math,
            self.tasks,
            sampler_anchor_id,
            feeds=self.feeds,
            on_export=self.on_export,
        )
        return execution


@dataclass(frozen=True, slots=True)
class ConditionalProducerTask:
    original: TaskGraphImplementation

    @property
    def implementation_id(self) -> ImplementationId:
        return self.original.implementation_id

    def execute(self, context: StateTaskContext) -> ProjectionResiduals:
        context.check_page_execution()
        if context.held_input is None:
            return self.original.execute(context)
        context.trace.record(TraceEventKind.INPUT_SKIPPED, context.invocation_id)
        return {}


@dataclass(frozen=True, slots=True)
class PublishStateTask:
    implementation_id: ImplementationId
    resource: StateResource

    def memory(self, data: TensorFrame, batch: TensorBatch) -> MemoryUsage:
        """Publication changes ownership metadata; PTTS already funds its tensor pages."""
        return MemoryUsage(0, 0)

    def input_shapes(
        self, rows: int, hidden: int, training_shapes: tuple[tuple[int, ...], ...]
    ) -> tuple[tuple[int, ...], ...]:
        return ()

    def resident(self, capture_bytes: int) -> int:
        return 0

    def execute(self, context: StateTaskContext) -> ProjectionResiduals:
        write = context.pending_write(self.resource)
        if write is not None:
            context.publish_owned_write(write)
        return {}


@dataclass(frozen=True, slots=True)
class StateDiagnostics:
    active_states: int
    inference_pins: int
    loss_count: int
    update_count: int
    trained_rows: int
    completed_sampler_pins: int
    pending_state_releases: int
    state_usage: PageBundleUsage
    hold_count: int = 0
    last_hold: HeldInput | None = None
    # Why the model runs without CUDA graphs (ModelConfig.cuda_graphs); empty when it uses them.
    eager_reason: str = ""
    numeric_health: tuple[FieldNumericHealth, ...] = ()
    # Actual vLLM allocation and concurrency at max_model_len, per TP rank.
    kv_cache_blocks: int = 0
    kv_cache_max_concurrency: float = 0.0

    def logical_counts(self) -> tuple[int, ...]:
        """Rank-invariant progress, excluding allocator/event timing and device identities."""
        return (
            self.active_states,
            self.inference_pins,
            self.loss_count,
            self.update_count,
            self.trained_rows,
            self.completed_sampler_pins,
            self.hold_count,
        )


class PartialCommitError(RuntimeError):
    def __init__(self, context: StateTaskContext, receipt: PagePublication) -> None:
        super().__init__(
            "the state update is published but finishing the step failed; finish it again (finish_step) without rerunning the step"
        )
        self.context = context
        self.receipt = receipt


class StateExecution:
    def __init__(
        self,
        arena: PageBundleArena,
        directory: RequestStateDirectory,
        math: FunctionMath,
        implementations: tuple[TaskGraphImplementation, ...],
        sampler_anchor_id: AnchorId,
        *,
        feeds: tuple[FeedBinding, ...] = (),
        on_export: Callable[[TraceEvent], None] | None = None,
    ) -> None:
        self.arena, self.directory, self.math = arena, directory, math
        self.sampler_anchor_id = sampler_anchor_id
        self._anchors: dict[str, int] = {anchor.symbol: anchor.ordinal for anchor in arena.plan.timeline.anchors}
        self.trace = ExecutionTrace(arena.plan.schedule, on_export=on_export)
        self._values = tuple(
            (task.symbol, output.name) for task in arena.plan.semantic_graph.tasks for output in task.outputs
        )
        bindings = {implementation.implementation_id: implementation for implementation in implementations}
        assert len(bindings) == len(implementations)
        assert set(bindings) == {task.implementation_id for task in self.plan.tasks}
        # Rust numbers tasks densely, so position i holds the implementation of task i.
        self._implementations = tuple(
            bindings[task.implementation_id] for task in sorted(self.plan.tasks, key=lambda task: task.task_id)
        )
        self._active: dict[int, StateTaskContext] = {}
        self._next_invocation = 1
        self._loss_count = self._update_count = self._trained_rows = self._sampler_count = 0
        self._training_counters = DeviceCounters(self.device)
        self.feeds = feeds
        self.side_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda" and any(task.lane == "side" for task in self.plan.tasks)
            else None
        )
        prefetched = {
            self.arena.plan.semantic_graph.tasks[task.task_id].symbol
            for task in self.plan.tasks
            if task.lane == "side" and task.read_origin != "latest"
        }
        self._prefetch_consumers = frozenset(
            TaskId(index)
            for index, task in enumerate(self.arena.plan.semantic_graph.tasks)
            if any(
                isinstance(port.source, TaskResultInputSource) and port.source.task in prefetched
                for port in task.inputs
            )
        )
        self.post_sample = self._post_sample_overlap()
        self._pending_borrows: tuple[BorrowedStateRoot, ...] | None = None
        self._borrow_failed = False
        self._successful_inputs: dict[RequestIdentity, tuple[tuple[StateRef, ...], int]] = {}
        self._hold_count = 0
        self._last_hold: HeldInput | None = None

    def _post_sample_overlap(self) -> PostSampleOverlap | None:
        """Bind the latest-version writer lane and the retained-reader policy."""
        from pllm.execution.post_sample import PostSampleOverlap  # tensor_frame imports this module

        tasks = self.plan.tasks
        if not any(task.read_origin == "latest" for task in tasks):
            return None
        stream = (
            None
            if self.device.type == "cpu"
            else (
                self.side_stream
                if any(task.lane == "side" for task in tasks)
                else torch.cuda.current_stream(self.device)
            )
        )
        anchor = min(task.anchor_id for task in tasks if task.read_origin == "latest")
        return PostSampleOverlap(self.arena, self.directory, max(task.read_distance for task in tasks), stream, anchor)

    def close(self) -> None:
        """Finish asynchronous exports when direct ownership or a backend session ends; idempotent."""
        self.trace.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.close()

    def discard_borrows(self) -> None:
        """Retain failed grants until all GPU work ends, then discard their aliases."""
        roots = list(self._pending_borrows or ())
        for context in self._active.values():
            roots.extend(context.borrowed_roots)
            context.value = None
            context.results.clear()
        if roots and self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        for root in roots:
            root.discard()
        self._pending_borrows = None
        self._borrow_failed = True

    def _require_borrow_live(self) -> None:
        # Only feed (borrowing) paths set _borrow_failed; test_video covers the fail-stop.
        if self._borrow_failed:
            raise RuntimeError(
                "this engine stopped after a fed state failed to arrive; create a new engine with pllm.make_llm"
            )

    def stage_readonly(self, roots: tuple[BorrowedStateRoot, ...]) -> None:
        """Internal transport handoff; no change to workload authoring signatures."""
        self._require_borrow_live()
        self._pending_borrows = roots  # Retain even invalid incoming mappings until the session terminates.
        try:
            for root in roots:
                root.claim(self.arena.allocation_ranges)
        except Exception:
            self._borrow_failed = True
            raise

    def stage_hold(self, identities: tuple[RequestIdentity, ...], reason: str, *, step_id: int) -> HeldInput:
        """Return the HOLD branch with exact versions from a successful commit."""
        self._require_borrow_live()
        try:
            if (
                self.arena.plan.semantic_graph.input_groups
                and not self.arena.plan.semantic_graph.input_groups[0].allow_absent
            ):
                raise RuntimeError('missing activation input: on_missing="stop" stops the request')
            return HeldInput(
                reason, step_id, tuple((identity, *self._successful_inputs[identity]) for identity in identities)
            )
        except Exception:
            self._borrow_failed = True
            raise

    @property
    def plan(self) -> ScheduleDecision:
        return self.arena.plan.schedule

    def _release_values(self, context: StateTaskContext, values: tuple[int, ...]) -> None:
        for value in values:
            context.results.pop(self._values[value], None)

    def implementation(self, identity: ImplementationId) -> TaskGraphImplementation:
        return next(item for item in self._implementations if item.implementation_id == identity)

    def rebind_implementation(self, implementation: TaskGraphImplementation) -> None:
        """Install a locally verified lowering while no original invocation is active."""
        original = self.implementation(implementation.implementation_id)
        self._implementations = tuple(implementation if item is original else item for item in self._implementations)

    @property
    def device(self) -> torch.device:
        return self.arena.device

    def admit(
        self, identities: tuple[RequestIdentity, ...], stream: torch.cuda.Stream | None = None
    ) -> tuple[StateRef, ...]:
        self._require_borrow_live()
        if not self.directory.states:
            return ()
        receipt = self.directory.admit(identities, stream, self.math.initialize)
        # One visible ref per unique admitted key, whether created or joined.
        refs = self.latest(identities)
        if receipt is None:
            self.trace.record(TraceEventKind.ADMIT, 0, identities=identities, refs=refs)
        else:
            self.trace.record(
                TraceEventKind.ADMIT, 0, identities=identities, refs=refs, operation_id=receipt.operation_id
            )
        return refs

    def latest(self, identities: tuple[RequestIdentity, ...]) -> tuple[StateRef, ...]:
        """The newest published version of each admitted state of `identities`; `()` names the instance states."""
        return self.directory.latest_refs(identities)

    def admit_instances(self, stream: torch.cuda.Stream | None = None) -> None:
        """Admit instance-scoped state once, when the execution is installed; requests admit their own state."""
        if self.directory.instance_states:
            self.admit((), stream)

    def begin(
        self,
        rows: PackedRows,
        stream: torch.cuda.Stream | None = None,
        *,
        history: tuple[StateRefs, ...] = (),
        held_input: HeldInput | None = None,
        retain_gradients: bool = True,
    ) -> StateTaskContext:
        """Open a state invocation; token-only consumers can omit returned gradient snapshots."""
        self._require_borrow_live()
        self.arena.collect_completed()
        selection = self.directory.select(rows, history)
        pin = self.arena.pin_exact(selection.refs, stream)
        context = StateTaskContext(selection, self.arena, pin, self.trace, self._next_invocation)
        context.held_input = held_input
        context.retain_gradients = retain_gradients
        self._next_invocation += 1
        self._active[context.invocation_id] = context
        if self.feeds and context.held_input is None:
            assert self._pending_borrows is not None
            context.borrowed_roots = self._pending_borrows
            self._pending_borrows = None
            try:
                for root in context.borrowed_roots:
                    root.bind(pin)
            except Exception as exc:
                self._failed(context, exc)
        self.trace.record(
            TraceEventKind.BEGIN,
            context.invocation_id,
            refs=pin.refs,
            pin_id=pin.pin_id,
            identities=selection.rows.identities,
        )
        return context

    def start_step(
        self,
        value: StepInput,
        stream: torch.cuda.Stream | None = None,
        *,
        history: tuple[StateRefs, ...] = (),
        held_input: HeldInput | None = None,
        retain_gradients: bool = True,
    ) -> StateTaskContext:
        """Prepare one invocation, retaining gradients for direct step results unless the consumer opts out."""
        context = self.begin(
            self.math.rows(value), stream, history=history, held_input=held_input, retain_gradients=retain_gradients
        )
        self.submit_input(context, value)
        return context

    def advance_to(self, context: StateTaskContext, symbol: str) -> None:
        """Run every anchor up to `symbol`, handing the step to the post-sample overlap before its first latest read."""
        target = self._anchors[symbol]
        for ordinal in range(context.last_anchor_id + 1, target + 1):
            if self.post_sample is not None and ordinal == self.post_sample.anchor:
                self.post_sample.hand_off(context)
            self.execute_anchor(context, AnchorId(ordinal))

    def step(
        self, value: StepInput, stream: torch.cuda.Stream | None = None, *, feedback: SamplerFeedback | None = None
    ) -> TensorOutput:
        """Run without an external sampler borrower using the same staged lifecycle."""
        if stream is None and self.side_stream is not None:
            stream = torch.cuda.current_stream(self.device)
        context = self.start_step(value, stream)
        try:
            for anchor in self.arena.plan.timeline.anchors:
                self.advance_to(context, anchor.symbol)
                if anchor.ordinal == self.sampler_anchor_id:
                    self.complete_sampler(context, stream, feedback=feedback)
            return self.finish_step(context, stream)
        except WindowBackpressure:
            # The convenience call exposes no cursor; staged callers keep it through execute_anchor.
            self.cancel_step(context)
            raise

    def submit_input(
        self,
        context: StateTaskContext,
        value: StepInput,
        readiness: CudaStreamReadiness | None = None,
    ) -> None:
        """Consume original inputs, ordering an external CUDA producer when supplied."""
        self._require_borrow_live()
        try:
            if readiness is not None:
                assert context.pin.stream is not None
                readiness.order_on(context.pin.stream)
            context.value = self.math.prepare(value, context)
        except Exception as exc:
            self.trace.record(TraceEventKind.FAILED, context.invocation_id)
            # Failed preparation can leave aliases in traceback frames; test_video retains grants.
            if context.borrowed_roots:
                self._failed(context, exc)
            self.cancel_step(context)
            raise

    def execute_anchor(self, context: StateTaskContext, anchor_id: AnchorId) -> None:
        """Advance the step through every task at this anchor, retaining its cursor on backpressure."""
        self._require_borrow_live()
        try:
            context.check_page_execution()
            plan = self.plan
            dense_anchor = int(anchor_id)
            assert not context.finished and 0 <= dense_anchor and context.last_anchor_id <= dense_anchor
            assert context.next_task == len(plan.tasks) or dense_anchor <= plan.tasks[context.next_task].anchor_id
            while context.next_task < len(plan.tasks) and int(plan.tasks[context.next_task].anchor_id) == dense_anchor:
                task = plan.tasks[context.next_task]
                self.execute_task(context, task)
                context.next_task += 1
            context.last_anchor_id = dense_anchor
        except WindowBackpressure:
            # No cell of the pending task issued; its original cursor and pins remain usable.
            raise
        except Exception as exc:
            self._failed(context, exc)

    def execute_task(self, context: StateTaskContext, task: PhysicalTask) -> None:
        """Borrow the same exact versions on the chosen lane until its kernels finish."""
        prefetch = task.lane == "side" and task.read_origin != "latest"
        stream = self.side_stream if prefetch else context.pin.stream
        if prefetch:
            context.prefetch_stream = stream
            assert stream is not None
            stream.wait_stream(torch.cuda.current_stream(self.device))
        elif stream is not None and task.task_id in self._prefetch_consumers and context.prefetch_stream is not None:
            stream.wait_stream(context.prefetch_stream)
        with nullcontext() if stream is None else torch.cuda.stream(stream):
            pin = context.pin
            if prefetch:
                context.pin = self.arena.pin_exact(pin.refs, stream)
            try:
                with self.trace.task(context.invocation_id, task.task_id, stream):
                    self._implementations[int(task.task_id)].execute(context)
            finally:
                if prefetch:
                    self.arena.release_pin(context.pin)
                context.pin = pin
            if task.release_values:
                self._release_values(context, task.release_values)

    def execute_part(
        self,
        context: StateTaskContext,
        value: StepInput,
        task_id: TaskId,
        *,
        complete: bool,
    ) -> ProjectionResiduals:
        """Supply actual projection operands to one segment of a bound in-forward task."""
        self._require_borrow_live()
        try:
            context.check_page_execution()
            context.value = self.math.prepare(value, context)
            plan = self.plan
            assert not context.finished and context.next_task < len(plan.tasks)
            task = plan.tasks[context.next_task]
            assert task.task_id == task_id and task.anchor_id >= context.last_anchor_id
            with self.trace.task(context.invocation_id, task_id, context.pin.stream):
                result = self._implementations[int(task_id)].execute(context)
            context.last_anchor_id = int(task.anchor_id)
            if complete:
                if task.release_values:
                    self._release_values(context, task.release_values)
                context.next_task += 1
            return result
        except Exception as exc:
            self._failed(context, exc)

    def sampler_guidance(
        self,
        context: StateTaskContext,
        stream: torch.cuda.Stream | None = None,
    ) -> tuple[PackedSamplerGuidanceBatch, ...]:
        self._require_borrow_live()
        assert context.guidance
        if stream is not None:
            assert context.guidance_ready is not None
            context.guidance_ready.order_on(stream)
            context.sampler_stream = stream
        return context.guidance

    def complete_sampler(
        self,
        context: StateTaskContext,
        stream: torch.cuda.Stream | None = None,
        *,
        feedback: SamplerFeedback | None = None,
    ) -> None:
        self._require_borrow_live()
        self._order_sampler(context, stream)
        context.sampler_feedback = feedback
        self._sampler_count += 1
        self.trace.record(TraceEventKind.SAMPLER_COMPLETE, context.invocation_id, pin_id=context.pin.pin_id)

    def finish_step(self, context: StateTaskContext, stream: torch.cuda.Stream | None = None) -> TensorOutput:
        """Complete the step; pass the stream that will read the result so it is ordered after the writer lane.

        The vLLM session passes none: it discards results, and waiting would serialize the overlap.
        """
        self._require_borrow_live()
        if context.prefetch_stream is not None and stream is not None:
            stream.wait_stream(context.prefetch_stream)
        try:
            result = self.math.output(context)
            if not context.finished:
                assert context.next_task == len(self.plan.tasks)
                context.finished = True
            self._release(context)
        except Exception as exc:
            self._failed(context, exc)
            raise
        if self.post_sample is not None and self.post_sample.stream is not None and stream is not None:
            CudaStreamReadiness(self.post_sample.stream).order_on(stream)
        return result

    def cancel_step(self, context: StateTaskContext) -> None:
        context.cancel_requested = True
        self._require_borrow_live()
        # Cancellation cannot prove that remote aliases were discarded; test_video checks no ACK.
        if context.borrowed_roots or context.held_input is not None:
            self._failed(context, RuntimeError("borrowed invocation cancelled; resources retained"))
        self._order_sampler(context, context.sampler_stream)
        context.finish_pages(("abort",))
        self.trace.record(TraceEventKind.ABORT, context.invocation_id)
        self._release(context)

    def cancel_active(self) -> None:
        for context in tuple(self._active.values()):
            self.cancel_step(context)
        self.release_history()

    def release_history(self) -> None:
        """Release retained reader versions before retirement or drain, also after cancellation."""
        if self.post_sample is not None:
            self.post_sample.release_history()

    def _failed(self, context: StateTaskContext, exc: Exception) -> NoReturn:
        context.cancel_requested = True
        self.trace.record(
            TraceEventKind.FAILED,
            context.invocation_id,
            operation_id=None if context.publication is None else context.publication.operation_id,
        )
        # Quarantine borrowed failures, including exception-frame aliases; test_video covers both publication outcomes.
        if context.borrowed_roots or context.held_input is not None:
            self._borrow_failed = True
            if context.publication is None:
                try:
                    context.finish_pages(("abort",))
                except Exception as cleanup:
                    exc.add_note(f"local abort also failed; resources retained: {cleanup}")
                raise exc
        if context.publication is not None:
            raise PartialCommitError(context, context.publication) from exc
        self.cancel_step(context)
        raise exc

    def _release(self, context: StateTaskContext) -> None:
        # Discard aliases before borrowed ACKs and the commit; test_video checks ACK/failure ordering.
        context.value = None
        context.results.clear()
        context.loss = None
        context.guidance = ()
        context.sampler_feedback = None
        context.guidance_ready = None
        for root in context.borrowed_roots:
            root.close()
        context.finish_pages()
        self._active.pop(context.invocation_id)
        release = next(receipt for receipt in context.releases if receipt.pin is context.pin)
        self.trace.record(
            TraceEventKind.COMMITTED,
            context.invocation_id,
            pin_id=context.pin.pin_id,
            release_id=release.release_id,
            operation_id=None if context.publication is None else context.publication.operation_id,
        )
        if context.publication is not None or context.finished:
            self._loss_count += context.loss_count
            self._update_count += context.update_count
            self._trained_rows += context.trained_rows
            if context.training_counters is not None:
                self._training_counters.add(context.training_counters, context.pin.stream)
        if context.finished and self.arena.plan.semantic_graph.input_groups and context.held_input is None:
            for identity, refs in context.injected_refs.items():
                self._successful_inputs[identity] = tuple(sorted(refs)), context.input_step_id
        if context.finished and context.held_input is not None:
            self._hold_count += 1
            self._last_hold = context.held_input

    def _order_sampler(
        self,
        context: StateTaskContext,
        stream: torch.cuda.Stream | None,
    ) -> None:
        consumer = context.pin.stream
        if stream is not None and consumer is not None:
            CudaStreamReadiness(stream).order_on(consumer)

    def retire(self, identities: tuple[RequestIdentity, ...], *, wait: bool = False) -> tuple[StateRef, ...]:
        for identity in identities:
            self._successful_inputs.pop(identity, None)
        if not self.directory.states:
            self.arena.collect_completed(wait=wait)
            self.trace.record(TraceEventKind.RETIRE, 0, identities=identities)
            return ()
        if not identities:
            for context in tuple(self._active.values()):
                if any(address.key is None for address in context.selection.addresses):
                    self.cancel_step(context)
        elif self.directory.request_states:
            retiring = set(identities)
            # Only the retiring requests' own steps cancel; requests sharing a
            # route lineage keep running, and the lineage outlives the last
            # holder until dormant retirement.
            for context in tuple(self._active.values()):
                if any(row.identity in retiring for row in context.selection.rows):
                    self.cancel_step(context)
        self.arena.collect_completed(wait=wait)
        refs = self.directory.retire(identities)
        self.trace.record(TraceEventKind.RETIRE, 0, identities=identities, refs=refs)
        return refs

    def retire_dormant(self, *, wait: bool = False) -> tuple[StateRef, ...]:
        self.arena.collect_completed(wait=wait)
        refs = self.directory.retire_dormant()
        self.trace.record(TraceEventKind.RETIRE, 0, identities=(), refs=refs)
        return refs

    def snapshot(self, ref: StateRef, stream: torch.cuda.Stream | None = None) -> tuple[torch.Tensor, ...]:
        pin = self.arena.pin_exact(StateRefs.of((ref,)), stream)
        try:
            return self.arena.snapshot(pin, ref)
        finally:
            self.arena.wait_release(self.arena.release_pin(pin))

    def diagnostics(self) -> StateDiagnostics:
        """Exact counts for every step submitted so far; waits on the device counters' latest copy, so keep it off per-step paths."""
        self.arena.collect_completed()
        updates, rows = self._training_counters.read(wait=True)
        return StateDiagnostics(
            self.directory.active_count,
            self.directory.inference_pin_count,
            self._loss_count,
            self._update_count + updates,
            self._trained_rows + rows,
            self._sampler_count,
            self.arena.pending_pin_releases,
            self.arena.usage(),
            self._hold_count,
            self._last_hold,
            numeric_health=self.arena.numeric_health(),
        )
