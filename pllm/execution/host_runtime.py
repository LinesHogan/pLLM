"""Workload-independent physical facts for synchronous typed host execution."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable, Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import Generic, Protocol, TypeVar

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from pllm.contracts.codec import boundary_digest
from pllm.contracts.feedback import Completion, CompletionBatch, FinishReason
from pllm.contracts.request_identity import RequestIdentity
from pllm.contracts.semantic_graph import (
    AnchorId,
    AnchorSymbol,
    SchedulerTimeline,
    SemanticGraph,
    TaskResourceFact,
    TaskSymbol,
    TimelineAnchor,
    ValueResourceFact,
)
from pllm.contracts.state_representation import StateDeviceKind
from pllm.contracts.tensor_layout import TensorLayoutRequest
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.declaration import Builder, Call, HostOperation, When, native
from pllm.execution.execution import TaskGraphExecution, TaskGraphImplementation
from pllm.execution.ptts.page_bundle import PageBundleArena
from pllm.execution.ptts.page_bundle_planning import PageBundleProviderFacts, compile_page_bundle_plan
from pllm.execution.scheduler import compile_schedule
from pllm.execution.state_execution import MathBinding, StateExecution
from pllm.execution.torch_private import MemoryTrace, MemoryUsage
from pllm.execution.trace import ExecutionTrace, TracePhase

RequestT = TypeVar("RequestT")
TransactionT = TypeVar("TransactionT")
ResultT = TypeVar("ResultT")
RequestT_contra = TypeVar("RequestT_contra", contravariant=True)
TransactionT_contra = TypeVar("TransactionT_contra", contravariant=True)
ResultT_co = TypeVar("ResultT_co", covariant=True)
ContextT = TypeVar("ContextT")
SnapshotT = TypeVar("SnapshotT")
CompiledT = TypeVar("CompiledT", covariant=True)
InputT = TypeVar("InputT")
DataT = TypeVar("DataT")


class HostBinding(Protocol[CompiledT]):
    def compile(self) -> CompiledT: ...


class HostWorkload(ABC, Generic[CompiledT]):
    @abstractmethod
    def declare_host(self, facts: HostRuntimeFacts) -> HostBinding[CompiledT]: ...

    def compile_host(self, facts: HostRuntimeFacts) -> CompiledT:
        return self.declare_host(facts).compile()


@dataclass(frozen=True, slots=True)
class TensorHostBinding(Generic[InputT, DataT, ResultT]):
    graph: SemanticGraph
    layouts: TensorLayoutRequest
    math: MathBinding[InputT, DataT, ResultT]
    facts: HostRuntimeFacts
    sampler_anchor: AnchorSymbol = AnchorSymbol("host.next")
    timeline: SchedulerTimeline | None = None

    def compile(self) -> StateExecution[InputT, DataT, ResultT]:
        facts = self.facts
        timeline = facts.timeline() if self.timeline is None else self.timeline
        limits = tuple(
            value for value in (timeline.memory_budget_bytes, facts.memory_budget_bytes) if value is not None
        )
        timeline = replace(timeline, memory_budget_bytes=min(limits, default=None))
        providers = (
            (
                PageBundleProviderFacts(
                    f"canonical.page_bundle_{facts.device.type}@1",
                    3,
                    StateDeviceKind(facts.device.type),
                    facts.device.index or 0,
                ),
            )
            if self.layouts.layouts
            else ()
        )
        plan = compile_page_bundle_plan(
            (self.graph, self.layouts),
            timeline,
            providers,
            1,
            facts.state_memory_budget_bytes,
            facts.device,
        )
        (anchor,) = (AnchorId(item.ordinal) for item in timeline.anchors if item.symbol == self.sampler_anchor)
        return self.math.bind(PageBundleArena(plan, device=facts.device), anchor)


@dataclass(frozen=True, slots=True)
class TransactionHostBinding(Generic[RequestT, TransactionT, ContextT, SnapshotT, ResultT]):
    graph: SemanticGraph
    state_factory: Callable[[], HostTaskState[TransactionT, ContextT, SnapshotT, ResultT]]
    tasks: tuple[TaskGraphImplementation[ContextT, None], ...]
    service: HostTransactionService[RequestT, TransactionT, ResultT]
    facts: HostRuntimeFacts
    timeline: SchedulerTimeline | None = None

    def compile(self) -> HostTransactionCoordinator[RequestT, TransactionT, ContextT, SnapshotT, ResultT]:
        facts = self.facts
        timeline = facts.timeline() if self.timeline is None else self.timeline
        limits = tuple(
            value for value in (timeline.memory_budget_bytes, facts.memory_budget_bytes) if value is not None
        )
        timeline = replace(timeline, memory_budget_bytes=min(limits, default=None))
        execution = HostExecution(self.graph, timeline, self.state_factory, self.tasks, facts.device)
        return HostTransactionCoordinator(self.service, execution)


@dataclass(frozen=True, slots=True)
class CompletionInput:
    """One owned completion/hidden boundary, shared by host consumers."""

    completions: CompletionBatch
    hidden: torch.Tensor
    readiness: CudaStreamReadiness | None = None

    @classmethod
    def example(cls, facts: HostRuntimeFacts, device: torch.device = torch.device("meta")) -> CompletionInput:
        """A shape-only input with domain-neutral completion metadata."""
        if facts.max_input_rows is None:
            raise ValueError("finite completion costs require max_input_rows")
        entries = tuple(
            Completion(RequestIdentity(str(i), i, 0), i, str(i), (), "", FinishReason.LENGTH)
            for i in range(facts.max_input_rows)
        )
        return cls(
            CompletionBatch(entries), torch.empty((len(entries), facts.hidden_size), dtype=torch.float32, device=device)
        )

    @classmethod
    def capture(
        cls, completions: CompletionBatch, hidden: torch.Tensor, stream: torch.cuda.Stream | None = None
    ) -> CompletionInput:
        if (
            (hidden.device.type == "cuda") != (stream is not None)
            or (stream is not None and stream.device != hidden.device)
            or hidden.ndim != 2
            or hidden.shape[0] != len(completions.entries)
            or hidden.device.type not in ("cpu", "cuda")
        ):
            raise ValueError("completion input requires rows aligned to completions on its producer stream")
        with nullcontext() if stream is None else torch.cuda.stream(stream), torch.inference_mode(False):
            owned = hidden.detach().to(dtype=torch.float32, copy=True)
            return cls(completions, owned, CudaStreamReadiness(stream) if stream is not None else None)


class CompletionCapture:
    """Write final sampling rows directly into the original owned output tensor."""

    def __init__(self, identities: tuple[RequestIdentity, ...], hidden: torch.Tensor) -> None:
        self.identities, self.hidden = identities, hidden
        self._indices = {identity: index for index, identity in enumerate(identities)}
        self._written: set[RequestIdentity] = set()
        self._readiness = (
            CudaStreamReadiness(torch.cuda.current_stream(hidden.device)) if hidden.device.type == "cuda" else None
        )

    def write(self, identities: tuple[RequestIdentity, ...], rows: torch.Tensor) -> None:
        assert rows.shape == (len(identities), self.hidden.shape[1])
        with torch.inference_mode(False), torch.no_grad():
            if self._readiness is not None:
                self._readiness.order_on(torch.cuda.current_stream(self.hidden.device))
                self.hidden.record_stream(torch.cuda.current_stream(self.hidden.device))
            for identity, row in zip(identities, rows.unbind(), strict=True):
                self.hidden[self._indices[identity]].copy_(row.detach(), non_blocking=False)
                self._written.add(identity)
            if self.hidden.device.type == "cuda":
                self._readiness = CudaStreamReadiness(torch.cuda.current_stream(self.hidden.device))

    def require_complete(self) -> None:
        if len(self._written) != len(self.identities):
            raise RuntimeError("completion capture is missing submitted identities")

    def synchronize(self) -> None:
        if self._readiness is not None:
            self._readiness.synchronize()

    def finish(self, completions: CompletionBatch) -> CompletionInput:
        self.require_complete()
        assert tuple(entry.identity for entry in completions.entries) == self.identities
        return CompletionInput(completions, self.hidden, self._readiness)


class HostTaskState(Protocol[TransactionT_contra, ContextT, SnapshotT, ResultT_co]):
    """Workload mathematics and reversible state, without transaction ownership."""

    @property
    def version(self) -> int: ...

    def prepare(self, transaction: TransactionT_contra) -> ContextT:
        """Validate/order input readiness without mutating observable state."""
        ...

    def snapshot(self) -> SnapshotT: ...

    def restore(self, snapshot: SnapshotT) -> None: ...

    def output(self, context: ContextT) -> ResultT_co: ...


@dataclass(frozen=True, slots=True)
class HostReceipt(Generic[ResultT]):
    receipt_id: int
    result: ResultT
    completion: torch.cuda.Event | None


@dataclass(frozen=True, slots=True)
class _Pending(Generic[ResultT, SnapshotT]):
    receipt: HostReceipt[ResultT]
    snapshot: SnapshotT
    stream: torch.cuda.Stream | None


@dataclass(frozen=True, slots=True)
class _Failed:
    pass


@dataclass(frozen=True, slots=True)
class HostDiagnostics:
    transaction_count: int
    state_version: int


class HostExecution(Generic[TransactionT, ContextT, SnapshotT, ResultT]):
    """One receipt, rollback and completion owner for Rust-planned host task graphs."""

    def __init__(
        self,
        graph: SemanticGraph,
        timeline: SchedulerTimeline,
        state_factory: Callable[[], HostTaskState[TransactionT, ContextT, SnapshotT, ResultT]],
        implementations: tuple[TaskGraphImplementation[ContextT, None], ...],
        device: torch.device,
    ) -> None:
        self.decision = compile_schedule(graph, timeline)
        self.trace = ExecutionTrace(self.decision)
        self._runner = TaskGraphExecution(self.decision, implementations, self.trace)
        self._state = state_factory()
        self._device = device
        self._initialization = CudaStreamReadiness(torch.cuda.current_stream(device)) if device.type == "cuda" else None
        self._active: _Pending[ResultT, SnapshotT] | _Failed | None = None
        self._next_receipt_id = 1
        self._transaction_count = 0

    def execute(self, transaction: TransactionT) -> HostReceipt[ResultT]:
        if self._active is not None:
            raise RuntimeError("host execution is pending or fail-stopped")
        receipt_id = self._next_receipt_id
        self._next_receipt_id += 1
        self.trace.record(TracePhase.BEGIN, receipt_id)
        stream = torch.cuda.current_stream(self._device) if self._device.type == "cuda" else None
        if stream is not None and self._initialization is not None:
            self._initialization.order_on(stream)
        # Readiness is ordered before the snapshot; no workload owns a second cursor.
        snapshot: SnapshotT | None = None
        with torch.inference_mode(False), torch.enable_grad():
            try:
                context = self._state.prepare(transaction)
                snapshot = self._state.snapshot()
                result = self._state.output(self._runner.run(context, receipt_id))
                completion = None
                if stream is not None:
                    completion = torch.cuda.Event(blocking=False)
                    completion.record(stream)
                    completion.synchronize()
            except Exception:
                self.trace.record(TracePhase.FAILED, receipt_id)
                if snapshot is not None:
                    self._restore(snapshot, stream)
                    self.trace.record(TracePhase.ABORT, receipt_id)
                raise
        receipt = HostReceipt(receipt_id, result, completion)
        self._active = _Pending(receipt, snapshot, stream)
        self.trace.record(TracePhase.PREPARED, receipt_id)
        return receipt

    def validate_receipt(self, receipt: HostReceipt[ResultT]) -> _Pending[ResultT, SnapshotT]:
        match self._active:
            case _Pending(receipt=issued) as pending if issued is receipt:
                return pending
            case _:
                raise RuntimeError("host execution receipt is not active")

    def commit(self, receipt: HostReceipt[ResultT]) -> None:
        self.validate_receipt(receipt)
        self._transaction_count += 1
        self._active = None
        self.trace.record(TracePhase.SETTLED, receipt.receipt_id)

    def abort(self, receipt: HostReceipt[ResultT]) -> None:
        pending = self.validate_receipt(receipt)
        self._restore(pending.snapshot, pending.stream)
        self._active = None
        self.trace.record(TracePhase.ABORT, receipt.receipt_id)

    def _restore(self, snapshot: SnapshotT, stream: torch.cuda.Stream | None) -> None:
        try:
            if stream is None:
                self._state.restore(snapshot)
            else:
                stream.synchronize()
                with torch.cuda.stream(stream):
                    self._state.restore(snapshot)
                stream.synchronize()
        except Exception:
            self._active = _Failed()
            raise

    def diagnostics(self) -> HostDiagnostics:
        return HostDiagnostics(self._transaction_count, self._state.version)


class HostTransactionService(Protocol[RequestT_contra, TransactionT, ResultT]):
    def begin(self, request: RequestT_contra) -> TransactionT | None: ...

    def commit(self, transaction: TransactionT, receipt: HostReceipt[ResultT]) -> None:
        """Publish atomically: an exception must leave the transaction abortable."""
        ...

    def abort(self, transaction: TransactionT) -> None: ...


@dataclass(frozen=True, slots=True)
class NoHostRequest:
    """Closed request value for a service driven only by its owned queue."""


NO_HOST_REQUEST = NoHostRequest()


class HostSettlementError(RuntimeError, Generic[ResultT]):
    """The service committed; settle this exact receipt without replaying input."""

    def __init__(self, receipt: HostReceipt[ResultT]) -> None:
        super().__init__("host publication committed; execution settlement remains pending")
        self.receipt = receipt


@dataclass(frozen=True, slots=True)
class HostTransactionCoordinator(Generic[RequestT, TransactionT, ContextT, SnapshotT, ResultT]):
    """Atomically settle one typed host transaction and execution receipt."""

    service: HostTransactionService[RequestT, TransactionT, ResultT]
    execution: HostExecution[TransactionT, ContextT, SnapshotT, ResultT]

    def run(self, request: RequestT) -> ResultT | None:
        transaction = self.service.begin(request)
        if transaction is None:
            return None
        try:
            receipt = self.execution.execute(transaction)
        except Exception:
            self.service.abort(transaction)
            raise
        try:
            self.execution.validate_receipt(receipt)
            self.service.commit(transaction, receipt)
        except Exception:
            try:
                self.execution.abort(receipt)
            finally:
                self.service.abort(transaction)
            raise
        try:
            self.execution.trace.record(TracePhase.PUBLISH, receipt.receipt_id)
            self.execution.commit(receipt)
        except Exception as exc:
            self.execution.trace.record(TracePhase.FAILED, receipt.receipt_id)
            raise HostSettlementError(receipt) from exc
        return receipt.result


@dataclass(frozen=True, slots=True)
class HostRuntimeFacts:
    device: torch.device
    hidden_size: int = 0
    initialization_seed: int = 0
    state_memory_budget_bytes: int = 64 * 1024 * 1024
    max_input_rows: int | None = None
    memory_budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if (
            self.device.type not in ("cpu", "cuda")
            or self.hidden_size < 0
            or self.initialization_seed < 0
            or self.state_memory_budget_bytes < 0
            or self.max_input_rows is not None
            and self.max_input_rows < 0
            or self.memory_budget_bytes is not None
            and self.memory_budget_bytes < 0
        ):
            raise ValueError(
                "HostRuntimeFacts needs a cpu or cuda device and nonnegative hidden_size, initialization_seed, "
                f"state_memory_budget_bytes, max_input_rows and memory_budget_bytes; got {self}"
            )

    def timeline(
        self,
        extra_capabilities: tuple[str, ...] = (),
    ) -> SchedulerTimeline:
        compute_capability = "compute/cpu@1" if self.device.type == "cpu" else "compute/cuda@1"
        return SchedulerTimeline(
            memory_budget_bytes=self.memory_budget_bytes,
            anchors=(
                TimelineAnchor(AnchorSymbol("host.input.ready"), 0),
                TimelineAnchor(AnchorSymbol("host.compute.end"), 1),
                TimelineAnchor(AnchorSymbol("host.output.publish"), 2),
                TimelineAnchor(AnchorSymbol("host.next"), 3),
            ),
            capabilities=tuple(
                sorted(
                    (
                        compute_capability,
                        "host/service@1",
                        *extra_capabilities,
                    )
                )
            ),
        )


def transaction(
    name: str,
    facts: HostRuntimeFacts,
    *,
    config: Hashable,
    state_factory: Callable[[], HostTaskState[TransactionT, ContextT, SnapshotT, ResultT]],
    service: HostTransactionService[RequestT, TransactionT, ResultT],
    update: Callable[[ContextT], None],
    query: Callable[[ContextT], None],
    publish: Callable[[ContextT], None],
    example: Callable[[HostRuntimeFacts], tuple[ContextT, SnapshotT]] | None = None,
) -> TransactionHostBinding[RequestT, TransactionT, ContextT, SnapshotT, ResultT]:
    """Bind native structured math; retain the original state, service and context."""
    b = Builder[ContextT](name, boundary_digest(config, facts))
    resource = b.state("state")
    changed = b.compute(
        "update",
        host_function=update,
        inputs={"context": b.input(name, "transaction", f"{name}.transaction@1")},
        outputs=("context",),
        changes=resource,
        when=When.HOST_COMPUTE,
        implementations=(native(f"{name}:update@1", f"compute/{facts.device.type}@1"),),
    )
    queried = b.compute(
        "query",
        host_function=query,
        inputs={"context": changed["context"]},
        outputs=("context",),
        reads=(resource.version(changed),),
        when=When.HOST_OUTPUT,
        implementations=(native(f"{name}:query@1", f"compute/{facts.device.type}@1"),),
    )
    published = b.compute(
        "publish",
        host_function=publish,
        inputs={"context": queried["context"]},
        outputs=("result",),
        result_schema=f"{name}.result@1",
        reads=(resource.version(changed),),
        when=When.HOST_COMMIT,
        implementations=(native(f"{name}:publish@1", "host/service@1", host=True),),
    )
    timeline = None
    if example is not None:
        timeline = _transaction_timeline(facts, example, b.host_operations, (changed, queried, published))
    return TransactionHostBinding(b.graph, state_factory, b.bind_host(), service, facts, timeline)


def _transaction_timeline(
    facts: HostRuntimeFacts,
    example: Callable[[HostRuntimeFacts], tuple[ContextT, SnapshotT]],
    operations: Mapping[TaskSymbol, HostOperation[ContextT]],
    calls: tuple[Call, ...],
) -> SchedulerTimeline:
    """Run the transaction calls in order on fake tensors, with the live context and its rollback snapshot."""
    if facts.memory_budget_bytes is None:
        return facts.timeline()
    if facts.max_input_rows is None:
        raise ValueError("finite transaction costs require max_input_rows")
    costs: list[tuple[Call, MemoryUsage]] = []
    with FakeTensorMode(), MemoryTrace() as lifetime:
        context, snapshot = example(facts)
        resident = lifetime.live
        for call in calls:
            usage = operations[call.symbol].memory(context)
            # The context carries every output forward, so a call's output is all growth since the example.
            costs.append((call, MemoryUsage(usage.workspace, max(0, lifetime.live - resident))))
        del snapshot
    return measured_timeline(facts, resident, costs, ((facts.max_input_rows, facts.hidden_size),))


def measured_timeline(
    facts: HostRuntimeFacts,
    reserved_bytes: int,
    costs: list[tuple[Call, MemoryUsage]],
    input_shapes: tuple[tuple[int, ...], ...] = (),
) -> SchedulerTimeline:
    """The host timeline with each call's measured workspace and the bytes its one output keeps."""
    timeline = facts.timeline()
    costs = sorted(costs, key=lambda cost: cost[0].symbol)
    return replace(
        timeline,
        reserved_bytes=(reserved_bytes,) * len(timeline.anchors),
        task_resources=tuple(
            TaskResourceFact(call.symbol, usage.workspace, input_shapes=input_shapes) for call, usage in costs
        ),
        value_resources=tuple(ValueResourceFact(call.symbol, call.outputs[0], usage.output) for call, usage in costs),
    )


__all__ = [
    "HostRuntimeFacts",
    "measured_timeline",
    "transaction",
    "CompletionInput",
    "HostWorkload",
    "HostExecution",
    "HostReceipt",
    "HostTransactionCoordinator",
    "HostTransactionService",
    "NO_HOST_REQUEST",
    "NoHostRequest",
]
