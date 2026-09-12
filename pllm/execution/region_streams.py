"""Borrow invocation streams and consume native phase/placement accessors."""

from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import torch

from pllm import _core
from pllm.contracts.region_program import Program
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_streams import RegionStreamExploration, RegionStreamPlan
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.region_frame import lookup
from pllm.execution.region_storage import RegionImportSource, import_region_functions
from pllm.execution.region_values import Bank, Data


def lower_region_streams(program: Program, schedule: str, cells: RegionImportSource, *,
                         axis: int, lanes: int, retained: tuple[RetainedAllocation, ...] = (), pipeline_depth: int = 1) -> RegionStreamPlan:
    return decode_boundary(_core.lower_region_streams(encode_boundary(program), schedule, axis, lanes,
        encode_boundary(import_region_functions(program, cells)), encode_boundary(retained), pipeline_depth), RegionStreamPlan)


def validate_region_streams(program: Program, cells: RegionImportSource, plan: RegionStreamPlan,
                            retained: tuple[RetainedAllocation, ...] = ()) -> None:
    _core.validate_region_streams(encode_boundary(program), encode_boundary(import_region_functions(program, cells)),
                                 encode_boundary(retained), encode_boundary(plan))


def explore_region_streams(program: Program, cells: RegionImportSource, *, lanes: int,
                           max_candidates: int = 4, retained: tuple[RetainedAllocation, ...] = (),
                           max_pipeline_depth: int = 1) -> RegionStreamExploration:
    """Find physical phases in the shared ISL catalog for borrowed stream capacity."""
    return decode_boundary(_core.explore_region_streams(encode_boundary(program),
        encode_boundary(import_region_functions(program, cells)), encode_boundary(retained), lanes, max_candidates, max_pipeline_depth),
        RegionStreamExploration)


def validate_stream_budget(plan: RegionStreamPlan, limits: tuple[RegionPayloadLimit, ...]) -> None:
    budgets = {limit.device: limit.bytes for limit in limits}
    if len(budgets) != len(limits) or any(type(n) is not int or n < 0 for n in budgets.values()):
        raise ValueError("stream payload limits must be distinct nonnegative device budgets")
    for device in plan.devices:
        if device.device not in budgets or device.peak_bytes_upper_bound > budgets[device.device]:
            raise ValueError("concurrent stream payload exceeds the invocation's available budget")


def record_stream(value: Data, stream: torch.cuda.Stream) -> None:
    if isinstance(value, torch.Tensor):
        if value.device != stream.device:
            raise ValueError("stream execution requires tensors on its borrowed device")
        value.record_stream(stream)
    elif isinstance(value, Bank):
        for element in value.elements:
            record_stream(element, stream)


@dataclass(frozen=True, slots=True)
class RegionStreamBindings:
    """Streams supplied by the original invocation; lane zero is its owner."""

    owner: torch.cuda.Stream
    auxiliary: tuple[torch.cuda.Stream, ...]

    def validate_available(self) -> None:
        streams = (self.owner, *self.auxiliary)
        if not 2 <= len(streams) <= 8 or any(type(s) is not torch.cuda.Stream for s in streams):
            raise ValueError("stream bindings require two to eight borrowed streams")
        if any(s.device != self.owner.device for s in streams) or len({s.cuda_stream for s in streams}) != len(streams):
            raise ValueError("physical lanes require distinct streams on the owner's device")

    def validate(self, plan: RegionStreamPlan) -> None:
        self.validate_available()
        if len(self.auxiliary) + 1 != plan.lanes:
            raise ValueError("stream bindings must cover the physical lanes exactly")
        if any(d.device != str(self.owner.device) for d in plan.devices):
            raise ValueError("stream plan belongs to another tensor or retention device")


@dataclass(frozen=True, slots=True)
class _PendingPhase:
    phase: tuple[int, ...]
    completions: Mapping[int, CudaStreamReadiness]
    values: tuple[Data, ...]


class RegionStreamFrame:
    """A bounded completion window for one fixed plan, under the original owner.

    GPU waits follow the native per-lane dependency frontier. References survive
    logical last use until their phase finishes; a full window applies host
    backpressure before another phase can allocate. Opaque host/page operations
    drain prior work. This object neither owns pages nor chooses a schedule.
    """

    def __init__(self, plan: RegionStreamPlan, bindings: RegionStreamBindings, sizes: Mapping[str, int],
                 hold_values: Callable[[], tuple[Data, ...]]) -> None:
        self._streams = (bindings.owner, *bindings.auxiliary)
        self._placements = {table.input_id: table for table in plan.placements}
        self._waits = tuple((wait.lane, {table.input_id: table for table in wait.predecessors}) for wait in plan.waits)
        self._host_fences = frozenset(plan.host_fences)
        self._depth = plan.pipeline_depth
        self._hold_values = hold_values
        self._pending: deque[_PendingPhase] = deque()
        self._events: dict[tuple[int, ...], Mapping[int, CudaStreamReadiness]] = {}
        self._completed: tuple[int, ...] | None = None
        self._waited: dict[tuple[int, int], tuple[int, ...]] = {}
        self._sizes = sizes
        self._phase: tuple[int, ...] | None = None
        self._active: set[int] = {0}
        # Called after the owner has copied the entry values. No auxiliary
        # stream may read those copies (or prior owner work) before this event.
        ready = CudaStreamReadiness(bindings.owner)
        for stream in bindings.auxiliary:
            ready.order_on(stream)

    def enter(self, phase: tuple[int, ...], statement_id: int, coordinates: tuple[int, ...]) -> torch.cuda.Stream:
        if self._phase != phase:
            if self._phase is not None:
                if phase < self._phase:
                    raise RuntimeError("stream cursor moved backwards in phase time")
                self._seal()
                while len(self._pending) >= self._depth:
                    self._retire()
            self._phase = phase
        placement = lookup(self._placements.get(statement_id), coordinates, self._sizes)
        if placement is None or len(placement[1]) != 1:
            raise RuntimeError("statement has no verified physical stream")
        lane, = placement[1]
        if not 0 <= lane < len(self._streams):
            raise RuntimeError("physical stream is outside the borrowed lanes")
        if statement_id in self._host_fences:
            # A CPU callback or page operation can inspect state immediately;
            # queued CUDA waits alone do not order those host observations.
            self._drain()
            self._join(self._record())
        for producer, tables in self._waits:
            if producer == lane:
                continue  # Stream issue order already covers this frontier.
            predecessor = lookup(tables.get(statement_id), coordinates, self._sizes)
            if predecessor is None:
                continue
            required = predecessor[1]
            if required >= phase:
                raise RuntimeError("pipeline predecessor does not precede its consumer phase")
            if self._completed is not None and required <= self._completed:
                continue
            previous = self._waited.get((lane, producer))
            if previous is not None and required <= previous:
                continue
            events = self._events.get(required)
            if events is None or producer not in events:
                raise RuntimeError("pipeline consumer has no issued producer completion")
            events[producer].order_on(self._streams[lane])
            self._waited[(lane, producer)] = required
        self._active.add(lane)
        return self._streams[lane]

    def _record(self) -> dict[int, CudaStreamReadiness]:
        return {lane: CudaStreamReadiness(self._streams[lane]) for lane in sorted(self._active)}

    def _join(self, completions: Mapping[int, CudaStreamReadiness]) -> None:
        # Record all producers before inserting owner waits, avoiding an event
        # that accidentally includes a later join to another producer.
        for readiness in completions.values():
            readiness.order_on(self._streams[0])
        for readiness in completions.values():
            readiness.synchronize()

    def _seal(self) -> None:
        if self._phase is None:
            return
        events = self._record()
        values = self._hold_values() if self._depth > 1 else ()
        self._pending.append(_PendingPhase(self._phase, events, values))
        self._events[self._phase] = events
        self._active.clear()

    def _retire(self) -> None:
        pending = self._pending[0]
        self._join(pending.completions)
        self._completed = pending.phase
        del self._events[pending.phase]
        self._pending.popleft()

    def _drain(self) -> None:
        while self._pending:
            self._retire()

    def order_results(self) -> None:
        """Order final owner-side copies after every borrowed producer lane."""
        ready = tuple(CudaStreamReadiness(stream) for stream in self._streams[1:])
        for completion in ready:
            completion.order_on(self._streams[0])
        # Final completion must include these copies, even when the last
        # scheduled statement ran solely on an auxiliary lane.
        self._active.add(0)

    def complete(self) -> None:
        if self._phase is None:
            self._join(self._record())
        else:
            self._seal()
            self._drain()
        self._active.clear()
