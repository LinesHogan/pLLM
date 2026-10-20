"""Borrow invocation streams and consume native phase/placement accessors."""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import torch

from pllm import _core
from pllm.contracts.region_program import Program
from pllm.contracts.region_proposal import ClosedRegionWindow, RegionScheduleProposal
from pllm.contracts.region_retention import RetainedAllocation
from pllm.contracts.region_selection import RegionPayloadLimit, RegionSelectionPolicy
from pllm.contracts.region_streams import RegionStreamExploration, RegionStreamPlan
from pllm.contracts.semantic_graph import TaskSourceRegion
from pllm.execution.codec import decode_boundary, encode_boundary
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.region_compilation import analyze_region_program
from pllm.execution.region_frame import lookup
from pllm.execution.region_storage import RegionImportSource, analyze_region_payload, import_region_functions
from pllm.execution.region_values import Bank, Data


def plan_closed_window(
    source: TaskSourceRegion, policy: RegionSelectionPolicy, *, parallel: bool, budget: int | None = None
) -> ClosedRegionWindow:
    """Derive and sign an engine cut before the enclosing task is partitioned."""
    assert source.window is not None
    return decode_boundary(
        _core.plan_closed_region_window(
            encode_boundary(source.program),
            encode_boundary(source.imports),
            encode_boundary(source.window),
            encode_boundary(policy),
            parallel,
            budget,
        ),
        ClosedRegionWindow,
    )


def lower_region_streams(
    program: Program,
    schedule: str,
    cells: RegionImportSource,
    *,
    axis: int,
    lanes: int,
    retained: tuple[RetainedAllocation, ...] = (),
    pipeline_depth: int = 1,
) -> RegionStreamPlan:
    return decode_boundary(
        _core.lower_region_streams(
            encode_boundary(program),
            schedule,
            axis,
            lanes,
            encode_boundary(import_region_functions(program, cells)),
            encode_boundary(retained),
            pipeline_depth,
        ),
        RegionStreamPlan,
    )


def explore_region_streams(
    program: Program,
    cells: RegionImportSource,
    *,
    schedule: str,
    retained: tuple[RetainedAllocation, ...] = (),
    payload_limits: tuple[RegionPayloadLimit, ...] | None = None,
) -> RegionStreamExploration:
    """First funded two-stream mapping of this order; no schedule search."""
    return decode_boundary(
        _core.explore_region_streams(
            encode_boundary(program),
            encode_boundary(import_region_functions(program, cells)),
            encode_boundary(retained),
            schedule,
            None if payload_limits is None else encode_boundary(payload_limits),
        ),
        RegionStreamExploration,
    )


def plan_region_schedule(
    program: Program,
    cells: RegionImportSource,
    *,
    payload_limits: tuple[RegionPayloadLimit, ...] | None = None,
    retained: tuple[RetainedAllocation, ...] = (),
    parallel: bool = False,
    policy: Literal["auto", "serial"] = "auto",
) -> RegionScheduleProposal | None:
    """Choose one funded order, then its first legal fixed two-stream mapping.

    Source facts stay unchanged. No window/tile catalog or timing is involved;
    the reference remains the admission baseline, never an execution retry.
    """
    budgets = None if payload_limits is None else {limit.device: limit.bytes for limit in payload_limits}
    if budgets is not None and (
        len(budgets) != len(payload_limits or ()) or any(type(n) is not int or n < 0 for n in budgets.values())
    ):
        raise ValueError("schedule budgets must be distinct nonnegative device bounds")
    analysis = analyze_region_program(program, optimize=policy == "auto", parallel_only=True)
    proposals = tuple(
        RegionScheduleProposal(analyze_region_payload(program, candidate.schedule_map, cells, retained))
        for candidate in reversed(analysis.candidates)
    )
    if parallel and policy == "auto":
        for proposal in proposals:
            mapped = arrange_region_schedule(
                program,
                proposal.payload.capacity.schedule_map,
                cells,
                parallel=True,
                retained=retained,
                payload_limits=payload_limits,
            )
            if mapped.stream_plan is not None:
                return mapped
    for proposal in proposals:
        if budgets is None or all(
            limit.device in budgets and limit.bytes <= budgets[limit.device] for limit in proposal.peak_bytes
        ):
            return proposal
    return None


def arrange_region_schedule(
    program: Program,
    schedule: str,
    cells: RegionImportSource,
    *,
    parallel: bool,
    retained: tuple[RetainedAllocation, ...] = (),
    payload_limits: tuple[RegionPayloadLimit, ...] | None = None,
) -> RegionScheduleProposal:
    """One physical arrangement for an explicitly selected advanced order."""
    if parallel:
        found = explore_region_streams(
            program, cells, schedule=schedule, retained=retained, payload_limits=payload_limits
        )
        if found.candidates:
            (plan,) = found.candidates
            return RegionScheduleProposal(plan.payload, plan)
    return RegionScheduleProposal(analyze_region_payload(program, schedule, cells, retained))


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


class RegionStreamFrame:
    """Device dependency and window-credit events; no host completion waits."""

    def __init__(self, plan: RegionStreamPlan, bindings: RegionStreamBindings, sizes: Mapping[str, int]) -> None:
        self._streams = (bindings.owner, *bindings.auxiliary)
        self._placements = {table.input_id: table for table in plan.placements}
        self._waits = tuple((wait.lane, {table.input_id: table for table in wait.predecessors}) for wait in plan.waits)
        self._depth = plan.pipeline_depth
        self._pending: deque[tuple[tuple[int, ...], dict[int, torch.cuda.Event]]] = deque()
        self._events: dict[tuple[int, ...], dict[int, torch.cuda.Event]] = {}
        self._event_pool = deque(
            {lane: torch.cuda.Event() for lane in range(len(self._streams))} for _ in range(plan.pipeline_depth + 1)
        )
        self._waited: dict[tuple[int, int], tuple[int, ...]] = {}
        self._sizes = sizes
        self._phase: tuple[int, ...] | None = None
        self._active: set[int] = {0}
        self._ordered_through: tuple[int, ...] | None = None
        ready = CudaStreamReadiness(bindings.owner)
        for stream in bindings.auxiliary:
            ready.order_on(stream)

    def enter(self, phase: tuple[int, ...], statement_id: int, coordinates: tuple[int, ...]) -> torch.cuda.Stream:
        if self._phase != phase:
            if self._phase is not None:
                assert phase > self._phase
                self._seal()
            if len(self._pending) >= self._depth:
                retired, credit = self._pending.popleft()
                for stream in self._streams:
                    for event in self._events[retired].values():
                        stream.wait_event(event)
                del self._events[retired]
                self._event_pool.append(credit)
                self._ordered_through = retired
            self._phase = phase
        placement = lookup(self._placements.get(statement_id), coordinates, self._sizes)
        assert placement is not None and len(placement[1]) == 1
        (lane,) = placement[1]
        stream = self._streams[lane]
        for producer, tables in self._waits:
            if producer == lane:
                continue
            predecessor = lookup(tables.get(statement_id), coordinates, self._sizes)
            if predecessor is None:
                continue
            required = predecessor[1]
            assert required < phase
            if self._ordered_through is not None and required <= self._ordered_through:
                continue  # Every lane already carries that retired phase's device wait.
            previous = self._waited.get((lane, producer))
            if previous is None or required > previous:
                stream.wait_event(self._events[required][producer])
                self._waited[(lane, producer)] = required
        self._active.add(lane)
        return stream

    def _seal(self) -> None:
        assert self._phase is not None
        events = self._event_pool.popleft()
        for lane in sorted(self._active):
            events[lane].record(self._streams[lane])
        self._pending.append((self._phase, events))
        self._events[self._phase] = {lane: events[lane] for lane in self._active}
        self._active.clear()

    def order_results(self) -> None:
        """Join every issued lane before owner-side materialization or settlement."""
        completions = tuple(CudaStreamReadiness(stream) for stream in self._streams[1:])
        for completion in completions:
            completion.order_on(self._streams[0])

    def complete(self) -> None:
        """Cancellation and normal return both enqueue the same owner join."""
        self.order_results()
        self._events.clear()
        self._pending.clear()
        self._active.clear()
