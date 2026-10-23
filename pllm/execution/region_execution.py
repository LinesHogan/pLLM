"""Execute source-verified region instructions with imported cells and PTTS effects.

The loop cursor and mathematical value frame implement a fixed native plan.
Resource access goes through the original invocation's page capabilities.
"""

from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass

import torch

from pllm.contracts.region_bindings import RegionBindings
from pllm.contracts.region_imports import RegionTensorImport
from pllm.contracts.region_program import Call, Program, bind_sizes
from pllm.contracts.region_proposal import RegionScheduleProposal
from pllm.contracts.region_schedule import ScheduleInstance, Statement
from pllm.contracts.region_selection import RegionPayloadLimit
from pllm.contracts.region_streams import RegionStateWindow, RegionStreamPlan
from pllm.contracts.schedule_instructions import LoweredSchedule
from pllm.execution.cuda_stream import CudaStreamReadiness
from pllm.execution.exported_cell import ExportedCell
from pllm.execution.region_compilation import lower_region_bindings, lower_region_schedule
from pllm.execution.region_control import RegionActions
from pllm.execution.region_frame import RegionValueFrame, RegionValueLayout, Routes, expand_accesses
from pllm.execution.region_native import (
    AffineScanPrimitive,
    BorrowedTensorPrimitive,
    CompiledPrimitive,
    FunctionalKernelPrimitive,
    NativePrimitive,
    RegionFunction,
)
from pllm.execution.region_resources import RegionResources
from pllm.execution.region_schedule import RegionScheduleExecution
from pllm.execution.region_storage import import_region_functions
from pllm.execution.region_streams import (
    RegionStreamBindings,
    RegionStreamFrame,
    record_stream,
    validate_stream_budget,
)
from pllm.execution.region_values import Data, copy_data
from pllm.execution.region_window import RegionWindow, WindowBackpressure

PreparedWindow = tuple[tuple[ScheduleInstance, ...], RegionValueLayout, Routes, tuple[tuple[int, Routes], ...]]


@dataclass(frozen=True, slots=True)
class RegionExecutionResult:
    """Owned results, ordered on the invocation stream; completion can order other consumers."""

    values: tuple[Data, ...]
    invocations: int
    # Outer scheduled frame only; not a GPU memory budget or nested-slot count.
    peak_live_values: int
    released_values: int
    window_reuses: int = 0
    completion: CudaStreamReadiness | None = None


@dataclass(frozen=True, slots=True)
class RegionExecutionPlan:
    """Owner-free source plan; local functions are supplied only when binding.

    This is an in-process composition of existing contracts, not another wire
    format. Numerical alternatives still require the selection/replay path.
    """

    program: Program
    imports: tuple[RegionTensorImport, ...]
    proposal: RegionScheduleProposal
    payload_limits: tuple[RegionPayloadLimit, ...]
    state_windows: tuple[RegionStateWindow, ...] = ()

    def __post_init__(self) -> None:
        if import_region_functions(self.program, self.imports) != self.imports:
            raise ValueError("execution plan requires canonical source imports")
        # Local derivation is authoritative: assembled facts are checked for
        # plain consistency, never re-derived from source a second time.
        if self.stream_plan is not None and self.stream_plan.payload != self.proposal.payload:
            raise ValueError("execution plan and stream plan require the same payload")
        budgets = {limit.device: limit.bytes for limit in self.payload_limits}
        if len(budgets) != len(self.payload_limits) or any(
            type(value) is not int or value < 0 for value in budgets.values()
        ):
            raise ValueError("execution plan budgets must be distinct nonnegative device bounds")
        if any(
            limit.device not in budgets or limit.bytes > budgets[limit.device] for limit in self.proposal.peak_bytes
        ):
            raise ValueError("execution plan exceeds its original payload reservation")

    @property
    def stream_plan(self) -> RegionStreamPlan | None:
        return self.proposal.stream_plan

    def bind(self, cells: Mapping[str, RegionFunction]) -> "RegionInstructionExecution":
        if any(
            isinstance(cell, (CompiledPrimitive, AffineScanPrimitive))
            or isinstance(cell, ExportedCell)
            and cell.transformations
            for cell in cells.values()
        ):
            raise ValueError("transformed cells require a numerical selection decision")
        if import_region_functions(self.program, cells) != self.imports:
            raise ValueError("local functions differ from the planned source imports")
        schedule = self.proposal.payload.capacity.schedule_map
        return RegionInstructionExecution(
            self.program,
            lower_region_schedule(self.program, schedule),
            lower_region_bindings(self.program, schedule),
            cells,
            stream_plan=self.stream_plan,
            payload_limits=self.payload_limits,
            state_windows=self.state_windows,
        )


class RegionInstructionExecution:
    """Prepare each concrete window once, then execute its table with fresh frames."""

    def __init__(
        self,
        program: Program,
        lowered: LoweredSchedule,
        bindings: RegionBindings,
        cells: Mapping[str, RegionFunction],
        *,
        stream_plan: RegionStreamPlan | None = None,
        payload_limits: tuple[RegionPayloadLimit, ...] = (),
        state_windows: tuple[RegionStateWindow, ...] = (),
    ) -> None:
        self._schedule = RegionScheduleExecution(program, lowered)
        self._prepared: dict[tuple[tuple[str, int], ...], PreparedWindow] = {}
        if bindings.program_digest != lowered.program_digest or bindings.schedule_map != lowered.schedule_map:
            raise ValueError("tensor bindings and loop instructions must describe the same schedule")
        if stream_plan is not None:
            validate_stream_budget(stream_plan, payload_limits)
            if bindings.schedule_map != stream_plan.payload.capacity.schedule_map:
                raise ValueError("physical stream plan differs from the bound phase schedule")
        if set(cells) != {f.name for f in program.functions}:
            raise ValueError("region cell bindings must be complete and exact")
        for declaration in program.functions:
            cell = cells[declaration.name]
            if type(cell) not in (
                ExportedCell,
                NativePrimitive,
                FunctionalKernelPrimitive,
                CompiledPrimitive,
                BorrowedTensorPrimitive,
                AffineScanPrimitive,
            ):
                raise ValueError("region calls require an exported cell or explicit native import")
            if cell.declaration != declaration:
                raise ValueError("cell differs from its imported source declaration")
        self._program = program
        self.payload_limits = payload_limits
        self._lowered = lowered
        self._bindings = bindings
        self._cells = dict(cells)
        operations = program.entry.block.operations
        self._borrow_inputs = (
            len(operations) == 1
            and isinstance(operations[0], Call)
            and isinstance(cells[operations[0].function], AffineScanPrimitive)
        )
        self._state_windows = stream_plan.state_windows if stream_plan is not None else state_windows
        self._actions = RegionActions(program, cells)
        self._stream_plan = stream_plan
        self._window = None if stream_plan is None else RegionWindow(stream_plan.slots)
        self._tensor_borrows = {
            name: cell.implementation for name, cell in cells.items() if isinstance(cell, BorrowedTensorPrimitive)
        }

    def _prepare(self, parameters: tuple[tuple[str, int], ...]) -> PreparedWindow:
        if parameters in self._prepared:
            return self._prepared[parameters]
        sizes = dict(parameters)
        domains = {statement.id: statement.extents for statement in self._lowered.statements}
        times = expand_accesses(self._bindings.times, domains, sizes)
        instances = tuple(
            ScheduleInstance(item.statement_id, item.coordinates, times[item.statement_id, item.coordinates][1])
            for item in self._schedule.invocations(sizes)
        )
        plan = self._stream_plan
        self._prepared[parameters] = (
            instances,
            RegionValueLayout(self._bindings, sizes, self._program.resources),
            {} if plan is None else expand_accesses(plan.placements, domains, sizes),
            ()
            if plan is None
            else tuple((wait.lane, expand_accesses(wait.predecessors, domains, sizes)) for wait in plan.waits),
        )
        return self._prepared[parameters]

    @property
    def stream_plan(self) -> RegionStreamPlan | None:
        return self._stream_plan

    @property
    def cells(self) -> Mapping[str, RegionFunction]:
        """The bound local implementations, as checked at bind time."""
        return self._cells

    def _invoke(
        self,
        frame: RegionValueFrame,
        statement: Statement,
        coordinates: tuple[int, ...],
        pages: RegionResources | None,
        sizes: Mapping[str, int],
        stream: torch.cuda.Stream | None = None,
    ) -> None:
        # Keep operand/result temporaries in one invocation's scope. Otherwise
        # Python's loop locals can retain the previous result through the next
        # kernel, after the native last-use plan has released its frame slot.
        ins = tuple(frame.read(i, coordinates) for i in statement.inputs)
        if stream is not None and self._window is None:
            for value in ins:
                record_stream(value, stream)
        try:
            result = self._actions.execute(self._lowered.actions[statement.id], ins, pages, sizes)
            for output, value in zip(statement.outputs, result, strict=True):
                frame.put(output, coordinates, value)
        finally:
            if self._window is not None:
                self._window.finish_reads()

    def run(
        self,
        inputs: Mapping[str, Data],
        *,
        pages: RegionResources | None = None,
        sizes: Mapping[str, int] = {},
        streams: RegionStreamBindings | None = None,
    ) -> RegionExecutionResult:
        sizes = bind_sizes(self._program, sizes)
        if streams is None and self._stream_plan is not None:
            raise ValueError("borrowed streams and a verified physical stream plan must be supplied together")
        instances, layout, placements, waits = self._prepare(tuple(sorted(sizes.items())))
        owner = None if streams is None else streams.owner
        if streams is not None:
            if self._stream_plan is None:
                streams.validate_available()
            else:
                streams.validate(self._stream_plan)
                if torch.cuda.is_current_stream_capturing():
                    raise ValueError(
                        "run region windows eagerly; graph replay needs device-side version and completion state"
                    )
            if pages is not None:
                pages.validate_stream(streams.owner)
            if self._stream_plan is None:
                streams = None  # Available lanes can be unused by the installed plan.
        if self._program.resources and pages is None:
            raise ValueError("resource programs require PTTS primitive bindings")
        if pages is not None:
            pages.validate(self._program, sizes)
            pages.validate_cells(layout.resource_uses)
            pages.validate_native_kernels(
                {name: cell.implementation for name, cell in self._cells.items() if isinstance(cell, NativePrimitive)}
            )
            pages.validate_tensor_borrows(self._tensor_borrows)
        scope = pages.scope() if pages is not None else (nullcontext() if owner is None else torch.cuda.stream(owner))
        window_scope = nullcontext() if self._window is None else self._window.reserve()
        page_window = nullcontext() if pages is None or not self._state_windows else pages.window(self._state_windows)
        with torch.no_grad(), window_scope, page_window, scope:
            if streams is not None:
                for value in inputs.values():
                    record_stream(value, streams.owner)
            frame = RegionValueFrame(layout, inputs, sizes, self._window, borrow_inputs=self._borrow_inputs)
            count = 0
            concurrent: RegionStreamFrame | None = None
            try:
                if streams is not None and self._stream_plan is not None:
                    concurrent = RegionStreamFrame(self._stream_plan, streams, placements, waits)
                for invocation in instances:
                    statement = self._lowered.statements[invocation.statement_id]
                    stream = (
                        None
                        if concurrent is None
                        else concurrent.enter(invocation.time, statement.id, invocation.coordinates)
                    )
                    # GPU credit waits and slot events protect physical reuse;
                    # logical lifetimes can advance without observing completion.
                    frame.advance(invocation.time)
                    with nullcontext() if stream is None else torch.cuda.stream(stream):
                        self._invoke(frame, statement, invocation.coordinates, pages, sizes, stream)
                    count += 1
                if concurrent is not None:
                    concurrent.order_results()
                values = frame.results()
                if self._tensor_borrows or self._window is not None:
                    # An exported result (including a nested bank/control
                    # result) can alias a borrowed model parameter. Materialize
                    # the complete public result tuple; AER charges this ABI.
                    try:
                        values = tuple(copy_data(value) for value in values)
                    finally:
                        if self._window is not None:
                            self._window.finish_reads()
                if streams is not None:
                    for value in values:
                        record_stream(value, streams.owner)
                return RegionExecutionResult(
                    values,
                    count,
                    frame.peak_live_values,
                    frame.released_values,
                    0 if self._window is None else self._window.reuses,
                    None if streams is None else CudaStreamReadiness(streams.owner),
                )
            except WindowBackpressure as exc:
                raise RuntimeError("an executing cell requested backpressure after window admission") from exc
            finally:
                try:
                    if concurrent is not None:
                        concurrent.complete()
                finally:
                    frame.close()


class RegionExecution(RegionInstructionExecution):
    """Bind original source imports; optimized variants need selection replay."""

    def __init__(
        self, program: Program, lowered: LoweredSchedule, bindings: RegionBindings, cells: Mapping[str, RegionFunction]
    ) -> None:
        if any(
            isinstance(cell, (CompiledPrimitive, AffineScanPrimitive))
            or isinstance(cell, ExportedCell)
            and cell.transformations
            for cell in cells.values()
        ):
            raise ValueError("transformed cells require a numerical selection decision")
        super().__init__(program, lowered, bindings, cells)
