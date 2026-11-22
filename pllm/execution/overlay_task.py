"""In-forward overlays at model projections, run as ordinary state tasks.

Owns the overlay update rules (ChunkUpdate, GradientUpdate), the per-invocation projection driver
(ProjectionTask), OverlayTransition (one task's request/target adaptation) and ProgramOverlay. model_effect_binding
declares them; the vLLM backend calls ProjectionTask from its projection hooks.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, NamedTuple

import torch

from pllm.contracts.hidden_locator import HiddenPhase
from pllm.contracts.packed_rows import PackedPhaseRow, PackedPhaseRowBatch
from pllm.contracts.parameter_overlay import (
    FieldView,
    ParameterOverlayOperator,
    Projection,
    ProjectionDelta,
    ProjectionOperands,
    ProjectionShape,
)
from pllm.contracts.request_identity import RequestIdentity, RequestKeyField
from pllm.contracts.sampler import PackedSamplerRowBatch
from pllm.contracts.semantic_graph import StateResource, TaskId, TaskSymbol
from pllm.contracts.tensor_layout import LayoutFieldId
from pllm.contracts.visibility import VisibilitySource
from pllm.execution.parameter_projection import parameter_correction
from pllm.execution.ptts.request_state import RequestStateSelection
from pllm.execution.scan import execute_scan, prepare_scan
from pllm.execution.scan_analysis import chunk_readout
from pllm.execution.scan_execution import PreparedScan, Step
from pllm.execution.tensor_frame import OverlayEntry, OverlayTargetResult, TensorFrame, Tensors, UpdateOutput
from pllm.execution.tensor_program import PoolSGD, TaskOutput, TensorBatch, read_batch
from pllm.execution.torch_private import MemoryTrace
from pllm.kernels.page_copy import chunk_capacity, prepend_rows

if TYPE_CHECKING:
    from pllm.execution.overlay_batch import OverlayBatch
    from pllm.execution.state_execution import StateTaskContext
    from pllm.execution.step_driver import StepDriver


def join_tail(
    tails: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    length: torch.Tensor,
    chunk: int,
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    """Return bounded columns and their device-resident valid row count."""
    return prepend_rows(tails, rows, length, chunk), length + len(rows[0])


def save_tail(
    tails: tuple[torch.Tensor, ...],
    rows: tuple[torch.Tensor, ...],
    length: torch.Tensor,
    valid: torch.Tensor,
    trained: torch.Tensor,
) -> torch.Tensor:
    """Save unfinished valid rows; capacity padding never becomes retained history."""
    remaining = valid - trained
    for tail, values in zip(tails, rows, strict=True):
        positions = torch.arange(len(tail), device=length.device)
        selected = values.index_select(0, (positions + trained).clamp_max(len(values) - 1))
        tail.copy_(selected.masked_fill((positions >= remaining).view(-1, *((1,) * (tail.ndim - 1))), 0))
    length.copy_(remaining)
    return remaining


@dataclass(frozen=True)
class ChunkUpdate:
    """Authored recurrence metadata used by both custom-forward and batched overlay execution."""

    step: Step
    chunk: int
    arguments: tuple[str, ...]
    invariants: tuple[str, ...] = ()

    def __call__(self, **values: torch.Tensor) -> UpdateOutput:
        plan = self.prepare(values)
        count = values["_pending_length"].new_full((), len(values[self.arguments[1]]))
        return self.counted(plan, count, values)

    def prepare(self, values: Tensors) -> PreparedScan:
        """Inspect storage and select policy outside the mapped mathematical body."""
        carry, *columns = self.arguments
        over = tuple(
            values[name].new_empty(
                (chunk_capacity(len(values[name]), self.chunk), *values[name].shape[1:]),
                dtype=torch.promote_types(values[name].dtype, values[f"_pending_rows_{name}"].dtype),
            )
            for name in columns
        )
        return prepare_scan(
            self.step,
            values[carry],
            over,
            self.chunk,
            tuple(values[name] for name in self.invariants),
            valid_rows=values["_pending_length"],
        )

    def counted(self, plan: PreparedScan, count: torch.Tensor, values: Tensors) -> UpdateOutput:
        """Advance complete chunks only; padded requests and rows contribute no updates."""
        carry, *columns = self.arguments
        rows = tuple(values[name] for name in columns)
        tails = tuple(values[f"_pending_rows_{name}"] for name in columns)
        length = values["_pending_length"]
        joined, _ = join_tail(tails, rows, length, self.chunk)
        valid = length + count
        positions = torch.arange(len(rows[0]), device=length.device) + length
        trained = valid // self.chunk * self.chunk
        updated, outputs = execute_scan(
            plan, values[carry], joined, tuple(values[name] for name in self.invariants), trained
        )
        values[carry].copy_(updated)
        remaining = save_tail(tails, joined, length, valid, trained)
        return UpdateOutput(
            torch.cat(outputs).index_select(0, positions),
            trained > 0,
            trained // self.chunk,
            trained,
            {"tail_rows": remaining},
        )

    def readout(self, values: Tensors) -> Callable[..., torch.Tensor] | None:
        """Prepare the chunk's unchanged read arithmetic for eager tails and captured selection."""
        carry, *columns = self.arguments
        samples = (
            values[carry],
            *(values[f"_pending_rows_{name}"] for name in columns),
            *(values[name] for name in self.invariants),
        )
        return chunk_readout(self.step, tuple((t.shape, t.stride(), t.dtype, t.device) for t in samples))

    def contiguous(self, values: Tensors, plan: PreparedScan, offset: int) -> UpdateOutput:
        """Use the scheduler's contiguous token frontier without reading a device counter.

        Complete chunks run the authored step in order, keeping its GEMM shapes and rounding. The incomplete
        tail only evaluates its readout, so a carry that no chunk completes is never written.
        """
        carry, *columns = self.arguments
        pending = offset % self.chunk
        total = pending + len(values[columns[0]])
        full = total // self.chunk * self.chunk
        readout = self.readout(values)
        # A selected (non-serial) scan is specialized for a device-counted capacity shape.
        if readout is None or (full and plan.analysis is not None):
            return self(**values)
        tails = tuple(values[f"_pending_rows_{name}"] for name in columns)
        rows = tuple(
            torch.cat((tail[:pending], values[name])) if pending else values[name]
            for tail, name in zip(tails, columns, strict=True)
        )
        invariants = tuple(values[name] for name in self.invariants)
        outputs: list[torch.Tensor] = []
        if full:
            updated, scanned = execute_scan(plan, values[carry], tuple(row[:full] for row in rows), invariants, None)
            values[carry].copy_(updated)
            outputs.extend(scanned)
        length = values["_pending_length"]
        trained = length.new_full((), full)
        remaining = save_tail(tails, rows, length, length.new_full((), total), trained)
        if total > full:
            outputs.append(readout(values[carry], *tails, *invariants)[: total - full])
        return UpdateOutput(
            torch.cat(outputs)[pending:], trained > 0, trained // self.chunk, trained, {"tail_rows": remaining}
        )


def overlay_apply(effect: ProjectionDelta, hidden: torch.Tensor, values: Tensors) -> torch.Tensor:
    """Read the declared overlay operator for ordinary and chunk-exclusive projections."""
    return parameter_correction(hidden, tuple(values[name] for name in effect.applied_fields), effect.operator)


def count_updates(results: Sequence[UpdateOutput]) -> torch.Tensor:
    """The device [applied updates, trained rows] summed over results, in two kernels."""
    return torch.stack((*(r.update_count for r in results), *(r.trained_rows for r in results))).view(2, -1).sum(1)


Gradient = Callable[[Tensors, Tensors, Tensors, torch.Tensor], Tensors]


def overlay_gradient(function: Callable[..., torch.Tensor], *, batched: bool = False) -> Gradient:
    """Compose request mapping with differentiation before compiling; broadcast frozen weights."""

    def loss(parameters: Tensors, operands: Tensors, constants: Tensors, valid: torch.Tensor) -> torch.Tensor:
        rows = function(**parameters, **operands, **constants)
        if rows.shape != valid.shape:
            raise ValueError("autograd overlays must return one loss per input row")
        return torch.where(valid, rows.to(torch.promote_types(rows.dtype, torch.float32)), 0).sum()

    eager = torch.func.grad(loss)
    if batched:
        # Folding requests into the frozen-base GEMM changes CUDA's BF16 reduction.
        # Compile the chunks together while retaining each source GEMM's shape.
        eager = torch.vmap(eager, in_dims=(0, 0, None, 0), chunk_size=1)
    compiled = torch.compile(  # pyright: ignore[reportUnknownMemberType]
        eager, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False}
    )

    def gradient(parameters: Tensors, operands: Tensors, constants: Tensors, valid: torch.Tensor) -> Tensors:
        return (eager if valid.device.type != "cuda" or MemoryTrace.active.get() is not None else compiled)(
            parameters, operands, constants, valid
        )

    return gradient


@dataclass(frozen=True)
class GradientUpdate:
    """Chunk gradient sums and optimizer masks over independent request parameters."""

    gradient: Gradient
    mapped_gradient: Gradient
    optimizer: PoolSGD
    effect: ProjectionDelta
    chunk: int
    flush: bool
    bindings: Mapping[str, str]

    @property
    def input_name(self) -> str:
        """The authored argument carrying arriving rows."""
        return next(name for name, source in self.bindings.items() if source == "input")

    def __call__(self, **values: torch.Tensor) -> UpdateOutput:
        count = values["_pending_length"].new_full((), len(values[self.input_name]))
        return self.advance(values, count, batched=False)

    def prepare(self, values: Tensors) -> None:
        """Autograd has no scan selection."""

    def rows(
        self, values: Tensors, count: torch.Tensor, offset: torch.Tensor, start: int
    ) -> tuple[Tensors, torch.Tensor, torch.Tensor]:
        """Gather a request's chunk and read its incoming parameters before the update."""
        inputs = {
            name: values[name] for name, source in self.bindings.items() if source in ("input", "target", "embedding")
        }
        hidden = inputs[self.input_name]
        indices = torch.arange(self.chunk, device=hidden.device) + start - offset
        valid = (indices >= 0) & (indices < count)
        rows = {name: value.index_select(0, indices.clamp(0, len(hidden) - 1)) for name, value in inputs.items()}
        state = {source: values[name] for name, source in self.bindings.items() if source in self.effect.applied_fields}
        return rows, valid, overlay_apply(self.effect, rows[self.input_name], state)

    def apply(self, values: Tensors, gradients: Tensors, valid: torch.Tensor, complete: torch.Tensor) -> None:
        """Accumulate each request's valid rows and apply its own completion mask."""
        length = values["_pending_length"]
        length.add_(valid.sum())
        parameters = tuple(values[name] for name in gradients)
        sums = tuple(values[f"_pending_grad_{name}"] for name in gradients)
        for total, grad in zip(sums, gradients.values(), strict=True):
            total.add_(grad)
        self.optimizer.tensors(
            parameters,
            tuple(torch.where(complete, total / length.clamp_min(1), 0) for total in sums),
        )
        for total in sums:
            total.masked_fill_(complete, 0)
        length.masked_fill_(complete, 0)

    def advance(self, values: Tensors, count: torch.Tensor, *, batched: bool) -> UpdateOutput:
        """Keep per-request accumulation and SGD eager around the compiled loss derivative."""
        parameters = {
            name: values[name] for name, source in self.bindings.items() if source in self.effect.applied_fields
        }
        constants = {name: values[name] for name, source in self.bindings.items() if source == "weight"}
        fields = {name: value for name, value in values.items() if name not in constants}
        hidden = fields[self.input_name]
        length = fields["_pending_length"]
        offset = length.clone()
        rows = hidden.shape[int(batched)]
        updates, trained = torch.zeros_like(length), torch.zeros_like(length)
        outputs: list[torch.Tensor] = []
        select = torch.vmap(self.rows, in_dims=(0, 0, 0, None)) if batched else self.rows
        apply = torch.vmap(self.apply, out_dims=None) if batched else self.apply
        gradient = self.mapped_gradient if batched else self.gradient
        with torch.inference_mode(False), torch.enable_grad():
            for start in range(0, rows + self.chunk - 1, self.chunk):
                operands, valid, output = select(fields, count, offset, start)
                outputs.append(output)
                gradients = gradient(
                    {
                        name: value.to(torch.promote_types(value.dtype, torch.float32)).clone()
                        for name, value in parameters.items()
                    },
                    operands,
                    constants,
                    valid,
                )
                total = length + valid.sum(-1)
                complete = (total == self.chunk) | (
                    self.flush & fields["_prompt_end"] & (start + self.chunk >= offset + count) & (total > 0)
                )
                updates.add_(complete)
                trained.add_(torch.where(complete, total, 0))
                apply(fields, gradients, valid, complete)
        output = torch.cat(outputs, dim=int(batched))
        positions = torch.arange(rows, device=hidden.device) + offset.unsqueeze(-1)
        output = (
            torch.vmap(torch.index_select, in_dims=(0, None, 0))(output, 0, positions)
            if batched
            else output.index_select(0, positions)
        )
        return UpdateOutput(output, updates > 0, updates, trained, {"tail_rows": length.clone()})


class ProjectionTask:
    """One invocation's projection segments; private writes remain owned by StateExecution."""

    def __init__(
        self,
        driver: StepDriver,
        context: StateTaskContext,
        rows: PackedSamplerRowBatch,
        task: TaskId,
        targets: tuple[Projection, ...],
    ) -> None:
        self.driver, self.context, self.rows, self.task = driver, context, rows, task
        self.pending = set(targets)

    def prepare(self, math: OverlayTransition) -> tuple[TensorBatch, tuple[tuple[LayoutFieldId, ...], ...]]:
        """Open the private transition before replay; carries with incomplete chunks stay shared."""
        selection = self.context.selection.for_resource(math.resource)
        written = math.row_written_fields(
            self.rows.row_batch, self.context.selection, tuple(range(len(selection.refs)))
        )
        self.context.prepare_owned_write(selection.refs, fields=written, copy_fields=None)
        return read_batch(self.context, "private", math.resource), written

    def complete(
        self,
        operands: tuple[ProjectionOperands, ...],
        updates: Mapping[tuple[RequestIdentity, Projection], UpdateOutput],
    ) -> None:
        """Settle captured results through the same task, counters and publication as eager execution."""
        from pllm.execution.step_input import StepInput

        self.pending.clear()
        self.driver.part(
            self.context,
            StepInput(self.rows.row_batch, self.rows, operands=operands, overlay_updates=updates),
            self.task,
            complete=True,
        )

    def run(
        self, operands: tuple[ProjectionOperands, ...], *, complete: bool = True
    ) -> Mapping[tuple[RequestIdentity, Projection], torch.Tensor]:
        """Execute target segments; complete=False keeps targets open across a prompt/decode split."""
        from pllm.execution.step_input import StepInput

        targets = {operand.target for operand in operands}
        assert targets and targets <= self.pending
        if complete:
            self.pending.difference_update(targets)
        return self.driver.part(
            self.context,
            StepInput(self.rows.row_batch, self.rows, operands=operands),
            self.task,
            complete=not self.pending,
        )


@dataclass(frozen=True, slots=True)
class OverlayTransition:
    """One overlay task's mathematics: groups operands per target and phase, runs each group and reports the
    residuals, the changed states and device counters.

    Binding replaces this immutable callable (for instance to install its executor); the TensorOperation that
    runs it keeps private-page access.
    """

    buffers: tuple[str, ...]
    projections: tuple[ProjectionShape, ...]
    phases: tuple[HiddenPhase, ...]
    apply: Callable[[torch.Tensor, Tensors], torch.Tensor]
    function: Callable[..., UpdateOutput]
    arguments: Mapping[str, str]
    visibility: VisibilitySource
    task: TaskSymbol
    operator: ParameterOverlayOperator
    resource: StateResource
    invariants: Mapping[str, tuple[str, tuple[FieldView, ...]]]
    chunk: ChunkUpdate | GradientUpdate | None = None
    executor: OverlayBatch | None = None
    keys: tuple[RequestKeyField, ...] = (RequestKeyField.REQUEST_ID,)
    custom_forward: bool = False

    @property
    def targets(self) -> tuple[Projection, ...]:
        """The model-order targets of this transition's rank-local projections."""
        return tuple(projection.target for projection in self.projections)

    def invariant_values(self, target: int, batch: TensorBatch) -> Tensors:
        """Borrow each shared field at this projection's layer without request copies."""
        roots = {
            resource: batch.parameter_root(resource)
            for resource in dict.fromkeys(resource for resource, _ in self.invariants.values())
        }
        return {
            name: views[target].read(roots[resource][views[target].field])
            for name, (resource, views) in self.invariants.items()
        }

    def __call__(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        batch = batch.for_resource(self.resource)
        return self._output(data, batch, self._updates(data, batch))

    def trace(self, data: TensorFrame, batch: TensorBatch) -> TaskOutput:
        """Planning's run on maximum-shape meta operands; checks the author's results once, so steps trust them."""
        batch = batch.for_resource(self.resource)
        updates = self._updates(data, batch)
        self.check(data.step.operands, updates)
        return self._output(data, batch, updates)

    def _updates(self, data: TensorFrame, batch: TensorBatch) -> dict[tuple[RequestIdentity, Projection], UpdateOutput]:
        """Each visible request's update of each target."""
        operands = {
            (operand.identity, operand.target): operand
            for operand in data.step.operands
            if self.visibility.in_forward or operand.phase in self.phases
        }
        indices = dict(zip((row.identity for row in batch.selection.rows), batch.selection.state_indices, strict=True))
        updates: dict[tuple[RequestIdentity, Projection], UpdateOutput] = {}
        with torch.no_grad():
            for target_index, target in enumerate(self.targets):
                for train in (False, True):
                    group = tuple(
                        operand
                        for (_, name), operand in operands.items()
                        if name == target and (operand.phase in self.phases) == train
                    )
                    if not group:
                        continue
                    group_indices = tuple(indices[operand.identity] for operand in group)
                    if data.step.overlay_updates:
                        results = tuple(data.step.overlay_updates[o.identity, target] for o in group)
                    elif self.executor is None:
                        values = self.private_values(target_index, group_indices, batch)
                        results = tuple(self.result(o, v) for o, v in zip(group, values, strict=True))
                    else:
                        results = self.executor.run(group, target_index, group_indices, batch)
                    updates.update(((o.identity, target), result) for o, result in zip(group, results, strict=True))
        return updates

    def _output(
        self,
        data: TensorFrame,
        batch: TensorBatch,
        updates: Mapping[tuple[RequestIdentity, Projection], UpdateOutput],
    ) -> TaskOutput:
        """Record one entry per updated request, and the residuals and counters of the step."""
        if not updates:
            return TaskOutput()
        selected = dict.fromkeys(
            (row.identity, index)
            for row, index in zip(batch.selection.rows, batch.selection.state_indices, strict=True)
        )
        start = len(data.entries)
        changed: dict[int, torch.Tensor] = {}
        for identity, index in selected:
            results = tuple(
                OverlayTargetResult(target, updates[identity, target])
                for target in self.targets
                if (identity, target) in updates
            )
            if results:
                data.entries.append(
                    OverlayEntry(identity, batch.selection.refs[int(index)], batch.after_refs[int(index)], results)
                )
                changed[index] = results[0].update.changed
                for result in results[1:]:
                    changed[index] = changed[index] | result.update.changed
        return TaskOutput(
            residuals={
                (entry.identity, target.target): target.update.output
                for entry in data.entries[start:]
                for target in entry.targets
            },
            changed=tuple(changed),
            update_mask=torch.stack(tuple(changed.values())),
            counters=count_updates(tuple(updates.values())),
        )

    def private_values(self, target: int, indices: tuple[int, ...], batch: TensorBatch) -> tuple[Tensors, ...]:
        """Borrow each request's private pages and the target's shared fields.

        The task already owns these pages; borrowing their views avoids copying every dense carry twice.
        """
        width = len(self.buffers)
        fields = tuple(batch.field_slots(target * width + i) for i in range(width))
        invariants = self.invariant_values(target, batch)
        return tuple(
            {
                **{name: pool[slots[index]] for name, (pool, slots) in zip(self.buffers, fields, strict=True)},
                **invariants,
            }
            for index in indices
        )

    def chunk_offset(self, row: PackedPhaseRow, count: int) -> int | None:
        """The contiguous training frontier, when the backend supplied complete row metadata."""
        if (
            isinstance(self.chunk, ChunkUpdate)
            and self.visibility.in_forward
            and RequestKeyField.REQUEST_ID in self.keys
            and not self.custom_forward
            and row.prompt_length
            and row.phase in self.phases
        ):
            offset = row.token_offset + 1 - count
            if HiddenPhase.PREFILL not in self.phases:
                offset -= row.prompt_length
            return offset
        return None

    def row_written_fields(
        self, rows: PackedPhaseRowBatch, selection: RequestStateSelection, selected: tuple[int, ...]
    ) -> tuple[tuple[LayoutFieldId, ...], ...]:
        """Share unchanged chunk carries; tails still receive private writable pages."""
        fields = tuple(LayoutFieldId(i) for i in range(len(self.targets) * len(self.buffers)))
        if not isinstance(self.chunk, ChunkUpdate):
            return (fields,) * len(selected)
        carry = self.buffers.index(self.arguments[self.chunk.arguments[0]])
        tails = tuple(field for field in fields if field % len(self.buffers) != carry)
        counts = Counter(row.identity for row in rows.rows if row.phase in self.phases)
        by_state = dict(zip(selection.for_resource(self.resource).state_indices, selection.rows, strict=True))
        writes: list[tuple[LayoutFieldId, ...]] = []
        for index in selected:
            row = by_state[index]
            count = counts[row.identity]
            offset = self.chunk_offset(row, count)
            complete = offset is None or offset % self.chunk.chunk + count >= self.chunk.chunk
            writes.append(fields if complete else tails)
        return tuple(writes)

    def written_fields(
        self, data: TensorFrame, selection: RequestStateSelection, selected: tuple[int, ...]
    ) -> tuple[tuple[LayoutFieldId, ...], ...]:
        """Share unchanged chunk carries during eager task execution."""
        return self.row_written_fields(data.step.rows, selection, selected)

    def bind(
        self,
        values: Tensors,
        hidden: torch.Tensor,
        desired: torch.Tensor,
        embedding: torch.Tensor,
        weight: torch.Tensor,
        prompt_end: torch.Tensor,
    ) -> Tensors:
        """Bind state, shared fields and projection operands to the author's parameter names.

        Arguments whose field is absent from `values` stay unbound; decode reads its tails in place.
        """
        inputs = {
            **values,
            "input": hidden,
            "target": desired,
            "embedding": embedding,
            "weight": weight,
            "prompt_end": prompt_end,
        }
        return {key: inputs[source] for key, source in self.arguments.items() if source in inputs}

    def inputs(self, operand: ProjectionOperands, values: Tensors) -> Tensors:
        """Bind one request's projection operands."""
        return self.bind(
            values, operand.hidden, operand.desired, operand.embedding, operand.base_weight, operand.prompt_end
        )

    def result(self, operand: ProjectionOperands, values: Tensors) -> UpdateOutput:
        """Run one request eagerly."""
        if operand.phase not in self.phases:
            return UpdateOutput.unchanged(self.apply(operand.hidden, values))
        return self.function(**self.inputs(operand, values))

    def check(
        self,
        operands: Sequence[ProjectionOperands],
        updates: Mapping[tuple[RequestIdentity, Projection], UpdateOutput],
    ) -> None:
        """Check the author's results once, from declaration's run on maximum-shape meta operands.

        Every update kind returns its residual and counters here, so no step checks them again.
        """
        hidden = {(operand.identity, operand.target): operand.hidden for operand in operands}
        for (identity, target), result in updates.items():
            rows = hidden[identity, target]
            size = self.projections[self.targets.index(target)].output_size
            if type(result) is not UpdateOutput:
                raise TypeError(
                    f"{self.task} must return pllm.UpdateOutput(residual, changed, update_count, trained_rows); "
                    f"got {type(result).__name__}"
                )
            if (
                result.output.shape != (len(rows), size)
                or result.output.dtype not in (rows.dtype, torch.float32)
                or result.output.device != rows.device
            ):
                raise ValueError(
                    f"{self.task} must return the residual for {target} as [rows, {size}] in the input "
                    f"dtype or float32 on the input's device; got {tuple(result.output.shape)} "
                    f"{result.output.dtype} on {result.output.device}"
                )
            if any(
                value.shape != () or value.device != rows.device or value.dtype != dtype
                for value, dtype in (
                    (result.changed, torch.bool),
                    (result.update_count, torch.int64),
                    (result.trained_rows, torch.int64),
                )
            ):
                raise ValueError(
                    f"{self.task} must return changed (bool) and update_count and trained_rows (int64) "
                    "as 0-d tensors on the input's device"
                )


class ProgramOperand(NamedTuple):
    """Capture outputs without retaining the cold invocation's request or phase."""

    target: Projection
    hidden: torch.Tensor
    desired: torch.Tensor
    weight: torch.Tensor
    embedding: torch.Tensor


def join_encounters(
    operands: Sequence[ProjectionOperands], key: Callable[[ProjectionOperands], Hashable]
) -> tuple[ProjectionOperands, ...]:
    """Concatenate the rows of operands sharing a key, in first-seen order; the last encounter's prompt end wins."""
    groups: dict[Hashable, list[ProjectionOperands]] = {}
    for operand in operands:
        groups.setdefault(key(operand), []).append(operand)
    return tuple(
        replace(
            records[0],
            hidden=torch.cat(tuple(record.hidden for record in records)),
            desired=torch.cat(tuple(record.desired for record in records)),
            embedding=torch.cat(tuple(record.embedding for record in records)),
            prompt_end=records[-1].prompt_end,
        )
        for records in groups.values()
    )


class ProgramOverlay:
    """Read and capture at projections; apply grouped operands within one owned transition."""

    def __init__(
        self,
        math: OverlayTransition,
        batch: TensorBatch,
        entry: TensorBatch,
        data: TensorFrame,
    ) -> None:
        self.math, self.data = math, data
        self.batch, self.entry = batch.for_resource(math.resource), entry.for_resource(math.resource)
        self.entry_fields: tuple[torch.Tensor, ...] = ()
        self.fields: tuple[torch.Tensor, ...] = ()
        self.saved: tuple[torch.Tensor, ...] = ()
        self.gate = torch.zeros((1, 1), device=batch.roots[0][0].device)
        self.prompt_end = torch.zeros((), device=batch.roots[0][0].device, dtype=torch.bool)
        self.pending: list[ProjectionOperands] = []
        self.ordinary: list[ProjectionOperands] = []
        self.results: list[OverlayTargetResult] = []
        self.row = 0
        self.in_program = self.private = False

    def stage(self, row: int, in_program: bool) -> None:
        """Resolve input/private roots once, outside cold execution and graph replay."""
        self.row, self.in_program = row, in_program
        if not self.fields:
            self.entry_fields, self.fields, self.saved = (
                tuple(torch.empty_like(field) for field in self.batch.roots[0]) for _ in range(3)
            )
        index = int(self.batch.selection.state_indices[row])
        for ports, fields in (
            (self.entry_fields, self.entry.roots[index]),
            (self.saved, self.batch.roots[index]),
        ):
            for target, source in zip(ports, fields, strict=True):
                target.copy_(source)

    def reset(self) -> None:
        """Restore every cold execution from this token's incoming private fields."""
        for target, source in zip(self.fields, self.saved, strict=True):
            target.copy_(source)
        self.pending, self.results = [], []
        self.private = self.math.visibility.in_forward
        self.gate.zero_()

    def values(self, target: Projection, private: bool) -> Tensors:
        width = len(self.math.buffers)
        start = self.math.targets.index(target) * width
        return {
            **self.math.invariant_values(self.math.targets.index(target), self.batch),
            **dict(
                zip(
                    self.math.buffers,
                    (self.fields if private else self.entry_fields)[start : start + width],
                    strict=True,
                )
            ),
        }

    def run(
        self, operands: tuple[ProjectionOperands, ...]
    ) -> Mapping[tuple[RequestIdentity, Projection], torch.Tensor]:
        """Own operands immediately because later model calls may mutate their inputs."""
        outputs: dict[tuple[RequestIdentity, Projection], torch.Tensor] = {}
        for operand in operands:
            captured = replace(
                operand,
                hidden=operand.hidden.detach().clone(),
                desired=operand.desired.detach().clone(),
                embedding=operand.embedding.detach().clone(),
            )
            if not self.in_program and self.math.visibility.in_forward:
                result = self.math.result(captured, self.values(operand.target, True))
                self.results.append(OverlayTargetResult(operand.target, result))
                output = result.output
            else:
                visible = {
                    name: torch.where(
                        self.gate.reshape(()).bool(),
                        value,
                        self.values(operand.target, self.math.visibility.in_forward)[name],
                    )
                    for name, value in self.values(operand.target, True).items()
                }
                output = self.math.apply(operand.hidden, visible)
                self.pending.append(captured)
            outputs[operand.identity, operand.target] = output
        return outputs

    def apply_updates(self) -> None:
        """Concatenate every encounter at a target before invoking the author's batch math."""
        for operand in join_encounters(self.pending, lambda operand: operand.target):
            result = self.math.result(operand, self.values(operand.target, True))
            self.results.append(OverlayTargetResult(operand.target, result))
        if self.pending:
            self.private = True
            self.gate.fill_(1.0)
        self.pending = []

    def operands(self) -> tuple[ProgramOperand, ...]:
        """Export graph outputs independently of the cold request identity."""
        return tuple(
            ProgramOperand(operand.target, operand.hidden, operand.desired, operand.base_weight, operand.embedding)
            for operand in self.pending
        )

    def collect(self, results: tuple[OverlayTargetResult, ...], pending: tuple[ProgramOperand, ...]) -> None:
        """Reconstruct invocation metadata from device outputs, including cached graph replays."""
        row = self.data.step.rows.inspect_row(self.row)
        index = int(self.batch.selection.state_indices[self.row])
        if self.batch.after_refs[int(index)] != self.batch.selection.refs[int(index)]:
            for target, source in zip(self.batch.roots[index], self.fields, strict=True):
                target.copy_(source)
        copied = tuple(
            OverlayTargetResult(
                item.target,
                replace(
                    item.update,
                    output=item.update.output.clone(),
                    changed=item.update.changed.clone(),
                    update_count=item.update.update_count.clone(),
                    trained_rows=item.update.trained_rows.clone(),
                    metrics={key: value.clone() for key, value in item.update.metrics.items()},
                ),
            )
            for item in results
        )
        if copied:
            self.data.entries.append(
                OverlayEntry(
                    row.identity, self.batch.selection.refs[int(index)], self.batch.after_refs[int(index)], copied
                )
            )
        if not self.in_program:
            self.ordinary.extend(
                ProjectionOperands(
                    row.identity,
                    operand.target,
                    row.phase,
                    operand.hidden.clone(),
                    operand.desired.clone(),
                    operand.weight,
                    operand.hidden.new_full(
                        (), row.token_offset + len(operand.hidden) == row.prompt_length, dtype=torch.bool
                    ),
                    operand.embedding.clone(),
                )
                for operand in pending
            )

    def finish(self) -> TaskOutput:
        """Apply ordinary pinned rows once as a batch; trailing operands inside a custom forward are discarded."""
        grouped = join_encounters(self.ordinary, lambda operand: (operand.identity, operand.target))
        self.math(replace(self.data, step=replace(self.data.step, operands=grouped)), self.batch)
        updates = tuple(target.update for entry in self.data.entries for target in entry.targets)
        return TaskOutput(counters=count_updates(updates) if updates else None)
